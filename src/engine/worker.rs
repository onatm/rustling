//! The thread that owns the model: it loads it, takes requests one at a time, and keeps the prompt cache on disk
//! up to date between them.

use super::*;

/// How long the model must be idle before the prompt cache is brought up to date on disk.
const IDLE_BEFORE_SAVING: std::time::Duration = std::time::Duration::from_secs(2);
/// How long before the conversation in the session is kept as a snapshot and saved with the rest.
const IDLE_BEFORE_KEEPING: std::time::Duration = std::time::Duration::from_secs(60);

/// Loads the model, reports readiness, then serves jobs one at a time until every sender is gone.
pub(super) fn worker(
    model_dir: &Path,
    options: EngineOptions,
    queue: std_mpsc::Receiver<Job>,
    waiting: Arc<AtomicUsize>,
    ready: std_mpsc::Sender<Result<()>>,
) {
    let loaded = (|| -> Result<_> {
        let config = Config::load(model_dir)?;
        let config = match options.passes {
            Some(passes) => config.with_passes(passes)?,
            None => config,
        };
        Ok((config, Checkpoint::open(model_dir)?, Prompter::load(model_dir)?))
    })();
    let (config, checkpoint, prompter) = match loaded {
        Ok(loaded) => loaded,
        Err(error) => return drop(ready.send(Err(error))),
    };
    let model = match Model::load(&config, &checkpoint, options.model) {
        Ok(model) => model,
        Err(error) => return drop(ready.send(Err(error))),
    };
    let draft_weights = match options.draft_confidence {
        None => None,
        Some(_) => match crate::weights::checkpoint::companion_dir(model_dir).and_then(|dir| Checkpoint::open_file(&dir.join("mtp.safetensors"))) {
            Ok(weights) => Some(weights),
            Err(error) => return drop(ready.send(Err(error.context("opening the draft model")))),
        },
    };
    let draft = match (&draft_weights, options.draft_confidence) {
        (Some(weights), Some(confidence)) => match Draft::load(&config, weights) {
            Ok(draft) => Some(draft.with_min_confidence(confidence)),
            Err(error) => return drop(ready.send(Err(error))),
        },
        _ => None,
    };
    let drafting_pays = !model.decodes_on_gpu() || crate::generate::draft::tried_on_gpu();
    if draft.is_some() && !drafting_pays {
        tracing::warn!("--draft is not used: tokens are generated on the GPU, where one costs less than checking a proposal");
    }
    let draft = draft.filter(|_| drafting_pays);
    // Bring the weights into memory and run a token through the model, so the first request is not the slow one.
    let started = std::time::Instant::now();
    let bytes = checkpoint.read_through();
    if let Err(error) = model.warm_up() {
        return drop(ready.send(Err(error.context("warming up"))));
    }
    let backend = match (model.uses_gpu(), model.decodes_on_gpu()) {
        (_, true) => "Metal for long prompts and for generating",
        (true, false) => "Metal for large batches",
        (false, false) => "CPU",
    };
    let drafting = if draft.is_some() { ", with the draft model" } else { "" };
    tracing::info!(
        "model loaded ({backend}{drafting}); read {:.1} GB and warmed up in {:.1}s",
        bytes as f64 / 1e9,
        started.elapsed().as_secs_f32()
    );
    tracing::info!("context window: {} tokens", options.context);
    let _ = ready.send(Ok(()));

    let mut session = match &draft {
        Some(draft) => Session::new(&model).with_window(options.context).with_copying(options.copying).with_draft(draft),
        None => Session::new(&model).with_window(options.context).with_copying(options.copying),
    };
    // What snapshots depend on: one made with another model, pass count or arithmetic must not be reused.
    // The leading number changes with the layout of a saved state.
    let fingerprint = format!(
        "state 3|{}|{} bytes|{} visits|{:?}|{}|draft {}",
        model_dir.display(),
        checkpoint.maps().iter().map(|map| map.len()).sum::<usize>(),
        config.text_config.visits().len(),
        options.model.activations,
        // The fast path's states differ from the exact ones, so the two must not share snapshots.
        if cfg!(feature = "split-attention") && model.decodes_on_gpu() { "fast-split" }
        else if model.on_fast_path() { "fast" } else { "exact" },
        draft.is_some(),
    );
    let mut cache = match &options.cache_dir {
        Some(dir) => match PromptCache::load(dir, &fingerprint, options.cache_bytes) {
            Ok(cache) => cache,
            Err(error) => {
                tracing::warn!("starting with an empty prompt cache: {error:#}");
                PromptCache::new(options.cache_bytes)
            }
        },
        None => PromptCache::new(options.cache_bytes),
    };
    if !cache.is_empty() {
        tracing::info!("prompt cache: {} processed prompts, {:.1} GiB", cache.len(), cache.bytes() as f64 / GIB);
    }

    let mut number = 0;
    // When the session's conversation last changed, while it has not been kept as a snapshot.
    let mut live_since: Option<std::time::Instant> = None;
    loop {
        // Between requests, what the cache has gained goes to disk: only the new snapshots are written, and a
        // server that is killed loses nothing but the conversation it was in.
        let job = match queue.recv_timeout(IDLE_BEFORE_SAVING) {
            Ok(job) => job,
            Err(std_mpsc::RecvTimeoutError::Timeout) => {
                // The conversation in the session is only a snapshot once something displaces it. Left alone
                // for a while, it is kept as one, so that it too outlives the server; not sooner, because that
                // writes its whole state, a gigabyte for a long one, and an agent's turns come seconds apart.
                if live_since.is_some_and(|since| since.elapsed() >= IDLE_BEFORE_KEEPING) && options.cache_dir.is_some() {
                    cache.keep(&session);
                    live_since = None;
                }
                if let Some(dir) = options.cache_dir.as_ref().filter(|_| cache.unsaved()) {
                    match cache.save(dir, &fingerprint) {
                        Ok(written) => tracing::debug!("saved {written} processed prompts to {}", dir.display()),
                        Err(error) => tracing::warn!("could not save the prompt cache: {error:#}"),
                    }
                }
                continue;
            }
            Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
        };
        waiting.fetch_sub(1, Ordering::Relaxed);
        let events = job.events.clone();
        number += 1;
        let handled =
            handle(number, &job, &options, &config, &prompter, &mut session, &mut cache, &waiting);
        if let Err(failure) = handled {
            match failure.kind {
                // Asked for: nothing went wrong.
                FailureKind::Cancelled => tracing::info!("[{number}] cancelled"),
                _ => tracing::warn!("[{number}] failed ({:?}): {}", failure.kind, failure.message),
            }
            // The receiver may already be gone, when the failure is the client hanging up.
            let _ = events.blocking_send(Event::Failed(failure));
        }
        live_since = Some(std::time::Instant::now());
    }

    // Shutting down: the conversation in the session is worth keeping too.
    if let Some(dir) = &options.cache_dir {
        cache.keep(&session);
        let (count, bytes) = (cache.len(), cache.bytes());
        match cache.save(dir, &fingerprint) {
            Ok(_) => tracing::info!("{count} processed prompts ({:.1} GiB) kept in {}", bytes as f64 / GIB, dir.display()),
            Err(error) => tracing::warn!("could not save the prompt cache: {error:#}"),
        }
    }
}
