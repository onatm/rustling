//! The thread that owns the model: it takes requests one at a time, prepares each prompt and generates the reply.

use super::*;

/// Loads the model, reports readiness, then serves jobs one at a time until every sender is gone.
pub(crate) fn worker(model_dir: &Path, options: ServerOptions, queue: std_mpsc::Receiver<Job>, ready: std_mpsc::Sender<Result<()>>) {
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
        Some(_) => match rustling::weights::checkpoint::companion_dir(model_dir).and_then(|dir| Checkpoint::open_file(&dir.join("mtp.safetensors"))) {
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
    if draft.is_some() && model.decodes_on_gpu() {
        eprintln!("--draft is not used: tokens are generated on the GPU, where one costs less than checking a proposal");
    }
    let draft = draft.filter(|_| !model.decodes_on_gpu());
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
    eprintln!(
        "model loaded ({backend}{drafting}); read {:.1} GB and warmed up in {:.1}s",
        bytes as f64 / 1e9,
        started.elapsed().as_secs_f32()
    );
    eprintln!("context window: {} tokens", options.context);
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
        if options.model.backend == Backend::MetalFast { "fast" } else { "exact" },
        draft.is_some(),
    );
    let mut cache = match &options.cache_dir {
        Some(dir) => match PromptCache::load(dir, &fingerprint, options.cache_bytes) {
            Ok(cache) => cache,
            Err(error) => {
                eprintln!("starting with an empty prompt cache: {error:#}");
                PromptCache::new(options.cache_bytes)
            }
        },
        None => PromptCache::new(options.cache_bytes),
    };
    if !cache.is_empty() {
        eprintln!("prompt cache: {} processed prompts, {:.1} GiB", cache.len(), cache.bytes() as f64 / GIB);
    }

    let mut memory = Memory::default();
    let mut calls = 0usize;
    for (number, job) in queue.into_iter().enumerate() {
        let events = job.events.clone();
        let number = number + 1;
        let handled =
            handle(number, &job, &options, &config, &prompter, &mut session, &mut cache, &mut memory, &mut calls);
        if let Err(failure) = handled {
            eprintln!("[{number}] failed ({}): {}", failure.status.as_u16(), failure.message);
            // The receiver may already be gone, when the failure is the client hanging up.
            let _ = events.blocking_send(Event::Failed { status: failure.status, message: failure.message });
        }
    }

    // Shutting down: the conversation in the session is worth keeping too.
    if let Some(dir) = &options.cache_dir {
        cache.keep(&session);
        let (count, bytes) = (cache.len(), cache.bytes());
        match cache.save(dir, &fingerprint) {
            Ok(()) => eprintln!("saved {count} processed prompts ({:.1} GiB) to {}", bytes as f64 / GIB, dir.display()),
            Err(error) => eprintln!("could not save the prompt cache: {error:#}"),
        }
    }
}

pub(crate) struct Failure {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
}

impl Failure {
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, message: message.into() }
    }
}

impl From<anyhow::Error> for Failure {
    fn from(error: anyhow::Error) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, message: format!("{error:#}") }
    }
}

/// The reasoning behind replies this server generated, keyed by the reply itself.
///
/// Clients usually send the conversation back without the assistant's reasoning. The model did see that
/// reasoning, though, and so did the cached state; putting it back makes the next prompt extend the cached one
/// instead of forcing the whole conversation to be reprocessed.
#[derive(Default)]
pub(crate) struct Memory {
    pub(crate) replies: VecDeque<(String, String)>,
}

impl Memory {
    /// What identifies a reply: its text and tool calls, which clients do send back.
    pub(crate) fn key(message: &Value) -> String {
        let calls: Vec<Value> = message["tool_calls"]
            .as_array()
            .map(|calls| calls.iter().map(|c| json!([c["function"]["name"], c["function"]["arguments"]])).collect())
            .unwrap_or_default();
        json!([message["content"].as_str().unwrap_or_default(), calls]).to_string()
    }

    pub(crate) fn remember(&mut self, message: &Value) {
        if let Some(reasoning) = message["reasoning_content"].as_str() {
            self.replies.push_back((Self::key(message), reasoning.to_owned()));
            if self.replies.len() > REMEMBERED_REPLIES {
                self.replies.pop_front();
            }
        }
    }

    pub(crate) fn recall(&self, message: &Value) -> Option<&str> {
        let key = Self::key(message);
        self.replies.iter().rev().find(|(k, _)| *k == key).map(|(_, reasoning)| reasoning.as_str())
    }
}

/// Brings OpenAI-style messages to what the chat template takes.
pub(crate) fn normalise(messages: &mut [Value], memory: &Memory) -> Result<(), Failure> {
    for message in messages {
        if message["role"] == "developer" {
            message["role"] = "system".into();
        }
        if let Some(parts) = message["content"].as_array() {
            let text_only = parts.iter().all(|p| p.is_string() || matches!(p["type"].as_str(), None | Some("text" | "input_text")));
            if !text_only {
                return Err(Failure::bad_request("only text content is supported"));
            }
        }
        if message["role"] != "assistant" {
            continue;
        }
        // OpenAI sends tool call arguments as a JSON string; the template wants the object.
        if let Some(calls) = message["tool_calls"].as_array_mut() {
            for call in calls {
                if let Some(text) = call["function"]["arguments"].as_str() {
                    let parsed: Value = serde_json::from_str(text)
                        .map_err(|e| Failure::bad_request(format!("tool call arguments are not valid JSON: {e}")))?;
                    call["function"]["arguments"] = parsed;
                }
            }
        }
        if message["content"].is_null() {
            message["content"] = "".into();
        }
        if message.get("reasoning_content").is_none_or(Value::is_null)
            && let Some(reasoning) = memory.recall(message)
        {
            message["reasoning_content"] = reasoning.into();
        }
    }
    Ok(())
}

/// Tokens per second, for the log.
pub(crate) fn rate(tokens: usize, seconds: f32) -> f32 {
    tokens as f32 / seconds.max(1e-3)
}

pub(crate) fn handle(
    number: usize,
    job: &Job,
    options: &ServerOptions,
    config: &Config,
    prompter: &Prompter,
    session: &mut Session,
    cache: &mut PromptCache,
    memory: &mut Memory,
    calls: &mut usize,
) -> Result<(), Failure> {
    let request = &job.request;
    let gone = || anyhow!("the client disconnected");
    let send = |event: Event| job.events.blocking_send(event).map_err(|_| gone());

    let mut messages = request.messages.clone();
    normalise(&mut messages, memory)?;
    let sampling = SamplerConfig {
        temperature: request.temperature.unwrap_or(options.sampling.temperature),
        top_k: request.top_k.unwrap_or(options.sampling.top_k),
        top_p: request.top_p.unwrap_or(options.sampling.top_p),
    };
    if sampling.temperature < 0.0 || !(0.0..=1.0).contains(&sampling.top_p) {
        return Err(Failure::bad_request("temperature must not be negative and top_p must be between 0 and 1"));
    }
    let stops: Vec<&str> = match &request.stop {
        None => vec![],
        Some(StopSequences::One(stop)) => vec![stop.as_str()],
        Some(StopSequences::Many(stops)) => stops.iter().map(String::as_str).collect(),
    };
    let stops: Vec<&str> = stops.into_iter().filter(|s| !s.is_empty()).collect();
    let limits = Limits {
        max_tokens: request.max_completion_tokens.or(request.max_tokens).unwrap_or(options.max_tokens).max(1),
        stop_on_loop: options.stop_on_loop,
    };

    // Template errors (an unknown role, a bad effort level) are the client's to fix.
    // The header, system messages and tool declarations, is what other conversations with the same setup start
    // with. It renders on its own to a prefix of the whole prompt.
    let leading = messages.iter().take_while(|message| message["role"] == "system").count();
    let header = prompter
        .render(&Value::Array(messages[..leading].to_vec()), request.tools.as_ref(), request.reasoning_effort.as_ref())
        .ok();
    let rendered = prompter
        .render(&Value::Array(messages), request.tools.as_ref(), request.reasoning_effort.as_ref())
        .map_err(|e| Failure::bad_request(format!("{e:#}")))?;
    let ids = prompter.encode(&rendered)?;
    let header = match header {
        Some(header) => Some(prompter.encode(&header)?).filter(|header| ids.starts_with(header)).map(|h| h.len()),
        None => None,
    };
    // A prompt that does not fit is the client's to shorten.
    if ids.len() >= session.window() {
        return Err(Failure::bad_request(format!(
            "the prompt is {} tokens, which leaves no room in the context window of {} (set with --context)",
            ids.len(),
            session.window()
        )));
    }
    send(Event::Started)?;
    eprintln!(
        "[{number}] {} messages, {} tools, {} prompt tokens",
        request.messages.len(),
        request.tools.as_ref().and_then(Value::as_array).map_or(0, Vec::len),
        ids.len()
    );
    let started = std::time::Instant::now();
    let mut reported = started;
    let mut watch = |processed: usize| {
        if reported.elapsed().as_secs() >= 5 {
            reported = std::time::Instant::now();
            eprintln!("[{number}] prompt: {processed} of {} tokens after {:.0}s", ids.len(), started.elapsed().as_secs_f32());
        }
        // A client that has gone away is not worth finishing the prompt for; what was processed stays.
        !job.events.is_closed()
    };
    let cached_tokens = cache.prepare_watched(session, &ids, header, &mut watch).map_err(|error| {
        match job.events.is_closed() {
            true => gone(),
            false => error,
        }
    })?;
    let prompt_seconds = started.elapsed().as_secs_f32();
    let processed = ids.len() - cached_tokens;
    eprintln!(
        "[{number}] prompt: {cached_tokens} tokens reused, {processed} processed in {prompt_seconds:.1}s ({:.0} tokens/s)",
        rate(processed, prompt_seconds)
    );
    let started = std::time::Instant::now();

    // Streamed text is the growth of the reply's reasoning and content; tool calls go out whole once closed.
    let mut sampler = Sampler::new(sampling, request.seed);
    let mut generated: Vec<u32> = Vec::new();
    let (mut reasoning_sent, mut content_sent, mut calls_sent) = (0, 0, 0);
    let mut hit_stop = false;
    let mut emit = |generated: &[u32], finished: bool| -> Result<bool> {
        let raw = prompter.decode(generated)?;
        if raw.ends_with('\u{FFFD}') && !finished {
            // Mid-character: wait for the rest.
            return Ok(true);
        }
        let segments = parse_reply(&raw);
        let join = |kind: SegmentKind| -> String {
            segments.iter().filter(|s| s.kind == kind).map(|s| s.text.as_str()).collect()
        };
        let reasoning = join(SegmentKind::Thinking);
        if reasoning.len() > reasoning_sent {
            send(Event::Reasoning(reasoning[reasoning_sent..].to_owned()))?;
            reasoning_sent = reasoning.len();
        }

        let content = join(SegmentKind::Text);
        let stop_at = stops.iter().filter_map(|stop| content.find(stop)).min();
        // Without a match yet, hold back a tail that could still turn into one.
        let longest = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        let held = if finished { 0 } else { longest.saturating_sub(1) };
        let mut upto = stop_at.unwrap_or(content.len().saturating_sub(held));
        while !content.is_char_boundary(upto) {
            upto -= 1;
        }
        if upto > content_sent {
            send(Event::Content(content[content_sent..upto].to_owned()))?;
            content_sent = upto;
        }
        if stop_at.is_some() {
            hit_stop = true;
            return Ok(false);
        }

        let closed_calls = segments.iter().filter(|s| s.kind == SegmentKind::ToolCall && s.closed);
        for (index, segment) in closed_calls.enumerate().skip(calls_sent) {
            let call: Value = serde_json::from_str(&segment.text).unwrap_or_default();
            *calls += 1;
            send(Event::ToolCall {
                index,
                id: format!("call_{calls}"),
                name: call["name"].as_str().unwrap_or_default().to_owned(),
                arguments: call["args"].to_string(),
            })?;
            calls_sent = index + 1;
        }
        Ok(true)
    };
    let drafted_before = session.drafting();
    let (_, stop) = session.generate(&mut sampler, limits, config.eos_token_id, |token, _| {
        generated.push(token);
        emit(&generated, false)
    })?;
    emit(&generated, true)?;

    let segments = parse_reply(&prompter.decode(&generated)?);
    memory.remember(&reply_message(&segments));
    let finish_reason = match stop {
        _ if hit_stop => "stop",
        Stop::EndOfTurn | Stop::Requested if calls_sent > 0 => "tool_calls",
        Stop::EndOfTurn | Stop::Requested => "stop",
        // A loop cut short is reported like any other truncated reply.
        Stop::MaxTokens | Stop::Loop | Stop::ContextFull => "length",
    };
    let seconds = started.elapsed().as_secs_f32();
    let drafted = session.drafting();
    let drafting = match drafted.rounds > drafted_before.rounds || drafted.rested > drafted_before.rested {
        true => format!(
            "; draft: {} of {} proposed tokens accepted, rested for {}",
            drafted.accepted - drafted_before.accepted,
            drafted.proposed - drafted_before.proposed,
            drafted.rested - drafted_before.rested
        ),
        false => String::new(),
    };
    let copied = match drafted.copied_proposed > drafted_before.copied_proposed {
        true => format!(
            "; copied: {} of {} proposed tokens accepted in {} rounds",
            drafted.copied_accepted - drafted_before.copied_accepted,
            drafted.copied_proposed - drafted_before.copied_proposed,
            drafted.copied_rounds - drafted_before.copied_rounds
        ),
        false => String::new(),
    };
    eprintln!(
        "[{number}] reply: {} tokens in {seconds:.1}s ({:.1} tokens/s), {finish_reason}{drafting}{copied}",
        generated.len(),
        rate(generated.len(), seconds)
    );
    let usage = Usage { prompt_tokens: ids.len(), cached_tokens, completion_tokens: generated.len() };
    send(Event::Done { finish_reason, usage })?;
    Ok(())
}
