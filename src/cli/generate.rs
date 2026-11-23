//! `generate`: one reply to one message.

use crate::*;

pub(crate) fn print_stage_times(label: &str) {
    let stages = rustling::timing::take();
    let total: f64 = stages.iter().map(|(_, seconds)| seconds).sum();
    let parts: Vec<String> = stages
        .iter()
        .filter(|(_, seconds)| *seconds > 0.0)
        .map(|(stage, seconds)| format!("{stage:?} {seconds:.1}s"))
        .collect();
    eprintln!("{label} stages ({total:.1}s in all): {}", parts.join(", "));
}

pub(crate) fn stop_note(stop: Stop, limits: Limits) -> Option<String> {
    match stop {
        Stop::EndOfTurn | Stop::Requested => None,
        Stop::MaxTokens => Some(format!("stopped at the limit of {} tokens", limits.max_tokens)),
        Stop::Loop => Some("stopped: the output was repeating itself (--allow-loops to continue)".to_owned()),
        Stop::ContextFull => Some("stopped: the context window is full (--context sets its size)".to_owned()),
    }
}

pub(crate) fn generate(model_dir: &std::path::Path, text: &str, run: &RunArgs, top: bool, timing: bool) -> Result<()> {
    let config = load_config(model_dir)?;
    let checkpoint = Checkpoint::open(model_dir)?;
    let prompter = Prompter::load(model_dir)?;
    let started = Instant::now();
    let model = Model::load(&config, &checkpoint, run.model())?;
    let backend = match (model.uses_gpu(), model.decodes_on_gpu()) {
        (_, true) => "Metal for long prompts and for generating",
        (true, false) => "Metal for large batches",
        (false, false) => "CPU",
    };
    tracing::info!("loaded in {:.1}s ({backend})", started.elapsed().as_secs_f32());

    let messages = serde_json::json!([{ "role": "user", "content": text }]);
    let rendered = prompter.render(&messages, None, run.effort().as_ref())?;
    let prompt_ids = prompter.encode(&rendered)?;
    tracing::debug!("{rendered}");

    // The prompt goes through in one call; after that each new token is a call of its own.
    let draft_weights = run.draft_weights(model_dir)?;
    let draft = draft_weights
        .as_ref()
        .map(|weights| Draft::load(&config, weights).map(|draft| draft.with_min_confidence(run.draft_confidence)))
        .transpose()?;
    let draft = draft.filter(|_| drafting_pays(&model));
    let mut session = match &draft {
        Some(draft) => Session::new(&model).with_window(run.window()).with_copying(!run.no_copying).with_draft(draft),
        None => Session::new(&model).with_window(run.window()).with_copying(!run.no_copying),
    };
    let started = Instant::now();
    session.set_context(&prompt_ids)?;
    let prompt_seconds = started.elapsed().as_secs_f32();
    if timing {
        print_stage_times("prompt");
        let redone = rustling::model::GPU_EXPERTS_REDONE.load(std::sync::atomic::Ordering::Relaxed);
        if redone > 0 {
            tracing::info!("{redone} experts were recomputed on the CPU after the GPU returned NaN");
        }
        let redone = rustling::model::GPU_ATTENTION_REDONE.load(std::sync::atomic::Ordering::Relaxed);
        if redone > 0 {
            tracing::info!("{redone} stretches of attention were recomputed on the CPU after the GPU returned NaN");
        }
        let redone = rustling::model::GPU_PRODUCTS_REDONE.load(std::sync::atomic::Ordering::Relaxed);
        if redone > 0 {
            let on_cpu = rustling::model::GPU_PRODUCTS_ON_CPU.load(std::sync::atomic::Ordering::Relaxed);
            tracing::info!("{redone} batches of matrix products were recomputed after the GPU returned NaN, {on_cpu} of them on the CPU");
        }
    }

    let started = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut printed = 0;
    let (_, stop) = session.generate(&mut run.sampler()?, run.limits(), config.eos_token_id, |token, logits| {
        generated.push(token);
        if top {
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.select_nth_unstable_by(5, |&a, &b| logits[b].total_cmp(&logits[a]));
            order[..5].sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let candidates: Vec<String> = order[..5]
                .iter()
                .map(|&id| Ok(format!("{:?} {:.2}", prompter.decode(&[id as u32])?, logits[id])))
                .collect::<Result<_>>()?;
            eprintln!("[{}] {}", prompt_ids.len() + generated.len() - 1, candidates.join(" | "));
        }
        // Print whole characters only: a token can end in the middle of a multi-byte character.
        let text = prompter.decode(&generated)?;
        if !text.ends_with('\u{FFFD}') {
            print!("{}", &text[printed..]);
            std::io::stdout().flush()?;
            printed = text.len();
        }
        Ok(true)
    })?;
    println!();
    let seconds = started.elapsed().as_secs_f32();
    if timing {
        print_stage_times("generation");
    }
    if let Some(note) = stop_note(stop, run.limits()) {
        tracing::info!("{note}");
    }
    let drafting = session.drafting();
    if drafting.copied_proposed > 0 {
        tracing::info!(
            "copied: {} of {} proposed tokens accepted in {} rounds",
            drafting.copied_accepted, drafting.copied_proposed, drafting.copied_rounds
        );
    }
    if draft.is_some() {
        tracing::info!(
            "draft: {} of {} proposed tokens accepted over {} rounds ({:.1} tokens per round); rested for {} tokens",
            drafting.accepted,
            drafting.proposed,
            drafting.rounds,
            (drafting.rounds + drafting.accepted) as f32 / drafting.rounds.max(1) as f32,
            drafting.rested
        );
    }
    tracing::info!(
        "prompt: {} tokens in {prompt_seconds:.1}s; generated {} tokens in {seconds:.1}s ({:.2} tokens/s)",
        prompt_ids.len(),
        generated.len(),
        generated.len() as f32 / seconds
    );
    Ok(())
}
