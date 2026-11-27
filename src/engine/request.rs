//! Handling one request: render and tokenize the conversation, bring the session to it (reusing what the cache
//! holds), then generate the reply and stream it as events.

use super::*;

/// Tokens per second, for the log.
pub(super) fn rate(tokens: usize, seconds: f32) -> f32 {
    tokens as f32 / seconds.max(1e-3)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn handle(
    number: usize,
    job: &Job,
    options: &EngineOptions,
    config: &Config,
    prompter: &Prompter,
    session: &mut Session,
    cache: &mut PromptCache,
    waiting: &AtomicUsize,
) -> Result<(), Failure> {
    let request = &job.request;
    let gone = || anyhow!("the client disconnected");
    let send = |event: Event| job.events.blocking_send(event).map_err(|_| gone());
    let cancelled = || request.cancel.is_cancelled();
    if cancelled() {
        return Err(Failure::cancelled());
    }

    let messages = &request.messages;
    let sampling = SamplerConfig {
        temperature: request.temperature.unwrap_or(options.sampling.temperature),
        top_k: request.top_k.unwrap_or(options.sampling.top_k),
        top_p: request.top_p.unwrap_or(options.sampling.top_p),
    };
    if sampling.temperature < 0.0 || !(0.0..=1.0).contains(&sampling.top_p) {
        return Err(Failure::bad_request("temperature must not be negative and top_p must be between 0 and 1"));
    }
    let stops: Vec<&str> = request.stop.iter().map(String::as_str).filter(|s| !s.is_empty()).collect();
    let limits = Limits {
        max_tokens: request.max_tokens.unwrap_or(options.max_tokens).max(1),
        stop_on_loop: options.stop_on_loop,
    };

    // Template errors (an unknown role, a bad effort level) are the client's to fix.
    // The header, system messages and tool declarations, is what other conversations with the same setup start
    // with. It renders on its own to a prefix of the whole prompt.
    // Without reasoning, the prompt asks for none unless the request says otherwise.
    let no_reasoning = options.no_reasoning || request.no_reasoning;
    let effort = request.reasoning_effort.clone().or_else(|| no_reasoning.then(|| json!("none")));
    let leading = messages.iter().take_while(|message| message["role"] == "system").count();
    let header = prompter
        .render(&Value::Array(messages[..leading].to_vec()), request.tools.as_ref(), effort.as_ref())
        .ok();
    let rendered = prompter
        .render(&Value::Array(messages.clone()), request.tools.as_ref(), effort.as_ref())
        .map_err(|e| Failure::bad_request(format!("{e:#}")))?;
    let ids = prompter.encode(&rendered)?;
    let header = match header {
        Some(header) => Some(prompter.encode(&header)?).filter(|header| ids.starts_with(header)).map(|h| h.len()),
        None => None,
    };
    // A prompt that does not fit is the client's to shorten.
    if ids.len() >= session.window() {
        return Err(Failure {
            kind: FailureKind::ContextOverflow,
            message: format!(
                "the prompt is {} tokens, which leaves no room in the context window of {} (set with --context)",
                ids.len(),
                session.window()
            ),
        });
    }
    send(Event::Started)?;
    let tools = request.tools.as_ref().and_then(Value::as_array).map_or(0, Vec::len);
    tracing::info!("[{number}] {} messages, {tools} tools, {} prompt tokens", request.messages.len(), ids.len());
    let started = std::time::Instant::now();
    let mut reported = started;
    let mut watch = |processed: usize| {
        if reported.elapsed().as_secs() >= 5 {
            reported = std::time::Instant::now();
            tracing::debug!("[{number}] prompt: {processed} of {} tokens after {:.0}s", ids.len(), started.elapsed().as_secs_f32());
        }
        // A client that has gone away has often only run out of patience, and will ask again: its prompt is
        // finished for it, unless another request is waiting for the model. What was processed stays either way.
        // A cancelled request stops at once.
        !cancelled() && (!job.events.is_closed() || waiting.load(Ordering::Relaxed) == 0)
    };
    let cached_tokens = cache.prepare_watched(session, &ids, header, &mut watch).map_err(|error| {
        match (cancelled(), job.events.is_closed()) {
            (true, _) => Failure::cancelled(),
            (false, true) => gone().into(),
            (false, false) => error.into(),
        }
    })?;
    let prompt_seconds = started.elapsed().as_secs_f32();
    let processed = ids.len() - cached_tokens;
    tracing::info!(
        "[{number}] prompt: {cached_tokens} tokens reused, {processed} processed in {prompt_seconds:.1}s ({:.0} tokens/s)",
        rate(processed, prompt_seconds)
    );
    // Work the GPU got wrong and that was done again: worth knowing about, since it costs time.
    let redone = |count: &std::sync::atomic::AtomicUsize| count.swap(0, std::sync::atomic::Ordering::Relaxed);
    let (products, on_cpu) = (redone(&crate::model::GPU_PRODUCTS_REDONE), redone(&crate::model::GPU_PRODUCTS_ON_CPU));
    let (experts, attention) = (redone(&crate::model::GPU_EXPERTS_REDONE), redone(&crate::model::GPU_ATTENTION_REDONE));
    if products + experts + attention > 0 {
        tracing::warn!(
            "[{number}] the GPU returned NaN and work was redone: {products} batches of products ({on_cpu} of them on the \
             CPU), {experts} experts, {attention} stretches of attention"
        );
    }
    let started = std::time::Instant::now();

    // Streamed text is the growth of the reply's reasoning and content; tool calls go out whole once closed.
    let mut sampler = Sampler::new(sampling, request.seed);
    if no_reasoning {
        sampler = sampler.without(prompter.reasoning_token()?);
    }
    if request.no_tool_calls {
        for token in prompter.tool_call_tokens()? {
            sampler = sampler.without(token);
        }
    }
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
            send(Event::ToolCall {
                index,
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
        if cancelled() {
            return Ok(false);
        }
        emit(&generated, false)
    })?;
    if cancelled() {
        tracing::info!("[{number}] cancelled after {} tokens", generated.len());
        return Err(Failure::cancelled());
    }
    emit(&generated, true)?;

    let finish = match stop {
        _ if hit_stop => Finish::Stop,
        Stop::EndOfTurn | Stop::Requested if calls_sent > 0 => Finish::ToolCalls,
        Stop::EndOfTurn | Stop::Requested => Finish::Stop,
        // A loop cut short is reported like any other truncated reply.
        Stop::MaxTokens | Stop::Loop | Stop::ContextFull => Finish::Length,
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
    tracing::info!(
        "[{number}] reply: {} tokens in {seconds:.1}s ({:.1} tokens/s), {}{drafting}{copied}",
        generated.len(),
        rate(generated.len(), seconds),
        finish.as_str()
    );
    let usage = Usage {
        prompt_tokens: ids.len(),
        cached_tokens,
        completion_tokens: generated.len(),
        prompt_ms: (prompt_seconds * 1000.0) as u64,
        reply_ms: (seconds * 1000.0) as u64,
    };
    send(Event::Done { finish, usage })?;
    Ok(())
}
