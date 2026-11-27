//! The engine through its public API: a request's [`Cancel`] stops it queued or mid-reply, and the engine serves
//! the next request as usual; a request without tool calls gets text. Needs the model; skipped otherwise. Run with
//! `--release`.

use rustling::engine::{Cancel, Engine, EngineOptions, Event, FailureKind, Request};
use rustling::generate::sampler::SamplerConfig;
use rustling::model::{Activations, Backend, Options};
use rustling::weights::checkpoint::{resolve_checkpoint, resolve_model};
use serde_json::json;

fn options() -> EngineOptions {
    EngineOptions {
        model: Options { activations: Activations::Quantised, backend: Backend::Auto },
        sampling: SamplerConfig { temperature: 0.0, top_k: 0, top_p: 1.0 },
        max_tokens: 400,
        stop_on_loop: true,
        draft_confidence: None,
        copying: true,
        no_reasoning: false,
        passes: None,
        context: 4096,
        cache_bytes: 0,
        cache_dir: None,
    }
}

fn request(text: &str) -> Request {
    Request {
        reasoning_effort: Some(json!("none")),
        ..Request::new(vec![json!({ "role": "user", "content": text })])
    }
}

#[test]
fn cancelled_requests_stop_and_the_engine_carries_on() {
    let Ok(model_dir) = resolve_checkpoint() else {
        eprintln!("skipping: model not found");
        return;
    };
    let engine = Engine::start(&model_dir, options()).unwrap();
    let client = engine.client();

    // Cancelled before the engine takes it up: it never starts.
    let cancel = Cancel::new();
    cancel.cancel();
    let mut events = client.submit(Request { cancel, ..request("Hello") }).unwrap();
    match events.blocking_recv() {
        Some(Event::Failed(failure)) => assert_eq!(failure.kind, FailureKind::Cancelled),
        other => panic!("expected a cancelled request, got {other:?}"),
    }

    // Cancelled mid-reply: a few pieces of text, then it stops well short of the limit.
    let cancel = Cancel::new();
    let mut events = client.submit(Request { cancel: cancel.clone(), ..request("Count from one to two hundred.") }).unwrap();
    let mut pieces = 0;
    let last = loop {
        match events.blocking_recv() {
            Some(Event::Reasoning(_) | Event::Content(_)) => {
                pieces += 1;
                if pieces == 5 {
                    cancel.cancel();
                }
            }
            Some(Event::Started) => {}
            other => break other,
        }
    };
    match last {
        Some(Event::Failed(failure)) => assert_eq!(failure.kind, FailureKind::Cancelled),
        other => panic!("expected the reply to be cancelled, got {other:?}"),
    }
    assert!(pieces < 20, "{pieces} pieces arrived after cancelling");

    // The next request is served as usual.
    let mut events = client.submit(request("What is the capital of France? One word.")).unwrap();
    let mut text = String::new();
    loop {
        match events.blocking_recv() {
            Some(Event::Content(piece)) => text.push_str(&piece),
            Some(Event::Done { .. }) => break,
            Some(Event::Failed(failure)) => panic!("{failure}"),
            Some(_) => {}
            None => panic!("the engine dropped the request"),
        }
    }
    assert!(text.contains("Paris"), "{text}");

    drop(client);
    engine.shutdown().unwrap();
}

#[test]
fn a_reply_without_tool_calls_answers_in_text() {
    // The model the chat runs, which is the one that calls tools.
    let Ok(model_dir) = resolve_model(None) else {
        eprintln!("skipping: model not found");
        return;
    };
    let engine = Engine::start(&model_dir, options()).unwrap();
    let client = engine.client();
    let tools = json!([{ "type": "function", "function": {
        "name": "get_weather",
        "description": "The current weather in a city.",
        "parameters": { "type": "object", "properties": { "city": { "type": "string" } }, "required": ["city"] },
    } }]);
    let ask = |no_tool_calls: bool| {
        let mut messages = vec![json!({ "role": "user", "content": "What is the weather in Paris right now? Use the get_weather tool." })];
        if no_tool_calls {
            // What a subagent's last step says, alongside the tools being off.
            messages.push(json!({ "role": "system", "content": "This is your last step, and tools are off for it. Write your final reply now from what you have found, and say what you did not get to." }));
        }
        let req = Request {
            tools: Some(tools.clone()),
            no_tool_calls,
            reasoning_effort: Some(json!("none")),
            ..Request::new(messages)
        };
        let mut events = client.submit(req).unwrap();
        let (mut calls, mut text) = (0, String::new());
        loop {
            match events.blocking_recv() {
                Some(Event::ToolCall { .. }) => calls += 1,
                Some(Event::Content(piece)) => text.push_str(&piece),
                Some(Event::Done { .. }) => return (calls, text),
                Some(Event::Failed(failure)) => panic!("{failure}"),
                Some(_) => {}
                None => panic!("the engine went away"),
            }
        }
    };
    // With the tools on, the model calls one. With calls off, it answers in text: it is told so, as a subagent's last
    // step is; without being told, a model set on calling a tool may just stop.
    let (calls, _) = ask(false);
    assert_eq!(calls, 1, "the model was expected to call the tool");
    let (calls, text) = ask(true);
    assert_eq!(calls, 0);
    assert!(!text.trim().is_empty(), "no text either");
    drop(client);
    engine.shutdown().unwrap();
}
