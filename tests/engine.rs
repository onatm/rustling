//! The engine through its public API: a request's [`Cancel`] stops it queued or mid-reply, and the engine serves
//! the next request as usual. Needs the model; skipped otherwise. Run with `--release`.

use rustling::engine::{Cancel, Engine, EngineOptions, Event, FailureKind, Request};
use rustling::generate::sampler::SamplerConfig;
use rustling::model::{Activations, Backend, Options};
use rustling::weights::checkpoint::resolve_checkpoint;
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
