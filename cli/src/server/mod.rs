//! An OpenAI-compatible HTTP server: `POST /v1/chat/completions` (streaming or not), `POST /v1/responses`,
//! `GET /v1/models`, `GET /health`.
//!
//! The model runs in the [engine](rustling::engine), which takes requests one at a time; this module only
//! translates between HTTP and its requests and events.

use std::convert::Infallible;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rustling::engine::{self, Client, Engine, EngineOptions, Failure, FailureKind, Request, Usage};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

mod openai;
mod responses;

use openai::*;
use responses::*;

#[derive(Clone)]
struct App {
    client: Client,
    clients: Clients,
}

pub struct ServerOptions {
    pub host: String,
    pub port: u16,
    pub engine: EngineOptions,
}

/// The fields of a chat completion request that the server acts on. Unknown fields are ignored.
#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Value>,
    tools: Option<Value>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    /// A level name or a number, passed to the chat template.
    reasoning_effort: Option<Value>,
    stop: Option<StopSequences>,
}

impl ChatRequest {
    fn into_request(self) -> Request {
        Request {
            messages: self.messages,
            tools: self.tools,
            max_tokens: self.max_completion_tokens.or(self.max_tokens),
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
            reasoning_effort: self.reasoning_effort,
            cancel: Default::default(),
            no_reasoning: false,
            stop: match self.stop {
                None => vec![],
                Some(StopSequences::One(stop)) => vec![stop],
                Some(StopSequences::Many(stops)) => stops,
            },
        }
    }
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
}

fn usage_json(usage: Usage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
        "prompt_tokens_details": { "cached_tokens": usage.cached_tokens },
    })
}

pub fn serve(model_dir: &Path, options: ServerOptions) -> Result<()> {
    let address = format!("{}:{}", options.host, options.port);
    let engine = Engine::start(model_dir, options.engine)?;
    let client = engine.client();

    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    let served = runtime.block_on(async move {
        let app = Router::new()
            .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/responses", post(responses))
            .with_state(App { client, clients: Clients::default() });
        let listener = tokio::net::TcpListener::bind(&address).await.with_context(|| format!("binding {address}"))?;
        tracing::info!("listening on http://{address}/v1");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    });
    // Dropping the runtime drops the last client, so the model thread saves its cache and ends.
    drop(runtime);
    engine.shutdown()?;
    served
}

async fn models(State(app): State<App>) -> Json<Value> {
    let model = app.client.model_name();
    Json(json!({ "object": "list", "data": [{ "id": model, "object": "model", "owned_by": "thinkingmachines" }] }))
}

fn failure_response(failure: &Failure) -> Response {
    let status = match failure.kind {
        FailureKind::BadRequest | FailureKind::ContextOverflow => StatusCode::BAD_REQUEST,
        FailureKind::Internal | FailureKind::Cancelled => StatusCode::INTERNAL_SERVER_ERROR,
    };
    error_response(status, &failure.message)
}

/// Queues the request and waits for the engine to accept it, which decides the status code.
async fn start(app: &App, mut request: Request) -> Result<mpsc::Receiver<Event>, Failure> {
    app.clients.normalise(&mut request.messages)?;
    let mut received = app.client.submit(request)?;
    match received.recv().await {
        Some(engine::Event::Started) => Ok(relay(received, &app.clients)),
        Some(engine::Event::Failed(failure)) => Err(failure),
        _ => Err(Failure { kind: FailureKind::Internal, message: "the model thread dropped the request".into() }),
    }
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let kind = if status.is_client_error() { "invalid_request_error" } else { "server_error" };
    (status, Json(json!({ "error": { "message": message, "type": kind } }))).into_response()
}

fn unix_time() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

async fn chat_completions(State(app): State<App>, body: axum::body::Bytes) -> Response {
    // Parsed by hand so a malformed body gets an OpenAI-style error instead of axum's plain text.
    let request: ChatRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, &format!("invalid request body: {error}")),
    };
    let (stream, include_usage) = (request.stream, request.stream_options.as_ref().is_some_and(|o| o.include_usage));
    let mut received = match start(&app, request.into_request()).await {
        Ok(received) => received,
        Err(failure) => return failure_response(&failure),
    };
    let model: std::sync::Arc<str> = app.client.model_name().into();
    let id = format!("chatcmpl-{:x}", SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    let created = unix_time();

    if stream {
        let chunk = move |delta: Value, finish_reason: Option<&str>| {
            json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": &*model,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
            })
        };
        // A task turns worker events into SSE payloads. When the client goes away the payload channel closes,
        // the task ends, and the worker's next send fails, which stops generation.
        let (payloads, outgoing) = mpsc::channel::<String>(64);
        tokio::spawn(async move {
            let first = chunk(json!({ "role": "assistant", "content": "" }), None);
            if payloads.send(first.to_string()).await.is_err() {
                return;
            }
            loop {
                // Nothing is sent while a prompt is processed, so the client leaving is watched for directly:
                // dropping `received` is what tells the worker to stop.
                let event = tokio::select! {
                    _ = payloads.closed() => return,
                    event = received.recv() => event,
                };
                let Some(event) = event else { return };
                let (done, chunks) = match event {
                    Event::Started => (false, vec![]),
                    Event::Reasoning(text) => (false, vec![chunk(json!({ "reasoning_content": text }), None)]),
                    Event::Content(text) => (false, vec![chunk(json!({ "content": text }), None)]),
                    Event::ToolCall { index, id, name, arguments } => {
                        let call = json!({
                            "index": index, "id": id, "type": "function",
                            "function": { "name": name, "arguments": arguments },
                        });
                        (false, vec![chunk(json!({ "tool_calls": [call] }), None)])
                    }
                    Event::Done { finish, usage } => {
                        let mut last = chunk(json!({}), Some(finish.as_str()));
                        if include_usage {
                            last["usage"] = usage_json(usage);
                        }
                        (true, vec![last])
                    }
                    Event::Failed(failure) => {
                        (true, vec![json!({ "error": { "message": failure.message, "type": "server_error" } })])
                    }
                };
                for chunk in chunks {
                    if payloads.send(chunk.to_string()).await.is_err() {
                        return;
                    }
                }
                if done {
                    // The stream's terminator is the bare text, not JSON.
                    let _ = payloads.send("[DONE]".to_owned()).await;
                    return;
                }
            }
        });
        let events = ReceiverStream::new(outgoing).map(|data| Ok::<_, Infallible>(SseEvent::default().data(data)));
        return Sse::new(events).keep_alive(KeepAlive::default()).into_response();
    }

    let (mut reasoning, mut content, mut tool_calls) = (String::new(), String::new(), Vec::new());
    while let Some(event) = received.recv().await {
        match event {
            Event::Started => {}
            Event::Reasoning(text) => reasoning.push_str(&text),
            Event::Content(text) => content.push_str(&text),
            Event::ToolCall { id, name, arguments, .. } => {
                tool_calls.push(json!({ "id": id, "type": "function", "function": { "name": name, "arguments": arguments } }));
            }
            Event::Failed(failure) => return failure_response(&failure),
            Event::Done { finish, usage } => {
                let mut message = json!({ "role": "assistant", "content": content });
                if !reasoning.is_empty() {
                    message["reasoning_content"] = reasoning.into();
                }
                if !tool_calls.is_empty() {
                    message["tool_calls"] = tool_calls.into();
                }
                return Json(json!({
                    "id": id, "object": "chat.completion", "created": created, "model": &*model,
                    "choices": [{ "index": 0, "message": message, "finish_reason": finish.as_str() }],
                    "usage": usage_json(usage),
                }))
                .into_response();
            }
        }
    }
    error_response(StatusCode::INTERNAL_SERVER_ERROR, "generation ended without a result")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_responses_conversation_becomes_chat_messages() {
        let body = json!({
            "model": "m", "stream": true, "instructions": "Be brief.", "reasoning": { "effort": "low" },
            "input": [
                { "type": "message", "role": "developer", "content": [{ "type": "input_text", "text": "Rules." }] },
                { "type": "message", "role": "user", "content": [{ "type": "input_text", "text": "List files." }] },
                { "type": "reasoning", "id": "rs_1", "summary": [{ "type": "summary_text", "text": "Use ls." }] },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "Looking." }] },
                { "type": "function_call", "call_id": "call_1", "name": "shell", "arguments": "{\"cmd\":\"ls\"}" },
                { "type": "custom_tool_call", "call_id": "call_2", "name": "apply_patch", "input": "*** Begin Patch" },
                { "type": "function_call_output", "call_id": "call_1", "output": "a.txt" },
                { "type": "custom_tool_call_output", "call_id": "call_2", "output": "Done" },
                { "type": "web_search_call", "id": "ws_1" },
            ],
            "tools": [
                { "type": "function", "name": "shell", "description": "Run.", "parameters": { "type": "object" } },
                { "type": "custom", "name": "apply_patch", "description": "Patch." },
                { "type": "web_search" },
            ],
        });
        let (request, custom) = chat_from_responses(&body).unwrap();
        assert_eq!(custom, ["apply_patch"]);
        assert!(request.stream);
        assert_eq!(request.reasoning_effort, Some(json!("low")));
        let roles: Vec<&str> = request.messages.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["system", "system", "user", "assistant", "tool", "tool"]);
        // The model's reasoning, text and two calls are one turn again.
        let turn = &request.messages[3];
        assert_eq!((&turn["reasoning_content"], &turn["content"]), (&json!("Use ls."), &json!("Looking.")));
        assert_eq!(turn["tool_calls"][0]["function"], json!({ "name": "shell", "arguments": "{\"cmd\":\"ls\"}" }));
        assert_eq!(turn["tool_calls"][1]["function"]["arguments"], json!("{\"input\":\"*** Begin Patch\"}"));
        assert_eq!(request.messages[4], json!({ "role": "tool", "tool_call_id": "call_1", "content": "a.txt" }));
        let tools = request.tools.unwrap();
        assert_eq!(tools.as_array().unwrap().len(), 2);
        assert_eq!(tools[1]["function"]["parameters"]["required"], json!(["input"]));
        // Images are refused, as in the chat endpoint.
        let image = json!({ "input": [{ "role": "user", "content": [{ "type": "input_image", "image_url": "x" }] }] });
        assert!(chat_from_responses(&image).is_err());
    }

    #[test]
    fn a_reply_streams_as_responses_items() {
        let mut items = ResponseItems { id: "resp_1".into(), custom: vec!["apply_patch".into()], output: Vec::new(), open: None };
        let kinds = |events: &[Value]| -> Vec<String> { events.iter().map(|e| e["type"].as_str().unwrap().to_owned()).collect() };
        let events = items.take(Event::Reasoning("Think".into()));
        assert_eq!(kinds(&events), ["response.output_item.added", "response.reasoning_summary_text.delta"]);
        // Text ends the reasoning item and starts a message.
        let events = items.take(Event::Content("Hi".into()));
        assert_eq!(kinds(&events), ["response.output_item.done", "response.output_item.added", "response.output_text.delta"]);
        assert_eq!(events[0]["item"]["summary"][0]["text"], "Think");
        assert_eq!(kinds(&items.take(Event::Content(" there".into()))), ["response.output_text.delta"]);
        let call = |name: &str, arguments: &str| Event::ToolCall { index: 0, id: "call_1".into(), name: name.into(), arguments: arguments.into() };
        let events = items.take(call("shell", "{\"cmd\":\"ls\"}"));
        assert_eq!(kinds(&events), ["response.output_item.done", "response.output_item.done"]);
        assert_eq!(events[0]["item"]["content"][0]["text"], "Hi there");
        assert_eq!(events[1]["item"]["type"], "function_call");
        // A free-form tool gets its plain text back.
        let events = items.take(call("apply_patch", "{\"input\":\"*** Begin Patch\"}"));
        assert_eq!((&events[0]["item"]["type"], &events[0]["item"]["input"]), (&json!("custom_tool_call"), &json!("*** Begin Patch")));
        let usage = Usage { prompt_tokens: 10, cached_tokens: 4, completion_tokens: 6, ..Usage::default() };
        let events = items.take(Event::Done { finish: rustling::engine::Finish::Stop, usage });
        assert_eq!(kinds(&events), ["response.completed"]);
        let response = &events[0]["response"];
        assert_eq!(response["output"].as_array().unwrap().len(), 4);
        assert_eq!(response["usage"]["input_tokens_details"]["cached_tokens"], 4);
    }
}
