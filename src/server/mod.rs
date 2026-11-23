//! An OpenAI-compatible HTTP server: `POST /v1/chat/completions` (streaming or not), `GET /v1/models`,
//! `GET /health`.
//!
//! One worker thread owns the model and a single [`Session`]. Requests queue for it. A request whose prompt
//! extends what the session last saw, as the next turn of the same conversation does, only pays for its new
//! tokens; anything else starts over.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rustling::generate::cache::PromptCache;
use rustling::weights::checkpoint::Checkpoint;
use rustling::config::Config;
use rustling::generate::draft::Draft;
use rustling::model::{Model, Options};
use rustling::prompt::{Prompter, SegmentKind, parse_reply, reply_message};
use rustling::generate::sampler::{Sampler, SamplerConfig};
use rustling::generate::session::{Limits, Session, Stop};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

mod responses;
mod worker;

use responses::*;
use worker::*;

/// The name the API reports for the model: the checkpoint's own, lower-cased, with the pass count when one was
/// chosen. A Hugging Face cache directory is `models--<owner>--<name>/snapshots/<revision>`.
fn model_name(model_dir: &Path, passes: Option<usize>) -> String {
    let from_cache = model_dir.ancestors().filter_map(|dir| dir.file_name()?.to_str()).find_map(|name| {
        name.strip_prefix("models--").and_then(|rest| rest.rsplit("--").next())
    });
    // A GGUF file is named after itself, without the extension.
    let gguf = model_dir.extension().is_some_and(|extension| extension == "gguf");
    let own = if gguf { model_dir.file_stem() } else { model_dir.file_name() }.and_then(|name| name.to_str());
    let name = if gguf { own } else { from_cache.or(own) }.unwrap_or("model").to_lowercase();
    match passes {
        Some(passes) => format!("{name}-{passes}-passes"),
        None => name,
    }
}

#[derive(Clone)]
struct App {
    jobs: std_mpsc::Sender<Job>,
    model: std::sync::Arc<str>,
}

const GIB: f64 = (1u64 << 30) as f64;

/// How many of its own replies the server remembers the reasoning of; see [`Memory`].
const REMEMBERED_REPLIES: usize = 64;

pub struct ServerOptions {
    pub host: String,
    pub port: u16,
    pub model: Options,
    /// Sampling when a request does not say.
    pub sampling: SamplerConfig,
    /// The reply length limit when a request does not say.
    pub max_tokens: usize,
    pub stop_on_loop: bool,
    /// Generate with the draft model, proposing only runs it is at least this confident of.
    pub draft_confidence: Option<f32>,
    /// Guess at coming tokens by copying from the context.
    pub copying: bool,
    /// Never let a reply open a reasoning block, whatever effort the request asks for.
    pub no_reasoning: bool,
    /// Overrides how many times the looped layers run, for a variable-loop checkpoint.
    pub passes: Option<usize>,
    /// The context window in tokens, prompt and reply together.
    pub context: usize,
    /// How much memory processed prompts may take up, for reuse by later requests.
    pub cache_bytes: usize,
    /// Where to keep the processed prompts between runs.
    pub cache_dir: Option<PathBuf>,
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

/// How many requests are queued for the model and not yet taken up.
static WAITING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct Job {
    request: ChatRequest,
    events: mpsc::Sender<Event>,
}

/// What the worker reports back while handling a request.
enum Event {
    Started,
    Reasoning(String),
    Content(String),
    ToolCall { index: usize, id: String, name: String, arguments: String },
    Done { finish_reason: &'static str, usage: Usage },
    Failed { status: StatusCode, message: String },
}

#[derive(Clone, Copy)]
struct Usage {
    prompt_tokens: usize,
    /// Prompt tokens whose state was already cached from the previous request.
    cached_tokens: usize,
    completion_tokens: usize,
}

impl Usage {
    fn json(self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "prompt_tokens_details": { "cached_tokens": self.cached_tokens },
        })
    }
}

pub fn serve(model_dir: &Path, options: ServerOptions) -> Result<()> {
    let (jobs, queue) = std_mpsc::channel::<Job>();
    let (ready, loaded) = std_mpsc::channel::<Result<()>>();
    let address = format!("{}:{}", options.host, options.port);
    let model: std::sync::Arc<str> = model_name(model_dir, options.passes).into();
    let model_dir: PathBuf = model_dir.to_owned();
    let worker =
        std::thread::Builder::new().name("model".into()).spawn(move || worker(&model_dir, options, queue, ready))?;
    loaded.recv().context("the model thread stopped while loading")??;

    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    let served = runtime.block_on(async move {
        let app = Router::new()
            .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/responses", post(responses))
            .with_state(App { jobs, model });
        let listener = tokio::net::TcpListener::bind(&address).await.with_context(|| format!("binding {address}"))?;
        tracing::info!("listening on http://{address}/v1");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    });
    // Dropping the runtime drops the last sender to the model thread, which then saves its cache and ends.
    drop(runtime);
    worker.join().map_err(|_| anyhow!("the model thread panicked"))?;
    served
}

async fn models(State(app): State<App>) -> Json<Value> {
    Json(json!({ "object": "list", "data": [{ "id": &*app.model, "object": "model", "owned_by": "thinkingmachines" }] }))
}

fn error_response(status: StatusCode, message: &str) -> Response {
    let kind = if status.is_client_error() { "invalid_request_error" } else { "server_error" };
    (status, Json(json!({ "error": { "message": message, "type": kind } }))).into_response()
}

fn unix_time() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

async fn chat_completions(State(App { jobs, model }): State<App>, body: axum::body::Bytes) -> Response {
    // Parsed by hand so a malformed body gets an OpenAI-style error instead of axum's plain text.
    let request: ChatRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, &format!("invalid request body: {error}")),
    };
    let (stream, include_usage) = (request.stream, request.stream_options.as_ref().is_some_and(|o| o.include_usage));
    let (events, mut received) = mpsc::channel(64);
    WAITING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if jobs.send(Job { request, events }).is_err() {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "the model thread has stopped");
    }
    let id = format!("chatcmpl-{:x}", SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    let created = unix_time();

    // The first event says whether the request was accepted, which decides the status code.
    match received.recv().await {
        Some(Event::Started) => {}
        Some(Event::Failed { status, message }) => return error_response(status, &message),
        _ => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "the model thread dropped the request"),
    }

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
                    Event::Done { finish_reason, usage } => {
                        let mut last = chunk(json!({}), Some(finish_reason));
                        if include_usage {
                            last["usage"] = usage.json();
                        }
                        (true, vec![last])
                    }
                    Event::Failed { message, .. } => {
                        (true, vec![json!({ "error": { "message": message, "type": "server_error" } })])
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
            Event::Failed { status, message } => return error_response(status, &message),
            Event::Done { finish_reason, usage } => {
                let mut message = json!({ "role": "assistant", "content": content });
                if !reasoning.is_empty() {
                    message["reasoning_content"] = reasoning.into();
                }
                if !tool_calls.is_empty() {
                    message["tool_calls"] = tool_calls.into();
                }
                return Json(json!({
                    "id": id, "object": "chat.completion", "created": created, "model": &*model,
                    "choices": [{ "index": 0, "message": message, "finish_reason": finish_reason }],
                    "usage": usage.json(),
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
    fn the_reported_model_name_comes_from_the_checkpoint() {
        let cached = Path::new("/home/u/.cache/huggingface/hub/models--thinkingmachines--TML-d24-varloop-research/snapshots/657b53");
        assert_eq!(model_name(cached, Some(2)), "tml-d24-varloop-research-2-passes");
        assert_eq!(model_name(Path::new("/models/Inkling-2-Mini"), None), "inkling-2-mini");
        assert_eq!(model_name(Path::new("/models/x/Inkling-2-Mini-Q4_K_M.gguf"), None), "inkling-2-mini-q4_k_m");
    }

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
        let usage = Usage { prompt_tokens: 10, cached_tokens: 4, completion_tokens: 6 };
        let events = items.take(Event::Done { finish_reason: "stop", usage });
        assert_eq!(kinds(&events), ["response.completed"]);
        let response = &events[0]["response"];
        assert_eq!(response["output"].as_array().unwrap().len(), 4);
        assert_eq!(response["usage"]["input_tokens_details"]["cached_tokens"], 4);
    }

    #[test]
    fn remembered_reasoning_is_put_back() {
        let mut memory = Memory::default();
        memory.remember(&json!({"role": "assistant", "content": "Paris.", "reasoning_content": "Easy."}));
        let mut messages = vec![
            json!({"role": "developer", "content": "Be brief."}),
            json!({"role": "user", "content": "Capital of France?"}),
            json!({"role": "assistant", "content": "Paris."}),
            json!({"role": "assistant", "content": "Something the server never said."}),
        ];
        assert!(normalise(&mut messages, &memory).is_ok());
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[2]["reasoning_content"], "Easy.");
        assert!(messages[3].get("reasoning_content").is_none());
    }

    #[test]
    fn replies_are_told_apart_by_their_tool_calls() {
        let mut memory = Memory::default();
        let call = |city: &str| json!({"type": "function", "function": {"name": "weather", "arguments": {"city": city}}});
        memory.remember(&json!({"role": "assistant", "content": "", "reasoning_content": "Look it up.", "tool_calls": [call("Paris")]}));
        // As a client sends it back: null content, arguments as a JSON string.
        let sent = |city: &str| {
            json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": format!(r#"{{"city":"{city}"}}"#)}}
            ]})
        };
        let mut messages = vec![sent("Paris"), sent("Rome")];
        assert!(normalise(&mut messages, &memory).is_ok());
        assert_eq!(messages[0]["tool_calls"][0]["function"]["arguments"], json!({"city": "Paris"}));
        assert_eq!(messages[0]["content"], "");
        assert_eq!(messages[0]["reasoning_content"], "Look it up.");
        assert!(messages[1].get("reasoning_content").is_none());
    }

    #[test]
    fn unsupported_input_is_a_client_error() {
        let memory = Memory::default();
        let mut images = vec![json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]})];
        assert!(normalise(&mut images, &memory).is_err_and(|f| f.status == StatusCode::BAD_REQUEST));
        let mut bad_arguments = vec![json!({"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{"}}]})];
        assert!(normalise(&mut bad_arguments, &memory).is_err_and(|f| f.status == StatusCode::BAD_REQUEST));
    }
}
