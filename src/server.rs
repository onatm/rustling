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
use rustling::cache::PromptCache;
use rustling::checkpoint::Checkpoint;
use rustling::config::Config;
use rustling::draft::Draft;
use rustling::model::{Backend, Model, Options};
use rustling::prompt::{Prompter, SegmentKind, parse_reply, reply_message};
use rustling::sampler::{Sampler, SamplerConfig};
use rustling::session::{Limits, Session, Stop};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

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
        eprintln!("listening on http://{address}/v1");
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

/// The text of a Responses content value: a string, or a list of parts of which the text ones are joined.
fn responses_text(content: &Value) -> Result<String, String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => parts
            .iter()
            .map(|part| match part["type"].as_str() {
                Some("input_text" | "output_text" | "text" | "summary_text" | "reasoning_text") => {
                    Ok(part["text"].as_str().unwrap_or_default())
                }
                other => Err(format!("only text content is supported, not {}", other.unwrap_or("untyped parts"))),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|texts| texts.concat()),
        Value::Null => Ok(String::new()),
        other => Ok(other.to_string()),
    }
}

/// A Responses API request as the chat request it amounts to, and the names of its free-form ("custom") tools.
///
/// The input is a flat list of items. What the model produced in one turn (reasoning, text, tool calls) comes
/// back as separate items in a row, and is put together again as one assistant message, so the prompt is the
/// one the model saw when it generated them.
fn chat_from_responses(body: &Value) -> Result<(ChatRequest, Vec<String>), String> {
    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = body["instructions"].as_str().filter(|text| !text.is_empty()) {
        messages.push(json!({ "role": "system", "content": instructions }));
    }
    // The assistant message being put together, if the last items were the model's.
    let mut turn: Option<Value> = None;
    let close = |turn: &mut Option<Value>, messages: &mut Vec<Value>| messages.extend(turn.take());
    let items = match &body["input"] {
        Value::String(text) => vec![json!({ "type": "message", "role": "user", "content": text })],
        Value::Array(items) => items.clone(),
        _ => return Err("`input` must be a string or a list of items".to_owned()),
    };
    for item in &items {
        let kind = item["type"].as_str().unwrap_or("message");
        let call = |name: &Value, arguments: Value| {
            json!({ "id": item["call_id"], "type": "function", "function": { "name": name, "arguments": arguments } })
        };
        match kind {
            "message" if item["role"] != "assistant" => {
                close(&mut turn, &mut messages);
                let role = match item["role"].as_str() {
                    Some("developer" | "system") => "system",
                    _ => "user",
                };
                messages.push(json!({ "role": role, "content": responses_text(&item["content"])? }));
            }
            "function_call_output" | "custom_tool_call_output" => {
                close(&mut turn, &mut messages);
                let output = responses_text(&item["output"])?;
                messages.push(json!({ "role": "tool", "tool_call_id": item["call_id"], "content": output }));
            }
            "message" | "reasoning" | "function_call" | "custom_tool_call" => {
                let message = turn.get_or_insert_with(|| json!({ "role": "assistant", "content": "" }));
                match kind {
                    "message" => {
                        let text = format!("{}{}", message["content"].as_str().unwrap_or_default(), responses_text(&item["content"])?);
                        message["content"] = text.into();
                    }
                    "reasoning" => {
                        let source = if item["content"].is_array() { &item["content"] } else { &item["summary"] };
                        let text = responses_text(source)?;
                        if !text.is_empty() {
                            message["reasoning_content"] = text.into();
                        }
                    }
                    _ => {
                        // Arguments go on as the JSON text the chat API carries them in.
                        let arguments = match kind {
                            "function_call" => item["arguments"].clone(),
                            _ => json!({ "input": item["input"] }).to_string().into(),
                        };
                        let calls = message.as_object_mut().expect("an object").entry("tool_calls").or_insert_with(|| json!([]));
                        calls.as_array_mut().expect("a list").push(call(&item["name"], arguments));
                    }
                }
            }
            // Items a hosted model would understand and this one has no use for.
            _ => {}
        }
    }
    close(&mut turn, &mut messages);

    // Function tools are declared flat here and nested in the chat API. A free-form tool takes one string.
    let mut custom = Vec::new();
    let mut tools = Vec::new();
    for tool in body["tools"].as_array().map(Vec::as_slice).unwrap_or_default() {
        match tool["type"].as_str() {
            Some("function") => tools.push(json!({
                "type": "function",
                "function": { "name": tool["name"], "description": tool["description"], "parameters": tool["parameters"] },
            })),
            Some("custom") => {
                custom.extend(tool["name"].as_str().map(str::to_owned));
                tools.push(json!({
                    "type": "function",
                    "function": {
                        "name": tool["name"], "description": tool["description"],
                        "parameters": {
                            "type": "object", "required": ["input"],
                            "properties": { "input": { "type": "string", "description": "The tool's input, as plain text." } },
                        },
                    },
                }));
            }
            // Hosted tools (web search and the like) are not available here.
            _ => {}
        }
    }
    let number = |name: &str| body[name].as_f64().map(|value| value as f32);
    let request = ChatRequest {
        messages,
        tools: (!tools.is_empty()).then(|| Value::Array(tools)),
        stream: body["stream"].as_bool().unwrap_or(false),
        stream_options: None,
        max_tokens: body["max_output_tokens"].as_u64().map(|tokens| tokens as usize),
        max_completion_tokens: None,
        temperature: number("temperature"),
        top_p: number("top_p"),
        top_k: None,
        seed: None,
        reasoning_effort: Some(body["reasoning"]["effort"].clone()).filter(|effort| !effort.is_null()),
        stop: None,
    };
    Ok((request, custom))
}

/// Turns the worker's events for one reply into Responses API output items and stream events.
struct ResponseItems {
    id: String,
    custom: Vec<String>,
    /// Finished items, in order.
    output: Vec<Value>,
    /// The reasoning or message item being streamed: its kind and text so far.
    open: Option<(&'static str, String)>,
}

impl ResponseItems {
    fn item_id(&self, prefix: &str) -> String {
        format!("{prefix}_{}_{}", self.id, self.output.len())
    }

    fn item(&self, kind: &str, text: &str) -> Value {
        match kind {
            "reasoning" => json!({
                "type": "reasoning", "id": self.item_id("rs"),
                "summary": [{ "type": "summary_text", "text": text }],
            }),
            _ => json!({
                "type": "message", "role": "assistant", "id": self.item_id("msg"), "status": "completed",
                "content": [{ "type": "output_text", "text": text, "annotations": [] }],
            }),
        }
    }

    /// Ends the item being streamed, if there is one.
    fn close(&mut self, events: &mut Vec<Value>) {
        if let Some((kind, text)) = self.open.take() {
            let item = self.item(kind, &text);
            events.push(json!({ "type": "response.output_item.done", "output_index": self.output.len(), "item": item }));
            self.output.push(item);
        }
    }

    /// Adds `delta` to the item of this kind, starting it (and ending one of the other kind) as needed.
    fn text(&mut self, kind: &'static str, delta: String, events: &mut Vec<Value>) {
        if self.open.as_ref().is_some_and(|(open, _)| *open != kind) {
            self.close(events);
        }
        let index = self.output.len();
        if self.open.is_none() {
            events.push(json!({ "type": "response.output_item.added", "output_index": index, "item": self.item(kind, "") }));
            self.open = Some((kind, String::new()));
        }
        let (event, id) = match kind {
            "reasoning" => ("response.reasoning_summary_text.delta", self.item_id("rs")),
            _ => ("response.output_text.delta", self.item_id("msg")),
        };
        events.push(json!({
            "type": event, "item_id": id, "output_index": index, "content_index": 0, "summary_index": 0, "delta": delta,
        }));
        self.open.as_mut().expect("opened above").1.push_str(&delta);
    }

    /// The stream events, or the final item, that a worker event amounts to.
    fn take(&mut self, event: Event) -> Vec<Value> {
        let mut events = Vec::new();
        match event {
            Event::Started => {}
            Event::Reasoning(text) => self.text("reasoning", text, &mut events),
            Event::Content(text) => self.text("message", text, &mut events),
            Event::ToolCall { id, name, arguments, .. } => {
                self.close(&mut events);
                let item = match self.custom.contains(&name) {
                    // A free-form tool's input was wrapped in an object for the model; unwrap it.
                    true => {
                        let wrapped: Value = serde_json::from_str(&arguments).unwrap_or_default();
                        let input = wrapped["input"].as_str().map_or_else(|| arguments.clone(), str::to_owned);
                        json!({ "type": "custom_tool_call", "id": self.item_id("ctc"), "status": "completed", "call_id": id, "name": name, "input": input })
                    }
                    false => json!({
                        "type": "function_call", "id": self.item_id("fc"), "status": "completed",
                        "call_id": id, "name": name, "arguments": arguments,
                    }),
                };
                events.push(json!({ "type": "response.output_item.done", "output_index": self.output.len(), "item": item }));
                self.output.push(item);
            }
            Event::Done { usage, .. } => {
                self.close(&mut events);
                events.push(json!({ "type": "response.completed", "response": self.response("completed", Some(usage)) }));
            }
            Event::Failed { message, .. } => {
                let mut response = self.response("failed", None);
                response["error"] = json!({ "code": "server_error", "message": message });
                events.push(json!({ "type": "response.failed", "response": response }));
            }
        }
        events
    }

    fn response(&self, status: &str, usage: Option<Usage>) -> Value {
        let usage = usage.map(|usage| {
            json!({
                "input_tokens": usage.prompt_tokens,
                "input_tokens_details": { "cached_tokens": usage.cached_tokens },
                "output_tokens": usage.completion_tokens,
                "output_tokens_details": { "reasoning_tokens": 0 },
                "total_tokens": usage.prompt_tokens + usage.completion_tokens,
            })
        });
        json!({
            "id": self.id, "object": "response", "created_at": unix_time(), "status": status,
            "output": self.output, "usage": usage,
        })
    }
}

/// `POST /v1/responses`: OpenAI's Responses API, as far as a coding agent uses it. Each request carries the
/// whole conversation (nothing is stored between requests), tools are functions or free-form, and the reply
/// streams as reasoning, text and tool-call items.
async fn responses(State(App { jobs, .. }): State<App>, body: axum::body::Bytes) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(body) => body,
        Err(error) => return error_response(StatusCode::BAD_REQUEST, &format!("invalid request body: {error}")),
    };
    let (request, custom) = match chat_from_responses(&body) {
        Ok(converted) => converted,
        Err(message) => return error_response(StatusCode::BAD_REQUEST, &message),
    };
    let stream = request.stream;
    let (events, mut received) = mpsc::channel(64);
    if jobs.send(Job { request, events }).is_err() {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "the model thread has stopped");
    }
    match received.recv().await {
        Some(Event::Started) => {}
        Some(Event::Failed { status, message }) => return error_response(status, &message),
        _ => return error_response(StatusCode::INTERNAL_SERVER_ERROR, "the model thread dropped the request"),
    }
    let id = format!("resp_{:x}", SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    let mut items = ResponseItems { id, custom, output: Vec::new(), open: None };

    if stream {
        let (payloads, outgoing) = mpsc::channel::<Value>(64);
        tokio::spawn(async move {
            let created = json!({ "type": "response.created", "response": items.response("in_progress", None) });
            if payloads.send(created).await.is_err() {
                return;
            }
            loop {
                // As in the chat endpoint: the client leaving while a prompt is processed must reach the worker.
                let event = tokio::select! {
                    _ = payloads.closed() => return,
                    event = received.recv() => event,
                };
                let Some(event) = event else { return };
                let last = matches!(event, Event::Done { .. } | Event::Failed { .. });
                for payload in items.take(event) {
                    if payloads.send(payload).await.is_err() {
                        return;
                    }
                }
                if last {
                    return;
                }
            }
        });
        let events = ReceiverStream::new(outgoing).map(|payload| {
            let kind = payload["type"].as_str().unwrap_or_default().to_owned();
            Ok::<_, Infallible>(SseEvent::default().event(kind).data(payload.to_string()))
        });
        return Sse::new(events).keep_alive(KeepAlive::default()).into_response();
    }

    while let Some(event) = received.recv().await {
        match event {
            Event::Failed { status, message } => return error_response(status, &message),
            Event::Done { usage, .. } => {
                items.take(Event::Done { finish_reason: "stop", usage });
                return Json(items.response("completed", Some(usage))).into_response();
            }
            event => drop(items.take(event)),
        }
    }
    error_response(StatusCode::INTERNAL_SERVER_ERROR, "generation ended without a result")
}

/// Loads the model, reports readiness, then serves jobs one at a time until every sender is gone.
fn worker(model_dir: &Path, options: ServerOptions, queue: std_mpsc::Receiver<Job>, ready: std_mpsc::Sender<Result<()>>) {
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
        Some(_) => match rustling::checkpoint::companion_dir(model_dir).and_then(|dir| Checkpoint::open_file(&dir.join("mtp.safetensors"))) {
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
    // Bring the weights into memory and run a token through the model, so the first request is not the slow one.
    let started = std::time::Instant::now();
    let bytes = checkpoint.read_through();
    if let Err(error) = model.warm_up() {
        return drop(ready.send(Err(error.context("warming up"))));
    }
    let backend = if model.uses_gpu() { "Metal for large batches" } else { "CPU" };
    let drafting = if draft.is_some() { ", with the draft model" } else { "" };
    eprintln!(
        "model loaded ({backend}{drafting}); read {:.1} GB and warmed up in {:.1}s",
        bytes as f64 / 1e9,
        started.elapsed().as_secs_f32()
    );
    let _ = ready.send(Ok(()));

    let mut session = match &draft {
        Some(draft) => Session::new(&model).with_window(options.context).with_draft(draft),
        None => Session::new(&model).with_window(options.context),
    };
    // What snapshots depend on: one made with another model, pass count or arithmetic must not be reused.
    // The leading number changes with the layout of a saved state.
    let fingerprint = format!(
        "state 2|{}|{} bytes|{} visits|{:?}|{}|draft {}",
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

struct Failure {
    status: StatusCode,
    message: String,
}

impl Failure {
    fn bad_request(message: impl Into<String>) -> Self {
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
struct Memory {
    replies: VecDeque<(String, String)>,
}

impl Memory {
    /// What identifies a reply: its text and tool calls, which clients do send back.
    fn key(message: &Value) -> String {
        let calls: Vec<Value> = message["tool_calls"]
            .as_array()
            .map(|calls| calls.iter().map(|c| json!([c["function"]["name"], c["function"]["arguments"]])).collect())
            .unwrap_or_default();
        json!([message["content"].as_str().unwrap_or_default(), calls]).to_string()
    }

    fn remember(&mut self, message: &Value) {
        if let Some(reasoning) = message["reasoning_content"].as_str() {
            self.replies.push_back((Self::key(message), reasoning.to_owned()));
            if self.replies.len() > REMEMBERED_REPLIES {
                self.replies.pop_front();
            }
        }
    }

    fn recall(&self, message: &Value) -> Option<&str> {
        let key = Self::key(message);
        self.replies.iter().rev().find(|(k, _)| *k == key).map(|(_, reasoning)| reasoning.as_str())
    }
}

/// Brings OpenAI-style messages to what the chat template takes.
fn normalise(messages: &mut [Value], memory: &Memory) -> Result<(), Failure> {
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
fn rate(tokens: usize, seconds: f32) -> f32 {
    tokens as f32 / seconds.max(1e-3)
}

fn handle(
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
    eprintln!(
        "[{number}] reply: {} tokens in {seconds:.1}s ({:.1} tokens/s), {finish_reason}{drafting}",
        generated.len(),
        rate(generated.len(), seconds)
    );
    let usage = Usage { prompt_tokens: ids.len(), cached_tokens, completion_tokens: generated.len() };
    send(Event::Done { finish_reason, usage })?;
    Ok(())
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
