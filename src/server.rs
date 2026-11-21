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
use rustling::checkpoint::Checkpoint;
use rustling::config::Config;
use rustling::draft::Draft;
use rustling::model::{Model, Options};
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
    let own = model_dir.file_name().and_then(|name| name.to_str());
    let name = from_cache.or(own).unwrap_or("model").to_lowercase();
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
    std::thread::Builder::new().name("model".into()).spawn(move || worker(&model_dir, options, queue, ready))?;
    loaded.recv().context("the model thread stopped while loading")??;

    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    runtime.block_on(async move {
        let app = Router::new()
            .route("/health", get(|| async { Json(json!({ "status": "ok" })) }))
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(App { jobs, model });
        let listener = tokio::net::TcpListener::bind(&address).await.with_context(|| format!("binding {address}"))?;
        eprintln!("listening on http://{address}/v1");
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
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
            while let Some(event) = received.recv().await {
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
        Some(_) => match Checkpoint::open_file(&model_dir.join("mtp.safetensors")) {
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
    let backend = if model.uses_gpu() { "Metal for large batches" } else { "CPU" };
    let drafting = if draft.is_some() { ", with the draft model" } else { "" };
    eprintln!("model loaded ({backend}{drafting})");
    let _ = ready.send(Ok(()));

    let mut session = match &draft {
        Some(draft) => Session::new(&model).with_draft(draft),
        None => Session::new(&model),
    };
    let mut memory = Memory::default();
    let mut calls = 0usize;
    for job in queue {
        let events = job.events.clone();
        let handled = handle(&job, &options, &config, &prompter, &mut session, &mut memory, &mut calls);
        if let Err(failure) = handled {
            // The receiver may already be gone, when the failure is the client hanging up.
            let _ = events.blocking_send(Event::Failed { status: failure.status, message: failure.message });
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

fn handle(
    job: &Job,
    options: &ServerOptions,
    config: &Config,
    prompter: &Prompter,
    session: &mut Session,
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
    let rendered = prompter
        .render(&Value::Array(messages), request.tools.as_ref(), request.reasoning_effort.as_ref())
        .map_err(|e| Failure::bad_request(format!("{e:#}")))?;
    let ids = prompter.encode(&rendered)?;
    send(Event::Started)?;
    let cached_tokens = session.set_context(&ids)?;

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
        Stop::MaxTokens | Stop::Loop => "length",
    };
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
