//! The engine: one thread that owns the model, a single [`Session`] and the prompt cache, and serves requests
//! one at a time.
//!
//! A request whose prompt extends what the session last saw, as the next turn of the same conversation does,
//! only pays for its new tokens. Other prompts start from the longest processed prefix the cache holds, in
//! memory or on disk. Both the HTTP server and an in-process agent talk to the engine through a [`Client`].
//!
//! ```no_run
//! # fn demo(model_dir: &std::path::Path, options: rustling::engine::EngineOptions) -> anyhow::Result<()> {
//! use rustling::engine::{Engine, Event, Request};
//!
//! let engine = Engine::start(model_dir, options)?;
//! let mut events = engine.client().submit(Request::new(vec![
//!     serde_json::json!({ "role": "user", "content": "What is the capital of France?" }),
//! ]))?;
//! while let Some(event) = events.blocking_recv() {
//!     if let Event::Content(text) = event {
//!         print!("{text}");
//!     }
//! }
//! engine.shutdown()
//! # }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::config::Config;
use crate::generate::cache::PromptCache;
use crate::generate::draft::Draft;
use crate::generate::sampler::{Sampler, SamplerConfig};
use crate::generate::session::{Limits, Session, Stop};
use crate::model::{Model, Options};
use crate::prompt::{Prompter, SegmentKind, parse_reply};
use crate::weights::checkpoint::Checkpoint;

mod request;
mod worker;

use request::*;
use worker::*;

const GIB: f64 = (1u64 << 30) as f64;

/// How the engine runs the model, and the defaults for requests that do not say.
pub struct EngineOptions {
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

/// One request: a conversation as the model's chat template takes it, and how to generate the reply.
///
/// Messages are JSON objects with a `role` (`system`, `user`, `assistant` or `tool`) and text `content`. An
/// assistant message carries its reasoning as `reasoning_content` and its tool calls as `tool_calls`, each with
/// an `id` and a `function` of `name` and `arguments`, the arguments an object; a `tool` message names the call it
/// answers with `tool_call_id`. Send the reasoning back with the reply it came with: the prompt then extends the
/// processed one instead of starting over. The chat form of the OpenAI API is close to this; translating it is
/// the HTTP server's job.
#[derive(Clone, Debug, Default)]
pub struct Request {
    pub messages: Vec<Value>,
    /// OpenAI-style function tools, passed to the chat template.
    pub tools: Option<Value>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
    /// A level name (`"none"` to `"max"`) or a number in `0.0..=0.99`, passed to the chat template.
    pub reasoning_effort: Option<Value>,
    /// Keep this reply from reasoning at all, as [`EngineOptions::no_reasoning`] does for every reply: the
    /// model cannot open a reasoning block. The effort is `"none"` unless the request gives one.
    pub no_reasoning: bool,
    /// Keep this reply from calling tools: the tokens that open a tool call are never chosen, so the model answers in
    /// text. The tools stay declared, so the prompt is the same as with them.
    pub no_tool_calls: bool,
    /// Text that ends the reply where it first appears; it is not sent.
    pub stop: Vec<String>,
    /// Stops the request where it is, queued, processing its prompt or generating, once set.
    pub cancel: Cancel,
}

/// A flag that stops a [`Request`]. Clone it before submitting the request and set it from anywhere.
///
/// Dropping the event channel also stops a request, but not at once: a prompt being processed is finished for
/// a client that went away, so that a retry finds it processed, unless another request is waiting. A cancelled
/// request stops at the next check, which comes every few hundred prompt tokens and at every generated token,
/// and what was processed until then stays in the session. It ends with [`FailureKind::Cancelled`].
#[derive(Clone, Debug, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

impl Request {
    pub fn new(messages: Vec<Value>) -> Self {
        Self { messages, ..Self::default() }
    }
}

/// What the engine reports while handling a request: [`Started`](Event::Started) or [`Failed`](Event::Failed)
/// first, then text as it is generated, and [`Done`](Event::Done) or [`Failed`](Event::Failed) last.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The request was accepted and its prompt is being processed.
    Started,
    Reasoning(String),
    Content(String),
    /// A tool call, sent whole once the model has closed it: the `index`th of the reply. `arguments` is JSON
    /// text. Calls have no ids; the caller gives them whatever ids its protocol needs.
    ToolCall { index: usize, name: String, arguments: String },
    Done { finish: Finish, usage: Usage },
    Failed(Failure),
}

/// Why a reply ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// The model ended its turn, or a stop sequence appeared.
    Stop,
    /// The model ended its turn after calling tools.
    ToolCalls,
    /// The reply ran into the token limit or the context window, or was cut short repeating itself.
    Length,
}

impl Finish {
    /// The OpenAI `finish_reason`.
    pub fn as_str(self) -> &'static str {
        match self {
            Finish::Stop => "stop",
            Finish::ToolCalls => "tool_calls",
            Finish::Length => "length",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    /// Prompt tokens whose state was already processed, from the session or the cache.
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    /// How long the prompt's new tokens took to process, in milliseconds.
    pub prompt_ms: u64,
    /// How long the reply took to generate, in milliseconds.
    pub reply_ms: u64,
}

/// A request that could not be served.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    pub kind: FailureKind,
    pub message: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// Something in the request is the client's to fix: unsupported content, bad arguments, a template error.
    BadRequest,
    /// The prompt leaves no room for a reply in the context window.
    ContextOverflow,
    /// The engine failed, or the model thread has stopped.
    Internal,
    /// The request's [`Cancel`] was set.
    Cancelled,
}

impl Failure {
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self { kind: FailureKind::BadRequest, message: message.into() }
    }

    pub(crate) fn cancelled() -> Self {
        Self { kind: FailureKind::Cancelled, message: "the request was cancelled".into() }
    }
}

impl From<anyhow::Error> for Failure {
    fn from(error: anyhow::Error) -> Self {
        Self { kind: FailureKind::Internal, message: format!("{error:#}") }
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Failure {}

struct Job {
    request: Request,
    events: mpsc::Sender<Event>,
}

/// The model thread. [`Client`]s send it requests; it ends, saving the prompt cache, once the engine and every
/// client are gone. Dropping the engine waits for that, so drop the clients first.
pub struct Engine {
    client: Client,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// A handle for sending requests to an [`Engine`]. Cheap to clone and shareable between threads.
#[derive(Clone)]
pub struct Client {
    jobs: std_mpsc::Sender<Job>,
    /// How many requests are queued for the model and not yet taken up.
    waiting: Arc<AtomicUsize>,
    name: Arc<str>,
    context: usize,
}

impl Engine {
    /// Loads the model on its own thread and returns once it is ready to serve.
    pub fn start(model_dir: &Path, options: EngineOptions) -> Result<Engine> {
        let (jobs, queue) = std_mpsc::channel::<Job>();
        let (ready, loaded) = std_mpsc::channel::<Result<()>>();
        let waiting = Arc::new(AtomicUsize::new(0));
        let name: Arc<str> = model_name(model_dir, options.passes).into();
        let context = options.context;
        let model_dir: PathBuf = model_dir.to_owned();
        let queued = waiting.clone();
        let thread = std::thread::Builder::new()
            .name("model".into())
            .spawn(move || worker(&model_dir, options, queue, queued, ready))?;
        loaded.recv().context("the model thread stopped while loading")??;
        Ok(Engine { client: Client { jobs, waiting, name, context }, thread: Some(thread) })
    }

    pub fn client(&self) -> Client {
        self.client.clone()
    }

    /// Waits for the model thread to finish, which it does once every [`Client`] is gone, and to save the prompt
    /// cache.
    pub fn shutdown(mut self) -> Result<()> {
        self.join()
    }

    fn join(&mut self) -> Result<()> {
        let Some(thread) = self.thread.take() else { return Ok(()) };
        // Replacing the engine's own sender leaves only the clients' ones.
        let (closed, _) = std_mpsc::channel();
        drop(std::mem::replace(&mut self.client.jobs, closed));
        thread.join().map_err(|_| anyhow!("the model thread panicked"))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if let Err(error) = self.join() {
            tracing::warn!("{error:#}");
        }
    }
}

impl Client {
    /// Queues a request. The events arrive on the returned channel; dropping it cancels the request, as a client
    /// that hangs up does.
    pub fn submit(&self, request: Request) -> Result<mpsc::Receiver<Event>, Failure> {
        let (events, received) = mpsc::channel(64);
        self.waiting.fetch_add(1, Ordering::Relaxed);
        if self.jobs.send(Job { request, events }).is_err() {
            self.waiting.fetch_sub(1, Ordering::Relaxed);
            return Err(Failure { kind: FailureKind::Internal, message: "the model thread has stopped".into() });
        }
        Ok(received)
    }

    /// The model's name, as the checkpoint gives it.
    pub fn model_name(&self) -> &str {
        &self.name
    }

    /// The context window in tokens.
    pub fn context(&self) -> usize {
        self.context
    }
}

/// The name reported for the model: the checkpoint's own, lower-cased, with the pass count when one was chosen.
/// A Hugging Face cache directory is `models--<owner>--<name>/snapshots/<revision>`.
pub fn model_name(model_dir: &Path, passes: Option<usize>) -> String {
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
}
