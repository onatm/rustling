//! `chat`: a coding agent built on manifatura, with the model in the same process.
//!
//! The engine runs on its own thread; [`LocalProvider`] hands it the agent's requests. The agent has three
//! tools (`read`, `grep`, `glob`), each asking before it runs, a system prompt built from the `AGENTS.md` files of the working directory and
//! the tools' own guidelines, and sessions kept as JSON lines, so a conversation can be resumed later. The screen
//! is in [`ui`].

pub(crate) mod mcp;
mod provider;
pub(crate) mod reasoning;
pub(crate) mod ui;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use manifatura::kit::context::{ContextFiles, SystemPrompt};
use manifatura::kit::{self, Answer, Summarizer};
use manifatura::store::JsonlStore;
use manifatura::{Agent, Hooks, Message, Runtime, Session, SessionId, SessionMeta, Store, ToolCall, hook};
use rustling::engine::{Client, EngineOptions, Usage};
use tokio::sync::{mpsc, oneshot, watch};

use provider::LocalProvider;
use ui::Ask;
use ui::logs::Logs;

/// Who the agent is. The tools' guidelines, the `AGENTS.md` files, the directory and the date follow it.
const BASE: &str = include_str!("system.md");

/// The tools the agent registers, from manifatura's built-ins. Each asks before it runs, unless `--yes`.
const TOOLS: &[&str] = &["read", "grep", "glob"];

pub(crate) struct ChatOptions {
    /// Text added to the system prompt, after the context files.
    pub(crate) system: Option<String>,
    /// Open the most recent session for this directory.
    pub(crate) resume_latest: bool,
    /// Open this session; `Some(None)` lists them instead.
    pub(crate) resume: Option<Option<String>>,
    /// Run every tool without asking.
    pub(crate) yes: bool,
    /// How much the model reasons to begin with; `/effort` changes it.
    pub(crate) effort: reasoning::Effort,
    pub(crate) max_tokens: usize,
    /// Keep the log, behind a button on the screen.
    pub(crate) logs: bool,
}

pub(crate) fn chat(model_dir: &Path, engine: EngineOptions, options: ChatOptions, mut logs: Logs) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let store = Arc::new(JsonlStore::new(sessions_dir(&cwd)?)?);
    let runtime = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    if options.resume == Some(None) {
        let sessions = runtime.block_on(recent_sessions(store.as_ref()))?;
        if sessions.is_empty() {
            println!("No sessions in this directory yet.");
        }
        for session in sessions {
            println!("{}  {}", session.id.as_str(), ago(session.updated));
        }
        return Ok(());
    }
    use std::io::IsTerminal;
    anyhow::ensure!(
        std::io::stdin().is_terminal() && std::io::stdout().is_terminal(),
        "rustling chat needs a terminal; for one reply, use `rustling generate`, or serve the model with `rustling serve`"
    );
    let mcp = match mcp::config_dir() {
        Some(dir) => mcp::configured(&dir.join("mcp.toml"))?,
        None => Vec::new(),
    };
    let start = ui::Start { model_dir: model_dir.to_owned(), engine, cwd, store: store.clone(), options, mcp };
    let (engine, session, chatted) = runtime.block_on(ui::run(start, &mut logs));
    // Only a session that was used is kept, and worth coming back to.
    let session = session.filter(|id| {
        runtime.block_on(store.list()).is_ok_and(|sessions| sessions.iter().any(|s| s.id == *id))
    });
    // The runtime holds the agent and with it the engine's last clients; once it is gone the engine saves its
    // cache and ends.
    drop(runtime);
    if let Some(id) = session {
        // The screen may leave the cursor mid-line.
        println!("\r\x1b[2KSession {} · continue it with: rustling chat --resume {}", id.as_str(), id.as_str());
    }
    let shut = engine.map(|engine| engine.shutdown()).unwrap_or(Ok(()));
    logs.print_rest();
    chatted.and(shut)
}

/// The agent and its session, set up once the model is loaded.
pub(crate) struct Prepared {
    pub(crate) runtime: Runtime,
    pub(crate) agent: Agent,
    pub(crate) session: Session,
    /// The session's conversation so far, when it was resumed.
    pub(crate) history: Vec<Message>,
    pub(crate) store: Arc<JsonlStore>,
    pub(crate) summarizer: Arc<Summarizer>,
    /// The effort replies are asked for, which `/effort` changes.
    pub(crate) reasoning: reasoning::Reasoning,
    /// The instruction files the prompt includes.
    pub(crate) files: Vec<PathBuf>,
    /// The MCP servers that started, with their tools' names and whether those are offered directly.
    pub(crate) servers: Vec<(String, Vec<String>, bool)>,
    /// What went wrong starting the others.
    pub(crate) problems: Vec<String>,
}

/// Builds the agent over the engine: its tools, instructions and hooks, the runtime, and the session to open.
pub(crate) async fn prepare(
    client: Client,
    cwd: &Path,
    store: Arc<JsonlStore>,
    options: &ChatOptions,
    asks: mpsc::UnboundedSender<Ask>,
    stats: watch::Sender<Option<Usage>>,
    started: Vec<Result<mcp::Started, String>>,
) -> Result<Prepared> {
    let window = client.context();
    let env = manifatura::tools::Local::new(cwd)?;
    let mut tools = manifatura::tools::by_name(TOOLS).map_err(anyhow::Error::msg)?;
    // The tools that ask: the built-ins, and those of servers configured with `approve`.
    let mut asking: Vec<String> = TOOLS.iter().map(|&name| name.to_owned()).collect();
    // MCP tools come after the built-ins. A server's tools are deferred unless it is `eager`: the model finds them
    // with find_tools, so their schemas stay out of the prompt until it needs them.
    let (mut servers, mut problems) = (Vec::new(), Vec::new());
    for started in started {
        match started {
            Ok(started) => {
                let config = started.server.config();
                let names: Vec<String> = started.tools.iter().map(|tool| tool.name().to_owned()).collect();
                if config.approve {
                    asking.extend(names.iter().cloned());
                }
                servers.push((started.name, names, config.eager));
                tools.extend(started.tools);
            }
            Err(problem) => problems.push(problem),
        }
    }
    let deferred = tools.iter().any(|tool| tool.is_deferred());
    let mut files = ContextFiles::new().max_bytes(16 * 1024);
    if let Some(dir) = mcp::config_dir() {
        files = files.user_file(dir.join("AGENTS.md"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        files = files.home(home);
    }
    let files = files.load(&env).await;
    let mut prompt = SystemPrompt::new(BASE.trim()).tools(&tools).context_files(&files);
    if let Some(system) = &options.system {
        prompt = prompt.section(system);
    }
    let instructions = prompt.cwd(cwd).date(today()).build();

    let summarizer = Arc::new(Summarizer::default().keep_recent(window as u64 / 3).max_tokens(2048));
    let mut agent = Agent::new(instructions).name("rustling").tools(tools);
    // The catalog goes first, so the hooks after it see the tool a run_tool call stands for.
    if deferred {
        agent = agent.hooks(kit::catalog());
    }
    agent = agent.hooks(kit::repair()).hooks(cap_results(window));
    if !options.yes {
        // "Always" allows the tool for the rest of the session. A turn interrupted while asking drops the question.
        agent = agent.hooks(kit::approve(asking, move |call: ToolCall| {
            let asks = asks.clone();
            async move {
                let (reply, answer) = oneshot::channel();
                if asks.send(Ask { call, reply }).is_err() {
                    return Answer::Deny(Some("nobody is there to ask".into()));
                }
                answer.await.unwrap_or(Answer::Deny(None))
            }
        }));
    }
    let agent = agent
        .hooks(kit::compaction::trim_results(window as u64 / 2, 2_000))
        .hooks(kit::compaction::on_overflow(summarizer.clone()))
        .hooks(kit::system_delta());

    let reasoning = reasoning::Reasoning::new(options.effort.clone());
    let provider = LocalProvider::new(client, reasoning.clone(), options.max_tokens, stats);
    let runtime = Runtime::builder().provider(provider).store(store.clone()).env(env).build();

    let known = store.list().await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let id = match (&options.resume, options.resume_latest) {
        (Some(Some(id)), _) => {
            let id = SessionId::from(id.as_str());
            if !known.iter().any(|s| s.id == id) {
                bail!("no session {} in this directory; `rustling chat --resume` lists them", id.as_str());
            }
            id
        }
        (_, true) => known.iter().max_by_key(|s| s.updated).map_or_else(SessionId::new, |latest| latest.id.clone()),
        _ => SessionId::new(),
    };
    // Reading a session writes the header of a new one; a new session is only written once it is used, so that
    // starting a chat and quitting leaves nothing behind.
    let existed = known.iter().any(|s| s.id == id);
    let session = runtime.session(&agent, id);
    let history = match existed {
        true => session.history().await?,
        false => Vec::new(),
    };
    let files = files.into_iter().map(|file| file.path).collect();
    Ok(Prepared { runtime, agent, session, history, store, summarizer, reasoning, files, servers, problems })
}

/// Where this directory's sessions are kept: `rustling/sessions/<directory>` under `$XDG_DATA_HOME`, or under
/// `~/.local/share`.
fn sessions_dir(cwd: &Path) -> Result<PathBuf> {
    let data = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).filter(|path| path.is_absolute());
    let data = match data {
        Some(data) => data,
        None => PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?).join(".local/share"),
    };
    Ok(data.join("rustling/sessions").join(slug(cwd)))
}

/// A directory name for a path: `/Users/me/src/app` is `Users-me-src-app`.
fn slug(path: &Path) -> String {
    let mut slug = String::new();
    for c in path.to_string_lossy().chars() {
        if c.is_ascii_alphanumeric() || c == '.' || c == '_' {
            slug.push(c);
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    slug.trim_matches('-').to_owned()
}

/// The same recency ordering for the picker, completions and the command-line listing.
async fn recent_sessions(store: &dyn Store) -> Result<Vec<SessionMeta>> {
    let mut sessions = store.list().await?;
    sessions.sort_by_key(|s| std::cmp::Reverse(s.updated));
    Ok(sessions)
}

/// How long ago a time in milliseconds since the epoch was, roughly.
fn ago(ms: u64) -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64);
    let minutes = now.saturating_sub(ms) / 60_000;
    match minutes {
        0 => "just now".into(),
        1..=59 => format!("{minutes} min ago"),
        60..=1439 => format!("{} h ago", minutes / 60),
        _ => format!("{} days ago", minutes / 1440),
    }
}

/// Today's local date, for the system prompt. Day granularity keeps the prompt the same all day, so its
/// processed state stays reusable.
fn today() -> String {
    chrono::Local::now().format("%Y-%m-%d").to_string()
}

/// Keeps a tool result to about a quarter of the window, its head and tail, so one call cannot fill the
/// context. The built-in tools allow 50 KB, which is most of a 16k-token window.
fn cap_results(window: usize) -> Hooks {
    let max = (window * 7 / 8).max(4_000); // about a quarter of the window, at 3.5 characters a token
    Hooks::new().on(move |mut e: hook::ToolResult, _cx| async move {
        if let Some(capped) = cap(&e.result.output, max) {
            e.result.output = capped;
        }
        e
    })
}

/// `output` cut to its first two thirds and last third of `max` bytes, with a note between; `None` if it fits.
fn cap(output: &str, max: usize) -> Option<String> {
    if output.len() <= max {
        return None;
    }
    let head = (0..=max * 2 / 3).rev().find(|&i| output.is_char_boundary(i)).unwrap_or(0);
    let tail = (output.len() - max / 3..output.len()).find(|&i| output.is_char_boundary(i)).unwrap_or(output.len());
    Some(format!(
        "{}\n[… {} characters left out; read less at a time, e.g. with offset and limit …]\n{}",
        &output[..head],
        tail - head,
        &output[tail..]
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_directory_becomes_a_slug() {
        assert_eq!(slug(Path::new("/Users/me/src/my app")), "Users-me-src-my-app");
    }

    #[test]
    fn todays_date_is_a_calendar_date() {
        let date = today();
        assert_eq!(date.len(), 10);
        assert!(date.starts_with("20"), "{date}");
    }

    #[test]
    fn long_results_keep_their_head_and_tail() {
        let output = format!("{}{}", "é".repeat(2_500), "z".repeat(5_000));
        let text = cap(&output, 4_000).unwrap();
        assert!(text.starts_with("éé") && text.ends_with("zzz") && text.contains("characters left out"), "{text}");
        assert!(text.len() < 4_200);
        assert_eq!(cap("short", 4_000), None);
    }
}
