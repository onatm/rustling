//! Chat on the terminal's main screen. The transcript goes into native scrollback;
//! thinking and replies stream into the transcript above the pinned controls.

mod approval;
mod files;
mod layout;
mod screen;
pub(crate) mod logs;
mod text;
mod view;

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{
    self as terminal_events, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event as Input, KeyCode,
    KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::{execute, terminal};
use self::screen::ChatScreen;
use manifatura::kit::Answer;
use manifatura::skills::Skill;
use manifatura::{Abort, Event, Session, SessionId, SessionMeta};
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui_textarea::{TextArea, WrapMode};
use rustling::engine::{Engine, EngineOptions, Usage};
use tokio::sync::{broadcast, mpsc, watch};

pub(crate) use self::approval::Ask;
use self::logs::{LogLine, Logs};
use self::view::{Activity, Item, Level, View, dim, explore_message, tokens};
use super::reasoning::{Effort, LEVELS};
use super::{ChatOptions, Prepared, ago, recent_sessions};

const SPINNER: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
const KEPT_LOG_LINES: usize = 200;
/// The commands, their argument and what they do, for completion and `/help`.
const COMMANDS: &[(&str, &str, &str)] = &[
    ("/new", "", "start a new session"),
    ("/resume", "[ID]", "pick a session or open it by ID"),
    ("/sessions", "", "list this directory's sessions"),
    ("/compact", "[focus]", "summarise older messages to make room"),
    ("/effort", "[level]", "how much the model reasons: off, none … max, or 0–0.99"),
    ("/status", "", "show model, session and token statistics"),
    ("/tools", "", "show tool output from the current or last turn"),
    ("/mcp", "", "the MCP servers and their tools"),
    ("/skills", "", "the skills; start one with /name [request]"),
    ("/explore", "<question>", "hand a question about the code to the explore subagent"),
    ("/help", "", "what the keys and commands do"),
    ("/exit", "", "quit"),
];
const MAX_COMPLETIONS: usize = 8;

/// One way to complete what is typed.
#[derive(Clone, Debug)]
struct Completion {
    /// The input it becomes.
    text: String,
    label: String,
    about: String,
    /// Whether Enter on it runs it straight away (it needs no argument).
    runs: bool,
}

/// How many log lines the open panel shows.
const LOG_PANEL: usize = 10;
const LOG_BUTTON: &str = " logs ";

/// Everything the screen needs to start: the engine is loaded once it is up.
pub(crate) struct Start {
    pub(crate) model_dir: PathBuf,
    pub(crate) engine: EngineOptions,
    pub(crate) cwd: PathBuf,
    pub(crate) store: Arc<manifatura::store::JsonlStore>,
    pub(crate) options: ChatOptions,
    /// The MCP servers to start beside the model.
    pub(crate) mcp: Vec<(String, manifatura::mcp::ServerConfig)>,
}

struct App {
    view: View,
    input: TextArea<'static>,
    files: files::FilePicker,
    history: Vec<String>,
    /// The recalled entry and the draft to restore when Down reaches the end.
    browsing: Option<(usize, TextArea<'static>)>,
    log: VecDeque<LogLine>,
    /// Whether the log is kept at all (`--logs`), and whether its panel is open.
    logs: bool,
    logs_open: bool,
    /// Where the `logs` button was drawn, in terminal cells, for clicks.
    log_button: Option<Rect>,
    /// The highlighted completion, while the input is a command being typed.
    picked: usize,
    completions_hidden: bool,
    /// This directory's sessions, for completing `/resume`.
    sessions: Vec<SessionMeta>,
    /// Selected row while the session picker is open.
    resume_pick: Option<usize>,
    asks: VecDeque<Ask>,
    /// Set by a first Ctrl-C with nothing to stop; a second one quits.
    armed: bool,
    tick: usize,
    quit: bool,
    abort: Option<Abort>,
    /// A running `/explore`: its question, and what it reports.
    exploring: Option<(String, mpsc::UnboundedReceiver<Explored>)>,
    /// The agent, once the model is loaded.
    ready: Option<Prepared>,
    events: Option<broadcast::Receiver<Event>>,
    last: Option<Usage>,
    /// Tokens generated this session.
    generated: u64,
    model: String,
    window: usize,
    /// The effort until the agent is ready, which then holds it.
    effort: String,
    yes: bool,
}

/// What a one-line input could complete to: the commands and skills it begins, or after `/resume `, the sessions.
fn completions(line: &str, sessions: &[SessionMeta], skills: &[Skill]) -> Vec<Completion> {
    if !line.starts_with('/') {
        return Vec::new();
    }
    if let Some(partial) = line.strip_prefix("/effort ") {
        let partial = partial.trim().to_lowercase();
        return LEVELS
            .iter()
            .filter(|(level, _)| level.starts_with(&partial) && *level != partial)
            .map(|(level, about)| Completion {
                text: format!("/effort {level}"),
                label: (*level).to_owned(),
                about: (*about).to_owned(),
                runs: true,
            })
            .collect();
    }
    if let Some(partial) = line.strip_prefix("/resume ") {
        let partial = partial.trim();
        return sessions
            .iter()
            .filter(|s| s.id.as_str().starts_with(partial) && s.id.as_str() != partial)
            .take(MAX_COMPLETIONS)
            .map(|s| Completion {
                text: format!("/resume {}", s.id.as_str()), label: s.id.as_str().into(), about: ago(s.updated), runs: true,
            })
            .collect();
    }
    if line.contains(' ') {
        return Vec::new();
    }
    COMMANDS
        .iter()
        .filter(|(name, ..)| name.starts_with(line))
        .map(|(name, argument, about)| Completion {
            text: match argument.is_empty() {
                true => (*name).to_owned(),
                false => format!("{name} "),
            },
            label: format!("{name} {argument}").trim_end().to_owned(),
            about: (*about).to_owned(),
            runs: argument.is_empty() || *name == "/resume",
        })
        .chain(
            skills
                .iter()
                .map(|skill| (format!("/{}", skill.name), skill))
                .filter(|(name, _)| name.starts_with(line) && !is_command(name))
                .map(|(name, skill)| Completion {
                    text: format!("{name} "),
                    label: format!("{name} [request]"),
                    about: skill.description.clone(),
                    runs: false,
                }),
        )
        .collect()
}

/// Whether `name` is one of the chat's own commands, which come before a skill of the same name.
fn is_command(name: &str) -> bool {
    matches!(name, "/quit") || COMMANDS.iter().any(|(command, ..)| *command == name)
}

/// Runs the screen until the person quits. Returns the engine, so the caller can shut it down once the runtime
/// holding its clients is gone, and the session to mention on the way out.
pub(crate) async fn run(start: Start, logs: &mut Logs) -> (Option<Engine>, Option<SessionId>, Result<()>) {
    let mut term = match ChatScreen::open() {
        Ok(term) => term,
        Err(error) => {
            ratatui::restore();
            return (None, None, Err(error.into()));
        }
    };
    let enhanced = matches!(terminal::supports_keyboard_enhancement(), Ok(true));
    let mut out = std::io::stdout();
    let _ = execute!(out, EnableBracketedPaste);
    // Clicks reach the screen only with the mouse captured, which also takes the scroll wheel from the terminal:
    // only when there is a log to open.
    let mouse = start.options.logs;
    if mouse {
        let _ = execute!(out, EnableMouseCapture);
    }
    if enhanced {
        let _ = execute!(out, PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES));
    }

    // Terminal input is read on its own thread, polling, so that the terminal is free between polls to answer
    // the viewport's questions about the cursor.
    let (keys, mut input) = mpsc::unbounded_channel();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let reading = stop.clone();
    std::thread::spawn(move || {
        while !reading.load(std::sync::atomic::Ordering::Relaxed) {
            if terminal_events::poll(Duration::from_millis(100)).unwrap_or(false) {
                let Ok(event) = terminal_events::read() else { break };
                if keys.send(event).is_err() {
                    break;
                }
            }
        }
    });

    let (engine, session, result) = screen(&mut term, start, logs, &mut input).await;

    stop.store(true, std::sync::atomic::Ordering::Relaxed);

    if enhanced {
        let _ = execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
    }
    if mouse {
        let _ = execute!(std::io::stdout(), DisableMouseCapture);
    }
    let _ = execute!(std::io::stdout(), DisableBracketedPaste);
    let _ = term.finish();
    ratatui::restore();
    (engine, session, result)
}

async fn screen(
    term: &mut ChatScreen,
    start: Start,
    logs: &mut Logs,
    keys: &mut mpsc::UnboundedReceiver<Input>,
) -> (Option<Engine>, Option<SessionId>, Result<()>) {
    let Start { model_dir, engine: engine_options, cwd, store, options, mcp } = start;
    // The servers start while the model loads.
    let mut servers = Some(tokio::spawn(super::mcp::start(mcp)));
    let mut app = App::new(rustling::engine::model_name(&model_dir, engine_options.passes), engine_options.context, cwd.clone(), &options);
    let directory = std::env::var_os("HOME")
        .and_then(|home| cwd.strip_prefix(home).ok().map(|relative| Path::new("~").join(relative)))
        .unwrap_or_else(|| cwd.clone());
    app.view.push(Item::Header { directory: directory.display().to_string(), model: app.model.clone() });
    let (ask_tx, mut asks) = mpsc::unbounded_channel::<Ask>();
    let (stats_tx, mut stats) = watch::channel::<Option<Usage>>(None);
    let mut loading = Some(tokio::task::spawn_blocking(move || Engine::start(&model_dir, engine_options)));
    let mut engine: Option<Engine> = None;
    let mut ticker = tokio::time::interval(Duration::from_millis(100));

    let result: Result<()> = async {
        loop {
            // A turn can stop waiting for its approval when interrupted.
            app.asks.retain(|ask| !ask.reply.is_closed());
            term.draw(&mut app)?;
            if app.quit {
                return Ok(());
            }
            let waiting_for_engine = loading.is_some();
            tokio::select! {
                input = keys.recv() => match input {
                    Some(input) => app.input(input).await,
                    None => return Ok(()),
                },
                event = recv(&mut app.events) => match event {
                    Ok(event) => {
                        app.view.apply(&event);
                        // A first turn is what puts a new session in the store.
                        if let (Event::TurnCompleted { .. }, Some(ready)) = (&event, &app.ready) {
                            match recent_sessions(ready.store.as_ref()).await {
                                Ok(sessions) => app.sessions = sessions,
                                Err(error) => app.view.notice(Level::Warn, format!("Could not list sessions: {error}")),
                            }
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => app.events = None,
                },
                Some(ask) = asks.recv() => app.asks.push_back(ask),
                Some(explored) = recv_explored(&mut app.exploring) => app.explored(explored).await,
                () = app.files.receive() => {}
                Some(line) = logs.0.recv() => app.log_line(line),
                Ok(()) = stats.changed() => {
                    let usage = *stats.borrow_and_update();
                    if let Some(usage) = usage {
                        app.generated += usage.completion_tokens as u64;
                        app.last = Some(usage);
                    }
                }
                loaded = async { loading.as_mut().expect("checked").await }, if waiting_for_engine => {
                    loading = None;
                    let started = loaded??;
                    let mcp = servers.take().expect("taken once").await?;
                    let prepared = super::prepare(started.client(), &cwd, store.clone(), &options, ask_tx.clone(), stats_tx.clone(), mcp).await?;
                    engine = Some(started);
                    app.ready(prepared).await?;
                }
                _ = ticker.tick(), if app.view.is_busy() || waiting_for_engine => app.tick += 1,
            }
        }
    }
    .await;
    // The agent holds the engine's clients; it goes first, so the engine can end.
    let id = app.ready.as_ref().map(|ready| ready.session.id().clone());
    drop(app);
    (engine, id, result)
}

/// What a running `/explore` sends the screen: the lines it logs, then its report.
enum Explored {
    Line(String),
    Done(manifatura::ToolResult),
}

/// The turn and call ids `/explore` shows its work under.
const EXPLORE: &str = "explore";

/// The running `/explore`'s next message, or never when none runs.
async fn recv_explored(exploring: &mut Option<(String, mpsc::UnboundedReceiver<Explored>)>) -> Option<Explored> {
    match exploring {
        Some((_, rx)) => rx.recv().await,
        None => std::future::pending().await,
    }
}

/// The next session event, or never when there is no session yet.
async fn recv(events: &mut Option<broadcast::Receiver<Event>>) -> Result<Event, broadcast::error::RecvError> {
    match events {
        Some(events) => events.recv().await,
        None => std::future::pending().await,
    }
}

fn composer() -> TextArea<'static> {
    let mut input = TextArea::default();
    input.set_wrap_mode(WrapMode::Glyph);
    input.set_cursor_line_style(Style::new());
    input.set_placeholder_style(dim());
    input
}

impl App {
    fn new(model: String, window: usize, cwd: PathBuf, options: &ChatOptions) -> Self {
        Self {
            view: View::new(),
            input: composer(),
            files: files::FilePicker::new(cwd),
            history: Vec::new(),
            browsing: None,
            log: VecDeque::new(),
            logs: options.logs,
            logs_open: false,
            log_button: None,
            picked: 0,
            completions_hidden: false,
            sessions: Vec::new(),
            resume_pick: None,
            asks: VecDeque::new(),
            armed: false,
            tick: 0,
            quit: false,
            abort: None,
            exploring: None,
            ready: None,
            events: None,
            last: None,
            generated: 0,
            model,
            window,
            effort: options.effort.label(),
            yes: options.yes,
        }
    }

    async fn ready(&mut self, prepared: Prepared) -> Result<()> {
        let session = &prepared.session;
        if !prepared.history.is_empty() {
            self.view.notice(Level::Info, format!("Resumed session {}", session.id().as_str()));
            self.view.replay(&prepared.history);
        }
        for problem in &prepared.problems {
            self.view.notice(Level::Warn, problem.clone());
        }
        self.events = Some(session.subscribe());
        self.sessions = recent_sessions(prepared.store.as_ref()).await?;
        self.ready = Some(prepared);
        Ok(())
    }

    fn log_line(&mut self, line: LogLine) {
        self.log.push_back(line);
        if self.log.len() > KEPT_LOG_LINES {
            self.log.pop_front();
        }
    }

    async fn input(&mut self, input: Input) {
        match input {
            Input::Key(key) if key.kind != KeyEventKind::Release => self.key(key).await,
            Input::Mouse(MouseEvent { kind: MouseEventKind::Down(MouseButton::Left), column, row, .. }) => {
                let on_button = self.log_button.is_some_and(|button| button.contains((column, row).into()));
                if on_button {
                    self.logs_open = !self.logs_open;
                }
            }
            Input::Paste(text) if self.asks.is_empty() && self.resume_pick.is_none() => {
                self.picked = 0;
                self.completions_hidden = false;
                self.input.insert_str(text.replace("\r\n", "\n").replace('\r', "\n"));
            }
            _ => {}
        }
        self.files.sync(&self.input, self.completions_hidden);
    }

    async fn key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        if !(ctrl && key.code == KeyCode::Char('c')) {
            self.armed = false;
        }
        if let Some(ask) = self.asks.front_mut() {
            let answer = if ctrl && key.code == KeyCode::Char('c') {
                self.interrupt();
                Some(Answer::Deny(Some("the user interrupted".into())))
            } else {
                ask.key(key)
            };
            if let Some(answer) = answer {
                let ask = self.asks.pop_front().expect("checked");
                let _ = ask.reply.send(answer);
            }
            return;
        }
        if let Some(selected) = self.resume_pick {
            let last = self.sessions.len().saturating_sub(1);
            match (key.code, ctrl) {
                (KeyCode::Up, _) => self.resume_pick = Some(selected.saturating_sub(1)),
                (KeyCode::Down, _) => self.resume_pick = Some((selected + 1).min(last)),
                (KeyCode::Esc, _) | (KeyCode::Char('c'), true) => self.resume_pick = None,
                (KeyCode::Enter, _) => {
                    self.resume_pick = None;
                    if let Some(session) = self.sessions.get(selected) {
                        self.resume(session.id.clone()).await;
                    }
                }
                _ => {}
            }
            return;
        }
        if self.completion_key(key) {
            return;
        }
        if !matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Tab) {
            self.picked = 0;
        }
        match (key.code, ctrl) {
            (KeyCode::Char('c'), true) => {
                if self.view.is_busy() {
                    self.interrupt();
                } else if !self.text().is_empty() {
                    self.input = composer();
                    self.browsing = None;
                } else if self.armed {
                    self.quit = true;
                } else {
                    self.armed = true;
                }
            }
            (KeyCode::Char('d'), true) if self.text().is_empty() && !self.view.is_busy() => self.quit = true,
            (KeyCode::Char('l'), true) if self.logs => self.logs_open = !self.logs_open,
            (KeyCode::Esc, _) if self.view.is_busy() => self.interrupt(),
            (KeyCode::Enter, _) => {
                let newline = key.modifiers.intersects(KeyModifiers::SHIFT | KeyModifiers::ALT);
                // A trailing backslash is a new line, for terminals that cannot tell Shift+Enter from Enter.
                let continued = self.input.lines().last().is_some_and(|line| line.ends_with('\\'));
                if newline || continued {
                    if continued {
                        self.input.delete_char();
                    }
                    self.input.insert_newline();
                } else {
                    let text = self.text();
                    if self.ready.is_none() && !text.trim_start().starts_with('/') {
                        return; // Keep the draft and cursor until it can actually be sent.
                    }
                    if !text.trim().is_empty() {
                        self.input = composer();
                        self.browsing = None;
                        self.submit(text.trim().to_owned()).await;
                    }
                }
            }
            (KeyCode::Up, false) if self.input.cursor().0 == 0 => self.browse(-1),
            (KeyCode::Down, false) if self.input.cursor().0 + 1 >= self.input.lines().len() => self.browse(1),
            _ => {
                self.input.input(key);
            }
        }
    }

    fn text(&self) -> String {
        self.input.lines().join("\n")
    }

    /// A completion consumes Enter when it inserts a file or a command argument.
    /// Fully specified slash commands continue to the normal submission handler.
    fn completion_key(&mut self, key: KeyEvent) -> bool {
        let files = self.files.active();
        let options = self.completions();
        if key.code == KeyCode::Esc && (files || !options.is_empty() && !self.view.is_busy()) {
            self.completions_hidden = true;
            return true;
        }
        if !matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Esc) {
            self.completions_hidden = false;
        }
        let code = match (key.code, key.modifiers) {
            (KeyCode::Char('p'), KeyModifiers::CONTROL) if files => KeyCode::Up,
            (KeyCode::Char('n'), KeyModifiers::CONTROL) if files => KeyCode::Down,
            (_, modifiers) if modifiers.contains(KeyModifiers::CONTROL) => return false,
            (code, _) => code,
        };
        let picked = self.picked.min(options.len().saturating_sub(1));
        let Some(option) = options.get(picked) else {
            return files && (matches!(code, KeyCode::Up | KeyCode::Down | KeyCode::Tab)
                || code == KeyCode::Enter && key.modifiers.is_empty() && self.files.searching());
        };
        match code {
            KeyCode::Up => self.picked = picked.checked_sub(1).unwrap_or(options.len() - 1),
            KeyCode::Down => self.picked = (picked + 1) % options.len(),
            KeyCode::Tab => self.complete(option),
            KeyCode::Enter if key.modifiers.is_empty() && (files || self.text().trim() != option.text.trim()) => {
                self.complete(option);
                return !option.runs;
            }
            _ => return false,
        }
        true
    }

    /// Matches for the active file mention or slash command.
    fn completions(&self) -> Vec<Completion> {
        if self.files.active() && self.asks.is_empty() && !self.completions_hidden {
            return self.files.matches.iter().map(|path| Completion {
                text: path.clone(), label: path.clone(), about: String::new(), runs: false,
            }).collect();
        }
        match (self.input.lines(), self.asks.is_empty() && !self.completions_hidden) {
            ([line], true) => {
                let skills = self.ready.as_ref().map_or(&[][..], |ready| ready.skills.skills());
                completions(line, &self.sessions, skills)
            }
            _ => Vec::new(),
        }
    }

    fn complete(&mut self, completion: &Completion) {
        if self.files.active() {
            files::insert_path(&mut self.input, &completion.text);
        } else {
            self.input = composer();
            self.input.insert_str(&completion.text);
        }
        self.picked = 0;
    }

    /// Up and Down through what was sent before.
    fn browse(&mut self, step: isize) {
        if self.history.is_empty() {
            return;
        }
        let last = self.history.len() as isize;
        let at = self.browsing.as_ref().map_or(last, |(i, _)| *i as isize) + step;
        if at >= last {
            if let Some((_, draft)) = self.browsing.take() {
                self.input = draft;
            }
            return;
        }
        let at = at.max(0) as usize;
        match &mut self.browsing {
            Some((i, _)) => *i = at,
            None => self.browsing = Some((at, std::mem::replace(&mut self.input, composer()))),
        }
        let mut input = composer();
        input.insert_str(&self.history[at]);
        self.input = input;
    }

    fn interrupt(&mut self) {
        if let Some(abort) = &self.abort {
            abort.abort();
        }
    }

    async fn submit(&mut self, text: String) {
        if self.history.last() != Some(&text) {
            self.history.push(text.clone());
        }
        if text.starts_with('/') {
            return self.command(&text).await;
        }
        self.send(text.clone(), text).await;
    }

    /// Sends `text` to the agent, shown as `shown`: as a new turn, or while it works, before its next step.
    async fn send(&mut self, shown: String, text: String) {
        let Some(ready) = &self.ready else {
            self.view.notice(Level::Info, "the model is still loading");
            return;
        };
        if self.exploring.is_some() {
            self.view.notice(Level::Info, "explore is still working; send this once it reports, or press Esc to stop it");
            return;
        }
        if self.view.is_busy() {
            // Sent while the agent works: it reads it before its next step.
            match ready.session.steer(text) {
                true => self.view.push(Item::User(format!("{shown}  (sent while working)"))),
                false => self.view.notice(Level::Warn, "could not send that while the agent works"),
            }
            return;
        }
        self.view.push(Item::User(shown));
        self.prompt(text).await;
    }

    /// Starts a turn for `text`, which the transcript already shows.
    async fn prompt(&mut self, text: String) {
        let Some(ready) = &self.ready else { return };
        match ready.session.prompt(text).await {
            Ok(turn) => {
                self.abort = Some(turn.abort_handle());
                // The screen follows the session's events; the turn's own result adds nothing to them.
                tokio::spawn(async move {
                    let _ = turn.wait().await;
                });
            }
            Err(error) => self.view.notice(Level::Error, error.to_string()),
        }
    }

    async fn command(&mut self, line: &str) {
        let (command, argument) = line.split_once(' ').map_or((line, ""), |(c, a)| (c, a.trim()));
        if matches!(command, "/exit" | "/quit") {
            self.quit = true;
            return;
        }
        if command == "/help" {
            for (name, argument, about) in COMMANDS {
                let name = format!("{name} {argument}");
                self.view.notice(Level::Info, format!("{:<18}{about}", name.trim_end()));
            }
            self.view.notice(
                Level::Info,
                "Enter sends · Shift+Enter inserts a line · @ picks a file · Tab completes · Esc or Ctrl-C stops the agent · Ctrl-D or Ctrl-C twice quits · Up and Down \
                 recall what you sent · with --logs, click logs (or Ctrl-L) for the log",
            );
            return;
        }
        if command == "/tools" {
            self.view.show_tools();
            return;
        }
        if command == "/status" {
            let details = self.status();
            self.view.notice(Level::Info, details);
            return;
        }
        let Some(ready) = &mut self.ready else {
            self.view.notice(Level::Info, "the model is still loading");
            return;
        };
        if self.view.is_busy() {
            self.view.notice(Level::Info, format!("{command} waits until the agent is done"));
            return;
        }
        let skill = command.strip_prefix('/').filter(|name| !is_command(command) && ready.skills.get(name).is_some());
        if let Some(name) = skill {
            return self.start_skill(name, argument, line).await;
        }
        match command {
            "/explore" if argument.is_empty() => self.view.notice(Level::Info, "/explore <question>: what to find out about the code"),
            "/explore" => self.explore(argument, line),
            "/new" => {
                let session = ready.runtime.session(&ready.agent, SessionId::new());
                self.view.replay(&[]);
                self.view.notice(Level::Info, format!("new session {}", session.id().as_str()));
                self.switch(session);
            }
            "/resume" if !argument.is_empty() => self.resume(SessionId::from(argument)).await,
            "/resume" | "/sessions" => {
                match recent_sessions(ready.store.as_ref()).await {
                    Err(error) => self.view.notice(Level::Error, format!("Could not list sessions: {error}")),
                    Ok(sessions) => {
                        self.sessions = sessions;
                        if self.sessions.is_empty() {
                            self.view.notice(Level::Info, "No sessions in this directory yet");
                        } else if command == "/resume" {
                            self.resume_pick = Some(0);
                        } else {
                            let current = ready.session.id();
                            for row in &self.sessions {
                                let mark = if &row.id == current { "● " } else { "  " };
                                self.view.notice(Level::Info, format!("{mark}{}  {}", row.id.as_str(), ago(row.updated)));
                            }
                        }
                    }
                }
            }
            "/mcp" => {
                if ready.servers.is_empty() && ready.problems.is_empty() {
                    let path = super::mcp::config_dir().map_or_else(|| "mcp.toml".to_owned(), |dir| dir.join("mcp.toml").display().to_string());
                    self.view.notice(Level::Info, format!("no MCP servers; add them to {path}"));
                }
                for (name, tools, eager) in &ready.servers {
                    let offered = if *eager { "offered" } else { "found with find_tools" };
                    self.view.notice(Level::Info, format!("{name} ({offered}): {}", tools.join(", ")));
                }
                for problem in &ready.problems {
                    self.view.notice(Level::Warn, problem.clone());
                }
            }
            "/effort" if argument.is_empty() => {
                let levels: Vec<&str> = LEVELS.iter().map(|(level, _)| *level).collect();
                self.view.notice(
                    Level::Info,
                    format!("effort is {}; set it with /effort {} or a number from 0 to 0.99", ready.reasoning.get().label(), levels.join("|")),
                );
            }
            "/effort" => match Effort::parse(argument) {
                Ok(effort) if effort == ready.reasoning.get() => {
                    self.view.notice(Level::Info, format!("effort is already {}", effort.label()))
                }
                Ok(effort) => {
                    let what = match effort {
                        Effort::Off => "off: replies start at once, without reasoning".to_owned(),
                        _ => effort.label(),
                    };
                    ready.reasoning.set(effort);
                    // The effort is part of the prompt, near its start.
                    self.view.notice(Level::Info, format!("effort {what}; the next reply processes the conversation again"));
                }
                Err(error) => self.view.notice(Level::Warn, error),
            },
            "/compact" => {
                let focus = (!argument.is_empty()).then(|| argument.to_owned());
                self.view.activity = Activity::Compacting;
                let result = ready.session.compact(ready.summarizer.clone(), focus).await;
                self.view.activity = Activity::Idle;
                match result {
                    Ok(_) => {}
                    Err(error) => self.view.notice(Level::Error, format!("could not compact: {error}")),
                }
            }
            "/skills" => {
                if ready.skills.is_empty() {
                    self.view.notice(Level::Info, format!("no skills; add them to {}/<name>/SKILL.md", super::PROJECT_SKILLS));
                }
                for skill in ready.skills.skills() {
                    let from = skill.path().map_or_else(|| "built in".to_owned(), |path| path.display().to_string());
                    self.view.notice(Level::Info, format!("/{}  {}  ({from})", skill.name, skill.description));
                }
            }
            _ => self.view.notice(Level::Warn, format!("unknown command {command}; /help lists them")),
        }
    }

    /// Starts the skill `name` from `/name request`: its instructions go to the agent ahead of the request, so it
    /// needs no step to load them. The transcript shows what was typed.
    async fn start_skill(&mut self, name: &str, request: &str, line: &str) {
        let Some(ready) = &self.ready else { return };
        let Some(instructions) = ready.skills.activation(name, ready.env.as_ref()).await else { return };
        let request = match request {
            "" => format!("Use the {name} skill."),
            request => request.to_owned(),
        };
        self.send(line.to_owned(), format!("{instructions}\n\n{request}")).await;
    }

    /// Starts the explore subagent on `question` from `/explore question`, without waiting for the model to choose
    /// it. The transcript shows it as a task; its report then goes to the agent with the question.
    fn explore(&mut self, question: &str, line: &str) {
        let Some(ready) = &self.ready else { return };
        self.view.push(Item::User(line.to_owned()));
        self.view.apply(&Event::TurnStarted { turn: EXPLORE.into() });
        self.view.apply(&Event::ActionStarted {
            turn: EXPLORE.into(),
            call_id: EXPLORE.into(),
            name: manifatura::subagents::TASK_TOOL.into(),
            input: serde_json::json!({ "agent": EXPLORE, "description": question }),
        });
        let abort = Abort::new();
        self.abort = Some(abort.clone());
        let (tx, rx) = mpsc::unbounded_channel();
        let (subagents, session, prompt) = (ready.subagents.clone(), ready.session.clone(), question.to_owned());
        tokio::spawn(async move {
            let lines = tx.clone();
            let log = move |line| {
                let _ = lines.send(Explored::Line(line));
            };
            let report = subagents.run(&session, EXPLORE, &prompt, &abort, log).await;
            let _ = tx.send(Explored::Done(report));
        });
        self.exploring = Some((question.to_owned(), rx));
    }

    async fn explored(&mut self, explored: Explored) {
        let report = match explored {
            Explored::Line(text) => {
                return self.view.apply(&Event::ActionLog { turn: EXPLORE.into(), call_id: EXPLORE.into(), text });
            }
            Explored::Done(report) => report,
        };
        let Some((question, _)) = self.exploring.take() else { return };
        let aborted = self.abort.take().is_some_and(|abort| abort.is_aborted());
        self.view.apply(&Event::ActionResult {
            turn: EXPLORE.into(),
            call_id: EXPLORE.into(),
            name: manifatura::subagents::TASK_TOOL.into(),
            output: report.output.clone(),
            is_error: report.is_error,
        });
        let outcome = if aborted { manifatura::Outcome::Aborted } else { manifatura::Outcome::Completed };
        self.view.apply(&Event::TurnCompleted { turn: EXPLORE.into(), outcome, error: None, usage: Default::default() });
        if !aborted {
            self.prompt(explore_message(&question, &report.output)).await;
        }
    }

    async fn resume(&mut self, id: SessionId) {
        let Some(ready) = &self.ready else { return };
        // Opening an unknown ID creates a session header. Recheck the store even for a picker selection.
        match recent_sessions(ready.store.as_ref()).await {
            Ok(sessions) if sessions.iter().any(|s| s.id == id) => {}
            Ok(_) => {
                self.view.notice(Level::Warn, format!("No session {} here; /resume lists them", id.as_str()));
                return;
            }
            Err(error) => {
                self.view.notice(Level::Error, format!("Could not list sessions: {error}"));
                return;
            }
        }
        let session = ready.runtime.session(&ready.agent, id);
        match session.history().await {
            Ok(history) => {
                self.view.notice(Level::Info, format!("Resumed session {} ({} messages)", session.id().as_str(), history.len()));
                self.view.replay(&history);
                self.switch(session);
            }
            Err(error) => self.view.notice(Level::Error, format!("Could not resume session: {error}")),
        }
    }

    fn switch(&mut self, session: Session) {
        self.events = Some(session.subscribe());
        self.abort = None;
        self.last = None;
        self.generated = 0;
        if let Some(ready) = &mut self.ready {
            ready.session = session;
        }
    }

    fn effort_label(&self) -> String {
        self.ready.as_ref().map_or_else(|| self.effort.clone(), |ready| ready.reasoning.get().label())
    }

    fn status(&self) -> String {
        let mut parts = vec![self.model.clone()];
        parts.push(format!("effort {}", self.effort_label()));
        if let Some(usage) = self.last {
            let used = (usage.prompt_tokens + usage.completion_tokens) as u64;
            parts.push(format!(
                "ctx {}% {}/{}",
                used * 100 / self.window.max(1) as u64,
                tokens(used),
                tokens(self.window as u64)
            ));
            let new = usage.prompt_tokens - usage.cached_tokens;
            let reused = usage.cached_tokens * 100 / usage.prompt_tokens.max(1);
            let prefill = match usage.prompt_ms {
                0 => String::new(),
                ms => format!(" at {:.0} tok/s", new as f64 * 1000.0 / ms as f64),
            };
            parts.push(format!("in {} ({reused}% reused){prefill}", tokens(new as u64)));
            let decode = match usage.reply_ms {
                0 => String::new(),
                ms => format!(" at {:.1} tok/s", usage.completion_tokens as f64 * 1000.0 / ms as f64),
            };
            parts.push(format!("out {}{decode}", tokens(usage.completion_tokens as u64)));
        } else {
            parts.push(format!("ctx {}", tokens(self.window as u64)));
        }
        if self.generated > 0 {
            parts.push(format!("{} generated", tokens(self.generated)));
        }
        if self.yes {
            parts.push("runs tools without asking".to_owned());
        }
        if let Some(ready) = &self.ready {
            let id = ready.session.id().as_str();
            parts.push(format!("session {id}"));
            for path in &ready.files {
                parts.push(format!("instructions from {}", path.display()));
            }
            if !ready.skills.is_empty() {
                parts.push(format!("skills: {}", ready.skills.names().join(", ")));
            }
        }
        parts.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn loading_app() -> App {
        let options = ChatOptions {
            system: None, resume_latest: false, resume: None, yes: false,
            effort: Effort::Default, max_tokens: 4096, logs: false,
        };
        App::new("inkling-2-mini".into(), 16_384, PathBuf::from("."), &options)
    }

    #[tokio::test]
    async fn loading_and_approval_preserve_the_composer() {
        use ratatui::{Terminal, TerminalOptions, Viewport};
        use ratatui::backend::TestBackend;

        let mut app = loading_app();
        app.input(Input::Paste("draft\r\nsecond line".into())).await;
        let enter = KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE);
        app.key(enter).await;
        assert_eq!(app.text(), "draft\nsecond line");
        assert!(app.history.is_empty());

        let (reply, answer) = tokio::sync::oneshot::channel();
        let call = manifatura::ToolCall { id: "r".into(), name: "read".into(), input: serde_json::json!({"path": "README.md"}) };
        app.asks.push_back(Ask::new(call.clone(), reply));
        let (reply, mut next) = tokio::sync::oneshot::channel();
        app.asks.push_back(Ask::new(call, reply));
        for (width, height) in [(80, 24), (60, 16)] {
            let backend = TestBackend::new(width, height);
            let mut term = Terminal::with_options(backend, TerminalOptions { viewport: Viewport::Inline(height / 2) }).unwrap();
            term.draw(|f| app.draw(f)).unwrap();
            let rendered = term.backend().to_string();
            for label in ["Read this file?", "README.md", "› 1. Allow once (y)", "2. Allow this tool for the session (a)", "3. Deny (n)", "Enter confirm"] {
                assert!(rendered.contains(label), "{rendered}");
            }
            assert!(!rendered.contains("second line"));
        }
        app.key(KeyEvent::from(KeyCode::Down)).await;
        app.key(KeyEvent::from(KeyCode::Down)).await;
        app.key(KeyEvent::from(KeyCode::Enter)).await;
        assert!(matches!(answer.await, Ok(Answer::Deny(None))));
        app.key(KeyEvent::new_with_kind(KeyCode::Enter, KeyModifiers::NONE, KeyEventKind::Repeat)).await;
        app.input(Input::Paste("y".into())).await;
        assert!(matches!(next.try_recv(), Err(tokio::sync::oneshot::error::TryRecvError::Empty)));
        app.key(KeyEvent::from(KeyCode::Enter)).await;
        assert_eq!(next.await.unwrap(), Answer::Allow);
        assert_eq!(app.text(), "draft\nsecond line");
    }

    #[tokio::test]
    async fn prompt_and_footer_stay_at_the_bottom_as_input_grows() {
        use ratatui::{Terminal, TerminalOptions, Viewport};
        use ratatui::backend::TestBackend;

        let mut app = loading_app();
        for (width, height) in [(80, 24), (60, 16), (100, 32)] {
            for draft in ["/", "first line\nsecond line", "wrap ".repeat(22).as_str()] {
                app.input = composer();
                app.input.insert_str(draft);
                let rows = app.pane_height(width, height);
                let area = Rect::new(0, height - rows, width, rows);
                let mut term = Terminal::with_options(TestBackend::new(width, height), TerminalOptions { viewport: Viewport::Fixed(area) }).unwrap();
                term.draw(|f| app.draw(f)).unwrap();
                let buf = term.backend().buffer();
                assert_eq!(buf[(0, height - 2)].symbol(), " ");
                assert_eq!(buf[(0, height - 1)].symbol(), "h", "high effort in footer");
                if draft == "/" {
                    assert_eq!(buf[(0, height - 3)].symbol(), "›");
                    assert!(term.backend().to_string().contains("/new"));
                }
            }
        }
        app.input = composer();
        app.input.insert_str("/st");
        app.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)).await;
        assert!(app.completions().is_empty());
        assert_eq!(app.text(), "/st");
        app.history.push("earlier question".into());
        app.browse(-1);
        app.browse(1);
        assert_eq!(app.text(), "/st");
    }

    #[test]
    fn draft_and_sent_message_share_padding_and_text_alignment() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;
        use ratatui::style::{Color, Modifier};
        use ratatui::widgets::Paragraph;

        let mut app = loading_app();
        let draft = "short\n12345678901";
        app.input.insert_str(draft);
        let (width, height) = (13, 20);
        let mut term = Terminal::new(TestBackend::new(width, height)).unwrap();
        term.draw(|f| {
            let sent = Item::User(draft.into()).lines(width.into());
            f.render_widget(Paragraph::new(sent), f.area());
            app.draw(f);
        }).unwrap();
        let buf = term.backend().buffer();
        let surface = buf[(0, 1)].bg;
        assert_ne!(surface, Color::Reset);
        for top in [1, height - 6] {
            for y in top..top + 5 {
                assert!((0..width).all(|x| buf[(x, y)].bg == surface), "missing shading on row {y}");
                assert_eq!(buf[(width - 1, y)].symbol(), " ");
            }
            for y in [top, top + 4] {
                assert!((0..width).all(|x| buf[(x, y)].symbol() == " "));
            }
            assert_eq!(buf[(0, top + 1)].symbol(), "›");
            assert_eq!(buf[(2, top + 1)].symbol(), "s");
            assert!(!buf[(2, top + 1)].modifier.contains(Modifier::BOLD));
            assert_eq!(buf[(2, top + 2)].symbol(), "1");
            assert_eq!(buf[(2, top + 3)].symbol(), "1");
        }
        assert!(buf[(0, 2)].modifier.contains(Modifier::DIM));
        assert!(buf[(0, height - 5)].modifier.contains(Modifier::BOLD));
    }

    #[tokio::test]
    async fn at_opens_a_file_picker_and_escape_keeps_the_draft() {
        use ratatui::{Terminal, backend::TestBackend};

        let mut app = loading_app();
        app.input(Input::Paste("Compare\nthe @".into())).await;
        let mut term = Terminal::new(TestBackend::new(70, 12)).unwrap();
        term.draw(|f| app.draw(f)).unwrap();
        assert!(term.backend().to_string().contains("Type a file name"));
        app.input(Input::Key(KeyEvent::from(KeyCode::Esc))).await;
        term.draw(|f| app.draw(f)).unwrap();
        assert!(!term.backend().to_string().contains("Type a file name"));
        assert_eq!(app.text(), "Compare\nthe @");
    }

    #[tokio::test]
    async fn file_selection_replaces_only_the_active_mention() {
        use std::fs;
        use ratatui_textarea::CursorMove;

        let root = std::env::temp_dir().join(format!("rustling-file-picker-{}", std::process::id()));
        fs::create_dir_all(root.join("docs")).unwrap();
        fs::create_dir_all(root.join(".git")).unwrap();
        for name in ["docs/report one.md", "docs/report two.md", "report.tmp", ".git/report"] {
            fs::write(root.join(name), "").unwrap();
        }
        fs::write(root.join(".gitignore"), "*.tmp\n").unwrap();
        let mut app = loading_app();
        app.files = files::FilePicker::new(root.clone());
        app.input(Input::Paste("@previous\nRésumé @rpt with me".into())).await;
        app.input.move_cursor(CursorMove::Jump(1, 9));
        app.input(Input::Key(KeyEvent::from(KeyCode::Left))).await;
        assert!(app.files.searching());
        app.files.receive().await;
        let mut matches = app.files.matches.clone();
        matches.sort();
        assert_eq!(matches, ["docs/report one.md", "docs/report two.md"]);
        let chosen = app.files.matches[1].clone();
        app.input(Input::Key(KeyEvent::from(KeyCode::Down))).await;
        app.input(Input::Key(KeyEvent::from(KeyCode::Enter))).await;
        assert_eq!(app.text(), format!("@previous\nRésumé \"{chosen}\" with me"));
        assert!(!app.files.active());
        assert!(app.history.is_empty());
        app.input(Input::Paste("and ".into())).await;
        assert!(app.text().ends_with("and with me"));
        app.input(Input::Key(KeyEvent::from(KeyCode::End))).await;
        app.input(Input::Paste(" dev@example.org".into())).await;
        assert!(!app.files.active());
        app.input(Input::Paste(" @zzz".into())).await;
        app.input(Input::Paste("q".into())).await;
        app.files.receive().await;
        assert_eq!(app.files.hint(), Some("No matching files"));
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn picker_scrolls_to_the_selection_and_escape_restores_input() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut app = loading_app();
        app.sessions = (0..12).map(|i| SessionMeta {
            id: SessionId::from(format!("session-{i:02}").as_str()), agent: None, updated: 0,
        }).collect();
        app.resume_pick = Some(0);
        for _ in 0..20 {
            app.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)).await;
        }
        let mut term = Terminal::new(TestBackend::new(60, 8)).unwrap();
        term.draw(|f| app.draw(f)).unwrap();
        let rendered = term.backend().to_string();
        assert!(rendered.contains("› session-11") && !rendered.contains("session-00"), "{rendered}");
        app.key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE)).await;
        assert_eq!(app.resume_pick, Some(10));
        app.input(Input::Paste("ignored in picker".into())).await;
        app.key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)).await;
        assert_eq!(app.resume_pick, None);
        assert!(app.text().is_empty());
    }

    #[test]
    fn commands_and_sessions_complete() {
        let sessions = ["01a2-aaaa", "01b3-bbbb"].map(|id| SessionMeta {
            id: SessionId::from(id), updated: 0, agent: None,
        });
        let names = |line: &str| -> Vec<String> { completions(line, &sessions, &[]).into_iter().map(|c| c.text).collect() };
        assert_eq!(names("/re"), ["/resume "]);
        assert_eq!(names("/s"), ["/sessions", "/status", "/skills"]);
        assert_eq!(names("/"), COMMANDS.iter().map(|(n, a, _)| if a.is_empty() { n.to_string() } else { format!("{n} ") }).collect::<Vec<_>>());
        assert_eq!(names("/resume 01b"), ["/resume 01b3-bbbb"]);
        assert_eq!(names("/resume 01a2-aaaa"), Vec::<String>::new(), "a complete id needs no completing");
        assert!(names("hello").is_empty() && names("/compact focus").is_empty());
        assert!(completions("/new", &sessions, &[])[0].runs && !completions("/compact", &sessions, &[])[0].runs);
        assert_eq!(names("/effort o"), ["/effort off"]);
        assert_eq!(names("/effort m").len(), 3, "medium, minimal and max");
    }

    #[test]
    fn skills_complete_after_the_commands_they_do_not_shadow() {
        let skill = |name: &str| {
            Skill::parse(&format!("---\nname: {name}\ndescription: About {name}.\n---\nBody.")).unwrap()
        };
        let skills = [skill("codebase-tour"), skill("status")];
        let found = completions("/", &[], &skills);
        let tour = found.iter().find(|c| c.text == "/codebase-tour ").expect("the skill is offered");
        assert!(!tour.runs && tour.about == "About codebase-tour.");
        assert_eq!(found.iter().filter(|c| c.text.starts_with("/status")).count(), 1, "the command wins");
        assert_eq!(completions("/co", &[], &skills).iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), ["/compact ", "/codebase-tour "]);
    }
}
