//! Chat on the terminal's main screen. The transcript goes into native scrollback;
//! thinking and replies stream into the transcript above the pinned controls.

mod approval;
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
use manifatura::{Abort, Event, Session, SessionId, SessionMeta};
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui_textarea::{TextArea, WrapMode};
use rustling::engine::{Engine, EngineOptions, Usage};
use tokio::sync::{broadcast, mpsc, watch};

pub(crate) use self::approval::Ask;
use self::logs::{LogLine, Logs};
use self::view::{Activity, Item, Level, View, dim, tokens};
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

/// What a one-line input could complete to: the commands it begins, or after `/resume `, the sessions.
fn completions(line: &str, sessions: &[SessionMeta]) -> Vec<Completion> {
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
        .collect()
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
    let mut app = App::new(rustling::engine::model_name(&model_dir, engine_options.passes), engine_options.context, &options);
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
    fn new(model: String, window: usize, options: &ChatOptions) -> Self {
        Self {
            view: View::new(),
            input: composer(),
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
                self.completions_hidden = false;
                self.input.insert_str(text.replace("\r\n", "\n").replace('\r', "\n"));
            }
            _ => {}
        }
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
        let completions = self.completions();
        if key.code == KeyCode::Esc && !completions.is_empty() && !self.view.is_busy() {
            self.completions_hidden = true;
            return;
        }
        if !matches!(key.code, KeyCode::Up | KeyCode::Down | KeyCode::Esc) {
            self.completions_hidden = false;
        }
        if !completions.is_empty() && !ctrl {
            let picked = self.picked.min(completions.len() - 1);
            match key.code {
                KeyCode::Up => {
                    self.picked = picked.checked_sub(1).unwrap_or(completions.len() - 1);
                    return;
                }
                KeyCode::Down => {
                    self.picked = (picked + 1) % completions.len();
                    return;
                }
                KeyCode::Tab => {
                    self.complete(&completions[picked]);
                    return;
                }
                KeyCode::Enter if key.modifiers.is_empty() && self.text().trim() != completions[picked].text.trim() => {
                    let completion = completions[picked].clone();
                    self.complete(&completion);
                    // A command that takes nothing more runs at once; one that takes an argument waits for it.
                    if !completion.runs {
                        return;
                    }
                }
                _ => {}
            }
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

    /// What the input could complete to, while it is a command being typed on one line: the commands it begins,
    /// or after `/resume `, the sessions.
    fn completions(&self) -> Vec<Completion> {
        match (self.input.lines(), self.asks.is_empty() && !self.completions_hidden) {
            ([line], true) => completions(line, &self.sessions),
            _ => Vec::new(),
        }
    }

    fn complete(&mut self, completion: &Completion) {
        let mut input = composer();
        input.insert_str(&completion.text);
        self.input = input;
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
        let Some(ready) = &self.ready else {
            self.view.notice(Level::Info, "the model is still loading");
            return;
        };
        if self.view.is_busy() {
            // Sent while the agent works: it reads it before its next step.
            match ready.session.steer(text.clone()) {
                true => self.view.push(Item::User(format!("{text}  (sent while working)"))),
                false => self.view.notice(Level::Warn, "could not send that while the agent works"),
            }
            return;
        }
        self.view.push(Item::User(text.clone()));
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
                "Enter sends · Shift+Enter inserts a line · Tab completes a command · Esc or Ctrl-C stops the agent · Ctrl-D or Ctrl-C twice quits · Up and Down \
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
        match command {
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
            _ => self.view.notice(Level::Warn, format!("unknown command {command}; /help lists them")),
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
        App::new("inkling-2-mini".into(), 16_384, &options)
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
        let names = |line: &str| -> Vec<String> { completions(line, &sessions).into_iter().map(|c| c.text).collect() };
        assert_eq!(names("/re"), ["/resume "]);
        assert_eq!(names("/s"), ["/sessions", "/status"]);
        assert_eq!(names("/"), COMMANDS.iter().map(|(n, a, _)| if a.is_empty() { n.to_string() } else { format!("{n} ") }).collect::<Vec<_>>());
        assert_eq!(names("/resume 01b"), ["/resume 01b3-bbbb"]);
        assert_eq!(names("/resume 01a2-aaaa"), Vec::<String>::new(), "a complete id needs no completing");
        assert!(names("hello").is_empty() && names("/compact focus").is_empty());
        assert!(completions("/new", &sessions)[0].runs && !completions("/compact", &sessions)[0].runs);
        assert_eq!(names("/effort o"), ["/effort off"]);
        assert_eq!(names("/effort m").len(), 3, "medium, minimal and max");
    }
}
