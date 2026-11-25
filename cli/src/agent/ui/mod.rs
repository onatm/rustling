//! The chat screen: an inline viewport at the bottom of the terminal, with finished messages written into the
//! terminal's own scrollback above it, so they scroll, select and copy as any output does.
//!
//! The viewport holds, from the top: the reply as it streams and the tools running, a line saying what the
//! agent is doing, the input (or an approval question in its place), a strip with the latest log lines, and a
//! line of statistics.

pub(crate) mod logs;
mod text;
mod view;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{
    self as terminal_events, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture, Event as Input, KeyCode,
    KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::{execute, terminal};
use manifatura::kit::Answer;
use manifatura::{Abort, Event, Session, SessionId, ToolCall};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Widget};
use ratatui::{DefaultTerminal, Frame, TerminalOptions, Viewport};
use ratatui_textarea::TextArea;
use rustling::engine::{Engine, EngineOptions, Usage};
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use tracing::Level as LogLevel;

use self::logs::{LogLine, Logs};
use self::view::{Activity, Item, Level, View, dim, tokens, tool_summary};
use super::reasoning::{Effort, LEVELS};
use super::{ChatOptions, Prepared};

/// An approval the agent is waiting for.
pub(crate) struct Ask {
    pub(crate) call: ToolCall,
    pub(crate) reply: oneshot::Sender<Answer>,
}

const SPINNER: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
const KEPT_LOG_LINES: usize = 200;
/// The commands, their argument and what they do, for completion and `/help`.
const COMMANDS: &[(&str, &str, &str)] = &[
    ("/new", "", "start a new session"),
    ("/resume", "ID", "switch to another session"),
    ("/sessions", "", "list this directory's sessions"),
    ("/compact", "[focus]", "summarise older messages to make room"),
    ("/effort", "[level]", "how much the model reasons: off, none … max, or 0–0.99"),
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
    /// Where Up and Down are in the history, while browsing it.
    browsing: Option<usize>,
    log: VecDeque<LogLine>,
    /// Whether the log is kept at all (`--logs`), and whether its panel is open.
    logs: bool,
    logs_open: bool,
    /// Where the `logs` button was drawn, in terminal cells, for clicks.
    log_button: Option<Rect>,
    /// The highlighted completion, while the input is a command being typed.
    picked: usize,
    /// This directory's sessions, for completing `/resume`.
    sessions: Vec<String>,
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
fn completions(line: &str, sessions: &[String]) -> Vec<Completion> {
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
            .filter(|id| id.starts_with(partial) && id.as_str() != partial)
            .take(MAX_COMPLETIONS)
            .map(|id| Completion { text: format!("/resume {id}"), label: id.clone(), about: String::new(), runs: true })
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
            runs: argument.is_empty(),
        })
        .collect()
}

/// The ids of this directory's sessions, most recent first.
async fn session_ids(ready: &Prepared) -> Vec<String> {
    super::session_lines(ready.store.as_ref())
        .await
        .into_iter()
        .filter_map(|line| line.split_whitespace().next().map(str::to_owned))
        .collect()
}

/// Hands out the viewport's rows from the top down.
struct Rows {
    area: Rect,
    top: u16,
}

impl Rows {
    fn take(&mut self, height: u16) -> Rect {
        let height = height.min(self.area.bottom().saturating_sub(self.top));
        let rect = Rect { y: self.top, height, ..self.area };
        self.top += height;
        rect
    }
}

/// An approval question: the call, and the keys that answer it.
fn question_lines(ask: &Ask, width: usize) -> Vec<Line<'static>> {
    let accent = Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD);
    let mut lines = vec![Line::from(vec![Span::styled("? ", accent), Span::styled(format!("Allow {}", ask.call.name), accent)])];
    let summary = vec![Span::raw("  "), Span::raw(tool_summary(&ask.call.name, &ask.call.input))];
    lines.extend(text::wrap(summary, width, "  ").into_iter().take(6));
    lines.push(Line::from(vec![
        Span::styled("  [y]", accent),
        Span::raw(" yes  "),
        Span::styled("[a]", accent),
        Span::raw(" always this session  "),
        Span::styled("[n]", accent),
        Span::raw(" no"),
    ]));
    lines
}

/// A line of the log panel, coloured by level.
fn log_line(line: &LogLine, width: usize) -> Line<'static> {
    let style = match line.level {
        LogLevel::ERROR => Style::new().fg(Color::Red),
        LogLevel::WARN => Style::new().fg(Color::Yellow),
        _ => dim(),
    };
    Line::from(vec![
        Span::raw("  "),
        Span::styled(view::clip(&line.text, width.saturating_sub(3)), style),
    ])
}

/// Runs the screen until the person quits. Returns the engine, so the caller can shut it down once the runtime
/// holding its clients is gone, and the session to mention on the way out.
pub(crate) async fn run(start: Start, logs: &mut Logs) -> (Option<Engine>, Option<SessionId>, Result<()>) {
    let rows = terminal::size().map(|(_, rows)| rows).unwrap_or(24);
    let height = rows.saturating_sub(2).clamp(10, 24);
    let mut term = ratatui::init_with_options(TerminalOptions { viewport: Viewport::Inline(height) });
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

    let (engine, session, result) = screen(&mut term, height, start, logs, &mut input).await;

    stop.store(true, std::sync::atomic::Ordering::Relaxed);

    if enhanced {
        let _ = execute!(std::io::stdout(), PopKeyboardEnhancementFlags);
    }
    if mouse {
        let _ = execute!(std::io::stdout(), DisableMouseCapture);
    }
    let _ = execute!(std::io::stdout(), DisableBracketedPaste);
    let _ = term.clear();
    ratatui::restore();
    (engine, session, result)
}

async fn screen(
    term: &mut DefaultTerminal,
    height: u16,
    start: Start,
    logs: &mut Logs,
    keys: &mut mpsc::UnboundedReceiver<Input>,
) -> (Option<Engine>, Option<SessionId>, Result<()>) {
    let Start { model_dir, engine: engine_options, cwd, store, options, mcp } = start;
    // The servers start while the model loads.
    let mut servers = Some(tokio::spawn(super::mcp::start(mcp)));
    let mut app = App {
        view: View::new(),
        input: composer(),
        history: Vec::new(),
        browsing: None,
        log: VecDeque::new(),
        logs: options.logs,
        logs_open: false,
        log_button: None,
        picked: 0,
        sessions: Vec::new(),
        asks: VecDeque::new(),
        armed: false,
        tick: 0,
        quit: false,
        abort: None,
        ready: None,
        events: None,
        last: None,
        generated: 0,
        model: rustling::engine::model_name(&model_dir, engine_options.passes),
        window: engine_options.context,
        effort: options.effort.label(),
        yes: options.yes,
    };
    let (ask_tx, mut asks) = mpsc::unbounded_channel::<Ask>();
    let (stats_tx, mut stats) = watch::channel::<Option<Usage>>(None);
    let mut loading = Some(tokio::task::spawn_blocking(move || Engine::start(&model_dir, engine_options)));
    let mut engine: Option<Engine> = None;
    let mut ticker = tokio::time::interval(Duration::from_millis(100));

    let result: Result<()> = async {
        loop {
            flush(term, &mut app.view, height)?;
            term.draw(|f| app.draw(f))?;
            if app.quit {
                return Ok(());
            }
            // Approvals the turn stopped waiting for (it was interrupted) go.
            app.asks.retain(|ask| !ask.reply.is_closed());
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
                            app.sessions = session_ids(ready).await;
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
    input.set_block(Block::bordered().border_style(dim()));
    input.set_cursor_line_style(Style::new());
    input.set_placeholder_text("Ask the agent, or /help");
    input.set_placeholder_style(dim());
    input
}

/// Writes finished items into the scrollback above the viewport, a screenful of rows at a time (more at once
/// garbles the screen).
fn flush(term: &mut DefaultTerminal, view: &mut View, height: u16) -> Result<()> {
    let items = view.take_committed();
    if items.is_empty() {
        return Ok(());
    }
    let size = term.size()?;
    let lines: Vec<Line<'static>> = items.iter().flat_map(|item| item.lines(size.width as usize)).collect();
    let room = usize::from(size.height.saturating_sub(height)).max(1);
    for chunk in lines.chunks(room) {
        let chunk = chunk.to_vec();
        term.insert_before(chunk.len() as u16, move |buf| Paragraph::new(chunk).render(buf.area, buf))?;
    }
    Ok(())
}

impl App {
    async fn ready(&mut self, prepared: Prepared) -> Result<()> {
        let session = &prepared.session;
        match prepared.history.len() {
            0 => self.view.notice(Level::Info, format!("rustling · {} · new session {}", self.model, session.id().as_str())),
            n => {
                self.view.replay(&prepared.history);
                self.view.notice(Level::Info, format!("rustling · {} · resumed session {} ({n} messages)", self.model, session.id().as_str()));
            }
        }
        for (name, tools, eager) in &prepared.servers {
            let offered = if *eager { "offered" } else { "found with find_tools" };
            self.view.notice(Level::Info, format!("MCP {name}: {} tools, {offered}", tools.len()));
        }
        for problem in &prepared.problems {
            self.view.notice(Level::Warn, problem.clone());
        }
        if !prepared.files.is_empty() {
            let paths: Vec<String> = prepared.files.iter().map(|path| path.display().to_string()).collect();
            self.view.notice(Level::Info, format!("instructions from {}", paths.join(", ")));
        }
        let logs = if self.logs { " · click logs or press Ctrl-L for the log" } else { "" };
        self.view.notice(Level::Info, format!("Enter sends · Shift+Enter new line · Esc interrupts · /help for commands{logs}"));
        self.events = Some(session.subscribe());
        self.sessions = session_ids(&prepared).await;
        self.ready = Some(prepared);
        Ok(())
    }

    fn log_line(&mut self, line: LogLine) {
        self.log.push_back(line);
        if self.log.len() > KEPT_LOG_LINES {
            self.log.pop_front();
        }
    }

    fn busy(&self) -> bool {
        self.view.is_busy() || !self.asks.is_empty()
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
            Input::Paste(text) if self.asks.is_empty() => {
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
        if !self.asks.is_empty() {
            let answer = match key.code {
                KeyCode::Char('y' | 'Y') | KeyCode::Enter => Some(Answer::Allow),
                KeyCode::Char('a' | 'A') => Some(Answer::AllowAlways),
                KeyCode::Char('n' | 'N') | KeyCode::Esc => Some(Answer::Deny(None)),
                KeyCode::Char('c') if ctrl => {
                    self.interrupt();
                    Some(Answer::Deny(Some("the user interrupted".into())))
                }
                _ => None,
            };
            if let Some(answer) = answer {
                let ask = self.asks.pop_front().expect("checked");
                let _ = ask.reply.send(answer);
            }
            return;
        }
        let completions = self.completions();
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
        match (self.input.lines(), self.asks.is_empty()) {
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
        let at = self.browsing.map_or(last, |i| i as isize) + step;
        if at >= last {
            self.browsing = None;
            self.input = composer();
            return;
        }
        let at = at.clamp(0, last - 1) as usize;
        self.browsing = Some(at);
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
                "Tab completes a command · Esc or Ctrl-C stops the agent · Ctrl-D or Ctrl-C twice quits · Up and Down \
                 recall what you sent · with --logs, click logs (or Ctrl-L) for the log",
            );
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
                self.view.notice(Level::Info, format!("new session {}", session.id().as_str()));
                self.switch(session);
            }
            "/resume" if !argument.is_empty() => {
                let id = SessionId::from(argument);
                let known = ready.store_list().await;
                if !known.contains(&id) {
                    self.view.notice(Level::Warn, format!("no session {argument} here; /sessions lists them"));
                    return;
                }
                let session = ready.runtime.session(&ready.agent, id);
                match session.history().await {
                    Ok(history) => {
                        self.view.replay(&history);
                        self.view.notice(Level::Info, format!("resumed session {argument} ({} messages)", history.len()));
                    }
                    Err(error) => self.view.notice(Level::Error, error.to_string()),
                }
                self.switch(session);
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
            "/sessions" => {
                let current = ready.session.id().clone();
                for line in super::session_lines(ready.store.as_ref()).await {
                    let mark = if line.starts_with(current.as_str()) { "● " } else { "  " };
                    self.view.notice(Level::Info, format!("{mark}{line}"));
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

    fn switch(&mut self, session: Session) {
        self.events = Some(session.subscribe());
        self.abort = None;
        self.last = None;
        self.generated = 0;
        if let Some(ready) = &mut self.ready {
            ready.session = session;
        }
    }

    /// Draws, from the top of the viewport down: what is being written, what the agent is doing, the input (or a
    /// question), the log strip and the statistics. They sit right under the last message, as the conversation
    /// continues there; the rows the viewport has to spare stay empty below them.
    fn draw(&mut self, f: &mut Frame) {
        let area = f.area();
        let width = area.width as usize;

        let activity = self.activity_line();
        let activity_height = u16::from(!activity.spans.is_empty());
        let (question, input_height) = match self.asks.front() {
            Some(ask) => {
                let lines = question_lines(ask, width);
                let height = lines.len() as u16;
                (Some(lines), height)
            }
            None => (None, (self.input.lines().len() as u16).clamp(1, 6) + 2),
        };
        let completions = self.completions();
        let picked = self.picked.min(completions.len().saturating_sub(1));
        let completion_lines: Vec<Line> = completions
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let (mark, style) = match i == picked {
                    true => ("› ", Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    false => ("  ", Style::new()),
                };
                Line::from(vec![Span::styled(format!("{mark}{:<18}", c.label), style), Span::styled(c.about.clone(), dim())])
            })
            .collect();
        let completion_height = completion_lines.len() as u16;
        let panel = match self.logs_open {
            true => self.log.len().clamp(1, LOG_PANEL) as u16 + 1,
            false => 0,
        };
        let chrome = activity_height + input_height + completion_height + panel + 1;
        let live = self.view.live_lines(width, area.height.saturating_sub(chrome) as usize);

        let mut rows = Rows { area, top: area.y };
        f.render_widget(Paragraph::new(live.clone()), rows.take(live.len() as u16));
        f.render_widget(Paragraph::new(activity), rows.take(activity_height));
        match question {
            Some(lines) => f.render_widget(Paragraph::new(lines), rows.take(input_height)),
            None => f.render_widget(&self.input, rows.take(input_height)),
        }
        f.render_widget(Paragraph::new(completion_lines), rows.take(completion_height));
        if panel > 0 {
            let shown = (panel - 1) as usize;
            let mut lines = vec![Line::styled("─ log (click logs or Ctrl-L to close) ".to_owned() + &"─".repeat(width.saturating_sub(38)), dim())];
            lines.extend(self.log.iter().skip(self.log.len().saturating_sub(shown)).map(|line| log_line(line, width)));
            if self.log.is_empty() {
                lines.push(Line::styled("  nothing logged yet", dim()));
            }
            f.render_widget(Paragraph::new(lines), rows.take(panel));
        }
        let stats = rows.take(1);
        let button = LOG_BUTTON.len() as u16;
        match self.logs {
            true if stats.width > button + 10 => {
                let text = Rect { width: stats.width - button - 1, ..stats };
                f.render_widget(Paragraph::new(self.stats_line()), text);
                let rect = Rect { x: stats.right() - button, width: button, ..stats };
                let style = match self.logs_open {
                    true => Style::new().fg(Color::Black).bg(Color::Cyan),
                    false => dim().add_modifier(Modifier::REVERSED),
                };
                f.render_widget(Paragraph::new(Line::styled(LOG_BUTTON, style)), rect);
                self.log_button = Some(rect);
            }
            _ => f.render_widget(Paragraph::new(self.stats_line()), stats),
        }
    }

    fn activity_line(&self) -> Line<'static> {
        let spinner = Span::styled(format!("{} ", SPINNER[self.tick % SPINNER.len()]), Style::new().fg(Color::Cyan));
        if self.ready.is_none() {
            return Line::from(vec![spinner, Span::styled("Loading the model", Style::new().add_modifier(Modifier::BOLD))]);
        }
        if !self.asks.is_empty() {
            return Line::from(vec![
                Span::styled("? ", Style::new().fg(Color::Yellow)),
                Span::styled("Waiting for you", Style::new().add_modifier(Modifier::BOLD)),
            ]);
        }
        if self.armed {
            return Line::styled("Press Ctrl-C again to quit", Style::new().fg(Color::Yellow));
        }
        let what = match &self.view.activity {
            Activity::Idle => return Line::default(),
            Activity::Thinking => "Thinking".to_owned(),
            Activity::Generating => "Writing".to_owned(),
            Activity::Running(names) => format!("Running {}", names.join(", ")),
            Activity::Compacting => "Compacting".to_owned(),
        };
        let seconds = self.view.since.map_or(0, |since| since.elapsed().as_secs());
        Line::from(vec![
            spinner,
            Span::styled(what, Style::new().add_modifier(Modifier::BOLD)),
            Span::styled(format!(" · {seconds}s · Esc to interrupt · Enter sends a note"), dim()),
        ])
    }

    /// `inkling-2-mini · effort low · ctx 9% 1.4k/16k · last: 19 new (99% reused) · 74 out at 25 tok/s · …`
    fn stats_line(&self) -> Line<'static> {
        let mut parts = vec![self.model.clone()];
        let effort = self.ready.as_ref().map_or_else(|| self.effort.clone(), |ready| ready.reasoning.get().label());
        parts.push(format!("effort {effort}"));
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
            parts.push(format!("session {}", id.rsplit('-').next().unwrap_or(id)));
        }
        let mut line = vec![Span::styled(parts.join(" · "), dim())];
        if self.busy() {
            line.insert(0, Span::styled("● ", Style::new().fg(Color::Cyan)));
        }
        Line::from(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_and_sessions_complete() {
        let sessions = vec!["01a2-aaaa".to_owned(), "01b3-bbbb".to_owned()];
        let names = |line: &str| -> Vec<String> { completions(line, &sessions).into_iter().map(|c| c.text).collect() };
        assert_eq!(names("/re"), ["/resume "]);
        assert_eq!(names("/s"), ["/sessions"]);
        assert_eq!(names("/"), COMMANDS.iter().map(|(n, a, _)| if a.is_empty() { n.to_string() } else { format!("{n} ") }).collect::<Vec<_>>());
        assert_eq!(names("/resume 01b"), ["/resume 01b3-bbbb"]);
        assert_eq!(names("/resume 01a2-aaaa"), Vec::<String>::new(), "a complete id needs no completing");
        assert!(names("hello").is_empty() && names("/compact focus").is_empty());
        assert!(completions("/new", &sessions)[0].runs && !completions("/compact", &sessions)[0].runs);
        assert_eq!(names("/effort o"), ["/effort off"]);
        assert_eq!(names("/effort m").len(), 3, "medium, minimal and max");
    }
}
