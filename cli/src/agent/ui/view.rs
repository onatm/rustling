//! What the screen shows of a session: finished items, written once into the terminal's scrollback, and the
//! live part (the reply as it streams, the tools running) redrawn below them. It is fed manifatura's session
//! events only.

use std::time::Instant;

use manifatura::{Content, Event, Message, Outcome, Role};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::Value;

use super::text::{markdown, plain, wrap};

/// A finished piece of the transcript.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Item {
    User(String),
    Assistant(String),
    /// Reasoning, collapsed to its first line once the answer starts, with how long it went on.
    Reasoning { first: String, seconds: u64 },
    Tool { name: String, summary: String, output: String, is_error: bool },
    Notice(Level, String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Level {
    Info,
    Warn,
    Error,
}

pub(crate) fn dim() -> Style {
    Style::new().fg(Color::DarkGray)
}

impl Item {
    /// The item as wrapped lines, after a blank line that separates it from the one before.
    pub(crate) fn lines(&self, width: usize) -> Vec<Line<'static>> {
        let mut out = vec![Line::default()];
        match self {
            Item::User(text) => {
                let style = Style::new().add_modifier(Modifier::BOLD);
                for (i, line) in text.split('\n').enumerate() {
                    let prefix = if i == 0 { "› " } else { "  " };
                    let spans = vec![Span::styled(prefix, style.fg(Color::Cyan)), Span::styled(line.to_owned(), style)];
                    out.extend(wrap(spans, width, "  "));
                }
            }
            Item::Assistant(text) => out.extend(markdown(text, width)),
            Item::Reasoning { first, seconds } => {
                let style = dim().add_modifier(Modifier::ITALIC);
                let line = format!("∴ Thought for {seconds}s · {first}");
                out.push(Line::styled(clip(&line, width.saturating_sub(1).max(8)), style));
            }
            Item::Tool { name, summary, output, is_error } => {
                let (mark, colour) = if *is_error { ("✗", Color::Red) } else { ("●", Color::Green) };
                let spans = vec![
                    Span::styled(format!("{mark} "), Style::new().fg(colour)),
                    Span::styled(name.clone(), Style::new().add_modifier(Modifier::BOLD)),
                    Span::raw(" "),
                    Span::styled(summary.clone(), dim()),
                ];
                out.extend(wrap(spans, width, "  "));
                out.extend(preview(output, width, *is_error));
            }
            Item::Notice(level, text) => {
                let style = match level {
                    Level::Info => dim(),
                    Level::Warn => Style::new().fg(Color::Yellow),
                    Level::Error => Style::new().fg(Color::Red),
                };
                out.extend(plain(text, width, style));
            }
        }
        out
    }
}

/// The first lines of a tool's result, under the call.
fn preview(output: &str, width: usize, is_error: bool) -> Vec<Line<'static>> {
    const SHOW: usize = 4;
    let lines: Vec<&str> = output.lines().filter(|l| !l.trim().is_empty()).collect();
    let style = if is_error { Style::new().fg(Color::Red) } else { dim() };
    let mut out = Vec::new();
    for (i, line) in lines.iter().take(SHOW).enumerate() {
        let prefix = if i == 0 { "  ⎿ " } else { "    " };
        // Tabs, as `read` puts after its line numbers, would take no room on screen.
        let line = line.replace('\t', "  ");
        out.push(Line::from(vec![Span::styled(prefix, dim()), Span::styled(clip(&line, width.saturating_sub(4).max(8)), style)]));
    }
    if lines.len() > SHOW {
        out.push(Line::styled(format!("    … {} more lines", lines.len() - SHOW), dim()));
    }
    out
}

pub(crate) fn clip(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_owned(),
    }
}

/// A call in a line: its path or pattern when it has one, else its input.
pub(crate) fn tool_summary(name: &str, input: &Value) -> String {
    let text = match name {
        "read" => input["path"].as_str().map(str::to_owned),
        "grep" | "glob" => input["pattern"].as_str().map(str::to_owned),
        "run_tool" => input["name"].as_str().map(|name| format!("{name} {}", input["input"])),
        // An MCP tool: what it searches for or opens, when it says.
        _ => ["query", "url", "urls", "path", "name", "text"].iter().find_map(|key| match &input[*key] {
            serde_json::Value::String(text) => Some(text.clone()),
            serde_json::Value::Null => None,
            other => Some(other.to_string()),
        }),
    }
    .unwrap_or_else(|| input.to_string());
    clip(&text.replace('\n', " ⏎ "), 160)
}

/// What the agent is doing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Activity {
    Idle,
    Thinking,
    Generating,
    Running(Vec<String>),
    Compacting,
}

struct Running {
    call_id: String,
    name: String,
    summary: String,
}

/// The display state of one session.
///
/// A reply goes to the scrollback as it streams, not when it ends: the answer paragraph by paragraph (at blank
/// lines outside code blocks, so its markdown renders as a whole). Reasoning stays live while it goes on, its
/// latest lines showing, and goes to the scrollback as one line once the answer (or a tool call) starts. Only
/// the part still being written is live, so text never moves once it is on the screen.
pub(crate) struct View {
    committed: Vec<Item>,
    /// The reply's text so far, and how much of it is already in the scrollback.
    streaming: String,
    streamed: usize,
    /// Its reasoning, and when the reasoning started.
    reasoning: String,
    thinking_since: Option<Instant>,
    running: Vec<Running>,
    pub(crate) activity: Activity,
    /// When the turn started.
    pub(crate) since: Option<Instant>,
}

impl View {
    pub(crate) fn new() -> Self {
        Self {
            committed: Vec::new(),
            streaming: String::new(),
            streamed: 0,
            reasoning: String::new(),
            thinking_since: None,
            running: Vec::new(),
            activity: Activity::Idle,
            since: None,
        }
    }

    pub(crate) fn is_busy(&self) -> bool {
        self.activity != Activity::Idle
    }

    pub(crate) fn push(&mut self, item: Item) {
        self.committed.push(item);
    }

    pub(crate) fn notice(&mut self, level: Level, text: impl Into<String>) {
        self.push(Item::Notice(level, text.into()));
    }

    /// The items finished since the last call, oldest first, for the scrollback.
    pub(crate) fn take_committed(&mut self) -> Vec<Item> {
        std::mem::take(&mut self.committed)
    }

    /// Ends the reasoning: it goes to the scrollback as one line.
    fn collapse_reasoning(&mut self) {
        let reasoning = std::mem::take(&mut self.reasoning);
        let seconds = self.thinking_since.take().map_or(0, |since| since.elapsed().as_secs());
        if let Some(first) = reasoning.lines().map(str::trim).find(|line| !line.is_empty()) {
            self.push(Item::Reasoning { first: first.to_owned(), seconds });
        }
    }

    /// Moves the reply's finished paragraphs to the scrollback; with `all`, the rest too.
    fn commit_text(&mut self, all: bool) {
        let rest = &self.streaming[self.streamed..];
        let end = match all {
            true => rest.len(),
            false => paragraph_end(&self.streaming[..self.streamed], rest),
        };
        if end == 0 {
            return;
        }
        let text = rest[..end].trim_end();
        if !text.trim().is_empty() {
            // Paragraphs keep the blank line between them, as the whole reply rendered would have.
            self.push(Item::Assistant(text.to_owned()));
        }
        self.streamed += end;
    }

    /// Ends the reply being streamed, whatever is left of it going to the scrollback.
    fn finish_reply(&mut self) {
        self.collapse_reasoning();
        self.commit_text(true);
        self.streaming.clear();
        self.streamed = 0;
    }

    /// A stored conversation as items, when a session is resumed.
    pub(crate) fn replay(&mut self, messages: &[Message]) {
        let mut calls = std::collections::HashMap::new();
        for message in messages {
            for content in &message.content {
                match (message.role, content) {
                    (Role::User, Content::Text { text }) => self.push(Item::User(text.clone())),
                    (Role::Assistant, Content::Text { text }) if !text.trim().is_empty() => {
                        self.push(Item::Assistant(text.clone()))
                    }
                    (Role::Assistant, Content::ToolCall { id, name, input }) => {
                        calls.insert(id.clone(), (name.clone(), tool_summary(name, input)));
                    }
                    (Role::Tool, Content::ToolResult { call_id, output, is_error }) => {
                        let (name, summary) = calls.remove(call_id).unwrap_or_else(|| ("tool".into(), String::new()));
                        self.push(Item::Tool { name, summary, output: output.clone(), is_error: *is_error });
                    }
                    _ => {}
                }
            }
        }
    }

    pub(crate) fn apply(&mut self, event: &Event) {
        match event {
            Event::TurnStarted { .. } => {
                self.activity = Activity::Thinking;
                self.since = Some(Instant::now());
            }
            Event::StepStarted { .. } => self.activity = Activity::Thinking,
            Event::ReasoningDelta { text, .. } => {
                self.thinking_since.get_or_insert_with(Instant::now);
                self.reasoning.push_str(text);
                self.activity = Activity::Thinking;
            }
            Event::TextDelta { text, .. } => {
                // The reasoning is over once the answer starts.
                self.collapse_reasoning();
                self.streaming.push_str(text);
                self.commit_text(false);
                self.activity = Activity::Generating;
            }
            Event::MessageAppended { message, .. } if message.role == Role::Assistant => {
                // A reply that did not stream (none do here, but a provider may) is shown whole.
                if self.streaming.is_empty() {
                    self.streaming = message.text();
                }
                self.finish_reply();
            }
            Event::ActionStarted { call_id, name, input, .. } => {
                self.running.push(Running { call_id: call_id.clone(), name: name.clone(), summary: tool_summary(name, input) });
                self.activity = Activity::Running(self.running.iter().map(|r| r.name.clone()).collect());
            }
            Event::ActionResult { call_id, name, output, is_error, .. } => {
                let summary = match self.running.iter().position(|r| &r.call_id == call_id) {
                    Some(i) => self.running.remove(i).summary,
                    None => String::new(),
                };
                self.push(Item::Tool { name: name.clone(), summary, output: output.clone(), is_error: *is_error });
                self.activity = match self.running.is_empty() {
                    true => Activity::Thinking,
                    false => Activity::Running(self.running.iter().map(|r| r.name.clone()).collect()),
                };
            }
            Event::Compacted { tokens_before, tokens_after, .. } => {
                self.notice(Level::Info, format!("context compacted: {} → {} tokens", tokens(*tokens_before), tokens(*tokens_after)));
            }
            Event::TurnCompleted { outcome, error, .. } => {
                // What an interrupted reply got to stays on the screen.
                self.finish_reply();
                self.running.clear();
                self.activity = Activity::Idle;
                self.since = None;
                match outcome {
                    Outcome::Aborted => self.notice(Level::Warn, "interrupted"),
                    Outcome::Failed => {
                        self.notice(Level::Error, format!("error: {}", error.as_deref().unwrap_or("the turn failed")))
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    /// The live part: what is being written and not yet in the scrollback (the reasoning's latest lines, the
    /// answer's unfinished paragraph) and the tools running; at most its last `max` lines.
    pub(crate) fn live_lines(&self, width: usize, max: usize) -> Vec<Line<'static>> {
        let mut out = Vec::new();
        if !self.reasoning.trim().is_empty() {
            let style = dim().add_modifier(Modifier::ITALIC);
            let mut lines = Vec::new();
            for line in self.reasoning.lines().map(str::trim).filter(|line| !line.is_empty()) {
                lines.extend(wrap(vec![Span::styled(format!("∴ {line}"), style)], width, "  "));
            }
            out.extend(lines.into_iter().rev().take(LIVE_REASONING).rev());
        }
        let text = &self.streaming[self.streamed..];
        if !text.trim().is_empty() {
            out.extend(markdown(text, width));
        }
        for running in &self.running {
            out.extend(wrap(
                vec![
                    Span::styled("◌ ", Style::new().fg(Color::Yellow)),
                    Span::styled(running.name.clone(), Style::new().add_modifier(Modifier::BOLD)),
                    Span::raw(" "),
                    Span::styled(running.summary.clone(), dim()),
                ],
                width,
                "  ",
            ));
        }
        if out.len() > max {
            out.drain(..out.len() - max);
        }
        out
    }
}

/// Where the finished paragraphs of `rest` end: after its last blank line that is outside a code block,
/// counting the fences in `before` (the part of the reply already shown) too. 0 when no paragraph has ended.
fn paragraph_end(before: &str, rest: &str) -> usize {
    let mut fences = before.lines().filter(|l| l.trim_start().starts_with("```")).count();
    let mut end = 0;
    let mut at = 0;
    let mut previous_blank = false;
    for line in rest.split_inclusive('\n') {
        at += line.len();
        if !line.ends_with('\n') {
            break; // still being written
        }
        if line.trim_start().starts_with("```") {
            fences += 1;
            // A closed code block ends a paragraph of its own.
            if fences % 2 == 0 {
                end = at;
            }
        } else if line.trim().is_empty() && fences % 2 == 0 && !previous_blank {
            end = at;
        }
        previous_blank = line.trim().is_empty();
    }
    end
}

/// How many lines of reasoning show while it goes on.
const LIVE_REASONING: usize = 6;

/// `12.3k`, `850`.
pub(crate) fn tokens(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        _ => format!("{:.1}k", n as f64 / 1_000.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifatura::Usage;
    use serde_json::json;

    #[test]
    fn events_become_items() {
        let mut view = View::new();
        let turn = "t".to_owned();
        view.apply(&Event::TurnStarted { turn: turn.clone() });
        view.apply(&Event::TextDelta { turn: turn.clone(), text: "Let me look.".into() });
        assert_eq!(view.activity, Activity::Generating);
        assert_eq!(view.live_lines(40, 10).len(), 1);
        view.apply(&Event::MessageAppended { turn: turn.clone(), message: Message::assistant("Let me look."), usage: None });
        view.apply(&Event::ActionStarted {
            turn: turn.clone(),
            call_id: "1".into(),
            name: "grep".into(),
            input: json!({ "pattern": "fn main" }),
        });
        assert_eq!(view.activity, Activity::Running(vec!["grep".into()]));
        view.apply(&Event::ActionResult {
            turn: turn.clone(),
            call_id: "1".into(),
            name: "grep".into(),
            output: "src/main.rs:1: fn main() {\nsrc/bin.rs:3: fn main() {".into(),
            is_error: false,
        });
        view.apply(&Event::TurnCompleted { turn, outcome: Outcome::Failed, error: Some("boom".into()), usage: Usage::default() });
        let items = view.take_committed();
        assert_eq!(items[0], Item::Assistant("Let me look.".into()));
        assert!(matches!(&items[1], Item::Tool { summary, .. } if summary == "fn main"));
        assert_eq!(items[2], Item::Notice(Level::Error, "error: boom".into()));
        assert!(!view.is_busy());
        assert!(items[1].lines(40).len() >= 4);
    }

    #[test]
    fn replies_reach_the_scrollback_as_they_stream() {
        let mut view = View::new();
        let turn = "t".to_owned();
        let delta = |view: &mut View, text: &str, reasoning: bool| {
            let text = text.to_owned();
            let turn = turn.clone();
            view.apply(&match reasoning {
                true => Event::ReasoningDelta { turn, text },
                false => Event::TextDelta { turn, text },
            });
        };
        // Reasoning stays live, all of it, while it goes on.
        delta(&mut view, "First thought.\nSecond", true);
        delta(&mut view, " thought.\nThird", true);
        assert!(view.take_committed().is_empty());
        assert_eq!(view.live_lines(40, 10).len(), 3);
        // Once the answer starts it is one line; the answer goes in paragraph by paragraph, not inside a code block.
        delta(&mut view, "One.\n\n```\na\n\nb\n", false);
        let items = view.take_committed();
        assert_eq!(items, [Item::Reasoning { first: "First thought.".into(), seconds: 0 }, Item::Assistant("One.".into())]);
        assert_eq!(items[0].lines(80).len(), 2, "a blank line, then the one line");
        delta(&mut view, "```\nAfter.", false);
        assert_eq!(view.take_committed(), [Item::Assistant("```\na\n\nb\n```".into())]);
        view.apply(&Event::MessageAppended { turn: turn.clone(), message: Message::assistant("ignored"), usage: None });
        assert_eq!(view.take_committed(), [Item::Assistant("After.".into())]);
        assert!(view.live_lines(40, 10).is_empty());
        // Reasoning that ends in a tool call, with no answer, collapses as well.
        delta(&mut view, "Look at the file.", true);
        view.apply(&Event::MessageAppended { turn, message: Message::new(Role::Assistant, Vec::new()), usage: None });
        assert!(matches!(&view.take_committed()[..], [Item::Reasoning { first, .. }] if first == "Look at the file."));
    }

    #[test]
    fn token_counts_are_short() {
        assert_eq!(tokens(850), "850");
        assert_eq!(tokens(12_345), "12.3k");
    }
}
