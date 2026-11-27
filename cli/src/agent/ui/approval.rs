//! The focused tool approval that temporarily occupies the chat composer.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use manifatura::{ToolCall, kit::Answer};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use tokio::sync::oneshot;

use super::text::plain;
use super::view::{clip, copper, dim, user_message_style};

const CHOICES: [(&str, &str, Answer); 3] = [
    ("Allow once (y)", "Once (y)", Answer::Allow),
    ("Allow this tool for the session (a)", "Allow tool for session (a)", Answer::AllowAlways),
    ("Deny (n)", "Deny (n)", Answer::Deny(None)),
];

pub(crate) struct Ask {
    call: ToolCall,
    /// The subagent that asks, when it is not the chat itself.
    from: Option<String>,
    pub(super) reply: oneshot::Sender<Answer>,
    selected: usize,
}

impl Ask {
    pub(crate) fn new(call: ToolCall, reply: oneshot::Sender<Answer>) -> Self {
        Self { call, from: None, reply, selected: 0 }
    }

    /// The question as a subagent's.
    pub(crate) fn from(mut self, subagent: Option<String>) -> Self {
        self.from = subagent;
        self
    }

    pub(super) fn key(&mut self, key: KeyEvent) -> Option<Answer> {
        if key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER) {
            return None;
        }
        let choice = match key.code {
            KeyCode::Up => {
                self.selected = self.selected.checked_sub(1).unwrap_or(CHOICES.len() - 1);
                return None;
            }
            KeyCode::Down => {
                self.selected = (self.selected + 1) % CHOICES.len();
                return None;
            }
            _ if key.kind != KeyEventKind::Press => return None,
            KeyCode::Enter => self.selected,
            KeyCode::Char('1' | 'y' | 'Y') => 0,
            KeyCode::Char('2' | 'a' | 'A') => 1,
            KeyCode::Char('3' | 'n' | 'N') | KeyCode::Esc => 2,
            _ => return None,
        };
        Some(CHOICES[choice].2.clone())
    }

    fn target(&self) -> String {
        let input = &self.call.input;
        let field = match self.call.name.as_str() {
            "read" => "path",
            "glob" | "grep" => "pattern",
            _ => return input.to_string(),
        };
        if input.as_object().is_some_and(|fields| fields.len() == 1) && let Some(value) = input[field].as_str() {
            return value.into();
        }
        input.to_string()
    }

    fn lines(&self, width: u16, height: u16) -> Vec<Line<'static>> {
        let height = usize::from(height);
        let width = usize::from(width);
        if height == 0 {
            return Vec::new();
        }
        let title = match self.call.name.as_str() {
            "read" => "Read this file?".to_owned(),
            "grep" => "Search file contents?".to_owned(),
            "glob" => "Find matching files?".to_owned(),
            name => format!("Run {name}?"),
        };
        let title = match &self.from {
            Some(subagent) => format!("{subagent} · {title}"),
            None => title,
        };
        let target = plain(&self.target(), width.saturating_sub(3), Style::new());
        // Short terminals keep the selected choice visible. On larger ones,
        // spend the remaining rows on the target before adding blank padding.
        let title_rows = usize::from(height >= 2);
        let target_min = usize::from(height >= 3);
        let hint_rows = usize::from(height >= 4);
        let choices = CHOICES.len().min(height - title_rows - target_min - hint_rows);
        let target_rows = target.len().min(height - title_rows - hint_rows - choices);
        let spare = height - title_rows - target_rows - hint_rows - choices;
        let mut rows = Vec::new();
        if spare >= 2 {
            rows.push(Line::default());
        }
        if title_rows > 0 {
            rows.push(Line::styled(format!("  {}", clip(&title, width.saturating_sub(2))), Style::new().add_modifier(Modifier::BOLD)));
        }
        let truncated = target.len() > target_rows;
        for (i, mut line) in target.into_iter().take(target_rows).enumerate() {
            line.spans.insert(0, Span::raw("  "));
            if truncated && i + 1 == target_rows {
                line.push_span("…");
            }
            rows.push(line);
        }
        if spare >= 1 {
            rows.push(Line::default());
        }
        let first = (self.selected + 1).saturating_sub(choices);
        for (i, (full, short, _)) in CHOICES.iter().enumerate().skip(first).take(choices) {
            let (mark, style) = if i == self.selected { ("›", copper()) } else { (" ", Style::new()) };
            let label = if full.len() + 5 <= width { full } else { short };
            rows.push(Line::styled(format!("{mark} {}. {}", i + 1, clip(label, width.saturating_sub(5))), style));
        }
        if spare >= 3 {
            rows.push(Line::default());
        }
        if hint_rows > 0 {
            let hint = if width >= 38 { "  ↑↓ select · Enter confirm · Esc deny" } else { "  ↑↓ select · Enter · Esc" };
            rows.push(Line::styled(hint, dim()));
        }
        rows
    }

    pub(super) fn height(&self, width: u16, limit: u16) -> u16 {
        self.lines(width, limit).len() as u16
    }

    pub(super) fn draw(&self, f: &mut Frame, area: Rect) {
        let lines = self.lines(area.width, area.height);
        let height = lines.len() as u16;
        let area = Rect { y: area.bottom() - height, height, ..area };
        let body = Rect { height: height.saturating_sub(u16::from(height >= 4)), ..area };
        f.render_widget(Block::new().style(user_message_style()), body);
        f.render_widget(Paragraph::new(lines), area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_approval_keeps_the_tool_arguments_and_selected_choice_visible() {
        let call = ToolCall { id: "1".into(), name: "lookup".into(), input: serde_json::json!({"id": 42}) };
        let mut ask = Ask::new(call, oneshot::channel().0);
        assert_eq!(ask.key(KeyEvent::from(KeyCode::Up)), None);
        let rows: Vec<_> = ask.lines(30, 4).iter().map(Line::to_string).collect();
        assert_eq!(rows, ["  Run lookup?", "  {\"id\":42}", "› 3. Deny (n)", "  ↑↓ select · Enter · Esc"]);
        for (key, answer) in [('1', Answer::Allow), ('2', Answer::AllowAlways), ('3', Answer::Deny(None))] {
            assert_eq!(ask.key(KeyEvent::from(KeyCode::Char(key))), Some(answer));
        }
        assert_eq!(ask.key(KeyEvent::from(KeyCode::Esc)), Some(Answer::Deny(None)));
        let ask = ask.from(Some("explore".into()));
        assert!(ask.lines(60, 8).iter().any(|line| line.to_string().contains("explore · ")));
    }
}
