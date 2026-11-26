//! The editable rows below the terminal's scrollback.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Padding};
use tracing::Level as LogLevel;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::view::{clip, copper, dim, prompt_text_width, tokens, user_message_style};
use super::{Activity, App, LOG_BUTTON, LOG_PANEL, SPINNER, ago};

/// Take rows from the top of the remaining area.
fn take(area: &mut Rect, count: u16) -> Rect {
    let height = count.min(area.height);
    let taken = Rect { height, ..*area };
    area.y += height;
    area.height -= height;
    taken
}

/// Keep the filename visible when its parent directories exceed the menu width.
fn path_label(path: &str, width: usize) -> String {
    if path.width() <= width {
        return path.to_owned();
    }
    let mut room = width.saturating_sub(1);
    let mut start = path.len();
    for (i, ch) in path.char_indices().rev() {
        let cells = ch.width().unwrap_or(0);
        if cells > room {
            break;
        }
        room -= cells;
        start = i;
    }
    format!("…{}", &path[start..])
}

impl App {
    pub(super) fn pane_height(&self, width: u16, height: u16) -> u16 {
        let limit = (height / 2).clamp(6, 12).min(height.saturating_sub(2).max(1));
        if let Some(ask) = self.asks.front() {
            return ask.height(width, limit);
        }
        if self.resume_pick.is_some() {
            return (self.sessions.len() + 2).min(usize::from(limit)) as u16;
        }
        let editor = self.editor_rows(width);
        let logs = if self.logs_open { self.log.len().clamp(1, LOG_PANEL) + 1 } else { 0 };
        let extras = logs + self.completions().len() + usize::from(self.files.hint().is_some());
        // Leave a blank row between the transcript and the controls when space permits.
        (editor + 2 + extras.min(usize::from(limit)) as u16).min(limit)
    }

    fn editor_rows(&self, width: u16) -> u16 {
        let columns = prompt_text_width(width.into());
        let mut rows = 0;
        for line in self.input.lines() {
            rows += line.width().div_ceil(columns).max(1);
            if rows >= 6 {
                return 8;
            }
        }
        rows as u16 + 2
    }

    pub(super) fn draw(&mut self, f: &mut Frame) {
        let mut rows = f.area();
        if let Some(ask) = self.asks.front() {
            self.log_button = None;
            ask.draw(f, rows);
            return;
        }
        if let Some(selected) = self.resume_pick {
            self.draw_sessions(f, rows, selected);
            return;
        }
        let width = rows.width as usize;
        // Budget the editor first. Suggestions and logs only use spare rows.
        let reserved = if self.files.active() && rows.height >= 4 { 2 } else { 0 };
        let editor = self.editor_rows(rows.width).min(rows.height - reserved);
        let mut spare = rows.height - editor;
        let footer = spare.min(1);
        spare -= footer;
        let completions = self.completions();
        let hint = self.files.hint();
        let menu = (completions.len() + usize::from(hint.is_some())).min(spare as usize) as u16;
        spare -= menu;
        let logs = if self.logs_open { (1 + self.log.len().clamp(1, LOG_PANEL)).min(spare as usize) as u16 } else { 0 };
        take(&mut rows, spare - logs);
        let picked = self.picked.min(completions.len().saturating_sub(1));
        let first = (picked + 1).saturating_sub(menu as usize);
        if let Some(hint) = hint && menu > 0 {
            f.render_widget(Line::styled(format!("  {hint}"), dim()), take(&mut rows, 1));
        }
        for (i, completion) in completions.iter().enumerate().skip(first).take(menu as usize) {
            let (mark, style) = if i == picked { ("› ", copper()) } else { ("  ", Style::new()) };
            let line = if self.files.active() {
                Line::styled(format!("{mark}{}", path_label(&completion.label, width.saturating_sub(2))), style)
            } else {
                Line::from(vec![
                    Span::styled(format!("{mark}{:<18}", completion.label), style),
                    Span::styled(completion.about.as_str(), dim()),
                ])
            };
            f.render_widget(line, take(&mut rows, 1));
        }
        if logs > 0 {
            f.render_widget(Line::styled("─ Log · Ctrl-L to close", dim()), take(&mut rows, 1));
            if self.log.is_empty() && logs > 1 {
                f.render_widget(Line::styled("  Nothing logged yet", dim()), take(&mut rows, 1));
            }
            for line in self.log.iter().skip(self.log.len().saturating_sub(usize::from(logs - 1))) {
                let style = match line.level {
                    LogLevel::ERROR => Style::new().fg(Color::Red),
                    LogLevel::WARN => Style::new().fg(Color::Yellow),
                    _ => dim(),
                };
                f.render_widget(Line::styled(clip(&line.text, width), style), take(&mut rows, 1));
            }
        }
        self.draw_composer(f, take(&mut rows, editor));
        self.draw_footer(f, take(&mut rows, footer));
    }

    fn draw_sessions(&mut self, f: &mut Frame, mut rows: Rect, selected: usize) {
        self.log_button = None;
        let needed = (self.sessions.len() + 2).min(rows.height.into()) as u16;
        let unused = rows.height - needed;
        take(&mut rows, unused);
        f.render_widget(Line::styled("Resume a session", copper()), take(&mut rows, 1));
        let count = usize::from(rows.height.saturating_sub(1));
        let first = (selected + 1).saturating_sub(count);
        let current = self.ready.as_ref().map(|ready| ready.session.id());
        for (i, session) in self.sessions.iter().enumerate().skip(first).take(count) {
            let (mark, style) = if selected == i { ("› ", copper()) } else { ("  ", Style::new()) };
            let current = if current == Some(&session.id) { " · current" } else { "" };
            let age = format!("  {}{current}", ago(session.updated));
            let available = (rows.width as usize).saturating_sub(2 + age.len());
            let row = Line::from(vec![
                Span::styled(mark, style),
                Span::styled(clip(session.id.as_str(), available), style),
                Span::styled(age, dim()),
            ]);
            f.render_widget(row, take(&mut rows, 1));
        }
        f.render_widget(Line::styled("↑↓ select · Enter resume · Esc cancel", dim()), take(&mut rows, 1));
    }

    fn draw_composer(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::new().style(user_message_style()).padding(Padding::vertical(u16::from(area.height >= 3)));
        let inner = block.inner(area);
        f.render_widget(block, area);
        let gutter = inner.width.min(2);
        f.render_widget(Span::styled("›", Style::new().add_modifier(Modifier::BOLD)), inner);
        let width = (prompt_text_width(inner.width.into()) as u16).min(inner.width - gutter);
        if width == 0 {
            return;
        }
        let placeholder = if self.view.is_busy() { "Send a follow-up…" } else { "Ask Rustling…" };
        self.input.set_placeholder_text(placeholder);
        f.render_widget(&self.input, Rect { x: inner.x + gutter, width, ..inner });
    }

    /// Use the same leading separator as a finished transcript item, space permitting.
    pub(super) fn transcript_preview(&self, width: usize, available: usize) -> Vec<Line<'static>> {
        if available == 0 {
            return Vec::new();
        }
        let mut lines = self.view.live_lines(width, available);
        if lines.is_empty() {
            let activity = self.activity_line(width);
            if activity.spans.is_empty() {
                return lines;
            }
            lines.push(activity);
        }
        if lines.len() < available {
            lines.insert(0, Line::default());
        }
        lines
    }

    fn activity_line(&self, width: usize) -> Line<'static> {
        if !self.asks.is_empty() {
            return Line::default();
        }
        if self.armed {
            return Line::styled("Press Ctrl-C again to quit", Style::new().fg(Color::Yellow));
        }
        let label = if self.ready.is_none() {
            "Loading model…".into()
        } else {
            match self.view.activity {
                Activity::Idle => return Line::default(),
                Activity::Thinking => "Thinking".into(),
                Activity::Generating => "Writing".into(),
                Activity::Running => self.view.running_label(),
                Activity::Compacting => "Compacting".into(),
            }
        };
        let time = self.view.since.map_or_else(String::new, |since| format!(" · {}s", since.elapsed().as_secs()));
        Line::from(vec![
            Span::styled(format!("{} ", SPINNER[self.tick % SPINNER.len()]), copper()),
            Span::raw(clip(&label, width.saturating_sub(2 + Line::raw(&time).width()))),
            Span::styled(time, dim()),
        ])
    }

    fn draw_footer(&mut self, f: &mut Frame, area: Rect) {
        self.log_button = None;
        if area.is_empty() {
            return;
        }
        let button_width = LOG_BUTTON.len() as u16;
        let body = if self.logs && area.width > button_width + 10 {
            let button = Rect { x: area.right() - button_width, width: button_width, ..area };
            f.render_widget(Line::styled(LOG_BUTTON, Style::new().add_modifier(Modifier::REVERSED)), button);
            self.log_button = Some(button);
            Rect { width: area.width - button_width - 1, ..area }
        } else {
            area
        };
        let context = match self.last {
            Some(usage) => {
                let used = usage.prompt_tokens as u64 + usage.completion_tokens as u64;
                format!("{}% context used", used * 100 / self.window.max(1) as u64)
            }
            None => format!("{} context", tokens(self.window as u64)),
        };
        let mut left = format!("{} effort · {context}", self.effort_label());
        if self.yes {
            left.push_str(" · auto-allow");
        }
        let hint = if self.files.active() {
            "↑↓ select · Tab/Enter insert · Esc close"
        } else if self.view.is_busy() && body.width < 65 {
            "Esc interrupt"
        } else if self.view.is_busy() {
            "Enter follow-up · Esc interrupt"
        } else if !self.completions().is_empty() {
            "↑↓ select · Tab complete · Esc close"
        } else {
            "@ files · / commands"
        };
        let gap = (body.width as usize).saturating_sub(Line::raw(&left).width() + Line::raw(hint).width());
        if gap > 0 {
            left.push_str(&" ".repeat(gap));
            left.push_str(hint);
        } else if self.view.is_busy() || self.files.active() {
            left = hint.to_owned();
        }
        f.render_widget(Line::styled(left, dim()), body);
    }
}
