//! The editable rows below the terminal's scrollback.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use tracing::Level as LogLevel;

use super::view::{clip, copper, dim, tokens, tool_summary};
use super::{Activity, App, Ask, LOG_BUTTON, LOG_PANEL, SPINNER, text};

/// Take rows from the top of the remaining area.
fn take(area: &mut Rect, count: u16) -> Rect {
    let height = count.min(area.height);
    let taken = Rect { height, ..*area };
    area.y += height;
    area.height -= height;
    taken
}

fn approval(ask: &Ask, width: usize, height: usize) -> Vec<Line<'static>> {
    let amber = Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD);
    let action = match ask.call.name.as_str() {
        "read" => "file read",
        other => other,
    };
    let mut lines = vec![Line::styled(format!("▏ ? Allow {action}?"), amber)];
    let choices = if width >= 43 { "[y] Once  [a] Always this session  [n] Deny" } else { "[y] Once  [a] Session  [n] Deny" };
    let choices = text::wrap(vec![Span::styled(choices, amber)], width, "  ");
    let summary = tool_summary(&ask.call.name, &ask.call.input);
    let room = height.saturating_sub(lines.len() + choices.len());
    let target = text::wrap(vec![Span::raw(format!("  {summary}"))], width, "  ");
    lines.extend(target.into_iter().take(room));
    lines.extend(choices);
    lines
}

impl App {
    pub(super) fn draw(&mut self, f: &mut Frame) {
        let mut rows = f.area();
        let width = rows.width as usize;
        // Budget the editor or approval first. Suggestions and logs only use spare rows.
        let question = self.asks.front().map(|ask| approval(ask, width, rows.height as usize));
        let editor = question.as_ref().map_or_else(
            || self.input.lines().len().clamp(1, 6) + 2,
            Vec::len,
        ).min(rows.height as usize) as u16;
        let mut spare = rows.height - editor;
        let footer = spare.min(1);
        spare -= footer;
        let activity = self.activity_line(width);
        let activity_rows = u16::from(!activity.spans.is_empty()).min(spare);
        spare -= activity_rows;
        let completions = self.completions();
        let menu = completions.len().min(spare as usize) as u16;
        spare -= menu;
        let logs = if self.logs_open { (1 + self.log.len().clamp(1, LOG_PANEL)).min(spare as usize) as u16 } else { 0 };
        let live = self.view.live_lines(width, (spare - logs) as usize);
        let live_area = take(&mut rows, live.len() as u16);
        f.render_widget(Paragraph::new(live), live_area);
        f.render_widget(activity, take(&mut rows, activity_rows));

        let edit_area = take(&mut rows, editor);
        if let Some(lines) = question {
            f.render_widget(Paragraph::new(lines), edit_area);
        } else {
            self.draw_composer(f, edit_area);
        }
        let picked = self.picked.min(completions.len().saturating_sub(1));
        let first = (picked + 1).saturating_sub(menu as usize);
        for (i, completion) in completions.iter().enumerate().skip(first).take(menu as usize) {
            let (mark, style) = if i == picked { ("› ", copper()) } else { ("  ", Style::new()) };
            let line = Line::from(vec![
                Span::styled(format!("{mark}{:<18}", completion.label), style),
                Span::styled(completion.about.as_str(), dim()),
            ]);
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
        self.draw_footer(f, take(&mut rows, footer));
    }

    fn draw_composer(&mut self, f: &mut Frame, area: Rect) {
        let block = Block::new().borders(Borders::TOP | Borders::BOTTOM).border_style(dim());
        let inner = if area.height >= 3 {
            let inner = block.inner(area);
            f.render_widget(block, area);
            inner
        } else {
            area
        };
        let gutter = inner.width.min(2);
        f.render_widget(Span::styled("›", copper()), inner);
        let placeholder = if self.view.is_busy() { "Send a follow-up…" } else { "Ask Rustling…" };
        self.input.set_placeholder_text(placeholder);
        f.render_widget(&self.input, Rect { x: inner.x + gutter, width: inner.width - gutter, ..inner });
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
        let hint = if !self.asks.is_empty() {
            ""
        } else if self.view.is_busy() && body.width < 65 {
            "Esc interrupt"
        } else if self.view.is_busy() {
            "Enter follow-up · Esc interrupt"
        } else {
            "/ commands"
        };
        let gap = (body.width as usize).saturating_sub(Line::raw(&left).width() + Line::raw(hint).width());
        if gap > 0 {
            left.push_str(&" ".repeat(gap));
            left.push_str(hint);
        } else if self.view.is_busy() && self.asks.is_empty() {
            left = hint.to_owned();
        }
        f.render_widget(Line::styled(left, dim()), body);
    }
}
