//! Write finished rows directly to the terminal. Ratatui owns only the bottom pane.

use std::io::{self, stdout};

use ratatui::backend::{Backend, ClearType};
use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect, Size};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Widget};
use ratatui::{DefaultTerminal, TerminalOptions, Viewport};
use ratatui_crossterm::CrosstermBackend;

use super::App;

pub(super) struct ChatScreen {
    term: DefaultTerminal,
    pane: Rect,
    /// The first unused transcript row. Park the hidden cursor here between frames so
    /// the terminal tracks the end of the transcript even while it reflows during resize.
    next: u16,
}

impl ChatScreen {
    pub(super) fn open() -> io::Result<Self> {
        let term = ratatui::try_init_with_options(TerminalOptions { viewport: Viewport::Fixed(Rect::ZERO) })?;
        Ok(Self { term, pane: Rect::ZERO, next: 0 })
    }

    fn position_pane(&mut self, pane: Rect) -> io::Result<()> {
        if self.pane == pane {
            return Ok(());
        }
        let size_changed = self.pane.width != pane.width || self.pane.bottom() != pane.bottom();
        if size_changed {
            self.next = self.term.get_cursor_position()?.y;
        }
        let backend = self.term.backend_mut();
        backend.set_cursor_position(Position::new(0, self.next))?;
        backend.clear_region(ClearType::AfterCursor)?;
        if self.next > pane.y {
            // All transient content was cleared. These newlines move just the history
            // that no longer fits into the terminal's own scrollback.
            backend.set_cursor_position(Position::new(0, pane.bottom().saturating_sub(1)))?;
            backend.append_lines(self.next - pane.y)?;
            self.next = pane.y;
        }
        // Terminal::resize clears the entire display on width shrink, including native history.
        self.term = DefaultTerminal::with_options(CrosstermBackend::new(stdout()), TerminalOptions { viewport: Viewport::Fixed(pane) })?;
        self.pane = pane;
        Ok(())
    }

    pub(super) fn draw(&mut self, app: &mut App) -> io::Result<()> {
        let Size { width, height } = self.term.size()?;
        if width == 0 || height == 0 {
            return Ok(());
        }
        let dock = app.pane_height(width, height);
        self.position_pane(Rect::new(0, height - dock, width, dock))?;
        let items = app.view.take_committed();
        for item in items {
            self.write_history(item.lines(usize::from(width)))?;
        }
        self.term.draw(|f| app.draw(f))?;
        self.term.set_cursor_position((0, self.next))?;
        Ok(())
    }

    fn write_history(&mut self, lines: Vec<Line<'static>>) -> io::Result<()> {
        let room = self.pane.y;
        if room < 2 {
            // In a one- or two-row terminal, scroll the whole display. Erase the
            // composer first so its contents never become transcript rows.
            self.term.draw(|_| {})?;
        }
        let backend = self.term.backend_mut();
        for chunk in lines.chunks(usize::from(room.max(1))) {
            let count = chunk.len() as u16;
            let overflow = count.saturating_sub(room.max(1) - self.next);
            if overflow > 0 {
                if room == 1 {
                    backend.set_cursor_position(Position::new(0, self.pane.bottom() - 1))?;
                    backend.append_lines(overflow)?;
                } else {
                    backend.scroll_region_up(0..room, overflow)?;
                }
                self.next -= overflow;
            }
            let area = Rect::new(0, self.next, self.pane.width, count);
            let empty = Buffer::empty(area);
            let mut written = empty.clone();
            Paragraph::new(chunk.to_vec()).render(area, &mut written);
            backend.draw(empty.diff_iter(&written))?;
            if room == 0 {
                backend.set_cursor_position(Position::new(0, self.pane.bottom() - 1))?;
                backend.append_lines(count)?;
            } else {
                self.next += count;
            }
        }
        Ok(())
    }

    pub(super) fn finish(&mut self) -> io::Result<()> {
        self.term.set_cursor_position((0, self.next))?;
        self.term.backend_mut().clear_region(ClearType::AfterCursor)
    }
}
