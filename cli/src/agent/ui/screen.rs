//! Write finished rows directly to the terminal. Ratatui owns only the bottom pane.

use std::io::{self, stdout};

use ratatui::backend::{Backend, ClearType};
use ratatui::buffer::Buffer;
use ratatui::layout::{Position, Rect, Size};
use ratatui::text::Line;
use ratatui::widgets::{Paragraph, Widget};
use ratatui::{DefaultTerminal, TerminalOptions, Viewport};
use ratatui_crossterm::CrosstermBackend;

use super::{App, Item};

pub(super) struct ChatScreen {
    term: DefaultTerminal,
    pane: Rect,
    transcript: Transcript,
}

#[derive(Default)]
struct Transcript {
    /// The first uncommitted row. The hidden cursor parks here so the terminal
    /// tracks it while reflowing history on resize.
    next: u16,
    /// The last preview drawn from that row, for diffing the next frame.
    preview: Buffer,
}

impl ChatScreen {
    pub(super) fn open() -> io::Result<Self> {
        let term = ratatui::try_init_with_options(TerminalOptions { viewport: Viewport::Fixed(Rect::ZERO) })?;
        Ok(Self { term, pane: Rect::ZERO, transcript: Transcript::default() })
    }

    fn position_pane(&mut self, pane: Rect) -> io::Result<()> {
        if self.pane == pane {
            return Ok(());
        }
        let size_changed = self.pane.width != pane.width || self.pane.bottom() != pane.bottom();
        if size_changed {
            self.transcript.next = self.term.get_cursor_position()?.y;
        }
        let backend = self.term.backend_mut();
        backend.set_cursor_position(Position::new(0, self.transcript.next))?;
        backend.clear_region(ClearType::AfterCursor)?;
        if self.transcript.next > pane.y {
            // All transient content was cleared. These newlines move just the history
            // that no longer fits into the terminal's own scrollback.
            backend.set_cursor_position(Position::new(0, pane.bottom().saturating_sub(1)))?;
            backend.append_lines(self.transcript.next - pane.y)?;
            self.transcript.next = pane.y;
        }
        // Terminal::resize clears the entire display on width shrink, including native history.
        self.term = DefaultTerminal::with_options(CrosstermBackend::new(stdout()), TerminalOptions { viewport: Viewport::Fixed(pane) })?;
        self.transcript.preview = Buffer::empty(Rect::ZERO);
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
        let live = app.transcript_preview(width.into(), self.pane.y.into());
        if self.pane.y < 2 && (!items.is_empty() || live.len() > usize::from(self.pane.y - self.transcript.next)) {
            // These terminals have to scroll the entire display, including the editor.
            self.term.draw(|_| {})?;
        }
        self.transcript.draw(self.term.backend_mut(), self.pane, items, live)?;
        self.term.draw(|f| app.draw(f))?;
        self.term.set_cursor_position((0, self.transcript.next))?;
        Ok(())
    }

    pub(super) fn finish(&mut self) -> io::Result<()> {
        self.term.set_cursor_position((0, self.transcript.next))?;
        self.term.backend_mut().clear_region(ClearType::AfterCursor)
    }
}

impl Transcript {
    fn draw<B: Backend>(&mut self, backend: &mut B, pane: Rect, items: Vec<Item>, live: Vec<Line<'static>>) -> Result<(), B::Error> {
        let height = live.len() as u16;
        if !items.is_empty() || height > pane.y - self.next {
            // Reclaim the live rows before committing an item or scrolling the transcript.
            self.paint_preview(backend, pane.width, Vec::new())?;
        }
        for item in items {
            self.write_history(backend, pane, item.lines(pane.width.into()))?;
        }
        self.make_room(backend, pane, height)?;
        self.paint_preview(backend, pane.width, live)
    }

    fn make_room<B: Backend>(&mut self, backend: &mut B, pane: Rect, count: u16) -> Result<(), B::Error> {
        let overflow = count.saturating_sub(pane.y.max(1) - self.next);
        if overflow > 0 {
            if pane.y == 1 {
                backend.set_cursor_position(Position::new(0, pane.bottom() - 1))?;
                backend.append_lines(overflow)?;
            } else {
                backend.scroll_region_up(0..pane.y, overflow)?;
            }
            self.next -= overflow;
        }
        Ok(())
    }

    /// Redraw only changed cells. Include old trailing rows in the diff when a
    /// paragraph shrinks or reasoning collapses into a summary.
    fn paint_preview<B: Backend>(&mut self, backend: &mut B, width: u16, live: Vec<Line<'static>>) -> Result<(), B::Error> {
        let height = live.len() as u16;
        let rows = height.max(self.preview.area.height);
        if rows == 0 {
            return Ok(());
        }
        let area = Rect::new(0, self.next, width, rows);
        self.preview.resize(area);
        let mut frame = Buffer::empty(area);
        Paragraph::new(live).render(area, &mut frame);
        backend.draw(self.preview.diff_iter(&frame))?;
        frame.resize(Rect { height, ..area });
        self.preview = frame;
        Ok(())
    }

    fn write_history<B: Backend>(&mut self, backend: &mut B, pane: Rect, lines: Vec<Line<'static>>) -> Result<(), B::Error> {
        let room = pane.y;
        for chunk in lines.chunks(usize::from(room.max(1))) {
            let count = chunk.len() as u16;
            self.make_room(backend, pane, count)?;
            let area = Rect::new(0, self.next, pane.width, count);
            let empty = Buffer::empty(area);
            let mut written = empty.clone();
            Paragraph::new(chunk.to_vec()).render(area, &mut written);
            backend.draw(empty.diff_iter(&written))?;
            if room == 0 {
                backend.set_cursor_position(Position::new(0, pane.bottom() - 1))?;
                backend.append_lines(count)?;
            } else {
                self.next += count;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use manifatura::{Event, Message};
    use ratatui::backend::TestBackend;

    fn row(buffer: &Buffer, y: u16) -> String {
        (0..buffer.area.width).map(|x| buffer[(x, y)].symbol()).collect::<String>().trim_end().into()
    }

    #[test]
    fn live_text_and_committed_text_occupy_the_same_rows() {
        let mut app = super::super::tests::loading_app();
        let pane = Rect::new(0, 20, 80, 4);
        let mut backend = TestBackend::new(80, 24);
        let mut transcript = Transcript::default();
        let mut draw = |app: &mut App, backend: &mut TestBackend| {
            let live = app.transcript_preview(80, 20);
            transcript.draw(backend, pane, app.view.take_committed(), live).unwrap();
        };
        let user = Item::User("question".into());
        let first = user.lines(80).len() as u16 + 1;
        app.view.push(user);
        app.view.apply(&Event::ReasoningDelta { turn: "t".into(), text: "start\nsecond\nthird".into() });
        draw(&mut app, &mut backend);
        assert_eq!(row(backend.buffer(), first), "∴ start");

        app.view.apply(&Event::TextDelta { turn: "t".into(), text: "The answer".into() });
        draw(&mut app, &mut backend);
        assert!(row(backend.buffer(), first).contains("Thought for"));
        let answer = first + 2;
        assert_eq!(row(backend.buffer(), answer), "The answer");
        assert_eq!(row(backend.buffer(), first + 1), "");

        app.view.apply(&Event::TextDelta { turn: "t".into(), text: " is here.\n\nNext".into() });
        draw(&mut app, &mut backend);
        assert_eq!(row(backend.buffer(), answer), "The answer is here.");
        assert_eq!(row(backend.buffer(), answer + 2), "Next");
        app.view.apply(&Event::MessageAppended { turn: "t".into(), message: Message::assistant("The answer is here.\n\nNext"), usage: None });
        draw(&mut app, &mut backend);
        assert_eq!(row(backend.buffer(), answer + 2), "Next");
        backend.assert_scrollback_empty();
    }

    #[test]
    fn an_overflowing_preview_enters_scrollback_only_when_committed() {
        let pane = Rect::new(0, 5, 50, 4);
        let mut backend = TestBackend::new(50, 9);
        let mut transcript = Transcript::default();
        let item = Item::Assistant((0..9).map(|i| format!("code line {i}")).collect::<Vec<_>>().join("\n\n"));
        let lines = item.lines(50);
        let tail = lines[lines.len() - 5..].to_vec();
        transcript.draw(&mut backend, pane, vec![Item::Notice(super::super::Level::Info, "earlier".into())], tail).unwrap();
        assert_eq!(row(backend.buffer(), 4), "code line 8");
        assert_eq!(row(backend.scrollback(), 1), "earlier");
        transcript.draw(&mut backend, pane, vec![item], Vec::new()).unwrap();
        let saved: Vec<_> = (0..backend.scrollback().area.height).map(|y| row(backend.scrollback(), y))
            .chain((0..pane.y).map(|y| row(backend.buffer(), y))).filter(|s| s.starts_with("code line")).collect();
        assert_eq!(saved, (0..9).map(|i| format!("code line {i}")).collect::<Vec<_>>());
    }
}
