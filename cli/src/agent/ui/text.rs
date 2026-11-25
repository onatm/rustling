//! Wrapping styled text to lines of a known count, which writing into the terminal's scrollback needs, and
//! markdown through `tui-markdown`.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// Wraps spans to `width` columns, breaking at spaces and splitting words longer than a line. `indent` starts
/// each continuation line.
pub(crate) fn wrap(spans: Vec<Span<'static>>, width: usize, indent: &str) -> Vec<Line<'static>> {
    let width = width.max(8);
    let indent_width = indent.width();
    let mut lines = Vec::new();
    let mut line: Vec<Span<'static>> = Vec::new();
    let mut column = 0;
    let mut new_line = |line: &mut Vec<Span<'static>>, column: &mut usize| {
        lines.push(tidy(std::mem::take(line)));
        line.push(Span::raw(indent.to_owned()));
        *column = indent_width;
    };
    for span in spans {
        let style = span.style;
        for token in tokens(&span.content) {
            let token_width = token.width();
            if column + token_width <= width {
                column += token_width;
                line.push(Span::styled(token, style));
            } else if token.starts_with(' ') {
                // A break: the spaces go.
                new_line(&mut line, &mut column);
            } else if token_width <= width - indent_width && column > indent_width {
                new_line(&mut line, &mut column);
                column += token_width;
                line.push(Span::styled(token, style));
            } else {
                // A word longer than a line is split.
                let mut piece = String::new();
                for c in token.chars() {
                    let c_width = c.width().unwrap_or(0);
                    if column + c_width > width {
                        line.push(Span::styled(std::mem::take(&mut piece), style));
                        new_line(&mut line, &mut column);
                    }
                    piece.push(c);
                    column += c_width;
                }
                line.push(Span::styled(piece, style));
            }
        }
    }
    lines.push(tidy(line));
    lines
}

/// Runs of spaces and runs of everything else, alternating.
fn tokens(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut space = None;
    for c in text.chars() {
        let is_space = c == ' ';
        match out.last_mut() {
            Some(token) if space == Some(is_space) => token.push(c),
            _ => out.push(c.to_string()),
        }
        space = Some(is_space);
    }
    out
}

/// Drops empty spans and trailing spaces, and merges neighbours of the same style.
fn tidy(spans: Vec<Span<'static>>) -> Line<'static> {
    let mut out: Vec<Span<'static>> = Vec::new();
    for span in spans.into_iter().filter(|s| !s.content.is_empty()) {
        match out.last_mut() {
            Some(last) if last.style == span.style => last.content = format!("{}{}", last.content, span.content).into(),
            _ => out.push(span),
        }
    }
    while let Some(last) = out.last_mut() {
        let trimmed = last.content.trim_end().to_owned();
        if trimmed.is_empty() {
            out.pop();
        } else {
            last.content = trimmed.into();
            break;
        }
    }
    Line::from(out)
}

/// Plain text in one style, wrapped, each of its lines on its own.
pub(crate) fn plain(text: &str, width: usize, style: Style) -> Vec<Line<'static>> {
    text.split('\n').flat_map(|line| wrap(vec![Span::styled(line.to_owned(), style)], width, "")).collect()
}

/// Markdown as wrapped lines: headings, lists, quotes, code and inline styles, as `tui-markdown` renders them.
pub(crate) fn markdown(text: &str, width: usize) -> Vec<Line<'static>> {
    let rendered = tui_markdown::from_str(text);
    let mut out = Vec::new();
    for line in rendered.lines {
        let spans: Vec<Span<'static>> = line
            .spans
            .into_iter()
            .map(|span| Span::styled(span.content.into_owned(), line.style.patch(span.style)))
            .collect();
        // Continuation lines of a list item or quote keep its indent.
        let indent: String = spans.first().map_or(String::new(), |s| {
            let lead = s.content.len() - s.content.trim_start_matches([' ', '>', '-', '*', '•']).len();
            " ".repeat(s.content[..lead].width())
        });
        out.extend(wrap(spans, width, &indent));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line]) -> Vec<String> {
        lines.iter().map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect()).collect()
    }

    #[test]
    fn words_wrap_at_spaces_and_long_ones_split() {
        let lines = wrap(vec![Span::raw("the quick brown fox")], 10, "  ");
        assert_eq!(text(&lines), ["the quick", "  brown", "  fox"]);
        let lines = wrap(vec![Span::raw("abcdefghijkl")], 8, "");
        assert_eq!(text(&lines), ["abcdefgh", "ijkl"]);
    }

    #[test]
    fn markdown_keeps_its_words() {
        let lines = markdown("# Title\n\nSome **bold** text and `code`.", 40);
        let all = text(&lines).join("\n");
        assert!(all.contains("Title") && all.contains("bold") && all.contains("code"), "{all}");
    }
}
