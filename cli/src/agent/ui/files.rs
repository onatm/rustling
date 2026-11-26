//! Workspace path completion for the token under the composer cursor.

use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use ignore::WalkBuilder;
use nucleo_matcher::pattern::{Atom, AtomKind, CaseMatching, Normalization};
use nucleo_matcher::{Config, Matcher, Utf32Str};
use ratatui_textarea::{CursorMove, TextArea};
use tokio::sync::oneshot;

use super::MAX_COMPLETIONS;

struct Mention<'a> {
    query: &'a str,
    start_column: usize,
    following_space: bool,
}

impl<'a> Mention<'a> {
    fn at(input: &'a TextArea) -> Option<Self> {
        if input.selection_range().is_some() {
            return None;
        }
        let cursor = input.cursor();
        let line = &input.lines()[cursor.0];
        if !line.contains('@') {
            return None;
        }
        let byte = line.char_indices().nth(cursor.1).map_or(line.len(), |(i, _)| i);
        let before = line[..byte].rsplit(char::is_whitespace).next()?;
        let after = line[byte..].split(char::is_whitespace).next()?;
        let end = byte + after.len();
        let query = line[byte - before.len()..end].strip_prefix('@')?;
        Some(Self {
            query,
            start_column: cursor.1 - before.chars().count(),
            following_space: line[end..].starts_with(' '),
        })
    }
}

pub(super) fn insert_path(input: &mut TextArea, path: &str) {
    let Some(mention) = Mention::at(input) else { return };
    let count = 1 + mention.query.chars().count() + usize::from(mention.following_space);
    let cursor = input.cursor();
    if let (Ok(row), Ok(col)) = (cursor.0.try_into(), mention.start_column.try_into()) {
        input.move_cursor(CursorMove::Jump(row, col));
    } else {
        // Jump takes u16 coordinates. Relative movement also works in larger pastes.
        for _ in mention.start_column..cursor.1 {
            input.move_cursor(CursorMove::Back);
        }
    }
    input.delete_str(count);
    let path = if path.contains(char::is_whitespace) || path.contains('"') {
        serde_json::Value::String(path.to_owned()).to_string()
    } else {
        path.to_owned()
    };
    input.insert_str(format!("{path} "));
}

pub(super) struct FilePicker {
    root: PathBuf,
    query: Option<String>,
    pending: Option<oneshot::Receiver<Vec<String>>>,
    pub(super) matches: Vec<String>,
}

impl FilePicker {
    pub(super) fn new(root: PathBuf) -> Self {
        Self { root, query: None, pending: None, matches: Vec::new() }
    }

    pub(super) fn active(&self) -> bool {
        self.query.is_some()
    }

    pub(super) fn searching(&self) -> bool {
        self.pending.is_some()
    }

    pub(super) fn hint(&self) -> Option<&'static str> {
        match self.query.as_deref()? {
            "" => Some("Type a file name to search"),
            _ if self.pending.is_some() => Some("Searching files…"),
            _ if self.matches.is_empty() => Some("No matching files"),
            _ => None,
        }
    }

    pub(super) fn sync(&mut self, input: &TextArea, dismissed: bool) {
        let query = if dismissed { None } else { Mention::at(input).map(|mention| mention.query) };
        if query == self.query.as_deref() {
            return;
        }
        self.query = query.map(str::to_owned);
        self.matches.clear();
        // Dropping the receiver cancels the previous walk and discards stale results.
        self.pending = None;
        if let Some(query) = query.filter(|q| !q.is_empty()) {
            let (mut tx, rx) = oneshot::channel();
            let root = self.root.clone();
            let query = query.to_owned();
            tokio::spawn(async move {
                // Coalesce a burst of typing before starting filesystem work.
                tokio::select! {
                    _ = tx.closed() => return,
                    _ = tokio::time::sleep(Duration::from_millis(50)) => {}
                }
                tokio::task::spawn_blocking(move || {
                    let matches = search(&root, &query, || tx.is_closed());
                    let _ = tx.send(matches);
                });
            });
            self.pending = Some(rx);
        }
    }

    pub(super) async fn receive(&mut self) {
        let Some(rx) = &mut self.pending else { return std::future::pending().await };
        self.matches = rx.await.unwrap_or_default();
        self.pending = None;
    }
}

fn search(root: &Path, query: &str, cancelled: impl Fn() -> bool) -> Vec<String> {
    let walk = WalkBuilder::new(root).hidden(false).require_git(false)
        .filter_entry(|entry| entry.file_name() != ".git").build();
    let mut matcher = Matcher::new(Config::DEFAULT.match_paths());
    let pattern = Atom::new(query, CaseMatching::Smart, Normalization::Smart, AtomKind::Fuzzy, false);
    let mut scratch = Vec::new();
    let mut best = BTreeSet::new();
    for entry in walk {
        if cancelled() {
            break;
        }
        let Ok(entry) = entry else { continue };
        let Some(kind) = entry.file_type() else { continue };
        if !(kind.is_file() || kind.is_symlink() && entry.path().is_file()) {
            continue;
        }
        let Some(path) = entry.path().strip_prefix(root).ok().and_then(Path::to_str) else { continue };
        if path.chars().any(char::is_control) {
            continue;
        }
        if let Some(score) = pattern.score(Utf32Str::new(path, &mut scratch), &mut matcher) {
            best.insert((Reverse(score), path.len(), path.to_owned()));
            if best.len() > MAX_COMPLETIONS {
                best.pop_last();
            }
        }
    }
    best.into_iter().map(|(_, _, path)| path).collect()
}
