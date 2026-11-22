//! Processed prompts kept for reuse. A request whose prompt starts with something already processed (an earlier
//! turn of the same conversation, or a system prompt and tool list shared with other conversations) starts from
//! there instead of from nothing.
//!
//! The model's state can only be extended, or rolled back a few tokens, so reuse needs a snapshot at exactly the
//! shared point. Snapshots are taken where sharing is likely or has been seen:
//!
//! - the end of a conversation, when another conversation displaces it from the session;
//! - the end of the prompt's header (system messages and tool declarations), which other conversations with
//!   the same setup begin with;
//! - wherever a new prompt stops agreeing with one seen before, which marks a prefix the two share.
//!
//! Snapshots are large (hundreds of megabytes to gigabytes), so they share a byte budget and the least recently
//! used go first.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::session::{Session, Snapshot};

/// A header shorter than this is not worth a snapshot of its own.
const MIN_HEADER_TOKENS: usize = 64;
/// A snapshot must save at least this many tokens over the one before it.
const MIN_GAIN_TOKENS: usize = 32;

const FILE_NAME: &str = "prompt-cache.bin";

pub struct PromptCache {
    entries: Vec<Entry>,
    budget_bytes: usize,
    /// Counts uses, to order entries by recency.
    clock: u64,
}

#[derive(Serialize, Deserialize)]
struct Entry {
    snapshot: Snapshot,
    /// The end of a conversation so far, as opposed to a point inside a prompt. A later end of the same
    /// conversation replaces it.
    tip: bool,
    used: u64,
}

/// What is kept on disk: the snapshots, and what they were made with.
#[derive(Serialize, Deserialize)]
struct Saved {
    fingerprint: String,
    entries: Vec<Entry>,
}

/// Where to start processing a prompt from, and where to stop on the way to take snapshots.
#[derive(Debug, PartialEq, Eq)]
pub struct Plan {
    pub source: Source,
    /// How many of the prompt's tokens the source already covers.
    pub start: usize,
    /// Prefix lengths of the prompt to snapshot, in increasing order.
    pub boundaries: Vec<usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Source {
    /// The session as it stands.
    Live,
    /// The snapshot at this index.
    Snapshot(usize),
    /// Nothing reusable: start from an empty context.
    Scratch,
}

/// Decides how to process `prompt` given the token sequences of the snapshots on hand and of the live session.
/// `header` is the length of the prompt's system-and-tools header, when it has one.
pub fn plan(snapshots: &[&[u32]], live: &[u32], prompt: &[u32], header: Option<usize>) -> Plan {
    // The longest thing on hand that the prompt starts with; the live session wins a tie, as it costs no copy.
    let mut source = Source::Scratch;
    let mut start = 0;
    for (index, tokens) in snapshots.iter().enumerate() {
        if tokens.len() > start && prompt.starts_with(tokens) {
            (source, start) = (Source::Snapshot(index), tokens.len());
        }
    }
    if !live.is_empty() && live.len() >= start && prompt.starts_with(live) {
        (source, start) = (Source::Live, live.len());
    }

    // The prompt may agree with something on hand beyond that point without a snapshot there to use. This time
    // the tokens have to be processed; a snapshot at the end of the agreement saves them next time.
    let common = |tokens: &&[u32]| tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    let shared = snapshots.iter().chain([&live]).map(common).max().unwrap_or(0);

    let mut boundaries = Vec::new();
    if let Some(header) = header
        && header >= MIN_HEADER_TOKENS
        && header >= start + MIN_GAIN_TOKENS
        && header <= prompt.len()
    {
        boundaries.push(header);
    }
    if shared >= start + MIN_GAIN_TOKENS {
        boundaries.push(shared);
    }
    boundaries.sort_unstable();
    boundaries.dedup();
    // Two snapshots a few tokens apart are not worth the memory; keep the later.
    let spaced = |i: usize| boundaries.get(i + 1).is_none_or(|next| next - boundaries[i] >= MIN_GAIN_TOKENS);
    let boundaries = (0..boundaries.len()).filter(|&i| spaced(i)).map(|i| boundaries[i]).collect();
    Plan { source, start, boundaries }
}

impl PromptCache {
    /// A cache holding at most about `budget_bytes` of snapshots. With a budget of zero it keeps nothing and the
    /// session alone carries a conversation from one turn to the next.
    pub fn new(budget_bytes: usize) -> Self {
        Self { entries: Vec::new(), budget_bytes, clock: 0 }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn bytes(&self) -> usize {
        self.entries.iter().map(|entry| entry.snapshot.bytes()).sum()
    }

    /// Brings `session` to exactly `prompt`, reusing whatever is on hand, and takes the snapshots the plan calls
    /// for along the way. Returns how many of the prompt's tokens did not need processing.
    pub fn prepare(&mut self, session: &mut Session, prompt: &[u32], header: Option<usize>) -> Result<usize> {
        self.prepare_watched(session, prompt, header, &mut |_| true)
    }

    /// [`PromptCache::prepare`], with `progress` as in [`Session::set_context_watched`].
    pub fn prepare_watched(
        &mut self,
        session: &mut Session,
        prompt: &[u32],
        header: Option<usize>,
        progress: &mut dyn FnMut(usize) -> bool,
    ) -> Result<usize> {
        let tokens: Vec<&[u32]> = self.entries.iter().map(|entry| entry.snapshot.tokens()).collect();
        let plan = plan(&tokens, session.context(), prompt, header);

        if plan.source != Source::Live {
            // Taken out before anything is added, which could move or evict entries.
            let chosen = match plan.source {
                Source::Snapshot(index) => {
                    self.clock += 1;
                    self.entries[index].used = self.clock;
                    // A conversation's end is about to be extended, which makes this copy of it obsolete: move
                    // it. A point inside a prompt stays for the next conversation that shares it.
                    Some(match self.entries[index].tip {
                        true => self.entries.remove(index).snapshot,
                        false => self.entries[index].snapshot.clone(),
                    })
                }
                _ => None,
            };
            // Another conversation takes over the session; keep the one that was in it.
            if !session.context().is_empty() {
                let displaced = session.take_snapshot();
                self.insert(displaced, true);
            }
            if let Some(snapshot) = chosen {
                session.restore(snapshot)?;
            }
        }
        for &boundary in &plan.boundaries {
            session.set_context_watched(&prompt[..boundary], progress)?;
            self.insert(session.snapshot(), false);
        }
        session.set_context_watched(prompt, progress)?;
        Ok(plan.start)
    }

    /// Keeps the session's present context for later, as the end of a conversation.
    pub fn keep(&mut self, session: &Session) {
        if !session.context().is_empty() {
            self.insert(session.snapshot(), true);
        }
    }

    fn insert(&mut self, snapshot: Snapshot, tip: bool) {
        if self.budget_bytes == 0 || snapshot.bytes() > self.budget_bytes {
            return;
        }
        self.clock += 1;
        if let Some(existing) = self.entries.iter_mut().find(|entry| entry.snapshot.tokens() == snapshot.tokens()) {
            existing.used = self.clock;
            return;
        }
        if tip {
            // Earlier ends of the same conversation are prefixes of this one, and no longer needed.
            self.entries.retain(|entry| !(entry.tip && snapshot.tokens().starts_with(entry.snapshot.tokens())));
        }
        self.entries.push(Entry { snapshot, tip, used: self.clock });
        while self.bytes() > self.budget_bytes {
            let oldest = (0..self.entries.len()).min_by_key(|&i| self.entries[i].used).expect("over budget, so not empty");
            self.entries.remove(oldest);
        }
    }

    /// Writes the cache to `dir`. `fingerprint` names what the snapshots were made with (model, pass count,
    /// arithmetic); [`PromptCache::load`] only accepts a file with the same one.
    pub fn save(self, dir: &Path, fingerprint: &str) -> Result<()> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(FILE_NAME);
        // Written beside the real file and renamed, so an interrupted save cannot leave half a cache.
        let partial = dir.join(format!("{FILE_NAME}.partial"));
        let file = File::create(&partial).with_context(|| format!("creating {}", partial.display()))?;
        let saved = Saved { fingerprint: fingerprint.to_owned(), entries: self.entries };
        postcard::to_io(&saved, BufWriter::new(file)).with_context(|| format!("writing {}", partial.display()))?;
        std::fs::rename(&partial, &path).with_context(|| format!("renaming to {}", path.display()))?;
        Ok(())
    }

    /// Reads back a cache written by [`PromptCache::save`]. A missing file, or one made with something else, gives
    /// an empty cache.
    pub fn load(dir: &Path, fingerprint: &str, budget_bytes: usize) -> Result<Self> {
        let mut cache = Self::new(budget_bytes);
        let path = dir.join(FILE_NAME);
        let Ok(file) = File::open(&path) else { return Ok(cache) };
        let mut scratch = vec![0u8; 1 << 16];
        let (saved, _): (Saved, _) = postcard::from_io((BufReader::new(file), &mut scratch))
            .with_context(|| format!("reading {}; delete it to start with an empty cache", path.display()))?;
        if saved.fingerprint != fingerprint {
            return Ok(cache);
        }
        // Oldest first, so the budget drops the right ones if it has shrunk.
        let mut entries = saved.entries;
        entries.sort_by_key(|entry| entry.used);
        for entry in entries {
            cache.insert(entry.snapshot, entry.tip);
        }
        Ok(cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(range: std::ops::Range<u32>) -> Vec<u32> {
        range.collect()
    }

    #[test]
    fn the_next_turn_extends_the_live_session() {
        let live = ids(0..200);
        let prompt = ids(0..260);
        let plan = plan(&[], &live, &prompt, Some(100));
        // The header is already behind the live context, so nothing to snapshot.
        assert_eq!(plan, Plan { source: Source::Live, start: 200, boundaries: vec![] });
    }

    #[test]
    fn a_first_prompt_snapshots_its_header() {
        let prompt = ids(0..500);
        assert_eq!(plan(&[], &[], &prompt, Some(300)), Plan { source: Source::Scratch, start: 0, boundaries: vec![300] });
        // A short header is not worth it.
        assert_eq!(plan(&[], &[], &prompt, Some(40)).boundaries, Vec::<usize>::new());
        assert_eq!(plan(&[], &[], &prompt, None).boundaries, Vec::<usize>::new());
    }

    #[test]
    fn a_second_conversation_starts_from_the_shared_header() {
        let header = ids(0..300);
        let live: Vec<u32> = ids(0..300).into_iter().chain(1000..1200).collect();
        let prompt: Vec<u32> = ids(0..300).into_iter().chain(2000..2100).collect();
        let plan = plan(&[&header], &live, &prompt, Some(300));
        assert_eq!(plan, Plan { source: Source::Snapshot(0), start: 300, boundaries: vec![] });
    }

    #[test]
    fn a_shared_prefix_without_a_snapshot_gets_one() {
        // No header known, but the new prompt agrees with the live conversation for its first 150 tokens.
        let live: Vec<u32> = ids(0..150).into_iter().chain(1000..1100).collect();
        let prompt: Vec<u32> = ids(0..150).into_iter().chain(2000..2100).collect();
        assert_eq!(plan(&[], &live, &prompt, None), Plan { source: Source::Scratch, start: 0, boundaries: vec![150] });
    }

    #[test]
    fn asking_again_snapshots_the_end_of_the_prompt() {
        // The live session holds the prompt and the reply to it; the same prompt arrives again.
        let prompt = ids(0..120);
        let live = ids(0..180);
        assert_eq!(plan(&[], &live, &prompt, None), Plan { source: Source::Scratch, start: 0, boundaries: vec![120] });
        // With that snapshot on hand, a third time costs nothing.
        assert_eq!(plan(&[&prompt], &live, &prompt, None), Plan { source: Source::Snapshot(0), start: 120, boundaries: vec![] });
    }

    #[test]
    fn the_longest_usable_snapshot_wins_and_close_boundaries_merge() {
        let (short, long, other) = (ids(0..100), ids(0..250), ids(500..900));
        let prompt = ids(0..400);
        let plan = plan(&[&short, &other, &long], &[], &prompt, Some(260));
        // The header at 260 is only ten tokens past the snapshot at 250.
        assert_eq!(plan, Plan { source: Source::Snapshot(2), start: 250, boundaries: vec![] });
    }
}
