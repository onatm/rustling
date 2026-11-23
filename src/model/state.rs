//! What the tokens seen so far leave behind for the tokens to come, and stepping back in it.

use super::*;

/// Everything the tokens seen so far left behind for the tokens to come.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    /// Number of tokens already processed, which is the position of the next one.
    pub(super) position: usize,
    pub(super) visits: Vec<VisitState>,
}

impl State {
    pub fn position(&self) -> usize {
        self.position
    }

    /// Roughly how much memory the state holds.
    pub fn bytes(&self) -> usize {
        let floats: usize = self
            .visits
            .iter()
            .map(|v| v.convs.iter().map(Vec::len).sum::<usize>())
            .sum();
        floats * size_of::<f32>() + self.visits.iter().map(|v| v.keys.bytes() + v.values.bytes()).sum::<usize>()
    }

    /// The size of what each visit holds: the root of the summed squares of its keys, its values and its
    /// convolution inputs. Anything not finite in a visit shows as a size that is not finite, and two states
    /// that should be alike can be compared visit by visit.
    pub fn sizes(&self) -> Vec<[f64; 3]> {
        let size = |values: &mut dyn Iterator<Item = &f32>| values.map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        self.visits
            .iter()
            .map(|visit| {
                let table = |table: &Table| size(&mut (0..table.heads()).flat_map(|head| table.head(head)));
                [table(&visit.keys), table(&visit.values), size(&mut visit.convs.iter().flatten())]
            })
            .collect()
    }

    /// Whether the state has the shape `model` expects, as a check on one read back from disk.
    pub fn fits(&self, model: &Model) -> bool {
        let fresh = model.new_state();
        self.visits.len() == fresh.visits.len()
            && self.visits.iter().zip(&fresh.visits).all(|(mine, theirs)| {
                // Both tables hold the tokens from `first` on.
                let held = self.position.saturating_sub(mine.first);
                mine.conv_channels == theirs.conv_channels
                    && mine.first <= self.position
                    && [&mine.keys, &mine.values].iter().all(|table| table.heads() == theirs.keys.heads() && table.rows() == held)
            })
    }

    /// Forgets the last `tokens` tokens, as if they had never been processed. Only the most recent
    /// [`MAX_ROLLBACK`] tokens can be taken back.
    pub fn truncate(&mut self, tokens: usize) {
        assert!(tokens <= MAX_ROLLBACK && tokens <= self.position, "cannot take back {tokens} tokens");
        self.position -= tokens;
        for visit in &mut self.visits {
            for table in [&mut visit.keys, &mut visit.values] {
                table.truncate(table.rows() - tokens);
            }
            for (history, channels) in visit.convs.iter_mut().zip(visit.conv_channels) {
                history.truncate(history.len() - tokens * channels);
            }
        }
    }
}

/// A place in a [`State`] to come back to after any number of further tokens, made by [`Model::mark`]. A visit
/// that keeps every token's keys and values only ever adds to them, so its part of the mark is their count; a
/// sliding-window visit drops old ones as it goes, so the mark holds a copy of its window. The size is therefore
/// fixed, a few hundred megabytes, however long the context.
#[derive(Clone)]
pub struct Mark {
    pub(super) position: usize,
    pub(super) visits: Vec<VisitMark>,
}

#[derive(Clone)]
pub(super) struct VisitMark {
    pub(super) first: usize,
    pub(super) convs: [Vec<f32>; 4],
    /// The keys and values, for a visit that drops old ones.
    pub(super) window: Option<(Table, Table)>,
}

impl State {
    /// Goes back to `mark`, as if the tokens since had never been processed. The state must have come from the
    /// marked one by processing more tokens.
    pub fn rewind(&mut self, mark: &Mark) {
        assert!(mark.position <= self.position && mark.visits.len() == self.visits.len(), "not a mark of this state");
        for (visit, marked) in self.visits.iter_mut().zip(&mark.visits) {
            visit.convs = marked.convs.clone();
            match &marked.window {
                Some((keys, values)) => {
                    (visit.keys, visit.values, visit.first) = (keys.clone(), values.clone(), marked.first);
                }
                None => {
                    assert_eq!(visit.first, marked.first, "a visit that keeps everything dropped something");
                    let held = mark.position - visit.first;
                    visit.keys.truncate(held);
                    visit.values.truncate(held);
                }
            }
        }
        self.position = mark.position;
    }
}

/// How many of the most recent tokens [`State::truncate`] can take back: each convolution remembers this many
/// inputs beyond the ones it needs.
pub const MAX_ROLLBACK: usize = 16;

#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub(super) struct VisitState {
    /// Keys and values of earlier tokens, starting at position `first`: one table per key head, `head_dim` per
    /// token, so that attention reads each head's as one run of memory. Sliding-window visits drop what has
    /// left the window.
    pub(super) keys: Table,
    pub(super) values: Table,
    pub(super) first: usize,
    /// The recent inputs of the key, value, post-attention and post-MLP convolutions: the `kernel - 1` they
    /// need, and up to [`MAX_ROLLBACK`] more so recent tokens can be taken back.
    pub(super) convs: [Vec<f32>; 4],
    /// The width of each convolution's input.
    pub(super) conv_channels: [usize; 4],
}

impl VisitState {
    /// Records the inputs the convolutions took for one token computed elsewhere.
    #[cfg(target_os = "macos")]
    pub(super) fn push_conv_inputs(&mut self, cfg: &TextConfig, conv_inputs: [&[f32]; 4]) {
        for ((history, input), channels) in self.convs.iter_mut().zip(conv_inputs).zip(self.conv_channels) {
            history.extend_from_slice(input);
            let keep = (cfg.sconv_kernel_size - 1 + MAX_ROLLBACK) * channels;
            history.drain(..history.len().saturating_sub(keep));
        }
    }

    /// Drops the keys and values that have left a sliding layer's window, once a batch of them has: dropping
    /// from the front moves everything kept, megabytes per visit, so until then they sit unread at the front.
    pub(super) fn drop_stale(&mut self, cfg: &TextConfig, local: bool, position: usize) {
        self.drop_stale_from(cfg, local, position, STALE_BATCH);
    }

    /// As [`VisitState::drop_stale`], for `batch` stale tokens or more.
    pub(super) fn drop_stale_from(&mut self, cfg: &TextConfig, local: bool, position: usize, batch: usize) {
        let window_start = if local { (position + 1).saturating_sub(cfg.sliding_window_size) } else { 0 };
        let stale = window_start.saturating_sub(self.first).min(self.keys.rows());
        if stale >= batch {
            self.keys.drop_front(stale);
            self.values.drop_front(stale);
            self.first += stale;
        }
    }
}

