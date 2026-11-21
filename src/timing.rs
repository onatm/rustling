//! Wall-clock totals for the stages of generation, cheap enough to leave on. Each stage runs on the calling
//! thread (fanning out to the pool inside), so its total is elapsed time, not CPU time. The draft head's total
//! includes the vocabulary projections it makes, which are also counted under `Logits`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Residual-block reads, message merges and norms ahead of attention and the experts.
    Inputs,
    /// Query, key, value, relative, sink and gate projections.
    Projections,
    /// Key/value convolutions, clipping, attention scores and mixing.
    Attention,
    /// Output projection and the post-attention convolution.
    AttentionOutput,
    /// Routing and the expert matrices.
    Experts,
    /// The post-expert convolution and residual add.
    ExpertOutput,
    /// The output projection onto the vocabulary.
    Logits,
    /// The draft model's layers, for a block of proposals.
    DraftLayers,
    /// The draft model's recurrent head, including its vocabulary projections.
    DraftHead,
    /// Adding processed tokens to the draft model's context.
    DraftContext,
}

const STAGES: [Stage; 10] = [
    Stage::Inputs,
    Stage::Projections,
    Stage::Attention,
    Stage::AttentionOutput,
    Stage::Experts,
    Stage::ExpertOutput,
    Stage::Logits,
    Stage::DraftLayers,
    Stage::DraftHead,
    Stage::DraftContext,
];

static NANOS: [AtomicU64; STAGES.len()] = [const { AtomicU64::new(0) }; STAGES.len()];

/// Runs `work`, adding its elapsed time to `stage`.
pub fn time<T>(stage: Stage, work: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let out = work();
    since(stage, started);
    out
}

/// Adds the time elapsed since `started` to `stage`, for stages that do not fit in a closure.
pub fn since(stage: Stage, started: Instant) {
    NANOS[stage as usize].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// Seconds spent in each stage since the last call, which resets the totals.
pub fn take() -> Vec<(Stage, f64)> {
    STAGES.iter().map(|&stage| (stage, NANOS[stage as usize].swap(0, Ordering::Relaxed) as f64 * 1e-9)).collect()
}
