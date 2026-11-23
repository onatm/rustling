//! The small operations the layers are built from.

use super::*;

/// Learned-query attention over the normed residual blocks plus the block in progress; one output per query.
///
/// `blocks` are finished blocks, already normed with [`residual_normed`], which a block is once for all the reads
/// that follow; `partial` is normed here.
pub(super) fn attn_res(queries: &[f32], blocks: &[&Seq], partial: &Seq, h: usize) -> Vec<Seq> {
    let scale = (h as f32).powf(-0.5);
    // One token's outputs for all queries, back to back.
    let read = |t: usize| -> Vec<f32> {
        let mut current = partial[t * h..][..h].to_vec();
        rms_norm(&mut current, None, RESIDUAL_NORM_EPS);
        let states: Vec<&[f32]> = blocks.iter().map(|block| &block[t * h..][..h]).chain([&current[..]]).collect();
        let mut out = vec![0.0f32; queries.len()];
        for (query, out) in queries.chunks_exact(h).zip(out.chunks_exact_mut(h)) {
            let logits: Vec<f32> = states.iter().map(|s| dot(query, s) * scale).collect();
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            for (weight, state) in weights.iter().zip(&states) {
                out.iter_mut().zip(*state).for_each(|(o, s)| *o += weight / total * s);
            }
        }
        out
    };
    let per_token = per_token(partial.len() / h, read);
    (0..queries.len() / h)
        .map(|q| per_token.iter().flat_map(|token| token[q * h..][..h].iter().copied()).collect())
        .collect()
}

/// A finished residual block as [`attn_res`] reads it: each token's state normed.
pub(super) fn residual_normed(mut block: Seq, h: usize) -> Seq {
    block.chunks_exact_mut(h).for_each(|x| rms_norm(x, None, RESIDUAL_NORM_EPS));
    block
}

/// Runs `work` for each token, across threads once there are enough tokens to be worth waking the pool.
pub(super) fn per_token<T: Send>(tokens: usize, work: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    match tokens < PARALLEL_MIN_TOKENS {
        true => (0..tokens).map(work).collect(),
        false => (0..tokens).into_par_iter().map(work).collect(),
    }
}

/// Prepends the convolution's remembered inputs to `x` and remembers the most recent of the result.
/// Returns the joined context and how many remembered vectors lead it.
pub(super) fn with_history(history: &mut ConvHistory, x: &[f32], channels: usize, k: usize) -> (Vec<f32>, usize) {
    let seen = history.len() / channels;
    let (front, back) = history.as_slices();
    let context = [front, back, x].concat();
    let keep = (k - 1 + MAX_ROLLBACK).min(context.len() / channels) * channels;
    history.clear();
    history.extend(context[context.len() - keep..].iter().copied());
    (context, seen)
}

/// Causal depthwise convolution with a residual: `(conv(x) + x) / sqrt(kernel + 1)`. `weight` is
/// `[channels, kernel]` with the last tap on the current token. `history` carries the inputs of earlier calls.
pub(super) fn static_conv(x: &[f32], weight: &[f32], channels: usize, k: usize, history: &mut ConvHistory) -> Seq {
    let scale = 1.0 / ((k + 1) as f32).sqrt();
    let (context, seen) = with_history(history, x, channels, k);
    let mut out = vec![0.0f32; x.len()];
    for (i, out) in out.chunks_exact_mut(channels).enumerate() {
        let t = seen + i;
        for lag in 0..k.min(t + 1) {
            let past = &context[(t - lag) * channels..][..channels];
            for (c, (o, p)) in out.iter_mut().zip(past).enumerate() {
                *o += weight[c * k + (k - 1 - lag)] * p;
            }
        }
        let current = &context[t * channels..][..channels];
        out.iter_mut().zip(current).for_each(|(o, c)| *o = (*o + c) * scale);
    }
    out
}

pub(crate) fn rms_norm(x: &mut [f32], weight: Option<&[f32]>, eps: f32) {
    let mean_square = dot(x, x) / x.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    match weight {
        Some(weight) => x.iter_mut().zip(weight).for_each(|(v, w)| *v *= scale * w),
        None => x.iter_mut().for_each(|v| *v *= scale),
    }
}

/// Scales a vector down to `threshold` when its L2 norm is larger.
pub(super) fn l2_clip(x: &mut [f32], threshold: f32) {
    let norm = dot(x, x).sqrt();
    let scale = (threshold / (norm + 1e-6)).min(1.0);
    x.iter_mut().for_each(|v| *v *= scale);
}

pub(super) fn add(a: &[f32], b: &[f32]) -> Seq {
    a.iter().zip(b).map(|(a, b)| a + b).collect()
}

pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 16;
    let (a_chunks, a_rest) = a.as_chunks::<LANES>();
    let (b_chunks, b_rest) = b.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (a, b) in a_chunks.iter().zip(b_chunks) {
        for i in 0..LANES {
            acc[i] += a[i] * b[i];
        }
    }
    acc.iter().sum::<f32>() + a_rest.iter().zip(b_rest).map(|(a, b)| a * b).sum::<f32>()
}
