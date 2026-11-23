//! The DSpark draft model (`mtp.safetensors`): a three-layer transformer that proposes the next several tokens in
//! one go, for the main model to check in a single batch.
//!
//! It never sees tokens other than the last one. Its context is the main model's own hidden states: for every
//! token the main model has processed, four of its states are mixed (with learned weights, one mix per draft
//! layer) into keys and values that the draft attends over. A block is the last sampled token, the anchor,
//! followed by mask positions; the block attends to that context and to itself in both directions, and each mask
//! position's output predicts the token there. A small recurrent head then walks the block left to right, nudging
//! each position's prediction by the tokens chosen before it.
//!
//! Ported from vLLM's `models/inkling2/nvidia/dspark.py`. Its matrices are rounded to bytes when loaded and
//! multiplied in integers: a draft that is slightly off only gets fewer proposals accepted, it cannot change what
//! is generated.

use std::sync::OnceLock;

use anyhow::{Context, Result, ensure};
use crate::weights::ggml::Format;

use rayon::prelude::*;

use crate::accel::{self, Rows};
use crate::weights::checkpoint::Checkpoint;
use crate::config::{Config, DraftConfig};
use crate::model::{Mat, Model, Seq, apply_many, rms_norm, sigmoid};
use crate::timing::{self, Stage};

pub struct Draft<'a> {
    cfg: DraftConfig,
    hidden: usize,
    /// One row per draft layer, then one for the recurrent head: how to weigh the main model's tapped states.
    mix: Vec<Vec<f32>>,
    layers: Vec<Layer<'a>>,
    norm: Vec<f32>,
    mask_embed: Vec<f32>,
    head: Head<'a>,
    /// Rotary frequencies, one per pair of head dimensions.
    frequencies: Vec<f32>,
    /// A proposal ends where the draft's confidence in the whole run so far would fall below this.
    min_confidence: f32,
    /// The main model's output weights rounded to bytes, made on first use; nothing when the model's own are
    /// quantised already.
    unembed: OnceLock<Option<Mat<'static>>>,
    /// What each direction of the recurrent head's correction adds to every token's logit: the output weights
    /// times the head's output matrix, `[vocabulary, rank]`, made on first use. With it a proposal projects its
    /// positions onto the vocabulary once, all together, and each step adds a correction a ninth of that size,
    /// where projecting every corrected state on its own was most of the draft's work.
    corrections: OnceLock<Vec<f32>>,
}

/// A rejected proposal wastes the work of checking it and everything proposed after it, so only confident runs
/// are worth proposing. Measured on an M1 Max, speed keeps improving up to about here and flattens beyond.
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.8;

/// Whether to propose with the draft even when tokens are generated on the GPU, where a proposal made on the CPU
/// costs more than the tokens it saves. For measuring that: set `RUSTLING_DRAFT_ON_GPU`.
pub fn tried_on_gpu() -> bool {
    std::env::var_os("RUSTLING_DRAFT_ON_GPU").is_some()
}

/// Tokens per task when a correction is added to every token's logit.
const CORRECTION_TOKENS: usize = 8192;

struct Layer<'a> {
    local: bool,
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    /// Norm of the mixed main-model states that become this layer's context.
    context_norm: Vec<f32>,
    q: Mat<'a>,
    k: Mat<'a>,
    v: Mat<'a>,
    o: Mat<'a>,
    k_context: Mat<'a>,
    v_context: Mat<'a>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    k_context_norm: Vec<f32>,
    gate: Mat<'a>,
    up: Mat<'a>,
    down: Mat<'a>,
}

/// The recurrent ("Markov") head.
struct Head<'a> {
    rank: usize,
    context_norm: Vec<f32>,
    /// Token embeddings of the previously chosen token, `[vocab, rank]` in bfloat16.
    previous: &'a [u8],
    /// Maps `[state, previous token, position's hidden state, context]` to `[gate, candidate, output]`.
    step: Mat<'a>,
    step_bias: Vec<f32>,
    /// Maps the output back to a correction of the hidden state.
    out: Mat<'a>,
    out_bias: Vec<f32>,
}

/// The draft's view of the conversation: context keys and values for every token the main model has processed.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct DraftState {
    /// Per layer, `kv_dim` values per token.
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// The recurrent head's context, from the last token processed.
    head_context: Vec<f32>,
    tokens: usize,
}

impl DraftState {
    pub fn tokens(&self) -> usize {
        self.tokens
    }

    /// The part of the state that later tokens overwrite: with it and the token count, [`DraftState::rewind`]
    /// comes back to this point. The keys and values are only ever added to.
    pub(crate) fn head_context(&self) -> &[f32] {
        &self.head_context
    }

    /// Forgets everything after the first `tokens` tokens; `head_context` is what the state held then.
    pub(crate) fn rewind(&mut self, tokens: usize, head_context: Vec<f32>) {
        assert!(tokens > 0 && tokens <= self.tokens, "cannot rewind a draft of {} tokens to {tokens}", self.tokens);
        for table in self.keys.iter_mut().chain(&mut self.values) {
            table.truncate(table.len() / self.tokens * tokens);
        }
        (self.head_context, self.tokens) = (head_context, tokens);
    }

    /// Roughly how much memory the state holds.
    pub fn bytes(&self) -> usize {
        let floats: usize = self.keys.iter().chain(&self.values).map(Vec::len).sum::<usize>() + self.head_context.len();
        floats * size_of::<f32>()
    }
}

impl<'a> Draft<'a> {
    /// Loads the draft from its own checkpoint file. `config` is the main model's, which describes the draft too.
    pub fn load(config: &Config, weights: &'a Checkpoint) -> Result<Self> {
        let cfg = config.mtp_config.clone().context("this checkpoint has no draft model (mtp_config)")?;
        let text = &config.text_config;
        let h = cfg.hidden_size;
        ensure!(h == text.hidden_size, "the draft's width differs from the model's");
        ensure!(cfg.markov_head_type == "rnn_hidden", "unsupported draft head {}", cfg.markov_head_type);
        ensure!(cfg.head_dim % 2 == 0 && cfg.num_attention_heads % cfg.num_key_value_heads == 0, "bad draft heads");
        let taps = cfg.target_layer_ids.len();
        ensure!(
            cfg.target_layer_ids.last() == Some(&text.visits().len()),
            "the draft reads the model at fixed layer visits, so it only works at the checkpoint's own pass count"
        );

        let vec = |name: &str, len: usize| -> Result<Vec<f32>> {
            let dense = weights.dense(name)?;
            ensure!(dense.data.len() == len, "{name}: expected {len} values, found shape {:?}", dense.shape);
            Ok(dense.data)
        };
        // The matrices are packed as bytes at load. A draft that is slightly off only has fewer proposals
        // accepted, and its own work is what every round pays before the main model checks anything.
        let mat = |name: &str, rows: usize, cols: usize| Mat::open(weights, name, rows, cols).map(Mat::rounded_to_bytes);

        let (q_dim, kv_dim) = (cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim);
        let layers = (0..cfg.num_hidden_layers)
            .map(|l| {
                let p = format!("layers.{l}");
                Ok(Layer {
                    local: cfg.local_layer_ids.contains(&l),
                    input_norm: vec(&format!("{p}.input_layernorm.weight"), h)?,
                    post_attention_norm: vec(&format!("{p}.post_attention_layernorm.weight"), h)?,
                    context_norm: vec(&format!("{p}.dflare_norm.weight"), h)?,
                    q: mat(&format!("{p}.self_attn.q_proj.weight"), q_dim, h)?,
                    k: mat(&format!("{p}.self_attn.k_proj.weight"), kv_dim, h)?,
                    v: mat(&format!("{p}.self_attn.v_proj.weight"), kv_dim, h)?,
                    o: mat(&format!("{p}.self_attn.o_proj.weight"), h, q_dim)?,
                    k_context: mat(&format!("{p}.self_attn.k_proj_ctx.weight"), kv_dim, h)?,
                    v_context: mat(&format!("{p}.self_attn.v_proj_ctx.weight"), kv_dim, h)?,
                    q_norm: vec(&format!("{p}.self_attn.q_norm.weight"), cfg.head_dim)?,
                    k_norm: vec(&format!("{p}.self_attn.k_norm.weight"), cfg.head_dim)?,
                    k_context_norm: vec(&format!("{p}.self_attn.k_norm_ctx.weight"), cfg.head_dim)?,
                    gate: mat(&format!("{p}.mlp.gate_proj.weight"), cfg.intermediate_size, h)?,
                    up: mat(&format!("{p}.mlp.up_proj.weight"), cfg.intermediate_size, h)?,
                    down: mat(&format!("{p}.mlp.down_proj.weight"), h, cfg.intermediate_size)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Each row of the fuse weights becomes a set of mixing weights through a softmax.
        let fuse = vec("fuse_weight", (cfg.num_hidden_layers + 1) * taps)?;
        let mix = fuse
            .chunks_exact(taps)
            .map(|row| {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = row.iter().map(|w| (w - max).exp()).collect();
                let total: f32 = weights.iter().sum();
                weights.iter().map(|w| w / total).collect()
            })
            .collect();

        let rank = cfg.markov_rank;
        let (format, _, previous) = weights.rows("markov_head.markov_w1.weight")?;
        ensure!(format == Format::Bf16 && previous.len() == text.vocab_size * rank * 2, "unexpected markov_w1");
        let head = Head {
            rank,
            context_norm: vec("markov_head.dflare_norm.weight", h)?,
            previous,
            step: mat("markov_head.rnn_proj.weight", 3 * rank, 2 * rank + 2 * h)?,
            step_bias: vec("markov_head.rnn_proj.bias", 3 * rank)?,
            out: mat("markov_head.hidden_out.weight", h, rank)?,
            out_bias: vec("markov_head.hidden_out.bias", h)?,
        };
        let pairs = cfg.head_dim / 2;
        let frequencies = (0..pairs).map(|j| cfg.rope_theta.powf(-(j as f32) / pairs as f32)).collect();
        Ok(Self {
            norm: vec("norm.weight", h)?,
            mask_embed: vec("mask_embed", h)?,
            hidden: h,
            mix,
            layers,
            head,
            frequencies,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            unembed: OnceLock::new(),
            corrections: OnceLock::new(),
            cfg,
        })
    }

    /// The main-model visits to tap with [`Model::forward_tapped`]; the final state is the last tap and comes
    /// with every forward pass.
    pub fn taps(&self) -> &[usize] {
        &self.cfg.target_layer_ids[..self.cfg.target_layer_ids.len() - 1]
    }

    /// Sets how sure the draft must be of a run of tokens to propose it: 0 always proposes a full block.
    pub fn with_min_confidence(mut self, min_confidence: f32) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// The most tokens one proposal holds.
    pub fn proposal_len(&self) -> usize {
        self.cfg.block_size - 1
    }

    pub fn new_state(&self) -> DraftState {
        let layers = self.layers.len();
        DraftState { keys: vec![Vec::new(); layers], values: vec![Vec::new(); layers], head_context: Vec::new(), tokens: 0 }
    }

    /// Adds the next `tokens` tokens to the draft's context. `taps` are the main model's tapped states followed by
    /// its final state, each holding at least `tokens` vectors, of which the first `tokens` are used.
    pub fn extend(&self, state: &mut DraftState, taps: &[&[f32]], tokens: usize) {
        let (h, eps, dim) = (self.hidden, self.cfg.rms_norm_eps, self.cfg.head_dim);
        assert_eq!(taps.len(), self.cfg.target_layer_ids.len());
        if tokens == 0 {
            return;
        }
        let started = std::time::Instant::now();
        // The tapped states of token `t`, mixed with one row of weights and normed.
        let mixed = |t: usize, row: &[f32], norm: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; h];
            for (tap, weight) in taps.iter().zip(row) {
                out.iter_mut().zip(&tap[t * h..][..h]).for_each(|(o, x)| *o += x * weight);
            }
            rms_norm(&mut out, Some(norm), eps);
            out
        };
        for (l, layer) in self.layers.iter().enumerate() {
            let context: Seq = (0..tokens).flat_map(|t| mixed(t, &self.mix[l], &layer.context_norm)).collect();
            let mut projected = apply_many(None, &[(&layer.k_context, &context), (&layer.v_context, &context)]);
            let (values, mut keys) = (projected.remove(1), projected.remove(0));
            for (t, key) in keys.chunks_exact_mut(layer.k_context.rows).enumerate() {
                for head in key.chunks_exact_mut(dim) {
                    rms_norm(head, Some(&layer.k_context_norm), eps);
                    self.rotate(head, state.tokens + t);
                }
            }
            state.keys[l].extend(keys);
            state.values[l].extend(values);
        }
        state.head_context = mixed(tokens - 1, &self.mix[self.layers.len()], &self.head.context_norm);
        state.tokens += tokens;
        timing::since(Stage::DraftContext, started);
    }

    /// Rotary position embedding of one head vector, halves paired: the first half with the second.
    fn rotate(&self, head: &mut [f32], position: usize) {
        let (first, second) = head.split_at_mut(self.frequencies.len());
        for ((a, b), frequency) in first.iter_mut().zip(second).zip(&self.frequencies) {
            let (sin, cos) = (position as f32 * frequency).sin_cos();
            (*a, *b) = (*a * cos - *b * sin, *b * cos + *a * sin);
        }
    }

    /// Proposes the tokens that follow `anchor`, the token sampled after everything in `state`. Greedy: each is
    /// the draft's most likely token given the ones before it. The proposal stops, possibly empty, where the
    /// draft's probability for the run so far drops below its confidence threshold.
    pub fn propose(&self, state: &DraftState, anchor: u32, model: &Model) -> Result<Vec<u32>> {
        ensure!(state.tokens > 0, "the draft needs at least one processed token for context");
        let (h, eps, block) = (self.hidden, self.cfg.rms_norm_eps, self.cfg.block_size);
        let normed = |x: &Seq, weight: &[f32]| -> Seq {
            let mut out = x.clone();
            out.chunks_exact_mut(h).for_each(|x| rms_norm(x, Some(weight), eps));
            out
        };

        let started = std::time::Instant::now();
        // The block: the anchor's embedding, then masks. Pre-norm residual layers.
        let mut residual = model.embed(anchor)?;
        (1..block).for_each(|_| residual.extend_from_slice(&self.mask_embed));
        for (l, layer) in self.layers.iter().enumerate() {
            let attended = self.attention(l, layer, &normed(&residual, &layer.input_norm), state);
            residual.iter_mut().zip(&attended).for_each(|(r, a)| *r += a);

            let x = normed(&residual, &layer.post_attention_norm);
            let mut projected = apply_many(None, &[(&layer.gate, &x), (&layer.up, &x)]);
            let (up, gate) = (projected.remove(1), projected.remove(0));
            let activated: Seq = gate.iter().zip(&up).map(|(g, u)| g * sigmoid(*g) * u).collect();
            let mlp = layer.down.apply(&activated);
            residual.iter_mut().zip(&mlp).for_each(|(r, m)| *r += m);
        }
        let hidden = normed(&residual, &self.norm);
        timing::since(Stage::DraftLayers, started);
        let started = std::time::Instant::now();

        // The recurrent head, left to right over the mask positions. A position's logits are the output weights
        // applied to its hidden state plus the head's correction; the first part does not depend on the tokens
        // chosen so far, so positions can share a pass over the weights.
        let head = &self.head;
        let rank = head.rank;
        let (vocab, cap) = (model.config().unpadded_vocab_size, model.config().final_logit_softcapping);
        let unembed = self.unembed.get_or_init(|| model.packed_unembed()).as_ref().unwrap_or(model.unembed());
        let corrections = self.corrections.get_or_init(|| self.corrections(unembed, vocab));
        let biased: Seq = hidden[h..].chunks_exact(h).flat_map(|x| x.iter().zip(&head.out_bias).map(|(x, bias)| x + bias)).collect();
        // The first position alone, then the rest together once the draft has turned out sure of the first:
        // where it is not, the proposal ends there and the rest would be wasted.
        let mut uncorrected = unembed.apply(&biased[..h]);

        let mut memory = vec![0.0f32; rank];
        let mut previous = anchor;
        let mut proposal = Vec::with_capacity(block - 1);
        let mut confidence = 1.0f32;
        for (i, position) in hidden.chunks_exact(h).skip(1).enumerate() {
            if i == 1 {
                uncorrected.extend(unembed.apply(&biased[h..]));
            }
            let uncorrected = &uncorrected[i * unembed.rows..][..vocab];
            let row = &head.previous[previous as usize * rank * 2..][..rank * 2];
            let embedding = row.as_chunks::<2>().0.iter().map(|b| half::bf16::from_le_bytes(*b).to_f32());
            let input: Vec<f32> =
                memory.iter().copied().chain(embedding).chain(position.iter().copied()).chain(state.head_context.iter().copied()).collect();
            let mut raw = head.step.apply_one(&input);
            raw.iter_mut().zip(&head.step_bias).for_each(|(r, b)| *r += b);
            let (gate, rest) = raw.split_at(rank);
            let (candidate, output) = rest.split_at(rank);
            for ((m, g), c) in memory.iter_mut().zip(gate).zip(candidate) {
                let keep = sigmoid(*g);
                *m = keep * *m + (1.0 - keep) * c.tanh();
            }
            let latent: Vec<f32> = output.iter().map(|o| o.tanh()).collect();
            let mut logits = vec![0.0f32; vocab];
            let latent = Rows { values: &latent, rows: rank, cols: 1, stride: 1 };
            // Reading the table is what takes the time, so the cores share it.
            logits.par_chunks_mut(CORRECTION_TOKENS).zip(corrections.par_chunks(CORRECTION_TOKENS * rank)).for_each(|(logits, table)| {
                let table = Rows { values: table, rows: logits.len(), cols: rank, stride: rank };
                accel::multiply(table, latent, false, 1.0, logits);
            });
            logits.iter_mut().zip(uncorrected).for_each(|(l, u)| *l = (*l + u) / cap);
            accel::tanh_in_place(&mut logits);
            logits.iter_mut().for_each(|l| *l *= cap);
            let best = (0..vocab).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).expect("a vocabulary");
            // The draft's probability for its choice, and with it for the run of choices so far.
            let top = logits[best];
            logits.iter_mut().for_each(|l| *l -= top);
            accel::exp_in_place(&mut logits);
            let total: f64 = logits.iter().map(|l| *l as f64).sum();
            confidence *= (1.0 / total) as f32;
            if confidence < self.min_confidence {
                break;
            }
            previous = best as u32;
            proposal.push(previous);
        }
        timing::since(Stage::DraftHead, started);
        Ok(proposal)
    }

    /// The output weights times the head's output matrix, a row of `rank` values per token.
    fn corrections(&self, unembed: &Mat, vocab: usize) -> Vec<f32> {
        const TOKENS: usize = 256;
        let (h, rank) = (self.hidden, self.head.rank);
        let out = self.head.out.widened(0, h);
        let mut table = vec![0.0f32; vocab * rank];
        table.par_chunks_mut(TOKENS * rank).enumerate().for_each(|(chunk, table)| {
            let count = table.len() / rank;
            let rows = unembed.widened(chunk * TOKENS, count);
            let rows = Rows { values: &rows, rows: count, cols: h, stride: h };
            accel::multiply(rows, Rows { values: &out, rows: h, cols: rank, stride: rank }, false, 1.0, table);
        });
        table
    }

    /// Block attention: every block position attends to the context (the window before the block, in local
    /// layers) and to the whole block, itself and later positions included.
    fn attention(&self, l: usize, layer: &Layer, x: &Seq, state: &DraftState) -> Seq {
        let cfg = &self.cfg;
        let (heads, kv_heads, dim) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let (q_dim, kv_dim, group) = (heads * dim, kv_heads * dim, heads / kv_heads);
        let block = x.len() / self.hidden;

        let mut projected = apply_many(None, &[(&layer.q, x), (&layer.k, x), (&layer.v, x)]);
        let (values, mut keys, mut queries) = (projected.remove(2), projected.remove(1), projected.remove(0));
        for (i, query) in queries.chunks_exact_mut(q_dim).enumerate() {
            for head in query.chunks_exact_mut(dim) {
                rms_norm(head, Some(&layer.q_norm), cfg.rms_norm_eps);
                self.rotate(head, state.tokens + i);
            }
        }
        for (i, key) in keys.chunks_exact_mut(kv_dim).enumerate() {
            for head in key.chunks_exact_mut(dim) {
                rms_norm(head, Some(&layer.k_norm), cfg.rms_norm_eps);
                self.rotate(head, state.tokens + i);
            }
        }

        let first = if layer.local { state.tokens.saturating_sub(cfg.sliding_window_size) } else { 0 };
        let context_keys = &state.keys[l][first * kv_dim..];
        let context_values = &state.values[l][first * kv_dim..];
        let context = context_keys.len() / kv_dim;
        let scaling = (dim as f32).powf(-0.5);
        // A key head at a time, each on its own core: its query heads at every block position against all the
        // keys, as matrix products. In the layer that sees the whole conversation this is most of a proposal's
        // work once the context is long.
        let rows = block * group;
        let attend = |kv: usize| -> Vec<f32> {
            let mut mine = vec![0.0f32; rows * dim];
            for (i, mine) in mine.chunks_exact_mut(group * dim).enumerate() {
                mine.copy_from_slice(&queries[i * q_dim + kv * group * dim..][..group * dim]);
            }
            let mine = Rows { values: &mine, rows, cols: dim, stride: dim };
            // This head's band of a table holding every head's vectors for each token.
            fn band(table: &[f32], kv: usize, dim: usize, stride: usize) -> Rows<'_> {
                Rows { values: &table[kv * dim..], rows: table.len() / stride, cols: dim, stride }
            }
            let band = |table| band(table, kv, dim, kv_dim);
            let (mut far, mut near) = (vec![0.0f32; rows * context], vec![0.0f32; rows * block]);
            accel::multiply(mine, band(context_keys), true, scaling, &mut far);
            accel::multiply(mine, band(&keys), true, scaling, &mut near);
            // One softmax over the context and the block together.
            for (far, near) in far.chunks_exact_mut(context).zip(near.chunks_exact_mut(block)) {
                let max = far.iter().chain(near.iter()).copied().fold(f32::NEG_INFINITY, f32::max);
                far.iter_mut().chain(near.iter_mut()).for_each(|s| *s -= max);
                accel::exp_in_place(far);
                accel::exp_in_place(near);
                let total: f32 = far.iter().chain(near.iter()).sum();
                far.iter_mut().chain(near.iter_mut()).for_each(|w| *w /= total);
            }
            let (mut attended, mut within) = (vec![0.0f32; rows * dim], vec![0.0f32; rows * dim]);
            let weights = Rows { values: &far, rows, cols: context, stride: context };
            accel::multiply(weights, band(context_values), false, 1.0, &mut attended);
            let weights = Rows { values: &near, rows, cols: block, stride: block };
            accel::multiply(weights, band(&values), false, 1.0, &mut within);
            attended.iter_mut().zip(&within).for_each(|(a, w)| *a += w);
            attended
        };
        let attended: Vec<Vec<f32>> = (0..kv_heads).into_par_iter().map(attend).collect();
        let mut out = vec![0.0f32; block * q_dim];
        for (kv, attended) in attended.iter().enumerate() {
            for (row, attended) in attended.chunks_exact(dim).enumerate() {
                let unit = (row / group) * heads + kv * group + row % group;
                out[unit * dim..][..dim].copy_from_slice(attended);
            }
        }
        layer.o.apply(&out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::checkpoint::{companion_dir, resolve_model};
    use crate::model::{Activations, Backend, Options};

    /// What a proposal costs with a long conversation behind it, without paying to process one: the draft's
    /// context is filled with made-up keys and values, which cost its attention the same as real ones.
    /// `cargo test --release --lib a_proposal_at_long_context -- --ignored --nocapture`; `CONTEXT` sets the length.
    #[test]
    #[ignore]
    fn a_proposal_at_long_context() {
        let context: usize = std::env::var("CONTEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(12_000);
        let named = std::env::var("RUSTLING_GGUF").unwrap_or_else(|_| crate::weights::checkpoint::DEFAULT_MODEL.to_owned());
        let path = resolve_model(Some(std::path::Path::new(&named))).unwrap();
        let weights = Checkpoint::open(&path).unwrap();
        let config = Config::load(&path).unwrap();
        let model = Model::load(&config, &weights, Options { activations: Activations::Float, backend: Backend::Cpu }).unwrap();
        let draft_weights = Checkpoint::open_file(&companion_dir(&path).unwrap().join("mtp.safetensors")).unwrap();
        let draft = Draft::load(&config, &draft_weights).unwrap().with_min_confidence(0.0);

        let kv_dim = draft.cfg.num_key_value_heads * draft.cfg.head_dim;
        let made_up = |seed: usize| -> Vec<f32> {
            (0..context * kv_dim).map(|i| (((i * 2_654_435_761 + seed) >> 7) % 2001) as f32 / 1000.0 - 1.0).collect()
        };
        let mut state = draft.new_state();
        for l in 0..draft.layers.len() {
            (state.keys[l], state.values[l]) = (made_up(l), made_up(l + 99));
        }
        state.head_context = vec![0.1; draft.hidden];
        state.tokens = context;

        draft.propose(&state, 17, &model).unwrap();
        let mut best = f64::INFINITY;
        let mut stages = Vec::new();
        for round in 0..12 {
            timing::take();
            let started = std::time::Instant::now();
            let proposal = draft.propose(&state, 100 + round, &model).unwrap();
            let took = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(proposal.len(), draft.proposal_len());
            if took < best {
                best = took;
                stages = timing::take().iter().filter(|(_, s)| *s > 0.0).map(|(stage, s)| format!("{stage:?} {:.1}", s * 1000.0)).collect();
            }
        }
        eprintln!("a proposal at {context} tokens: {best:.1} ms at best ({})", stages.join(", "));
    }
}
