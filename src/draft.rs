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
//! Ported from vLLM's `models/inkling2/nvidia/dspark.py`. Arithmetic is plain f32: a draft that is slightly off
//! only gets fewer proposals accepted, it cannot change what is generated.

use anyhow::{Context, Result, ensure};
use safetensors::Dtype;

use crate::checkpoint::Checkpoint;
use crate::config::{Config, DraftConfig};
use crate::model::{Mat, Model, Seq, apply_many, dot, rms_norm, sigmoid};
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
}

/// A rejected proposal wastes the work of checking it and everything proposed after it, so only confident runs
/// are worth proposing. Measured on an M1 Max, speed keeps improving up to about here and flattens beyond.
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.8;

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
        let mat = |name: &str, rows: usize, cols: usize| -> Result<Mat<'a>> {
            let (info, bytes) = weights.bytes(name)?;
            ensure!(info.dtype == Dtype::BF16, "{name}: expected bfloat16, found {:?}", info.dtype);
            ensure!(bytes.len() == rows * cols * 2, "{name}: expected {rows}x{cols}, found shape {:?}", info.shape);
            Ok(Mat { rows, cols, bytes })
        };

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
        let (info, previous) = weights.bytes("markov_head.markov_w1.weight")?;
        ensure!(info.dtype == Dtype::BF16 && previous.len() == text.vocab_size * rank * 2, "unexpected markov_w1");
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

        // The recurrent head, left to right over the mask positions.
        let head = &self.head;
        let rank = head.rank;
        let mut memory = vec![0.0f32; rank];
        let mut previous = anchor;
        let mut proposal = Vec::with_capacity(block - 1);
        let mut confidence = 1.0f32;
        for position in hidden.chunks_exact(h).skip(1) {
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
            // The head's output corrects the hidden state before the shared output projection.
            let latent: Vec<f32> = output.iter().map(|o| o.tanh()).collect();
            let mut corrected = head.out.apply_one(&latent);
            for ((c, bias), x) in corrected.iter_mut().zip(&head.out_bias).zip(position) {
                *c += bias + x;
            }
            let logits = model.logits(&corrected)?;
            let best = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).expect("a vocabulary");
            // The draft's probability for its choice, and with it for the run of choices so far.
            let total: f64 = logits.iter().map(|l| ((l - logits[best]) as f64).exp()).sum();
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
        let scaling = (dim as f32).powf(-0.5);
        let mut out = vec![0.0f32; block * q_dim];
        for (unit, out) in out.chunks_exact_mut(dim).enumerate() {
            let (i, head) = (unit / heads, unit % heads);
            let kv = head / group;
            let query = &queries[i * q_dim + head * dim..][..dim];
            let sources = [(context_keys, context_values), (&keys[..], &values[..])];
            let scores: Vec<f32> = sources
                .iter()
                .flat_map(|(keys, _)| keys.chunks_exact(kv_dim).map(|key| dot(query, &key[kv * dim..][..dim]) * scaling))
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            let all_values = sources.iter().flat_map(|(_, values)| values.chunks_exact(kv_dim));
            for (weight, value) in weights.iter().zip(all_values) {
                let p = weight / total;
                out.iter_mut().zip(&value[kv * dim..][..dim]).for_each(|(o, v)| *o += p * v);
            }
        }
        layer.o.apply(&out)
    }
}
