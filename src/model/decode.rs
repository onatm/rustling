//! Generating a token entirely on the GPU, from a GGUF file.
//!
//! Token by token, the CPU path is held back less by arithmetic than by coordination: a token is hundreds of
//! short bursts of parallel work, each of which has to wake the thread pool, and anything else running on the
//! machine takes cores away. Here one token is a single submission to the GPU: every matrix product, norm,
//! convolution, attention and expert of all the layer visits, laid out once as a fixed list of kernels over one
//! scratch buffer and run again for each token with only the token's position changed.
//!
//! The model's state stays what it is everywhere else, the [`State`]. Its tables of keys and values are moved
//! into buffers the GPU shares the first time a token runs here; from then on the kernels read and add to the
//! same rows the CPU paths do, so prompts, batches and rollbacks there mix freely with tokens from here and
//! nothing is kept twice.
//!
//! The arithmetic is float throughout, where the CPU rounds the inputs of a quantised matrix to bytes, so the
//! two agree only approximately. This is how llama.cpp computes on a GPU too.

use anyhow::{Context, Result, ensure};

use super::{Controller, KvConv, Mat, Model, RESIDUAL_NORM_EPS, State};
use crate::ggml::Format;
use crate::gpu::{Gpu, Grid, Kernels, Shared, Step};

/// Tokens the full layers' tables hold at first; they double when a conversation outgrows them.
const FIRST_CAPACITY: usize = 8192;
/// The most tokens one submission takes: a token and as many proposals as can be taken back after it.
pub(crate) const MAX_BATCH: usize = super::MAX_ROLLBACK + 1;

// Buffers, by their place in the list.
const SCRATCH: usize = 0;
const CONSTANTS: usize = 1;
const SCORES: usize = 2;
const TABLES: usize = 3;

// Kernels, by their place in the two lists they are built from.
const ROW_KERNELS: [&str; 5] = ["row_q4k", "row_q5_0", "row_q6k", "row_q8_0", "row_f32"];
const STEP_KERNELS: [&str; 14] = [
    "rms_norm", "scale_clip", "residual_read", "merge", "conv", "distance_bias", "attention_scores",
    "attention_weights", "attention_values", "store", "route", "activation", "combine", "soft_cap",
];
const RMS_NORM: usize = 5;
const SCALE_CLIP: usize = 6;
const RESIDUAL_READ: usize = 7;
const MERGE: usize = 8;
const CONV: usize = 9;
const DISTANCE_BIAS: usize = 10;
const ATTENTION_SCORES: usize = 11;
const ATTENTION_WEIGHTS: usize = 12;
const ATTENTION_VALUES: usize = 13;
const STORE: usize = 14;
const ROUTE: usize = 15;
const ACTIVATION: usize = 16;
const COMBINE: usize = 17;
const SOFT_CAP: usize = 18;

fn row_kernel(format: Format) -> Result<usize> {
    Ok(match format {
        Format::Q4K => 0,
        Format::Q5_0 => 1,
        Format::Q6K => 2,
        Format::Q8_0 => 3,
        Format::F32 => 4,
        Format::Bf16 => anyhow::bail!("no GPU kernel for bfloat16 rows"),
    })
}

/// A value passed to a kernel: its parameter structs are made of 32-bit words.
#[derive(Clone, Copy)]
enum Word {
    Count(usize),
    Real(f32),
    /// Floats from one token's copy of the scratch layout to the next, known once the layout is complete.
    TokenStride,
}

fn words(values: &[Word], token_stride: usize) -> Vec<u8> {
    values
        .iter()
        .flat_map(|word| match *word {
            Word::Count(count) => (count as u32).to_le_bytes(),
            Word::Real(real) => real.to_le_bytes(),
            Word::TokenStride => (token_stride as u32).to_le_bytes(),
        })
        .collect()
}

/// Where one layer visit's results are in the scratch buffer, and its tables' buffers.
struct VisitSlots {
    /// The hidden state entering the visit.
    input: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    output: usize,
    /// The inputs of the four convolutions this token: raw keys, raw values, attention output, expert output.
    conv_inputs: [usize; 4],
    /// Where the `kernel - 1` inputs before them are expected.
    histories: [usize; 4],
    /// Where the kernels expect the visit's first position and its tables' capacities.
    info: usize,
    local: bool,
}

/// What a batch of tokens left behind.
pub(crate) struct Tokens {
    /// The final normed state of each token, one after another.
    pub last: Vec<f32>,
    /// For each of the visits asked for, the hidden state entering it for each token.
    pub taps: Vec<Vec<f32>>,
    /// Each token's soft-capped logits over the usable vocabulary.
    pub logits: Vec<Vec<f32>>,
}

pub(crate) struct Decoder {
    kernels: [Kernels; 2],
    steps: Vec<Step>,
    buffers: Vec<Shared>,
    visits: Vec<VisitSlots>,
    embed: usize,
    last: usize,
    logits: usize,
    /// Tokens the full layers' tables hold.
    capacity: usize,
    /// Floats from one token's copy of the scratch layout to the next.
    token_stride: usize,
    /// Seconds the GPU spent on the last token.
    pub(crate) gpu_seconds: f64,
    /// The scratch buffer of [`Decoder::experts`], and how many tokens it holds.
    short: Option<(usize, Shared)>,
}

/// Lays out the program: every helper appends kernels and returns where in the scratch buffer its result is.
struct Builder<'m, 'a> {
    model: &'m Model<'a>,
    gpu: &'m Gpu,
    /// The kernels so far, their parameters still to be packed.
    steps: Vec<(usize, Vec<(usize, usize)>, Vec<Word>, Grid)>,
    constants: Vec<f32>,
    weights: Vec<Shared>,
    first_weight: usize,
    /// Floats of scratch handed out so far.
    used: usize,
    visits: Vec<VisitSlots>,
    capacity: usize,
    /// Eight zero-based indices, for products whose matrices are not chosen by a kernel.
    in_order: usize,
}

impl<'m, 'a> Builder<'m, 'a> {
    fn alloc(&mut self, floats: usize) -> usize {
        let at = self.used;
        self.used += floats.next_multiple_of(4);
        at
    }

    fn constant(&mut self, values: &[f32]) -> (usize, usize) {
        let at = self.constants.len();
        self.constants.extend_from_slice(values);
        self.constants.resize(self.constants.len().next_multiple_of(4), 0.0);
        (CONSTANTS, at * 4)
    }

    fn weight(&mut self, bytes: &[u8]) -> Result<(usize, usize)> {
        let (buffer, offset) = self.gpu.wrap(bytes)?;
        self.weights.push(buffer);
        Ok((self.first_weight + self.weights.len() - 1, offset))
    }

    /// Appends a kernel. Every kernel's parameters end with the token stride.
    fn step(&mut self, kernel: usize, buffers: Vec<(usize, usize)>, params: &[Word], grid: Grid) {
        let params = params.iter().copied().chain([Word::TokenStride]).collect();
        self.steps.push((kernel, buffers, params, grid));
    }

    /// `matrices` products of a stack's matrices, chosen by `order`, with inputs `x_stride` floats apart.
    #[allow(clippy::too_many_arguments)]
    fn rows(
        &mut self,
        format: Format,
        bytes: &[u8],
        (rows, cols): (usize, usize),
        matrices: usize,
        order: (usize, usize),
        x: usize,
        x_stride: usize,
    ) -> Result<usize> {
        let row_bytes = format.row_bytes(cols).context("rows of whole blocks")?;
        ensure!(cols % 64 == 0, "the row kernels take 64 columns at a time");
        let out = self.alloc(rows * matrices);
        let weights = self.weight(bytes)?;
        // More rows to a group of threads for a large product, where starting threads is what costs.
        let together = match rows * matrices {
            work if work >= 16384 => 8,
            work if work >= 1024 => 4,
            _ => 1,
        };
        // An order in the scratch buffer is each token's own.
        let own_order = usize::from(order.0 == SCRATCH);
        let params = [
            Word::Count(rows),
            Word::Count(cols),
            Word::Count(rows * row_bytes),
            Word::Count(together),
            Word::Count(x_stride),
            Word::TokenStride,
            Word::Count(own_order),
        ];
        let buffers = vec![weights, (SCRATCH, x * 4), (SCRATCH, out * 4), order];
        // Not through `step`: the stride is not last here.
        self.steps.push((row_kernel(format)?, buffers, params.to_vec(), Grid::Groups(rows.div_ceil(together), matrices)));
        Ok(out)
    }

    fn product(&mut self, matrix: &Mat, x: usize) -> Result<usize> {
        self.rows(matrix.format, &matrix.bytes, (matrix.rows, matrix.cols), 1, (CONSTANTS, self.in_order * 4), x, 0)
    }

    /// `count` vectors of `n`, each normed, into `out`.
    fn norm_into(&mut self, x: usize, out: usize, weight: Option<&[f32]>, n: usize, count: usize, eps: f32) {
        let weights = weight.map_or((CONSTANTS, 0), |weight| self.constant(weight));
        // Enough threads that each has only a few elements.
        let width = (n / 256).clamp(1, 8);
        let params = [Word::Count(n), Word::Count(usize::from(weight.is_some())), Word::Real(eps), Word::Count(width)];
        let grid = Grid::Wide { groups: count, threads: 32 * width };
        self.step(RMS_NORM, vec![(SCRATCH, x * 4), (SCRATCH, out * 4), weights], &params, grid);
    }

    fn norm(&mut self, x: usize, weight: Option<&[f32]>, n: usize, count: usize, eps: f32) -> usize {
        let out = self.alloc(n * count);
        self.norm_into(x, out, weight, n, count, eps);
        out
    }

    fn scale_clip(&mut self, x: usize, scale: Option<&[f32]>, n: usize, count: usize, threshold: f32, long_range: bool) -> usize {
        let out = self.alloc(n * count);
        let scales = scale.map_or((CONSTANTS, 0), |scale| self.constant(scale));
        let params = [
            Word::Count(n),
            Word::Count(usize::from(scale.is_some())),
            Word::Real(threshold),
            Word::Count(usize::from(long_range)),
        ];
        let buffers = vec![(SCRATCH, x * 4), (SCRATCH, out * 4), scales, (SCRATCH, 0)];
        self.step(SCALE_CLIP, buffers, &params, Grid::Groups(count, 1));
        out
    }

    /// One read of the residual stream per query: over the finished `blocks` and the block in progress.
    fn read(&mut self, queries: &[f32], blocks: &[usize], partial: usize) -> usize {
        let h = self.model.cfg.hidden_size;
        assert!(blocks.len() <= 5, "the read kernel takes at most five finished blocks");
        let count = queries.len() / h;
        let out = self.alloc(queries.len());
        let mut buffers = vec![self.constant(queries), (SCRATCH, partial * 4), (SCRATCH, out * 4)];
        // Unused block bindings still have to point somewhere.
        buffers.extend((0..5).map(|i| (SCRATCH, blocks.get(i).copied().unwrap_or(partial) * 4)));
        let width = (h / 256).clamp(1, 8);
        let params = [Word::Count(h), Word::Count(blocks.len()), Word::Real(RESIDUAL_NORM_EPS), Word::Count(width)];
        self.step(RESIDUAL_READ, buffers, &params, Grid::Wide { groups: count, threads: 32 * width });
        out
    }

    /// A previous pass's message merged into `hidden` through the controller's gates.
    fn merge(&mut self, controller: &Controller, hidden: usize, message: usize) -> Result<usize> {
        let (h, eps) = (self.model.cfg.hidden_size, self.model.cfg.rms_norm_eps);
        let pair = self.alloc(2 * h);
        self.norm_into(hidden, pair, None, h, 1, eps);
        self.norm_into(message, pair + h, None, h, 1, eps);
        let low = self.product(&controller.down, pair)?;
        let (forget, gain) = (self.product(&controller.forget, low)?, self.product(&controller.gain, low)?);
        let out = self.alloc(h);
        let buffers = [hidden, message, forget, gain, out].map(|at| (SCRATCH, at * 4)).to_vec();
        self.step(MERGE, buffers, &[], Grid::Threads(h, 1));
        Ok(out)
    }

    /// The short convolution over this token's `x` and the inputs before it, which are expected at the
    /// returned history slot. `taps` selects the dynamic form; `residual` is added to the result when given.
    fn conv(&mut self, x: usize, weight: &[f32], channels: usize, taps: Option<usize>, residual: Option<usize>) -> (usize, usize) {
        let cfg = &self.model.cfg;
        let (k, head_dim) = (cfg.sconv_kernel_size, cfg.head_dim);
        let history = self.alloc((k - 1) * channels);
        let out = self.alloc(channels);
        let scale = 1.0 / ((if taps.is_some() { 2 * k + 1 } else { k + 1 }) as f32).sqrt();
        let params = [
            Word::Count(channels),
            Word::Count(k),
            Word::Count(head_dim),
            Word::Count(usize::from(taps.is_some())),
            Word::Real(scale),
            Word::Count(usize::from(residual.is_some())),
        ];
        let buffers = vec![
            (SCRATCH, x * 4),
            (SCRATCH, history * 4),
            self.constant(weight),
            (SCRATCH, taps.unwrap_or(x) * 4),
            (SCRATCH, residual.unwrap_or(x) * 4),
            (SCRATCH, out * 4),
            (SCRATCH, 0),
        ];
        self.step(CONV, buffers, &params, Grid::Threads(channels, 1));
        (out, history)
    }

    /// One layer visit: returns where its output is.
    fn visit(&mut self, l: usize, input: usize, blocks: &mut Vec<usize>, messages: Option<[usize; 4]>) -> Result<usize> {
        let model = self.model;
        let cfg = &model.cfg;
        let layer = &model.layers[l];
        let (h, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
        let (heads, kv_heads, dim, d_rel) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.d_rel);
        let kv_dim = kv_heads * dim;
        let local = layer.local;
        let visit = self.visits.len();
        let starts_block = l % cfg.n_layers_per_attn_res_block == 0;
        let normed_input = starts_block.then(|| self.norm(input, None, h, 1, RESIDUAL_NORM_EPS));
        let gates = layer.controllers.as_ref().zip(messages);

        // Attention inputs: read from the residual blocks, merge the previous pass's messages, then norm.
        let reads: Vec<usize> = match &layer.attn_query {
            Some(query) => {
                let out = self.read(query, blocks, input);
                (0..query.len() / h).map(|i| out + i * h).collect()
            }
            None => vec![input],
        };
        let mut inputs = [0usize; 3];
        for i in 0..3 {
            // A sliding layer without messages gives its query, keys and values the same input.
            if i > 0 && local && gates.is_none() {
                inputs[i] = inputs[0];
                continue;
            }
            let read = reads[if local { 0 } else { i }];
            let merged = match gates {
                Some((controllers, messages)) => self.merge(&controllers[i], read, messages[i])?,
                None => read,
            };
            inputs[i] = self.norm(merged, Some(&layer.input_norms[if local { 0 } else { i }]), h, 1, eps);
        }
        let [q_in, k_in, v_in] = inputs;

        let q = self.product(&layer.wq, q_in)?;
        let k_raw = self.product(&layer.wk, k_in)?;
        let v_raw = self.product(&layer.wv, v_in)?;
        let rel = self.product(&layer.wr, q_in)?;
        let sinks = self.product(&layer.sink, q_in)?;
        let gate = self.product(&layer.gate, q_in)?;
        let q = self.scale_clip(q, Some(&layer.q_scale), dim, heads, cfg.qk_l2_clip_threshold, !local);

        let kv_conv = |builder: &mut Self, conv: &KvConv, raw: usize, dynamic_input: usize| -> Result<(usize, usize)> {
            Ok(match conv {
                KvConv::Static(weight) => builder.conv(raw, weight, kv_dim, None, None),
                KvConv::Dynamic { static_w, proj } => {
                    let taps = builder.product(proj, dynamic_input)?;
                    builder.conv(raw, static_w, kv_dim, Some(taps), None)
                }
            })
        };
        let (k, k_history) = kv_conv(self, &layer.k_conv, k_raw, k_in)?;
        let keys = self.scale_clip(k, None, dim, kv_heads, cfg.qk_l2_clip_threshold, false);
        let (v, v_history) = kv_conv(self, &layer.v_conv, v_raw, v_in)?;
        let values = self.norm(v, None, dim, kv_heads, eps);

        // The token's keys and values join the layer's tables before it attends, as it sees itself. The tables
        // are the state's own, bound when the program runs.
        let tables = [TABLES + 2 * visit, TABLES + 2 * visit + 1];
        let info = self.alloc(4);
        for (which, (from, table)) in [keys, values].into_iter().zip(tables).enumerate() {
            let params = [kv_heads, dim, which].map(Word::Count);
            let buffers = vec![(SCRATCH, from * 4), (table, 0), (SCRATCH, 0), (SCRATCH, info * 4)];
            self.step(STORE, buffers, &params, Grid::Threads(kv_dim, 1));
        }

        let rel = self.scale_clip(rel, Some(&layer.rel_scale), d_rel, heads, cfg.rel_l2_clip_threshold, !local);
        let extent = cfg.rel_extent;
        let bias = self.alloc(heads * extent);
        let buffers = vec![(SCRATCH, rel * 4), self.constant(&layer.rel_proj), (SCRATCH, bias * 4)];
        self.step(DISTANCE_BIAS, buffers, &[d_rel, extent].map(Word::Count), Grid::Threads(extent, heads));

        let value_width = if local { 2 } else { 4 };
        let attention = [
            Word::Count(heads),
            Word::Count(heads / kv_heads),
            Word::Count(dim),
            Word::Count(usize::from(local)),
            Word::Count(cfg.sliding_window_size),
            Word::Count(extent),
            Word::Real((dim as f32).powf(-0.5)),
            Word::Count(self.capacity),
            // Enough threads to a head that each has few keys: a window's worth, or a long conversation's.
            Word::Count(if local { 8 } else { 32 }),
            Word::Count(value_width),
        ];
        let weights_grid = Grid::Wide { groups: heads, threads: 32 * if local { 8 } else { 32 } };
        let buffers = vec![(SCRATCH, q * 4), (tables[0], 0), (SCRATCH, bias * 4), (SCORES, 0), (SCRATCH, 0), (SCRATCH, info * 4)];
        self.step(ATTENTION_SCORES, buffers, &attention, Grid::Keys { local, rows: kv_heads });
        self.step(ATTENTION_WEIGHTS, vec![(SCORES, 0), (SCRATCH, sinks * 4), (SCRATCH, 0)], &attention, weights_grid);
        let attended = self.alloc(heads * dim);
        let buffers = vec![(SCORES, 0), (tables[1], 0), (SCRATCH, gate * 4), (SCRATCH, attended * 4), (SCRATCH, 0), (SCRATCH, info * 4)];
        ensure!(heads / kv_heads <= 4, "the attention kernels hold four query heads to a key head");
        self.step(ATTENTION_VALUES, buffers, &attention, Grid::Wide { groups: heads * dim / 16, threads: 32 * value_width });
        let attn = self.product(&layer.wo, attended)?;

        // The first layer of a block starts a fresh residual; the block's input becomes a source instead.
        let (residual, attn_history) = self.conv(attn, &layer.attn_conv, h, None, (!starts_block).then_some(input));
        let mut sources = blocks.clone();
        sources.extend(normed_input);

        let mut x = self.read(&layer.mlp_query, &sources, residual);
        if let Some((controllers, messages)) = gates {
            x = self.merge(&controllers[3], x, messages[3])?;
        }
        let x = self.norm(x, Some(&layer.mlp_norm), h, 1, eps);

        let scores = self.product(&layer.router, x)?;
        let top = cfg.num_experts_per_tok;
        ensure!(cfg.n_routed_experts <= 256 && top <= 8, "the routing kernel holds 256 experts and picks 8");
        let (order, weight) = (self.alloc(top), self.alloc(top));
        let buffers = vec![(SCRATCH, scores * 4), self.constant(&layer.router_bias), (SCRATCH, order * 4), (SCRATCH, weight * 4)];
        let params = [Word::Count(cfg.n_routed_experts), Word::Count(top), Word::Real(cfg.route_scale)];
        self.step(ROUTE, buffers, &params, Grid::Wide { groups: 1, threads: cfg.n_routed_experts.next_multiple_of(32) });

        let experts = &layer.experts;
        let (gate_up_format, down_format) =
            (experts.gate_up.ggml().context("experts in a GGUF format")?, experts.down.ggml().context("experts in a GGUF format")?);
        ensure!(!experts.interleaved, "the activation kernel expects the gate rows before the up rows");
        let inter = cfg.intermediate_size;
        let picked = (SCRATCH, order * 4);
        let products = self.rows(gate_up_format, experts.gate_up.raw_parts().0, (2 * inter, h), top, picked, x, 0)?;
        let activated = self.alloc(top * inter);
        let params = [Word::Count(inter), Word::Real(cfg.mlp_gate_softcap), Word::Real(cfg.mlp_up_softcap)];
        self.step(ACTIVATION, vec![(SCRATCH, products * 4), (SCRATCH, activated * 4)], &params, Grid::Threads(inter, top));
        let outputs = self.rows(down_format, experts.down.raw_parts().0, (h, inter), top, picked, activated, inter)?;
        let mlp = self.alloc(h);
        let buffers = vec![(SCRATCH, outputs * 4), (SCRATCH, weight * 4), (SCRATCH, mlp * 4)];
        self.step(COMBINE, buffers, &[h, top].map(Word::Count), Grid::Threads(h, 1));

        let (output, mlp_history) = self.conv(mlp, &layer.mlp_conv, h, None, Some(residual));
        // A block boundary files the layer's input as a finished block.
        blocks.extend(normed_input);
        self.visits.push(VisitSlots {
            input,
            output,
            conv_inputs: [k_raw, v_raw, attn, mlp],
            histories: [k_history, v_history, attn_history, mlp_history],
            info,
            local,
        });
        Ok(output)
    }
}

impl Decoder {
    pub(crate) fn new(model: &Model, gpu: &Gpu) -> Result<Self> {
        Self::with_capacity(model, gpu, FIRST_CAPACITY)
    }

    fn with_capacity(model: &Model, gpu: &Gpu, capacity: usize) -> Result<Self> {
        let cfg = &model.cfg;
        let h = cfg.hidden_size;
        let (start, end) = (cfg.loop_config.core_start, cfg.loop_config.core_end);
        let visit_count = cfg.visits().len();
        let mut builder = Builder {
            model,
            gpu,
            steps: Vec::new(),
            constants: Vec::new(),
            weights: Vec::new(),
            first_weight: TABLES + 2 * visit_count,
            // The token's position and scale come first.
            used: 4,
            visits: Vec::new(),
            capacity,
            in_order: 0,
        };
        let in_order: Vec<f32> = (0..8u32).map(f32::from_bits).collect();
        builder.in_order = builder.constant(&in_order).1 / 4;

        // The same order of visits as `Model::run_in_pool`: prelude, the core once per pass, then the coda.
        let embed = builder.alloc(h);
        let mut hidden = embed;
        let mut blocks: Vec<usize> = Vec::new();
        for l in 0..start {
            hidden = builder.visit(l, hidden, &mut blocks, None)?;
        }
        let (prelude_hidden, prelude_blocks) = (hidden, blocks.clone());
        for l in start..end {
            hidden = builder.visit(l, hidden, &mut blocks, None)?;
        }
        for _ in 1..cfg.loop_config.num_passes {
            // Every pass restarts from the prelude and sees the previous pass only through messages.
            let previous_blocks = &blocks[prelude_blocks.len() + 1..];
            ensure!(previous_blocks.len() == 1, "message reads assume a loop core of exactly two blocks");
            let (previous_block, previous_hidden) = (previous_blocks[0], hidden);
            hidden = prelude_hidden;
            blocks = prelude_blocks.clone();
            for l in start..end {
                let read = builder.read(&model.message_queries[l - start], &[previous_block], previous_hidden);
                let messages = [read, read + h, read + 2 * h, read + 3 * h];
                hidden = builder.visit(l, hidden, &mut blocks, Some(messages))?;
            }
        }
        for l in end..cfg.num_hidden_layers {
            hidden = builder.visit(l, hidden, &mut blocks, None)?;
        }
        let read = builder.read(&model.output_query, &blocks, hidden);
        let last = builder.norm(read, Some(&model.final_norm), h, 1, cfg.rms_norm_eps);
        let (unembed, vocab) = (&model.unembed, cfg.unpadded_vocab_size);
        let in_order = (CONSTANTS, builder.in_order * 4);
        let logits = builder.rows(unembed.format, &unembed.bytes, (vocab, h), 1, in_order, last, 0)?;
        let params = [Word::Count(vocab), Word::Real(cfg.final_logit_softcapping)];
        builder.step(SOFT_CAP, vec![(SCRATCH, logits * 4)], &params, Grid::Threads(vocab, 1));

        let token_stride = builder.used;
        let steps = std::mem::take(&mut builder.steps)
            .into_iter()
            .map(|(kernel, buffers, params, grid)| Step { kernel, buffers, params: words(&params, token_stride), grid })
            .collect();
        let mut buffers = vec![
            gpu.shared(MAX_BATCH * token_stride * 4)?,
            gpu.shared(builder.constants.len() * 4)?,
            gpu.shared(MAX_BATCH * cfg.num_attention_heads * capacity * 4)?,
        ];
        buffers[CONSTANTS].floats_mut()[..builder.constants.len()].copy_from_slice(&builder.constants);
        // Places for the state's tables, which are swapped in for the length of a run.
        for _ in 0..2 * builder.visits.len() {
            buffers.push(gpu.shared(16)?);
        }
        buffers.append(&mut builder.weights);
        Ok(Self {
            kernels: [
                gpu.kernels(include_str!("../gguf.metal"), &ROW_KERNELS)?,
                gpu.kernels(include_str!("../decode.metal"), &STEP_KERNELS)?,
            ],
            steps,
            buffers,
            visits: builder.visits,
            embed,
            last,
            logits,
            capacity,
            token_stride,
            gpu_seconds: 0.0,
            short: None,
        })
    }

    /// Runs a batch of tokens, each seeing the ones before it, and adds them to `state`. `embedded` holds their
    /// embeddings one after another.
    pub(crate) fn tokens(&mut self, model: &Model, gpu: &Gpu, state: &mut State, embedded: &[f32], taps: &[usize]) -> Result<Tokens> {
        let cfg = &model.cfg;
        let (h, k, stride) = (cfg.hidden_size, cfg.sconv_kernel_size, self.token_stride);
        let count = embedded.len() / h;
        ensure!((1..=MAX_BATCH).contains(&count), "a batch of {count} tokens");
        let position = state.position;
        if position + count > self.capacity {
            *self = Self::with_capacity(model, gpu, (position + count).next_power_of_two().max(2 * self.capacity))?;
        }
        // The state's tables: old rows dropped as at the start of any pass, room made for the batch's rows, and
        // the buffers themselves put where the kernels are bound to them.
        self.prepare_tables(model, gpu, state, count)?;
        self.exchange_tables(state);

        let scratch = &mut self.buffers[SCRATCH];
        for t in 0..count {
            scratch.words_mut()[t * stride] = (position + t) as u32;
            let floats = scratch.floats_mut();
            floats[t * stride + 1] =
                1.0 + cfg.log_scaling_alpha * ((position + t + 1) as f32 / cfg.log_scaling_n_floor).max(1.0).ln();
            floats[t * stride + self.embed..][..h].copy_from_slice(&embedded[t * h..][..h]);
        }
        // What the convolutions saw before the batch, where the first token's copy of the layout expects it.
        let floats = scratch.floats_mut();
        for (slots, visit) in self.visits.iter().zip(&state.visits) {
            for ((history, at), channels) in visit.convs.iter().zip(slots.histories).zip(visit.conv_channels) {
                // The most recent inputs, ending where the kernel expects the latest.
                let wanted = ((k - 1) * channels).min(history.len());
                floats[at + (k - 1) * channels - wanted..][..wanted].copy_from_slice(&history[history.len() - wanted..]);
            }
        }

        let end = position + count;
        let keys = [end.min(cfg.sliding_window_size), end];
        let ran = gpu.run(&[&self.kernels[0], &self.kernels[1]], &self.steps, &self.buffers, keys, count);
        // The tables go back to the state whatever happened; only a run that worked adds to it.
        self.exchange_tables(state);
        self.gpu_seconds = ran?;

        let floats = self.buffers[SCRATCH].floats();
        for visit in &mut state.visits {
            // The kernels wrote the batch's keys and values into the tables.
            visit.keys.added(count);
            visit.values.added(count);
        }
        for t in 0..count {
            let mine = &floats[t * stride..][..stride];
            for (slots, visit) in self.visits.iter().zip(&mut state.visits) {
                let conv_inputs: [&[f32]; 4] = std::array::from_fn(|c| &mine[slots.conv_inputs[c]..][..visit.conv_channels[c]]);
                visit.push_conv_inputs(cfg, conv_inputs);
            }
        }
        state.position = end;

        let of_each = |at: usize, len: usize| -> Vec<&[f32]> { (0..count).map(|t| &floats[t * stride + at..][..len]).collect() };
        Ok(Tokens {
            last: of_each(self.last, h).concat(),
            taps: taps.iter().map(|&visit| of_each(self.visits[visit].input, h).concat()).collect(),
            logits: of_each(self.logits, cfg.unpadded_vocab_size).into_iter().map(<[f32]>::to_vec).collect(),
        })
    }

    /// A layer's experts for some tokens of a batch, the row kernels multiplying straight from the stored
    /// blocks, each token's experts at a time. `routes` holds each token's experts and their weights. Returns
    /// `tokens x hidden` outputs.
    pub(crate) fn experts(&mut self, model: &Model, gpu: &Gpu, layer: &super::Layer, x: &[f32], routes: &[(u32, f32)]) -> Result<Vec<f32>> {
        let cfg = &model.cfg;
        let (h, inter, top) = (cfg.hidden_size, cfg.intermediate_size, cfg.num_experts_per_tok);
        let tokens = x.len() / h;
        let experts = &layer.experts;
        let (gate_up_format, down_format) =
            (experts.gate_up.ggml().context("experts in a GGUF format")?, experts.down.ggml().context("experts in a GGUF format")?);
        ensure!(!experts.interleaved && routes.len() == tokens * top && top % 4 == 0, "bad expert batch");
        // A token's block of the scratch buffer: its input, its experts and their weights, the first products,
        // the activations, the second products and its output.
        let (order, weight, first) = (h, h + top, h + 2 * top);
        let active = first + top * 2 * inter;
        let second = active + top * inter;
        let out = second + top * h;
        let stride = out + h;
        if self.short.as_ref().is_none_or(|(held, _)| *held < tokens) {
            self.short = Some((tokens, gpu.shared(tokens * stride * 4)?));
        }
        let (held, mut scratch) = self.short.take().expect("just made");
        for (t, (x, routes)) in x.chunks_exact(h).zip(routes.chunks_exact(top)).enumerate() {
            for (k, &(expert, _)) in routes.iter().enumerate() {
                scratch.words_mut()[t * stride + order + k] = expert;
            }
            let block = &mut scratch.floats_mut()[t * stride..][..stride];
            block[..h].copy_from_slice(x);
            for (k, &(_, weight_of)) in routes.iter().enumerate() {
                block[weight + k] = weight_of;
            }
        }

        let (gate_up, gate_up_at) = gpu.wrap(experts.gate_up.raw_parts().0)?;
        let (down, down_at) = gpu.wrap(experts.down.raw_parts().0)?;
        let buffers = [scratch, gate_up, down];
        let rows = |format: Format, buffer: (usize, usize), (rows, cols): (usize, usize), x: usize, x_stride: usize, to: usize| -> Result<Step> {
            let row_bytes = format.row_bytes(cols).context("rows of whole blocks")?;
            let params = [rows, cols, rows * row_bytes, 8, x_stride, stride, 1].map(Word::Count);
            Ok(Step {
                kernel: row_kernel(format)?,
                buffers: vec![buffer, (0, x * 4), (0, to * 4), (0, order * 4)],
                params: words(&params, stride),
                grid: Grid::Groups(rows.div_ceil(8), top),
            })
        };
        let activation = [Word::Count(inter), Word::Real(cfg.mlp_gate_softcap), Word::Real(cfg.mlp_up_softcap), Word::TokenStride];
        let steps = [
            rows(gate_up_format, (1, gate_up_at), (2 * inter, h), 0, 0, first)?,
            Step { kernel: ACTIVATION, buffers: vec![(0, first * 4), (0, active * 4)], params: words(&activation, stride), grid: Grid::Threads(inter, top) },
            rows(down_format, (2, down_at), (h, inter), active, inter, second)?,
            Step {
                kernel: COMBINE,
                buffers: vec![(0, second * 4), (0, weight * 4), (0, out * 4)],
                params: words(&[Word::Count(h), Word::Count(top), Word::TokenStride], stride),
                grid: Grid::Threads(h, 1),
            },
        ];
        let ran = gpu.run(&[&self.kernels[0], &self.kernels[1]], &steps, &buffers, [0, 0], tokens);
        let [scratch, ..] = buffers;
        let outputs = (0..tokens).flat_map(|t| scratch.floats()[t * stride + out..][..h].iter().copied()).collect();
        self.short = Some((held, scratch));
        ran?;
        Ok(outputs)
    }

    /// Gets the state's tables ready for `count` more tokens: rows that have left a sliding layer's window
    /// dropped, as at the start of any pass; the tables in shared buffers with room for the new rows; and where
    /// they start and how much they hold written where the kernels look.
    fn prepare_tables(&mut self, model: &Model, gpu: &Gpu, state: &mut State, count: usize) -> Result<()> {
        let words = self.buffers[SCRATCH].words_mut();
        for (slots, visit) in self.visits.iter().zip(&mut state.visits) {
            visit.drop_stale(&model.cfg, slots.local, state.position);
            for table in [&mut visit.keys, &mut visit.values] {
                table.share(gpu)?;
                table.reserve(count);
            }
            words[slots.info] = visit.first as u32;
            words[slots.info + 1] = visit.keys.capacity() as u32;
            words[slots.info + 2] = visit.values.capacity() as u32;
        }
        Ok(())
    }

    /// Swaps the state's tables with the buffers in the places the kernels are bound to: in before a run, and
    /// back out after it.
    fn exchange_tables(&mut self, state: &mut State) {
        for (v, visit) in state.visits.iter_mut().enumerate() {
            for (which, table) in [&mut visit.keys, &mut visit.values].into_iter().enumerate() {
                let shared = table.shared_mut().expect("shared before the first exchange");
                std::mem::swap(shared, &mut self.buffers[TABLES + 2 * v + which]);
            }
        }
    }

    /// GPU milliseconds per kernel for a token at the tables' present state, and how often each runs.
    #[cfg(test)]
    pub(crate) fn profile(&mut self, model: &Model, gpu: &Gpu, state: &mut State) -> Vec<(&'static str, usize, f64)> {
        self.prepare_tables(model, gpu, state, 1).unwrap();
        self.exchange_tables(state);
        self.buffers[SCRATCH].words_mut()[0] = state.position as u32;
        let keys = [(state.position + 1).min(model.cfg.sliding_window_size), state.position + 1];
        let names: Vec<&'static str> = ROW_KERNELS.iter().chain(&STEP_KERNELS).copied().collect();
        let mut profile = Vec::new();
        for (kernel, name) in names.iter().enumerate() {
            let steps: Vec<Step> = self
                .steps
                .iter()
                .filter(|step| step.kernel == kernel)
                .map(|step| Step { kernel, buffers: step.buffers.clone(), params: step.params.clone(), grid: step.grid })
                .collect();
            if steps.is_empty() {
                continue;
            }
            let run = || gpu.run(&[&self.kernels[0], &self.kernels[1]], &steps, &self.buffers, keys, 1).unwrap();
            let best = (0..3).map(|_| run()).fold(f64::MAX, f64::min);
            profile.push((*name, steps.len(), best * 1e3));
        }
        self.exchange_tables(state);
        profile
    }

    /// The output of each layer visit for the last token, for checking against the CPU.
    #[cfg(test)]
    pub(crate) fn visit_outputs(&self, h: usize) -> Vec<Vec<f32>> {
        let floats = self.buffers[SCRATCH].floats();
        self.visits.iter().map(|visit| floats[visit.output..][..h].to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{Checkpoint, resolve_model};
    use crate::config::Config;
    use crate::model::{Activations, Backend, FLOAT_REFERENCE, Options};

    /// The float reference is a switch for the whole process, and the tests are heavy enough on the GPU that
    /// their timings mean nothing side by side, so they take turns.
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn my_turn() -> std::sync::MutexGuard<'static, ()> {
        TURN.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The experts as the decoder's kernels do them for part of a short batch, against the CPU in floats.
    #[test]
    #[ignore]
    fn the_experts_of_a_short_batch_match_the_cpu_in_floats() {
        let _turn = my_turn();
        let path = resolve_model(Some(std::path::Path::new("lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M"))).unwrap();
        let weights = Checkpoint::open(&path).unwrap();
        let config = Config::load(&path).unwrap();
        let options = Options { activations: Activations::Float, backend: Backend::MetalFast };
        let model = Model::load(&config, &weights, options).unwrap();
        let gpu = model.gpu.as_ref().unwrap();
        let h = model.cfg.hidden_size;
        // Inputs of about the size the experts see: normed states.
        let x: Vec<f32> = (0..40 * h).map(|i| (((i * 2_654_435_761) >> 9) % 2001) as f32 / 1000.0 - 1.0).collect();
        let mut decoder = Decoder::new(&model, gpu).unwrap();
        for l in [0, 7, 23] {
            let layer = &model.layers[l];
            FLOAT_REFERENCE.store(true, std::sync::atomic::Ordering::Relaxed);
            let expected = model.experts(layer, &x);
            FLOAT_REFERENCE.store(false, std::sync::atomic::Ordering::Relaxed);
            let actual = decoder.experts(&model, gpu, layer, &x, &model.routes(layer, &x)).unwrap();
            let diff: f64 = actual.iter().zip(&expected).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
            let apart = (diff / expected.iter().map(|b| (*b as f64).powi(2)).sum::<f64>()).sqrt();
            eprintln!("layer {l}: apart by {apart:.1e}");
            assert!(apart < 1e-5, "layer {l}: the experts differ by {apart:.1e}");
        }
    }

    /// A batch must give each of its tokens exactly what it gets alone after the ones before it: at the start
    /// of a conversation, where the convolutions run out of history, and across the point where the sliding
    /// layers' rings wrap. `cargo test --release --lib decode::tests::a_batch -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn a_batch_matches_tokens_one_at_a_time() {
        let _turn = my_turn();
        let path = resolve_model(Some(std::path::Path::new("lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M"))).unwrap();
        let weights = Checkpoint::open(&path).unwrap();
        let config = Config::load(&path).unwrap();
        let options = Options { activations: Activations::Float, backend: Backend::MetalFast };
        let model = Model::load(&config, &weights, options).unwrap();
        let h = model.cfg.hidden_size;
        let ids = |from: u32, count: u32| -> Vec<u32> { (from..from + count).map(|i| 500 + (i * 7919) % 50_000).collect() };
        for (prompt, batches) in [(0u32, vec![2u32, 17, 5]), (1075, vec![17, 9, 17])] {
            let (mut together, mut alone) = (model.new_state(), model.new_state());
            for chunk in ids(0, prompt).chunks(275) {
                model.forward(chunk, &mut together, false).unwrap();
                model.forward(chunk, &mut alone, false).unwrap();
            }
            let mut from = prompt;
            for count in batches {
                let batch = ids(from, count);
                from += count;
                let started = std::time::Instant::now();
                let all = model.forward(&batch, &mut together, false).unwrap();
                let seconds = started.elapsed().as_secs_f64();
                let logits = all.logits.expect("a batch from the GPU has logits");
                for (t, &id) in batch.iter().enumerate() {
                    let one = model.forward(&[id], &mut alone, false).unwrap();
                    assert!(one.last[..] == all.last[t * h..][..h], "token {t} of a batch of {count} after {prompt} differs");
                    assert!(one.logits.unwrap()[0] == logits[t], "logits of token {t} of a batch of {count} differ");
                }
                eprintln!("a batch of {count} at {}: {:.1} ms, {:.1} ms a token", from - count, seconds * 1e3, seconds * 1e3 / count as f64);
            }
        }
    }

    /// Where a token's time goes with a long conversation behind it.
    /// `CONTEXT=15000 cargo test --release --lib decode::tests::profile -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn profile_at_long_context() {
        let _turn = my_turn();
        let context: usize = std::env::var("CONTEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(15_000);
        let path = resolve_model(Some(std::path::Path::new("lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M"))).unwrap();
        let weights = Checkpoint::open(&path).unwrap();
        let config = Config::load(&path).unwrap();
        let options = Options { activations: Activations::Float, backend: Backend::MetalFast };
        let model = Model::load(&config, &weights, options).unwrap();
        let gpu = model.gpu.as_ref().unwrap();
        let mut decoder = Decoder::with_capacity(&model, gpu, (context + 1).next_power_of_two()).unwrap();
        // A state of that length, its tables filled with made-up rows.
        let cfg = &model.cfg;
        let mut state = model.new_state();
        state.position = context;
        for (visit, held) in cfg.visits().iter().zip(&mut state.visits) {
            let rows = if cfg.is_local(visit.layer) { context.min(cfg.sliding_window_size - 1) } else { context };
            let table = || super::super::Table::from_heads(vec![vec![0.25f32; rows * cfg.head_dim]; cfg.num_key_value_heads], cfg.head_dim);
            (held.keys, held.values, held.first) = (table(), table(), context - rows);
        }
        let mut total = 0.0;
        for (name, count, ms) in decoder.profile(&model, gpu, &mut state) {
            total += ms;
            eprintln!("  {name}: {count} dispatches, {ms:.2} ms");
        }
        eprintln!("  in all {total:.1} ms");
    }

    /// Tokens from the GPU mixed with everything else that changes a state: prompts and batches on the CPU,
    /// tokens taken back, a rewind to a mark, and a context long enough for the sliding layers' rings to wrap.
    /// After each, the token must be exactly what another decoder computes from a copy of the state.
    /// `cargo test --release --lib decode::tests::keeps -- --ignored`.
    #[test]
    #[ignore]
    fn keeps_its_tables_in_step_with_the_state() {
        let _turn = my_turn();
        let named = std::env::var("RUSTLING_GGUF").unwrap_or_else(|_| "lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M".to_owned());
        let path = resolve_model(Some(std::path::Path::new(&named))).unwrap();
        let weights = Checkpoint::open(&path).unwrap();
        let config = Config::load(&path).unwrap();
        let options = Options { activations: Activations::Float, backend: Backend::MetalFast };
        let model = Model::load(&config, &weights, options).unwrap();
        let gpu = model.gpu.as_ref().unwrap();
        let mut refilled = Decoder::new(&model, gpu).unwrap();
        let mut next = 0u32;
        let mut ids = |count: usize| -> Vec<u32> {
            (0..count)
                .map(|_| {
                    next += 1;
                    500 + (next * 7919) % 50_000
                })
                .collect()
        };
        let mut state = model.new_state();
        let mut checked = 0;
        // One token through the model's own decoder, checked against a refill.
        let mut single = |state: &mut State, id: u32, what: &str| {
            let mut copy = state.clone();
            let expected = refilled.tokens(&model, gpu, &mut copy, &model.embed(id).unwrap(), &[]).unwrap();
            let actual = model.forward(&[id], state, false).unwrap();
            assert!(actual.last == expected.last, "{what}: the token differs from one computed on refilled tables");
            assert!(actual.logits == Some(expected.logits), "{what}: logits differ");
            checked += 1;
        };

        // A prompt on the CPU paths, long enough that the rings wrap during what follows.
        for chunk in ids(1100).chunks(275) {
            model.forward(chunk, &mut state, false).unwrap();
        }
        single(&mut state, ids(1)[0], "after a prompt");
        single(&mut state, ids(1)[0], "after a token");
        model.forward(&ids(5), &mut state, false).unwrap();
        single(&mut state, ids(1)[0], "after a batch");
        model.forward(&ids(9), &mut state, false).unwrap();
        state.truncate(6);
        single(&mut state, ids(1)[0], "after taking tokens back");
        // Tokens from the GPU taken back, as when the next request repeats the prompt.
        let mark = model.mark(&state);
        for id in ids(70) {
            single(&mut state, id, "in a run of tokens");
        }
        model.forward(&ids(4), &mut state, false).unwrap();
        state.rewind(&mark);
        single(&mut state, ids(1)[0], "after a rewind");
        // A rewind further than the rings' slack.
        let mark = model.mark(&state);
        for chunk in ids(200).chunks(100) {
            model.forward(chunk, &mut state, false).unwrap();
        }
        single(&mut state, ids(1)[0], "after a long batch");
        state.rewind(&mark);
        single(&mut state, ids(1)[0], "after a long rewind");
        // And a copy of the state carries on where the original was.
        let mut copy = state.clone();
        single(&mut state, ids(1)[0], "the original");
        single(&mut copy, ids(1)[0], "a copy");
        single(&mut state, ids(1)[0], "the original again");
        eprintln!("{checked} tokens matched at position {}", state.position());
    }

    /// Each layer visit of each token against the CPU computing in floats, which is the same arithmetic in
    /// another order. `cargo test --release --lib decode::tests -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn matches_the_cpu_in_floats() {
        let _turn = my_turn();
        let named = std::env::var("RUSTLING_GGUF").unwrap_or_else(|_| "lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M".to_owned());
        let path = resolve_model(Some(std::path::Path::new(&named))).unwrap();
        let weights = Checkpoint::open(&path).unwrap();
        let config = Config::load(&path).unwrap();
        let options = Options { activations: Activations::Float, backend: Backend::MetalFast };
        let model = Model::load(&config, &weights, options).unwrap();
        let gpu = model.gpu.as_ref().unwrap();
        let h = model.cfg.hidden_size;
        FLOAT_REFERENCE.store(true, std::sync::atomic::Ordering::Relaxed);

        let ids: Vec<u32> = (0..12).map(|i| 1000 + 37 * i).collect();
        let mut decoder = Decoder::new(&model, gpu).unwrap();
        let (mut on_cpu, mut on_gpu) = (model.new_state(), model.new_state());
        let apart = |a: &[f32], b: &[f32]| {
            let diff: f64 = a.iter().zip(b).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
            (diff / b.iter().map(|b| (*b as f64).powi(2)).sum::<f64>()).sqrt()
        };
        for (t, &id) in ids.iter().enumerate() {
            let expected = model.forward(&[id], &mut on_cpu, true).unwrap();
            let started = std::time::Instant::now();
            let token = decoder.tokens(&model, gpu, &mut on_gpu, &model.embed(id).unwrap(), &[]).unwrap();
            let last = token.last;
            let logits_apart = apart(&token.logits[0], &model.logits(&expected.last).unwrap());
            let seconds = started.elapsed().as_secs_f64();
            let visits = decoder.visit_outputs(h);
            let by_visit: Vec<f64> = visits.iter().zip(&expected.hidden).map(|(a, b)| apart(a, b)).collect();
            let worst = by_visit.iter().copied().fold(0.0, f64::max);
            let first_off = by_visit.iter().position(|d| *d > 1e-3);
            eprintln!(
                "token {t}: final state apart by {:.1e}, logits by {logits_apart:.1e}, worst visit {worst:.1e}, first visit off {first_off:?}; {:.1} ms, GPU {:.1} ms",
                apart(&last, &expected.last),
                seconds * 1e3,
                decoder.gpu_seconds * 1e3
            );
            if t == 0 || first_off.is_some() {
                let shown: Vec<String> = by_visit.iter().map(|d| format!("{d:.0e}")).collect();
                eprintln!("  by visit: {}", shown.join(" "));
            }
        }
        FLOAT_REFERENCE.store(false, std::sync::atomic::Ordering::Relaxed);
        for (name, count, ms) in decoder.profile(&model, gpu, &mut on_gpu) {
            eprintln!("  {name}: {count} dispatches, {ms:.2} ms");
        }
    }
}
