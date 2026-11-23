//! Text-only forward pass, correctness first: f32 throughout, weights decoded on the fly.
//!
//! Ported from transformers' `modeling_inkling2_loop.py`. A sequence of `T` vectors of width `D` is a flat
//! `Vec<f32>` of length `T * D`.
//!
//! Decoding is incremental: [`State`] carries what earlier tokens left behind, so each call only computes the
//! tokens it is given. The only cross-token state is the attention keys and values and the short convolutions'
//! recent inputs, and every layer visit has its own copy of both.

use anyhow::{Context, Result, ensure};
use half::bf16;
use crate::weights::ggml::{Format, INT8_MAX_BATCH, Int8Activation, dot_row_int8_many, quantize_int8};
use rayon::prelude::*;

use crate::accel;
use crate::weights::checkpoint::{Checkpoint, ExpertMatrix, Experts};
use crate::config::{Config, TextConfig};
#[cfg(target_os = "macos")]
use crate::gpu::{AttendHead, AttendShape, DenseJob, ExpertJob, Gpu, Stack, WholeExperts};
use crate::timing::{self, Stage, time};
use crate::trace;
use crate::weights::quant::{
    NVFP4_MAX_BATCH, Nvfp4Activation, dot_f32_row_many, quantize_nvfp4,
    requantize_mxfp8,
};

pub(crate) type Seq = Vec<f32>;

/// `UnweightedRMSNorm` in the reference is built with its default epsilon in the attention residuals.
const RESIDUAL_NORM_EPS: f32 = 1e-6;

/// Below this many multiply-adds a batch of matrix products runs on the calling thread: waking the pool costs more.
const PARALLEL_MIN_WORK: usize = 1 << 20;

/// Batches of at least this many tokens go to the GPU when there is one. Below it, submitting the work costs
/// more than it saves; the results are the same either way.
const GPU_MIN_TOKENS: usize = 32;

/// The fraction of an expert layer's work given to the GPU; the CPU does the rest at the same time. The two are
/// about equally fast at it.
#[cfg(target_os = "macos")]
const GPU_EXPERT_SHARE: f32 = 0.5;

/// The fraction of a GGUF expert layer's work given to the GPU on the fast path, the CPU doing the rest at the
/// same time. Measured on an M1 Max: the two get in each other's way, so the GPU alone is nearly as fast, and
/// this split is a little faster still.
#[cfg(target_os = "macos")]
const GPU_FAST_EXPERT_SHARE: f32 = 0.75;

/// How many experts' GPU results had to be recomputed on the CPU, for the timing report.
pub static GPU_EXPERTS_REDONE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How many stretches of attention had to be recomputed on the CPU, for the timing report.
pub static GPU_ATTENTION_REDONE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How many batches of plain matrix products, and of the controllers' merges, had to be recomputed because the
/// GPU's fast product returned values that are not numbers.
pub static GPU_PRODUCTS_REDONE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How many of those came back wrong from the GPU a second time, in short runs, and went to the CPU.
pub static GPU_PRODUCTS_ON_CPU: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Counts a batch of fast products that came back with NaN: at the first try, or at the second in short runs.
fn products_redone(in_runs: bool) {
    let count = if in_runs { &GPU_PRODUCTS_ON_CPU } else { &GPU_PRODUCTS_REDONE };
    count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Prompt tokens per pass on the fast path. Unpacking the weights for the GPU costs the same however many tokens
/// follow, so long passes pay off: a 2,400-token prompt took 29 s in passes of 512 and 23 s in passes of 2,048,
/// and a 15,000-token one 211 s in passes of 2,048 and 175 s in passes of 4,096. Longer made no difference.
const FAST_PROMPT_CHUNK: usize = 4096;

/// The most tokens of a sliding layer whose fast attention is computed together.
#[cfg(target_os = "macos")]
const FAST_LOCAL_STRETCH: usize = 1024;

/// The share of a short batch's tokens whose experts the GPU does with the decoder's kernels while the CPU does
/// the others'. With 150 tokens at 10,000 tokens of context the experts took 7.4 ms a token at this share,
/// 8.4 ms at 0.3, 8.9 ms at 0.6 and 10.8 ms on the CPU alone.
#[cfg(target_os = "macos")]
const GPU_ROW_EXPERT_SHARE: f32 = 0.45;

/// From this many tokens a GGUF layer's experts run as one submission on the GPU, which unpacks every expert's
/// weights: about 6 s a pass on an M1 Max however few tokens there are. A shorter batch does better with the GPU
/// unpacking only the busiest experts and the CPU doing the rest: 282 tokens took 4.1 s that way against 6.1 s,
/// and 1,202 tokens 12.5 s against 13.1 s.
#[cfg(target_os = "macos")]
const WHOLE_EXPERTS_MIN_TOKENS: usize = 1536;

/// A stretch of fast attention goes to the GPU when its rows times its keys come to at least this.
#[cfg(target_os = "macos")]
const GPU_ATTENTION_MIN_WEIGHTS: usize = 1 << 20;

/// Prompt tokens per pass otherwise: enough to share each read of the weights, with memory still bounded.
const PROMPT_CHUNK: usize = 512;

/// Batches of at least this many tokens take the fast attention, which pays off much sooner than the GPU does: the
/// last stretch of a long prompt after the part a cached one covered, and the handful of tokens checked at a time
/// with a draft model, where at 15,000 tokens of context it takes a third of the time. A single token is quicker
/// on the exact path.
#[cfg(target_os = "macos")]
const FAST_ATTENTION_MIN_TOKENS: usize = 2;

/// The most weights (tokens' heads times keys) the fast attention holds at once, which bounds its memory to a
/// quarter of a gigabyte however long the context.
#[cfg(target_os = "macos")]
const FAST_ATTENTION_WEIGHTS: usize = 1 << 26;

/// Batches of at least this many tokens take the GPU's fast path when it was asked for: tuned float matrix products
/// in place of the exact kernels. Below it the fast path's fixed costs outweigh its speed.
#[cfg(target_os = "macos")]
const FAST_MIN_TOKENS: usize = 256;

/// (expert, token) products per GPU submission, which bounds the size of the output buffer.
#[cfg(target_os = "macos")]
const GPU_PAIR_BATCH: usize = 4096;

/// Stands in for the Metal backend where there is none; it has no values, so `Option<Gpu>` is always `None`.
#[cfg(not(target_os = "macos"))]
pub(crate) enum Gpu {}

/// Fewer tokens than this are handled on the calling thread in the cheap per-token steps.
const PARALLEL_MIN_TOKENS: usize = 8;

/// Attention goes across threads, a head each, from this many products of a query value with a key value. A
/// lone token reaches it with about fifty tokens of context.
const ATTENTION_MIN_WORK: usize = 1 << 17;

/// Tokens that attend together, and the keys they take at a time: the queries of a block and one stretch of keys
/// and values fit in a core's cache, so the context is read once per block.
const ATTENTION_BLOCK: usize = 32;
const ATTENTION_KEYS: usize = 64;

/// How many tokens' keys and values a sliding-window layer lets pile up beyond its window before dropping them.
const STALE_BATCH: usize = 128;

/// Rows per parallel task in a matrix product.
const TASK_ROWS: usize = 128;

/// Tokens that share one pass over a block of weight rows. Large enough that the weights are read once for many
/// tokens, small enough that the tokens' vectors stay in cache while the rows go by.
const TOKEN_TILE: usize = NVFP4_MAX_BATCH;

/// Tokens per tile in the dense products, where each token's vector is four times the size.
const DENSE_TILE: usize = 16;

#[cfg(target_os = "macos")]
mod attention;
mod controller;
mod decode;
mod experts;
mod mat;
mod ops;
mod state;
mod table;

#[allow(unused_imports)]
use experts::*;
pub(crate) use mat::*;
#[allow(unused_imports)]
pub(crate) use ops::*;
pub use state::{MAX_ROLLBACK, Mark, State};
use state::*;

use table::Table;

/// Makes the CPU multiply quantised matrices by float inputs, as the GPU decoder does, where it normally rounds
/// the inputs to bytes first. Slow; for checking the decoder against.
pub(crate) static FLOAT_REFERENCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn float_reference() -> bool {
    FLOAT_REFERENCE.load(std::sync::atomic::Ordering::Relaxed)
}

pub struct Model<'a> {
    cfg: TextConfig,
    embed: Mat<'a>,
    unembed: Mat<'a>,
    activations: Activations,
    gpu: Option<Gpu>,
    /// Generates single tokens on the GPU; built at the first one. Nothing when that failed.
    #[cfg(target_os = "macos")]
    decoder: std::sync::OnceLock<Option<std::sync::Mutex<decode::Decoder>>>,
    embed_norm: Vec<f32>,
    final_norm: Vec<f32>,
    output_query: Vec<f32>,
    /// Per looped layer: the query, key, value and MLP message queries, `4 * hidden`.
    message_queries: Vec<Vec<f32>>,
    layers: Vec<Layer<'a>>,
}

struct Layer<'a> {
    /// Whether this is the first layer, the one [`trace`] captures.
    traced: bool,
    local: bool,
    wq: Mat<'a>,
    wk: Mat<'a>,
    wv: Mat<'a>,
    wr: Mat<'a>,
    wo: Mat<'a>,
    gate: Mat<'a>,
    sink: Mat<'a>,
    q_scale: Vec<f32>,
    rel_scale: Vec<f32>,
    /// `[d_rel, rel_extent]`: one bias-versus-distance profile per relative dimension.
    rel_proj: Vec<f32>,
    k_conv: KvConv<'a>,
    v_conv: KvConv<'a>,
    attn_conv: Vec<f32>,
    mlp_conv: Vec<f32>,
    /// One norm (sliding layers) or separate query, key and value norms (full layers).
    input_norms: Vec<Vec<f32>>,
    mlp_norm: Vec<f32>,
    /// Absent in layer 0, which reads its input directly.
    attn_query: Option<Vec<f32>>,
    mlp_query: Vec<f32>,
    router: Mat<'a>,
    router_bias: Vec<f32>,
    experts: Experts<'a>,
    /// Message gates for the query, key, value and MLP inputs; looped layers only.
    controllers: Option<[Controller<'a>; 4]>,
}

enum KvConv<'a> {
    /// `[channels, kernel]`, last tap on the current token.
    Static(Vec<f32>),
    /// `static_w` is `[kernel, channels]` with tap 0 on the current token; `proj` adds per-token, per-head taps.
    Dynamic { static_w: Vec<f32>, proj: Mat<'a> },
}

struct Controller<'a> {
    down: Mat<'a>,
    forget: Mat<'a>,
    gain: Mat<'a>,
}

/// What one call computed, for the tokens it was given. `hidden` is filled only when tracing.
pub struct Forward {
    /// The output of each layer visit, in execution order.
    pub hidden: Vec<Seq>,
    /// The final normed state, `tokens * hidden`, ready for [`Model::logits`].
    pub last: Seq,
    /// The hidden state entering each visit asked for with [`Model::forward_tapped`], in visit order.
    pub taps: Vec<Seq>,
    /// The logits after each token, when the pass computed them along the way: [`Model::logits`] of each state
    /// in `last`, to the path's own rounding.
    pub logits: Option<Vec<Vec<f32>>>,
}

impl<'a> Model<'a> {
    pub fn load(config: &Config, checkpoint: &'a Checkpoint, options: Options) -> Result<Self> {
        let gpu = open_gpu(checkpoint, options.backend)?;
        let cfg = config.text_config.clone();
        let h = cfg.hidden_size;
        let vec = |name: &str, len: usize| -> Result<Vec<f32>> {
            let dense = checkpoint.dense(name)?;
            ensure!(dense.data.len() == len, "{name}: expected {len} values, found shape {:?}", dense.shape);
            Ok(dense.data)
        };
        let mat = |name: &str, rows: usize, cols: usize| Mat::open(checkpoint, name, rows, cols);

        let layers = (0..cfg.num_hidden_layers)
            .into_par_iter()
            .map(|l| {
                let p = format!("model.llm.layers.{l}");
                let local = cfg.is_local(l);
                let (heads, kv_heads, head_dim, k) =
                    (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.sconv_kernel_size);
                let (q_dim, kv_dim) = (heads * head_dim, kv_heads * head_dim);
                let kv_conv = |which: &str| -> Result<KvConv<'a>> {
                    Ok(if local {
                        KvConv::Static(vec(&format!("{p}.attn.{which}_sconv.weight"), kv_dim * k)?)
                    } else {
                        KvConv::Dynamic {
                            static_w: vec(&format!("{p}.attn.{which}_sconv.static_w"), k * kv_dim)?,
                            proj: mat(&format!("{p}.attn.{which}_sconv.weight_proj"), kv_heads * k, h)?,
                        }
                    })
                };
                let input_norms = if local {
                    vec![vec(&format!("{p}.attn_norm.weight"), h)?]
                } else {
                    ["q", "k", "v"]
                        .iter()
                        .map(|w| vec(&format!("{p}.{w}_input_norm.weight"), h))
                        .collect::<Result<_>>()?
                };
                let controller = |which: &str| -> Result<Controller<'a>> {
                    let rank = cfg.loop_config.gate_rank;
                    let c = format!("{p}.controllers.{which}");
                    Ok(Controller {
                        down: mat(&format!("{c}.down_weight"), rank, 2 * h)?,
                        forget: mat(&format!("{c}.forget_weight"), h, rank)?,
                        gain: mat(&format!("{c}.gain_weight"), h, rank)?,
                    })
                };
                // attn_queries.{i} belongs to layer i + 1: one query in sliding layers, three in full layers.
                let attn_query = match l {
                    0 => None,
                    _ => Some(vec(&format!("model.llm.attn_res.attn_queries.{}", l - 1), if local { h } else { 3 * h })?),
                };
                Ok(Layer {
                    traced: l == 0,
                    local,
                    wq: mat(&format!("{p}.attn.wq_du.weight"), q_dim, h)?,
                    wk: mat(&format!("{p}.attn.wk_dv.weight"), kv_dim, h)?,
                    wv: mat(&format!("{p}.attn.wv_dv.weight"), kv_dim, h)?,
                    wr: mat(&format!("{p}.attn.wr_du.weight"), heads * cfg.d_rel, h)?,
                    wo: mat(&format!("{p}.attn.wo_ud.weight"), h, q_dim)?,
                    gate: mat(&format!("{p}.attn.gate_attn.weight"), q_dim, h)?,
                    sink: mat(&format!("{p}.attn.attn_sink_proj.weight"), heads, h)?,
                    q_scale: vec(&format!("{p}.attn.q_l2_clip.scale"), q_dim)?,
                    rel_scale: vec(&format!("{p}.attn.rel_l2_clip.scale"), heads * cfg.d_rel)?,
                    rel_proj: vec(&format!("{p}.attn.rel_logits_proj.proj"), cfg.d_rel * cfg.rel_extent)?,
                    k_conv: kv_conv("k")?,
                    v_conv: kv_conv("v")?,
                    attn_conv: vec(&format!("{p}.attn_sconv.weight"), h * k)?,
                    mlp_conv: vec(&format!("{p}.mlp_sconv.weight"), h * k)?,
                    input_norms,
                    mlp_norm: vec(&format!("{p}.mlp_norm.weight"), h)?,
                    attn_query,
                    mlp_query: vec(&format!("model.llm.attn_res.mlp_queries.{l}"), h)?,
                    router: mat(&format!("{p}.mlp.gate.weight"), cfg.n_routed_experts, h)?,
                    router_bias: vec(&format!("{p}.mlp.gate.bias"), cfg.n_routed_experts)?,
                    experts: checkpoint.experts(l)?,
                    controllers: match cfg.is_looped(l) {
                        true => Some([controller("attn")?, controller("k")?, controller("v")?, controller("mlp")?]),
                        false => None,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()
            .context("loading decoder layers")?;

        // The checkpoint stores [query, mlp] pairs and [key, value] pairs per looped layer, in two tensors.
        let core = cfg.loop_config.core_end - cfg.loop_config.core_start;
        let message_queries = if checkpoint.contains("model.llm.middle_cycle_cross_pass_queries") {
            let q_mlp = vec("model.llm.middle_cycle_cross_pass_queries", 2 * core * h)?;
            let k_v = vec("model.llm.middle_cycle_cross_pass_kv_queries", 2 * core * h)?;
            (0..core)
                .map(|i| {
                    let pair = |source: &[f32], j: usize| source[(2 * i + j) * h..(2 * i + j + 1) * h].to_vec();
                    [pair(&q_mlp, 0), pair(&k_v, 0), pair(&k_v, 1), pair(&q_mlp, 1)].concat()
                })
                .collect()
        } else {
            // A GGUF file already has them per layer, in this order.
            let start = cfg.loop_config.core_start;
            (start..start + core)
                .map(|l| vec(&format!("model.llm.layers.{l}.message_queries"), 4 * h))
                .collect::<Result<_>>()?
        };

        ensure!(cfg.rel_extent == cfg.sliding_window_size, "relative extent and sliding window are assumed equal");
        Ok(Self {
            embed_norm: vec("model.llm.embed_norm.weight", h)?,
            final_norm: vec("model.llm.norm.weight", h)?,
            output_query: vec("model.llm.attn_res.output_query", h)?,
            message_queries,
            layers,
            cfg,
            embed: mat("model.llm.embed.weight", config.text_config.vocab_size, h)?,
            unembed: mat("model.llm.unembed.weight", config.text_config.vocab_size, h)?,
            // A GGUF file's experts are not in the formats the activations are rounded to.
            activations: match checkpoint.is_gguf() {
                true => Activations::Float,
                false => options.activations,
            },
            gpu,
            #[cfg(target_os = "macos")]
            decoder: std::sync::OnceLock::new(),
        })
    }

    /// Gets the model ready to answer at full speed: runs one token through every layer and the output weights,
    /// which starts the thread pool and brings the weights that generation reads into memory. Without it the
    /// first request pays for both.
    pub fn warm_up(&self) -> Result<()> {
        let last = self.forward(&[0], &mut self.new_state(), false)?.last;
        self.logits(&last).map(drop)
    }

    /// Whether to imitate transformers' bfloat16 rounding.
    fn bf16(&self) -> bool {
        self.activations == Activations::FloatBf16
    }

    /// How many prompt tokens to give [`Model::forward`] at a time. Any split gives the same states on the exact
    /// paths; this one is the fastest that keeps memory bounded.
    pub fn prompt_chunk(&self) -> usize {
        #[cfg(target_os = "macos")]
        if self.gpu.as_ref().is_some_and(Gpu::fast) {
            return FAST_PROMPT_CHUNK;
        }
        PROMPT_CHUNK
    }

    /// Whether long prompts take the fast path, whose states match the exact paths' only to rounding.
    pub fn on_fast_path(&self) -> bool {
        #[cfg(target_os = "macos")]
        return self.gpu.as_ref().is_some_and(Gpu::fast);
        #[cfg(not(target_os = "macos"))]
        false
    }

    /// Whether large batches run on the GPU.
    pub fn uses_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// The output weights rounded to bytes, for a reader that can do with approximate logits. Nothing when they
    /// are quantised already.
    pub(crate) fn packed_unembed(&self) -> Option<Mat<'static>> {
        self.unembed.packed()
    }

    /// The output weights: a row per token, padding included.
    pub(crate) fn unembed(&self) -> &Mat<'a> {
        &self.unembed
    }

    pub fn config(&self) -> &TextConfig {
        &self.cfg
    }

    /// Marks where `state` stands, for [`State::rewind`].
    pub fn mark(&self, state: &State) -> Mark {
        let visits = self
            .cfg
            .visits()
            .iter()
            .zip(&state.visits)
            .map(|(visit, held)| VisitMark {
                first: held.first,
                convs: held.convs.clone(),
                window: self.cfg.is_local(visit.layer).then(|| (held.keys.clone(), held.values.clone())),
            })
            .collect();
        Mark { position: state.position, visits }
    }

    /// A copy of `state` holding only what the tokens to come can read: the next pass would drop the rest
    /// anyway. After a long prompt the sliding layers still hold the whole of its last chunk, four times their
    /// window, which is most of what a copy would otherwise take.
    pub fn trimmed(&self, state: &State) -> State {
        let mut copy = state.clone();
        self.trim(&mut copy);
        copy
    }

    /// Drops from `state` what the tokens to come cannot read; see [`Model::trimmed`].
    pub fn trim(&self, state: &mut State) {
        let position = state.position;
        for (visit, at) in state.visits.iter_mut().zip(self.cfg.visits()) {
            visit.drop_stale_from(&self.cfg, self.cfg.is_local(at.layer), position, 1);
        }
    }

    /// A fresh context with no tokens in it.
    pub fn new_state(&self) -> State {
        let cfg = &self.cfg;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let conv_channels = [kv_dim, kv_dim, cfg.hidden_size, cfg.hidden_size];
        let table = Table::new(cfg.num_key_value_heads, cfg.head_dim);
        let fresh = VisitState { keys: table.clone(), values: table, conv_channels, ..Default::default() };
        let visits = cfg.visits().iter().map(|_| fresh.clone()).collect();
        State { position: 0, visits }
    }

    /// Runs the decoder over `ids`, which continue the context in `state`. With `trace`, keeps the output of
    /// every layer visit.
    pub fn forward(&self, ids: &[u32], state: &mut State, trace: bool) -> Result<Forward> {
        self.run(ids, state, trace, &[])
    }

    /// As [`Model::forward`], also keeping the hidden state that enters each of the visits in `taps`.
    pub fn forward_tapped(&self, ids: &[u32], state: &mut State, taps: &[usize]) -> Result<Forward> {
        self.run(ids, state, false, taps)
    }

    /// The normed embedding of one token: what the first layer reads.
    pub(crate) fn embed(&self, id: u32) -> Result<Vec<f32>> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        ensure!((id as usize) < cfg.vocab_size, "token id {id} is out of range");
        let mut x = vec![0.0f32; h];
        self.embed.format.decode_row(self.embed.row(id as usize), &mut x);
        rms_norm(&mut x, Some(&self.embed_norm), cfg.rms_norm_eps);
        round_all(self.bf16(), &mut x);
        Ok(x)
    }

    /// Runs on a thread of the pool. From outside it, every one of the hundreds of parallel steps in a forward
    /// pass would be handed over and waited for separately, which costs more than the steps themselves when
    /// generating token by token.
    fn run(&self, ids: &[u32], state: &mut State, trace: bool, taps: &[usize]) -> Result<Forward> {
        #[cfg(target_os = "macos")]
        if !trace
            && (1..=decode::MAX_BATCH).contains(&ids.len())
            && let Some(forward) = self.decode(ids, state, taps)?
        {
            return Ok(forward);
        }
        in_pool(|| self.run_in_pool(ids, state, trace, taps))
    }

    /// Frees the memory the GPU paths for long batches work in, several gigabytes after a long prompt. Worth
    /// doing when a prompt is done: generating does not use it.
    pub fn release_scratch(&self) {
        #[cfg(target_os = "macos")]
        if let Some(gpu) = &self.gpu {
            gpu.release_scratch();
        }
    }

    /// Whether tokens are generated on the GPU, singly and in the short batches that check proposed tokens. A
    /// token then costs less than a draft model's proposal takes to make on the CPU, so a draft cannot gain
    /// anything; proposals that cost nothing to make, copied from the context, still can.
    pub fn decodes_on_gpu(&self) -> bool {
        #[cfg(target_os = "macos")]
        return self.gpu.as_ref().is_some_and(Gpu::fast) && self.layers[0].experts.gate_up.ggml().is_some();
        #[cfg(not(target_os = "macos"))]
        false
    }

    /// The GPU decoder, built at first use; nothing when the model is not set up for it or building it failed.
    #[cfg(target_os = "macos")]
    fn decoder(&self) -> Option<&std::sync::Mutex<decode::Decoder>> {
        let gpu = self.gpu.as_ref().filter(|_| self.decodes_on_gpu())?;
        let decoder = self.decoder.get_or_init(|| match decode::Decoder::new(self, gpu) {
            Ok(decoder) => Some(std::sync::Mutex::new(decoder)),
            Err(error) => {
                gpu_failed(&error);
                None
            }
        });
        decoder.as_ref()
    }

    /// A token or a short batch on the GPU, when the model is set up for it: a GGUF file on the fast backend.
    /// Nothing otherwise, or when the GPU fails, and the caller takes the CPU path with the state untouched.
    #[cfg(target_os = "macos")]
    fn decode(&self, ids: &[u32], state: &mut State, taps: &[usize]) -> Result<Option<Forward>> {
        let Some(gpu) = self.gpu.as_ref().filter(|_| self.decodes_on_gpu() && !float_reference()) else { return Ok(None) };
        let Some(decoder) = self.decoder() else { return Ok(None) };
        let mut embedded = Vec::with_capacity(ids.len() * self.cfg.hidden_size);
        for &id in ids {
            embedded.extend(self.embed(id)?);
        }
        let started = std::time::Instant::now();
        let result = decoder.lock().expect("decoder lock").tokens(self, gpu, state, &embedded, taps);
        timing::since(Stage::Token, started);
        match result {
            Ok(tokens) => Ok(Some(Forward { hidden: Vec::new(), last: tokens.last, taps: tokens.taps, logits: Some(tokens.logits) })),
            Err(error) => {
                gpu_failed(&error);
                Ok(None)
            }
        }
    }

    fn run_in_pool(&self, ids: &[u32], state: &mut State, trace: bool, taps: &[usize]) -> Result<Forward> {
        let cfg = &self.cfg;
        let (h, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
        let (start, end) = (cfg.loop_config.core_start, cfg.loop_config.core_end);

        let mut hidden: Seq = Vec::with_capacity(ids.len() * h);
        for &id in ids {
            hidden.extend(self.embed(id)?);
        }
        trace::record("embed", &hidden);

        let mut tapped = Vec::new();
        let mut visits = Vec::new();
        let mut blocks: Vec<Seq> = Vec::new();
        let mut run = |layers: std::ops::Range<usize>,
                       pass: usize,
                       hidden: &mut Seq,
                       blocks: &mut Vec<Seq>,
                       previous: Option<(&Seq, &Seq)>| {
            for l in layers {
                let messages = previous.map(|(block, last)| {
                    let read = attn_res(&self.message_queries[l - start], &[block], last, h);
                    <[Seq; 4]>::try_from(read).unwrap_or_else(|_| unreachable!("four message queries"))
                });
                // Visits run prelude, then the core once per pass, then the coda.
                let visit = match l {
                    l if l < start => l,
                    l if l < end => start + pass * (end - start) + (l - start),
                    l => l + (cfg.loop_config.num_passes - 1) * (end - start),
                };
                if taps.contains(&visit) {
                    tapped.push(hidden.clone());
                }
                let out = self.layer(l, hidden, blocks, messages.as_ref(), &mut state.visits[visit], state.position);
                // A block boundary files the layer's input as a finished block.
                if l % cfg.n_layers_per_attn_res_block == 0 {
                    blocks.push(residual_normed(std::mem::take(hidden), h));
                }
                *hidden = out;
                if trace {
                    visits.push(hidden.clone());
                }
            }
        };

        run(0..start, 0, &mut hidden, &mut blocks, None);
        let (prelude_hidden, prelude_blocks) = (hidden.clone(), blocks.clone());
        run(start..end, 0, &mut hidden, &mut blocks, None);
        for pass in 1..cfg.loop_config.num_passes {
            // Every pass restarts from the prelude and sees the previous pass only through messages, which are
            // read from that pass's blocks after its first (the first is just the prelude output again).
            let previous_blocks = &blocks[prelude_blocks.len() + 1..];
            ensure!(previous_blocks.len() == 1, "message reads assume a loop core of exactly two blocks");
            let (previous_block, previous_hidden) = (previous_blocks[0].clone(), hidden);
            hidden = prelude_hidden.clone();
            blocks = prelude_blocks.clone();
            run(start..end, pass, &mut hidden, &mut blocks, Some((&previous_block, &previous_hidden)));
        }
        run(end..cfg.num_hidden_layers, 0, &mut hidden, &mut blocks, None);
        drop(run);
        state.position += ids.len();

        let block_refs: Vec<&Seq> = blocks.iter().collect();
        let mut last = attn_res(&self.output_query, &block_refs, &hidden, h).remove(0);
        last.chunks_exact_mut(h).for_each(|x| rms_norm(x, Some(&self.final_norm), eps));
        round_all(self.bf16(), &mut last);
        Ok(Forward { hidden: visits, last, taps: tapped, logits: None })
    }

    /// Soft-capped logits over the usable vocabulary for one final state.
    pub fn logits(&self, state: &[f32]) -> Result<Vec<f32>> {
        let started = std::time::Instant::now();
        let logits = in_pool(|| self.project(state));
        timing::since(Stage::Logits, started);
        logits
    }

    /// [`Model::logits`] for several final states at once, `states` holding one after another. The output
    /// weights are read once for all of them; each result is identical to what `logits` gives on its own.
    pub fn logits_many(&self, states: &[f32]) -> Result<Vec<Vec<f32>>> {
        let (h, cap, vocab) = (self.cfg.hidden_size, self.cfg.final_logit_softcapping, self.cfg.unpadded_vocab_size);
        let xs: Vec<&[f32]> = states.chunks_exact(h).collect();
        if xs.len() == 1 {
            return Ok(vec![self.logits(states)?]);
        }
        let started = std::time::Instant::now();
        let bf16 = self.bf16();
        let unembed = &self.unembed;
        let rounded: Vec<Int8Activation> = if unembed.int8() { xs.iter().map(|x| quantize_int8(x)).collect() } else { Vec::new() };
        let rounded: Vec<&Int8Activation> = rounded.iter().collect();
        // Per token: one value for each state.
        let mut by_token = vec![0.0f32; vocab * xs.len()];
        by_token.par_chunks_mut(1024 * xs.len()).enumerate().for_each_init(
            || vec![0.0f32; h],
            |widened, (chunk, out)| {
                for (i, out) in out.chunks_exact_mut(xs.len()).enumerate() {
                    let token = chunk * 1024 + i;
                    if unembed.int8() {
                        for (xs, out) in rounded.chunks(INT8_MAX_BATCH).zip(out.chunks_mut(INT8_MAX_BATCH)) {
                            dot_row_int8_many(unembed.format, unembed.row(token), xs, out);
                        }
                    } else {
                        unembed.format.decode_row(unembed.row(token), widened);
                        dot_f32_row_many(widened, &xs, out);
                    }
                    for sum in out {
                        let rounded = if bf16 { round_bf16(*sum) } else { *sum };
                        *sum = (rounded / cap).tanh() * cap;
                    }
                }
            },
        );
        timing::since(Stage::Logits, started);
        Ok((0..xs.len()).map(|x| by_token.iter().skip(x).step_by(xs.len()).copied().collect()).collect())
    }

    fn project(&self, state: &[f32]) -> Result<Vec<f32>> {
        let cap = self.cfg.final_logit_softcapping;
        let unembed = &self.unembed;
        let rounded = unembed.int8().then(|| quantize_int8(state));
        let mut logits = vec![0.0f32; self.cfg.unpadded_vocab_size];
        logits.par_chunks_mut(1024).enumerate().for_each_init(Vec::new, |scratch, (chunk, logits)| {
            for (i, logit) in logits.iter_mut().enumerate() {
                let token = chunk * 1024 + i;
                let sum = match &rounded {
                    Some(rounded) => {
                        let mut sum = [0.0f32];
                        dot_row_int8_many(unembed.format, unembed.row(token), &[rounded], &mut sum);
                        sum[0]
                    }
                    None => unembed.format.dot_row(unembed.row(token), state, scratch),
                };
                let sum = if self.bf16() { round_bf16(sum) } else { sum };
                *logit = (sum / cap).tanh() * cap;
            }
        });
        Ok(logits)
    }

    /// One layer visit over new tokens starting at `position`, reading and extending that visit's `state`.
    fn layer(
        &self,
        l: usize,
        input: &Seq,
        blocks: &[Seq],
        messages: Option<&[Seq; 4]>,
        state: &mut VisitState,
        position: usize,
    ) -> Seq {
        let cfg = &self.cfg;
        let (h, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
        let layer = &self.layers[l];
        let mut block_refs: Vec<&Seq> = blocks.iter().collect();
        let starts_block = l % cfg.n_layers_per_attn_res_block == 0;
        let normed_input = starts_block.then(|| residual_normed(input.clone(), h));
        let gates = layer.controllers.as_ref().zip(messages);
        let bf16 = self.bf16();

        // Attention inputs: read from the residual blocks, merge the previous pass's messages, then norm.
        let (q_in, k_in, v_in) = time(Stage::Inputs, || {
            let reads = match &layer.attn_query {
                Some(query) => attn_res(query, &block_refs, input, h),
                None => vec![input.clone()],
            };
            let input = |i: usize| {
                let read = &reads[if layer.local { 0 } else { i }];
                let mut x = match gates {
                    Some((controllers, messages)) => controllers[i].merge(self.gpu.as_ref(), read, &messages[i], h, eps, bf16),
                    None => read.clone(),
                };
                let norm = &layer.input_norms[if layer.local { 0 } else { i }];
                x.chunks_exact_mut(h).for_each(|x| rms_norm(x, Some(norm), eps));
                round_all(bf16, &mut x);
                x
            };
            // The three merges are independent and, for a lone token, each runs on one thread.
            match gates {
                Some(_) => {
                    let (q, (k, v)) = rayon::join(|| input(0), || rayon::join(|| input(1), || input(2)));
                    (q, k, v)
                }
                None => (input(0), input(1), input(2)),
            }
        });

        let tracing = layer.traced && trace::enabled();
        if tracing {
            trace::record("attn_in", &q_in);
        }
        let attn = self.attention(layer, &q_in, &k_in, &v_in, state, position);
        if tracing {
            trace::record("attn", &attn);
        }
        let attn = time(Stage::AttentionOutput, || {
            let mut attn = static_conv(&attn, &layer.attn_conv, h, cfg.sconv_kernel_size, &mut state.convs[2]);
            round_all(bf16, &mut attn);
            attn
        });
        if tracing {
            trace::record("attn_conv", &attn);
        }

        // The first layer of a block starts a fresh residual; the block's input becomes a source instead.
        let residual = if let Some(normed_input) = &normed_input {
            block_refs.push(normed_input);
            attn
        } else {
            add(input, &attn)
        };

        let x = time(Stage::Inputs, || {
            let mut x = attn_res(&layer.mlp_query, &block_refs, &residual, h).remove(0);
            if let Some((controllers, messages)) = gates {
                x = controllers[3].merge(self.gpu.as_ref(), &x, &messages[3], h, eps, bf16);
            }
            x.chunks_exact_mut(h).for_each(|x| rms_norm(x, Some(&layer.mlp_norm), eps));
            round_all(bf16, &mut x);
            x
        });
        let mlp = time(Stage::Experts, || self.experts(layer, &x));
        if tracing {
            trace::record("mlp_in", &x);
            trace::record("mlp", &mlp);
        }
        let out = time(Stage::ExpertOutput, || {
            let mut mlp = static_conv(&mlp, &layer.mlp_conv, h, cfg.sconv_kernel_size, &mut state.convs[3]);
            round_all(bf16, &mut mlp);
            if tracing {
                trace::record("mlp_conv", &mlp);
            }
            let mut out = add(&residual, &mlp);
            round_all(bf16, &mut out);
            out
        });
        if tracing {
            trace::record("out", &out);
        }
        out
    }

}

/// Where the heavy matrix products run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// The GPU when there is one, otherwise the CPU. For a checkpoint that is the exact kernels on large
    /// batches, with the CPU's results bit for bit; for a GGUF file, which has no exact kernels, the fast path.
    #[default]
    Auto,
    Cpu,
    /// As `Auto`, but failing to set up the GPU is an error.
    Metal,
    /// The GPU with a faster path for long prompts, whose results match the others' only to rounding. With
    /// quantised activations that small difference grows into a different, equally valid result.
    MetalFast,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    pub activations: Activations,
    pub backend: Backend,
}

#[cfg(target_os = "macos")]
fn open_gpu(checkpoint: &Checkpoint, backend: Backend) -> Result<Option<Gpu>> {
    // The exact GPU kernels read the checkpoint's own formats, so a GGUF file has only the fast path.
    if checkpoint.is_gguf() && !matches!(backend, Backend::MetalFast | Backend::Auto) {
        ensure!(backend != Backend::Metal, "GGUF weights have no exact Metal kernels; use `metal-fast` or the CPU");
        return Ok(None);
    }
    match backend {
        Backend::Auto if checkpoint.is_gguf() => Ok(Gpu::new(&checkpoint.maps(), true).ok()),
        Backend::Cpu => Ok(None),
        Backend::Auto => Ok(Gpu::new(&checkpoint.maps(), false).ok()),
        Backend::Metal => Gpu::new(&checkpoint.maps(), false).map(Some).context("setting up Metal"),
        Backend::MetalFast => Gpu::new(&checkpoint.maps(), true).map(Some).context("setting up Metal"),
    }
}

#[cfg(not(target_os = "macos"))]
fn open_gpu(_checkpoint: &Checkpoint, backend: Backend) -> Result<Option<Gpu>> {
    ensure!(
        !matches!(backend, Backend::Metal | Backend::MetalFast),
        "the Metal backend is only available on macOS"
    );
    Ok(None)
}

/// Reports a GPU failure once; the caller carries on with the CPU, which gives the same results.
#[cfg(target_os = "macos")]
fn gpu_failed(error: &anyhow::Error) {
    static REPORTED: std::sync::Once = std::sync::Once::new();
    REPORTED.call_once(|| tracing::warn!("Metal backend failed, continuing on the CPU: {error:#}"));
}

/// Which arithmetic the expert layers use for their inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activations {
    /// f32 throughout: the model as transformers defines it, without its bfloat16 rounding.
    Float,
    /// As `Float`, but rounding to bfloat16 wherever transformers does when it runs the model in bfloat16. Only
    /// useful for checking against transformers' outputs.
    FloatBf16,
    /// Rounded to the expert weights' own format, as the model was trained and as SGLang and vLLM serve it.
    #[default]
    Quantised,
}

/// Runs `work` on a thread of the pool, or here when this already is one.
fn in_pool<T: Send>(work: impl FnOnce() -> T + Send) -> T {
    match rayon::current_thread_index() {
        Some(_) => work(),
        None => rayon::scope(|_| work()),
    }
}

fn round_bf16(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}

/// Rounds every value to bfloat16 when `on`; otherwise leaves them alone.
fn round_all(on: bool, values: &mut [f32]) {
    if on {
        values.iter_mut().for_each(|v| *v = round_bf16(*v));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_conv_is_causal_and_ends_on_the_current_token() {
        // One channel, kernel 2, taps [previous, current] = [10, 1].
        let out = static_conv(&[1.0, 2.0, 3.0], &[10.0, 1.0], 1, 2, &mut Vec::new());
        let scale = 1.0 / 3f32.sqrt();
        let expected = [(1.0 + 1.0) * scale, (10.0 + 2.0 + 2.0) * scale, (20.0 + 3.0 + 3.0) * scale];
        assert_eq!(out, expected);

        // Fed one token at a time, the history carries the earlier inputs across calls.
        let mut history = Vec::new();
        let stepped: Vec<f32> =
            [1.0, 2.0, 3.0].iter().flat_map(|x| static_conv(&[*x], &[10.0, 1.0], 1, 2, &mut history)).collect();
        assert_eq!(stepped, expected);
        // All three inputs are still remembered: the one the kernel needs, and more for taking tokens back.
        assert_eq!(history, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn attn_res_with_zero_query_averages_normed_states() {
        let (a, b) = (residual_normed(vec![3.0, 0.0], 2), vec![0.0, 5.0]);
        let out = attn_res(&[0.0, 0.0], &[&a], &b, 2).remove(0);
        // Each state norms to RMS 1, i.e. [sqrt(2), 0] and [0, sqrt(2)]; a zero query weights them equally.
        let half = 2f32.sqrt() / 2.0;
        assert!((out[0] - half).abs() < 1e-5 && (out[1] - half).abs() < 1e-5, "{out:?}");
    }

    #[test]
    fn l2_clip_only_shrinks() {
        let mut long = [3.0, 4.0];
        l2_clip(&mut long, 1.0);
        assert!((long[0] - 0.6).abs() < 1e-5 && (long[1] - 0.8).abs() < 1e-5);
        let mut short = [0.3, 0.4];
        l2_clip(&mut short, 1.0);
        assert_eq!(short, [0.3, 0.4]);
    }

    /// Token-by-token speed with a long context, without paying to process one: the state is filled with made-up
    /// keys and values, which cost attention the same as real ones.
    /// `cargo test --release --lib decode_at_long_context -- --ignored --nocapture`; `CONTEXT` sets the length
    /// and `BATCH` how many tokens go through at a time.
    #[test]
    #[ignore]
    fn decode_at_long_context() {
        use crate::weights::checkpoint::resolve_model;
        let context: usize = std::env::var("CONTEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(12_000);
        let named = std::env::var("RUSTLING_GGUF").unwrap_or_else(|_| "lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M".to_owned());
        let path = resolve_model(Some(std::path::Path::new(&named))).unwrap();
        let weights = Checkpoint::open(&path).unwrap();
        let config = Config::load(&path).unwrap();
        for backend in [Backend::Cpu, Backend::MetalFast] {
            let model = Model::load(&config, &weights, Options { activations: Activations::Float, backend }).unwrap();
            let cfg = model.config().clone();
            let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
            let mut state = model.new_state();
            state.position = context;
            for (visit, held) in cfg.visits().iter().zip(&mut state.visits) {
                let kept = if cfg.is_local(visit.layer) { context.min(cfg.sliding_window_size - 1) } else { context };
                let values = |seed: usize| -> Vec<Vec<f32>> {
                    let head = |kv: usize| (0..kept * kv_dim / cfg.num_key_value_heads)
                        .map(|i| (((i * 2_654_435_761 + seed + kv) >> 7) % 2001) as f32 / 1000.0 - 1.0)
                        .collect();
                    (0..cfg.num_key_value_heads).map(head).collect()
                };
                let table = |seed: usize| Table::from_heads(values(seed), cfg.head_dim);
                (held.keys, held.values, held.first) = (table(visit.index), table(visit.index + 99), context - kept);
            }
            model.forward(&[17], &mut state, false).unwrap();
            crate::timing::take();
            let started = std::time::Instant::now();
            // `BATCH` tokens per pass, as when a draft's proposals are checked.
            let batch: u32 = std::env::var("BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            let tokens = 24.max(batch);
            for pass in 0..tokens / batch {
                let ids: Vec<u32> = (0..batch).map(|i| 100 + pass * batch + i).collect();
                model.forward(&ids, &mut state, false).unwrap();
            }
            let each = started.elapsed().as_secs_f64() * 1000.0 / tokens as f64;
            let stages: Vec<String> = crate::timing::take()
                .iter()
                .filter(|(_, seconds)| *seconds > 0.0)
                .map(|(stage, seconds)| format!("{stage:?} {:.1}", seconds * 1000.0 / tokens as f64))
                .collect();
            eprintln!("{backend:?} at {context} tokens: {each:.1} ms per token without logits ({})", stages.join(", "));
        }
    }
}
