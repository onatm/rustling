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
use crate::ggml::{Format, INT8_MAX_BATCH, Int8Activation, dot_row_int8_many, quantize_int8};
use rayon::prelude::*;

use crate::accel;
use crate::checkpoint::{Checkpoint, ExpertMatrix, Experts};
use crate::config::{Config, TextConfig};
#[cfg(target_os = "macos")]
use crate::gpu::{AttendHead, AttendShape, DenseJob, ExpertJob, Gpu, Stack, WholeExperts};
use crate::timing::{self, Stage, time};
use crate::trace;
use crate::quant::{
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

/// How many batches of plain matrix products, and of the controllers' merges, had to be recomputed on the CPU
/// because the GPU's fast product returned values that are not numbers.
pub static GPU_PRODUCTS_REDONE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

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
mod decode;
mod table;

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

/// Everything the tokens seen so far left behind for the tokens to come.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct State {
    /// Number of tokens already processed, which is the position of the next one.
    position: usize,
    visits: Vec<VisitState>,
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
    position: usize,
    visits: Vec<VisitMark>,
}

#[derive(Clone)]
struct VisitMark {
    first: usize,
    convs: [Vec<f32>; 4],
    /// The keys and values, for a visit that drops old ones.
    window: Option<(Table, Table)>,
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
struct VisitState {
    /// Keys and values of earlier tokens, starting at position `first`: one table per key head, `head_dim` per
    /// token, so that attention reads each head's as one run of memory. Sliding-window visits drop what has
    /// left the window.
    keys: Table,
    values: Table,
    first: usize,
    /// The recent inputs of the key, value, post-attention and post-MLP convolutions: the `kernel - 1` they
    /// need, and up to [`MAX_ROLLBACK`] more so recent tokens can be taken back.
    convs: [Vec<f32>; 4],
    /// The width of each convolution's input.
    conv_channels: [usize; 4],
}

impl VisitState {
    /// Records the inputs the convolutions took for one token computed elsewhere.
    #[cfg(target_os = "macos")]
    fn push_conv_inputs(&mut self, cfg: &TextConfig, conv_inputs: [&[f32]; 4]) {
        for ((history, input), channels) in self.convs.iter_mut().zip(conv_inputs).zip(self.conv_channels) {
            history.extend_from_slice(input);
            let keep = (cfg.sconv_kernel_size - 1 + MAX_ROLLBACK) * channels;
            history.drain(..history.len().saturating_sub(keep));
        }
    }

    /// Drops the keys and values that have left a sliding layer's window, once a batch of them has: dropping
    /// from the front moves everything kept, megabytes per visit, so until then they sit unread at the front.
    fn drop_stale(&mut self, cfg: &TextConfig, local: bool, position: usize) {
        let window_start = if local { (position + 1).saturating_sub(cfg.sliding_window_size) } else { 0 };
        let stale = window_start.saturating_sub(self.first).min(self.keys.rows());
        if stale >= STALE_BATCH {
            self.keys.drop_front(stale);
            self.values.drop_front(stale);
            self.first += stale;
        }
    }
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

    fn attention(
        &self,
        layer: &Layer,
        q_in: &Seq,
        k_in: &Seq,
        v_in: &Seq,
        state: &mut VisitState,
        position: usize,
    ) -> Seq {
        let cfg = &self.cfg;
        let (heads, kv_heads, dim, d_rel) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.d_rel);
        let (q_dim, kv_dim, group) = (heads * dim, kv_heads * dim, heads / kv_heads);
        let tokens = q_in.len() / cfg.hidden_size;
        // Full-attention layers scale queries and relative states up logarithmically at long range.
        let tau = |t: usize| match layer.local {
            true => 1.0,
            false => 1.0 + cfg.log_scaling_alpha * ((t + 1) as f32 / cfg.log_scaling_n_floor).max(1.0).ln(),
        };

        // All input projections in one parallel region.
        let mut projected = time(Stage::Projections, || {
            apply_many(self.gpu.as_ref(), &[
                (&layer.wq, q_in),
                (&layer.wk, k_in),
                (&layer.wv, v_in),
                (&layer.wr, q_in),
                (&layer.sink, q_in),
                (&layer.gate, q_in),
            ])
        });
        let attention_started = std::time::Instant::now();
        // In the bfloat16 imitation every projection and every step after it lands on bfloat16 values, as each is
        // a separate bfloat16 tensor in transformers.
        let bf16 = self.bf16();
        projected.iter_mut().for_each(|projection| round_all(bf16, projection));
        let (gate, sinks, mut rel) = (projected.remove(5), projected.remove(4), projected.remove(3));
        let (v, k, mut q) = (projected.remove(2), projected.remove(1), projected.remove(0));
        q.chunks_exact_mut(q_dim).enumerate().for_each(|(i, q)| {
            for (head, scale) in q.chunks_exact_mut(dim).zip(layer.q_scale.chunks_exact(dim)) {
                head.iter_mut().zip(scale).for_each(|(x, s)| *x *= s);
                l2_clip(head, cfg.qk_l2_clip_threshold);
                round_all(bf16, head);
                head.iter_mut().for_each(|x| *x *= tau(position + i));
                round_all(bf16, head);
            }
        });
        let mut k = layer.k_conv.apply(&k, k_in, kv_dim, cfg, &mut state.convs[0], bf16);
        round_all(bf16, &mut k);
        k.chunks_exact_mut(dim).for_each(|head| l2_clip(head, cfg.qk_l2_clip_threshold));
        round_all(bf16, &mut k);
        let mut v = layer.v_conv.apply(&v, v_in, kv_dim, cfg, &mut state.convs[1], bf16);
        round_all(bf16, &mut v);
        v.chunks_exact_mut(dim).for_each(|head| rms_norm(head, None, cfg.rms_norm_eps));
        round_all(bf16, &mut v);
        let tracing = layer.traced && trace::enabled();
        if tracing {
            trace::record("q", &q);
            trace::record("k", &k);
            trace::record("v", &v);
        }

        // Sliding layers see the last `sliding_window_size` tokens; full layers see everything.
        let window_start = |t: usize| if layer.local { (t + 1).saturating_sub(cfg.sliding_window_size) } else { 0 };
        state.drop_stale(cfg, layer.local, position);
        state.keys.push(&k);
        state.values.push(&v);
        // Each key head's rows, oldest first.
        let k: Vec<&[f32]> = (0..kv_heads).map(|kv| state.keys.head(kv)).collect();
        let v: Vec<&[f32]> = (0..kv_heads).map(|kv| state.values.head(kv)).collect();
        let (k, v, cached_from) = (&k, &v, state.first);

        rel.chunks_exact_mut(heads * d_rel).enumerate().for_each(|(i, rel)| {
            for (head, scale) in rel.chunks_exact_mut(d_rel).zip(layer.rel_scale.chunks_exact(d_rel)) {
                head.iter_mut().zip(scale).for_each(|(x, s)| *x *= s);
                l2_clip(head, cfg.rel_l2_clip_threshold);
                round_all(bf16, head);
                head.iter_mut().for_each(|x| *x *= tau(position + i));
                round_all(bf16, head);
            }
        });

        let scaling = (dim as f32).powf(-0.5);
        let extent = cfg.rel_extent;
        let mut out = vec![0.0f32; tokens * q_dim];
        // One unit of work per (token, head): many tokens or a long context both spread across threads.
        let attend = |(unit, out): (usize, &mut [f32])| {
            // `i` indexes the new tokens, `t` and `s` are absolute positions.
            let (i, head) = (unit / heads, unit % heads);
            let t = position + i;
            let first = window_start(t);
            let kv = head / group;
            let query = &q[i * q_dim + head * dim..][..dim];
            let rel = &rel[(i * heads + head) * d_rel..][..d_rel];
            // bias[d] for backward distance d; zero beyond the relative extent.
            let mut bias = vec![0.0f32; (t + 1 - first).min(extent)];
            for (r, rel) in rel.iter().enumerate() {
                let profile = &layer.rel_proj[r * extent..][..bias.len()];
                bias.iter_mut().zip(profile).for_each(|(b, p)| *b += rel * p);
            }
            round_all(bf16, &mut bias);
            if tracing {
                // As the reference lays it out: [query token][head][key token], zero for keys after the query.
                for (distance, value) in bias.iter().enumerate() {
                    let offset = (i * heads + head) * tokens + (i - distance);
                    trace::record_at("bias", tokens * heads * tokens, offset, &[*value]);
                }
            }
            // The sink is one extra logit with a zero value: it only absorbs probability mass.
            let sink = sinks[i * heads + head];
            let keys = t + 1 - first;
            let (k, v) = (&k[kv], &v[kv]);
            let base = (first - cached_from) * dim;
            if bf16 {
                // The reference's steps one by one, each landing on bfloat16.
                let mut weights: Vec<f32> = (first..=t)
                    .map(|s| {
                        let key = &k[(s - cached_from) * dim..][..dim];
                        // The product, the scaling and the bias are three bfloat16 results in transformers.
                        let score = round_bf16(round_bf16(dot(query, key)) * scaling);
                        round_bf16(score + bias.get(t - s).copied().unwrap_or(0.0))
                    })
                    .collect();
                let max = weights.iter().copied().fold(sink, f32::max);
                weights.iter_mut().for_each(|w| *w = (*w - max).exp());
                let total = weights.iter().sum::<f32>() + (sink - max).exp();
                for (weight, s) in weights.iter().zip(first..=t) {
                    let value = &v[(s - cached_from) * dim..][..dim];
                    let p = round_bf16(weight / total);
                    out.iter_mut().zip(value).for_each(|(o, x)| *o += p * x);
                }
            } else {
                let mut weights: Vec<f32> = (0..keys)
                    .map(|j| {
                        // Key `j` is `keys - 1 - j` positions back.
                        let key = &k[base + j * dim..][..dim];
                        dot(query, key) * scaling + bias.get(keys - 1 - j).copied().unwrap_or(0.0)
                    })
                    .collect();
                let max = weights.iter().copied().fold(sink, f32::max);
                weights.iter_mut().for_each(|w| *w = (*w - max).exp());
                let total = weights.iter().sum::<f32>() + (sink - max).exp();
                for (j, weight) in weights.iter().enumerate() {
                    let p = weight / total;
                    out.iter_mut().zip(&v[base + j * dim..][..dim]).for_each(|(o, x)| *o += p * x);
                }
            }
            round_all(bf16, out);
            let gate = &gate[unit * dim..][..dim];
            out.iter_mut().zip(gate).for_each(|(o, g)| *o *= sigmoid(*g));
            round_all(bf16, out);
        };
        // The same sums in the same order, arranged around the keys instead: a block of tokens and the query
        // heads that share a key head take each stretch of keys and values together, so those are read once per
        // block and not once per token and head. With a long context that reading is most of the work.
        let attend_block = |&(block, kv): &(usize, usize)| -> Vec<f32> {
            let (i0, i1) = (block * ATTENTION_BLOCK, ((block + 1) * ATTENTION_BLOCK).min(tokens));
            let firsts: Vec<usize> = (i0..i1).map(|i| window_start(position + i)).collect();
            // One row of weights per (token, head), starting as the bias of each key.
            let mut weights: Vec<Vec<f32>> = Vec::with_capacity((i1 - i0) * group);
            for i in i0..i1 {
                let (t, first) = (position + i, firsts[i - i0]);
                let keys = t + 1 - first;
                for head in kv * group..(kv + 1) * group {
                    let rel = &rel[(i * heads + head) * d_rel..][..d_rel];
                    let mut bias = vec![0.0f32; keys.min(extent)];
                    for (r, rel) in rel.iter().enumerate() {
                        let profile = &layer.rel_proj[r * extent..][..bias.len()];
                        bias.iter_mut().zip(profile).for_each(|(b, p)| *b += rel * p);
                    }
                    let mut row = vec![0.0f32; keys];
                    bias.iter().enumerate().for_each(|(distance, value)| row[keys - 1 - distance] = *value);
                    weights.push(row);
                }
            }
            // Calls `visit(row, token, head, first key, keys)` for every stretch of keys, in key order.
            let stretches = |visit: &mut dyn FnMut(usize, usize, usize, usize, std::ops::Range<usize>)| {
                let (lowest, end) = (firsts[0], position + i1);
                for stretch in (lowest..end).step_by(ATTENTION_KEYS) {
                    for i in i0..i1 {
                        let first = firsts[i - i0];
                        let keys = stretch.max(first)..(stretch + ATTENTION_KEYS).min(position + i + 1);
                        if keys.is_empty() {
                            continue;
                        }
                        for g in 0..group {
                            visit((i - i0) * group + g, i, kv * group + g, first, keys.clone());
                        }
                    }
                }
            };
            let (k, v) = (&k[kv], &v[kv]);
            let at = |s: usize| (s - cached_from) * dim;
            stretches(&mut |row, i, head, first, keys| {
                let query = &q[i * q_dim + head * dim..][..dim];
                let weights = &mut weights[row];
                for s in keys {
                    weights[s - first] = dot(query, &k[at(s)..][..dim]) * scaling + weights[s - first];
                }
            });
            for (row, weights) in weights.iter_mut().enumerate() {
                let sink = sinks[(i0 + row / group) * heads + kv * group + row % group];
                let max = weights.iter().copied().fold(sink, f32::max);
                weights.iter_mut().for_each(|w| *w = (*w - max).exp());
                let total = weights.iter().sum::<f32>() + (sink - max).exp();
                weights.iter_mut().for_each(|w| *w /= total);
            }
            let mut out = vec![0.0f32; (i1 - i0) * group * dim];
            stretches(&mut |row, _, _, first, keys| {
                let out = &mut out[row * dim..][..dim];
                for s in keys {
                    let p = weights[row][s - first];
                    out.iter_mut().zip(&v[at(s)..][..dim]).for_each(|(o, x)| *o += p * x);
                }
            });
            for (row, out) in out.chunks_exact_mut(dim).enumerate() {
                let unit = (i0 + row / group) * heads + kv * group + row % group;
                out.iter_mut().zip(&gate[unit * dim..][..dim]).for_each(|(o, g)| *o *= sigmoid(*g));
            }
            out
        };
        // A long batch on the fast path: scores and outputs are matrix products of the block's queries and
        // weights with the keys and values, done by a tuned routine. Several times faster with a long context,
        // and equal to the loops above only to rounding.
        #[cfg(target_os = "macos")]
        let fast = self.gpu.as_ref().is_some_and(Gpu::fast) && tokens >= FAST_ATTENTION_MIN_TOKENS && !bf16 && !tracing;
        #[cfg(not(target_os = "macos"))]
        let fast = false;
        #[cfg(target_os = "macos")]
        if fast {
            use accel::Rows;
            // A stretch of tokens at a time, as many as keep the weights within bounds.
            let longest = position + tokens - window_start(position);
            let stretch = (FAST_ATTENTION_WEIGHTS / (group * longest)).clamp(1, tokens);
            // A sliding layer's rows see only a window's worth of the keys their stretch covers, and the rest of
            // each product is wasted, so its stretches are kept short.
            let stretch = if layer.local { stretch.min(FAST_LOCAL_STRETCH) } else { stretch };
            for i0 in (0..tokens).step_by(stretch) {
                let count = stretch.min(tokens - i0);
                // Every key any of these tokens can see; a row uses its own part of them.
                let lowest = window_start(position + i0);
                let keys = position + i0 + count - lowest;
                let rows = count * group;
                // One key head's share: its query heads' outputs for these tokens, before the gate.
                let attend = |kv: usize| -> Vec<f32> {
                    let from = (lowest - cached_from) * dim;
                    let mut queries = vec![0.0f32; rows * dim];
                    for (i, queries) in queries.chunks_exact_mut(group * dim).enumerate() {
                        queries.copy_from_slice(&q[(i0 + i) * q_dim + kv * group * dim..][..group * dim]);
                    }
                    let mut weights = vec![0.0f32; rows * keys];
                    let mine = Rows { values: &queries, rows, cols: dim, stride: dim };
                    let context = Rows { values: &k[kv][from..], rows: keys, cols: dim, stride: dim };
                    accel::multiply(mine, context, true, scaling, &mut weights);
                    weights.par_chunks_exact_mut(keys).enumerate().for_each(|(row, weights)| {
                        let (i, head) = (i0 + row / group, kv * group + row % group);
                        let t = position + i;
                        let visible = window_start(t) - lowest..t + 1 - lowest;
                        weights[..visible.start].fill(0.0);
                        weights[visible.end..].fill(0.0);
                        let weights = &mut weights[visible];
                        let rel = &rel[(i * heads + head) * d_rel..][..d_rel];
                        let mut bias = vec![0.0f32; weights.len().min(extent)];
                        for (r, rel) in rel.iter().enumerate() {
                            let profile = &layer.rel_proj[r * extent..][..bias.len()];
                            bias.iter_mut().zip(profile).for_each(|(b, p)| *b += rel * p);
                        }
                        weights.iter_mut().rev().zip(&bias).for_each(|(w, b)| *w += b);
                        let sink = sinks[i * heads + head];
                        let max = weights.iter().copied().fold(sink, f32::max);
                        weights.iter_mut().for_each(|w| *w -= max);
                        accel::exp_in_place(weights);
                        let total = weights.iter().sum::<f32>() + (sink - max).exp();
                        weights.iter_mut().for_each(|w| *w /= total);
                    });
                    let mut attended = vec![0.0f32; rows * dim];
                    let mine = Rows { values: &weights, rows, cols: keys, stride: keys };
                    let context = Rows { values: &v[kv][from..], rows: keys, cols: dim, stride: dim };
                    accel::multiply(mine, context, false, 1.0, &mut attended);
                    attended
                };
                // A long stretch goes to the GPU whole: the same products and softmax, an order of magnitude
                // faster once every row meets thousands of keys. Anything it gets wrong, the CPU redoes.
                let on_gpu = || -> Option<Vec<Vec<f32>>> {
                    let gpu = self.gpu.as_ref().filter(|_| rows * keys >= GPU_ATTENTION_MIN_WEIGHTS)?;
                    let from = (lowest - cached_from) * dim;
                    // Each key head's rows: for every token, the query heads that share it.
                    let gathered: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..kv_heads)
                        .into_par_iter()
                        .map(|kv| {
                            let (mut queries, mut rels, mut row_sinks) = (Vec::new(), Vec::new(), Vec::new());
                            for i in i0..i0 + count {
                                queries.extend_from_slice(&q[i * q_dim + kv * group * dim..][..group * dim]);
                                rels.extend_from_slice(&rel[(i * heads + kv * group) * d_rel..][..group * d_rel]);
                                row_sinks.extend_from_slice(&sinks[i * heads + kv * group..][..group]);
                            }
                            (queries, rels, row_sinks)
                        })
                        .collect();
                    let jobs: Vec<AttendHead> = gathered
                        .iter()
                        .enumerate()
                        .map(|(kv, (queries, rels, row_sinks))| AttendHead {
                            queries,
                            keys: &k[kv][from..],
                            values: &v[kv][from..],
                            rel: rels,
                            sinks: row_sinks,
                        })
                        .collect();
                    let shape = AttendShape {
                        rows,
                        keys,
                        dim,
                        d_rel,
                        extent,
                        profiles: &layer.rel_proj,
                        share: group,
                        first: position + i0,
                        lowest,
                        local: layer.local,
                        window: cfg.sliding_window_size,
                        scaling,
                    };
                    match gpu.attend_fast(&jobs, &shape) {
                        Ok(outputs) if outputs.par_iter().all(|head| head.iter().all(|value| value.is_finite())) => Some(outputs),
                        Ok(_) => {
                            GPU_ATTENTION_REDONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            None
                        }
                        Err(error) => {
                            gpu_failed(&error);
                            None
                        }
                    }
                };
                // On the CPU, a few tokens make products too small for the routine to spread over the cores
                // itself, so the key heads go a core each; a long batch fills the cores from inside one product.
                let attended: Vec<Vec<f32>> = match on_gpu() {
                    Some(attended) => attended,
                    None if count < FAST_MIN_TOKENS => (0..kv_heads).into_par_iter().map(attend).collect(),
                    None => (0..kv_heads).map(attend).collect(),
                };
                for (kv, attended) in attended.iter().enumerate() {
                    for (row, attended) in attended.chunks_exact(dim).enumerate() {
                        let unit = (i0 + row / group) * heads + kv * group + row % group;
                        let gate = &gate[unit * dim..][..dim];
                        let out = &mut out[unit * dim..][..dim];
                        out.iter_mut().zip(attended).zip(gate).for_each(|((o, a), g)| *o = a * sigmoid(*g));
                    }
                }
            }
        }
        let keys = position + tokens - window_start(position + tokens - 1);
        let serial = tokens * keys * q_dim < ATTENTION_MIN_WORK;
        if fast {
        } else if bf16 || tracing {
            // Step by step as the reference does it, a token and head at a time.
            match serial {
                true => out.chunks_mut(dim).enumerate().for_each(attend),
                false => out.par_chunks_mut(dim).enumerate().for_each(attend),
            }
        } else {
            let units: Vec<(usize, usize)> =
                (0..tokens.div_ceil(ATTENTION_BLOCK)).flat_map(|block| (0..kv_heads).map(move |kv| (block, kv))).collect();
            let results: Vec<Vec<f32>> = match serial {
                true => units.iter().map(attend_block).collect(),
                false => units.par_iter().map(attend_block).collect(),
            };
            for (&(block, kv), result) in units.iter().zip(&results) {
                for (row, result) in result.chunks_exact(dim).enumerate() {
                    let unit = (block * ATTENTION_BLOCK + row / group) * heads + kv * group + row % group;
                    out[unit * dim..][..dim].copy_from_slice(result);
                }
            }
        }
        timing::since(Stage::Attention, attention_started);
        time(Stage::AttentionOutput, || {
            let mut projected = apply_many(self.gpu.as_ref(), &[(&layer.wo, &out)]).remove(0);
            round_all(bf16, &mut projected);
            projected
        })
    }

    /// Each token's experts and the weights of their outputs, the highest-ranked first: `num_experts_per_tok`
    /// entries per token.
    fn routes(&self, layer: &Layer, x: &Seq) -> Vec<(u32, f32)> {
        let cfg = &self.cfg;
        let top_k = cfg.num_experts_per_tok;
        let mut routes = Vec::with_capacity(x.len() / cfg.hidden_size * top_k);
        let scores = layer.router.apply(x);
        if layer.traced {
            trace::record("router_logits", &scores);
        }
        for scores in scores.chunks_exact(cfg.n_routed_experts) {
            let scores: Vec<f32> = scores.iter().map(|s| sigmoid(*s)).collect();
            let mut order: Vec<usize> = (0..scores.len()).collect();
            order.sort_by(|&a, &b| (scores[b] + layer.router_bias[b]).total_cmp(&(scores[a] + layer.router_bias[a])));
            let chosen = &order[..top_k];
            let total: f32 = chosen.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20;
            routes.extend(chosen.iter().map(|&e| (e as u32, scores[e] / total * cfg.route_scale)));
        }
        routes
    }

    /// Routes each token to its top experts and sums their weighted outputs.
    fn experts(&self, layer: &Layer, x: &Seq) -> Seq {
        let cfg = &self.cfg;
        let (h, top_k) = (cfg.hidden_size, cfg.num_experts_per_tok);

        let routes = self.routes(layer, x);
        // tokens_of[expert] lists the (token, weight) pairs routed to it.
        let mut tokens_of: Vec<Vec<(usize, f32)>> = vec![Vec::new(); cfg.n_routed_experts];
        for (at, &(expert, weight)) in routes.iter().enumerate() {
            tokens_of[expert as usize].push((at / top_k, weight));
        }

        // A long batch from a GGUF file on the fast path: the whole step on the GPU.
        #[cfg(target_os = "macos")]
        if let Some(out) = self.experts_on_gpu(layer, x, &routes) {
            return out;
        }
        // A batch too short for that, but more than the decoder takes whole: the GPU does the experts of the
        // first tokens with the decoder's kernels while the CPU does the rest's, each at about its own pace.
        #[cfg(target_os = "macos")]
        let shared = {
            let tokens = x.len() / h;
            let count = (tokens as f32 * GPU_ROW_EXPERT_SHARE) as usize;
            (tokens < FAST_MIN_TOKENS && count > 0 && !float_reference()).then_some(count).and_then(|count| {
                let gpu = self.gpu.as_ref()?;
                let decoder = self.decoder()?;
                // The CPU keeps only the later tokens.
                for tokens in &mut tokens_of {
                    tokens.retain(|&(t, _)| t >= count);
                }
                Some((gpu, decoder, count))
            })
        };
        #[cfg(target_os = "macos")]
        if let Some((gpu, decoder, count)) = shared {
            let (first, cpu) = std::thread::scope(|scope| {
                let on_gpu = scope.spawn(|| decoder.lock().expect("decoder lock").experts(self, gpu, layer, &x[..count * h], &routes[..count * top_k]));
                let cpu = self.expert_outputs_ggml(layer, x, &tokens_of);
                (on_gpu.join().expect("GPU thread panicked"), cpu)
            });
            let mut out = vec![0.0f32; x.len()];
            match first {
                Ok(first) => out[..count * h].copy_from_slice(&first),
                Err(error) => {
                    // The CPU does them after all.
                    gpu_failed(&error);
                    let mut theirs: Vec<Vec<(usize, f32)>> = vec![Vec::new(); cfg.n_routed_experts];
                    for (at, &(expert, weight)) in routes[..count * top_k].iter().enumerate() {
                        theirs[expert as usize].push((at / top_k, weight));
                    }
                    for (t, contribution) in self.expert_outputs_ggml(layer, x, &theirs) {
                        out[t * h..][..h].iter_mut().zip(contribution).for_each(|(o, c)| *o += c);
                    }
                }
            }
            for (t, contribution) in cpu {
                out[t * h..][..h].iter_mut().zip(contribution).for_each(|(o, c)| *o += c);
            }
            return out;
        }

        let contributions = match self.activations {
            _ if layer.experts.gate_up.ggml().is_some() && !float_reference() => self.expert_outputs_ggml(layer, x, &tokens_of),
            Activations::Float | Activations::FloatBf16 => self.expert_outputs_float(layer, x, &tokens_of),
            Activations::Quantised => self.expert_outputs_quantised(layer, x, &tokens_of),
        };
        let mut out = vec![0.0f32; x.len()];
        for (t, contribution) in contributions {
            out[t * h..][..h].iter_mut().zip(contribution).for_each(|(o, c)| *o += c);
        }
        round_all(self.bf16(), &mut out);
        out
    }

    /// The experts of a long batch entirely on the GPU, when the model is set up for it; nothing otherwise, or
    /// when the GPU fails or returns values that are not finite, and the caller takes the usual route.
    #[cfg(target_os = "macos")]
    fn experts_on_gpu(&self, layer: &Layer, x: &Seq, routes: &[(u32, f32)]) -> Option<Seq> {
        let cfg = &self.cfg;
        let experts = &layer.experts;
        let gpu = self.gpu.as_ref().filter(|gpu| gpu.fast() && x.len() / cfg.hidden_size >= WHOLE_EXPERTS_MIN_TOKENS)?;
        let (gate_up, down) = (experts.gate_up.ggml()?, experts.down.ggml()?);
        if experts.interleaved || float_reference() {
            return None;
        }
        // A batch this short does not repay unpacking every expert's weights; the decoder's kernels multiply
        // straight from the stored blocks.
        let job = WholeExperts {
            gate_up: (experts.gate_up.raw_parts().0, gate_up),
            down: (experts.down.raw_parts().0, down),
            experts: cfg.n_routed_experts,
            hidden: cfg.hidden_size,
            units: cfg.intermediate_size,
            chosen: cfg.num_experts_per_tok,
            x,
            routes,
            gate_cap: cfg.mlp_gate_softcap,
            up_cap: cfg.mlp_up_softcap,
        };
        match gpu.experts_whole(&job) {
            Ok(out) if out.par_iter().all(|value| value.is_finite()) => Some(out),
            Ok(_) => {
                GPU_EXPERTS_REDONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                None
            }
            Err(error) => {
                gpu_failed(&error);
                None
            }
        }
    }

    /// Expert outputs with f32 activations, as transformers computes them. Returns `(token, output)` pairs,
    /// already scaled by the routing weight.
    fn expert_outputs_float(&self, layer: &Layer, x: &Seq, tokens_of: &[Vec<(usize, f32)>]) -> Vec<(usize, Vec<f32>)> {
        let cfg = &self.cfg;
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let softcap = |x: f32, cap: f32| (x / cap).tanh() * cap;
        // The bfloat16 imitation rounds the weights too, since transformers stores the dequantised experts in
        // bfloat16, and every product and the activation after them.
        let bf16 = self.bf16();
        let r = |v: f32| if bf16 { round_bf16(v) } else { v };
        tokens_of
            .par_iter()
            .enumerate()
            .filter(|(_, tokens)| !tokens.is_empty())
            .flat_map_iter(|(expert, tokens)| {
                // One token, as in token-by-token decoding: multiply straight from the stored weights.
                if let ([(t, weight)], false) = (&tokens[..], bf16) {
                    let (t, weight) = (*t, *weight);
                    let x = &x[t * h..][..h];
                    let acts: Vec<f32> = (0..inter)
                        .into_par_iter()
                        .with_min_len(64)
                        .map(|unit| {
                            let (gate, up) = layer.experts.gate_up_rows(unit);
                            let gate = layer.experts.gate_up.dot_row(expert, gate, x);
                            let up = layer.experts.gate_up.dot_row(expert, up, x);
                            softcap(gate, cfg.mlp_gate_softcap) * sigmoid(gate) * softcap(up, cfg.mlp_up_softcap)
                        })
                        .collect();
                    let out: Vec<f32> = (0..h)
                        .into_par_iter()
                        .with_min_len(64)
                        .map(|r| layer.experts.down.dot_row(expert, r, &acts) * weight)
                        .collect();
                    return vec![(t, out)];
                }
                // Several tokens: decode each weight row once and apply it to every token routed here.
                let mut row = vec![0.0f32; h];
                let mut acts = vec![0.0f32; tokens.len() * inter];
                for unit in 0..inter {
                    layer.experts.decode_gate_row(expert, unit, &mut row);
                    round_all(bf16, &mut row);
                    let gates: Vec<f32> = tokens.iter().map(|&(t, _)| r(dot(&row, &x[t * h..][..h]))).collect();
                    layer.experts.decode_up_row(expert, unit, &mut row);
                    round_all(bf16, &mut row);
                    for (i, (&(t, _), gate)) in tokens.iter().zip(gates).enumerate() {
                        let up = r(dot(&row, &x[t * h..][..h]));
                        acts[i * inter + unit] =
                            r(softcap(gate, cfg.mlp_gate_softcap) * sigmoid(gate) * softcap(up, cfg.mlp_up_softcap));
                    }
                }
                let mut outs = vec![vec![0.0f32; h]; tokens.len()];
                let mut row = vec![0.0f32; inter];
                for row_index in 0..h {
                    layer.experts.down.decode_row(expert, row_index, &mut row);
                    round_all(bf16, &mut row);
                    for (i, (out, &(_, weight))) in outs.iter_mut().zip(tokens).enumerate() {
                        out[row_index] = r(dot(&row, &acts[i * inter..][..inter])) * weight;
                    }
                }
                tokens.iter().map(|&(t, _)| t).zip(outs).collect::<Vec<_>>()
            })
            .collect()
    }

    /// Expert outputs from a GGUF file's quantised experts, in integer arithmetic: the input of each of the two
    /// matrices is rounded to bytes first, as ggml does on a CPU. Returns `(token, output)` pairs, already scaled
    /// by the routing weight. Every token takes the same arithmetic however many share its expert.
    fn expert_outputs_ggml(&self, layer: &Layer, x: &Seq, tokens_of: &[Vec<(usize, f32)>]) -> Vec<(usize, Vec<f32>)> {
        let cfg = &self.cfg;
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let experts = &layer.experts;
        let inputs: Vec<Int8Activation> = x.par_chunks_exact(h).with_min_len(8).map(quantize_int8).collect();
        // `matrix x` for every activation: `rows` values for the first, then for the second, and so on.
        let multiply = |matrix: &ExpertMatrix, expert: usize, xs: &[&Int8Activation]| -> Vec<f32> {
            let mut out = vec![0.0f32; xs.len() * matrix.rows];
            let mut ranges: Vec<_> = out.chunks_exact_mut(matrix.rows).map(|out| out.chunks_mut(TASK_ROWS)).collect();
            let tasks: Vec<(usize, Vec<&mut [f32]>)> = (0..matrix.rows.div_ceil(TASK_ROWS))
                .map(|range| (range * TASK_ROWS, ranges.iter_mut().map(|r| r.next().expect("a range each")).collect()))
                .collect();
            tasks.into_par_iter().for_each(|(first, mut outs)| {
                let mut dots = [0.0f32; INT8_MAX_BATCH];
                for (xs, outs) in xs.chunks(INT8_MAX_BATCH).zip(outs.chunks_mut(INT8_MAX_BATCH)) {
                    for r in 0..outs[0].len() {
                        let (format, row) = matrix.ggml_row(expert, first + r);
                        dot_row_int8_many(format, row, xs, &mut dots[..xs.len()]);
                        outs.iter_mut().zip(&dots).for_each(|(out, dot)| out[r] = *dot);
                    }
                }
            });
            out
        };
        // The CPU route, for the experts named.
        let on_cpu = |chosen: &[usize]| -> Vec<(usize, Vec<f32>)> {
          chosen
            .par_iter()
            .map(|&expert| (expert, &tokens_of[expert]))
            .filter(|(_, tokens)| !tokens.is_empty())
            .flat_map_iter(|(expert, tokens)| {
                let xs: Vec<&Int8Activation> = tokens.iter().map(|(t, _)| &inputs[*t]).collect();
                let gate_up = multiply(&experts.gate_up, expert, &xs);
                let activations: Vec<Int8Activation> = gate_up
                    .chunks_exact(2 * inter)
                    .map(|gate_up| {
                        let (mut gate, mut up): (Vec<f32>, Vec<f32>) =
                            (0..inter).map(|unit| experts.gate_up_rows(unit)).map(|(g, u)| (gate_up[g], gate_up[u])).unzip();
                        capped_swiglu(&mut gate, &mut up, cfg.mlp_gate_softcap, cfg.mlp_up_softcap, 1.0);
                        quantize_int8(&gate)
                    })
                    .collect();
                let down = multiply(&experts.down, expert, &activations.iter().collect::<Vec<_>>());
                let outputs = down.chunks_exact(h).zip(tokens);
                outputs.map(|(out, &(t, weight))| (t, out.iter().map(|o| o * weight).collect())).collect::<Vec<_>>()
            })
            .collect()
        };

        // A long batch on the fast path: the GPU takes the first experts, unpacked to float and multiplied by a
        // tuned product, while the CPU does the rest as above. The GPU's inputs are not rounded to bytes, so its
        // results match the CPU's only approximately.
        #[cfg(target_os = "macos")]
        if let Some(gpu) = &self.gpu
            && gpu.fast()
            && inputs.len() >= FAST_MIN_TOKENS
            && let (Some(gate_up_format), Some(down_format)) = (experts.gate_up.ggml(), experts.down.ggml())
        {
            // The GPU takes the experts with the most tokens, where one product covers the most work, up to its
            // share; the CPU takes the many with few.
            let total: usize = tokens_of.iter().map(Vec::len).sum();
            let count = tokens_of.len();
            let mut busiest: Vec<usize> = (0..count).collect();
            busiest.sort_by_key(|&e| std::cmp::Reverse(tokens_of[e].len()));
            let mut taken = 0;
            let split = busiest
                .iter()
                .position(|&e| {
                    taken += tokens_of[e].len();
                    taken as f32 > GPU_FAST_EXPERT_SHARE * total as f32
                })
                .unwrap_or(count);
            let (for_gpu, for_cpu) = busiest.split_at(split);
            // Returns the outputs, and the experts whose results were not all finite (see `Gpu::experts_fast`).
            let on_gpu = || -> Result<(Vec<(usize, Vec<f32>)>, Vec<usize>)> {
                let mut firsts: Vec<Vec<&[f32]>> = vec![Vec::new(); count];
                for &e in for_gpu {
                    firsts[e] = tokens_of[e].iter().map(|&(t, _)| &x[t * h..][..h]).collect();
                }
                let (rows, cols) = (experts.gate_up.rows, experts.gate_up.cols);
                let stack = Stack::Ggml(gate_up_format);
                let gate_up = gpu.experts_fast(experts.gate_up.raw_parts(), (count, rows, cols), stack, &firsts, 0)?;
                let acts: Vec<Vec<f32>> = gate_up
                    .par_iter()
                    .map(|products| {
                        let mut acts = Vec::with_capacity(products.len() / 2);
                        for products in products.chunks_exact(rows) {
                            let (mut gate, mut up): (Vec<f32>, Vec<f32>) =
                                (0..inter).map(|unit| experts.gate_up_rows(unit)).map(|(g, u)| (products[g], products[u])).unzip();
                            capped_swiglu(&mut gate, &mut up, cfg.mlp_gate_softcap, cfg.mlp_up_softcap, 1.0);
                            acts.extend(gate);
                        }
                        acts
                    })
                    .collect();
                let seconds: Vec<Vec<&[f32]>> = acts.iter().map(|acts| acts.chunks_exact(inter).collect()).collect();
                let (rows, cols) = (experts.down.rows, experts.down.cols);
                let stack = Stack::Ggml(down_format);
                let down = gpu.experts_fast(experts.down.raw_parts(), (count, rows, cols), stack, &seconds, 1)?;
                // A NaN in the first product carries through the activation into the whole of the second.
                let finite: Vec<bool> = down.par_iter().map(|products| products.iter().all(|v| v.is_finite())).collect();
                let (mut outputs, mut failed) = (Vec::new(), Vec::new());
                for (expert, (products, tokens)) in down.iter().zip(tokens_of).enumerate() {
                    if !finite[expert] {
                        failed.push(expert);
                        continue;
                    }
                    for (product, &(t, weight)) in products.chunks_exact(rows).zip(tokens) {
                        outputs.push((t, product.iter().map(|v| v * weight).collect()));
                    }
                }
                Ok((outputs, failed))
            };
            let (gpu_outputs, mut cpu_outputs) = std::thread::scope(|scope| {
                let gpu_work = scope.spawn(on_gpu);
                let cpu_outputs = on_cpu(for_cpu);
                (gpu_work.join().expect("GPU thread panicked"), cpu_outputs)
            });
            let mut outputs = match gpu_outputs {
                Ok((mut outputs, failed)) => {
                    GPU_EXPERTS_REDONE.fetch_add(failed.len(), std::sync::atomic::Ordering::Relaxed);
                    outputs.append(&mut on_cpu(&failed));
                    outputs
                }
                Err(error) => {
                    gpu_failed(&error);
                    on_cpu(for_gpu)
                }
            };
            outputs.append(&mut cpu_outputs);
            return outputs;
        }
        on_cpu(&(0..tokens_of.len()).collect::<Vec<_>>())
    }

    /// Expert outputs with the activations quantised as in training: each expert input is rounded to the format
    /// of the weights it meets (NVFP4 or MXFP8), and each result to bfloat16, following vLLM's reference.
    ///
    /// Tokens routed to the same expert are processed in tiles, but every token takes the same arithmetic
    /// whether it arrives alone or in a tile, so the rounding does not depend on how the sequence was split
    /// across calls.
    fn expert_outputs_quantised(
        &self,
        layer: &Layer,
        x: &Seq,
        tokens_of: &[Vec<(usize, f32)>],
    ) -> Vec<(usize, Vec<f32>)> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let softcap = |x: f32, cap: f32| (x / cap).tanh() * cap;
        let experts = &layer.experts;

        enum Input {
            Nvfp4(Nvfp4Activation),
            Mxfp8(Vec<f32>),
        }
        let quantise = |mut values: Vec<f32>| {
            values.iter_mut().for_each(|v| *v = round_bf16(*v));
            match experts.gate_up.is_nvfp4() {
                true => Input::Nvfp4(quantize_nvfp4(&values)),
                false => {
                    requantize_mxfp8(&mut values);
                    Input::Mxfp8(values)
                }
            }
        };
        // One expert matrix applied to a tile of inputs: every weight row is read once for the whole tile.
        // Returns one output vector per input.
        let apply = |matrix: &ExpertMatrix, expert: usize, inputs: &[&Input]| -> Vec<Vec<f32>> {
            let mut nvfp4 = Vec::new();
            let mut mxfp8 = Vec::new();
            for input in inputs {
                match input {
                    Input::Nvfp4(x) => nvfp4.push(x),
                    Input::Mxfp8(x) => mxfp8.push(x.as_slice()),
                }
            }
            let scratch = || vec![0.0f32; if mxfp8.is_empty() { 0 } else { matrix.cols }];
            let dots = |decoded: &mut Vec<f32>, row: usize| -> [f32; TOKEN_TILE] {
                let mut out = [0.0f32; TOKEN_TILE];
                if mxfp8.is_empty() {
                    matrix.dot_row_quantized_many(expert, row, &nvfp4, &mut out[..nvfp4.len()]);
                } else {
                    matrix.decode_row(expert, row, decoded);
                    dot_f32_row_many(decoded, &mxfp8, &mut out[..mxfp8.len()]);
                }
                out.map(round_bf16)
            };
            let rows = (0..matrix.rows).into_par_iter().with_min_len(64);
            // A lone input, as in token-by-token decoding, needs no regrouping of the per-row results.
            if inputs.len() == 1 {
                return vec![rows.map_init(scratch, |decoded, row| dots(decoded, row)[0]).collect()];
            }
            let per_row: Vec<[f32; TOKEN_TILE]> = rows.map_init(scratch, dots).collect();
            (0..inputs.len()).map(|i| per_row.iter().map(|row| row[i]).collect()).collect()
        };

        // A token's input is quantised once and shared by all the experts it is routed to.
        let inputs: Vec<Input> = x.par_chunks_exact(h).map(|x| quantise(x.to_vec())).collect();

        // The CPU route, for experts `first_expert` onwards.
        let on_cpu = |first_expert: usize, tokens_of: &[Vec<(usize, f32)>]| -> Vec<(usize, Vec<f32>)> {
            tokens_of
                .par_iter()
                .enumerate()
                .filter(|(_, tokens)| !tokens.is_empty())
                .flat_map_iter(|(index, tokens)| {
                    let expert = first_expert + index;
                    let mut outputs = Vec::with_capacity(tokens.len());
                    for tile in tokens.chunks(TOKEN_TILE) {
                        let tile_inputs: Vec<&Input> = tile.iter().map(|&(t, _)| &inputs[t]).collect();
                        let gate_up = apply(&experts.gate_up, expert, &tile_inputs);
                        // The routing weight goes in before the second quantisation.
                        let acts: Vec<Input> = gate_up
                            .iter()
                            .zip(tile)
                            .map(|(gate_up, &(_, weight))| {
                                let act = gate_up.as_chunks::<2>().0.iter().map(|&[gate, up]| {
                                    softcap(gate, cfg.mlp_gate_softcap)
                                        * sigmoid(gate)
                                        * softcap(up, cfg.mlp_up_softcap)
                                        * weight
                                });
                                quantise(act.collect())
                            })
                            .collect();
                        let down = apply(&experts.down, expert, &acts.iter().collect::<Vec<_>>());
                        outputs.extend(tile.iter().map(|&(t, _)| t).zip(down));
                    }
                    outputs
                })
                .collect()
        };

        // A long batch, where the fast path is allowed: every expert's tokens go through one tuned float product
        // per matrix. The activations are the same quantised values as below, so only the order of the sums, and
        // with it the last bits before rounding, differs.
        #[cfg(target_os = "macos")]
        if let Some(gpu) = &self.gpu
            && gpu.fast()
            && inputs.len() >= FAST_MIN_TOKENS
        {
            let count = tokens_of.len();
            let stack = if experts.gate_up.is_nvfp4() { Stack::Nvfp4 } else { Stack::Mxfp8 };
            // An input as plain floats and the scale left out of them: NVFP4 keeps its outer scale apart so the
            // floats are exact; an MXFP8 input is already the values it stands for.
            let activation = |input: &Input| -> (Vec<f32>, f32) {
                match input {
                    Input::Nvfp4(x) => (x.unscaled(), x.outer_scale()),
                    Input::Mxfp8(x) => (x.clone(), 1.0),
                }
            };
            let run = || -> Result<Vec<(usize, Vec<f32>)>> {
                let unscaled: Vec<(Vec<f32>, f32)> = inputs.par_iter().map(activation).collect();
                let firsts: Vec<Vec<&[f32]>> =
                    tokens_of.iter().map(|tokens| tokens.iter().map(|&(t, _)| unscaled[t].0.as_slice()).collect()).collect();
                let (rows, cols) = (experts.gate_up.rows, experts.gate_up.cols);
                let gate_up = gpu.experts_fast(experts.gate_up.raw_parts(), (count, rows, cols), stack, &firsts, 0)?;

                // Scale, round, activate with the routing weight, and quantise again, per (expert, token).
                let acts: Vec<Vec<(Vec<f32>, f32)>> = gate_up
                    .par_iter()
                    .zip(tokens_of)
                    .enumerate()
                    .map(|(expert, (products, tokens))| {
                        let per_token = products.par_chunks_exact(rows).zip(tokens).map(|(products, &(t, weight))| {
                            let scale = experts.gate_up.outer_scale(expert) * unscaled[t].1;
                            let (mut gate, mut up): (Vec<f32>, Vec<f32>) = products
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .map(|&[gate, up]| (round_bf16(gate * scale), round_bf16(up * scale)))
                                .unzip();
                            capped_swiglu(&mut gate, &mut up, cfg.mlp_gate_softcap, cfg.mlp_up_softcap, weight);
                            activation(&quantise(gate))
                        });
                        per_token.collect()
                    })
                    .collect();
                let seconds: Vec<Vec<&[f32]>> =
                    acts.iter().map(|acts| acts.iter().map(|(values, _)| values.as_slice()).collect()).collect();
                let (rows, cols) = (experts.down.rows, experts.down.cols);
                let down = gpu.experts_fast(experts.down.raw_parts(), (count, rows, cols), stack, &seconds, 1)?;

                // In expert order, then token order: the order the other routes produce their outputs in.
                let mut outputs = Vec::new();
                for (expert, ((products, acts), tokens)) in down.iter().zip(&acts).zip(tokens_of).enumerate() {
                    // The GPU's product now and then returns NaN (see `Gpu::experts_fast`), which carries through
                    // to here; such an expert is done again on the CPU.
                    if !products.iter().all(|v| v.is_finite()) {
                        GPU_EXPERTS_REDONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        outputs.append(&mut on_cpu(expert, &tokens_of[expert..=expert]));
                        continue;
                    }
                    for ((product, (_, act_scale)), &(t, _)) in products.chunks_exact(rows).zip(acts).zip(tokens) {
                        let scale = experts.down.outer_scale(expert) * act_scale;
                        outputs.push((t, product.iter().map(|v| round_bf16(v * scale)).collect()));
                    }
                }
                Ok(outputs)
            };
            match run() {
                Ok(outputs) => return outputs,
                Err(error) => gpu_failed(&error),
            }
        }

        // Enough tokens, a GPU and NVFP4 weights: the GPU takes the first experts and the CPU the rest, at the
        // same time. Both give the same results, so the split only affects speed. On the GPU both matrices run
        // there, with the activation function and the second quantisation in between done here.
        #[cfg(target_os = "macos")]
        if let Some(gpu) = &self.gpu
            && inputs.len() >= GPU_MIN_TOKENS
            && let Some((gate_up_values, gate_up_scales, gate_up_outer)) = experts.gate_up.nvfp4_parts()
            && let Some((down_values, down_scales, down_outer)) = experts.down.nvfp4_parts()
        {
            let nvfp4 = |input: &'_ Input| -> Option<Nvfp4Activation> {
                match input {
                    Input::Nvfp4(x) => Some(x.clone()),
                    Input::Mxfp8(_) => None,
                }
            };
            let activations: Vec<Nvfp4Activation> = inputs.iter().filter_map(nvfp4).collect();
            let activations: Vec<&Nvfp4Activation> = activations.iter().collect();
            // The GPU's experts: as many from the front as hold its share of the routed tokens.
            let total: usize = tokens_of.iter().map(Vec::len).sum();
            let mut taken = 0;
            let split = tokens_of
                .iter()
                .position(|tokens| {
                    taken += tokens.len();
                    taken as f32 > GPU_EXPERT_SHARE * total as f32
                })
                .unwrap_or(tokens_of.len());
            // In expert order, then token order: the order the CPU route produces its outputs in.
            let routed: Vec<(usize, usize, f32)> = tokens_of[..split]
                .iter()
                .enumerate()
                .flat_map(|(expert, tokens)| tokens.iter().map(move |&(t, weight)| (expert, t, weight)))
                .collect();
            let on_gpu = || -> Result<Vec<(usize, Vec<f32>)>> {
                let mut outputs = Vec::with_capacity(routed.len());
                for batch in routed.chunks(GPU_PAIR_BATCH) {
                    let pairs: Vec<_> = batch.iter().map(|&(expert, t, _)| (expert, t, gate_up_outer[expert])).collect();
                    let gate_up = gpu.experts(&ExpertJob {
                        values: gate_up_values,
                        scales: gate_up_scales,
                        rows: experts.gate_up.rows,
                        cols: experts.gate_up.cols,
                        activations: &activations,
                        pairs: &pairs,
                    })?;
                    let acts: Vec<Nvfp4Activation> = gate_up
                        .par_chunks_exact(experts.gate_up.rows)
                        .zip(batch)
                        .filter_map(|(gate_up, &(_, _, weight))| {
                            let act = gate_up.as_chunks::<2>().0.iter().map(|&[gate, up]| {
                                softcap(gate, cfg.mlp_gate_softcap) * sigmoid(gate) * softcap(up, cfg.mlp_up_softcap) * weight
                            });
                            nvfp4(&quantise(act.collect()))
                        })
                        .collect();
                    let acts: Vec<&Nvfp4Activation> = acts.iter().collect();
                    let pairs: Vec<_> =
                        batch.iter().enumerate().map(|(i, &(expert, _, _))| (expert, i, down_outer[expert])).collect();
                    let down = gpu.experts(&ExpertJob {
                        values: down_values,
                        scales: down_scales,
                        rows: experts.down.rows,
                        cols: experts.down.cols,
                        activations: &acts,
                        pairs: &pairs,
                    })?;
                    let tokens = batch.iter().map(|&(_, t, _)| t);
                    outputs.extend(tokens.zip(down.chunks_exact(experts.down.rows).map(<[f32]>::to_vec)));
                }
                Ok(outputs)
            };
            let (gpu_outputs, mut cpu_outputs) = std::thread::scope(|scope| {
                let gpu_work = scope.spawn(on_gpu);
                let cpu_outputs = on_cpu(split, &tokens_of[split..]);
                (gpu_work.join().expect("GPU thread panicked"), cpu_outputs)
            });
            match gpu_outputs {
                Ok(mut outputs) => {
                    outputs.append(&mut cpu_outputs);
                    return outputs;
                }
                Err(error) => gpu_failed(&error),
            }
        }
        on_cpu(0, tokens_of)
    }
}

/// Where the heavy matrix products run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// The GPU for large batches when there is one, otherwise the CPU. Either way the results are the same,
    /// bit for bit.
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
    if checkpoint.is_gguf() && backend != Backend::MetalFast {
        ensure!(backend != Backend::Metal, "GGUF weights have no exact Metal kernels; use `metal-fast` or the CPU");
        return Ok(None);
    }
    match backend {
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
    REPORTED.call_once(|| eprintln!("Metal backend failed, continuing on the CPU: {error:#}"));
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

/// `gate[i] = softcap(gate[i]) * sigmoid(gate[i]) * softcap(up[i]) * weight`, the experts' activation, over whole
/// vectors, with vectorised `tanh` and `exp`. `up` is used as working space.
fn capped_swiglu(gate: &mut [f32], up: &mut [f32], gate_cap: f32, up_cap: f32, weight: f32) {
    let mut capped: Vec<f32> = gate.iter().map(|g| g / gate_cap).collect();
    let mut negated: Vec<f32> = gate.iter().map(|g| -g).collect();
    up.iter_mut().for_each(|u| *u /= up_cap);
    accel::tanh_in_place(&mut capped);
    accel::exp_in_place(&mut negated);
    accel::tanh_in_place(up);
    for (((gate, capped), exp), up) in gate.iter_mut().zip(&capped).zip(&negated).zip(up.iter()) {
        *gate = (capped * gate_cap) * (1.0 / (1.0 + exp)) * (up * up_cap) * weight;
    }
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

impl KvConv<'_> {
    fn apply(
        &self,
        x: &Seq,
        dynamic_input: &Seq,
        channels: usize,
        cfg: &TextConfig,
        history: &mut Vec<f32>,
        bf16: bool,
    ) -> Seq {
        let k = cfg.sconv_kernel_size;
        match self {
            KvConv::Static(weight) => static_conv(x, weight, channels, k, history),
            KvConv::Dynamic { static_w, proj } => {
                // Taps are per channel (static) plus per token and key/value head (dynamic); tap `lag` reaches
                // `lag` tokens back.
                let mut taps = proj.apply(dynamic_input);
                round_all(bf16, &mut taps);
                let head_dim = channels / cfg.num_key_value_heads;
                let scale = 1.0 / ((2 * k + 1) as f32).sqrt();
                let (context, seen) = with_history(history, x, channels, k);
                let mut out = vec![0.0f32; x.len()];
                for (i, out) in out.chunks_exact_mut(channels).enumerate() {
                    let t = seen + i;
                    for lag in 0..k.min(t + 1) {
                        let past = &context[(t - lag) * channels..][..channels];
                        for (c, (o, p)) in out.iter_mut().zip(past).enumerate() {
                            let tap = static_w[lag * channels + c] + taps[i * proj.rows + (c / head_dim) * k + lag];
                            *o += tap * p;
                        }
                    }
                    let current = &context[t * channels..][..channels];
                    out.iter_mut().zip(current).for_each(|(o, c)| *o = (*o + c) * scale);
                }
                out
            }
        }
    }
}

impl Controller<'_> {
    /// `(1 - forget) * hidden + gain * message`, with both gates read from the normed pair through a low rank.
    fn merge(&self, gpu: Option<&Gpu>, hidden: &Seq, message: &Seq, h: usize, eps: f32, bf16: bool) -> Seq {
        // A long batch on the fast path: the three products on the GPU, which is several times quicker at them
        // than every core together.
        #[cfg(target_os = "macos")]
        if let Some(gpu) = gpu.filter(|gpu| gpu.fast() && hidden.len() / h >= FAST_MIN_TOKENS && !bf16)
            && [&self.down, &self.forget, &self.gain].iter().all(|matrix| matrix.format == Format::F32)
        {
            let mut pairs = vec![0.0f32; 2 * hidden.len()];
            pairs.par_chunks_exact_mut(2 * h).zip(hidden.par_chunks_exact(h)).zip(message.par_chunks_exact(h)).for_each(
                |((pair, hidden), message)| {
                    let (a, b) = pair.split_at_mut(h);
                    a.copy_from_slice(hidden);
                    b.copy_from_slice(message);
                    rms_norm(a, None, eps);
                    rms_norm(b, None, eps);
                },
            );
            match gpu.gates_fast(&pairs, &self.down.bytes, &self.forget.bytes, &self.gain.bytes, (self.down.rows, h)) {
                Ok((forget, gain)) if !forget.par_iter().chain(&gain).all(|value| value.is_finite()) => {
                    GPU_PRODUCTS_REDONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Ok((mut forget, mut gain)) => {
                    let mut merged = vec![0.0f32; hidden.len()];
                    (merged.par_chunks_mut(h), forget.par_chunks_mut(h), gain.par_chunks_mut(h), hidden.par_chunks(h), message.par_chunks(h))
                        .into_par_iter()
                        .for_each(|(merged, forget, gain, hidden, message)| {
                            accel::tanh_in_place(forget);
                            accel::tanh_in_place(gain);
                            for i in 0..h {
                                merged[i] = (1.0 - forget[i]) * hidden[i] + gain[i] * message[i];
                            }
                        });
                    return merged;
                }
                Err(error) => gpu_failed(&error),
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = gpu;
        let merged = per_token(hidden.len() / h, |t| {
            let (hidden, message) = (&hidden[t * h..][..h], &message[t * h..][..h]);
            let mut pair = [hidden, message].concat();
            let (a, b) = pair.split_at_mut(h);
            rms_norm(a, None, eps);
            rms_norm(b, None, eps);
            round_all(bf16, &mut pair);
            let mut low = self.down.apply_one(&pair);
            round_all(bf16, &mut low);
            let (mut forget, mut gain) = (self.forget.apply_one(&low), self.gain.apply_one(&low));
            accel::tanh_in_place(&mut forget);
            accel::tanh_in_place(&mut gain);
            (0..h).map(|i| (1.0 - forget[i]) * hidden[i] + gain[i] * message[i]).collect::<Vec<f32>>()
        });
        merged.concat()
    }
}

/// Row-major dense matrix in its stored form (bfloat16 in the checkpoint, a block format in a GGUF file),
/// borrowed from the weights and applied as `y = W x`.
pub(crate) struct Mat<'a> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) format: Format,
    row_bytes: usize,
    pub(crate) bytes: std::borrow::Cow<'a, [u8]>,
}

impl<'a> Mat<'a> {
    pub(crate) fn open(weights: &'a Checkpoint, name: &str, rows: usize, cols: usize) -> Result<Self> {
        let (format, shape, bytes) = weights.rows(name)?;
        let row_bytes = format.row_bytes(cols).with_context(|| format!("{name}: rows of {cols} in {format:?}"))?;
        ensure!(bytes.len() == rows * row_bytes, "{name}: expected {rows}x{cols}, found shape {shape:?}");
        ensure!(format != Format::F32 || weights.is_gguf(), "{name}: expected bfloat16");
        Ok(Mat { rows, cols, format, row_bytes, bytes: bytes.into() })
    }

    /// A copy with every row packed as `Q8_0`, so that it is multiplied in integer arithmetic: several times
    /// faster than bfloat16, and off by a little. The matrix itself when it is quantised already, or its rows do
    /// not make whole blocks.
    pub(crate) fn rounded_to_bytes(self) -> Mat<'a> {
        self.packed().unwrap_or(self)
    }

    /// The copy [`Mat::rounded_to_bytes`] makes, or nothing where that returns the matrix itself.
    pub(crate) fn packed(&self) -> Option<Mat<'static>> {
        if self.format.is_quantised() || !self.cols.is_multiple_of(crate::ggml::INT8_BLOCK) {
            return None;
        }
        let format = Format::Q8_0;
        let row_bytes = format.row_bytes(self.cols).expect("whole blocks");
        let mut bytes = vec![0u8; self.rows * row_bytes];
        bytes.par_chunks_exact_mut(row_bytes).zip(self.bytes.par_chunks_exact(self.row_bytes)).for_each_init(
            || vec![0.0f32; self.cols],
            |widened, (packed, row)| {
                self.format.decode_row(row, widened);
                packed.copy_from_slice(&crate::ggml::encode_q8_0(widened));
            },
        );
        Some(Mat { rows: self.rows, cols: self.cols, format, row_bytes, bytes: bytes.into() })
    }

    pub(crate) fn row(&self, row: usize) -> &[u8] {
        &self.bytes[row * self.row_bytes..][..self.row_bytes]
    }

    /// Rows `first..first + count` as floats, one after another.
    pub(crate) fn widened(&self, first: usize, count: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; count * self.cols];
        for (r, out) in out.chunks_exact_mut(self.cols).enumerate() {
            self.format.decode_row(self.row(first + r), out);
        }
        out
    }

    /// Whether the matrix is multiplied in integer arithmetic, against activations rounded to bytes.
    fn int8(&self) -> bool {
        self.format.is_quantised() && self.cols.is_multiple_of(crate::ggml::INT8_BLOCK) && !float_reference()
    }

    /// `W x` for each of the rounded activations `xs`, rows `first..first + outs[i].len()`, into `outs`.
    fn apply_int8(&self, first: usize, xs: &[&Int8Activation], outs: &mut [&mut [f32]]) {
        let mut dots = [0.0f32; INT8_MAX_BATCH];
        for (xs, outs) in xs.chunks(INT8_MAX_BATCH).zip(outs.chunks_mut(INT8_MAX_BATCH)) {
            for r in 0..outs[0].len() {
                dot_row_int8_many(self.format, self.row(first + r), xs, &mut dots[..xs.len()]);
                outs.iter_mut().zip(&dots).for_each(|(out, dot)| out[r] = *dot);
            }
        }
    }

    /// Applies the matrix to every vector of a sequence.
    pub(crate) fn apply(&self, xs: &[f32]) -> Seq {
        apply_many(None, &[(self, xs)]).remove(0)
    }

    /// Applies the matrix to one vector on the calling thread.
    pub(crate) fn apply_one(&self, x: &[f32]) -> Vec<f32> {
        if self.int8() {
            let mut out = vec![0.0f32; self.rows];
            self.apply_int8(0, &[&quantize_int8(x)], &mut [&mut out[..]]);
            return out;
        }
        let mut scratch = Vec::new();
        self.bytes.chunks_exact(self.row_bytes).map(|row| self.format.dot_row(row, x, &mut scratch)).collect()
    }
}

/// Applies each matrix to its sequence. The products share one parallel region, split into row ranges, because
/// waking the thread pool separately for every matrix costs more than the smaller products themselves.
///
/// Tokens go through in tiles: each weight row is widened once and applied to every token of the tile, so a long
/// sequence reads the weights once per tile instead of once per token. A lone token skips the widening; the two
/// routes give the same result bit for bit.
///
/// With a GPU and enough tokens the whole batch goes there instead; the results are identical.
pub(crate) fn apply_many(gpu: Option<&Gpu>, jobs: &[(&Mat, &[f32])]) -> Vec<Seq> {
    #[cfg(target_os = "macos")]
    if let Some(gpu) = gpu
        && jobs.iter().any(|(m, xs)| xs.len() / m.cols >= GPU_MIN_TOKENS)
    {
        let fast = gpu.fast() && jobs.iter().any(|(m, xs)| xs.len() / m.cols >= FAST_MIN_TOKENS);
        // The exact kernel reads bfloat16; the fast path also unpacks a GGUF file's block formats.
        let suits = |m: &Mat| match m.format {
            Format::Bf16 => m.cols % 16 == 0,
            format => fast && format.is_quantised() && m.cols % 32 == 0,
        };
        if jobs.iter().all(|(m, _)| suits(m)) {
        let dense: Vec<DenseJob> = jobs
            .iter()
            .map(|(m, xs)| DenseJob { format: m.format, weights: &m.bytes, rows: m.rows, cols: m.cols, xs })
            .collect();
        match if fast { gpu.dense_fast(&dense) } else { gpu.dense(&dense) } {
            // The fast product now and then returns NaN (see `Gpu::experts_fast`), which nothing downstream
            // would catch: it would sit in the state and spoil every token after it.
            Ok(outs) if !fast || outs.par_iter().all(|out| out.iter().all(|value| value.is_finite())) => return outs,
            Ok(_) => drop(GPU_PRODUCTS_REDONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)),
            Err(error) => gpu_failed(&error),
        }
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = gpu;
    let mut outs: Vec<Seq> = jobs.iter().map(|(m, xs)| vec![0.0f32; xs.len() / m.cols * m.rows]).collect();
    let work: usize = jobs.iter().map(|(m, xs)| xs.len() * m.rows).sum();

    // One task per (matrix, tile of tokens, range of rows); it owns that range of each token's output.
    // Quantised matrices meet their inputs rounded to bytes, once for all the rows.
    let rounded: Vec<Vec<Int8Activation>> = jobs
        .iter()
        .map(|(m, xs)| match m.int8() {
            true => xs.chunks_exact(m.cols).map(quantize_int8).collect(),
            false => Vec::new(),
        })
        .collect();
    let mut tasks = Vec::new();
    for (((m, xs), out), rounded) in jobs.iter().zip(&mut outs).zip(&rounded) {
        let mut ranges: Vec<_> = out.chunks_exact_mut(m.rows).map(|out| out.chunks_mut(TASK_ROWS)).collect();
        let tiles = xs.chunks(DENSE_TILE * m.cols).zip(ranges.chunks_mut(DENSE_TILE)).enumerate();
        for (t, (tile, ranges)) in tiles {
            let rounded = rounded.get(t * DENSE_TILE..).unwrap_or_default();
            for range in 0..m.rows.div_ceil(TASK_ROWS) {
                let outs: Vec<&mut [f32]> = ranges.iter_mut().map(|r| r.next().expect("a range per token")).collect();
                tasks.push((*m, tile, rounded, range * TASK_ROWS, outs));
            }
        }
    }
    let run = |(m, xs, rounded, first, mut outs): (&Mat, &[f32], &[Int8Activation], usize, Vec<&mut [f32]>)| {
        if m.int8() {
            let xs: Vec<&Int8Activation> = rounded.iter().take(outs.len()).collect();
            return m.apply_int8(first, &xs, &mut outs);
        }
        let rows = m.bytes[first * m.row_bytes..].chunks_exact(m.row_bytes).take(outs[0].len());
        if let [out] = &mut outs[..] {
            let mut scratch = Vec::new();
            out.iter_mut().zip(rows).for_each(|(o, row)| *o = m.format.dot_row(row, xs, &mut scratch));
            return;
        }
        let xs: Vec<&[f32]> = xs.chunks_exact(m.cols).collect();
        let mut widened = vec![0.0f32; m.cols];
        let mut dots = vec![0.0f32; xs.len()];
        for (r, row) in rows.enumerate() {
            m.format.decode_row(row, &mut widened);
            dot_f32_row_many(&widened, &xs, &mut dots);
            outs.iter_mut().zip(&dots).for_each(|(out, dot)| out[r] = *dot);
        }
    };
    match work < PARALLEL_MIN_WORK {
        true => tasks.into_iter().for_each(run),
        false => tasks.into_par_iter().for_each(run),
    }
    outs
}

/// Learned-query attention over the normed residual blocks plus the block in progress; one output per query.
///
/// `blocks` are finished blocks, already normed with [`residual_normed`], which a block is once for all the reads
/// that follow; `partial` is normed here.
fn attn_res(queries: &[f32], blocks: &[&Seq], partial: &Seq, h: usize) -> Vec<Seq> {
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
fn residual_normed(mut block: Seq, h: usize) -> Seq {
    block.chunks_exact_mut(h).for_each(|x| rms_norm(x, None, RESIDUAL_NORM_EPS));
    block
}

/// Runs `work` for each token, across threads once there are enough tokens to be worth waking the pool.
fn per_token<T: Send>(tokens: usize, work: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    match tokens < PARALLEL_MIN_TOKENS {
        true => (0..tokens).map(work).collect(),
        false => (0..tokens).into_par_iter().map(work).collect(),
    }
}

/// Prepends the convolution's remembered inputs to `x` and remembers the most recent of the result.
/// Returns the joined context and how many remembered vectors lead it.
fn with_history(history: &mut Vec<f32>, x: &[f32], channels: usize, k: usize) -> (Vec<f32>, usize) {
    let seen = history.len() / channels;
    let context = [history.as_slice(), x].concat();
    let keep = (k - 1 + MAX_ROLLBACK).min(context.len() / channels) * channels;
    *history = context[context.len() - keep..].to_vec();
    (context, seen)
}

/// Causal depthwise convolution with a residual: `(conv(x) + x) / sqrt(kernel + 1)`. `weight` is
/// `[channels, kernel]` with the last tap on the current token. `history` carries the inputs of earlier calls.
fn static_conv(x: &[f32], weight: &[f32], channels: usize, k: usize, history: &mut Vec<f32>) -> Seq {
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
fn l2_clip(x: &mut [f32], threshold: f32) {
    let norm = dot(x, x).sqrt();
    let scale = (threshold / (norm + 1e-6)).min(1.0);
    x.iter_mut().for_each(|v| *v *= scale);
}

fn add(a: &[f32], b: &[f32]) -> Seq {
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
        use crate::checkpoint::resolve_model;
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
