//! Memory-mapped access to the weights, under the published checkpoint's tensor names: either that checkpoint
//! (safetensors shards) or a GGUF conversion of it, whose tensors are presented under the same names and layouts.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::weights::ggml::Format;
use crate::weights::quant;

/// How a tensor's values are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stored {
    /// Rows of weights in one of the formats [`Format`] can decode.
    Rows(Format),
    /// Bytes with a meaning of their own: packed NVFP4 nibbles, or MXFP8 block exponents.
    U8,
    /// 8-bit floats (E4M3): NVFP4 block scales, or MXFP8 values.
    E4m3,
}

/// How a GGUF tensor's layout differs from the published checkpoint's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Same,
    /// Stored `[b, a]` where the checkpoint has `[a, b]`.
    Transposed,
    /// A convolution stored `[channels, kernel]` with the last tap on the current token, where the checkpoint has
    /// `[kernel, channels]` with the first tap on it.
    ConvTaps,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub stored: Stored,
    /// The shape under the checkpoint's layout.
    pub shape: Vec<usize>,
    layout: Layout,
    file: usize,
    start: usize,
    end: usize,
}

impl TensorInfo {
    pub fn byte_len(&self) -> usize {
        self.end - self.start
    }
}

pub struct Checkpoint {
    files: Vec<Mmap>,
    tensors: BTreeMap<String, TensorInfo>,
    gguf: bool,
}

impl Checkpoint {
    /// Opens the weights at `model`: every `model-*.safetensors` shard of a checkpoint directory, or a GGUF file.
    /// Nothing is read until a tensor is touched.
    pub fn open(model: &Path) -> Result<Self> {
        if is_gguf(model) {
            return Self::open_gguf(model);
        }
        let model_dir = model;
        let mut paths: Vec<PathBuf> = std::fs::read_dir(model_dir)
            .with_context(|| format!("listing {}", model_dir.display()))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|path| {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
                name.starts_with("model") && name.ends_with(".safetensors")
            })
            .collect();
        paths.sort();
        ensure!(!paths.is_empty(), "no model*.safetensors in {}", model_dir.display());

        Self::map(&paths)
    }

    /// Opens a single safetensors file, such as the draft model beside the main checkpoint.
    pub fn open_file(path: &Path) -> Result<Self> {
        Self::map(&[path.to_path_buf()])
    }

    fn map(paths: &[PathBuf]) -> Result<Self> {
        let mut files = Vec::new();
        let mut tensors = BTreeMap::new();
        for (file, path) in paths.iter().enumerate() {
            let handle = File::open(path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: the checkpoint is treated as read-only; truncating it while mapped is undefined behaviour.
            let mmap = unsafe { Mmap::map(&handle) }.with_context(|| format!("mapping {}", path.display()))?;
            let (header_len, metadata) =
                SafeTensors::read_metadata(&mmap).map_err(|e| anyhow!("{}: {e}", path.display()))?;
            let data_start = header_len + 8;
            for (name, info) in metadata.tensors() {
                let (start, end) = info.data_offsets;
                let stored = match info.dtype {
                    Dtype::F32 => Stored::Rows(Format::F32),
                    Dtype::BF16 => Stored::Rows(Format::Bf16),
                    Dtype::U8 => Stored::U8,
                    Dtype::F8_E4M3 => Stored::E4m3,
                    other => bail!("{name}: unsupported type {other:?}"),
                };
                let info = TensorInfo {
                    stored,
                    shape: info.shape.clone(),
                    layout: Layout::Same,
                    file,
                    start: data_start + start,
                    end: data_start + end,
                };
                ensure!(info.end <= mmap.len(), "{name} runs past the end of {}", path.display());
                ensure!(tensors.insert(name.clone(), info).is_none(), "{name} appears in two shards");
            }
            files.push(mmap);
        }
        Ok(Self { files, tensors, gguf: false })
    }

    /// Opens a GGUF conversion of the checkpoint. Its tensors are filed under the checkpoint's names, so the rest
    /// of the engine does not need to know which kind of file it is reading.
    pub fn open_gguf(path: &Path) -> Result<Self> {
        let reader = gguf_rs_lib::reader::file_reader::open_gguf_file(path)
            .map_err(|e| anyhow!("reading {}: {e}", path.display()))?;
        let architecture = reader.metadata().get("general.architecture").and_then(|v| v.as_str()).unwrap_or_default();
        ensure!(architecture == "inkling2_loop", "{}: architecture {architecture:?} is not supported", path.display());
        let handle = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: as for the safetensors shards.
        let mmap = unsafe { Mmap::map(&handle) }.with_context(|| format!("mapping {}", path.display()))?;

        let names: Vec<&str> = reader.tensor_names();
        let full_layers: Vec<String> = names
            .iter()
            .filter_map(|name| name.strip_suffix(".attn_k_conv_proj.weight").map(|block| format!("{block}.")))
            .collect();
        let mut tensors = BTreeMap::new();
        for tensor in reader.tensor_infos() {
            use gguf_rs_lib::format::types::GGUFTensorType as Type;
            let name = tensor.name();
            let full = full_layers.iter().any(|block| name.starts_with(block.as_str()));
            let Some((name, layout)) = checkpoint_name(name, full) else { continue };
            let format = match tensor.tensor_type() {
                Type::F32 => Format::F32,
                Type::BF16 => Format::Bf16,
                Type::Q4_K => Format::Q4K,
                Type::Q5_0 => Format::Q5_0,
                Type::Q6_K => Format::Q6K,
                Type::Q8_0 => Format::Q8_0,
                other => bail!("{name} is stored as {other:?}; this engine reads F32, BF16, Q4_K, Q5_0, Q6_K and Q8_0"),
            };
            // GGUF lists dimensions innermost first.
            let stored_shape: Vec<usize> = tensor.shape().dims().iter().rev().map(|d| *d as usize).collect();
            let cols = *stored_shape.last().context("a tensor without dimensions")?;
            let rows: usize = stored_shape[..stored_shape.len() - 1].iter().product();
            let row_bytes =
                format.row_bytes(cols).with_context(|| format!("{name}: rows of {cols} do not fit {format:?} blocks"))?;
            let start = (reader.tensor_data_offset() + tensor.data_offset()) as usize;
            let end = start + rows * row_bytes;
            ensure!(end <= mmap.len(), "{name} runs past the end of {}", path.display());
            let shape = match layout {
                Layout::Same => stored_shape,
                Layout::Transposed | Layout::ConvTaps => stored_shape.into_iter().rev().collect(),
            };
            tensors.insert(name, TensorInfo { stored: Stored::Rows(format), shape, layout, file: 0, start, end });
        }
        ensure!(!tensors.is_empty(), "{} has none of the model's tensors", path.display());
        Ok(Self { files: vec![mmap], tensors, gguf: true })
    }

    /// Reads every byte of the weights once, so that they are in memory before the first request needs them.
    /// Returns how many bytes that was.
    pub fn read_through(&self) -> usize {
        use rayon::prelude::*;
        let touch = |chunk: &[u8]| chunk.iter().step_by(4096).fold(0u8, |sum, byte| sum.wrapping_add(*byte));
        for map in &self.files {
            std::hint::black_box(map.par_chunks(1 << 24).map(touch).reduce(|| 0, u8::wrapping_add));
        }
        self.files.iter().map(|map| map.len()).sum()
    }

    /// Whether the weights come from a GGUF conversion.
    pub fn is_gguf(&self) -> bool {
        self.gguf
    }

    /// The memory-mapped shards, whole.
    pub fn maps(&self) -> Vec<&[u8]> {
        self.files.iter().map(|map| &map[..]).collect()
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.tensors.iter().map(|(name, info)| (name.as_str(), info))
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).ok_or_else(|| anyhow!("tensor {name} is not in the checkpoint"))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// The raw little-endian bytes of a tensor.
    pub fn bytes(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let info = self.info(name)?;
        Ok((info, &self.files[info.file][info.start..info.end]))
    }

    /// A tensor decoded to f32, in the checkpoint's layout.
    pub fn dense(&self, name: &str) -> Result<Dense> {
        let (info, bytes) = self.bytes(name)?;
        let Stored::Rows(format) = info.stored else { bail!("{name} is {:?}, not a tensor of weights", info.stored) };
        let count: usize = info.shape.iter().product();
        let mut data = vec![0.0f32; count];
        // Stored rows run along the last stored dimension, which a changed layout puts first.
        let cols = match info.layout {
            Layout::Same => info.shape.last().copied().unwrap_or(1),
            Layout::Transposed | Layout::ConvTaps => info.shape[0],
        };
        let row_bytes = format.row_bytes(cols).with_context(|| format!("{name}: bad row length"))?;
        for (row, out) in bytes.chunks_exact(row_bytes).zip(data.chunks_exact_mut(cols)) {
            format.decode_row(row, out);
        }
        if info.layout != Layout::Same {
            ensure!(info.shape.len() == 2, "{name}: only matrices change layout");
            let (a, b) = (info.shape[0], info.shape[1]);
            let stored = std::mem::replace(&mut data, vec![0.0f32; count]);
            for i in 0..a {
                // Reversing the taps puts the current token's first, as the checkpoint has it.
                let from = if info.layout == Layout::ConvTaps { a - 1 - i } else { i };
                for j in 0..b {
                    data[i * b + j] = stored[j * a + from];
                }
            }
        }
        Ok(Dense { shape: info.shape.clone(), data })
    }

    /// A matrix of weights in its stored form, for tensors whose layout matches the checkpoint's.
    pub fn rows(&self, name: &str) -> Result<(Format, &[usize], &[u8])> {
        let (info, bytes) = self.bytes(name)?;
        match (info.stored, info.layout) {
            (Stored::Rows(format), Layout::Same) => Ok((format, &info.shape, bytes)),
            other => bail!("{name}: {other:?} cannot be read in place"),
        }
    }

    /// The routed experts of one decoder layer, still in their stored precision.
    pub fn experts(&self, layer: usize) -> Result<Experts<'_>> {
        let prefix = format!("model.llm.layers.{layer}.mlp.experts");
        let gate_up = self.expert_matrix(&format!("{prefix}.w13_weight"))?;
        let down = self.expert_matrix(&format!("{prefix}.w2_weight"))?;
        ensure!(gate_up.experts == down.experts, "layer {layer}: expert counts differ");
        ensure!(gate_up.rows == 2 * down.cols && gate_up.cols == down.rows, "layer {layer}: expert shapes disagree");
        Ok(Experts { gate_up, down, interleaved: !self.gguf })
    }

    fn expert_matrix(&self, name: &str) -> Result<ExpertMatrix<'_>> {
        let (info, values) = self.bytes(name)?;
        if let Stored::Rows(format) = info.stored {
            ensure!(info.shape.len() == 3, "{name}: expected one matrix per expert");
            let (experts, rows, cols) = (info.shape[0], info.shape[1], info.shape[2]);
            format.row_bytes(cols).with_context(|| format!("{name}: bad row length"))?;
            let (values_per_byte, block) = (0, format.block().0);
            return Ok(ExpertMatrix {
                format: ExpertFormat::Ggml(format),
                experts,
                rows,
                cols,
                values,
                scales: &[],
                values_per_byte,
                block,
            });
        }
        let (scale_info, scales) = self.bytes(&format!("{name}_scale"))?;
        ensure!(info.shape.len() == 3 && scale_info.shape.len() == 3, "{name}: expected 3-D weights and scales");
        let (experts, rows) = (info.shape[0], info.shape[1]);
        ensure!(scale_info.shape[..2] == [experts, rows], "{name}: scales are not per row");

        let format = match (info.stored, scale_info.stored) {
            (Stored::U8, Stored::E4m3) => {
                let amax = self.dense(&format!("{name}_amax"))?;
                let denominator = self.dense(&format!("{name}_outer_scale_denominator"))?;
                ensure!(amax.shape == [experts] && denominator.shape == [1], "{name}: unexpected outer scale shapes");
                let denominator = denominator.data[0];
                ensure!(denominator.is_finite() && denominator > 0.0, "{name}: bad outer scale denominator");
                ExpertFormat::Nvfp4 { outer_scales: amax.data.iter().map(|amax| amax / denominator).collect() }
            }
            (Stored::E4m3, Stored::U8) => ExpertFormat::Mxfp8,
            other => bail!("{name}: unsupported weight/scale dtypes {other:?}"),
        };
        let (values_per_byte, block) = match format {
            ExpertFormat::Nvfp4 { .. } => (2, quant::NVFP4_BLOCK),
            ExpertFormat::Mxfp8 => (1, quant::MXFP8_BLOCK),
            ExpertFormat::Ggml(_) => unreachable!("handled above"),
        };
        let cols = info.shape[2] * values_per_byte;
        ensure!(scale_info.shape[2] * block == cols, "{name}: scales do not cover blocks of {block}");
        Ok(ExpertMatrix { format, experts, rows, cols, values, scales, values_per_byte, block })
    }
}

/// A dense tensor held as f32, row-major.
#[derive(Debug, Clone)]
pub struct Dense {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone)]
pub enum ExpertFormat {
    /// One `amax / denominator` scale per expert, on top of the per-block scales.
    Nvfp4 { outer_scales: Vec<f32> },
    Mxfp8,
    /// A GGUF block format, each row standing on its own.
    Ggml(Format),
}

/// A stack of per-expert matrices `[experts, rows, cols]` in block floating point.
pub struct ExpertMatrix<'a> {
    pub format: ExpertFormat,
    pub experts: usize,
    pub rows: usize,
    pub cols: usize,
    values: &'a [u8],
    scales: &'a [u8],
    values_per_byte: usize,
    block: usize,
}

impl<'a> ExpertMatrix<'a> {
    /// Decodes row `row` of expert `expert` into `out` (`cols` values).
    pub fn decode_row(&self, expert: usize, row: usize, out: &mut [f32]) {
        let (values, scales) = self.row(expert, row);
        match &self.format {
            ExpertFormat::Nvfp4 { outer_scales } => quant::decode_nvfp4_row(values, scales, outer_scales[expert], out),
            ExpertFormat::Mxfp8 => quant::decode_mxfp8_row(values, scales, out),
            ExpertFormat::Ggml(format) => format.decode_row(values, out),
        }
    }

    /// The block format of a GGUF expert matrix; `None` for the checkpoint's own formats.
    pub fn ggml(&self) -> Option<Format> {
        match self.format {
            ExpertFormat::Ggml(format) => Some(format),
            _ => None,
        }
    }

    /// The stored bytes of one row of a GGUF expert matrix.
    pub fn ggml_row(&self, expert: usize, row: usize) -> (Format, &'a [u8]) {
        let ExpertFormat::Ggml(format) = self.format else { panic!("not a GGUF expert matrix") };
        let row_bytes = format.row_bytes(self.cols).expect("checked when opened");
        (format, &self.values[(expert * self.rows + row) * row_bytes..][..row_bytes])
    }

    /// `row . x` for row `row` of expert `expert`, computed from the stored form without decoding it.
    pub fn dot_row(&self, expert: usize, row: usize, x: &[f32]) -> f32 {
        let (values, scales) = self.row(expert, row);
        match &self.format {
            ExpertFormat::Nvfp4 { outer_scales } => quant::dot_nvfp4_row(values, scales, outer_scales[expert], x),
            ExpertFormat::Mxfp8 => quant::dot_mxfp8_row(values, scales, x),
            ExpertFormat::Ggml(format) => format.dot_row(values, x, &mut Vec::new()),
        }
    }

    /// For NVFP4: the packed values and block scales of every expert, and the per-expert outer scales.
    pub fn nvfp4_parts(&self) -> Option<(&[u8], &[u8], &[f32])> {
        match &self.format {
            ExpertFormat::Nvfp4 { outer_scales } => Some((self.values, self.scales, outer_scales)),
            ExpertFormat::Mxfp8 | ExpertFormat::Ggml(_) => None,
        }
    }

    /// The stored values and block scales of every expert, whatever the format.
    pub fn raw_parts(&self) -> (&'a [u8], &'a [u8]) {
        (self.values, self.scales)
    }

    /// The scale applied to an expert's whole matrix on top of its block scales: 1 for MXFP8, which has none.
    pub fn outer_scale(&self, expert: usize) -> f32 {
        match &self.format {
            ExpertFormat::Nvfp4 { outer_scales } => outer_scales[expert],
            ExpertFormat::Mxfp8 | ExpertFormat::Ggml(_) => 1.0,
        }
    }

    pub fn is_nvfp4(&self) -> bool {
        matches!(self.format, ExpertFormat::Nvfp4 { .. })
    }

    /// `row . x` for an NVFP4 row and each of up to [`quant::NVFP4_MAX_BATCH`] NVFP4 activations, in integer
    /// arithmetic within each block.
    pub fn dot_row_quantized_many(&self, expert: usize, row: usize, xs: &[&quant::Nvfp4Activation], out: &mut [f32]) {
        let (values, scales) = self.row(expert, row);
        match &self.format {
            ExpertFormat::Nvfp4 { outer_scales } => {
                quant::dot_nvfp4_row_quantized_many(values, scales, outer_scales[expert], xs, out)
            }
            ExpertFormat::Mxfp8 | ExpertFormat::Ggml(_) => panic!("only NVFP4 experts take NVFP4 activations"),
        }
    }

    fn row(&self, expert: usize, row: usize) -> (&[u8], &[u8]) {
        assert!(expert < self.experts && row < self.rows);
        if let ExpertFormat::Ggml(_) = self.format {
            return (self.ggml_row(expert, row).1, &[]);
        }
        let index = expert * self.rows + row;
        let row_bytes = self.cols / self.values_per_byte;
        let row_scales = self.cols / self.block;
        (&self.values[index * row_bytes..(index + 1) * row_bytes], &self.scales[index * row_scales..(index + 1) * row_scales])
    }
}

/// The experts of one layer: `gate_up` is `[experts, 2 * intermediate, hidden]`, `down` is `[experts, hidden, intermediate]`.
pub struct Experts<'a> {
    pub gate_up: ExpertMatrix<'a>,
    pub down: ExpertMatrix<'a>,
    /// The checkpoint interleaves the two halves of `gate_up`: row `2i` is gate unit `i`, row `2i + 1` is up unit
    /// `i`. Otherwise all the gate rows come first.
    pub interleaved: bool,
}

impl Experts<'_> {
    pub fn intermediate(&self) -> usize {
        self.down.cols
    }

    /// The rows of `gate_up` holding gate unit `unit` and up unit `unit`.
    pub fn gate_up_rows(&self, unit: usize) -> (usize, usize) {
        match self.interleaved {
            true => (2 * unit, 2 * unit + 1),
            false => (unit, self.intermediate() + unit),
        }
    }

    pub fn decode_gate_row(&self, expert: usize, unit: usize, out: &mut [f32]) {
        self.gate_up.decode_row(expert, self.gate_up_rows(unit).0, out);
    }

    pub fn decode_up_row(&self, expert: usize, unit: usize, out: &mut [f32]) {
        self.gate_up.decode_row(expert, self.gate_up_rows(unit).1, out);
    }
}

/// The checkpoint's name for a GGUF tensor, and how its layout differs. `full` says whether the tensor belongs to a
/// layer that attends over the whole context, whose key and value convolutions are stored differently.
fn checkpoint_name(gguf: &str, full: bool) -> Option<(String, Layout)> {
    let root = "model.llm";
    let same = |name: String| Some((name, Layout::Same));
    match gguf {
        "token_embd.weight" => return same(format!("{root}.embed.weight")),
        "output.weight" => return same(format!("{root}.unembed.weight")),
        "token_embd_norm.weight" => return same(format!("{root}.embed_norm.weight")),
        "output_norm.weight" => return same(format!("{root}.norm.weight")),
        "output_res_query.weight" => return same(format!("{root}.attn_res.output_query")),
        _ => {}
    }
    let (layer, tensor) = gguf.strip_prefix("blk.")?.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let p = format!("{root}.layers.{layer}");
    let controller = |tensor: &str| -> Option<String> {
        let (which, part) = tensor.strip_prefix("loop_")?.strip_suffix(".weight")?.split_once("_ctrl_")?;
        let which = match which {
            "q" => "attn",
            "ffn" => "mlp",
            other => other,
        };
        Some(format!("{p}.controllers.{which}.{part}_weight"))
    };
    let name = match tensor {
        "attn_q.weight" => format!("{p}.attn.wq_du.weight"),
        "attn_k.weight" => format!("{p}.attn.wk_dv.weight"),
        "attn_v.weight" => format!("{p}.attn.wv_dv.weight"),
        "attn_rel.weight" => format!("{p}.attn.wr_du.weight"),
        "attn_output.weight" => format!("{p}.attn.wo_ud.weight"),
        "attn_gate.weight" => format!("{p}.attn.gate_attn.weight"),
        "attn_sink_proj.weight" => format!("{p}.attn.attn_sink_proj.weight"),
        "attn_q_clip.weight" => format!("{p}.attn.q_l2_clip.scale"),
        "attn_rel_clip.weight" => format!("{p}.attn.rel_l2_clip.scale"),
        "attn_rel_bias.weight" => return Some((format!("{p}.attn.rel_logits_proj.proj"), Layout::Transposed)),
        "attn_k_conv.weight" | "attn_v_conv.weight" => {
            let which = &tensor[5..6];
            return Some(match full {
                true => (format!("{p}.attn.{which}_sconv.static_w"), Layout::ConvTaps),
                false => (format!("{p}.attn.{which}_sconv.weight"), Layout::Same),
            });
        }
        "attn_k_conv_proj.weight" => format!("{p}.attn.k_sconv.weight_proj"),
        "attn_v_conv_proj.weight" => format!("{p}.attn.v_sconv.weight_proj"),
        "attn_norm.weight" => format!("{p}.attn_norm.weight"),
        "attn_q_inp_norm.weight" => format!("{p}.q_input_norm.weight"),
        "attn_k_inp_norm.weight" => format!("{p}.k_input_norm.weight"),
        "attn_v_inp_norm.weight" => format!("{p}.v_input_norm.weight"),
        "attn_post_conv.weight" => format!("{p}.attn_sconv.weight"),
        "ffn_post_conv.weight" => format!("{p}.mlp_sconv.weight"),
        "ffn_norm.weight" => format!("{p}.mlp_norm.weight"),
        "ffn_gate_inp.weight" => format!("{p}.mlp.gate.weight"),
        "exp_probs_b.bias" => format!("{p}.mlp.gate.bias"),
        "ffn_gate_up_exps.weight" => format!("{p}.mlp.experts.w13_weight"),
        "ffn_down_exps.weight" => format!("{p}.mlp.experts.w2_weight"),
        "ffn_res_query.weight" => format!("{root}.attn_res.mlp_queries.{layer}"),
        // The checkpoint files a layer's attention queries under the layer before it.
        "attn_res_query.weight" => format!("{root}.attn_res.attn_queries.{}", layer.checked_sub(1)?),
        // The query, key, value and MLP message queries of a looped layer, which the checkpoint splits across two
        // tensors; this name exists only for GGUF files.
        "loop_msg_query.weight" => format!("{p}.message_queries"),
        other => controller(other)?,
    };
    Some((name, Layout::Same))
}

/// The model used when none is named: a 4-bit conversion, which is what the GPU generates from.
pub const DEFAULT_MODEL: &str = "lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M";
/// The published checkpoint. Conversions borrow its config, tokenizer and chat template.
pub const BASE_MODEL: &str = "thinkingmachines/Inkling-2-Mini";

/// The published checkpoint among the downloaded models, for the tests and tools that need it and no other.
pub fn resolve_checkpoint() -> Result<PathBuf> {
    resolve_model(Some(Path::new(BASE_MODEL)))
}

/// Finds the model's weights: a checkpoint directory, or a GGUF file.
///
/// `explicit` (or else `$RUSTLING_MODEL`) is one of
/// - a checkpoint directory or a `.gguf` file;
/// - a Hugging Face model name such as `thinkingmachines/TML-d24-varloop-research`;
/// - the name of a GGUF repository and the quantisation wanted from it, such as
///   `unsloth/Inkling-2-Mini-GGUF:Q4_K_M`.
///
/// Names are looked up among the models already downloaded: the Hugging Face cache, then LM Studio's models.
/// With nothing given, [`DEFAULT_MODEL`]. Nothing is downloaded here; see [`super::hub::ensure_model`].
pub fn resolve_model(explicit: Option<&Path>) -> Result<PathBuf> {
    let named = explicit.map(Path::to_path_buf).or_else(|| std::env::var_os("RUSTLING_MODEL").map(PathBuf::from));
    let name = match named {
        Some(path) if path.is_dir() || is_gguf(&path) && path.is_file() => return Ok(path),
        Some(path) => path.to_string_lossy().into_owned(),
        None => DEFAULT_MODEL.to_owned(),
    };
    let (repo, quantisation) = match name.split_once(':') {
        Some((repo, quantisation)) => (repo, Some(quantisation)),
        None => (name.as_str(), None),
    };
    ensure!(
        repo.split('/').count() == 2,
        "{name} is not a directory, a .gguf file, or a model name like {BASE_MODEL} or {DEFAULT_MODEL}"
    );
    let home = PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?);
    let hub = match std::env::var_os("HF_HUB_CACHE") {
        Some(path) => PathBuf::from(path),
        None => match std::env::var_os("HF_HOME") {
            Some(hf_home) => PathBuf::from(hf_home).join("hub"),
            None => home.join(".cache/huggingface/hub"),
        },
    };
    let snapshots = hub.join(format!("models--{}/snapshots", repo.replace('/', "--")));
    let snapshot = std::fs::read_dir(&snapshots)
        .ok()
        .and_then(|entries| entries.filter_map(|e| e.ok()).max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok()))
        .map(|entry| entry.path());
    let places: Vec<PathBuf> = snapshot.into_iter().chain([home.join(".lmstudio/models").join(repo)]).collect();

    let mut seen = Vec::new();
    for place in places.iter().filter(|place| place.is_dir()) {
        let has = |wanted: fn(&Path) -> bool| -> Vec<PathBuf> {
            let files = std::fs::read_dir(place).into_iter().flatten().filter_map(|e| e.ok()).map(|e| e.path());
            let mut files: Vec<PathBuf> = files.filter(|path| wanted(path)).collect();
            files.sort();
            files
        };
        if quantisation.is_none() && !has(|path| path.extension().is_some_and(|e| e == "safetensors")).is_empty() {
            return Ok(place.clone());
        }
        // The vision projector and the draft model sit beside the main model's files.
        let ggufs = has(|path| {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_ascii_lowercase();
            is_gguf(path) && !name.starts_with("mmproj") && !name.starts_with("dspark")
        });
        let matching: Vec<&PathBuf> = ggufs
            .iter()
            .filter(|path| {
                let stem = path.file_stem().and_then(|n| n.to_str()).unwrap_or_default().to_ascii_lowercase();
                quantisation.is_none_or(|q| stem.ends_with(&q.to_ascii_lowercase()))
            })
            .collect();
        if let [only] = matching[..] {
            return Ok(only.clone());
        }
        seen.extend(ggufs.iter().filter_map(|path| Some(path.file_name()?.to_string_lossy().into_owned())));
    }
    match seen.is_empty() {
        true => bail!("{repo} is not among the downloaded models; fetch it with `rustling pull {name}`"),
        false => bail!("{name} does not pick out one file; {repo} has: {}", seen.join(", ")),
    }
}

fn is_gguf(path: &Path) -> bool {
    path.extension().is_some_and(|extension| extension == "gguf")
}

/// Where the files that go with the weights are: the config, the tokenizer, the chat template and the draft model.
/// A checkpoint directory holds its own. A GGUF file borrows them from the published checkpoint it was converted
/// from, `$RUSTLING_BASE_MODEL` or else Inkling 2 Mini, which must be downloaded too (the weights are not needed).
pub fn companion_dir(model: &Path) -> Result<PathBuf> {
    if !is_gguf(model) {
        return Ok(model.to_path_buf());
    }
    let base = std::env::var_os("RUSTLING_BASE_MODEL").map(PathBuf::from).unwrap_or_else(|| BASE_MODEL.into());
    let dir = resolve_model(Some(&base)).with_context(|| {
        format!(
            "a GGUF file needs the config and tokenizer of the checkpoint it came from, {}; `rustling pull` \
             fetches them",
            base.display()
        )
    })?;
    ensure!(dir.is_dir(), "{} is not a checkpoint directory", dir.display());
    Ok(dir)
}
