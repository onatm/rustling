//! Memory-mapped access to the published checkpoint, using the checkpoint's own tensor names.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail, ensure};
use half::bf16;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::quant;

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
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
}

impl Checkpoint {
    /// Opens every `model-*.safetensors` shard in `model_dir`. Nothing is read until a tensor is touched.
    pub fn open(model_dir: &Path) -> Result<Self> {
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
                let info = TensorInfo {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    file,
                    start: data_start + start,
                    end: data_start + end,
                };
                ensure!(info.end <= mmap.len(), "{name} runs past the end of {}", path.display());
                ensure!(tensors.insert(name.clone(), info).is_none(), "{name} appears in two shards");
            }
            files.push(mmap);
        }
        Ok(Self { files, tensors })
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

    /// A dense tensor (F32 or BF16) widened to f32.
    pub fn dense(&self, name: &str) -> Result<Dense> {
        let (info, bytes) = self.bytes(name)?;
        let data = match info.dtype {
            Dtype::F32 => bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
            Dtype::BF16 => bytes.as_chunks::<2>().0.iter().map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
            other => bail!("{name} is {other:?}, not a dense float tensor"),
        };
        Ok(Dense { shape: info.shape.clone(), data })
    }

    /// The routed experts of one decoder layer, still in their stored precision.
    pub fn experts(&self, layer: usize) -> Result<Experts<'_>> {
        let prefix = format!("model.llm.layers.{layer}.mlp.experts");
        let gate_up = self.expert_matrix(&format!("{prefix}.w13_weight"))?;
        let down = self.expert_matrix(&format!("{prefix}.w2_weight"))?;
        ensure!(gate_up.experts == down.experts, "layer {layer}: expert counts differ");
        ensure!(gate_up.rows == 2 * down.cols && gate_up.cols == down.rows, "layer {layer}: expert shapes disagree");
        Ok(Experts { gate_up, down })
    }

    fn expert_matrix(&self, name: &str) -> Result<ExpertMatrix<'_>> {
        let (info, values) = self.bytes(name)?;
        let (scale_info, scales) = self.bytes(&format!("{name}_scale"))?;
        ensure!(info.shape.len() == 3 && scale_info.shape.len() == 3, "{name}: expected 3-D weights and scales");
        let (experts, rows) = (info.shape[0], info.shape[1]);
        ensure!(scale_info.shape[..2] == [experts, rows], "{name}: scales are not per row");

        let format = match (info.dtype, scale_info.dtype) {
            (Dtype::U8, Dtype::F8_E4M3) => {
                let amax = self.dense(&format!("{name}_amax"))?;
                let denominator = self.dense(&format!("{name}_outer_scale_denominator"))?;
                ensure!(amax.shape == [experts] && denominator.shape == [1], "{name}: unexpected outer scale shapes");
                let denominator = denominator.data[0];
                ensure!(denominator.is_finite() && denominator > 0.0, "{name}: bad outer scale denominator");
                ExpertFormat::Nvfp4 { outer_scales: amax.data.iter().map(|amax| amax / denominator).collect() }
            }
            (Dtype::F8_E4M3, Dtype::U8) => ExpertFormat::Mxfp8,
            other => bail!("{name}: unsupported weight/scale dtypes {other:?}"),
        };
        let (values_per_byte, block) = match format {
            ExpertFormat::Nvfp4 { .. } => (2, quant::NVFP4_BLOCK),
            ExpertFormat::Mxfp8 => (1, quant::MXFP8_BLOCK),
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

impl ExpertMatrix<'_> {
    /// Decodes row `row` of expert `expert` into `out` (`cols` values).
    pub fn decode_row(&self, expert: usize, row: usize, out: &mut [f32]) {
        let (values, scales) = self.row(expert, row);
        match &self.format {
            ExpertFormat::Nvfp4 { outer_scales } => quant::decode_nvfp4_row(values, scales, outer_scales[expert], out),
            ExpertFormat::Mxfp8 => quant::decode_mxfp8_row(values, scales, out),
        }
    }

    /// `row . x` for row `row` of expert `expert`, computed from the stored form without decoding it.
    pub fn dot_row(&self, expert: usize, row: usize, x: &[f32]) -> f32 {
        let (values, scales) = self.row(expert, row);
        match &self.format {
            ExpertFormat::Nvfp4 { outer_scales } => quant::dot_nvfp4_row(values, scales, outer_scales[expert], x),
            ExpertFormat::Mxfp8 => quant::dot_mxfp8_row(values, scales, x),
        }
    }

    /// For NVFP4: the packed values and block scales of every expert, and the per-expert outer scales.
    pub fn nvfp4_parts(&self) -> Option<(&[u8], &[u8], &[f32])> {
        match &self.format {
            ExpertFormat::Nvfp4 { outer_scales } => Some((self.values, self.scales, outer_scales)),
            ExpertFormat::Mxfp8 => None,
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
            ExpertFormat::Mxfp8 => panic!("MXFP8 experts take dequantised activations"),
        }
    }

    fn row(&self, expert: usize, row: usize) -> (&[u8], &[u8]) {
        assert!(expert < self.experts && row < self.rows);
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
}

impl Experts<'_> {
    pub fn intermediate(&self) -> usize {
        self.down.cols
    }

    /// The checkpoint interleaves the two halves of `w13`: row `2i` is gate unit `i`, row `2i + 1` is up unit `i`.
    pub fn decode_gate_row(&self, expert: usize, unit: usize, out: &mut [f32]) {
        self.gate_up.decode_row(expert, 2 * unit, out);
    }

    pub fn decode_up_row(&self, expert: usize, unit: usize, out: &mut [f32]) {
        self.gate_up.decode_row(expert, 2 * unit + 1, out);
    }
}

/// The model whose snapshot is used when none is named.
const DEFAULT_MODEL: &str = "thinkingmachines/Inkling-2-Mini";

/// Finds the model directory. `explicit` (or else `$RUSTLING_MODEL`) is a directory, or a Hugging Face model name
/// such as `thinkingmachines/TML-d24-varloop-research` to look up in the local Hugging Face cache. With neither,
/// Inkling 2 Mini from the cache.
pub fn resolve_model_dir(explicit: Option<&Path>) -> Result<PathBuf> {
    let named = explicit.map(Path::to_path_buf).or_else(|| std::env::var_os("RUSTLING_MODEL").map(PathBuf::from));
    let name = match named {
        Some(path) if path.is_dir() => return Ok(path),
        Some(path) => path.to_string_lossy().into_owned(),
        None => DEFAULT_MODEL.to_owned(),
    };
    ensure!(name.split('/').count() == 2, "{name} is neither a directory nor a model name like {DEFAULT_MODEL}");
    let hub = match std::env::var_os("HF_HUB_CACHE") {
        Some(path) => PathBuf::from(path),
        None => match std::env::var_os("HF_HOME") {
            Some(home) => PathBuf::from(home).join("hub"),
            None => PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?).join(".cache/huggingface/hub"),
        },
    };
    let snapshots = hub.join(format!("models--{}/snapshots", name.replace('/', "--")));
    let newest = std::fs::read_dir(&snapshots)
        .with_context(|| format!("{name} is not in the Hugging Face cache; fetch it with `hf download {name}`"))?
        .filter_map(|entry| entry.ok())
        .max_by_key(|entry| entry.metadata().and_then(|m| m.modified()).ok())
        .with_context(|| format!("no snapshots in {}", snapshots.display()))?;
    Ok(newest.path())
}
