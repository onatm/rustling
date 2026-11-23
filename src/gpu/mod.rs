//! Metal backend for the matrix products that dominate prompt processing. See `shaders.metal` for the kernels
//! and why their results equal the CPU's bit for bit.
//!
//! Weights are never copied: the checkpoint's memory map is handed to Metal as it is. Everything else (inputs,
//! outputs) lives in small shared buffers made per call.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, ensure};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResource,
    MTLResourceOptions, MTLSize,
};

use crate::weights::ggml::Format;
use crate::weights::quant::{E4M3, Nvfp4Activation};

mod exact;
mod fast;
mod steps;

pub(crate) use steps::*;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// The most bytes of unpacked expert weights kept in one buffer.
const WIDENED_BYTES: usize = 1 << 30;

/// How many rows the left operand of a fast product has where it must not fail: every expert product, and the
/// plain products when they are redone. Apple's matrix product has returned NaN for half the outputs of products
/// of some other sizes, repeatably for a given sequence of work, and none of this one. Longer operands go
/// through in runs of this many rows, the last padded with zeros. With runs of 512 a few expert products of a
/// 3,774-token prompt still came back with NaN; with 256 none did.
const PRODUCT_RUN: usize = 256;

/// Apple silicon pages; `newBufferWithBytesNoCopy` needs whole pages.
const PAGE: usize = 16384;

pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    dense: Pipeline,
    experts: Pipeline,
    widen_bf16: Pipeline,
    widen_nvfp4: Pipeline,
    widen_mxfp8: Pipeline,
    /// Unpacking kernels for a GGUF file's block formats: Q4_K, Q5_0, Q6_K, Q8_0.
    widen_ggml: [Pipeline; 4],
    /// Direct products for short expert tails, in the same format order as `widen_ggml`.
    tail_ggml: [Pipeline; 4],
    /// Turns a long batch's attention scores into weights.
    prompt_weights: Pipeline,
    /// The steps between a long batch's two expert products: gather, activation, weighted sum.
    expert_steps: [Pipeline; 3],
    /// Held while a message merge's gates use their scratch buffers.
    gates: Mutex<()>,
    e4m3: Buffer,
    /// Working buffers of the fast path, reused from call to call and grown as needed; `(bytes, buffer)` by slot.
    scratch: Mutex<Vec<Option<(usize, Buffer)>>>,
    /// Whether large batches may take the fast path, whose results match the exact kernels only to rounding.
    fast: bool,
    /// Memory maps that may be wrapped without copying, as `(address, length)`.
    mapped: Vec<(usize, usize)>,
    /// No-copy buffers over page-aligned spans of the maps, keyed by `(address, length)`.
    wrapped: Mutex<HashMap<(usize, usize), Buffer>>,
}

// SAFETY: Metal devices, queues, pipeline states and buffers are documented as safe to use from any thread.
// Command buffers and encoders are not, and never leave the function that creates them.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

/// A GGUF layer's experts applied to a long batch, for [`Gpu::experts_whole`].
pub struct WholeExperts<'a> {
    /// The stored bytes and block format of the two stacks: `experts` matrices of `2 * units x hidden`, with
    /// each expert's gate rows before its up rows, and of `hidden x units`.
    pub gate_up: (&'a [u8], Format),
    pub down: (&'a [u8], Format),
    pub experts: usize,
    pub hidden: usize,
    pub units: usize,
    /// Experts to a token.
    pub chosen: usize,
    /// `tokens x hidden`: the tokens' inputs.
    pub x: &'a [f32],
    /// `chosen` per token: the expert and the weight of its output.
    pub routes: &'a [(u32, f32)],
    pub gate_cap: f32,
    pub up_cap: f32,
}

/// One key head's part of an attention job: its query heads' rows, and the keys and values they attend over.
pub struct AttendHead<'a> {
    /// `rows x dim`: for each token, the query heads that share this key head.
    pub queries: &'a [f32],
    /// At least `keys x dim` each, oldest first.
    pub keys: &'a [f32],
    pub values: &'a [f32],
    /// `rows x d_rel`: each row's relative state.
    pub rel: &'a [f32],
    /// One sink per row.
    pub sinks: &'a [f32],
}

/// What the heads of an attention job share.
pub struct AttendShape<'a> {
    pub rows: usize,
    pub keys: usize,
    pub dim: usize,
    pub d_rel: usize,
    pub extent: usize,
    /// `d_rel x extent`: the bias each relative dimension gives at each distance.
    pub profiles: &'a [f32],
    /// Rows to a token.
    pub share: usize,
    /// The position of the first row's token, and of the first key.
    pub first: usize,
    pub lowest: usize,
    pub local: bool,
    pub window: usize,
    pub scaling: f32,
}

/// One dense product: `rows x cols` weights, bfloat16 or in a GGUF block format, applied to each `cols`-wide
/// vector in `xs`. Only the fast path takes block formats.
pub struct DenseJob<'a> {
    pub format: Format,
    pub weights: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub xs: &'a [f32],
}

/// A stack of NVFP4 expert matrices and the (expert, activation) products wanted from it.
pub struct ExpertJob<'a> {
    /// Packed values for every expert, `[experts, rows, cols / 2]`.
    pub values: &'a [u8],
    /// Block scales for every expert, `[experts, rows, cols / 16]`.
    pub scales: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub activations: &'a [&'a Nvfp4Activation],
    /// `(expert, index into activations, the expert's outer scale)`.
    pub pairs: &'a [(usize, usize, f32)],
}

/// How a stack of expert matrices is stored.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stack {
    Nvfp4,
    Mxfp8,
    Ggml(Format),
}

/// Vectors handled by one GPU thread; `SHARE` in the shaders.
const SHARE: usize = 4;

/// `Group` in the shaders: up to `SHARE` activations for one expert.
#[repr(C)]
#[derive(Clone, Copy)]
struct Group {
    expert: u32,
    count: u32,
    outer_scale: f32,
    unused: u32,
    activations: [u32; SHARE],
}

impl Gpu {
    /// Sets up the default device. `maps` are the memory-mapped files whose contents will be passed as weights;
    /// they must stay mapped for as long as the `Gpu` is used.
    pub fn new(maps: &[&[u8]], fast: bool) -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice().context("no Metal device")?;
        let queue = device.newCommandQueue().context("creating a Metal command queue")?;

        let options = MTLCompileOptions::new();
        // Safe math keeps IEEE semantics, which the bit-for-bit match with the CPU kernels depends on.
        options.setMathMode(MTLMathMode::Safe);
        let source = NSString::from_str(include_str!("shaders/exact.metal"));
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| anyhow!("compiling shaders: {}", e.localizedDescription()))?;
        let pipeline = |name: &str| -> Result<Pipeline> {
            let function =
                library.newFunctionWithName(&NSString::from_str(name)).with_context(|| format!("no kernel {name}"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| anyhow!("building kernel {name}: {}", e.localizedDescription()))
        };
        let (dense, experts) = (pipeline("dense")?, pipeline("experts")?);
        let (widen_bf16, widen_nvfp4, widen_mxfp8) =
            (pipeline("widen_bf16")?, pipeline("widen_nvfp4")?, pipeline("widen_mxfp8")?);
        let source = NSString::from_str(include_str!("shaders/gguf.metal"));
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| anyhow!("compiling the GGUF shaders: {}", e.localizedDescription()))?;
        let pipeline = |name: &str| -> Result<Pipeline> {
            let function =
                library.newFunctionWithName(&NSString::from_str(name)).with_context(|| format!("no kernel {name}"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| anyhow!("building kernel {name}: {}", e.localizedDescription()))
        };
        let widen_ggml = [pipeline("widen_q4k")?, pipeline("widen_q5_0")?, pipeline("widen_q6k")?, pipeline("widen_q8_0")?];
        let tail_ggml = [pipeline("tail_q4k")?, pipeline("tail_q5_0")?, pipeline("tail_q6k")?, pipeline("tail_q8_0")?];
        let source = NSString::from_str(include_str!("shaders/attend.metal"));
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .map_err(|e| anyhow!("compiling the attention shaders: {}", e.localizedDescription()))?;
        let pipeline = |name: &str| -> Result<Pipeline> {
            let function =
                library.newFunctionWithName(&NSString::from_str(name)).with_context(|| format!("no kernel {name}"))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| anyhow!("building kernel {name}: {}", e.localizedDescription()))
        };
        let prompt_weights = pipeline("prompt_weights")?;
        let expert_steps = [pipeline("gather_rows")?, pipeline("expert_activation")?, pipeline("combine_routes")?];

        let mut gpu = Self {
            e4m3: copied(&device, &E4M3[..])?,
            device,
            queue,
            dense,
            experts,
            widen_bf16,
            widen_nvfp4,
            widen_mxfp8,
            widen_ggml,
            tail_ggml,
            prompt_weights,
            expert_steps,
            gates: Mutex::new(()),
            scratch: Mutex::new(Vec::new()),
            fast,
            mapped: Vec::new(),
            wrapped: Mutex::new(HashMap::new()),
        };
        gpu.mapped = maps.iter().map(|map| (map.as_ptr() as usize, map.len())).collect();
        Ok(gpu)
    }

    /// A buffer holding `bytes` and the offset of `bytes` within it. Bytes inside a registered memory map are
    /// wrapped in place; anything else is copied.
    fn weights(&self, bytes: &[u8]) -> Result<(Buffer, usize)> {
        let (address, len) = (bytes.as_ptr() as usize, bytes.len());
        let inside = |&(base, map_len): &(usize, usize)| address >= base && address + len <= base + map_len;
        if !self.mapped.iter().any(inside) {
            return Ok((copied(&self.device, bytes)?, 0));
        }
        // Whole pages around the bytes. The span stays inside the mapping, which itself covers whole pages.
        let start = address / PAGE * PAGE;
        let span = (address + len).div_ceil(PAGE) * PAGE - start;
        let mut wrapped = self.wrapped.lock().expect("buffer cache lock");
        let buffer = match wrapped.get(&(start, span)) {
            Some(buffer) => buffer.clone(),
            None => {
                ensure!(span <= self.device.maxBufferLength(), "a {span}-byte tensor exceeds the Metal buffer limit");
                let pointer = NonNull::new(start as *mut c_void).expect("mapped memory is not null");
                // SAFETY: the span is page-aligned, lies within a live memory map (see `new`), and is only read.
                let buffer = unsafe {
                    self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                        pointer,
                        span,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
                }
                .context("Metal refused to wrap the checkpoint's memory map")?;
                wrapped.insert((start, span), buffer.clone());
                buffer
            }
        };
        Ok((buffer, address - start))
    }

    /// Whether large batches may take the fast, inexact path.
    pub fn fast(&self) -> bool {
        self.fast
    }

    /// A working buffer of at least `bytes`, kept under `slot` for reuse. Its contents are whatever the last use
    /// left there.
    /// Lets go of the working buffers of the long-batch paths, which are gigabytes after a long prompt. They are
    /// made again when next needed.
    pub fn release_scratch(&self) {
        self.scratch.lock().expect("scratch lock").clear();
    }

    fn scratch(&self, slot: usize, bytes: usize) -> Result<Buffer> {
        let mut scratch = self.scratch.lock().expect("scratch lock");
        if scratch.len() <= slot {
            scratch.resize(slot + 1, None);
        }
        match &scratch[slot] {
            Some((size, buffer)) if *size >= bytes => Ok(buffer.clone()),
            _ => {
                // Drop the old one first, so the two are never held together.
                scratch[slot] = None;
                let buffer = empty(&self.device, bytes)?;
                scratch[slot] = Some((bytes, buffer.clone()));
                Ok(buffer)
            }
        }
    }

}

fn finish(commands: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
    match commands.error() {
        Some(error) => Err(anyhow!("Metal command buffer failed: {}", error.localizedDescription())),
        None => Ok(()),
    }
}

/// A shared buffer holding a copy of `data`.
fn copied<T: Copy>(device: &ProtocolObject<dyn MTLDevice>, data: &[T]) -> Result<Buffer> {
    let bytes = size_of_val(data).max(1);
    let pointer = NonNull::new(data.as_ptr() as *mut c_void).unwrap_or(NonNull::dangling());
    // SAFETY: Metal copies `size_of_val(data)` bytes out of `data` during the call; a zero-length slice is given
    // one byte of length only when the pointer is still valid to read, so guard that case.
    if data.is_empty() {
        return empty(device, 1);
    }
    unsafe { device.newBufferWithBytes_length_options(pointer, bytes, MTLResourceOptions::StorageModeShared) }
        .context("allocating a Metal buffer")
}

fn empty(device: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Result<Buffer> {
    device
        .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
        .context("allocating a Metal buffer")
}

fn read(buffer: &Buffer, len: usize) -> Vec<f32> {
    // SAFETY: the buffer is shared-mode and at least `len` floats long, Metal buffers are page-aligned, and the
    // command buffer that wrote it has completed.
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const f32, len) }.to_vec()
}

#[cfg(test)]
mod tests;
