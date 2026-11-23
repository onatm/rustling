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

use crate::ggml::Format;
use crate::quant::{E4M3, Nvfp4Activation};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// The most bytes of unpacked expert weights kept in one buffer.
const WIDENED_BYTES: usize = 1 << 30;

/// The most input vectors given to one expert product on the fast path. With runs of 512 a few products of a
/// 3,774-token prompt still came back with NaN; with 256 none did.
const EXPERT_RUN: usize = 256;

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
        let source = NSString::from_str(include_str!("shaders.metal"));
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
        let source = NSString::from_str(include_str!("gguf.metal"));
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
        let source = NSString::from_str(include_str!("attend.metal"));
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

    /// Runs every job in one command buffer. Returns, per job, `vectors * rows` values laid out vector by vector.
    pub fn dense(&self, jobs: &[DenseJob]) -> Result<Vec<Vec<f32>>> {
        // Metal hands back objects that are only let go when a pool drains, and the threads this runs on
        // have none of their own: without one here, every call would leave its command buffers behind.
        objc2::rc::autoreleasepool(|_| {
            let commands = self.queue.commandBuffer().context("creating a command buffer")?;
            let mut outputs = Vec::new();
            for job in jobs {
                let vectors = job.xs.len() / job.cols;
                ensure!(job.format == Format::Bf16, "the exact dense kernel reads bfloat16");
                ensure!(job.weights.len() == job.rows * job.cols * 2 && job.xs.len() == vectors * job.cols, "bad dense job");
                ensure!(job.cols % 16 == 0, "the dense kernel takes 16 columns per step");
                let (weights, offset) = self.weights(job.weights)?;
                ensure!(offset % 4 == 0, "weights are not word-aligned in the checkpoint");
                let x = copied(&self.device, job.xs)?;
                let out = empty(&self.device, vectors * job.rows * 4)?;
                let params = [job.rows as u32, job.cols as u32, vectors as u32];
                let buffers = [(&weights, offset), (&x, 0), (&out, 0)];
                self.encode(&commands, &self.dense, &buffers, &params, (job.rows, vectors.div_ceil(SHARE)))?;
                outputs.push((out, vectors * job.rows));
            }
            commands.commit();
            commands.waitUntilCompleted();
            finish(&commands)?;
            Ok(outputs.into_iter().map(|(buffer, len)| read(&buffer, len)).collect())
        })
    }

    /// Returns `pairs * rows` bfloat16-rounded products, pair by pair.
    pub fn experts(&self, job: &ExpertJob) -> Result<Vec<f32>> {
        // Metal hands back objects that are only let go when a pool drains, and the threads this runs on
        // have none of their own: without one here, every call would leave its command buffers behind.
        objc2::rc::autoreleasepool(|_| {
            let (row_bytes, blocks) = (job.cols / 2, job.cols / 16);
            ensure!(job.cols % 64 == 0, "the expert kernel takes four blocks per step");
            ensure!(job.activations.iter().all(|a| a.len() == job.cols), "activation width does not match the experts");
            if job.pairs.is_empty() {
                return Ok(Vec::new());
            }
            // Activations side by side, in the layout the kernel indexes.
            let mut even = Vec::with_capacity(job.activations.len() * row_bytes);
            let mut odd = Vec::with_capacity(job.activations.len() * row_bytes);
            let mut block_scales = Vec::with_capacity(job.activations.len() * blocks);
            let mut outer = Vec::with_capacity(job.activations.len());
            for activation in job.activations {
                let (e, o, scales, outer_scale) = activation.parts();
                even.extend_from_slice(e);
                odd.extend_from_slice(o);
                block_scales.extend_from_slice(scales);
                outer.push(outer_scale);
            }
            // Consecutive pairs for the same expert share GPU threads, `SHARE` at a time. `slots[i]` is where
            // pair `i` lands in the output.
            let mut groups: Vec<Group> = Vec::new();
            let mut slots = Vec::with_capacity(job.pairs.len());
            for &(expert, activation, outer_scale) in job.pairs {
                let open = groups.last().is_some_and(|g| g.expert == expert as u32 && (g.count as usize) < SHARE);
                if !open {
                    // Unused slots point at a real activation so the kernel reads valid memory; it never stores them.
                    let activations = [activation as u32; SHARE];
                    groups.push(Group { expert: expert as u32, count: 0, outer_scale, unused: 0, activations });
                }
                let index = groups.len() - 1;
                let group = &mut groups[index];
                group.activations[group.count as usize] = activation as u32;
                slots.push(index * SHARE + group.count as usize);
                group.count += 1;
            }

            let (values, values_offset) = self.weights(job.values)?;
            let (scales, scales_offset) = self.weights(job.scales)?;
            ensure!(values_offset % 4 == 0, "expert weights are not word-aligned in the checkpoint");
            let out = empty(&self.device, groups.len() * SHARE * job.rows * 4)?;
            let buffers = [
                (&values, values_offset),
                (&scales, scales_offset),
                (&copied(&self.device, &even)?, 0),
                (&copied(&self.device, &odd)?, 0),
                (&copied(&self.device, &block_scales)?, 0),
                (&copied(&self.device, &outer)?, 0),
                (&copied(&self.device, &groups)?, 0),
                (&self.e4m3, 0),
                (&out, 0),
            ];
            let params = [job.rows as u32, job.cols as u32, groups.len() as u32];
            let commands = self.queue.commandBuffer().context("creating a command buffer")?;
            self.encode(&commands, &self.experts, &buffers, &params, (job.rows, groups.len()))?;
            commands.commit();
            commands.waitUntilCompleted();
            finish(&commands)?;
            let all = read(&out, groups.len() * SHARE * job.rows);
            Ok(slots.iter().flat_map(|&slot| all[slot * job.rows..][..job.rows].iter().copied()).collect())
        })
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

    /// [`Gpu::dense`] on the fast path: the weights are widened to float and multiplied by Apple's tuned matrix
    /// product. Several times faster for many vectors, and equal to `dense` only up to rounding.
    pub fn dense_fast(&self, jobs: &[DenseJob]) -> Result<Vec<Vec<f32>>> {
        // Metal hands back objects that are only let go when a pool drains, and the threads this runs on
        // have none of their own: without one here, every call would leave its command buffers behind.
        objc2::rc::autoreleasepool(|_| {
            let commands = self.queue.commandBuffer().context("creating a command buffer")?;
            let mut outputs = Vec::new();
            for (index, job) in jobs.iter().enumerate() {
                let vectors = job.xs.len() / job.cols;
                let row_bytes = job.format.row_bytes(job.cols).context("rows do not make whole blocks")?;
                ensure!(job.weights.len() == job.rows * row_bytes && job.xs.len() == vectors * job.cols, "bad dense job");
                let (weights, offset) = self.weights(job.weights)?;
                let widened = self.scratch(2 * index, job.rows * job.cols * 4)?;
                if job.format == Format::Bf16 {
                    ensure!(job.cols % 16 == 0, "the widening kernel takes 16 columns per step");
                    ensure!(offset % 4 == 0, "weights are not word-aligned in the checkpoint");
                    let params = [job.rows as u32, job.cols as u32, 0];
                    let buffers = [(&weights, offset), (&widened, 0)];
                    self.encode(&commands, &self.widen_bf16, &buffers, &params, (job.cols / 8, job.rows))?;
                } else {
                    let first = copied(&self.device, &[0u32])?;
                    let params = [job.rows as u32, job.cols as u32, 1];
                    let buffers = [(&weights, offset), (&first, 0), (&widened, 0)];
                    self.encode(&commands, self.widen_ggml(job.format)?, &buffers, &params, (job.cols.div_ceil(256), job.rows))?;
                }
                let x = copied(&self.device, job.xs)?;
                let out = self.scratch(2 * index + 1, vectors * job.rows * 4)?;
                // out = x . widened^T: one row per vector.
                self.multiply(&commands, (&x, 0, vectors, job.cols), (&widened, 0, job.rows, job.cols), (&out, 0), 1);
                outputs.push((out, vectors * job.rows));
            }
            commands.commit();
            commands.waitUntilCompleted();
            finish(&commands)?;
            Ok(outputs.into_iter().map(|(buffer, len)| read(&buffer, len)).collect())
        })
    }

    fn widen_ggml(&self, format: Format) -> Result<&Pipeline> {
        Ok(match format {
            Format::Q4K => &self.widen_ggml[0],
            Format::Q5_0 => &self.widen_ggml[1],
            Format::Q6K => &self.widen_ggml[2],
            Format::Q8_0 => &self.widen_ggml[3],
            other => anyhow::bail!("no kernel unpacks {other:?}"),
        })
    }

    /// Each expert of a stack applied to its own vectors, on the fast path.
    ///
    /// `values` and `scales` hold `experts` matrices of `rows x cols`, stored as `stack` says (a GGUF stack has
    /// no separate scales).
    /// `vectors[expert]` are that expert's inputs, each `cols` wide. Returns, per expert, its results end to end
    /// (`rows` values per input). For NVFP4 that is without the expert's outer scale, which the caller applies.
    /// `slot` (0 or 1) keeps a layer's two stacks apart.
    ///
    /// Every expert gets a product of its own.
    ///
    /// The results must be checked: on an M1 Max, Apple's matrix product now and then returns NaN for half the
    /// outputs of a run of inputs, from clean inputs and weights, repeatably for the same data. The callers
    /// recompute any expert whose results are not all finite.
    pub fn experts_fast(
        &self,
        (values, scales): (&[u8], &[u8]),
        (experts, rows, cols): (usize, usize, usize),
        stack: Stack,
        vectors: &[Vec<&[f32]>],
        slot: usize,
    ) -> Result<Vec<Vec<f32>>> {
        // Metal hands back objects that are only let go when a pool drains, and the threads this runs on
        // have none of their own: without one here, every call would leave its command buffers behind.
        objc2::rc::autoreleasepool(|_| {
            ensure!(vectors.len() == experts, "one list of vectors per expert");
            let weights = experts * rows * cols;
            // NVFP4 packs two values per byte in blocks of 16; MXFP8 is a byte per value in blocks of 32. `block` is
            // the values one GPU thread unpacks.
            let (block, widen) = match stack {
                Stack::Nvfp4 => {
                    ensure!(values.len() == weights / 2 && scales.len() == weights / 16, "bad expert stack");
                    (16, &self.widen_nvfp4)
                }
                Stack::Mxfp8 => {
                    ensure!(values.len() == weights && scales.len() == weights / 32, "bad expert stack");
                    (32, &self.widen_mxfp8)
                }
                Stack::Ggml(format) => {
                    let row_bytes = format.row_bytes(cols).context("rows do not make whole blocks")?;
                    ensure!(values.len() == experts * rows * row_bytes, "bad expert stack");
                    (32, self.widen_ggml(format)?)
                }
            };
            ensure!(cols % 32 == 0 && vectors.iter().flatten().all(|v| v.len() == cols), "bad expert inputs");
            let mut order: Vec<usize> = (0..experts).filter(|&e| !vectors[e].is_empty()).collect();
            let mut outputs = vec![Vec::new(); experts];
            if order.is_empty() {
                return Ok(outputs);
            }
            order.sort_by_key(|&e| std::cmp::Reverse(vectors[e].len()));

            // Products as (position in `order`, first vector, vectors): an expert's vectors in runs of at most
            // `run`, since Apple's product has returned NaN for runs of around nine hundred.
            let run = EXPERT_RUN;
            let groups: Vec<(usize, usize, usize)> = order
                .iter()
                .enumerate()
                .flat_map(|(position, &expert)| {
                    let all = vectors[expert].len();
                    (0..all).step_by(run).map(move |start| (position, start, run.min(all - start)))
                })
                .collect();

            let (values_buffer, values_offset) = self.weights(values)?;
            ensure!(values_offset % 4 == 0 || matches!(stack, Stack::Ggml(_)), "expert weights are not word-aligned");
            // Slots after the dense ones.
            // Every product takes a full run, the last of an expert padded with zero vectors: one shape for all.
            let padded = groups.len() * run;
            // The unpacked weights go in several buffers of at most a gigabyte: Apple's product returns NaN for
            // matrices that start more than two gigabytes into a buffer.
            let per_part = (WIDENED_BYTES / (rows * cols * 4)).max(1);
            let widened: Vec<Buffer> = order
                .chunks(per_part)
                .enumerate()
                .map(|(part, experts)| self.scratch(40 + 16 * slot + part, experts.len() * rows * cols * 4))
                .collect::<Result<_>>()?;
            ensure!(widened.len() <= 16, "too many experts for the scratch buffers");
            let x = self.scratch(18 + slot, padded * cols * 4)?;
            let out = self.scratch(20 + slot, padded * rows * 4)?;

            // Inputs product by product.
            // SAFETY: the buffer is shared-mode, at least `padded * cols` floats long, and nothing else uses it until
            // the command buffer below is committed.
            let inputs = unsafe { std::slice::from_raw_parts_mut(x.contents().as_ptr() as *mut f32, padded * cols) };
            let mut rest = inputs;
            for &(position, start, count) in &groups {
                let (mine, after) = rest.split_at_mut(run * cols);
                let (real, padding) = mine.split_at_mut(count * cols);
                for (slot, vector) in real.chunks_exact_mut(cols).zip(&vectors[order[position]][start..start + count]) {
                    slot.copy_from_slice(vector);
                }
                padding.fill(0.0);
                rest = after;
            }

            let commands = self.queue.commandBuffer().context("creating a command buffer")?;
            let order_buffer = copied(&self.device, &order.iter().map(|&e| e as u32).collect::<Vec<u32>>())?;
            let scales_buffer = match stack {
                Stack::Ggml(_) => None,
                _ => Some(self.weights(scales)?),
            };
            for (part, (experts, widened)) in order.chunks(per_part).zip(&widened).enumerate() {
                let params = [rows as u32, cols as u32, experts.len() as u32];
                let chosen = (&order_buffer, part * per_part * 4);
                match &scales_buffer {
                    // The GGUF kernels take 256 values to a thread.
                    None => {
                        let buffers = [(&values_buffer, values_offset), chosen, (widened, 0)];
                        self.encode(&commands, widen, &buffers, &params, (cols.div_ceil(256), experts.len() * rows))?;
                    }
                    Some((scales_buffer, scales_offset)) => {
                        let buffers =
                            [(&values_buffer, values_offset), (scales_buffer, *scales_offset), (&self.e4m3, 0), chosen, (widened, 0)];
                        self.encode(&commands, widen, &buffers, &params, (cols / block, experts.len() * rows))?;
                    }
                }
            }
            for (index, &(position, ..)) in groups.iter().enumerate() {
                self.multiply(
                    &commands,
                    (&x, index * run * cols * 4, run, cols),
                    (&widened[position / per_part], (position % per_part) * rows * cols * 4, rows, cols),
                    (&out, index * run * rows * 4),
                    1,
                );
            }
            commands.commit();
            commands.waitUntilCompleted();
            finish(&commands)?;

            // SAFETY: as for `inputs`; the command buffer that wrote it has completed.
            let results = unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, padded * rows) };
            // Each expert's runs are next to each other, in order.
            for (index, &(position, _, count)) in groups.iter().enumerate() {
                outputs[order[position]].extend_from_slice(&results[index * run * rows..][..count * rows]);
            }
            Ok(outputs)
        })
    }

    /// The gates of a message merge for a long batch: `forget = F (D pair)` and `gain = G (D pair)` for every
    /// token's `pair`, where the three matrices are plain floats of `rank x width`, `hidden x rank` and
    /// `hidden x rank`. Returns the two, `tokens x hidden` each, before their `tanh`.
    pub fn gates_fast(&self, pairs: &[f32], down: &[u8], forget: &[u8], gain: &[u8], (rank, hidden): (usize, usize)) -> Result<(Vec<f32>, Vec<f32>)> {
        // Metal hands back objects that are only let go when a pool drains, and the threads this runs on
        // have none of their own: without one here, every call would leave its command buffers behind.
        objc2::rc::autoreleasepool(|_| {
            let width = down.len() / 4 / rank;
            let tokens = pairs.len() / width;
            ensure!(
                down.len() == rank * width * 4 && forget.len() == hidden * rank * 4 && gain.len() == forget.len() && pairs.len() == tokens * width,
                "bad gate job"
            );
            // Merges run side by side on the CPU; here they share scratch buffers and take turns.
            let _turn = self.gates.lock().expect("gate lock");
            let buffer = |slot: usize, floats: usize| self.scratch(96 + slot, floats * 4);
            let (x, low) = (buffer(0, tokens * width)?, buffer(1, tokens * rank)?);
            let (forgets, gains) = (buffer(2, tokens * hidden)?, buffer(3, tokens * hidden)?);
            // SAFETY: the buffer is shared-mode, at least `pairs.len()` floats long, and nothing else uses it until
            // the submission below completes.
            unsafe { std::ptr::copy_nonoverlapping(pairs.as_ptr(), x.contents().as_ptr() as *mut f32, pairs.len()) };
            let ((down, down_at), (forget, forget_at), (gain, gain_at)) = (self.weights(down)?, self.weights(forget)?, self.weights(gain)?);
            let commands = self.queue.commandBuffer().context("creating a command buffer")?;
            self.multiply(&commands, (&x, 0, tokens, width), (&down, down_at, rank, width), (&low, 0), 1);
            self.multiply(&commands, (&low, 0, tokens, rank), (&forget, forget_at, hidden, rank), (&forgets, 0), 1);
            self.multiply(&commands, (&low, 0, tokens, rank), (&gain, gain_at, hidden, rank), (&gains, 0), 1);
            commands.commit();
            commands.waitUntilCompleted();
            finish(&commands)?;
            Ok((read(&forgets, tokens * hidden), read(&gains, tokens * hidden)))
        })
    }

    /// A GGUF layer's experts for a long batch, start to finish in one submission: each token's input laid out
    /// under the experts it is routed to, the first product, the activation, the second product, and each
    /// token's weighted sum. Returns `tokens x hidden` outputs.
    ///
    /// Nothing comes back to the CPU in between, which is most of what the two products cost when run apart.
    /// Every product takes exactly [`EXPERT_RUN`] rows, padded with zeros: Apple's routine has returned NaN
    /// for half the outputs of products of some other sizes, and none of this one. The result should still be
    /// checked.
    pub fn experts_whole(&self, job: &WholeExperts) -> Result<Vec<f32>> {
        // Metal hands back objects that are only let go when a pool drains, and the threads this runs on
        // have none of their own: without one here, every call would leave its command buffers behind.
        objc2::rc::autoreleasepool(|_| {
            let &WholeExperts { experts, hidden, units, chosen, .. } = job;
            let tokens = job.x.len() / hidden;
            ensure!(job.x.len() == tokens * hidden && job.routes.len() == tokens * chosen, "bad expert job");
            ensure!(hidden % 256 == 0 || hidden % 32 == 0, "bad expert width");
            // Rows: for each expert in use, its tokens in runs of `run`, the last padded.
            let mut tokens_of: Vec<Vec<(u32, u32)>> = vec![Vec::new(); experts];
            for (at, &(expert, _)) in job.routes.iter().enumerate() {
                tokens_of[expert as usize].push(((at / chosen) as u32, at as u32));
            }
            let run = EXPERT_RUN;
            let order: Vec<u32> = (0..experts as u32).filter(|&e| !tokens_of[e as usize].is_empty()).collect();
            // Products as (position in `order`, first row).
            let mut products: Vec<(usize, usize)> = Vec::new();
            let mut token_of: Vec<u32> = Vec::new();
            let mut row_of = vec![0u32; tokens * chosen];
            for (position, &expert) in order.iter().enumerate() {
                for piece in tokens_of[expert as usize].chunks(run) {
                    products.push((position, token_of.len()));
                    for &(token, route) in piece {
                        row_of[route as usize] = token_of.len() as u32;
                        token_of.push(token);
                    }
                    token_of.resize(token_of.len().next_multiple_of(run), u32::MAX);
                }
            }
            let rows = token_of.len();
            let weights: Vec<f32> = job.routes.iter().map(|&(_, weight)| weight).collect();

            let buffer = |slot: usize, bytes: usize| self.scratch(80 + slot, bytes);
            let (x, token_buffer, laid) = (buffer(0, tokens * hidden * 4)?, buffer(1, rows * 4)?, buffer(2, rows * hidden * 4)?);
            // The first product's results, and later the second's, which are the same size.
            let wide = buffer(3, rows * 2 * units.max(hidden / 2) * 4)?;
            let (active, row_buffer, weight_buffer) = (buffer(4, rows * units * 4)?, buffer(5, tokens * chosen * 4)?, buffer(6, tokens * chosen * 4)?);
            let out = buffer(7, tokens * hidden * 4)?;
            // SAFETY: each buffer is shared-mode and at least as long as what is written, and nothing else uses the
            // scratch slots until the submission below completes.
            unsafe {
                std::ptr::copy_nonoverlapping(job.x.as_ptr(), x.contents().as_ptr() as *mut f32, job.x.len());
                std::ptr::copy_nonoverlapping(token_of.as_ptr(), token_buffer.contents().as_ptr() as *mut u32, rows);
                std::ptr::copy_nonoverlapping(row_of.as_ptr(), row_buffer.contents().as_ptr() as *mut u32, row_of.len());
                std::ptr::copy_nonoverlapping(weights.as_ptr(), weight_buffer.contents().as_ptr() as *mut f32, weights.len());
            }
            let order_buffer = copied(&self.device, &order)?;

            let commands = self.queue.commandBuffer().context("creating a command buffer")?;
            let step = |pipeline: &Pipeline, buffers: &[(&Buffer, usize)], params: &[u32], grid: (usize, usize)| -> Result<()> {
                let encoder = commands.computeCommandEncoder().context("creating a compute encoder")?;
                encoder.setComputePipelineState(pipeline);
                // SAFETY: the buffers are held until the submission completes; Metal copies the parameters now.
                unsafe {
                    for (index, (buffer, offset)) in buffers.iter().enumerate() {
                        encoder.setBuffer_offset_atIndex(Some(buffer), *offset, index);
                    }
                    let pointer = NonNull::new(params.as_ptr() as *mut c_void).expect("parameters");
                    encoder.setBytes_length_atIndex(pointer, size_of_val(params), buffers.len());
                }
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: grid.0, height: grid.1, depth: 1 },
                    MTLSize { width: 32, height: 1, depth: 1 },
                );
                encoder.endEncoding();
                Ok(())
            };
            // One matrix stack's part: its chosen experts unpacked, then a product per run of rows.
            let stack = |values: &[u8], format: Format, (out_rows, cols): (usize, usize), from: &Buffer, to: &Buffer, slot: usize| -> Result<()> {
                let row_bytes = format.row_bytes(cols).context("rows do not make whole blocks")?;
                ensure!(values.len() == experts * out_rows * row_bytes, "bad expert stack");
                let (values_buffer, values_offset) = self.weights(values)?;
                let per_part = (WIDENED_BYTES / (out_rows * cols * 4)).max(1);
                let widened: Vec<Buffer> = order
                    .chunks(per_part)
                    .enumerate()
                    .map(|(part, experts)| self.scratch(40 + 16 * slot + part, experts.len() * out_rows * cols * 4))
                    .collect::<Result<_>>()?;
                ensure!(widened.len() <= 16, "too many experts for the scratch buffers");
                for (part, (experts, widened)) in order.chunks(per_part).zip(&widened).enumerate() {
                    let params = [out_rows as u32, cols as u32, experts.len() as u32];
                    let buffers = [(&values_buffer, values_offset), (&order_buffer, part * per_part * 4), (widened, 0)];
                    self.encode(&commands, self.widen_ggml(format)?, &buffers, &params, (cols.div_ceil(256), experts.len() * out_rows))?;
                }
                for &(position, first) in &products {
                    self.multiply(
                        &commands,
                        (from, first * cols * 4, run, cols),
                        (&widened[position / per_part], (position % per_part) * out_rows * cols * 4, out_rows, cols),
                        (to, first * out_rows * 4),
                        1,
                    );
                }
                Ok(())
            };

            step(&self.expert_steps[0], &[(&x, 0), (&token_buffer, 0), (&laid, 0)], &[hidden as u32], (hidden / 4, rows))?;
            stack(job.gate_up.0, job.gate_up.1, (2 * units, hidden), &laid, &wide, 0)?;
            let params = [units as u32, job.gate_cap.to_bits(), job.up_cap.to_bits(), 0];
            step(&self.expert_steps[1], &[(&wide, 0), (&active, 0)], &params, (units, rows))?;
            stack(job.down.0, job.down.1, (hidden, units), &active, &wide, 1)?;
            let buffers = [(&wide, 0), (&row_buffer, 0), (&weight_buffer, 0), (&out, 0)];
            step(&self.expert_steps[2], &buffers, &[hidden as u32, chosen as u32], (hidden / 4, tokens))?;
            commands.commit();
            commands.waitUntilCompleted();
            finish(&commands)?;
            Ok(read(&out, tokens * hidden))
        })
    }

    /// Attention for a stretch of a long batch, a key head at a time in one submission: scores as a product of
    /// the head's queries and keys, weights by a softmax over each row's window with its distance bias and sink,
    /// and outputs as a product of the weights and values. Returns each head's `rows x dim` outputs, before the
    /// output gate.
    ///
    /// As with [`Gpu::experts_fast`], the results must be checked for values that are not finite.
    pub fn attend_fast(&self, heads: &[AttendHead], shape: &AttendShape) -> Result<Vec<Vec<f32>>> {
        // Metal hands back objects that are only let go when a pool drains, and the threads this runs on
        // have none of their own: without one here, every call would leave its command buffers behind.
        objc2::rc::autoreleasepool(|_| {
            let &AttendShape { rows, keys, dim, d_rel, extent, .. } = shape;
            ensure!(shape.profiles.len() == d_rel * extent, "bad distance profiles");
            for head in heads {
                ensure!(
                    head.queries.len() == rows * dim
                        && head.keys.len() >= keys * dim
                        && head.values.len() >= keys * dim
                        && head.rel.len() == rows * d_rel
                        && head.sinks.len() == rows,
                    "bad attention job"
                );
            }
            let count = heads.len();
            let buffer = |slot: usize, floats: usize| self.scratch(24 + slot, floats * 4);
            let (queries, tables, values) = (buffer(0, count * rows * dim)?, buffer(1, count * keys * dim)?, buffer(2, count * keys * dim)?);
            let (rel, sinks, profiles) = (buffer(3, count * rows * d_rel)?, buffer(4, count * rows)?, buffer(5, d_rel * extent)?);
            let (bias, scores, out) = (buffer(6, rows * extent)?, buffer(7, rows * keys)?, buffer(8, count * rows * dim)?);
            // SAFETY: each buffer is shared-mode and at least as long as what is written, and nothing else uses the
            // scratch slots until the submission below completes.
            let fill = |buffer: &Buffer, at: usize, data: &[f32]| unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), (buffer.contents().as_ptr() as *mut f32).add(at), data.len());
            };
            fill(&profiles, 0, shape.profiles);
            for (h, head) in heads.iter().enumerate() {
                fill(&queries, h * rows * dim, head.queries);
                fill(&tables, h * keys * dim, &head.keys[..keys * dim]);
                fill(&values, h * keys * dim, &head.values[..keys * dim]);
                fill(&rel, h * rows * d_rel, head.rel);
                fill(&sinks, h * rows, head.sinks);
            }

            let commands = self.queue.commandBuffer().context("creating a command buffer")?;
            let width = (keys / 256).clamp(1, 16);
            let params = [
                keys as u32,
                shape.share as u32,
                shape.first as u32,
                shape.lowest as u32,
                u32::from(shape.local),
                shape.window as u32,
                extent as u32,
                width as u32,
            ];
            for h in 0..count {
                self.product(&commands, (&rel, h * rows * d_rel * 4, rows, d_rel), (&profiles, 0, d_rel, extent), (&bias, 0), 1, false, 1.0);
                let (mine, table) = ((&queries, h * rows * dim * 4, rows, dim), (&tables, h * keys * dim * 4, keys, dim));
                self.product(&commands, mine, table, (&scores, 0), 1, true, shape.scaling as f64);
                let encoder = commands.computeCommandEncoder().context("creating a compute encoder")?;
                encoder.setComputePipelineState(&self.prompt_weights);
                // SAFETY: the buffers are held until the submission completes, and the offsets are within them.
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(&scores), 0, 0);
                    encoder.setBuffer_offset_atIndex(Some(&bias), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&sinks), h * rows * 4, 2);
                    encoder.setBytes_length_atIndex(NonNull::from(&params).cast::<c_void>(), size_of_val(&params), 3);
                }
                encoder.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: rows, height: 1, depth: 1 },
                    MTLSize { width: 32 * width, height: 1, depth: 1 },
                );
                encoder.endEncoding();
                let table = (&values, h * keys * dim * 4, keys, dim);
                self.product(&commands, (&scores, 0, rows, keys), table, (&out, h * rows * dim * 4), 1, false, 1.0);
            }
            commands.commit();
            commands.waitUntilCompleted();
            finish(&commands)?;
            let results = read(&out, count * rows * dim);
            Ok(results.chunks_exact(rows * dim).map(<[f32]>::to_vec).collect())
        })
    }

    /// Encodes `out = left . right^T` for `count` matrices laid end to end in each buffer, as float.
    /// `left` is `(buffer, byte offset, rows, cols)` per matrix and `right` likewise; they share `cols`. `out` is
    /// `(buffer, byte offset)`.
    fn multiply(
        &self,
        commands: &ProtocolObject<dyn MTLCommandBuffer>,
        left: (&Buffer, usize, usize, usize),
        right: (&Buffer, usize, usize, usize),
        out: (&Buffer, usize),
        count: usize,
    ) {
        self.product(commands, left, right, out, count, true, 1.0);
    }

    /// Encodes `out = scale * left . right` (or `left . right^T` with `transposed`) for `count` matrices laid
    /// end to end in each buffer, as float. Each operand is `(buffer, byte offset, rows, cols)` per matrix.
    #[allow(clippy::too_many_arguments)]
    fn product(
        &self,
        commands: &ProtocolObject<dyn MTLCommandBuffer>,
        left: (&Buffer, usize, usize, usize),
        right: (&Buffer, usize, usize, usize),
        out: (&Buffer, usize),
        count: usize,
        transposed: bool,
        scale: f64,
    ) {
        use objc2::AnyThread;
        use objc2_metal_performance_shaders::{MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};
        let ((left, left_offset, vectors, cols), (right, right_offset, right_rows, right_cols)) = (left, right);
        // How many columns the result has.
        let rows = if transposed { right_rows } else { right_cols };
        // SAFETY: the callers size each buffer for `count` matrices of the stated shape past its offset, and hold
        // the buffers until the command buffer completes.
        unsafe {
            let matrix = |buffer: &Buffer, offset: usize, r: usize, c: usize| {
                let descriptor = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
                    r, c, count, c * 4, r * c * 4, MPSDataType::Float32,
                );
                MPSMatrix::initWithBuffer_offset_descriptor(MPSMatrix::alloc(), buffer, offset, &descriptor)
            };
            let kernel = MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(), &self.device, false, transposed, vectors, rows, cols, scale, 0.0,
            );
            kernel.setBatchSize(count);
            kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                commands,
                &matrix(left, left_offset, vectors, cols),
                &matrix(right, right_offset, right_rows, right_cols),
                &matrix(out.0, out.1, vectors, rows),
            );
        }
    }

    /// Encodes one kernel over a `grid.0 x grid.1` grid. `params` goes in the slot after the buffers.
    fn encode(
        &self,
        commands: &ProtocolObject<dyn MTLCommandBuffer>,
        pipeline: &Pipeline,
        buffers: &[(&Buffer, usize)],
        params: &[u32; 3],
        grid: (usize, usize),
    ) -> Result<()> {
        let encoder = commands.computeCommandEncoder().context("creating a compute encoder")?;
        encoder.setComputePipelineState(pipeline);
        // SAFETY: each buffer is alive until the command buffer completes (the callers hold them), offsets are
        // within the buffers, and `params` is copied by Metal during the call.
        unsafe {
            for (index, (buffer, offset)) in buffers.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(buffer), *offset, index);
            }
            let pointer = NonNull::from(params).cast::<c_void>();
            encoder.setBytes_length_atIndex(pointer, size_of_val(params), buffers.len());
        }
        let width = pipeline.threadExecutionWidth();
        let height = pipeline.maxTotalThreadsPerThreadgroup() / width;
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize { width: grid.0, height: grid.1, depth: 1 },
            MTLSize { width, height, depth: 1 },
        );
        encoder.endEncoding();
        Ok(())
    }
}

/// A buffer shared with the GPU that the caller also reads and writes directly, between runs.
pub(crate) struct Shared {
    buffer: Buffer,
    /// How many bytes the caller may touch: none for weights wrapped in place, which are only bound.
    bytes: usize,
}

// SAFETY: as for `Gpu`: Metal buffers may be used from any thread, and access to the contents goes through
// `&self`/`&mut self`.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    /// Another zeroed buffer on the same device.
    pub(crate) fn sibling(&self, bytes: usize) -> Result<Shared> {
        let bytes = bytes.next_multiple_of(16).max(16);
        Ok(Shared { buffer: empty(&self.buffer.device(), bytes)?, bytes })
    }

    pub(crate) fn floats(&self) -> &[f32] {
        // SAFETY: a shared-mode buffer of `bytes` bytes, page-aligned, that no kernel is writing: runs are
        // synchronous and take the buffers by reference.
        unsafe { std::slice::from_raw_parts(self.buffer.contents().as_ptr() as *const f32, self.bytes / 4) }
    }

    pub(crate) fn floats_mut(&mut self) -> &mut [f32] {
        // SAFETY: as `floats`, and `&mut self` rules out any other view.
        unsafe { std::slice::from_raw_parts_mut(self.buffer.contents().as_ptr() as *mut f32, self.bytes / 4) }
    }

    pub(crate) fn words_mut(&mut self) -> &mut [u32] {
        // SAFETY: as `floats_mut`.
        unsafe { std::slice::from_raw_parts_mut(self.buffer.contents().as_ptr() as *mut u32, self.bytes / 4) }
    }
}

/// Compiled kernels, addressed by their place in the list they were built from.
pub(crate) struct Kernels(Vec<Pipeline>);

// SAFETY: pipeline states are immutable once built and documented as thread-safe.
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

/// One kernel of a program: which buffers it is given, at which byte offsets, its parameters and its grid.
pub(crate) struct Step {
    pub kernel: usize,
    /// `(buffer, byte offset)` for each binding in order; the parameters take the next binding.
    pub buffers: Vec<(usize, usize)>,
    pub params: Vec<u8>,
    pub grid: Grid,
}

#[derive(Clone, Copy)]
pub(crate) enum Grid {
    /// A thread per cell.
    Threads(usize, usize),
    /// A group of 32 threads per cell.
    Groups(usize, usize),
    /// So many groups, each of the given number of threads (a multiple of 32).
    Wide { groups: usize, threads: usize },
    /// A thread per visible key and row; how many keys are visible is known only when the program runs.
    Keys { local: bool, rows: usize },
}

impl Gpu {
    /// A zeroed buffer of `bytes` bytes.
    pub(crate) fn shared(&self, bytes: usize) -> Result<Shared> {
        let bytes = bytes.next_multiple_of(16).max(16);
        Ok(Shared { buffer: empty(&self.device, bytes)?, bytes })
    }

    /// Weights as a buffer and the offset they start at: wrapped in place when they lie in the checkpoint's
    /// memory map, which must then outlive the buffer.
    pub(crate) fn wrap(&self, bytes: &[u8]) -> Result<(Shared, usize)> {
        let (buffer, offset) = self.weights(bytes)?;
        Ok((Shared { buffer, bytes: 0 }, offset))
    }

    /// Compiles `source` and builds the kernels named.
    pub(crate) fn kernels(&self, source: &str, names: &[&str]) -> Result<Kernels> {
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&MTLCompileOptions::new()))
            .map_err(|e| anyhow!("compiling kernels: {}", e.localizedDescription()))?;
        let build = |name: &&str| -> Result<Pipeline> {
            let function = library.newFunctionWithName(&NSString::from_str(name)).with_context(|| format!("no kernel {name}"))?;
            self.device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| anyhow!("building kernel {name}: {}", e.localizedDescription()))
        };
        Ok(Kernels(names.iter().map(build).collect::<Result<_>>()?))
    }

    /// Runs `steps` in order as one submission and waits for it, every grid `tokens` deep. `keys` is how many
    /// keys a sliding layer and a full layer see at most, for the grids that depend on it. Returns the seconds
    /// the GPU spent.
    pub(crate) fn run(&self, kernels: &[&Kernels], steps: &[Step], buffers: &[Shared], keys: [usize; 2], tokens: usize) -> Result<f64> {
        // Metal hands back objects that are only let go when a pool drains, and the threads this runs on
        // have none of their own: without one here, every call would leave its command buffers behind.
        objc2::rc::autoreleasepool(|_| {
            let pipelines: Vec<&Pipeline> = kernels.iter().flat_map(|kernels| &kernels.0).collect();
            let commands = self.queue.commandBuffer().context("creating a command buffer")?;
            let encoder = commands.computeCommandEncoder().context("creating a compute encoder")?;
            let wide = MTLSize { width: 32, height: 1, depth: 1 };
            for step in steps {
                encoder.setComputePipelineState(pipelines[step.kernel]);
                // SAFETY: the buffers outlive the submission, which is waited for below; the offsets were laid out
                // within them by the program's builder; Metal copies the parameter bytes during the call.
                unsafe {
                    for (index, &(buffer, offset)) in step.buffers.iter().enumerate() {
                        encoder.setBuffer_offset_atIndex(Some(&buffers[buffer].buffer), offset, index);
                    }
                    if !step.params.is_empty() {
                        let pointer = NonNull::new(step.params.as_ptr() as *mut c_void).expect("parameters");
                        encoder.setBytes_length_atIndex(pointer, step.params.len(), step.buffers.len());
                    }
                }
                let size = |width: usize, height: usize| MTLSize { width, height, depth: tokens };
                match step.grid {
                    Grid::Threads(width, height) => encoder.dispatchThreads_threadsPerThreadgroup(size(width, height), wide),
                    Grid::Groups(width, height) => encoder.dispatchThreadgroups_threadsPerThreadgroup(size(width, height), wide),
                    Grid::Wide { groups, threads } => encoder
                        .dispatchThreadgroups_threadsPerThreadgroup(size(groups, 1), MTLSize { width: threads, height: 1, depth: 1 }),
                    Grid::Keys { local, rows } => {
                        encoder.dispatchThreads_threadsPerThreadgroup(size(keys[usize::from(!local)], rows), wide)
                    }
                }
            }
            encoder.endEncoding();
            commands.commit();
            commands.waitUntilCompleted();
            finish(&commands)?;
            Ok(commands.GPUEndTime() - commands.GPUStartTime())
        })
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
mod tests {
    use super::*;
    use crate::quant::{dot_bf16_row, dot_nvfp4_row_quantized, quantize_nvfp4};

    fn bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 56) as u8
            })
            .collect()
    }

    fn floats(len: usize, seed: u64) -> Vec<f32> {
        bytes(len, seed).iter().map(|b| (*b as f32 - 127.5) / 64.0).collect()
    }

    #[test]
    fn dense_matches_the_cpu_kernel_exactly() {
        let gpu = Gpu::new(&[], true).unwrap();
        // A small case, and the real width with a vector count that is not a multiple of the share.
        for (rows, cols, vectors) in [(5, 48, 1), (33, 2304, 7)] {
            // Small bfloat16 values: sign, a modest exponent, random mantissa.
            let weights: Vec<u8> =
                bytes(rows * cols * 2, 1).as_chunks::<2>().0.iter().flat_map(|b| [b[0], 0x3C | (b[1] & 0x83)]).collect();
            let xs = floats(vectors * cols, 2);
            let out = gpu.dense(&[DenseJob { format: Format::Bf16, weights: &weights, rows, cols, xs: &xs }]).unwrap().remove(0);
            for v in 0..vectors {
                for r in 0..rows {
                    let cpu = dot_bf16_row(&weights[r * cols * 2..][..cols * 2], &xs[v * cols..][..cols]);
                    assert_eq!(out[v * rows + r], cpu, "{rows}x{cols}: vector {v}, row {r}");
                }
            }
        }
    }

    /// What a token's matrix products cost with a group of 32 threads sharing 1, 2, 4 or 8 rows, and what a
    /// dispatch costs by itself: `cargo test --release --lib gpu::tests::row_kernels -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn row_kernels() {
        use crate::checkpoint::{Checkpoint, resolve_model};
        let path = resolve_model(Some(std::path::Path::new("lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M"))).unwrap();
        let weights = Checkpoint::open(&path).unwrap();
        let gpu = Gpu::new(&weights.maps(), false).unwrap();
        let kernels = gpu.kernels(include_str!("gguf.metal"), &["row_q4k", "row_q5_0", "row_q6k", "row_q8_0", "row_f32"]).unwrap();
        let index = |format: Format| match format {
            Format::Q4K => 0,
            Format::Q5_0 => 1,
            Format::Q6K => 2,
            Format::Q8_0 => 3,
            other => panic!("{other:?}"),
        };
        let word = |values: &[usize]| -> Vec<u8> { values.iter().flat_map(|v| (*v as u32).to_le_bytes()).collect() };
        // Buffers: inputs, outputs, the matrices' order, then the weights.
        let mut buffers = vec![gpu.shared(8 * 3072 * 4).unwrap(), gpu.shared(200_058 * 4).unwrap(), gpu.shared(32).unwrap()];
        buffers[0].floats_mut().copy_from_slice(&floats(8 * 3072, 5));
        buffers[2].words_mut().copy_from_slice(&[0, 1, 2, 3, 4, 5, 6, 7]);

        for (count, rows) in [(300usize, 1usize), (3000, 1), (300, 2304)] {
            let step = || Step { kernel: 4, buffers: vec![(0, 0), (0, 0), (1, 0), (2, 0)], params: word(&[rows, 64, 256, 1, 0, 0, 0]), grid: Grid::Groups(rows, 1) };
            let steps: Vec<Step> = (0..count).map(|_| step()).collect();
            let best = (0..3).map(|_| gpu.run(&[&kernels], &steps, &buffers, [0, 0], 1).unwrap()).fold(f64::MAX, f64::min);
            eprintln!("{count} dispatches of {rows} short rows: {:.2} ms ({:.1} us each)", best * 1e3, best * 1e6 / count as f64);
        }

        // (name, rows, cols, matrices)
        let mut jobs: Vec<(String, usize, usize, usize)> = Vec::new();
        for visit in 0..48 {
            let p = format!("model.llm.layers.{}", visit % 24);
            for (name, rows, cols) in [
                ("attn.wq_du.weight", 3072, 2304),
                ("attn.wk_dv.weight", 1024, 2304),
                ("attn.wv_dv.weight", 1024, 2304),
                ("attn.wr_du.weight", 384, 2304),
                ("attn.gate_attn.weight", 3072, 2304),
                ("attn.wo_ud.weight", 2304, 3072),
            ] {
                jobs.push((format!("{p}.{name}"), rows, cols, 1));
            }
            jobs.push((format!("{p}.mlp.experts.w13_weight"), 2304, 2304, 8));
            jobs.push((format!("{p}.mlp.experts.w2_weight"), 2304, 1152, 8));
        }
        jobs.push(("model.llm.unembed.weight".to_owned(), 200_058, 2304, 1));
        let mut prepared = Vec::new();
        for (name, rows, cols, matrices) in &jobs {
            let (format, _, bytes) = weights.rows(name).unwrap();
            let (buffer, offset) = gpu.wrap(bytes).unwrap();
            buffers.push(buffer);
            prepared.push((index(format), buffers.len() - 1, offset, *rows, *cols, rows * format.row_bytes(*cols).unwrap(), *matrices));
        }
        let groups: [(&str, &dyn Fn(&str) -> bool); 5] = [
            ("projections", &|n: &str| n.contains(".attn.")),
            ("gate_up", &|n: &str| n.contains("w13")),
            ("down", &|n: &str| n.contains("w2_")),
            ("logits", &|n: &str| n.contains("unembed")),
            ("everything", &|_: &str| true),
        ];
        for (label, wanted) in groups {
            let times: Vec<String> = [1usize, 2, 4, 8]
                .iter()
                .map(|&together| {
                    let steps: Vec<Step> = jobs
                        .iter()
                        .zip(&prepared)
                        .filter(|((name, ..), _)| wanted(name))
                        .map(|(_, &(kernel, buffer, offset, rows, cols, stride, matrices))| Step {
                            kernel,
                            buffers: vec![(buffer, offset), (0, 0), (1, 0), (2, 0)],
                            params: word(&[rows, cols, stride, together, cols, 0, 0]),
                            grid: Grid::Groups(rows.div_ceil(together), matrices),
                        })
                        .collect();
                    let best = (0..3).map(|_| gpu.run(&[&kernels], &steps, &buffers, [0, 0], 1).unwrap()).fold(f64::MAX, f64::min);
                    format!("{:.1}", best * 1e3)
                })
                .collect();
            eprintln!("{label}: {} ms with a group of threads to 1, 2, 4, 8 rows", times.join(", "));
        }
    }

    /// What the pieces of a long prompt's chunk cost on the fast path, one layer visit's worth:
    /// `cargo test --release --lib gpu::tests::prompt_parts -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn prompt_parts() {
        use crate::checkpoint::{Checkpoint, resolve_model};
        use std::time::Instant;
        let path = resolve_model(Some(std::path::Path::new("lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M"))).unwrap();
        let weights = Checkpoint::open(&path).unwrap();
        let gpu = Gpu::new(&weights.maps(), true).unwrap();
        let tokens: usize = std::env::var("TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
        let best = |work: &dyn Fn()| (0..3).map(|_| { let t = Instant::now(); work(); t.elapsed().as_secs_f64() }).fold(f64::MAX, f64::min);

        let xs = floats(tokens * 3072, 7);
        let mut dense_products = 0usize;
        let mut jobs = Vec::new();
        for (name, rows, cols) in [
            ("attn.wq_du.weight", 3072, 2304),
            ("attn.wk_dv.weight", 1024, 2304),
            ("attn.wv_dv.weight", 1024, 2304),
            ("attn.wr_du.weight", 384, 2304),
            ("attn.gate_attn.weight", 3072, 2304),
            ("attn.wo_ud.weight", 2304, 3072),
        ] {
            let (format, _, bytes) = weights.rows(&format!("model.llm.layers.3.{name}")).unwrap();
            jobs.push(DenseJob { format, weights: bytes, rows, cols, xs: &xs[..tokens * cols] });
            dense_products += rows * cols * tokens;
        }
        let seconds = best(&|| drop(gpu.dense_fast(&jobs).unwrap()));
        eprintln!("projections, {tokens} tokens: {:.0} ms, {:.0} G products/s", seconds * 1e3, dense_products as f64 / seconds / 1e9);

        // Every expert with the same share of the tokens' eight choices.
        let experts = weights.experts(3).unwrap();
        let each = tokens * 8 / 128;
        for (label, matrix, slot) in [("gate_up", &experts.gate_up, 0), ("down", &experts.down, 1)] {
            let inputs = floats(each * matrix.cols, 9);
            let vectors: Vec<Vec<&[f32]>> = (0..128).map(|_| inputs.chunks_exact(matrix.cols).collect()).collect();
            let stack = Stack::Ggml(matrix.ggml().unwrap());
            let shape = (128, matrix.rows, matrix.cols);
            let seconds = best(&|| drop(gpu.experts_fast(matrix.raw_parts(), shape, stack, &vectors, slot).unwrap()));
            let products = 128 * each * matrix.rows * matrix.cols;
            eprintln!("{label}, {each} tokens to each of 128 experts: {:.0} ms, {:.0} G products/s", seconds * 1e3, products as f64 / seconds / 1e9);
            let none: Vec<Vec<&[f32]>> = (0..128).map(|_| inputs.chunks_exact(matrix.cols).take(1).collect()).collect();
            let seconds = best(&|| drop(gpu.experts_fast(matrix.raw_parts(), shape, stack, &none, slot).unwrap()));
            eprintln!("  of which unpacking the weights (one token each): {:.0} ms", seconds * 1e3);
        }
    }

    /// Kernel throughput at prompt-like sizes: `cargo test --release --lib gpu::tests::bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench() {
        use std::time::Instant;
        let gpu = Gpu::new(&[], true).unwrap();
        let (rows, cols, vectors) = (3072, 2304, 512);
        let weights: Vec<u8> =
            bytes(rows * cols * 2, 1).as_chunks::<2>().0.iter().flat_map(|b| [b[0], 0x3C | (b[1] & 0x83)]).collect();
        let xs = floats(vectors * cols, 2);
        let job = [DenseJob { format: Format::Bf16, weights: &weights, rows, cols, xs: &xs }];
        gpu.dense(&job).unwrap();
        let started = Instant::now();
        let out = gpu.dense(&job).unwrap().remove(0);
        let seconds = started.elapsed().as_secs_f64();
        let products = (rows * cols * vectors) as f64;
        eprintln!("dense: {seconds:.3}s, {:.1} G products/s", products / seconds / 1e9);
        let (v, r) = (vectors - 1, rows - 1);
        assert_eq!(out[v * rows + r], dot_bf16_row(&weights[r * cols * 2..][..cols * 2], &xs[v * cols..][..cols]));

        let (experts, rows, cols, tokens) = (16, 2304, 2304, 256);
        let values = bytes(experts * rows * cols / 2, 3);
        let scales: Vec<u8> =
            bytes(experts * rows * cols / 16, 4).into_iter().map(|b| if b & 0x7F == 0x7F { b & 0xF0 } else { b }).collect();
        let activations: Vec<Nvfp4Activation> = (0..tokens as u64).map(|seed| quantize_nvfp4(&floats(cols, 10 + seed))).collect();
        let refs: Vec<&Nvfp4Activation> = activations.iter().collect();
        // Sorted by expert, as the model submits them.
        let mut pairs: Vec<(usize, usize, f32)> = (0..tokens * 8).map(|i| (i % experts, i / 8, 0.011)).collect();
        pairs.sort_by_key(|pair| pair.0);
        let job = ExpertJob { values: &values, scales: &scales, rows, cols, activations: &refs, pairs: &pairs };
        gpu.experts(&job).unwrap();
        let started = Instant::now();
        let out = gpu.experts(&job).unwrap();
        let seconds = started.elapsed().as_secs_f64();
        let products = (rows * cols * pairs.len()) as f64;
        eprintln!("experts: {seconds:.3}s, {:.1} G products/s", products / seconds / 1e9);
        let (p, r) = (pairs.len() - 1, rows - 1);
        let row = pairs[p].0 * rows + r;
        let cpu = dot_nvfp4_row_quantized(
            &values[row * cols / 2..][..cols / 2],
            &scales[row * cols / 16..][..cols / 16],
            pairs[p].2,
            &activations[pairs[p].1],
        );
        assert_eq!(out[p * rows + r], half::bf16::from_f32(cpu).to_f32());
    }

    #[test]
    fn the_fast_dense_path_matches_to_rounding() {
        let gpu = Gpu::new(&[], true).unwrap();
        let (rows, cols, vectors) = (40, 2304, 70);
        let weights: Vec<u8> =
            bytes(rows * cols * 2, 5).as_chunks::<2>().0.iter().flat_map(|b| [b[0], 0x3C | (b[1] & 0x83)]).collect();
        let xs = floats(vectors * cols, 6);
        let job = [DenseJob { format: Format::Bf16, weights: &weights, rows, cols, xs: &xs }];
        let (fast, exact) = (gpu.dense_fast(&job).unwrap().remove(0), gpu.dense(&job).unwrap().remove(0));
        // Compared with the size of the terms summed: a result near zero from cancellation is still that accurate.
        for v in 0..vectors {
            for r in 0..rows {
                let row = weights[r * cols * 2..][..cols * 2].as_chunks::<2>().0;
                let magnitude: f32 = row.iter().zip(&xs[v * cols..]).map(|(w, x)| (half::bf16::from_le_bytes(*w).to_f32() * x).abs()).sum();
                let (a, b) = (fast[v * rows + r], exact[v * rows + r]);
                assert!((a - b).abs() <= 1e-5 * magnitude, "vector {v}, row {r}: {a} against {b}");
            }
        }
    }

    #[test]
    fn the_fast_expert_path_matches_to_rounding() {
        let gpu = Gpu::new(&[], true).unwrap();
        let (experts, rows, cols) = (6, 24, 2304);
        let values = bytes(experts * rows * cols / 2, 7);
        let scales: Vec<u8> =
            bytes(experts * rows * cols / 16, 8).into_iter().map(|b| if b & 0x7F == 0x7F { b & 0xF0 } else { b }).collect();
        // Very uneven counts, one expert with none: several groups, with padding inside them.
        let counts = [3usize, 40, 0, 7, 21, 1];
        let activations: Vec<Vec<Nvfp4Activation>> = counts
            .iter()
            .enumerate()
            .map(|(e, &n)| (0..n as u64).map(|i| quantize_nvfp4(&floats(cols, 100 * e as u64 + i))).collect())
            .collect();
        let unscaled: Vec<Vec<Vec<f32>>> =
            activations.iter().map(|list| list.iter().map(Nvfp4Activation::unscaled).collect()).collect();
        let vectors: Vec<Vec<&[f32]>> = unscaled.iter().map(|list| list.iter().map(Vec::as_slice).collect()).collect();
        let out = gpu.experts_fast((&values, &scales), (experts, rows, cols), Stack::Nvfp4, &vectors, 0).unwrap();
        let mut decoded = vec![0.0f32; cols];
        for (expert, list) in activations.iter().enumerate() {
            assert_eq!(out[expert].len(), list.len() * rows);
            for (i, activation) in list.iter().enumerate() {
                for r in 0..rows {
                    let row = expert * rows + r;
                    // The exact kernel with outer weight scale 1, without the activation's outer scale.
                    let exact = dot_nvfp4_row_quantized(
                        &values[row * cols / 2..][..cols / 2],
                        &scales[row * cols / 16..][..cols / 16],
                        1.0,
                        activation,
                    ) / activation.outer_scale();
                    let fast = out[expert][i * rows + r];
                    // Compared with the size of the terms summed, which nearly cancel for random weights.
                    crate::quant::decode_nvfp4_row(
                        &values[row * cols / 2..][..cols / 2],
                        &scales[row * cols / 16..][..cols / 16],
                        1.0,
                        &mut decoded,
                    );
                    let magnitude: f32 = decoded.iter().zip(&unscaled[expert][i]).map(|(w, x)| (w * x).abs()).sum();
                    assert!((fast - exact).abs() <= 1e-5 * magnitude, "expert {expert}, input {i}, row {r}: {fast} against {exact}");
                }
            }
        }
    }

    /// Checks `fast` products of `x` against the rows decoded on the CPU.
    fn assert_close(format: Format, rows: &[u8], cols: usize, x: &[f32], fast: &[f32], what: &str) {
        let row_bytes = format.row_bytes(cols).unwrap();
        let mut decoded = vec![0.0f32; cols];
        for (r, (row, fast)) in rows.chunks_exact(row_bytes).zip(fast).enumerate() {
            format.decode_row(row, &mut decoded);
            let exact = crate::quant::dot_f32_row(&decoded, x);
            let magnitude: f32 = decoded.iter().zip(x).map(|(w, x)| (w * x).abs()).sum();
            assert!((fast - exact).abs() <= 1e-5 * magnitude, "{format:?} {what}, row {r}: {fast} against {exact}");
        }
    }

    #[test]
    fn the_fast_attention_matches_plain_sums() {
        let gpu = Gpu::new(&[], true).unwrap();
        // Five tokens at positions 40..45, three query heads to a key head, two key heads.
        let (count, share, dim, d_rel, extent, window) = (5usize, 3usize, 16usize, 4usize, 12usize, 9usize);
        let (rows, first) = (count * share, 40usize);
        let profiles = floats(d_rel * extent, 1);
        for local in [false, true] {
            // The table starts at the oldest key the first token sees.
            let lowest = if local { first + 1 - window } else { 0 };
            let keys = first + count - lowest;
            let data: Vec<_> = (0..2u64)
                .map(|h| (floats(rows * dim, 10 + h), floats(keys * dim, 20 + h), floats(keys * dim, 30 + h), floats(rows * d_rel, 40 + h), floats(rows, 50 + h)))
                .collect();
            let heads: Vec<AttendHead> = data
                .iter()
                .map(|(queries, keys, values, rel, sinks)| AttendHead { queries, keys, values, rel, sinks })
                .collect();
            let shape = AttendShape { rows, keys, dim, d_rel, extent, profiles: &profiles, share, first, lowest, local, window, scaling: 0.25 };
            let out = gpu.attend_fast(&heads, &shape).unwrap();
            for (h, (queries, table, values, rel, sinks)) in data.iter().enumerate() {
                for row in 0..rows {
                    let t = first + row / share;
                    let from = if local { t + 1 - window } else { 0 };
                    let query = &queries[row * dim..][..dim];
                    let scores: Vec<f32> = (from..=t)
                        .map(|s| {
                            let key = &table[(s - lowest) * dim..][..dim];
                            let bias: f32 = match t - s < extent {
                                true => (0..d_rel).map(|r| rel[row * d_rel + r] * profiles[r * extent + (t - s)]).sum(),
                                false => 0.0,
                            };
                            query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * 0.25 + bias
                        })
                        .collect();
                    let highest = scores.iter().copied().fold(sinks[row], f32::max);
                    let total: f32 = scores.iter().map(|s| (s - highest).exp()).sum::<f32>() + (sinks[row] - highest).exp();
                    for c in 0..dim {
                        let expected: f32 = (from..=t)
                            .zip(&scores)
                            .map(|(s, score)| (score - highest).exp() / total * values[(s - lowest) * dim + c])
                            .sum();
                        let got = out[h][row * dim + c];
                        assert!((got - expected).abs() < 1e-4, "local {local}, head {h}, row {row}, {c}: {got} against {expected}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_whole_expert_step_matches_plain_sums() {
        use crate::ggml::encode_q8_0;
        let gpu = Gpu::new(&[], true).unwrap();
        let (experts, hidden, units, chosen, tokens) = (5usize, 64usize, 32usize, 2usize, 300usize);
        let decoded = |rows: usize, cols: usize, seed: u64| -> (Vec<u8>, Vec<f32>) {
            let bytes: Vec<u8> = floats(rows * cols, seed).chunks_exact(cols).flat_map(encode_q8_0).collect();
            let mut values = vec![0.0f32; rows * cols];
            for (row, values) in bytes.chunks_exact(cols / 32 * 34).zip(values.chunks_exact_mut(cols)) {
                Format::Q8_0.decode_row(row, values);
            }
            (bytes, values)
        };
        let (gate_up, gate_up_values) = decoded(experts * 2 * units, hidden, 1);
        let (down, down_values) = decoded(experts * hidden, units, 2);
        let x = floats(tokens * hidden, 3);
        // Most tokens to one expert, so that it takes more than one run of rows, and one expert unused.
        let routes: Vec<(u32, f32)> = (0..tokens).flat_map(|t| [(0u32, 0.75f32), (1 + (t % 3) as u32, 0.25 + t as f32 / 1000.0)]).collect();
        let job = WholeExperts {
            gate_up: (&gate_up, Format::Q8_0),
            down: (&down, Format::Q8_0),
            experts,
            hidden,
            units,
            chosen,
            x: &x,
            routes: &routes,
            gate_cap: 4.0,
            up_cap: 25.0,
        };
        let out = gpu.experts_whole(&job).unwrap();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
        for t in [0, 1, 255, 256, 257, tokens - 1] {
            let input = &x[t * hidden..][..hidden];
            let mut expected = vec![0.0f32; hidden];
            for &(expert, weight) in &routes[t * chosen..][..chosen] {
                let rows = &gate_up_values[expert as usize * 2 * units * hidden..][..2 * units * hidden];
                let active: Vec<f32> = (0..units)
                    .map(|u| {
                        let (gate, up) = (dot(&rows[u * hidden..][..hidden], input), dot(&rows[(units + u) * hidden..][..hidden], input));
                        (gate / 4.0).tanh() * 4.0 / (1.0 + (-gate).exp()) * (up / 25.0).tanh() * 25.0
                    })
                    .collect();
                let rows = &down_values[expert as usize * hidden * units..][..hidden * units];
                for (c, expected) in expected.iter_mut().enumerate() {
                    *expected += weight * dot(&rows[c * units..][..units], &active);
                }
            }
            for (c, expected) in expected.iter().enumerate() {
                let got = out[t * hidden + c];
                assert!((got - expected).abs() < 1e-3 * (1.0 + expected.abs()), "token {t}, {c}: {got} against {expected}");
            }
        }
    }

    #[test]
    fn the_fast_gates_match_plain_sums() {
        let gpu = Gpu::new(&[], true).unwrap();
        let (tokens, width, rank, hidden) = (7usize, 48usize, 6usize, 20usize);
        let (pairs, down, forget, gain) = (floats(tokens * width, 1), floats(rank * width, 2), floats(hidden * rank, 3), floats(hidden * rank, 4));
        let raw = |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
        let (forgets, gains) = gpu.gates_fast(&pairs, &raw(&down), &raw(&forget), &raw(&gain), (rank, hidden)).unwrap();
        for t in 0..tokens {
            let low: Vec<f32> = (0..rank).map(|r| (0..width).map(|c| down[r * width + c] * pairs[t * width + c]).sum()).collect();
            for i in 0..hidden {
                let expected: f32 = (0..rank).map(|r| forget[i * rank + r] * low[r]).sum();
                assert!((forgets[t * hidden + i] - expected).abs() < 1e-4, "forget, token {t}, {i}");
                let expected: f32 = (0..rank).map(|r| gain[i * rank + r] * low[r]).sum();
                assert!((gains[t * hidden + i] - expected).abs() < 1e-4, "gain, token {t}, {i}");
            }
        }
    }

    #[test]
    fn the_fast_paths_unpack_gguf_blocks() {
        use crate::ggml::random_row;
        let gpu = Gpu::new(&[], true).unwrap();
        for (f, format) in [Format::Q4K, Format::Q5_0, Format::Q6K, Format::Q8_0].into_iter().enumerate() {
            // A dense matrix.
            let (rows, cols, vectors) = (37, 768, 5);
            let weights: Vec<u8> = (0..rows).flat_map(|r| random_row(format, cols, (100 * f + r) as u32)).collect();
            let xs = floats(vectors * cols, 40 + f as u64);
            let out = gpu.dense_fast(&[DenseJob { format, weights: &weights, rows, cols, xs: &xs }]).unwrap().remove(0);
            for (i, x) in xs.chunks_exact(cols).enumerate() {
                assert_close(format, &weights, cols, x, &out[i * rows..][..rows], "dense");
            }

            // A stack of experts with uneven numbers of inputs.
            let (experts, rows, cols) = (5, 12, 512);
            let stack: Vec<u8> =
                (0..experts * rows).flat_map(|r| random_row(format, cols, (1000 * f + r) as u32)).collect();
            let counts = [3usize, 0, 11, 1, 6];
            let inputs: Vec<Vec<Vec<f32>>> = counts
                .iter()
                .enumerate()
                .map(|(e, &n)| (0..n as u64).map(|i| floats(cols, 300 * e as u64 + i)).collect())
                .collect();
            let vectors: Vec<Vec<&[f32]>> = inputs.iter().map(|list| list.iter().map(Vec::as_slice).collect()).collect();
            let out = gpu.experts_fast((&stack, &[]), (experts, rows, cols), Stack::Ggml(format), &vectors, 0).unwrap();
            let matrix_bytes = rows * format.row_bytes(cols).unwrap();
            for (expert, list) in inputs.iter().enumerate() {
                assert_eq!(out[expert].len(), list.len() * rows);
                for (i, x) in list.iter().enumerate() {
                    let matrix = &stack[expert * matrix_bytes..][..matrix_bytes];
                    assert_close(format, matrix, cols, x, &out[expert][i * rows..][..rows], "experts");
                }
            }
        }
    }

    /// The expert stacks of a long prompt: a hundred matrices of the model's size and thousands of inputs.
    /// `cargo test --release --lib the_fast_path_holds_at_full_size -- --ignored`.
    #[test]
    #[ignore]
    fn the_fast_path_holds_at_full_size() {
        use crate::ggml::random_row;
        let gpu = Gpu::new(&[], true).unwrap();
        let (format, experts, rows, cols) = (Format::Q4K, 100, 2304, 2304);
        let stack: Vec<u8> = (0..experts * rows).flat_map(|r| random_row(format, cols, r as u32)).collect();
        // The numbers of inputs one layer of a 2,048-token pass gave its busiest experts.
        let counts: Vec<usize> = [(5, 1839), (6, 746), (4, 282), (6, 138), (9, 67), (30, 20), (40, 3)]
            .iter()
            .flat_map(|&(experts, inputs)| std::iter::repeat_n(inputs, experts))
            .collect();
        assert_eq!(counts.len(), experts);
        let inputs: Vec<Vec<Vec<f32>>> = counts
            .iter()
            .enumerate()
            .map(|(e, &n)| (0..n as u64).map(|i| floats(cols, 1000 * e as u64 + i)).collect())
            .collect();
        let vectors: Vec<Vec<&[f32]>> = inputs.iter().map(|list| list.iter().map(Vec::as_slice).collect()).collect();
        let out = gpu.experts_fast((&stack, &[]), (experts, rows, cols), Stack::Ggml(format), &vectors, 0).unwrap();
        for (expert, out) in out.iter().enumerate() {
            let bad: Vec<usize> =
                out.chunks_exact(rows).enumerate().filter(|(_, r)| r.iter().any(|v| !v.is_finite())).map(|(i, _)| i).collect();
            assert!(bad.is_empty(), "expert {expert} ({} inputs): bad {:?}..{:?} ({})", counts[expert], bad.first(), bad.last(), bad.len());
        }
        let matrix_bytes = rows * format.row_bytes(cols).unwrap();
        for expert in [0, 1, 49, 50, 98, 99] {
            let list = &inputs[expert];
            for i in [0, list.len() - 1] {
                let matrix = &stack[expert * matrix_bytes..][..matrix_bytes];
                assert_close(format, matrix, cols, &list[i], &out[expert][i * rows..][..rows], &format!("expert {expert}"));
            }
        }
    }

    #[test]
    fn the_fast_mxfp8_path_matches_to_rounding() {
        let gpu = Gpu::new(&[], true).unwrap();
        let (experts, rows, cols) = (4, 20, 1152);
        let values: Vec<u8> =
            bytes(experts * rows * cols, 9).into_iter().map(|b| if b & 0x7F == 0x7F { b & 0xF0 } else { b }).collect();
        // Block exponents near 127 keep the products in range; one zero byte exercises the subnormal scale.
        let mut scales: Vec<u8> = bytes(experts * rows * cols / 32, 10).iter().map(|b| 115 + b % 20).collect();
        scales[5] = 0;
        let counts = [2usize, 0, 9, 5];
        let inputs: Vec<Vec<Vec<f32>>> = counts
            .iter()
            .enumerate()
            .map(|(e, &n)| (0..n as u64).map(|i| floats(cols, 200 * e as u64 + i)).collect())
            .collect();
        let vectors: Vec<Vec<&[f32]>> = inputs.iter().map(|list| list.iter().map(Vec::as_slice).collect()).collect();
        let out = gpu.experts_fast((&values, &scales), (experts, rows, cols), Stack::Mxfp8, &vectors, 0).unwrap();
        let mut decoded = vec![0.0f32; cols];
        for (expert, list) in inputs.iter().enumerate() {
            assert_eq!(out[expert].len(), list.len() * rows);
            for (i, x) in list.iter().enumerate() {
                for r in 0..rows {
                    let row = expert * rows + r;
                    crate::quant::decode_mxfp8_row(&values[row * cols..][..cols], &scales[row * cols / 32..][..cols / 32], &mut decoded);
                    let exact = crate::quant::dot_f32_row(&decoded, x);
                    let magnitude: f32 = decoded.iter().zip(x).map(|(w, x)| (w * x).abs()).sum();
                    let fast = out[expert][i * rows + r];
                    assert!((fast - exact).abs() <= 1e-5 * magnitude, "expert {expert}, input {i}, row {r}: {fast} against {exact}");
                }
            }
        }
    }

    #[test]
    fn experts_match_the_cpu_kernel_exactly() {
        let gpu = Gpu::new(&[], true).unwrap();
        let (experts, rows, cols) = (3, 40, 2304);
        let values = bytes(experts * rows * cols / 2, 3);
        let scales: Vec<u8> =
            bytes(experts * rows * cols / 16, 4).into_iter().map(|b| if b & 0x7F == 0x7F { b & 0xF0 } else { b }).collect();
        let activations: Vec<Nvfp4Activation> = (0..5).map(|seed| quantize_nvfp4(&floats(cols, 10 + seed))).collect();
        let refs: Vec<&Nvfp4Activation> = activations.iter().collect();
        // Runs of the same expert shorter than, equal to and longer than one thread's share.
        let pairs = [
            (0, 0, 0.011f32),
            (0, 1, 0.011),
            (1, 3, 0.017),
            (2, 0, 0.02),
            (2, 1, 0.02),
            (2, 2, 0.02),
            (2, 3, 0.02),
            (2, 4, 0.02),
            (0, 4, 0.011),
        ];
        let job = ExpertJob { values: &values, scales: &scales, rows, cols, activations: &refs, pairs: &pairs };
        let out = gpu.experts(&job).unwrap();
        for (p, &(expert, activation, outer)) in pairs.iter().enumerate() {
            for r in 0..rows {
                let row = expert * rows + r;
                let cpu = dot_nvfp4_row_quantized(
                    &values[row * cols / 2..][..cols / 2],
                    &scales[row * cols / 16..][..cols / 16],
                    outer,
                    &activations[activation],
                );
                let rounded = half::bf16::from_f32(cpu).to_f32();
                assert_eq!(out[p * rows + r], rounded, "pair {p}, row {r}");
            }
        }
    }
}
