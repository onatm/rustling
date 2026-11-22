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
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions,
    MTLSize,
};

use crate::quant::{E4M3, Nvfp4Activation};

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

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

/// One dense product: `rows x cols` bfloat16 weights applied to each `cols`-wide vector in `xs`.
pub struct DenseJob<'a> {
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

        let mut gpu = Self {
            e4m3: copied(&device, &E4M3[..])?,
            device,
            queue,
            dense,
            experts,
            widen_bf16,
            widen_nvfp4,
            widen_mxfp8,
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
        let commands = self.queue.commandBuffer().context("creating a command buffer")?;
        let mut outputs = Vec::new();
        for job in jobs {
            let vectors = job.xs.len() / job.cols;
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
    }

    /// Returns `pairs * rows` bfloat16-rounded products, pair by pair.
    pub fn experts(&self, job: &ExpertJob) -> Result<Vec<f32>> {
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
    }

    /// Whether large batches may take the fast, inexact path.
    pub fn fast(&self) -> bool {
        self.fast
    }

    /// A working buffer of at least `bytes`, kept under `slot` for reuse. Its contents are whatever the last use
    /// left there.
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
        let commands = self.queue.commandBuffer().context("creating a command buffer")?;
        let mut outputs = Vec::new();
        for (index, job) in jobs.iter().enumerate() {
            let vectors = job.xs.len() / job.cols;
            ensure!(job.weights.len() == job.rows * job.cols * 2 && job.xs.len() == vectors * job.cols, "bad dense job");
            ensure!(job.cols % 16 == 0, "the widening kernel takes 16 columns per step");
            let (weights, offset) = self.weights(job.weights)?;
            ensure!(offset % 4 == 0, "weights are not word-aligned in the checkpoint");
            let widened = self.scratch(2 * index, job.rows * job.cols * 4)?;
            let params = [job.rows as u32, job.cols as u32, 0];
            self.encode(&commands, &self.widen_bf16, &[(&weights, offset), (&widened, 0)], &params, (job.cols / 8, job.rows))?;
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
    }

    /// Each expert of a stack applied to its own vectors, on the fast path.
    ///
    /// `values` and `scales` hold `experts` matrices of `rows x cols`, in NVFP4 or MXFP8 as `nvfp4` says.
    /// `vectors[expert]` are that expert's inputs, each `cols` wide. Returns, per expert, its results end to end
    /// (`rows` values per input). For NVFP4 that is without the expert's outer scale, which the caller applies.
    /// `slot` (0 or 1) keeps a layer's two stacks apart.
    ///
    /// Experts get very different numbers of inputs, and one tuned product wants the same number for every matrix
    /// in it. So the experts are sorted by how many they have and multiplied in groups of similar size, each
    /// padded with zeros to its largest member.
    pub fn experts_fast(
        &self,
        (values, scales): (&[u8], &[u8]),
        (experts, rows, cols): (usize, usize, usize),
        nvfp4: bool,
        vectors: &[Vec<&[f32]>],
        slot: usize,
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(vectors.len() == experts, "one list of vectors per expert");
        // NVFP4 packs two values per byte in blocks of 16; MXFP8 is a byte per value in blocks of 32.
        let (values_per_byte, block, widen) = if nvfp4 { (2, 16, &self.widen_nvfp4) } else { (1, 32, &self.widen_mxfp8) };
        let weights = experts * rows * cols;
        ensure!(values.len() == weights / values_per_byte && scales.len() == weights / block, "bad expert stack");
        ensure!(cols % 32 == 0 && vectors.iter().flatten().all(|v| v.len() == cols), "bad expert inputs");
        let mut order: Vec<usize> = (0..experts).filter(|&e| !vectors[e].is_empty()).collect();
        let mut outputs = vec![Vec::new(); experts];
        if order.is_empty() {
            return Ok(outputs);
        }
        order.sort_by_key(|&e| std::cmp::Reverse(vectors[e].len()));

        // Groups of consecutive experts in that order, as (first position, experts, vectors each). An expert joins
        // a group while it has at least half as many vectors as the group's first, so padding at most doubles
        // the work.
        let mut groups: Vec<(usize, usize, usize)> = Vec::new();
        for (position, &expert) in order.iter().enumerate() {
            match groups.last_mut() {
                Some((_, count, most)) if vectors[expert].len() * 2 >= *most => *count += 1,
                _ => groups.push((position, 1, vectors[expert].len())),
            }
        }

        let (values_buffer, values_offset) = self.weights(values)?;
        let (scales_buffer, scales_offset) = self.weights(scales)?;
        ensure!(values_offset % 4 == 0, "expert weights are not word-aligned in the checkpoint");
        // Slots after the dense ones.
        let padded: usize = groups.iter().map(|&(_, count, most)| count * most).sum();
        let widened = self.scratch(16 + slot, order.len() * rows * cols * 4)?;
        let x = self.scratch(18 + slot, padded * cols * 4)?;
        let out = self.scratch(20 + slot, padded * rows * 4)?;

        // Inputs group by group, expert by expert, zero where an expert has fewer than its group's most.
        // SAFETY: the buffer is shared-mode, at least `padded * cols` floats long, and nothing else uses it until
        // the command buffer below is committed.
        let inputs = unsafe { std::slice::from_raw_parts_mut(x.contents().as_ptr() as *mut f32, padded * cols) };
        let mut rest = inputs;
        for &(first, count, most) in &groups {
            for &expert in &order[first..first + count] {
                let (mine, after) = rest.split_at_mut(most * cols);
                for (slot, vector) in mine.chunks_exact_mut(cols).zip(vectors[expert].iter().map(Some).chain(std::iter::repeat(None))) {
                    match vector {
                        Some(vector) => slot.copy_from_slice(vector),
                        None => slot.fill(0.0),
                    }
                }
                rest = after;
            }
        }

        let commands = self.queue.commandBuffer().context("creating a command buffer")?;
        let order_buffer = copied(&self.device, &order.iter().map(|&e| e as u32).collect::<Vec<u32>>())?;
        let buffers = [
            (&values_buffer, values_offset),
            (&scales_buffer, scales_offset),
            (&self.e4m3, 0),
            (&order_buffer, 0),
            (&widened, 0),
        ];
        let params = [rows as u32, cols as u32, order.len() as u32];
        self.encode(&commands, widen, &buffers, &params, (cols / block, order.len() * rows))?;
        let mut offset = 0;
        for &(first, count, most) in &groups {
            self.multiply(
                &commands,
                (&x, offset * cols * 4, most, cols),
                (&widened, first * rows * cols * 4, rows, cols),
                (&out, offset * rows * 4),
                count,
            );
            offset += count * most;
        }
        commands.commit();
        commands.waitUntilCompleted();
        finish(&commands)?;

        // SAFETY: as for `inputs`; the command buffer that wrote it has completed.
        let results = unsafe { std::slice::from_raw_parts(out.contents().as_ptr() as *const f32, padded * rows) };
        let mut offset = 0;
        for &(first, count, most) in &groups {
            for &expert in &order[first..first + count] {
                outputs[expert] = results[offset * rows..][..vectors[expert].len() * rows].to_vec();
                offset += most;
            }
        }
        Ok(outputs)
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
        use objc2::AnyThread;
        use objc2_metal_performance_shaders::{MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};
        let ((left, left_offset, vectors, cols), (right, right_offset, rows, _)) = (left, right);
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
                MPSMatrixMultiplication::alloc(), &self.device, false, true, vectors, rows, cols, 1.0, 0.0,
            );
            kernel.setBatchSize(count);
            kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                commands,
                &matrix(left, left_offset, vectors, cols),
                &matrix(right, right_offset, rows, cols),
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
            let out = gpu.dense(&[DenseJob { weights: &weights, rows, cols, xs: &xs }]).unwrap().remove(0);
            for v in 0..vectors {
                for r in 0..rows {
                    let cpu = dot_bf16_row(&weights[r * cols * 2..][..cols * 2], &xs[v * cols..][..cols]);
                    assert_eq!(out[v * rows + r], cpu, "{rows}x{cols}: vector {v}, row {r}");
                }
            }
        }
    }

    /// What a whole token's matrix products cost on the GPU from a GGUF file, in one command buffer, against the
    /// CPU's results: `cargo test --release --lib gpu::tests::gguf_token -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gguf_token() {
        use crate::checkpoint::{Checkpoint, resolve_model};
        use crate::ggml::Format;
        use std::time::Instant;
        let path = resolve_model(Some(std::path::Path::new("lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M"))).unwrap();
        let weights = Checkpoint::open(&path).unwrap();
        let gpu = Gpu::new(&weights.maps(), false).unwrap();
        let options = MTLCompileOptions::new();
        let library = gpu
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(include_str!("gguf.metal")), Some(&options))
            .map_err(|e| e.localizedDescription().to_string())
            .unwrap();
        let pipeline = |name: &str| {
            let function = library.newFunctionWithName(&NSString::from_str(name)).unwrap();
            gpu.device.newComputePipelineStateWithFunction_error(&function).unwrap()
        };
        let kernels = [pipeline("mv_q4k"), pipeline("mv_q5_0"), pipeline("mv_q6k"), pipeline("mv_q8_0")];
        let kernel = |format: Format| match format {
            Format::Q4K => &kernels[0],
            Format::Q5_0 => &kernels[1],
            Format::Q6K => &kernels[2],
            Format::Q8_0 => &kernels[3],
            other => panic!("{other:?}"),
        };
        let x = floats(8 * 3072, 5);
        let x_buffer = copied(&gpu.device, &x).unwrap();
        // (name, rows, cols, matrices)
        let mut jobs: Vec<(String, usize, usize, usize)> = Vec::new();
        for visit in 0..48 {
            let l = visit % 24;
            let p = format!("model.llm.layers.{l}");
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
            jobs.push((format!("{p}.mlp.experts.w13_weight"), 4608, 2304, 8));
            jobs.push((format!("{p}.mlp.experts.w2_weight"), 2304, 1152, 8));
        }
        jobs.push(("model.llm.unembed.weight".to_owned(), 200_058, 2304, 1));
        let out = empty(&gpu.device, 200_058 * 4).unwrap();
        let prepared: Vec<_> = jobs
            .iter()
            .map(|(name, rows, cols, matrices)| {
                let (format, _, bytes) = weights.rows(name).unwrap();
                let (buffer, offset) = gpu.weights(bytes).unwrap();
                let stride = (rows * format.row_bytes(*cols).unwrap()) as u32;
                (format, buffer, offset, [*rows as u32, *cols as u32, stride], *matrices)
            })
            .collect();
        let products: usize = jobs.iter().map(|(_, rows, cols, matrices)| rows * cols * matrices).sum();

        for (label, one_encoder) in [("one encoder", true), ("an encoder per kernel", false), ("one encoder", true)] {
            let started = Instant::now();
            let commands = gpu.queue.commandBuffer().unwrap();
            let mut shared = one_encoder.then(|| commands.computeCommandEncoder().unwrap());
            for (format, buffer, offset, params, matrices) in &prepared {
                let pipeline = kernel(*format);
                let own = (!one_encoder).then(|| commands.computeCommandEncoder().unwrap());
                let encoder = shared.as_ref().or(own.as_ref()).unwrap();
                encoder.setComputePipelineState(pipeline);
                unsafe {
                    encoder.setBuffer_offset_atIndex(Some(buffer), *offset, 0);
                    encoder.setBuffer_offset_atIndex(Some(&x_buffer), 0, 1);
                    encoder.setBuffer_offset_atIndex(Some(&out), 0, 2);
                    encoder.setBytes_length_atIndex(NonNull::from(params).cast::<c_void>(), 12, 3);
                }
                let width = pipeline.threadExecutionWidth();
                encoder.dispatchThreads_threadsPerThreadgroup(
                    MTLSize { width: params[0] as usize, height: *matrices, depth: 1 },
                    MTLSize { width, height: 1, depth: 1 },
                );
                if let Some(own) = own {
                    own.endEncoding();
                }
            }
            if let Some(shared) = shared.take() {
                shared.endEncoding();
            }
            let encoded = started.elapsed().as_secs_f64();
            commands.commit();
            commands.waitUntilCompleted();
            finish(&commands).unwrap();
            let seconds = started.elapsed().as_secs_f64();
            eprintln!(
                "{label}: {} kernels, {:.2} G products in {:.1} ms ({:.1} ms encoding), GPU time {:.1} ms",
                prepared.len(),
                products as f64 / 1e9,
                seconds * 1e3,
                encoded * 1e3,
                (commands.GPUEndTime() - commands.GPUStartTime()) * 1e3,
            );
        }

        // Where the time goes, and what a tuned Q4_K kernel changes.
        let fast = pipeline("mv_q4k_fast");
        let sums: Vec<f32> = x.chunks(32).map(|c| c.iter().sum()).collect();
        let sums_buffer = copied(&gpu.device, &sums).unwrap();
        let groups: [(&str, Box<dyn Fn(&str) -> bool>, bool); 6] = [
            ("dense", Box::new(|n: &str| n.contains(".attn.")), false),
            ("dense, tuned Q4_K", Box::new(|n: &str| n.contains(".attn.")), true),
            ("gate_up", Box::new(|n: &str| n.contains("w13")), false),
            ("gate_up, tuned Q4_K", Box::new(|n: &str| n.contains("w13")), true),
            ("down", Box::new(|n: &str| n.contains("w2_")), false),
            ("logits", Box::new(|n: &str| n.contains("unembed")), false),
        ];
        for (label, wanted, tuned) in &groups {
            let mut best = f64::MAX;
            let mut check = 0.0f32;
            for _ in 0..3 {
                let commands = gpu.queue.commandBuffer().unwrap();
                let encoder = commands.computeCommandEncoder().unwrap();
                for ((name, ..), (format, buffer, offset, params, matrices)) in jobs.iter().zip(&prepared) {
                    if !wanted(name) {
                        continue;
                    }
                    let tuned = *tuned && *format == Format::Q4K;
                    let pipeline = if tuned { &fast } else { kernel(*format) };
                    encoder.setComputePipelineState(pipeline);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(buffer), *offset, 0);
                        encoder.setBuffer_offset_atIndex(Some(&x_buffer), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(&out), 0, 2);
                        encoder.setBytes_length_atIndex(NonNull::from(params).cast::<c_void>(), 12, 3);
                        encoder.setBuffer_offset_atIndex(Some(&sums_buffer), 0, 4);
                    }
                    let width = pipeline.threadExecutionWidth();
                    let rows = params[0] as usize / if tuned { 2 } else { 1 };
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: rows, height: *matrices, depth: 1 },
                        MTLSize { width, height: 1, depth: 1 },
                    );
                }
                encoder.endEncoding();
                commands.commit();
                commands.waitUntilCompleted();
                best = best.min(commands.GPUEndTime() - commands.GPUStartTime());
                check = read(&out, 4608)[4607];
            }
            eprintln!("{label}: {:.1} ms (last value {check})", best * 1e3);
        }

        // A visit's experts as their own submission: what sharing them between the CPU and the GPU would cost.
        for share in [2usize, 3, 4, 8] {
            let started = Instant::now();
            let mut gpu_time = 0.0;
            for visit in 0..48 {
                let commands = gpu.queue.commandBuffer().unwrap();
                let encoder = commands.computeCommandEncoder().unwrap();
                for (format, buffer, offset, params, _) in &prepared[visit * 8 + 6..visit * 8 + 8] {
                    let tuned = *format == Format::Q4K;
                    let pipeline = if tuned { &fast } else { kernel(*format) };
                    encoder.setComputePipelineState(pipeline);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(buffer), *offset, 0);
                        encoder.setBuffer_offset_atIndex(Some(&x_buffer), 0, 1);
                        encoder.setBuffer_offset_atIndex(Some(&out), 0, 2);
                        encoder.setBytes_length_atIndex(NonNull::from(params).cast::<c_void>(), 12, 3);
                        encoder.setBuffer_offset_atIndex(Some(&sums_buffer), 0, 4);
                    }
                    let width = pipeline.threadExecutionWidth();
                    let rows = params[0] as usize / if tuned { 2 } else { 1 };
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: rows, height: share, depth: 1 },
                        MTLSize { width, height: 1, depth: 1 },
                    );
                }
                encoder.endEncoding();
                commands.commit();
                commands.waitUntilCompleted();
                gpu_time += commands.GPUEndTime() - commands.GPUStartTime();
            }
            let wall = started.elapsed().as_secs_f64();
            eprintln!(
                "{share} experts per visit on the GPU: {:.2} ms a visit, of which the GPU worked {:.2} ms",
                wall / 48.0 * 1e3,
                gpu_time / 48.0 * 1e3
            );
        }

        // Each kernel against the CPU decoder, on real rows.
        for (name, rows, cols) in [
            ("model.llm.layers.3.attn.wq_du.weight", 3072, 2304),
            ("model.llm.layers.0.attn.wv_dv.weight", 1024, 2304),
            ("model.llm.layers.3.mlp.experts.w2_weight", 2304, 1152),
            ("model.llm.layers.21.mlp.experts.w2_weight", 2304, 1152),
        ] {
            let (format, _, bytes) = weights.rows(name).unwrap();
            let (buffer, offset) = gpu.weights(bytes).unwrap();
            let row_bytes = format.row_bytes(cols).unwrap();
            let params = [rows as u32, cols as u32, (rows * row_bytes) as u32];
            let commands = gpu.queue.commandBuffer().unwrap();
            gpu.encode(&commands, kernel(format), &[(&buffer, offset), (&x_buffer, 0), (&out, 0)], &params, (rows, 1)).unwrap();
            commands.commit();
            commands.waitUntilCompleted();
            finish(&commands).unwrap();
            let got = read(&out, rows);
            let mut decoded = vec![0.0f32; cols];
            let mut worst = 0.0f64;
            for r in [0, 1, rows / 2, rows - 1] {
                format.decode_row(&bytes[r * row_bytes..][..row_bytes], &mut decoded);
                let exact: f64 = decoded.iter().zip(&x).map(|(w, x)| *w as f64 * *x as f64).sum();
                let size: f64 = decoded.iter().zip(&x).map(|(w, x)| (*w as f64 * *x as f64).abs()).sum();
                worst = worst.max((got[r] as f64 - exact).abs() / size);
            }
            eprintln!("{format:?}: worst error {worst:.1e} of the terms' size");
            assert!(worst < 1e-5, "{format:?}");
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
        let job = [DenseJob { weights: &weights, rows, cols, xs: &xs }];
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
        let job = [DenseJob { weights: &weights, rows, cols, xs: &xs }];
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
        let out = gpu.experts_fast((&values, &scales), (experts, rows, cols), true, &vectors, 0).unwrap();
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
        let out = gpu.experts_fast((&values, &scales), (experts, rows, cols), false, &vectors, 0).unwrap();
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
