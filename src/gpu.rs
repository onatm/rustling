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
    e4m3: Buffer,
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
    pub fn new(maps: &[&[u8]]) -> Result<Self> {
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

        let mut gpu = Self {
            e4m3: copied(&device, &E4M3[..])?,
            device,
            queue,
            dense,
            experts,
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
        let gpu = Gpu::new(&[]).unwrap();
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

    /// Kernel throughput at prompt-like sizes: `cargo test --release --lib gpu::tests::bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench() {
        use std::time::Instant;
        let gpu = Gpu::new(&[]).unwrap();
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
    fn experts_match_the_cpu_kernel_exactly() {
        let gpu = Gpu::new(&[]).unwrap();
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
