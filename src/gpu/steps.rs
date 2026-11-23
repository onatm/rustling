//! What the token decoder builds on: buffers shared with the CPU, and lists of kernel steps run in one
//! submission.

use super::*;

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
#[derive(Clone)]
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
    /// One group per block of full-attention keys and KV head, for a single token.
    KeyBlocks { rows: usize, block: usize },
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
        self.run_steps(kernels, steps.iter().map(|step| (step, tokens)), buffers, keys)
    }

    /// Like `run`, but each step can cover a different number of tokens in the same submission.
    pub(crate) fn run_steps<'s>(&self, kernels: &[&Kernels], steps: impl Iterator<Item = (&'s Step, usize)>, buffers: &[Shared], keys: [usize; 2]) -> Result<f64> {
        // Metal hands back objects that are only let go when a pool drains, and the threads this runs on
        // have none of their own: without one here, every call would leave its command buffers behind.
        objc2::rc::autoreleasepool(|_| {
            let pipelines: Vec<&Pipeline> = kernels.iter().flat_map(|kernels| &kernels.0).collect();
            let commands = self.queue.commandBuffer().context("creating a command buffer")?;
            let encoder = commands.computeCommandEncoder().context("creating a compute encoder")?;
            let wide = MTLSize { width: 32, height: 1, depth: 1 };
            for (step, tokens) in steps {
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
                    Grid::KeyBlocks { rows, block } => encoder.dispatchThreadgroups_threadsPerThreadgroup(
                        size(keys[1].div_ceil(block), rows), MTLSize { width: block, height: 1, depth: 1 },
                    ),
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
