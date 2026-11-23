//! The fast paths for long prompts: weights widened to float on the GPU and multiplied by Apple's tuned matrix
//! product, for plain matrices, experts, attention and the controllers' merges. Results match the CPU's only
//! to rounding.

use super::*;

fn ggml_pipeline_index(format: Format) -> Result<usize> {
    Ok(match format {
        Format::Q4K => 0,
        Format::Q5_0 => 1,
        Format::Q6K => 2,
        Format::Q8_0 => 3,
        Format::Nvfp4 => 4,
        Format::Q5K => 5,
        other => anyhow::bail!("no kernel unpacks {other:?}"),
    })
}

impl Gpu {
    /// [`Gpu::dense`] on the fast path: the weights are widened to float and multiplied by Apple's tuned matrix
    /// product. Several times faster for many vectors, and equal to `dense` only up to rounding.
    ///
    /// The product takes all of a job's vectors at once, which is quickest and which Apple's routine now and
    /// then answers with NaN; the caller checks. `in_runs` takes them [`PRODUCT_RUN`] at a time instead, about a
    /// fifth slower over a long prompt and not known to fail: the way to redo a batch that came back wrong.
    pub fn dense_fast(&self, jobs: &[DenseJob], in_runs: bool) -> Result<Vec<Vec<f32>>> {
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
                // A fresh buffer is zeroed, which is the padding of the last run.
                let run = if in_runs { PRODUCT_RUN } else { vectors.max(1) };
                let padded = vectors.next_multiple_of(run);
                let x = empty(&self.device, padded * job.cols * 4)?;
                // SAFETY: the buffer is shared-mode and at least `job.xs.len()` floats long.
                unsafe { std::ptr::copy_nonoverlapping(job.xs.as_ptr(), x.contents().as_ptr() as *mut f32, job.xs.len()) };
                let out = self.scratch(2 * index + 1, padded * job.rows * 4)?;
                // out = x . widened^T: one row per vector.
                self.multiply_runs(&commands, (&x, vectors, job.cols), (&widened, 0, job.rows, job.cols), &out, run);
                outputs.push((out, vectors * job.rows));
            }
            commands.commit();
            commands.waitUntilCompleted();
            finish(&commands)?;
            Ok(outputs.into_iter().map(|(buffer, len)| read(&buffer, len)).collect())
        })
    }

    pub(super) fn widen_ggml(&self, format: Format) -> Result<&Pipeline> {
        Ok(&self.widen_ggml[ggml_pipeline_index(format)?])
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
            let run = PRODUCT_RUN;
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
    /// `hidden x rank`. Returns the two, `tokens x hidden` each, before their `tanh`. `in_runs` is as for
    /// [`Gpu::dense_fast`].
    pub fn gates_fast(
        &self,
        pairs: &[f32],
        down: &[u8],
        forget: &[u8],
        gain: &[u8],
        (rank, hidden): (usize, usize),
        in_runs: bool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
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
            let run = if in_runs { PRODUCT_RUN } else { tokens.max(1) };
            let padded = tokens.next_multiple_of(run);
            let (x, low) = (buffer(0, padded * width)?, buffer(1, padded * rank)?);
            let (forgets, gains) = (buffer(2, padded * hidden)?, buffer(3, padded * hidden)?);
            // SAFETY: the buffer is shared-mode, at least `padded * width` floats long, and nothing else uses it
            // until the submission below completes. The rows past the last token are the padding, zeroed.
            unsafe {
                let floats = x.contents().as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(pairs.as_ptr(), floats, pairs.len());
                std::ptr::write_bytes(floats.add(pairs.len()), 0, (padded - tokens) * width);
            }
            let ((down, down_at), (forget, forget_at), (gain, gain_at)) = (self.weights(down)?, self.weights(forget)?, self.weights(gain)?);
            let commands = self.queue.commandBuffer().context("creating a command buffer")?;
            self.multiply_runs(&commands, (&x, tokens, width), (&down, down_at, rank, width), &low, run);
            self.multiply_runs(&commands, (&low, tokens, rank), (&forget, forget_at, hidden, rank), &forgets, run);
            self.multiply_runs(&commands, (&low, tokens, rank), (&gain, gain_at, hidden, rank), &gains, run);
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
    /// Apple's routine always takes [`PRODUCT_RUN`] rows, padded when needed to avoid shapes that returned
    /// NaN. Very short runs multiply directly from compressed weights instead. Results are still checked.
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
            let run = PRODUCT_RUN;
            // At 3,058 prompt tokens on M4 Max this cut scheduled rows by 21% across the 48 visits.
            const DIRECT_ROWS: usize = 32;
            #[derive(Clone, Copy)]
            enum Matrix {
                Direct(usize),  // Expert ID in the compressed stack.
                Widened(usize), // Position in the compact unpacking order.
            }
            let mut order = Vec::new();
            let mut products = Vec::new();
            let mut token_of: Vec<u32> = Vec::new();
            let mut row_of = vec![0u32; tokens * chosen];
            for (expert, routed) in tokens_of.iter().enumerate() {
                let position = order.len();
                if routed.len() > DIRECT_ROWS {
                    order.push(expert as u32);
                }
                for piece in routed.chunks(run) {
                    let first = token_of.len();
                    let matrix = if piece.len() > DIRECT_ROWS { Matrix::Widened(position) } else { Matrix::Direct(expert) };
                    products.push((matrix, first, piece.len()));
                    for &(token, route) in piece {
                        row_of[route as usize] = token_of.len() as u32;
                        token_of.push(token);
                    }
                    if piece.len() > DIRECT_ROWS {
                        token_of.resize(first + run, u32::MAX);
                    }
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
            let order_buffer = (!order.is_empty()).then(|| copied(&self.device, &order)).transpose()?;

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
                let scales = job.scales[slot];
                ensure!(scales.is_empty() || scales.len() == 4*experts, "bad expert scales");
                let scale = |expert: usize| if scales.is_empty() { 1.0 } else { f32::from_le_bytes(scales[4*expert..][..4].try_into().unwrap()) };
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
                    let buffers = [(&values_buffer, values_offset), (order_buffer.as_ref().expect("nonempty order"), part * per_part * 4), (widened, 0)];
                    self.encode(&commands, self.widen_ggml(format)?, &buffers, &params, (cols.div_ceil(256), experts.len() * out_rows))?;
                }
                for &(matrix, first, count) in &products {
                    match matrix {
                        Matrix::Direct(expert) => {
                            let index = ggml_pipeline_index(format)?;
                            ensure!(out_rows % 4 == 0, "bad expert width");
                            let at = values_offset + expert * out_rows * row_bytes;
                            let buffers = [(&values_buffer, at), (from, first * cols * 4), (to, first * out_rows * 4)];
                            step(&self.tail_ggml[index], &buffers, &[out_rows as u32, cols as u32, scale(expert).to_bits()], (out_rows * 8, count))?;
                        }
                        Matrix::Widened(position) => self.product(
                            &commands,
                            (from, first * cols * 4, run, cols),
                            (&widened[position / per_part], (position % per_part) * out_rows * cols * 4, out_rows, cols),
                            (to, first * out_rows * 4),
                            1,
                            true,
                            scale(order[position] as usize) as f64,
                        ),
                    }
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
    pub(super) fn multiply(
        &self,
        commands: &ProtocolObject<dyn MTLCommandBuffer>,
        left: (&Buffer, usize, usize, usize),
        right: (&Buffer, usize, usize, usize),
        out: (&Buffer, usize),
        count: usize,
    ) {
        self.product(commands, left, right, out, count, true, 1.0);
    }

    /// Encodes `out = left . right^T` for a `left` of any number of rows, as products of `run` rows each.
    /// `left` is `(buffer, rows, cols)` and `out` a buffer, both from their starts and both long enough for the
    /// rows rounded up to a whole run; `left`'s rows past the last are the padding and should be zero.
    pub(super) fn multiply_runs(
        &self,
        commands: &ProtocolObject<dyn MTLCommandBuffer>,
        (left, rows, cols): (&Buffer, usize, usize),
        right: (&Buffer, usize, usize, usize),
        out: &Buffer,
        run: usize,
    ) {
        for from in (0..rows).step_by(run) {
            self.multiply(commands, (left, from * cols * 4, run, cols), right, (out, from * right.2 * 4), 1);
        }
    }

    /// Encodes `out = scale * left . right` (or `left . right^T` with `transposed`) for `count` matrices laid
    /// end to end in each buffer, as float. Each operand is `(buffer, byte offset, rows, cols)` per matrix.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn product(
        &self,
        commands: &ProtocolObject<dyn MTLCommandBuffer>,
        left: (&Buffer, usize, usize, usize),
        right: (&Buffer, usize, usize, usize),
        out: (&Buffer, usize),
        count: usize,
        transposed: bool,
        scale: f64,
    ) {
        self.product_as(commands, left, right, out, count, transposed, scale, false)
    }

    /// [`Gpu::product`] over 32-bit floats, or with `half` over 16-bit ones.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn product_as(
        &self,
        commands: &ProtocolObject<dyn MTLCommandBuffer>,
        left: (&Buffer, usize, usize, usize),
        right: (&Buffer, usize, usize, usize),
        out: (&Buffer, usize),
        count: usize,
        transposed: bool,
        scale: f64,
        half: bool,
    ) {
        use objc2::AnyThread;
        use objc2_metal_performance_shaders::{MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication};
        let ((left, left_offset, vectors, cols), (right, right_offset, right_rows, right_cols)) = (left, right);
        // How many columns the result has.
        let rows = if transposed { right_rows } else { right_cols };
        let size = if half { 2 } else { 4 };
        // SAFETY: the callers size each buffer for `count` matrices of the stated shape past its offset, and hold
        // the buffers until the command buffer completes.
        unsafe {
            let matrix = |buffer: &Buffer, offset: usize, r: usize, c: usize| {
                let descriptor = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
                    r, c, count, c * size, r * c * size, if half { MPSDataType::Float16 } else { MPSDataType::Float32 },
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
    pub(super) fn encode(
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
