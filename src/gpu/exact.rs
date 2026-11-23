//! The kernels that repeat the CPU's arithmetic step for step, so their results are identical to it.

use super::*;

impl Gpu {
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
}
