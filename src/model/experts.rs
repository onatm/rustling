//! The expert layers: routing each token to its experts and running them, in each arithmetic and on each device.

use super::*;

impl<'a> Model<'a> {
    /// Each token's experts and the weights of their outputs, the highest-ranked first: `num_experts_per_tok`
    /// entries per token.
    pub(super) fn routes(&self, layer: &Layer, x: &Seq) -> Vec<(u32, f32)> {
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
    pub(super) fn experts(&self, layer: &Layer, x: &Seq) -> Seq {
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
    pub(super) fn experts_on_gpu(&self, layer: &Layer, x: &Seq, routes: &[(u32, f32)]) -> Option<Seq> {
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
            scales: [experts.gate_up.raw_parts().1, experts.down.raw_parts().1],
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
    pub(super) fn expert_outputs_float(&self, layer: &Layer, x: &Seq, tokens_of: &[Vec<(usize, f32)>]) -> Vec<(usize, Vec<f32>)> {
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
    pub(super) fn expert_outputs_ggml(&self, layer: &Layer, x: &Seq, tokens_of: &[Vec<(usize, f32)>]) -> Vec<(usize, Vec<f32>)> {
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
                        let scale = matrix.outer_scale(expert);
                        outs.iter_mut().zip(&dots).for_each(|(out, dot)| out[r] = *dot * scale);
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
                    .enumerate()
                    .map(|(expert, products)| {
                        let scale = experts.gate_up.outer_scale(expert);
                        let mut acts = Vec::with_capacity(products.len() / 2);
                        for products in products.chunks_exact(rows) {
                            let (mut gate, mut up): (Vec<f32>, Vec<f32>) =
                                (0..inter).map(|unit| experts.gate_up_rows(unit)).map(|(g, u)| (products[g]*scale, products[u]*scale)).unzip();
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
                        outputs.push((t, product.iter().map(|v| v * experts.down.outer_scale(expert) * weight).collect()));
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
    pub(super) fn expert_outputs_quantised(
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

/// `gate[i] = softcap(gate[i]) * sigmoid(gate[i]) * softcap(up[i]) * weight`, the experts' activation, over whole
/// vectors, with vectorised `tanh` and `exp`. `up` is used as working space.
pub(super) fn capped_swiglu(gate: &mut [f32], up: &mut [f32], gate_cap: f32, up_cap: f32, weight: f32) {
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
