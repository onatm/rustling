//! Attention: queries against the keys and values the state holds, on the CPU or the GPU.

use super::*;

impl<'a> Model<'a> {
    pub(super) fn attention(
        &self,
        layer: &Layer,
        q_in: &Seq,
        k_in: &Seq,
        v_in: &Seq,
        state: &mut VisitState,
        position: usize,
    ) -> Seq {
        let cfg = &self.cfg;
        let (heads, kv_heads, dim, d_rel) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.d_rel);
        let (q_dim, kv_dim, group) = (heads * dim, kv_heads * dim, heads / kv_heads);
        let tokens = q_in.len() / cfg.hidden_size;
        // Full-attention layers scale queries and relative states up logarithmically at long range.
        let tau = |t: usize| match layer.local {
            true => 1.0,
            false => 1.0 + cfg.log_scaling_alpha * ((t + 1) as f32 / cfg.log_scaling_n_floor).max(1.0).ln(),
        };

        // All input projections in one parallel region.
        let mut projected = time(Stage::Projections, || {
            apply_many(self.gpu.as_ref(), &[
                (&layer.wq, q_in),
                (&layer.wk, k_in),
                (&layer.wv, v_in),
                (&layer.wr, q_in),
                (&layer.sink, q_in),
                (&layer.gate, q_in),
            ])
        });
        let attention_started = std::time::Instant::now();
        // In the bfloat16 imitation every projection and every step after it lands on bfloat16 values, as each is
        // a separate bfloat16 tensor in transformers.
        let bf16 = self.bf16();
        projected.iter_mut().for_each(|projection| round_all(bf16, projection));
        let (gate, sinks, mut rel) = (projected.remove(5), projected.remove(4), projected.remove(3));
        let (v, k, mut q) = (projected.remove(2), projected.remove(1), projected.remove(0));
        q.chunks_exact_mut(q_dim).enumerate().for_each(|(i, q)| {
            for (head, scale) in q.chunks_exact_mut(dim).zip(layer.q_scale.chunks_exact(dim)) {
                head.iter_mut().zip(scale).for_each(|(x, s)| *x *= s);
                l2_clip(head, cfg.qk_l2_clip_threshold);
                round_all(bf16, head);
                head.iter_mut().for_each(|x| *x *= tau(position + i));
                round_all(bf16, head);
            }
        });
        let mut k = layer.k_conv.apply(&k, k_in, kv_dim, cfg, &mut state.convs[0], bf16);
        round_all(bf16, &mut k);
        k.chunks_exact_mut(dim).for_each(|head| l2_clip(head, cfg.qk_l2_clip_threshold));
        round_all(bf16, &mut k);
        let mut v = layer.v_conv.apply(&v, v_in, kv_dim, cfg, &mut state.convs[1], bf16);
        round_all(bf16, &mut v);
        v.chunks_exact_mut(dim).for_each(|head| rms_norm(head, None, cfg.rms_norm_eps));
        round_all(bf16, &mut v);
        let tracing = layer.traced && trace::enabled();
        if tracing {
            trace::record("q", &q);
            trace::record("k", &k);
            trace::record("v", &v);
        }

        // Sliding layers see the last `sliding_window_size` tokens; full layers see everything.
        let window_start = |t: usize| if layer.local { (t + 1).saturating_sub(cfg.sliding_window_size) } else { 0 };
        state.drop_stale(cfg, layer.local, position);
        state.keys.push(&k);
        state.values.push(&v);
        // Each key head's rows, oldest first.
        let k: Vec<&[f32]> = (0..kv_heads).map(|kv| state.keys.head(kv)).collect();
        let v: Vec<&[f32]> = (0..kv_heads).map(|kv| state.values.head(kv)).collect();
        let (k, v, cached_from) = (&k, &v, state.first);

        rel.chunks_exact_mut(heads * d_rel).enumerate().for_each(|(i, rel)| {
            for (head, scale) in rel.chunks_exact_mut(d_rel).zip(layer.rel_scale.chunks_exact(d_rel)) {
                head.iter_mut().zip(scale).for_each(|(x, s)| *x *= s);
                l2_clip(head, cfg.rel_l2_clip_threshold);
                round_all(bf16, head);
                head.iter_mut().for_each(|x| *x *= tau(position + i));
                round_all(bf16, head);
            }
        });

        let scaling = (dim as f32).powf(-0.5);
        let extent = cfg.rel_extent;
        let mut out = vec![0.0f32; tokens * q_dim];
        // One unit of work per (token, head): many tokens or a long context both spread across threads.
        let attend = |(unit, out): (usize, &mut [f32])| {
            // `i` indexes the new tokens, `t` and `s` are absolute positions.
            let (i, head) = (unit / heads, unit % heads);
            let t = position + i;
            let first = window_start(t);
            let kv = head / group;
            let query = &q[i * q_dim + head * dim..][..dim];
            let rel = &rel[(i * heads + head) * d_rel..][..d_rel];
            // bias[d] for backward distance d; zero beyond the relative extent.
            let mut bias = vec![0.0f32; (t + 1 - first).min(extent)];
            for (r, rel) in rel.iter().enumerate() {
                let profile = &layer.rel_proj[r * extent..][..bias.len()];
                bias.iter_mut().zip(profile).for_each(|(b, p)| *b += rel * p);
            }
            round_all(bf16, &mut bias);
            if tracing {
                // As the reference lays it out: [query token][head][key token], zero for keys after the query.
                for (distance, value) in bias.iter().enumerate() {
                    let offset = (i * heads + head) * tokens + (i - distance);
                    trace::record_at("bias", tokens * heads * tokens, offset, &[*value]);
                }
            }
            // The sink is one extra logit with a zero value: it only absorbs probability mass.
            let sink = sinks[i * heads + head];
            let keys = t + 1 - first;
            let (k, v) = (&k[kv], &v[kv]);
            let base = (first - cached_from) * dim;
            if bf16 {
                // The reference's steps one by one, each landing on bfloat16.
                let mut weights: Vec<f32> = (first..=t)
                    .map(|s| {
                        let key = &k[(s - cached_from) * dim..][..dim];
                        // The product, the scaling and the bias are three bfloat16 results in transformers.
                        let score = round_bf16(round_bf16(dot(query, key)) * scaling);
                        round_bf16(score + bias.get(t - s).copied().unwrap_or(0.0))
                    })
                    .collect();
                let max = weights.iter().copied().fold(sink, f32::max);
                weights.iter_mut().for_each(|w| *w = (*w - max).exp());
                let total = weights.iter().sum::<f32>() + (sink - max).exp();
                for (weight, s) in weights.iter().zip(first..=t) {
                    let value = &v[(s - cached_from) * dim..][..dim];
                    let p = round_bf16(weight / total);
                    out.iter_mut().zip(value).for_each(|(o, x)| *o += p * x);
                }
            } else {
                let mut weights: Vec<f32> = (0..keys)
                    .map(|j| {
                        // Key `j` is `keys - 1 - j` positions back.
                        let key = &k[base + j * dim..][..dim];
                        dot(query, key) * scaling + bias.get(keys - 1 - j).copied().unwrap_or(0.0)
                    })
                    .collect();
                let max = weights.iter().copied().fold(sink, f32::max);
                weights.iter_mut().for_each(|w| *w = (*w - max).exp());
                let total = weights.iter().sum::<f32>() + (sink - max).exp();
                for (j, weight) in weights.iter().enumerate() {
                    let p = weight / total;
                    out.iter_mut().zip(&v[base + j * dim..][..dim]).for_each(|(o, x)| *o += p * x);
                }
            }
            round_all(bf16, out);
            let gate = &gate[unit * dim..][..dim];
            out.iter_mut().zip(gate).for_each(|(o, g)| *o *= sigmoid(*g));
            round_all(bf16, out);
        };
        // The same sums in the same order, arranged around the keys instead: a block of tokens and the query
        // heads that share a key head take each stretch of keys and values together, so those are read once per
        // block and not once per token and head. With a long context that reading is most of the work.
        let attend_block = |&(block, kv): &(usize, usize)| -> Vec<f32> {
            let (i0, i1) = (block * ATTENTION_BLOCK, ((block + 1) * ATTENTION_BLOCK).min(tokens));
            let firsts: Vec<usize> = (i0..i1).map(|i| window_start(position + i)).collect();
            // One row of weights per (token, head), starting as the bias of each key.
            let mut weights: Vec<Vec<f32>> = Vec::with_capacity((i1 - i0) * group);
            for i in i0..i1 {
                let (t, first) = (position + i, firsts[i - i0]);
                let keys = t + 1 - first;
                for head in kv * group..(kv + 1) * group {
                    let rel = &rel[(i * heads + head) * d_rel..][..d_rel];
                    let mut bias = vec![0.0f32; keys.min(extent)];
                    for (r, rel) in rel.iter().enumerate() {
                        let profile = &layer.rel_proj[r * extent..][..bias.len()];
                        bias.iter_mut().zip(profile).for_each(|(b, p)| *b += rel * p);
                    }
                    let mut row = vec![0.0f32; keys];
                    bias.iter().enumerate().for_each(|(distance, value)| row[keys - 1 - distance] = *value);
                    weights.push(row);
                }
            }
            // Calls `visit(row, token, head, first key, keys)` for every stretch of keys, in key order.
            let stretches = |visit: &mut dyn FnMut(usize, usize, usize, usize, std::ops::Range<usize>)| {
                let (lowest, end) = (firsts[0], position + i1);
                for stretch in (lowest..end).step_by(ATTENTION_KEYS) {
                    for i in i0..i1 {
                        let first = firsts[i - i0];
                        let keys = stretch.max(first)..(stretch + ATTENTION_KEYS).min(position + i + 1);
                        if keys.is_empty() {
                            continue;
                        }
                        for g in 0..group {
                            visit((i - i0) * group + g, i, kv * group + g, first, keys.clone());
                        }
                    }
                }
            };
            let (k, v) = (&k[kv], &v[kv]);
            let at = |s: usize| (s - cached_from) * dim;
            stretches(&mut |row, i, head, first, keys| {
                let query = &q[i * q_dim + head * dim..][..dim];
                let weights = &mut weights[row];
                for s in keys {
                    weights[s - first] = dot(query, &k[at(s)..][..dim]) * scaling + weights[s - first];
                }
            });
            for (row, weights) in weights.iter_mut().enumerate() {
                let sink = sinks[(i0 + row / group) * heads + kv * group + row % group];
                let max = weights.iter().copied().fold(sink, f32::max);
                weights.iter_mut().for_each(|w| *w = (*w - max).exp());
                let total = weights.iter().sum::<f32>() + (sink - max).exp();
                weights.iter_mut().for_each(|w| *w /= total);
            }
            let mut out = vec![0.0f32; (i1 - i0) * group * dim];
            stretches(&mut |row, _, _, first, keys| {
                let out = &mut out[row * dim..][..dim];
                for s in keys {
                    let p = weights[row][s - first];
                    out.iter_mut().zip(&v[at(s)..][..dim]).for_each(|(o, x)| *o += p * x);
                }
            });
            for (row, out) in out.chunks_exact_mut(dim).enumerate() {
                let unit = (i0 + row / group) * heads + kv * group + row % group;
                out.iter_mut().zip(&gate[unit * dim..][..dim]).for_each(|(o, g)| *o *= sigmoid(*g));
            }
            out
        };
        // A long batch on the fast path: scores and outputs are matrix products of the block's queries and
        // weights with the keys and values, done by a tuned routine. Several times faster with a long context,
        // and equal to the loops above only to rounding.
        #[cfg(target_os = "macos")]
        let fast = self.gpu.as_ref().is_some_and(Gpu::fast) && tokens >= FAST_ATTENTION_MIN_TOKENS && !bf16 && !tracing;
        #[cfg(not(target_os = "macos"))]
        let fast = false;
        #[cfg(target_os = "macos")]
        if fast {
            use accel::Rows;
            // A stretch of tokens at a time, as many as keep the weights within bounds.
            let longest = position + tokens - window_start(position);
            let stretch = (FAST_ATTENTION_WEIGHTS / (group * longest)).clamp(1, tokens);
            // A sliding layer's rows see only a window's worth of the keys their stretch covers, and the rest of
            // each product is wasted, so its stretches are kept short.
            let stretch = if layer.local { stretch.min(FAST_LOCAL_STRETCH) } else { stretch };
            for i0 in (0..tokens).step_by(stretch) {
                let count = stretch.min(tokens - i0);
                // Every key any of these tokens can see; a row uses its own part of them.
                let lowest = window_start(position + i0);
                let keys = position + i0 + count - lowest;
                let rows = count * group;
                // One key head's share: its query heads' outputs for these tokens, before the gate.
                let attend = |kv: usize| -> Vec<f32> {
                    let from = (lowest - cached_from) * dim;
                    let mut queries = vec![0.0f32; rows * dim];
                    for (i, queries) in queries.chunks_exact_mut(group * dim).enumerate() {
                        queries.copy_from_slice(&q[(i0 + i) * q_dim + kv * group * dim..][..group * dim]);
                    }
                    let mut weights = vec![0.0f32; rows * keys];
                    let mine = Rows { values: &queries, rows, cols: dim, stride: dim };
                    let context = Rows { values: &k[kv][from..], rows: keys, cols: dim, stride: dim };
                    accel::multiply(mine, context, true, scaling, &mut weights);
                    weights.par_chunks_exact_mut(keys).enumerate().for_each(|(row, weights)| {
                        let (i, head) = (i0 + row / group, kv * group + row % group);
                        let t = position + i;
                        let visible = window_start(t) - lowest..t + 1 - lowest;
                        weights[..visible.start].fill(0.0);
                        weights[visible.end..].fill(0.0);
                        let weights = &mut weights[visible];
                        let rel = &rel[(i * heads + head) * d_rel..][..d_rel];
                        let mut bias = vec![0.0f32; weights.len().min(extent)];
                        for (r, rel) in rel.iter().enumerate() {
                            let profile = &layer.rel_proj[r * extent..][..bias.len()];
                            bias.iter_mut().zip(profile).for_each(|(b, p)| *b += rel * p);
                        }
                        weights.iter_mut().rev().zip(&bias).for_each(|(w, b)| *w += b);
                        let sink = sinks[i * heads + head];
                        let max = weights.iter().copied().fold(sink, f32::max);
                        weights.iter_mut().for_each(|w| *w -= max);
                        accel::exp_in_place(weights);
                        let total = weights.iter().sum::<f32>() + (sink - max).exp();
                        weights.iter_mut().for_each(|w| *w /= total);
                    });
                    let mut attended = vec![0.0f32; rows * dim];
                    let mine = Rows { values: &weights, rows, cols: keys, stride: keys };
                    let context = Rows { values: &v[kv][from..], rows: keys, cols: dim, stride: dim };
                    accel::multiply(mine, context, false, 1.0, &mut attended);
                    attended
                };
                // A long stretch goes to the GPU whole: the same products and softmax, an order of magnitude
                // faster once every row meets thousands of keys. Anything it gets wrong, the CPU redoes.
                let on_gpu = || -> Option<Vec<Vec<f32>>> {
                    let gpu = self.gpu.as_ref().filter(|_| rows * keys >= GPU_ATTENTION_MIN_WEIGHTS)?;
                    let from = (lowest - cached_from) * dim;
                    // Each key head's rows: for every token, the query heads that share it.
                    let gathered: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..kv_heads)
                        .into_par_iter()
                        .map(|kv| {
                            let (mut queries, mut rels, mut row_sinks) = (Vec::new(), Vec::new(), Vec::new());
                            for i in i0..i0 + count {
                                queries.extend_from_slice(&q[i * q_dim + kv * group * dim..][..group * dim]);
                                rels.extend_from_slice(&rel[(i * heads + kv * group) * d_rel..][..group * d_rel]);
                                row_sinks.extend_from_slice(&sinks[i * heads + kv * group..][..group]);
                            }
                            (queries, rels, row_sinks)
                        })
                        .collect();
                    let jobs: Vec<AttendHead> = gathered
                        .iter()
                        .enumerate()
                        .map(|(kv, (queries, rels, row_sinks))| AttendHead {
                            queries,
                            keys: &k[kv][from..],
                            values: &v[kv][from..],
                            rel: rels,
                            sinks: row_sinks,
                        })
                        .collect();
                    let shape = AttendShape {
                        rows,
                        keys,
                        dim,
                        d_rel,
                        extent,
                        profiles: &layer.rel_proj,
                        share: group,
                        first: position + i0,
                        lowest,
                        local: layer.local,
                        window: cfg.sliding_window_size,
                        scaling,
                    };
                    match gpu.attend_fast(&jobs, &shape) {
                        Ok(outputs) if outputs.par_iter().all(|head| head.iter().all(|value| value.is_finite())) => Some(outputs),
                        Ok(_) => {
                            GPU_ATTENTION_REDONE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            None
                        }
                        Err(error) => {
                            gpu_failed(&error);
                            None
                        }
                    }
                };
                // On the CPU, a few tokens make products too small for the routine to spread over the cores
                // itself, so the key heads go a core each; a long batch fills the cores from inside one product.
                let attended: Vec<Vec<f32>> = match on_gpu() {
                    Some(attended) => attended,
                    None if count < FAST_MIN_TOKENS => (0..kv_heads).into_par_iter().map(attend).collect(),
                    None => (0..kv_heads).map(attend).collect(),
                };
                for (kv, attended) in attended.iter().enumerate() {
                    for (row, attended) in attended.chunks_exact(dim).enumerate() {
                        let unit = (i0 + row / group) * heads + kv * group + row % group;
                        let gate = &gate[unit * dim..][..dim];
                        let out = &mut out[unit * dim..][..dim];
                        out.iter_mut().zip(attended).zip(gate).for_each(|((o, a), g)| *o = a * sigmoid(*g));
                    }
                }
            }
        }
        let keys = position + tokens - window_start(position + tokens - 1);
        let serial = tokens * keys * q_dim < ATTENTION_MIN_WORK;
        if fast {
        } else if bf16 || tracing {
            // Step by step as the reference does it, a token and head at a time.
            match serial {
                true => out.chunks_mut(dim).enumerate().for_each(attend),
                false => out.par_chunks_mut(dim).enumerate().for_each(attend),
            }
        } else {
            let units: Vec<(usize, usize)> =
                (0..tokens.div_ceil(ATTENTION_BLOCK)).flat_map(|block| (0..kv_heads).map(move |kv| (block, kv))).collect();
            let results: Vec<Vec<f32>> = match serial {
                true => units.iter().map(attend_block).collect(),
                false => units.par_iter().map(attend_block).collect(),
            };
            for (&(block, kv), result) in units.iter().zip(&results) {
                for (row, result) in result.chunks_exact(dim).enumerate() {
                    let unit = (block * ATTENTION_BLOCK + row / group) * heads + kv * group + row % group;
                    out[unit * dim..][..dim].copy_from_slice(result);
                }
            }
        }
        timing::since(Stage::Attention, attention_started);
        time(Stage::AttentionOutput, || {
            let mut projected = apply_many(self.gpu.as_ref(), &[(&layer.wo, &out)]).remove(0);
            round_all(bf16, &mut projected);
            projected
        })
    }
}

impl KvConv<'_> {
    fn apply(
        &self,
        x: &Seq,
        dynamic_input: &Seq,
        channels: usize,
        cfg: &TextConfig,
        history: &mut ConvHistory,
        bf16: bool,
    ) -> Seq {
        let k = cfg.sconv_kernel_size;
        match self {
            KvConv::Static(weight) => static_conv(x, weight, channels, k, history),
            KvConv::Dynamic { static_w, proj } => {
                // Taps are per channel (static) plus per token and key/value head (dynamic); tap `lag` reaches
                // `lag` tokens back.
                let mut taps = proj.apply(dynamic_input);
                round_all(bf16, &mut taps);
                let head_dim = channels / cfg.num_key_value_heads;
                let scale = 1.0 / ((2 * k + 1) as f32).sqrt();
                let (context, seen) = with_history(history, x, channels, k);
                let mut out = vec![0.0f32; x.len()];
                for (i, out) in out.chunks_exact_mut(channels).enumerate() {
                    let t = seen + i;
                    for lag in 0..k.min(t + 1) {
                        let past = &context[(t - lag) * channels..][..channels];
                        for (c, (o, p)) in out.iter_mut().zip(past).enumerate() {
                            let tap = static_w[lag * channels + c] + taps[i * proj.rows + (c / head_dim) * k + lag];
                            *o += tap * p;
                        }
                    }
                    let current = &context[t * channels..][..channels];
                    out.iter_mut().zip(current).for_each(|(o, c)| *o = (*o + c) * scale);
                }
                out
            }
        }
    }
}
