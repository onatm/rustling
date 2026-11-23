use super::*;
use crate::weights::quant::{dot_bf16_row, dot_nvfp4_row_quantized, quantize_nvfp4};

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
fn split_attention_matches_stable_softmax_with_sink_and_distance_bias() {
    let gpu = Gpu::new(&[], true).unwrap();
    let kernels = gpu.kernels(include_str!("shaders/decode.metal"), &["attention_blocks", "attention_reduce"]).unwrap();
    let (dim, heads, extent) = (128usize, 6usize, 5usize);
    for seen in [127usize, 128, 129, 8193] {
        let blocks = seen.div_ceil(128);
        let (kc, vc) = (seen + 6, seen + 11);
        let (q, k, v) = (floats(heads * dim, 1), floats(2 * kc * dim, 2), floats(2 * vc * dim, 3));
        let bias: Vec<_> = floats(heads * extent, 4).into_iter().map(|x| x * 6.).collect();
        let sinks = [-10., 2., 12., 100., -2., 9.];
        let gate = floats(heads * dim, 5);
        let shared = |x: &[f32]| {
            let mut buf = gpu.shared(x.len() * 4).unwrap();
            buf.floats_mut()[..x.len()].copy_from_slice(x);
            buf
        };
        let buffers = [
            shared(&q), shared(&k), shared(&v), shared(&bias), shared(&[f32::from_bits((seen - 1) as u32)]),
            shared(&[0., f32::from_bits(kc as u32), f32::from_bits(vc as u32)]), gpu.shared(blocks * heads * 132 * 4).unwrap(),
            shared(&sinks), shared(&gate), gpu.shared(heads * dim * 4).unwrap(),
        ];
        let scale = (dim as f32).powf(-0.5);
        let steps = [
            Step { kernel: 0, buffers: (0..7).map(|i| (i, 0)).collect(),
                params: [extent as u32, scale.to_bits()].into_iter().flat_map(u32::to_le_bytes).collect(),
                grid: Grid::KeyBlocks { rows: 2, block: 128 } },
            Step { kernel: 1, buffers: [6, 7, 8, 4, 9].map(|i| (i, 0)).to_vec(), params: vec![],
                grid: Grid::Wide { groups: heads, threads: 256 } },
        ];
        gpu.run(&[&kernels], &steps, &buffers, [seen, seen], 1).unwrap();
        for head in 0..heads {
            let scores: Vec<_> = (0..seen).map(|j| {
                let key = (head / 3 * kc + j) * dim;
                let dot: f64 = (0..dim).map(|i| q[head * dim + i] as f64 * k[key + i] as f64).sum();
                let back = seen - 1 - j;
                dot * scale as f64 + if back < extent { bias[head * extent + back] as f64 } else { 0. }
            }).collect();
            let max = scores.iter().copied().fold(sinks[head] as f64, f64::max);
            let w: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
            let mass = w.iter().sum::<f64>() + (sinks[head] as f64 - max).exp();
            for i in 0..dim {
                let at = head * dim + i;
                let sum: f64 = w.iter().enumerate().map(|(j, w)| w * v[(head / 3 * vc + j) * dim + i] as f64).sum();
                let want = sum / mass / (1. + (-gate[at] as f64).exp());
                assert!((buffers[9].floats()[at] as f64 - want).abs() < 1e-5, "head {head}, component {i}, keys {seen}");
            }
        }
    }
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
    use crate::weights::checkpoint::{Checkpoint, resolve_model};
    let path = resolve_model(Some(std::path::Path::new("lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M"))).unwrap();
    let weights = Checkpoint::open(&path).unwrap();
    let gpu = Gpu::new(&weights.maps(), false).unwrap();
    let kernels = gpu.kernels(include_str!("shaders/gguf.metal"), &["row_q4k", "row_q5_0", "row_q6k", "row_q8_0", "row_f32"]).unwrap();
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
    use crate::weights::checkpoint::{Checkpoint, resolve_model};
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
    let seconds = best(&|| drop(gpu.dense_fast(&jobs, false).unwrap()));
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
    // More vectors than a run, and not a whole number of runs.
    let (rows, cols, vectors) = (40, 2304, 300);
    let weights: Vec<u8> =
        bytes(rows * cols * 2, 5).as_chunks::<2>().0.iter().flat_map(|b| [b[0], 0x3C | (b[1] & 0x83)]).collect();
    let xs = floats(vectors * cols, 6);
    let job = [DenseJob { format: Format::Bf16, weights: &weights, rows, cols, xs: &xs }];
    let (fast, exact) = (gpu.dense_fast(&job, false).unwrap().remove(0), gpu.dense(&job).unwrap().remove(0));
    // Taken a run at a time, the products are the same sums.
    let in_runs = gpu.dense_fast(&job, true).unwrap().remove(0);
    assert_eq!(in_runs.len(), fast.len());
    assert!(in_runs.iter().zip(&fast).all(|(a, b)| (a - b).abs() <= 1e-3 * (1.0 + b.abs())), "runs differ from the whole");
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
                crate::weights::quant::decode_nvfp4_row(
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
        let exact = crate::weights::quant::dot_f32_row(&decoded, x);
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
    use crate::weights::ggml::encode_q8_0;
    let gpu = Gpu::new(&[], true).unwrap();
    // One full run followed by a 17-row direct tail. The other experts' padded runs therefore
    // start at non-multiples of PRODUCT_RUN; check both layouts and their combined results.
    let (experts, hidden, units, chosen, tokens) = (5usize, 64usize, 32usize, 2usize, 273usize);
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
    // Expert 0 is direct-only, 1 has both kinds of run, 2 and 3 are widened, and 4 is unused.
    // The first 16 tokens alone also exercise a submission with no widening at all.
    let routes: Vec<(u32, f32)> = (0..tokens).flat_map(|t| {
        let other = if t < 32 { 0 } else { 2 + (t % 2) as u32 };
        [(1u32, 0.75f32), (other, 0.25 + t as f32 / 1000.0)]
    }).collect();
    let gate_scale = [0.5f32, 0.25, 1.5, 1., 0.75];
    let down_scale = [1.5f32, 0.75, 0.5, 1., 0.25];
    let scale_bytes = |scales: &[f32]| scales.iter().flat_map(|s| s.to_le_bytes()).collect::<Vec<_>>();
    let (gate_bytes, down_bytes) = (scale_bytes(&gate_scale), scale_bytes(&down_scale));
    let job = WholeExperts {
        scales: [&gate_bytes, &down_bytes],
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
    let short = gpu.experts_whole(&WholeExperts { x: &x[..16 * hidden], routes: &routes[..16 * chosen], ..job }).unwrap();
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
    for t in [0, 1, 255, 256, 257, tokens - 1] {
        let input = &x[t * hidden..][..hidden];
        let mut expected = vec![0.0f32; hidden];
        for &(expert, weight) in &routes[t * chosen..][..chosen] {
            let rows = &gate_up_values[expert as usize * 2 * units * hidden..][..2 * units * hidden];
            let active: Vec<f32> = (0..units)
                .map(|u| {
                    let (gate, up) = (dot(&rows[u * hidden..][..hidden], input), dot(&rows[(units + u) * hidden..][..hidden], input));
                    let (gate, up) = (gate * gate_scale[expert as usize], up * gate_scale[expert as usize]);
                    (gate / 4.0).tanh() * 4.0 / (1.0 + (-gate).exp()) * (up / 25.0).tanh() * 25.0
                })
                .collect();
            let rows = &down_values[expert as usize * hidden * units..][..hidden * units];
            for (c, expected) in expected.iter_mut().enumerate() {
                *expected += weight * down_scale[expert as usize] * dot(&rows[c * units..][..units], &active);
            }
        }
        for (c, expected) in expected.iter().enumerate() {
            let got = out[t * hidden + c];
            assert!((got - expected).abs() < 1e-3 * (1.0 + expected.abs()), "token {t}, {c}: {got} against {expected}");
            if t < 16 {
                assert!((short[t * hidden + c] - expected).abs() < 1e-3 * (1.0 + expected.abs()), "direct-only token {t}, {c}");
            }
        }
    }
}

#[test]
fn the_fast_gates_match_plain_sums() {
    let gpu = Gpu::new(&[], true).unwrap();
    let (tokens, width, rank, hidden) = (263usize, 48usize, 6usize, 20usize);
    let (pairs, down, forget, gain) = (floats(tokens * width, 1), floats(rank * width, 2), floats(hidden * rank, 3), floats(hidden * rank, 4));
    let raw = |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    for in_runs in [false, true] {
    let (forgets, gains) = gpu.gates_fast(&pairs, &raw(&down), &raw(&forget), &raw(&gain), (rank, hidden), in_runs).unwrap();
    assert_eq!((forgets.len(), gains.len()), (tokens * hidden, tokens * hidden));
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
}

#[test]
fn the_fast_paths_unpack_gguf_blocks() {
    use crate::weights::ggml::random_row;
    let gpu = Gpu::new(&[], true).unwrap();
    for (f, format) in [Format::Q4K, Format::Q5_0, Format::Q6K, Format::Q8_0, Format::Q5K, Format::Nvfp4].into_iter().enumerate() {
        // A dense matrix.
        let (rows, cols, vectors) = (37, 768, 5);
        let weights: Vec<u8> = (0..rows).flat_map(|r| random_row(format, cols, (100 * f + r) as u32)).collect();
        let xs = floats(vectors * cols, 40 + f as u64);
        let out = gpu.dense_fast(&[DenseJob { format, weights: &weights, rows, cols, xs: &xs }], false).unwrap().remove(0);
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
    use crate::weights::ggml::random_row;
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
                crate::weights::quant::decode_mxfp8_row(&values[row * cols..][..cols], &scales[row * cols / 32..][..cols / 32], &mut decoded);
                let exact = crate::weights::quant::dot_f32_row(&decoded, x);
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

/// Whether Apple's matrix product is quicker over 16-bit floats, at the shape of an expert's first product.
/// `cargo test --release --lib half_products -- --ignored --nocapture`.
#[test]
#[ignore]
fn half_products() {
    let gpu = Gpu::new(&[], true).unwrap();
    for (count, vectors, rows, cols) in [(128usize, 256usize, 2304usize, 2304usize), (128, 256, 2304, 1152), (1, 4096, 3072, 2304)] {
        for half in [false, true] {
            let size = if half { 2 } else { 4 };
            let fill = |floats: usize| -> Buffer {
                // 0.5 in either width, so the sums are ordinary numbers.
                let bytes: Vec<u8> = match half {
                    true => (0..floats).flat_map(|_| 0x3800u16.to_le_bytes()).collect(),
                    false => (0..floats).flat_map(|_| 0.5f32.to_le_bytes()).collect(),
                };
                copied(&gpu.device, &bytes).unwrap()
            };
            let (left, right) = (fill(count * vectors * cols), fill(count * rows * cols));
            let out = empty(&gpu.device, count * vectors * rows * size).unwrap();
            let mut best = f64::MAX;
            for _ in 0..4 {
                let started = std::time::Instant::now();
                objc2::rc::autoreleasepool(|_| {
                    let commands = gpu.queue.commandBuffer().unwrap();
                    gpu.product_as(&commands, (&left, 0, vectors, cols), (&right, 0, rows, cols), (&out, 0), count, true, 1.0, half);
                    commands.commit();
                    commands.waitUntilCompleted();
                    finish(&commands).unwrap();
                });
                best = best.min(started.elapsed().as_secs_f64());
            }
            // SAFETY: the buffer holds at least one value of either width.
            let first = unsafe {
                let pointer = out.contents().as_ptr();
                if half { half::f16::from_bits(*(pointer as *const u16)).to_f32() } else { *(pointer as *const f32) }
            };
            let products = (count * vectors * rows * cols) as f64;
            eprintln!(
                "{count} x ({vectors} x {cols}) . ({rows} x {cols})^T, {}: {:.1} ms, {:.0} G products/s, first value {first} (expected {})",
                if half { "16-bit" } else { "32-bit" },
                best * 1e3,
                products / best / 1e9,
                0.25 * cols as f32
            );
        }
    }
}
