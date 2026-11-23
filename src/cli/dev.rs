//! Commands for checking the engine against dumps from transformers.

use crate::*;

pub(crate) fn compare(model_dir: &std::path::Path, golden: &std::path::Path, activations: Activations) -> Result<()> {
    let bytes = std::fs::read(golden)?;
    let golden = SafeTensors::deserialize(&bytes)?;
    let f32s = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
        let view = golden.tensor(name)?;
        let data = view.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        Ok((view.shape().to_vec(), data))
    };
    let ids: Vec<u32> =
        golden.tensor("input_ids")?.data().as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as u32).collect();
    let (shape, states) = f32s("hidden_states")?;
    let (_, logits) = f32s("logits")?;

    let config = load_config(model_dir)?;
    let checkpoint = Checkpoint::open(model_dir)?;
    let model = Model::load(&config, &checkpoint, Options { activations, backend: Backend::Cpu })?;
    let forward = model.forward(&ids, &mut model.new_state(), true)?;
    let visits = model.config().visits();
    // The reference may record the embedding ahead of the layer outputs; the visits are always the tail.
    anyhow::ensure!(shape[0] >= visits.len(), "golden has {} states, expected at least {}", shape[0], visits.len());
    let per_state = shape[1] * shape[2];
    let offset = (shape[0] - visits.len()) * per_state;

    // Relative error: |ours - reference| / |reference|, over the whole sequence.
    let relative = |ours: &[f32], reference: &[f32]| -> f64 {
        let diff: f64 = ours.iter().zip(reference).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let norm: f64 = reference.iter().map(|b| (*b as f64).powi(2)).sum();
        (diff / norm).sqrt()
    };
    println!("{} tokens", ids.len());
    // transformers reports the final normed state in place of the last layer's raw output, so that visit has no
    // counterpart; the final state is compared below instead.
    for (visit, ours) in visits.iter().zip(&forward.hidden).take(visits.len() - 1) {
        let reference = &states[offset + visit.index * per_state..][..per_state];
        println!(
            "visit {:>2}  layer {:>2} pass {}  relative error {:.2e}",
            visit.index,
            visit.layer,
            visit.pass,
            relative(ours, reference)
        );
    }
    let hidden = model.config().hidden_size;
    let vocab = model.config().unpadded_vocab_size;
    let reference_last = &states[states.len() - per_state..];
    println!("final normed state relative error {:.2e}", relative(&forward.last, reference_last));

    let prompter = Prompter::load(model_dir)?;
    let mut agree = 0;
    for (t, state) in forward.last.chunks_exact(hidden).enumerate() {
        let ours = model.logits(state)?;
        let reference = &logits[t * vocab..][..vocab];
        let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
        let (mine, theirs) = (argmax(&ours), argmax(reference));
        agree += (mine == theirs) as usize;
        if mine != theirs {
            // How close the two candidates are on each side shows whether this is a near-tie or a real difference.
            println!(
                "  position {t}: ours {:?} (ours {:.3}, reference {:.3}), reference {:?} (ours {:.3}, reference {:.3})",
                prompter.decode(&[mine as u32])?,
                ours[mine],
                reference[mine],
                prompter.decode(&[theirs as u32])?,
                ours[theirs],
                reference[theirs]
            );
        }
        if t + 1 == ids.len() {
            println!("last-token logits relative error {:.2e}", relative(&ours, reference));
        }
    }
    println!("top-1 agreement {agree}/{}", ids.len());
    Ok(())
}

pub(crate) fn probe(model_dir: &std::path::Path, golden: &std::path::Path, activations: Activations) -> Result<()> {
    let bytes = std::fs::read(golden)?;
    let golden = SafeTensors::deserialize(&bytes)?;
    let ids: Vec<u32> =
        golden.tensor("input_ids")?.data().as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as u32).collect();

    let config = load_config(model_dir)?;
    let checkpoint = Checkpoint::open(model_dir)?;
    let model = Model::load(&config, &checkpoint, Options { activations, backend: Backend::Cpu })?;
    rustling::trace::start();
    model.forward(&ids, &mut model.new_state(), false)?;
    let ours = rustling::trace::take();

    println!("{:<14} {:>10} {:>12} {:>12}", "tensor", "values", "rel. error", "max |diff|");
    for name in [
        "embed", "attn_in", "q", "k", "v", "bias", "attn", "attn_conv", "mlp_in", "router_logits", "mlp", "mlp_conv", "out",
    ] {
        let (Ok(view), Some(ours)) = (golden.tensor(name), ours.get(name)) else {
            println!("{name:<14} missing on one side");
            continue;
        };
        let reference: Vec<f32> = view.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        if reference.len() != ours.len() {
            println!("{name:<14} {} values here, {} in the reference {:?}", ours.len(), reference.len(), view.shape());
            continue;
        }
        let diff: f64 = ours.iter().zip(&reference).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let norm: f64 = reference.iter().map(|b| (*b as f64).powi(2)).sum();
        let max = ours.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("{name:<14} {:>10} {:>12.2e} {:>12.3e}", ours.len(), (diff / norm).sqrt(), max);
    }
    Ok(())
}
