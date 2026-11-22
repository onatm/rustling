//! The forward pass against transformers' hidden states and logits, from `tests/golden/hidden_default.safetensors`
//! (`uv run tools/golden.py hidden "Hi there"`). Skipped when the model or the dump is missing. Run with `--release`.
//!
//! transformers computes in bfloat16, rounding every intermediate to eight bits of mantissa, and those roundings
//! compound over 48 layer visits. So even with the rounding imitated, agreement is to within several percent, not
//! exact; `tests/noise.rs` shows that this is the size rounding alone accounts for. The bounds here are loose
//! guards against a broken layer, which produces errors of order one.

use std::path::Path;

use rustling::checkpoint::{Checkpoint, resolve_model};
use rustling::config::Config;
use rustling::model::{Activations, Backend, Model, Options};
use safetensors::SafeTensors;

fn relative(ours: &[f32], reference: &[f32]) -> f64 {
    let diff: f64 = ours.iter().zip(reference).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let norm: f64 = reference.iter().map(|b| (*b as f64).powi(2)).sum();
    (diff / norm).sqrt()
}

#[test]
fn float_path_tracks_transformers() {
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/hidden_default.safetensors");
    let (Ok(bytes), Ok(model_dir)) = (std::fs::read(&golden), resolve_model(None)) else {
        eprintln!("skipping: needs the model and {}", golden.display());
        return;
    };
    let Ok(checkpoint) = Checkpoint::open(&model_dir) else {
        eprintln!("skipping: checkpoint not found");
        return;
    };
    let golden = SafeTensors::deserialize(&bytes).unwrap();
    let floats = |name: &str| -> Vec<f32> {
        golden.tensor(name).unwrap().data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    };
    let ids: Vec<u32> = golden
        .tensor("input_ids")
        .unwrap()
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b) as u32)
        .collect();
    let (states, logits) = (floats("hidden_states"), floats("logits"));

    let config = Config::load(&model_dir).unwrap();
    let options = Options { activations: Activations::FloatBf16, backend: Backend::Cpu };
    let model = Model::load(&config, &checkpoint, options).unwrap();
    let forward = model.forward(&ids, &mut model.new_state(), true).unwrap();

    // The dump holds the embedding, then one state per visit, except that the last entry is the final normed
    // state instead of the last layer's raw output.
    let visits = forward.hidden.len();
    let per_state = ids.len() * model.config().hidden_size;
    assert_eq!(states.len(), (visits + 1) * per_state);
    for (visit, ours) in forward.hidden.iter().enumerate().take(visits - 1) {
        let error = relative(ours, &states[(visit + 1) * per_state..][..per_state]);
        assert!(error < 0.15, "visit {visit}: relative error {error:.3}");
    }
    let error = relative(&forward.last, &states[visits * per_state..]);
    assert!(error < 0.15, "final state: relative error {error:.3}");

    let (hidden, vocab) = (model.config().hidden_size, model.config().unpadded_vocab_size);
    let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
    let (mut agree, mut worst) = (0, 0.0f64);
    for (t, state) in forward.last.chunks_exact(hidden).enumerate() {
        let ours = model.logits(state).unwrap();
        let reference = &logits[t * vocab..][..vocab];
        agree += (argmax(&ours) == argmax(reference)) as usize;
        worst = worst.max(relative(&ours, reference));
    }
    eprintln!("worst logits relative error {worst:.4}, top-1 agreement {agree}/{}", ids.len());
    assert!(worst < 0.25, "logits: worst relative error {worst:.3}");
    assert!(agree * 10 >= ids.len() * 8, "top-1 agreement {agree}/{}", ids.len());
}
