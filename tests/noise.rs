//! How far bfloat16 rounding alone moves the outputs: the plain float path against the same path with
//! transformers' rounding imitated. This is the scale against which the gap to transformers should be read.
//! Needs the model and `tests/golden/hidden_default.safetensors` (for its token ids and reference states).

use std::path::Path;

use rustling::weights::checkpoint::{Checkpoint, resolve_checkpoint};
use rustling::config::Config;
use rustling::model::{Activations, Backend, Model, Options};
use safetensors::SafeTensors;

fn relative(ours: &[f32], reference: &[f32]) -> f64 {
    let diff: f64 = ours.iter().zip(reference).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
    let norm: f64 = reference.iter().map(|b| (*b as f64).powi(2)).sum();
    (diff / norm).sqrt()
}

#[test]
fn rounding_noise_explains_the_gap_to_transformers() {
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/hidden_default.safetensors");
    let (Ok(bytes), Ok(model_dir)) = (std::fs::read(&golden), resolve_checkpoint()) else {
        eprintln!("skipping: needs the model and {}", golden.display());
        return;
    };
    let Ok(checkpoint) = Checkpoint::open(&model_dir) else {
        eprintln!("skipping: checkpoint not found");
        return;
    };
    let golden = SafeTensors::deserialize(&bytes).unwrap();
    let ids: Vec<u32> =
        golden.tensor("input_ids").unwrap().data().as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as u32).collect();
    let states: Vec<f32> =
        golden.tensor("hidden_states").unwrap().data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();

    let config = Config::load(&model_dir).unwrap();
    let run = |activations| {
        let model = Model::load(&config, &checkpoint, Options { activations, backend: Backend::Cpu }).unwrap();
        let forward = model.forward(&ids, &mut model.new_state(), true).unwrap();
        (forward.last, forward.hidden)
    };
    let ((float, float_visits), (imitation, imitation_visits)) = (run(Activations::Float), run(Activations::FloatBf16));
    let hidden = config.text_config.hidden_size;

    // Per visit, to show whether any one stage (the first full-attention layer at visit 5, the first pass with
    // messages from visit 18) stands out. The dump holds the embedding first; its last entry is the final state.
    let per_state = ids.len() * hidden;
    eprintln!("visit  rounding alone  imitation vs transformers  ratio");
    for visit in 0..float_visits.len() - 1 {
        let noise = relative(&float_visits[visit], &imitation_visits[visit]);
        let gap = relative(&imitation_visits[visit], &states[(visit + 1) * per_state..][..per_state]);
        eprintln!("v{visit:>4}  {noise:>14.4}  {gap:>25.4}  {:>5.2}", gap / noise);
        // Early on, the imitation reproduces most of transformers' own rounding, so it sits closer to
        // transformers than plain float does. Deeper in, the two rounding histories decorrelate and the gap
        // tends to sqrt(2) times the noise.
        if visit < 6 {
            assert!(gap < noise, "visit {visit}: gap {gap:.4} is not below the rounding noise {noise:.4}");
        }
        assert!(gap < 3.0 * noise, "visit {visit}: gap {gap:.4} is far above the rounding noise {noise:.4}");
    }
    let reference = &states[states.len() - ids.len() * hidden..];

    eprintln!("position  rounding alone  imitation vs transformers");
    for t in 0..ids.len() {
        let slice = |v: &[f32]| v[t * hidden..][..hidden].to_vec();
        let noise = relative(&slice(&float), &slice(&imitation));
        let gap = relative(&slice(&imitation), &slice(reference));
        eprintln!("{t:>8}  {noise:>14.4}  {gap:>25.4}");
    }
    let (noise, gap) = (relative(&float, &imitation), relative(&imitation, reference));
    eprintln!("overall   {noise:>14.4}  {gap:>25.4}");
    // If the port were wrong somewhere, the gap to transformers would stand well clear of what rounding does.
    assert!(gap < 2.0 * noise, "gap to transformers {gap:.4} is more than twice the rounding noise {noise:.4}");
}
