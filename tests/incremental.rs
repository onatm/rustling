//! Incremental decoding must give the same states as one pass over the whole sequence, however the tokens are
//! split across calls. Skipped when the model is not on this machine. Run with `--release`: debug builds are slow.

use rustling::checkpoint::{Checkpoint, resolve_model_dir};
use rustling::config::Config;
use rustling::model::{Activations, Backend, Model, Options};
use rustling::prompt::Prompter;

#[test]
fn stepping_matches_one_pass() {
    let Ok(model_dir) = resolve_model_dir(None) else {
        eprintln!("skipping: model not found");
        return;
    };
    let Ok(checkpoint) = Checkpoint::open(&model_dir) else {
        eprintln!("skipping: checkpoint not found");
        return;
    };
    let config = Config::load(&model_dir).unwrap();
    let prompter = Prompter::load(&model_dir).unwrap();
    for activations in [Activations::Float, Activations::Quantised] {
        let options = Options { activations, backend: Backend::Cpu };
        stepping_matches(&Model::load(&config, &checkpoint, options).unwrap(), &prompter, activations);
    }
}

fn stepping_matches(model: &Model, prompter: &Prompter, activations: Activations) {
    // Long enough that the short convolutions (kernel 4) have full histories for several tokens.
    let ids = prompter.encode("<|message_user|><|content_text|>Count: one two three four five<|end_message|>").unwrap();
    assert!(ids.len() >= 10);

    let whole = model.forward(&ids, &mut model.new_state(), true).unwrap();

    // A chunk, then single tokens, as generation does.
    let split = 4;
    let mut state = model.new_state();
    let mut parts = vec![model.forward(&ids[..split], &mut state, true).unwrap()];
    for id in &ids[split..] {
        parts.push(model.forward(&[*id], &mut state, true).unwrap());
    }
    assert_eq!(state.position(), ids.len());

    let max_diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let scale = |a: &[f32]| a.iter().map(|a| a.abs()).fold(0.0f32, f32::max);
    for visit in 0..whole.hidden.len() {
        let stepped: Vec<f32> = parts.iter().flat_map(|part| part.hidden[visit].iter().copied()).collect();
        let diff = max_diff(&whole.hidden[visit], &stepped);
        assert!(diff <= 1e-4 * scale(&whole.hidden[visit]), "{activations:?} visit {visit}: max difference {diff}");
    }
    let stepped: Vec<f32> = parts.iter().flat_map(|part| part.last.iter().copied()).collect();
    let diff = max_diff(&whole.last, &stepped);
    assert!(diff <= 1e-4 * scale(&whole.last), "{activations:?} final state: max difference {diff}");
    eprintln!("{activations:?}: final state max difference {diff:e}");
}
