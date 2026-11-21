//! The Metal backend must reproduce the CPU's results exactly, on the real weights. Skipped when the model is not
//! on this machine or there is no GPU. Run with `--release`.
#![cfg(target_os = "macos")]

use rustling::checkpoint::{Checkpoint, resolve_model_dir};
use rustling::config::Config;
use rustling::model::{Activations, Backend, Model, Options};
use rustling::prompt::Prompter;

#[test]
fn metal_matches_cpu_exactly() {
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
    // Long enough that every batch is past the size at which work moves to the GPU.
    let text = "<|message_user|><|content_text|>".to_owned() + &"The quick brown fox jumps over the lazy dog. ".repeat(8);
    let ids = prompter.encode(&text).unwrap();
    assert!(ids.len() >= 64, "{} tokens", ids.len());

    for activations in [Activations::Quantised, Activations::Float] {
        let load = |backend| Model::load(&config, &checkpoint, Options { activations, backend });
        let Ok(metal) = load(Backend::Metal) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        assert!(metal.uses_gpu());
        let cpu = load(Backend::Cpu).unwrap();
        let on_metal = metal.forward(&ids, &mut metal.new_state(), true).unwrap();
        let on_cpu = cpu.forward(&ids, &mut cpu.new_state(), true).unwrap();
        for (visit, (a, b)) in on_metal.hidden.iter().zip(&on_cpu.hidden).enumerate() {
            assert!(a == b, "{activations:?}: visit {visit} differs between Metal and CPU");
        }
        assert!(on_metal.last == on_cpu.last, "{activations:?}: final state differs between Metal and CPU");
    }
}
