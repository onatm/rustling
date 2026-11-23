//! The Metal backend must reproduce the CPU's results exactly, on the real weights. Skipped when the model is not
//! on this machine or there is no GPU. Run with `--release`.
#![cfg(target_os = "macos")]

use rustling::weights::checkpoint::{Checkpoint, resolve_checkpoint};
use rustling::config::Config;
use rustling::model::{Activations, Backend, Model, Options};
use rustling::prompt::Prompter;

#[test]
fn metal_matches_cpu_exactly() {
    let Ok(model_dir) = resolve_checkpoint() else {
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

/// The fast path for long prompts sums in a different order from the exact kernels, so its results differ in the
/// last bits. What that does depends on the arithmetic:
///
/// - With float activations nothing amplifies the difference, and the two paths agree closely.
/// - With quantised activations the model is chaotic in this respect: a last-bit difference flips a few 4-bit
///   codes, each flip is a large change, and the states drift far apart over the 48 layer visits (around a
///   quarter of their size on this text) while mostly still predicting the same tokens. Any two implementations
///   that differ at all do this, so neither path is the more correct one; the test only guards against the fast
///   path being broken, which would send agreement towards zero.
#[test]
fn the_fast_path_tracks_the_exact_one() {
    let Ok(model_dir) = resolve_checkpoint() else {
        eprintln!("skipping: model not found");
        return;
    };
    let Ok(checkpoint) = Checkpoint::open(&model_dir) else {
        eprintln!("skipping: checkpoint not found");
        return;
    };
    let config = Config::load(&model_dir).unwrap();
    let prompter = Prompter::load(&model_dir).unwrap();
    // Ordinary prose, past the length at which the fast path takes over.
    let prose = "Rivers shape the land they cross. A young river runs fast and cuts a narrow valley; further down it slows, widens and drops the sand and silt it carried, building plains that people have farmed for thousands of years. Towns grew where rivers could be forded or bridged, and many of the oldest roads follow them. A river is also a boundary, a source of power for mills and turbines, and a route for moving heavy goods cheaply. Floods are the price of living beside one: the same water that renews the soil can destroy a harvest, so communities built levees, dug channels and learned to read the seasons. Today engineers argue about how much to control a river and how much to leave it room, since straightened channels move water quickly but pass the flood on to whoever lives downstream. ";
    let ids = prompter.encode(&format!("<|message_user|><|content_text|>{}<|end_message|>", prose.repeat(2))).unwrap();
    assert!(ids.len() >= 256, "{} tokens", ids.len());
    let hidden = config.text_config.hidden_size;

    for activations in [Activations::Float, Activations::Quantised] {
        let load = |backend| Model::load(&config, &checkpoint, Options { activations, backend });
        let Ok(fast) = load(Backend::MetalFast) else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let exact = load(Backend::Cpu).unwrap();
        let on_fast = fast.forward(&ids, &mut fast.new_state(), false).unwrap().last;
        let on_exact = exact.forward(&ids, &mut exact.new_state(), false).unwrap().last;

        let diff: f64 = on_fast.iter().zip(&on_exact).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let difference = (diff / on_exact.iter().map(|b| (*b as f64).powi(2)).sum::<f64>()).sqrt();
        let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
        // Next-token predictions at a spread of positions.
        let sampled: Vec<usize> = (0..ids.len()).step_by(ids.len() / 24).collect();
        let agree = sampled
            .iter()
            .filter(|&&t| {
                let (a, b) = (&on_fast[t * hidden..][..hidden], &on_exact[t * hidden..][..hidden]);
                argmax(&fast.logits(a).unwrap()) == argmax(&exact.logits(b).unwrap())
            })
            .count();
        eprintln!(
            "{activations:?}: fast against exact over {} tokens: relative difference {difference:.4}, top-1 agreement {agree}/{}",
            ids.len(),
            sampled.len()
        );
        match activations {
            Activations::Quantised => {
                assert!(difference < 0.6, "relative difference {difference:.3}");
                assert!(agree * 4 >= sampled.len() * 3, "top-1 agreement {agree}/{}", sampled.len());
            }
            // Attention scores are sums of large terms that mostly cancel, so a different order of summing moves
            // them in the fifth digit. Now and then that tips which experts a token is routed to, and the states
            // part by a percent or two (0.02 measured) while predicting the same tokens.
            _ => {
                assert!(difference < 0.05, "relative difference {difference:.5}");
                assert_eq!(agree, sampled.len());
            }
        }
    }
}
