//! A GGUF conversion must behave like the checkpoint it was made from. Needs both on this machine: the published
//! checkpoint, and a Q4_K_M conversion named by `$RUSTLING_GGUF` or found among LM Studio's models. Skipped
//! otherwise. Run with `--release`.

use std::path::Path;

use rustling::checkpoint::{Checkpoint, resolve_model};
use rustling::config::Config;
use rustling::model::{Activations, Backend, Model, Options};
use rustling::prompt::Prompter;

const DEFAULT_GGUF: &str = "lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M";

/// The converted weights are a lossy copy, so the two models cannot agree exactly. A wrong tensor mapping or a
/// broken kernel would make them unrelated, which is what this guards against: over ordinary prose they must
/// mostly predict the same next token.
#[test]
fn a_gguf_conversion_predicts_what_the_checkpoint_does() {
    let named = std::env::var("RUSTLING_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_owned());
    let (Ok(gguf_path), Ok(model_dir)) = (resolve_model(Some(Path::new(&named))), resolve_model(None)) else {
        eprintln!("skipping: needs the checkpoint and {named}");
        return;
    };
    let (Ok(converted), Ok(original)) = (Checkpoint::open(&gguf_path), Checkpoint::open(&model_dir)) else {
        eprintln!("skipping: weights not found");
        return;
    };
    assert!(converted.is_gguf() && !original.is_gguf());
    let config = Config::load(&gguf_path).unwrap();
    let prompter = Prompter::load(&gguf_path).unwrap();
    let prose = "Rivers shape the land they cross. A young river runs fast and cuts a narrow valley; further down it slows, widens and drops the sand and silt it carried, building plains that people have farmed for thousands of years. Towns grew where rivers could be forded or bridged, and many of the oldest roads follow them.";
    let ids = prompter.encode(&format!("<|message_user|><|content_text|>{prose}<|end_message|>")).unwrap();
    let hidden = config.text_config.hidden_size;

    let options = Options { activations: Activations::Float, backend: Backend::Cpu };
    let predictions = |weights| -> Vec<usize> {
        let model = Model::load(&config, weights, options).unwrap();
        let last = model.forward(&ids, &mut model.new_state(), false).unwrap().last;
        let logits = model.logits_many(&last).unwrap();
        assert_eq!(logits.len(), last.len() / hidden);
        logits.iter().map(|l| (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap()).collect()
    };
    let (from_gguf, from_checkpoint) = (predictions(&converted), predictions(&original));
    let agree = from_gguf.iter().zip(&from_checkpoint).filter(|(a, b)| a == b).count();
    let share = agree as f64 / ids.len() as f64;
    eprintln!("top-1 agreement {agree}/{} ({share:.2})", ids.len());
    // Q4_K_M measured 0.80 here, with or without rounding the activations to bytes.
    assert!(share > 0.7, "only {agree} of {} predictions agree", ids.len());
}

/// How fast the cores can read weights at all, which bounds generation: every token streams most of the file.
/// `cargo test --release --test gguf bandwidth -- --ignored --nocapture`.
#[test]
#[ignore]
fn bandwidth() {
    use rayon::prelude::*;
    let named = std::env::var("RUSTLING_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_owned());
    let weights = Checkpoint::open(&resolve_model(Some(Path::new(&named))).unwrap()).unwrap();
    let map = weights.maps()[0];
    let span = &map[1 << 30..5 << 30];
    let sum = |chunk: &[u8]| chunk.as_chunks::<8>().0.iter().fold(0u64, |s, b| s.wrapping_add(u64::from_le_bytes(*b)));
    for threads in [1, 4, 8, 10] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let started = std::time::Instant::now();
            let total: u64 = pool.install(|| span.par_chunks(1 << 20).map(sum).reduce(|| 0, u64::wrapping_add));
            std::hint::black_box(total);
            best = best.min(started.elapsed().as_secs_f64());
        }
        eprintln!("{threads} threads: {:.0} GB/s", span.len() as f64 / best / 1e9);
    }
}

/// What the vocabulary projection costs alone and in batches.
/// `cargo test --release --test gguf logits_cost -- --ignored --nocapture`.
#[test]
#[ignore]
fn logits_cost() {
    let named = std::env::var("RUSTLING_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_owned());
    let path = resolve_model(Some(Path::new(&named))).unwrap();
    let weights = Checkpoint::open(&path).unwrap();
    let config = Config::load(&path).unwrap();
    let model = Model::load(&config, &weights, Options { activations: Activations::Float, backend: Backend::Cpu }).unwrap();
    let hidden = config.text_config.hidden_size;
    let states: Vec<f32> = (0..8 * hidden).map(|i| ((i * 7919 % 2003) as f32 / 1000.0) - 1.0).collect();
    for batch in [1, 2, 4, 8] {
        let started = std::time::Instant::now();
        for _ in 0..20 {
            std::hint::black_box(model.logits_many(&states[..batch * hidden]).unwrap());
        }
        let each = started.elapsed().as_secs_f64() * 1000.0 / 20.0;
        eprintln!("batch {batch}: {each:.2} ms, {:.2} ms per state", each / batch as f64);
    }
}
