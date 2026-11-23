//! How far each way of running a GGUF conversion strays from the checkpoint it was made from, measured on text
//! of your choosing: the checkpoint predicts the next token at every position, and each path is scored on how
//! close its own predictions are. Needs the checkpoint and a Q4_K_M conversion, as `tests/gguf.rs` does.
//!
//! `RUSTLING_QUALITY=agent.txt,code.rs cargo test --release --test quality -- --ignored --nocapture`
//!
//! A file that starts with `<|` is taken as a rendered prompt; anything else is wrapped in a user message.

use std::path::Path;

use rustling::weights::checkpoint::{Checkpoint, resolve_model};
use rustling::config::Config;
use rustling::model::{Activations, Backend, Model, Options, State};
use rustling::prompt::Prompter;

const DEFAULT_GGUF: &str = "lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M";
/// The tokens at the end of each text that the generating paths are also run over, one by one.
const TAIL: usize = 2600;
/// The size of the additions an agent makes between turns, which have a path of their own.
const ADDITION: usize = 150;
/// The checkpoint counts as sure of a token from this probability.
const SURE: f64 = 0.9;

/// One path's results over a text: the final states, and the logits where the path makes its own.
struct Run {
    name: &'static str,
    /// The first position covered.
    from: usize,
    states: Vec<f32>,
    logits: Option<Vec<Vec<f32>>>,
}

#[derive(Default, Clone)]
struct Score {
    positions: usize,
    agree: usize,
    divergence: f64,
    worst: f64,
    /// The path's own surprise at the token that actually came next.
    surprise: f64,
    sure: usize,
    sure_agree: usize,
}

fn log_probabilities(logits: &[f32]) -> Vec<f64> {
    let top = logits.iter().copied().fold(f32::MIN, f32::max) as f64;
    let total: f64 = logits.iter().map(|&l| (l as f64 - top).exp()).sum();
    logits.iter().map(|&l| l as f64 - top - total.ln()).collect()
}

fn best(values: &[f64]) -> usize {
    (0..values.len()).max_by(|&a, &b| values[a].total_cmp(&values[b])).unwrap()
}

impl Score {
    fn add(&mut self, reference: &[f64], ours: &[f64], next: u32) {
        let (theirs, mine) = (best(reference), best(ours));
        let divergence: f64 = reference.iter().zip(ours).map(|(r, o)| r.exp() * (r - o)).sum();
        self.positions += 1;
        self.agree += (theirs == mine) as usize;
        self.divergence += divergence;
        self.worst = self.worst.max(divergence);
        self.surprise -= ours[next as usize];
        if reference[theirs].exp() >= SURE {
            self.sure += 1;
            self.sure_agree += (theirs == mine) as usize;
        }
    }

    fn report(&self, name: &str) {
        if self.positions == 0 {
            return;
        }
        let n = self.positions as f64;
        eprintln!(
            "  {name:<22} same token {:>6.2}%   when sure {:>6.2}% ({} of {})   divergence {:.4} (worst {:.2})   perplexity {:.3}",
            100.0 * self.agree as f64 / n,
            100.0 * self.sure_agree as f64 / self.sure.max(1) as f64,
            self.sure - self.sure_agree,
            self.sure,
            self.divergence / n,
            self.worst,
            (self.surprise / n).exp(),
        );
    }
}

/// The final states for `ids`, processed as a prompt is: in the model's own chunks.
fn as_a_prompt(model: &Model, ids: &[u32], state: &mut State) -> Vec<f32> {
    let mut states = Vec::new();
    for chunk in ids.chunks(model.prompt_chunk()) {
        states.extend(model.forward(chunk, state, false).unwrap().last);
    }
    model.release_scratch();
    states
}

#[test]
#[ignore]
fn paths_against_the_checkpoint() {
    let files = std::env::var("RUSTLING_QUALITY").expect("RUSTLING_QUALITY names the texts, separated by commas");
    let named = std::env::var("RUSTLING_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_owned());
    let gguf_path = resolve_model(Some(Path::new(&named))).unwrap();
    let model_dir = resolve_model(None).unwrap();
    let (converted, original) = (Checkpoint::open(&gguf_path).unwrap(), Checkpoint::open(&model_dir).unwrap());
    assert!(converted.is_gguf() && !original.is_gguf());
    let config = Config::load(&gguf_path).unwrap();
    let prompter = Prompter::load(&gguf_path).unwrap();
    let hidden = config.text_config.hidden_size;
    let one = |text: &str| -> u32 {
        let ids = prompter.encode(text).unwrap();
        assert_eq!(ids.len(), 1, "{text} is not one token");
        ids[0]
    };
    let (model_turn, turn_end) = (one("<|message_model|>"), one("<|end_message|>"));

    let load = |weights, activations, backend| Model::load(&config, weights, Options { activations, backend }).unwrap();
    let reference = load(&original, Activations::Quantised, Backend::Auto);
    let in_floats = load(&original, Activations::Float, Backend::Auto);
    let on_cpu = load(&converted, Activations::Float, Backend::Cpu);
    let on_gpu = load(&converted, Activations::Float, Backend::MetalFast);
    assert!(on_gpu.decodes_on_gpu());

    for file in files.split(',') {
        let text = std::fs::read_to_string(file).unwrap();
        let text = match text.starts_with("<|") {
            true => text,
            false => format!("<|message_user|><|content_text|>{text}<|end_message|>"),
        };
        let ids = prompter.encode(&text).unwrap();
        let tail = ids.len() - TAIL.min(ids.len() / 2);
        // Whether the token after each position is the model's own writing: what a reply is made of.
        let mut writing = false;
        let replies: Vec<bool> = (0..ids.len() - 1)
            .map(|t| {
                writing = match ids[t] {
                    id if id == model_turn => true,
                    id if id == turn_end => false,
                    _ => writing,
                };
                writing
            })
            .collect();
        eprintln!("\n{file}: {} tokens, {} of them the model's own; the last {} also one by one", ids.len(), replies.iter().filter(|r| **r).count(), ids.len() - tail);

        let started = std::time::Instant::now();
        let whole = |name, model: &Model| Run { name, from: 0, states: as_a_prompt(model, &ids, &mut model.new_state()), logits: None };
        let truth = whole("checkpoint", &reference);
        let mut runs = vec![whole("checkpoint in floats", &in_floats), whole("file, CPU", &on_cpu), whole("file, GPU prompt", &on_gpu)];
        // The paths that generate: a prompt up to the tail, then the tail as generation and as additions see it.
        let mut state = on_gpu.new_state();
        let before = as_a_prompt(&on_gpu, &ids[..tail], &mut state);
        // The same prompt a second time, to tell a state that came out wrong from one that is always so.
        let mut again = on_gpu.new_state();
        as_a_prompt(&on_gpu, &ids[..tail], &mut again);
        let not_finite = |values: &[f32]| values.iter().filter(|v| !v.is_finite()).count();
        eprintln!("  prompt before the tail: {} values not finite", not_finite(&before));
        for (visit, (first, second)) in state.sizes().iter().zip(again.sizes()).enumerate() {
            for (part, (a, b)) in ["keys", "values", "convolution inputs"].iter().zip(first.iter().zip(second)) {
                if !a.is_finite() || !b.is_finite() || (a - b).abs() > 0.02 * b.abs() {
                    eprintln!("  STATE: visit {visit} {part}: {a} the first time, {b} the second");
                }
            }
        }
        drop(again);
        let (mut states, mut logits, mut stepped) = (Vec::new(), Vec::new(), state.clone());
        let mut first_bad = None;
        for (i, &id) in ids[tail..].iter().enumerate() {
            let mut forward = on_gpu.forward(&[id], &mut stepped, false).unwrap();
            let own = forward.logits.take().expect("the GPU decoder gives logits");
            if first_bad.is_none() && (not_finite(&forward.last) > 0 || not_finite(&own[0]) > 0) {
                first_bad = Some(i);
                eprintln!("  TOKENS: first result not finite at token {i} of the tail");
                for (visit, sizes) in stepped.sizes().iter().enumerate().filter(|(_, s)| s.iter().any(|v| !v.is_finite())) {
                    eprintln!("  TOKENS: visit {visit} holds {sizes:?}");
                }
            }
            logits.extend(own);
            states.extend(forward.last);
        }
        runs.push(Run { name: "file, GPU tokens", from: tail, states, logits: Some(logits) });
        let mut states = Vec::new();
        for chunk in ids[tail..].chunks(ADDITION) {
            states.extend(on_gpu.forward(chunk, &mut state, false).unwrap().last);
        }
        eprintln!("  additions: {} values not finite", not_finite(&states));
        runs.push(Run { name: "file, additions", from: tail, states, logits: None });
        let redone = |count: &std::sync::atomic::AtomicUsize| count.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!(
            "  redone on the CPU so far: {} batches of products, {} experts, {} stretches of attention",
            redone(&rustling::model::GPU_PRODUCTS_REDONE),
            redone(&rustling::model::GPU_EXPERTS_REDONE),
            redone(&rustling::model::GPU_ATTENTION_REDONE)
        );
        eprintln!("  (ran in {:.0}s)", started.elapsed().as_secs_f64());

        // Scores over everything, over the tail, and over the model's own writing in each.
        let mut scores = vec![[Score::default(), Score::default(), Score::default(), Score::default()]; runs.len()];
        let positions: Vec<usize> = (0..ids.len() - 1).collect();
        for chunk in positions.chunks(32) {
            let (first, count) = (chunk[0], chunk.len());
            let of = |model: &Model, states: &[f32], from: usize| -> Vec<Vec<f32>> {
                model.logits_many(&states[(first - from) * hidden..][..count * hidden]).unwrap()
            };
            let truth: Vec<Vec<f64>> = of(&reference, &truth.states, 0).iter().map(|l| log_probabilities(l)).collect();
            for (run, scores) in runs.iter().zip(&mut scores) {
                if first < run.from {
                    continue;
                }
                let logits = match (&run.logits, run.name) {
                    (Some(own), _) => own[first - run.from..][..count].to_vec(),
                    (None, "checkpoint in floats") => of(&reference, &run.states, run.from),
                    (None, _) => of(&on_cpu, &run.states, run.from),
                };
                for ((&t, truth), logits) in chunk.iter().zip(&truth).zip(&logits) {
                    let ours = log_probabilities(logits);
                    let kinds = [true, t >= tail, replies[t], t >= tail && replies[t]];
                    for (score, _) in scores.iter_mut().zip(kinds).filter(|(_, counted)| *counted) {
                        score.add(truth, &ours, ids[t + 1]);
                    }
                }
            }
        }
        for (kind, title) in ["every position", "the tail", "the model's own writing", "the model's own writing in the tail"].iter().enumerate() {
            eprintln!(" {title}:");
            for (run, scores) in runs.iter().zip(&scores) {
                scores[kind].report(run.name);
            }
        }
    }
}
