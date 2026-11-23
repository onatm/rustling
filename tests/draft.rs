//! The draft model, and copying from the context, must only change speed. Needs the model, and `mtp.safetensors` for the drafting test; skipped
//! otherwise. Run with `--release`.

use rustling::weights::checkpoint::{Checkpoint, resolve_model};
use rustling::config::Config;
use rustling::generate::draft::Draft;
use rustling::model::{Activations, Backend, Model, Options};
use rustling::prompt::Prompter;
use rustling::generate::sampler::{Sampler, SamplerConfig};
use rustling::generate::session::{Limits, Session};

struct Loaded {
    config: Config,
    checkpoint: Checkpoint,
    prompter: Prompter,
    draft_weights: Option<Checkpoint>,
}

fn load() -> Option<Loaded> {
    let model_dir = resolve_model(None).ok()?;
    let checkpoint = Checkpoint::open(&model_dir).ok()?;
    Some(Loaded {
        config: Config::load(&model_dir).ok()?,
        checkpoint,
        prompter: Prompter::load(&model_dir).ok()?,
        draft_weights: Checkpoint::open_file(&model_dir.join("mtp.safetensors")).ok(),
    })
}

#[test]
fn taking_tokens_back_leaves_no_trace() {
    let Some(loaded) = load() else {
        eprintln!("skipping: model not found");
        return;
    };
    let options = Options { activations: Activations::Quantised, backend: Backend::Cpu };
    let model = Model::load(&loaded.config, &loaded.checkpoint, options).unwrap();
    let ids = loaded
        .prompter
        .encode("<|message_user|><|content_text|>Count: one two three four five six seven eight nine ten eleven<|end_message|>")
        .unwrap();
    let (base, rest) = ids.split_at(ids.len() - 9);
    let (wrong, right) = rest.split_at(6);

    // Straight through: the base, two of the "wrong" tokens, then the "right" ones.
    let mut straight = model.new_state();
    model.forward(base, &mut straight, false).unwrap();
    model.forward(&wrong[..2], &mut straight, false).unwrap();
    let expected = model.forward(right, &mut straight, false).unwrap();

    // With a detour: all six "wrong" tokens, four taken back.
    let mut detour = model.new_state();
    model.forward(base, &mut detour, false).unwrap();
    model.forward(wrong, &mut detour, false).unwrap();
    detour.truncate(4);
    assert_eq!(detour.position(), base.len() + 2);
    let actual = model.forward(right, &mut detour, false).unwrap();
    assert!(actual.last == expected.last, "state after taking tokens back differs");

    // Batched logits equal one-at-a-time logits.
    let hidden = model.config().hidden_size;
    let many = model.logits_many(&expected.last).unwrap();
    for (state, batched) in expected.last.chunks_exact(hidden).zip(&many) {
        assert!(model.logits(state).unwrap() == *batched, "batched logits differ");
    }
}

#[test]
fn drafting_does_not_change_greedy_output() {
    let Some(loaded) = load() else {
        eprintln!("skipping: model not found");
        return;
    };
    let Some(draft_weights) = &loaded.draft_weights else {
        eprintln!("skipping: mtp.safetensors not found");
        return;
    };
    let options = Options { activations: Activations::Quantised, backend: Backend::Cpu };
    let model = Model::load(&loaded.config, &loaded.checkpoint, options).unwrap();
    // A low threshold, so plenty of proposals are made and a good share of them rejected.
    let draft = Draft::load(&loaded.config, draft_weights).unwrap().with_min_confidence(0.1);
    let messages = serde_json::json!([{ "role": "user", "content": "Write a Python function that adds two numbers." }]);
    let rendered = loaded.prompter.render(&messages, None, Some(&serde_json::json!("none"))).unwrap();
    let ids = loaded.prompter.encode(&rendered).unwrap();
    let limits = Limits { max_tokens: 60, stop_on_loop: false };
    let greedy = || Sampler::new(SamplerConfig { temperature: 0.0, ..Default::default() }, None);

    let mut plain = Session::new(&model).with_copying(false);
    plain.set_context(&ids).unwrap();
    let (expected, expected_stop) = plain.generate(&mut greedy(), limits, loaded.config.eos_token_id, |_, _| Ok(true)).unwrap();

    let mut drafted = Session::new(&model).with_draft(&draft);
    drafted.set_context(&ids).unwrap();
    let (actual, stop) = drafted.generate(&mut greedy(), limits, loaded.config.eos_token_id, |_, _| Ok(true)).unwrap();

    assert_eq!(actual, expected, "drafting changed the greedy output");
    assert_eq!(stop, expected_stop);
    let stats = drafted.drafting();
    eprintln!("{stats:?}");
    assert!(stats.accepted > 0 && stats.accepted < stats.proposed, "expected both accepted and rejected proposals");
    // Both sessions end with the same context: everything generated except the token that ended generation.
    assert_eq!(drafted.context(), plain.context());
}

#[test]
fn copying_does_not_change_greedy_output() {
    let Some(loaded) = load() else {
        eprintln!("skipping: model not found");
        return;
    };
    let options = Options { activations: Activations::Quantised, backend: Backend::Cpu };
    let model = Model::load(&loaded.config, &loaded.checkpoint, options).unwrap();
    // A reply that repeats its prompt, and then itself.
    let line = "The quick brown fox jumps over the lazy dog near the old stone bridge at dawn.";
    let messages = serde_json::json!([{ "role": "user", "content": format!("Write this line out three times, each on its own line, and nothing else: {line}") }]);
    let rendered = loaded.prompter.render(&messages, None, Some(&serde_json::json!("none"))).unwrap();
    let ids = loaded.prompter.encode(&rendered).unwrap();
    let limits = Limits { max_tokens: 90, stop_on_loop: false };
    let greedy = || Sampler::new(SamplerConfig { temperature: 0.0, ..Default::default() }, None);

    let mut plain = Session::new(&model).with_copying(false);
    plain.set_context(&ids).unwrap();
    let (expected, expected_stop) = plain.generate(&mut greedy(), limits, loaded.config.eos_token_id, |_, _| Ok(true)).unwrap();
    assert_eq!(plain.drafting().copied_proposed, 0);

    let mut copying = Session::new(&model);
    copying.set_context(&ids).unwrap();
    let (actual, stop) = copying.generate(&mut greedy(), limits, loaded.config.eos_token_id, |_, _| Ok(true)).unwrap();

    assert_eq!(actual, expected, "copying changed the greedy output");
    assert_eq!(stop, expected_stop);
    let stats = copying.drafting();
    eprintln!("{stats:?}");
    assert!(stats.copied_accepted > 0, "expected some of the reply to be copied");
    assert_eq!(copying.context(), plain.context());
}
