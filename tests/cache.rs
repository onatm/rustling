//! Reusing processed prompts must be invisible apart from the time it saves: a session brought to a prompt through
//! the cache has to end in exactly the state a fresh session reaches. Needs the model; skipped otherwise. Run with
//! `--release`.

use std::path::Path;

use rustling::generate::cache::PromptCache;
use rustling::weights::checkpoint::{Checkpoint, resolve_checkpoint};
use rustling::config::Config;
use rustling::model::{Activations, Backend, Model, Options};
use rustling::prompt::Prompter;
use rustling::generate::session::Session;

const GIB: usize = 1 << 30;

#[cfg(target_os = "macos")]
#[test]
fn gpu_branches_keep_a_cached_prefix_independent() {
    use rustling::weights::checkpoint::{DEFAULT_MODEL, resolve_model};
    let Ok(path) = resolve_model(Some(Path::new(DEFAULT_MODEL))) else { return };
    let weights = Checkpoint::open(&path).unwrap();
    let config = Config::load(&path).unwrap();
    let model = Model::load(&config, &weights, Options { activations: Activations::Quantised, backend: Backend::MetalFast }).unwrap();
    let prefix: Vec<u32> = (500..596).collect();
    let mut reference = Session::new(&model);
    reference.set_context(&prefix).unwrap();
    let saved = reference.snapshot();
    let mut session = Session::new(&model);
    let mut cache = PromptCache::new(GIB);
    for (i, suffix) in [vec![601, 602], vec![701, 702, 703], vec![801]].iter().enumerate() {
        let prompt = [prefix.as_slice(), suffix].concat();
        reference.restore(saved.clone()).unwrap();
        reference.set_context(&prompt).unwrap();
        let reused = cache.prepare(&mut session, &prompt, Some(prefix.len())).unwrap();
        assert_eq!(reused, if i == 0 { 0 } else { prefix.len() });
        assert!(session.logits().unwrap() == reference.logits().unwrap(), "branch {i} changed logits");
    }
}

#[test]
fn cached_prompts_give_the_same_state_as_fresh_ones() {
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
    let options = Options { activations: Activations::Quantised, backend: Backend::Cpu };
    let model = Model::load(&config, &checkpoint, options).unwrap();

    // A header long enough to be worth a snapshot, shared by two conversations.
    let system = "You answer questions about a small bookshop. ".to_owned()
        + &(1..=8).map(|i| format!("Rule {i}: answer questions on topic {i} briefly. ")).collect::<String>();
    let encode = |messages: serde_json::Value| -> Vec<u32> {
        prompter.encode(&prompter.render(&messages, None, Some(&serde_json::json!("none"))).unwrap()).unwrap()
    };
    let header = encode(serde_json::json!([{"role": "system", "content": system}]));
    let conversation = |question: &str| {
        encode(serde_json::json!([{"role": "system", "content": system}, {"role": "user", "content": question}]))
    };
    let (a, b) = (conversation("Do you sell maps?"), conversation("What are your opening hours?"));
    assert!(a.starts_with(&header) && b.starts_with(&header) && header.len() >= 64, "{} header tokens", header.len());
    // A's next turn: its first prompt, a reply, and a follow-up question.
    let a_next: Vec<u32> = a.iter().copied().chain(prompter.encode("<|message_model|><|content_text|>Yes.<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>And guides?<|end_message|>").unwrap()).collect();

    let fresh = |prompt: &[u32]| {
        let mut session = Session::new(&model);
        session.set_context(prompt).unwrap();
        session.logits().unwrap()
    };

    let mut cache = PromptCache::new(4 * GIB);
    let mut session = Session::new(&model);
    // First conversation: nothing to reuse, and the header gets a snapshot.
    assert_eq!(cache.prepare(&mut session, &a, Some(header.len())).unwrap(), 0);
    assert_eq!(cache.len(), 1);
    assert!(session.logits().unwrap() == fresh(&a), "first prompt differs");
    // Second conversation: starts from the header. The first conversation is kept.
    assert_eq!(cache.prepare(&mut session, &b, Some(header.len())).unwrap(), header.len());
    assert_eq!(cache.len(), 2);
    assert!(session.logits().unwrap() == fresh(&b), "prompt restored from the header differs");
    // Back to the first conversation: all of its earlier prompt is reused.
    assert_eq!(cache.prepare(&mut session, &a_next, Some(header.len())).unwrap(), a.len());
    let expected = fresh(&a_next);
    assert!(session.logits().unwrap() == expected, "prompt restored from a kept conversation differs");

    // Saved and read back, the cache serves the same prompt from disk, in full this time.
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("prompt-cache-test");
    let _ = std::fs::remove_dir_all(&dir);
    cache.keep(&session);
    assert!(cache.unsaved());
    let written = cache.save(&dir, "test model").unwrap();
    assert_eq!(written, cache.len(), "every snapshot is new to the directory");
    // A second save finds nothing changed and writes nothing.
    assert!(!cache.unsaved());
    assert_eq!(cache.save(&dir, "test model").unwrap(), 0);
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), cache.len(), "one file per snapshot");
    assert!(PromptCache::load(&dir, "another model", 4 * GIB).unwrap().is_empty());
    let mut cache = PromptCache::load(&dir, "test model", 4 * GIB).unwrap();
    assert!(!cache.is_empty());
    let mut session = Session::new(&model);
    assert_eq!(cache.prepare(&mut session, &a_next, Some(header.len())).unwrap(), a_next.len());
    assert!(session.logits().unwrap() == expected, "prompt restored from disk differs");

    // A budget too small for anything keeps nothing, and everything still works.
    let mut cache = PromptCache::load(&dir, "test model", 1024).unwrap();
    assert!(cache.is_empty());
    let mut session = Session::new(&model);
    assert_eq!(cache.prepare(&mut session, &b, Some(header.len())).unwrap(), 0);
    assert!(session.logits().unwrap() == fresh(&b));
    let _ = std::fs::remove_dir_all(&dir);

    // A prompt asked again after its reply: the session steps back to where the prompt ended, with nothing
    // processed and no snapshot taken. Long enough that the sliding-window layers have dropped tokens since,
    // which is the part of the state that stepping back has to restore from a copy.
    let text = |tokens: std::ops::Range<usize>| -> Vec<u32> {
        let words: String = tokens.map(|i| format!("item {i} is on shelf {}. ", i * 7 % 13)).collect();
        prompter.encode(&words).unwrap()
    };
    let prompt: Vec<u32> = a.iter().copied().chain(text(0..200)).take(1060).collect();
    assert_eq!(prompt.len(), 1060);
    let reply = text(300..340);
    let longer: Vec<u32> = prompt.iter().chain(&reply[..100]).copied().collect();
    let longest: Vec<u32> = prompt.iter().chain(&reply[..150]).copied().collect();
    let mut cache = PromptCache::new(4 * GIB);
    let mut session = Session::new(&model);
    cache.prepare(&mut session, &prompt, None).unwrap();
    let at_prompt = session.logits().unwrap();
    // The reply, in two steps as generation would add it.
    session.set_context(&longer).unwrap();
    session.set_context(&longest).unwrap();
    assert_eq!(cache.prepare(&mut session, &prompt, None).unwrap(), prompt.len());
    assert!(cache.is_empty(), "stepping back needs no snapshot");
    assert!(session.logits().unwrap() == at_prompt, "the prompt stepped back to differs");
    // And a different continuation from there matches a fresh session.
    let other: Vec<u32> = prompt.iter().copied().chain(text(400..410)).collect();
    session.set_context(&longest).unwrap();
    assert_eq!(cache.prepare(&mut session, &other, None).unwrap(), prompt.len());
    assert!(session.logits().unwrap() == fresh(&other), "a continuation after stepping back differs");
}
