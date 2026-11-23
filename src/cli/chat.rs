//! `chat`: an interactive conversation.

use crate::*;

pub(crate) fn chat(model_dir: &std::path::Path, system: Option<&str>, run: &RunArgs) -> Result<()> {
    const DIM: &str = "\x1b[2m";
    const RESET: &str = "\x1b[0m";

    let config = load_config(model_dir)?;
    let checkpoint = Checkpoint::open(model_dir)?;
    let prompter = Prompter::load(model_dir)?;
    let model = Model::load(&config, &checkpoint, run.model())?;
    let mut sampler = run.sampler()?;
    let draft_weights = run.draft_weights(model_dir)?;
    let draft = draft_weights
        .as_ref()
        .map(|weights| Draft::load(&config, weights).map(|draft| draft.with_min_confidence(run.draft_confidence)))
        .transpose()?;
    let draft = draft.filter(|_| drafting_pays(&model));
    let new_session = || match &draft {
        Some(draft) => Session::new(&model).with_window(run.window()).with_copying(!run.no_copying).with_draft(draft),
        None => Session::new(&model).with_window(run.window()).with_copying(!run.no_copying),
    };
    let mut session = new_session();
    let mut messages: Vec<serde_json::Value> = Vec::new();
    if let Some(system) = system {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    let system_messages = messages.len();

    eprintln!("/reset starts over, /exit or Ctrl-D quits.");
    let mut editor = rustyline::DefaultEditor::new()?;
    loop {
        let line = match editor.readline("> ") {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Eof | rustyline::error::ReadlineError::Interrupted) => break,
            Err(error) => return Err(error.into()),
        };
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                messages.truncate(system_messages);
                session = new_session();
                continue;
            }
            text => {
                editor.add_history_entry(text)?;
                messages.push(serde_json::json!({ "role": "user", "content": text }));
            }
        }

        // The whole conversation is rendered each turn; the session only processes what it has not seen.
        let rendered = prompter.render(&serde_json::Value::Array(messages.clone()), None, run.effort().as_ref())?;
        let ids = prompter.encode(&rendered)?;
        let started = Instant::now();
        let reused = session.set_context(&ids)?;
        let prompt_seconds = started.elapsed().as_secs_f32();

        // Stream the reply part by part: reasoning dimmed, the answer plain.
        let started = Instant::now();
        let mut generated: Vec<u32> = Vec::new();
        let mut shown = (0, 0); // (part index, bytes of it printed)
        let mut show = |generated: &[u32]| -> Result<()> {
            let raw = prompter.decode(generated)?;
            if raw.ends_with('\u{FFFD}') {
                return Ok(());
            }
            let segments = parse_reply(&raw);
            for (index, segment) in segments.iter().enumerate().skip(shown.0) {
                let from = if index == shown.0 { shown.1 } else { 0 };
                let style = if segment.kind == SegmentKind::Thinking { DIM } else { "" };
                print!("{style}{}{RESET}", &segment.text[from.min(segment.text.len())..]);
                if segment.closed {
                    println!();
                }
                shown = if segment.closed { (index + 1, 0) } else { (index, segment.text.len()) };
            }
            std::io::stdout().flush()?;
            Ok(())
        };
        let (_, stop) = session.generate(&mut sampler, run.limits(), config.eos_token_id, |token, _| {
            generated.push(token);
            show(&generated)?;
            Ok(true)
        })?;
        let seconds = started.elapsed().as_secs_f32();

        let segments = parse_reply(&prompter.decode(&generated)?);
        if segments.last().is_some_and(|segment| !segment.closed) {
            println!();
        }
        messages.push(reply_message(&segments));
        if let Some(note) = stop_note(stop, run.limits()) {
            eprintln!("{DIM}{note}{RESET}");
        }
        eprintln!(
            "{DIM}{} new prompt tokens in {prompt_seconds:.1}s ({reused} reused); {} tokens at {:.1}/s{RESET}",
            ids.len() - reused,
            generated.len(),
            generated.len() as f32 / seconds
        );
    }
    Ok(())
}
