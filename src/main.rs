use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use safetensors::SafeTensors;
use clap::{Parser, Subcommand};
use rustling::checkpoint::{Checkpoint, ExpertFormat, ExpertMatrix, resolve_model};
use rustling::config::Config;
use rustling::draft::Draft;
use rustling::model::{Activations, Backend, Model, Options};
use rustling::prompt::{Prompter, SegmentKind, parse_reply, reply_message};
use rustling::sampler::{Sampler, SamplerConfig};
use rustling::session::{Limits, Session, Stop};

mod server;

#[derive(Parser)]
#[command(about = "Inference engine for Inkling 2 Mini")]
struct Cli {
    /// Model directory, or a model name to find in the Hugging Face cache. Defaults to $RUSTLING_MODEL, then
    /// thinkingmachines/Inkling-2-Mini.
    #[arg(long, global = true)]
    model: Option<PathBuf>,
    /// How many times to run the looped layers, for a variable-loop checkpoint
    /// (thinkingmachines/TML-d24-varloop-research, trained with 1 to 5). Fewer is faster and less capable.
    #[arg(long, global = true)]
    passes: Option<usize>,
    /// Worker threads. Defaults to the number of performance cores on Apple silicon, where work that lands on
    /// an efficiency core holds up the rest, and to every core elsewhere.
    #[arg(long, global = true, value_parser = clap::value_parser!(u64).range(1..))]
    threads: Option<u64>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Summarise the config, the layer visit order and the checkpoint's tensors.
    Inspect,
    /// Decode one expert and print statistics of its weights.
    Expert {
        #[arg(long)]
        layer: usize,
        #[arg(long)]
        expert: usize,
    },
    /// Render a single user message through the chat template and tokenize it.
    Prompt {
        text: String,
        /// Level name (none, minimal, low, medium, high, xhigh, max) or a number in 0.0..=0.99.
        #[arg(long)]
        effort: Option<String>,
    },
    /// Generate a reply to a single user message, printing the raw token text.
    Generate {
        text: String,
        #[command(flatten)]
        run: RunArgs,
        /// Also print the five most likely tokens at each step.
        #[arg(long)]
        top: bool,
        /// Print how the forward pass's time divides between its stages, for the prompt and for generation.
        #[arg(long)]
        timing: bool,
    },
    /// Serve an OpenAI-compatible API. The sampling flags set the defaults for requests that do not say.
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        /// Memory, in GiB, for keeping processed prompts so later requests that share them start from there.
        /// 0 keeps only the latest conversation.
        #[arg(long, default_value_t = 4.0)]
        cache_gb: f64,
        /// Keep the processed prompts in this directory between runs.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        #[command(flatten)]
        run: RunArgs,
    },
    /// Chat interactively. Each turn reuses the state of the conversation so far.
    Chat {
        /// A system message to start the conversation with.
        #[arg(long)]
        system: Option<String>,
        #[command(flatten)]
        run: RunArgs,
    },
    /// Compare the forward pass with a dump written by `tools/golden.py hidden`.
    /// Compare the first layer's intermediate tensors with a dump written by `tools/golden.py layer0`.
    Probe {
        golden: PathBuf,
        #[arg(long, value_enum, default_value_t = ActivationsArg::FloatBf16)]
        activations: ActivationsArg,
    },
    Compare {
        golden: PathBuf,
        /// Expert arithmetic. transformers does not quantise activations, so `float` is the like-for-like choice.
        #[arg(long, value_enum, default_value_t = ActivationsArg::Float)]
        activations: ActivationsArg,
    },
}

/// Options shared by the commands that run the model.
#[derive(clap::Args)]
struct RunArgs {
    /// Level name (none, minimal, low, medium, high, xhigh, max) or a number in 0.0..=0.99.
    #[arg(long)]
    effort: Option<String>,
    /// The most tokens to generate per reply.
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    /// The context window: the most tokens a prompt and its reply may take together. Longer prompts are refused
    /// and replies stop when it is full. Unlimited by default (the model was trained with up to 1,048,576).
    #[arg(long, value_name = "TOKENS", value_parser = clap::value_parser!(u64).range(2..))]
    context: Option<u64>,
    /// Sampling temperature; 0 always takes the most likely token. The model's own default is 1.
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    /// Sample only from the K most likely tokens; 0 disables the limit.
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    /// Sample only from the most likely tokens that together reach this probability; 1 disables the limit.
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// Seed for reproducible sampling.
    #[arg(long)]
    seed: Option<u64>,
    /// Keep generating even when the output is repeating itself.
    #[arg(long)]
    allow_loops: bool,
    /// Use the draft model beside the checkpoint (`mtp.safetensors`) to propose several tokens at a time.
    /// The output is distributed as without it.
    #[arg(long)]
    draft: bool,
    /// How sure the draft must be of a run of tokens to propose it, from 0 (always propose a full block) to 1.
    #[arg(long, default_value_t = rustling::draft::DEFAULT_MIN_CONFIDENCE)]
    draft_confidence: f32,
    /// Do not guess at coming tokens by copying from the context. Copying speeds up replies that repeat text
    /// already in the conversation, such as edits to a file, and rests where its guesses fail.
    #[arg(long)]
    no_copying: bool,
    /// Expert arithmetic: `quantised` as the model was trained, or `float` as transformers computes it.
    #[arg(long, value_enum, default_value_t = ActivationsArg::Quantised)]
    activations: ActivationsArg,
    /// Where the heavy matrix products run: `auto` uses the GPU for large batches when there is one, with
    /// results identical to the CPU's. `metal-fast` is quicker on long prompts but not bit-identical.
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,
}

impl RunArgs {
    fn sampler(&self) -> Result<Sampler> {
        anyhow::ensure!(self.temperature >= 0.0, "temperature must not be negative");
        anyhow::ensure!((0.0..=1.0).contains(&self.top_p), "top-p must be between 0 and 1");
        let config = SamplerConfig { temperature: self.temperature, top_k: self.top_k, top_p: self.top_p };
        Ok(Sampler::new(config, self.seed))
    }

    fn model(&self) -> Options {
        Options { activations: self.activations.into(), backend: self.backend.into() }
    }

    /// The draft model's weights, when `--draft` was given.
    fn draft_weights(&self, model_dir: &std::path::Path) -> Result<Option<Checkpoint>> {
        match self.draft {
            false => Ok(None),
            true => {
                let path = rustling::checkpoint::companion_dir(model_dir)?.join("mtp.safetensors");
                anyhow::ensure!(
                    path.exists(),
                    "{} is missing; fetch it with `hf download thinkingmachines/Inkling-2-Mini mtp.safetensors`",
                    path.display()
                );
                Ok(Some(Checkpoint::open_file(&path)?))
            }
        }
    }

    /// The context window in tokens.
    fn window(&self) -> usize {
        self.context.map_or(usize::MAX, |tokens| tokens as usize)
    }

    fn limits(&self) -> Limits {
        Limits { max_tokens: self.max_tokens, stop_on_loop: !self.allow_loops }
    }

    /// The reasoning effort as the chat template takes it: a number or a level name.
    fn effort(&self) -> Option<serde_json::Value> {
        self.effort.as_deref().map(|e| match e.parse::<f64>() {
            Ok(number) => serde_json::json!(number),
            Err(_) => serde_json::json!(e),
        })
    }
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum BackendArg {
    Auto,
    Cpu,
    Metal,
    /// Metal with a faster path for prompts of 256 tokens or more. Its results are not bit-identical to the other
    /// backends', and with quantised activations can be a different, equally valid reply.
    MetalFast,
}

impl From<BackendArg> for Backend {
    fn from(arg: BackendArg) -> Self {
        match arg {
            BackendArg::Auto => Backend::Auto,
            BackendArg::Cpu => Backend::Cpu,
            BackendArg::Metal => Backend::Metal,
            BackendArg::MetalFast => Backend::MetalFast,
        }
    }
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum ActivationsArg {
    #[value(alias = "quantized")]
    Quantised,
    Float,
    /// Float with transformers' bfloat16 rounding imitated; for checking against its outputs.
    FloatBf16,
}

impl From<ActivationsArg> for Activations {
    fn from(arg: ActivationsArg) -> Self {
        match arg {
            ActivationsArg::Quantised => Activations::Quantised,
            ActivationsArg::Float => Activations::Float,
            ActivationsArg::FloatBf16 => Activations::FloatBf16,
        }
    }
}

/// The `--passes` override, for [`load_config`].
static PASSES: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();

/// The model's config with the command line's pass count applied.
fn load_config(model_dir: &std::path::Path) -> Result<Config> {
    let config = Config::load(model_dir)?;
    match PASSES.get().copied().flatten() {
        Some(passes) => config.with_passes(passes),
        None => Ok(config),
    }
}

/// Whether a draft model can speed this model up; says why not when it cannot.
fn drafting_pays(model: &Model) -> bool {
    if model.decodes_on_gpu() {
        eprintln!("--draft is not used: tokens are generated on the GPU, where one costs less than checking a proposal");
    }
    !model.decodes_on_gpu()
}

/// How many performance cores an Apple silicon machine has; nothing elsewhere, or if the system will not say.
fn performance_cores() -> Option<usize> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let output = std::process::Command::new("sysctl").args(["-n", "hw.perflevel0.physicalcpu"]).output().ok()?;
    String::from_utf8(output.stdout).ok()?.trim().parse().ok().filter(|&cores| cores > 0)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let model_dir = resolve_model(cli.model.as_deref())?;
    PASSES.set(cli.passes).expect("set once");
    if let Some(threads) = cli.threads.map(|threads| threads as usize).or_else(performance_cores) {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    }
    match cli.command {
        Command::Inspect => inspect(&model_dir),
        Command::Expert { layer, expert } => expert_stats(&model_dir, layer, expert),
        Command::Prompt { text, effort } => prompt(&model_dir, &text, effort.as_deref()),
        Command::Generate { text, run, top, timing } => generate(&model_dir, &text, &run, top, timing),
        Command::Chat { system, run } => chat(&model_dir, system.as_deref(), &run),
        Command::Serve { host, port, cache_gb, cache_dir, run } => {
            anyhow::ensure!(cache_gb >= 0.0, "--cache-gb must not be negative");
            run.sampler()?;
            anyhow::ensure!(run.effort.is_none(), "--effort does not apply to serve: requests set reasoning_effort");
            let options = server::ServerOptions {
                host,
                port,
                model: run.model(),
                sampling: SamplerConfig { temperature: run.temperature, top_k: run.top_k, top_p: run.top_p },
                max_tokens: run.max_tokens,
                stop_on_loop: !run.allow_loops,
                draft_confidence: run.draft.then_some(run.draft_confidence),
                copying: !run.no_copying,
                passes: cli.passes,
                context: run.window(),
                cache_bytes: (cache_gb * (1u64 << 30) as f64) as usize,
                cache_dir,
            };
            server::serve(&model_dir, options)
        }
        Command::Probe { golden, activations } => probe(&model_dir, &golden, activations.into()),
        Command::Compare { golden, activations } => compare(&model_dir, &golden, activations.into()),
    }
}

fn inspect(model_dir: &std::path::Path) -> Result<()> {
    let config = load_config(model_dir)?;
    let text = &config.text_config;
    println!("model      {} at {}", config.model_type, model_dir.display());
    println!("decoder    {} layers, hidden {}, vocab {}", text.num_hidden_layers, text.hidden_size, text.vocab_size);
    println!(
        "experts    top-{} of {}, intermediate {}",
        text.num_experts_per_tok, text.n_routed_experts, text.intermediate_size
    );

    let visits = text.visits();
    let order: Vec<String> = visits
        .iter()
        .map(|v| format!("{}{}", v.layer, if text.is_local(v.layer) { "" } else { "*" }))
        .collect();
    println!("visits     {} (* = full attention): {}", visits.len(), order.join(" "));

    let checkpoint = Checkpoint::open(model_dir)?;
    let mut groups: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for (name, info) in checkpoint.tensors() {
        let group = match name.split('.').collect::<Vec<_>>().as_slice() {
            ["model", "llm", "layers", _, rest @ ..] if rest.contains(&"experts") => "llm experts".to_owned(),
            ["model", "llm", "layers", _, "controllers", ..] => "llm loop controllers".to_owned(),
            ["model", "llm", "layers", ..] => "llm layers (dense)".to_owned(),
            ["model", "llm", "embed" | "unembed", ..] => "llm embed + unembed".to_owned(),
            ["model", "llm", ..] => "llm other".to_owned(),
            ["model", "visual", ..] => "vision tower".to_owned(),
            _ => "other".to_owned(),
        };
        let entry = groups.entry(group).or_default();
        entry.0 += 1;
        entry.1 += info.byte_len();
    }
    println!("tensors");
    for (group, (count, bytes)) in &groups {
        println!("  {group:<22} {count:>5} tensors {:>8.2} GiB", *bytes as f64 / (1u64 << 30) as f64);
    }
    for layer in [0, text.num_hidden_layers - 1] {
        let experts = checkpoint.experts(layer)?;
        let format = match experts.gate_up.format {
            ExpertFormat::Nvfp4 { .. } => "NVFP4",
            ExpertFormat::Mxfp8 => "MXFP8",
            ExpertFormat::Ggml(_) => "GGUF blocks",
        };
        println!(
            "  layer {layer:<2} experts: {format}, gate_up {}x{}x{}, down {}x{}x{}",
            experts.gate_up.experts,
            experts.gate_up.rows,
            experts.gate_up.cols,
            experts.down.experts,
            experts.down.rows,
            experts.down.cols
        );
    }
    Ok(())
}

fn expert_stats(model_dir: &std::path::Path, layer: usize, expert: usize) -> Result<()> {
    let checkpoint = Checkpoint::open(model_dir)?;
    let experts = checkpoint.experts(layer)?;
    anyhow::ensure!(expert < experts.gate_up.experts, "layer has {} experts", experts.gate_up.experts);
    for (name, matrix) in [("gate_up", &experts.gate_up), ("down", &experts.down)] {
        let stats = Stats::of(matrix, expert);
        println!(
            "layer {layer} expert {expert} {name:<8} {}x{}  min {:+.5} max {:+.5} mean {:+.6} rms {:.5} zeros {:.1}% non-finite {}",
            matrix.rows,
            matrix.cols,
            stats.min,
            stats.max,
            stats.sum / stats.count as f64,
            (stats.sum_sq / stats.count as f64).sqrt(),
            100.0 * stats.zeros as f64 / stats.count as f64,
            stats.non_finite
        );
    }
    Ok(())
}

#[derive(Default)]
struct Stats {
    min: f32,
    max: f32,
    sum: f64,
    sum_sq: f64,
    zeros: usize,
    non_finite: usize,
    count: usize,
}

impl Stats {
    fn of(matrix: &ExpertMatrix, expert: usize) -> Self {
        let mut stats = Stats { min: f32::INFINITY, max: f32::NEG_INFINITY, ..Default::default() };
        let mut row = vec![0.0f32; matrix.cols];
        for r in 0..matrix.rows {
            matrix.decode_row(expert, r, &mut row);
            for &v in &row {
                stats.min = stats.min.min(v);
                stats.max = stats.max.max(v);
                stats.sum += v as f64;
                stats.sum_sq += (v as f64) * (v as f64);
                stats.zeros += (v == 0.0) as usize;
                stats.non_finite += !v.is_finite() as usize;
            }
            stats.count += row.len();
        }
        stats
    }
}

/// Renders one user message and tokenizes it.
fn user_prompt(prompter: &Prompter, text: &str, effort: Option<&str>) -> Result<(String, Vec<u32>)> {
    let messages = serde_json::json!([{ "role": "user", "content": text }]);
    let effort = effort.map(|e| match e.parse::<f64>() {
        Ok(number) => serde_json::json!(number),
        Err(_) => serde_json::json!(e),
    });
    let rendered = prompter.render(&messages, None, effort.as_ref())?;
    let ids = prompter.encode(&rendered)?;
    Ok((rendered, ids))
}

fn prompt(model_dir: &std::path::Path, text: &str, effort: Option<&str>) -> Result<()> {
    let prompter = Prompter::load(model_dir)?;
    let (rendered, ids) = user_prompt(&prompter, text, effort)?;
    println!("{rendered}");
    println!("{} tokens: {ids:?}", ids.len());
    Ok(())
}

fn print_stage_times(label: &str) {
    let stages = rustling::timing::take();
    let total: f64 = stages.iter().map(|(_, seconds)| seconds).sum();
    let parts: Vec<String> = stages
        .iter()
        .filter(|(_, seconds)| *seconds > 0.0)
        .map(|(stage, seconds)| format!("{stage:?} {seconds:.1}s"))
        .collect();
    eprintln!("{label} stages ({total:.1}s in all): {}", parts.join(", "));
}

fn stop_note(stop: Stop, limits: Limits) -> Option<String> {
    match stop {
        Stop::EndOfTurn | Stop::Requested => None,
        Stop::MaxTokens => Some(format!("stopped at the limit of {} tokens", limits.max_tokens)),
        Stop::Loop => Some("stopped: the output was repeating itself (--allow-loops to continue)".to_owned()),
        Stop::ContextFull => Some("stopped: the context window is full (--context sets its size)".to_owned()),
    }
}

fn generate(model_dir: &std::path::Path, text: &str, run: &RunArgs, top: bool, timing: bool) -> Result<()> {
    let config = load_config(model_dir)?;
    let checkpoint = Checkpoint::open(model_dir)?;
    let prompter = Prompter::load(model_dir)?;
    let started = Instant::now();
    let model = Model::load(&config, &checkpoint, run.model())?;
    let backend = if model.uses_gpu() { "Metal for large batches" } else { "CPU" };
    eprintln!("loaded in {:.1}s ({backend})", started.elapsed().as_secs_f32());

    let messages = serde_json::json!([{ "role": "user", "content": text }]);
    let rendered = prompter.render(&messages, None, run.effort().as_ref())?;
    let prompt_ids = prompter.encode(&rendered)?;
    eprintln!("{rendered}");

    // The prompt goes through in one call; after that each new token is a call of its own.
    let draft_weights = run.draft_weights(model_dir)?;
    let draft = draft_weights
        .as_ref()
        .map(|weights| Draft::load(&config, weights).map(|draft| draft.with_min_confidence(run.draft_confidence)))
        .transpose()?;
    let draft = draft.filter(|_| drafting_pays(&model));
    let mut session = match &draft {
        Some(draft) => Session::new(&model).with_window(run.window()).with_copying(!run.no_copying).with_draft(draft),
        None => Session::new(&model).with_window(run.window()).with_copying(!run.no_copying),
    };
    let started = Instant::now();
    session.set_context(&prompt_ids)?;
    let prompt_seconds = started.elapsed().as_secs_f32();
    if timing {
        print_stage_times("prompt");
        let redone = rustling::model::GPU_EXPERTS_REDONE.load(std::sync::atomic::Ordering::Relaxed);
        if redone > 0 {
            eprintln!("{redone} experts were recomputed on the CPU after the GPU returned NaN");
        }
        let redone = rustling::model::GPU_ATTENTION_REDONE.load(std::sync::atomic::Ordering::Relaxed);
        if redone > 0 {
            eprintln!("{redone} stretches of attention were recomputed on the CPU after the GPU returned NaN");
        }
    }

    let started = Instant::now();
    let mut generated: Vec<u32> = Vec::new();
    let mut printed = 0;
    let (_, stop) = session.generate(&mut run.sampler()?, run.limits(), config.eos_token_id, |token, logits| {
        generated.push(token);
        if top {
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.select_nth_unstable_by(5, |&a, &b| logits[b].total_cmp(&logits[a]));
            order[..5].sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            let candidates: Vec<String> = order[..5]
                .iter()
                .map(|&id| Ok(format!("{:?} {:.2}", prompter.decode(&[id as u32])?, logits[id])))
                .collect::<Result<_>>()?;
            eprintln!("[{}] {}", prompt_ids.len() + generated.len() - 1, candidates.join(" | "));
        }
        // Print whole characters only: a token can end in the middle of a multi-byte character.
        let text = prompter.decode(&generated)?;
        if !text.ends_with('\u{FFFD}') {
            print!("{}", &text[printed..]);
            std::io::stdout().flush()?;
            printed = text.len();
        }
        Ok(true)
    })?;
    println!();
    let seconds = started.elapsed().as_secs_f32();
    if timing {
        print_stage_times("generation");
    }
    if let Some(note) = stop_note(stop, run.limits()) {
        eprintln!("{note}");
    }
    let drafting = session.drafting();
    if drafting.copied_proposed > 0 {
        eprintln!(
            "copied: {} of {} proposed tokens accepted in {} rounds",
            drafting.copied_accepted, drafting.copied_proposed, drafting.copied_rounds
        );
    }
    if draft.is_some() {
        eprintln!(
            "draft: {} of {} proposed tokens accepted over {} rounds ({:.1} tokens per round); rested for {} tokens",
            drafting.accepted,
            drafting.proposed,
            drafting.rounds,
            (drafting.rounds + drafting.accepted) as f32 / drafting.rounds.max(1) as f32,
            drafting.rested
        );
    }
    eprintln!(
        "prompt: {} tokens in {prompt_seconds:.1}s; generated {} tokens in {seconds:.1}s ({:.2} tokens/s)",
        prompt_ids.len(),
        generated.len(),
        generated.len() as f32 / seconds
    );
    Ok(())
}

fn chat(model_dir: &std::path::Path, system: Option<&str>, run: &RunArgs) -> Result<()> {
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

fn compare(model_dir: &std::path::Path, golden: &std::path::Path, activations: Activations) -> Result<()> {
    let bytes = std::fs::read(golden)?;
    let golden = SafeTensors::deserialize(&bytes)?;
    let f32s = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
        let view = golden.tensor(name)?;
        let data = view.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        Ok((view.shape().to_vec(), data))
    };
    let ids: Vec<u32> =
        golden.tensor("input_ids")?.data().as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as u32).collect();
    let (shape, states) = f32s("hidden_states")?;
    let (_, logits) = f32s("logits")?;

    let config = load_config(model_dir)?;
    let checkpoint = Checkpoint::open(model_dir)?;
    let model = Model::load(&config, &checkpoint, Options { activations, backend: Backend::Cpu })?;
    let forward = model.forward(&ids, &mut model.new_state(), true)?;
    let visits = model.config().visits();
    // The reference may record the embedding ahead of the layer outputs; the visits are always the tail.
    anyhow::ensure!(shape[0] >= visits.len(), "golden has {} states, expected at least {}", shape[0], visits.len());
    let per_state = shape[1] * shape[2];
    let offset = (shape[0] - visits.len()) * per_state;

    // Relative error: |ours - reference| / |reference|, over the whole sequence.
    let relative = |ours: &[f32], reference: &[f32]| -> f64 {
        let diff: f64 = ours.iter().zip(reference).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let norm: f64 = reference.iter().map(|b| (*b as f64).powi(2)).sum();
        (diff / norm).sqrt()
    };
    println!("{} tokens", ids.len());
    // transformers reports the final normed state in place of the last layer's raw output, so that visit has no
    // counterpart; the final state is compared below instead.
    for (visit, ours) in visits.iter().zip(&forward.hidden).take(visits.len() - 1) {
        let reference = &states[offset + visit.index * per_state..][..per_state];
        println!(
            "visit {:>2}  layer {:>2} pass {}  relative error {:.2e}",
            visit.index,
            visit.layer,
            visit.pass,
            relative(ours, reference)
        );
    }
    let hidden = model.config().hidden_size;
    let vocab = model.config().unpadded_vocab_size;
    let reference_last = &states[states.len() - per_state..];
    println!("final normed state relative error {:.2e}", relative(&forward.last, reference_last));

    let prompter = Prompter::load(model_dir)?;
    let mut agree = 0;
    for (t, state) in forward.last.chunks_exact(hidden).enumerate() {
        let ours = model.logits(state)?;
        let reference = &logits[t * vocab..][..vocab];
        let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap();
        let (mine, theirs) = (argmax(&ours), argmax(reference));
        agree += (mine == theirs) as usize;
        if mine != theirs {
            // How close the two candidates are on each side shows whether this is a near-tie or a real difference.
            println!(
                "  position {t}: ours {:?} (ours {:.3}, reference {:.3}), reference {:?} (ours {:.3}, reference {:.3})",
                prompter.decode(&[mine as u32])?,
                ours[mine],
                reference[mine],
                prompter.decode(&[theirs as u32])?,
                ours[theirs],
                reference[theirs]
            );
        }
        if t + 1 == ids.len() {
            println!("last-token logits relative error {:.2e}", relative(&ours, reference));
        }
    }
    println!("top-1 agreement {agree}/{}", ids.len());
    Ok(())
}

fn probe(model_dir: &std::path::Path, golden: &std::path::Path, activations: Activations) -> Result<()> {
    let bytes = std::fs::read(golden)?;
    let golden = SafeTensors::deserialize(&bytes)?;
    let ids: Vec<u32> =
        golden.tensor("input_ids")?.data().as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as u32).collect();

    let config = load_config(model_dir)?;
    let checkpoint = Checkpoint::open(model_dir)?;
    let model = Model::load(&config, &checkpoint, Options { activations, backend: Backend::Cpu })?;
    rustling::trace::start();
    model.forward(&ids, &mut model.new_state(), false)?;
    let ours = rustling::trace::take();

    println!("{:<14} {:>10} {:>12} {:>12}", "tensor", "values", "rel. error", "max |diff|");
    for name in [
        "embed", "attn_in", "q", "k", "v", "bias", "attn", "attn_conv", "mlp_in", "router_logits", "mlp", "mlp_conv", "out",
    ] {
        let (Ok(view), Some(ours)) = (golden.tensor(name), ours.get(name)) else {
            println!("{name:<14} missing on one side");
            continue;
        };
        let reference: Vec<f32> = view.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        if reference.len() != ours.len() {
            println!("{name:<14} {} values here, {} in the reference {:?}", ours.len(), reference.len(), view.shape());
            continue;
        }
        let diff: f64 = ours.iter().zip(&reference).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let norm: f64 = reference.iter().map(|b| (*b as f64).powi(2)).sum();
        let max = ours.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("{name:<14} {:>10} {:>12.2e} {:>12.3e}", ours.len(), (diff / norm).sqrt(), max);
    }
    Ok(())
}
