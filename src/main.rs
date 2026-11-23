use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use safetensors::SafeTensors;
use clap::{Parser, Subcommand};
use rustling::weights::checkpoint::{Checkpoint, ExpertFormat, ExpertMatrix, DEFAULT_MODEL};
use rustling::config::Config;
use rustling::generate::draft::Draft;
use rustling::model::{Activations, Backend, Model, Options};
use rustling::prompt::{Prompter, SegmentKind, parse_reply, reply_message};
use rustling::generate::sampler::{Sampler, SamplerConfig};
use rustling::generate::session::{Limits, Session, Stop};

mod cli;
mod server;

use cli::args::*;
use cli::chat::*;
use cli::dev::*;
use cli::generate::*;
use cli::inspect::*;

#[derive(Parser)]
#[command(about = "Inference engine for Inkling 2 Mini")]
struct Cli {
    /// A model directory, a `.gguf` file, or a model name, which is downloaded if it is not on this machine.
    /// Defaults to $RUSTLING_MODEL, then lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M.
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
    /// How much to report on stderr: `info` gives loading, prompt and reply speeds; `debug` adds progress through
    /// long prompts and the rendered prompt; `warn` only problems; `off` nothing. Also `$RUSTLING_LOG`.
    #[arg(long, global = true, value_enum, env = "RUSTLING_LOG", default_value_t = LogLevel::Info)]
    log_level: LogLevel,
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum LogLevel {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Sends the log to stderr at the level asked for. A server's lines carry the time; a command's do not.
fn start_logging(level: LogLevel, timed: bool) {
    use tracing_subscriber::Layer;
    use tracing_subscriber::filter::LevelFilter;
    let level = match level {
        LogLevel::Off => LevelFilter::OFF,
        LogLevel::Error => LevelFilter::ERROR,
        LogLevel::Warn => LevelFilter::WARN,
        LogLevel::Info => LevelFilter::INFO,
        LogLevel::Debug => LevelFilter::DEBUG,
        LogLevel::Trace => LevelFilter::TRACE,
    };
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    // The libraries underneath, the Hub client above all, log a good deal of their own: only their problems.
    let ours = tracing_subscriber::filter::Targets::new().with_default(LevelFilter::WARN.min(level)).with_target("rustling", level);
    // Colour only where someone is looking: a log sent to a file should hold plain text.
    let coloured = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let log = tracing_subscriber::fmt::layer().with_writer(std::io::stderr).with_target(false).with_ansi(coloured);
    match timed {
        true => tracing_subscriber::registry().with(log.with_filter(ours)).init(),
        false => tracing_subscriber::registry().with(log.without_time().with_filter(ours)).init(),
    }
}

#[derive(Subcommand)]
enum Command {
    /// Download a model to the Hugging Face cache. The commands that run a model do this themselves the first
    /// time; this is for fetching ahead.
    Pull {
        /// A model name, such as `thinkingmachines/Inkling-2-Mini` or a GGUF repository and quantisation like
        /// `lmstudio-community/Inkling-2-Mini-GGUF:Q4_K_M`. Defaults to `--model`, then the default model.
        name: Option<String>,
        /// Also fetch the draft model, which `--draft` needs.
        #[arg(long)]
        draft: bool,
    },
    /// Summarise the config, the layer visit order and the checkpoint's tensors.
    Inspect,
    /// Decode one expert and print statistics of its weights.
    #[command(hide = true)]
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
    /// Compare the first layer's intermediate tensors with a dump written by `tools/golden.py layer0`.
    #[command(hide = true)]
    Probe {
        golden: PathBuf,
        #[arg(long, value_enum, default_value_t = ActivationsArg::FloatBf16)]
        activations: ActivationsArg,
    },
    /// Compare the forward pass with a dump written by `tools/golden.py hidden`.
    #[command(hide = true)]
    Compare {
        golden: PathBuf,
        /// Expert arithmetic. transformers does not quantise activations, so `float` is the like-for-like choice.
        #[arg(long, value_enum, default_value_t = ActivationsArg::Float)]
        activations: ActivationsArg,
    },
}

fn main() -> Result<()> {
    // The Hub client reports download statistics to Hugging Face unless told not to. Nothing else here phones
    // home, so neither does that, unless the user has asked for it by name.
    if std::env::var_os("HF_XET_TELEMETRY_ENABLED").is_none() {
        // SAFETY: nothing else is running yet to read the environment at the same time.
        unsafe { std::env::set_var("HF_HUB_DISABLE_TELEMETRY", "1") };
    }
    let cli = Cli::parse();
    start_logging(cli.log_level, matches!(cli.command, Command::Serve { .. }));
    let drafting = match &cli.command {
        Command::Generate { run, .. } | Command::Chat { run, .. } | Command::Serve { run, .. } => run.draft,
        _ => false,
    };
    let model_dir = match &cli.command {
        Command::Pull { name, draft } => {
            let named = name.clone().or_else(|| cli.model.as_ref().map(|model| model.to_string_lossy().into_owned()));
            let name = named.or_else(|| std::env::var("RUSTLING_MODEL").ok()).unwrap_or_else(|| DEFAULT_MODEL.to_owned());
            return rustling::weights::hub::pull(&name, *draft);
        }
        // What these look at is the published checkpoint's own formats.
        Command::Expert { .. } | Command::Probe { .. } | Command::Compare { .. } if cli.model.is_none() => {
            rustling::weights::checkpoint::resolve_checkpoint()?
        }
        _ => rustling::weights::hub::ensure_model(cli.model.as_deref(), drafting)?,
    };
    PASSES.set(cli.passes).expect("set once");
    if let Some(threads) = cli.threads.map(|threads| threads as usize).or_else(performance_cores) {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    }
    match cli.command {
        Command::Pull { .. } => unreachable!("handled above"),
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
