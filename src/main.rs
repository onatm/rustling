use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use safetensors::SafeTensors;
use clap::{Parser, Subcommand};
use rustling::weights::checkpoint::{Checkpoint, ExpertFormat, ExpertMatrix, resolve_model};
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
