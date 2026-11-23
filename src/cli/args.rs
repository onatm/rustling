//! The options shared by the commands that run the model, and the defaults that depend on the machine.

use crate::*;

/// Options shared by the commands that run the model.
#[derive(clap::Args)]
pub(crate) struct RunArgs {
    /// Level name (none, minimal, low, medium, high, xhigh, max) or a number in 0.0..=0.99.
    #[arg(long)]
    pub(crate) effort: Option<String>,
    /// The most tokens to generate per reply. A coding agent's turn can run to a few thousand: its reasoning,
    /// then a whole file in a tool call.
    #[arg(long, default_value_t = 4096)]
    pub(crate) max_tokens: usize,
    /// The context window: the most tokens a prompt and its reply may take together. Longer prompts are refused
    /// and replies stop when it is full. The default depends on the chip (16,384 on an M1, more on later ones),
    /// chosen so that a prompt filling the window is still processed in a couple of minutes; the model itself
    /// takes up to 1,048,576.
    #[arg(long, value_name = "TOKENS", value_parser = clap::value_parser!(u64).range(2..))]
    pub(crate) context: Option<u64>,
    /// Sampling temperature; 0 always takes the most likely token. The model's own default is 1.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) temperature: f32,
    /// Sample only from the K most likely tokens; 0 disables the limit.
    #[arg(long, default_value_t = 0)]
    pub(crate) top_k: usize,
    /// Sample only from the most likely tokens that together reach this probability; 1 disables the limit.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) top_p: f32,
    /// Seed for reproducible sampling.
    #[arg(long)]
    pub(crate) seed: Option<u64>,
    /// Keep generating even when the output is repeating itself.
    #[arg(long)]
    pub(crate) allow_loops: bool,
    /// Use the draft model beside the checkpoint (`mtp.safetensors`) to propose several tokens at a time.
    /// The output is distributed as without it.
    #[arg(long)]
    pub(crate) draft: bool,
    /// How sure the draft must be of a run of tokens to propose it, from 0 (always propose a full block) to 1.
    #[arg(long, default_value_t = rustling::generate::draft::DEFAULT_MIN_CONFIDENCE)]
    pub(crate) draft_confidence: f32,
    /// Do not guess at coming tokens by copying from the context. Copying speeds up replies that repeat text
    /// already in the conversation, such as edits to a file, and rests where its guesses fail.
    #[arg(long)]
    pub(crate) no_copying: bool,
    /// Expert arithmetic: `quantised` as the model was trained, or `float` as transformers computes it.
    #[arg(long, value_enum, default_value_t = ActivationsArg::Quantised)]
    pub(crate) activations: ActivationsArg,
    /// Where the model runs. `auto` uses the GPU when there is one: for a GGUF file that means prompts and
    /// generation both; for a checkpoint, large batches with results identical to the CPU's. `metal-fast` is the
    /// checkpoint's quicker path for long prompts, not bit-identical. `cpu` uses no GPU.
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,
}

impl RunArgs {
    pub(crate) fn sampler(&self) -> Result<Sampler> {
        anyhow::ensure!(self.temperature >= 0.0, "temperature must not be negative");
        anyhow::ensure!((0.0..=1.0).contains(&self.top_p), "top-p must be between 0 and 1");
        let config = SamplerConfig { temperature: self.temperature, top_k: self.top_k, top_p: self.top_p };
        Ok(Sampler::new(config, self.seed))
    }

    pub(crate) fn model(&self) -> Options {
        Options { activations: self.activations.into(), backend: self.backend.into() }
    }

    /// The draft model's weights, when `--draft` was given.
    pub(crate) fn draft_weights(&self, model_dir: &std::path::Path) -> Result<Option<Checkpoint>> {
        match self.draft {
            false => Ok(None),
            true => {
                let path = rustling::weights::checkpoint::companion_dir(model_dir)?.join("mtp.safetensors");
                anyhow::ensure!(
                    path.exists(),
                    "{} is missing; fetch it with `rustling pull --draft`",
                    path.display()
                );
                Ok(Some(Checkpoint::open_file(&path)?))
            }
        }
    }

    /// The context window in tokens.
    pub(crate) fn window(&self) -> usize {
        self.context.map_or_else(default_context, |tokens| tokens as usize)
    }

    pub(crate) fn limits(&self) -> Limits {
        Limits { max_tokens: self.max_tokens, stop_on_loop: !self.allow_loops }
    }

    /// The reasoning effort as the chat template takes it: a number or a level name.
    pub(crate) fn effort(&self) -> Option<serde_json::Value> {
        self.effort.as_deref().map(|e| match e.parse::<f64>() {
            Ok(number) => serde_json::json!(number),
            Err(_) => serde_json::json!(e),
        })
    }
}

#[derive(Clone, Copy, clap::ValueEnum)]
pub(crate) enum BackendArg {
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
pub(crate) enum ActivationsArg {
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
pub(crate) static PASSES: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();

/// The model's config with the command line's pass count applied.
pub(crate) fn load_config(model_dir: &std::path::Path) -> Result<Config> {
    let config = Config::load(model_dir)?;
    match PASSES.get().copied().flatten() {
        Some(passes) => config.with_passes(passes),
        None => Ok(config),
    }
}

/// Whether a draft model can speed this model up; says why not when it cannot.
pub(crate) fn drafting_pays(model: &Model) -> bool {
    let pays = !model.decodes_on_gpu() || rustling::generate::draft::tried_on_gpu();
    if !pays {
        tracing::warn!("--draft is not used: tokens are generated on the GPU, where one costs less than checking a proposal");
    }
    pays
}

/// The context window for each generation of Apple silicon, when none is asked for. Time decides it, not memory:
/// a prompt that fills the window should be processed in about two minutes, which on an M1 Max is 16,384 tokens.
/// Only that entry is measured; the later ones scale it by how much faster each generation's GPU is, roughly.
pub(crate) const CONTEXT_BY_CHIP: [(&str, usize); 5] =
    [("Apple M1", 16_384), ("Apple M2", 20_480), ("Apple M3", 24_576), ("Apple M4", 32_768), ("Apple M5", 40_960)];

/// For a chip the table does not know: the first entry's.
pub(crate) const CONTEXT_OTHERWISE: usize = 16_384;

/// The context window for a chip named as the system names it, such as "Apple M1 Max".
pub(crate) fn context_for(chip: &str) -> usize {
    let known = CONTEXT_BY_CHIP.iter().find(|(name, _)| {
        // "Apple M1" must not claim an "Apple M10".
        chip.strip_prefix(name).is_some_and(|rest| !rest.starts_with(|c: char| c.is_ascii_digit()))
    });
    known.map_or(CONTEXT_OTHERWISE, |&(_, tokens)| tokens)
}

pub(crate) fn default_context() -> usize {
    let chip = match cfg!(target_os = "macos") {
        true => std::process::Command::new("sysctl").args(["-n", "machdep.cpu.brand_string"]).output().ok(),
        false => None,
    };
    context_for(chip.and_then(|output| String::from_utf8(output.stdout).ok()).as_deref().unwrap_or("").trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub(crate) fn the_context_follows_the_chip() {
        assert_eq!(context_for("Apple M1 Max"), 16_384);
        assert_eq!(context_for("Apple M4 Max"), 32_768);
        assert_eq!(context_for("Apple M5"), 40_960);
        assert_eq!(context_for("Apple M10 Max"), CONTEXT_OTHERWISE);
        assert_eq!(context_for("Intel(R) Core(TM) i9"), CONTEXT_OTHERWISE);
        assert_eq!(context_for(""), CONTEXT_OTHERWISE);
    }
}

/// How many performance cores an Apple silicon machine has; nothing elsewhere, or if the system will not say.
pub(crate) fn performance_cores() -> Option<usize> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let output = std::process::Command::new("sysctl").args(["-n", "hw.perflevel0.physicalcpu"]).output().ok()?;
    String::from_utf8(output.stdout).ok()?.trim().parse().ok().filter(|&cores| cores > 0)
}
