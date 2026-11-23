//! Fetching models from the Hugging Face Hub into its shared cache, where [`resolve_model`] finds them and where
//! other tools that use the Hub will find them too.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};
use hf_hub::HFClientSync;
use hf_hub::repository::RepoTreeEntry;
use indicatif::{ProgressBar, ProgressStyle};

use super::checkpoint::{BASE_MODEL, DEFAULT_MODEL, companion_dir, resolve_model};

/// The files that go with any weights: the config, the tokenizer and the chat template.
const COMPANIONS: [&str; 2] = ["*.json", "*.jinja"];
/// The draft model, which only `--draft` uses.
const DRAFT: &str = "mtp.safetensors";

/// Finds the model as [`resolve_model`] does, first downloading it when it is named and not on this machine.
/// `draft` also fetches the draft model.
pub fn ensure_model(explicit: Option<&Path>, draft: bool) -> Result<PathBuf> {
    let found = resolve_model(explicit).and_then(|model| {
        let companions = companion_dir(&model)?;
        ensure!(!draft || companions.join(DRAFT).is_file(), "the draft model is not downloaded");
        Ok(model)
    });
    let Err(missing) = found else { return found };
    let named = explicit.map(Path::to_path_buf).or_else(|| std::env::var_os("RUSTLING_MODEL").map(PathBuf::from));
    let name = match &named {
        // Something that looks like a place on this machine is not a model to go and fetch.
        Some(path) if path.exists() || path.is_absolute() || path.extension().is_some() => return Err(missing),
        Some(path) => path.to_string_lossy().into_owned(),
        None => DEFAULT_MODEL.to_owned(),
    };
    pull(&name, draft).with_context(|| format!("downloading {name}"))?;
    resolve_model(explicit)
}

/// Downloads a model by name: a checkpoint repository such as `thinkingmachines/Inkling-2-Mini`, or a GGUF
/// repository and the quantisation wanted from it, such as `unsloth/Inkling-2-Mini-GGUF:Q4_K_M`. A GGUF
/// file brings along the config and tokenizer of the checkpoint it was converted from.
pub fn pull(name: &str, draft: bool) -> Result<()> {
    let (repo, quantisation) = match name.split_once(':') {
        Some((repo, quantisation)) => (repo, Some(quantisation)),
        None => (name, None),
    };
    let Some((owner, model)) = repo.split_once('/').filter(|(_, model)| !model.contains('/')) else {
        bail!("{name} is not a model name like {BASE_MODEL} or {DEFAULT_MODEL}");
    };
    let client = HFClientSync::new().context("setting up the Hugging Face client")?;
    let repository = client.model(owner, model);
    let files: Vec<(String, u64)> = repository
        .list_tree()
        .recursive(true)
        .send()
        .with_context(|| format!("listing the files of {repo}"))?
        .into_iter()
        .filter_map(|entry| match entry {
            RepoTreeEntry::File { path, size, .. } => Some((path, size)),
            RepoTreeEntry::Directory { .. } => None,
        })
        .collect();

    let ggufs: Vec<&(String, u64)> = files
        .iter()
        .filter(|(path, _)| {
            let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
            name.ends_with(".gguf") && !name.starts_with("mmproj") && !name.starts_with("dspark")
        })
        .collect();
    if ggufs.is_empty() {
        ensure!(quantisation.is_none(), "{repo} has no GGUF files to pick {} from", quantisation.unwrap_or_default());
        let wanted = |path: &str| draft || path != DRAFT;
        let bytes: u64 = files.iter().filter(|(path, _)| wanted(path)).map(|(_, size)| size).sum();
        tracing::info!("downloading {repo} ({:.1} GB) to the Hugging Face cache", bytes as f64 / 1e9);
        let ignored = if draft { Vec::new() } else { vec![DRAFT.to_owned()] };
        repository.snapshot_download().ignore_patterns(ignored).progress(bar()).send()?;
        return Ok(());
    }

    let matching: Vec<&&(String, u64)> = ggufs
        .iter()
        .filter(|(path, _)| {
            let stem = path.trim_end_matches(".gguf").to_ascii_lowercase();
            quantisation.is_none_or(|q| stem.ends_with(&q.to_ascii_lowercase()))
        })
        .collect();
    let [(file, bytes)] = matching[..] else {
        let names: Vec<&str> = ggufs.iter().map(|(path, _)| path.as_str()).collect();
        bail!("{name} does not pick out one file; {repo} has: {}", names.join(", "));
    };
    tracing::info!("downloading {file} ({:.1} GB) from {repo} to the Hugging Face cache", *bytes as f64 / 1e9);
    repository.download_file().filename(file.clone()).progress(bar()).send()?;

    // The checkpoint the file was converted from lends its config, tokenizer and chat template.
    let base = std::env::var("RUSTLING_BASE_MODEL").unwrap_or_else(|_| BASE_MODEL.to_owned());
    let (owner, model) = base.split_once('/').with_context(|| format!("{base} is not a model name"))?;
    let mut patterns: Vec<String> = COMPANIONS.iter().map(|pattern| (*pattern).to_owned()).collect();
    patterns.extend(draft.then(|| DRAFT.to_owned()));
    tracing::info!("downloading the config and tokenizer from {base}");
    client.model(owner, model).snapshot_download().allow_patterns(patterns).progress(bar()).send()?;
    Ok(())
}

/// A progress bar fed by the download.
struct Bar {
    bar: ProgressBar,
    /// Bytes done per file: progress arrives as changes to single files.
    files: Mutex<HashMap<String, u64>>,
}

fn bar() -> Progress {
    let style = ProgressStyle::with_template("{bar:40} {bytes}/{total_bytes} {bytes_per_sec} eta {eta}")
        .expect("a valid template");
    Progress::new(Bar { bar: ProgressBar::hidden().with_style(style), files: Mutex::default() })
}

impl ProgressHandler for Bar {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else { return };
        match event {
            DownloadEvent::Start { total_bytes, .. } => {
                self.bar.set_draw_target(indicatif::ProgressDrawTarget::stderr());
                self.bar.set_length(*total_bytes);
            }
            DownloadEvent::Progress { files } => {
                let mut done = self.files.lock().expect("progress lock");
                for file in files {
                    done.insert(file.filename.clone(), file.bytes_completed);
                }
                self.bar.set_position(done.values().sum());
            }
            DownloadEvent::AggregateProgress { bytes_completed, total_bytes, .. } => {
                self.bar.set_length((*total_bytes).max(self.bar.length().unwrap_or(0)));
                self.bar.set_position(*bytes_completed);
            }
            DownloadEvent::Complete => self.bar.finish_and_clear(),
        }
    }
}
