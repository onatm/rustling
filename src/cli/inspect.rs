//! Commands that look at the model without running it: its layout, an expert's weights, a rendered prompt.

use crate::*;

pub(crate) fn inspect(model_dir: &std::path::Path) -> Result<()> {
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

pub(crate) fn expert_stats(model_dir: &std::path::Path, layer: usize, expert: usize) -> Result<()> {
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
pub(crate) struct Stats {
    pub(crate) min: f32,
    pub(crate) max: f32,
    pub(crate) sum: f64,
    pub(crate) sum_sq: f64,
    pub(crate) zeros: usize,
    pub(crate) non_finite: usize,
    pub(crate) count: usize,
}

impl Stats {
    pub(crate) fn of(matrix: &ExpertMatrix, expert: usize) -> Self {
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
pub(crate) fn user_prompt(prompter: &Prompter, text: &str, effort: Option<&str>) -> Result<(String, Vec<u32>)> {
    let messages = serde_json::json!([{ "role": "user", "content": text }]);
    let effort = effort.map(|e| match e.parse::<f64>() {
        Ok(number) => serde_json::json!(number),
        Err(_) => serde_json::json!(e),
    });
    let rendered = prompter.render(&messages, None, effort.as_ref())?;
    let ids = prompter.encode(&rendered)?;
    Ok((rendered, ids))
}

pub(crate) fn prompt(model_dir: &std::path::Path, text: &str, effort: Option<&str>) -> Result<()> {
    let prompter = Prompter::load(model_dir)?;
    let (rendered, ids) = user_prompt(&prompter, text, effort)?;
    println!("{rendered}");
    println!("{} tokens: {ids:?}", ids.len());
    Ok(())
}
