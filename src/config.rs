//! `config.json` of an Inkling 2 Mini checkpoint, as published (not the transformers-normalised form).

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub eos_token_id: u32,
    pub text_config: TextConfig,
    /// The DSpark draft model shipped beside the checkpoint, if any.
    pub mtp_config: Option<DraftConfig>,
}

/// The draft model in `mtp.safetensors`: a small transformer that proposes several next tokens at once from the
/// main model's hidden states.
#[derive(Debug, Clone, Deserialize)]
pub struct DraftConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// Layers that see only the sliding window of context before the block.
    pub local_layer_ids: Vec<usize>,
    pub sliding_window_size: usize,
    /// One anchor token plus `block_size - 1` drafted positions.
    pub block_size: usize,
    pub markov_rank: usize,
    pub markov_head_type: String,
    /// Layer visits of the main model whose input the draft reads; the last entry stands for the final normed
    /// state.
    pub target_layer_ids: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: f32,
    pub log_scaling_alpha: f32,
    pub rms_norm_eps: f32,
    pub qk_l2_clip_threshold: f32,
    pub rel_l2_clip_threshold: f32,
    /// Sliding-window layers; every other layer attends over the full context.
    pub local_layer_ids: Vec<usize>,
    pub sconv_kernel_size: usize,
    pub final_logit_softcapping: f32,
    pub sliding_window_size: usize,
    pub n_layers_per_attn_res_block: usize,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    /// Width of one expert's hidden layer (`moe_intermediate_size` in transformers).
    pub intermediate_size: usize,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
    pub route_scale: f32,
    pub loop_config: LoopConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoopConfig {
    /// First looped layer.
    pub core_start: usize,
    /// One past the last looped layer.
    pub core_end: usize,
    pub num_passes: usize,
    pub gate_rank: usize,
    pub is_varloop: bool,
}

impl Config {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let config: Config = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        config.text_config.validate()?;
        Ok(config)
    }
}

impl Config {
    /// Runs the looped layers `passes` times instead of the checkpoint's own count. Only checkpoints trained for a
    /// variable number of passes take this; fewer passes are faster and less capable, more are slower.
    pub fn with_passes(mut self, passes: usize) -> Result<Self> {
        let l = &mut self.text_config.loop_config;
        ensure!(passes >= 1, "the looped layers must run at least once");
        ensure!(
            l.is_varloop || passes == l.num_passes,
            "this checkpoint always runs its looped layers {} times; only a variable-loop checkpoint \
             (thinkingmachines/TML-d24-varloop-research) can run them {passes} times",
            l.num_passes
        );
        l.num_passes = passes;
        Ok(self)
    }
}

impl TextConfig {
    fn validate(&self) -> Result<()> {
        let block = self.n_layers_per_attn_res_block;
        let l = &self.loop_config;
        ensure!(
            0 < l.core_start && l.core_start < l.core_end && l.core_end <= self.num_hidden_layers,
            "loop core {}..{} does not fit in {} layers",
            l.core_start,
            l.core_end,
            self.num_hidden_layers
        );
        ensure!(
            l.core_start.is_multiple_of(block) && l.core_end.is_multiple_of(block),
            "loop core must align to attention-residual blocks of {block} layers"
        );
        ensure!(l.num_passes >= 1, "loop needs at least one pass");
        ensure!(self.num_attention_heads.is_multiple_of(self.num_key_value_heads), "query heads must be a multiple of KV heads");
        Ok(())
    }

    pub fn is_local(&self, layer: usize) -> bool {
        self.local_layer_ids.contains(&layer)
    }

    pub fn is_looped(&self, layer: usize) -> bool {
        (self.loop_config.core_start..self.loop_config.core_end).contains(&layer)
    }

    /// The order layers are applied in for one token: prelude, the core once per pass, then the coda.
    pub fn visits(&self) -> Vec<Visit> {
        let l = &self.loop_config;
        let mut visits = Vec::new();
        let mut push = |layer: usize, pass: usize| visits.push(Visit { index: 0, layer, pass });
        (0..l.core_start).for_each(|layer| push(layer, 0));
        for pass in 0..l.num_passes {
            (l.core_start..l.core_end).for_each(|layer| push(layer, pass));
        }
        (l.core_end..self.num_hidden_layers).for_each(|layer| push(layer, 0));
        for (index, visit) in visits.iter_mut().enumerate() {
            visit.index = index;
        }
        visits
    }
}

/// One application of a layer. Each visit owns its own KV cache and convolution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Visit {
    /// Position in execution order.
    pub index: usize,
    /// Which stored layer's weights are used.
    pub layer: usize,
    /// Loop pass, 0 outside the core.
    pub pass: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(is_varloop: bool) -> Config {
        let json = serde_json::json!({
            "model_type": "inkling2_loop", "eos_token_id": 200006,
            "text_config": {
                "hidden_size": 2304, "num_hidden_layers": 24, "vocab_size": 201024, "unpadded_vocab_size": 200058,
                "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128, "d_rel": 16, "rel_extent": 1024,
                "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-6,
                "qk_l2_clip_threshold": 22.6, "rel_l2_clip_threshold": 8.0,
                "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
                "sconv_kernel_size": 4, "final_logit_softcapping": 30.0, "sliding_window_size": 1024,
                "n_layers_per_attn_res_block": 6, "n_routed_experts": 128, "num_experts_per_tok": 8,
                "intermediate_size": 1152, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
                "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": is_varloop},
            },
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn pass_count_sets_the_visit_order() {
        assert_eq!(config(false).text_config.visits().len(), 48);
        for (passes, visits) in [(1, 24), (2, 36), (3, 48), (5, 72)] {
            let text = config(true).with_passes(passes).unwrap().text_config;
            let order = text.visits();
            assert_eq!(order.len(), visits, "{passes} passes");
            // Prelude once, the core once per pass, the coda once.
            assert_eq!(order.iter().filter(|v| v.layer == 3).count(), 1);
            assert_eq!(order.iter().filter(|v| v.layer == 9).count(), passes);
            assert_eq!(order.iter().filter(|v| v.layer == 20).count(), 1);
            assert!(order.iter().enumerate().all(|(i, v)| v.index == i));
        }
    }

    #[test]
    fn only_variable_loop_checkpoints_change_their_pass_count() {
        assert!(config(false).with_passes(2).is_err());
        assert!(config(false).with_passes(3).is_ok());
        assert!(config(true).with_passes(0).is_err());
    }
}
