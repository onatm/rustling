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
