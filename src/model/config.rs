use burn::config::Config;
use burn::nn::{EmbeddingConfig, RmsNormConfig};
use burn::tensor::backend::Backend;

// Import your custom BitLinearConfig
use crate::model::bitlinear::BitLinearConfig;

// Fixed imports to point to the new model subdirectory
use crate::model::layer::QuantCBLayer;
use crate::model::model::{QuantCB, AttnRes}; 
use crate::model::mtp::MTPConfig;

#[derive(Config, Debug)]
pub struct QuantCBConfig {
    #[config(default = 0)]
    pub vocab_size: usize,

    #[config(default = 512)]
    pub d_model: usize,

    #[config(default = 8)]
    pub n_heads: usize,

    #[config(default = 4)]
    pub n_layers: usize,

    #[config(default = 8)]
    pub n_experts: usize,

    #[config(default = 2)]
    pub top_k: usize,

    #[config(default = 256)]
    pub max_seq_len: usize,

    #[config(default = 0.1)]
    pub dropout: f64,

    #[config(default = 64)]
    pub d_c: usize,

    #[config(default = 64)]
    pub d_c_q: usize,

    #[config(default = 16)]
    pub d_head_c: usize,

    #[config(default = 16)]
    pub d_rope: usize,

    #[config(default = 2)]
    pub loop_depth: usize, 
}

impl QuantCBConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> QuantCB<B> {
        // 1. Core Architecture
        let token_embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        
        // 2. Sequential Transformation Layers (The "Body")
        let layers = (0..self.n_layers)
            .map(|_| QuantCBLayer::init(self, device))
            .collect();

        // 3. THE HYBRID UPGRADES (The "Brain")
        let loop_block = QuantCBLayer::init(self, device);
        let attn_res = AttnRes::new(self, device);

        // 4. Output & Auxiliary Heads (Now using BitLinearConfig)
        let norm_f = RmsNormConfig::new(self.d_model).init(device);
        
        // Output head converted to Ternary
        let output = BitLinearConfig::new(self.d_model, self.vocab_size).init(device);

        let mtp_config = MTPConfig {
            d_model: self.d_model,
            n_heads: self.n_heads,
            vocab_size: self.vocab_size,
        };
        let mtp = mtp_config.init(device);

        // Auxiliary probes converted to Ternary
        let hallucination_probe = BitLinearConfig::new(self.d_model, 1).init(device);
        let thinking_gate = BitLinearConfig::new(self.d_model, 1).init(device);

        // 5. Final Assembly
        QuantCB {
            token_embedding,
            layers,
            loop_block,
            attn_res,
            norm_f,
            output,
            mtp,
            hallucination_probe,
            thinking_gate,
            loop_depth: self.loop_depth,
        }
    }
}