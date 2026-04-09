use burn::config::Config;
use burn::nn::{EmbeddingConfig, RmsNormConfig, LinearConfig};
use burn::tensor::backend::Backend;

use crate::layer::QuantCBLayer;
use crate::model::QuantCB;
use crate::mtp::MTPConfig;

#[derive(Config)]
pub struct QuantCBConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
    pub d_c: usize,
    pub d_c_q: usize,
    pub d_head_c: usize,
    pub d_rope: usize,
}

impl QuantCBConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> QuantCB<B> {
        let token_embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        
        let layers = (0..self.n_layers)
            .map(|_| QuantCBLayer::init(self, device))
            .collect();

        let norm_f = RmsNormConfig::new(self.d_model).init(device);
        let output = LinearConfig::new(self.d_model, self.vocab_size).init(device);

        let mtp_config = MTPConfig {
            d_model: self.d_model,
            n_heads: self.n_heads,
            vocab_size: self.vocab_size,
        };
        let mtp = mtp_config.init(device);

        let hallucination_probe = LinearConfig::new(self.d_model, 1).init(device);

        // Initialize the Thinking Gate
        // This projects the hidden state to a single value to control feedback
        let thinking_gate = LinearConfig::new(self.d_model, 1).init(device);

        QuantCB {
            token_embedding,
            layers,
            norm_f,
            output,
            mtp,
            hallucination_probe,
            thinking_gate,
        }
    }
}