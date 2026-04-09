use burn::config::Config;
use burn::nn::{EmbeddingConfig, LayerNormConfig, LinearConfig};
use burn::tensor::backend::Backend;

use crate::layer::QuantCBLayer;
use crate::model::QuantCB;

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
    
    // New MLA Architecture Parameters
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

        let norm_f = LayerNormConfig::new(self.d_model).init(device);
        let output = LinearConfig::new(self.d_model, self.vocab_size).init(device);

        QuantCB {
            token_embedding,
            layers,
            norm_f,
            output,
        }
    }
}