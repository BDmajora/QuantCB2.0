use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::config::QuantCBConfig;
use crate::mla::{MLAConfig, MultiHeadLatentAttention};
use crate::moe::MoELayer;

/// A unified Decoder Block containing Latent Attention and Mixture of Experts
#[derive(Module, Debug)]
pub struct QuantCBLayer<B: Backend> {
    mla: MultiHeadLatentAttention<B>,
    moe: MoELayer<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
}

impl<B: Backend> QuantCBLayer<B> {
    pub fn init(config: &QuantCBConfig, device: &B::Device) -> Self {
        let mla_config = MLAConfig::new(
            config.d_model,
            config.n_heads,
            config.d_c,
            config.d_c_q,
            config.d_head_c,
            config.d_rope,
        );

        Self {
            mla: mla_config.init(device),
            moe: MoELayer::init(config.d_model, config.n_experts, config.top_k, device),
            norm1: LayerNormConfig::new(config.d_model).init(device),
            norm2: LayerNormConfig::new(config.d_model).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Pre-norm architecture
        let residual = x.clone();
        let x = self.norm1.forward(x);
        let x = residual + self.mla.forward(x);

        let residual = x.clone();
        let x = self.norm2.forward(x);
        residual + self.moe.forward(x)
    }
}