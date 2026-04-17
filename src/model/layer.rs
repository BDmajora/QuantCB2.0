use burn::module::Module;
use burn::nn::{RmsNorm, RmsNormConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::model::config::QuantCBConfig;
use crate::model::kv_cache::KVCache;
use crate::model::mla::{MLAConfig, MultiHeadLatentAttention};
use crate::model::moe::MoELayer;

#[derive(Module, Debug)]
pub struct QuantCBLayer<B: Backend> {
    mla: MultiHeadLatentAttention<B>,
    moe: MoELayer<B>,
    norm1: RmsNorm<B>,
    norm2: RmsNorm<B>,
}

impl<B: Backend> QuantCBLayer<B> {
    pub fn init(config: &QuantCBConfig, device: &B::Device) -> Self {
        // MLA initialized with BitLinear projections (ternary)
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
            // MoE initialized with BitLinear experts (ternary)
            moe: MoELayer::init(
                config.d_model, 
                config.n_experts, 
                config.top_k, 
                device
            ),
            // RmsNorm is kept in higher precision (standard Burn implementation)
            // to maintain numerical stability between ternary blocks.
            norm1: RmsNormConfig::new(config.d_model).init(device),
            norm2: RmsNormConfig::new(config.d_model).init(device),
        }
    }

    pub fn forward(
        &self, 
        x: Tensor<B, 3>, 
        cache: Option<KVCache<B>>
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Option<KVCache<B>>) {
        // --- Multi-Head Latent Attention Block ---
        let residual = x.clone();
        
        // Normalize before the ternary projections
        let x_norm = self.norm1.forward(x);
        
        let (mla_out, new_cache) = self.mla.forward(x_norm, cache);
        
        // Residual connection: FP32/16 + Ternary-Output
        let x = residual + mla_out;

        // --- Mixture of Experts Block ---
        let residual = x.clone();
        
        // Second normalization to stabilize the MoE router inputs
        let x_norm = self.norm2.forward(x);
        
        let (moe_out, routing_loss) = self.moe.forward(x_norm);
        
        // Final output for this layer
        let out = residual + moe_out;

        (out, routing_loss, Some(new_cache))
    }
}