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
    // The "stabilizer" scale factor
    residual_scale: f32,
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

        // STABILIZATION LOGIC:
        // We scale the contribution of each ternary block by 1/sqrt(depth).
        // If your loop_depth is 3, this is ~0.57. If it's 12, it's ~0.28.
        let scale = 1.0 / (config.loop_depth as f32).sqrt().max(1.0);

        Self {
            mla: mla_config.init(device),
            moe: MoELayer::init(
                config.d_model, 
                config.n_experts, 
                config.top_k, 
                device
            ),
            norm1: RmsNormConfig::new(config.d_model).init(device),
            norm2: RmsNormConfig::new(config.d_model).init(device),
            residual_scale: scale,
        }
    }

    pub fn forward(
        &self, 
        x: Tensor<B, 3>, 
        cache: Option<KVCache<B>>
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Option<KVCache<B>>) {
        // --- Multi-Head Latent Attention Block ---
        let residual = x.clone();
        
        // 1. Pre-Norm (crucial for ternary stability)
        let x_norm = self.norm1.forward(x);
        
        let (mla_out, new_cache) = self.mla.forward(x_norm, cache);
        
        // 2. Scaled Residual Addition
        // This prevents the ternary MLA from exploding the hidden state variance
        let x = residual + mla_out.mul_scalar(self.residual_scale);

        // --- Mixture of Experts Block ---
        let residual = x.clone();
        
        // 3. Pre-Norm for MoE
        let x_norm = self.norm2.forward(x);
        
        let (moe_out, routing_loss) = self.moe.forward(x_norm);
        
        // 4. Scaled Residual Addition
        // This keeps the hidden state range consistent for the next layer (or the output head)
        let out = residual + moe_out.mul_scalar(self.residual_scale);

        (out, routing_loss, Some(new_cache))
    }
}