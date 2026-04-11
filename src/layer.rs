use burn::module::Module;
use burn::nn::{RmsNorm, RmsNormConfig};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::config::QuantCBConfig;
use crate::kv_cache::KVCache;
use crate::mla::{MLAConfig, MultiHeadLatentAttention};
use crate::moe::MoELayer;

#[derive(Module, Debug)]
pub struct QuantCBLayer<B: Backend> {
    mla: MultiHeadLatentAttention<B>,
    moe: MoELayer<B>,
    norm1: RmsNorm<B>,
    norm2: RmsNorm<B>,
}

impl<B: Backend> QuantCBLayer<B> {
    pub fn init(config: &QuantCBConfig, device: &B::Device) -> Self {
        let mla_config = MLAConfig::new(
            config.d_model, config.n_heads, config.d_c, 
            config.d_c_q, config.d_head_c, config.d_rope,
        );

        Self {
            mla: mla_config.init(device),
            moe: MoELayer::init(config.d_model, config.n_experts, config.top_k, device),
            norm1: RmsNormConfig::new(config.d_model).init(device),
            norm2: RmsNormConfig::new(config.d_model).init(device),
        }
    }

    pub fn forward(
        &self, 
        x: Tensor<B, 3>, 
        cache: Option<KVCache<B>>
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Option<KVCache<B>>) {
        let residual = x.clone();
        let x = self.norm1.forward(x);
        
        // Unpack the tuple from MLA and add the attention output to the residual
        let (mla_out, new_cache) = self.mla.forward(x, cache);
        let x = residual + mla_out;

        let residual = x.clone();
        let x = self.norm2.forward(x);
        
        let (moe_out, routing_loss) = self.moe.forward(x);
        
        (residual + moe_out, routing_loss, Some(new_cache))
    }
}