use burn::module::Module;
// Swapping standard Linear for BitLinear
use crate::model::bitlinear::{BitLinear, BitLinearConfig}; 
use burn::nn::{Embedding, RmsNorm, RmsNormConfig};
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::model::layer::QuantCBLayer;
use crate::model::mtp::MultiTokenPrediction; 
use crate::model::kv_cache::KVCache;
use crate::model::config::QuantCBConfig;

#[derive(Module, Debug)]
pub struct AttnRes<B: Backend> {
    pub q_proj: BitLinear<B>,
    pub k_proj: BitLinear<B>,
    pub v_proj: BitLinear<B>,
    pub norm: RmsNorm<B>, // Swapped to RmsNorm for BitNet consistency
    pub d_model: usize,
}

impl<B: Backend> AttnRes<B> {
    pub fn new(config: &QuantCBConfig, device: &B::Device) -> Self {
        Self {
            q_proj: BitLinearConfig::new(config.d_model, config.d_model).init(device),
            k_proj: BitLinearConfig::new(config.d_model, config.d_model).init(device),
            v_proj: BitLinearConfig::new(config.d_model, config.d_model).init(device),
            norm: RmsNormConfig::new(config.d_model).init(device),
            d_model: config.d_model,
        }
    }

    pub fn forward(&self, current_update: Tensor<B, 3>, history: &[Tensor<B, 3>]) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_model] = current_update.dims();
        let depth = history.len();

        if depth == 0 {
            return current_update;
        }

        // Maintain the 4D optimization for WGPU efficiency
        let query = current_update.clone().reshape([batch_size, seq_len, 1, d_model]);
        
        let mut history_tensors = Vec::with_capacity(depth);
        for h in history {
            history_tensors.push(h.clone().reshape([batch_size, seq_len, 1, d_model]));
        }
        
        let memory = Tensor::cat(history_tensors, 2);

        let q = self.q_proj.forward(query);
        let k = self.k_proj.forward(memory.clone());
        let v = self.v_proj.forward(memory);

        let scale = (self.d_model as f64).sqrt();
        let k_t = k.swap_dims(2, 3); 
        
        let scores = q.matmul(k_t).div_scalar(scale);
        let weights = softmax(scores, 3); 

        let context = weights.matmul(v);
        let context = context.reshape([batch_size, seq_len, d_model]);

        self.norm.forward(current_update + context)
    }
}

#[derive(Module, Debug)]
pub struct QuantCB<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub layers: Vec<QuantCBLayer<B>>,
    pub loop_block: QuantCBLayer<B>, 
    pub attn_res: AttnRes<B>,
    pub norm_f: RmsNorm<B>,
    pub output: BitLinear<B>, 
    pub mtp: MultiTokenPrediction<B>, 
    pub hallucination_probe: BitLinear<B>,
    pub thinking_gate: BitLinear<B>, 
    pub loop_depth: usize, 
}

impl<B: Backend> QuantCB<B> {
    fn process_layers(
        &self, 
        mut x: Tensor<B, 3>, 
        mut layer_caches: Vec<Option<KVCache<B>>>
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Vec<Option<KVCache<B>>>) {
        let device = x.device();
        let mut total_routing_loss = Tensor::<B, 1>::zeros([1], &device);
        let mut updated_caches = Vec::with_capacity(self.layers.len());
        
        for layer in &self.layers {
            let current_cache = if layer_caches.is_empty() { None } else { layer_caches.remove(0) };
            let (out, layer_loss, new_cache) = layer.forward(x, current_cache);
            x = out;
            total_routing_loss = total_routing_loss + layer_loss;
            updated_caches.push(new_cache);
        }
        (x, total_routing_loss, updated_caches)
    }

    // FIX: Added current_loop_depth to signature
    pub fn forward(
        &self, 
        input: Tensor<B, 2, Int>,
        caches: Option<Vec<Option<KVCache<B>>>>,
        current_loop_depth: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>, Vec<Option<KVCache<B>>>, Tensor<B, 3>) {
        let x_emb = self.token_embedding.forward(input);
        let [batch_size, seq_len, _] = x_emb.dims();
        let device = x_emb.device();
        
        let mut current_input = x_emb.clone();
        let mut gate = Tensor::<B, 3>::zeros([batch_size, seq_len, 1], &device);
        let mut total_aux_loss = Tensor::<B, 1>::zeros([1], &device);
        
        // Use current_loop_depth for history allocation
        let mut history: Vec<Tensor<B, 3>> = Vec::with_capacity(current_loop_depth);

        let active_caches = caches.unwrap_or_else(|| KVCache::init_empty_list(self.layers.len()));

        // --- Ternary Iterative Latent Reasoning ---
        // FIX: Loop now respects the dynamic depth passed from the scheduler
        for _ in 0..current_loop_depth {
            let (h_out, routing_loss, _) = self.loop_block.forward(current_input.clone(), None);
            total_aux_loss = total_aux_loss + routing_loss;
            
            let h_refined = self.attn_res.forward(h_out, &history);
            history.push(h_refined.clone());

            gate = sigmoid(self.thinking_gate.forward(h_refined.clone()));
            
            let p_safe = gate.clone().clamp(1e-7, 1.0 - 1e-7);
            let one_minus_p_safe = p_safe.clone().neg().add_scalar(1.0);
            let gate_entropy = (p_safe.clone().mul(p_safe.log()).neg() + one_minus_p_safe.clone().mul(one_minus_p_safe.log()).neg()).mean(); 
            total_aux_loss = total_aux_loss + gate_entropy;

            current_input = x_emb.clone() + (h_refined * gate.clone());
        }

        let (x_final, routing_loss_final, final_caches) = self.process_layers(current_input, active_caches);
        total_aux_loss = total_aux_loss + routing_loss_final;

        let h_base = self.norm_f.forward(x_final);
        
        let logits = self.output.forward(h_base.clone());
        let probe_logits = self.hallucination_probe.forward(h_base.clone());
        let hallucination_probs = sigmoid(probe_logits);

        (logits, hallucination_probs, gate, total_aux_loss, final_caches, h_base)
    }

    // FIX: Added current_loop_depth to signature to match Trainer call
    pub fn forward_mtp(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        caches: Option<Vec<Option<KVCache<B>>>>,
        current_loop_depth: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>, Vec<Option<KVCache<B>>>) {
        // Pass dynamic depth through to base forward
        let (main_logits, hallucination_probs, gate, total_aux_loss, final_caches, h_base) = 
            self.forward(input, caches, current_loop_depth);

        let (mtp_logits, mtp_loss) = self.mtp.forward(
            h_base,
            targets,
            &self.token_embedding,
            &self.output, 
        );

        (main_logits, mtp_logits, mtp_loss, hallucination_probs, gate, total_aux_loss, final_caches)
    }
}