use burn::module::Module;
use burn::nn::{Embedding, RmsNorm, Linear};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::layer::QuantCBLayer;
use crate::mtp::MultiTokenPrediction; 
use crate::kv_cache::KVCache;

#[derive(Module, Debug)]
pub struct QuantCB<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub layers: Vec<QuantCBLayer<B>>,
    pub norm_f: RmsNorm<B>,
    pub output: Linear<B>,
    pub mtp: MultiTokenPrediction<B>, 
    pub hallucination_probe: Linear<B>,
    pub thinking_gate: Linear<B>,
    pub num_recurrent_steps: usize,
}

impl<B: Backend> QuantCB<B> {
    fn process_layers(
        &self, 
        mut x: Tensor<B, 3>, 
        mut layer_caches: Vec<Option<KVCache<B>>>
    ) -> (Tensor<B, 3>, Tensor<B, 1>, Vec<Option<KVCache<B>>>) {
        let mut total_routing_loss = Tensor::<B, 1>::zeros([1], &x.device());
        let mut updated_caches = Vec::with_capacity(self.layers.len());
        
        for layer in &self.layers {
            // Extract the cache for the current layer if it exists
            let current_cache = if layer_caches.is_empty() {
                None
            } else {
                layer_caches.remove(0)
            };
            
            let (out, layer_loss, new_cache) = layer.forward(x, current_cache);
            x = out;
            total_routing_loss = total_routing_loss + layer_loss;
            // FIX: new_cache is already an Option, so we just push it directly
            updated_caches.push(new_cache);
        }
        (x, total_routing_loss, updated_caches)
    }

    pub fn forward(
        &self, 
        input: Tensor<B, 2, Int>,
        caches: Option<Vec<Option<KVCache<B>>>>
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>, Vec<Option<KVCache<B>>>) {
        let x_emb = self.token_embedding.forward(input);
        
        let [batch_size, seq_len, _] = x_emb.dims();
        let device = x_emb.device();
        
        let mut current_input = x_emb.clone();
        let mut gate = Tensor::<B, 3>::zeros([batch_size, seq_len, 1], &device);
        let mut total_aux_loss = Tensor::<B, 1>::zeros([1], &device);

        let active_caches = caches.unwrap_or_else(|| KVCache::init_empty_list(self.layers.len()));

        // Recurrent Thinking Steps: We pass empty caches here so intermediate 
        // states don't pollute the KV memory.
        for _ in 0..self.num_recurrent_steps {
            let (h_out, routing_loss, _) = self.process_layers(
                current_input, 
                KVCache::init_empty_list(self.layers.len())
            );
            total_aux_loss = total_aux_loss + routing_loss;
            
            gate = sigmoid(self.thinking_gate.forward(h_out.clone()));
            current_input = x_emb.clone() + (h_out * gate.clone());
        }

        // Final Pass: Apply and update the actual KV caches
        let (x_final, routing_loss_final, final_caches) = self.process_layers(current_input, active_caches);
        total_aux_loss = total_aux_loss + routing_loss_final;

        let h_base = self.norm_f.forward(x_final);
        let logits = self.output.forward(h_base.clone());

        let probe_logits = self.hallucination_probe.forward(h_base);
        let hallucination_probs = sigmoid(probe_logits);

        (logits, hallucination_probs, gate, total_aux_loss, final_caches)
    }

    pub fn forward_mtp(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        caches: Option<Vec<Option<KVCache<B>>>>
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>, Vec<Option<KVCache<B>>>) {
        let x_emb = self.token_embedding.forward(input);

        let [batch_size, seq_len, _] = x_emb.dims();
        let device = x_emb.device();

        let mut current_input = x_emb.clone();
        let mut gate = Tensor::<B, 3>::zeros([batch_size, seq_len, 1], &device);
        let mut total_aux_loss = Tensor::<B, 1>::zeros([1], &device);

        let active_caches = caches.unwrap_or_else(|| KVCache::init_empty_list(self.layers.len()));

        for _ in 0..self.num_recurrent_steps {
            let (h_out, routing_loss, _) = self.process_layers(
                current_input, 
                KVCache::init_empty_list(self.layers.len())
            );
            total_aux_loss = total_aux_loss + routing_loss;
            
            gate = sigmoid(self.thinking_gate.forward(h_out.clone()));
            current_input = x_emb.clone() + (h_out * gate.clone());
        }

        let (x_final, routing_loss_final, final_caches) = self.process_layers(current_input, active_caches);
        total_aux_loss = total_aux_loss + routing_loss_final;

        let h_base = self.norm_f.forward(x_final);
        let main_logits = self.output.forward(h_base.clone());

        let probe_logits = self.hallucination_probe.forward(h_base.clone());
        let hallucination_probs = sigmoid(probe_logits);

        let (mtp_logits, mtp_loss) = self.mtp.forward(
            h_base,
            targets,
            &self.token_embedding,
            &self.output,
        );

        (main_logits, mtp_logits, mtp_loss, hallucination_probs, gate, total_aux_loss, final_caches)
    }
}