use burn::module::Module;
// Swapping out standard Linear for our custom BitNet 1.58b implementation
use crate::model::bitlinear::{BitLinear, BitLinearConfig}; 
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct Expert<B: Backend> {
    ff1: BitLinear<B>,
    ff2: BitLinear<B>,
}

impl<B: Backend> Expert<B> {
    pub fn init(d_model: usize, device: &B::Device) -> Self {
        Self {
            // Using BitLinearConfig for ternary weight {-1, 0, 1} support
            ff1: BitLinearConfig::new(d_model, d_model * 4).init(device),
            ff2: BitLinearConfig::new(d_model * 4, d_model).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.ff1.forward(x);
        let x = gelu(x);
        self.ff2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct MoELayer<B: Backend> {
    experts: Vec<Expert<B>>,
    // In a full BitNet transition, even the router benefits from ternary quantization
    router: BitLinear<B>, 
    top_k: usize,
}

impl<B: Backend> MoELayer<B> {
    pub fn init(d_model: usize, n_experts: usize, top_k: usize, device: &B::Device) -> Self {
        let experts = (0..n_experts)
            .map(|_| Expert::init(d_model, device))
            .collect();
        let router = BitLinearConfig::new(d_model, n_experts).init(device);

        Self {
            experts,
            router,
            top_k,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [batch_size, seq_len, d_model] = x.dims();
        let device = x.device();
        
        // Pass through BitLinear Router
        let router_logits = self.router.forward(x.clone());
        let routing_probs = softmax(router_logits, 2); 
        
        let (top_k_weights, top_k_indices) = routing_probs.clone().topk_with_indices(self.top_k, 2);
        
        // Normalize weights over selected experts to maintain variance
        let top_k_weights = top_k_weights.clone() / (top_k_weights.sum_dim(2) + 1e-10);

        let mut final_output = Tensor::<B, 3>::zeros([batch_size, seq_len, d_model], &device);
        let mut routing_loss = Tensor::<B, 1>::zeros([1], &device);
        
        let b_s_float = (batch_size * seq_len) as f32;

        for i in 0..self.experts.len() {
            // Mask for tokens routed to expert i
            let expert_mask = top_k_indices.clone().equal_elem(i as i32).float();
            
            // Scaled contribution of this expert for the specific tokens
            let combined_weight = (expert_mask.clone() * top_k_weights.clone()).sum_dim(2);
            
            let expert_output = self.experts[i].forward(x.clone());
            final_output = final_output + (expert_output * combined_weight);
            
            // --- Load Balancing Loss (Essential for BitNet MoE Stability) ---
            let fraction_routed = expert_mask.sum() / b_s_float;
            
            let prob_i = routing_probs.clone()
                .slice([0..batch_size, 0..seq_len, i..(i+1)])
                .mean();
                
            routing_loss = routing_loss + (fraction_routed * prob_i);
        }

        // Apply standard balancing alpha (0.01)
        let alpha = 0.01;
        routing_loss = routing_loss * (self.experts.len() as f32 * alpha);

        (final_output, routing_loss)
    }
}