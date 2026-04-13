use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
pub struct Expert<B: Backend> {
    ff1: Linear<B>,
    ff2: Linear<B>,
}

impl<B: Backend> Expert<B> {
    pub fn init(d_model: usize, device: &B::Device) -> Self {
        Self {
            ff1: LinearConfig::new(d_model, d_model * 4).init(device),
            ff2: LinearConfig::new(d_model * 4, d_model).init(device),
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
    router: Linear<B>,
    top_k: usize,
}

impl<B: Backend> MoELayer<B> {
    pub fn init(d_model: usize, n_experts: usize, top_k: usize, device: &B::Device) -> Self {
        let experts = (0..n_experts)
            .map(|_| Expert::init(d_model, device))
            .collect();
        let router = LinearConfig::new(d_model, n_experts).init(device);

        Self {
            experts,
            router,
            top_k,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [batch_size, seq_len, d_model] = x.dims();
        let device = x.device();
        
        let router_logits = self.router.forward(x.clone());
        let routing_probs = softmax(router_logits, 2); 
        
        let (top_k_weights, top_k_indices) = routing_probs.clone().topk_with_indices(self.top_k, 2);
        let top_k_weights = top_k_weights.clone() / top_k_weights.sum_dim(2);

        let mut final_output = Tensor::<B, 3>::zeros([batch_size, seq_len, d_model], &device);
        let mut routing_loss = Tensor::<B, 1>::zeros([1], &device);
        
        let b_s_float = (batch_size * seq_len) as f32;

        for i in 0..self.experts.len() {
            // Standard Expert Application
            let expert_mask = top_k_indices.clone().equal_elem(i as i32).float();
            let combined_weight = (expert_mask.clone() * top_k_weights.clone()).sum_dim(2);
            let expert_output = self.experts[i].forward(x.clone());
            final_output = final_output + (expert_output * combined_weight);
            
            // --- Load Balancing Loss Calculation ---
            // 1. Calculate f_i (Fraction of tokens routed to expert i)
            let fraction_routed = expert_mask.sum() / b_s_float;
            
            // 2. Calculate P_i (Mean router probability for expert i)
            let prob_i = routing_probs.clone()
                .slice([0..batch_size, 0..seq_len, i..(i+1)])
                .mean();
                
            routing_loss = routing_loss + (fraction_routed * prob_i);
        }

        // Multiply by E (Number of Experts) and alpha (0.01)
        let alpha = 0.01;
        routing_loss = routing_loss * (self.experts.len() as f32 * alpha);

        (final_output, routing_loss)
    }
}