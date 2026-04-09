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

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_model] = x.dims();
        let device = x.device();
        
        let router_logits = self.router.forward(x.clone());
        let weights = softmax(router_logits, 2); 
        
        let (top_k_weights, top_k_indices) = weights.topk_with_indices(self.top_k, 2);
        let top_k_weights = top_k_weights.clone() / top_k_weights.sum_dim(2);

        let mut final_output = Tensor::<B, 3>::zeros([batch_size, seq_len, d_model], &device);

        for i in 0..self.experts.len() {
            let expert_mask = top_k_indices.clone().equal_elem(i as i32).float();
            let combined_weight = (expert_mask * top_k_weights.clone()).sum_dim(2);
            let expert_output = self.experts[i].forward(x.clone());
            final_output = final_output + (expert_output * combined_weight);
        }

        final_output
    }
}