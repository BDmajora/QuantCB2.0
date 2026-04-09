use burn::module::Module;
use burn::nn::{Embedding, LayerNorm, Linear};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::layer::QuantCBLayer;

#[derive(Module, Debug)]
pub struct QuantCB<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub layers: Vec<QuantCBLayer<B>>, // Replaced standard encoder
    pub norm_f: LayerNorm<B>,
    pub output: Linear<B>,
}

impl<B: Backend> QuantCB<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.token_embedding.forward(input);

        // Pass through all MLA + MoE layers
        for layer in &self.layers {
            x = layer.forward(x);
        }

        // Final normalization and projection
        let x = self.norm_f.forward(x);
        self.output.forward(x)
    }
}