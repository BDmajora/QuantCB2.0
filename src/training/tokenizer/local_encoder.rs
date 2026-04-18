use burn::nn::{Linear, LinearConfig, Embedding, EmbeddingConfig};
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Int};

#[derive(Module, Debug)]
pub struct LocalEncoder<B: Backend> {
    pub embedding: Embedding<B>,
    pub output_layer: Linear<B>,
}

impl<B: Backend> LocalEncoder<B> {
    pub fn new(device: &B::Device) -> Self {
        let embedding = EmbeddingConfig::new(262, 128).init(device);
        let output_layer = LinearConfig::new(128, 256).init(device);

        Self {
            embedding,
            output_layer,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // This makes 'embedding' and 'output_layer' read
        let x = self.embedding.forward(input);
        self.output_layer.forward(x)
    }
}