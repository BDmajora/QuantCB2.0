use burn::module::Module;
use burn::nn::{Embedding, LayerNorm, Linear};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::layer::QuantCBLayer;
use crate::mtp::MultiTokenPrediction; // New import

#[derive(Module, Debug)]
pub struct QuantCB<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub layers: Vec<QuantCBLayer<B>>,
    pub norm_f: LayerNorm<B>,
    pub output: Linear<B>,
    pub mtp: MultiTokenPrediction<B>, // Now owns the MTP block
}

impl<B: Backend> QuantCB<B> {
    /// Standard forward pass
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut x = self.token_embedding.forward(input);

        for layer in &self.layers {
            x = layer.forward(x);
        }

        let x = self.norm_f.forward(x);
        self.output.forward(x)
    }

    /// Forward pass with Multi-Token Prediction (Training/Verification)
    /// Returns (Main Logits, MTP Logits, MTP Loss)
    pub fn forward_mtp(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>) {
        let mut x = self.token_embedding.forward(input);

        for layer in &self.layers {
            x = layer.forward(x);
        }

        // Hidden state used for both main prediction and MTP fusion
        let h_base = self.norm_f.forward(x);
        
        // Main branch: predicts t+1
        let main_logits = self.output.forward(h_base.clone());

        // MTP branch: predicts t+2 using h_base and t+1 targets
        let (mtp_logits, mtp_loss) = self.mtp.forward(
            h_base,
            targets,
            &self.token_embedding,
            &self.output,
        );

        (main_logits, mtp_logits, mtp_loss)
    }
}