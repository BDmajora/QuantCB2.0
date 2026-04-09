use burn::module::Module;
use burn::nn::{Embedding, RmsNorm, Linear};
use burn::tensor::activation::sigmoid;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use crate::layer::QuantCBLayer;
use crate::mtp::MultiTokenPrediction; 

#[derive(Module, Debug)]
pub struct QuantCB<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub layers: Vec<QuantCBLayer<B>>,
    pub norm_f: RmsNorm<B>,
    pub output: Linear<B>,
    pub mtp: MultiTokenPrediction<B>, 
    pub hallucination_probe: Linear<B>, // The new probe
}

impl<B: Backend> QuantCB<B> {
    /// Standard forward pass
    /// Returns (Logits, Hallucination Probability)
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut x = self.token_embedding.forward(input);

        for layer in &self.layers {
            x = layer.forward(x);
        }

        let h_base = self.norm_f.forward(x);
        let logits = self.output.forward(h_base.clone());

        // Probe: project hidden state to 1, then apply sigmoid for probability
        let probe_logits = self.hallucination_probe.forward(h_base);
        let hallucination_probs = sigmoid(probe_logits);

        (logits, hallucination_probs)
    }

    /// Forward pass with Multi-Token Prediction (Training/Verification)
    /// Returns (Main Logits, MTP Logits, MTP Loss, Hallucination Probability)
    pub fn forward_mtp(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 3>) {
        let mut x = self.token_embedding.forward(input);

        for layer in &self.layers {
            x = layer.forward(x);
        }

        // Hidden state used for main prediction, MTP fusion, and the probe
        let h_base = self.norm_f.forward(x);
        
        // Main branch: predicts t+1
        let main_logits = self.output.forward(h_base.clone());

        // Probe branch
        let probe_logits = self.hallucination_probe.forward(h_base.clone());
        let hallucination_probs = sigmoid(probe_logits);

        // MTP branch: predicts t+2 using h_base and t+1 targets
        let (mtp_logits, mtp_loss) = self.mtp.forward(
            h_base,
            targets,
            &self.token_embedding,
            &self.output,
        );

        (main_logits, mtp_logits, mtp_loss, hallucination_probs)
    }
}