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
    pub hallucination_probe: Linear<B>,
    pub thinking_gate: Linear<B>, // The new Gate
}

impl<B: Backend> QuantCB<B> {
    /// Helper to pass a tensor through the transformer layer stack
    fn process_layers(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }

    /// Standard forward pass with Gated Recurrent Feedback
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let x_emb = self.token_embedding.forward(input);

        // --- Pass 1: Initial "Thought" ---
        let h_initial = self.process_layers(x_emb.clone());

        // --- Gated Feedback Mechanism ---
        // Calculate gate value (0.0 to 1.0) based on the initial pass
        let gate = sigmoid(self.thinking_gate.forward(h_initial.clone()));
        
        // Recurrent Feedback: Mix original embedding with processed state
        let x_refined = x_emb + (h_initial * gate.clone());

        // --- Pass 2: Refined Processing ---
        let x_final = self.process_layers(x_refined);

        let h_base = self.norm_f.forward(x_final);
        let logits = self.output.forward(h_base.clone());

        let probe_logits = self.hallucination_probe.forward(h_base);
        let hallucination_probs = sigmoid(probe_logits);

        (logits, hallucination_probs, gate)
    }

    /// Forward pass with Multi-Token Prediction and Thinking Gate
    pub fn forward_mtp(
        &self,
        input: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>, Tensor<B, 3>, Tensor<B, 3>) {
        let x_emb = self.token_embedding.forward(input);

        // Recurrent logic repeated for MTP branch consistency
        let h_initial = self.process_layers(x_emb.clone());
        let gate = sigmoid(self.thinking_gate.forward(h_initial.clone()));
        let x_refined = x_emb + (h_initial * gate.clone());
        let x_final = self.process_layers(x_refined);

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

        (main_logits, mtp_logits, mtp_loss, hallucination_probs, gate)
    }
}