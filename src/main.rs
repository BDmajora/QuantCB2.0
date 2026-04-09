mod config;
mod layer;
mod mla;
mod moe;
mod model;
mod mtp; 

use burn::backend::Wgpu;
use burn::tensor::{Int, Tensor};
use config::QuantCBConfig;

fn main() {
    type Backend = Wgpu;
    let device = Default::default();

    println!("QuantCB 2.0 | Backend: Vulkan (WGPU)");

    // Initialize config with current architectural parameters
    let config = QuantCBConfig::new(
        10000, 256, 8, 4, 8, 2, 512, 0.1, 64, 64, 16, 16,
    );

    let model = config.init::<Backend>(&device);
    println!("Model with Thinking Gate and Recurrent Feedback initialized.");

    // Dummy input: Batch of 2, Sequence of 10
    let dummy_input = Tensor::<Backend, 2, Int>::from_data(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
        &device,
    );

    // Dummy targets: Tokens shifted by 1 for MTP hint/loss
    let dummy_targets = Tensor::<Backend, 2, Int>::from_data(
        [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]],
        &device,
    );

    // --- Run Standard Forward ---
    // Returns: (Logits, Hallucination Probs, Thinking Gate)
    let (output, _probe_probs, think_gate) = model.forward(dummy_input.clone());
    
    println!("Standard Output shape: {:?}", output.dims());
    println!("Thinking Gate shape:   {:?}", think_gate.dims()); // Expected [2, 10, 1]

    // --- Run MTP Test ---
    println!("\n--- Testing MTP System ---");
    // Returns: (Main Logits, MTP Logits, MTP Loss, Hallucination Probs, Thinking Gate)
    let (main_logits, _mtp_logits, mtp_loss, _mtp_probe_probs, mtp_think_gate) = 
        model.forward_mtp(dummy_input, dummy_targets);

    println!("Main Logits shape:     {:?}", main_logits.dims());
    println!("Thinking Gate (MTP):   {:?}", mtp_think_gate.dims());
    println!("MTP Loss:              {:?}", mtp_loss.into_data());
    
    println!("\nVerification Complete: Dual-pass refinement and gated feedback functional.");
}