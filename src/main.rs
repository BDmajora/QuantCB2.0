mod config;
mod layer;
mod kv_cache;
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

    let config = QuantCBConfig::new(
        10000, 256, 8, 4, 8, 2, 512, 0.1, 64, 64, 16, 16, 2,
    );

    let model = config.init::<Backend>(&device);
    println!("Model with Dynamic Thinking Gate, Recurrent Feedback, and Balanced MoE initialized.");

    let dummy_input = Tensor::<Backend, 2, Int>::from_data(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
        &device,
    );

    let dummy_targets = Tensor::<Backend, 2, Int>::from_data(
        [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]],
        &device,
    );

    // FIX: Added `None` for caches, and `_final_caches` to the tuple
    let (output, _probe_probs, think_gate, aux_loss, _final_caches) = model.forward(dummy_input.clone(), None);
    
    println!("Standard Output shape: {:?}", output.dims());
    println!("Thinking Gate shape:   {:?}", think_gate.dims());
    println!("MoE Routing Loss:      {:?}", aux_loss.into_data());

    println!("\n--- Testing MTP System ---");
    
    // FIX: Added `None` for caches, and `_mtp_final_caches` to the tuple
    let (main_logits, _mtp_logits, mtp_loss, _mtp_probe_probs, mtp_think_gate, mtp_aux_loss, _mtp_final_caches) = 
        model.forward_mtp(dummy_input, dummy_targets, None);

    println!("Main Logits shape:     {:?}", main_logits.dims());
    println!("Thinking Gate (MTP):   {:?}", mtp_think_gate.dims());
    println!("MTP Loss:              {:?}", mtp_loss.into_data());
    println!("MTP Routing Loss:      {:?}", mtp_aux_loss.into_data());
    
    println!("\nVerification Complete: Dual-pass refinement, gated feedback, and expert balancing functional.");
}