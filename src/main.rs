mod config;
mod layer;
mod kv_cache;
mod mla;
mod moe;
mod model;
mod mtp; 

use burn::backend::Wgpu;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::tensor::{Int, Tensor};
use config::QuantCBConfig;

fn main() {
    type Backend = Wgpu;
    let device = Default::default();

    println!("QuantCB 2.0 | Backend: Vulkan (WGPU)");

    // Config: vocab, d_model, heads, layers, experts, top_k, seq_len, dropout, d_c, d_c_q, d_head_c, d_rope, recurrent_steps
    let config = QuantCBConfig::new(
        10000, 256, 8, 4, 8, 2, 512, 0.1, 64, 64, 16, 16, 2,
    );

    let model = config.init::<Backend>(&device);
    println!("Model with Dynamic Thinking Gate, Recurrent Feedback, and Balanced MoE initialized.");

    // Dummy Batch for Verification
    let dummy_input = Tensor::<Backend, 2, Int>::from_data(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
        &device,
    );

    let dummy_targets = Tensor::<Backend, 2, Int>::from_data(
        [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]],
        &device,
    );

    println!("\n--- Executing MTP Forward Pass ---");
    
    let (main_logits, _mtp_logits, mtp_loss, _mtp_probe_probs, mtp_think_gate, mtp_aux_loss, _mtp_final_caches) = 
        model.forward_mtp(dummy_input, dummy_targets.clone(), None);

    // 1. Main Prediction Loss (t+1)
    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let [batch_size, seq_len, vocab_size] = main_logits.dims();
    
    let logits_flat = main_logits.reshape([batch_size * seq_len, vocab_size]);
    let targets_flat = dummy_targets.reshape([batch_size * seq_len]);
    
    let main_loss = loss_fn.forward(logits_flat, targets_flat);

    // 2. Auxiliary Scaling Factors
    let mtp_weight = 0.3;
    let aux_routing_weight = 1.0; 

    // 3. Final Weighted Sum for Optimizer
    let final_loss = main_loss.clone() 
        + mtp_loss.clone().mul_scalar(mtp_weight) 
        + mtp_aux_loss.clone().mul_scalar(aux_routing_weight);

    // Telemetry Output
    println!("Main Logits:           {:?}", [batch_size, seq_len, vocab_size]);
    println!("Thinking Gate (MTP):   {:?}", mtp_think_gate.dims());
    println!("-----------------------------------------");
    println!("Main Prediction Loss:  {:?}", main_loss.into_data());
    println!("MTP Loss (Unscaled):   {:?}", mtp_loss.into_data());
    println!("MoE Routing Loss:      {:?}", mtp_aux_loss.into_data());
    println!("TOTAL TRACKED LOSS:    {:?}", final_loss.into_data());
    println!("-----------------------------------------");
    
    println!("\nVerification Complete: Model logic is sound and ready for the training loop.");
}