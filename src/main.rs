mod config;
mod layer;
mod mla;
mod moe;
mod model;
mod mtp; // Ensure mtp module is registered

use burn::backend::Wgpu;
use burn::tensor::{Int, Tensor};
use config::QuantCBConfig;

fn main() {
    type Backend = Wgpu;
    let device = Default::default();

    println!("QuantCB 2.0 | Backend: Vulkan (WGPU)");

    let config = QuantCBConfig::new(
        10000, 256, 8, 4, 8, 2, 512, 0.1, 64, 64, 16, 16,
    );

    let model = config.init::<Backend>(&device);
    println!("Model with MTP initialized successfully.");

    // Dummy input: Tokens [1..10]
    let dummy_input = Tensor::<Backend, 2, Int>::from_data(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
        &device,
    );

    // Dummy targets: Tokens [2..11] (t+1 tokens for MTP "hint")
    let dummy_targets = Tensor::<Backend, 2, Int>::from_data(
        [[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]],
        &device,
    );

    // Run Standard Forward
    let output = model.forward(dummy_input.clone());
    println!("Standard Output shape: {:?}", output.dims());

    // Run MTP Test
    println!("\n--- Testing MTP System ---");
    let (main_logits, mtp_logits, mtp_loss) = model.forward_mtp(dummy_input, dummy_targets);

    println!("Main Logits shape: {:?}", main_logits.dims());
    println!("MTP Logits shape:  {:?}", mtp_logits.dims());
    println!("MTP Loss:          {:?}", mtp_loss.into_data());
    println!("MTP Verification Complete.");
}