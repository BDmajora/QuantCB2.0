mod config;
mod layer;
mod mla;
mod moe;
mod model;

use burn::backend::Wgpu;
use burn::tensor::{Int, Tensor};
use config::QuantCBConfig;

fn main() {
    // Explicitly using the Wgpu backend for Vulkan support
    type Backend = Wgpu;
    let device = Default::default();

    println!("QuantCB 2.0 | Backend: Vulkan (WGPU)");

    // Initialize the configuration with new MLA specific dimensions
    let config = QuantCBConfig::new(
        10000, // vocab_size
        256,   // d_model
        8,     // n_heads
        4,     // n_layers
        8,     // n_experts
        2,     // top_k
        512,   // max_seq_len
        0.1,   // dropout
        64,    // d_c (KV Latent compression dim)
        64,    // d_c_q (Q Latent compression dim)
        16,    // d_head_c (NOPE dim per head)
        16,    // d_rope (RoPE dim per head)
    );

    // Initialize the model on the GPU
    let model = config.init::<Backend>(&device);
    println!("MoE Top-K + MLA Model initialized on GPU successfully.");

    // Dummy input for verification
    let dummy_input = Tensor::<Backend, 2, Int>::from_data(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]],
        &device,
    );

    // Run the forward pass
    let output = model.forward(dummy_input);
    
    println!("Output tensor shape: {:?}", output.dims());
}