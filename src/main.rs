// src/main.rs
mod config;
mod layer;
mod kv_cache;
mod mla;
mod moe;
mod model;
mod mtp;
mod trainer_config;
mod trainer;
mod training_data;
mod dataset;
mod batcher;
mod generator;
mod learning;

// Tokenizer Sub-modules
mod bpe_types;
mod bpe_trainer;
mod bpe_encoder;
mod tokenizer;

fn main() {
    println!("QuantCB 2.0 | Backend: Vulkan (WGPU)");
    trainer::run();
}