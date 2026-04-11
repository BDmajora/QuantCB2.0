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
mod tokenizer;
mod dataset;
mod batcher;
mod generator;

fn main() {
    println!("QuantCB 2.0 | Backend: Vulkan (WGPU)");
    trainer::run();
}