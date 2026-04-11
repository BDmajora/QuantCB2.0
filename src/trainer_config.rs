// src/trainer_config.rs
use burn::config::Config;
use burn::optim::AdamConfig;
use crate::config::QuantCBConfig;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: QuantCBConfig,
    pub optimizer: AdamConfig,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    #[config(default = 512)]
    pub seq_len: usize,
    #[config(default = 16)] 
    pub batch_size: usize,
    #[config(default = 1000)]
    pub max_iterations: usize,
    #[config(default = 42)]
    pub seed: u64,
}