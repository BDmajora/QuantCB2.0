use burn::config::Config;
use burn::optim::AdamConfig;
use crate::config::QuantCBConfig;

pub const TAG_TRUTH: &str = "<|truth|>";
pub const TAG_HALLUCINATE: &str = "<|hallucinate|>";
pub const TAG_SHAKESPEARE: &str = "<|shakespeare|>";

#[derive(Config)]
pub struct TrainingConfig {
    pub model: QuantCBConfig,
    pub optimizer: AdamConfig,

    // --- MIXTURE OF EXPERTS ---
    #[config(default = 8)]
    pub num_experts: usize,
    
    #[config(default = 0.15)]
    pub corruption_rate: f32, 

    // --- LOOPLM LATENT REASONING ---
    #[config(default = 2)]
    pub loop_depth: usize, 
    
    #[config(default = 0.05)]
    pub entropy_reg_weight: f32, 

    // --- STABLE CORE HYPERPARAMETERS ---
    #[config(default = 3e-4)]
    pub learning_rate: f64,
    
    // NEW: Matches Python's GRAD_CLIP for MoE stability
    #[config(default = 1.0)]
    pub clip_grad_norm: f64, 
    
    #[config(default = 16)] 
    pub batch_size: usize,

    #[config(default = 256)]
    pub seq_len: usize,
    
    #[config(default = 20000)]
    pub max_iterations: usize,
    
    #[config(default = 42)]
    pub seed: u64,
}