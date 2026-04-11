use burn::config::Config;
use burn::optim::AdamConfig;
use crate::config::QuantCBConfig;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: QuantCBConfig,
    pub optimizer: AdamConfig,
    
    // --- LOOPLM LATENT REASONING HYPERPARAMETERS ---
    #[config(default = 3)]
    pub loop_depth: usize, // Controls the depth of latent space iterations
    
    #[config(default = 0.01)]
    pub entropy_reg_weight: f32, // Regularization for learned depth allocation

    // --- CORE HYPERPARAMETERS ---
    #[config(default = 3e-4)]
    pub learning_rate: f64,
    
    #[config(default = 1.0)]
    pub grad_clip: f32,
    
    #[config(default = 50)]
    pub warmup_steps: usize,
    
    #[config(default = 0.1)]
    pub weight_decay: f32,

    #[config(default = 256)]
    pub seq_len: usize,
    
    #[config(default = 256)]
    pub batch_size: usize,
    
    #[config(default = 6000)]
    pub max_iterations: usize,
    
    #[config(default = 42)]
    pub seed: u64,
}