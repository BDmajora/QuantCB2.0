use burn::config::Config;
use burn::optim::AdamConfig;
use crate::model::config::QuantCBConfig;

pub const TAG_TRUTH: &str = "<|truth|>";
pub const TAG_HALLUCINATE: &str = "<|hallucinate|>";
pub const TAG_SHAKESPEARE: &str = "<|shakespeare|>";
pub const TAG_WIKI: &str = "<|wiki|>";

#[derive(Config)]
pub struct TrainingConfig {
    pub model: QuantCBConfig,
    pub optimizer: AdamConfig,

    // --- BITNET 1.58-BIT CORE ---
    #[config(default = 8)]
    pub activation_bits: usize, // BitNet b1.58 usually uses 8-bit activations
    #[config(default = 1e-5)]
    pub quant_eps: f32,

    // --- TOKENIZER ---
    #[config(default = 8192)]
    pub tokenizer_vocab_size: usize,

    // --- MIXTURE OF EXPERTS ---
    #[config(default = 8)]
    pub num_experts: usize,
    #[config(default = 0.15)]
    pub corruption_rate: f32, 

    // --- LOOPLM LATENT REASONING ---
    #[config(default = 3)]
    pub loop_depth: usize, 
    #[config(default = 0.05)]
    pub entropy_reg_weight: f32, 

    // --- KIMI TEAM ATTNRES ---
    #[config(default = true)]
    pub use_attn_res: bool,
    #[config(default = 4)]
    pub attn_res_heads: usize, 

    // --- MULTI-TOKEN PREDICTION ---
    #[config(default = 0.1)]
    pub mtp_loss_weight: f32,

    // --- STABLE CORE HYPERPARAMETERS ---
    #[config(default = 3e-4)]
    pub learning_rate: f64,
    #[config(default = 1.0)]
    pub clip_grad_norm: f64, 
    #[config(default = 8)] 
    pub batch_size: usize,
    #[config(default = 128)]
    pub seq_len: usize,
    #[config(default = 20000)]
    pub max_iterations: usize,
    #[config(default = 42)]
    pub seed: u64,
}