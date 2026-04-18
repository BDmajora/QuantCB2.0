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
    pub activation_bits: usize, 
    #[config(default = 1e-5)]
    pub quant_eps: f32,
    #[config(default = 8.0)]
    pub min_temp: f32,
    #[config(default = 15.0)]
    pub max_temp: f32,

    // --- BLT: BYTE LATENT TRANSFORMER ---
    // The vocab is strictly the 256 byte values + special tags
    #[config(default = 262)] 
    pub byte_vocab_size: usize,
    
    // The hidden dimension of the small, fast local encoder
    #[config(default = 128)]
    pub local_dim: usize,
    
    // The threshold at which uncertainty (entropy) forces a new patch
    #[config(default = 0.65)]
    pub entropy_threshold: f32,
    
    // Hard limits on patch sizes to prevent OOM
    #[config(default = 16)]
    pub max_patch_bytes: usize,

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

    // --- KIMI TEAM ATTNRES ---
    #[config(default = true)]
    pub use_attn_res: bool,
    #[config(default = 4)]
    pub attn_res_heads: usize, 

    // --- MULTI-TOKEN PREDICTION ---
    #[config(default = 0.1)]
    pub mtp_loss_weight: f32,

    // --- STABLE CORE HYPERPARAMETERS ---
    #[config(default = 8e-4)]
    pub learning_rate: f64,
    #[config(default = 1.0)]
    pub clip_grad_norm: f64, 
    #[config(default = 8)] // Back to 8 assuming VRAM holds
    pub batch_size: usize,
    #[config(default = 512)] // This now refers to 512 PATCHES, not bytes. Huge context upgrade.
    pub seq_len: usize,
    #[config(default = 20000)]
    pub max_iterations: usize,
    #[config(default = 42)]
    pub seed: u64,
}