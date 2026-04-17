use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
// Swapping standard Linear for our BitNet 1.58b BitLinear
use crate::model::bitlinear::{BitLinear, BitLinearConfig}; 
use burn::nn::{Embedding, RmsNorm, RmsNormConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[derive(Config, Debug)]
pub struct MTPConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub vocab_size: usize,
}

#[derive(Module, Debug)]
pub struct MultiTokenPrediction<B: Backend> {
    // Ternary Fusion Projections
    pub proj_h: BitLinear<B>,
    pub proj_emb: BitLinear<B>,
    pub ln_fusion: RmsNorm<B>,

    // Ternary Transformer Block
    pub norm1: RmsNorm<B>,
    pub attn_qkv: BitLinear<B>,
    pub attn_out: BitLinear<B>,
    pub norm2: RmsNorm<B>,
    pub mlp_fc1: BitLinear<B>,
    pub mlp_fc2: BitLinear<B>,

    pub n_heads: usize,
    pub vocab_size: usize,
}

impl MTPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiTokenPrediction<B> {
        let d_model = self.d_model;
        
        MultiTokenPrediction {
            // Initializing with BitLinearConfig for ternary {-1, 0, 1} weights
            proj_h: BitLinearConfig::new(d_model, d_model).init(device),
            proj_emb: BitLinearConfig::new(d_model, d_model).init(device),
            ln_fusion: RmsNormConfig::new(d_model).init(device),

            norm1: RmsNormConfig::new(d_model).init(device),
            // Combining QKV into one BitLinear for efficiency
            attn_qkv: BitLinearConfig::new(d_model, d_model * 3).init(device),
            attn_out: BitLinearConfig::new(d_model, d_model).init(device),
            
            norm2: RmsNormConfig::new(d_model).init(device),
            mlp_fc1: BitLinearConfig::new(d_model, d_model * 4).init(device),
            mlp_fc2: BitLinearConfig::new(d_model * 4, d_model).init(device),

            n_heads: self.n_heads,
            vocab_size: self.vocab_size,
        }
    }
}

impl<B: Backend> MultiTokenPrediction<B> {
    pub fn forward(
        &self,
        h_base: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        shared_emb: &Embedding<B>,
        // Important: If the base model is BitNet, the shared head must also be BitLinear
        shared_head: &BitLinear<B>, 
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [batch_size, seq_len, d_model] = h_base.dims();
        let device = h_base.device();

        // 1. Get embeddings for the 'hint' tokens (t+1)
        let x_embed = shared_emb.forward(targets.clone());

        // 2. DeepSeek-V3 Additive Fusion (Now via ternary projections)
        let fused_h = self.proj_h.forward(h_base);
        let fused_emb = self.proj_emb.forward(x_embed);
        let fused = (fused_h + fused_emb).mul_scalar(0.5);
        let mut x = self.ln_fusion.forward(fused);

        // 3. Ternary Transformer Block
        let x_norm1 = self.norm1.forward(x.clone());

        // --- Ternary Self-Attention ---
        let qkv = self.attn_qkv.forward(x_norm1);
        let head_dim = d_model / self.n_heads;

        // Slice out Q, K, V from the ternary projection result
        let q = qkv.clone().slice([0..batch_size, 0..seq_len, 0..d_model]);
        let k = qkv.clone().slice([0..batch_size, 0..seq_len, d_model..(2 * d_model)]);
        let v = qkv.slice([0..batch_size, 0..seq_len, (2 * d_model)..(3 * d_model)]);

        let reshape_mha = |t: Tensor<B, 3>| {
            t.reshape([batch_size, seq_len, self.n_heads, head_dim])
             .swap_dims(1, 2) 
        };

        let q = reshape_mha(q);
        let k = reshape_mha(k);
        let v = reshape_mha(v);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let k_t = k.transpose();
        let mut attn_scores = q.matmul(k_t).mul_scalar(scale);

        attn_scores = self.apply_causal_mask(attn_scores, seq_len);
        let attn_probs = softmax(attn_scores, 3);
        
        let attn_output = attn_probs.matmul(v);
        let attn_output = attn_output
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, d_model]);
            
        let attn_output = self.attn_out.forward(attn_output);
        x = x + attn_output; 

        // --- Ternary MLP ---
        let x_norm2 = self.norm2.forward(x.clone());
        let mut mlp_out = self.mlp_fc1.forward(x_norm2);
        mlp_out = gelu(mlp_out);
        mlp_out = self.mlp_fc2.forward(mlp_out);

        let x_mtp = x + mlp_out; 

        // 4. Predict t+2 using the shared BitLinear head
        let logits = shared_head.forward(x_mtp);

        // 5. Loss Calculation (Ensuring gradients flow back to both MTP and Base)
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        let logits_flat = logits.clone().reshape([batch_size * seq_len, self.vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);
        
        let loss = loss_fn.forward(logits_flat, targets_flat);

        (logits, loss)
    }

    fn apply_causal_mask(&self, scores: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
        let device = scores.device();
        let mask = Tensor::arange(0..seq_len as i64, &device)
            .reshape([seq_len, 1])
            .greater_equal(Tensor::arange(0..seq_len as i64, &device).reshape([1, seq_len]));
            
        let mask = mask.unsqueeze::<4>().expand(scores.dims());
        scores.mask_fill(mask.bool_not(), f32::NEG_INFINITY)
    }
}