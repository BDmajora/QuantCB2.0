use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Embedding, LayerNorm, LayerNormConfig, Linear, LinearConfig};
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
    // Fusion Projections
    pub proj_h: Linear<B>,
    pub proj_emb: Linear<B>,
    pub ln_fusion: LayerNorm<B>,

    // Transformer Block
    pub norm1: LayerNorm<B>,
    pub attn_qkv: Linear<B>,
    pub attn_out: Linear<B>,
    pub norm2: LayerNorm<B>,
    pub mlp_fc1: Linear<B>,
    pub mlp_fc2: Linear<B>,

    pub n_heads: usize,
    pub vocab_size: usize,
}

impl MTPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiTokenPrediction<B> {
        let d_model = self.d_model;
        
        MultiTokenPrediction {
            // Additive fusion linear layers
            proj_h: LinearConfig::new(d_model, d_model).with_bias(false).init(device),
            proj_emb: LinearConfig::new(d_model, d_model).with_bias(false).init(device),
            ln_fusion: LayerNormConfig::new(d_model).init(device),

            // Transformer block
            norm1: LayerNormConfig::new(d_model).init(device),
            attn_qkv: LinearConfig::new(d_model, d_model * 3).init(device),
            attn_out: LinearConfig::new(d_model, d_model).init(device),
            
            norm2: LayerNormConfig::new(d_model).init(device),
            // Assuming standard MLP expansion ratio of 4
            mlp_fc1: LinearConfig::new(d_model, d_model * 4).init(device),
            mlp_fc2: LinearConfig::new(d_model * 4, d_model).init(device),

            n_heads: self.n_heads,
            vocab_size: self.vocab_size,
        }
    }
}

impl<B: Backend> MultiTokenPrediction<B> {
    /// Forward pass for the MTP module.
    /// Takes the base model's hidden states, the target tokens, and references 
    /// to the shared embedding and output head from the base model.
    pub fn forward(
        &self,
        h_base: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
        shared_emb: &Embedding<B>,
        shared_head: &Linear<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let [batch_size, seq_len, d_model] = h_base.dims();
        let device = h_base.device();

        // ==========================================
        // 1. Get embeddings for the 'hint' tokens
        // ==========================================
        let x_embed = shared_emb.forward(targets.clone());

        // ==========================================
        // 2. DeepSeek-V3 style additive fusion
        // ==========================================
        let fused_h = self.proj_h.forward(h_base);
        let fused_emb = self.proj_emb.forward(x_embed);
        let fused = (fused_h + fused_emb).mul_scalar(0.5);
        let mut x = self.ln_fusion.forward(fused);

        // ==========================================
        // 3. Transformer Block
        // ==========================================
        
        // --- 3a. Norm 1 ---
        let x_norm1 = self.norm1.forward(x.clone());

        // --- 3b. Self-Attention ---
        let qkv = self.attn_qkv.forward(x_norm1);
        let head_dim = d_model / self.n_heads;

        // Split Q, K, V
        let q = qkv.clone().slice([0..batch_size, 0..seq_len, 0..d_model]);
        let k = qkv.clone().slice([0..batch_size, 0..seq_len, d_model..(2 * d_model)]);
        let v = qkv.slice([0..batch_size, 0..seq_len, (2 * d_model)..(3 * d_model)]);

        let reshape_mha = |t: Tensor<B, 3>| {
            t.reshape([batch_size, seq_len, self.n_heads, head_dim])
             .swap_dims(1, 2) // Outputs [B, n_heads, T, head_dim]
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

        x = x + attn_output; // Residual 1

        // --- 3c. Norm 2 ---
        let x_norm2 = self.norm2.forward(x.clone());

        // --- 3d. MLP ---
        let mut mlp_out = self.mlp_fc1.forward(x_norm2);
        mlp_out = gelu(mlp_out);
        mlp_out = self.mlp_fc2.forward(mlp_out);

        let x_mtp = x + mlp_out; // Residual 2

        // ==========================================
        // 4. Predict t+2 using the shared head
        // ==========================================
        let logits = shared_head.forward(x_mtp);

        // ==========================================
        // 5. Calculate Loss
        // ==========================================
        let loss_fn = CrossEntropyLossConfig::new().init(&device);
        
        // Flatten to [B * T, vocab_size] and [B * T] for loss computation
        let logits_flat = logits.clone().reshape([batch_size * seq_len, self.vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);
        
        let loss = loss_fn.forward(logits_flat, targets_flat);

        (logits, loss)
    }

    /// Reused causal mask implementation to prevent future token viewing
    fn apply_causal_mask(&self, scores: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
        let device = scores.device();
        
        let mask = Tensor::arange(0..seq_len as i64, &device)
            .reshape([seq_len, 1])
            .greater_equal(Tensor::arange(0..seq_len as i64, &device).reshape([1, seq_len]));
            
        let mask = mask.unsqueeze::<4>().expand(scores.dims());

        scores.mask_fill(mask.bool_not(), f32::NEG_INFINITY)
    }
}