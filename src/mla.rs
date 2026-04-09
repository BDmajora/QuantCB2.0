use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Config, Debug)]
pub struct MLAConfig {
    pub d_model: usize,
    pub n_heads: usize,
    /// Latent dimension for KV compression (c_kv)
    pub d_c: usize,
    /// Latent dimension for Q compression (c_q)
    pub d_c_q: usize,
    /// Dimension per head for the Non-Position (NOPE) elements
    pub d_head_c: usize,
    /// Dimension per head for the Rotary Position Embedding (RoPE) elements
    pub d_rope: usize,
}

#[derive(Module, Debug)]
pub struct MultiHeadLatentAttention<B: Backend> {
    // Query Projections
    w_dq: Linear<B>, // Down-project X -> c_q
    w_uq: Linear<B>, // Up-project c_q -> Q_NOPE
    w_qr: Linear<B>, // Project c_q -> Q_PE (RoPE queries)

    // Key/Value Projections
    w_dkv: Linear<B>, // Down-project X -> c_kv
    w_uk: Linear<B>,  // Up-project c_kv -> K_NOPE
    w_uv: Linear<B>,  // Up-project c_kv -> V_NOPE
    w_kr: Linear<B>,  // Project X -> K_PE (RoPE keys)

    // Output Projection
    o_proj: Linear<B>,

    n_heads: usize,
    d_head_c: usize,
    d_rope: usize,
}

impl MLAConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadLatentAttention<B> {
        let nope_dim = self.n_heads * self.d_head_c;
        let rope_dim = self.n_heads * self.d_rope;

        MultiHeadLatentAttention {
            w_dq: LinearConfig::new(self.d_model, self.d_c_q).init(device),
            w_uq: LinearConfig::new(self.d_c_q, nope_dim).init(device),
            w_qr: LinearConfig::new(self.d_c_q, rope_dim).init(device),

            w_dkv: LinearConfig::new(self.d_model, self.d_c).init(device),
            w_uk: LinearConfig::new(self.d_c, nope_dim).init(device),
            w_uv: LinearConfig::new(self.d_c, nope_dim).init(device),
            w_kr: LinearConfig::new(self.d_model, rope_dim).init(device),

            o_proj: LinearConfig::new(nope_dim, self.d_model).init(device),

            n_heads: self.n_heads,
            d_head_c: self.d_head_c,
            d_rope: self.d_rope,
        }
    }
}

impl<B: Backend> MultiHeadLatentAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _d_model] = x.dims();

        // ==========================================
        // 1. Query Compression & Projection
        // ==========================================
        let c_q = self.w_dq.forward(x.clone());
        let q_nope = self.w_uq.forward(c_q.clone()); 
        let q_pe = self.w_qr.forward(c_q);           

        // ==========================================
        // 2. Key/Value Compression
        // ==========================================
        let c_kv = self.w_dkv.forward(x.clone());
        let k_nope = self.w_uk.forward(c_kv.clone()); 
        let v_nope = self.w_uv.forward(c_kv);         
        let k_pe = self.w_kr.forward(x);              

        // ==========================================
        // 3. Reshape for Multi-Head Attention
        // ==========================================
        let q_nope = self.reshape_for_mha(q_nope, batch_size, seq_len, self.d_head_c);
        let q_pe = self.reshape_for_mha(q_pe, batch_size, seq_len, self.d_rope);
        
        let k_nope = self.reshape_for_mha(k_nope, batch_size, seq_len, self.d_head_c);
        let k_pe = self.reshape_for_mha(k_pe, batch_size, seq_len, self.d_rope);
        
        let v = self.reshape_for_mha(v_nope, batch_size, seq_len, self.d_head_c);

        // ==========================================
        // 4. Concatenate NOPE and PE parts
        // ==========================================
        let q = Tensor::cat(vec![q_nope, q_pe], 3);
        let k = Tensor::cat(vec![k_nope, k_pe], 3);

        // ==========================================
        // 5. Scaled Dot-Product Attention
        // ==========================================
        let scale = 1.0 / ((self.d_head_c + self.d_rope) as f32).sqrt();
        
        let k_t = k.transpose(); 
        let mut attn_scores = q.matmul(k_t).mul_scalar(scale);

        // Apply causal mask
        attn_scores = self.apply_causal_mask(attn_scores, seq_len);
        let attn_probs = softmax(attn_scores, 3);

        let context = attn_probs.matmul(v); 

        // ==========================================
        // 6. Output Projection
        // ==========================================
        let context_flat = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.n_heads * self.d_head_c]);

        self.o_proj.forward(context_flat)
    }

    fn reshape_for_mha(
        &self,
        tensor: Tensor<B, 3>,
        batch_size: usize,
        seq_len: usize,
        dim: usize,
    ) -> Tensor<B, 4> {
        tensor
            .reshape([batch_size, seq_len, self.n_heads, dim])
            .swap_dims(1, 2)
    }

    /// Fixed: Uses bool_not() to correctly handle Burn's Bool Tensor API
    fn apply_causal_mask(&self, scores: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
        let device = scores.device();
        
        let mask = Tensor::arange(0..seq_len as i64, &device)
            .reshape([seq_len, 1])
            .greater_equal(Tensor::arange(0..seq_len as i64, &device).reshape([1, seq_len]));
            
        let mask = mask.unsqueeze::<4>().expand(scores.dims());

        // Inverting the boolean mask to fill future tokens
        scores.mask_fill(mask.bool_not(), f32::NEG_INFINITY)
    }
}