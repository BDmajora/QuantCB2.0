use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use crate::model::kv_cache::KVCache;

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
    pub fn forward(
        &self, 
        x: Tensor<B, 3>, 
        cache: Option<KVCache<B>>
    ) -> (Tensor<B, 3>, KVCache<B>) {
        let [batch_size, seq_len_q, _d_model] = x.dims();

        // 1. Query Projections
        let c_q = self.w_dq.forward(x.clone());
        let q_nope = self.w_uq.forward(c_q.clone()); 
        let q_pe = self.w_qr.forward(c_q);           

        // 2. Key/Value Projections
        let c_kv = self.w_dkv.forward(x.clone());
        let k_nope_3d = self.w_uk.forward(c_kv.clone()); 
        let v_nope_3d = self.w_uv.forward(c_kv);         
        let k_pe_3d = self.w_kr.forward(x);              

        // Combine NOPE and PE for the 3D cache representation
        let k_3d = Tensor::cat(vec![k_nope_3d, k_pe_3d], 2);
        let v_3d = v_nope_3d;

        // 3. Update Memory Cache BEFORE reshaping to 4D
        let (k_ctx_3d, v_ctx_3d, updated_cache) = KVCache::update(cache, k_3d, v_3d);

        // 4. Reshape for Multi-Head
        let [_, seq_len_kv, _] = k_ctx_3d.dims();

        let q_nope = self.reshape_for_mha(q_nope, batch_size, seq_len_q, self.d_head_c);
        let q_pe = self.reshape_for_mha(q_pe, batch_size, seq_len_q, self.d_rope);
        let q = Tensor::cat(vec![q_nope, q_pe], 3);
        
        let k_ctx = self.reshape_for_mha(k_ctx_3d, batch_size, seq_len_kv, self.d_head_c + self.d_rope);
        let v_ctx = self.reshape_for_mha(v_ctx_3d, batch_size, seq_len_kv, self.d_head_c);

        // 5. Attention Math
        let scale = 1.0 / ((self.d_head_c + self.d_rope) as f32).sqrt();
        
        let k_t = k_ctx.transpose(); 
        let mut attn_scores = q.matmul(k_t).mul_scalar(scale);

        // 6. Causal Masking
        attn_scores = self.apply_causal_mask(attn_scores, seq_len_q, seq_len_kv);
        let attn_probs = softmax(attn_scores, 3);

        let context = attn_probs.matmul(v_ctx); 

        // 7. Output Projection
        let context_flat = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len_q, self.n_heads * self.d_head_c]);

        (self.o_proj.forward(context_flat), updated_cache)
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

    fn apply_causal_mask(
        &self, 
        scores: Tensor<B, 4>, 
        seq_len_q: usize, 
        seq_len_kv: usize
    ) -> Tensor<B, 4> {
        let device = scores.device();
        
        let offset = seq_len_kv - seq_len_q;

        let mask = Tensor::arange(0..seq_len_q as i64, &device)
            .reshape([seq_len_q, 1])
            .add_scalar(offset as i64)
            .greater_equal(Tensor::arange(0..seq_len_kv as i64, &device).reshape([1, seq_len_kv]));
            
        let mask = mask.unsqueeze::<4>().expand(scores.dims());

        scores.mask_fill(mask.bool_not(), f32::NEG_INFINITY)
    }
}