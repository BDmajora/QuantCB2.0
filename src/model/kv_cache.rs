use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct KVCache<B: Backend> {
    pub k: Tensor<B, 3>,
    pub v: Tensor<B, 3>,
}

impl<B: Backend> KVCache<B> {
    /// Creates a new cache or updates an existing one by concatenating new tokens.
    pub fn update(
        cache: Option<Self>,
        new_k: Tensor<B, 3>,
        new_v: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Self) {
        match cache {
            Some(mut existing) => {
                // Dim 1 is the sequence length: [batch, seq, d_model]
                let k_full = Tensor::cat(vec![existing.k, new_k], 1);
                let v_full = Tensor::cat(vec![existing.v, new_v], 1);
                
                existing.k = k_full.clone();
                existing.v = v_full.clone();
                
                (k_full, v_full, existing)
            }
            None => {
                let new_cache = Self {
                    k: new_k.clone(),
                    v: new_v.clone(),
                };
                (new_k, new_v, new_cache)
            }
        }
    }

    /// Helper to initialize an empty vector of caches for the whole model
    pub fn init_empty_list(num_layers: usize) -> Vec<Option<Self>> {
        vec![None; num_layers]
    }
}