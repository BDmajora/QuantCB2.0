use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Int, ElementConversion}; 
use crate::model::QuantCB;
use crate::tokenizer::BPETokenizer;
use crate::kv_cache::KVCache;

pub struct TextGenerator;

impl TextGenerator {
    pub fn generate<B: Backend>(
        model: &QuantCB<B>,
        tokenizer: &BPETokenizer,
        device: &B::Device,
        prompt: &str,
        max_len: usize,
    ) -> String {
        let mut tokens = tokenizer.encode(prompt);
        let mut cache: Option<Vec<Option<KVCache<B>>>> = None;

        for _ in 0..max_len {
            // Incremental decoding: if cache exists, only process the last token
            let input_tokens = if cache.is_some() {
                let last = *tokens.last().unwrap_or(&0); 
                vec![last]
            } else {
                tokens.clone()
            };

            // Create tensor from current token(s)
            let input_tensor = Tensor::<B, 1, Int>::from_ints(
                input_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>().as_slice(),
                device
            ).unsqueeze::<2>(); // [1, seq_len]

            // Forward pass
            let (logits, _, _, _, next_cache, _) = model.forward(input_tensor, cache);
            cache = Some(next_cache);

            let [_, seq_len, vocab_size] = logits.dims();
            
            // Get the last logit
            let last_token_logits = logits.slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size]);
            
            // FIX: Convert to u32 first, then cast to usize
            let next_token_id = last_token_logits
                .argmax(2)
                .into_data()
                .value[0]
                .elem::<u32>() as usize;

            tokens.push(next_token_id);
            
            // Optional check for end-of-text if your tokenizer defines it
            // if Some(next_token_id) == tokenizer.special_tokens.get("<|endoftext|>").copied() { break; }
        }
        
        tokenizer.decode(&tokens)
    }
}