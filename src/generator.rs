use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Int, Data, Shape, ElementConversion};
use crate::model::QuantCB;
use crate::tokenizer::CharacterTokenizer;
use crate::kv_cache::KVCache;

pub struct TextGenerator;

impl TextGenerator {
    pub fn generate<B: Backend>(
        model: &QuantCB<B>,
        tokenizer: &CharacterTokenizer,
        device: &B::Device,
        prompt: &str,
        max_len: usize,
    ) -> String {
        let mut tokens = tokenizer.encode(prompt);
        let mut cache: Option<Vec<Option<KVCache<B>>>> = None;

        for _ in 0..max_len {
            // If we have a cache, only process the very last token
            let input_tokens = if cache.is_some() {
                vec![*tokens.last().unwrap()]
            } else {
                tokens.clone()
            };

            let input_data = Data::new(
                input_tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                Shape::new([1, input_tokens.len()]),
            ).convert();

            let input_tensor = Tensor::<B, 2, Int>::from_data(input_data, device);

            // Use the base forward pass (6th return element is h_base, which we ignore)
            let (logits, _, _, _, next_cache, _) = model.forward(input_tensor, cache);
            cache = Some(next_cache);

            let [_, seq_len, vocab_size] = logits.dims();
            // Get logits for the last generated token
            let last_token_logits = logits.slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size]);
            let next_token = last_token_logits.argmax(2).into_data().value[0].elem::<u32>() as usize;

            tokens.push(next_token);
            
            // Optional: stop if model predicts end-of-text if you have that token
        }
        tokenizer.decode(&tokens)
    }
}