use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Int, ElementConversion};
use crate::model::QuantCB;
use crate::tokenizer::BPETokenizer;

pub struct TextGenerator;

impl TextGenerator {
    pub fn generate<B: Backend>(
        model: &QuantCB<B>,
        tokenizer: &BPETokenizer,
        device: &B::Device,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        rep_penalty: f32, 
    ) -> String {
        let mut tokens = tokenizer.encode(prompt); // This is Vec<u32>
        
        for _ in 0..max_tokens {
            let context_len = tokens.len();
            
            let data_vec: Vec<B::IntElem> = tokens
                .iter()
                .map(|&t| (t as i64).elem())
                .collect();

            let input_tensor = Tensor::<B, 1, Int>::from_data(
                burn::tensor::Data::new(data_vec, [context_len].into()),
                device,
            ).unsqueeze::<2>(); // Shape: [1, seq_len]

            let dummy: Tensor<B, 2, Int> = Tensor::zeros([1, context_len], device);
            
            let (logits, _, _, _, _, _, _) = model.forward_mtp(input_tensor, dummy, None);
            let [_, seq_len, vocab_size] = logits.dims();
            
            let last_logits_tensor = logits
                .slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size])
                .reshape([vocab_size]);
                
            let mut last_logits = last_logits_tensor.into_data().value;

            // Repetition Penalty
            let window_size = usize::min(context_len, 64);
            for i in (context_len - window_size)..context_len {
                let tok_idx = tokens[i] as usize;
                if tok_idx < last_logits.len() {
                    let val = last_logits[tok_idx].elem::<f32>();
                    if val > 0.0 {
                        last_logits[tok_idx] = B::FloatElem::from_elem(val / rep_penalty);
                    } else {
                        last_logits[tok_idx] = B::FloatElem::from_elem(val * rep_penalty);
                    }
                }
            }

            // Greedy Sampling
            let mut max_val = f32::NEG_INFINITY;
            let mut next_token = 0;
            
            for (i, &logit) in last_logits.iter().enumerate() {
                let adjusted_logit = logit.elem::<f32>() / f32::max(temperature, 0.01);
                
                if adjusted_logit > max_val {
                    max_val = adjusted_logit;
                    next_token = i; // i is usize
                }
            }
            
            // FIX: Cast usize next_token to u32
            tokens.push(next_token as u32);
        }
        
        tokenizer.decode(&tokens)
    }
}