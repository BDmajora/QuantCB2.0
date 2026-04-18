use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Int, ElementConversion};
use crate::model::QuantCB; 
use crate::training::BPETokenizer;

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
        let mut tokens = tokenizer.encode(prompt); 
        let vocab_size_actual = tokenizer.vocab_size();
        
        for _ in 0..max_tokens {
            let context_len = tokens.len();
            
            // Build input tensor from current token list
            let data_vec: Vec<B::IntElem> = tokens
                .iter()
                .map(|&t| (t as i64).elem())
                .collect();

            let input_tensor = Tensor::<B, 2, Int>::from_data(
                burn::tensor::Data::new(data_vec, [1, context_len].into()),
                device,
            ); 

            // Dummy targets for MTP (required by the signature, though not used for loss here)
            let dummy: Tensor<B, 2, Int> = Tensor::zeros([1, context_len], device);
            
            // FIX: Pass model.loop_depth as the 4th argument to satisfy the new signature
            let (logits, _, _, _, _, _, _) = model.forward_mtp(
                input_tensor, 
                dummy, 
                None, 
                model.loop_depth
            );
            
            // Fix E0282: Explicitly type the dims array
            let [batch, seq_len, vocab_size]: [usize; 3] = logits.dims();
            let last_logits_tensor = logits
                .slice([0..batch, (seq_len - 1)..seq_len, 0..vocab_size])
                .reshape([vocab_size]);
                
            let mut last_logits: Vec<f32> = last_logits_tensor.into_data().convert::<f32>().value;

            // --- Bit-Stable Repetition Penalty ---
            let window_size = usize::min(context_len, 64);
            let recent_tokens = &tokens[tokens.len().saturating_sub(window_size)..];
            
            for &prev_token in recent_tokens {
                let idx = prev_token as usize;
                if idx < last_logits.len() {
                    let val = last_logits[idx]; 
                    if val > 0.0 {
                        last_logits[idx] = val / rep_penalty;
                    } else {
                        last_logits[idx] = val * rep_penalty;
                    }
                }
            }

            // --- Safe Greedy/Top-1 Sampling ---
            let mut max_val = f32::NEG_INFINITY;
            let mut next_token = 0;
            let temp = f32::max(temperature, 0.01);
            
            for (i, &logit) in last_logits.iter().enumerate() {
                let adjusted = logit / temp; 
                
                if adjusted > max_val {
                    max_val = adjusted;
                    next_token = i; 
                }
            }
            
            if next_token >= vocab_size_actual {
                break;
            }

            tokens.push(next_token as u32);
        }
        
        tokenizer.decode(&tokens)
    }
}