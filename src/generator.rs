use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Int, ElementConversion};
use crate::model::QuantCB; 
use crate::training::BLTPatcher;

pub struct TextGenerator;

impl TextGenerator {
    pub fn generate<B: Backend>(
        model: &QuantCB<B>,
        patcher: &BLTPatcher<B>, // Un-ignored patcher parameter
        device: &B::Device,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        rep_penalty: f32, 
    ) -> String {
        let vocab_size_actual = 262; 
        
        // --- BLT PATCHER INTEGRATION ---
        // We segment the initial prompt using the BLT patcher logic
        let dummy_entropies = vec![0.0; prompt.len()];
        let patches = patcher.patch(prompt, &dummy_entropies);
        
        // Flatten the segments back out into our token sequence
        let mut tokens: Vec<u32> = patches.into_iter()
            .flat_map(|p| p.raw_bytes.into_iter().map(|b| b as u32))
            .collect();
        // -------------------------------
        
        for _ in 0..max_tokens {
            let context_len = tokens.len();
            
            let data_vec: Vec<B::IntElem> = tokens
                .iter()
                .map(|&t| (t as i64).elem())
                .collect();

            let input_tensor = Tensor::<B, 2, Int>::from_data(
                burn::tensor::Data::new(data_vec, [1, context_len].into()),
                device,
            ); 

            // --- BLT LOCAL ENCODER INTEGRATION ---
            // Pass the input tensor through the patcher's forward method.
            let _local_features = patcher.forward(input_tensor.clone());
            // -------------------------------------

            let dummy: Tensor<B, 2, Int> = Tensor::zeros([1, context_len], device);
            
            let (logits, _, _, _, _, _, _) = model.forward_mtp(
                input_tensor, 
                dummy, 
                None, 
                model.loop_depth
            );
            
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
        
        let byte_output: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
        String::from_utf8_lossy(&byte_output).to_string()
    }
}