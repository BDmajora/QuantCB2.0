// src/training_data.rs
use rand::Rng;
use reqwest::blocking::get;
// Import the Config struct itself instead of the constant
pub use crate::trainer_config::{
    TrainingConfig,
    TAG_TRUTH, 
    TAG_HALLUCINATE, 
    TAG_SHAKESPEARE, 
};

pub struct TrainingDataSources;

impl TrainingDataSources {
    /// Now accepts &TrainingConfig to access the dynamic corruption_rate
    pub fn load_tiny_shakespeare(config: &TrainingConfig) -> String {
        println!("--- Loading Tiny Shakespeare (Network Fetch) ---");
        let url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
        
        let response = get(url).expect("Network failure: check your internet connection.");
        let text = response.text().expect("Text parsing failure: could not read response body.");

        let mut combined_raw_text = String::new();
        let mut rng = rand::thread_rng();

        for chunk in text.split("\n\n") {
            if chunk.trim().is_empty() { continue; }
            
            // 1. Determine if this chunk is truth or hallucination
            // We cast corruption_rate to f64 because gen_bool requires f64
            let is_truth = rng.gen_bool(1.0 - config.corruption_rate as f64);
            let tag = if is_truth { TAG_TRUTH } else { TAG_HALLUCINATE };
            
            // 2. Corrupt the text if it's a hallucination
            let final_chunk = if is_truth {
                chunk.to_string()
            } else {
                Self::corrupt_logic(chunk)
            };
            
            // 3. Assemble: <|tag|> <|source|> Text...
            combined_raw_text.push_str(&format!("{} {} {}\n", tag, TAG_SHAKESPEARE, final_chunk));
        }
        
        combined_raw_text
    }

    fn corrupt_logic(text: &str) -> String {
        let mut tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.len() < 2 { return text.to_string(); }
        
        let mut rng = rand::thread_rng();
        
        let num_swaps = (tokens.len() / 10).max(1);
        
        for _ in 0..num_swaps {
            let idx = rng.gen_range(0..tokens.len() - 1);
            tokens.swap(idx, idx + 1);
        }
        
        tokens.join(" ")
    }
}