use rand::Rng;
use reqwest::blocking::get;

pub use crate::trainer_config::{
    TrainingConfig,
    TAG_TRUTH, 
    TAG_HALLUCINATE, 
    TAG_SHAKESPEARE, 
};

pub struct TrainingDataSources;

impl TrainingDataSources {
    /// Fetches the Complete Works of Shakespeare from Project Gutenberg.
    pub fn load_complete_shakespeare(config: &TrainingConfig) -> String {
        println!("--- Loading COMPLETE Shakespeare (Project Gutenberg) ---");
        
        // URL for the Project Gutenberg "Complete Works of William Shakespeare"
        let url = "https://www.gutenberg.org/cache/epub/100/pg100.txt";
        
        let response = get(url).expect("Network failure: check your internet connection.");
        let mut text = response.text().expect("Text parsing failure: could not read response body.");

        // 1. Clean up Project Gutenberg noise (Legal headers/footers)
        // We find the markers Gutenberg uses to denote the start/end of the actual book.
        if let Some(start_idx) = text.find("*** START OF THE PROJECT GUTENBERG EBOOK") {
            let actual_start = text[start_idx..].find("\n").unwrap_or(0);
            text = text[start_idx + actual_start..].to_string();
        }
        
        if let Some(end_idx) = text.find("*** END OF THE PROJECT GUTENBERG EBOOK") {
            text.truncate(end_idx);
        }

        let mut combined_raw_text = String::new();
        let mut rng = rand::thread_rng();

        // 2. Chunking logic
        // The complete works uses a mix of \r\n and \n. We normalize and split.
        let normalized_text = text.replace("\r\n", "\n");
        
        // Splitting by triple newlines often separates plays/scenes better in the big file
        for chunk in normalized_text.split("\n\n\n") {
            let clean_chunk = chunk.trim();
            if clean_chunk.is_empty() || clean_chunk.len() < 50 { continue; }
            
            // Determine if truth or hallucination
            let is_truth = rng.gen_bool(1.0 - config.corruption_rate as f64);
            let tag = if is_truth { TAG_TRUTH } else { TAG_HALLUCINATE };
            
            // Corrupt if necessary
            let final_chunk = if is_truth {
                clean_chunk.to_string()
            } else {
                Self::corrupt_logic(clean_chunk)
            };
            
            // Assemble with tags
            combined_raw_text.push_str(&format!("{} {} {}\n", tag, TAG_SHAKESPEARE, final_chunk));
        }
        
        println!("Finished processing {} bytes of the Bard.", combined_raw_text.len());
        combined_raw_text
    }

    fn corrupt_logic(text: &str) -> String {
        let mut tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.len() < 5 { return text.to_string(); }
        
        let mut rng = rand::thread_rng();
        
        // Increase swap intensity for the larger dataset to create more "obvious" hallucinations
        let num_swaps = (tokens.len() / 8).max(2);
        
        for _ in 0..num_swaps {
            let idx = rng.gen_range(0..tokens.len() - 1);
            tokens.swap(idx, idx + 1);
        }
        
        tokens.join(" ")
    }
}