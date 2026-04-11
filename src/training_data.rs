// src/training_data.rs
use rand::Rng;
use reqwest::blocking::get;

pub const TAG_TRUTH: &str = "<|truth|>";
pub const TAG_HALLUCINATE: &str = "<|hallucinate|>";
pub const TAG_SHAKESPEARE: &str = "<|shakespeare|>";
pub const CORRUPTION_RATE: f64 = 0.15;

pub struct TrainingDataSources;

impl TrainingDataSources {
    pub fn load_tiny_shakespeare() -> String {
        println!("--- Loading Tiny Shakespeare (Network Fetch) ---");
        let url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";
        let response = get(url).expect("Network failure");
        let text = response.text().expect("Text parsing failure");

        let mut combined_raw_text = String::new();
        let mut rng = rand::thread_rng();

        for chunk in text.split("\n\n") {
            if chunk.trim().is_empty() { continue; }
            let mut final_chunk = chunk.to_string();
            let tag = if rng.gen_bool(1.0 - CORRUPTION_RATE) {
                TAG_TRUTH
            } else {
                final_chunk = Self::corrupt_logic(&final_chunk);
                TAG_HALLUCINATE
            };
            combined_raw_text.push_str(&format!("{} {} {}\n", tag, TAG_SHAKESPEARE, final_chunk));
        }
        combined_raw_text
    }

    fn corrupt_logic(text: &str) -> String {
        let mut tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.len() < 2 { return text.to_string(); }
        let mut rng = rand::thread_rng();
        for _ in 0..(tokens.len() / 10).max(1) {
            let idx = rng.gen_range(0..tokens.len() - 1);
            tokens.swap(idx, idx + 1);
        }
        tokens.join(" ")
    }
}