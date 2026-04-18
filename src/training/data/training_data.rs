use reqwest::Client;
use tokio::sync::mpsc;
use serde_json::Value;
use rand::Rng;

use crate::training::trainer::trainer_config::{
    TrainingConfig, TAG_TRUTH, TAG_HALLUCINATE, TAG_WIKI, TAG_SHAKESPEARE
};

pub struct VolatileDataPipeline {
    shakespeare_buffer: Vec<String>,
    active_wiki_buffer: Vec<String>,
    wiki_receiver: Option<mpsc::Receiver<Vec<String>>>,
    corruption_rate: f32,
}

impl VolatileDataPipeline {
    pub fn new() -> Self {
        Self {
            shakespeare_buffer: Vec::new(),
            active_wiki_buffer: Vec::new(),
            wiki_receiver: None,
            corruption_rate: 0.0,
        }
    }

    pub async fn load_shakespeare(&mut self, config: &TrainingConfig) -> String {
        println!("--- Downloading Complete Works of Shakespeare to RAM ---");
        let client = Client::builder()
            .user_agent("QuantCB-Trainer/2.0")
            .build()
            .unwrap();

        let raw_content = client.get("https://www.gutenberg.org/cache/epub/100/pg100.txt")
            .send()
            .await
            .expect("Failed to fetch Shakespeare")
            .text()
            .await
            .expect("Failed to read text");

        let mut processing_text = raw_content.clone();
        
        if let Some(start_idx) = processing_text.find("*** START OF THE PROJECT GUTENBERG EBOOK") {
            let remainder = &processing_text[start_idx..];
            let actual_start = remainder.find('\n').unwrap_or(0);
            processing_text = remainder[actual_start..].trim().to_string();
        }
        
        if let Some(end_idx) = processing_text.find("*** END OF THE PROJECT GUTENBERG EBOOK") {
            processing_text.truncate(end_idx);
        }

        let normalized = processing_text.replace("\r\n", "\n");
        let mut rng = rand::thread_rng();

        for chunk in normalized.split("\n\n") {
            let clean = chunk.trim();
            if clean.len() < 50 { continue; }

            let is_truth = rng.gen_bool(1.0 - config.corruption_rate as f64);
            let tag = if is_truth { TAG_TRUTH } else { TAG_HALLUCINATE };
            let final_chunk = if is_truth {
                clean.to_string()
            } else {
                Self::corrupt_logic(clean)
            };

            self.shakespeare_buffer.push(format!("{} {} {}", tag, TAG_SHAKESPEARE, final_chunk));
        }

        println!("Loaded {} Shakespeare chunks into RAM.", self.shakespeare_buffer.len());
        raw_content
    }

    pub fn get_random_shakespeare(&self) -> String {
        if self.shakespeare_buffer.is_empty() {
            return "No Shakespeare data loaded.".to_string();
        }
        let mut rng = rand::thread_rng();
        let idx = rng.gen_range(0..self.shakespeare_buffer.len());
        self.shakespeare_buffer[idx].clone()
    }

    pub async fn get_next_wiki_text(&mut self) -> String {
        if self.active_wiki_buffer.is_empty() {
            if let Some(rx) = self.wiki_receiver.as_mut() {
                match rx.recv().await {
                    Some(new_chunk) => {
                        self.active_wiki_buffer = new_chunk;
                    }
                    None => {
                        return self.get_random_shakespeare();
                    }
                }
            } else {
                return self.get_random_shakespeare();
            }
        }

        self.active_wiki_buffer.pop().unwrap_or_else(|| self.get_random_shakespeare())
    }

    pub fn start_huggingface_chunker(&mut self, config: &TrainingConfig) {
        self.corruption_rate = config.corruption_rate;
        // Fix 3.1: Increase channel buffer from 2 to 8
        let (tx, rx) = mpsc::channel::<Vec<String>>(8);
        self.wiki_receiver = Some(rx);

        let c_rate = config.corruption_rate;

        tokio::spawn(async move {
            let client = Client::builder().user_agent("QuantCB-Trainer/2.0").build().unwrap();
            
            let chunk_urls = vec![
                "https://huggingface.co/datasets/v_m_s/wikipedia_20231101_simple_jsonl/resolve/main/data.jsonl",
                // Add more shards here if available, or loop the same URL
            ];

            // Fix 3.2: Wrap the URL iteration in an infinite loop so it cycles
            loop {
                for url in &chunk_urls {
                    let mut backoff = 1;
                    let raw_text = loop {
                        match client.get(*url).send().await {
                            Ok(res) => {
                                if let Ok(text) = res.text().await { break text; }
                            }
                            Err(e) => {
                                println!("Download failed: {}. Retrying in {}s", e, backoff);
                                tokio::time::sleep(tokio::time::Duration::from_secs(backoff)).await;
                                backoff = (backoff * 2).min(30);
                            }
                        }
                    };

                    let mut processed_chunk = Vec::new();
                    
                    {
                        let mut rng = rand::thread_rng();

                        for line in raw_text.lines() {
                            if let Ok(json) = serde_json::from_str::<Value>(line) {
                                if let Some(text) = json["text"].as_str() {
                                    if text.len() < 100 { continue; }
                                    let is_truth = rng.gen_bool(1.0 - c_rate as f64);
                                    let tag = if is_truth { TAG_TRUTH } else { TAG_HALLUCINATE };
                                    let final_text = if is_truth {
                                        text.to_string()
                                    } else {
                                        Self::corrupt_logic(text)
                                    };
                                    processed_chunk.push(format!("{} {} {}", tag, TAG_WIKI, final_text));
                                }
                            }
                        }
                    } 

                    // Return completely out of the spawned task if the receiver is dropped
                    if tx.send(processed_chunk).await.is_err() { 
                        return; 
                    }
                }
            }
        });
    }

    fn corrupt_logic(text: &str) -> String {
        let mut tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.len() < 5 { return text.to_string(); }
        
        let mut rng = rand::thread_rng();
        let num_swaps = (tokens.len() / 8).max(2);
        
        for _ in 0..num_swaps {
            let idx = rng.gen_range(0..tokens.len() - 1);
            tokens.swap(idx, idx + 1);
        }
        
        tokens.join(" ")
    }
}