use rand::Rng;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use reqwest::blocking::Client;
use std::time::Duration;

// 1. IMPORT the config and the tags from your config file
use crate::trainer_config::TrainingConfig;

// 2. RE-EXPORT the tags so trainer.rs can still find them here
pub use crate::trainer_config::{
    TAG_TRUTH, 
    TAG_HALLUCINATE, 
    TAG_SHAKESPEARE,
    TAG_WIKI,
};

pub struct TrainingDataSources;

impl TrainingDataSources {
    // ==========================================
    // SHAKESPEARE LOGIC (In-Memory Processing)
    // ==========================================
    
    pub fn load_complete_shakespeare(config: &TrainingConfig) -> String {
        let cache_dir = "data/cache";
        let cache_path = format!("{}/shakespeare.txt", cache_dir);
        
        if !Path::new(cache_dir).exists() {
            fs::create_dir_all(cache_dir).expect("Failed to create cache directory");
        }
        
        let text = if Path::new(&cache_path).exists() {
            println!("--- Loading Shakespeare from: {} ---", cache_path);
            fs::read_to_string(&cache_path).expect("Failed to read local cache file.")
        } else {
            println!("--- Cache not found. Downloading from Project Gutenberg ---");
            
            let client = Client::builder()
                .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
                .timeout(Duration::from_secs(120))
                .build()
                .unwrap();

            let url = "https://www.gutenberg.org/cache/epub/100/pg100.txt";
            
            let response = client.get(url)
                .send()
                .and_then(|r| r.error_for_status()) 
                .expect("Network failure: Project Gutenberg is currently unreachable.");
            
            let raw_content = response.text().expect("Failed to read response body.");

            fs::write(&cache_path, &raw_content).expect("Failed to write to cache.");
            raw_content
        };

        // Pass to processing logic
        Self::process_text(text, config)
    }

    fn process_text(text: String, config: &TrainingConfig) -> String {
        let mut processing_text = text;

        // 1. Clean up Project Gutenberg noise
        if let Some(start_idx) = processing_text.find("*** START OF THE PROJECT GUTENBERG EBOOK") {
            let remainder = &processing_text[start_idx..];
            let actual_start = remainder.find("\n").unwrap_or(0);
            processing_text = remainder[actual_start..].trim().to_string();
        }
        
        if let Some(end_idx) = processing_text.find("*** END OF THE PROJECT GUTENBERG EBOOK") {
            processing_text.truncate(end_idx);
        }

        let mut combined_raw_text = String::new();
        let mut rng = rand::thread_rng();

        // 2. Chunking logic (Fixing the \n\n delimiter)
        let normalized_text = processing_text.replace("\r\n", "\n");
        
        for chunk in normalized_text.split("\n\n") {
            let clean_chunk = chunk.trim();
            if clean_chunk.is_empty() || clean_chunk.len() < 50 { continue; }
            
            let is_truth = rng.gen_bool(1.0 - config.corruption_rate as f64);
            let tag = if is_truth { TAG_TRUTH } else { TAG_HALLUCINATE };
            
            let final_chunk = if is_truth {
                clean_chunk.to_string()
            } else {
                Self::corrupt_logic(clean_chunk)
            };
            
            combined_raw_text.push_str(&format!("{} {} {}\n", tag, TAG_SHAKESPEARE, final_chunk));
        }
        
        println!("Finished processing {} bytes of the Bard.", combined_raw_text.len());
        combined_raw_text
    }

    // ==========================================
    // WIKIPEDIA LOGIC (Streaming Disk-to-Disk)
    // ==========================================
    
    /// Checks for cached Wikipedia data, streams and processes raw dump if missing.
    /// Returns the PATH to the cache file (not the String content!) to prevent OOM.
    pub fn prepare_wikipedia(config: &TrainingConfig) -> String {
        let cache_dir = "data/cache";
        let raw_dir = "data/raw";
        let processed_path = format!("{}/wiki_processed.txt", cache_dir);
        let raw_path = format!("{}/wikipedia_pre_2022.jsonl", raw_dir);

        if !Path::new(cache_dir).exists() {
            fs::create_dir_all(cache_dir).expect("Failed to create cache directory");
        }

        if Path::new(&processed_path).exists() {
            println!("--- Processed Wikipedia cache found at: {} ---", processed_path);
            return processed_path; 
        } else {
            println!("--- No processed cache found. Processing raw Wikipedia dump... ---");
            if !Path::new(&raw_path).exists() {
                panic!("Raw Wikipedia dump not found at {}. Please download a pre-2022 dump first.", raw_path);
            }
            
            Self::process_wikipedia_streaming(&raw_path, &processed_path, config);
            processed_path
        }
    }

    fn process_wikipedia_streaming(input_path: &str, output_path: &str, config: &TrainingConfig) {
        let input_file = File::open(input_path).expect("Could not open raw wiki file");
        let reader = BufReader::new(input_file);
        let mut output_file = File::create(output_path).expect("Could not create wiki cache file");
        
        let mut rng = rand::thread_rng();
        let mut lines_processed = 0;

        for line in reader.lines() {
            if let Ok(content) = line {
                // 1. Basic Cleaning
                let clean_content = content.trim();
                // Skip lines that are too short to provide useful context
                if clean_content.len() < 100 { continue; }

                // 2. Tagging Logic
                let is_truth = rng.gen_bool(1.0 - config.corruption_rate as f64);
                let quality_tag = if is_truth { TAG_TRUTH } else { TAG_HALLUCINATE };
                
                let final_text = if is_truth {
                    clean_content.to_string()
                } else {
                    Self::corrupt_logic(clean_content)
                };

                // 3. Write directly to Cache
                writeln!(output_file, "{} {} {}", quality_tag, TAG_WIKI, final_text)
                    .expect("Disk full or write failure");

                lines_processed += 1;
                if lines_processed % 100_000 == 0 {
                    println!("--- Cached {} Wikipedia articles ---", lines_processed);
                }
            }
        }
        
        println!("--- Finished Wikipedia Processing. Processed {} total items. ---", lines_processed);
    }

    // ==========================================
    // SHARED UTILITIES
    // ==========================================
    
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