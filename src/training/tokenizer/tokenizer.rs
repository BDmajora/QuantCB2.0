use std::collections::HashMap;
use regex::Regex;
use crate::training::tokenizer::bpe_trainer::BPETrainer;
use crate::training::tokenizer::bpe_encoder::BPEEncoder;

pub struct BPETokenizer {
    pub merges: HashMap<(u32, u32), u32>,
    pub vocab: HashMap<u32, Vec<u8>>,
    pub special_tokens: HashMap<String, u32>,
    pub(crate) special_regex: Option<Regex>,
}

impl BPETokenizer {
    pub fn new(special_tags: &[&str]) -> Self {
        let mut vocab = HashMap::new();
        let mut special_tokens = HashMap::new();

        // Initialize base vocabulary with single bytes
        for i in 0..=255 {
            vocab.insert(i as u32, vec![i as u8]);
        }

        // Add special tokens to vocab and mapping
        let mut next_id = 256;
        for &tag in special_tags {
            special_tokens.insert(tag.to_owned(), next_id);
            vocab.insert(next_id, tag.as_bytes().to_vec());
            next_id += 1;
        }

        // BUILD REGEX: Sort by length descending to ensure "greedy" matching.
        // This prevents matching a prefix (e.g., "<pad>") when a longer 
        // special token exists (e.g., "<pad_token>").
        let special_regex = if !special_tags.is_empty() {
            let mut sorted_tags = special_tags.to_vec();
            sorted_tags.sort_by(|a, b| b.len().cmp(&a.len()));
            
            let escaped: Vec<String> = sorted_tags.iter().map(|s| regex::escape(s)).collect();
            Some(Regex::new(&escaped.join("|")).unwrap())
        } else {
            None
        };

        Self { 
            merges: HashMap::new(), 
            vocab, 
            special_tokens, 
            special_regex 
        }
    }

    pub fn train(&mut self, text: &str, target_vocab_size: usize) {
        BPETrainer::train(self, text, target_vocab_size);
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        BPEEncoder::encode(self, text)
    }

    pub fn vocab_size(&self) -> usize { 
        self.vocab.len() 
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        // Optimization: Pre-allocate capacity to reduce reallocations.
        // On average, 1 token is roughly 4 bytes of text.
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        
        for &id in ids {
            if let Some(b) = self.vocab.get(&id) {
                bytes.extend_from_slice(b);
            } else {
                // Fallback for unknown IDs
                bytes.push(b'?');
            }
        }
        
        // Handle potential invalid UTF-8 sequences gracefully
        String::from_utf8_lossy(&bytes).into_owned()
    }
}