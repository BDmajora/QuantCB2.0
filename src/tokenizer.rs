// src/tokenizer.rs
use std::collections::HashMap;

pub struct CharacterTokenizer {
    char_to_id: HashMap<char, usize>,
    id_to_char: HashMap<usize, char>,
}

impl CharacterTokenizer {
    pub fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        let mut char_to_id = HashMap::new();
        let mut id_to_char = HashMap::new();
        for (i, &c) in chars.iter().enumerate() {
            char_to_id.insert(c, i);
            id_to_char.insert(i, c);
        }
        Self { char_to_id, id_to_char }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars().map(|c| *self.char_to_id.get(&c).unwrap_or(&0)).collect()
    }
    
    // NEW: Needed to generate text every 100 steps
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|id| self.id_to_char.get(id).unwrap_or(&'?')).collect()
    }

    pub fn vocab_size(&self) -> usize { self.char_to_id.len() }
}