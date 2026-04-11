// src/tokenizer.rs
use std::collections::HashMap;
use regex::Regex;

pub struct BPETokenizer {
    merges: HashMap<(usize, usize), usize>,
    pub vocab: HashMap<usize, Vec<u8>>,
    
    // Special tokens map & pre-compiled isolator regex
    pub special_tokens: HashMap<String, usize>,
    special_regex: Option<Regex>,
    
    // Fast encoding maps (Mimicking the Python 1.0 'Turbo' PUA logic)
    id_to_char: HashMap<usize, char>,
    char_to_id: HashMap<char, usize>,
    merge_rules: Vec<((char, char), char)>,
    
    offset: u32,
}

impl BPETokenizer {
    pub fn new(special_tokens_list: &[&str]) -> Self {
        let offset = 0xE000; // Unicode Private Use Area
        let mut vocab = HashMap::new();
        let mut id_to_char = HashMap::new();
        let mut char_to_id = HashMap::new();
        let mut special_tokens = HashMap::new();

        // Initialize 0-255 byte-level tokens
        for i in 0..=255 {
            vocab.insert(i, vec![i as u8]);
            let c = std::char::from_u32(offset + i as u32).unwrap();
            id_to_char.insert(i, c);
            char_to_id.insert(c, i);
        }

        // Initialize special tokens immediately after bytes
        let mut next_id = 256;
        for &st in special_tokens_list {
            special_tokens.insert(st.to_string(), next_id);
            vocab.insert(next_id, st.as_bytes().to_vec());
            next_id += 1;
        }

        // Compile regex to quickly isolate special tags
        let special_regex = if !special_tokens_list.is_empty() {
            let escaped: Vec<String> = special_tokens_list.iter().map(|s| regex::escape(s)).collect();
            let pattern = escaped.join("|");
            Some(Regex::new(&pattern).unwrap())
        } else {
            None
        };

        Self {
            merges: HashMap::new(),
            vocab,
            special_tokens,
            special_regex,
            id_to_char,
            char_to_id,
            merge_rules: Vec::new(),
            offset,
        }
    }

    /// Helper to mimic Python's re.findall(r'\S+|\s+', text)
    fn split_text_with_spaces(text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current_chunk = String::new();
        let mut is_space_chunk = false;

        for c in text.chars() {
            let is_space = c.is_whitespace();
            if current_chunk.is_empty() {
                is_space_chunk = is_space;
                current_chunk.push(c);
            } else if is_space == is_space_chunk {
                current_chunk.push(c);
            } else {
                result.push(current_chunk);
                current_chunk = String::from(c);
                is_space_chunk = is_space;
            }
        }
        if !current_chunk.is_empty() {
            result.push(current_chunk);
        }
        result
    }

    pub fn train(&mut self, text: &str, target_vocab_size: usize) {
        let current_base_vocab = 256 + self.special_tokens.len();
        if target_vocab_size <= current_base_vocab { return; }

        // Strip special tokens out so they aren't merged
        let normal_text = if let Some(re) = &self.special_regex {
            re.split(text).collect::<Vec<_>>().join(" ")
        } else {
            text.to_string()
        };

        // 1. Initial Vocab: Map raw words to their Unicode representations
        let mut word_freqs: HashMap<String, usize> = HashMap::new();
        let words = Self::split_text_with_spaces(&normal_text);
        
        for word in words {
            let unicode_word: String = word.bytes()
                .map(|b| std::char::from_u32(self.offset + b as u32).unwrap())
                .collect();
            *word_freqs.entry(unicode_word).or_insert(0) += 1;
        }

        let mut current_vocab_size = current_base_vocab;
        let num_merges = target_vocab_size - current_base_vocab;
        println!("--- Starting BPE training for {} merges ---", num_merges);

        for i in 0..num_merges {
            // 2. Count all adjacent pairs
            let mut pairs: HashMap<(char, char), usize> = HashMap::new();
            for (word, freq) in &word_freqs {
                let chars: Vec<char> = word.chars().collect();
                if chars.len() < 2 { continue; }
                for window in chars.windows(2) {
                    *pairs.entry((window[0], window[1])).or_insert(0) += freq;
                }
            }

            if pairs.is_empty() { break; }

            // 3. Find the most frequent pair
            let best_pair = pairs.into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(pair, _)| pair);

            if let Some((p0, p1)) = best_pair {
                let new_id = current_vocab_size;
                let new_char = std::char::from_u32(self.offset + new_id as u32).unwrap();
                
                let id0 = self.char_to_id[&p0];
                let id1 = self.char_to_id[&p1];
                
                // Record the merge
                self.merges.insert((id0, id1), new_id);
                self.id_to_char.insert(new_id, new_char);
                self.char_to_id.insert(new_char, new_id);
                self.merge_rules.push(((p0, p1), new_char));
                
                // Update vocab map for decoding
                let mut new_bytes = self.vocab[&id0].clone();
                new_bytes.extend(self.vocab[&id1].iter());
                self.vocab.insert(new_id, new_bytes);

                // 4. Perform the global merge using high-speed string replacement
                let best_pair_str = format!("{}{}", p0, p1);
                let new_char_str = new_char.to_string();
                
                let mut new_word_freqs = HashMap::with_capacity(word_freqs.len());
                let mut unique_words = 0;
                for (word, freq) in word_freqs {
                    if word.contains(&best_pair_str) {
                        new_word_freqs.insert(word.replace(&best_pair_str, &new_char_str), freq);
                    } else {
                        new_word_freqs.insert(word, freq);
                    }
                    unique_words += 1;
                }
                word_freqs = new_word_freqs;
                current_vocab_size += 1;
                
                if i % 100 == 0 || i == num_merges - 1 {
                    println!("Merge {}/{}: ({}, {}) -> {} | Unique words: {}", 
                        i + 1, num_merges, id0, id1, new_id, unique_words);
                }
            } else {
                break;
            }
        }
        println!("Success: Learned {} merges. Vocab Size: {}", self.merges.len(), current_vocab_size);
    }

    /// Internal function handling standard PUA BPE encoding
    fn encode_chunk(&self, text: &str) -> Vec<usize> {
        if text.is_empty() { return vec![]; }
        
        let mut current_str: String = text.bytes()
            .map(|b| self.id_to_char[&(b as usize)])
            .collect();

        for ((p0, p1), new_char) in &self.merge_rules {
            let pattern = format!("{}{}", p0, p1);
            let replacement = new_char.to_string();
            current_str = current_str.replace(&pattern, &replacement);
        }

        current_str.chars()
            .map(|c| *self.char_to_id.get(&c).unwrap_or(&0))
            .collect()
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        if text.is_empty() { return vec![]; }
        
        let mut ids = Vec::new();

        if let Some(re) = &self.special_regex {
            let mut last_end = 0;
            for mat in re.find_iter(text) {
                if mat.start() > last_end {
                    ids.extend(self.encode_chunk(&text[last_end..mat.start()]));
                }
                // Inject the special token ID directly
                ids.push(self.special_tokens[mat.as_str()]);
                last_end = mat.end();
            }
            if last_end < text.len() {
                ids.extend(self.encode_chunk(&text[last_end..]));
            }
        } else {
            // No special tokens, encode everything normally
            ids.extend(self.encode_chunk(text));
        }
        
        ids
    }
    
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(b) = self.vocab.get(id) {
                bytes.extend_from_slice(b);
            } else {
                bytes.push(b'?'); 
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize { 
        self.vocab.len() 
    }
}