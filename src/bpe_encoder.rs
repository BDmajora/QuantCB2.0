use rayon::prelude::*;
use crate::tokenizer::BPETokenizer;

pub struct BPEEncoder;

impl BPEEncoder {
    pub fn encode(tokenizer: &BPETokenizer, text: &str) -> Vec<u32> {
        if let Some(re) = &tokenizer.special_regex {
            let mut ids = Vec::new();
            let mut last = 0;
            
            for mat in re.find_iter(text) {
                if mat.start() > last {
                    ids.extend(Self::encode_chunk(tokenizer, &text[last..mat.start()]));
                }
                ids.push(*tokenizer.special_tokens.get(mat.as_str()).expect("Special token missing"));
                last = mat.end();
            }
            
            if last < text.len() { 
                ids.extend(Self::encode_chunk(tokenizer, &text[last..])); 
            }
            ids
        } else {
            Self::encode_chunk(tokenizer, text)
        }
    }

    fn encode_chunk(tokenizer: &BPETokenizer, text: &str) -> Vec<u32> {
        text.par_split_whitespace()
            .flat_map(|word| {
                let mut ids: Vec<u32> = word.bytes().map(|b| b as u32).collect();
                while ids.len() >= 2 {
                    // FIX: Explicitly tell Rust what types live inside this Option
                    let mut best_merge: Option<(u32, u32, u32)> = None; 
                    
                    for window in ids.windows(2) {
                        if let Some(&new_id) = tokenizer.merges.get(&(window[0], window[1])) {
                            match best_merge {
                                None => best_merge = Some((new_id, window[0], window[1])),
                                Some((rank, ..)) if new_id < rank => best_merge = Some((new_id, window[0], window[1])),
                                _ => {}
                            }
                        }
                    }
                    if let Some((new_id, p0, p1)) = best_merge {
                        let mut next_ids = Vec::with_capacity(ids.len());
                        let mut i = 0;
                        while i < ids.len() {
                            if i < ids.len() - 1 && ids[i] == p0 && ids[i+1] == p1 {
                                next_ids.push(new_id); i += 2;
                            } else {
                                next_ids.push(ids[i]); i += 1;
                            }
                        }
                        ids = next_ids;
                    } else { break; }
                }
                ids
            }).collect()
    }
}