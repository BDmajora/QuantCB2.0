use std::collections::{HashMap, BinaryHeap};
use rayon::prelude::*;
use crate::bpe_types::{MergeCandidate, WordList};
use crate::tokenizer::BPETokenizer;

pub struct BPETrainer;

impl BPETrainer {
    pub fn train(tokenizer: &mut BPETokenizer, text: &str, target_vocab_size: usize) {
        let current_base = 256 + tokenizer.special_tokens.len() as u32;
        if (target_vocab_size as u32) <= current_base { return; }

        println!("--- Preparing Tokenizer Data Structures ---");
        let words: HashMap<&str, usize> = text.par_split_whitespace()
            .fold(HashMap::new, |mut acc, word| {
                *acc.entry(word).or_insert(0) += 1;
                acc
            })
            .reduce(HashMap::new, |mut a, b| {
                for (k, v) in b { *a.entry(k).or_insert(0) += v; }
                a
            });

        let mut pair_counts: HashMap<(u32, u32), i32> = HashMap::new();
        let mut word_structures: Vec<WordList> = Vec::new();

        for (word_str, freq) in words {
            let bytes: Vec<u32> = word_str.bytes().map(|b| b as u32).collect();
            let n = bytes.len();
            if n < 1 { continue; }

            let mut prev = vec![-1; n];
            let mut next = vec![-1; n];
            for i in 0..n {
                if i > 0 { prev[i] = (i - 1) as i32; }
                if i < n - 1 { 
                    next[i] = (i + 1) as i32; 
                    *pair_counts.entry((bytes[i], bytes[i+1])).or_insert(0) += freq as i32;
                }
            }
            word_structures.push(WordList { ids: bytes, prev, next, freq: freq as i32 });
        }

        let mut pq: BinaryHeap<MergeCandidate> = pair_counts.iter()
            .map(|(&pair, &freq)| MergeCandidate { freq, pair })
            .collect();

        let mut current_id = current_base;
        let num_merges = target_vocab_size as u32 - current_base;

        println!("--- Learning BPE Merges (PQ Loop) ---");
        for _ in 0..num_merges {
            let mut best_pair = None;
            while let Some(cand) = pq.pop() {
                if let Some(&count) = pair_counts.get(&cand.pair) {
                    if count == cand.freq && count > 0 {
                        best_pair = Some(cand.pair);
                        break;
                    }
                }
            }

            let (p0, p1) = match best_pair { Some(p) => p, None => break };
            tokenizer.merges.insert((p0, p1), current_id);
            
            let mut new_bytes = tokenizer.vocab[&p0].clone();
            new_bytes.extend(&tokenizer.vocab[&p1]);
            tokenizer.vocab.insert(current_id, new_bytes);

            for word in &mut word_structures {
                let mut i = 0;
                while i < word.ids.len() {
                    if word.ids[i] == p0 && word.next[i] != -1 {
                        let j = word.next[i] as usize;
                        if word.ids[j] == p1 {
                            if word.prev[i] != -1 {
                                let h = word.prev[i] as usize;
                                *pair_counts.entry((word.ids[h], p0)).or_insert(0) -= word.freq;
                            }
                            if word.next[j] != -1 {
                                let k = word.next[j] as usize;
                                *pair_counts.entry((p1, word.ids[k])).or_insert(0) -= word.freq;
                            }

                            word.ids[i] = current_id;
                            let next_next = word.next[j];
                            word.next[i] = next_next;
                            if next_next != -1 { word.prev[next_next as usize] = i as i32; }
                            
                            if word.prev[i] != -1 {
                                let h = word.prev[i] as usize;
                                let p = (word.ids[h], current_id);
                                let count = pair_counts.entry(p).or_insert(0);
                                *count += word.freq;
                                pq.push(MergeCandidate { freq: *count, pair: p });
                            }
                            if word.next[i] != -1 {
                                let k = word.next[i] as usize;
                                let p = (current_id, word.ids[k]);
                                let count = pair_counts.entry(p).or_insert(0);
                                *count += word.freq;
                                pq.push(MergeCandidate { freq: *count, pair: p });
                            }
                            i = if next_next == -1 { word.ids.len() } else { next_next as usize };
                            continue;
                        }
                    }
                    i += 1;
                }
            }
            pair_counts.remove(&(p0, p1));
            current_id += 1;
        }
    }
}