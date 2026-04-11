// src/dataset.rs
use burn::data::dataset::Dataset;

pub struct TextDataset {
    data: Vec<usize>,
    seq_len: usize,
}

impl TextDataset {
    pub fn new(encoded_data: Vec<usize>, seq_len: usize) -> Self {
        Self { data: encoded_data, seq_len }
    }
}

impl Dataset<Vec<usize>> for TextDataset {
    fn get(&self, index: usize) -> Option<Vec<usize>> {
        if index + self.seq_len + 1 > self.data.len() { return None; }
        Some(self.data[index..index + self.seq_len + 1].to_vec())
    }
    fn len(&self) -> usize {
        if self.data.len() <= self.seq_len + 1 { 0 } else { self.data.len() - self.seq_len - 1 }
    }
}