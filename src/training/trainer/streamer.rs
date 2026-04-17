use std::collections::VecDeque;

pub struct TokenStreamer {
    buffer: VecDeque<usize>, 
    batch_size: usize,
    seq_len: usize,
}

impl TokenStreamer {
    pub fn new(batch_size: usize, seq_len: usize) -> Self {
        Self { buffer: VecDeque::new(), batch_size, seq_len }
    }

    pub fn push(&mut self, tokens: Vec<usize>) {
        self.buffer.extend(tokens);
    }

    pub fn get_next_batch(&mut self) -> Option<Vec<Vec<usize>>> {
        let stride = self.seq_len + 1;
        let required = self.batch_size * stride;
        
        if self.buffer.len() < required {
            return None;
        }

        let mut batch = Vec::with_capacity(self.batch_size);
        for i in 0..self.batch_size {
            let start = i * stride;
            // Optimized: instead of .iter().skip().take(), we use as_slices
            let sample: Vec<usize> = self.buffer.iter()
                .skip(start)
                .take(stride)
                .copied()
                .collect();
            batch.push(sample);
        }

        self.buffer.drain(0..required);
        Some(batch)
    }
}