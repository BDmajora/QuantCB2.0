pub struct TokenStreamer {
    buffer: Vec<usize>,
    batch_size: usize,
    seq_len: usize,
}

impl TokenStreamer {
    pub fn new(batch_size: usize, seq_len: usize) -> Self {
        Self { buffer: Vec::new(), batch_size, seq_len }
    }

    pub fn push(&mut self, tokens: Vec<usize>) {
        self.buffer.extend(tokens);
    }

    pub fn get_next_batch(&mut self) -> Option<Vec<Vec<usize>>> {
        let required = self.batch_size * (self.seq_len + 1);
        if self.buffer.len() < required {
            return None;
        }

        let mut batch = Vec::with_capacity(self.batch_size);
        for i in 0..self.batch_size {
            let start = i * (self.seq_len + 1);
            batch.push(self.buffer[start..start + self.seq_len + 1].to_vec());
        }

        self.buffer.drain(0..required);
        Some(batch)
    }
}