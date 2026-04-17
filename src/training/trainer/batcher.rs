// src/batcher.rs
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, Data, Shape};

#[derive(Clone)]
pub struct QuantCBBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> QuantCBBatcher<B> {
    pub fn new(device: B::Device) -> Self { Self { device } }
}

#[derive(Debug, Clone)]
pub struct QuantCBBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<Vec<usize>, QuantCBBatch<B>> for QuantCBBatcher<B> {
    fn batch(&self, items: Vec<Vec<usize>>) -> QuantCBBatch<B> {
        // 1. Defensive Check: Ensure we actually have data to batch
        if items.is_empty() || items[0].len() < 2 {
            panic!("Batcher received empty or too-short sequences. Check your TokenStreamer logic.");
        }

        let batch_size = items.len();
        let seq_len = items[0].len() - 1;
        
        // 2. Pre-allocate with i64 to match Burn's native IntElem
        let mut inputs_flat = Vec::with_capacity(batch_size * seq_len);
        let mut targets_flat = Vec::with_capacity(batch_size * seq_len);

        for item in items {
            // Ensure every item in the batch has the same length to avoid Shape mismatches
            let current_seq_len = item.len().saturating_sub(1);
            let actual_len = std::cmp::min(seq_len, current_seq_len);

            for i in 0..actual_len {
                inputs_flat.push(item[i] as i64);
                targets_flat.push(item[i + 1] as i64);
            }
        }
        
        // 3. Construct Tensors
        // Using .convert() here is correct to ensure the data matches the Backend's precision
        let inputs_data = Data::new(inputs_flat, Shape::new([batch_size, seq_len])).convert();
        let inputs = Tensor::<B, 2, Int>::from_data(inputs_data, &self.device);
        
        let targets_data = Data::new(targets_flat, Shape::new([batch_size, seq_len])).convert();
        let targets = Tensor::<B, 2, Int>::from_data(targets_data, &self.device);
        
        QuantCBBatch { inputs, targets }
    }
}