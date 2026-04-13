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
        let batch_size = items.len();
        let seq_len = items[0].len() - 1;
        let mut inputs_flat = Vec::with_capacity(batch_size * seq_len);
        let mut targets_flat = Vec::with_capacity(batch_size * seq_len);

        for item in items {
            for i in 0..seq_len {
                inputs_flat.push(item[i] as i32);
                targets_flat.push(item[i + 1] as i32);
            }
        }
        
        // Wrapped dimensions in Shape::new() and added .convert() 
        // to securely cast generic int types to the backend's explicit IntElem
        let inputs_data = Data::new(inputs_flat, Shape::new([batch_size, seq_len])).convert();
        let inputs = Tensor::<B, 2, Int>::from_data(inputs_data, &self.device);
        
        let targets_data = Data::new(targets_flat, Shape::new([batch_size, seq_len])).convert();
        let targets = Tensor::<B, 2, Int>::from_data(targets_data, &self.device);
        
        QuantCBBatch { inputs, targets }
    }
}