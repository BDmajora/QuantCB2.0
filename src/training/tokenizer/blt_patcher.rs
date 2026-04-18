use crate::training::tokenizer::entropy_patcher::{EntropyPatcher, BytePatch};
use crate::training::tokenizer::local_encoder::LocalEncoder;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, Int};

pub struct BLTPatcher<B: Backend> {
    pub patcher: EntropyPatcher,
    pub local_encoder: LocalEncoder<B>,
}

impl<B: Backend> BLTPatcher<B> {
    pub fn new(threshold: f32, tags: Vec<&'static str>, device: &B::Device) -> Self {
        Self {
            patcher: EntropyPatcher::new(threshold, 16, tags), 
            local_encoder: LocalEncoder::new(device),
        }
    }

    pub fn patch(&self, text: &str, byte_entropies: &[f32]) -> Vec<BytePatch> {
        // Reads 'self.patcher' and calls its method
        self.patcher.segment_into_patches(text, byte_entropies)
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        // Reads 'self.local_encoder' and calls its method
        self.local_encoder.forward(input)
    }
}