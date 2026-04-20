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

    /// Normal patching using default config
    pub fn patch(&self, text: &str, byte_entropies: &[f32]) -> Vec<BytePatch> {
        self.patcher.segment_into_patches(text, byte_entropies)
    }

    /// Dynamic patching using the scheduler's threshold
    pub fn patch_with_threshold(
        &self, 
        text: &str, 
        byte_entropies: &[f32], 
        threshold: f32
    ) -> Vec<BytePatch> {
        // Now calls the method we just added to EntropyPatcher
        self.patcher.segment_with_threshold(text, byte_entropies, threshold)
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.local_encoder.forward(input)
    }
}