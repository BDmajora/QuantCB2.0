use rand::Rng;
use burn::tensor::backend::Backend;
use crate::training::tokenizer::BLTPatcher;
use crate::training::data::training_data::VolatileDataPipeline;
use super::streamer::TokenStreamer;

/// Feeds raw text data into the token streamer by converting it into patches.
/// 
/// This function now accepts an `entropy_threshold` which allows the training loop
/// to dynamically adjust patch granularity on the fly.
pub async fn feed_streamer<B: Backend>(
    rng: &mut impl Rng,
    pipeline: &mut VolatileDataPipeline,
    patcher: &BLTPatcher<B>,
    streamer: &mut TokenStreamer,
    entropy_threshold: f32, 
) -> f32 {
    // 1. Data Selection: 15% Shakespeare for flavor, 85% Wiki for knowledge
    let text = if rng.gen_bool(0.15) { 
        pipeline.get_random_shakespeare() 
    } else { 
        pipeline.get_next_wiki_text().await 
    };
    
    // 2. Prepare entropies: 
    // In a full implementation, these might come from a dedicated entropy model.
    // For this bridge, we provide a slice matching the text length.
    let dummy_entropies = vec![0.0; text.len()];
    
    // 3. Patching: Call the newly implemented 'patch_with_threshold' method.
    // This resolves the previous E0599 error.
    let patches = patcher.patch_with_threshold(&text, &dummy_entropies, entropy_threshold);
    
    // 4. Metrics: Calculate average patch length for the scheduler's feedback loop
    let patch_count = patches.len();
    let total_bytes: usize = patches.iter().map(|p| p.raw_bytes.len()).sum();
    let avg_len = if patch_count > 0 { 
        total_bytes as f32 / patch_count as f32 
    } else { 
        0.0 
    };

    // 5. Processing: Flatten the patches into a raw byte stream for the TokenStreamer
    let patched_bytes: Vec<usize> = patches.into_iter()
        .flat_map(|patch| patch.raw_bytes.into_iter().map(|b| b as usize))
        .collect();
        
    // 6. Finalize: Push to the streamer and return avg_len to update rolling averages
    streamer.push(patched_bytes);
    
    avg_len
}