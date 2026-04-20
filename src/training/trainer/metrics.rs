use std::time::Instant;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::optim::Optimizer;
use crate::model::QuantCB;
use crate::training::tokenizer::BLTPatcher;
use super::checkpoint::CheckpointManager;
use super::sampler::run_sample_block;

pub fn log_and_save<B: AutodiffBackend, O: Optimizer<QuantCB<B>, B>>(
    step: usize,
    loss: f32,
    avg_patch_len: f32,
    entropy_threshold: f32, // Added to track plateau-fighting logic
    lr: f64,
    current_temp: f32,
    current_loop_depth: usize,
    mtp_loss_weight: f32,
    wake_ups: u32, 
    batch_size: usize,
    seq_len: usize,
    t0: &mut Instant,
    model: &QuantCB<B>,
    optimizer: &O,
    patcher: &BLTPatcher<B::InnerBackend>,
    device: &<B as Backend>::Device,
    checkpoints: &CheckpointManager,
) {
    // 1. Console Logging
    if step % 10 == 0 {
        let elapsed = t0.elapsed().as_secs_f64().max(0.001);
        // TPS accounts for the sequence length and batch size
        let tps = (batch_size * seq_len * 10) as f64 / elapsed;
        
        println!(
            "Step {:4} | Loss: {:.4} | AvgP: {:.2} | Entr: {:.3} | LR: {:.2e} | Temp: {:.1} | Depth: {} | MTP: {:.3} | Wakes: {} | TPS: {:.0}", 
            step, 
            loss, 
            avg_patch_len, 
            entropy_threshold, // Injected into the log
            lr, 
            current_temp, 
            current_loop_depth, 
            mtp_loss_weight, 
            wake_ups, 
            tps
        );
        *t0 = Instant::now();
    }

    // 2. Sampling (Inference check)
    if step > 0 && step % 100 == 0 { 
        run_sample_block(step, model, patcher, device); 
    }

    // 3. Checkpointing
    if step > 0 && step % 500 == 0 {
        println!("--- Saving Checkpoint [Step {}] ---", step);
        checkpoints.save(
            model, 
            optimizer, 
            step, 
            lr, 
            mtp_loss_weight, 
            wake_ups
        );
    }
}