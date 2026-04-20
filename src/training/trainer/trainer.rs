use std::time::Instant;
use burn::optim::AdamConfig;
use burn::grad_clipping::GradientClippingConfig;
use burn::tensor::backend::Backend;
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher; 

use crate::training::trainer::trainer_config::{
    TrainingConfig, TAG_TRUTH, TAG_HALLUCINATE, TAG_SHAKESPEARE, TAG_WIKI
}; 
use crate::model::config::QuantCBConfig;
use crate::training::trainer::batcher::QuantCBBatcher;
use crate::training::tokenizer::BLTPatcher; 
use crate::training::data::training_data::VolatileDataPipeline;
use crate::training::data::learning::DynamicScheduler;

use super::checkpoint::CheckpointManager;
use super::streamer::TokenStreamer;
use super::core::QuantCBTrainer;

use super::data_stream::feed_streamer;
use super::metrics::log_and_save;

pub async fn run() {
    type TrainBackend = Autodiff<Wgpu>; 
    let device: <TrainBackend as Backend>::Device = Default::default();
    
    let optim_config = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));
    
    let mut config = TrainingConfig::new(QuantCBConfig::new(), optim_config);
    let mut pipeline = VolatileDataPipeline::new();
    let checkpoints = CheckpointManager::new("./modeloutputs");

    let _raw_shakespeare = pipeline.load_shakespeare(&config).await;
    
    let special_tags = vec![TAG_TRUTH, TAG_HALLUCINATE, TAG_SHAKESPEARE, TAG_WIKI];
    // Initialize the patcher with the starting threshold from config
    let patcher = BLTPatcher::<Wgpu>::new(config.entropy_threshold, special_tags, &device);
    
    config.model = config.model.with_vocab_size(config.byte_vocab_size); 

    let (model, optimizer) = checkpoints.load_or_init::<TrainBackend>(&config, &device);
    let (mut step, resumed_lr, resumed_mtp, resumed_wake_ups) = 
        checkpoints.load_state(config.learning_rate, config.mtp_loss_weight);

    let mut trainer = QuantCBTrainer::new(
        model, 
        optimizer, 
        resumed_lr, 
        config.entropy_reg_weight, 
        resumed_mtp
    );
    
    let mut scheduler = DynamicScheduler::new(config.learning_rate, config.mtp_loss_weight, config.model.loop_depth);
    
    if step > 0 { 
        scheduler.current_lr = resumed_lr; 
        scheduler.current_mtp_weight = resumed_mtp;
        scheduler.wake_ups = resumed_wake_ups; 
    }

    let batcher = QuantCBBatcher::<TrainBackend>::new(device.clone());
    let mut streamer = TokenStreamer::new(config.batch_size, config.seq_len);
    let mut rng = rand::thread_rng();
    
    pipeline.start_huggingface_chunker(&config);
    let mut t0 = Instant::now();

    // EMA for tracking patch efficiency
    let mut rolling_avg_patch = 4.0; 
    let mut current_loop_depth = 1;
    let mut current_temp = config.min_temp; 
    
    // --- FIX: Initialize the dynamic entropy threshold variable ---
    let mut current_entropy_threshold = config.entropy_threshold;

    println!("\n Launching QuantCB 2.0 Optimized Loop ");

    loop {
        if step >= config.max_iterations { break; }

        // --- FIX: Pass current_entropy_threshold to the data streamer ---
        // This allows the patcher to adapt its segmentation based on scheduler feedback
        let current_patch_len = feed_streamer(
            &mut rng, 
            &mut pipeline, 
            &patcher, 
            &mut streamer,
            current_entropy_threshold
        ).await;
        
        rolling_avg_patch = 0.95 * rolling_avg_patch + 0.05 * current_patch_len;

        if let Some(batch_items) = streamer.get_next_batch() {
            let batch = batcher.batch(batch_items);

            let loss = trainer.train_step(
                batch, 
                current_loop_depth, 
                current_temp
            );
            
            // Scheduler calculates new hyperparameters and the updated threshold
            let state = scheduler.step(loss, rolling_avg_patch);
            
            // Sync trainer and loop variables with new scheduler state
            trainer.lr = state.lr;
            trainer.mtp_loss_weight = state.mtp_weight;
            current_loop_depth = state.loop_depth;
            current_temp = state.temperature; 
            
            // --- FIX: Resolve E0609 by updating threshold from scheduler state ---
            current_entropy_threshold = state.entropy_threshold; 
            
            log_and_save(
                step, 
                loss, 
                rolling_avg_patch,
                current_entropy_threshold, // Log the active threshold
                trainer.lr, 
                current_temp, 
                current_loop_depth, 
                trainer.mtp_loss_weight, 
                scheduler.wake_ups, 
                config.batch_size, 
                config.seq_len, 
                &mut t0, 
                &trainer.model, 
                &trainer.optimizer, 
                &patcher, 
                &device, 
                &checkpoints
            );

            step += 1;
        }
    }
}