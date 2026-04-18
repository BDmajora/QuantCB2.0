use std::time::Instant;
use burn::optim::{Optimizer, AdamConfig};
use burn::grad_clipping::GradientClippingConfig;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::ElementConversion; 
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher; 
use burn::module::AutodiffModule; 
use rand::Rng;

// Internal Imports
use crate::training::trainer::trainer_config::{
    TrainingConfig, TAG_TRUTH, TAG_HALLUCINATE, TAG_SHAKESPEARE, TAG_WIKI
}; 
use crate::model::config::QuantCBConfig;
use crate::model::QuantCB; 
use crate::training::trainer::batcher::{QuantCBBatch, QuantCBBatcher};
use crate::training::tokenizer::BLTPatcher; 
use crate::training::data::training_data::VolatileDataPipeline;
use crate::generator::TextGenerator; 
use crate::training::data::learning::DynamicScheduler;

// New Logic Extractions
use super::mode::SampleMode;
use super::checkpoint::CheckpointManager;
use super::streamer::TokenStreamer;

pub struct QuantCBTrainer<B: AutodiffBackend, O: Optimizer<QuantCB<B>, B>> {
    pub model: QuantCB<B>,
    pub optimizer: O,
    pub lr: f64,
    pub entropy_reg_weight: f32, 
    pub mtp_loss_weight: f32,
}

impl<B: AutodiffBackend, O: Optimizer<QuantCB<B>, B>> QuantCBTrainer<B, O> {
    pub fn new(model: QuantCB<B>, optimizer: O, lr: f64, entropy_reg_weight: f32, mtp_loss_weight: f32) -> Self {
        Self { model, optimizer, lr, entropy_reg_weight, mtp_loss_weight }
    }

    pub fn train_step(
        &mut self, 
        batch: QuantCBBatch<B>, 
        current_loop_depth: usize, 
        temperature: f32
    ) -> f32 {
        let (main_logits, _, mtp_loss, _, _, aux_loss, _) = 
            self.model.forward_mtp(batch.inputs, batch.targets.clone(), None, current_loop_depth);

        let [batch_size, seq_len, vocab_size] = main_logits.dims();
        let num_elements = (batch_size * seq_len) as f32;
        
        let stabilized_logits = main_logits.div_scalar(temperature);
        
        let loss_fn = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&stabilized_logits.device());
        
        let logits_flat = stabilized_logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);
        
        let base_loss = loss_fn.forward(logits_flat, targets_flat);
        
        let normalized_mtp = mtp_loss.div_scalar(num_elements);
        let normalized_aux = aux_loss.div_scalar(num_elements);

        let total_loss = base_loss
            + normalized_mtp.mul_scalar(self.mtp_loss_weight) 
            + normalized_aux.mul_scalar(self.entropy_reg_weight);
        
        let grads = total_loss.backward();
        
        self.model = self.optimizer.step(
            self.lr, 
            self.model.clone(), 
            burn::optim::GradientsParams::from_grads(grads, &self.model)
        );
        
        total_loss.into_scalar().elem::<f32>()
    }
}

pub async fn run() {
    type TrainBackend = Autodiff<Wgpu>; 
    let device: <TrainBackend as Backend>::Device = Default::default();
    
    let optim_config = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));
    
    let mut config = TrainingConfig::new(QuantCBConfig::new(), optim_config);
    let mut pipeline = VolatileDataPipeline::new();
    let checkpoints = CheckpointManager::new("./modeloutputs");

    let _raw_shakespeare = pipeline.load_shakespeare(&config).await;
    
    // --- BLT ARCHITECTURE INITIALIZATION ---
    let special_tags = vec![TAG_TRUTH, TAG_HALLUCINATE, TAG_SHAKESPEARE, TAG_WIKI];
    
    // Initialize patcher on the Inner Backend (Wgpu)
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

    let mut current_loop_depth = 1;
    let mut current_temp = config.min_temp; 

    println!("\n--- Launching QuantCB 2.0 Optimized Loop ---");

    loop {
        if step >= config.max_iterations { break; }

        let text = if rng.gen_bool(0.15) { 
            pipeline.get_random_shakespeare() 
        } else { 
            pipeline.get_next_wiki_text().await 
        };
        
        // --- BLT PATCHER INTEGRATION ---
        let dummy_entropies = vec![0.0; text.len()];
        let patches = patcher.patch(&text, &dummy_entropies);
        
        let patched_bytes: Vec<usize> = patches.into_iter()
            .flat_map(|patch| patch.raw_bytes.into_iter().map(|b| b as usize))
            .collect();
            
        streamer.push(patched_bytes);

        if let Some(batch_items) = streamer.get_next_batch() {
            let batch = batcher.batch(batch_items);
            
            // --- BLT LOCAL ENCODER INTEGRATION ---
            // FIX: Use .inner() to convert Tensor<Autodiff<Wgpu>> to Tensor<Wgpu>
            // to match the patcher's backend and silence the E0308 error.
            let _local_features = patcher.forward(batch.inputs.clone().inner());

            let loss = trainer.train_step(
                batch, 
                current_loop_depth, 
                current_temp
            );
            
            let state = scheduler.step(loss);
            trainer.lr = state.lr;
            trainer.mtp_loss_weight = state.mtp_weight;
            current_loop_depth = state.loop_depth;
            current_temp = state.temperature; 
            
            if step % 10 == 0 {
                let elapsed = t0.elapsed().as_secs_f64().max(0.001);
                let tps = (config.batch_size * config.seq_len * 10) as f64 / elapsed;
                
                println!(
                    "Step {:4} | Loss: {:.4} | LR: {:.2e} | Temp: {:.1} | Depth: {} | MTP: {:.3} | Wakes: {} | TPS: {:.0}", 
                    step, loss, trainer.lr, current_temp, current_loop_depth, trainer.mtp_loss_weight, scheduler.wake_ups, tps
                );
                t0 = Instant::now();
            }

            if step > 0 && step % 100 == 0 { 
                run_sample_block(step, &trainer.model, &patcher, &device); 
            }

            if step > 0 && step % 500 == 0 {
                println!("--- Saving Checkpoint [Step {}] ---", step);
                checkpoints.save(
                    &trainer.model, 
                    &trainer.optimizer, 
                    step, 
                    trainer.lr, 
                    trainer.mtp_loss_weight, 
                    scheduler.wake_ups
                );
            }
            step += 1;
        }
    }
}

fn run_sample_block<B: AutodiffBackend>(
    step: usize, 
    model: &QuantCB<B>, 
    patcher: &BLTPatcher<B::InnerBackend>,
    device: &B::Device
) {
    let mode = SampleMode::get_for_step(step, 100);
    let prompt = mode.build_prompt(TAG_TRUTH);
    
    let valid_model = model.clone().valid();
    
    let output = TextGenerator::generate(&valid_model, patcher, device, &prompt, 60, 0.8, 1.2);
    
    println!("\n[Step {} - {}]\n>> {}\n", step, mode.name, output);
}