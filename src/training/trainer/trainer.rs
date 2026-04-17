use std::time::Instant;
use burn::optim::{Optimizer, AdamConfig};
// FIXED: GradientClippingConfig lives in its own module, not inside optim
use burn::grad_clipping::GradientClippingConfig;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::backend::{Autodiff, Wgpu};
use burn::data::dataloader::batcher::Batcher; 
use burn::module::AutodiffModule; // <-- ADDED: Required to unlock .valid()
use rand::Rng;

// Internal Imports
use crate::training::trainer::trainer_config::{
    TrainingConfig, TAG_TRUTH, TAG_HALLUCINATE, TAG_SHAKESPEARE, TAG_WIKI
}; 
use crate::model::config::QuantCBConfig;
use crate::model::QuantCB; 
use crate::training::trainer::batcher::{QuantCBBatch, QuantCBBatcher};
use crate::training::BPETokenizer; 
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

    pub fn train_step(&mut self, batch: QuantCBBatch<B>) -> f32 {
        let (main_logits, _, mtp_loss, _, _, aux_loss, _) = 
            self.model.forward_mtp(batch.inputs, batch.targets.clone(), None);

        let [batch_size, seq_len, vocab_size] = main_logits.dims();
        let loss_fn = burn::nn::loss::CrossEntropyLossConfig::new().init(&main_logits.device());
        
        let logits_flat = main_logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);
        
        let total_loss = loss_fn.forward(logits_flat, targets_flat)
            + mtp_loss.mul_scalar(self.mtp_loss_weight) 
            + aux_loss.mul_scalar(self.entropy_reg_weight);
        
        let grads = total_loss.backward();
        
        self.model = self.optimizer.step(
            self.lr, 
            self.model.clone(), 
            burn::optim::GradientsParams::from_grads(grads, &self.model)
        );
        
        burn::tensor::ElementConversion::elem::<f32>(total_loss.into_data().value[0])
    }
}

pub async fn run() {
    type TrainBackend = Autodiff<Wgpu>; 
    let device: <TrainBackend as Backend>::Device = Default::default();
    
    // Gradient clipping is essential for STE stability
    let optim_config = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));
    
    let mut config = TrainingConfig::new(QuantCBConfig::new(), optim_config);
    let mut pipeline = VolatileDataPipeline::new();
    let checkpoints = CheckpointManager::new("./modeloutputs");

    let raw_shakespeare = pipeline.load_shakespeare(&config).await;
    let special_tags = vec![TAG_TRUTH, TAG_HALLUCINATE, TAG_SHAKESPEARE, TAG_WIKI];
    
    let mut tokenizer = BPETokenizer::new(&special_tags);
    tokenizer.train(&raw_shakespeare, config.tokenizer_vocab_size);
    config.model = config.model.with_vocab_size(tokenizer.vocab_size());

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
    
    let mut scheduler = DynamicScheduler::new(config.learning_rate, config.mtp_loss_weight);
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

    println!("\n--- Launching QuantCB 2.0 Optimized Loop ---");

    loop {
        if step >= config.max_iterations { break; }

        let text = if rng.gen_bool(0.15) { 
            pipeline.get_random_shakespeare() 
        } else { 
            pipeline.get_next_wiki_text().await 
        };
        
        streamer.push(tokenizer.encode(&text).into_iter().map(|id| id as usize).collect());

        if let Some(batch_items) = streamer.get_next_batch() {
            let loss = trainer.train_step(batcher.batch(batch_items));
            let state = scheduler.step(loss);
            
            trainer.lr = state.lr;
            trainer.mtp_loss_weight = state.mtp_weight;
            
            if step % 10 == 0 {
                let elapsed = t0.elapsed().as_secs_f64().max(0.001);
                let tps = (config.batch_size * config.seq_len * 10) as f64 / elapsed;
                
                println!(
                    "Step {:4} | Loss: {:.4} | LR: {:.2e} | MTP: {:.3} | Wakes: {} | TPS: {:.0}", 
                    step, loss, trainer.lr, trainer.mtp_loss_weight, scheduler.wake_ups, tps
                );
                t0 = Instant::now();
            }

            if step > 0 && step % 100 == 0 { 
                run_sample_block(step, &trainer.model, &tokenizer, &device); 
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
    tokenizer: &BPETokenizer, 
    device: &B::Device
) {
    let mode = SampleMode::get_for_step(step, 100);
    let prompt = mode.build_prompt(TAG_TRUTH);
    
    // <-- ADDED: Unpack the model into its non-differentiable form for inference.
    // Burn ensures that B::Device and B::InnerBackend::Device are the exact same type, 
    // so we can safely pass the device reference forward.
    let valid_model = model.clone().valid();

    // Pass the valid_model instead of the autodiff model
    let output = TextGenerator::generate(&valid_model, tokenizer, device, &prompt, 60, 0.8, 1.2);
    
    println!("\n[Step {} - {}]\n>> {}\n", step, mode.name, output);
}