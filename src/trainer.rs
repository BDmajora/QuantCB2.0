// src/trainer.rs
use std::time::Instant;
use burn::optim::{Optimizer, GradientsParams};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::ElementConversion;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::data::dataloader::DataLoaderBuilder;
use burn::backend::{Autodiff, Wgpu};

use crate::trainer_config::TrainingConfig; 
use crate::model::QuantCB;
use crate::batcher::{QuantCBBatch, QuantCBBatcher};
use crate::tokenizer::BPETokenizer; // Updated Tokenizer
use crate::dataset::TextDataset;
use crate::training_data::{TrainingDataSources, TAG_TRUTH, TAG_SHAKESPEARE};
use crate::generator::TextGenerator; 

pub struct QuantCBTrainer<B: AutodiffBackend, O: Optimizer<QuantCB<B>, B>> {
    pub model: QuantCB<B>,
    pub optimizer: O,
    pub lr: f64,
    pub entropy_reg_weight: f32, 
}

impl<B: AutodiffBackend, O: Optimizer<QuantCB<B>, B>> QuantCBTrainer<B, O> {
    pub fn new(model: QuantCB<B>, optimizer: O, lr: f64, entropy_reg_weight: f32) -> Self {
        Self { model, optimizer, lr, entropy_reg_weight }
    }

    pub fn train_step(&mut self, batch: QuantCBBatch<B>) -> f32 {
        let (main_logits, _, mtp_loss, _, _, aux_loss, _) = 
            self.model.forward_mtp(batch.inputs, batch.targets.clone(), None);

        let [batch_size, seq_len, vocab_size] = main_logits.dims();
        let loss_fn = CrossEntropyLossConfig::new().init(&main_logits.device());
        
        let logits_flat = main_logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);
        let main_loss = loss_fn.forward(logits_flat, targets_flat);

        let total_loss = main_loss 
            + mtp_loss.mul_scalar(0.3) 
            + aux_loss.mul_scalar(self.entropy_reg_weight);
        
        let grads = total_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optimizer.step(self.lr, self.model.clone(), grads_params);

        total_loss.into_data().value[0].elem::<f32>()
    }
}

pub fn run() {
    type TrainBackend = Autodiff<Wgpu>; 
    let device: <TrainBackend as Backend>::Device = Default::default();

    let raw_text = TrainingDataSources::load_tiny_shakespeare();
    
    // Initialize tokenizer with special tokens to bypass byte-splitting
    let special_tags = vec![TAG_TRUTH, TAG_SHAKESPEARE];
    let mut tokenizer = BPETokenizer::new(&special_tags);
    
    // Train the BPE Tokenizer up to an arbitrary 2048 vocab size
    println!("Training BPE Tokenizer...");
    tokenizer.train(&raw_text, 2048);

    let model_config = crate::config::QuantCBConfig::new(
        tokenizer.vocab_size(), 
        256,                    
        8,                      
        4,                      
        8,                      
        2,                      
        512,                    
        0.1,                    
        64,                     
        64,                     
        16,                     
        16,                     
        2,                      
    );

    let config = TrainingConfig::new(
        model_config,
        burn::optim::AdamConfig::new(),
    );

    let dataset = TextDataset::new(tokenizer.encode(&raw_text), config.seq_len);
    let batcher = QuantCBBatcher::<TrainBackend>::new(device.clone());
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(dataset);

    let mut model = config.model.init::<TrainBackend>(&device);
    
    model.loop_depth = config.loop_depth;

    let optimizer = config.optimizer.init();
    let mut trainer = QuantCBTrainer::new(
        model, 
        optimizer, 
        config.learning_rate, 
        config.entropy_reg_weight
    );

    println!("\n--- Launching LoopLM (Ouro) Training Pipeline ---");
    let mut t0 = Instant::now();
    let mut step = 0;

    for batch in dataloader.iter() {
        let loss = trainer.train_step(batch);
        
        if step > 0 && step % 10 == 0 {
            let dt = t0.elapsed().as_secs_f64();
            let tps = (config.batch_size * config.seq_len * 10) as f64 / f64::max(dt, 0.001);
            println!("Step {:4} | Loss: {:.4} | TPS: {:.0}", step, loss, tps);
            t0 = Instant::now();
        }

        if step > 0 && step % 100 == 0 {
            let prompt = format!("{} {} \nFirst Citizen:", TAG_TRUTH, TAG_SHAKESPEARE);
            let generated = TextGenerator::generate(&trainer.model, &tokenizer, &device, &prompt, 60);
            println!("\n[Sample at Step {}]\n>> {}\n", step, generated);
        }

        step += 1;
        if step >= config.max_iterations { break; }
    }
}