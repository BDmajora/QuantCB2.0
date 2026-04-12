use std::time::Instant;
use std::fs; 
use burn::optim::{Optimizer, GradientsParams};
use burn::grad_clipping::GradientClippingConfig;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::ElementConversion; 
use burn::nn::loss::CrossEntropyLossConfig;
use burn::data::dataloader::DataLoaderBuilder;
use burn::backend::{Autodiff, Wgpu};
use burn::module::Module; 
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

use crate::trainer_config::{TrainingConfig, TAG_TRUTH, TAG_HALLUCINATE, TAG_SHAKESPEARE}; 
use crate::config::QuantCBConfig;
use crate::model::QuantCB;
use crate::batcher::{QuantCBBatch, QuantCBBatcher};
use crate::tokenizer::BPETokenizer; 
use crate::dataset::TextDataset;
use crate::training_data::TrainingDataSources;
use crate::generator::TextGenerator; 
use crate::learning::DynamicScheduler;

pub struct QuantCBTrainer<B: AutodiffBackend, O: Optimizer<QuantCB<B>, B>> {
    pub model: QuantCB<B>,
    pub optimizer: O,
    pub lr: f64,
    pub entropy_reg_weight: f32, 
    pub mtp_loss_weight: f32,
}

impl<B: AutodiffBackend, O: Optimizer<QuantCB<B>, B>> QuantCBTrainer<B, O> {
    pub fn new(
        model: QuantCB<B>, 
        optimizer: O, 
        lr: f64, 
        entropy_reg_weight: f32, 
        mtp_loss_weight: f32
    ) -> Self {
        Self { model, optimizer, lr, entropy_reg_weight, mtp_loss_weight }
    }

    pub fn train_step(&mut self, batch: QuantCBBatch<B>) -> f32 {
        // Forward pass with Multi-Token Prediction support
        let (main_logits, _, mtp_loss, _, _, aux_loss, _) = 
            self.model.forward_mtp(batch.inputs, batch.targets.clone(), None);

        let [batch_size, seq_len, vocab_size] = main_logits.dims();
        let loss_fn = CrossEntropyLossConfig::new().init(&main_logits.device());
        
        let logits_flat = main_logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_len]);
        let main_loss = loss_fn.forward(logits_flat, targets_flat);

        // total_loss uses the dynamic weights managed by the scheduler
        let total_loss = main_loss 
            + mtp_loss.mul_scalar(self.mtp_loss_weight) 
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

    let output_dir = "modeloutputs";
    fs::create_dir_all(output_dir).ok();

    let mut config = TrainingConfig::new(
        QuantCBConfig::new(), 
        burn::optim::AdamConfig::new()
    );

    println!("--- Loading Shakespeare Dataset ---");
    let raw_text = TrainingDataSources::load_complete_shakespeare(&config);
    
    let special_tags = vec![TAG_TRUTH, TAG_HALLUCINATE, TAG_SHAKESPEARE];
    let mut tokenizer = BPETokenizer::new(&special_tags);
    
    tokenizer.train(&raw_text, config.tokenizer_vocab_size);

    println!("--- Encoding Dataset ---");
    let encoded_data = tokenizer.encode(&raw_text)
        .into_iter()
        .map(|id| id as usize)
        .collect();
        
    let dataset = TextDataset::new(encoded_data, config.seq_len);
    let batcher = QuantCBBatcher::<TrainBackend>::new(device.clone());
    let dataloader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .build(dataset);

    config.model = config.model
        .with_vocab_size(tokenizer.vocab_size())
        .with_loop_depth(config.loop_depth);

    let model = config.model.init::<TrainBackend>(&device);
    
    let optimizer = config.optimizer
        .with_grad_clipping(Some(GradientClippingConfig::Norm(config.clip_grad_norm as f32)))
        .init();

    // 1. Initialize Trainer
    let mut trainer = QuantCBTrainer::new(
        model, 
        optimizer, 
        config.learning_rate, 
        config.entropy_reg_weight,
        config.mtp_loss_weight,
    );

    // FIX: Pass initial MTP weight as the second argument
    let mut scheduler = DynamicScheduler::new(config.learning_rate, config.mtp_loss_weight);

    println!("\n--- Launching QuantCB 2.0 Training Pipeline ---");
    println!("Architecture: Kimi AttnRes + LoopLM Dynamic Routing enabled.");
    
    let mut t0 = Instant::now();
    let mut step = 0;

    for batch in dataloader.iter() {
        let loss = trainer.train_step(batch);
        
        // FIX: Unpack SchedulerState to update both LR and MTP Weight
        let sched_state = scheduler.step(loss);
        trainer.lr = sched_state.lr;
        trainer.mtp_loss_weight = sched_state.mtp_weight;
        
        if step > 0 && step % 10 == 0 {
            let dt = t0.elapsed().as_secs_f64();
            let tps = (config.batch_size * config.seq_len * 10) as f64 / f64::max(dt, 0.001);
            
            // Added MTP Weight to the logs so you can watch the scheduler "think"
            println!(
                "Step {:4} | Loss: {:.4} | LR: {:.2e} | MTP-W: {:.3} | TPS: {:.0}", 
                step, loss, trainer.lr, trainer.mtp_loss_weight, tps
            );
            t0 = Instant::now();
        }

        if step > 0 && step % 100 == 0 {
            let prompt = format!("{} {} \nShylock:", TAG_TRUTH, TAG_SHAKESPEARE);
            let output = TextGenerator::generate(&trainer.model, &tokenizer, &device, &prompt, 60, 0.8, 1.2);
            println!("\n[Sample at Step {}]\n>> {}\n", step, output);
        }

        if step > 0 && step % 500 == 0 {
            let save_path = format!("{}/checkpoint", output_dir);
            let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
            recorder
                .record(trainer.model.clone().into_record(), save_path.into())
                .expect("Failed to save periodic checkpoint");
        }

        step += 1;
        if step >= config.max_iterations { break; }
    }

    let final_path = format!("{}/final_model", output_dir);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    recorder
        .record(trainer.model.clone().into_record(), final_path.into())
        .expect("Failed to save final model");
        
    println!("Final weights saved to {}. Done!", output_dir);
}