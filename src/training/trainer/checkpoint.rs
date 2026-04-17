use std::fs;
use std::path::Path;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::module::Module; 
use burn::optim::Optimizer;
use crate::model::model::QuantCB;
use crate::training::trainer::trainer_config::TrainingConfig;

pub struct CheckpointManager {
    pub dir: String,
    recorder: BinFileRecorder<FullPrecisionSettings>,
}

impl CheckpointManager {
    pub fn new(dir: &str) -> Self {
        fs::create_dir_all(dir).ok();
        Self { 
            dir: dir.to_string(), 
            recorder: BinFileRecorder::default() 
        }
    }

    pub fn load_or_init<B: AutodiffBackend>(
        &self, 
        config: &TrainingConfig, 
        device: &B::Device
    ) -> (QuantCB<B>, impl Optimizer<QuantCB<B>, B>) {
        let mut model = config.model.init::<B>(device);
        let mut optim = config.optimizer.clone().init();
        
        let model_path = format!("{}/checkpoint_model", self.dir);
        let optim_path = format!("{}/checkpoint_optim", self.dir);

        if Path::new(&format!("{}.bin", model_path)).exists() {
            if let Ok(record) = self.recorder.load(model_path.into(), device) {
                model = model.load_record(record);
            }
            if let Ok(record) = self.recorder.load(optim_path.into(), device) {
                optim = optim.load_record(record);
            }
        }
        (model, optim)
    }

    // SIGNATURE UPDATED: Added mtp_weight
    pub fn save<B: AutodiffBackend, O: Optimizer<QuantCB<B>, B>>(
        &self, 
        model: &QuantCB<B>, 
        optimizer: &O, 
        step: usize, 
        lr: f64,
        mtp_weight: f32, 
        wake_ups: u32
    ) {
        let model_path = format!("{}/checkpoint_model", self.dir);
        let optim_path = format!("{}/checkpoint_optim", self.dir);
        let state_path = format!("{}/checkpoint_state.txt", self.dir);
        
        let _ = self.recorder.record(model.clone().into_record(), model_path.into());
        let _ = self.recorder.record(optimizer.to_record(), optim_path.into());
        
        // Now writing 4 values: step, lr, mtp, wakes
        let _ = fs::write(&state_path, format!("{},{},{},{}", step, lr, mtp_weight, wake_ups));
    }

    // SIGNATURE UPDATED: Returns (step, lr, mtp, wakes)
    pub fn load_state(&self, default_lr: f64, default_mtp: f32) -> (usize, f64, f32, u32) {
        let state_path = format!("{}/checkpoint_state.txt", self.dir);
        if let Ok(content) = fs::read_to_string(state_path) {
            let parts: Vec<&str> = content.trim().split(',').collect();
            if parts.len() >= 4 {
                let step = parts[0].parse().unwrap_or(0);
                let lr = parts[1].parse().unwrap_or(default_lr);
                let mtp = parts[2].parse().unwrap_or(default_mtp);
                let wake_ups = parts[3].parse().unwrap_or(0);
                return (step, lr, mtp, wake_ups);
            }
        }
        (0, default_lr, default_mtp, 0)
    }
}