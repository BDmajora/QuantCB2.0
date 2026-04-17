use std::fs;
use std::path::Path;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn::module::Module; 
use burn::optim::Optimizer;
use crate::model::model::QuantCB;
use crate::training::trainer::trainer_config::TrainingConfig;
// Add serde for robust state saving
use serde::{Serialize, Deserialize}; 

#[derive(Serialize, Deserialize)]
pub struct TrainingState {
    pub step: usize,
    pub lr: f64,
    pub mtp_weight: f32,
    pub wake_ups: u32,
    // Add bit-specific metadata here later if needed
}

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
        let model = config.model.init::<B>(device);
        let optim = config.optimizer.clone().init();
        
        let model_path = format!("{}/checkpoint_model", self.dir);
        let optim_path = format!("{}/checkpoint_optim", self.dir);

        // Check if the binary file exists before trying to load
        if Path::new(&format!("{}.bin", model_path)).exists() {
            println!("Restoring model and optimizer from {}", self.dir);
            let model_record = self.recorder.load(model_path.into(), device)
                .expect("Failed to load model record");
            let optim_record = self.recorder.load(optim_path.into(), device)
                .expect("Failed to load optimizer record");
            
            return (model.load_record(model_record), optim.load_record(optim_record));
        }

        (model, optim)
    }

    pub fn save<B: AutodiffBackend, O: Optimizer<QuantCB<B>, B>>(
        &self, 
        model: &QuantCB<B>, 
        optimizer: &O, 
        step: usize, 
        lr: f64,
        mtp_weight: f32, 
        wake_ups: u32
    ) {
        // PREVENTION: Don't save at step 0 to avoid locking up during initialization
        if step == 0 { return; }

        let model_path = format!("{}/checkpoint_model", self.dir);
        let optim_path = format!("{}/checkpoint_optim", self.dir);
        let state_path = format!("{}/checkpoint_state.json", self.dir);
        
        // Record weights
        let _ = self.recorder.record(model.clone().into_record(), model_path.into());
        let _ = self.recorder.record(optimizer.to_record(), optim_path.into());
        
        // Robust State Saving
        let state = TrainingState { step, lr, mtp_weight, wake_ups };
        if let Ok(json) = serde_json::to_string(&state) {
            let _ = fs::write(state_path, json);
        }
    }

    pub fn load_state(&self, default_lr: f64, default_mtp: f32) -> (usize, f64, f32, u32) {
        let state_path = format!("{}/checkpoint_state.json", self.dir);
        
        if let Ok(content) = fs::read_to_string(state_path) {
            if let Ok(state) = serde_json::from_str::<TrainingState>(&content) {
                return (state.step, state.lr, state.mtp_weight, state.wake_ups);
            }
        }
        
        // Fallback for old .txt format or first run
        (0, default_lr, default_mtp, 0)
    }
}