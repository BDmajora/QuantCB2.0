use std::collections::VecDeque;

pub struct DynamicScheduler {
    pub current_lr: f64,
    pub min_lr: f64,
    pub max_lr: f64, // Now used for Recovery!
    
    window_size: usize,
    loss_window: VecDeque<f32>,
    prev_mean_loss: Option<f32>,
    
    jitter_threshold: f32,
    plateau_epsilon: f32,
    decay_factor: f64,
    
    patience: usize,
    steps_since_update: usize,
}

impl DynamicScheduler {
    pub fn new(initial_lr: f64) -> Self {
        Self {
            current_lr: initial_lr,
            min_lr: 1e-6,          
            max_lr: initial_lr * 1.5, // Ceiling for recovery
            
            window_size: 50,       
            loss_window: VecDeque::with_capacity(50),
            prev_mean_loss: None,
            
            jitter_threshold: 0.15, 
            plateau_epsilon: 0.01,  
            decay_factor: 0.5,      
            
            patience: 200,          
            steps_since_update: 0,
        }
    }

    pub fn step(&mut self, loss: f32) -> f64 {
        if self.loss_window.len() == self.window_size {
            self.loss_window.pop_front();
        }
        self.loss_window.push_back(loss);
        self.steps_since_update += 1;

        if self.loss_window.len() < self.window_size || self.steps_since_update < self.patience {
            return self.current_lr;
        }

        let window_len = self.loss_window.len() as f32;
        let sum: f32 = self.loss_window.iter().sum();
        let mean = sum / window_len;
        
        let variance: f32 = self.loss_window.iter()
            .map(|&val| (val - mean).powi(2))
            .sum::<f32>() / window_len;
            
        let std_dev = variance.sqrt();
        let mut lr_changed = false;

        // 1. Stability Check (Jitter Sensor)
        if std_dev > self.jitter_threshold {
            println!(" [Scheduler] High Jitter (σ={:.4}). Decaying LR.", std_dev);
            self.current_lr = (self.current_lr * self.decay_factor).max(self.min_lr);
            lr_changed = true;
        } 
        // 2. Progress & Recovery Check
        else if let Some(prev_mean) = self.prev_mean_loss {
            if mean > prev_mean - self.plateau_epsilon {
                println!(" [Scheduler] Plateau. Mean: {:.4} -> {:.4}. Decaying LR.", prev_mean, mean);
                self.current_lr = (self.current_lr * 0.8).max(self.min_lr); 
                lr_changed = true;
            } 
            // NEW: RECOVERY LOGIC
            // If jitter is very low (1/3 of threshold) and we're under the max, nudge it up
            else if std_dev < (self.jitter_threshold / 3.0) && self.current_lr < self.max_lr {
                let old_lr = self.current_lr;
                self.current_lr = (self.current_lr * 1.05).min(self.max_lr);
                if (self.current_lr - old_lr).abs() > 1e-9 {
                    println!(" [Scheduler] High Stability. Recovering LR: {:.2e}", self.current_lr);
                    // We don't set lr_changed to true here because we don't want to 
                    // flush the window for a tiny 5% nudge.
                }
            }
        }

        if lr_changed {
            self.steps_since_update = 0;
            self.prev_mean_loss = None;
            self.loss_window.clear(); 
        } else if self.steps_since_update % self.window_size == 0 {
            self.prev_mean_loss = Some(mean);
        }

        self.current_lr
    }
}