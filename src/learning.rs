use std::f32;

pub struct SchedulerState {
    pub lr: f64,
    pub mtp_weight: f32,
}

pub struct DynamicScheduler {
    pub current_lr: f64,
    pub current_mtp_weight: f32,
    pub base_lr: f64,
    pub min_lr: f64,
    pub max_mtp_weight: f32,
    pub wake_ups: u32,
    
    short_ema: f32,
    long_ema: f32,
    
    cooldown: usize,
    active_cooldown: usize,
    steps_since_action: usize,
    total_steps: usize,
    
    // New: Explicit state for post-slap recovery
    is_in_recovery: bool,
}

impl DynamicScheduler {
    pub fn new(initial_lr: f64, initial_mtp: f32) -> Self {
        Self {
            current_lr: initial_lr,
            current_mtp_weight: initial_mtp,
            base_lr: initial_lr,
            min_lr: 1.0e-6,
            max_mtp_weight: 0.50,
            wake_ups: 0,
            
            short_ema: 0.0,
            long_ema: 0.0,
            
            cooldown: 50,
            active_cooldown: 50,
            steps_since_action: 0,
            total_steps: 0,
            is_in_recovery: false,
        }
    }

    pub fn step(&mut self, loss: f32) -> SchedulerState {
        self.total_steps += 1;
        self.steps_since_action += 1;

        // Check if we are in the 500-step post-slap grace period
        if self.is_in_recovery {
            if self.steps_since_action >= 500 {
                self.is_in_recovery = false;
                self.steps_since_action = 0;
            }
            return SchedulerState { lr: self.current_lr, mtp_weight: self.current_mtp_weight };
        }

        // Initialize EMAs
        if self.total_steps == 1 {
            self.short_ema = loss;
            self.long_ema = loss;
            return SchedulerState { lr: self.current_lr, mtp_weight: self.current_mtp_weight };
        }

        // Standard EMA update
        let alpha_short = 2.0 / (15.0 + 1.0);
        let alpha_long = 2.0 / (100.0 + 1.0);
        self.short_ema = alpha_short * loss + (1.0 - alpha_short) * self.short_ema;
        self.long_ema = alpha_long * loss + (1.0 - alpha_long) * self.long_ema;

        if self.total_steps < 100 || self.steps_since_action < self.active_cooldown {
            return SchedulerState { lr: self.current_lr, mtp_weight: self.current_mtp_weight };
        }

        let trend_ratio = self.short_ema / self.long_ema;

        if trend_ratio > 1.05 {
            self.current_lr *= 0.5;
            self.current_mtp_weight = (self.current_mtp_weight * 0.7).max(0.05);
            self.steps_since_action = 0;
            self.active_cooldown = self.cooldown;
            
            println!("Divergence Detected: Cutting LR to {:.2e} | Dropping MTP to {:.3}", self.current_lr, self.current_mtp_weight);
        }
        else if trend_ratio > 0.995 {
            if self.current_lr <= self.min_lr * 1.1 {
                self.wake_ups += 1;
                self.current_lr = self.base_lr * (0.8_f64).powi(self.wake_ups as i32).max(5e-6);
                self.current_mtp_weight = (0.15 + (self.wake_ups as f32 * 0.05)).min(self.max_mtp_weight);
                
                self.short_ema = loss;
                self.long_ema = loss;

                // Enter Recovery Mode
                self.is_in_recovery = true;
                self.steps_since_action = 0;

                println!("WAKE UP SLAP #{}: Restarting LR to {:.2e} | MTP to {:.3} | Recovery mode active for 500 steps", 
                    self.wake_ups, self.current_lr, self.current_mtp_weight);
            } else {
                self.current_lr *= 0.85;
                self.current_mtp_weight = (self.current_mtp_weight * 0.9).max(0.05);
                self.steps_since_action = 0;
                self.active_cooldown = self.cooldown;

                println!("Plateau Detected: Decaying LR to {:.2e} | Easing MTP to {:.3}", self.current_lr, self.current_mtp_weight);
            }
        }
        else if trend_ratio < 0.97 {
            self.current_mtp_weight = (self.current_mtp_weight * 1.02).min(self.max_mtp_weight);
            self.steps_since_action = 0;
            self.active_cooldown = self.cooldown;
        }

        SchedulerState { lr: self.current_lr, mtp_weight: self.current_mtp_weight }
    }
}