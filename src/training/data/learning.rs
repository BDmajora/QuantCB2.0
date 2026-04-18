use std::f32;

pub struct SchedulerState {
    pub lr: f64,
    pub mtp_weight: f32,
    pub loop_depth: usize, 
    pub temperature: f32, // NEW
}

pub struct DynamicScheduler {
    pub current_lr: f64,
    pub current_mtp_weight: f32,
    pub base_lr: f64,
    pub min_lr: f64,
    pub max_mtp_weight: f32,
    pub wake_ups: u32,
    pub target_loop_depth: usize, 
    
    short_ema: f32,
    long_ema: f32,
    cooldown: usize,
    active_cooldown: usize,
    steps_since_action: usize,
    total_steps: usize,
    is_in_recovery: bool,
}

impl DynamicScheduler {
    pub fn new(initial_lr: f64, initial_mtp: f32, target_loop_depth: usize) -> Self {
        Self {
            current_lr: initial_lr,
            current_mtp_weight: initial_mtp,
            base_lr: initial_lr,
            min_lr: 1.1e-4, 
            max_mtp_weight: 0.50,
            wake_ups: 0,
            target_loop_depth,
            short_ema: 0.0,
            long_ema: 0.0,
            cooldown: 50,
            active_cooldown: 50,
            steps_since_action: 0,
            total_steps: 0,
            is_in_recovery: false,
        }
    }

    fn get_current_loop_depth(&self, current_loss: f32) -> usize {
        let reference_loss = if self.short_ema == 0.0 { current_loss } else { self.short_ema };
        if reference_loss > 8.0 { 1 }
        else if reference_loss > 6.0 { 2 }
        else if reference_loss > 4.5 { (self.target_loop_depth / 2).max(2) }
        else { self.target_loop_depth }
    }

    // NEW: Dynamic Temperature Logic
    fn get_current_temperature(&self, current_loss: f32) -> f32 {
        let ref_loss = if self.short_ema == 0.0 { current_loss } else { self.short_ema };
        
        if ref_loss > 9.0 {
            8.0 // High gradient flow during early chaos
        } else if ref_loss > 7.0 {
            11.5 // Transitioning
        } else {
            15.0 // Stable BitNet quantization range
        }
    }

    pub fn step(&mut self, loss: f32) -> SchedulerState {
        self.total_steps += 1;
        self.steps_since_action += 1;

        let current_depth = self.get_current_loop_depth(loss);
        let current_temp = self.get_current_temperature(loss);

        if self.is_in_recovery {
            if self.steps_since_action >= 500 {
                self.is_in_recovery = false;
                self.steps_since_action = 0;
            }
            return SchedulerState { 
                lr: self.current_lr, 
                mtp_weight: self.current_mtp_weight, 
                loop_depth: current_depth,
                temperature: current_temp
            };
        }

        if self.total_steps == 1 {
            self.short_ema = loss;
            self.long_ema = loss;
            return SchedulerState { 
                lr: self.current_lr, 
                mtp_weight: self.current_mtp_weight, 
                loop_depth: current_depth,
                temperature: current_temp
            };
        }

        let alpha_short = 2.0 / (15.0 + 1.0);
        let alpha_long = 2.0 / (100.0 + 1.0);
        self.short_ema = alpha_short * loss + (1.0 - alpha_short) * self.short_ema;
        self.long_ema = alpha_long * loss + (1.0 - alpha_long) * self.long_ema;

        if self.total_steps < 100 || self.steps_since_action < self.active_cooldown {
            return SchedulerState { 
                lr: self.current_lr, 
                mtp_weight: self.current_mtp_weight, 
                loop_depth: current_depth,
                temperature: current_temp
            };
        }

        let trend_ratio = self.short_ema / self.long_ema;

        if trend_ratio > 1.05 {
            self.current_lr *= 0.5;
            self.current_mtp_weight = (self.current_mtp_weight * 0.7).max(0.10); 
            self.steps_since_action = 0;
            self.active_cooldown = self.cooldown;
        }
        else if trend_ratio > 0.995 {
            if self.current_lr <= self.min_lr {
                self.wake_ups += 1;
                self.current_lr = self.base_lr; 
                self.current_mtp_weight = (0.20 + (self.wake_ups as f32 * 0.05)).min(self.max_mtp_weight);
                self.short_ema = loss;
                self.long_ema = loss;
                self.is_in_recovery = true;
                self.steps_since_action = 0;
            } else {
                self.current_lr *= 0.85;
                self.current_mtp_weight = (self.current_mtp_weight * 0.9).max(0.12);
                self.steps_since_action = 0;
                self.active_cooldown = self.cooldown;
            }
        }
        else if trend_ratio < 0.97 {
            self.current_mtp_weight = (self.current_mtp_weight * 1.02).min(self.max_mtp_weight);
            self.steps_since_action = 0;
            self.active_cooldown = self.cooldown;
        }

        SchedulerState { 
            lr: self.current_lr, 
            mtp_weight: self.current_mtp_weight, 
            loop_depth: current_depth,
            temperature: current_temp
        }
    }
}