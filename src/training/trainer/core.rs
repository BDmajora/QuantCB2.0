use burn::tensor::backend::AutodiffBackend;
use burn::optim::Optimizer;
use burn::tensor::ElementConversion;
use crate::model::QuantCB;
use crate::training::trainer::batcher::QuantCBBatch;

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