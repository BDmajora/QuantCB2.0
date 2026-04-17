use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

#[derive(Config, Debug)]
pub struct BitLinearConfig {
    pub d_in: usize,
    pub d_out: usize,
    #[config(default = false)]
    pub bias: bool,
}

impl BitLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BitLinear<B> {
        let weight = Tensor::random([self.d_in, self.d_out], Distribution::Normal(0.0, 0.02), device);
        
        let bias = if self.bias {
            Some(Param::from_tensor(Tensor::zeros([self.d_out], device)))
        } else {
            None
        };

        BitLinear {
            weight: Param::from_tensor(weight),
            bias,
        }
    }
}

#[derive(Module, Debug)]
pub struct BitLinear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> BitLinear<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let w = self.weight.val();
        let eps = 1e-5;
        
        // ----------------------------------------------------
        // 1. Weight Quantization to {-1, 0, 1} with STE
        // ----------------------------------------------------
        // FIX: Added .detach() so the optimizer doesn't loop back through the scale factor
        let gamma_w = w.clone().abs().mean().detach(); 
        let w_scaled = w.div(gamma_w.clone().add_scalar(eps).unsqueeze::<2>());
        let w_clamped = w_scaled.clamp(-1.0, 1.0);
        
        // Manual Rounding
        let w_quant_hard = w_clamped.clone().sign() * (w_clamped.clone().abs().add_scalar(0.5)).int().float();
        
        // Straight-Through Estimator: Use hard value for forward, clamped value for backward
        let w_quant = (w_quant_hard - w_clamped.clone()).detach() + w_clamped;

        // ----------------------------------------------------
        // 2. Activation Quantization to 8-bit with STE
        // ----------------------------------------------------
        // FIX: Added .detach() to prevent gradient explosion through activation scaling
        let gamma_x = x.clone().abs().max_dim(D - 1).detach(); 
        let x_scaled = (x.mul_scalar(127.0)).div(gamma_x.clone().add_scalar(eps));
        let x_clamped = x_scaled.clamp(-128.0, 127.0);
        
        // Manual Rounding
        let x_quant_hard = x_clamped.clone().sign() * (x_clamped.clone().abs().add_scalar(0.5)).int().float();
        
        // Straight-Through Estimator
        let x_quant = (x_quant_hard - x_clamped.clone()).detach() + x_clamped;

        // ----------------------------------------------------
        // 3. MatMul (Broadcasting weights to Rank D)
        // ----------------------------------------------------
        let w_quant_d = w_quant.unsqueeze::<D>();
        let out_quant = x_quant.matmul(w_quant_d);

        // ----------------------------------------------------
        // 4. Dequantization
        // ----------------------------------------------------
        let gamma_w_d = gamma_w.unsqueeze::<D>();
        let scale = (gamma_x * gamma_w_d).div_scalar(127.0);
        
        let mut out = out_quant * scale;

        if let Some(bias) = &self.bias {
            out = out + bias.val().unsqueeze::<D>();
        }

        out
    }
}