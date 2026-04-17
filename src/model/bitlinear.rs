use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor, ElementConversion};

#[derive(Config, Debug)]
pub struct BitLinearConfig {
    pub d_in: usize,
    pub d_out: usize,
    #[config(default = false)]
    pub bias: bool,
}

impl BitLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BitLinear<B> {
        // Zero-centered initialization to prevent massive positive drift
        let std_dev = (1.0 / self.d_in as f64).sqrt();
        let weight = Tensor::<B, 2>::random(
            [self.d_out, self.d_in], 
            Distribution::Normal(0.0, std_dev), 
            device
        );

        let gamma_w = Tensor::<B, 1>::ones([1], device);
        
        let bias = if self.bias {
            // Corrected: Explicitly wrap with Param::from_tensor
            Some(Param::from_tensor(Tensor::<B, 1>::zeros([self.d_out], device)))
        } else {
            None
        };

        BitLinear {
            // Corrected: Explicitly wrap with Param::from_tensor
            weight: Param::from_tensor(weight),
            gamma_w: Param::from_tensor(gamma_w),
            bias,
            d_in: self.d_in,
            d_out: self.d_out,
        }
    }
}

#[derive(Module, Debug)]
pub struct BitLinear<B: Backend> {
    pub weight: Param<Tensor<B, 2>>, 
    pub gamma_w: Param<Tensor<B, 1>>, 
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub d_in: usize,
    pub d_out: usize,
}

impl<B: Backend> BitLinear<B> {
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let eps: B::FloatElem = 1e-5.elem();
        
        // Extract tensors from Param wrappers for the forward calculation
        let weight_tensor = self.weight.val();
        let gamma_w_tensor = self.gamma_w.val();
        
        // --- 1. Activation Quantization (8-bit) ---
        let gamma_x = x.clone().abs().max_dim(D - 1).detach(); 
        
        let x_scaled = x.clone().mul_scalar(127.0).div(gamma_x.clone().add_scalar(eps));
        
        // FIX: Symmetric round for both positive and negative activations
        let x_quant = x_scaled.clone()
            .add(x_scaled.sign().mul_scalar(0.5))
            .int()
            .float()
            .clamp(-128.0, 127.0);

        // --- 2. Weight Quantization (Ternary STE) ---
        let w_scale = weight_tensor.clone().abs().mean().detach(); 
        
        let w_scale_2d = w_scale.reshape([1, 1]);
        let w_scaled = weight_tensor.clone().div(w_scale_2d.add_scalar(eps));
        
        // WORKAROUND: Simulate .round() for weights (shift based on sign)
        let w_quant = w_scaled.clone()
            .add(w_scaled.sign().mul_scalar(0.5))
            .int()
            .float()
            .clamp(-1.0, 1.0);

        // STE Trick: w_quant in forward pass, gradients route to weight_tensor
        let weight_for_forward = w_quant.detach() - weight_tensor.clone().detach() + weight_tensor.clone();

        // --- 3. Linear Projection ---
        let mut weight_shape = [1; D];
        weight_shape[D - 2] = self.d_in;
        weight_shape[D - 1] = self.d_out;
        
        let weight_projector: Tensor<B, D> = weight_for_forward.transpose().reshape(weight_shape);
        let out = x_quant.matmul(weight_projector);

        // --- 4. Rescale & Bias ---
        let gamma_w_expanded: Tensor<B, D> = gamma_w_tensor.reshape([1; D]);
        
        let out = out.mul(gamma_w_expanded).mul(gamma_x).div_scalar(127.0);
        
        if let Some(bias_param) = &self.bias {
            let mut bias_shape = [1; D];
            bias_shape[D - 1] = self.d_out;
            let bias_expanded: Tensor<B, D> = bias_param.val().reshape(bias_shape);
            out.add(bias_expanded)
        } else {
            out
        }
    }
}