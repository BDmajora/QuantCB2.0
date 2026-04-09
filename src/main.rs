use burn::backend::Wgpu;
use burn::tensor::Tensor;

fn main() {
    // Explicitly using the Wgpu backend for Vulkan support
    type Backend = Wgpu;
    let device = Default::default();

    println!("QuantCB 2.0 | Backend: Vulkan (WGPU)");

    let tensor = Tensor::<Backend, 1>::from_data([1.0, 2.0, 3.0], &device);

    println!("Tensor initialized on GPU: {}", tensor);
}