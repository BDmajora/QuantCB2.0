mod model;
mod training;
mod generator;

fn main() {
    println!("QuantCB 2.0 | Backend: Vulkan (WGPU)");

    // Accessing the run function inside src/training/trainer/trainer.rs
    training::trainer::trainer::run();
}