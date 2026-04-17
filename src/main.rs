mod model;
mod training;
mod generator;

#[tokio::main]
async fn main() {
    println!("QuantCB 2.0 | Backend: Vulkan (WGPU)");

    // Accessing the run function inside src/training/trainer/trainer.rs
    // Because run() is now an async function, we must await it!
    training::trainer::trainer::run().await;
}