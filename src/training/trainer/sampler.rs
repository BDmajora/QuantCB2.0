use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule; // <--- FIX: Added this trait import
use crate::model::QuantCB;
use crate::training::tokenizer::BLTPatcher;
use crate::generator::TextGenerator;
use super::mode::SampleMode;
use crate::training::trainer::trainer_config::TAG_TRUTH;

pub fn run_sample_block<B: AutodiffBackend>(
    step: usize, 
    model: &QuantCB<B>, 
    patcher: &BLTPatcher<B::InnerBackend>,
    device: &B::Device
) {
    let mode = SampleMode::get_for_step(step, 100);
    let prompt = mode.build_prompt(TAG_TRUTH);
    
    // Now .valid() will work because AutodiffModule is in scope
    let valid_model = model.clone().valid();
    
    let output = TextGenerator::generate(&valid_model, patcher, device, &prompt, 60, 0.8, 1.2);
    
    println!("\n[Step {} - {}]\n>> {}\n", step, mode.name, output);
}