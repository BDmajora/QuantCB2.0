pub mod data;
pub mod tokenizer;
pub mod trainer;

// 1. RE-EXPORT: Pull the config from the 'trainer' sub-module 
// so it's accessible as crate::training::trainer_config
pub use crate::training::trainer::trainer_config;

// Re-exporting common items for cleaner imports elsewhere
pub use data::learning::DynamicScheduler;

// 2. Fixed the name to BPETokenizer
pub use tokenizer::tokenizer::BPETokenizer;

// Note: 'pub use trainer::trainer::Trainer' remains removed 
// until the struct is actually defined.