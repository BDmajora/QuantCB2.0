#[derive(Debug, Clone)]
pub struct BytePatch {
    pub raw_bytes: Vec<u8>,
    pub _start_idx: usize,
    pub _end_idx: usize,
}

pub struct EntropyPatcher {
    pub entropy_threshold: f32,
    pub max_patch_size: usize,
    pub special_tags: Vec<Vec<u8>>, 
}

impl EntropyPatcher {
    pub fn new(entropy_threshold: f32, max_patch_size: usize, special_tags: Vec<&'static str>) -> Self {
        Self {
            entropy_threshold,
            max_patch_size,
            special_tags: special_tags.iter().map(|s| s.as_bytes().to_vec()).collect(),
        }
    }

    /// Uses the internal default threshold
    pub fn segment_into_patches(&self, text: &str, byte_entropies: &[f32]) -> Vec<BytePatch> {
        self.segment_with_threshold(text, byte_entropies, self.entropy_threshold)
    }

    /// The new core logic that accepts a dynamic threshold from the scheduler
    pub fn segment_with_threshold(
        &self, 
        text: &str, 
        byte_entropies: &[f32], 
        threshold: f32
    ) -> Vec<BytePatch> {
        let bytes = text.as_bytes();
        let mut patches = Vec::new();
        let mut i = 0;

        while i < bytes.len() {
            // 1. Check for special tags
            let mut matched_tag_len = None;
            for tag in &self.special_tags {
                if bytes[i..].starts_with(tag) {
                    matched_tag_len = Some(tag.len());
                    break;
                }
            }

            if let Some(tag_len) = matched_tag_len {
                patches.push(BytePatch {
                    raw_bytes: bytes[i..i + tag_len].to_vec(),
                    _start_idx: i,
                    _end_idx: i + tag_len,
                });
                i += tag_len;
                continue;
            }

            // 2. Standard Logic
            let mut current_patch = Vec::new();
            let patch_start = i;
            
            while i < bytes.len() {
                let b = bytes[i];
                let entropy = byte_entropies.get(i).cloned().unwrap_or(0.0);
                current_patch.push(b);

                // FIX: Use the 'threshold' argument passed into the function, NOT self.entropy_threshold
                let hit_entropy = entropy > threshold;
                let hit_limit = current_patch.len() >= self.max_patch_size;
                let is_boundary = b == b' ' || b == b'\n' || b == b'\t';

                let next_idx = i + 1;
                let next_is_tag = if next_idx < bytes.len() {
                    self.special_tags.iter().any(|t| bytes[next_idx..].starts_with(t))
                } else {
                    false
                };

                if hit_entropy || hit_limit || is_boundary || next_is_tag {
                    patches.push(BytePatch {
                        raw_bytes: current_patch,
                        _start_idx: patch_start,
                        _end_idx: i + 1,
                    });
                    i += 1;
                    break;
                }
                i += 1;
            }
        }
        patches
    }
}