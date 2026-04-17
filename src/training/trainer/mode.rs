pub struct SampleMode {
    pub name: &'static str,
    pub tag: &'static str,
    pub prefix: &'static str,
}

impl SampleMode {
    pub const ALL: [SampleMode; 2] = [
        SampleMode { name: "BARD", tag: "<|shakespeare|>", prefix: "Shylock:" },
        SampleMode { name: "WIKI", tag: "<|wiki|>", prefix: "Wikipedia:" },
    ];

    pub fn get_for_step(step: usize, interval: usize) -> &'static Self {
        let index = (step / interval) % Self::ALL.len();
        &Self::ALL[index]
    }

    pub fn build_prompt(&self, truth_tag: &str) -> String {
        // BitNet 1.58-bit models benefit from a very structured start 
        // to stabilize the initial 8-bit activation quantization.
        format!("<|bos|> {} {} \n{}", truth_tag, self.tag, self.prefix)
    }
}