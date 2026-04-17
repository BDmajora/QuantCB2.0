pub struct SampleMode {
    pub name: &'static str,
    pub tag: &'static str,
    pub prefix: &'static str,
}

impl SampleMode {
    // We only keep the two active contestants for now
    pub const ALL: [SampleMode; 2] = [
        SampleMode { 
            name: "BARD", 
            tag: "<|shakespeare|>", 
            prefix: "Shylock:" 
        },
        SampleMode { 
            name: "WIKI", 
            tag: "<|wiki|>", 
            prefix: "Wikipedia:" 
        },
    ];

    /// Dynamically selects a mode based on the current step
    pub fn get_for_step(step: usize, interval: usize) -> &'static Self {
        let index = (step / interval) % Self::ALL.len();
        &Self::ALL[index]
    }

    /// Helper to format the prompt string
    pub fn build_prompt(&self, truth_tag: &str) -> String {
        format!("{} {} \n{}", truth_tag, self.tag, self.prefix)
    }
}