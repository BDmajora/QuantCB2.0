use std::cmp::Ordering;

#[derive(Eq, PartialEq)]
pub struct MergeCandidate {
    pub freq: i32,
    pub pair: (u32, u32),
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.freq.cmp(&other.freq)
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct WordList {
    pub ids: Vec<u32>,
    pub prev: Vec<i32>,
    pub next: Vec<i32>,
    pub freq: i32,
}