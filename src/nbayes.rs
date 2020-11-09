use crate::{Model, Instance};

pub struct NBayes {
    ntraining: usize,
    ns_h: [usize; 2],
    n_eh: Vec<Vec<[usize; 2]>>,
}

impl NBayes {
    fn score_label(&self, instance: &Instance, label: u8) -> f64 {
        let mut logpr_eh = 0.0f64;
        for (i, &f) in instance.features.iter().enumerate() {
            let count = self.n_eh[label as usize][i as usize][f as usize];
            let c = count as f64;
            let t = self.ns_h[label as usize] as f64;
            logpr_eh += f64::log2((c + 0.5) / (t + 0.5));
        }
        let pr_h = self.ns_h[label as usize] as f64 / self.ntraining as f64;
        logpr_eh * pr_h
    }
}

pub fn train(samples: &[&Instance]) -> Box<NBayes> {
    let ntraining = samples.len();
    let nfeatures = samples[0].features.len();

    let n_nh = samples.iter().filter(|&s| s.label == 0).count();
    let n_h = ntraining - n_nh;
    let ns_h = [n_nh, n_h];

    let mut n_eh = Vec::new();
    for label in 0..2 {
        let mut lcounts = Vec::with_capacity(nfeatures);
        for f in 0..nfeatures {
            let mut counts = [0usize; 2];
            for inst in samples.iter() {
                if inst.label == label {
                    counts[inst.features[f] as usize] += 1;
                }
            }
            lcounts.push(counts);
        }
        n_eh.push(lcounts);
    }
    Box::new(NBayes { ntraining, ns_h, n_eh })
}

impl Model for NBayes {
    fn classify(&self, inst: &Instance) -> bool {
        let s_neg = self.score_label(inst, 0);
        let s_pos = self.score_label(inst, 1);
        s_pos > s_neg
    }
}
