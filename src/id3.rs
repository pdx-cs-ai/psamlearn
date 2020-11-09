//! ID3 Decision Tree learner.

use std::collections::HashSet;

use crate::{Instance, Model};

enum DTree {
    Branch {
        f: usize,
        pos: Box<DTree>,
        neg: Box<DTree>,
    },
    Label(bool),
}

fn count_labels(insts: &[&Instance]) -> (usize, usize) {
    let ninsts = insts.len();

    let mut np = 0;
    for i in insts {
        np += i.label as usize;
    }
    (np, ninsts - np)
}

fn split<'a>(insts: &[&'a Instance], f: usize) -> (Vec<&'a Instance>, Vec<&'a Instance>) {
    let mut pos = Vec::new();
    let mut neg = Vec::new();
    for &i in insts {
        if i.features[f] > 0 {
            pos.push(i);
        } else {
            neg.push(i);
        }
    }
    (pos, neg)
}

fn entropy(insts: &[&Instance]) -> f64 {
    let (np, nn) = count_labels(insts);
    if np == 0 || nn == 0 {
        return 0.0;
    }
    let ninsts = np + nn;

    let pr_p = np as f64 / ninsts as f64;
    let pr_n = nn as f64 / ninsts as f64;
    -pr_p * f64::log2(pr_p) - pr_n * f64::log2(pr_n)
}

fn chi_square(pos: usize, neg: usize) -> f64 {
    let pos = pos as f64;
    let neg = neg as f64;

    let avg = (pos + neg) / 2.0;
    let dpos = pos - avg;
    let dneg = neg - avg;
    (dpos * dpos + dneg * dneg) / avg
}

impl DTree {
    fn make_dtree(
        insts: &[&Instance],
        mut used: HashSet<usize>,
        u: f64,
        min_gain: Option<f64>,
        min_chisquare: Option<f64>,
    ) -> DTree {
        assert!(!insts.is_empty());
        let nfeatures = insts[0].features.len();

        let (np, nn) = count_labels(insts);
        let ninsts = np + nn;

        if used.len() == nfeatures {
            return DTree::Label(np > nn);
        }

        if let Some(min_chisquare) = min_chisquare {
            let chs = chi_square(np, nn);
            if chs < min_chisquare {
                return DTree::Label(np > nn);
            }
        }

        let mut best_f = None;
        let mut best_gain = None;
        let mut best_split = None;
        for f in 0..nfeatures {
            if used.contains(&f) {
                continue;
            }
            let (pos, neg) = split(insts, f);
            let npos = pos.len();
            let nneg = neg.len();
            if npos == 0 || nneg == 0 {
                continue;
            }
            let u_pos = entropy(&pos);
            let u_neg = entropy(&neg);
            let pr_pos = npos as f64 / ninsts as f64;
            let pr_neg = nneg as f64 / ninsts as f64;
            let gain = u - pr_pos * u_pos - pr_neg * u_neg;
            if gain <= 0.0 {
                // XXX Numerical errors can lead to tiny
                // negative gains.
                continue;
            }
            let save = match best_gain {
                None => true,
                Some(best_gain) => gain > best_gain,
            };
            if save {
                best_gain = Some(gain);
                best_f = Some(f);
                best_split = Some(((pos, u_pos), (neg, u_neg)));
            }
        }

        if best_f.is_none() {
            return DTree::Label(np > nn);
        }

        if let Some(min_gain) = min_gain {
            if best_gain.unwrap() < min_gain {
                return DTree::Label(np > nn);
            }
        }

        let f = best_f.unwrap();
        let ((pos, pu), (neg, nu)) = best_split.unwrap();
        used.insert(f);

        let pos = DTree::make_dtree(&pos, used.clone(), pu, min_gain, min_chisquare);
        let neg = DTree::make_dtree(&neg, used, nu, min_gain, min_chisquare);
        DTree::Branch { pos: Box::new(pos), neg: Box::new(neg), f }
    }

    fn new(
        insts: &[&Instance],
        min_gain: Option<f64>,
        min_chisquare: Option<f64>,
    ) -> Self {
        DTree::make_dtree(insts, HashSet::new(), entropy(insts), min_gain, min_chisquare)
    }

    fn classify(&self, inst: &Instance) -> bool {
        match self {
            DTree::Label(l) => *l,
            DTree::Branch { f, pos, neg } => {
                if inst.features[*f] > 0 {
                    pos.classify(inst)
                } else {
                    neg.classify(inst)
                }
            },
        }
    }
}

/// Model info.
pub struct ID3(DTree);

/// Build our model from the training instance data.
pub fn train(insts: &[&Instance], min_gain: Option<f64>, min_chisquare: Option<f64>) -> Box<ID3> {
    let dtree = DTree::new(insts, min_gain, min_chisquare);
    Box::new(ID3(dtree))
}

impl Model for ID3 {
    fn classify(&self, inst: &Instance) -> bool {
        self.0.classify(inst)
    }
}
