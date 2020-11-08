use num_bigint::BigUint;

use crate::{Model, Instance};

struct BVI {
    label: u8,
    features: BigUint,
}

pub struct KNN {
    k: usize,
    instances: Vec<BVI>,
}

fn collect_features(features: &[u8]) -> BigUint {
    features
        .iter()
        .enumerate()
        .fold(BigUint::from(0usize), |u, (i, &b)| {
            let b: BigUint = b.into();
            u | (b << i)
        })
}

fn count_ones(b: BigUint) -> u32 {
    b
        .to_u32_digits()
        .into_iter()
        .map(|d| d.count_ones())
        .sum()
}

pub fn train(k: usize, instances: &[&Instance]) -> Box<KNN> {
    let instances: Vec<BVI> = instances
        .iter()
        .map(|&i| BVI {
            label: i.label,
            features: collect_features(&i.features),
        })
        .collect();
    Box::new(KNN {
        k,
        instances,
    })
}

impl Model for KNN {
    fn classify(&self, instance: &Instance) -> bool {
        let instance = BVI {
            label: instance.label,
            features: collect_features(&instance.features),
        };
        let mut info: Vec<(u8, u64)> = self.instances
            .iter()
            .map(|i| {
                let h: u64 = count_ones(&i.features ^ &instance.features).into();
                (i.label, h)
            })
            .collect();
        info.sort_by_key(|x| x.1);
        let lweights: [usize; 2] = info
            .into_iter()
            .take(self.k)
            .fold([0usize; 2], |mut counts, (c, _)| {
                counts[c as usize] += 1;
                counts
            });
        lweights[1] > lweights[0]
    }
}
