use bvec::BVec;

use crate::{Model, Instance};

struct BVI {
    label: u8,
    features: BVec,
}

pub struct KNN {
    k: usize,
    instances: Vec<BVI>,
}

pub fn train(k: usize, instances: &[&Instance]) -> Box<KNN> {
    let instances: Vec<BVI> = instances
        .iter()
        .map(|&i| BVI {
            label: i.label,
            features: i.features.iter().map(|&i| i > 0).collect(),
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
            features: instance.features.iter().map(|&i| i > 0).collect(),
        };
        let mut info: Vec<(u8, u64)> = self.instances
            .iter()
            .map(|i| {
                let h: u64 = (&i.features ^ &instance.features).count_ones();
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
