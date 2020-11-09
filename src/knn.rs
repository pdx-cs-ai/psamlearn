//! k-Nearest Neighbor learner.

use num_bigint::BigUint;

use crate::{Model, Instance};

/// Instance with features compiled to a bit vector
/// represented as a big integer. This allows much faster
/// distance computation.
struct BVI {
    label: u8,
    features: BigUint,
}

/// Model info.
pub struct KNN {
    /// Number of neighbors to use in voting.
    k: usize,
    /// Compiled instances to select from for voting.
    instances: Vec<BVI>,
}

/// Compile the feature list to a big integer.
fn collect_features(features: &[u8]) -> BigUint {
    features
        .iter()
        .enumerate()
        .fold(BigUint::from(0usize), |u, (i, &b)| {
            let b: BigUint = b.into();
            u | (b << i)
        })
}

/// Return the number of one bits ("popcount") in a big
/// integer. See `num-bigint` Issue
/// [#174](https://github.com/rust-num/num-bigint/issues/174).
fn count_ones(b: BigUint) -> u64 {
    b
        .to_u32_digits()
        .into_iter()
        .map(|d| d.count_ones() as u64)
        .sum()
}

/// Build our model from the training instance data.
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
        // Compile test instance.
        let instance = BVI {
            label: instance.label,
            features: collect_features(&instance.features),
        };

        // Sort instance labels by increasing instance
        // distance.
        let mut info: Vec<(u8, u64)> = self.instances
            .iter()
            .map(|i| {
                let h = count_ones(&i.features ^ &instance.features);
                (i.label, h)
            })
            .collect();
        info.sort_by_key(|x| x.1);

        // For each label, compute the counts among the *k*
        // nearest neighbors.
        let lweights: [usize; 2] = info
            .into_iter()
            .take(self.k)
            .fold([0usize; 2], |mut counts, (c, _)| {
                counts[c as usize] += 1;
                counts
            });

        // Whichever count is larger wins.
        lweights[1] > lweights[0]
    }
}
