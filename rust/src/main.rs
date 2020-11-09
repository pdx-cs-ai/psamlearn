//! Machine learning driver geared toward PSAM data. Accepts
//! binary features and classifications.

mod args;
mod evenchunks;

mod id3;
mod knn;
mod nbayes;

use args::*;
use evenchunks::*;

use rand::seq::SliceRandom;
use serde::Deserialize;

/// Instance from corpus.
#[derive(Deserialize, Debug, Clone)]
pub struct Instance {
    /// Instance name.
    name: String,
    /// Instance classification label.
    label: u8,
    /// Instance feature labels.
    features: Vec<u8>,
}

/// Trait representing results of training a particular
/// kind of model.
pub trait Model {
    /// Classify an instance based on this model's data.
    fn classify(&self, instance: &Instance) -> bool;
}

fn main() {
    let args = args::parse();

    // Readable corpus file.
    let corpus: Box<dyn std::io::Read> = match args.features {
        None => Box::new(std::io::stdin()),
        Some(name) => Box::new(std::fs::File::open(name).unwrap_or_else(|e| {
            eprintln!("could not open features: {}", e);
            std::process::exit(1);
        })),
    };

    // Read corpus instances.
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(corpus);
    let mut instances: Vec<Instance> = rdr
        .deserialize()
        .map(|r| {
            r.unwrap_or_else(|e| {
                eprintln!("could not read record: {}", e);
                std::process::exit(1);
            })
        })
        .collect();

    // Shuffle instances.
    let mut rng = rand::thread_rng();
    instances.shuffle(&mut rng);

    // Find a list of training / test splits.
    let samples: Vec<(Vec<&Instance>, &[Instance])> = match args.crossval {
        // Single 50/50 split.
        None => {
            let split = instances.len() / 2;
            vec![(instances[split..].iter().collect(), &instances[..split])]
        }
        Some(mut n) => {
            if n == 0 {
                n = instances.len();
            }
            // Crossval splits.
            let chunks: Vec<&[Instance]> = EvenChunks::nchunks(&instances, n).collect();
            (0..n)
                .map(|i| {
                    let left = chunks[..i].iter().cloned().flatten();
                    let right = chunks[i + 1..].iter().cloned().flatten();
                    (left.chain(right).collect(), chunks[i])
                })
                .collect()
        }
    };

    // Find a training function based on kind of learning.
    let train: Box<dyn Fn(&[&Instance]) -> Box<dyn Model>> = match args.algorithm {
        ArgsAlg::NBayes(NBayesArgs {}) => Box::new(|i| nbayes::train(i)),
        ArgsAlg::KNN(KNNArgs { k }) => match k {
            None => Box::new(|i| knn::train(5, i)),
            Some(k) => Box::new(move |i| knn::train(k, i)),
        },
        ArgsAlg::ID3(ID3Args {
            min_gain,
            min_chisquare,
        }) => Box::new(move |i| id3::train(i, min_gain, min_chisquare)),
    };

    // Run testing, report results.
    for (tr, cl) in samples {
        let model = train(&tr);
        let mut stats = [0usize; 4];
        for c in cl.iter() {
            let actual = c.label;
            let predicted = model.classify(c);
            let ix = match (actual, predicted) {
                (0, false) => 0,
                (0, true) => 1,
                (1, false) => 2,
                (1, true) => 3,
                _ => panic!("internal error: bad stats"),
            };
            stats[ix] += 1;
        }
        let ntraining: usize = stats.iter().sum();
        let accuracy = (stats[0] + stats[3]) as f64 / ntraining as f64;
        println!("{:?} {:.3}", stats, accuracy);
    }
}
