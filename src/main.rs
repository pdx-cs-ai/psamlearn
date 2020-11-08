mod evenchunks;
mod knn;
mod nbayes;

use evenchunks::*;

use argh::FromArgs;
use rand::seq::SliceRandom;
use serde::Deserialize;

#[derive(FromArgs)]
/// Machine learning with binary features and classification.
struct Args {
    /// n-way cross-validation (0 for LOOCV)
    #[argh(option)]
    crossval: Option<usize>,
    /// learning algorithm
    #[argh(subcommand)]
    algorithm: ArgsAlg,
    /// csv feature file (default stdin)
    #[argh(positional)]
    features: Option<String>,
}

#[derive(FromArgs)]
#[argh(subcommand)]
enum ArgsAlg {
    NBayes(NBayesArgs),
    KNN(KNNArgs),
}

#[derive(FromArgs)]
/// Naïve Bayes classifier.
#[argh(subcommand, name = "nbayes")]
struct NBayesArgs {}

#[derive(FromArgs)]
/// k-Nearest Neighbor classifier.
#[argh(subcommand, name = "knn")]
struct KNNArgs {
    #[argh(option, short='k')]
    /// k (default 5)
    k: Option<usize>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Instance {
    name: String,
    label: u8,
    features: Vec<u8>,
}

pub trait Model {
    fn classify(&self, instance: &Instance) -> bool;
}

fn main() {
    let args: Args = argh::from_env();
    let features: Box<dyn std::io::Read> = match args.features {
        None => Box::new(std::io::stdin()),
        Some(name) => Box::new(std::fs::File::open(name).unwrap_or_else(
            |e| {
                eprintln!("could not open features: {}", e);
                std::process::exit(1);
            }
        )),
    };

    let mut rdr = csv::ReaderBuilder::new().has_headers(false).from_reader(features);
    let mut instances: Vec<Instance> =
        rdr.deserialize().map(|r| r.unwrap_or_else(
            |e| {
                eprintln!("could not read record: {}", e);
                std::process::exit(1);
            },
        )).collect();

    let mut rng = rand::thread_rng();
    instances.shuffle(&mut rng);

    let samples: Vec<(Vec<&Instance>, &[Instance])> =
        match args.crossval {
            None => {
                let split = instances.len() / 2;
                vec![(
                    instances[split..].iter().collect(),
                    &instances[..split],
                )]
            },
        Some(mut n) => {
            if n == 0 {
                n = instances.len();
            }
            let chunks: Vec<&[Instance]> =
                EvenChunks::nchunks(&instances, n).collect();
            (0..n).map(|i| {
                let left = chunks[..i].iter().cloned().flatten();
                let right = chunks[i+1..].iter().cloned().flatten();
                (left.chain(right).collect(), chunks[i])
            }).collect()
        }
    };

    let train: Box<dyn Fn(&[&Instance])->Box<dyn Model>> =
        match args.algorithm {
            ArgsAlg::NBayes(NBayesArgs{}) => Box::new(|i| nbayes::train(i)),
            ArgsAlg::KNN(KNNArgs{k}) => match k {
                None => Box::new(|i| knn::train(5, i)),
                Some(k) => Box::new(move |i| knn::train(k, i)),
            },
        };

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
