use argh::FromArgs;
use rand::seq::SliceRandom;
use serde::Deserialize;

#[derive(FromArgs)]
/// Machine learning with binary features and classification.
struct Args {
    /// n-way cross-validation (0 for LOOCV)
    #[argh(option)]
    crossval: Option<u64>,
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

#[derive(Deserialize, Debug)]
pub struct Instance {
    name: String,
    class: u8,
    features: Vec<u8>,
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
    let mut records: Vec<Instance> =
        rdr.deserialize().map(|r| r.unwrap_or_else(
            |e| {
                eprintln!("could not read record: {}", e);
                std::process::exit(1);
            },
        )).collect();

    let mut rng = rand::thread_rng();
    records.shuffle(&mut rng);

    match args.crossval {
        None => println!("split50"),
        Some(0) => println!("loocv"),
        Some(n) => println!("crossval {}", n),
    }
    match args.algorithm {
        ArgsAlg::NBayes(NBayesArgs{}) => println!("nbayes"),
        ArgsAlg::KNN(KNNArgs{k}) => match k {
            None => println!("nbayes 5"),
            Some(k) => println!("nbayes {}", k),
        },
    }
}
