use argh::FromArgs;

#[derive(FromArgs)]
/// Machine learning with binary features and classification.
struct Args {
    /// n-way cross-validation (0 for LOOCV)
    #[argh(option)]
    crossval: Option<u64>,
    /// learning algorithm
    #[argh(subcommand)]
    algorithm: ArgsAlg,
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

fn main() {
    let args: Args = argh::from_env();
    match args.algorithm {
        ArgsAlg::NBayes(NBayesArgs{}) => println!("nbayes"),
        ArgsAlg::KNN(KNNArgs{k}) => match k {
            None => println!("nbayes 5"),
            Some(k) => println!("nbayes {}", k),
        },
    }
    match args.crossval {
        None => println!("split50"),
        Some(0) => println!("loocv"),
        Some(n) => println!("crossval {}", n),
    }
}
