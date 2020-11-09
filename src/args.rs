//! Argument parsing for program.

use argh::FromArgs;

#[derive(FromArgs)]
/// Machine learning with binary features and classification.
pub struct Args {
    /// n-way cross-validation (0 for LOOCV)
    #[argh(option)]
    pub crossval: Option<usize>,
    /// learning algorithm
    #[argh(subcommand)]
    pub algorithm: ArgsAlg,
    /// csv feature file (default stdin)
    #[argh(positional)]
    pub features: Option<String>,
}

#[derive(FromArgs)]
#[argh(subcommand)]
pub enum ArgsAlg {
    NBayes(NBayesArgs),
    KNN(KNNArgs),
    ID3(ID3Args),
}

#[derive(FromArgs)]
/// Naïve Bayes classifier.
#[argh(subcommand, name = "nbayes")]
pub struct NBayesArgs {}

#[derive(FromArgs)]
/// ID3 classifier.
#[argh(subcommand, name = "id3")]
pub struct ID3Args {
    #[argh(option, short = 'g')]
    /// minimum gain to split (recommend 0.05)
    pub min_gain: Option<f64>,
    #[argh(option, short = 'c')]
    /// minimum chi_square to split (recommend 3.841 for p=0.05)
    pub min_chisquare: Option<f64>,
}

#[derive(FromArgs)]
/// k-Nearest Neighbor classifier.
#[argh(subcommand, name = "knn")]
pub struct KNNArgs {
    #[argh(option, short = 'k')]
    /// k (default 5)
    pub k: Option<usize>,
}

/// Parse arguments and return top-level argument structure.
pub fn parse() -> Args {
    argh::from_env()
}
