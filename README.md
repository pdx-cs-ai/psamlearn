# psamlearn: machine-learning classification on the PSAM spam database
Bart Massey

This repository contains a couple of things:

* Instances (feature-vectors) from the
  [PSAM](http://www.cs.pdx.edu/~bart/papers/spam.pdf) spam
  corpus of some years ago. Each `csv` file contains
  instances consisting of a name, a class (1 for spam, 0 for
  ham), and a vector of features obtained via
  big-bag-of-words and
  [SpamAssassin](https://spamassassin.apache.org/) analyses.

* Rust code for machine-learning classifiers for the
  instances.

## Build and Run

To build this, say `cargo build --release`.

To run Naïve Bayes on the "personal" corpus with 10-way
cross-validation, say

    cargo run -- --crossval 10 personal.csv nbayes

To run KNN on the "personal" corpus with 10-way
cross-validation and a neighbor distance of 5, say

    cargo run -- --crossval 10 personal.csv knn -k 5

In both cases, the output will consist of the accuracy for
each cross-validation split.

For further information on usage, try `cargo run -- --help`.
