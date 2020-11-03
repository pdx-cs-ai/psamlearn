# psambayes: Naïve Bayesian classification on the PSAM spam database
Bart Massey

This repository contains a couple of things:

* Instances (feature-vectors) from the
  [PSAM](http://www.cs.pdx.edu/~bart/papers/spam.pdf) spam
  corpus of some years ago. Each `csv` file contains
  instances consisting of a name, a class (1 for spam, 0 for
  ham), and a vector of features obtained via
  big-bag-of-words and
  [SpamAssassin](https://spamassassin.apache.org/) analyses.

* Python code for a Naïve Bayes classifier for the
  instances. The classifier uses log-likelihood comparisons
  with *m-*estimation and performs *n-*way cross-validation.

  The classifier achieves about 95% accuracy on the
  `personal.csv` corpus with 10-way cross-validation.

## Running

To run on the "personal" corpus with 10-way
cross-validation, say

    python3 nbayes.py personal.csv 10

The output will consist of the accuracy for each
cross-validation split.
