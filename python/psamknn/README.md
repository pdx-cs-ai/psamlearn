# psamknn: k-Nearest Neighbor classification on the PSAM spam database
Bart Massey

This repository contains a couple of things:

* Instances (feature-vectors) from the
  [PSAM](http://www.cs.pdx.edu/~bart/papers/spam.pdf) spam
  corpus of some years ago. Each `csv` file contains
  instances consisting of a name, a class (1 for spam, 0 for
  ham), and a vector of features obtained via
  big-bag-of-words and
  [SpamAssassin](https://spamassassin.apache.org/) analyses.

* Python code for a k-Nearest Neighbor classifier for the
  instances.

## Running

You will need the Python `gmpy2` package to run this. On
Debian, you can say:

    apt install libmpfr-dev
    apt install libmpc-dev
    pip3 install gmpy2

To run on the "personal" corpus with 10-way
cross-validation and a neighbor distance of 5, say

    python3 knn.py personal.csv 10 5

The output will consist of the accuracy for each
cross-validation split.
