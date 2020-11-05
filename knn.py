#!/usr/bin/python3
# Naive Bayes classification of binary instances.
# Bart Massey

import csv, math, random, sys

# Show individual instance results.
TRACE = True

# Class of instances.
class Instance(object):
    def __init__(self, row):
        self.name = row[0]
        self.label = int(row[1])
        self.features = [int(f) for f in row[2:]]

# Read instances.
with open(sys.argv[1], "r") as f:
    reader = csv.reader(f)
    instances = [Instance(row) for row in reader]
random.shuffle(instances)
ninstances = len(instances)

# Splits for cross-validation.
nsplits = int(sys.argv[2])

# Number of neighbors to consider
k = int(sys.argv[3])

# Number of features per instance. XXX Should be same for
# all instances.
nfeatures = len(instances[0].features)

# Hamming distance between feature vectors.
def hamming(f1, f2):
    return sum([abs(f1[i] - f2[i]) for i in range(nfeatures)])

# Try training on the training instances and then
# classifying the test instances.  Return the classification
# accuracy.
def try_tc(training, test):
    # Split into training and test set.
    ntraining = len(training)
    ntest = len(test)

    # Score test instances.
    correct = 0
    half = k / 2
    for inst in test:
        ordering = list(training)
        random.shuffle(ordering)
        ordering = sorted(
            ordering,
            key=lambda i: hamming(inst.features, i.features),
        )
        nspam = sum([i.label for i in ordering[:k]])
        guess = nspam > half
        if TRACE:
            print(inst.name, inst.label, guess)
        correct += int(inst.label == guess)

    return correct / ntest

# Number of instances per split.
nsplit = math.ceil(ninstances / nsplits)
splits = list()
for i in range(0, ninstances, nsplit):
    training = instances[0:i]
    if i + nsplit < ninstances:
        training += instances[i + nsplit:ninstances]
        test = instances[i:i + nsplit]
    else:
        test = instances[i:]
    print(try_tc(training, test))
