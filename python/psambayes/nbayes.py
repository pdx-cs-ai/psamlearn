#!/usr/bin/python3
# Naive Bayes classification of binary instances.
# Bart Massey

import csv, math, random, sys

# Show individual instance results.
TRACE = False

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

# Number of features per instance. XXX Should be same for
# all instances.
nfeatures = len(instances[0].features)

# Try training on the training instances and then
# classifying the test instances.  Return the classification
# accuracy.
def try_tc(training, test):
    # Split into training and test set.
    ntraining = len(training)
    ntest = len(test)

    # Split training into hams and spams.
    spams = [i for i in training if i.label == 1]
    nspams = len(spams)
    hams = [i for i in training if i.label == 0]
    nhams = len(hams)
    nsH = [nhams, nspams]

    # Probability that a training message is spam.
    prH = nspams / ntraining
    prnotH = nhams / ntraining
    prsH = [prH, prnotH]

    # Per-feature probabilities of evidence given
    # hypothesis. Access is nsEH[label][feature][fval].
    nsEH = list()
    for label in [0, 1]:
        lcounts = list()
        for f in range(nfeatures):
            counts = [0, 0]
            for inst in training:
                if inst.label == label:
                    counts[inst.features[f]] += 1
            lcounts.append(counts)
        nsEH.append(lcounts)

    # Return a score proportional to the Naïve Bayes
    # log-likelihood that an instance has the given label.
    def score_label(instance, label):
        # Compute probability of evidence given hypothesis.
        logprEH = list()
        for f in range(nfeatures):
            count = nsEH[label][f][instance.features[f]]
            logprEH.append(math.log2((count + 0.5) / (nsH[label] + 0.5)))

        return sum(logprEH) * prsH[label]

    # Score test instances.
    matrix = [[0] * 2 for _ in range(2)]
    for inst in test:
        ss = score_label(inst, 1)
        sh = score_label(inst, 0)
        guess = int(ss > sh)
        if TRACE:
            print(inst.name, inst.label, guess)
        matrix[inst.label][guess] += 1

    return matrix

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
    ntest = len(test)
    matrix = try_tc(training, test)
    accuracy = (matrix[0][0] + matrix[1][1]) / ntest
    fpr = matrix[0][1] / ntest
    fnr = matrix[1][0] / ntest
    print(f"acc:{accuracy:.3f} fpr:{fpr:.3f}  fnr:{fnr:.3f}")
