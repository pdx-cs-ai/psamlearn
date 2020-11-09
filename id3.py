#!/usr/bin/python3
# Naive Bayes classification of binary instances.
# Bart Massey

import csv, math, random, sys

# XXX Recursively building a tree; depth will depend on
# number of instance features.
sys.setrecursionlimit(10000)

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

# Gain threshold for continued splitting.
if len(sys.argv) > 3:
    min_gain = float(sys.argv[3])
else:
    min_gain = 0.05

# Significance threshold for continued splitting.
if len(sys.argv) > 4:
    min_chisquare = float(sys.argv[4])
else:
    # statistic with 1 DOF corresponds to p = 0.1
    # https://stattrek.com/online-calculator/chi-square.aspx
    min_chisquare = 0.211

# Number of features per instance. XXX Should be same for
# all instances.
nfeatures = len(instances[0].features)

def chi_square(pos, neg):
    avg = (pos + neg) / 2
    dpos = pos - avg
    dneg = neg - avg
    return (dpos * dpos + dneg * dneg) / avg;

def count_labels(insts):
    ninsts = len(insts)

    np = 0
    for i in insts:
        np += i.label

    return np, ninsts - np

def entropy(insts):
    np, nn = count_labels(insts)
    ninsts = np + nn

    if np == 0 or nn == 0:
        return 0

    pr_p = np / ninsts
    pr_n = nn / ninsts

    return -pr_p * math.log2(pr_p) - pr_n * math.log2(pr_n)

def majority(insts):
    np, nn = count_labels(insts)
    return int(np > nn)

def split(insts, f):
    splits = [[] for _ in range(2)]
    for i in insts:
        splits[i.features[f]].append(i)
    return splits[1], splits[0]

class DTree(object):
    def __init__(self, insts, used=None, u=None):
        if used is None:
            used = set()
        else:
            used = set(used)

        self.label = None

        if len(used) == nfeatures:
            self.label = majority(insts)
            return

        np, nn = count_labels(insts)
        ninsts = np + nn
        if chi_square(np, nn) < min_chisquare:
            self.label = int(np > nn)
            return

        if u is None:
            u = entropy(insts)
        self.u = u

        best_f = None
        best_gain = None
        best_split = None
        for f in range(nfeatures):
            if f in used:
                continue
            pos, neg = split(insts, f)
            npos = len(pos)
            nneg = len(neg)
            if npos == 0 or nneg == 0:
                continue
            u_pos = entropy(pos)
            u_neg = entropy(neg)
            pr_pos = npos / ninsts
            pr_neg = nneg / ninsts
            gain = u - pr_pos * u_pos - pr_neg * u_neg
            if gain <= 0:
                # XXX Numerical errors can lead to tiny
                # negative gains.
                continue
            if best_gain is None or gain > best_gain:
                best_gain = gain
                best_f = f
                best_split = ((pos, u_pos), (neg, u_neg))

        if best_f is None or best_gain < min_gain:
            self.label = majority(insts)
            return
        
        ps, ns = best_split
        pos, pu = ps
        neg, nu = ns
        used.add(best_f)
        self.f = best_f
        self.pos = DTree(pos, used=used, u=pu)
        self.neg = DTree(neg, used=used, u=nu)
        
    def classify(self, inst):
        if self.label is not None:
            return self.label
        if inst.features[self.f] > 0:
            return self.pos.classify(inst)
        else:
            return self.neg.classify(inst)

# Try training on the training instances and then
# classifying the test instances.  Return the classification
# accuracy.
def try_tc(training, test):
    # Build a decision tree for the training data.
    tree = DTree(training)

    # Score test instances.
    matrix = [[0] * 2 for _ in range(2)]
    for inst in test:
        guess = tree.classify(inst)
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
