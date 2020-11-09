#!/usr/bin/python3
# Naive Bayes classification of binary instances.
# Bart Massey

import csv, math, random, sys

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

# Number of features per instance. XXX Should be same for
# all instances.
nfeatures = len(instances[0].features)

def entropy(insts):
    ninsts = len(insts)
    if ninsts == 0:
        return 0

    np = 0
    for i in insts:
        np += i.label
    if np == 0:
        return 0
    nn = ninsts - np
    if nn == 0:
        return 0

    pr_p = np / ninsts
    pr_n = nn / ninsts

    return -pr_p * math.log2(pr_p) - pr_n * math.log2(pr_n)

def majority(insts):
    ninsts = len(insts)

    np = 0
    for i in insts:
        np += i.label
    
    return int(np > ninsts / 2)

def split(insts, f):
    splits = [[] for _ in range(2)]
    for i in insts:
        splits[i.features[f]].append(i)
    return splits[1], splits[0]

class DTree(object):
    def __init__(self, insts, used=None, u=None):
        if used is None:
            used = set()

        self.label = None
        if len(used) == nfeatures:
            self.label = majority(insts)
            return

        if u is None:
            u = entropy(insts)
        self.u = u

        best_f = None
        best_du = None
        best_split = None
        for f in range(nfeatures):
            if f in used:
                continue
            pos, neg = split(insts, f)
            pu = entropy(pos)
            nu = entropy(neg)
            du = u - pu - nu
            if best_du is None or du > best_du:
                best_du = du
                best_f = f
                best_split = ((pos, pu), (neg, nu))
        assert best_f is not None
        
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
