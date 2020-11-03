#!/usr/bin/python3

import csv, random, sys

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

# Number of instances.
ninstances = len(instances)

# Number of features per instance. XXX Should be same for
# all instances.
nfeatures = len(instances[0].features)

random.shuffle(instances)

# Split into training and test set.
split = ninstances // 2
training = instances[split:]
ntraining = len(training)
test = instances[:split]
ntest = len(test)

spams = [i for i in training if i.label == 1]
nspams = len(spams)
hams = [i for i in training if i.label == 0]
nhams = len(hams)

prH = nspams / ntraining

def product(vals):
    p = 1
    for v in vals:
        p *= v
    return p

def score_spam(instance):
    # Compute probability of evidence given hypothesis.
    prEH = list()
    for f in range(nfeatures):
        count = 0
        for tr in spams:
            if tr.features[f] == instance.features[f]:
                count += 1
        prEH.append(count / nspams)

    return product(prEH) * prH

def score_ham(instance):
    # Compute probability of evidence given hypothesis.
    prEH = list()
    for f in range(nfeatures):
        count = 0
        for tr in hams:
            if tr.features[f] == instance.features[f]:
                count += 1
        prEH.append(count / nhams)

    return product(prEH) * prH


for inst in test:
    ss = score_spam(inst)
    sh = score_ham(inst)
    print(inst.name, inst.label, ss > sh, ss, sh)
