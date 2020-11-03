#!/usr/bin/python3

import csv, sys

class Instance(object):
    def __init__(self, row):
        self.name = row[0]
        self.label = int(row[1])
        self.features = [int(f) for f in row[2:]]

with open(sys.argv[1], "r") as f:
    reader = csv.reader(f)
    instances = [Instance(row) for row in reader]

nfeatures = len(instances[0].features)

print(f"n: {len(instances)}")
print(f"c: {sum([i.label for i in instances])}")
for i in range(nfeatures):
    print(f"f[{i}]: {sum([j.features[i] for j in instances])}")
