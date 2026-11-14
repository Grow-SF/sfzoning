"""The yardstick: the April 2000 lot table, with the lots where the map was judged by eye to print our code counted as ours.
truth_codes(g) -> (corrected codes, mask of corrected lots)"""
import csv

import numpy as np

import truth
from lotcode_lib import canon

K = truth.D.parent / "k8"


def truth_codes(g):
    code = np.array([canon(c) for c in g["code"].values], dtype=object)
    bl = {b: i for i, b in enumerate(g["blklot"].astype(str).values)}
    fixed = np.zeros(len(g), bool)
    for r in csv.DictReader(open(K / "judge2" / "final2.csv")):
        if r["verdict"] != "A":
            continue
        for b in r["blklots"].split():
            if b in bl:
                code[bl[b]] = canon(r["A"])
                fixed[bl[b]] = True
    return code.astype(str), fixed
