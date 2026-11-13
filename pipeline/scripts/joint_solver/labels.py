"""The printed labels and their readings, from one holdout's print set.

A label can have several readings (code, trust).  A label printed in a street also speaks for the lot across the
street, at a lower trust (its own label there, a 'far-side copy')."""
from dataclasses import dataclass

import numpy as np

from .codes import canon, is_overlay
from .data import INPUTS


@dataclass
class Readings:
    lot: np.ndarray        # the lot a reading speaks for
    code: np.ndarray       # index into names
    trust: np.ndarray      # the reader's trust in this reading, 0..1
    label: np.ndarray      # which printed label it comes from
    far: np.ndarray        # True for a street label's far-side copy
    names: list            # code names in the one spelling; a reading's code is the first name with its spelling


def print_set(q):
    return f"cwfwx{q}"


def readings(q):
    """every reading of holdout q's print set.  Overlay (SUD) names are dropped."""
    P = print_set(q)
    R = np.load(INPUTS / f"lotclf6_{P}.npz")
    raw_names = list(R["names"].astype(str))
    lab, code, trust = R["cj"], R["ck"], R["cr"].astype(float)
    under = np.load(INPUTS / f"labread2_{P}.npz")["lab_lot"]
    S = np.load(INPUTS / f"labside_{P}.npz")
    ratio = min(float(S["rB_st"]) / float(S["rA_st"]), 1.0)
    across = S["onst"] & ~S["isfw"] & (S["lotB"] >= 0)
    r = np.flatnonzero(across[lab])
    lot = np.r_[under[lab], S["lotB"][lab[r]]]
    code = np.r_[code, code[r]]
    trust = np.r_[trust, trust[r] * ratio]
    label = np.r_[lab, lab[r]]
    far = np.r_[np.zeros(len(lab), bool), np.ones(len(r), bool)]
    keep = (lot >= 0) & ~np.array([is_overlay(n) for n in raw_names])[code]
    names = [canon(n) for n in raw_names]
    first = {}
    for i, n in enumerate(names):
        first.setdefault(n, i)
    code = np.array([first[names[k]] for k in code], int)
    return Readings(lot[keep], code[keep], trust[keep], label[keep], far[keep], names)


def top_readings(q):
    """(label, lot under its text, top reading's code) for every label of holdout q's print set whose top reading is
    not an overlay name"""
    P = print_set(q)
    R = np.load(INPUTS / f"lotclf6_{P}.npz")
    raw_names = R["names"].astype(str)
    lab, code, trust = R["cj"], R["ck"], R["cr"].astype(float)
    under = np.load(INPUTS / f"labread2_{P}.npz")["lab_lot"]
    o = np.lexsort((-trust, lab))
    first = np.r_[True, lab[o][1:] != lab[o][:-1]]
    out = []
    for j, k in zip(lab[o][first], code[o][first]):
        if j < len(under) and under[j] >= 0 and not is_overlay(raw_names[k]):
            out.append((int(j), int(under[j]), canon(raw_names[k])))
    return out
