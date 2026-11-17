"""The printed labels and their readings, as one held-out quarter's fits see them.

A label is a print (a row of prints.parquet).  It can have several readings (code, trust): readings_{q}.npz, from
the trust fit that held quarter q out.  A reading speaks for the lot under the label's text (prints.parquet
`lab_lot`).  A label printed on an ordinary street also speaks for the lot across the street (streetside_{q}.npz
`lotB`), at its trust x min(rB_st / rA_st, 1): a far-side copy."""
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from sfzmaps.solver import runs
from sfzmaps.solver.codes import canon, is_overlay
from sfzmaps.solver.trust import load_prints


class Source(NamedTuple):
    """Where one set of labels is read from: the prints table and the folder of readings_{q}.npz / streetside_{q}.npz."""
    prints_file: Path
    labels_dir: Path


def solve_source() -> Source:
    """The labels the solve reads (`runs.SOLVE_*`)."""
    return Source(runs.SOLVE_PRINTS_FILE, runs.SOLVE_LABELS_DIR)


def scoring_source() -> Source:
    """The labels the score counts (`runs.SCORING_*`): fixed whatever the solve reads."""
    return Source(runs.SCORING_PRINTS_FILE, runs.SCORING_LABELS_DIR)


@dataclass
class Readings:
    lot: np.ndarray        # the lot a reading speaks for
    code: np.ndarray       # index into names
    trust: np.ndarray      # the reader's trust in this reading, 0..1
    label: np.ndarray      # which printed label it comes from (a prints.parquet row)
    far: np.ndarray        # True for a street label's far-side copy
    names: list            # code names in the one spelling; a reading's code is the first name with its spelling


def _load(q: str, source: Source):
    """readings_{q}.npz's names, print, code and trust per reading, and the lot under each print's text."""
    f = source.labels_dir / f"readings_{q}.npz"
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver labels`")
    R = np.load(f)
    under = load_prints(source.prints_file)["lab_lot"].to_numpy(np.int64)
    return (list(R["names"].astype(str)), R["print_index"].astype(np.int64), R["code_index"].astype(np.int64),
            R["trust"].astype(float), under)


def readings(q: str, vocabulary=None, source: Source | None = None) -> Readings:
    """Every reading of holdout q, with the far-side copies, from `source` (default: the solve's labels).  Overlay
    (SUD) names are dropped; so are readings naming no code in vocabulary (when given: overlay text such as
    'NORTH OF', 'RES.' is not a zoning code)."""
    source = source or solve_source()
    raw_names, lab, code, trust, under = _load(q, source)
    S = np.load(source.labels_dir / f"streetside_{q}.npz")
    rA, rB = float(S["rA_st"]), float(S["rB_st"])
    ratio = min(rB / rA, 1.0) if rA > 0 else 0.0
    across = S["onst"] & ~S["isfw"] & (S["lotB"] >= 0)
    r = np.flatnonzero(across[lab])
    lot = np.r_[under[lab], S["lotB"][lab[r]]]
    code = np.r_[code, code[r]]
    trust = np.r_[trust, trust[r] * ratio]
    label = np.r_[lab, lab[r]]
    far = np.r_[np.zeros(len(lab), bool), np.ones(len(r), bool)]
    keep = (lot >= 0) & ~np.array([is_overlay(n) for n in raw_names], bool)[code]
    names = [canon(n) for n in raw_names]
    first = {}
    for i, n in enumerate(names):
        first.setdefault(n, i)
    code = np.array([first[names[k]] for k in code], int)
    if vocabulary is not None:
        keep &= np.array([names[k] in vocabulary for k in code], bool)
    return Readings(lot[keep], code[keep], trust[keep], label[keep], far[keep], names)


def top_readings(q: str, source: Source) -> list[tuple[int, int, str]]:
    """(label, lot under its text, top reading's code) for every label of holdout q in `source` whose top reading (its
    most trusted; the first in row order on a tie) is not an overlay name.  `source` is required: a score takes
    `scoring_source()`, never the solve's."""
    raw_names, lab, code, trust, under = _load(q, source)
    o = np.lexsort((-trust, lab))
    first = np.r_[True, lab[o][1:] != lab[o][:-1]]
    out = []
    for j, k in zip(lab[o][first], code[o][first]):
        if j < len(under) and under[j] >= 0 and not is_overlay(raw_names[k]):
            out.append((int(j), int(under[j]), canon(raw_names[k])))
    return out
