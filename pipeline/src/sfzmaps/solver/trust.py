"""The trust of every reading of a printed code: the chance that a code a reader saw on a print is the district printed
there.

A print (`prints.parquet`) has up to five reads: Sol's, Terra's, the third read, its turned re-check and the turned
pass (READS).  Each different code among them is a reading of the print, one row of the feature table: print by
print, codes by name.  One logistic regression on the rows gives each reading its trust, the chance that its code is
the table's code of the lot under the print (`lab_lot`).  Its 23 features (FEATURES) say which reads name the code
and how sure they were, how the print sits (its height, in a street, a leader line, printed sideways), and, for the
prints found only by the turned pass, how far they are from another print, whether they are in a street and whether
the code is a single letter (easy to find by accident).

There is one fit per quarter of the city.  The fit that holds out quarter q learns only from readings whose lot (the
lot under the print's text, `lab_lot`) has a known code in the April 2000 table as downloaded
(`table.known_and_codes`; the by-eye corrections are never read) and lies outside q.  A reading is held out by the
quarter of that lot (`lots.quarter`, from the lot's centroid), not by the print's own position, so no fit learns
from a code of a lot in its own quarter, and the trust of q's readings is a held-out guess.  The weights are found by
Newton's method with a ridge on each weight (RIDGE, RIDGE_BY_FEATURE).

Writes, under data/solver/labels/:
  readings_{q}.npz  for each quarter q (nw, ne, sw, se), every reading and its trust from the fit that held q out:
                      names        the code vocabulary (str): the table's known codes with Sol's and Terra's reads,
                                   sorted; then the third read's and the re-check's other codes, sorted; then the
                                   turned pass's other codes, sorted.  Every one is in `codes.canon`'s one spelling,
                                   so the table's undashed spellings (NC1) and a print's overlay text after a
                                   district ('NC-2 & FRANCISCO') join their district's name (118 names on the 2000
                                   prints)
                      print_index  int64 per reading: the print's row in prints.parquet
                      code_index   int64 per reading: its code, an index into names
                      trust        float64 per reading: the fitted chance, at least TRUST_FLOOR
                    The solver (`solver/labels.py` and `placement.py`) reads them with the lot under each print,
                    prints.parquet's `lab_lot`.
  trust.json        per quarter: the weights by feature, the readings trained on and held out, and the held-out
                    log loss (against the table's codes)
"""
import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sfzmaps.solver import paths
from sfzmaps.solver.lots import Lots, load_lots

READS = ("sol_code", "terra_code", "third_code", "recheck_code", "turned_code")   # a print's reads, in this order

CONFIDENCE_CAP = 1e-3     # a reader's confidence c enters as -log(max(1 - c, CONFIDENCE_CAP))
DISTANCE_FLOOR_FT = 1.0   # a turned-pass-only print's distance d to the nearest other print enters as
DISTANCE_SCALE_FT = 100.0  # log(max(d, DISTANCE_FLOOR_FT)) - log(DISTANCE_SCALE_FT)
NEWTON_ITERATIONS = 200
NEWTON_TOLERANCE = 1e-8   # the fit stops when no weight moves more than this in a step
LOGIT_CLIP = 40.0         # the log odds are clipped to +-this before the logistic function
TRUST_FLOOR = 1e-6        # the least trust a reading is given
RIDGE = 1e-3              # each weight's ridge (the prior's strength), unless RIDGE_BY_FEATURE says otherwise

FEATURES = (
    "base",
    "Sol names it",
    "Terra names it",
    "the third read names it",
    "every read names it",
    "two or more reads name it",
    "Sol's confidence when Sol names it (log)",
    "the third read's confidence when it names it (log)",
    "print height (log, against the median)",
    "in a street",
    "no Sol read",
    "no Terra read",
    "the third read found no code",
    "the third read saw a leader line",
    "a code Sol reads on some print",
    "the re-check names it",
    "re-checked (printed sideways)",
    "the turned pass names it",
    "no turned-pass read",
    "found only by the turned pass",
    "turned pass only: distance to the nearest other print (log)",
    "turned pass only: in a street",
    "turned pass only: a single letter",
)
# The prints found only by the turned pass are few: their distance and street weights are held near zero, leaving
# them a plain rate; whether the code is a single letter keeps a weak ridge.
RIDGE_BY_FEATURE = {
    "turned pass only: distance to the nearest other print (log)": 1000.0,
    "turned pass only: in a street": 1000.0,
    "turned pass only: a single letter": 1e-3,
}


@dataclass
class FeatureTable:
    """The readings of every print, one row each: the print (`print_index`, its row in prints.parquet), the code
    (`code_index`, into `names`), the features (`values`, float64 (rows, len(FEATURES))), and the lot under the
    print (`lot`, -1 for none) with its table facts: `known` (a single known code), `quarter` (nw 0, ne 1, sw 2,
    se 3; -1 for no lot) and `target` (the reading's code is the lot's code)."""
    print_index: np.ndarray
    code_index: np.ndarray
    values: np.ndarray
    names: list[str]
    lot: np.ndarray
    known: np.ndarray
    quarter: np.ndarray
    target: np.ndarray


def vocabulary(reads: np.ndarray, table_codes) -> list[str]:
    """The code names: the table's known codes with Sol's and Terra's reads, sorted; then the third read's and the
    re-check's other codes, sorted; then the turned pass's other codes, sorted.  `reads`: (prints, READS)."""
    sol, terra, third, recheck, turned = (set(col) - {""} for col in reads.T)
    names = sorted((set(table_codes) - {""}) | sol | terra)
    names += sorted((third | recheck) - set(names))
    names += sorted(turned - set(names))
    return names


def _confidence(c: np.ndarray) -> np.ndarray:
    """A reader's confidence as a feature: -log(max(1 - c, CONFIDENCE_CAP)); 0 where it gave none."""
    c = np.where(np.isfinite(c), c, 0.0)
    return -np.log(np.maximum(1 - c, CONFIDENCE_CAP))


def _turned_only_distance(xy: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """For each print found only by the turned pass (kind 3), log(max(d, DISTANCE_FLOOR_FT)) - log(DISTANCE_SCALE_FT),
    d the distance (ft) to the nearest print of the other kinds; 0 for the others."""
    out = np.zeros(len(xy))
    turned, others = np.flatnonzero(kind == 3), xy[kind < 3]
    if len(turned) and len(others):
        d = np.hypot(xy[turned, None, 0] - others[None, :, 0], xy[turned, None, 1] - others[None, :, 1]).min(1)
        out[turned] = np.log(np.maximum(d, DISTANCE_FLOOR_FT)) - np.log(DISTANCE_SCALE_FT)
    return out


def features(prints: pd.DataFrame, lots: Lots) -> FeatureTable:
    """The feature table of every reading of `prints` (prints.parquet's columns), with the table facts of the lot
    under each print from `lots`."""
    reads = prints[list(READS)].to_numpy(dtype=object)
    names = vocabulary(reads, lots.codes[lots.known])
    index = {name: i for i, name in enumerate(names)}
    print_index, code = [], []
    for j, r in enumerate(reads):
        for c in sorted(set(r) - {""}):
            print_index.append(j)
            code.append(c)
    pj = np.array(print_index, np.int64)
    code = np.array(code, dtype=object)
    r = reads[pj].reshape(-1, len(READS))
    who = r == code[:, None]
    naming, having = who.sum(1), (r != "").sum(1)

    kind = prints["kind"].to_numpy()
    street = prints["street"].to_numpy(bool)
    height = np.abs(prints["box_y1"].to_numpy(float) - prints["box_y0"].to_numpy(float))
    median = np.nanmedian(height) if np.isfinite(height).any() else 1.0
    log_height = np.log(np.where(np.isfinite(height), height, median)) - np.log(median)
    turned_only = kind == 3
    distance = _turned_only_distance(prints[["x", "y"]].to_numpy(float), kind)
    sol_vocabulary = set(reads[:, 0]) - {""}
    only = turned_only[pj]
    columns = {
        "base": np.ones(len(pj)),
        "Sol names it": who[:, 0],
        "Terra names it": who[:, 1],
        "the third read names it": who[:, 2],
        "every read names it": naming == having,
        "two or more reads name it": naming >= 2,
        "Sol's confidence when Sol names it (log)":
            _confidence(prints["sol_confidence"].to_numpy(float))[pj] * who[:, 0],
        "the third read's confidence when it names it (log)":
            _confidence(prints["third_confidence"].to_numpy(float))[pj] * who[:, 2],
        "print height (log, against the median)": log_height[pj],
        "in a street": street[pj],
        "no Sol read": r[:, 0] == "",
        "no Terra read": r[:, 1] == "",
        "the third read found no code": prints["third_found"].to_numpy(bool)[pj] & (r[:, 2] == ""),
        "the third read saw a leader line": prints["third_leader"].to_numpy(bool)[pj],
        "a code Sol reads on some print": np.array([c in sol_vocabulary for c in code], bool),
        "the re-check names it": who[:, 3],
        "re-checked (printed sideways)": prints["recheck_found"].to_numpy(bool)[pj],
        "the turned pass names it": who[:, 4],
        "no turned-pass read": r[:, 4] == "",
        "found only by the turned pass": only,
        "turned pass only: distance to the nearest other print (log)": only * distance[pj],
        "turned pass only: in a street": only * street[pj],
        "turned pass only: a single letter": only * np.array([len(c) == 1 for c in code], bool),
    }
    assert tuple(columns) == FEATURES
    values = np.column_stack([np.asarray(v, float) for v in columns.values()]).reshape(len(pj), len(FEATURES))

    lot = prints["lab_lot"].to_numpy(np.int64)[pj]
    has = lot >= 0
    known = np.zeros(len(pj), bool)
    known[has] = lots.known[lot[has]]
    quarter = np.full(len(pj), -1, np.int8)
    quarter[has] = lots.quarter[lot[has]]
    target = known.copy()
    target[known] = code[known] == lots.codes[lot[known]]
    return FeatureTable(pj, np.array([index[c] for c in code], np.int64), values, names, lot, known, quarter, target)


def _logistic(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -LOGIT_CLIP, LOGIT_CLIP)))


def newton_fit(X: np.ndarray, y: np.ndarray, ridge) -> np.ndarray:
    """The weights of a logistic regression of y (0/1) on X: the maximum of the log likelihood less
    sum(ridge * w ** 2) / 2 (`ridge` one number or one per column), by Newton's method from zero, at most
    NEWTON_ITERATIONS steps, stopping when no weight moves more than NEWTON_TOLERANCE."""
    w = np.zeros(X.shape[1])
    ridge = np.broadcast_to(np.asarray(ridge, float), w.shape)
    for _ in range(NEWTON_ITERATIONS):
        p = _logistic(X @ w)
        hessian = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(hessian + np.diag(ridge), X.T @ (y - p) - ridge * w)
        w += step
        if np.abs(step).max() < NEWTON_TOLERANCE:
            break
    return w


def ridges() -> np.ndarray:
    """Each feature's ridge: RIDGE, or RIDGE_BY_FEATURE's."""
    return np.array([RIDGE_BY_FEATURE.get(name, RIDGE) for name in FEATURES])


def training_rows(table: FeatureTable, q: str) -> np.ndarray:
    """The readings the fit holding out quarter q learns from: the lot under the print's text has a known code and
    is outside q (the lot's quarter, not the print's position)."""
    return table.known & (table.quarter != paths.QUARTERS.index(q))


def fit(table: FeatureTable, q: str) -> np.ndarray:
    """The weights (one per FEATURES) of the fit that holds out quarter q: trained on `training_rows`, the target
    that the reading's code is its lot's code."""
    rows = training_rows(table, q)
    return newton_fit(table.values[rows], table.target[rows].astype(float), ridges())


def trust(table: FeatureTable, weights: np.ndarray) -> np.ndarray:
    """Each reading's trust under `weights`: the logistic of its features, at least TRUST_FLOOR."""
    return _logistic(table.values @ weights).clip(TRUST_FLOOR, 1)


def log_loss(chance: np.ndarray, target: np.ndarray) -> float:
    """The mean log loss of chances (clipped to [TRUST_FLOOR, 1 - TRUST_FLOOR]) against 0/1 targets."""
    p = np.clip(chance, TRUST_FLOOR, 1 - TRUST_FLOOR)
    return float(-np.mean(np.where(target, np.log(p), np.log(1 - p)))) if len(p) else float("nan")


def load_prints() -> pd.DataFrame:
    """Read prints.parquet (made by `sfzmaps solver prints`)."""
    f = paths.PRINTS_DIR / "prints.parquet"
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver prints`")
    return pd.read_parquet(f)


def build_readings() -> dict:
    """Fit the trust once per held-out quarter and write readings_{q}.npz and trust.json into LABELS_DIR.  Returns
    trust.json's summary and the seconds taken."""
    t0 = time.perf_counter()
    found = load_prints()
    table = features(found, load_lots())
    paths.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"prints": len(found), "readings": len(table.print_index), "names": len(table.names), "quarters": {}}
    for qi, q in enumerate(paths.QUARTERS):
        weights = fit(table, q)
        chance = trust(table, weights)
        np.savez_compressed(paths.LABELS_DIR / f"readings_{q}.npz", names=np.array(table.names),
                            print_index=table.print_index, code_index=table.code_index, trust=chance)
        held = table.known & (table.quarter == qi)
        summary["quarters"][q] = {
            "trained_on": int(training_rows(table, q).sum()), "held_out": int(held.sum()),
            "held_out_log_loss": round(log_loss(chance[held], table.target[held]), 4),
            "weights": {name: round(float(w), 4) for name, w in zip(FEATURES, weights)}}
    (paths.LABELS_DIR / "trust.json").write_text(json.dumps(summary, indent=1) + "\n")
    return {**summary, "seconds": round(time.perf_counter() - t0, 1)}
