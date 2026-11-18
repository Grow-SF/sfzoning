"""Shared pieces of spikes S0 and S1: how much the 2000 solver's map depends on its label reader.

A measurement, not shipped code (brief: .superpowers/sdd/2026-11-16-search-move-and-reader-spikes/task-S01-brief.md).

    uv run python scripts/spikes/reader_spike.py solve RUN Q     # one quarter's solve (a worker; see `run_jobs`)
    uv run python scripts/spikes/reader_spike.py check           # the plumbing checks (no solve)

A run is named by what its readings are (`kind_of`):
  control              the stage's readings (data/solver/labels/readings_{q}.npz) through this path: must give the
                       stage's `final` exactly
  s0                   the template reader's posterior as the only reading, used directly (`posterior_P`)
  s0-mix               the same posterior through the energy function's own mixing (`posterior_P_mixed`): a check
                       on how the posterior enters
  m<M>-r<R>-s<SEED>    the stage's readings degraded: M% of the labels missed, R% misread (`degrade`)
control, s0 and s0-mix are written under data/solver/experiments/spike-s0/runs/, the degraded runs under
.../spike-s1/runs/: {run}-{q}.npz (as `solve.solve_quarter` writes it), {run}-{q}.json (its log, the search's
energy, the sha256 of the solver files it ran), and the join {run}.npz (`solve.join`).

Every solve is the stage's own: the stage's walls (walls_{q}.npz), the stage's placement (data/solver/solve/
placement.npz, `placement.load_options`), the stage's `solve.solve_joint` with its defaults.  Only the readings
differ.  For s0 the label costs U come from `label_costs_from` (a copy of `energy.label_costs` that takes each
label's P instead of computing it from trusts), handed to `solve_joint` by swapping `energy.label_costs` for the
one call; `check` shows that `label_costs_from` with the function's own P gives the function's U bit for bit.

The measures are `report.six_measures` with no lot left out (the report's every-lot column: each run's data-error list
comes from its own holes, so runs compare on that column).  The labels-right measure's labels are always the stage's
top readings (`measures.Table`), whatever the run read: one measure for every run.

Each solve process is pinned to one thread (the BLAS variables set before numpy loads).  `run_jobs` runs at most
MAX_PROCS at once and starts one only while the 1-minute load is below LOAD_START."""
import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENCV_FOR_THREADS_NUM": "1", "NUMEXPR_NUM_THREADS": "1"}.items():
    os.environ.setdefault(_k, _v)

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from sfzmaps.solver import energy as E  # noqa: E402
from sfzmaps.solver import labels as labels_mod  # noqa: E402
from sfzmaps.solver import paths, report, runs, solve, threads  # noqa: E402
from sfzmaps.solver.city import city, load_run, run_file, save_run, wall_odds  # noqa: E402
from sfzmaps.solver.codes import canon, is_overlay, table_vocabulary  # noqa: E402
from sfzmaps.solver.labels import Readings, scoring_source, top_readings  # noqa: E402
from sfzmaps.solver.measures import Table, data_error_lots  # noqa: E402
from sfzmaps.solver.placement import load_options  # noqa: E402
from sfzmaps.solver.trust import READS, load_prints  # noqa: E402

QUARTERS = paths.QUARTERS
S0_DIR = runs.EXPERIMENTS_DIR / "spike-s0"
S1_DIR = runs.EXPERIMENTS_DIR / "spike-s1"
INPUTS = S0_DIR / "inputs"          # the code reader's records, copied from the scratchpad (df61853's reader)
RECORDS = ("full_fix1held.records", "full_fix1rest.records")
PRIOR = "prior3.json"               # the prior the records were scored with (full_fix1.sh)
MAX_PROCS = 8                       # quarter solves at once (one thread each)
MIN_PROCS = 4                       # spike solves kept running whatever the load (other agents' load included)
START_EVERY = 20                    # s between starts, so the 1-minute load can show each before the next
LOAD_START = 19.0                   # a solve starts only while the 1-minute load is below this (target: near 20)
BOOT_DRAWS = 1000
BOOT_SEED = 20261116
# The solver every spike solve runs: a copy of the package at PINNED_COMMIT (`git archive PINNED_COMMIT
# pipeline/src/sfzmaps`, extracted under S1_DIR/code/<commit>/), first on each solve process's path, so that edits
# another agent makes to src/ mid-run can't reach a solve.  Its search.py, solve.py and mincut.py are byte for byte
# the ones the stage's final was solved with (their sha256 in every solve's json).  The first solves (control, s0,
# s0-mix) ran from src/ before it changed (their json's hashes say so).
PINNED_COMMIT = "ed4cf98"
PINNED_SRC = S1_DIR / "code" / PINNED_COMMIT / "pipeline" / "src"
UNPINNED_RUNS = ("control", "s0", "s0-mix")
# The final every comparison is against: solver-solve's output at 44dffb1, solved with ed4cf98's search, the search
# every spike solve runs (R6).  The stage was rebuilt at 7cce81a (11-16, 18:04) with the compound fusion move on, so
# data/solver/solve no longer holds it; this is its copy from the DVC cache (dvc.lock at 44dffb1: data/solver/solve
# 65e317b6392a98a43c8747b4bdeb6244.dir), each file checked by its md5 there.
BASELINE_DIR = S0_DIR / "final-44dffb1"
BASELINE_MD5 = {"data_errors.npz": "c55765a5055501dbc08589dc931a2254", "placement.npz": "5b2b2e4f14176a0ca18d90c9eaf7b5c2",
                "report.json": "40ca9eed534be6edd15060e005c7a085", "runs/final.npz": "51e49dcc967dfac02d758eec5795cef0",
                "runs/final-nw.npz": "271197d3f3fcd4fa1024c77115b8012c",
                "runs/final-ne.npz": "6e22db442b48db31a5495cb4b0f229e2",
                "runs/final-sw.npz": "9b02ac294b057d089db2d0ef48580918",
                "runs/final-se.npz": "92a82f147f9b007b2f4ed5f96a13ecc6"}


@lru_cache(maxsize=1)
def baseline_dir() -> Path:
    """BASELINE_DIR, every file checked against its md5 in the DVC cache's record of 44dffb1's solve."""
    for rel, md5 in BASELINE_MD5.items():
        f = BASELINE_DIR / rel
        if not f.is_file() or hashlib.md5(f.read_bytes()).hexdigest() != md5:
            raise FileNotFoundError(f"{f}: missing or not 44dffb1's (md5 {md5}); copy it from the DVC cache")
    return BASELINE_DIR
SOLVER_FILES = ("search.py", "solve.py", "energy.py", "mincut.py", "labels.py", "placement.py", "city.py")
# the uniform share of the energy function's most confident P: a reading of trust 1 - 1e-3 (its clip) is right
# with q = expit(a + b logit(0.999)); the rest, 1 - q, is spread evenly.  The posterior is floored by the same
# share, so it is never surer than the function's surest reading.
_A, _B = E.TOP_READING_FIT
FLOOR = 1.0 - 1.0 / (1.0 + math.exp(-(_A + _B * math.log(0.999 / 0.001))))


# --- the readings --------------------------------------------------------------------------------------------------

@dataclass
class Raw:
    """readings_{q}.npz as `labels._load` gives it: the raw names, and per reading its print, code (into names) and
    trust; and the lot under each print's text."""
    names: list
    lab: np.ndarray
    code: np.ndarray
    trust: np.ndarray
    under: np.ndarray


def stage_raw(q: str) -> Raw:
    names, lab, code, trust, under = labels_mod._load(q)
    return Raw(list(names), lab.copy(), code.copy(), trust.copy(), under)


def readings_from(q: str, raw: Raw, vocabulary) -> Readings:
    """`labels.readings(q, vocabulary)` from `raw` instead of the file (the same lines; `check` compares them)."""
    raw_names, lab, code, trust, under = raw.names, raw.lab, raw.code, raw.trust, raw.under
    S = np.load(paths.LABELS_DIR / f"streetside_{q}.npz")
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


def far_ratio(q: str) -> float:
    """The far-side copies' trust factor of holdout q (as `labels.readings`)."""
    S = np.load(paths.LABELS_DIR / f"streetside_{q}.npz")
    rA, rB = float(S["rA_st"]), float(S["rB_st"])
    return min(rB / rA, 1.0) if rA > 0 else 0.0


@lru_cache(maxsize=1)
def vocabulary() -> frozenset:
    return frozenset(table_vocabulary(city().lots))


def charged_labels() -> set:
    """The labels (prints rows) the stage's energy charges in any quarter."""
    voc = vocabulary()
    return {int(x) for q in QUARTERS for x in np.unique(labels_mod.readings(q, voc).label)}


# --- the label costs -------------------------------------------------------------------------------------------------

def groups_of(rd: Readings) -> list[tuple[int, bool, int, np.ndarray]]:
    """(label, far, own lot, T) per charged label as `energy.label_costs` groups them, in its order: T[code] the
    label's largest trust in that code."""
    K = len(rd.names)
    groups = {}
    for r in range(len(rd.lot)):
        groups.setdefault((int(rd.label[r]), bool(rd.far[r])), []).append(r)
    out = []
    for (label, far), rows in groups.items():
        T = np.zeros(K)
        for r in rows:
            T[rd.code[r]] = max(T[rd.code[r]], rd.trust[r])
        out.append((label, far, int(rd.lot[rows[0]]), T))
    return out


def function_P(T: np.ndarray, K: int) -> np.ndarray:
    """The P(code) `energy.label_costs` makes from a label's trusts (its lines, unchanged)."""
    top = int(np.argmax(T))
    t = np.clip(T[top], 1e-3, 1 - 1e-3)
    q = 1 / (1 + np.exp(-(E.TOP_READING_FIT[0] + E.TOP_READING_FIT[1] * np.log(t / (1 - t)))))
    rest = T.copy()
    rest[top] = 0.0
    rest = (rest + 1.0 / K) / (rest.sum() + 1.0)
    return q * np.eye(K)[top] + (1 - q) * rest


def label_costs_from(n_lots: int, K: int, charges, options=None, weight=1.0):
    """`energy.label_costs` with each label's P given: charges are (label, far, own lot, P), in the order to charge
    them.  The same lines otherwise (the placement's options, the far-side copies on their own lot)."""
    U = np.zeros((n_lots, K))
    terms = 0
    for label, far, own, P in charges:
        where = [(own, 1.0)]
        if options is not None and not far and label in options:
            stay = [o for o in options[label] if o[2]]
            if stay and stay[0][0] == own:
                where = [(o[0], o[1]) for o in options[label]]
        for lot, pi in where:
            U[lot] -= weight * np.log(pi * P + (1 - pi) / K)
            terms += 1
    return U, len(charges), terms


@contextmanager
def label_costs_given(U: np.ndarray, n_labels: int, n_terms: int):
    """`energy.label_costs` answers with U for the one solve inside (solve_joint calls it once)."""
    original = E.label_costs
    calls = []

    def given(n_lots, n_codes, rd, options=None, weight=1.0):
        if (n_lots, n_codes) != U.shape or weight != 1.0 or calls:
            raise ValueError("label_costs_given: called with another shape or weight, or twice")
        calls.append(1)
        return U.copy(), n_labels, n_terms

    E.label_costs = given
    try:
        yield
    finally:
        E.label_costs = original
    if not calls:
        raise ValueError("label_costs_given: solve_joint never asked for the label costs")


# --- the template reader's posterior ---------------------------------------------------------------------------------

def load_records() -> dict:
    """(prints row, point) -> (key, kind, point label?, Evidence, seconds), both record files."""
    out = {}
    for name in RECORDS:
        with open(INPUTS / name, "rb") as fh:
            while True:
                try:
                    rec = pickle.load(fh)
                except EOFError:
                    break
                if rec[0] in out:
                    raise ValueError(f"record {rec[0]} twice")
                out[rec[0]] = rec
    return out


def posteriors() -> tuple[dict, float]:
    """prints row -> the reader's posterior over its codes, "none" and "other" at the prior's temperature (32.7, the
    one both files hold), the default none and other masses (`posterior.Constants()`; Task 4 never fitted them); and
    that temperature.  The record of an answered case on no print (row -1) is left out (no solver label).  Each
    record's point is checked against prints.parquet's."""
    from sfzmaps.solver.codereader import posterior as P
    prior = P.Prior.load(INPUTS / PRIOR)
    pr = load_prints()
    out = {}
    for (row, point), rec in load_records().items():
        if row < 0:
            continue
        if (float(pr["page_x"].iloc[row]), float(pr["page_y"].iloc[row])) != tuple(point):
            raise ValueError(f"record of print {row} is at {point}, prints.parquet's elsewhere")
        if row in out:
            raise ValueError(f"two records for print {row}")
        out[row] = P.posterior_of(rec[3], prior, P.Constants())
    return out, prior.pose.temperature


def posterior_codes(post: dict, names: list) -> np.ndarray:
    """The posterior's code masses on `names` (rd.names, codes in canon spelling; a code at its first name)."""
    first = {}
    for i, n in enumerate(names):
        first.setdefault(n, i)
    P = np.zeros(len(names))
    for h, p in post.items():
        if h in ("none", "other"):
            continue
        P[first[canon(h)]] += p
    return P


def posterior_P(post: dict, names: list, ratio: float = 1.0) -> np.ndarray:
    """The label's P from the reader's posterior, used directly: each code its posterior mass (x `ratio` for a far-
    side copy, as `labels.readings` scales a far copy's trust); the mass left ("none", "other", and on a far copy what
    the ratio took off) spread evenly over the K names, as the function spreads its rest; then mixed with 1/K at
    FLOOR, the uniform share of the function's surest P, so no code is impossible."""
    K = len(names)
    P = ratio * posterior_codes(post, names)
    P = P + (1.0 - P.sum()) / K
    return (1.0 - FLOOR) * P + FLOOR / K


def posterior_P_mixed(post: dict, names: list, ratio: float = 1.0) -> np.ndarray:
    """The check variant: the posterior through the function's own mixing.  q = the top code's mass (x ratio), the
    rest = the other codes' masses (x ratio), and P = q at the top + (1 - q) x (rest + 1/K) / (rest.sum() + 1), as
    `energy.label_costs` does with q from the trust fit and the rest from the other readings' trusts."""
    K = len(names)
    T = ratio * posterior_codes(post, names)
    top = int(np.argmax(T))
    q = float(np.clip(T[top], 1e-3, 1 - 1e-3))
    rest = T.copy()
    rest[top] = 0.0
    rest = (rest + 1.0 / K) / (rest.sum() + 1.0)
    return q * np.eye(K)[top] + (1 - q) * rest


def s0_problem(q: str, mixed: bool = False):
    """Quarter q's S0 readings: the charges (the stage's charged labels and far-side copies, in its order, each with
    the reader's P), the pseudo-readings the tie-break's paths read (each charge's top code at its lot), and how
    many charged labels have no record (left uncharged: a miss)."""
    rd = labels_mod.readings(q, vocabulary())
    post, _ = posteriors()
    ratio = far_ratio(q)
    make = posterior_P_mixed if mixed else posterior_P
    charges, missing = [], set()
    for label, far, own, _ in groups_of(rd):
        if label not in post:
            missing.add(label)
            continue
        charges.append((label, far, own, make(post[label], rd.names, ratio if far else 1.0)))
    top = [int(np.argmax(posterior_codes(post[c[0]], rd.names))) for c in charges]
    pseudo = Readings(np.array([c[2] for c in charges], int), np.array(top, int), np.ones(len(charges)),
                      np.array([c[0] for c in charges], int), np.array([c[1] for c in charges], bool), rd.names)
    return charges, pseudo, len(missing)


# --- S1: degraded readings -------------------------------------------------------------------------------------------

RUN_RE = re.compile(r"^m(?P<m>[0-9.]+)-r(?P<r>[0-9.]+)-s(?P<seed>[0-9]+)$")


def s1_name(m: float, r: float, seed: int) -> str:
    """A degraded run's name, rates in percent."""
    return f"m{m:g}-r{r:g}-s{seed}"


def eligible(raw: Raw) -> np.ndarray:
    """Per reading: kept by `labels.readings` (a lot under the print, no overlay name, a table code)."""
    voc = vocabulary()
    ok_name = np.array([(not is_overlay(n)) and canon(n) in voc for n in raw.names], bool)
    return ok_name[raw.code] & (raw.under[raw.lab] >= 0)


def confusion() -> dict:
    """The misread model: code -> (replacement codes sorted, their chances), from the disagreements among the saved
    readers.  Each print whose reads (prints.parquet's sol, terra, third, recheck and turned codes, in `canon`'s
    spelling, table codes only) name two or more codes counts each ordered pair once.  Codes no reader ever confused
    get none here: `replacement` falls back to the codes printed in the label's quarter."""
    voc = vocabulary()
    pr = load_prints()
    pairs = Counter()
    for row in pr[list(READS)].itertuples(index=False):
        cs = sorted({canon(str(x)) for x in row if isinstance(x, str) and x} & voc)
        for a in cs:
            for b in cs:
                if a != b:
                    pairs[(a, b)] += 1
    model = {}
    for (a, b), n in sorted(pairs.items()):
        model.setdefault(a, {})[b] = n
    return {a: (sorted(d), np.array([d[k] for k in sorted(d)], float) / sum(d.values())) for a, d in model.items()}


def quarter_codes() -> dict:
    """The fallback: per quarter k, the codes printed there (each charged label's top reading in the readings of
    the fit that held its lot's quarter out, counted once per label) and their shares."""
    C = city()
    voc = vocabulary()
    out = {}
    for k, q in enumerate(QUARTERS):
        n = Counter(code for _, lot, code in top_readings(q, scoring_source())
                if C.quarter[lot] == k and code in voc)
        out[k] = (sorted(n), np.array([n[c] for c in sorted(n)], float))
    return out


def replacement(code: str, u: float, quarter: int, model: dict, fallback: dict) -> tuple[str, bool]:
    """The code a misread of `code` reads, by the inverse of the model's chances at u (the fallback's, without
    `code`, when the model has none for it); and whether the fallback gave it."""
    if code in model:
        names, p = model[code]
        fell = False
    else:
        names, n = fallback[quarter]
        keep = np.array([c != code for c in names])
        names, p = [c for c, k in zip(names, keep) if k], n[keep] / n[keep].sum()
        fell = True
    i = min(int(np.searchsorted(np.cumsum(p), u, side="right")), len(names) - 1)
    return names[i], fell


def draws(seed: int, n_prints: int) -> np.ndarray:
    """Three uniforms per print (miss, misread, replacement), the same for every quarter and every rate of a
    seed: a label missed at 5% is missed at 10% too (common random numbers)."""
    return np.random.default_rng(seed).random((3, n_prints))


def degrade(raw: Raw, q: str, m: float, r: float, seed: int, model=None, fallback=None, detail: bool = False):
    """The readings of holdout q with the labels missed and misread (rates as fractions of the charged labels,
    `eligible`): a missed label loses every reading; a misread label's top reading (its largest trust among the kept
    readings; the first on a tie) names the replacement code (`replacement`), its trust kept.  A missed label is
    not also misread, so the misread share among all labels is r (1 - m).  Returns (raw, info), and with `detail`
    also {misread label: (its top code, the code it now reads)}."""
    C = city()
    model = confusion() if model is None else model
    fallback = quarter_codes() if fallback is None else fallback
    u = draws(seed, len(raw.under))
    ok = eligible(raw)
    labels = np.unique(raw.lab[ok])
    missed = set(labels[u[0, labels] < m].tolist())
    misread = set(labels[(u[1, labels] < r) & (u[0, labels] >= m)].tolist())
    first_raw = {}
    for i, n in enumerate(raw.names):
        first_raw.setdefault(canon(n), i)
    code = raw.code.copy()
    fell = 0
    swaps = {}
    for label in sorted(misread):
        rows = np.flatnonzero((raw.lab == label) & ok)
        top = rows[int(np.argmax(raw.trust[rows]))]
        new, fb = replacement(canon(raw.names[code[top]]), float(u[2, label]), int(C.quarter[raw.under[label]]),
                              model, fallback)
        swaps[int(label)] = (canon(raw.names[code[top]]), new)
        code[top] = first_raw[new]
        fell += fb
    keep = ~np.isin(raw.lab, list(missed))
    out = Raw(raw.names, raw.lab[keep], code[keep], raw.trust[keep], raw.under)
    info = {"labels": int(len(labels)), "missed": len(missed), "misread": len(misread), "fallback": fell}
    return (out, info, swaps) if detail else (out, info)


# --- S1 with unsure misreads (fix round 1, I4) -------------------------------------------------------------------------

UNSURE_SUFFIX = "-u"


@lru_cache(maxsize=1)
def reader_misread_masses() -> tuple:
    """The template reader's own misreads on the owner's answered labels the solver charges (its top a code, not
    the answer): (its mass on that wrong code, its mass on the answer), in print order.  65 pairs: the wrong code's
    mass averages 0.46, the answer's 0.15."""
    post, _ = posteriors()
    charged = charged_labels()
    out = []
    for lab in sorted(pickle.load(open(INPUTS / "labs.pkl", "rb")), key=lambda lab: lab.index):
        if not lab.answer or lab.index not in charged:
            continue
        p = post[lab.index]
        top = max(p, key=p.get)
        truth = canon(lab.answer)
        if top in ("none", "other") or canon(top) == truth:
            continue
        out.append((float(p[top]), float(sum(v for h, v in p.items() if h not in ("none", "other")
                                              and canon(h) == truth))))
    return tuple(out)


def unsure_P(new: str, old: str, wrong: float, right: float, names: list, ratio: float = 1.0) -> np.ndarray:
    """A misread label's P as the template reader misreads: the wrong code `new` at mass `wrong`, the code it
    replaced (the label's top reading) at `right`, the rest spread evenly over the K names; a far-side copy's masses
    times `ratio`, what that takes off spread evenly too; then the floor, as `posterior_P`."""
    K = len(names)
    first = {}
    for i, n in enumerate(names):
        first.setdefault(n, i)
    P = np.zeros(K)
    P[first[new]] += ratio * wrong
    P[first[old]] += ratio * right
    P = P + (1.0 - P.sum()) / K
    return (1.0 - FLOOR) * P + FLOOR / K


def unsure_problem(q: str, m: float, r: float, seed: int):
    """Quarter q of an S1 point with unsure misreads: the same misses, misread labels and replacement codes as the
    sure run of that point and seed (the same draws), the readings to the tie-break's paths as there; each misread
    label's own-lot and far-side charges get `unsure_P`, its (wrong, right) masses one of the reader's own misreads
    (`reader_misread_masses`, picked by a fourth uniform per print from the seed's own generator); every other
    charge keeps the function's P from its readings (`function_P`)."""
    raw, info, swaps = degrade(stage_raw(q), q, m, r, seed, detail=True)
    rd = readings_from(q, raw, vocabulary())
    K = len(rd.names)
    pairs = reader_misread_masses()
    u4 = np.random.default_rng([seed, 4]).random(len(raw.under))
    ratio = far_ratio(q)
    charges = []
    for label, far, own, T in groups_of(rd):
        if label in swaps:
            old, new = swaps[label]
            wrong, right = pairs[min(int(u4[label] * len(pairs)), len(pairs) - 1)]
            charges.append((label, far, own, unsure_P(new, old, wrong, right, rd.names, ratio if far else 1.0)))
        else:
            charges.append((label, far, own, function_P(T, K)))
    return charges, rd, info


# --- solving ---------------------------------------------------------------------------------------------------------

def kind_of(run: str) -> tuple[str, Path]:
    if run in ("control", "control-pinned", "s0", "s0-mix"):
        return run.split("-pinned")[0], S0_DIR
    if run == "control-s0path":             # the stage's readings through S0's path (fix round 1, M1)
        return "control-s0path", S0_DIR
    if RUN_RE.match(run):
        return "s1", S1_DIR
    if run.endswith(UNSURE_SUFFIX) and RUN_RE.match(run[:-len(UNSURE_SUFFIX)]):
        return "s1u", S1_DIR
    raise ValueError(f"{run!r}: not a spike run (control, s0, s0-mix, m<M>-r<R>-s<SEED>[-u])")


def solver_hashes() -> dict:
    src = Path(solve.__file__).parent
    return {f: hashlib.sha256((src / f).read_bytes()).hexdigest()[:16] for f in SOLVER_FILES}


def quarter_energy(run: str, q: str):
    """Quarter q's energy as `solve.solve_joint` builds it from the run's readings (U with the tie-break, less each
    row's minimum; the walls' odds), and its code names."""
    C = city()
    odds = wall_odds(q)
    options = load_options(q, baseline_dir())
    kind, _ = kind_of(run)
    voc = vocabulary()
    if kind == "control":
        rd = readings_from(q, stage_raw(q), voc)
        U, _, _ = E.label_costs(C.n, len(rd.names), rd, options, 1.0)
    elif kind == "s1":
        g = RUN_RE.match(run)
        rd = readings_from(q, degrade(stage_raw(q), q, float(g["m"]) / 100, float(g["r"]) / 100, int(g["seed"]))[0],
                           voc)
        U, _, _ = E.label_costs(C.n, len(rd.names), rd, options, 1.0)
    elif kind == "s1u":
        g = RUN_RE.match(run[:-len(UNSURE_SUFFIX)])
        charges, rd, _ = unsure_problem(q, float(g["m"]) / 100, float(g["r"]) / 100, int(g["seed"]))
        U, _, _ = label_costs_from(C.n, len(rd.names), charges, options, 1.0)
    else:
        charges, rd, _ = s0_problem(q, mixed=(kind == "s0-mix"))
        U, _, _ = label_costs_from(C.n, len(rd.names), charges, options, 1.0)
    U = U + solve_tie_break() * E.path_costs(C.n, len(rd.names), C.a, C.b, odds, rd)
    U -= U.min(1, keepdims=True)
    return U, odds, rd.names


def solve_tie_break() -> float:
    import inspect
    return inspect.signature(solve.solve_joint).parameters["tie_break"].default


def answer_index(run: str, q: str, names: list) -> np.ndarray:
    """A run's quarter-q solve (the whole city) as indices into `names` (a code at its first name)."""
    folder = baseline_dir() if run == runs.FINAL_RUN else kind_of(run)[1]
    Z = np.load(run_file(run, f"-{q}", folder), allow_pickle=True)
    first = {}
    for i, n in enumerate(names):
        first.setdefault(n, i)
    return np.array([first[canon(n)] for n in np.asarray(Z["names"]).astype(str)[Z["pred"]]], int)


def search_gap(run: str, q: str) -> dict:
    """The run's own quarter-q energy at final's quarter-q answer, minus at the run's answer.  Below 0: the run's
    search stopped above a map (final's) that its own energy prefers.  That is all it shows: it splits no wrong lots
    between the search and the readings, and says nothing of where the run's own optimum lies.  It can only show a
    miss where final's map is a low-energy answer under the run's readings, as it stays when labels are dropped but
    not when they are misread.  Written to runs/{run}-{q}.gap.json."""
    C = city()
    U, odds, names = quarter_energy(run, q)
    x, xf = answer_index(run, q, names), answer_index(runs.FINAL_RUN, q, names)
    own = E.energy(U, C.a, C.b, odds, x, 1.0)
    out = {"run": run, "q": q, "own_energy": round(own, 3),
           "final_answer_minus_own": round(E.energy(U, C.a, C.b, odds, xf, 1.0) - own, 3),
           "lots_differing_from_final": int((x != xf).sum())}
    run_file(run, f"-{q}", kind_of(run)[1]).with_suffix(".gap.json").write_text(json.dumps(out) + "\n")
    return out


def solve_one(run: str, q: str) -> dict:
    """Solve quarter q of `run` with the stage's walls, placement and search; write {run}-{q}.npz and .json."""
    kind, folder = kind_of(run)
    hashes = solver_hashes()
    loaded = str(Path(solve.__file__).resolve().parent)
    if run not in UNPINNED_RUNS and not loaded.startswith(str(PINNED_SRC.resolve())):
        raise RuntimeError(f"{run}: the solver must load from the pinned copy {PINNED_SRC}, not {loaded}")
    C = city()
    odds = wall_odds(q)
    voc = vocabulary()
    lines = []

    def log(s):
        lines.append(f"{q}: {s}")

    info = {}
    U = None
    if kind == "control":
        rd = readings_from(q, stage_raw(q), voc)
    elif kind == "s1":
        g = RUN_RE.match(run)
        raw, info = degrade(stage_raw(q), q, float(g["m"]) / 100, float(g["r"]) / 100, int(g["seed"]))
        rd = readings_from(q, raw, voc)
    elif kind == "s1u":
        g = RUN_RE.match(run[:-len(UNSURE_SUFFIX)])
        charges, rd, info = unsure_problem(q, float(g["m"]) / 100, float(g["r"]) / 100, int(g["seed"]))
        U, n_labels, n_terms = label_costs_from(C.n, len(rd.names), charges, load_options(q, baseline_dir()), 1.0)
    elif kind == "control-s0path":
        rd = readings_from(q, stage_raw(q), voc)
        charges = [(lab, far, own, function_P(T, len(rd.names))) for lab, far, own, T in groups_of(rd)]
        U, n_labels, n_terms = label_costs_from(C.n, len(rd.names), charges, load_options(q, baseline_dir()), 1.0)
    else:
        charges, rd, info["no_record"] = s0_problem(q, mixed=(kind == "s0-mix"))
        info["charges"] = len(charges)
        options = load_options(q, baseline_dir())
        U, n_labels, n_terms = label_costs_from(C.n, len(rd.names), charges, options, 1.0)
    index = {}
    for i, n in enumerate(rd.names):
        index.setdefault(n, i)
    table_codes = np.array([index.get(canon(c), -1) if k else -1 for c, k in zip(C.table, C.known)])
    log(f"readings {len(rd.lot):,} on {len(np.unique(rd.lot)):,} lots")
    options = load_options(q, baseline_dir())
    t0 = time.perf_counter()
    if U is None:
        p, Uused = solve.solve_joint(C.n, C.a, C.b, odds, rd, options, table_codes=table_codes, log=log)
    else:
        with label_costs_given(U, n_labels, n_terms):
            p, Uused = solve.solve_joint(C.n, C.a, C.b, odds, rd, options, table_codes=table_codes, log=log)
    seconds = time.perf_counter() - t0
    names = rd.names + [""]
    save_run(run_file(run, f"-{q}", folder), np.array(names)[p])
    out = {"run": run, "q": q, "kind": kind, "info": info, "log": lines, "seconds": round(seconds, 1),
           "solver_loaded_from": loaded,
           "solver_sha256": hashes, "hashes_after": solver_hashes()}
    run_file(run, f"-{q}", folder).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def done(run: str, q: str) -> bool:
    _, folder = kind_of(run)
    return run_file(run, f"-{q}", folder).with_suffix(".json").is_file()


def pinned_environment() -> dict:
    """A solve process's environment: one thread (`threads.environment`), the pinned copy of the package first on
    the path, and this checkout's data and inputs (the copy's own config would look beside the copy)."""
    from sfzmaps import config
    if not (PINNED_SRC / "sfzmaps" / "solver" / "solve.py").is_file():
        raise FileNotFoundError(f"no pinned copy at {PINNED_SRC}: extract `git archive {PINNED_COMMIT} "
                                "pipeline/src/sfzmaps` into its parent's parent")
    return {**threads.environment(), "PYTHONPATH": str(PINNED_SRC), "SFZMAPS_DATA_DIR": str(config.DATA_DIR),
            "SFZMAPS_INPUTS_DIR": str(config.INPUTS_DIR)}


def solving() -> int:
    """How many spike solve processes run on the machine (any runner's)."""
    p = subprocess.run(["pgrep", "-f", "reader_spike.py solve"], capture_output=True, text=True)
    return len(p.stdout.split())


def in_progress(run: str, q: str) -> bool:
    """Whether a solve of (run, q) runs now (another runner's, say)."""
    return subprocess.run(["pgrep", "-f", f"reader_spike.py solve {run} {q}$"], capture_output=True).returncode == 0


def run_jobs(jobs: list[tuple[str, str]], max_procs: int = MAX_PROCS, load_start: float = LOAD_START,
             min_procs: int = MIN_PROCS) -> None:
    """Solve each (run, q) not yet solved nor being solved, one pinned process each (`pinned_environment`), at most
    max_procs at once on the machine (counting every runner's solves), one start per START_EVERY s, while the
    1-minute load is below load_start or fewer than MIN_PROCS spike solves run (other agents' load must not stop
    this one); then, once every job is solved, join every run whose four quarters are."""
    todo = [j for j in jobs if not done(*j) and not in_progress(*j)]
    me = Path(__file__).resolve()
    env = pinned_environment()
    live = []
    failed = []
    t0 = time.time()
    while todo or live:
        for j, p, fh in list(live):
            if p.poll() is not None:
                live.remove((j, p, fh))
                fh.close()
                if p.returncode != 0:
                    failed.append(j)
                print(f"[{time.time() - t0:6.0f} s] {j[0]} {j[1]} {'done' if p.returncode == 0 else 'FAILED'} "
                      f"(load {os.getloadavg()[0]:.1f}, {len(live)} running, {len(todo)} waiting)", flush=True)
        running = max(len(live), solving())
        if todo and running < max_procs and (os.getloadavg()[0] < load_start or running < min_procs):
            j = todo.pop(0)
            if done(*j) or in_progress(*j):                 # another runner took it meanwhile
                continue
            _, folder = kind_of(j[0])
            (folder / "runs").mkdir(parents=True, exist_ok=True)
            fh = open(folder / "runs" / f"{j[0]}-{j[1]}.stderr", "w")
            p = subprocess.Popen([sys.executable, str(me), "solve", *j], env=env, stdout=fh, stderr=subprocess.STDOUT)
            live.append((j, p, fh))
            print(f"[{time.time() - t0:6.0f} s] {j[0]} {j[1]} started (load {os.getloadavg()[0]:.1f}, "
                  f"{len(live)} running)", flush=True)
        time.sleep(START_EVERY)
    if failed:
        raise RuntimeError(f"failed: {failed}")
    while any(in_progress(*j) for j in jobs):              # another runner's solves of these jobs
        time.sleep(START_EVERY)
    for run in sorted({r for r, _ in jobs}):
        if all(done(run, q) for q in QUARTERS):
            subprocess.run([sys.executable, str(me), "join", run], env=env, check=True)


# --- measuring -------------------------------------------------------------------------------------------------------

def codes_of(run: str) -> np.ndarray:
    """A run's code per lot: 'final' (the stage's) or a spike run's join."""
    if run == runs.FINAL_RUN:
        return load_run(run, baseline_dir())
    return load_run(run, kind_of(run)[1])


def six(run: str) -> dict:
    """The run's six measures and companions, every lot scored (`report.six_measures`)."""
    return report.six_measures(codes_of(run), np.zeros(city().n, bool))


@lru_cache(maxsize=1)
def final_largest_incident() -> np.ndarray:
    """The lots of final's largest incident (1,838 lots RH-1 -> RH-1(S), the Mission-strip district; a search
    failure: the energy prefers the right map, docs/research/joint-solver/notes/2026-11-16-big-incidents.md #1).
    Fixed from final, the same lots for every run: 'wrong lots outside it' is a companion measure that a run
    escaping or keeping that one search trap does not swing by 1,800 lots."""
    C = city()
    codes = codes_of(runs.FINAL_RUN)
    a, b = C.a, C.b
    wrong = C.scored & (codes != C.table)
    key = np.char.add(np.char.add(C.table, "|"), codes)
    m = wrong[a] & wrong[b] & (key[a] == key[b])
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(C.n, C.n)),
                                   directed=False)
    c = Counter(comp[wrong].tolist()).most_common(1)[0][0]
    return wrong & (comp == c)


def wrong_outside_1(run: str) -> int:
    """Wrong lots outside final's largest incident (`final_largest_incident`)."""
    C = city()
    return int((C.scored & (codes_of(run) != C.table) & ~final_largest_incident()).sum())


class Blocks:
    """The measures split by the table's blocks (blklot's first four characters), for the block bootstrap: per
    block, wrong lots, incidents (each incident split over its lots, 1/size each, so they sum to the count) and the
    zoning labels counted and right (a label in the block of its lot), as `measures.score` and `incidents` count
    them."""

    def __init__(self):
        C = self.C = city()
        blk = C.lots["blklot"].astype(str).str[:4].values
        self.ids, self.of_lot = np.unique(blk, return_inverse=True)
        self.nb = len(self.ids)
        self.T = Table(np.zeros(C.n, bool))
        labels = [lot for k, q in enumerate(QUARTERS) for _, lot, _ in top_readings(q, scoring_source())
                  if C.quarter[lot] == k]
        if len(labels) != len(self.T.label_code):
            raise ValueError("labels differ from measures.Table's")
        self.label_block = self.of_lot[np.array(labels, int)]
        self.outside_1 = ~final_largest_incident()

    def stats(self, codes: np.ndarray) -> dict:
        C, T = self.C, self.T
        a, b = C.a, C.b
        wrong = C.scored & (codes != C.table)
        key = np.char.add(np.char.add(C.table, "|"), codes)
        m = wrong[a] & wrong[b] & (key[a] == key[b])
        _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(C.n, C.n)),
                                       directed=False)
        size = np.bincount(comp[wrong], minlength=C.n)
        share = np.where(wrong, 1.0 / np.maximum(size[comp], 1), 0.0)
        nd = T.district.max() + 1
        lots_in = np.bincount(T.district[C.scored], minlength=nd)
        right_in = np.bincount(T.district[(codes == C.table) & C.scored], minlength=nd)
        d = np.maximum(T.label_district, 0)
        counted = (T.label_district < 0) | (lots_in[d] > 0)
        right = (T.label_district >= 0) & (lots_in[d] > 0) & (right_in[d] > lots_in[d] / 2)
        zoning = counted & T.label_in_vocab
        return {"wrong": np.bincount(self.of_lot, wrong.astype(float), self.nb),
                "wrong_outside_1": np.bincount(self.of_lot, (wrong & self.outside_1).astype(float), self.nb),
                "incidents": np.bincount(self.of_lot, share, self.nb),
                "right": np.bincount(self.label_block[zoning], right[zoning].astype(float), self.nb),
                "counted": np.bincount(self.label_block[zoning], minlength=self.nb).astype(float)}

    def weights(self, draws: int = BOOT_DRAWS, seed: int = BOOT_SEED) -> np.ndarray:
        """draws x blocks: how often each block is drawn (blocks resampled with replacement), fixed seed."""
        rng = np.random.default_rng(seed)
        W = np.zeros((draws, self.nb), np.float64)
        for i in range(draws):
            W[i] = np.bincount(rng.integers(0, self.nb, self.nb), minlength=self.nb)
        return W


def boot_measures(W: np.ndarray, s: dict) -> dict:
    """Each bootstrap draw's wrong lots, incidents and labels right."""
    return {"wrong_lots": W @ s["wrong"], "wrong_lots_outside_1": W @ s["wrong_outside_1"],
            "incidents": W @ s["incidents"],
            "labels_right": (W @ s["right"]) / (W @ s["counted"])}


def interval(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# --- honest intervals for wrong lots (fix round 1, I1) ---------------------------------------------------------------

CLUSTER_SEED = BOOT_SEED + 2


def changed_sets_of(codes: np.ndarray, base: np.ndarray) -> list[np.ndarray]:
    """The lots whose code differs between two answers, joined through any border into connected sets (one search
    outcome each): the units a wrong-lot difference is made of (a lot with the same code in both is wrong in both
    or in neither)."""
    C = city()
    diff = codes != base
    m = diff[C.a] & diff[C.b]
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (C.a[m], C.b[m])), shape=(C.n, C.n)),
                                   directed=False)
    ix = np.flatnonzero(diff)
    order = np.argsort(comp[ix], kind="stable")
    return [g for g in np.split(ix[order], np.flatnonzero(np.diff(comp[ix][order])) + 1) if len(g)]


def set_deltas(codes: np.ndarray, base: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """Per changed set (`changed_sets_of`): scored wrong lots of `codes` minus of `base`, counting only the lots of
    `keep` (all when None)."""
    C = city()
    w = C.scored & (codes != C.table)
    wb = C.scored & (base != C.table)
    k = np.ones(C.n, bool) if keep is None else keep
    return np.array([int((w[g] & k[g]).sum()) - int((wb[g] & k[g]).sum()) for g in changed_sets_of(codes, base)])


def cluster_interval(deltas: np.ndarray, draws: int = BOOT_DRAWS, seed: int = CLUSTER_SEED) -> list[float]:
    """95% percentile interval of a wrong-lot difference with its changed sets resampled with replacement (one set:
    one search outcome, so a district recoloured by one outcome counts once, not as many independent blocks).  It
    covers which outcomes happened, given the run; not the seed (which labels were degraded), which only more runs
    measure."""
    if not len(deltas):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    sums = np.array([deltas[rng.integers(0, len(deltas), len(deltas))].sum() for _ in range(draws)])
    return interval(sums)


def t_interval(values) -> list[float]:
    """The mean's 95% t interval over independent replicates (seeds): mean ± t(0.975, n - 1) sd / sqrt(n)."""
    from scipy.stats import t
    v = np.asarray(values, float)
    if len(v) < 2:
        return [float("nan"), float("nan")]
    h = float(t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return [float(v.mean() - h), float(v.mean() + h)]


def paired(W: np.ndarray, s_run: dict, s_base: dict) -> dict:
    """run - base per draw (the same resample for both): the 95% percentile interval of each difference, and each
    run's own interval."""
    br, bb = boot_measures(W, s_run), boot_measures(W, s_base)
    return {k: {"diff_interval": interval(br[k] - bb[k]), "run_interval": interval(br[k]),
                "base_interval": interval(bb[k])} for k in br}


# --- the checks ------------------------------------------------------------------------------------------------------

def check() -> None:
    """The plumbing, before any solve: readings_from equals labels.readings, label_costs_from with the function's P
    equals energy.label_costs (bit for bit), and the block split sums to the measures of final."""
    C = city()
    voc = vocabulary()
    for q in QUARTERS:
        a, b = readings_from(q, stage_raw(q), voc), labels_mod.readings(q, voc)
        for f in ("lot", "code", "trust", "label", "far"):
            if not np.array_equal(getattr(a, f), getattr(b, f)):
                raise AssertionError(f"{q}: readings_from's {f} differs")
        if a.names != b.names:
            raise AssertionError(f"{q}: names differ")
        options = load_options(q, baseline_dir())
        K = len(b.names)
        U1, n1, t1 = E.label_costs(C.n, K, b, options, 1.0)
        U2, n2, t2 = label_costs_from(C.n, K, [(lab, far, own, function_P(T, K)) for lab, far, own, T in
                                               groups_of(b)], options, 1.0)
        if not (np.array_equal(U1, U2) and n1 == n2 and t1 == t2):
            raise AssertionError(f"{q}: label_costs_from differs from energy.label_costs")
        print(f"{q}: readings_from == labels.readings; label_costs_from == energy.label_costs "
              f"({n1:,} labels, {t1:,} terms)")
    model, fallback = confusion(), quarter_codes()
    base = {q: stage_raw(q) for q in QUARTERS}
    changed = {}
    for q in QUARTERS:
        raw = base[q]
        d, info = degrade(raw, q, 0.10, 0.10, 0, model, fallback)
        d5, _ = degrade(raw, q, 0.05, 0.05, 0, model, fallback)
        if not set(np.unique(d.lab)) <= set(np.unique(d5.lab)):
            raise AssertionError(f"{q}: a label missed at 5% is kept at 10%")
        kept = np.isin(raw.lab, d.lab)
        if not (np.array_equal(raw.trust[kept], d.trust) and np.array_equal(raw.lab[kept], d.lab)):
            raise AssertionError(f"{q}: a kept reading's trust or label changed")
        moved = d.code != raw.code[kept]
        if moved.sum() != info["misread"] or len(set(d.lab[moved])) != info["misread"]:
            raise AssertionError(f"{q}: not one changed reading per misread label")
        changed[q] = (frozenset(d.lab[moved].tolist()), frozenset(set(raw.lab.tolist()) - set(d.lab.tolist())))
    if len(set(changed.values())) != 1:
        raise AssertionError("the quarters miss or misread different labels")
    print(f"degrade: nested by rate, trusts kept, one reading changed per misread label, the same labels in every "
          f"quarter ({info})")
    B = Blocks()
    s = B.stats(codes_of(runs.FINAL_RUN))
    m = six(runs.FINAL_RUN)
    got = (int(round(s["wrong"].sum())), int(round(s["incidents"].sum())),
           round(s["right"].sum() / s["counted"].sum(), 6))
    want = (m["wrong_lots"], m["incidents"], m["labels_right"])
    if got != want:
        raise AssertionError(f"block split {got} != measures {want}")
    print(f"blocks: {B.nb:,}; the split sums to final's wrong lots, incidents and labels right {want}")
    labs = pickle.load(open(INPUTS / "labs.pkl", "rb"))
    pr = load_prints()
    bad = [lab.index for lab in labs if lab.index >= 0 and (lab.sheet != pr["sheet"].iloc[lab.index] or
                                                             tuple(lab.point) != (float(pr["page_x"].iloc[lab.index]),
                                                                                  float(pr["page_y"].iloc[lab.index])))]
    if bad:
        raise AssertionError(f"code-reader labels whose sheet or point differ from prints.parquet: {bad[:5]}")
    print(f"labs.pkl: every label's sheet and point equal prints.parquet's row ({len(labs):,} labels)")
    final = codes_of(runs.FINAL_RUN)
    d = set_deltas(codes_of("s0"), final) if done("s0", "se") else np.zeros(0)
    if len(d) and int(d.sum()) != six("s0")["wrong_lots"] - six(runs.FINAL_RUN)["wrong_lots"]:
        raise AssertionError("the changed sets' wrong-lot differences don't sum to the run's")
    print(f"changed sets: s0's {len(d)} sets sum to its wrong-lot difference ({int(d.sum()):+,})")


if __name__ == "__main__":
    if sys.argv[1] == "solve":
        out = solve_one(sys.argv[2], sys.argv[3])
        print(json.dumps({k: out[k] for k in ("run", "q", "info", "seconds")}))
    elif sys.argv[1] == "gap":                 # gap RUN... : each run's four quarters (`search_gap`)
        for run in sys.argv[2:]:
            for q in QUARTERS:
                if not run_file(run, f"-{q}", kind_of(run)[1]).with_suffix(".gap.json").is_file():
                    print(json.dumps(search_gap(run, q)), flush=True)
    elif sys.argv[1] == "join":
        if not str(Path(solve.__file__).resolve()).startswith(str(PINNED_SRC.resolve())):
            raise SystemExit("join: run pinned (reader_spike.pinned_environment)")
        solve.join(sys.argv[2], kind_of(sys.argv[2])[1])
    elif sys.argv[1] == "check":
        check()
    else:
        raise SystemExit(__doc__)
