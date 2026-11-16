"""Shared pieces of the incident scripts (scripts/incidents/): where they read and write, the incidents of a run, a
quarter's saved costs, the energy of an answer split into its terms, and a search confined to a set of lots.

Everything here reads the solver's stage outputs and never writes them.  The scripts write only into
data/solver/experiments/incidents-*/ (CENSUS_DIR, J7_DIR).

The energy (sfzmaps.solver.energy) of a quarter's solve, as its saved costs (runs/{run}-{q}-costs.npz) hold it:
  U[lot, code]   the label costs (each printed label charged once, spread by the placement over the lots it may
                 stand for) plus a tiny tie-break (energy.path_costs x 1e-4), less each lot's row minimum
  odds[pair]     the wall odds the solve used (wall_weight 1); a border whose lots differ where odds < 0 costs -odds,
                 one whose lots share a code where odds > 0 costs odds x same_weight (1)
`label_charges` rebuilds U from the readings and the placement charge by charge, so a change in U can be split into
the labels' own lots, the placement's moves and the street labels' far-side copies.

Importing this module pins the maths libraries and OpenCV to one thread for any process it starts (their environment
variables, unless set) and OpenCV in this one; run the scripts with the BLAS variables set too (sfzmaps.solver.threads
ONE_THREAD), since numpy reads them when it loads."""
import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENCV_FOR_THREADS_NUM": "1"}.items():
    os.environ.setdefault(_k, _v)

from collections import Counter  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from sfzmaps.solver import energy as E  # noqa: E402
from sfzmaps.solver import runs, search  # noqa: E402
from sfzmaps.solver.city import city, load_run, run_file  # noqa: E402
from sfzmaps.solver.codes import canon, table_vocabulary  # noqa: E402
from sfzmaps.solver.labels import readings  # noqa: E402
from sfzmaps.solver.placement import load_options  # noqa: E402

cv2.setNumThreads(1)
CENSUS_DIR = runs.EXPERIMENTS_DIR / "incidents-census"
J7_DIR = runs.EXPERIMENTS_DIR / "incidents-j7"          # import_j7.py: J7's answer (runs/j7.npz) and walls
J7_WALLS_RUN = "incidents-j7walls"                      # this build's solve with J7's walls (`solve --walls J7_DIR`)
SAME_WEIGHT = 1.0           # solve_joint's default, as the stage solves
TIE_BREAK = 1e-4            # solve_joint's default
KINDS = ("own", "stay", "move", "far")   # a charge: a label with no placement options on its own lot; the
                                         # placement's stay option; a move option; a street label's far-side copy


def run_codes(run: str) -> np.ndarray:
    """A run's code per lot (its join) by name: 'final' (the stage's), 'j7' (import_j7.py) or an experiment."""
    if run == "j7":
        return load_run("j7", J7_DIR)
    return load_run(run, runs.solve_dir(run))


def has_run(run: str) -> bool:
    folder = J7_DIR if run == "j7" else runs.solve_dir(run)
    return run_file(run, folder=folder).is_file()


def mask(n: int, ix) -> np.ndarray:
    m = np.zeros(n, bool)
    m[np.asarray(ix, int)] = True
    return m


def blocks(ix) -> list:
    """(block, lots) of the lots `ix`, most lots first"""
    blk = city().lots["blklot"].astype(str).str[:4].values
    return [[b, int(n)] for b, n in Counter(blk[np.asarray(ix, int)].tolist()).most_common()]


def incidents(codes: np.ndarray, min_lots: int = 1) -> list[dict]:
    """The incidents of `codes` as measures.incidents defines them (connected wrong scored lots sharing one table
    code and one wrong code), largest first (ties by first lot): lots, table, solved, quarters."""
    C = city()
    a, b = C.a, C.b
    wrong = C.scored & (codes != C.table)
    key = np.char.add(np.char.add(C.table, "|"), codes)
    m = wrong[a] & wrong[b] & (key[a] == key[b])
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(C.n, C.n)),
                                   directed=False)
    w = np.flatnonzero(wrong)
    order = np.argsort(comp[w], kind="stable")
    groups = np.split(w[order], np.flatnonzero(np.diff(comp[w][order])) + 1)
    out = []
    for g in groups:
        if len(g) >= min_lots:
            i = g[0]
            out.append({"lots": g, "table": str(C.table[i]), "solved": str(codes[i]),
                        "quarters": {k: int(v) for k, v in Counter(C.quarter[g].tolist()).items()}})
    out.sort(key=lambda d: (-len(d["lots"]), int(d["lots"][0])))
    return out


@dataclass
class Costs:
    """A quarter's saved costs (runs/{run}-{q}-costs.npz): U and odds as float64, the solve's codes (index), names
    in the one spelling, code index by name, and the table's code index per lot (-1 unknown or not a name)."""
    run: str
    q: str
    U: np.ndarray
    odds: np.ndarray
    codes: np.ndarray
    names: list
    index: dict
    table_codes: np.ndarray

    def to_index(self, names_per_lot, fallback):
        """codes by name -> indices, a name not among the solve's keeping `fallback`'s index"""
        return np.array([self.index.get(canon(c), f) if c else f for c, f in zip(names_per_lot, fallback)], int)


def load_costs(run: str, q: str) -> Costs:
    z = np.load(run_file(run, f"-{q}-costs", runs.solve_dir(run)))
    names = [canon(n) for n in z["names"]]
    index = {}
    for i, n in enumerate(names):
        index.setdefault(n, i)
    return Costs(run, q, z["U"].astype(float), z["odds"].astype(float), z["codes"].astype(int), names, index,
                 z["table_codes"].astype(int))


@dataclass
class Charges:
    """Every label charge of a quarter's U, as energy.label_costs makes them: charge j adds cost[j] (one value per
    code) to lot[j]; its label, kind (KINDS index) and placement chance pi."""
    lot: np.ndarray
    label: np.ndarray
    kind: np.ndarray
    pi: np.ndarray
    cost: np.ndarray
    top: np.ndarray        # the label's top reading (code index) and its trust
    top_trust: np.ndarray
    tie: np.ndarray        # TIE_BREAK x energy.path_costs, per lot and code


def label_charges(run: str, q: str, costs: Costs, walls_dir=None) -> Charges:
    """Rebuild the quarter's U charge by charge (energy.label_costs with the run's placement, weight 1), and the
    tie-break; checks that they give the saved U (to float32 rounding) and raises if not."""
    C = city()
    rd = readings(q, table_vocabulary(C.lots))
    if [canon(n) for n in rd.names] != costs.names:
        raise ValueError(f"{run} {q}: the readings' names differ from the saved costs'")
    options = load_options(q, runs.solve_dir(run), walls_dir)
    K = len(rd.names)
    groups = {}
    for r in range(len(rd.lot)):
        groups.setdefault((int(rd.label[r]), bool(rd.far[r])), []).append(r)
    lot, label, kind, pi, cost, top, top_trust = [], [], [], [], [], [], []
    for (lab, far), rows in groups.items():
        T = np.zeros(K)
        for r in rows:
            T[rd.code[r]] = max(T[rd.code[r]], rd.trust[r])
        t_top = int(np.argmax(T))
        t = np.clip(T[t_top], 1e-3, 1 - 1e-3)
        qq = 1 / (1 + np.exp(-(E.TOP_READING_FIT[0] + E.TOP_READING_FIT[1] * np.log(t / (1 - t)))))
        rest = T.copy()
        rest[t_top] = 0.0
        rest = (rest + 1.0 / K) / (rest.sum() + 1.0)
        P = qq * np.eye(K)[t_top] + (1 - qq) * rest
        own = int(rd.lot[rows[0]])
        where = [(own, 1.0, 3 if far else 0)]
        if not far and lab in options:
            stay = [o for o in options[lab] if o[2]]
            if stay and stay[0][0] == own:
                where = [(o[0], o[1], 1 if o[2] else 2) for o in options[lab]]
        for l_, p_, k_ in where:
            lot.append(l_)
            label.append(lab)
            kind.append(k_)
            pi.append(p_)
            cost.append(-np.log(p_ * P + (1 - p_) / K))
            top.append(t_top)
            top_trust.append(T[t_top])
    ch = Charges(np.array(lot, int), np.array(label, int), np.array(kind, int), np.array(pi), np.array(cost),
                 np.array(top, int), np.array(top_trust), np.zeros(0))
    U = np.zeros((C.n, K))
    np.add.at(U, ch.lot, ch.cost)
    ch.tie = TIE_BREAK * E.path_costs(C.n, K, C.a, C.b, costs.odds, rd)
    U = U + ch.tie
    U -= U.min(1, keepdims=True)
    err = float(np.abs(U - costs.U).max())
    if err > 1e-3:
        raise ValueError(f"{run} {q}: the rebuilt label costs differ from the saved U by up to {err:.2g}")
    return ch


def wall_costs(costs: Costs, x: np.ndarray, P: np.ndarray, same_weight: float = SAME_WEIGHT):
    """(cost of the borders P whose lots differ where odds < 0, cost of those whose lots share a code where
    odds > 0) under the answer x (code index per lot)"""
    C = city()
    o = costs.odds[P]
    differ = x[C.a[P]] != x[C.b[P]]
    return np.where(differ, np.maximum(-o, 0), 0.0), np.where(~differ, np.maximum(o, 0) * same_weight, 0.0)


def split(costs: Costs, ch: Charges, x0: np.ndarray, x1: np.ndarray, S: np.ndarray) -> dict:
    """The energy change from answer x0 to answer x1 (code index per lot; they differ only on the lots S), by term:
    labels on their own lot (labels_no_options + labels_stay: a label the placement keeps on the lot under its text
    counts here, so a label that should have moved but stayed shows as a label term), the placement's moves, the
    far-side copies, the tie-break, the walls where the lots differ with no line and where they share a code across
    a line; and the total, checked against the saved U."""
    C = city()
    inS = mask(C.n, S)
    if np.any((x0 != x1) & ~inS):
        raise ValueError("the answers differ outside S")
    P = np.flatnonzero(inS[C.a] | inS[C.b])
    d0, s0 = wall_costs(costs, x0, P)
    d1, s1 = wall_costs(costs, x1, P)
    j = np.flatnonzero(inS[ch.lot])
    dl = ch.cost[j, x1[ch.lot[j]]] - ch.cost[j, x0[ch.lot[j]]]
    by_kind = {k: float(dl[ch.kind[j] == i].sum()) for i, k in enumerate(KINDS)}
    tie = float((ch.tie[S, x1[S]] - ch.tie[S, x0[S]]).sum())
    dU = float((costs.U[S, x1[S]] - costs.U[S, x0[S]]).sum())
    out = {"labels": by_kind["own"] + by_kind["stay"], "labels_no_options": by_kind["own"],
           "labels_stay": by_kind["stay"], "placement": by_kind["move"], "far_side": by_kind["far"],
           "tie_break": tie, "walls_no_line": float(d1.sum() - d0.sum()),
           "walls_same_code_across_line": float(s1.sum() - s0.sum())}
    out["walls"] = out["walls_no_line"] + out["walls_same_code_across_line"]
    out["total"] = dU + out["walls"]
    out["check"] = float(dU - (sum(by_kind.values()) + tie))       # ~0: the charges rebuild U
    out["by_label"] = (ch.label[j], ch.kind[j], dl)
    return out


def energy_on(costs: Costs, x: np.ndarray, S: np.ndarray) -> float:
    """The part of the energy that depends on the lots S: their U and every border touching them."""
    C = city()
    inS = mask(C.n, S)
    P = np.flatnonzero(inS[C.a] | inS[C.b])
    d, s = wall_costs(costs, x, P)
    return float(costs.U[S, x[S]].sum() + d.sum() + s.sum())


def search_on(costs: Costs, x: np.ndarray, S: np.ndarray) -> np.ndarray:
    """The solver's own search (solve_joint's sequence of cuts, region and compound moves) confined to the lots S,
    every other lot fixed at x; returns the answer (code index per lot) it ends at, x outside S."""
    C = city()
    S = np.asarray(S, int)
    if not len(S):
        return x.copy()
    pos = np.full(C.n, -1)
    pos[S] = np.arange(len(S))
    ina, inb = pos[C.a] >= 0, pos[C.b] >= 0
    inner = np.flatnonzero(ina & inb)
    U = costs.U[S].copy()
    K = U.shape[1]
    for side, inside, other in ((ina & ~inb, C.a, C.b), (inb & ~ina, C.b, C.a)):   # a border to a fixed lot: a
        P = np.flatnonzero(side)                                                     # cost per code of the lot in S
        lot = pos[inside[P]]
        o = costs.odds[P]
        fixed = x[other[P]]
        cost = np.where(np.arange(K)[None, :] != fixed[:, None], np.maximum(-o, 0)[:, None],
                        (np.maximum(o, 0) * SAME_WEIGHT)[:, None])
        np.add.at(U, lot, cost)
    a, b, odds = pos[C.a[inner]], pos[C.b[inner]], costs.odds[inner]
    p = x[S].copy()
    p = search.expansion_moves(U, a, b, odds, 0.0, p)
    p = search.expansion_moves(U, a, b, odds, SAME_WEIGHT, p)
    for _ in range(4):
        e0 = E.energy(U, a, b, odds, p, SAME_WEIGHT)
        p = search.region_moves(U, a, b, odds, SAME_WEIGHT, p)
        p = search.compound_moves(U, a, b, odds, SAME_WEIGHT, p)
        p = search.expansion_moves(U, a, b, odds, SAME_WEIGHT, p)
        if E.energy(U, a, b, odds, p, SAME_WEIGHT) >= e0 - 1e-6:
            break
    out = x.copy()
    out[S] = p
    return out


def ring(S, hops: int) -> np.ndarray:
    """The lots S and every lot within `hops` neighbour steps of them."""
    C = city()
    m = mask(C.n, S)
    for _ in range(hops):
        m2 = m.copy()
        m2[C.a[m[C.b]]] = True
        m2[C.b[m[C.a]]] = True
        m = m2
    return np.flatnonzero(m)


def changed_set(costs: Costs, seed) -> np.ndarray:
    """The lots that change from the solve's answer to the table's around `seed`: the connected lots (through any
    border) whose solved code differs from the table's (lots the table knows by a code the solve has), holding the
    seed lots."""
    C = city()
    t = costs.table_codes
    diff = (t >= 0) & (costs.codes != t)
    m = diff[C.a] & diff[C.b]
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (C.a[m], C.b[m])), shape=(C.n, C.n)),
                                   directed=False)
    seed = np.asarray(seed, int)
    seed = seed[diff[seed]]
    return np.flatnonzero(np.isin(comp, np.unique(comp[seed])) & diff)
