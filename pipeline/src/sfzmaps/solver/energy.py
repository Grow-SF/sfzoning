"""The energy of a district map: one number over every lot's code, lower is better.

  E(codes) = sum over lots of label_cost[lot, code]
           + sum over borders whose two lots get different codes, where the odds say no line (odds < 0): -odds
           + sum over borders whose two lots get the same code, where the odds say a line (odds > 0): odds x same_weight

label_cost comes only from printed labels, each charged once (label_costs), plus a tiny tie-break."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

TOP_READING_FIT = (0.128, 1.085)   # logit(chance the top reading is right) = a + b logit(trust), fitted on the
                                   #   real map: trust 0.5 -> 53%, 0.9 -> 92.5%, 0.99 -> 99.4%


def wall_terms(a, b, odds, same_weight):
    """the borders that can cost anything: (a, b, cost if codes differ) where odds < 0, and
    (a, b, cost if codes are the same) where odds > 0"""
    differ = np.maximum(-odds, 0)
    d, s = differ > 0, odds > 0
    return a[d], b[d], differ[d], a[s], b[s], odds[s].astype(np.float64) * same_weight


def energy(U, a, b, odds, codes, same_weight):
    da, db, dc, sa, sb, sc = wall_terms(a, b, odds, same_weight)
    return float(U[np.arange(len(U)), codes].sum() + dc[codes[da] != codes[db]].sum()
                 + sc[codes[sa] == codes[sb]].sum())


def label_costs(n_lots, n_codes, rd, options=None, weight=1.0):
    """U[lot, code]: each printed label charged once, spread over the lots it may stand for.
    A label's readings give P(code): the top reading is right with its calibrated chance q; the rest goes to the
    other readings by their trusts and evenly to every code, so no code is ever impossible.  At each lot l the label
    may stand for (chance pi_l), U[l] gets -weight log(pi_l P + (1 - pi_l) / K).
    options: label -> [(lot, chance, is_stay)] from the placement model; without options (or for a far-side copy, or
    when the options were made for another lot) a label stands for its own lot with chance 1."""
    K = n_codes
    U = np.zeros((n_lots, K))
    groups = {}
    for r in range(len(rd.lot)):
        groups.setdefault((int(rd.label[r]), bool(rd.far[r])), []).append(r)
    terms = 0
    for (label, far), rows in groups.items():
        T = np.zeros(K)
        for r in rows:
            T[rd.code[r]] = max(T[rd.code[r]], rd.trust[r])
        top = int(np.argmax(T))
        t = np.clip(T[top], 1e-3, 1 - 1e-3)
        q = 1 / (1 + np.exp(-(TOP_READING_FIT[0] + TOP_READING_FIT[1] * np.log(t / (1 - t)))))
        rest = T.copy()
        rest[top] = 0.0
        rest = (rest + 1.0 / K) / (rest.sum() + 1.0)
        P = q * np.eye(K)[top] + (1 - q) * rest
        own = int(rd.lot[rows[0]])
        where = [(own, 1.0)]
        if options is not None and not far and label in options:
            stay = [o for o in options[label] if o[2]]
            if stay and stay[0][0] == own:
                where = [(o[0], o[1]) for o in options[label]]
        for lot, pi in where:
            U[lot] -= weight * np.log(pi * P + (1 - pi) / K)
            terms += 1
    return U, len(groups), terms


def path_costs(n_lots, n_codes, a, b, odds, rd, cap=50.0):
    """per code: the cheapest path from any of its readings' lots, each border crossed costing -log(chance of no
    line) = log(1 + e^odds); capped"""
    with np.errstate(over="ignore"):
        cost = np.minimum(np.log1p(np.exp(odds)), 1e5) + 1e-12
    G = coo_matrix((np.r_[cost, cost], (np.r_[a, b], np.r_[b, a])), shape=(n_lots, n_lots)).tocsr()
    C = np.full((n_lots, n_codes), cap)
    for c in np.unique(rd.code):
        C[:, c] = np.minimum(dijkstra(G, directed=False, indices=np.unique(rd.lot[rd.code == c]), min_only=True), cap)
    return C
