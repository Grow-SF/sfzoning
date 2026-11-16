"""The shared pieces of the search investigation (search_verify.py, search_diagnose.py, search_experiments.py): a
quarter's energy rebuilt from its held-out inputs exactly as `solve.solve_quarter` builds it, the solver's own search
sequence (`solve.solve_joint`), the expansion move's bounded energy, and the search variants measured as experiments.

Read-only on the solver: every function here calls the solver's own code (energy.label_costs, energy.path_costs,
energy.energy, search.expansion_moves / region_moves / compound_moves) or copies a piece of it to inspect it; nothing
here changes it.  Experiments write only into data/solver/experiments/incidents-search-*/."""
import os

for _k, _v in {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENCV_FOR_THREADS_NUM": "1"}.items():
    os.environ.setdefault(_k, _v)

from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

from sfzmaps.solver import energy as E  # noqa: E402
from sfzmaps.solver import runs, search  # noqa: E402
from sfzmaps.solver.city import city, run_file, wall_odds  # noqa: E402
from sfzmaps.solver.codes import canon, table_vocabulary  # noqa: E402
from sfzmaps.solver.labels import readings  # noqa: E402
from sfzmaps.solver.placement import load_options  # noqa: E402

SAME = 1.0                  # solve_joint's same_weight / wall_weight, as the stage solves
TIE_BREAK = 1e-4            # solve_joint's default
SEARCH_PREFIX = "incidents-search-"


@dataclass
class Problem:
    """A quarter's energy as `solve.solve_joint` minimises it (label_weight = wall_weight = same_weight = 1):
    U[lot, code] (float64), the pairs a, b and their odds; the code names, the stage's solved answer (index per
    lot, runs/final-{q}.npz) and the table's code index per lot (-1 unknown)."""
    q: str
    U: np.ndarray
    a: np.ndarray
    b: np.ndarray
    odds: np.ndarray
    names: list
    final: np.ndarray
    table_codes: np.ndarray
    n_labels: int
    n_terms: int

    def energy(self, x) -> float:
        return E.energy(self.U, self.a, self.b, self.odds, x, SAME)

    def to_index(self, codes_by_name) -> np.ndarray:
        idx = {}
        for i, n in enumerate(self.names):
            idx.setdefault(n, i)
        return np.array([idx[canon(c)] for c in codes_by_name], int)


def problem(q: str, run: str = runs.FINAL_RUN, true_borders: bool = False, drop_labels=()) -> Problem:
    """Rebuild quarter q's energy from its held-out inputs (the stage's walls, the readings, the run's placement),
    in float64, exactly as solve_quarter -> solve_joint does; the run's solved answer mapped to the same names.
    true_borders: the ceiling check's odds (solve_quarter --true-borders), the tie-break's paths on them too.
    drop_labels: labels (print indices) whose readings are left out, as if deleted (an experiment, never an input)."""
    C = city()
    odds = wall_odds(q)
    if true_borders:
        from sfzmaps.solver.solve import TRUE_BORDER_ODDS
        both = C.known[C.a] & C.known[C.b]
        odds = np.where(both, np.where(C.table[C.a] != C.table[C.b], TRUE_BORDER_ODDS, -TRUE_BORDER_ODDS), odds)
    rd = readings(q, table_vocabulary(C.lots))
    if len(drop_labels):
        from sfzmaps.solver.labels import Readings
        k = ~np.isin(rd.label, np.asarray(drop_labels))
        rd = Readings(rd.lot[k], rd.code[k], rd.trust[k], rd.label[k], rd.far[k], rd.names)
    index = {}
    for i, n in enumerate(rd.names):
        index.setdefault(n, i)
    table_codes = np.array([index.get(canon(c), -1) if k else -1 for c, k in zip(C.table, C.known)])
    options = load_options(q, runs.solve_dir(run))
    K = len(rd.names)
    U, n_labels, n_terms = E.label_costs(C.n, K, rd, options, 1.0)
    U = U + TIE_BREAK * E.path_costs(C.n, K, C.a, C.b, odds, rd)
    U -= U.min(1, keepdims=True)
    R = np.load(run_file(run, f"-{q}", runs.solve_dir(run)), allow_pickle=True)
    solved_names = np.asarray(R["names"]).astype(str)[R["pred"]]
    names = list(rd.names)
    if len(set(names)) != len(names):
        raise ValueError(f"{q}: the readings' names repeat a code; a saved run's names would be ambiguous")
    lookup = {n: i for i, n in enumerate(names)}
    final = np.array([lookup[n] for n in solved_names], int)
    return Problem(q, U, C.a, C.b, odds, [canon(n) for n in names], final, table_codes, n_labels, n_terms)


def fusion_terms(P: Problem, x, y):
    """The binary problem of fusing answers x (s = 0) and y (s = 1), exactly (no bound): unaries c0, c1 and per
    pair (pa, pb) its four costs e00, e01, e10, e11.  An expansion move to code al is the fusion with y = al
    everywhere."""
    a, b, t, ra, rb, rr = E.wall_terms(P.a, P.b, P.odds, SAME)
    N = len(x)
    c0, c1 = P.U[np.arange(N), x], P.U[np.arange(N), y]
    pa, pb = np.r_[a, ra], np.r_[b, rb]
    w = np.r_[t, rr]
    is_same = np.r_[np.zeros(len(a), bool), np.ones(len(ra), bool)]

    def V(la, lb):
        return np.where(is_same, w * (la == lb), w * (la != lb))

    return c0, c1, pa, pb, V(x[pa], x[pb]), V(x[pa], y[pb]), V(y[pa], x[pb]), V(y[pa], y[pb])


def binary_energy(terms, s) -> float:
    c0, c1, pa, pb, e00, e01, e10, e11 = terms
    sa, sb = s[pa], s[pb]
    pair = np.where(sa, np.where(sb, e11, e10), np.where(sb, e01, e00))
    return float(np.where(s, c1, c0).sum() + pair.sum())


def fuse(P: Problem, x, y, improve: bool = False):
    """QPBO fusion of x with the proposal y: the roof dual's labelled lots take their side, the rest keep x; kept
    only if the exact energy falls (it cannot rise, by autarky, but it is checked).  With `improve`, each connected
    piece of the lots the roof dual leaves open then takes y whole if that lowers the energy.  Returns (answer, lots
    labelled, lots where x and y differ, energy change)."""
    diff = x != y
    if not diff.any():
        return x, 0, 0, 0.0
    terms = fusion_terms(P, x, y)
    c0, c1, pa, pb, e00, e01, e10, e11 = terms
    # only the lots where x and y differ are variables; pairs with one side fixed fold into the other's unary
    var = np.flatnonzero(diff)
    pos = np.full(len(x), -1)
    pos[var] = np.arange(len(var))
    va, vb = diff[pa], diff[pb]
    u0, u1 = c0[var].copy(), c1[var].copy()
    one = va & ~vb                          # b fixed at x_b (== y_b): s_b irrelevant, use e_.0
    np.add.at(u0, pos[pa[one]], e00[one])
    np.add.at(u1, pos[pa[one]], e10[one])
    one = vb & ~va
    np.add.at(u0, pos[pb[one]], e00[one])
    np.add.at(u1, pos[pb[one]], e01[one])
    two = va & vb
    s, labelled = qpbo_binary(u0, u1, pos[pa[two]], pos[pb[two]], e00[two], e01[two], e10[two], e11[two])
    q = x.copy()
    q[var[s]] = y[var[s]]
    if improve and not labelled.all():
        # the lots the roof dual leaves open, by connected piece: each piece takes y whole when that lowers the
        # energy (a simple form of QPBO-I's improve step).  Open pieces touch only labelled lots, so each piece's
        # change is its own: computed for all at once from the binary terms.
        open_ = ~labelled
        ia, ib = pos[pa[two]], pos[pb[two]]
        f00, f01, f10, f11 = e00[two], e01[two], e10[two], e11[two]
        m = open_[ia] & open_[ib]
        _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (ia[m], ib[m])),
                                                  shape=(len(var), len(var))), directed=False)
        nc = comp.max() + 1
        delta = np.bincount(comp[open_], (u1 - u0)[open_], minlength=nc)
        delta += np.bincount(comp[ia[m]], (f11 - f00)[m], minlength=nc)
        oa = open_[ia] & ~open_[ib]                     # a open, b fixed at s_b
        sb_ = s[ib[oa]]
        delta += np.bincount(comp[ia[oa]], np.where(sb_, f11[oa] - f01[oa], f10[oa] - f00[oa]), minlength=nc)
        ob = open_[ib] & ~open_[ia]                     # b open, a fixed at s_a
        sa_ = s[ia[ob]]
        delta += np.bincount(comp[ib[ob]], np.where(sa_, f11[ob] - f10[ob], f01[ob] - f00[ob]), minlength=nc)
        flip = open_ & (delta[comp] < -1e-9)
        if flip.any():
            t_ = q.copy()
            t_[var[flip]] = y[var[flip]]
            if P.energy(t_) < P.energy(q) - 1e-6:
                q = t_
    dE = P.energy(q) - P.energy(x)
    if dE < -1e-6:
        return q, int(labelled.sum()), len(var), dE
    return x, int(labelled.sum()), len(var), 0.0


def run_folder(name: str):
    """an experiment's folder: data/solver/experiments/incidents-search-<name>/"""
    return runs.solve_dir(SEARCH_PREFIX + name)


def save_quarter(name: str, P: Problem, p) -> None:
    from sfzmaps.solver.city import save_run
    run = SEARCH_PREFIX + name
    save_run(run_file(run, f"-{P.q}", run_folder(name)), np.array(P.names + [""])[p])


def join_and_score(name: str) -> dict:
    """join the experiment's four quarters (solve.join) and its six measures on every lot (report.six_measures, no
    data errors left out: the one measure for comparing runs of this build)"""
    from sfzmaps.solver.city import load_run
    from sfzmaps.solver.report import six_measures
    from sfzmaps.solver.solve import join
    run = SEARCH_PREFIX + name
    join(run, run_folder(name))
    return six_measures(load_run(run, run_folder(name)), np.zeros(city().n, bool))


def final_measures() -> dict:
    import json
    return json.loads((runs.SOLVE_DIR / "report.json").read_text())["measures"]["every_lot"]


SIX = (("lines_drawn", "Lines drawn", "{:.1%}"), ("drawn_are_table_lines", "Drawn that are table lines", "{:.1%}"),
       ("labels_right", "Labels right", "{:.1%}"), ("holes", "Holes", "{:,}"), ("incidents", "Incidents", "{:,}"),
       ("wrong_lots", "Wrong lots", "{:,}"))


def measures_table(cols: dict) -> str:
    """a markdown table of the six measures, one column per run (name -> six_measures)"""
    head = "| Measure | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    return head + "".join(f"| {label} | " + " | ".join(fmt.format(m[k]) for m in cols.values()) + " |\n"
                          for k, label, fmt in SIX)


def solver_sequence(U, a, b, odds, start=None, steps=None, sweeps=5, outer=4):
    """solve_joint's search, move for move: from `start` (default U.argmin(1)), expansion without the same-code
    cost, expansion with it, then up to `outer` rounds of region, compound and expansion moves until the energy
    stops falling."""
    p = U.argmin(1) if start is None else start.copy()
    p = search.expansion_moves(U, a, b, odds, 0.0, p, sweeps=sweeps, log=steps)
    p = search.expansion_moves(U, a, b, odds, SAME, p, sweeps=sweeps, log=steps)
    p = polish(U, a, b, odds, p, steps, sweeps, outer)
    return p


def polish(U, a, b, odds, p, steps=None, sweeps=5, outer=4):
    """the loop at the end of solve_joint: region, compound and expansion moves until the energy stops falling"""
    for _ in range(outer):
        e0 = E.energy(U, a, b, odds, p, SAME)
        p = search.region_moves(U, a, b, odds, SAME, p, log=steps)
        p = search.compound_moves(U, a, b, odds, SAME, p, log=steps)
        p = search.expansion_moves(U, a, b, odds, SAME, p, sweeps=sweeps, log=steps)
        if E.energy(U, a, b, odds, p, SAME) >= e0 - 1e-6:
            break
    return p


def expansion_parts(U, a0, b0, odds, same, p, al):
    """The binary problem one expansion move to code `al` from `p` solves (search.expansion_moves' construction,
    copied to inspect it): returns (c0, c1, ea, eb, w, const) with the cut's energy of a switch set s (bool per lot)
    = const + sum(c1[s]) + sum(c0[~s]) + sum(w[~s[ea] & s[eb]]).  It bounds the same-code-across-a-line cost from
    above (the side less keen to switch carries it), so it equals the true energy at s = none and is never below
    it."""
    N, K = U.shape
    a, b, t, ra, rb, rr = E.wall_terms(a0, b0, odds, same)
    c0 = U[np.arange(N), p].copy()
    c1 = U[:, al].copy()
    A = t * (p[a] != p[b])
    Bc = t * (p[a] != al)
    Cc = t * (p[b] != al)
    np.add.at(c1, a, Cc - A)
    np.add.at(c1, b, -Cc)
    w = Bc + Cc - A
    const = float(A.sum())
    xa, xb = p[ra], p[rb]
    np.add.at(c1, rb, rr * ((xa == al) & (xb != al)))
    np.add.at(c1, ra, rr * ((xb == al) & (xa != al)))
    both = (xa != al) & (xb != al) & (xa != xb)
    a_keener = (U[ra, al] - U[ra, xa]) <= (U[rb, al] - U[rb, xb])
    np.add.at(c1, np.where(a_keener, rb, ra), rr * both)
    same_ = (xa != al) & (xa == xb)
    np.add.at(c0, np.where(a_keener, ra, rb), rr * same_)
    np.add.at(c1, np.where(a_keener, rb, ra), rr * same_)
    # the same-code cost of borders where one side already holds al (exact) and where both hold al (constant)
    const += float((rr * ((xa == al) & (xb == al))).sum())
    return c0, c1, a, b, w, const


def bounded_energy(parts, s) -> float:
    c0, c1, ea, eb, w, const = parts
    return float(const + c1[s].sum() + c0[~s].sum() + w[~s[ea] & s[eb]].sum())


def qpbo_binary(c0, c1, pa, pb, e00, e01, e10, e11, keep=None):
    """Roof duality (QPBO, Boros-Hammer; Kolmogorov-Rother 2007) for a binary energy
        sum_i c0[i] [s_i = 0] + c1[i] [s_i = 1] + sum_pairs E_{s_a s_b}
    by one maximum flow on the doubled graph (mincut.py).  Returns (labels, labelled): the labels of the nodes the
    roof dual fixes (persistency: some minimiser agrees with them there) and which nodes those are.  Unlabelled
    nodes come back as `keep` (default 0), which by autarky never raises the energy above the all-`keep`
    labelling's when keep is 0 everywhere (Kolmogorov-Rother 2007, Lempitsky et al. 2010)."""
    from sfzmaps.solver.mincut import min_cut
    N = len(c0)
    # reparametrise each pair: E = e00 + (e10-e00) sa + (e01-e00) sb + (e11-e10-e01+e00) sa sb
    lin_a = e10 - e00
    lin_b = e01 - e00
    q = e11 - e10 - e01 + e00            # q <= 0: submodular; q > 0: not
    u1 = c1 - c0
    u1 = u1.astype(float).copy()
    np.add.at(u1, pa, lin_a)
    np.add.at(u1, pb, lin_b)
    sub = q <= 0
    # submodular q sa sb (q<0) = -q (1-sa) sb ... write q sa sb = q sb - q (1-sa) sb: u1[b] += q, edge (a->b) -q
    # non-submodular q sa sb (q>0) = q sa sb, on the doubled graph with the complement nodes
    np.add.at(u1, pb[sub], q[sub])
    # doubled graph: node i (x_i) and node N+i (complement, 1 - x_i)
    # unaries: u1 s_i  ->  half on i, half on its complement (cost u1/2 if s_i=1; on complement cost u1/2 if bar=0)
    h = u1 / 2.0
    src = np.where(h > 0, h, 0.0)        # sink segment (s=1) pays the source cap
    snk = np.where(h < 0, -h, 0.0)
    idx = np.arange(N)
    src2 = np.concatenate([src, snk])   # the complements' terminal capacities are swapped
    snk2 = np.concatenate([snk, src])   # complement: in sink segment means bar=1 i.e. s=0, pays -h if h<0
    # submodular pairs: -q (1-sa) sb  with -q >= 0: edge a->b cap -q/2, and on complements edge bar_b -> bar_a
    sa, sb, qs = pa[sub], pb[sub], -q[sub] / 2.0
    # non-submodular pairs: q sa sb with q > 0 = q sa (1 - bar_b): edge a -> bar_b? (a in sink=1, bar_b in source=0)
    # cost when a=1 and bar_b=0: an edge from bar_b (source side) to a (sink side) with cap q/2, and symmetric
    na, nb, qn = pa[~sub], pb[~sub], q[~sub] / 2.0
    ea = np.concatenate([sa, sb + N, nb + N, na + N])
    eb = np.concatenate([sb, sa + N, na, nb])
    cap = np.concatenate([qs, qs, qn, qn])
    seg = min_cut(2 * N, src2, snk2, ea, eb, cap)[0]
    x, xbar = seg[:N], seg[N:]
    labelled = x != xbar
    out = np.zeros(N, bool) if keep is None else keep.astype(bool).copy()
    out[labelled] = x[labelled]
    return out, labelled
