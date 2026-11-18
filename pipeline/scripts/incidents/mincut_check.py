"""Check of the OR-Tools minimum cut (solver/mincut.py) that replaced PyMaxflow in `search.expansion_moves`.

For each quarter it replays the solver's search (search_lab.solver_sequence on the quarter's rebuilt energy, as the
stage builds it) following the PyMaxflow cut's moves, and at every expansion move's graph also cuts with the new
function: the two cuts' energies (the cost of the cut itself, which is the bounded binary energy of its switch set)
must agree within the scaling bound `min_cut` returns.  It prints the number of cuts, max |difference|, the largest
bound, and the seconds each spent.  PyMaxflow (GPL-3.0) is no longer a project dependency, so run it as

    uv run --with pymaxflow==1.3.2 python scripts/incidents/mincut_check.py nw ne sw se

(SciPy's `maximum_flow` was timed the same way before the choice: 29 times slower than PyMaxflow on quarter nw,
OR-Tools 2.5 times.)  Read-only on the solver."""
import sys
import time

import numpy as np

import search_lab as L
from sfzmaps.solver import mincut, search


def cut_cost(side, src, snk, ea, eb, w):
    return float(src[side].sum() + snk[~side].sum() + w[~side[ea] & side[eb]].sum())


def pymaxflow_cut(n, src, snk, ea, eb, w):
    import maxflow
    G = maxflow.Graph[float](n, len(ea))
    nodes = G.add_nodes(n)
    G.add_grid_tedges(nodes, src, snk)
    G.add_edges(ea.astype(np.int64), eb.astype(np.int64), w, np.zeros(len(w)))
    G.maxflow()
    return G.get_grid_segments(nodes)


def main(quarters):
    for q in quarters:
        P = L.problem(q)
        rec = {"diff": [], "bound": [], "t_py": 0.0, "t_new": 0.0, "n": 0}
        real = search.min_cut

        def both(n, src, snk, ea, eb, w, cba=None):
            t0 = time.perf_counter()
            ref = pymaxflow_cut(n, src, snk, ea, eb, w)
            t1 = time.perf_counter()
            side, bound = mincut.min_cut(n, src, snk, ea, eb, w)
            t2 = time.perf_counter()
            rec["t_py"] += t1 - t0
            rec["t_new"] += t2 - t1
            rec["n"] += 1
            rec["diff"].append(abs(cut_cost(side, src, snk, ea, eb, w) - cut_cost(ref, src, snk, ea, eb, w)))
            rec["bound"].append(bound)
            return ref, bound
        search.min_cut = both
        try:
            L.solver_sequence(P.U, P.a, P.b, P.odds)
        finally:
            search.min_cut = real
        print(f"{q}: {rec['n']} cuts, max |diff| {max(rec['diff']):.3e}, max bound {max(rec['bound']):.3e}, "
              f"within bound at all: {all(d <= b + 1e-9 for d, b in zip(rec['diff'], rec['bound']))}; "
              f"seconds pymaxflow {rec['t_py']:.1f} ortools {rec['t_new']:.1f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["nw", "ne", "sw", "se"])
