"""One s-t minimum cut with real-number capacities (the search's graph-cut step).

OR-Tools' max flow (Apache-2.0) takes whole numbers, so the capacities are scaled by a power of two chosen so that
every capacity added together stays under 2**50 (its flows are 64-bit) and rounded.  The rounding is the only
inexactness: each capacity moves by at most half a unit, so any cut's cost moves by at most half of `bound`
(returned: twice the rounding actually done, in the capacities' own units), and the cut returned costs at most the
true minimum plus `bound`.  Nodes the sink can still reach in the residual graph are the sink side, the smallest
minimum cut's sink side, so a node nothing pulls either way stays on the source side, and the cut is the same every
run.

Capacities follow the usual picture: a node pays `src_cap` if it lands on the sink side, `snk_cap` if it lands on
the source side, and an edge a -> b pays its capacity when a is on the source side and b on the sink side."""
import numpy as np
from ortools.graph.python.max_flow import SimpleMaxFlow

BUDGET = 2.0 ** 50


def min_cut(n, src_cap, snk_cap, ea, eb, cab, cba=None):
    """(sink_side, bound): sink_side[i] is True when node i ends on the sink side.  Edges ea[k] -> eb[k] have
    capacity cab[k] and, when `cba` is given, eb[k] -> ea[k] has capacity cba[k]; all capacities are >= 0.
    Raises ValueError for a capacity that is negative or not finite, a node index out of range or arrays of the
    wrong lengths, and RuntimeError if the max flow does not report an optimum."""
    ea = np.asarray(ea, np.int64)
    eb = np.asarray(eb, np.int64)
    parts = [("src_cap", src_cap, n), ("snk_cap", snk_cap, n), ("cab", cab, len(ea))] \
        + ([("cba", cba, len(ea))] if cba is not None else [])
    for name, v, length in parts:
        v = np.asarray(v, float)
        if v.shape != (length,):
            raise ValueError(f"min_cut: {name} has shape {v.shape}, not ({length},)")
        if not np.isfinite(v).all() or (v < 0).any():
            raise ValueError(f"min_cut: {name} holds a capacity that is negative or not finite")
    if eb.shape != ea.shape or (len(ea) and (min(ea.min(), eb.min()) < 0 or max(ea.max(), eb.max()) >= n)):
        raise ValueError(f"min_cut: edge ends must be two equal-length arrays of nodes 0..{n - 1}")
    tails = np.concatenate([np.full(n, n), np.arange(n), ea] + ([eb] if cba is not None else []))
    heads = np.concatenate([np.arange(n), np.full(n, n + 1), eb] + ([ea] if cba is not None else []))
    cap = np.concatenate([np.asarray(v, float) for _, v, _ in parts])
    total = float(cap.sum())
    scale = 1.0 if total <= 0 else 2.0 ** np.floor(np.log2(BUDGET / total))
    whole = np.rint(cap * scale)
    bound = 2.0 * float(np.abs(whole - cap * scale).sum()) / scale
    keep = whole > 0
    sink_side = np.zeros(n, bool)
    if keep.any():
        flow = SimpleMaxFlow()
        flow.add_arcs_with_capacity(tails[keep], heads[keep], whole[keep].astype(np.int64))
        status = flow.solve(n, n + 1)
        if status != SimpleMaxFlow.OPTIMAL:
            raise RuntimeError(f"min_cut: the max flow ended with status {status}")
        reach = np.asarray(flow.get_sink_side_min_cut(), np.int64)
        sink_side[reach[reach < n]] = True
    return sink_side, bound
