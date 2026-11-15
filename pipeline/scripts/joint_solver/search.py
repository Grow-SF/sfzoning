"""Searching for the lowest energy (energy.py).  Every move is priced with the exact energy and kept only if it
lowers it.

  expansion_moves   for each code in turn, the best set of lots to switch to it, by one minimum cut (PyMaxflow)
  region_moves      whole regions (lots of one code joined across borders with no line) take another code, or two
                    touching regions swap codes
  compound_moves    a region takes the code its own labels favour while the neighbour holding that code moves to its
                    own next best code
One-code-at-a-time cuts cannot fix two neighbouring districts holding each other's codes: either switch alone pays
the same-code-across-a-line cost against the other.  The region and compound moves can."""
from collections import Counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .energy import energy, wall_terms


def expansion_moves(U, a0, b0, odds, same_weight, start, sweeps=5, log=None):
    """switch-one-code-at-a-time minimum cuts.  The same-code cost of a border both of whose lots would switch away
    from different codes is bounded by charging it all to the side less willing to switch (exact unless that side
    switches alone); the exact energy decides whether a move is kept."""
    import maxflow
    N, K = U.shape
    a, b, t, ra, rb, rr = wall_terms(a0, b0, odds, same_weight)

    def en(p_):
        return float(U[np.arange(N), p_].sum() + t[p_[a] != p_[b]].sum() + rr[p_[ra] == p_[rb]].sum())

    p = start.copy()
    every = list(range(K))
    for sweep in range(sweeps):
        moved = 0
        order = [k for k, _ in Counter(p.tolist()).most_common()] + [k for k in every if k not in set(p.tolist())]
        for al in order:
            c0 = U[np.arange(N), p].copy()
            c1 = U[:, al].copy()
            fixed = p == al
            A = t * (p[a] != p[b])
            Bc = t * (p[a] != al)
            Cc = t * (p[b] != al)
            np.add.at(c1, a, Cc - A)
            np.add.at(c1, b, -Cc)
            w = Bc + Cc - A
            xa, xb = p[ra], p[rb]
            np.add.at(c1, rb, rr * ((xa == al) & (xb != al)))
            np.add.at(c1, ra, rr * ((xb == al) & (xa != al)))
            both = (xa != al) & (xb != al) & (xa != xb)
            a_keener = (U[ra, al] - U[ra, xa]) <= (U[rb, al] - U[rb, xb])
            np.add.at(c1, np.where(a_keener, rb, ra), rr * both)
            same = (xa != al) & (xa == xb)
            np.add.at(c0, np.where(a_keener, ra, rb), rr * same)
            np.add.at(c1, np.where(a_keener, rb, ra), rr * same)
            m = np.minimum(c0, c1)
            c0 -= m
            c1 -= m
            G = maxflow.Graph[float](N, len(a))
            nodes = G.add_nodes(N)
            G.add_grid_tedges(nodes, c1, c0)
            G.add_edges(a.astype(np.int64), b.astype(np.int64), np.maximum(w, 0.0), np.zeros(len(w)))
            G.maxflow()
            switch = G.get_grid_segments(nodes) & ~fixed
            if switch.any():
                q = p.copy()
                q[switch] = al
                if en(q) < en(p) - 1e-6:
                    p = q
                    moved += int(switch.sum())
        if log is not None:
            log.append(("expansion", sweep, moved, en(p)))
        if moved == 0:
            break
    return p


def _regions(a, b, odds, p):
    """regions: lots of one code joined across borders with no line (odds <= 0)"""
    N = len(p)
    same = (p[a] == p[b]) & (odds <= 0)
    _, reg = connected_components(coo_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(N, N)),
                                  directed=False)
    return reg, reg.max() + 1


def region_moves(U, a, b, odds, same_weight, p, rounds=50, log=None):
    """recode: a region takes whichever code lowers the energy most; swap: two touching regions exchange codes.
    The best non-overlapping improvements are applied together, then the exact energy is checked; repeated until
    nothing improves."""
    N, K = U.shape
    t = np.maximum(-odds, 0.0)
    rr = np.maximum(odds, 0.0) * same_weight
    p = p.copy()
    for rd in range(rounds):
        reg, R = _regions(a, b, odds, p)
        rc = np.zeros(R, int)
        rc[reg] = p
        SU = np.zeros((R, K))
        np.add.at(SU, reg, U)
        x = reg[a] != reg[b]
        ra_, rb_ = reg[a[x]], reg[b[x]]
        lo_r, hi_r = np.minimum(ra_, rb_), np.maximum(ra_, rb_)
        uk, inv = np.unique(lo_r.astype(np.int64) * R + hi_r, return_inverse=True)
        Tp = np.bincount(inv, t[x])                      # per touching pair of regions: no-line costs, line costs
        Rp = np.bincount(inv, rr[x])
        P1, P2 = uk // R, uk % R
        codes = np.arange(K)[None, :]
        E = SU.copy()                                    # E[region, code]: its energy at that code, neighbours as now
        np.add.at(E, P1, Tp[:, None] * (codes != rc[P2][:, None]) + Rp[:, None] * (codes == rc[P2][:, None]))
        np.add.at(E, P2, Tp[:, None] * (codes != rc[P1][:, None]) + Rp[:, None] * (codes == rc[P1][:, None]))
        best = E.argmin(1)
        gain = E[np.arange(R), rc] - E[np.arange(R), best]
        c1, c2 = rc[P1], rc[P2]
        e1 = E[P1, c2] - E[P1, c1] - (Rp - Tp)           # after a swap the pair still differs: undo the same-code
        e2 = E[P2, c1] - E[P2, c2] - (Rp - Tp)           #   cost each side counted against the other
        dsw = np.where(c1 != c2, e1 + e2, 0.0)
        cand = [(g_, 0, r_) for r_, g_ in zip(np.flatnonzero(gain > 1e-6), gain[gain > 1e-6])]
        cand += [(-d_, 1, i_) for i_, d_ in zip(np.flatnonzero(dsw < -1e-6), dsw[dsw < -1e-6])]
        cand.sort(key=lambda z: -z[0])
        newc = rc.copy()
        used = np.zeros(R, bool)
        moved = 0
        for _, kind, i_ in cand:
            if kind == 0:
                nb = np.r_[P2[P1 == i_], P1[P2 == i_]]
                if used[i_] or used[nb].any():
                    continue
                newc[i_] = best[i_]
                used[i_] = True
            else:
                r1, r2 = P1[i_], P2[i_]
                nb = np.r_[P2[(P1 == r1) | (P1 == r2)], P1[(P2 == r1) | (P2 == r2)]]
                if used[r1] or used[r2] or used[nb].any():
                    continue
                newc[r1], newc[r2] = rc[r2], rc[r1]
                used[r1] = used[r2] = True
            moved += 1
        q = newc[reg]
        e_old, e_new = energy(U, a, b, odds, p, same_weight), energy(U, a, b, odds, q, same_weight)
        if log is not None:
            log.append(("region", rd, moved, e_new))
        if moved == 0 or e_new >= e_old - 1e-6:
            break
        p = q
    return p


def compound_moves(U, a, b, odds, same_weight, p, passes=6, log=None):
    """a region whose own terms favour another code takes it; every touching region already holding that code moves
    at the same time to its own best code that none of its neighbours holds.  Fixes chains like: this district's
    code is held by a neighbour that should hold a third code."""
    N, K = U.shape
    p = p.copy()
    E0 = energy(U, a, b, odds, p, same_weight)
    for ps in range(passes):
        reg, R = _regions(a, b, odds, p)
        rc = np.zeros(R, int)
        rc[reg] = p
        SU = np.zeros((R, K))
        np.add.at(SU, reg, U)
        x = reg[a] != reg[b]
        nbr = coo_matrix((np.ones(int(x.sum())), (reg[a[x]], reg[b[x]])), shape=(R, R)).tocsr()
        nbr = ((nbr + nbr.T) > 0).tocsr()
        want = SU.argmin(1)
        gain = SU[np.arange(R), rc] - SU[np.arange(R), want]
        cand = np.flatnonzero((want != rc) & (gain > 1e-3))
        cand = cand[np.argsort(-gain[cand])]
        o = np.argsort(reg, kind="stable")
        st = np.searchsorted(reg[o], np.arange(R + 1))
        kept = 0
        for r_ in cand:
            g_ = want[r_]
            if rc[r_] == g_:
                continue
            q = p.copy()
            q[o[st[r_]:st[r_ + 1]]] = g_
            for n_ in nbr.indices[nbr.indptr[r_]:nbr.indptr[r_ + 1]]:
                if rc[n_] != g_:
                    continue
                held = set(rc[nbr.indices[nbr.indptr[n_]:nbr.indptr[n_ + 1]]].tolist()) | {g_}
                alt = next((c for c in np.argsort(SU[n_]) if c not in held), None)
                if alt is not None:
                    q[o[st[n_]:st[n_ + 1]]] = alt
            E1 = energy(U, a, b, odds, q, same_weight)
            if E1 < E0 - 1e-6:
                p, E0 = q, E1
                rc[reg] = p                              # regions keep their ids this pass; codes updated
                kept += 1
        if log is not None:
            log.append(("compound", ps, kept, E0))
        if kept == 0:
            break
    return p
