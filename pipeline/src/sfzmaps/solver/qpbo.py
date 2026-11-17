"""Roof duality (QPBO) for a binary energy, by one minimum cut (mincut.py), and its improve step: the search's fusion
move (search.fusion_move) solves its binary problem with them.

A binary energy over variables s (bool per variable):

    sum_i c0[i] [s_i = 0] + c1[i] [s_i = 1]  +  sum_k E_k(s_pa[k], s_pb[k]),   E_k(0, 0) = e00[k], (0, 1) = e01[k], ...

A pair is submodular when e00 + e11 <= e01 + e10; a minimum cut minimises an energy whose pairs all are.  Roof duality
(Hammer, Hansen & Simeone 1984; Boros & Hammer 2002; Kolmogorov & Rother 2007) takes any pairs: one minimum cut of
the doubled graph (each variable i and its complement i', with half of every term on each copy) labels the
variables whose two copies disagree and leaves the rest open.  Persistency: some minimiser of the energy agrees with
the labelled variables.  Autarky: putting the labelled variables' values into any labelling never raises its energy,
so the open variables can keep 0.  `improve` then gives each connected piece of open variables 1 whole when that
lowers the energy (a simple form of QPBO-I's improve step, Rother et al. 2007)."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from sfzmaps.solver.mincut import min_cut


def binary_energy(c0, c1, pa, pb, e00, e01, e10, e11, s) -> float:
    s = np.asarray(s, bool)
    sa, sb = s[pa], s[pb]
    pair = np.where(sa, np.where(sb, e11, e10), np.where(sb, e01, e00))
    return float(np.where(s, c1, c0).sum() + pair.sum())


def roof_dual(c0, c1, pa, pb, e00, e01, e10, e11):
    """(s, labelled): the roof dual's labels (open variables come back False) and which variables it labels."""
    N = len(c0)
    pa, pb = np.asarray(pa, np.int64), np.asarray(pb, np.int64)
    # each pair: e00 + (e10 - e00) s_a + (e01 - e00) s_b + q s_a s_b, q = e11 - e10 - e01 + e00 (q <= 0: submodular)
    q = e11 - e10 - e01 + e00
    u1 = np.asarray(c1, float) - np.asarray(c0, float)  # the cost of s_i = 1 over s_i = 0
    np.add.at(u1, pa, e10 - e00)
    np.add.at(u1, pb, e01 - e00)
    sub = q <= 0
    np.add.at(u1, pb[sub], q[sub])                      # q s_a s_b = q s_b - q (1 - s_a) s_b: an edge a -> b of -q
    # the doubled graph: node i is s_i (sink side = 1), node N + i its complement.  Half of u1 s_i on each copy: on
    # i, paid on the sink side when positive and on the source side when negative; on the complement the other way.
    h = u1 / 2.0
    pos, neg = np.maximum(h, 0.0), np.maximum(-h, 0.0)
    src = np.concatenate([pos, neg])
    snk = np.concatenate([neg, pos])
    # submodular -q (1 - s_a) s_b / 2: a -> b on the variables, b' -> a' on the complements.  Not submodular
    # q s_a s_b / 2 = q s_a (1 - s_b') / 2: b' -> a, and = q (1 - s_a') s_b / 2: a' -> b
    sa, sb, qs = pa[sub], pb[sub], -q[sub] / 2.0
    na, nb, qn = pa[~sub], pb[~sub], q[~sub] / 2.0
    ea = np.concatenate([sa, sb + N, nb + N, na + N])
    eb = np.concatenate([sb, sa + N, na, nb])
    cap = np.concatenate([qs, qs, qn, qn])
    side, _ = min_cut(2 * N, src, snk, ea, eb, cap)
    x, x_complement = side[:N], side[N:]
    labelled = x != x_complement
    return x & labelled, labelled


def improve(c0, c1, pa, pb, e00, e01, e10, e11, s, labelled):
    """s with each connected piece of the open variables (joined by pairs both of whose variables are open) set to
    1 whole when that lowers the energy, the labelled ones kept.  Two pieces never share a pair, so each piece's
    change is its own, and all are computed at once from the terms."""
    s = np.asarray(s, bool).copy()
    open_ = ~np.asarray(labelled, bool)
    if not open_.any():
        return s
    N = len(s)
    u = np.asarray(c1, float) - np.asarray(c0, float)
    both = open_[pa] & open_[pb]
    _, comp = connected_components(coo_matrix((np.ones(int(both.sum())), (pa[both], pb[both])), shape=(N, N)),
                                   directed=False)
    nc = comp.max() + 1
    delta = np.bincount(comp[open_], u[open_], minlength=nc)
    delta += np.bincount(comp[pa[both]], (e11 - e00)[both], minlength=nc)
    oa = open_[pa] & ~open_[pb]                          # a open, b labelled at s_b
    delta += np.bincount(comp[pa[oa]], np.where(s[pb[oa]], e11[oa] - e01[oa], e10[oa] - e00[oa]), minlength=nc)
    ob = open_[pb] & ~open_[pa]                          # b open, a labelled at s_a
    delta += np.bincount(comp[pb[ob]], np.where(s[pa[ob]], e11[ob] - e10[ob], e01[ob] - e00[ob]), minlength=nc)
    s[open_ & (delta[comp] < -1e-9)] = True
    return s
