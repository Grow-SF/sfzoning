"""The network of lot borders, and the raw scan's ink across every point of it.

Borders are where the grown lot grid (every pixel -> its nearest lot) changes, so a border sits on the lot line where
two lots touch and on the street's middle where they face each other across a street: where a district line is
expected.  Each crack between two pixels of different lots is one point (4 ft).
  paths      cracks chained through corners where only two lots meet; a path ends at a junction (3+ lots meet).
             Every path lies between one pair of lots, in order along its length.
  junctions  the corner each path end meets; borders meeting at a junction are neighbours (measures.py: holes)
  profile    the raw scan sampled across each point, -J..J px along the path's normal (hole causes)
Built once and kept in joint/border_paths.npz."""
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from .data import OUT, city

J = 20                                       # profile half-width, px (80 ft)
SMOOTH = 3                                   # direction smoothing, points each side
PATH = OUT / "border_paths.npz"


def border_paths():
    if not PATH.exists():
        _build()
    return np.load(PATH)


def _build():
    C = city()
    lot, grown = C.lot_raster, C.grown_raster
    H, W = grown.shape
    N = C.n
    keys = C.a.astype(np.int64) * N + C.b
    W1 = W + 1
    rv, cv = np.nonzero(grown[:, :-1] != grown[:, 1:])      # vertical cracks (r,c | r,c+1), horizontal (r,c | r+1,c)
    rh, ch = np.nonzero(grown[:-1, :] != grown[1:, :])
    nv, nh = len(rv), len(rh)
    L1 = np.r_[grown[rv, cv], grown[rh, ch]].astype(np.int64)
    L2 = np.r_[grown[rv, cv + 1], grown[rh + 1, ch]].astype(np.int64)
    lo, hi = np.minimum(L1, L2), np.maximum(L1, L2)
    k = np.searchsorted(keys, lo * N + hi)
    assert np.all(keys[k] == lo * N + hi)
    touch = np.r_[(lot[rv, cv] >= 0) & (lot[rv, cv + 1] >= 0), (lot[rh, ch] >= 0) & (lot[rh + 1, ch] >= 0)]
    c0 = np.r_[rv * W1 + cv + 1, (rh + 1) * W1 + ch]              # corners on the pixel-corner lattice
    c1 = np.r_[(rv + 1) * W1 + cv + 1, (rh + 1) * W1 + ch + 1]
    my = np.r_[rv + 0.5, rh + 1.0]
    mx = np.r_[cv + 1.0, ch + 0.5]
    ny0 = np.r_[np.zeros(nv), np.ones(nh)]                        # the crack's normal, from the pair's first lot
    nx0 = np.r_[np.ones(nv), np.zeros(nh)]                        #   to its second
    flip = L1 != lo
    ny0[flip] *= -1
    nx0[flip] *= -1
    E = len(k)
    cc = np.r_[c0, c1]
    _, inv = np.unique(cc, return_inverse=True)
    deg = np.bincount(inv)
    ci0, ci1 = inv[:E], inv[E:]
    two = deg == 2
    o = np.argsort(inv, kind="stable")                            # chain the two cracks at each degree-2 corner
    eo = np.r_[np.arange(E), np.arange(E)][o]
    io = inv[o]
    st = np.flatnonzero(np.r_[True, io[1:] != io[:-1]])
    pair = np.diff(np.r_[st, len(io)]) == 2
    ea, eb = eo[st[pair]], eo[st[pair] + 1]
    corner_of_link = io[st[pair]]
    G = sparse.coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(E, E)).tocsr()
    ncomp, comp = csgraph.connected_components(G, directed=False)
    isend = ~two[ci0] | ~two[ci1]                                 # path ends; closed loops cut at their first crack
    has_end = np.zeros(ncomp, bool)
    has_end[comp[isend]] = True
    loops = np.flatnonzero(~has_end)
    if len(loops):
        e0 = np.full(ncomp, E, np.int64)
        np.minimum.at(e0, comp, np.arange(E))
        cut = e0[loops]
        keep = ~(np.isin(ea, cut) & (ci0[ea] == corner_of_link) | np.isin(eb, cut) & (ci0[eb] == corner_of_link))
        ea, eb, corner_of_link = ea[keep], eb[keep], corner_of_link[keep]
        G = sparse.coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(E, E)).tocsr()
        isend = isend | (np.bincount(np.r_[ea, eb], minlength=E) < 2)
    start = np.full(ncomp, E, np.int64)
    ends = np.flatnonzero(isend)
    np.minimum.at(start, comp[ends], ends)
    pos = csgraph.dijkstra(G, directed=False, unweighted=True, indices=start, min_only=True).astype(np.int64)
    order = np.lexsort((pos, comp))                               # cracks by path, then along it
    comp_s, pos_s = comp[order], pos[order]
    assert np.all((np.diff(pos_s) == 1) | (np.diff(comp_s) != 0)), "a path is not a simple chain"
    pstart = np.flatnonzero(np.r_[True, comp_s[1:] != comp_s[:-1]])
    plen = np.diff(np.r_[pstart, E])
    pk = k[order[pstart]]
    assert np.all(k[order] == np.repeat(pk, plen)), "a path runs between more than one pair of lots"
    k, touch, my, mx, ny0, nx0 = k[order], touch[order], my[order], mx[order], ny0[order], nx0[order]
    ci0, ci1 = ci0[order], ci1[order]
    i_in = np.arange(E) - np.repeat(pstart, plen)
    ia = np.repeat(pstart, plen) + np.clip(i_in - SMOOTH, 0, None)
    ib = np.repeat(pstart, plen) + np.minimum(i_in + SMOOTH, np.repeat(plen, plen) - 1)
    ty, tx = my[ib] - my[ia], mx[ib] - mx[ia]
    tn = np.hypot(ty, tx)
    one = tn == 0
    ty, tx = np.where(one, -nx0, ty / np.maximum(tn, 1e-9)), np.where(one, ny0, tx / np.maximum(tn, 1e-9))
    ny, nx = tx, -ty
    sg = np.sign(ny * ny0 + nx * nx0)
    sg[sg == 0] = 1
    ny, nx = ny * sg, nx * sg
    ink, row0, col0 = C.scan()
    PH, PW = ink.shape
    prof = np.zeros((E, 2 * J + 1), bool)
    for j in range(-J, J + 1):
        pr = row0 - np.floor(my + j * ny).astype(np.int64)
        pc = np.floor(mx + j * nx).astype(np.int64) - col0
        ok = (pr >= 0) & (pr < PH) & (pc >= 0) & (pc < PW)
        v = np.zeros(E, bool)
        v[ok] = ink[pr[ok], pc[ok]]
        prof[:, j + J] = v

    def end_corner(e, other):
        """the corner of crack e not shared with its neighbour on the path"""
        return np.where((ci0[e] == ci0[other]) | (ci0[e] == ci1[other]), ci1[e], ci0[e])

    first, last = pstart, pstart + plen - 1
    jc_first = np.where(plen > 1, end_corner(first, np.where(plen > 1, pstart + 1, pstart)), ci0[first])
    jc_last = np.where(plen > 1, end_corner(last, np.where(plen > 1, pstart + plen - 2, pstart)), ci1[last])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PATH, k=k.astype(np.int32), touch=touch, my=my.astype(np.float32), mx=mx.astype(np.float32),
                        ny=ny.astype(np.float32), nx=nx.astype(np.float32), prof=prof, pstart=pstart, plen=plen,
                        pk=pk.astype(np.int32), jc_first=jc_first, jc_last=jc_last, J=J)
    print(f"border paths: {E:,} points, {ncomp:,} paths; saved {PATH}")


def junction_neighbours():
    """(i, j): pairs of borders (indices into the city's neighbour pairs) whose paths meet at a junction"""
    Z = border_paths()
    pk = Z["pk"].astype(np.int64)
    Je = np.r_[Z["jc_first"], Z["jc_last"]]
    Ke = np.r_[pk, pk]
    o = np.argsort(Je, kind="stable")
    Je, Ke = Je[o], Ke[o]
    st = np.flatnonzero(np.r_[True, Je[1:] != Je[:-1]])
    cnt = np.diff(np.r_[st, len(Je)])
    first, second = [], []
    for c in range(2, 6):
        s_ = st[cnt == c]
        for i in range(c):
            for j in range(c):
                if i != j:
                    first.append(Ke[s_ + i]), second.append(Ke[s_ + j])
    first, second = np.concatenate(first), np.concatenate(second)
    keep = first != second
    return first[keep], second[keep]
