"""Fine alignment: iterate scan block corners against modern DataSF block corners.

Starting from the rough fit (Task 8, within ~100-150 ft):
  0. Translation vote (once). Every scan corner, mapped by the start transform, votes for
     the displacement to each modern corner within the vote reach (VOTE_REACH x the
     first radius, capped at config.ROUGH_TOL_FT); the peak of the smoothed 2D histogram
     is the start's translation error. It is added to the mapped positions used for the
     first matching only (the transform itself is refit from pairs). Without it a start
     more than about one street width off locks a region onto the next street or block
     (seen at ~100 ft / 1.9 deg).
     Ambiguity guard: the vote returns a zero shift, and RefineResult.vote_status says
     why, when the peak lies within VOTE_EDGE_BINS bins of the reach (the true offset is
     probably outside the window: "edge") or when another smoothed local maximum, more
     than VOTE_PEAK_SEP_BINS bins from the top, is >= VOTE_SECOND_PEAK x the top and is a
     separate peak - the histogram dips below VOTE_VALLEY x its height on the straight
     line to the top ("ambiguous": on a periodic grid "one street over" and "one block
     over" tie, with near-empty bins between). The valley test matters: a rotated start
     smears the true offset into a broad ridge whose bumps are within 5-20% of each other
     but never dip below ~0.8 of the bump between them (measured on the synthetic 20x20
     sheet at 1-1.9 deg), whereas lattice alternatives dip to ~0.04. A confident wrong lock
     is worse than refining from the unshifted start.
Then each iteration:
  1. maps every scan corner to the world through the current transform;
  2. pairs scan and modern corners that are *mutual* nearest neighbours (KD-trees both
     ways) within the current radius. Corners sit ~25-80 ft apart across a street, so a
     one-way nearest neighbour easily grabs the far side; mutual NN plus a radius that
     shrinks only once the fit has settled keeps pairs on the correct side;
  3. rejects pairs whose residual r exceeds median(r) + MAD_K * MAD, MAD being the
     normal-consistent median absolute deviation 1.4826 * median|r - median(r)| (floored at
     MIN_MAD_FT so a near-perfect fit does not reject quantization noise). Only large
     residuals are rejected (one-sided): a small residual is never an outlier;
  4. refits a SheetTransform on the surviving pairs, as source="auto" GCPs (thinned evenly
     to at most MAX_LOOP_GCPS: the TPS solve is cubic in the GCP count).
The radius steps to the next value in `radii_ft` once the median shift of the mapped scan
corners between consecutive fits is < SHIFT_TOL_FT; the loop ends when that happens at
the last radius, or after `max_iter` refits. RefineResult.converged is False exactly when
`max_iter` refits were used up before the last radius settled.

Annealed smoothing: at radius stage i of n the TPS smoothing is
config.TPS_SMOOTHING x COARSE_SMOOTHING_STEP^(n-1-i) (1000, 100, 10, 1 for the default
four radii), so the wide-radius stages are nearly affine. A flexible warp at the wide
radii can carry a whole region one street over and still look self-consistent (seen with
a 40 ft / 1 deg start: q95 error 3.5 ft, some inliers 50-70 ft wrong); stiff early stages
fix the global misfit before local paper stretch is admitted. The final transform uses
config.TPS_SMOOTHING on all inliers.

Anchor GCPs: the final transform is fit on corner matches only, which are far denser and
more precise than label anchors. The start transform's non-auto GCPs (anchors, manual) are
added back only when fewer than MIN_CORNER_GCPS corner pairs survive, so a sheet with
almost no extracted linework still has a transform.

Labels (one per scan corner, in input order). A last matching pass at the final radius
picks the inlier pairs; the returned transform is refit on exactly those, and then every
other label, wx/wy and dist_ft is measured with that returned transform:
  * inlier        - a mutual NN pair that survived MAD rejection; exactly these are the
                    GCPs of the returned transform. wx/wy is the paired modern corner;
  * no_equivalent - no modern corner within the final radius of the scan corner mapped
                    by the returned transform (also: non-finite input rows, with
                    wx/wy = NaN and dist_ft = inf, and every row when ref_corners is
                    empty);
  * outlier       - a modern corner is within the final radius, but the pair is not
                    mutual or was rejected by MAD.
For non-inliers wx/wy is the nearest modern corner. dist_ft is always the distance from
the scan corner mapped by the returned transform to (wx, wy), so for an inlier it is the
fit residual.

Real change (classify_changes, Task 10). The owner's rule is to keep the historic
geometry where a block edge really moved (street widening, cut-back block face,
redevelopment). Such corners come out of refine as outlier (moved less than the final
radius) or no_equivalent (moved more). classify_changes, run after refine:
  1. candidates: finite scan corners with no modern corner within tol_ft;
  2. counterparts: each candidate may pair with any modern corner within COUNTERPART_FT
     (60 ft, beyond the final radius) that is not already the pair of an inlier; each
     such pair gives an offset vector v = mapped scan corner - modern corner;
  3. a vector's support is the number of distinct candidates (itself included) within
     CLUSTER_FT (600 ft) that have some vector within coherence_ft of it; each candidate
     keeps its best-supported vector (ties: the shortest). This picks a moved block's own
     modern corners over the corners across the street;
  4. clustering: connected components over candidates whose kept vectors differ by
     <= coherence_ft and whose positions are <= CLUSTER_FT apart (DBSCAN-like, min
     points = min_cluster); then each component is trimmed, dropping the member whose
     vector is farthest from the component median, until every pairwise vector
     difference is <= coherence_ft. Components still >= min_cluster are real_change;
  5. the transform is refit (source="auto", config.TPS_SMOOTHING) on the inliers minus
     real_change, and steps 1-5 repeat until the labels stop changing (at most
     MAX_CLASSIFY_ROUNDS rounds).
Matches not in a cluster keep their refine label (outlier / no_equivalent / inlier),
re-measured against the final transform. Real-change matches never enter the fit or the
score; changes_geojson records them with their offsets.

The warp stays smooth: TPS smoothing is never zero, and every pair whose residual is out of
line with the rest is dropped before refitting, so a block drawn in a different place on
the historic sheet is not pulled onto its modern position.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from geopandas import GeoDataFrame
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon, mapping

from sfzmaps import config
from sfzmaps.transform import GCP, SheetTransform

MAD_K = 3.0
MAD_SCALE = 1.4826       # MAD -> standard deviation for normal residuals
MIN_MAD_FT = 0.5         # floor on the scaled MAD (ft)
SHIFT_TOL_FT = 1.0       # median shift below which the fit has stopped moving
MIN_CORNER_GCPS = 10     # fewer corner pairs than this: keep the start's anchor GCPs
COARSE_SMOOTHING_STEP = 10.0  # TPS smoothing x this per radius step above the last
MAX_LOOP_GCPS = 1200     # inside the loop fit on at most this many pairs (speed)
VOTE_REACH = 1.5         # translation vote looks this x the first radius out ...
                         # ... capped at config.ROUGH_TOL_FT
VOTE_BIN_FT = 5.0
VOTE_SIGMA_BINS = 2.0
VOTE_EDGE_BINS = 2       # peak this close to the reach: "edge", zero shift
VOTE_SECOND_PEAK = 0.8   # another peak >= this x the top: "ambiguous", zero shift
VOTE_PEAK_SEP_BINS = 3   # local maxima nearer the top than this are the same peak
VOTE_VALLEY = 0.5        # ... and so are those not separated from it by a dip below this x

CLUSTER_FT = 600.0       # real-change corners link within this distance
COUNTERPART_FT = 60.0    # search radius for a moved corner's modern counterpart
MAX_CLASSIFY_ROUNDS = 5
NOEQ_LINK_FT = 300.0     # no_equivalent corners within this join one hull
NOEQ_BUFFER_FT = 15.0    # buffer for a no_equivalent singleton (or a 2-point/collinear hull)
GEOJSON_CRS = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2227"}}

Label = Literal["inlier", "real_change", "no_equivalent", "outlier"]
VoteStatus = Literal["ok", "none", "edge", "ambiguous"]


@dataclass
class Match:
    px: float
    py: float
    wx: float
    wy: float
    dist_ft: float
    label: Label
    ref_idx: int = -1    # row of (wx, wy) in ref_corners; -1 when there is none


@dataclass
class RefineResult:
    transform: SheetTransform
    matches: list[Match]
    iterations: int
    converged: bool = True               # False: max_iter used up before the final radius settled
    vote_shift_ft: tuple[float, float] = (0.0, 0.0)
    vote_status: VoteStatus = "none"
    final_radius_ft: float = 15.0
    ref_xy: np.ndarray | None = field(default=None, repr=False, compare=False)
    ref_block: np.ndarray | None = field(default=None, repr=False, compare=False)


def _pairs(world: np.ndarray, ref_tree: cKDTree, radius: float):
    """Mutual NN pairs within radius, MAD-filtered.

    Returns (nearest modern index per scan corner, its distance (inf if none within
    radius), boolean inlier mask over scan corners)."""
    d, j = ref_tree.query(world, distance_upper_bound=radius)
    within = np.isfinite(d)
    inl = np.zeros(len(world), bool)
    if within.any():
        idx = np.flatnonzero(within)
        # nearest mapped scan corner of each candidate modern corner
        _, back = cKDTree(world).query(ref_tree.data[j[idx]])
        inl[idx[back == idx]] = True
    if inl.any():
        r = d[inl]
        med = np.median(r)
        mad = max(MAD_SCALE * np.median(np.abs(r - med)), MIN_MAD_FT)
        bad = np.flatnonzero(inl)[r > med + MAD_K * mad]
        inl[bad] = False
    return j, d, inl


def _fit(P, W, j, inl, extra: list[GCP], smoothing: float,
         cap: int | None = None) -> SheetTransform | None:
    idx = np.flatnonzero(inl)
    if cap is not None and len(idx) > cap:
        idx = idx[::int(np.ceil(len(idx) / cap))]  # even thinning, deterministic
    gcps = [GCP(float(p[0]), float(p[1]), float(w[0]), float(w[1]), "auto")
            for p, w in zip(P[idx], W[j[idx]])]
    if len(gcps) < MIN_CORNER_GCPS:
        gcps += extra
    try:
        return SheetTransform(gcps, smoothing=smoothing).fit()
    except ValueError:
        return None


def _vote_reach(radii_ft) -> float:
    return float(min(VOTE_REACH * radii_ft[0], config.ROUGH_TOL_FT))


def _vote_shift(world: np.ndarray, ref_tree: cKDTree, reach: float) -> tuple[np.ndarray, str]:
    """Most common scan->modern displacement among all pairs within `reach` (ft).

    A coarse translation prior: every true pair votes for (about) the same offset while
    wrong pairs spread over many, so the smoothed 2D histogram peaks at the start's
    translation error even when that is larger than a street width. Returns (shift,
    status); the shift is zero unless status is "ok" (see the module docstring)."""
    zero = np.zeros(2)
    nbrs = ref_tree.query_ball_point(world, reach)
    counts = np.fromiter((len(n) for n in nbrs), int, len(nbrs))
    if counts.sum() == 0:
        return zero, "none"
    idx = np.concatenate([np.asarray(n, int) for n in nbrs if n])
    disp = ref_tree.data[idx] - np.repeat(world, counts, axis=0)
    nb = int(np.ceil(reach / VOTE_BIN_FT))
    edges = np.linspace(-nb * VOTE_BIN_FT, nb * VOTE_BIN_FT, 2 * nb + 1)
    h, _, _ = np.histogram2d(disp[:, 0], disp[:, 1], bins=[edges, edges])
    h = gaussian_filter(h, VOTE_SIGMA_BINS, mode="constant")
    ix, iy = np.unravel_index(int(np.argmax(h)), h.shape)
    centres = (edges[:-1] + edges[1:]) / 2
    peak = np.array([centres[ix], centres[iy]])
    if np.hypot(*peak) > reach - VOTE_EDGE_BINS * VOTE_BIN_FT:
        return zero, "edge"
    top = h[ix, iy]
    loc = np.argwhere((h == maximum_filter(h, size=3, mode="constant"))
                      & (h >= VOTE_SECOND_PEAK * top))
    for qx, qy in loc:
        sep = np.hypot(qx - ix, qy - iy)
        if sep <= VOTE_PEAK_SEP_BINS:
            continue
        t = np.linspace(0.0, 1.0, int(np.ceil(2 * sep)) + 1)
        line = h[np.rint(ix + (qx - ix) * t).astype(int), np.rint(iy + (qy - iy) * t).astype(int)]
        if line.min() < VOTE_VALLEY * h[qx, qy]:  # a separate peak, not a bump on a ridge
            return zero, "ambiguous"
    return peak, "ok"


def _smoothing(stage: int, n_stages: int) -> float:
    """Stiff warp at the wide radii, config.TPS_SMOOTHING at the last one."""
    return config.TPS_SMOOTHING * COARSE_SMOOTHING_STEP ** (n_stages - 1 - stage)


def _ref_arrays(ref_corners: GeoDataFrame):
    if len(ref_corners) == 0:
        return np.empty((0, 2)), None
    W = np.column_stack([ref_corners.geometry.x, ref_corners.geometry.y]).astype(float)
    blk = (np.asarray(ref_corners["block_num"]).astype(str)
           if "block_num" in ref_corners.columns else None)
    return W, blk


def _no_eq(px: float, py: float) -> Match:
    return Match(float(px), float(py), np.nan, np.nan, np.inf, "no_equivalent")


def refine(corners_px: np.ndarray, ref_corners: GeoDataFrame, start: SheetTransform, *,
           radii_ft: tuple[float, ...] = (120, 60, 30, 15), max_iter: int = 30) -> RefineResult:
    P_all = np.asarray(corners_px, float).reshape(-1, 2)
    ok = np.isfinite(P_all).all(axis=1)
    P = P_all[ok]
    W, ref_block = _ref_arrays(ref_corners)
    final_r = float(radii_ft[-1])

    def result(tr, matches, it, converged=True, shift=(0.0, 0.0), status="none"):
        return RefineResult(tr, matches, it, converged, (float(shift[0]), float(shift[1])),
                            status, final_r, W, ref_block)

    if len(W) == 0 or len(P) == 0:
        return result(start, [_no_eq(*p) for p in P_all], 0)

    ref_tree = cKDTree(W)
    extra = [GCP(g.px, g.py, g.wx, g.wy, g.source)
             for g in start.gcps if g.enabled and g.source != "auto"]

    tr = start
    world = tr.to_world(P)
    shift, status = _vote_shift(world, ref_tree, _vote_reach(radii_ft))
    world = world + shift
    ri, it, settled, failed = 0, 0, False, False
    while it < max_iter:
        j, _, inl = _pairs(world, ref_tree, radii_ft[ri])
        new = _fit(P, W, j, inl, extra, _smoothing(ri, len(radii_ft)), MAX_LOOP_GCPS)
        if new is None:
            failed = True
            break
        it += 1
        nworld = new.to_world(P)
        moved = float(np.median(np.linalg.norm(nworld - world, axis=1)))
        tr, world = new, nworld
        if moved < SHIFT_TOL_FT:
            if ri == len(radii_ft) - 1:
                settled = True
                break
            ri += 1
    converged = settled or failed  # only running out of max_iter is "not converged"

    # final labelling pass at the last radius; the returned transform is fit on exactly
    # the inlier pairs, then every distance is measured with it
    world = tr.to_world(P)  # drops the vote offset if no refit succeeded
    j, _, inl = _pairs(world, ref_tree, final_r)
    new = _fit(P, W, j, inl, extra, config.TPS_SMOOTHING)
    if new is not None:
        tr = new
        world = tr.to_world(P)
    else:
        inl[:] = False
    dn, jn = ref_tree.query(world)
    kept = iter(range(len(P)))
    matches = []
    for p, good in zip(P_all, ok):
        if not good:
            matches.append(_no_eq(*p))
            continue
        i = next(kept)
        k = int(j[i] if inl[i] else jn[i])
        if inl[i]:
            label = "inlier"
        elif dn[i] > final_r:
            label = "no_equivalent"
        else:
            label = "outlier"
        dist = float(np.linalg.norm(world[i] - W[k]))
        matches.append(Match(float(p[0]), float(p[1]), float(W[k, 0]), float(W[k, 1]),
                             dist, label, k))
    return result(tr, matches, it, converged, shift, status)


# --- real change ----------------------------------------------------------------------

def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def _real_change(world: np.ndarray, valid: np.ndarray, W: np.ndarray, tree: cKDTree,
                 taken: np.ndarray, tol_ft: float, min_cluster: int, coherence_ft: float):
    """One classification round (module docstring, steps 1-4).

    Returns (real-change mask, modern counterpart index (-1 if none)) over all rows."""
    n = len(world)
    rc, cp = np.zeros(n, bool), np.full(n, -1)
    vi = np.flatnonzero(valid)
    dn, _ = tree.query(world[vi])
    cand = vi[dn > tol_ft]
    if len(cand) < min_cluster:
        return rc, cp
    nbrs = tree.query_ball_point(world[cand], COUNTERPART_FT)
    owner = np.repeat(np.arange(len(cand)), [len(x) for x in nbrs])
    if not len(owner):
        return rc, cp
    kref = np.concatenate([np.asarray(x, int) for x in nbrs if x])
    free = ~np.isin(kref, taken)
    owner, kref = owner[free], kref[free]
    if not len(owner):
        return rc, cp
    pos = world[cand[owner]]
    vec = pos - W[kref]

    # support of each vector: distinct other candidates nearby with an agreeing vector
    pr = cKDTree(vec).query_pairs(coherence_ft, output_type="ndarray")
    if len(pr):
        pr = pr[(owner[pr[:, 0]] != owner[pr[:, 1]])
                & (np.linalg.norm(pos[pr[:, 0]] - pos[pr[:, 1]], axis=1) <= CLUSTER_FT)]
    both = np.vstack([pr, pr[:, ::-1]]) if len(pr) else np.empty((0, 2), int)
    uniq = np.unique(np.c_[both[:, 0], owner[both[:, 1]]], axis=0) if len(both) else both
    support = np.bincount(uniq[:, 0], minlength=len(vec)) + 1 if len(uniq) else np.ones(len(vec), int)

    # each candidate keeps its best-supported vector (ties: shortest)
    order = np.lexsort((np.hypot(*vec.T), -support, owner))
    first = np.r_[True, owner[order][1:] != owner[order][:-1]]
    best = order[first]                     # one vector index per owner that has any
    best = best[support[best] >= min_cluster]
    if len(best) < min_cluster:
        return rc, cp
    bv, bp = vec[best], pos[best]
    pr = cKDTree(bv).query_pairs(coherence_ft, output_type="ndarray")
    if len(pr):
        pr = pr[np.linalg.norm(bp[pr[:, 0]] - bp[pr[:, 1]], axis=1) <= CLUSTER_FT]
    comp = _components(len(best), pr[:, 0], pr[:, 1]) if len(pr) else np.arange(len(best))
    for c in np.unique(comp):
        mem = np.flatnonzero(comp == c)
        while len(mem) >= min_cluster:
            v = bv[mem]
            spread = np.linalg.norm(v[:, None] - v[None], axis=2).max()
            if spread <= coherence_ft:
                break
            mem = np.delete(mem, np.argmax(np.linalg.norm(v - np.median(v, 0), axis=1)))
        if len(mem) >= min_cluster:
            rows = cand[owner[best[mem]]]
            rc[rows] = True
            cp[rows] = kref[best[mem]]
    return rc, cp


def classify_changes(result: RefineResult, *, tol_ft: float = config.TOLERANCE_FT,
                     min_cluster: int = 3, coherence_ft: float = 4.0) -> RefineResult:
    """Label coherent clusters of large residuals real_change and refit without them.

    See the module docstring ("Real change"). Returns a new RefineResult; the input is
    not modified."""
    W, ms = result.ref_xy, result.matches
    if W is None or not len(W) or not ms:
        return result
    n = len(ms)
    P = np.array([[m.px, m.py] for m in ms], float)
    ok = np.isfinite(P).all(axis=1)
    base_inl = np.array([m.label == "inlier" for m in ms])
    pair_k = np.array([m.ref_idx for m in ms])
    extra = [g for g in result.transform.gcps if g.enabled and g.source != "auto"]
    tree = cKDTree(W)

    def mapped(tr):
        w = np.full((n, 2), np.nan)
        w[ok] = tr.to_world(P[ok])
        return w

    tr = result.transform
    rc, cp = np.zeros(n, bool), np.full(n, -1)
    for _ in range(MAX_CLASSIFY_ROUNDS):
        new_rc, new_cp = _real_change(mapped(tr), ok, W, tree, pair_k[base_inl & ~rc],
                                      tol_ft, min_cluster, coherence_ft)
        if (new_rc == rc).all() and (new_cp == cp).all():
            break
        rc, cp = new_rc, new_cp
        new = _fit(P, W, pair_k, base_inl & ~rc, extra, config.TPS_SMOOTHING)
        if new is not None:
            tr = new

    world = mapped(tr)
    dn, jn = np.full(n, np.inf), np.full(n, -1)
    if ok.any():
        dn[ok], jn[ok] = tree.query(world[ok])
    out = []
    for i, m in enumerate(ms):
        if not ok[i]:
            out.append(_no_eq(m.px, m.py))
            continue
        if rc[i]:
            k, label = cp[i], "real_change"
        elif base_inl[i]:
            k, label = pair_k[i], "inlier"
        else:
            k = jn[i]
            label = "no_equivalent" if dn[i] > result.final_radius_ft else "outlier"
        k = int(k)
        out.append(Match(m.px, m.py, float(W[k, 0]), float(W[k, 1]),
                         float(np.linalg.norm(world[i] - W[k])), label, k))
    return RefineResult(tr, out, result.iterations, result.converged, result.vote_shift_ft,
                        result.vote_status, result.final_radius_ft, W, result.ref_block)


def changes_geojson(result: RefineResult) -> dict:
    """GeoJSON FeatureCollection of real change and no-modern-equivalent areas.

    Coordinates are EPSG:2227 (US ft), declared with a legacy `crs` member (RFC 7946
    assumes WGS84, so readers must honour it). Features:
      * kind="real_change": a Point at the historic corner (scan corner mapped by the
        sheet transform), with dx_ft/dy_ft = historic - modern counterpart, dist_ft,
        modern_x/modern_y, px/py and block_num (the modern block, when known);
      * kind="no_equivalent": the convex hull of a cluster of no_equivalent corners
        (single linkage within NOEQ_LINK_FT), with n_corners. A singleton, or a hull
        that is a line, is buffered by NOEQ_BUFFER_FT."""
    tr, feats = result.transform, []
    rcm = [m for m in result.matches if m.label == "real_change"]
    if rcm:
        hw = tr.to_world([[m.px, m.py] for m in rcm])
        for m, w in zip(rcm, hw):
            props = {"kind": "real_change", "dx_ft": float(w[0] - m.wx),
                     "dy_ft": float(w[1] - m.wy), "dist_ft": m.dist_ft,
                     "modern_x": m.wx, "modern_y": m.wy, "px": m.px, "py": m.py}
            if result.ref_block is not None and m.ref_idx >= 0:
                props["block_num"] = str(result.ref_block[m.ref_idx])
            feats.append({"type": "Feature", "properties": props,
                          "geometry": {"type": "Point", "coordinates": [float(w[0]), float(w[1])]}})
    ne = np.array([[m.px, m.py] for m in result.matches
                   if m.label == "no_equivalent" and np.isfinite([m.px, m.py]).all()])
    if len(ne):
        w = tr.to_world(ne)
        pr = cKDTree(w).query_pairs(NOEQ_LINK_FT, output_type="ndarray")
        comp = _components(len(w), pr[:, 0], pr[:, 1]) if len(pr) else np.arange(len(w))
        for c in np.unique(comp):
            pts = w[comp == c]
            hull = MultiPoint([tuple(p) for p in pts]).convex_hull
            if not isinstance(hull, Polygon):
                hull = hull.buffer(NOEQ_BUFFER_FT)
            feats.append({"type": "Feature",
                          "properties": {"kind": "no_equivalent", "n_corners": int(len(pts))},
                          "geometry": mapping(hull)})
    return {"type": "FeatureCollection", "crs": GEOJSON_CRS, "features": feats}
