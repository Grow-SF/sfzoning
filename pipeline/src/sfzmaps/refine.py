"""Fine alignment: iterate scan block corners against modern DataSF block corners.

Starting from the rough fit (Task 8, within ~100-150 ft):
  0. Translation vote (once). Every scan corner, mapped by the start transform, votes for
     the displacement to each modern corner within VOTE_REACH x the first radius; the
     peak of the smoothed 2D histogram is the start's translation error. It is added to
     the mapped positions used for the first matching only (the transform itself is
     refit from pairs). Without it a start more than about one street width off locks a
     region onto the next street or block (seen at ~100 ft / 1.9 deg).
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
the last radius, or after `max_iter` refits.

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

Labels (one per scan corner, in input order), from a last matching pass at the final
radius with the final transform:
  * no_equivalent - no modern corner within the final radius;
  * inlier        - a mutual NN pair that survived MAD rejection; exactly these are the
                    GCPs of the returned transform (refit on them once more);
  * outlier       - a modern corner is within the final radius, but the pair is not
                    mutual or was rejected by MAD.
Real change is not classified here (Task 10). Match.wx/wy is always the nearest modern
corner and dist_ft the distance from the scan corner mapped by the returned transform to
it, so for an inlier it is the fit residual.

The warp stays smooth: TPS smoothing is never zero, and every pair whose residual is out of
line with the rest is dropped before refitting, so a block drawn in a different place on
the historic sheet is not pulled onto its modern position.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from geopandas import GeoDataFrame
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from sfzmaps import config
from sfzmaps.transform import GCP, SheetTransform

MAD_K = 3.0
MAD_SCALE = 1.4826       # MAD -> standard deviation for normal residuals
MIN_MAD_FT = 0.5         # floor on the scaled MAD (ft)
SHIFT_TOL_FT = 1.0       # median shift below which the fit has stopped moving
MIN_CORNER_GCPS = 10     # fewer corner pairs than this: keep the start's anchor GCPs
COARSE_SMOOTHING_STEP = 10.0  # TPS smoothing x this per radius step above the last
MAX_LOOP_GCPS = 1200     # inside the loop fit on at most this many pairs (speed)
VOTE_REACH = 1.5         # translation vote looks this x the first radius out
VOTE_BIN_FT = 5.0
VOTE_SIGMA_BINS = 2.0


@dataclass
class Match:
    px: float
    py: float
    wx: float
    wy: float
    dist_ft: float
    label: Literal["inlier", "real_change", "no_equivalent", "outlier"]


@dataclass
class RefineResult:
    transform: SheetTransform
    matches: list[Match]
    iterations: int


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


def _vote_shift(world: np.ndarray, ref_tree: cKDTree, reach: float) -> np.ndarray:
    """Most common scan->modern displacement among all pairs within `reach` (ft).

    A coarse translation prior: every true pair votes for (about) the same offset while
    wrong pairs spread over many, so the smoothed 2D histogram peaks at the start's
    translation error even when that is larger than a street width."""
    nbrs = ref_tree.query_ball_point(world, reach)
    counts = np.fromiter((len(n) for n in nbrs), int, len(nbrs))
    if counts.sum() == 0:
        return np.zeros(2)
    idx = np.concatenate([np.asarray(n, int) for n in nbrs if n])
    disp = ref_tree.data[idx] - np.repeat(world, counts, axis=0)
    nb = int(np.ceil(reach / VOTE_BIN_FT))
    edges = np.linspace(-nb * VOTE_BIN_FT, nb * VOTE_BIN_FT, 2 * nb + 1)
    h, _, _ = np.histogram2d(disp[:, 0], disp[:, 1], bins=[edges, edges])
    h = gaussian_filter(h, VOTE_SIGMA_BINS, mode="constant")
    ix, iy = np.unravel_index(int(np.argmax(h)), h.shape)
    centres = (edges[:-1] + edges[1:]) / 2
    return np.array([centres[ix], centres[iy]])


def _smoothing(stage: int, n_stages: int) -> float:
    """Stiff warp at the wide radii, config.TPS_SMOOTHING at the last one."""
    return config.TPS_SMOOTHING * COARSE_SMOOTHING_STEP ** (n_stages - 1 - stage)


def refine(corners_px: np.ndarray, ref_corners: GeoDataFrame, start: SheetTransform, *,
           radii_ft: tuple[float, ...] = (120, 60, 30, 15), max_iter: int = 30) -> RefineResult:
    P = np.asarray(corners_px, float).reshape(-1, 2)
    W = np.column_stack([ref_corners.geometry.x, ref_corners.geometry.y]).astype(float)
    ref_tree = cKDTree(W)
    extra = [GCP(g.px, g.py, g.wx, g.wy, g.source)
             for g in start.gcps if g.enabled and g.source != "auto"]

    tr = start
    world = tr.to_world(P) if len(P) else P
    if len(P):
        world = world + _vote_shift(world, ref_tree, VOTE_REACH * radii_ft[0])
    ri, it = 0, 0
    while len(P) and it < max_iter:
        j, _, inl = _pairs(world, ref_tree, radii_ft[ri])
        new = _fit(P, W, j, inl, extra, _smoothing(ri, len(radii_ft)), MAX_LOOP_GCPS)
        if new is None:
            break
        it += 1
        nworld = new.to_world(P)
        shift = float(np.median(np.linalg.norm(nworld - world, axis=1)))
        tr, world = new, nworld
        if shift < SHIFT_TOL_FT:
            if ri == len(radii_ft) - 1:
                break
            ri += 1

    # final labelling pass at the last radius; the returned transform is fit on exactly
    # the inlier pairs
    final_r = radii_ft[-1]
    if len(P):
        world = tr.to_world(P)  # drops the vote offset if no refit succeeded
        j, d, inl = _pairs(world, ref_tree, final_r)
        new = _fit(P, W, j, inl, extra, config.TPS_SMOOTHING)
        if new is not None:
            tr = new
            world = tr.to_world(P)
        else:
            inl[:] = False
        _, jn = ref_tree.query(world)
    matches = []
    for i in range(len(P)):
        k = j[i] if inl[i] else jn[i]
        dist = float(np.linalg.norm(world[i] - W[k]))
        if inl[i]:
            label = "inlier"
        elif not np.isfinite(d[i]):
            label = "no_equivalent"
        else:
            label = "outlier"
        matches.append(Match(float(P[i, 0]), float(P[i, 1]), float(W[k, 0]), float(W[k, 1]),
                             dist, label))
    return RefineResult(tr, matches, it)
