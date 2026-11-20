"""Held-out scores for the frame finder.

Every score is a probability-level measure with no cutoff. The log-loss is the
primary one. The edge distance (the 1/2-contour against the label boundary) is
reported on the side only, for comparison with the ICDAR 2021 MapSeg
competition, whose HD95 is the 95th percentile of the pooled two-way distances.

The labels are only so precise: their outlines sit a median of 4-6 px inside the
printed frame rule, never more than about 8 px (issue #71's label QA).  Px within
`UNSURE_PX` of a label outline, on either side and around holes too (`unsure`),
count in no score here and weigh 0 in both passes' training losses.  The edge
distance still measures the outlines themselves.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from sfzmaps.frame.data import ink_weight

BAND_PX = 60
UNSURE_PX = 8
_EPS = 1e-6


def _sides(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(m, d_in, d_out): the mask as bool, each inside px's distance to the nearest outside px and each outside px's
    to the nearest inside px (0 on the other side); None for a mask with no outline (all map or none)."""
    m = mask.astype(bool)
    if m.all() or not m.any():
        return None
    return m, ndimage.distance_transform_edt(m), ndimage.distance_transform_edt(~m)


def unsure(mask: np.ndarray, px: float = UNSURE_PX) -> np.ndarray:
    """Bool, True on the px within `px` of the mask's outline on either side: an inside px whose nearest outside px
    is at most `px` away, or an outside px whose nearest inside px is.  A hole's outline is part of the mask's, so it
    gets the strip too; the page's edge is not an outline.  All False for a mask with no outline."""
    sides = _sides(mask)
    if sides is None:
        return np.zeros(mask.shape, bool)
    m, d_in, d_out = sides
    return np.where(m, d_in <= px, d_out <= px)


def bands(mask: np.ndarray, band_px: int = BAND_PX) -> tuple[np.ndarray, np.ndarray]:
    """(frame_band, map_band): outside / inside the mask, beyond the uncertain strip (`unsure`) and within band_px of
    its edge.  Both empty for a mask with no outline."""
    sides = _sides(mask)
    if sides is None:
        empty = np.zeros(mask.shape, bool)
        return empty, empty.copy()
    m, d_in, d_out = sides
    return (~m) & (d_out > UNSURE_PX) & (d_out <= band_px), m & (d_in > UNSURE_PX) & (d_in <= band_px)


def _wmean(values: np.ndarray, w: np.ndarray, where: np.ndarray) -> float:
    ww = w[where].astype(np.float64)
    total = ww.sum()
    if total <= 0:
        return float("nan")
    return float((values[where].astype(np.float64) * ww).sum() / total)


def log_loss(p: np.ndarray, mask: np.ndarray, w: np.ndarray, where: np.ndarray) -> float:
    pc = np.clip(p, _EPS, 1 - _EPS).astype(np.float64)
    loss = np.where(mask.astype(bool), -np.log(pc), -np.log1p(-pc))
    return _wmean(loss, w, where)


def frame_called_map(p: np.ndarray, w: np.ndarray, frame_band: np.ndarray) -> float:
    return _wmean(p, w, frame_band)


def map_called_frame(p: np.ndarray, w: np.ndarray, map_band: np.ndarray) -> float:
    return _wmean(1.0 - p, w, map_band)


def _boundary(region: np.ndarray) -> np.ndarray:
    return region & ~ndimage.binary_erosion(region, border_value=1)


def edge_distance(p: np.ndarray, mask: np.ndarray) -> dict:
    """Median and HD95 (pooled two-way) between the 1/2-contour of p and the mask boundary."""
    a = _boundary(p >= 0.5)
    b = _boundary(mask.astype(bool))
    if not a.any() or not b.any():
        return {"median": float("inf"), "hd95": float("inf")}
    to_b = ndimage.distance_transform_edt(~b)[a]
    to_a = ndimage.distance_transform_edt(~a)[b]
    pooled = np.concatenate([to_b, to_a])
    return {"median": float(np.median(pooled)), "hd95": float(np.percentile(pooled, 95))}


def sheet_scores(p: np.ndarray, mask: np.ndarray, lum: np.ndarray) -> dict:
    """The held-out scores of a sheet: log-losses over the two bands and over all ink beyond the uncertain strip,
    the bands' mean probabilities of the wrong side, and (on the side) the edge distance."""
    w = ink_weight(lum)
    fb, mb = bands(mask)
    sure = ~unsure(mask)
    e = edge_distance(p, mask)
    return {
        "log_loss_frame_band": log_loss(p, mask, w, fb),
        "log_loss_map_band": log_loss(p, mask, w, mb),
        "log_loss_all_ink": log_loss(p, mask, w, sure),
        "frame_called_map": frame_called_map(p, w, fb),
        "map_called_frame": map_called_frame(p, w, mb),
        "edge_median_px": e["median"],
        "edge_hd95_px": e["hd95"],
    }
