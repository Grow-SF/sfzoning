"""Held-out scores for the frame finder.

Every score is a probability-level measure with no cutoff. The log-loss is the
primary one. The edge distance (the 1/2-contour against the label boundary) is
reported on the side only, for comparison with the ICDAR 2021 MapSeg
competition, whose HD95 is the 95th percentile of the pooled two-way distances.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from sfzmaps.frame.data import ink_weight

BAND_PX = 60
_EPS = 1e-6


def bands(mask: np.ndarray, band_px: int = BAND_PX) -> tuple[np.ndarray, np.ndarray]:
    """(frame_band, map_band): outside / inside the mask, within band_px of its edge."""
    m = mask.astype(bool)
    d_out = ndimage.distance_transform_edt(~m)
    d_in = ndimage.distance_transform_edt(m)
    return (~m) & (d_out <= band_px), m & (d_in <= band_px)


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
    w = ink_weight(lum)
    fb, mb = bands(mask)
    everywhere = np.ones(mask.shape, bool)
    e = edge_distance(p, mask)
    return {
        "log_loss_frame_band": log_loss(p, mask, w, fb),
        "log_loss_map_band": log_loss(p, mask, w, mb),
        "log_loss_all_ink": log_loss(p, mask, w, everywhere),
        "frame_called_map": frame_called_map(p, w, fb),
        "map_called_frame": map_called_frame(p, w, mb),
        "edge_median_px": e["median"],
        "edge_hd95_px": e["hd95"],
    }
