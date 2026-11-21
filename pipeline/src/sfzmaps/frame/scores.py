"""Held-out scores for the frame finder.

The goal is ink: frame ink left in the map and map ink cut away, not areas.  So the primary scores are the two ink
amounts and their cost at the owner's ratio (`decide.COST_FRAME`, `decide.COST_MAP`), counted on the decision itself;
shares are reported beside them, but amounts, not shares, are what is compared, because a share hides how much ink a
sheet has.  The multiclass log-loss of the chances is a proper scoring rule, so minimising it gives calibrated chances
(Gneiting & Raftery 2007).  HD95 (the 95th percentile of the pooled two-way boundary distances) is reported on the
side, for comparison with the ICDAR 2021 MapSeg competition.

The sheet is the unit of resampling (Efron & Tibshirani 1993): pixels within a sheet are far from independent, so
intervals come from resampling whole sheets, and two methods are compared on the same sheets (`paired_difference`).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from sfzmaps.frame.data import SheetId
from sfzmaps.frame.decide import COST_FRAME, COST_MAP

EPS = 1e-6          # numerical: clips chances away from 0 so the log-loss stays finite
BOOT_N = 10_000     # computational: Monte Carlo resamples
INTERVAL = (5, 95)  # reporting convention: percentile interval


def _boundary(region: np.ndarray) -> np.ndarray:
    return region & ~ndimage.binary_erosion(region, border_value=1)


def edge_distance(a: np.ndarray, b: np.ndarray) -> dict:
    """Median and HD95 (pooled two-way) between the boundaries of two bool regions; NaN when either has none."""
    ba, bb = _boundary(a.astype(bool)), _boundary(b.astype(bool))
    if not ba.any() or not bb.any():
        return {"median": float("nan"), "hd95": float("nan")}
    to_b = ndimage.distance_transform_edt(~bb)[ba]
    to_a = ndimage.distance_transform_edt(~ba)[bb]
    pooled = np.concatenate([to_b, to_a])
    return {"median": float(np.median(pooled)), "hd95": float(np.percentile(pooled, 95))}


def _share(part: float, total: float) -> float:
    return float(part / total) if total > 0 else float("nan")


def sheet_scores(p: np.ndarray, deleted: np.ndarray, cls: np.ndarray, lum: np.ndarray,
                 cost_frame: float = COST_FRAME, cost_map: float = COST_MAP) -> dict:
    """The held-out scores of a sheet: `p` the (3, h, w) chances, `deleted` the decision's outside | removed, `cls` the
    label's class mask (0 map, 1 outside, 2 removed block), `lum` the page's luminance in 0..1 (ink = 1 - lum)."""
    deleted = deleted.astype(bool)
    ink = 1.0 - lum.astype(np.float64)
    frame_side, map_side = cls != 0, cls == 0
    frame_kept = float(ink[frame_side & ~deleted].sum())
    map_cut = float(ink[map_side & deleted].sum())
    pc = np.clip(p.astype(np.float64), EPS, 1.0)
    picked = np.take_along_axis(pc, cls[None].astype(np.int64), axis=0)[0]
    total = ink.sum()
    return {
        "frame_kept": frame_kept,
        "map_cut": map_cut,
        "cost": cost_frame * frame_kept + cost_map * map_cut,
        "frame_kept_share": _share(frame_kept, ink[frame_side].sum()),
        "map_cut_share": _share(map_cut, ink[map_side].sum()),
        "log_loss": float((-np.log(picked) * ink).sum() / total) if total > 0 else float("nan"),
        "hd95_px": edge_distance(~deleted, map_side)["hd95"],
    }


def bootstrap_mean(values, n: int = BOOT_N, seed: int = 0) -> tuple[float, float, float]:
    """(mean, 5th, 95th percentile) of the means of sheets resampled with replacement; NaNs are dropped first."""
    v = np.asarray(list(values), np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, INTERVAL)
    return float(v.mean()), float(lo), float(hi)


def paired_difference(a: dict[SheetId, float], b: dict[SheetId, float], n: int = BOOT_N,
                      seed: int = 0) -> tuple[float, float, float]:
    """`bootstrap_mean` of a - b over the sheets in both."""
    return bootstrap_mean([a[k] - b[k] for k in sorted(a.keys() & b.keys())], n, seed)
