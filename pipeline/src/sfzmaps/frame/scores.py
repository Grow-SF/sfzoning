"""Held-out scores for the frame finder.

The goal is ink: frame ink left in the map and map ink cut away, not areas.  So the primary scores are the two ink
amounts and their cost at the owner's ratio (`decide.COST_FRAME`, `decide.COST_MAP`), counted on the decision itself;
shares are reported beside them, but amounts, not shares, are what is compared, because a share hides how much ink a
sheet has.  The multiclass log-loss of the chances is a proper scoring rule, so minimising it gives calibrated chances
(Gneiting & Raftery 2007).  HD95 (the 95th percentile of the pooled two-way boundary distances) is reported on the
side, for comparison with the ICDAR 2021 MapSeg competition.

The cost is split, for reporting only, into `cost_edge`, on ink within `SPLIT_PX` of a class change in the label (where
the layout's patch-sized wobble and the line placement decide), and `cost_layout`, the rest (where only a wrong
layout costs anything).

The sheet is the unit of resampling (Efron & Tibshirani 1993): pixels within a sheet are far from independent, so
intervals come from resampling whole sheets, and two methods are compared on the same sheets (`paired_difference`).
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from sfzmaps.frame.data import SheetId
from sfzmaps.frame.decide import COST_FRAME, COST_MAP

EPS = 1e-6          # numerical: clips chances away from 0 so the log-loss stays finite
BOOT_N = 10_000     # computational: Monte Carlo resamples
INTERVAL = (5, 95)  # reporting convention: percentile interval
SPLIT_PX = 100      # reporting split: ink within this many px of a class change in the label is "edge" ink


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


def percent(share: float) -> str:
    """A share as a percentage for the owner (three significant figures), "n/a" for NaN."""
    return "n/a" if share != share else f"{100.0 * share:.3g}%"


def _share(part: float, total: float) -> float:
    return float(part / total) if total > 0 else float("nan")


def near_class_change(cls: np.ndarray, px: float = SPLIT_PX) -> np.ndarray:
    """Bool, True on px within `px` (Euclidean, px centres) of a class change in `cls`: of a px whose 4-neighbour
    has another class (both px of such a pair count).  All False when `cls` has one class."""
    cls = np.asarray(cls)
    change = np.zeros(cls.shape, bool)
    dy, dx = cls[:-1] != cls[1:], cls[:, :-1] != cls[:, 1:]
    change[:-1] |= dy
    change[1:] |= dy
    change[:, :-1] |= dx
    change[:, 1:] |= dx
    if not change.any():
        return change
    # exact Euclidean distance to the nearest change px (float32; a quarter of scipy's memory on a 44 Mpx page)
    dist = cv2.distanceTransform((~change).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist <= px


def sheet_scores(p: np.ndarray, deleted: np.ndarray, cls: np.ndarray, lum: np.ndarray,
                 cost_frame: float = COST_FRAME, cost_map: float = COST_MAP) -> dict:
    """The held-out scores of a sheet: `p` the (3, h, w) chances, `deleted` the decision's outside | removed, `cls` the
    label's class mask (0 map, 1 outside, 2 removed block), `lum` the page's luminance in 0..1 (ink = 1 - lum).
    Ink amounts: `frame_kept` (outside-frame and removed-block ink left in; `outside_kept` and `block_kept` apart),
    `map_cut`, and each class's total (`frame_ink`, `map_ink`, `outside_ink`, `block_ink`) for pooling shares over
    sheets; `cost` at the given prices, split into `cost_edge` and `cost_layout` (see the module)."""
    deleted = deleted.astype(bool)
    ink = 1.0 - lum.astype(np.float64)
    frame_side, map_side = cls != 0, cls == 0
    kept_frame = frame_side & ~deleted
    cut_map = map_side & deleted
    frame_kept = float(ink[kept_frame].sum())
    map_cut = float(ink[cut_map].sum())
    outside_kept = float(ink[(cls == 1) & ~deleted].sum())
    block_kept = float(ink[(cls == 2) & ~deleted].sum())
    totals = {"frame_ink": float(ink[frame_side].sum()), "map_ink": float(ink[map_side].sum()),
              "outside_ink": float(ink[cls == 1].sum()), "block_ink": float(ink[cls == 2].sum())}
    cost = cost_frame * frame_kept + cost_map * map_cut
    near = near_class_change(cls)
    cost_edge = cost_frame * float(ink[kept_frame & near].sum()) + cost_map * float(ink[cut_map & near].sum())
    pc = np.clip(p.astype(np.float64), EPS, 1.0)
    picked = np.take_along_axis(pc, cls[None].astype(np.int64), axis=0)[0]
    total = ink.sum()
    return {
        "frame_kept": frame_kept,
        "map_cut": map_cut,
        "cost": cost,
        "cost_edge": cost_edge,
        "cost_layout": cost - cost_edge,
        "frame_kept_share": _share(frame_kept, totals["frame_ink"]),
        "map_cut_share": _share(map_cut, totals["map_ink"]),
        "outside_kept": outside_kept,
        "block_kept": block_kept,
        "outside_kept_share": _share(outside_kept, totals["outside_ink"]),
        "block_kept_share": _share(block_kept, totals["block_ink"]),
        **totals,
        "log_loss": float((-np.log(picked) * ink).sum() / total) if total > 0 else float("nan"),
        "hd95_px": edge_distance(~deleted, map_side)["hd95"],
    }


def realised(deleted: np.ndarray, cls: np.ndarray, lum: np.ndarray, cost_frame: float = COST_FRAME,
             cost_map: float = COST_MAP) -> dict:
    """The decision's realised cost alone (`frame_kept`, `map_cut`, `cost`, as `sheet_scores` counts them): what the
    shape numbers are chosen by, without the scores that cost a page-sized distance transform."""
    deleted = deleted.astype(bool)
    ink = 1.0 - lum.astype(np.float64)
    frame_kept = float(ink[(cls != 0) & ~deleted].sum())
    map_cut = float(ink[(cls == 0) & deleted].sum())
    return {"frame_kept": frame_kept, "map_cut": map_cut, "cost": cost_frame * frame_kept + cost_map * map_cut}


def pooled_shares(rows: list[dict]) -> dict:
    """Shares of ink pooled over sheets (each sheet's amounts summed, then divided; `sheet_scores` rows):
    frame (outside and removed blocks together), outside and removed-block ink left in, and map ink cut."""
    def tot(k):
        return float(sum(r[k] for r in rows))
    return {"frame_kept_share": _share(tot("frame_kept"), tot("frame_ink")),
            "outside_kept_share": _share(tot("outside_kept"), tot("outside_ink")),
            "block_kept_share": _share(tot("block_kept"), tot("block_ink")),
            "map_cut_share": _share(tot("map_cut"), tot("map_ink"))}


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
