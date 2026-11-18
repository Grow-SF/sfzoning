"""How well each placement fits one sheet's print, and which correction the warp takes."""
import numpy as np
from scipy.ndimage import map_coordinates

from sfzmaps.sheetwarp.probabilistic import ProbabilisticInputs
from sfzmaps.sheetwarp.targets import SidePoints, Targets, measure
from sfzmaps.transform import Correction


def scores(targets: Targets, sides: SidePoints, inputs: ProbabilisticInputs, first: Correction,
           probabilistic: Correction) -> dict:
    """The share of block-side points within 1 and 2 px of a printed line running their way, for today's placement
    and each correction; how far the placement's control points land from their matched scan pixels (median and
    99th percentile, px) with none and with each correction; and for the probabilistic correction its smallest cell
    determinant (below 0 it folds the scan over) and its largest correction (px)."""
    out = {}
    for name, correction in (("today", None), ("first", first), ("probabilistic", probabilistic)):
        d = measure(targets, sides, correction)
        out[f"within1_{name}"] = round(float((d <= 1).mean()), 3)
        out[f"within2_{name}"] = round(float((d <= 2).mean()), 3)
    gb, gx = inputs.gbase, inputs.gpx          # control points: where today's placement puts them, and their scan pixels
    e = np.linalg.norm(gb - gx, axis=1)
    out["control_miss_p50_today"] = round(float(np.median(e)), 2)
    out["control_miss_p99_today"] = round(float(np.percentile(e, 99)), 2)
    for name, correction in (("first", first), ("probabilistic", probabilistic)):
        G, sp = correction.grid, float(correction.spacing)
        q = gb - (np.asarray(correction.origin, float) - np.asarray(inputs.origin, float))   # both on the same crop
        u = np.stack([map_coordinates(G[k], [q[:, 1] / sp, q[:, 0] / sp], order=1, mode="constant", cval=0.0)
                      for k in (0, 1)], -1)
        e = np.linalg.norm(gb + u - gx, axis=1)
        out[f"control_miss_p50_{name}"] = round(float(np.median(e)), 2)
        out[f"control_miss_p99_{name}"] = round(float(np.percentile(e, 99)), 2)
    G, sp = probabilistic.grid, float(probabilistic.spacing)
    ux_x, uy_x = np.diff(G[0], axis=1) / sp, np.diff(G[1], axis=1) / sp
    ux_y, uy_y = np.diff(G[0], axis=0) / sp, np.diff(G[1], axis=0) / sp
    det = ((1 + 0.5 * (ux_x[1:] + ux_x[:-1])) * (1 + 0.5 * (uy_y[:, 1:] + uy_y[:, :-1]))
           - 0.5 * (ux_y[:, 1:] + ux_y[:, :-1]) * 0.5 * (uy_x[1:] + uy_x[:-1]))
    out["smallest_cell_probabilistic"] = round(float(det.min()), 3)
    out["largest_correction_px_probabilistic"] = round(float(np.linalg.norm(G, axis=0).max()), 1)
    return out


def choose(s: dict) -> str:
    """The probabilistic correction where it puts at least as many block-side points within 1 px of the print as
    the first and its smallest cell is above 0 (it never folds the scan over), else the first.  Today's placement
    is never a fallback: it isn't known to be right."""
    if s["within1_probabilistic"] >= s["within1_first"] and s["smallest_cell_probabilistic"] > 0:
        return "probabilistic"
    return "first"
