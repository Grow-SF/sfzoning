"""One sheet's scores, for today's placement and both corrections: the share of block-side points within 1 and 2 px of
a printed line running their way (measure_<fit>.npz); how far the placement's control points land from their matched
scan pixels (median and 99th percentile, px); and for the probabilistic correction its smallest cell determinant
(below 0 it folds the scan over) and its largest correction.  Appends a line to fits/scores.jsonl."""
import json

import numpy as np
from scipy.ndimage import map_coordinates

from warp_common import FITS, S, SERIES, SHEET

out = {"series": SERIES, "sheet": SHEET}
for f in ("today", "first", "probabilistic"):
    d = np.load(S + f"measure_{f}.npz")["d"]
    out[f"within1_{f}"] = round(float((d <= 1).mean()), 3)
    out[f"within2_{f}"] = round(float((d <= 2).mean()), 3)
I = np.load(S + "probabilistic_inputs.npz")
gb, gx = I["gbase"], I["gpx"]                # control points: where today's placement puts them, and their scan pixels
e = np.linalg.norm(gb - gx, axis=1)
out["control_miss_p50_today"] = round(float(np.median(e)), 2)
out["control_miss_p99_today"] = round(float(np.percentile(e, 99)), 2)
for f in ("first", "probabilistic"):
    Z = np.load(S + f"{f}_correction.npz")
    G, sp = Z["Gc"], float(Z["spacing"])
    q = gb - (np.asarray(Z["origin"], float) - np.asarray(I["origin"], float))     # both on the same crop: no shift
    u = np.stack([map_coordinates(G[k], [q[:, 1] / sp, q[:, 0] / sp], order=1, mode="constant", cval=0.0) for k in (0, 1)], -1)
    e = np.linalg.norm(gb + u - gx, axis=1)
    out[f"control_miss_p50_{f}"] = round(float(np.median(e)), 2)
    out[f"control_miss_p99_{f}"] = round(float(np.percentile(e, 99)), 2)
ux_x, uy_x = np.diff(G[0], axis=1) / sp, np.diff(G[1], axis=1) / sp     # G: the probabilistic correction
ux_y, uy_y = np.diff(G[0], axis=0) / sp, np.diff(G[1], axis=0) / sp
det = ((1 + 0.5 * (ux_x[1:] + ux_x[:-1])) * (1 + 0.5 * (uy_y[:, 1:] + uy_y[:, :-1]))
       - 0.5 * (ux_y[:, 1:] + ux_y[:, :-1]) * 0.5 * (uy_x[1:] + uy_x[:-1]))
out["smallest_cell_probabilistic"] = round(float(det.min()), 3)
out["largest_correction_px_probabilistic"] = round(float(np.linalg.norm(G, axis=0).max()), 1)
print(json.dumps(out))
with open(FITS / "scores.jsonl", "a") as fh:
    fh.write(json.dumps(out) + "\n")
