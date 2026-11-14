"""One sheet's summary, today vs the old correction (p1) vs the probabilistic one (prob6), same measures as the 2000
comparison: block-line points within 1 / 2 px of a printed same-way line (dm_NN_<fit>.npz), the warp's control points'
miss (px: where today's fit + correction puts each control point's map point vs its matched page px), the smallest
cell determinant of prob6 (below 0: a fold).  Appends a JSON line to allyears/run/probsum.jsonl.
usage: ay_probsum.py SERIES SHEET"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

SERIES, NN = sys.argv[1], sys.argv[2]
A = Path(__file__).resolve().parent
RUN = Path(os.environ.get("PW_RUN", A.parents[2] / "data" / "prob-warp" / "run"))
S = RUN / SERIES / NN
out = {"series": SERIES, "sheet": NN}
for f in ("today", "p1", "prob6"):
    p = S / f"dm_{NN}_{f}.npz"
    if p.exists():
        d = np.load(p)["d"]
        out[f"within1_{f}"] = round(float((d <= 1).mean()), 3)
        out[f"within2_{f}"] = round(float((d <= 2).mean()), 3)
I = np.load(S / f"probin_{NN}.npz")
gb, gx = I["gbase"], I["gpx"]
for f in ("p1", "prob6"):
    Z = np.load(S / f"confit_{NN}_{f}.npz")
    G, sp = Z["Gc"], float(Z["spacing"])
    o = np.asarray(Z["origin"], float) - np.asarray(I["origin"], float)     # both on reg2's crop: 0
    q = gb - o
    u = np.stack([map_coordinates(G[k], [q[:, 1] / sp, q[:, 0] / sp], order=1, mode="constant", cval=0.0) for k in (0, 1)], -1)
    e = np.linalg.norm(gb + u - gx, axis=1)
    out[f"gcp_miss_px_p50_{f}"] = round(float(np.median(e)), 2)
    out[f"gcp_miss_px_p99_{f}"] = round(float(np.percentile(e, 99)), 2)
    if f == "prob6":
        ux_x = np.diff(G[0], axis=1) / sp
        uy_x = np.diff(G[1], axis=1) / sp
        ux_y = np.diff(G[0], axis=0) / sp
        uy_y = np.diff(G[1], axis=0) / sp
        det = (1 + 0.5 * (ux_x[1:] + ux_x[:-1])) * (1 + 0.5 * (uy_y[:, 1:] + uy_y[:, :-1])) - 0.5 * (ux_y[:, 1:] + ux_y[:, :-1]) * 0.5 * (uy_x[1:] + uy_x[:-1])
        out["min_det_prob6"] = round(float(det.min()), 3)
        out["max_correction_px_prob6"] = round(float(np.linalg.norm(G, axis=0).max()), 1)
e0 = np.linalg.norm(gb - gx, axis=1)
out["gcp_miss_px_p50_today"] = round(float(np.median(e0)), 2)
out["gcp_miss_px_p99_today"] = round(float(np.percentile(e0, 99)), 2)
print(json.dumps(out))
with open(RUN / "probsum.jsonl", "a") as fh:
    fh.write(json.dumps(out) + "\n")
