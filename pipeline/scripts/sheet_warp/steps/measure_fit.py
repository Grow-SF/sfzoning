"""How well a placement fits the print: for every block-side point placed on the page, its distance to the nearest
printed block-line centre running its way (within about 20 degrees: fit_targets.npz).  FIT is "today" (the pipeline's
placement), "first" or "probabilistic" (today's placement with that correction added).
-> measure_<FIT>.npz (d: distance per point, px; sid: its side; px: where it is placed).  usage: measure_fit.py FIT"""
import json
import sys

import numpy as np
from scipy.ndimage import map_coordinates

from sfzmaps.transform import SheetTransform

from warp_common import D, S, SHEET, side_pixels

FIT = sys.argv[1]
T0 = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
FT = np.load(S + "fit_targets.npz")
TG = FT["targets"]
x0, y0 = (int(v) for v in FT["origin"])
B, h, w = TG.shape
SD = np.load(S + "block_side_points.npz")
p, pa, pb = side_pixels(T0, SD)
if FIT != "today":                           # a correction adds its field where today's placement puts each point
    Z = np.load(S + f"{FIT}_correction.npz")
    G, sp, o = Z["Gc"], float(Z["spacing"]), np.asarray(Z["origin"], float)

    def u(q):
        c = [(q[:, 1] - o[1]) / sp, (q[:, 0] - o[0]) / sp]
        return np.stack([map_coordinates(G[k], c, order=1, mode="constant", cval=0.0) for k in (0, 1)], -1)

    p, pa, pb = p + u(p), pa + u(pa), pb + u(pb)
dv = pb - pa
b = np.round((np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180) / 10.0).astype(int) % B
q = p - [x0, y0]
ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
d = np.full(len(p), np.inf)
d[ok] = TG[b[ok], np.round(q[ok, 1]).astype(int), np.round(q[ok, 0]).astype(int)].astype(float)
print(f"{SHEET} {FIT}: block-side points {len(d)}; within 1 px {np.mean(d <= 1):.1%}, 2 px {np.mean(d <= 2):.1%}, 4 px {np.mean(d <= 4):.1%}")
np.savez(S + f"measure_{FIT}.npz", d=d, sid=SD["sid"], px=p)
