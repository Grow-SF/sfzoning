"""The probabilistic correction's inputs, on the crop of fit_targets.npz: every block-side point placed by today's
placement, with its direction (one of 18, every 10 degrees); the street width (blocks_px.geojson); the placement's
control points, where today's placement puts them and where they are matched on the scan; and how much today's
placement stretches and shears across the sheet (the spread the correction's own stretch is held to).
With "extra_evidence": the extra evidence points (extra_evidence_points.npz), with their weights and kinds.
-> probabilistic_inputs.npz or extra_evidence_inputs.npz.  usage: probabilistic_inputs.py [extra_evidence]"""
import json
import sys

import numpy as np
import shapely
from scipy.spatial import cKDTree

from sfzmaps import linework
from sfzmaps.transform import SheetTransform

from warp_common import D, S, SERIES, SHEET, side_pixels

EXTRA = sys.argv[1:] == ["extra_evidence"]
TJ = json.loads((D / "transform.json").read_text())
T0 = SheetTransform.from_json(TJ)
FT = np.load(S + "fit_targets.npz")
x0, y0 = (int(v) for v in FT["origin"])
B, h, w = FT["targets"].shape
SW, _ = linework.estimate_street_px([shapely.geometry.shape(f["geometry"])
                                     for f in json.loads((D / "blocks_px.geojson").read_text())["features"]])
if EXTRA:
    SD = np.load(S + "extra_evidence_points.npz")
    pts, ta, tb = T0.to_pixel(SD["pts"]), T0.to_pixel(SD["ta"]), T0.to_pixel(SD["tb"])
else:
    SD = np.load(S + "block_side_points.npz")
    pts, ta, tb = side_pixels(T0, SD)
P = pts - [x0, y0]
dv = tb - ta
pbin = np.round((np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180) / 10.0).astype(int) % B
ok = np.isfinite(P).all(1) & np.isfinite(dv).all(1) & (P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)
gc = [q for q in TJ["gcps"] if q.get("enabled", True)]
GBASE = T0.to_pixel(np.array([[q["wx"], q["wy"]] for q in gc])) - [x0, y0]
GPX = np.array([[q["px"], q["py"]] for q in gc]) - [x0, y0]
# stretch and shear: today's placement's local Jacobian on a 50 px grid near the block sides, against its median
gy_, gx_ = np.mgrid[0:h:50, 0:w:50]
gq = np.c_[gx_.ravel(), gy_.ravel()].astype(float)
near = cKDTree(P[ok & (SD["kind"] == 0)] if EXTRA else P[ok]).query(gq)[0] <= 100
gq = gq[near] + [x0, y0]
J = np.stack([(T0.to_world(gq + [0.5, 0]) - T0.to_world(gq - [0.5, 0])), (T0.to_world(gq + [0, 0.5]) - T0.to_world(gq - [0, 0.5]))], -1)
K = J @ np.linalg.inv(np.median(J, axis=0)) - np.eye(2)
TAU_J = float(np.sqrt(np.mean(K ** 2)))
extra = dict(wt=SD["wt"][ok].astype(np.float32), kind=SD["kind"][ok]) if EXTRA else {}
np.savez(S + ("extra_evidence_inputs.npz" if EXTRA else "probabilistic_inputs.npz"), P=P[ok].astype(np.float32),
         bin=pbin[ok], street_px=SW, **extra, origin=np.array([x0, y0]),
         gbase=GBASE.astype(np.float32), gpx=GPX.astype(np.float32), tau_j=TAU_J)
print(f"{SERIES} {SHEET}: points {ok.sum()} of {len(ok)}; control points {len(gc)}; street width {SW:.1f} px; "
      f"stretch spread {TAU_J:.4f}; crop {w} x {h} at ({x0}, {y0})")
