"""(sides_plus: the evidence of dm_sides_plus_NN.npz, with its weights and kinds, into S/plus/probin_NN.npz)
probprep.py for any series/sheet (env SERIES, SHEET; ay.py): inputs of the probabilistic correction.  Every block-line
point (dm_sides) placed by today's fit (pix_NN.npz, the same to_pixel the old chain used), in reg2's crop, its
direction bin; the street width (blocks_px.geojson); the warp's control points; the spread of today's warp's stretch
and shear across the sheet (shape prior).  -> S/probin_NN.npz"""
import json

import cv2
import numpy as np
import shapely
from scipy.spatial import cKDTree

from ay import *  # noqa: F403
from sfzmaps import linework
from sfzmaps.transform import SheetTransform

TJ = json.loads((D / "transform.json").read_text())
T0 = SheetTransform.from_json(TJ)
R2 = np.load(S + f"reg2_{NN}.npz")
x0, y0 = (int(v) for v in R2["origin"])
B, h, w = R2["targets"].shape
B_ = json.loads((D / "blocks_px.geojson").read_text())
SW, _ = linework.estimate_street_px([shapely.geometry.shape(f["geometry"]) for f in B_["features"]])
SD = np.load(S + f"dm_sides_plus_{NN}.npz")
pts, ta, tb = T0.to_pixel(SD["pts"]), T0.to_pixel(SD["ta"]), T0.to_pixel(SD["tb"])
P = pts - [x0, y0]
dv = tb - ta
pbin = np.round((np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180) / 10.0).astype(int) % B
ok = np.isfinite(P).all(1) & np.isfinite(dv).all(1) & (P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)
gc = [q for q in TJ["gcps"] if q.get("enabled", True)]
GW = np.array([[q["wx"], q["wy"]] for q in gc])
GBASE = T0.to_pixel(GW) - [x0, y0]
GPX = np.array([[q["px"], q["py"]] for q in gc]) - [x0, y0]
gy_, gx_ = np.mgrid[0:h:50, 0:w:50]
gq = np.c_[gx_.ravel(), gy_.ravel()].astype(float)
near = cKDTree(P[ok & (SD["kind"] == 0)]).query(gq)[0] <= 100
gq = gq[near] + [x0, y0]
J = np.stack([(T0.to_world(gq + [0.5, 0]) - T0.to_world(gq - [0.5, 0])), (T0.to_world(gq + [0, 0.5]) - T0.to_world(gq - [0, 0.5]))], -1)
K = J @ np.linalg.inv(np.median(J, axis=0)) - np.eye(2)
TAU_J = float(np.sqrt(np.mean(K ** 2)))
Hp, Wp = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE).shape
import os  # noqa: E402
os.makedirs(S + "plus", exist_ok=True)
np.savez(S + f"plus/probin_{NN}.npz", P=P[ok].astype(np.float32), bin=pbin[ok], sid=SD["sid"][ok], keep=ok, street_px=SW,
         wt=SD["wt"][ok].astype(np.float32), kind=SD["kind"][ok],
         origin=np.array([x0, y0]), shape=np.array([Hp, Wp]), gbase=GBASE.astype(np.float32), gpx=GPX.astype(np.float32),
         tau_j=TAU_J)
print(f"{SERIES} {NN}: points {ok.sum()} of {len(ok)}; control points {len(gc)}; street width {SW:.1f} px; "
      f"stretch spread {TAU_J:.4f}; crop {w}x{h} at ({x0},{y0})")
