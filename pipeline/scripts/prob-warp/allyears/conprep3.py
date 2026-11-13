"""Correction inputs from the plain measure (distmeasure + heldrule): the side points of dm_sides_NN.npz placed by
today's fit; a side is held if heldrule says so (every measurable point within 1 px of its printed line), else
free.  Held sides: their points + normals (held across the line only).  Control points: held if every side at the
corner is held.  Thrown-out corners as before.  -> con_NN.npz (same fields as conprep2).  usage: conprep3.py NN"""
import json
import sys

import numpy as np
import shapely
from scipy.spatial import cKDTree

from sfzmaps import config
from sfzmaps.districts import read
from sfzmaps.transform import SheetTransform

from ay import *  # noqa: F403
TJ = json.loads((D / "transform.json").read_text())
T0 = SheetTransform.from_json(TJ)
R2 = np.load(S + f"reg2_{NN}.npz")
x0, y0 = (int(v) for v in R2["origin"])
B, h, w = R2["targets"].shape
SD = np.load(S + f"dm_sides_{NN}.npz")
held_side = np.load(S + f"dm_held_{NN}.npz")["held"]
sid = SD["sid"]
_pp, _pa, _pb = side_pixels(T0, SD)          # E4: computed once per sheet (bit-identical)
cp = _pp - [x0, y0]
dv = _pb - _pa
pdir = np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180
ok = np.isfinite(cp).all(1) & (cp[:, 0] >= 0) & (cp[:, 0] < w) & (cp[:, 1] >= 0) & (cp[:, 1] < h) & np.isfinite(pdir)
cp, sid, pdir = cp[ok], sid[ok], pdir[ok]
pbin = np.round(pdir / 10.0).astype(int) % B
held = held_side[sid]
th = np.radians(pdir)
nrm = np.column_stack([-np.sin(th), np.cos(th)])
gc_all = [g for g in TJ["gcps"] if g.get("enabled", True)]
gc = np.array([[g["px"], g["py"]] for g in gc_all], float)
gcr = T0.to_pixel(np.array([[g["wx"], g["wy"]] for g in gc_all], float)) - [x0, y0]
kt = cKDTree(cp)
gc_free = np.array([bool(np.any(~held_side[sid[kt.query_ball_point(q, 4.0)]])) if len(kt.query_ball_point(q, 4.0)) else False for q in gcr])
np.save(S + f"gcfree_{NN}.npy", gc_free)
lots = read.load_parcels(year=year())
part = PART()
area = part.difference(lots[lots["block_num"] == "1300"].union_all().buffer(8.0))
M = json.loads((D / "refine.json").read_text())["matches"]
rc = np.array([(m["px"], m["py"], m["wx"], m["wy"]) for m in M if m["label"] in ("no_equivalent", "outlier") and m["wx"] is not None
               and np.isfinite(m["dist_ft"]) and 15.0 < m["dist_ft"] <= 60.0], float).reshape(-1, 4)
rc = rc[shapely.contains_xy(area, rc[:, 2], rc[:, 3])] if len(rc) else rc
corner_rec = T0.to_pixel(rc[:, 2:4]) if len(rc) else np.zeros((0, 2))
gch = gc[~gc_free]
np.savez(S + f"con_{NN}.npz", held_line=cp[held].astype(np.float32), held_n=nrm[held].astype(np.float32), held_cp=(gch - [x0, y0]).astype(np.float32),
         held=np.vstack([cp[held], gch - [x0, y0]]).astype(np.float32), free=cp[~held].astype(np.float32), free_bin=pbin[~held],
         corner_rec=(corner_rec - [x0, y0]).astype(np.float32), corner_scan=(rc[:, :2] - [x0, y0]).astype(np.float32), origin=R2["origin"], shape=R2["shape"])
print(f"p{NN}: held sides {held_side.sum()} of {len(held_side)}; held points {held.sum()}, free points {(~held).sum()}; "
      f"control points held {(~gc_free).sum()}, free {gc_free.sum()}; thrown-out corners {len(rc)}")
