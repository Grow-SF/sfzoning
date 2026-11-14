"""What the first correction holds and what it moves, on the crop of fit_targets.npz:
  held sides           the points of the sides the print holds (held_sides.npz), with their normals: held across the
                       line, free to slide along it
  held control points  the placement's control points with no free side within 4 px
  thrown-out corners   block corners the pipeline's placement matched 15-60 ft away and threw out (refine.json), with
                       where the scan prints them
-> first_correction_inputs.npz."""
import json

import numpy as np
import shapely
from scipy.spatial import cKDTree

from sfzmaps.transform import SheetTransform

from warp_common import D, S, SHEET, lots_and_area, side_pixels

TJ = json.loads((D / "transform.json").read_text())
T0 = SheetTransform.from_json(TJ)
FT = np.load(S + "fit_targets.npz")
x0, y0 = (int(v) for v in FT["origin"])
B, h, w = FT["targets"].shape
SD = np.load(S + "block_side_points.npz")
held_side = np.load(S + "held_sides.npz")["held"]
sid = SD["sid"]
pp, pa, pb = side_pixels(T0, SD)
cp = pp - [x0, y0]
dv = pb - pa
pdir = np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180
ok = np.isfinite(cp).all(1) & (cp[:, 0] >= 0) & (cp[:, 0] < w) & (cp[:, 1] >= 0) & (cp[:, 1] < h) & np.isfinite(pdir)
cp, sid, pdir = cp[ok], sid[ok], pdir[ok]
held = held_side[sid]
th = np.radians(pdir)
nrm = np.column_stack([-np.sin(th), np.cos(th)])
gc_all = [g for g in TJ["gcps"] if g.get("enabled", True)]
gc = np.array([[g["px"], g["py"]] for g in gc_all], float)
gcr = T0.to_pixel(np.array([[g["wx"], g["wy"]] for g in gc_all], float)) - [x0, y0]
kt = cKDTree(cp)
gc_free = np.array([bool(np.any(~held_side[sid[kt.query_ball_point(q, 4.0)]])) if len(kt.query_ball_point(q, 4.0)) else False for q in gcr])
lots, area = lots_and_area()
M = json.loads((D / "refine.json").read_text())["matches"]
rc = np.array([(m["px"], m["py"], m["wx"], m["wy"]) for m in M if m["label"] in ("no_equivalent", "outlier") and m["wx"] is not None
               and np.isfinite(m["dist_ft"]) and 15.0 < m["dist_ft"] <= 60.0], float).reshape(-1, 4)
rc = rc[shapely.contains_xy(area, rc[:, 2], rc[:, 3])] if len(rc) else rc
corner_rec = T0.to_pixel(rc[:, 2:4]) if len(rc) else np.zeros((0, 2))
gch = gc[~gc_free]
np.savez(S + "first_correction_inputs.npz", held_line=cp[held].astype(np.float32), held_n=nrm[held].astype(np.float32),
         held_cp=(gch - [x0, y0]).astype(np.float32), corner_rec=(corner_rec - [x0, y0]).astype(np.float32),
         corner_scan=(rc[:, :2] - [x0, y0]).astype(np.float32), origin=FT["origin"], shape=FT["shape"])
print(f"{SHEET}: held sides {held_side.sum()} of {len(held_side)}; held points {held.sum()}, free points {(~held).sum()}; "
      f"control points held {(~gc_free).sum()}, free {gc_free.sum()}; thrown-out corners {len(rc)}")
