"""The block-side points: a point every 2 ft along every side of every block of the sheet's year inside its drawn map
(not the Presidio), each with the two ends of its tangent (3 ft either way).  Sides are cut where they turn more than
30 degrees.  -> block_side_points.npz (pts, sid: which side, ta, tb), and today's placement of them on the page
(block_side_points_on_page.npz)."""
import json

import cv2
import numpy as np
import shapely

from sfzmaps.transform import SheetTransform

from warp_common import D, S, SHEET, lots_and_area, side_pixels

T0 = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
lots, area = lots_and_area()
Hp, Wp = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE).shape
pg = shapely.Polygon(T0.to_world(np.array([[0, 0], [Wp, 0], [Wp, Hp], [0, Hp]], float)))
blocks = lots[lots.intersects(pg)].dissolve(by="block_num").geometry
pts, sid_, ta, tb = [], [], [], []
sid = 0
for g in blocks.boundary:
    for ring in shapely.get_parts(g):
        for ln in shapely.get_parts(ring.intersection(area)):
            if ln.geom_type != "LineString" or ln.length < 4:
                continue
            c = np.asarray(shapely.simplify(ln, 2.0).coords)[:, :2]
            if len(c) < 2:
                continue
            ang = np.degrees(np.arctan2(*np.diff(c, axis=0)[:, ::-1].T))
            turn = np.abs((np.diff(ang) + 180) % 360 - 180)
            cuts = [0] + [i + 1 for i in np.flatnonzero(turn > 30)] + [len(c) - 1]
            for a, b in zip(cuts[:-1], cuts[1:]):
                sl = shapely.LineString(c[a:b + 1])
                if sl.length < 2:
                    continue
                for v in np.arange(0, sl.length, 2.0):
                    pts.append(sl.interpolate(v).coords[0][:2]); sid_.append(sid)
                    ta.append(sl.interpolate(max(0, v - 3)).coords[0][:2]); tb.append(sl.interpolate(min(sl.length, v + 3)).coords[0][:2])
                sid += 1
np.savez(S + "block_side_points.npz", pts=np.array(pts), sid=np.array(sid_), ta=np.array(ta), tb=np.array(tb))
side_pixels(T0, np.load(S + "block_side_points.npz"))
print(f"{SHEET}: {len(pts)} block-side points on {sid} sides")
