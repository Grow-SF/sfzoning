"""overlay.py SERIES SHEET X0 Y0 SIZE [ZOOM]: crop of page.png with modern outlines (red), extracted blocks (blue), matches."""
import sys, json, pickle
import numpy as np, cv2
from pathlib import Path
from shapely.geometry import box, shape
from _common import blocks_corners, out_dir
from sfzmaps import config
from sfzmaps.transform import SheetTransform
D = out_dir()
series, sheet, x0, y0, S = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:6])
Z = int(sys.argv[6]) if len(sys.argv) > 6 else 3
d = config.WORK_DIR / series / sheet
img = cv2.imread(str(d / "page.png"), cv2.IMREAD_COLOR)
crop = img[y0:y0+S, x0:x0+S].copy()
crop = cv2.resize(crop, None, fx=Z, fy=Z, interpolation=cv2.INTER_NEAREST)
t = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
blocks, corners = blocks_corners()
cpx = np.array([[x0, y0], [x0+S, y0], [x0, y0+S], [x0+S, y0+S]], float)
cw = t.to_world(cpx)
bb = box(*cw.min(0), *cw.max(0))
def P(pts):
    q = t.to_pixel(np.asarray(pts, float))
    return np.round((q - [x0, y0]) * Z).astype(np.int32)
for g in blocks[blocks.intersects(bb)].geometry:
    for poly in getattr(g, "geoms", [g]):
        cv2.polylines(crop, [P(poly.exterior.coords)], True, (0, 0, 255), 1, cv2.LINE_AA)
bj = json.loads((d / "blocks_px.geojson").read_text())
for f in bj["features"]:
    g = shape(f["geometry"])
    if not g.intersects(box(x0, y0, x0+S, y0+S)): continue
    for poly in getattr(g, "geoms", [g]):
        q = np.round((np.asarray(poly.exterior.coords) - [x0, y0]) * Z).astype(np.int32)
        cv2.polylines(crop, [q], True, (255, 120, 0), 1, cv2.LINE_AA)
rf = json.loads((d / "refine.json").read_text())
col = {"inlier": (0, 160, 0), "outlier": (0, 200, 255), "real_change": (255, 0, 255), "no_equivalent": (128, 128, 128)}
for m in rf["matches"]:
    if m["px"] is None: continue
    if not (x0 <= m["px"] < x0+S and y0 <= m["py"] < y0+S): continue
    p = (int((m["px"]-x0)*Z), int((m["py"]-y0)*Z))
    cv2.circle(crop, p, 4, col[m["label"]], 2)
    if m["wx"] is not None and m["label"] != "no_equivalent":
        q = P([[m["wx"], m["wy"]]])[0]
        cv2.line(crop, p, tuple(int(v) for v in q), col[m["label"]], 1)
fn = D / f"ov_{series}_{sheet}_{x0}_{y0}.png"
cv2.imwrite(str(fn), crop); print(fn)
