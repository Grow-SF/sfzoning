"""gallery.py SERIES SHEET [MIN_FT] [N]: crops around scored corners with residual >= MIN_FT."""
import sys, json, pickle
import numpy as np, cv2
from pathlib import Path
from shapely.geometry import box, shape
from _common import blocks_corners, out_dir
from sfzmaps import config
from sfzmaps.transform import SheetTransform
D = out_dir()
series, sheet = sys.argv[1], sys.argv[2]
lo = float(sys.argv[3]) if len(sys.argv) > 3 else 9; N = int(sys.argv[4]) if len(sys.argv) > 4 else 16
d = config.WORK_DIR / series / sheet
img = cv2.imread(str(d / "page.png"), cv2.IMREAD_COLOR)
t = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
blocks, corners = blocks_corners()
bj = json.loads((d / "blocks_px.geojson").read_text()); ext = [shape(f["geometry"]) for f in bj["features"]]
rf = json.loads((d / "refine.json").read_text())
hi = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9
ms = [m for m in rf["matches"] if m["label"] in ("inlier", "outlier") and m["dist_ft"] is not None and lo <= m["dist_ft"] < hi]
rng = np.random.default_rng(1); ms = [ms[i] for i in rng.permutation(len(ms))[:N]]
S, Z = 70, 5; tiles = []
for m in ms:
    x0, y0 = int(m["px"]) - S // 2, int(m["py"]) - S // 2
    crop = cv2.resize(img[max(0,y0):y0+S, max(0,x0):x0+S], (S*Z, S*Z), interpolation=cv2.INTER_NEAREST)
    cw = t.to_world(np.array([[x0, y0], [x0+S, y0+S], [x0+S, y0], [x0, y0+S]], float)); bb = box(*cw.min(0), *cw.max(0))
    def Pp(pts): return np.round((t.to_pixel(np.asarray(pts, float)) - [x0, y0]) * Z).astype(np.int32)
    for g in blocks[blocks.intersects(bb)].geometry:
        for poly in getattr(g, "geoms", [g]): cv2.polylines(crop, [Pp(poly.exterior.coords)], True, (0, 0, 255), 1, cv2.LINE_AA)
    for g in ext:
        if g.intersects(box(x0, y0, x0+S, y0+S)):
            cv2.polylines(crop, [np.round((np.asarray(g.exterior.coords) - [x0, y0]) * Z).astype(np.int32)], True, (255, 120, 0), 1, cv2.LINE_AA)
    p = (int((m["px"]-x0)*Z), int((m["py"]-y0)*Z)); q = tuple(int(v) for v in Pp([[m["wx"], m["wy"]]])[0])
    cv2.circle(crop, p, 5, (0, 170, 0), 2); cv2.circle(crop, q, 5, (0, 0, 255), 2); cv2.line(crop, p, q, (0, 170, 0), 1)
    cv2.putText(crop, f"{m['dist_ft']:.1f}ft {m['label'][:3]}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 200), 2)
    tiles.append(crop)
while len(tiles) % 4: tiles.append(np.full_like(tiles[0], 255))
grid = np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)])
fn = D / f"gallery_{series}_{sheet}_{int(lo)}.png"; cv2.imwrite(str(fn), grid); print(fn, len(ms))
