"""The plain measure: for every point (every 2 ft) along every 2000 block side inside the drawn map (no Presidio),
placed on the page by a fit, its distance to the nearest printed block-line centre running the same way (within
~20 deg; reg2 maps).  No pieces, no sliding.
  side points (world) are built once and cached: dm_sides_NN.npz (point, side id, tangent ends)
  fit: "today" (transform.json) or a correction tag (today's fit + confit_NN_<tag>.npz, nothing re-fitted)
Prints: share of block-line length within 1 / 2 / 4 px, overall and in the Laurel Heights band; per side the
largest distance; area by area (250 px cells) when given a second fit to compare with.
Writes dm_NN_<fit>.npz (per point distance, side id, placed px) and pictures dm_NN_<fit>.png (whole drawn map)
and dm_NN_<fit>_laurel.png: points green <= 1 px, yellow 1-2, orange 2-4, red > 4.
usage: distmeasure.py NN FIT [COMPARE_FIT]"""
import json
import os
import sys

import cv2
import numpy as np
import shapely

from sfzmaps import config
from sfzmaps.districts import read
from sfzmaps.transform import SheetTransform

FIT = sys.argv[1]
from ay import *  # noqa: F403,E402
CMP = sys.argv[2] if len(sys.argv) > 2 else None
sys.path.insert(0, S)
from composed import Composed  # noqa: E402

T0 = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
R2 = np.load(S + f"reg2_{NN}.npz")
TG = R2["targets"]
x0, y0 = (int(v) for v in R2["origin"])
B, h, w = TG.shape
cache = S + f"dm_sides_{NN}.npz"
if not os.path.exists(cache):
    lots = read.load_parcels(year=year())
    part = PART()
    area = part.difference(lots[lots["block_num"] == "1300"].union_all().buffer(8.0))
    Hp, Wp = (int(v) for v in R2["shape"])
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
    np.savez(cache, pts=np.array(pts), sid=np.array(sid_), ta=np.array(ta), tb=np.array(tb))
SD = np.load(cache)
PW, SID, TA, TB = SD["pts"], SD["sid"], SD["ta"], SD["tb"]


def fit_of(name):
    if name == "today":
        return T0
    Z = np.load(S + f"confit_{NN}_{name}.npz")
    return Composed(T0, Z["Gc"], Z["spacing"], Z["origin"])


def measure(name):
    # E4: today's to_pixel of the side points once per sheet (side_pixels); a correction adds its u there, exactly as
    # Composed.to_pixel does (p = today's to_pixel(w); p + u(p))
    t = fit_of(name)
    p0, a0, b0 = side_pixels(T0, SD)
    if name == "today":
        p, pa, pb = p0, a0, b0
    else:
        p, pa, pb = p0 + t.u(p0), a0 + t.u(a0), b0 + t.u(b0)
    dv = pb - pa
    b = np.round((np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180) / 10.0).astype(int) % B
    q = p - [x0, y0]
    ok = np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
    d = np.full(len(p), np.inf)
    d[ok] = TG[b[ok], np.round(q[ok, 1]).astype(int), np.round(q[ok, 0]).astype(int)].astype(float)
    return d, p


d, p = measure(FIT)
LH = (p[:, 0] >= 2300) & (p[:, 0] < 2900) & (p[:, 1] >= 2950) & (p[:, 1] < 3190)
print(f"p{NN} {FIT}: block-line points {len(d)} (sides {SID.max() + 1}); within 1 px {np.mean(d <= 1):.1%}, 2 px {np.mean(d <= 2):.1%}, 4 px {np.mean(d <= 4):.1%}; "
      f"Laurel Heights band: within 1 px {np.mean(d[LH] <= 1):.1%}, 2 px {np.mean(d[LH] <= 2):.1%}")
smax = np.zeros(SID.max() + 1)
np.maximum.at(smax, SID, np.minimum(d, 99))
print(f"  sides wholly within 1 px {np.mean(smax <= 1):.1%}, within 2 px {np.mean(smax <= 2):.1%}")
np.savez(S + f"dm_{NN}_{FIT}.npz", d=d, sid=SID, px=p)
if CMP:
    d2, p2 = measure(CMP)
    C = 250
    key = (p[:, 0] // C).astype(int) * 1000 + (p[:, 1] // C).astype(int)
    out = []
    for k in np.unique(key):
        m = key == k
        if m.sum() < 60:
            continue
        a1, a2 = np.mean(d[m] <= 1), np.mean(d2[m] <= 1)
        out.append((k, a1, a2, m.sum()))
    ch = np.array([100 * (a2 - a1) for _, a1, a2, _ in out])
    good = [(k, a1, a2) for k, a1, a2, _ in out if a1 >= 0.9]
    print(f"  area by area ({len(out)} cells), within-1-px share {FIT} -> {CMP}: rose > 2 points {(ch > 2).sum()}, within 2 {(np.abs(ch) <= 2).sum()}, fell > 2 {(ch < -2).sum()}; "
          f"cells >= 90% under {FIT}: {len(good)}, of them fell > 2 points: {sum(1 for _, a1, a2 in good if a2 < a1 - 0.02)}")
    for k, a1, a2, n in sorted(out, key=lambda r: r[2] - r[1])[:5]:
        print(f"    cell x {k // 1000 * C}-{k // 1000 * C + C}, y {k % 1000 * C}-{k % 1000 * C + C}: {a1:.0%} -> {a2:.0%}")
page = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(np.where(page < 128, 200, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
for (x, y), v in zip(p, d):
    if not np.isfinite(x):
        continue
    col = (0, 160, 0) if v <= 1 else ((0, 210, 230) if v <= 2 else ((0, 140, 255) if v <= 4 else (0, 0, 230)))
    cv2.circle(img, (int(round(x)), int(round(y))), 2, col, -1)
ban = np.full((60, img.shape[1], 3), (235, 235, 235), np.uint8)
cv2.putText(ban, f"{SERIES} {NN} {FIT}: each block-line point, its distance to the printed line running its way: green <= 1 px, yellow 1-2, orange 2-4, red > 4",
            (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 160), 2, cv2.LINE_AA)
crop = img[y0:y0 + h, x0:x0 + w]
cv2.imwrite(S + f"dm_{NN}_{FIT}.png", np.vstack([ban[:, :crop.shape[1]], crop]))
