"""edges.py SERIES SHEET: per modern block edge, where is the drawn stroke and our extracted edge?

For every modern block edge >= 120 ft inside the matched area: sample every 8 ft (skipping 20 ft
at each end), map to pixels with the sheet transform, take the outward normal, and along it
find (a) the drawn ink stroke nearest the modern line (centre and outer/inner edge), and
(b) our extracted block polygon boundary. Offsets are signed, + = outward (into the street).
"""
import sys, json, pickle
import numpy as np, cv2
from pathlib import Path
from shapely.geometry import shape, LineString, Point
from shapely.strtree import STRtree
from _common import blocks_corners, out_dir
from sfzmaps import config, linework
from sfzmaps.transform import SheetTransform
D = out_dir()
series, sheet = sys.argv[1], sys.argv[2]
d = config.WORK_DIR / series / sheet
img = cv2.imread(str(d / "page.png"), cv2.IMREAD_UNCHANGED)
ink = linework.binarize(img)
sw = linework.stroke_width(ink)
t = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
blocks, corners = blocks_corners()
rf = json.loads((d / "refine.json").read_text())
inl = np.array([[m["wx"], m["wy"]] for m in rf["matches"] if m["label"] == "inlier"])
from scipy.spatial import cKDTree
itree = cKDTree(inl)
bj = json.loads((d / "blocks_px.geojson").read_text())
ext = [shape(f["geometry"]) for f in bj["features"]]
etree = STRtree([e.boundary for e in ext])
ftpx = None
REACH = 10  # px either side
rows = []
for bn, g in zip(blocks.block_num, blocks.geometry):
    for poly in getattr(g, "geoms", [g]):
        c = np.asarray(poly.simplify(2.0).exterior.coords)
        if poly.exterior.is_ccw is False:  # make outward normal = right-hand for ccw
            pass
        ccw = LineString(c).is_ring and Polygon_ccw if False else None
        for a, b in zip(c[:-1], c[1:]):
            L = np.hypot(*(b - a))
            if L < 120: continue
            mid = (a + b) / 2
            if itree.query(mid)[0] > 400: continue  # outside matched area
            dvec = (b - a) / L
            nrm = np.array([dvec[1], -dvec[0]])
            if poly.contains(Point(*(mid + 3 * nrm))): nrm = -nrm   # outward
            ss = np.arange(20, L - 20, 8)
            W = a + ss[:, None] * dvec
            Pp = t.to_pixel(W); Pn = t.to_pixel(W + nrm)  # 1 ft outward
            npx = Pn - Pp; fpp = np.linalg.norm(npx, axis=1)  # px per ft
            npx /= fpp[:, None]
            offs, outs, ins, exo = [], [], [], []
            for p, nv, f in zip(Pp, npx, fpp):
                if not np.isfinite(p).all(): continue
                st = np.arange(-REACH, REACH + 0.01, 0.25)
                q = p[None] + st[:, None] * nv[None]
                xi, yi = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
                if xi.min() < 0 or yi.min() < 0 or xi.max() >= ink.shape[1] or yi.max() >= ink.shape[0]: continue
                v = ink[yi, xi]
                if not v.any(): continue
                # runs of ink; pick the one nearest 0
                e = np.diff(np.r_[0, v.astype(int), 0]); s0 = np.flatnonzero(e == 1); s1 = np.flatnonzero(e == -1) - 1
                cen = (st[s0] + st[s1]) / 2
                k = np.argmin(np.abs(cen))
                if s1[k] - s0[k] > 4 * 4 * 2: continue  # > 8 px: solid fill, not a stroke
                offs.append(cen[k] / f); outs.append(st[s1[k]] / f); ins.append(st[s0[k]] / f)
                # extracted boundary crossing along the normal
                seg = LineString([q[0], q[-1]])
                best = None
                for j in etree.query(seg):
                    x = etree.geometries[j].intersection(seg)
                    for pt in getattr(x, "geoms", [x]):
                        if pt.is_empty or pt.geom_type != "Point": continue
                        s = np.dot(np.array(pt.coords[0]) - p, nv)
                        if best is None or abs(s) < abs(best): best = s
                exo.append(best / f if best is not None else np.nan)
            if len(offs) >= 5:
                rows.append((bn, mid[0], mid[1], len(offs), np.median(offs), np.std(offs),
                             np.median(outs), np.median(ins), np.nanmedian(exo) if np.isfinite(exo).any() else np.nan,
                             float(np.mean(fpp))))
R = np.array([r[1:] for r in rows], float)
np.save(D / f"edges_{series}_{sheet}.npy", R)
mx, my, n, cen, sd, out, inn, ex, fpp = R.T
print(f"{series}/{sheet}: stroke width {sw:.1f} px = {sw / np.median(fpp):.1f} ft; {len(R)} modern edges measured")
print(f" stroke centre offset vs modern line (+ = outward, ft): median {np.median(cen):+.2f}  IQR {np.quantile(cen,.25):+.2f}..{np.quantile(cen,.75):+.2f}  rms {np.sqrt(np.mean(cen**2)):.2f}")
print(f" stroke outer edge vs modern: median {np.median(out):+.2f};  inner edge: median {np.median(inn):+.2f}")
print(f" within-edge std of stroke centre (straightness/noise): median {np.median(sd):.2f} ft")
ok = np.isfinite(ex)
print(f" our extracted edge vs modern: median {np.median(ex[ok]):+.2f}  IQR {np.quantile(ex[ok],.25):+.2f}..{np.quantile(ex[ok],.75):+.2f} rms {np.sqrt(np.mean(ex[ok]**2)):.2f}")
print(f" our extracted edge vs stroke centre: median {np.median(ex[ok]-cen[ok]):+.2f}  IQR {np.quantile(ex[ok]-cen[ok],.25):+.2f}..{np.quantile(ex[ok]-cen[ok],.75):+.2f}")
for q in (0.5, 0.8, 0.9, 0.95):
    print(f"  |stroke centre offset| q{int(q*100)} {np.quantile(np.abs(cen), q):.1f} ft")
