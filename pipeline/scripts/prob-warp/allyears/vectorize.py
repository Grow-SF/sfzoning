"""Raster -> graph: the only step that looks at pixels.

The text-free ink (clean.png) is thinned to centre lines; junctions and line ends become nodes,
the runs between them edges (polylines in map feet, with the stroke width along them, in feet).
The code labels' boxes become map polygons.  An SVG or a vector graph would replace this step.

usage: vectorize.py SERIES SHEET x0,y0 OUTDIR      (x0,y0: the window's page origin)
writes OUTDIR/graph.json, OUTDIR/labels.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import shapely
from skimage.morphology import skeletonize

from sfzmaps import config
from sfzmaps.transform import SheetTransform

series, sheet, origin, outdir = sys.argv[1:5]
X0, Y0 = map(int, origin.split(","))
O = Path(outdir)
t = SheetTransform.from_json(json.loads((config.WORK_DIR / series / sheet / "transform.json").read_text()))
clean = cv2.imread(str(O / "clean.png"), cv2.IMREAD_GRAYSCALE)
H, W = clean.shape
ink = (clean < 128).astype(np.uint8)
half = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
sk = skeletonize(ink.astype(bool))
w_ = t.to_world(np.array([[X0, Y0], [X0 + 100, Y0]], float))
FT = float(np.linalg.norm(w_[1] - w_[0]) / 100)          # feet per pixel


def world(pts):                                            # (x, y) window px -> map feet
    return t.to_world(np.asarray(pts, float) + [X0, Y0])


# --- skeleton -> graph -------------------------------------------------------------
NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
pad = np.pad(sk, 1)
deg = sum(np.roll(np.roll(pad, dy, 0), dx, 1) for dy, dx in NB)[1:-1, 1:-1] * sk
isnode = sk & (deg != 2)
nn, nlab = cv2.connectedComponents(isnode.astype(np.uint8), connectivity=8)


def nbrs(y, x):
    for dy, dx in NB:
        yy, xx = y + dy, x + dx
        if 0 <= yy < H and 0 <= xx < W and sk[yy, xx]:
            yield yy, xx


ys_, xs_ = np.nonzero(nlab)                     # every junction's centre in one pass
kk = nlab[ys_, xs_]
cnt = np.bincount(kk, minlength=nn)
mx, my = np.bincount(kk, xs_, nn) / np.maximum(cnt, 1), np.bincount(kk, ys_, nn) / np.maximum(cnt, 1)
nodes = {k: (float(mx[k]), float(my[k])) for k in range(1, nn)}
visited = np.zeros_like(sk)
edges = []
for y, x in zip(*np.nonzero(isnode)):
    a = nlab[y, x]
    for q in nbrs(y, x):
        if isnode[q]:
            b = nlab[q]
            if b != a and a < b:
                edges.append((a, b, [(x, y), (q[1], q[0])]))
            continue
        if visited[q]:
            continue
        path, prev, cur = [(y, x), q], (y, x), q
        visited[q] = True
        end = None
        while True:
            nxt = [p for p in nbrs(*cur) if p != prev and p not in path[-3:]]
            hit = [p for p in nxt if isnode[p] and not (nlab[p] == a and len(path) < 3)]
            if hit:
                path.append(hit[0])
                end = nlab[hit[0]]
                break
            nxt = [p for p in nxt if not visited[p] and not isnode[p]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            visited[cur] = True
            path.append(cur)
        if end is None:                                  # ran into an already-traced run: stop there
            nn += 1
            nodes[nn] = (float(cur[1]), float(cur[0]))
            end = nn
        edges.append((a, end, [(p[1], p[0]) for p in path]))
# closed loops with no junction
for y, x in zip(*np.nonzero(sk & ~visited & ~isnode)):
    if visited[y, x]:
        continue
    nn += 1
    nodes[nn] = (float(x), float(y))
    path, prev, cur = [(y, x)], None, (y, x)
    visited[y, x] = True
    while True:
        nxt = [p for p in nbrs(*cur) if p != prev and not visited[p]]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        visited[cur] = True
        path.append(cur)
    path.append((y, x))
    edges.append((nn, nn, [(p[1], p[0]) for p in path]))

# spurs: an edge with a free end shorter than the stroke is thinning noise, not a line
degree = {}
for a, b, _ in edges:
    degree[a] = degree.get(a, 0) + 1
    degree[b] = degree.get(b, 0) + 1
out, simp = [], []
for a, b, pts in edges:
    P = np.array(pts, float)
    wid = 2 * half[P[:, 1].astype(int), P[:, 0].astype(int)]
    L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    if a != b and (degree[a] == 1 or degree[b] == 1) and L < np.median(wid):
        continue
    # ends at the node's centre, so edges meeting at a junction share an exact point
    P = np.vstack([nodes[a], P, nodes[b]])
    simp.append(np.asarray(shapely.simplify(shapely.LineString(P), 0.75).coords))
    out.append({"a": int(a), "b": int(b), "width_ft": np.round(wid * FT, 1).tolist(), "length_ft": round(L * FT, 1)})
# to map feet: every point in one call
allw = world(np.vstack(simp)) if simp else np.zeros((0, 2))
i = 0
for e, s_ in zip(out, simp):
    e["coords"] = np.round(allw[i:i + len(s_)], 2).tolist()
    i += len(s_)
used = sorted({e["a"] for e in out} | {e["b"] for e in out})
nw = world([nodes[k] for k in used])
gnodes = {int(k): np.round(v, 2).tolist() for k, v in zip(used, nw)}
json.dump({"ft_per_px": FT, "nodes": gnodes, "edges": out}, open(O / "graph.json", "w"))

# --- labels -> map polygons ----------------------------------------------------------
M = 80
labels = []
for f in json.load(open(O / "fits3.json")):
    if f["kind"] != "zoning code":
        continue
    a, b, c, e = [v - M for v in f["box"]]
    poly = world([(a, b), (c, b), (c, e), (a, e)])
    labels.append({"text": f["text"], "poly": np.round(poly, 2).tolist(), "height_ft": round(min(c - a, e - b) * FT, 1)})
json.dump(labels, open(O / "labels.json", "w"))
print(f"{len(gnodes)} nodes, {len(out)} edges, {len(labels)} labels; {FT:.2f} ft/px")
