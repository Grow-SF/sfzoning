"""The zoning graph: the ink graph with the known parcel edges deleted.

Heavy edges (district lines) are kept whole.  Every thin edge is cut along its length: the stretches
lying on a known 2000 parcel edge or street centreline (within the edge's half-width + ALIGN_FT, the
known line running the same way within ANGLE) are deleted; the stretches left over are kept as their
own edges.  Leftover bits shorter than the thin line's own width are dropped (corner rounding).

usage: zonegraph.py SERIES OUTDIR   -> OUTDIR/zgraph.json (nodes as before; edges: kept pieces)"""
import json
import sys

import geopandas as gpd
import numpy as np
import shapely
from shapely.ops import unary_union

from sfzmaps import config
from sfzmaps.districts import read

series, outdir = sys.argv[1:3]
G = json.load(open(f"{outdir}/graph.json"))
E = G["edges"]
ALIGN_FT, ANGLE, STEP = 10.0, 20.0, 4.0
import os as _os0  # noqa: E402
if _os0.environ.get("TEXTBOXES"):
    # the printed text left on the sheet (street names, block numbers, other: everything but zoning codes,
    # which are erased) is not line: an edge lying mostly (80%+ of its length) inside one of their boxes
    # (+2 px) is left out before anything else, so it neither becomes an edge nor sways the thin/heavy split
    from sfzmaps.transform import SheetTransform
    _t = SheetTransform.from_json(json.loads((config.WORK_DIR / series / _os0.environ["SHEET"] / "transform.json").read_text()))
    _tb = [k["box"] for k in json.load(open(_os0.environ["TEXTBOXES"])) if k["kind"] != "zoning code"]
    _U = unary_union([shapely.Polygon(_t.to_world(np.array([[a - 2, b - 2], [c + 2, b - 2], [c + 2, e + 2], [a - 2, e + 2]], float)))
                      for a, b, c, e in _tb])
    _ln = [shapely.LineString(np.asarray(e["coords"])[:, :2]) for e in E]
    _keep = [ln.length < 1e-9 or ln.intersection(_U).length < 0.8 * ln.length for ln in _ln]
    print(f"text boxes: {len(_tb)}; edges left out as text: {len(E) - sum(_keep)} of {len(E)}", flush=True)
    E = [e for e, k in zip(E, _keep) if k]
    G["edges"] = E
ws = np.array([float(np.median(e["width_ft"])) for e in E])
ls = np.array([max(shapely.LineString(e["coords"]).length, 1.0) for e in E])
# thin / heavy: the same peak split as graph7
BIN = 2.0
hist = np.bincount((ws / BIN).astype(int), weights=ls)
sm_ = np.convolve(hist, [1, 2, 1], mode="same")
thin_pk = int(np.argmax(sm_))
above = [i for i in range(thin_pk + 1, len(sm_) - 1) if sm_[i] >= sm_[i - 1] and sm_[i] >= sm_[i + 1]]
heavy_pk = max(above, key=lambda i: sm_[i])
T = BIN * (thin_pk + int(np.argmin(sm_[thin_pk:heavy_pk + 1])) + 0.5)
THIN_W = BIN * (thin_pk + 0.5)
import os as _os  # noqa: E402
if _os.environ.get("SPLIT_FT"):            # a fixed thin/heavy split (and thin width), for like-for-like checks
    T = float(_os.environ["SPLIT_FT"])
    THIN_W = float(_os.environ.get("THIN_FT", THIN_W))

lines = [shapely.LineString(np.asarray(e["coords"])[:, :2]) for e in E]
bbox = shapely.box(*unary_union(lines).bounds).buffer(300)
lots = read.load_parcels(year=read.series_year(series))
lots = lots[lots.intersects(bbox)]
streets = gpd.read_parquet(config.REF_DIR / "streets.parquet")
streets = streets[streets.intersects(bbox)]
segs = []
import os
if os.environ.get("BLOCKS"):
    # the raw scan (step 4a): only block outlines are printed; lot lines inside a block and street
    # centrelines aren't, and labels sit on them
    known = list(lots.dissolve(by="block_num").geometry.boundary)
else:
    known = list(lots.geometry.boundary) + list(streets.geometry)
for g in known:
    for part in shapely.get_parts(g):
        c = np.asarray(part.coords)[:, :2]
        segs += [c[i:i + 2] for i in range(len(c) - 1) if np.linalg.norm(c[i + 1] - c[i]) > 0.5]
seg_geom = shapely.linestrings(np.array(segs))
seg_dir = np.array([np.degrees(np.arctan2(*(s[1] - s[0])[::-1])) % 180 for s in segs])
tree = shapely.STRtree(seg_geom)

out, n_cut, kept_ft, deleted_ft, dels = [], 0, 0.0, 0.0, []
nodes_out = dict(G["nodes"])
next_id = max(int(k) for k in G["nodes"]) + 1
for e, ln, w in zip(E, lines, ws):
    if w >= T:
        out.append({**e, "kind": "heavy"})
        kept_ft += ln.length
        continue
    L = ln.length
    n = max(2, int(np.ceil(L / STEP)) + 1)
    s = np.linspace(0, L, n)
    pts = shapely.points(np.array([ln.interpolate(v).coords[0] for v in s]))
    reach = w / 2 + ALIGN_FT
    pi, si = tree.query(pts, predicate="dwithin", distance=reach)
    # local direction of the edge at each sample (over +-6 ft)
    a = np.array([ln.interpolate(max(0, v - 6)).coords[0] for v in s])
    b = np.array([ln.interpolate(min(L, v + 6)).coords[0] for v in s])
    d = np.degrees(np.arctan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])) % 180
    on = np.zeros(n, bool)
    da = np.abs(seg_dir[si] - d[pi])
    ok = np.minimum(da, 180 - da) <= ANGLE
    on[pi[ok]] = True
    if on.all():
        deleted_ft += L
        dels.append({"coords": [list(c)[:2] for c in ln.coords], "w_ft": float(np.median(e["width_ft"]))})
        continue
    # keep each run of off samples as its own piece
    runs, i = [], 0
    while i < n:
        if on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not on[j + 1]:
            j += 1
        runs.append((max(0, s[i] - STEP / 2), min(L, s[j] + STEP / 2)))
        i = j + 1
    if len(runs) > 1 or runs[0] != (0, L):
        n_cut += 1
    # the deleted stretches: between the kept runs, and kept runs too short to keep
    keep = [(r0, r1) for r0, r1 in runs if r1 - r0 >= THIN_W]
    edges_ = [0.0] + [v for r in keep for v in r] + [L]
    for g0, g1 in zip(edges_[::2], edges_[1::2]):
        if g1 - g0 > 0.5:
            dels.append({"coords": [list(c)[:2] for c in shapely.ops.substring(ln, g0, g1).coords], "w_ft": float(np.median(e["width_ft"]))})
    for r0, r1 in runs:
        if r1 - r0 < THIN_W:
            deleted_ft += r1 - r0
            continue
        piece = shapely.ops.substring(ln, r0, r1)
        pc = [list(c) for c in piece.coords]
        if r0 == 0:
            na = e["a"]
        else:
            na = next_id; nodes_out[str(na)] = pc[0][:2]; next_id += 1
        if r1 == L:
            nb = e["b"]
        else:
            nb = next_id; nodes_out[str(nb)] = pc[-1][:2]; next_id += 1
        out.append({"a": na, "b": nb,
                    "coords": [list(c) for c in piece.coords], "width_ft": e["width_ft"], "kind": "thin"})
        kept_ft += piece.length
    deleted_ft += L - sum(r1 - r0 for r0, r1 in runs)
json.dump({"nodes": nodes_out, "edges": out, "split_ft": T, "thin_ft": THIN_W, "heavy_ft": BIN * (heavy_pk + 0.5)}, open(f"{outdir}/zgraph.json", "w"))
st = {"edges_in": len(E), "edges_out": len(out), "heavy": sum(o["kind"] == "heavy" for o in out),
      "thin_pieces_kept": sum(o["kind"] == "thin" for o in out), "thin_edges_cut": n_cut,
      "kept_miles": round(kept_ft / 5280, 1), "deleted_miles": round(deleted_ft / 5280, 1)}
json.dump(st, open(f"{outdir}/zgraph_stats.json", "w"))
json.dump(dels, open(f"{outdir}/zgraph_deleted.json", "w"))
print(st)
