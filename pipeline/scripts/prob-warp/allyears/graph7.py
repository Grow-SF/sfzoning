"""Steps 6-7 on a line graph (map feet): zoning boxes and the leaders pointing into them.

Input: graph.json (nodes; edges as polylines with stroke widths), labels.json (code label
polygons), the year's lots and streets.  No pixels: a vectorized scan, an SVG or a supplied graph
all fit.
1. Widths split into thin and heavy (two groups, from the edges themselves).
2. Each edge against known geometry: "on" a lot or street line (near it and parallel along most of
   its length) or "off" it.
3. Walls = heavy edges (district lines, on or off known lines).  Dangling wall ends are joined to
   the nearest wall within a few line widths.  Boxes = faces of the walls.
4. Leaders = chains of thin, off-geometry edges.  A chain that ends at a code label points from
   it: its far end lies in the box it labels (on a wall: the smaller box beside, not the label's).
5. Codes: a box takes the labels inside it, minus those whose leader points elsewhere, plus the
   labels pointing into it.

usage: graph7.py SERIES OUTDIR x0,y0 (window page origin, for drawing only)
"""
import colorsys
import os
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import geopandas as gpd
import networkx as nx
import numpy as np
import shapely
from shapely.ops import polygonize, unary_union

from sfzmaps import config
from sfzmaps.districts import read
from sfzmaps.transform import SheetTransform

series, outdir, origin = sys.argv[1:4]
SHEET = os.environ.get("SHEET", "2000-digital-zoning-p07")   # the sheet (its transform and drawn footprint)
O = Path(outdir)
X0, Y0 = map(int, origin.split(","))
# the zoning graph (W1: the ink graph with the known parcel edges deleted) when there is one
ZG = (O / "zgraph.json").exists()
G = json.load(open(O / ("zgraph.json" if ZG else "graph.json")))
labels = json.load(open(O / "labels.json"))
nodes = {int(k): np.array(v) for k, v in G["nodes"].items()}
E = G["edges"]
for e in E:
    e["line"] = shapely.LineString(e["coords"])
    e["w"] = float(np.median(e["width_ft"]))
lab_polys = [shapely.Polygon(l["poly"]) for l in labels]

# --- 1. thin / heavy: Otsu on edge widths, weighted by length -------------------------
ws = np.array([e["w"] for e in E])
ls = np.array([max(e["line"].length, 1.0) for e in E])
# block outlines are the commonest line (the biggest peak of length by width); district lines the
# biggest peak above it; the split is the low point between them (other groups, hairlines or
# borders, don't move it)
BIN = 2.0
hist = np.bincount((ws / BIN).astype(int), weights=ls)
sm_ = np.convolve(hist, [1, 2, 1], mode="same")
thin_pk = int(np.argmax(sm_))
above = [i for i in range(thin_pk + 1, len(sm_) - 1) if sm_[i] >= sm_[i - 1] and sm_[i] >= sm_[i + 1]]
heavy_pk = max(above, key=lambda i: sm_[i])
T = BIN * (thin_pk + int(np.argmin(sm_[thin_pk:heavy_pk + 1])) + 0.5)
THIN_W, HEAVY_W = BIN * (thin_pk + 0.5), BIN * (heavy_pk + 0.5)
for e in E:
    e["heavy"] = e["w"] >= T
if ZG:      # the split was made on the whole ink graph, before the parcel edges were deleted
    T, THIN_W, HEAVY_W = G["split_ft"], G["thin_ft"], G["heavy_ft"]
    for e in E:
        e["heavy"] = e["kind"] == "heavy"

# --- 2. on / off known geometry -----------------------------------------------------------
bbox = shapely.box(*unary_union([e["line"] for e in E]).bounds).buffer(300)
lots = read.load_parcels(year=read.series_year(series))
lots = lots[lots.intersects(bbox)]
streets = gpd.read_parquet(config.REF_DIR / "streets.parquet")
streets = streets[streets.intersects(bbox)]
segs = []
for g in list(lots.geometry.boundary) + list(streets.geometry):
    for part in shapely.get_parts(g):
        c = np.asarray(part.coords)[:, :2]
        segs += [shapely.LineString(c[i:i + 2]) for i in range(len(c) - 1) if np.linalg.norm(c[i + 1] - c[i]) > 0.5]
tree = shapely.STRtree(segs)
ALIGN_FT = 10.0                    # how far the scan sits from the records (step 1: 99% within 2 px)
ANGLE = 20.0


def direction(a, b):
    v = np.asarray(b) - np.asarray(a)
    return np.degrees(np.arctan2(v[1], v[0])) % 180


for e in E:
    L = e["line"].length
    n = max(2, int(L / 8))
    pts = [e["line"].interpolate(s) for s in np.linspace(0, L, n)]
    # a point is on a known line when ANY known segment within reach runs the same way (at a block
    # corner the single nearest segment is often a lot line meeting the edge at a right angle)
    reach = e["w"] / 2 + ALIGN_FT
    on = 0
    for pt in pts:
        hits = [j for j in tree.query(pt.buffer(reach)) if segs[j].distance(pt) <= reach]
        if not hits:
            continue
        if L < 2 * e["w"]:
            on += 1
            continue
        p = e["line"].project(pt)
        a_, b_ = e["line"].interpolate(max(0, p - 6)), e["line"].interpolate(min(L, p + 6))
        de = direction(a_.coords[0], b_.coords[0])
        for j in hits:
            s = np.asarray(segs[j].coords)
            da = abs(direction(s[0], s[1]) - de)
            if min(da, 180 - da) <= ANGLE:
                on += 1
                break
    e["on"] = on >= 0.8 * n

# --- 3. walls and boxes ---------------------------------------------------------------------------
# walls: every heavy edge, plus a short edge joining wall pieces at both ends (at a junction the
# crossing line spoils the width measured there)
def _leave(e, node, reach):
    """Unit direction leaving `node` along edge e, over its first `reach` feet."""
    ln, L = e["line"], e["line"].length
    if node == e["a"]:
        p0, p1 = ln.interpolate(0), ln.interpolate(min(L, reach))
    else:
        p0, p1 = ln.interpolate(L), ln.interpolate(max(0, L - reach))
    v = np.asarray(p1.coords[0]) - np.asarray(p0.coords[0])
    nv = np.linalg.norm(v)
    return v / nv if nv else v


def _through(e, node, heavy_at, reach):
    """Does a heavy edge at `node` carry on in line with e (e arrives, the heavy edge leaves opposite)?"""
    d = _leave(e, node, reach)
    return any(float(np.dot(d, -_leave(h, node, reach))) >= np.cos(np.radians(35)) for h in heavy_at.get(node, []))


changed = True
while changed:
    changed = False
    heavy_at = defaultdict(list)
    for e in E:
        if e["heavy"]:
            heavy_at[e["a"]].append(e)
            heavy_at[e["b"]].append(e)
    for e in E:
        if (not e["heavy"] and e["a"] != e["b"] and e["line"].length < 3 * HEAVY_W
                and bool(heavy_at.get(e["a"])) and bool(heavy_at.get(e["b"]))
                and (_through(e, e["a"], heavy_at, HEAVY_W) or _through(e, e["b"], heavy_at, HEAVY_W))):
            e["heavy"] = True
            changed = True
# --- a wall the graph still carries: ink is continuous but text touching the line spoiled the widths
# measured there, so a few edges came out thin.  From each loose wall end, follow the drawn edges (any
# width) onward, leaving in line with the wall, to the nearest other wall within REACH; those are wall.
REACH = 4 * HEAVY_W
AG = nx.Graph()
for j, e in enumerate(E):
    if e["a"] != e["b"]:
        AG.add_edge(e["a"], e["b"], length=e["line"].length, j=j)
path_walls = 0
for _round in range(3):
    hv = defaultdict(list)
    for e in E:
        if e["heavy"]:
            hv[e["a"]].append(e)
            hv[e["b"]].append(e)
    loose_ = [k for k, es in hv.items() if len(es) == 1 and AG.degree(k) >= 2]
    added = 0
    for k in loose_:
        w0 = hv[k][0]
        if w0["line"].length < 2 * HEAVY_W:          # a dot or a scrap, not a line
            continue
        go = -_leave(w0, k, 2 * HEAVY_W)              # onward, the way the wall runs
        piece = {k}                                   # its own piece: nodes along it up to a junction
        node, prev = k, None
        for _ in range(50):
            nb = [x for x in (w0["a"], w0["b"]) if x != node] if prev is None else \
                 [x for e_ in hv[node] for x in (e_["a"], e_["b"]) if x not in (node, prev)]
            if len(hv[node]) > 2 or not nb:
                break
            prev, node = node, nb[0]
            piece.add(node)
        dist, paths = nx.single_source_dijkstra(AG, k, cutoff=REACH, weight="length")
        best = None
        for m, dm in dist.items():
            if m in piece or not hv.get(m) or len(paths[m]) < 2:
                continue
            e1 = E[AG[k][paths[m][1]]["j"]]
            if e1["heavy"] or float(np.dot(go, _leave(e1, k, HEAVY_W))) < np.cos(np.radians(35)):
                continue
            if best is None or dm < best[0]:
                best = (dm, paths[m])
        if best is None:
            continue
        for u, v in zip(best[1][:-1], best[1][1:]):
            e_ = E[AG[u][v]["j"]]
            if not e_["heavy"]:
                e_["heavy"] = True
                added += 1
    path_walls += added
    if not added:
        break
print("edges made wall along the drawn graph:", path_walls)
walls = [e for e in E if e["heavy"]]
wdeg = defaultdict(int)
for e in walls:
    wdeg[e["a"]] += 1
    wdeg[e["b"]] += 1
wall_lines = [e["line"] for e in walls]
# the window's frame closes boxes the window cuts (they're marked: their code may be outside)
allpts = np.vstack([np.asarray(e["coords"]) for e in E])
rect = shapely.MultiPoint(allpts).minimum_rotated_rectangle
# The sheet maps only part of its paper: the frame is the sheet's content (warped.footprint.geojson,
# the ink closed and filled).  District lines stop where its drawn blocks stop, short of that edge,
# and nothing is printed between; a loose line end that runs on into blank paper to the edge joins it.
fp = shapely.from_geojson(Path(os.environ.get("FOOTPRINT") or (config.WORK_DIR / series / SHEET / "warped.footprint.geojson")).read_text())
fp = max(shapely.get_parts(fp.intersection(rect)), key=lambda g: g.area)
cover = rect if (X0 or Y0) else shapely.Polygon(fp.exterior)   # where the paper's content ends
frame = rect.exterior if not (X0 or Y0) else cover.exterior   # a window's own edge closes it; a sheet has none
# CITY: all sheets joined; the drawing's own outline (the sheets' drawn areas together) closes the boxes at
# the city's edge, and a wall that runs on over blank paper to that outline is carried to it
CITY = bool(os.environ.get("CITY"))
if CITY:
    frame = cover.exterior
etree = shapely.STRtree([e["line"] for e in E])
wall_union = unary_union(wall_lines + [frame])
wtree = shapely.STRtree([e["line"] for e in walls])
# gaps in the printed line: a loose wall end joins the nearest other wall within a few line widths
GAP = 3 * HEAVY_W
# a break in a printed line leaves two loose ends facing each other; a leader leaves one.  So a
# loose end joins only another loose end (the nearest, both ways), or the window's frame.
joins, join_edges, loose, off_edge = [], [], [], []
JK = []   # each join's kind, for the step page
ends = [k for k, dgr in wdeg.items() if dgr == 1]
WG = nx.Graph()
for e in walls:
    WG.add_edge(e["a"], e["b"])
done = set()
# --- a wall broken by text: step 4 erased letters printed over a heavy line and took its ink.  Every
# erased label's box is known; the wall ends inside a box reconnect: two ends to each other, one end to
# the nearest wall vertex in the box.
_t = SheetTransform.from_json(json.loads((config.WORK_DIR / series / SHEET / "transform.json").read_text()))
text_joins = 0
TEXT_JOINS = []   # (join line coords, erased label box coords, label text)
_end_set = set(ends)
for fit in []:   # dropped (W2): the erase keeps the lines; 48 of 49 "breaks" still had their ink
    if not fit.get("ok"):
        continue
    bx0, by0, bx1, by1 = fit["box"]
    corners = np.array([[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1]], float) + [X0, Y0]
    tb = shapely.Polygon(_t.to_world(corners)).buffer(HEAVY_W / 2)
    inside = [k for k in _end_set - done if tb.contains(shapely.Point(nodes[k]))]
    # pair ends across the box: nearest first, never two ends of one short piece
    pairs = sorted((np.linalg.norm(nodes[a_] - nodes[b_]), a_, b_) for ia, a_ in enumerate(inside) for b_ in inside[ia + 1:]
                   if not (nx.has_path(WG, a_, b_) and nx.shortest_path_length(WG, a_, b_) <= 2))
    for _, a_, b_ in pairs:
        if a_ in done or b_ in done:
            continue
        joins.append(shapely.LineString([nodes[a_], nodes[b_]]))
        join_edges.append((a_, b_))
        TEXT_JOINS.append(([nodes[a_].tolist(), nodes[b_].tolist()], [list(c) for c in tb.exterior.coords], fit["text"]))
        done |= {a_, b_}
        text_joins += 1
    for k in [k for k in inside if k not in done]:
        best = None
        for j in wtree.query(tb):
            w = walls[j]
            if k in (w["a"], w["b"]) or any(nx.has_path(WG, k, n_) and nx.shortest_path_length(WG, k, n_) <= 2 for n_ in (w["a"], w["b"])):
                continue
            for c in np.asarray(w["coords"])[:, :2]:
                if tb.contains(shapely.Point(c)):
                    dd = np.linalg.norm(c - nodes[k])
                    if best is None or dd < best[0]:
                        near_node = w["a"] if np.linalg.norm(c - nodes[w["a"]]) <= np.linalg.norm(c - nodes[w["b"]]) else w["b"]
                        best = (dd, c, near_node)
        if best is not None:
            joins.append(shapely.LineString([nodes[k], best[1]]))
            join_edges.append((k, best[2]))
            TEXT_JOINS.append(([nodes[k].tolist(), list(best[1])], [list(c) for c in tb.exterior.coords], fit["text"]))
            done.add(k)
            text_joins += 1
print("wall breaks under erased text rejoined:", text_joins)
for k in ends:
    if k in done:
        continue
    p = shapely.Point(nodes[k])
    # not the other end of its own short piece
    others = [m for m in ends if m != k and m not in done and np.linalg.norm(nodes[m] - nodes[k]) <= GAP
              and not (nx.has_path(WG, k, m) and nx.shortest_path_length(WG, k, m) <= 2)]
    m = min(others, key=lambda m_: np.linalg.norm(nodes[m_] - nodes[k]), default=None)
    if m is not None:
        back = [x for x in ends if x != m and x not in done and np.linalg.norm(nodes[x] - nodes[m]) < np.linalg.norm(nodes[k] - nodes[m])
                and not (nx.has_path(WG, x, m) and nx.shortest_path_length(WG, x, m) <= 2)]
        if not back:
            joins.append(shapely.LineString([nodes[k], nodes[m]]))
            JK.append("end to end")
            join_edges.append((k, m))
            done |= {k, m}
            continue
    if p.distance(frame) <= GAP:
        joins.append(shapely.LineString([p, shapely.ops.nearest_points(p, frame)[1]]))
        JK.append("to the outline")
        join_edges.append((k, "frame"))
        done.add(k)
        continue
    # carried straight on, does the line reach the sheet's edge over blank paper?
    e_ = next(w for w in walls if k in (w["a"], w["b"]))
    d = -_leave(e_, k, 2 * HEAVY_W)
    if np.linalg.norm(d) and e_["line"].length >= 2 * HEAVY_W:
        far = np.hypot(*(np.array(cover.bounds[2:]) - cover.bounds[:2]))
        ray = shapely.LineString([nodes[k] + d * HEAVY_W, nodes[k] + d * far])
        hit = ray.intersection(cover.exterior)
        if not hit.is_empty:
            s0 = min(ray.project(q) for q in shapely.get_parts(hit))
            seg = shapely.LineString([ray.coords[0], ray.interpolate(s0).coords[0]])
            # it may pass thin block outlines on the way out, never another district line
            if not any(walls[i]["line"].intersects(seg) for i in wtree.query(seg)):
                # where it leaves the drawing, as a position along the drawing's outline
                off_edge.append((k, float(cover.exterior.project(shapely.Point(seg.coords[1])))))
                if CITY:
                    joins.append(shapely.LineString([nodes[k], seg.coords[1]]))
                    JK.append("run off the drawing")
                    join_edges.append((k, "frame"))
                done.add(k)
                continue
    loose.append(k)
# lines that run off the drawing continue on the next sheet.  Taken in order round the drawing's
# outline, each end connects to the next one (and the last to the first).
off_edge.sort(key=lambda r: r[1])
# the join follows the drawing's own outer edge: the path through the drawn lines that keeps closest
# to the outline (an edge costs its length times how far in from the outline it lies)
_ol = cover.exterior
EG = nx.Graph()
for e in E:
    if e["a"] != e["b"]:
        dd = _ol.distance(e["line"].centroid)
        EG.add_edge(e["a"], e["b"], w=e["line"].length * (1 + (dd / HEAVY_W) ** 2), line=e["line"])
edge_paths = []
# Removed: the drawing's outer edge isn't a line in the graph (blocks are islands), so a path between
# edge ends borrows interior lines and cuts districts.  Lines off the drawing stay open until the
# neighbouring sheets are stitched on.
for (a_, _), (b_, _) in []:
    if a_ == b_ or not nx.has_path(EG, a_, b_):
        continue
    pth = nx.dijkstra_path(EG, a_, b_, weight="w")
    for u, v in zip(pth[:-1], pth[1:]):
        joins.append(EG[u][v]["line"])
        join_edges.append((u, v))
    edge_paths.append([nodes[k].tolist() for k in pth])
print("line ends off the drawing:", len(off_edge))

# m2: a wall that stops just short of the wall it meets.  A loose end carried straight on, the way its
# wall runs, for up to RUN_ON line widths: the first other wall it meets is where it joins.
RUN_ON = float(os.environ.get("RUN_ON", "0"))
run_on_joins = []
if RUN_ON > 0:
    _wadj = defaultdict(list)
    for j, w in enumerate(walls):
        _wadj[w["a"]].append(j)
        _wadj[w["b"]].append(j)

    def _own(k):
        """Wall edges of k's own piece, k along degree-2 nodes to the first junction or end."""
        seen_e, node = set(), k
        while True:
            nxt = [j for j in _wadj[node] if j not in seen_e]
            if not nxt:
                break
            seen_e.add(nxt[0])
            w = walls[nxt[0]]
            node = w["b"] if w["a"] == node else w["a"]
            if wdeg[node] != 2:
                break
        return seen_e

    still = []
    for k in loose:
        e_ = walls[_wadj[k][0]]
        d = -_leave(e_, k, 2 * HEAVY_W)
        if not np.linalg.norm(d) or e_["line"].length < 2 * HEAVY_W:
            still.append(k)
            continue
        ray = shapely.LineString([nodes[k], nodes[k] + d * RUN_ON * HEAVY_W])
        mine = _own(k)
        best = None
        for j in wtree.query(ray):
            if j in mine:
                continue
            x = ray.intersection(walls[j]["line"])
            for q in shapely.get_parts(x):
                if q.is_empty:
                    continue
                s = min(ray.project(shapely.Point(c)) for c in q.coords) if q.geom_type != "Point" else ray.project(q)
                if s > 1e-6 and (best is None or s < best[0]):
                    best = (s, ray.interpolate(s), j)
        if best is None:
            still.append(k)
            continue
        w = walls[best[2]]
        hit = np.asarray(best[1].coords[0])
        near_node = w["a"] if np.linalg.norm(hit - nodes[w["a"]]) <= np.linalg.norm(hit - nodes[w["b"]]) else w["b"]
        joins.append(shapely.LineString([nodes[k], hit]))
        JK.append("run-on")
        join_edges.append((k, near_node))
        run_on_joins.append((int(k), round(best[0] / HEAVY_W, 2)))
    print(f"loose ends carried on to the wall ahead (<= {RUN_ON} line widths):", len(run_on_joins), "of", len(loose))
    loose = still


def own_chain(k):
    """Wall edges of k's own piece: from k along degree-2 nodes up to the first junction or end."""
    seen_e, node, prev = set(), k, None
    while True:
        nxt = [(j, w) for j, w in enumerate(walls) if node in (w["a"], w["b"]) and j not in seen_e]
        if not nxt:
            break
        j, w = nxt[0]
        seen_e.add(j)
        node = w["b"] if w["a"] == node else w["a"]
        if wdeg[node] != 2:
            break
    return seen_e


loose_near = []
DIAG_ROWS = []
for k in loose:
    mine = own_chain(k)
    p = shapely.Point(nodes[k])
    cand_ = [j for j in wtree.query(p.buffer(10 * HEAVY_W)) if j not in mine]
    dmin = min((walls[j]["line"].distance(p) for j in cand_), default=None)
    loose_near.append((int(k), None if dmin is None else round(dmin / HEAVY_W, 2)))
    if os.environ.get("DIAG") and dmin is not None and dmin <= 2 * HEAVY_W:
        jn = min(cand_, key=lambda j: walls[j]["line"].distance(p))
        q = np.asarray(shapely.ops.nearest_points(p, walls[jn]["line"])[1].coords[0])
        e0 = next((w for w in walls if k in (w["a"], w["b"])), None)
        go = -_leave(e0, k, 2 * HEAVY_W) if e0 is not None else np.zeros(2)
        v = q - nodes[k]
        ang = float(np.degrees(np.arccos(np.clip(np.dot(go, v) / (np.linalg.norm(v) or 1), -1, 1)))) if np.linalg.norm(go) else -1
        L0 = sum(walls[j]["line"].length for j in mine) / HEAVY_W
        DIAG_ROWS.append({"k": int(k), "d_w": round(dmin / HEAVY_W, 2), "angle": round(ang, 1), "own_len_w": round(L0, 1),
                          "own_edges": len(mine), "xy": nodes[k].round(1).tolist(), "hit_xy": q.round(1).tolist()})
_h = np.array([d for _, d in loose_near if d is not None])
print("loose ends:", len(loose), "nearest other wall (line widths): <=1", (_h <= 1).sum(), "<=2", ((_h > 1) & (_h <= 2)).sum(),
      "<=3", ((_h > 2) & (_h <= 3)).sum(), "<=5", ((_h > 3) & (_h <= 5)).sum(), "<=10", ((_h > 5) & (_h <= 10)).sum(), "none", sum(d is None for _, d in loose_near))
if DIAG_ROWS:
    json.dump(DIAG_ROWS, open(Path(outdir) / "loose_diag.json", "w"))
    print("DIAG rows", len(DIAG_ROWS))
    if os.environ.get("DIAG_ONLY"):
        sys.exit(0)
wall_lines.append(frame)

# --- bridges: edges on no loop (after the joins).  Outlines and walls close loops; a leader hangs
# off the graph ----------------------------------------------------------------------------------------
FG = nx.MultiGraph()
for j, e in enumerate(E):
    FG.add_edge(e["a"], e["b"], key=j)
for u, v in join_edges:
    FG.add_edge(u, v)
for e in walls:                                   # walls cut by the window close through the frame
    for n_ in (e["a"], e["b"]):
        if shapely.Point(nodes[n_]).distance(frame) < GAP:
            FG.add_edge(n_, "frame")
simple = nx.Graph()
for u, v in FG.edges():
    if u != v:
        simple.add_edge(u, v)
br = {frozenset(b) for b in nx.bridges(simple)}
for j, e in enumerate(E):
    e["bridge"] = (e["a"] != e["b"] and frozenset((e["a"], e["b"])) in br
                   and FG.number_of_edges(e["a"], e["b"]) == 1)
faces = [f for f in polygonize(unary_union(wall_lines + joins)) if f.area > (2 * HEAVY_W) ** 2]
ftree = shapely.STRtree(faces)
cut = {j for j, f in enumerate(faces) if f.exterior.distance(frame) < 1.0 and f.exterior.intersection(frame.buffer(1.0)).length > HEAVY_W}


def face_of(pt):
    hit = [i for i in ftree.query(pt, predicate="within")]
    return min(hit, key=lambda i: faces[i].area) if hit else None


host = {i: face_of(p.centroid) for i, p in enumerate(lab_polys)}

# --- 4. leaders ---------------------------------------------------------------------------------
cand = [e for e in E if e["bridge"] and not e["on"]]
LG = nx.Graph()
for e in cand:
    LG.add_edge(e["a"], e["b"], length=e["line"].length, line=e["line"])
leaders = []
LEAD_DIAG = []
for comp in nx.connected_components(LG):
    sub = LG.subgraph(comp)
    chain = unary_union([d["line"] for _, _, d in sub.edges(data=True)])
    best = None
    for i, lp in enumerate(lab_polys):
        reach = 0.6 * labels[i]["height_ft"]
        if chain.distance(lp) > reach:
            continue
        # the chain's node nearest the label is where it leaves it
        start = min(comp, key=lambda k: lp.distance(shapely.Point(nodes[k])))
        dstart = lp.distance(shapely.Point(nodes[start]))
        if dstart > reach:
            continue
        if best is None or dstart < best[0]:
            best = (dstart, i, start)
    if best is None:
        continue
    _, i, start = best
    dist = nx.single_source_dijkstra_path_length(sub, start, weight="length")
    end = max(dist, key=dist.get)
    if dist[end] < THIN_W:
        continue
    pt = shapely.Point(nodes[end])
    # a leader ends in a dot: a short heavy piece at (or right by) its tip
    _dot = any(E[jj]["heavy"] and E[jj]["line"].length < 2 * HEAVY_W
               for jj in etree.query(pt.buffer(HEAVY_W / 2)) if E[jj]["line"].distance(pt) <= HEAVY_W / 2)
    LEAD_DIAG.append((labels[i]["text"], round(dist[end] / labels[i]["height_ft"], 2), _dot))
    if not _dot or dist[end] < 0.5 * labels[i]["height_ft"]:      # no dot, or too short: a scrap
        continue
    # the box at the far end: the one it lies in, or (on a wall) the smaller box beside it
    # a label sits in the bigger area and points into a smaller box
    near = [j for j in ftree.query(pt.buffer(HEAVY_W)) if j != host[i]
            and (host[i] is None or faces[j].area < faces[host[i]].area)]
    tgt = min(near, key=lambda j: faces[j].area) if near else None
    path = nx.dijkstra_path(sub, start, end, weight="length")
    leaders.append({"label": i, "code": labels[i]["text"], "face": tgt, "length_ft": round(dist[end], 1),
                    "path": [nodes[k].tolist() for k in path]})

# --- 5. codes per box ----------------------------------------------------------------------------
codes = defaultdict(list)
moved = {l["label"] for l in leaders if l["face"] is not None}
# A label printed in a street (its text mostly off every lot) names the boxes it sits between: a
# corridor's label sits in the street, its two sides' boxes either side.  It goes to the boxes
# within half its text height that hold no label of their own, smaller than the one it stands in.
lot_tree = shapely.STRtree(list(lots.geometry))
own = defaultdict(int)
for i, f in host.items():
    if f is not None and i not in moved:
        own[f] += 1
street = {}
for i, lp in enumerate(lab_polys):
    f = host[i]
    if i in moved or not lp.area:
        continue
    on_lots = unary_union([lots.geometry.iloc[k] for k in lot_tree.query(lp)]).intersection(lp).area / lp.area if len(lot_tree.query(lp)) else 0.0
    # beside it: empty boxes smaller than its own; in a street any, on lots only those too small to
    # hold the label (a district too small for its text has it printed beside it)
    beside = [j for j in ftree.query(lp.buffer(0.5 * labels[i]["height_ft"]), predicate="intersects")
              if j != f and not own[j] and (f is None or faces[j].area < faces[f].area)
              and (on_lots < 0.5 or faces[j].area < 3 * lp.area)]
    if beside:
        street[i] = [int(j) for j in beside]
for i, f in host.items():
    if i in street:
        for j in street[i]:
            codes[j].append(labels[i]["text"])
        if f is None or own[f] > 1:     # the only label in its own box names that box too
            continue
    if f is not None and i not in moved:
        codes[f].append(labels[i]["text"])
for l in leaders:
    if l["face"] is not None:
        codes[l["face"]].append(l["code"])

# --- drawing (back to window pixels) ----------------------------------------------------------------
t = SheetTransform.from_json(json.loads((config.WORK_DIR / series / SHEET / "transform.json").read_text()))


# drawing only: the transform sampled once on a grid over the graph's extent, then interpolated
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

_b = unary_union([e["line"] for e in E]).bounds
_gx = np.linspace(_b[0] - 200, _b[2] + 200, 120)
_gy = np.linspace(_b[1] - 200, _b[3] + 200, 160)
_GX, _GY = np.meshgrid(_gx, _gy, indexing="ij")
_pp = t.to_pixel(np.c_[_GX.ravel(), _GY.ravel()]).reshape(len(_gx), len(_gy), 2)
_ix = RegularGridInterpolator((_gx, _gy), _pp[..., 0], bounds_error=False, fill_value=None)
_iy = RegularGridInterpolator((_gx, _gy), _pp[..., 1], bounds_error=False, fill_value=None)


def px(coords):
    c = np.asarray(coords, float)[:, :2]
    return np.round(np.c_[_ix(c), _iy(c)] - [X0, Y0]).astype(np.int32)


clean = cv2.imread(str(O / "clean.png"), cv2.IMREAD_GRAYSCALE)
H, W = clean.shape
base = cv2.addWeighted(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR), 0.25, np.full((H, W, 3), 255, np.uint8), 0.75, 0)
img = base.copy()
for e in E:
    col = (60, 60, 60) if e["heavy"] else ((230, 190, 140) if e["on"] else (0, 150, 255))
    cv2.polylines(img, [px(e["coords"])], False, col, 3 if e["heavy"] else 1)
for j in joins:
    cv2.polylines(img, [px(j.coords)], False, (200, 0, 200), 3)
for k in loose:
    P = px([nodes[k]])[0]
    cv2.drawMarker(img, tuple(int(v) for v in P), (200, 0, 200), cv2.MARKER_TILTED_CROSS, 14, 2)
for l in leaders:
    col = (0, 0, 230) if l["face"] is not None else (230, 120, 0)
    P = px(l["path"])
    cv2.polylines(img, [P], False, col, 3)
    cv2.circle(img, tuple(P[-1]), 6, col, 2)
cv2.imwrite(str(O / "g7_detect.png"), img)

col = np.full((H, W, 3), 255, np.uint8)
yy, xx = np.mgrid[0:H, 0:W]
hatch = ((xx + yy) % 10) < 3
pointed = {l["face"] for l in leaders if l["face"] is not None}
for j, f in enumerate(faces):
    ext = px(f.exterior.coords)
    bx0, by0 = np.maximum(ext.min(0), 0)
    bx1, by1 = np.minimum(ext.max(0) + 1, [W, H])
    if bx0 >= bx1 or by0 >= by1:
        continue
    sm = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
    cv2.fillPoly(sm, [ext - [bx0, by0]], 1)
    for ring in f.interiors:
        cv2.fillPoly(sm, [px(ring.coords) - [bx0, by0]], 0)
    m = sm.astype(bool)                  # all drawing on the face's own bounding box
    cv_ = col[by0:by1, bx0:bx1]
    hv = hatch[by0:by1, bx0:bx1]
    cs = codes.get(j, [])
    if len(set(cs)) == 1:
        h = (hash(cs[0]) % 997) / 997
        r_, g_, b_ = colorsys.hls_to_rgb(h, 0.82, 0.55)
        cv_[m] = (int(b_ * 255), int(g_ * 255), int(r_ * 255))
        if j in pointed:
            cv_[m & hv] = (0, 120, 0)
    elif len(set(cs)) > 1:
        cv_[m] = (235, 235, 235)
        cv_[m & hv] = (0, 150, 255)
    else:
        cv_[m] = (245, 245, 245)
for e in walls:
    cv2.polylines(col, [px(e["coords"])], False, (40, 40, 40), 2)
for j in joins:
    cv2.polylines(col, [px(j.coords)], False, (40, 40, 40), 2)
for i, l in enumerate(labels):
    P = px(l["poly"])
    cv2.putText(col, l["text"], (int(P[:, 0].min()), int(P[:, 1].mean()) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
for l in leaders:
    if l["face"] is not None:
        cv2.polylines(col, [px(l["path"])], False, (0, 0, 230), 2)
cv2.imwrite(str(O / "g7_after.png"), col)

stats = {"thin_heavy_split_ft": float(T), "thin_ft": round(THIN_W, 1), "heavy_ft": round(HEAVY_W, 1),
         "edges": len(E), "heavy": len(walls), "thin_on_known": sum(1 for e in E if not e["heavy"] and e["on"]),
         "thin_off_known": len(cand), "wall_gaps_joined": len(joins), "run_on_joins": len(run_on_joins), "text_breaks_rejoined": text_joins, "wall_edges_followed": path_walls, "boxes": len(faces),
         "leaders_to_a_label": len(leaders), "leaders_into_a_box": len(moved),
         "boxes_several_codes": sum(len(set(v)) > 1 for v in codes.values())}
json.dump({"stats": stats, "leaders": leaders, "codes": {str(k): v for k, v in codes.items()},
           "faces": [f.wkt for f in faces], "joins": [[np.round(np.asarray(j.coords), 2).tolist(), k_] for j, k_ in zip(joins, JK)], "cut": sorted(cut), "cover": cover.wkt, "off_edge": [(int(k), pos, nodes[k].tolist()) for k, pos in off_edge], "edge_paths": edge_paths, "text_joins": TEXT_JOINS, "edge_class": ["heavy" if e["heavy"] else ("thin_on" if e["on"] else "thin_off") for e in E], "edge_class_raw": ["heavy" if e["w"] >= T else ("thin_on" if e["on"] else "thin_off") for e in E], "loose_near": [(k, d, nodes[k].tolist()) for k, d in loose_near],
           "host": {str(i): f for i, f in host.items()}, "moved": sorted(moved), "street": {str(i): v for i, v in street.items()}},
          open(O / "graph7.json", "w"), default=float)
print(stats)
print("leader candidates", len(LEAD_DIAG), "with a dot", sum(d for _, _, d in LEAD_DIAG), "long (>=0.5 h)", sum(r >= 0.5 for _, r, _ in LEAD_DIAG), "both", sum(d and r >= 0.5 for _, r, d in LEAD_DIAG))
print("both:", sorted((t_, r) for t_, r, d in LEAD_DIAG if d and r >= 0.5))
for l in leaders:
    c = faces[l["face"]].centroid if l["face"] is not None else None
    print(" ", l["code"], "len", l["length_ft"], "-> box", l["face"], c and px([c.coords[0]])[0].tolist(), c and round(faces[l["face"]].area))
print("several:", {k: v for k, v in codes.items() if len(set(v)) > 1})
