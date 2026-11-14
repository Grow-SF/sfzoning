"""Extra evidence, for sheets whose own year has few blocks where the map prints a street grid: the block-side points
(block_side_points.npz, weight 1), plus
  later blocks   lots recorded after the map's year, merged by block: their outlines (land held whole in the map's year
                 and split up later, such as the Hunters Point shipyard, often had its streets printed long before);
  street edges   every street centre line (DataSF, current or retired) offset by half its width each way: the map
                 prints a street as its two edges.  Each street's width (by cnn) is the gap between the block outlines
                 either side, measured along it (median); a street with no block beside it takes its name's median,
                 else the sheet's median.
Each added point counts by how little the evidence already there explains it: w = 1 - exp(-d^2 / (2 s^2)), d its
distance to the nearest block-side point (for street edges: to block sides or later-block points), s half the
sheet's median street width.  So an edge with a block outline on it counts nothing, an edge with nothing near counts
in full.  Sides are cut into straight stretches and sampled every 2 ft with tangents, as block_side_points.py does.
-> extra_evidence_points.npz (pts, sid, ta, tb, wt, kind: 0 block side, 1 later block, 2 street edge)."""
import json

import cv2
import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import cKDTree

from warp_common import D, S, SHEET, lots_and_area, year
from sfzmaps import config
from sfzmaps.districts import lots as lotsmod
from sfzmaps.districts import read
from sfzmaps.transform import SheetTransform

T0 = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
SD = np.load(S + "block_side_points.npz")
Y = year()
lots, area = lots_and_area(cuts=1)
Hp, Wp = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE).shape
pg = shapely.Polygon(T0.to_world(np.array([[0, 0], [Wp, 0], [Wp, Hp], [0, Hp]], float)))
blocks = lots[lots.intersects(pg)].dissolve(by="block_num").geometry
bl_lines = shapely.union_all([g.boundary for g in blocks.values])


def sample(lines, sid0):
    """Straight stretches every 2 ft with tangents (as block_side_points.py); returns pts, sid, ta, tb."""
    pts, sid_, ta, tb = [], [], [], []
    sid = sid0
    for ln0 in lines:
        for ln in shapely.get_parts(ln0.intersection(area)):
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
                    pts.append(sl.interpolate(v).coords[0][:2])
                    sid_.append(sid)
                    ta.append(sl.interpolate(max(0, v - 3)).coords[0][:2])
                    tb.append(sl.interpolate(min(sl.length, v + 3)).coords[0][:2])
                sid += 1
    return np.array(pts).reshape(-1, 2), np.array(sid_, int), np.array(ta).reshape(-1, 2), np.array(tb).reshape(-1, 2), sid


# later blocks: lots recorded after the map's year, by block
ev = read.load_parcels(every=True)
ev = ev[ev.intersects(pg) & ev.geometry.notna()]
lv = lotsmod.lives().set_index("blklot").reindex(ev["blklot"].astype(str))
later = ev[lv["created"].values > Y + 0.5]
lb = later.dissolve(by="block_num").geometry if len(later) else gpd.GeoSeries([])
sid0 = int(SD["sid"].max()) + 1 if len(SD["sid"]) else 0
P1, I1, A1, B1, sid0 = sample([g.boundary for g in lb.values], sid0)

# street widths from the blocks either side
st = gpd.read_parquet(config.REF_DIR / "streets.parquet")
st = st[st.intersects(pg)]
bl_tree = shapely.STRtree(list(shapely.get_parts(bl_lines)))
bl_parts = list(shapely.get_parts(bl_lines))
widths = {}
for cnn, g in zip(st["cnn"].values, st.geometry.values):
    if g is None or g.length < 20:
        continue
    ws = []
    for t in np.linspace(0.2, 0.8, 4):
        p0, p1 = g.interpolate(t, normalized=True), g.interpolate(min(1.0, t + 0.02), normalized=True)
        dv = np.array(p1.coords[0][:2]) - np.array(p0.coords[0][:2])
        if np.linalg.norm(dv) < 1e-6:
            continue
        nrm = np.array([-dv[1], dv[0]]) / np.linalg.norm(dv)
        side = []
        for sgn in (1, -1):
            ray = shapely.LineString([p0.coords[0][:2], np.array(p0.coords[0][:2]) + sgn * 80 * nrm])
            hits = [ray.intersection(bl_parts[k]) for k in bl_tree.query(ray, predicate="intersects")]
            ds = [p0.distance(h) for h in hits if not h.is_empty]
            side.append(min(ds) if ds else None)
        if None not in side:
            ws.append(side[0] + side[1])
    if ws:
        widths[cnn] = float(np.median(ws))
name_w = {}
for cnn, nm in zip(st["cnn"].values, st["streetname"].values):
    if cnn in widths:
        name_w.setdefault(nm, []).append(widths[cnn])
name_w = {k: float(np.median(v)) for k, v in name_w.items()}
W_MED = float(np.median(list(widths.values()))) if widths else 60.0
edges = []
for cnn, nm, g in zip(st["cnn"].values, st["streetname"].values, st.geometry.values):
    if g is None or g.is_empty:
        continue
    wd = widths.get(cnn, name_w.get(nm, W_MED))
    for sgn in (1, -1):
        try:
            e = g.offset_curve(sgn * wd / 2)
        except Exception:
            continue
        if not e.is_empty:
            edges.append(e)
P2, I2, A2, B2, sid0 = sample(edges, sid0)

# weights: how little the evidence already there explains each added point
s = W_MED / 2
base = SD["pts"]
d1 = cKDTree(base).query(P1)[0] if len(P1) and len(base) else np.full(len(P1), np.inf)
w1 = 1 - np.exp(-d1 ** 2 / (2 * s ** 2))
known2 = np.vstack([base, P1]) if len(P1) else base
d2 = cKDTree(known2).query(P2)[0] if len(P2) and len(known2) else np.full(len(P2), np.inf)
w2 = 1 - np.exp(-d2 ** 2 / (2 * s ** 2))
pts = np.vstack([SD["pts"], P1, P2])
out = dict(pts=pts, sid=np.r_[SD["sid"], I1, I2], ta=np.vstack([SD["ta"], A1, A2]), tb=np.vstack([SD["tb"], B1, B2]),
           wt=np.r_[np.ones(len(SD["pts"])), w1, w2], kind=np.r_[np.zeros(len(SD["pts"]), int), np.ones(len(P1), int),
                                                                   np.full(len(P2), 2)])
np.savez(S + "extra_evidence_points.npz", **out)
print(f"{SHEET}: block-side points {len(SD['pts'])}; later-block points {len(P1)} (weight sum {w1.sum():.0f}); street-edge "
      f"points {len(P2)} (weight sum {w2.sum():.0f}); street widths measured on {len(widths)} of {len(st)} segments, "
      f"median {W_MED:.0f} ft")
