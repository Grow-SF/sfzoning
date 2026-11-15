"""Street edges and later blocks, for sheets whose own year has few blocks where the map prints a street grid: the
block-side points (weight 1), plus
  later blocks   lots recorded after the map's year, merged by block: their outlines (land held whole in the map's year
                 and split up later, such as the Hunters Point shipyard, often had its streets printed long before);
  street edges   every street centre line (DataSF, current or retired) offset by half its width each way: the map
                 prints a street as its two edges.  Each street's width (by cnn) is the gap between the block outlines
                 either side, measured along it (median); a street with no block beside it takes its name's median,
                 else the sheet's median.
Each added point counts by how little the evidence already there explains it: w = 1 - exp(-d^2 / (2 s^2)), d its
distance to the nearest block-side point (for street edges: to block sides or later-block points), s half the
sheet's median street width.  So an edge with a block outline on it counts nothing, an edge with nothing near counts
in full.  Sides are cut into straight stretches and sampled every 2 ft with tangents, as `block_side_points` does."""
import logging

import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import cKDTree

from sfzmaps import config
from sfzmaps.districts import lots as lotsmod
from sfzmaps.districts import read
from sfzmaps.sheetwarp.sheet import Sheet
from sfzmaps.sheetwarp.targets import SidePoints

log = logging.getLogger(__name__)

BLOCK_SIDE, LATER_BLOCK, STREET_EDGE = 0, 1, 2


def unexplained_weight(d: np.ndarray, s: float) -> np.ndarray:
    """How little the evidence already there explains a point at distance `d` (ft) from it: 0 on it, nearly 1 far
    from any; `s` is half the sheet's median street width."""
    return 1 - np.exp(-d ** 2 / (2 * s ** 2))


def _sample(lines, area, sid0):
    """Straight stretches every 2 ft with tangents (as `block_side_points`); returns pts, sid, ta, tb, next sid."""
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


def street_edges_and_later_blocks(sheet: Sheet, sides: SidePoints) -> SidePoints:
    """The block sides plus the later-block and street-edge points, with `wt` (the weights) and `kind` (0 block
    side, 1 later block, 2 street edge).  The Presidio is cut out of the area once (block sides: twice)."""
    T0 = sheet.transform
    Y = sheet.year
    lots, area = sheet.lots, sheet.area(cuts=1)
    Hp, Wp = sheet.page.shape
    pg = shapely.Polygon(T0.to_world(np.array([[0, 0], [Wp, 0], [Wp, Hp], [0, Hp]], float)))
    blocks = lots[lots.intersects(pg)].dissolve(by="block_num").geometry
    bl_lines = shapely.union_all([g.boundary for g in blocks.values])

    # later blocks: lots recorded after the map's year, by block
    ev = read.load_parcels(every=True)
    ev = ev[ev.intersects(pg) & ev.geometry.notna()]
    lv = lotsmod.lives().set_index("blklot").reindex(ev["blklot"].astype(str))
    later = ev[lv["created"].values > Y + 0.5]
    lb = later.dissolve(by="block_num").geometry if len(later) else gpd.GeoSeries([])
    sid0 = int(sides.sid.max()) + 1 if len(sides.sid) else 0
    P1, I1, A1, B1, sid0 = _sample([g.boundary for g in lb.values], area, sid0)

    # street widths from the blocks either side
    st = gpd.read_parquet(config.REF_DIR / "streets.parquet")
    st = st[st.intersects(pg)]
    bl_parts = list(shapely.get_parts(bl_lines))
    bl_tree = shapely.STRtree(bl_parts)
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
    P2, I2, A2, B2, sid0 = _sample(edges, area, sid0)

    # weights: how little the evidence already there explains each added point
    s = W_MED / 2
    base = sides.pts
    d1 = cKDTree(base).query(P1)[0] if len(P1) and len(base) else np.full(len(P1), np.inf)
    w1 = unexplained_weight(d1, s)
    known2 = np.vstack([base, P1]) if len(P1) else base
    d2 = cKDTree(known2).query(P2)[0] if len(P2) and len(known2) else np.full(len(P2), np.inf)
    w2 = unexplained_weight(d2, s)
    pts = np.vstack([sides.pts, P1, P2])
    ta = np.vstack([sides.ta, A1, A2])
    tb = np.vstack([sides.tb, B1, B2])
    log.info("%s/%s: block-side points %d; later-block points %d (weight sum %.0f); street-edge points %d (weight "
             "sum %.0f); street widths measured on %d of %d segments, median %.0f ft", sheet.series, sheet.name,
             len(sides.pts), len(P1), w1.sum(), len(P2), w2.sum(), len(widths), len(st), W_MED)
    # placed through the grid, one call on each whole array (its iteration stops on the batch's largest step)
    page_shape = sheet.page.shape
    return SidePoints(pts=pts, sid=np.r_[sides.sid, I1, I2], ta=ta, tb=tb, px=T0.grid_to_pixel(pts, page_shape),
                      pa=T0.grid_to_pixel(ta, page_shape), pb=T0.grid_to_pixel(tb, page_shape),
                      wt=np.r_[np.ones(len(sides.pts)), w1, w2],
                      kind=np.r_[np.full(len(sides.pts), BLOCK_SIDE), np.full(len(P1), LATER_BLOCK),
                                 np.full(len(P2), STREET_EDGE)])
