"""Shared by the gap scripts: the labels, the walls and the closed areas of a run, exactly as lotcode.py builds them
(4 ft pixels, row = (y - y0) / 4, not flipped)."""
import json
from pathlib import Path

import cv2
import numpy as np
import shapely
from shapely.ops import unary_union

from sfzmaps import config
from sfzmaps.transform import SheetTransform

RES = 4.0
K = Path(__file__).parent
SERIES = "2000-digital-zoning"
SHEETS = [f"{i:02d}" for i in range(1, 14)]


def labels(series=SERIES):
    W = config.WORK_DIR / series
    pipe = {}
    for nn in SHEETS:
        f = W / f"{series}-p{nn}" / "district_labels.json"
        pipe[nn] = json.load(open(f))["labels"] if f.exists() else []
    vocab = {l["code"].upper().strip() for ls in pipe.values() for l in ls}
    ctrs, codes = [], []
    for nn in SHEETS:
        sheet = f"{series}-p{nn}"
        t = SheetTransform.from_json(json.loads((W / sheet / "transform.json").read_text()))
        drawn = shapely.from_geojson((W / sheet / "warped.footprint.geojson").read_text())
        pl = np.array([[l["x"], l["y"]] for l in pipe[nn]], float).reshape(-1, 2)
        for b in json.load(open(K / f"p{nn}" / "textboxes_gpt-6.1-terra.json")):
            if b["kind"] != "zoning code":
                continue
            bx0, by0, bx1, by1 = b["box"]
            ctr = t.to_world(np.array([[(bx0 + bx1) / 2, (by0 + by1) / 2]], float))[0]
            if not shapely.contains(drawn, shapely.Point(ctr)):
                continue
            code = None
            if len(pl):
                ins = np.flatnonzero((pl[:, 0] >= bx0) & (pl[:, 0] <= bx1) & (pl[:, 1] >= by0) & (pl[:, 1] <= by1))
                if len(ins):
                    j = ins[np.argmin(np.hypot(pl[ins, 0] - (bx0 + bx1) / 2, pl[ins, 1] - (by0 + by1) / 2))]
                    code = pipe[nn][j]["code"].upper().strip()
            if code is None:
                tx = " ".join(b["text"].upper().split())
                if tx in vocab:
                    code = tx
            if code is None:
                continue
            ctrs.append(ctr)
            codes.append(code)
    return np.array(ctrs), np.array(codes, dtype=object)


def build(run="none"):
    """run: none (letters out), all, or a k8/city4a_TAG tag."""
    C = K / "city"
    F = np.load(K / "city4a_v10" / "lines8.npz", allow_pickle=True)
    Z = json.load(open(C / "zgraph.json"))
    G = json.load(open(C / "graph7.json"))
    E = Z["edges"]
    lines = [json.loads(s) for s in F["line_edges"]]
    isf = np.zeros(len(lines), bool)
    isf[F["false"]] = True
    gone = isf.copy()
    folder = K / run if (K / run / "graph7.json").exists() else None      # a chain folder: its own walls and joins
    if folder is not None:
        pass
    elif run == "all":
        gone[:] = False
    elif run != "none":
        gone[np.load(K / f"city4a_{run}" / "lines8.npz", allow_pickle=True)["drop"]] = True
    fp = shapely.from_geojson((C / "footprint.geojson").read_text())
    outline = shapely.boundary(unary_union([shapely.Polygon(p.exterior) for p in shapely.get_parts(fp)]))
    x0, y0, x1, y1 = fp.bounds
    x0, y0 = x0 - 40, y0 - 40
    W_, H_ = int((x1 + 40 - x0) / RES) + 1, int((y1 + 40 - y0) / RES) + 1
    wall = np.zeros((H_, W_), np.uint8)

    def px(c):
        return np.round((np.asarray(c, float)[:, :2] - (x0, y0)) / RES).astype(np.int32).reshape(-1, 1, 2)

    for part in shapely.get_parts(outline):
        cv2.polylines(wall, [px(shapely.get_coordinates(part))], False, 1, 1)
    def key(c):
        c = np.asarray(c, float)
        return (round(c[0, 0], 1), round(c[0, 1], 1), round(c[-1, 0], 1), round(c[-1, 1], 1), len(c))

    if folder is not None:
        Zf, Gf = json.load(open(folder / "zgraph.json")), json.load(open(folder / "graph7.json"))
        G = dict(G, joins=Gf["joins"])
        kept = {key(e["coords"]) for e, c in zip(Zf["edges"], Gf["edge_class"]) if c == "heavy"}
    for c, _ in G["joins"]:
        cv2.polylines(wall, [px(c)], False, 1, 1)
    heavy = {i for i, c in enumerate(G["edge_class"]) if c == "heavy"}
    wall_edges, edge_line = set(), {}
    for k, ch in enumerate(lines):
        for i in ch:
            edge_line[i] = k
        if folder is None and gone[k]:
            continue
        for i in ch:
            if (i in heavy) if folder is None else (key(E[i]["coords"]) in kept):
                cv2.polylines(wall, [px(E[i]["coords"])], False, 1, 1)
                wall_edges.add(i)
    if folder is not None:
        for i, e in enumerate(E):
            if i not in wall_edges and key(e["coords"]) in kept:
                cv2.polylines(wall, [px(e["coords"])], False, 1, 1)
                wall_edges.add(i)
        print(f"walls of {folder.name}: {len(wall_edges)} of its {len(kept)} heavy edges found in the city graph; joins {len(G['joins'])}", flush=True)
    wall[0, :] = wall[-1, :] = 1
    wall[:, 0] = wall[:, -1] = 1
    # the ink that is not a wall, by why: 1 thin on a lot line, 2 thin off, 3 heavy stroke of a letter, 4 heavy line the
    # run removed, 5 heavy edge in no line
    ink = np.zeros((H_, W_), np.uint8)
    for i, e in enumerate(E):
        if i in wall_edges:
            continue
        cls = G["edge_class"][i]
        if cls == "thin_on":
            v = 1
        elif cls == "thin_off":
            v = 2
        elif i in edge_line:
            v = 3 if isf[edge_line[i]] else 4
        else:
            v = 5
        cv2.polylines(ink, [px(e["coords"])], False, v, 1)
    ink[wall > 0] = 0
    ncomp, comp = cv2.connectedComponents((wall == 0).astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
    ctrs, codes = labels()
    sc, sr = np.round((ctrs[:, 0] - x0) / RES).astype(np.int64), np.round((ctrs[:, 1] - y0) / RES).astype(np.int64)
    lab_comp = comp[sr, sc]
    names = sorted(set(codes))
    ci = np.array([names.index(c) for c in codes])
    pairs = np.unique(np.c_[lab_comp, ci], axis=0)
    ncodes = np.bincount(pairs[:, 0], minlength=ncomp)
    ncodes[0] = 0
    return dict(Z=Z, G=G, E=E, lines=lines, isf=isf, gone=gone, wall=wall, ink=ink, comp=comp, ncomp=ncomp, ncodes=ncodes, x0=x0, y0=y0,
                W=W_, H=H_, px=px, wall_edges=wall_edges, edge_line=edge_line, ctrs=ctrs, codes=codes, names=names, lab_comp=lab_comp, fp=fp)


def scan_ink(B, series=SERIES):
    """the scan's own ink (warped.tif band 4, each sheet inside its drawn area) on the wall grid"""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    x0, y0, W_, H_ = B["x0"], B["y0"], B["W"], B["H"]
    out = np.zeros((H_, W_), bool)
    Wd = config.WORK_DIR / series
    for nn in SHEETS:
        d = Wd / f"{series}-p{nn}"
        drawn = shapely.from_geojson((d / "warped.footprint.geojson").read_text())
        bx0, by0, bx1, by1 = drawn.bounds
        c0, c1 = max(0, int((bx0 - x0) / RES)), min(W_ - 1, int((bx1 - x0) / RES) + 1)
        r0, r1 = max(0, int((by0 - y0) / RES)), min(H_ - 1, int((by1 - y0) / RES) + 1)
        with rasterio.open(d / "warped.tif") as src:
            win = from_bounds(x0 + (c0 - .5) * RES, y0 + (r0 - .5) * RES, x0 + (c1 + .5) * RES, y0 + (r1 + .5) * RES, src.transform)
            a = src.read(4, window=win, out_shape=(r1 - r0 + 1, c1 - c0 + 1), boundless=True, fill_value=0, resampling=Resampling.nearest) > 127
        m = np.zeros((r1 - r0 + 1, c1 - c0 + 1), np.uint8)
        for part in polys(drawn):      # (until 2026-11-09 only plain Polygons were filled: sheets 02-05 were dropped)
            for ring, v in [(part.exterior, 1)] + [(r, 0) for r in part.interiors]:
                p = np.round((shapely.get_coordinates(ring) - (x0 + c0 * RES, y0 + r0 * RES)) / RES).astype(np.int32)
                cv2.fillPoly(m, [p.reshape(-1, 1, 2)], v)
        out[r0:r1 + 1, c0:c1 + 1] |= a[::-1] & (m > 0)
    return out


def polys(g):
    """every Polygon inside a geometry, however deeply nested (a drawn area can be a collection of multipolygons)"""
    if g.geom_type == "Polygon":
        return [] if g.is_empty else [g]
    return [q for p in shapely.get_parts(g) if p is not g for q in polys(p)] if hasattr(g, "geoms") else []


def clean_ink(B):
    """the merged cleaned page the graph was read from (k8/city/clean.png, north up) on the wall grid"""
    pg = cv2.imread(str(K / "city" / "clean.png"), cv2.IMREAD_GRAYSCALE) < 128
    bx0, _, _, by1 = B["fp"].bounds
    X0, Y1 = np.floor(bx0 / 100) * 100, np.ceil(by1 / 100) * 100
    cc = np.floor((B["x0"] + np.arange(B["W"]) * RES - X0) / RES).astype(int)
    rr = np.floor((Y1 - (B["y0"] + np.arange(B["H"]) * RES)) / RES).astype(int)
    okc, okr = (cc >= 0) & (cc < pg.shape[1]), (rr >= 0) & (rr < pg.shape[0])
    out = np.zeros((B["H"], B["W"]), bool)
    out[np.ix_(okr, okc)] = pg[np.ix_(rr[okr], cc[okc])]
    return out


def lot_codes(B, code_px, year=2000.0):
    """each lot takes the code covering most of its pixels (as lotcode.py)"""
    import pandas as pd
    from sfzmaps.districts import read
    names, x0, y0, W_, H_ = B["names"], B["x0"], B["y0"], B["W"], B["H"]
    Kn = len(names)
    lots = read.load_parcels(year=year)
    lots = lots[shapely.intersects(lots.geometry.values, shapely.box(*B["fp"].bounds))].reset_index(drop=True)
    geoms = lots.geometry.values
    shape_of = pd.factorize(shapely.to_wkb(geoms))[0]
    first = pd.Series(np.arange(len(lots))).groupby(shape_of).first().values
    lotpx = np.full((H_, W_), -1, np.int32)
    for s, i in enumerate(first):
        for part in shapely.get_parts(geoms[i]):
            if part.geom_type == "Polygon" and not part.is_empty:
                cv2.fillPoly(lotpx, [B["px"](shapely.get_coordinates(part.exterior))], int(s))
    S_ = len(first)
    inlot = lotpx >= 0
    npx = np.bincount(lotpx[inlot], minlength=S_)
    both = inlot & (code_px >= 0)
    cnt = np.bincount(lotpx[both].astype(np.int64) * Kn + code_px[both], minlength=S_ * Kn).reshape(S_, Kn)
    code_s = np.where(cnt.sum(1) > 0, np.array(names, dtype=object)[cnt.argmax(1)], None)
    rp = shapely.get_coordinates(shapely.point_on_surface(geoms[first]))
    cc_, rr_ = np.clip(np.round((rp[:, 0] - x0) / RES).astype(int), 0, W_ - 1), np.clip(np.round((rp[:, 1] - y0) / RES).astype(int), 0, H_ - 1)
    at = code_px[rr_, cc_]
    none = npx == 0
    code_s[none] = [names[v] if v >= 0 else None for v in at[none]]
    return pd.DataFrame({"blklot": lots.blklot.astype(str).values, "code": code_s[shape_of], "npx": npx[shape_of], "area": shapely.area(geoms)})
