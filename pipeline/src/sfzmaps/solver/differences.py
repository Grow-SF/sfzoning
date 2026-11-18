"""The page of differences: a zoning map published from its series' solve against the map the site shows, for the
owner's eye.  $0: no model is asked, and nothing under data/out is written.

    uv run sfzmaps solver publish --series S --page DIR      (after the publish, its new map from out/<S>/districts)
    uv run python scripts/incidents/ship_differences.py --series S   (the new map published the stage's way into the
                                                                      experiments folder, never data/out)

1. The new map: out/<series>/districts as the publish stage wrote it (`new_dir`), or the series published the stage's
   way into `ship_dir`/new-map/ (`new_map`).  Either way, the solve's codes on the read's lots before the overrides
   (`assemble.published_read`), for the corrections' measure.
2. The site's map (`site_codes`): the series' code per lot in the site's archive (data/out/shared/districts.json's
   cards: what a lot card shows), or in a parcels.fgb.  A lot the archive leaves out as stacked on another lot's
   footprint (condos) takes that footprint's code.
3. The differences: the new map's lots whose code (one spelling) is not the site's, in connected areas: lots with the
   same (site, new) pair whose solve lots (the one under most of each lot) are the same or the solve's neighbours
   (<root>/lots/lots.npz pairs, across streets too).
4. The ground truth, whatever the series has:
   - the April 2000 table (TABLE_SERIES: 2000 and 1996-2000): each lot's table lot (the one under most of it) and its
     code with the by-eye corrections; the six measures (`report.six_measures`) of both maps and of the solve, each
     map reaching the table's lots by majority area, with a paired block bootstrap (published - new: wrong lots,
     labels right, incidents; 1,000 draws, seed 0, criteria_check's);
   - the site's corrections (every series): overrides.csv (by lot and by block, as the publish applies them) and the
     label fixes that give a code (label_fixes.csv, each placed on the map through its sheet's warp).  Both maps take
     the overrides, so the summary scores the reads before them, the solve's code and the old read's on the corrected
     lots; and both maps at the fixed labels.
   Each area's verdict is the table's where there is one, else the corrections': which code most of its coded lots
   back, the new map's, the site's, neither, or none.
5. The page: every area, largest first, with the scan around it (the series' merged page, <root>/page/ink.png on the
   solver's grid, one 4 ft grid px per image px: the scan's own resolution), the area's lots tinted, both codes, the
   ground truth and the lots.  Crops load on tap from crops/chunk-NNN.json.

Writes `ship_dir(series)` (compare.json, areas.csv, areas.json, lots.parquet; new-map/) and the page (index.html,
crops/) into `page_dir` (default `ship_dir`/differences)."""
from __future__ import annotations

import base64
import gzip
import io
import json
import shutil
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sfzmaps import config
from sfzmaps.solver import grid, paths, runs
from sfzmaps.solver.codes import canon

TABLE_SERIES = frozenset({"2000-digital-zoning", "1996-2000-zoning"})   # ground truth: the April 2000 table
DRAWS, SEED = 1000, 0             # the paired block bootstrap (criteria_check's)
MARGIN = 60                       # grid px of scan around an area (render_incident.py's)
PER_CHUNK = 100                   # crops per file the page fetches
STRIP = 14                        # px of the scale-bar strip below each crop
PAPER, INK = (245, 240, 229), (36, 33, 30)      # warp.PAPER_TINT, and the ink as the site's paper layer shows it
TINT, ALPHA = (31, 95, 191), 0.32               # the area's lots: the page's accent, see-through over the scan
TEMPLATE = Path(__file__).with_name("differences.html")


def label(series: str) -> str:
    """How the page names a series' map: '2000' for the 2000 digital map, else the series without '-zoning'."""
    from sfzmaps.districts import solved

    return "2000" if series == solved.SERIES else series.removesuffix("-zoning")


def ship_dir(series: str) -> Path:
    """Where the page's data and the experiment's new map go: data/solver/experiments/ship (2000), ship-<series>."""
    from sfzmaps.districts import solved

    return runs.EXPERIMENTS_DIR / ("ship" if series == solved.SERIES else f"ship-{series}")


# --- the maps -----------------------------------------------------------------------------------

def new_map(series: str, dest: Path) -> tuple[Path, pd.DataFrame]:
    """Publish `series` from its solve (as `solved.rooted` sets it) the stage's way into dest/districts (its heights
    copied beside it, as the stage finds them), never data/out; its districts folder and the solve's read."""
    from sfzmaps.districts import assemble
    from sfzmaps.districts.series import SERIES

    d = dest / "districts"
    d.mkdir(parents=True, exist_ok=True)
    for n in ("height.parquet", "height.json"):
        if (assemble.out_dir(series) / n).exists():
            shutil.copy(assemble.out_dir(series) / n, d / n)
    read = assemble.published_read(series)
    real = assemble.out_dir
    assemble.out_dir = lambda s, partial=False: d if s == series and not partial else real(s, partial=partial)
    try:
        assemble.publish(series, read, SERIES[series].year)
    finally:
        assemble.out_dir = real
    return d, read


def site_codes(series: str, blklots: pd.Series, geoms, published: str | Path = "shared") -> tuple[np.ndarray, str]:
    """The site's code (as written, "" for none) for each lot (`blklots`, `geoms`), and where it was read: the series'
    era of the site's archive ('shared': data/out/shared/districts.json's cards) or a parcels.fgb.  A lot the archive
    leaves out takes the code of a lot of the same footprint that it has (`shared._one_per_footprint`)."""
    import shapely

    blk = pd.Series(np.asarray(blklots).astype(str))
    if str(published) == "shared":
        ptr = json.loads((config.OUT_DIR / "shared" / "districts.json").read_text())
        cards = json.loads(gzip.decompress((config.OUT_DIR / ptr["cards"]).read_bytes()))
        if series not in cards["eras"]:
            raise ValueError(f"the site's archive {ptr['archive']} has no {series}")
        strings = cards["strings"]
        code = {b: strings[i] for b, i in zip(cards["blklot"], cards["eras"][series]["code"]) if i is not None}
        listed = set(cards["blklot"])
        where = f"the site's map (archive {Path(ptr['archive']).stem})"
    else:
        import pyogrio

        g = pyogrio.read_dataframe(published, columns=["blklot", "code"], read_geometry=False)
        g = g[g["code"].notna() & (g["code"] != "")]
        code, listed, where = dict(zip(g["blklot"].astype(str), g["code"])), set(g["blklot"].astype(str)), str(published)
    out = blk.map(code)
    lost = ~blk.isin(listed)
    if lost.any():
        wkb = pd.Series(shapely.to_wkb(np.asarray(geoms, dtype=object)))
        known = pd.DataFrame({"wkb": wkb[~lost], "code": out[~lost]}).dropna().drop_duplicates("wkb")
        out[lost] = wkb[lost].map(dict(zip(known["wkb"], known["code"])))
    return out.fillna("").astype(str).to_numpy(), where


def hosts(geoms, table_geoms) -> np.ndarray:
    """Each lot's table lot: the one covering most of it, else the nearest (-1 for a lot with no shape)."""
    import shapely
    from shapely import STRtree

    tg = shapely.make_valid(np.asarray(table_geoms, dtype=object))
    geoms = np.asarray(geoms, dtype=object)
    has = np.array([isinstance(g, shapely.Geometry) and not shapely.is_empty(g) for g in geoms], bool)
    idx = np.flatnonzero(has)
    lg = shapely.make_valid(geoms[idx])
    tree = STRtree(tg)
    li, ti = tree.query(lg, predicate="intersects")
    a = shapely.area(shapely.intersection(lg[li], tg[ti]))
    ov = pd.DataFrame({"lot": idx[li], "t": ti, "a": a})
    best = ov[ov["a"] > 0].sort_values(["lot", "a", "t"], ascending=[True, False, True]).drop_duplicates("lot")
    out = np.full(len(geoms), -1)
    out[best["lot"].to_numpy()] = best["t"].to_numpy()
    rest = np.flatnonzero(has & (out < 0))
    if len(rest):
        qi, qt = tree.query_nearest(lg[np.searchsorted(idx, rest)], all_matches=False)
        out[rest[qi]] = qt
    return out


def differences(new, site, solve) -> pd.DataFrame:
    """The new map's lots (`new`: blklot, code, geometry) with the site's code (`site`) beside, both in the one
    spelling too (c_new, c_pub), their block and solve lot (`host`, `solve`'s table), and the differing ones in
    connected areas (`area`, -1 where the codes agree)."""
    import geopandas as gpd
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from sfzmaps.districts import assemble

    df = gpd.GeoDataFrame({"blklot": new["blklot"].astype(str).to_numpy(),
                           "code_new": new["code"].where(new["code"].notna(), "").astype(str).to_numpy(),
                           "code_pub": site}, geometry=new.geometry.to_numpy(), crs=new.crs)
    for k in ("new", "pub"):
        df[f"c_{k}"] = [canon(c) for c in df[f"code_{k}"]]
    df["block"] = assemble.blocks_of(df["blklot"]).values
    table = _table(solve)
    df["host"] = hosts(df.geometry.to_numpy(), table.geometry.to_numpy())
    d = (df["c_new"] != df["c_pub"]).to_numpy() & (df["host"] >= 0).to_numpy()
    df["area"] = -1
    sub = df[d].copy()
    if not len(sub):
        return df
    sub["key"] = sub["c_pub"] + "|" + sub["c_new"]
    groups = sub[["host", "key"]].drop_duplicates().reset_index(drop=True)
    groups["gid"] = np.arange(len(groups))
    with np.load(solve.lots_file()) as z:
        pairs = pd.DataFrame({"a": z["pairs"][:, 0].astype(int), "b": z["pairs"][:, 1].astype(int)})
    e = pairs.merge(groups.rename(columns={"host": "a", "gid": "ga"}), on="a") \
             .merge(groups.rename(columns={"host": "b", "gid": "gb"}), on=["b", "key"])
    n = len(groups)
    _, comp = connected_components(coo_matrix((np.ones(len(e)), (e["ga"], e["gb"])), shape=(n, n)), directed=False)
    sub = sub.merge(groups, on=["host", "key"], how="left")
    df.loc[d, "area"] = comp[sub["gid"].to_numpy()]
    return df


def _table(solve):
    import geopandas as gpd

    return gpd.read_parquet(solve.table_file())


# --- the ground truth ---------------------------------------------------------------------------

def table_truth(df, solve) -> pd.Series:
    """Each lot's April 2000 table code (the table lot under most of it): the scored table's (its by-eye corrections
    applied), else the table's own words (a split 'RH-1/RH-2', or 'none')."""
    from sfzmaps.solver.city import city

    C = city()
    same = Path(solve.table_file()).resolve() == (paths.LOTS_DIR / "table.parquet").resolve()
    h = df["host"].to_numpy() if same else hosts(df.geometry.to_numpy(), C.lots.geometry.to_numpy())
    zoning = C.lots["zoning"].fillna("").astype(str).to_numpy()
    return pd.Series([("none" if i < 0 else C.table[i] if C.known[i] else (zoning[i] or "none")) for i in h],
                     index=df.index)


def corrections(series: str, df) -> tuple[pd.DataFrame, dict]:
    """The site's corrections on the new map's lots: per lot its corrected code (`x_code`, one spelling, "" for none)
    and how (`x_kind`: 'override' or 'label fix'; an override wins on a lot with both); and their counts."""
    import csv

    import shapely
    from shapely import STRtree

    from sfzmaps import inputs
    from sfzmaps.districts import assemble, lots
    from sfzmaps.districts.read import series_year
    from sfzmaps.transform import SheetTransform

    out = pd.DataFrame({"x_code": "", "x_kind": ""}, index=df.index)
    counts = {"override_lots": 0, "label_fixes": 0, "label_fixes_placed": 0}
    # label fixes that give a code, at the fix point where there is one (a leader line's box), else the label's
    path = inputs.label_fixes_path(series)
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if (r.get("fix_code") or "").strip() not in ("", "-")]
    counts["label_fixes"] = len(rows)
    if rows:
        tree, geoms = STRtree(shapely.make_valid(df.geometry.to_numpy())), {}
        for r in rows:
            folder = config.DATA_DIR / "warp" / series / r["sheet"]
            if not (folder / "transform.json").is_file():
                continue
            if r["sheet"] not in geoms:
                geoms[r["sheet"]] = SheetTransform.from_json(json.loads((folder / "transform.json").read_text()),
                                                            folder=folder)
            fx, fy = (r.get("fix_x") or "").strip(), (r.get("fix_y") or "").strip()
            x, y = (float(fx), float(fy)) if fx and fy else (float(r["x"]), float(r["y"]))
            pt = shapely.Point(*geoms[r["sheet"]].to_world(np.array([[x, y]]))[0][:2])
            hit = tree.query(pt, predicate="intersects")
            if len(hit):
                i = df.index[int(hit[0])]
                out.loc[i, ["x_code", "x_kind"]] = [canon(r["fix_code"]), "label fix"]
                counts["label_fixes_placed"] += 1
    # overrides, by lot and by block, exactly as the publish applies them
    if inputs.overrides_path(series).exists():
        yr = series_year(series)
        base = pd.DataFrame({"blklot": df["blklot"].to_numpy(), "code": None, "confidence": 0.0}, index=df.index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = assemble.apply_overrides(base, series, blocks=df["block"],
                                         remap=lots.era_lot_of(yr) if yr is not None else None)
        hit = o["code"].notna() & (o["code"].astype(str) != "")
        out.loc[hit, "x_code"] = [canon(c) for c in o.loc[hit, "code"]]
        out.loc[hit, "x_kind"] = "override"
        counts["override_lots"] = int(hit.sum())
    return out, counts


def correction_measures(series: str, df, read) -> dict:
    """The reads before the overrides on the overridden lots (the solve's code, `read`; the old read's,
    districts_raw.parquet), and both maps at the placed label fixes: how many each gets right."""
    from sfzmaps.districts import assemble

    old = pd.read_parquet(assemble._raw_path(series), columns=["blklot", "code"])
    old = dict(zip(old["blklot"].astype(str), old["code"].fillna("").astype(str)))
    solve = dict(zip(read["blklot"].astype(str), read["code"].fillna("").astype(str)))
    ov = df[df["x_kind"] == "override"]
    fx = df[df["x_kind"] == "label fix"]
    return {"override_lots": len(ov),
            "override_solve_right": int(sum(canon(solve.get(b, "")) == c for b, c in zip(ov["blklot"], ov["x_code"]))),
            "override_read_right": int(sum(canon(old.get(b, "")) == c for b, c in zip(ov["blklot"], ov["x_code"]))),
            "label_fix_lots": len(fx), "label_fix_new_right": int((fx["c_new"] == fx["x_code"]).sum()),
            "label_fix_site_right": int((fx["c_pub"] == fx["x_code"]).sum())}


# --- the table's measures (TABLE_SERIES) --------------------------------------------------------

def per_block(codes, blocks, nb):
    """Per block: the incidents (each incident split over its lots' blocks by lot), the zoning labels counted, the
    zoning labels right, the wrong lots; the totals as report.six_measures counts them (criteria_check's)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from sfzmaps.solver.city import city
    from sfzmaps.solver.measures import Table

    C = city()
    a, b = C.a, C.b
    wrong = C.scored & (codes != C.table)
    key = np.char.add(np.char.add(C.table, "|"), codes)
    m = wrong[a] & wrong[b] & (key[a] == key[b])
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(C.n, C.n)),
                                   directed=False)
    size = np.bincount(comp[wrong], minlength=comp.max() + 1)
    inc = np.bincount(blocks[wrong], 1.0 / size[comp[wrong]], minlength=nb)
    T = Table(np.zeros(C.n, bool))
    nd = T.district.max() + 1
    lots_in = np.bincount(T.district[C.scored], minlength=nd)
    right_in = np.bincount(T.district[(codes == C.table) & C.scored], minlength=nd)
    d = np.maximum(T.label_district, 0)
    counted = (T.label_district < 0) | (lots_in[d] > 0)
    right = (T.label_district >= 0) & (lots_in[d] > 0) & (right_in[d] > lots_in[d] / 2)
    zoning = counted & T.label_in_vocab
    label_block = blocks[T.label_lot]
    return {"incidents": inc, "zoning": np.bincount(label_block[zoning], minlength=nb).astype(float),
            "right": np.bincount(label_block[zoning & right], minlength=nb).astype(float),
            "wrong_lots": np.bincount(blocks[wrong], minlength=nb).astype(float)}


def bootstrap(fin, new, nb):
    """The paired block bootstrap's 95% interval of (new - fin) for incidents, labels right and wrong lots."""
    rng = np.random.default_rng(SEED)
    out = {"incidents": [], "labels_right": [], "wrong_lots": []}
    for _ in range(DRAWS):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        out["incidents"].append(w @ new["incidents"] - w @ fin["incidents"])
        out["labels_right"].append(w @ new["right"] / (w @ new["zoning"]) - w @ fin["right"] / (w @ fin["zoning"]))
        out["wrong_lots"].append(w @ new["wrong_lots"] - w @ fin["wrong_lots"])
    return {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in out.items()}


def table_measures(df, solve) -> dict:
    """The six measures on the April 2000 table's lots of the site's map and the new map (each table lot taking the
    map's code covering most of it) and, where the solve is on the table's lots, of the solve; the bootstrap."""
    from sfzmaps.districts import solved
    from sfzmaps.solver.city import city, load_run
    from sfzmaps.solver.measures import check_block_split
    from sfzmaps.solver.report import six_measures

    C = city()
    nothing = np.zeros(C.n, bool)
    tg, geoms = C.lots.geometry.to_numpy(), df.geometry.to_numpy()
    codes = {k: solved.majority_codes(tg, geoms, df[f"c_{c}"].to_numpy())[0]
             for k, c in (("published", "pub"), ("new", "new"))}
    with np.load(solve.run_file()) as R:
        on_table = len(R["pred"]) == C.n and Path(solve.table_file()).resolve() == \
            (paths.LOTS_DIR / "table.parquet").resolve()
    if on_table:
        codes["solve"] = load_run(runs.FINAL_RUN, solve.solve_dir)
    m = {k: six_measures(v, nothing) for k, v in codes.items()}
    _, blocks = np.unique(C.lots["blklot"].astype(str).str[:4].values, return_inverse=True)
    nb = int(blocks.max()) + 1
    pa, pb = per_block(codes["published"], blocks, nb), per_block(codes["new"], blocks, nb)
    check_block_split(pa, blocks, codes["published"], nothing, m["published"], "published")
    check_block_split(pb, blocks, codes["new"], nothing, m["new"], "new")
    return {"measures_every_lot": m, "bootstrap_95": {"published - new": bootstrap(pb, pa, nb)}, "draws": DRAWS,
            "seed": SEED, "blocks": nb}


# --- the areas ----------------------------------------------------------------------------------

def areas(df, table: bool) -> pd.DataFrame:
    """One row per area, largest first: lots, both codes (as the maps write them), the ground truth (the table's codes
    over its lots where `table`, else the corrections'), how many of its lots it backs each way, the verdict, the
    corrections on its lots, its blocks, its quarter (of the solver's split) and its box (map ft)."""
    rows = []
    for a, g in df[df["area"] >= 0].groupby("area"):
        truth = list(g["t_code"]) if table else [c for c in g["x_code"] if c]
        single = [t for t in truth if "/" not in t and t != "none"]
        c_new, c_pub = g["c_new"].iat[0], g["c_pub"].iat[0]
        backs_new = sum(canon(t) == c_new for t in single)
        backs_pub = sum(canon(t) == c_pub for t in single)
        verdict = ("new" if backs_new > len(single) / 2 else "published" if backs_pub > len(single) / 2
                   else "neither" if single else "none")
        x0, y0, x1, y1 = g.total_bounds
        fixes = Counter(f"{k} {c}" for k, c in zip(g["x_kind"], g["x_code"]) if c)
        rows.append({"area": int(a), "lots": len(g),
                     "published": Counter(g["code_pub"]).most_common(1)[0][0],
                     "new": Counter(g["code_new"]).most_common(1)[0][0], "c_pub": c_pub, "c_new": c_new,
                     "truth": Counter(truth).most_common(1)[0][0] if truth else "none",
                     "truth_codes": dict(Counter(truth).most_common()), "truth_coded": len(single),
                     "truth_backs_new": int(backs_new), "truth_backs_published": int(backs_pub), "verdict": verdict,
                     "corrections": dict(fixes.most_common()),
                     "blocks": sorted(set(g["block"].astype(str))),
                     "quarter": paths.QUARTERS[int(grid.quarter_of(np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]))[0])],
                     "box": [float(x0), float(y0), float(x1), float(y1)]})
    cols = ["area", "lots", "published", "new", "c_pub", "c_new", "truth", "truth_codes", "truth_coded",
            "truth_backs_new", "truth_backs_published", "verdict", "corrections", "blocks", "quarter", "box"]
    out = pd.DataFrame(rows, columns=cols)
    out = out.sort_values(["lots", "area"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# --- the page -----------------------------------------------------------------------------------

def load_ink(solve) -> np.ndarray | None:
    """The series' merged page's ink on the solver's grid (<root>/page/ink.png beside its solve), or None."""
    import cv2

    f = Path(solve.solve_dir).parent / "page" / "ink.png"
    a = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) if f.is_file() else None
    return None if a is None else a < grid.INK_BELOW


def crop(ink, geoms):
    """The scan around an area (its lots' box plus MARGIN, one grid px per image px), its lots tinted and outlined,
    a scale bar in a strip below; a palette PNG's bytes, the bar's length in ft, the crop's size."""
    import cv2
    import shapely
    from PIL import Image

    x0, y0, x1, y1 = shapely.total_bounds(geoms)
    (c0, r0), (c1, r1) = grid.to_grid([[x0, y1], [x1, y0]])
    c0, r0 = max(int(np.floor(c0)) - MARGIN, 0), max(int(np.floor(r0)) - MARGIN, 0)
    c1, r1 = min(int(np.ceil(c1)) + MARGIN, grid.WIDTH), min(int(np.ceil(r1)) + MARGIN, grid.HEIGHT)
    page = ink[r0:r1, c0:c1]
    mask = np.zeros(page.shape, np.uint8)
    for g in geoms:
        for poly in getattr(g, "geoms", [g]):
            if poly.geom_type != "Polygon":
                continue
            rings = [poly.exterior, *poly.interiors]
            pts = [np.round((grid.to_grid(np.asarray(r.coords)[:, :2]) - [c0, r0]) * 4).astype(np.int32) for r in rings]
            cv2.fillPoly(mask, pts[:1], 1, shift=2)
            if len(pts) > 1:
                cv2.fillPoly(mask, pts[1:], 0, shift=2)
    edge = mask.astype(bool) & ~cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool)
    idx = np.where(page, 1, 0).astype(np.uint8)
    idx[mask.astype(bool)] += 2
    idx[edge] = 4
    w = idx.shape[1]
    bar_ft = 100 if w < 400 else 500
    bar = int(bar_ft / grid.FT_PER_PX)
    strip = np.full((STRIP, w), 6, np.uint8)                  # below the scan, so it never covers ink
    strip[5:9, 6:6 + min(bar, w - 12)] = 5
    idx = np.vstack([idx, strip])
    h = idx.shape[0]
    tint = [tuple(round(ALPHA * t + (1 - ALPHA) * p) for t, p in zip(TINT, base)) for base in (PAPER, INK)]
    palette = [*PAPER, *INK, *tint[0], *tint[1], *TINT, 20, 20, 20, 255, 255, 255]
    im = Image.fromarray(idx, mode="P")
    im.putpalette(palette)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), bar_ft, (w, h)


def write_crops(df, A, ink, page_dir: Path) -> dict:
    """Every area's crop, PER_CHUNK to a file: crops/chunk-NNN.json {rank: data URI}.  {rank: (bar_ft, w, h)};
    (0, 0, 0) for every area when there is no scan."""
    d = page_dir / "crops"
    shutil.rmtree(d, ignore_errors=True)
    if ink is None:
        return {r: (0, 0, 0) for r in A["rank"]}
    d.mkdir(parents=True)
    geoms = {a: g.geometry.to_numpy() for a, g in df[df["area"] >= 0].groupby("area")}
    out, chunk = {}, {}
    for r in A.itertuples():
        png, bar_ft, (w, h) = crop(ink, geoms[r.area])
        chunk[str(r.rank)] = "data:image/png;base64," + base64.b64encode(png).decode()
        out[r.rank] = (bar_ft, w, h)
        if r.rank % PER_CHUNK == 0 or r.rank == len(A):
            (d / f"chunk-{(r.rank - 1) // PER_CHUNK:03d}.json").write_text(json.dumps(chunk, separators=(",", ":")))
            chunk = {}
    return out


def page_rows(A, crops) -> list:
    rows = []
    for r in A.itertuples():
        blocks = list(r.blocks)
        single = [c for c in r.truth_codes if "/" not in c and c != "none"]
        rows.append({"r": r.rank, "n": r.lots, "p": r.published, "w": r.new, "t": single[0] if single else r.truth,
                     "tc": [[k, v] for k, v in list(r.truth_codes.items())[:4]], "tx": max(len(r.truth_codes) - 4, 0),
                     "k": r.truth_coded, "bn": r.truth_backs_new, "bp": r.truth_backs_published, "v": r.verdict,
                     "fx": [[k, v] for k, v in list(r.corrections.items())[:6]],
                     "b": blocks[:10], "bx": max(len(blocks) - 10, 0), "q": r.quarter,
                     "bar": crops[r.rank][0], "cw": crops[r.rank][1], "ch": crops[r.rank][2]})
    return rows


def summary(series, A, made, m, fixes, n_lots, where, scan) -> dict:
    """What the page's header says: the series, the ground truth, the measures, the corrections, the counts."""
    table = series in TABLE_SERIES
    out = {"series": series, "label": label(series), "table": table, "site": where, "scan": scan,
           "areas": len(A), "lots": n_lots, "readLots": made["read_lots"], "coded": made["read_lots_coded"],
           "unknown": len(made["unknown_codes"]), "confidence": made["confidence"],
           "verdicts": {k: int(v) for k, v in Counter(A["verdict"]).items()}, "fixes": fixes, "rows": [], "solve": None}
    if m:
        M, ci = m["measures_every_lot"], m["bootstrap_95"]["published - new"]
        out["rows"] = [
            ["Wrong lots", f"{M['published']['wrong_lots']:,}", f"{M['new']['wrong_lots']:,}",
             _fewer(M["published"]["wrong_lots"] - M["new"]["wrong_lots"]),
             f"{ci['wrong_lots'][0]:,.0f} to {ci['wrong_lots'][1]:,.0f}"],
            ["Labels whose district comes out right", f"{M['published']['labels_right']:.1%}",
             f"{M['new']['labels_right']:.1%}",
             f"{(M['new']['labels_right'] - M['published']['labels_right']) * 100:+.1f} pts",
             f"{-ci['labels_right'][1] * 100:.1f} to {-ci['labels_right'][0] * 100:.1f} pts"],
            ["Wrong-lot incidents", f"{M['published']['incidents']:,}", f"{M['new']['incidents']:,}",
             _fewer(M["published"]["incidents"] - M["new"]["incidents"]),
             f"{ci['incidents'][0]:,.0f} to {ci['incidents'][1]:,.0f}"]]
        if "solve" in M:
            out["solve"] = [M["solve"]["wrong_lots"], M["solve"]["labels_right"], M["solve"]["incidents"]]
    return out


def _fewer(d: int) -> str:
    return f"{d:,} fewer" if d >= 0 else f"{-d:,} more"


def write_page(S, rows, page_dir: Path) -> Path:
    html = (TEMPLATE.read_text()
            .replace("__TITLE__", f"{S['label']} Map Differences")
            .replace("__SUMMARY__", json.dumps(S).replace("</", "<\\/"))
            .replace("__ROWS__", json.dumps(rows, separators=(",", ":")).replace("</", "<\\/")))
    out = page_dir / "index.html"
    out.write_text(html)
    return out


def make(series: str, *, root=None, published: str | Path = "shared", page_dir: Path | None = None,
         new_dir: Path | None = None, log=print) -> Path:
    """The page of differences of `series` (see the module's doc): the new map from `new_dir` (a districts folder the
    publish stage wrote) or published the stage's way into ship_dir(series)/new-map; against `published`.  Returns
    the page's index.html."""
    import geopandas as gpd

    from sfzmaps.districts import assemble, solved

    out_dir = ship_dir(series)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_dir = Path(page_dir or out_dir / "differences")
    page_dir.mkdir(parents=True, exist_ok=True)
    with solved.rooted(series, root) as solve:
        if new_dir is None:
            new_dir, read = new_map(series, out_dir / "new-map")
        else:
            read = assemble.published_read(series)
        source = solved.confidence_source(series)
        conf = solved.score(series) if source == "score" else solved.provenance(series)["mean_confidence"]
    read.to_parquet(out_dir / "districts_raw.parquet")
    new = gpd.read_file(Path(new_dir) / "parcels.fgb")
    coded = set(new.loc[new["code"].notna() & (new["code"] != ""), "blklot"].astype(str))
    lots = read["blklot"].astype(str)
    made = {"new_map": str(new_dir), "read_lots": len(read), "read_lots_coded": int(lots.isin(coded).sum()),
            "published_lots": len(new), "unknown_codes": json.loads((Path(new_dir) / "unknown_codes.json").read_text()),
            "confidence": {"from": source, "value": conf}}
    log(json.dumps(made))
    site, where = site_codes(series, new["blklot"], new.geometry.to_numpy(), published)
    df = differences(new, site, solve)
    x, fixes = corrections(series, df)
    df = df.join(x)
    table = series in TABLE_SERIES
    df["t_code"] = table_truth(df, solve) if table else ""
    fixes.update(correction_measures(series, df, read))
    m = table_measures(df, solve) if table else None
    A = areas(df, table)
    n_diff = int((df["area"] >= 0).sum())
    A.drop(columns=["truth_codes", "corrections", "blocks", "box"]).to_csv(out_dir / "areas.csv", index=False)
    A.to_json(out_dir / "areas.json", orient="records", indent=1)
    df.to_parquet(out_dir / "lots.parquet")
    (out_dir / "compare.json").write_text(json.dumps({"series": series, "site": where, "new_map": made,
                                                      "corrections": fixes, **(m or {}), "differing_lots": n_diff,
                                                      "areas": len(A)}, indent=1, default=float) + "\n")
    log(f"{series}: areas {len(A):,}, differing lots {n_diff:,}; corrections {fixes}")
    ink = load_ink(solve)
    crops = write_crops(df, A, ink, page_dir)
    return write_page(summary(series, A, made, m, fixes, n_diff, where, ink is not None), page_rows(A, crops),
                      page_dir)
