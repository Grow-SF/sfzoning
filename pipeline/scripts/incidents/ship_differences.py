"""The 2000 map published from the solve (the `solver-publish` stage) against the published map: the measures, and the
page of differences for the owner's eye.  $0: no model is asked, and nothing under data/out is written.

    uv run python scripts/incidents/ship_differences.py [--page-dir DIR]

1. The new map: the read's lots with the solve's codes (`assemble.published_read`), published as the stage publishes
   it (`assemble.publish`: unzoned blocks, overrides, heights, parks) into data/solver/experiments/ship/new-map/.
2. The measures: `report.six_measures` on every table lot, for the published map, the new map and the solve itself.
   Each map reaches the table's lots by lot ID (`solver.published.save_published`: the measure that gave the published
   map 35,455 wrong lots), with the paired block bootstrap (published - new) of `criteria_check.py`.  Beside it, both
   maps reach the table's lots by majority area instead (`districts.solved.majority_codes` the other way round), since
   1,286 scored table lots have no lot of the same ID on the published lots.
3. The differences: the lots whose published code (parcels.fgb, in the one spelling) is not the new map's, in
   connected areas: lots with the same (published, new) pair whose table lots (the one under most of each lot) are the
   same or neighbours (lots.npz pairs: the solver's neighbours, across streets too).
4. The page: every area, largest first, with the scan around it (the solver's merged page of the warped sheets,
   data/solver/page/ink.png, one 4 ft grid px per image px: the scan's own resolution), the area's lots tinted, both
   codes, the table's code and the lots.

Writes data/solver/experiments/ship/ (compare.json, compare.md, areas.csv, new-map/) and the page into --page-dir
(index.html, crops/)."""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import criteria_check as cc
from sfzmaps.districts import assemble, solved
from sfzmaps.districts.series import SERIES
from sfzmaps.solver import paths, runs
from sfzmaps.solver.city import city, load_run
from sfzmaps.solver.codes import canon
from sfzmaps.solver.measures import check_block_split
from sfzmaps.solver.published import save_published
from sfzmaps.solver.report import six_measures

S = solved.SERIES
OUT = runs.EXPERIMENTS_DIR / "ship"
NEW = OUT / "new-map"
KEYS = (("wrong_lots", "Wrong lots", "{:,}"),
        ("labels_right", "Zoning labels whose district comes out right", "{:.2%}"),
        ("incidents", "Wrong-lot incidents", "{:,}"), ("lines_drawn", "Table district lines the map draws", "{:.2%}"),
        ("drawn_are_table_lines", "Lines the map draws that are table lines", "{:.2%}"), ("holes", "Holes", "{:,}"))


def new_map():
    """Publish the new map into NEW/districts (the heights copied beside it, as the stage finds them) and save its read
    (the solve's codes on the read's lots, before the overrides) as NEW/districts_raw.parquet."""
    import geopandas as gpd

    d = NEW / "districts"
    d.mkdir(parents=True, exist_ok=True)
    for n in ("height.parquet", "height.json"):
        shutil.copy(assemble.out_dir(S) / n, d / n)
    read = assemble.published_read(S)
    read.to_parquet(NEW / "districts_raw.parquet")
    real = assemble.out_dir
    assemble.out_dir = lambda series, partial=False: d if series == S and not partial else real(series, partial=partial)
    try:
        assemble.publish(S, read, SERIES[S].year)
    finally:
        assemble.out_dir = real
    g = gpd.read_file(d / "parcels.fgb")
    coded = set(g.loc[g["code"].notna() & (g["code"] != ""), "blklot"].astype(str))
    lots = read["blklot"].astype(str)
    return {"read_lots": len(read), "read_lots_coded": int(lots.isin(coded).sum()),
            "read_lots_with_no_shape": int((~lots.isin(set(g["blklot"].astype(str)))).sum()),
            "published_lots": len(g), "unknown_codes": json.loads((d / "unknown_codes.json").read_text()),
            "confidence": float(read["confidence"].max()),
            "codes": {str(k): int(v) for k, v in g["code"].value_counts().items()}}


def by_area(parcels_file: Path) -> np.ndarray:
    """A published map brought to the table's lots by majority area: each table lot takes the code (one spelling)
    covering most of it."""
    import geopandas as gpd

    g = gpd.read_file(parcels_file)
    g = g[g.geometry.notna()]
    code = np.array([canon(c) if isinstance(c, str) else "" for c in g["code"]])
    got, _ = solved.majority_codes(city().lots.geometry.to_numpy(), g.geometry.to_numpy(), code)
    return got


def measures(read_files):
    """six_measures, the per-block split and the bootstrap: published and the new map by lot ID; the solve; both maps
    by majority area."""
    C = city()
    nothing = np.zeros(C.n, bool)
    save_published(name="ship-published")
    save_published(read_files["new"], name="ship-new-map")
    codes = {"published": load_run("ship-published"), "new": load_run("ship-new-map"), "solve": load_run("final"),
             "published_by_area": by_area(assemble.out_dir(S) / "parcels.fgb"),
             "new_by_area": by_area(NEW / "districts" / "parcels.fgb")}
    m = {k: six_measures(v, nothing) for k, v in codes.items()}
    _, blocks = np.unique(C.lots["blklot"].astype(str).str[:4].values, return_inverse=True)
    nb = int(blocks.max()) + 1
    out = {"measures_every_lot": m, "bootstrap_95": {}, "draws": cc.DRAWS, "seed": cc.SEED, "blocks": nb}
    for a, b in (("published", "new"), ("published_by_area", "new_by_area")):
        pa, pb = cc.per_block(codes[a], blocks, nb), cc.per_block(codes[b], blocks, nb)
        check_block_split(pa, blocks, codes[a], nothing, m[a], a)
        check_block_split(pb, blocks, codes[b], nothing, m[b], b)
        out["bootstrap_95"][f"{a} - {b}"] = cc.bootstrap(pb, pa, nb)
    raw_ids = set(pd.read_parquet(assemble._raw_path(S), columns=["blklot"])["blklot"].astype(str))
    ids = C.lots["blklot"].astype(str)
    out["scored_table_lots_with_no_lot_of_their_id"] = int((C.scored & ~ids.isin(raw_ids).values).sum())
    return out, codes


def hosts(geoms) -> np.ndarray:
    """Each lot's table lot: the one covering most of it, else the nearest (-1 for a lot with no shape)."""
    import shapely
    from shapely import STRtree

    tg = city().lots.geometry.to_numpy()
    geoms = np.asarray(geoms, dtype=object)
    has = np.array([isinstance(g, shapely.Geometry) and not shapely.is_empty(g) for g in geoms])
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
        qi, qt = tree.query_nearest(shapely.make_valid(geoms[rest]), all_matches=False)
        out[rest[qi]] = qt
    return out


def differences():
    """The published and the new parcels side by side (blklot, both codes as written and in the one spelling, block,
    geometry, table lot), the differing ones in connected areas (`area`, -1 where the codes agree)."""
    import geopandas as gpd

    old = gpd.read_file(assemble.out_dir(S) / "parcels.fgb")[["blklot", "code"]]
    new = gpd.read_file(NEW / "districts" / "parcels.fgb")[["blklot", "code", "geometry"]]
    df = new.merge(old, on="blklot", how="outer", suffixes=("_new", "_pub"))
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=new.crs)
    for k in ("new", "pub"):
        df[f"code_{k}"] = df[f"code_{k}"].where(df[f"code_{k}"].notna(), "")
        df[f"c_{k}"] = [canon(c) for c in df[f"code_{k}"]]
    df["block"] = assemble.blocks_of(df["blklot"].astype(str)).values
    df["host"] = hosts(df.geometry.to_numpy())
    d = (df["c_new"] != df["c_pub"]).to_numpy() & (df["host"] >= 0).to_numpy()
    df["area"] = -1
    sub = df[d].copy()
    sub["key"] = sub["c_pub"] + "|" + sub["c_new"]
    groups = sub[["host", "key"]].drop_duplicates().reset_index(drop=True)
    groups["gid"] = np.arange(len(groups))
    C = city()
    pairs = pd.DataFrame({"a": C.a, "b": C.b})
    e = pairs.merge(groups.rename(columns={"host": "a", "gid": "ga"}), on="a") \
             .merge(groups.rename(columns={"host": "b", "gid": "gb"}), on=["b", "key"])
    n = len(groups)
    _, comp = connected_components(coo_matrix((np.ones(len(e)), (e["ga"], e["gb"])), shape=(n, n)), directed=False)
    sub = sub.merge(groups, on=["host", "key"], how="left")
    df.loc[d, "area"] = comp[sub["gid"].to_numpy()]
    return df


def table_code(i: int, lots) -> str:
    """The table's code of table lot i: the scored table's (its by-eye corrections applied), else the table's own
    words (a split 'RH-1/RH-2', or none)."""
    C = city()
    return C.table[i] if C.known[i] else (str(lots["zoning"].iat[i]) or "none")


def areas(df) -> pd.DataFrame:
    """One row per area, largest first: lots, both codes (as the maps write them), the table's code (the commonest over
    its lots), how many of its lots the table backs each way, its blocks and its box (map ft)."""
    C = city()
    rows = []
    for a, g in df[df["area"] >= 0].groupby("area"):
        tab = [table_code(int(h), C.lots) for h in g["host"]]
        single = [t for t in tab if "/" not in t and t != "none"]
        backs_new = sum(canon(t) == g["c_new"].iat[0] for t in single)
        backs_pub = sum(canon(t) == g["c_pub"].iat[0] for t in single)
        verdict = ("new" if backs_new > len(single) / 2 else "published" if backs_pub > len(single) / 2
                   else "neither" if single else "none")
        x0, y0, x1, y1 = g.total_bounds
        rows.append({"area": int(a), "lots": len(g),
                     "published": Counter(g["code_pub"]).most_common(1)[0][0],
                     "new": Counter(g["code_new"]).most_common(1)[0][0],
                     "c_pub": g["c_pub"].iat[0], "c_new": g["c_new"].iat[0],
                     "table": Counter(tab).most_common(1)[0][0],
                     "table_codes": dict(Counter(tab).most_common()),
                     "table_coded": len(single), "table_backs_new": int(backs_new),
                     "table_backs_published": int(backs_pub), "verdict": verdict,
                     "blocks": sorted(set(g["block"].astype(str))),
                     "quarter": paths.QUARTERS[int(Counter(C.quarter[g["host"].to_numpy()]).most_common(1)[0][0])],
                     "box": [float(x0), float(y0), float(x1), float(y1)]})
    out = pd.DataFrame(rows).sort_values(["lots", "area"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# --- the page -----------------------------------------------------------------------------------

MARGIN = 60                      # grid px of scan around an area (render_incident.py's)
PER_CHUNK = 100                  # crops per file the page fetches
PAPER, INK = (245, 240, 229), (36, 33, 30)      # warp.PAPER_TINT, and the ink as the site's paper layer shows it
TINT, ALPHA = (31, 95, 191), 0.32               # the area's lots: the page's accent, see-through over the scan


def crop(ink, geoms):
    """The scan around an area (its lots' box plus MARGIN, one grid px per image px), its lots tinted and outlined,
    a scale bar at the bottom left; a palette PNG's bytes, the bar's length in ft, the crop's size."""
    import io

    import cv2
    import shapely
    from PIL import Image

    from sfzmaps.solver import grid

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
    strip = np.full((14, w), 6, np.uint8)                     # below the scan, so it never covers ink
    strip[5:9, 6:6 + min(bar, w - 12)] = 5
    idx = np.vstack([idx, strip])
    h = idx.shape[0]
    tint = [tuple(int(round(ALPHA * t + (1 - ALPHA) * p)) for t, p in zip(TINT, base)) for base in (PAPER, INK)]
    palette = [*PAPER, *INK, *tint[0], *tint[1], *TINT, 20, 20, 20, 255, 255, 255]
    im = Image.fromarray(idx, mode="P")
    im.putpalette(palette)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), bar_ft, (w, h)


def write_crops(df, A, page_dir: Path) -> dict:
    """Every area's crop, PER_CHUNK to a file: crops/chunk-NNN.json {rank: data URI}.
    Returns {rank: (bar_ft, w, h)}."""
    import base64

    from sfzmaps.solver import page

    ink = page.load_ink()
    lots = df[df["area"] >= 0]
    geoms = {a: g.geometry.to_numpy() for a, g in lots.groupby("area")}
    out, chunk = {}, {}
    d = page_dir / "crops"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
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
        single = [c for c in r.table_codes if "/" not in c and c != "none"]
        rows.append({"r": r.rank, "n": r.lots, "p": r.published, "w": r.new, "t": single[0] if single else r.table,
                     "tc": [[k, v] for k, v in list(r.table_codes.items())[:4]], "tx": max(len(r.table_codes) - 4, 0),
                     "k": r.table_coded, "bn": r.table_backs_new, "bp": r.table_backs_published, "v": r.verdict,
                     "b": blocks[:10], "bx": max(len(blocks) - 10, 0), "q": r.quarter,
                     "bar": crops[r.rank][0], "cw": crops[r.rank][1], "ch": crops[r.rank][2]})
    return rows


def write_page(A, crops, m, made, n_lots, page_dir: Path) -> Path:
    M = m["measures_every_lot"]
    ci = m["bootstrap_95"]["published - new"]
    summary = {
        "areas": len(A), "lots": n_lots, "readLots": made["read_lots"], "coded": made["read_lots_coded"],
        "unknown": len(made["unknown_codes"]), "confidence": made["confidence"],
        "rows": [
            ["Wrong lots", f"{M['published']['wrong_lots']:,}", f"{M['new']['wrong_lots']:,}",
             f"{M['published']['wrong_lots'] - M['new']['wrong_lots']:,} fewer",
             f"{ci['wrong_lots'][0]:,.0f} to {ci['wrong_lots'][1]:,.0f}"],
            ["Labels whose district comes out right", f"{M['published']['labels_right']:.1%}",
             f"{M['new']['labels_right']:.1%}",
             f"{(M['new']['labels_right'] - M['published']['labels_right']) * 100:.1f} pts more",
             f"{-ci['labels_right'][1] * 100:.1f} to {-ci['labels_right'][0] * 100:.1f} pts"],
            ["Wrong-lot incidents", f"{M['published']['incidents']:,}", f"{M['new']['incidents']:,}",
             f"{M['published']['incidents'] - M['new']['incidents']:,} fewer",
             f"{ci['incidents'][0]:,.0f} to {ci['incidents'][1]:,.0f}"]],
        "solve": [M["solve"]["wrong_lots"], M["solve"]["labels_right"], M["solve"]["incidents"]],
        "byArea": [M["published_by_area"]["wrong_lots"], M["new_by_area"]["wrong_lots"]],
        "noId": m["scored_table_lots_with_no_lot_of_their_id"],
        "verdicts": {k: int(v) for k, v in Counter(A["verdict"]).items()},
    }
    html = (Path(__file__).with_name("ship_differences.html").read_text()
            .replace("__SUMMARY__", json.dumps(summary).replace("</", "<\\/"))
            .replace("__ROWS__", json.dumps(page_rows(A, crops), separators=(",", ":")).replace("</", "<\\/")))
    out = page_dir / "index.html"
    out.write_text(html)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-dir", type=Path, default=OUT / "differences")
    ap.add_argument("--page-only", action="store_true", help="Rewrite the page from the saved areas and measures.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.page_dir.mkdir(parents=True, exist_ok=True)
    if args.page_only:
        saved = json.loads((OUT / "compare.json").read_text())
        made, m = saved["new_map"], saved
        import geopandas as gpd
        df, A = gpd.read_parquet(OUT / "lots.parquet"), pd.DataFrame(json.loads((OUT / "areas.json").read_text()))
    else:
        made = new_map()
        print(json.dumps({k: v for k, v in made.items() if k != "codes"}))
        m, _ = measures({"new": NEW / "districts_raw.parquet"})
        df = differences()
        A = areas(df)
        A.drop(columns=["table_codes", "blocks", "box"]).to_csv(OUT / "areas.csv", index=False)
        n_diff = int((df["area"] >= 0).sum())
        (OUT / "compare.json").write_text(json.dumps({"new_map": made, **m, "differing_lots": n_diff, "areas": len(A)},
                                                     indent=1, default=float) + "\n")
        df.to_parquet(OUT / "lots.parquet")
        A.to_json(OUT / "areas.json", orient="records", indent=1)
        for k, lab, fmt in KEYS:
            print(f"{lab:48}" + " ".join(f"{fmt.format(m['measures_every_lot'][r][k]):>12}" for r in
                                         ("published", "new", "solve", "published_by_area", "new_by_area")))
        print(json.dumps(m["bootstrap_95"]), m["scored_table_lots_with_no_lot_of_their_id"])
        print(f"areas {len(A):,}, differing lots {int((df['area'] >= 0).sum()):,}")
    crops = write_crops(df, A, args.page_dir)
    print(write_page(A, crops, m, made, int((df["area"] >= 0).sum()), args.page_dir))


if __name__ == "__main__":
    main()
