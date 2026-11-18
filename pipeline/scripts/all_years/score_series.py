"""Score a series' solve ($0, reads only): against its published map, the site's corrections and, for the 1996-2000
scan, the April 2000 table; and picture the largest areas where the solve and the published map differ.

    uv run python scripts/all_years/score_series.py SERIES [--crops N]

Runs on the 2000 map's paths (no --series): the series' solve is read from its folder
(data/solver/experiments/all-years/<series>/solve/runs/final.npz); every series is solved on the April 2000 table's
lots, so the 2000 map's lots (data/solver/lots) are the solve's lots, lot for lot.

The published map is the site's (data/out/<series>/districts/parcels.fgb: the reader's codes with the overrides and
parks applied), reached by lot ID, as `solver.published.save_published` reaches the 2000 map's.  Writes, into the
series' folder: score.json, differences.csv (the differing areas, largest first) and crops/<rank>_<what>.png."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pyogrio
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from sfzmaps import config
from sfzmaps.solver import grid, paths
from sfzmaps.solver.city import city, load_run, run_file
from sfzmaps.solver.codes import canon

KEYS = ("wrong_lots", "labels_right", "incidents", "lines_drawn", "drawn_are_table_lines", "holes")


def published_codes(series: str, blklots: np.ndarray) -> np.ndarray:
    """The site's code of each lot (by lot ID; "" for a lot it does not list or leaves blank), in the one spelling.
    For a height map, the period's published heights."""
    if series.endswith("-height"):
        import pandas as pd
        from sfzmaps.districts.series import SERIES
        d = pd.read_parquet(config.OUT_DIR / f"{SERIES[series].period}-zoning" / "districts" / "height.parquet",
                            columns=["blklot", "height_code"]).rename(columns={"height_code": "code"})
    else:
        d = pyogrio.read_dataframe(config.OUT_DIR / series / "districts" / "parcels.fgb", columns=["blklot", "code"],
                                   read_geometry=False)
    code = dict(zip(d["blklot"].astype(str), d["code"].fillna("").astype(str)))
    return np.array([canon(code[b]) if code.get(b, "") else "" for b in blklots])


def raw_codes(series: str, blklots: np.ndarray) -> np.ndarray:
    import pandas as pd
    if series.endswith("-height"):
        return published_codes(series, blklots)
    d = pd.read_parquet(config.WORK_DIR / series / "districts_raw.parquet", columns=["blklot", "code"])
    code = dict(zip(d["blklot"].astype(str), d["code"].fillna("").astype(str)))
    return np.array([canon(code[b]) if code.get(b, "") else "" for b in blklots])


def overrides_agreement(series: str, blklots: np.ndarray, solve: np.ndarray) -> dict:
    f = config.INPUTS_DIR / series / "overrides.csv"
    if not f.is_file():
        return {"rows": 0}
    row = {b: i for i, b in enumerate(blklots)}
    n = agree = missing = 0
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            i = row.get(str(r["blklot"]))
            if i is None or not r.get("code"):
                missing += 1
                continue
            n += 1
            agree += canon(r["code"]) == solve[i]
    return {"rows": n + missing, "on_solve_lots": n, "agree": agree, "share": round(agree / n, 4) if n else None}


def label_fix_agreement(series: str, solve: np.ndarray, grown: np.ndarray) -> dict:
    """Each hand label fix with a code (fix_code, not a deletion): the code the solve gives the lot under the label's
    point (through the sheet's corrected warp) against the fix's."""
    from sfzmaps.transform import SheetTransform
    f = config.INPUTS_DIR / series / "label_fixes.csv"
    if not f.is_file():
        return {"rows": 0}
    out = []
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            code = (r.get("fix_code") or "").strip()
            if not code or code == "-":
                continue
            folder = config.DATA_DIR / "warp" / series / r["sheet"]
            if not (folder / "transform.json").is_file():
                continue
            tr = SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)
            xy = tr.to_world(np.array([[float(r["x"]), float(r["y"])]]))
            c, rr = np.floor(grid.to_grid(xy)).astype(int)[0]
            if 0 <= rr < grown.shape[0] and 0 <= c < grown.shape[1]:
                out.append((canon(code), solve[grown[rr, c]]))
    agree = sum(a == b for a, b in out)
    return {"rows": len(out), "agree": agree, "share": round(agree / len(out), 4) if out else None,
            "disagree": [f"{a} -> solve {b}" for a, b in out if a != b][:20]}


def own_labels(folder: Path, pub: np.ndarray, maps: dict) -> dict:
    """The series' own printed labels as a check both maps take alike: each print not on street ground whose read
    (Sol's, else Terra's) is a code of the published map, and the share of them whose lot under the text has that
    code in each map.  The solve read these labels, so it is favoured; the published map was read from the same
    sheets by another reader.  A large shortfall still shows a solve gone wrong."""
    import pandas as pd
    f = folder / "prints" / "prints.parquet"
    if not f.is_file():
        return {}
    p = pd.read_parquet(f)
    read = np.where(p["sol_code"].astype(str) != "", p["sol_code"].astype(str), p["terra_code"].astype(str))
    vocab = set(pub) - {""}
    lot = p["lab_lot"].to_numpy(np.int64)
    keep = (lot >= 0) & ~p["street"].to_numpy(bool) & np.array([r in vocab for r in read])
    out = {"labels": int(keep.sum())}
    for name, codes in maps.items():
        out[name] = round(float((codes[lot[keep]] == read[keep]).mean()), 4) if keep.any() else None
    return out


def differing_areas(C, solve, pub, area):
    """Connected areas (lots joined where they touch) whose solve code and published code are both given and differ,
    largest by area first: (lots, area sq ft, published majority, solve majority)."""
    diff = (solve != "") & (pub != "") & (solve != pub)
    a, b = C.a, C.b
    touch = np.load(paths.LOTS_DIR / "lots.npz")["pairs"][:, 2] > 0
    m = touch & diff[a] & diff[b]
    _, comp = connected_components(coo_matrix((np.ones(int(m.sum())), (a[m], b[m])), shape=(C.n, C.n)),
                                   directed=False)
    ids = np.flatnonzero(diff)
    groups = {}
    for i in ids:
        groups.setdefault(comp[i], []).append(i)
    rows = []
    for lots_ in groups.values():
        lots_ = np.array(lots_)
        pv, pc = np.unique(pub[lots_], return_counts=True)
        sv, sc = np.unique(solve[lots_], return_counts=True)
        rows.append((lots_, float(area[lots_].sum()), pv[np.argmax(pc)], sv[np.argmax(sc)]))
    rows.sort(key=lambda r: -r[1])
    return rows


def crop_picture(series, lots_, C, solve, pub, out: Path) -> None:
    """The series' merged page (its printed ink) around an area, the area's lots outlined in red, other lots that
    differ in orange; the published and solved codes written above."""
    ink = cv2.imread(str(paths.solver_dir(series) / "page" / "ink.png"), cv2.IMREAD_GRAYSCALE)
    lot = np.load(paths.LOTS_DIR / "lots.npz")["lot"]
    geoms = C.lots.geometry.values
    xy = shapely.get_coordinates(shapely.envelope(shapely.union_all(geoms[lots_])))
    cr = grid.to_grid(xy)
    pad = 60
    c0, c1 = max(int(cr[:, 0].min()) - pad, 0), min(int(cr[:, 0].max()) + pad, ink.shape[1])
    r0, r1 = max(int(cr[:, 1].min()) - pad, 0), min(int(cr[:, 1].max()) + pad, ink.shape[0])
    big = max(c1 - c0, r1 - r0)
    if big > 1600:                                   # keep a picture readable: the area's centre, 1600 px across
        cx, cy = (c0 + c1) // 2, (r0 + r1) // 2
        c0, c1, r0, r1 = max(cx - 800, 0), min(cx + 800, ink.shape[1]), max(cy - 800, 0), min(cy + 800, ink.shape[0])
    view = cv2.cvtColor(ink[r0:r1, c0:c1], cv2.COLOR_GRAY2BGR)
    sub = lot[r0:r1, c0:c1]
    differ = (solve != pub) & (solve != "") & (pub != "")
    tint = view.copy()
    inside = np.isin(sub, lots_)
    other = (sub >= 0) & ~inside & differ[np.maximum(sub, 0)]
    tint[inside] = (0.6 * tint[inside] + 0.4 * np.array([60, 60, 255])).astype(np.uint8)
    tint[other] = (0.7 * tint[other] + 0.3 * np.array([0, 165, 255])).astype(np.uint8)
    edge = cv2.morphologyEx(inside.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    tint[edge] = (0, 0, 255)
    scale = max(1, 900 // max(tint.shape[:2]))
    if scale > 1:
        tint = cv2.resize(tint, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    head = np.full((60, tint.shape[1], 3), 255, np.uint8)
    pv = ", ".join(f"{c} {n}" for c, n in zip(*np.unique(pub[lots_], return_counts=True)))
    sv = ", ".join(f"{c} {n}" for c, n in zip(*np.unique(solve[lots_], return_counts=True)))
    cv2.putText(head, f"published: {pv}"[:110], (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(head, f"solve: {sv}"[:110], (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1)
    cv2.imwrite(str(out), np.vstack([head, tint]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series")
    ap.add_argument("--crops", type=int, default=8)
    args = ap.parse_args()
    series = args.series
    folder = paths.solver_dir(series)
    C = city()
    solve = load_run("final", folder / "solve")
    if len(solve) != C.n:
        raise ValueError(f"{run_file('final', folder=folder / 'solve')} has {len(solve)} lots, the table {C.n}")
    blklots = C.lots["blklot"].astype(str).values
    area = shapely.area(C.lots.geometry.values)
    pub = published_codes(series, blklots)
    raw = raw_codes(series, blklots)
    both = (solve != "") & (pub != "")
    out = {"series": series, "lots": int(C.n), "solve_coded": int((solve != "").sum()),
           "published_coded": int((pub != "").sum()), "both": int(both.sum()),
           "agree_share_lots": round(float((solve[both] == pub[both]).mean()), 4) if both.any() else None,
           "agree_share_area": round(float(area[both & (solve == pub)].sum() / area[both].sum()), 4)
           if both.any() else None,
           "agree_share_lots_raw_read": round(float((solve[both & (raw != "")] == raw[both & (raw != "")]).mean()), 4)
           if (both & (raw != "")).any() else None}
    out["overrides"] = overrides_agreement(series, blklots, solve)
    out["overrides_published"] = overrides_agreement(series, blklots, pub)
    grown = np.load(paths.LOTS_DIR / "lots.npz")["grown"]
    out["label_fixes"] = label_fix_agreement(series, solve, grown)
    out["label_fixes_published"] = label_fix_agreement(series, pub, grown)
    out["own_labels_right"] = own_labels(folder, pub, {"solve": solve, "published": pub})
    if series in paths.TABLE_SCORED_SERIES:         # the 1996-2000 scan: the April 2000 table, as the twin's score
        from sfzmaps.solver.report import six_measures
        nothing = np.zeros(C.n, bool)
        for key, codes in (("table_solve", solve), ("table_published_raw_read", raw), ("table_published_site", pub)):
            m = six_measures(codes, nothing)
            out[key] = {k: m[k] for k in KEYS}
    areas = differing_areas(C, solve, pub, area)
    out["differing_lots"] = int(((solve != pub) & both).sum())
    out["differing_areas"] = len(areas)
    crops = folder / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    with open(folder / "differences.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "lots", "area_sqft", "published", "solve", "x", "y", "blklots"])
        for k, (lots_, ar, pv, sv) in enumerate(areas[:200], 1):
            c = C.centroid[lots_].mean(0)
            w.writerow([k, len(lots_), round(ar), pv, sv, round(c[0]), round(c[1]), " ".join(blklots[lots_[:30]])])
    out["largest_areas"] = [{"lots": len(lots_), "area_acres": round(ar / 43560, 1), "published": pv, "solve": sv,
                             "x": round(float(C.centroid[lots_, 0].mean())),
                             "y": round(float(C.centroid[lots_, 1].mean()))}
                            for lots_, ar, pv, sv in areas[:10]]
    for k, (lots_, ar, pv, sv) in enumerate(areas[:args.crops], 1):
        safe = lambda s: "".join(ch if ch.isalnum() or ch in "-" else "_" for ch in s)   # noqa: E731
        crop_picture(series, lots_, C, solve, pub,
                     crops / f"{k:02d}_published-{safe(pv)}_solve-{safe(sv)}_{len(lots_)}lots.png")
    (folder / "score.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
