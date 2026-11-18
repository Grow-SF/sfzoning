"""The owner's review page of every series' solve against its published map ($0, reads only): one page with a series
picker, each series' scores and its largest differing areas, each with a crop of the series' own printed page.

    uv run python scripts/all_years/review_page.py [--areas 150] [--px 420]

Runs on the 2000 map's paths (no --series); reads each series' solve from data/solver/<series>/ (the stages' layout).
Writes data/solver/experiments/all-years/review/: index.html, and data/<series>.json per series (its scores and areas,
the crops as JPEG data URIs), fetched by the page when the series is picked."""
import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import shapely

from sfzmaps import config
from sfzmaps.districts import solved
from sfzmaps.districts.series import SERIES as INFO
from sfzmaps.solver import paths
from sfzmaps.solver.city import city, load_run

sys.path.insert(0, str(Path(__file__).parent))
import score_series as sc  # noqa: E402

OUT = config.DATA_DIR / "solver" / "experiments" / "all-years" / "review"
JPEG_QUALITY = 50        # the crops: about 60 MB for every series at 150 areas each
KEYS = ("wrong_lots", "labels_right", "incidents", "lines_drawn", "drawn_are_table_lines", "holes")


def crop(ink, lot, lots_, differ, box, px):
    """The series' page around an area (grid box r0, r1, c0, c1): its lots tinted red and outlined, other differing
    lots tinted amber; scaled to `px` on its long side.  JPEG bytes."""
    r0, r1, c0, c1 = box
    view = cv2.cvtColor(ink[r0:r1, c0:c1], cv2.COLOR_GRAY2BGR)
    sub = lot[r0:r1, c0:c1]
    inside = np.isin(sub, lots_)
    other = (sub >= 0) & ~inside & differ[np.maximum(sub, 0)]
    view[inside] = (0.62 * view[inside] + 0.38 * np.array([70, 70, 235])).astype(np.uint8)
    view[other] = (0.72 * view[other] + 0.28 * np.array([40, 170, 240])).astype(np.uint8)
    edge = cv2.morphologyEx(inside.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    view[edge] = (20, 20, 200)
    s = px / max(view.shape[:2])
    if s < 1:
        view = cv2.resize(view, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes()


def area_box(C, lots_, shape, pad=50, cap=900):
    xy = shapely.get_coordinates(shapely.envelope(shapely.union_all(C.lots.geometry.values[lots_])))
    cr = np.array([[(x - 5979300) / 4, (2124100 - y) / 4] for x, y in xy])
    c0, c1 = int(cr[:, 0].min()) - pad, int(cr[:, 0].max()) + pad
    r0, r1 = int(cr[:, 1].min()) - pad, int(cr[:, 1].max()) + pad
    if max(c1 - c0, r1 - r0) > cap:                  # a big area: its centre, cap px across
        cx, cy = (c0 + c1) // 2, (r0 + r1) // 2
        c0, c1, r0, r1 = cx - cap // 2, cx + cap // 2, cy - cap // 2, cy + cap // 2
    return max(r0, 0), min(r1, shape[0]), max(c0, 0), min(c1, shape[1])


def series_data(s, C, n_areas, px, to_ll, reference=None):
    folder = paths.solver_dir(s)
    run = folder / "solve" / "runs" / "final.npz"
    if not run.is_file():
        return None
    solve = load_run("final", folder / "solve")
    with np.load(run) as z:
        post = z["posterior"] if "posterior" in z.files else np.full(len(solve), np.nan)
    blk = C.lots["blklot"].astype(str).values
    area = shapely.area(C.lots.geometry.values)
    pub = sc.published_codes(s, blk, (reference or {}).get("use"))
    both = (solve != "") & (pub != "")
    out = {"series": s, "state": "PUBLISHED-FROM-SOLVE" if s in solved.SOLVED else "NEEDS-OWNER-LOOK",
           "role": INFO[s].role, "pattern": s in paths.PATTERN_SERIES, "lots_compared": int(both.sum()),
           "agree_lots": round(float((solve[both] == pub[both]).mean()), 4),
           "agree_area": round(float(area[both & (solve == pub)].sum() / area[both].sum()), 4),
           "own_labels": sc.own_labels(folder, pub, {"solve": solve, "published": pub}),
           "overrides": sc.overrides_agreement(s, blk, solve),
           "label_fixes": sc.label_fix_agreement(s, solve, np.load(paths.LOTS_DIR / "lots.npz")["grown"]),
           "label_fixes_published": sc.label_fix_agreement(s, pub, np.load(paths.LOTS_DIR / "lots.npz")["grown"])}
    ev = folder / "walls" / "evaluation.json"
    out["width_scale"] = json.loads(ev.read_text()).get("width_scale", 1.0) if ev.is_file() else None
    tr = folder / "labels" / "trust.json"
    if tr.is_file():
        t = json.loads(tr.read_text())
        out["prints"], out["reads"] = t.get("prints"), t.get("read_set")
    table = None
    if s in paths.TABLE_SCORED_SERIES:
        from sfzmaps.solver.report import six_measures
        nothing = np.zeros(C.n, bool)
        out["table"] = {name: {k: six_measures(codes, nothing)[k] for k in KEYS}
                        for name, codes in (("solve", solve), ("published", pub))}
        table = C.table
    areas = sc.differing_areas(C, solve, pub, area)
    out["differing_lots"] = int(((solve != pub) & both).sum())
    out["differing_areas"] = len(areas)
    ink = cv2.imread(str(folder / "page" / "ink.png"), cv2.IMREAD_GRAYSCALE)
    lot = C.lot_raster
    differ = (solve != pub) & both
    items = []
    for k, (lots_, ar, pv, sv) in enumerate(areas[:n_areas], 1):
        cx, cy = C.centroid[lots_].mean(0)
        lon, lat = to_ll(cx, cy)
        jpg = crop(ink, lot, lots_, differ, area_box(C, lots_, ink.shape), px)
        p = post[lots_]
        it = {"rank": k, "lots": len(lots_), "acres": round(ar / 43560, 2),
              "old": [[c, int(n)] for c, n in zip(*np.unique(pub[lots_], return_counts=True))],
              "new": [[c, int(n)] for c, n in zip(*np.unique(solve[lots_], return_counts=True))],
              # the 2000 solve keeps no posterior (it is scored on the table): JSON has no NaN
              "posterior": round(float(np.nanmean(p)), 3) if np.isfinite(p).any() else None,
              "lat": round(lat, 5), "lon": round(lon, 5),
              "blocks": sorted({b[:4] for b in blk[lots_]})[:6],
              "img": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}
        if table is not None:
            it["table"] = [[c, int(n)] for c, n in zip(*np.unique(table[lots_], return_counts=True)) if c]
        items.append(it)
    out["areas"] = items
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", type=int, default=150)
    ap.add_argument("--px", type=int, default=320)
    ap.add_argument("--series", nargs="*")
    ap.add_argument("--solve-root", type=Path, help="the folder holding each series' solver folder (default data/solver; the build's, read only)")
    ap.add_argument("--out", type=Path, default=OUT, help="where the page goes")
    ap.add_argument("--old-maps", type=Path, default=OUT.parent / "old-maps.json",
                    help="verify_old.py's check that each published map is the one main's dvc.lock records")
    a = ap.parse_args()
    if a.solve_root:
        paths.SOLVER_ROOT = a.solve_root.resolve()
    out_dir = a.out
    old = json.loads(a.old_maps.read_text()) if a.old_maps.is_file() else {}
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:2227", "EPSG:4326", always_xy=True)
    C = city()
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    index_file = out_dir / "series.json"
    index = json.loads(index_file.read_text()) if index_file.is_file() else {}
    for s in a.series or (paths.SERIES_2000, *paths.ALL_YEARS):
        d = series_data(s, C, a.areas, a.px, tr.transform, old.get(s))
        if d is not None:
            d["reference"] = {k: v for k, v in (old.get(s) or {}).items() if k != "use"} or None   # no local paths
        if d is None:
            print(s, "no solve yet")
            continue
        f = out_dir / "data" / f"{s}.json"
        f.write_text(json.dumps(d, separators=(",", ":")))
        index[s] = {k: v for k, v in d.items() if k != "areas"}
        print(s, d["state"], f"agree {d['agree_lots']:.1%}", f"own labels {d['own_labels']}",
              f"{f.stat().st_size / 1e6:.1f} MB")
    index_file.write_text(json.dumps(index, indent=1) + "\n")
    order = [s for s in (paths.SERIES_2000, *paths.ALL_YEARS) if s in index]
    html = (Path(__file__).parent / "review_template.html").read_text()
    html = html.replace("/*SERIES_INDEX*/null", json.dumps({"order": order, "series": index}, separators=(",", ":")))
    (out_dir / "index.html").write_text(html)
    total = sum(f.stat().st_size for f in (out_dir / "data").glob("*.json"))
    print(f"wrote {out_dir / 'index.html'}; data {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
