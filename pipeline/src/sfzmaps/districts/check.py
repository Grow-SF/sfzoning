"""A $0 accuracy check of the district reads (`sfzmaps districts --check`).

Reads only the saved raw reads (work/<series>/districts_raw.parquet, or <dir>/<series>_raw.parquet
for a snapshot or an experiment) and prints:

- review cases (inputs/district_cases.csv): per case, lots in its bbox carrying the known wrong
  code and lots carrying a right one.  Kinds: seam (a seam artifact), harm (an area a past rule
  change broke; must stay right), other (a known error not yet fixed);
- answer keys (tests/fixtures/district_keys): hand-read lots right;
- agreement between maps drawn on the same base: 1960/1964, and 1996-2000 p25-p36 vs 2000-digital
  p01-p12 (two scans of the same twelve drawings; lots read on the matching sheet in both).  It
  needs no key, so it is the cleanest measure of how repeatable the reader is;
- agreement between adjacent eras (real rezonings lower it too: compare, don't maximise);
- the largest clusters of lots where the same-base maps disagree: where to look for new cases.

city_map draws a series before | after (lots by zoning class, sheet footprints outlined): large
regressions that no case covers show up there (a split rule once turned ~1,100 Forest Hill lots
RH-2, caught only by eye).
"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from sfzmaps import config
from sfzmaps.districts.validate import _norm, key_report

CASES_CSV = config.INPUTS_DIR / "district_cases.csv"
KEYS_DIR = config.PIPELINE_DIR / "tests" / "fixtures" / "district_keys"
WORK = "work"

# (a, b, sheet pairs): with sheet pairs, only lots read on a sheet in a and its partner in b count
SAME_BASE: list[tuple[str, str, dict[str, str] | None]] = [
    ("1960", "1964", None),
    ("1996-2000-zoning", "2000-digital-zoning",
     {f"1996-2000-zoning-p{25 + i}": f"2000-digital-zoning-p{1 + i:02d}" for i in range(12)}),
]
CLUSTER_CELL_FT = 500.0   # disagreeing lots in touching cells of this size form one cluster
CLUSTER_MIN_LOTS = 20


def use_series() -> list[str]:
    from sfzmaps.districts.series import SERIES

    return [s for s, i in SERIES.items() if i.role == "use"]


def raw_path(source: str | Path, series: str) -> Path:
    if str(source) == WORK:
        return config.WORK_DIR / series / "districts_raw.parquet"
    return Path(source) / f"{series}_raw.parquet"


def loader(source: str | Path) -> Callable[[str], pd.DataFrame | None]:
    """series -> its raw reads (blklot, code normalised, source_sheet), None if there are none."""
    cache: dict[str, pd.DataFrame | None] = {}

    def get(series: str) -> pd.DataFrame | None:
        if series not in cache:
            p = raw_path(source, series)
            if not p.exists():
                cache[series] = None
            else:
                r = pd.read_parquet(p)
                if "source_sheet" not in r:
                    r["source_sheet"] = None
                cache[series] = r[["blklot", "code", "source_sheet"]].assign(code=r["code"].map(_norm))
        return cache[series]
    return get


def load_cases(path: Path | str = CASES_CSV) -> pd.DataFrame:
    c = pd.read_csv(path, dtype={"series": str}, keep_default_na=False)
    c["right"] = c["right"].map(lambda v: {_norm(x) for x in v.split(";") if x.strip()})
    c["wrong"] = c["wrong"].map(_norm)
    return c


def centroids() -> pd.DataFrame:
    from sfzmaps.districts.read import load_parcels

    p = load_parcels()
    c = p.geometry.centroid
    return pd.DataFrame({"blklot": p["blklot"].to_numpy(), "x": c.x.to_numpy(), "y": c.y.to_numpy()})


def score_cases(cases: pd.DataFrame, get, cent: pd.DataFrame) -> pd.DataFrame:
    """Per case: n lots read in the bbox, wrong (the known wrong code), right (a right code)."""
    rows = []
    for c in cases.itertuples():
        r = get(c.series)
        lots = cent[(cent.x >= c.x0) & (cent.x <= c.x1) & (cent.y >= c.y0) & (cent.y <= c.y1)]
        if r is None:
            rows.append((c.id, c.kind, None, None, None))
            continue
        m = lots.merge(r, on="blklot")
        rows.append((c.id, c.kind, len(m), int((m.code == c.wrong).sum()), int(m.code.isin(c.right).sum())))
    return pd.DataFrame(rows, columns=["id", "kind", "n", "wrong", "right"])


def pair(a: pd.DataFrame, b: pd.DataFrame, sheets: dict[str, str] | None = None) -> pd.DataFrame:
    """Lots coded in both (columns blklot, code_a, code_b), on matching sheets if `sheets`."""
    m = a.merge(b, on="blklot", suffixes=("_a", "_b")).dropna(subset=["code_a", "code_b"])
    if sheets is not None:
        m = m[m["source_sheet_a"].map(sheets) == m["source_sheet_b"]]
    return m


def agreement(a: pd.DataFrame, b: pd.DataFrame, sheets: dict[str, str] | None = None) -> tuple[float, int]:
    m = pair(a, b, sheets)
    return (float((m.code_a == m.code_b).mean()) if len(m) else float("nan")), len(m)


def key_scores(get) -> dict[str, tuple[int, int]]:
    out = {}
    for f in sorted(KEYS_DIR.glob("*.csv")):
        r = get(f.stem)
        if r is not None:
            k = key_report(r, f)
            out[f.stem] = (int(k.ok.sum()), len(k))
    return out


def clusters(a: pd.DataFrame, b: pd.DataFrame, cent: pd.DataFrame, sheets: dict[str, str] | None = None,
             cell: float = CLUSTER_CELL_FT, min_lots: int = CLUSTER_MIN_LOTS) -> pd.DataFrame:
    """Spatial clusters of lots where a and b disagree, per (code_a, code_b), largest first:
    code_a, code_b, n, x0, y0, x1, y1 (ft)."""
    from scipy import ndimage

    m = pair(a, b, sheets)
    m = m[m.code_a != m.code_b].merge(cent, on="blklot")
    rows = []
    for (ca, cb), g in m.groupby(["code_a", "code_b"]):
        if len(g) < min_lots:
            continue
        ix = ((g.x - g.x.min()) // cell).astype(int).to_numpy()
        iy = ((g.y - g.y.min()) // cell).astype(int).to_numpy()
        grid = np.zeros((iy.max() + 1, ix.max() + 1), bool)
        grid[iy, ix] = True
        lab, _ = ndimage.label(grid, structure=np.ones((3, 3)))
        for k, h in g.groupby(lab[iy, ix]):
            if len(h) >= min_lots:
                rows.append((ca, cb, len(h), round(h.x.min()), round(h.y.min()), round(h.x.max()), round(h.y.max())))
    out = pd.DataFrame(rows, columns=["code_a", "code_b", "n", "x0", "y0", "x1", "y1"])
    return out.sort_values("n", ascending=False, ignore_index=True)


def snapshot(dest: Path | str) -> list[Path]:
    """Copy every use series' current raw reads to dest/<series>_raw.parquet (a 'before')."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for s in use_series():
        p = raw_path(WORK, s)
        if p.exists():
            out.append(Path(shutil.copy2(p, dest / f"{s}_raw.parquet")))
    return out


def report(sources: dict[str, str | Path], n_clusters: int = 5) -> str:
    """The check as text, one column per source (the first is the reference for clusters)."""
    names = list(sources)
    gets = {n: loader(s) for n, s in sources.items()}
    cent = centroids()
    w = max(19, *(len(n) + 2 for n in names))
    head = lambda t: f"{t:34}" + "".join(f"{n:>{w}}" for n in names)  # noqa: E731
    lines = ["Review cases (wrong / right of n lots in the bbox)", head("case")]
    cases = load_cases()
    scored = {n: score_cases(cases, g, cent) for n, g in gets.items()}
    for i, c in enumerate(cases.itertuples()):
        cells = []
        for n in names:
            r = scored[n].iloc[i]
            cells.append(f"{'-':>{w}}" if pd.isna(r.n) else f"{f'{r.wrong:.0f}/{r.right:.0f} of {r.n:.0f}':>{w}}")
        lines.append(f"{c.id[:26]:26} {c.kind:7}" + "".join(cells))
    for kind in ("seam", "harm", "other"):
        cells = []
        for n in names:
            s = scored[n][scored[n].kind == kind]
            cells.append(f"{f'{s.wrong.sum():.0f}/{s.right.sum():.0f}':>{w}}")
        lines.append(f"{'TOTAL ' + kind:34}" + "".join(cells))

    lines += ["", "Answer keys (lots right)", head("series")]
    keys = {n: key_scores(g) for n, g in gets.items()}
    for s in sorted({k for v in keys.values() for k in v}):
        lines.append(f"{s:34}" + "".join(
            f"{(f'{keys[n][s][0]}/{keys[n][s][1]}' if s in keys[n] else '-'):>{w}}" for n in names))

    def agree_rows(pairs):
        for a, b, sheets in pairs:
            cells = []
            for n in names:
                ra, rb = gets[n](a), gets[n](b)
                cells.append(f"{'-':>{w}}" if ra is None or rb is None
                             else f"{agreement(ra, rb, sheets)[0]:>{w}.3f}")
            lines.append(f"{a + ' ~ ' + b:34}" + "".join(cells))

    lines += ["", "Same base drawing (share of lots coded alike; higher is better)", head("pair")]
    agree_rows(SAME_BASE)
    lines += ["", "Adjacent eras (rezonings lower it too)", head("pair")]
    us = use_series()
    agree_rows([(a, b, None) for a, b in zip(us, us[1:])])

    ref = gets[names[0]]
    lines += ["", f"Largest same-base disagreements in {names[0]} (candidate cases)"]
    for a, b, sheets in SAME_BASE:
        ra, rb = ref(a), ref(b)
        if ra is None or rb is None:
            continue
        cl = clusters(ra, rb, cent, sheets).head(n_clusters)
        for c in cl.itertuples():
            lines.append(f"  {a} {c.code_a} | {b} {c.code_b}: {c.n} lots  bbox {c.x0} {c.y0} {c.x1} {c.y1}")
    return "\n".join(lines)


# --- city map ---------------------------------------------------------------------------

CITY_BBOX = (5979000.0, 2084000.0, 6020000.0, 2124000.0)   # ft, EPSG:2227
MAP_COLOURS = {"P": (153, 153, 153), "IND": (139, 69, 19), "COM": (31, 119, 180),
               "R1": (238, 136, 136), "R2": (255, 153, 0), "R3+": (34, 170, 34)}   # RGB


def map_class(code) -> str | None:
    """Coarse class for the city map: P, IND, COM, R1 (single family), R2, R3+ (denser)."""
    c = _norm(code)
    if c is None:
        return None
    if c == "P" or c.startswith("P-") or c == "OS":
        return "P"
    if c.startswith(("M-", "PDR", "SLI", "SSO", "SPD", "SALI", "UMU")) or c == "C-M":
        return "IND"
    if c.startswith(("C", "NC", "RC", "RED", "SLR", "MUG", "MUO", "MUR")):
        return "COM"
    if c.startswith(("R-1", "RH-1", "1R")):
        return "R1"
    if c.startswith(("R-2", "RH-2", "2R")):
        return "R2"
    return "R3+"


def city_map(series: str, panels: dict[str, str | Path], out: Path | str, ft_per_px: float = 20.0,
             bbox: tuple[float, float, float, float] = CITY_BBOX) -> Path:
    """PNG with one panel per source (e.g. before | after): lots by map_class, sheet
    footprints outlined in black, uncoded lots white."""
    import cv2

    from sfzmaps import stitch
    from sfzmaps.districts import sampling
    from sfzmaps.districts.read import load_parcels

    x0, y0, x1, y1 = bbox
    W, H = int((x1 - x0) / ft_per_px), int((y1 - y0) / ft_per_px)

    def px(coords):
        a = np.asarray(coords)[:, :2]
        return np.int32(np.c_[(a[:, 0] - x0) / ft_per_px, (y1 - a[:, 1]) / ft_per_px])

    parcels = load_parcels()[["blklot", "geometry"]]
    fps = []
    for sid in sampling.sheet_order(series):
        try:
            fp = stitch.footprint(series, sid)
        except Exception:  # noqa: BLE001 - a sheet without a footprint is just not outlined
            continue
        fps += [px(q.exterior.coords) for q in getattr(fp, "geoms", [fp]) if q.geom_type == "Polygon"]
    imgs = []
    for name, src in panels.items():
        img = np.full((H, W, 3), 255, np.uint8)
        r = loader(src)(series)
        if r is not None:
            g = parcels.merge(r, on="blklot")
            for cls, rgb in MAP_COLOURS.items():
                polys = [px(q.exterior.coords) for geom in g.geometry[g.code.map(map_class) == cls]
                         if geom is not None for q in getattr(geom, "geoms", [geom])]
                cv2.fillPoly(img, polys, rgb[::-1])
        cv2.polylines(img, fps, True, (0, 0, 0), 1)
        cv2.putText(img, f"{series} {name}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        imgs.append(img)
    sep = np.full((H, 8, 3), 128, np.uint8)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack([x for im in imgs for x in (im, sep)][:-1]))
    return out
