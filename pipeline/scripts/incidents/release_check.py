"""Release checks by eye: a window of a series' published zoning over its own scan, before and after a rebuild.

    uv run python scripts/incidents/release_check.py NAME --box X0 Y0 X1 Y1 --scan SERIES \
        --map "before=PATH/parcels.fgb" --map "after=PATH/parcels.fgb" [--pad FT]

The backdrop is the series' warped sheets (data/work/<SERIES>/<sheet>/warped.tif, EPSG:2227) laid over white paper,
read at the sheets' own 4 ft a px and drawn 2x larger (each scan px a 2 x 2 block, no smoothing), the sheets as the
old pipeline warped them, so their seams show where they are.  Each map is a panel: its lots tinted by their
category's colour (`assemble.COLORS`), lot lines thin, and each code written once on every connected patch of its
lots in the window.  The box is in EPSG:2227 ft.  Writes data/solver/experiments/ship/release-checks/NAME.png; reads
only otherwise.
"""
import argparse

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from shapely.geometry import box as shapely_box

from sfzmaps import config
from sfzmaps.districts.assemble import COLORS
from sfzmaps.solver import runs

OUT = runs.EXPERIMENTS_DIR / "ship" / "release-checks"
FT_PER_PX = 4.0         # the warped sheets' own resolution
ZOOM = 2                # each scan px drawn as ZOOM x ZOOM
ALPHA = 0.42            # the category tint
MIN_LABEL_FT2 = 40_000  # a patch smaller than this is not labelled (too small to read)


def backdrop(series: str, bounds) -> np.ndarray:
    """The series' warped sheets inside `bounds` (x0, y0, x1, y1 ft) over white, RGB uint8, north up, at FT_PER_PX."""
    x0, y0, x1, y1 = bounds
    h, w = int(round((y1 - y0) / FT_PER_PX)), int(round((x1 - x0) / FT_PER_PX))
    files = []
    for f in sorted((config.WORK_DIR / series).glob("*/warped.tif")):
        with rasterio.open(f) as ds:
            b = ds.bounds
            if b.right > x0 and b.left < x1 and b.top > y0 and b.bottom < y1:
                files.append(f)
    if not files:
        return np.full((h, w, 3), 255, np.uint8)
    sets = [rasterio.open(f) for f in files]
    try:
        arr, _ = merge(sets, bounds=(x0, y0, x1, y1), res=FT_PER_PX, nodata=0)
    finally:
        for ds in sets:
            ds.close()
    arr = arr[:, :h, :w]
    rgb, alpha = arr[:3].transpose(1, 2, 0).astype(float), arr[3].astype(float)[..., None] / 255
    return np.clip(rgb * alpha + 255 * (1 - alpha), 0, 255).astype(np.uint8)


def _px(coords, bounds) -> np.ndarray:
    x0, _, _, y1 = bounds
    c = np.asarray(coords, float)
    return np.round(np.c_[(c[:, 0] - x0) / FT_PER_PX * ZOOM, (y1 - c[:, 1]) / FT_PER_PX * ZOOM]).astype(np.int32)


def _rings(geom):
    for poly in getattr(geom, "geoms", [geom]):
        if poly.geom_type == "Polygon":
            yield poly


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)     # BGR for cv2


def panel(img: np.ndarray, bounds, lots: gpd.GeoDataFrame, title: str) -> np.ndarray:
    """img (RGB at FT_PER_PX) drawn ZOOM x larger with the lots tinted, outlined and labelled; a title bar on top."""
    base = cv2.cvtColor(cv2.resize(img, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
    tint = base.copy()
    for geom, cat in zip(lots.geometry, lots["category"].fillna("unknown")):
        colour = _hex(COLORS.get(cat, "#ffffff"))
        for poly in _rings(geom):
            cv2.fillPoly(tint, [_px(poly.exterior.coords, bounds)] + [_px(r.coords, bounds) for r in poly.interiors],
                         colour)
    out = cv2.addWeighted(tint, ALPHA, base, 1 - ALPHA, 0)
    for geom in lots.geometry:
        for poly in _rings(geom):
            cv2.polylines(out, [_px(poly.exterior.coords, bounds)], True, (90, 90, 90), 1, cv2.LINE_AA)
    for code, group in lots.dropna(subset=["code"]).groupby("code"):
        merged = group.geometry.buffer(1.0).union_all()
        for part in getattr(merged, "geoms", [merged]):
            if part.area < MIN_LABEL_FT2:
                continue
            p = _px([part.representative_point().coords[0]], bounds)[0]
            (tw, th), _ = cv2.getTextSize(code, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (p[0] - tw // 2 - 2, p[1] - th // 2 - 3), (p[0] + tw // 2 + 2, p[1] + th // 2 + 3),
                          (255, 255, 255), -1)
            cv2.putText(out, code, (p[0] - tw // 2, p[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)
    bar = np.full((28, out.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--box", nargs=4, type=float, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--scan", required=True, help="the series whose warped sheets are the backdrop")
    ap.add_argument("--map", action="append", required=True, help="LABEL=PATH of a parcels.fgb (EPSG:2227)")
    ap.add_argument("--pad", type=float, default=250.0, help="ft added around the box")
    a = ap.parse_args()
    x0, y0, x1, y1 = a.box
    bounds = (x0 - a.pad, y0 - a.pad, x1 + a.pad, y1 + a.pad)
    img = backdrop(a.scan, bounds)
    window = shapely_box(*bounds)
    panels = []
    for label, path in (m.split("=", 1) for m in a.map):
        lots = gpd.read_file(path, bbox=bounds)
        lots = lots[lots.intersects(window)]
        panels.append(panel(img, bounds, lots, f"{label}: {len(lots):,} lots"))
    gap = np.full((panels[0].shape[0], 12, 3), 255, np.uint8)
    sheet = np.hstack([p if i == 0 else np.hstack([gap, p]) for i, p in enumerate(panels)])
    head = np.full((30, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"{a.name}  ({a.scan} scan, 2 ft a px)", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
                cv2.LINE_AA)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.name}.png"
    if not cv2.imwrite(str(out), np.vstack([head, sheet])):
        raise OSError(f"can't write {out}")
    print(out)


if __name__ == "__main__":
    main()
