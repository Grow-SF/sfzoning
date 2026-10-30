"""Synthetic warped sheets for the districts tests (EPSG:2227 RGBA GeoTIFFs under config.WORK_DIR).

``write_sheet`` lays out ``WORK_DIR/<series>/<sheet>/`` with warped.tif, score.json and
(optionally) warped.footprint.geojson, as the real pipeline would.  Content is chosen by
``fill``: a name from FILLS or any callable ``(rows, cols) -> uint8 gray array``; extend FILLS
(or pass your own callable) for new patterns.  ``alpha_box`` limits the valid-alpha area to a
sub-rectangle (x0, y0, x1, y1 in world feet); ``footprint`` is the geojson footprint box
(default: the raster bounds; None writes no file).
"""
from __future__ import annotations

import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from sfzmaps import config


def hatch(rows, cols, period=4):
    yy, xx = np.mgrid[:rows, :cols]
    return np.where((xx + yy) % period < 1, 0, 255).astype(np.uint8)


def crosshatch(rows, cols, period=4):
    yy, xx = np.mgrid[:rows, :cols]
    return np.where(((xx + yy) % period < 1) | ((xx - yy) % period < 1), 0, 255).astype(np.uint8)


def grid(rows, cols, period=8):
    yy, xx = np.mgrid[:rows, :cols]
    return np.where((xx % period < 1) | (yy % period < 1), 0, 255).astype(np.uint8)


def stipple(rows, cols, period=4):
    yy, xx = np.mgrid[:rows, :cols]
    return np.where((xx % period < 1) & (yy % period < 1), 0, 255).astype(np.uint8)


def checker(rows, cols, period=8):
    yy, xx = np.mgrid[:rows, :cols]
    return np.where(((xx // period) + (yy // period)) % 2 == 0, 0, 255).astype(np.uint8)


FILLS = {
    "blank": lambda r, c: np.full((r, c), 255, np.uint8),
    "solid": lambda r, c: np.zeros((r, c), np.uint8),
    "hatch": hatch,
    "crosshatch": crosshatch,
    "grid": grid,
    "stipple": stipple,
    "checker": checker,
}

# A fake 1948-style legend: code -> fill name.
FAKE_LEGEND = {"1R": "hatch", "2R": "blank", "C": "solid", "LI": "crosshatch", "HI": "grid",
               "U": "checker"}


def swatches(legend=None, h=24, w=48):
    """code -> RGB swatch (h x w x 3) for a fake legend."""
    legend = legend or FAKE_LEGEND
    return {code: np.repeat(FILLS[f](h, w)[..., None], 3, axis=2) for code, f in legend.items()}


def pattern_fill(x0, y0, h_ft, res, items):
    """A `fill` callable for write_sheet painting each (geometry, fill name) of `items` with its
    pattern on white paper, world coordinates as in write_sheet."""
    from rasterio import features

    def fill(rows, cols):
        tr = from_origin(x0, y0 + h_ft, res, res)
        out = np.full((rows, cols), 255, np.uint8)
        for geom, name in items:
            m = features.rasterize([(geom, 1)], out_shape=(rows, cols), transform=tr, fill=0,
                                   dtype="uint8") > 0
            out[m] = FILLS[name](rows, cols)[m]
        return out
    return fill


def write_legend_page(series, sheet_id, legend=None, *, at=(100, 100)):
    """page.png (white, with swatch rectangles and black borders) + page.json whose single
    mask box holds the legend.  Returns {code: (x0, y0, x1, y1)} swatch boxes in page px."""
    import cv2

    legend = legend or FAKE_LEGEND
    d = config.WORK_DIR / series / sheet_id
    d.mkdir(parents=True, exist_ok=True)
    page = np.full((800, 800, 3), 255, np.uint8)
    ax, ay = at
    boxes = {}
    for i, (code, name) in enumerate(legend.items()):
        x0, y0 = ax + 20 + (i % 2) * 200, ay + 20 + (i // 2) * 50
        x1, y1 = x0 + 80, y0 + 36
        page[y0:y1, x0:x1] = FILLS[name](y1 - y0, x1 - x0)[..., None]
        cv2.rectangle(page, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 0), 2)
        cv2.putText(page, code, (x1 + 8, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        boxes[code] = (x0, y0, x1, y1)
    cv2.imwrite(str(d / "page.png"), page)
    mb = {"x0": ax, "y0": ay, "x1": ax + 440, "y1": ay + 20 + 50 * ((len(legend) + 1) // 2) + 10}
    (d / "page.json").write_text(json.dumps({"kind": "map", "sheet_number": sheet_id,
                                             "neighbor_refs": [], "mask_boxes": [mb]}))
    return boxes, mb


def write_score(sheet_dir, passed=True, p50=3.0, reason=None):
    (sheet_dir / "score.json").write_text(json.dumps({
        "n_scored": 50, "insample_frac_within": 0.5, "insample_p50_ft": p50, "insample_p95_ft": 9.0,
        "insample_max_ft": 12.0, "n_real_change": 0, "n_no_equivalent": 0, "passed": passed,
        "reason": reason, "holdout_p50_ft": p50, "holdout_p95_ft": 9.0, "n_holdout": 50,
        "score_version": config.SCORE_VERSION}))


def write_sheet(series, sheet_id, x0, y0, w_ft, h_ft, *, res=1.0, fill="blank", passed=True,
                p50=3.0, alpha_box=None, footprint="bounds", paper="opaque"):
    """Write one warped sheet whose lower-left corner is (x0, y0); returns the warped.tif path.

    paper="transparent" mimics the real warp: alpha = darkness and RGB premultiplied toward
    black, so white paper is (0, 0, 0, 0)."""
    d = config.WORK_DIR / series / sheet_id
    d.mkdir(parents=True, exist_ok=True)
    cols, rows = int(round(w_ft / res)), int(round(h_ft / res))
    gray = (FILLS[fill] if isinstance(fill, str) else fill)(rows, cols)
    data = np.zeros((4, rows, cols), np.uint8)
    if paper == "transparent":
        data[3] = 255 - gray
    else:
        data[:3] = gray
        data[3] = 255
    if alpha_box is not None:
        ax0, ay0, ax1, ay1 = alpha_box
        top = y0 + h_ft
        a = np.zeros((rows, cols), np.uint8)
        c0, c1 = int((ax0 - x0) / res), int((ax1 - x0) / res)
        r0, r1 = int((top - ay1) / res), int((top - ay0) / res)
        a[max(r0, 0):r1, max(c0, 0):c1] = 255
        data[3] = np.minimum(data[3], a)
    tif = d / "warped.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=cols, height=rows, count=4, dtype="uint8",
                       crs=config.WORK_CRS, transform=from_origin(x0, y0 + h_ft, res, res)) as ds:
        ds.write(data)
        ds.colorinterp = [rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                          rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha]
    write_score(d, passed, p50)
    if footprint is not None:
        x1, y1 = x0 + w_ft, y0 + h_ft
        fx0, fy0, fx1, fy1 = (x0, y0, x1, y1) if footprint == "bounds" else footprint
        ring = [[fx0, fy0], [fx1, fy0], [fx1, fy1], [fx0, fy1], [fx0, fy0]]
        (d / "warped.footprint.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}))
    return tif
