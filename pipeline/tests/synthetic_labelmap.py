"""Synthetic label-and-boundary map (Reader B): thin block outlines, heavy district lines,
codes printed in the regions, plus a linear page->world transform and modern parcels.

Layout (page px): a heavy neatline frame; blocks of BW x BH px on a STEP_X x STEP_Y grid with
thin (THIN px) outlines; heavy (HEAVY px) district lines on street centres:
- a vertical line at x = 410 and x = 810 (full height), and a horizontal one at y = 410 left
  of x = 410, giving districts TL (x<410, y<410), BL (x<410, y>410), MID (410<x<810) and
  RIGHT (x>810);
- a heavy box around one block inside RIGHT (the ISLAND district).
World: 1 px = FT_PER_PX ft, x east, y north from (X0, Y0) at the page's top-left.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import geopandas as gpd
import numpy as np
from shapely.geometry import box

from sfzmaps import config
from sfzmaps.transform import GCP, SheetTransform

W, H = 1220, 860
FRAME = (10, 10, 1210, 850)    # neatline (outer edge of the heavy frame)
THIN, HEAVY = 2, 7
STEP_X, STEP_Y, BW, BH = 100, 80, 80, 60
ORIGIN = (20, 20)              # top-left block corner
X0, Y0, FT_PER_PX = 6_000_000.0, 2_100_000.0, 2.0
ISLAND_BLOCK = (9, 2)          # (column, row) of the island block
CENTRES = {"TL": (200, 200), "BL": (200, 600), "MID": (610, 400), "RIGHT": (1010, 650),
           "ISLAND": (950, 210)}


@dataclass
class LabelMap:
    img: np.ndarray            # gray uint8 page
    exclude: np.ndarray        # bool, True outside the neatline
    transform: SheetTransform
    parcels: gpd.GeoDataFrame  # EPSG:2227, blklot / block_num


def block_px(i: int, j: int) -> tuple[int, int, int, int]:
    x, y = ORIGIN[0] + i * STEP_X, ORIGIN[1] + j * STEP_Y
    return x, y, x + BW, y + BH


def blocks() -> list[tuple[int, int]]:
    return [(i, j) for j in range((FRAME[3] - ORIGIN[1]) // STEP_Y)
            for i in range((FRAME[2] - ORIGIN[0]) // STEP_X)
            if block_px(i, j)[2] < FRAME[2] - 5 and block_px(i, j)[3] < FRAME[3] - 5]


def to_world_xy(px: float, py: float) -> tuple[float, float]:
    return X0 + px * FT_PER_PX, Y0 - py * FT_PER_PX


def px_box_world(x0, y0, x1, y1):
    a, b = to_world_xy(x0, y1)
    c, d = to_world_xy(x1, y0)
    return box(a, b, c, d)


def linear_transform() -> SheetTransform:
    gcps = [GCP(px, py, *to_world_xy(px, py), "auto")
            for px in np.linspace(0, W, 5) for py in np.linspace(0, H, 5)]
    return SheetTransform(gcps, config.TPS_SMOOTHING)


def build(*, districts: bool = True, codes: dict[str, str] | None = None) -> LabelMap:
    """The synthetic page.  districts=False draws no heavy district lines (one region: the
    fallback case).  codes: region name -> text printed at CENTRES[name]."""
    img = np.full((H, W), 255, np.uint8)
    fx0, fy0, fx1, fy1 = FRAME
    cv2.rectangle(img, (fx0, fy0), (fx1 - 1, fy1 - 1), 0, HEAVY)
    for i, j in blocks():
        x0, y0, x1, y1 = block_px(i, j)
        cv2.rectangle(img, (x0, y0), (x1, y1), 0, THIN)
    if districts:
        h2 = HEAVY // 2
        img[fy0:fy1, 410 - h2:410 - h2 + HEAVY] = 0
        img[fy0:fy1, 810 - h2:810 - h2 + HEAVY] = 0
        img[410 - h2:410 - h2 + HEAVY, fx0:410] = 0
        x0, y0, x1, y1 = block_px(*ISLAND_BLOCK)
        cv2.rectangle(img, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), 0, HEAVY)
    for name, text in (codes or {}).items():
        cx, cy = CENTRES[name]
        cv2.putText(img, text, (cx - 25, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    exclude = np.ones((H, W), bool)
    exclude[fy0:fy1, fx0:fx1] = False
    rows = []
    for i, j in blocks():
        x0, y0, x1, y1 = block_px(i, j)
        blk = f"{j * 20 + i + 1:04d}"
        xm = (x0 + x1) / 2
        rows.append((blk + "001", blk, px_box_world(x0, y0, xm, y1)))
        rows.append((blk + "002", blk, px_box_world(xm, y0, x1, y1)))
    parcels = gpd.GeoDataFrame({"blklot": [r[0] for r in rows], "block_num": [r[1] for r in rows]},
                               geometry=[r[2] for r in rows], crs=config.WORK_CRS)
    return LabelMap(img, exclude, linear_transform(), parcels)


def write_sheet(series: str, sheet_id: str, lm: LabelMap, page_json: dict | None = None):
    """page.png, page.json and transform.json under WORK_DIR/<series>/<sheet_id>."""
    import json

    d = config.WORK_DIR / series / sheet_id
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "page.png"), lm.img)
    (d / "page.json").write_text(json.dumps(page_json or {"kind": "map", "mask_boxes": []}))
    (d / "transform.json").write_text(json.dumps({**lm.transform.to_json(), "origin": "auto"}))
    return d
