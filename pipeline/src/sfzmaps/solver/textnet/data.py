"""The text detector's data: each 2000 sheet's own page (`page.png`, the scan before any warp), with what it learns.

  ink        page < `page.INK_BELOW` (128), the model's only input
  text       1 inside one of Terra's text boxes (every kind; `PRINTS_DIR/terra_boxes.json`), drawn as the rotated
             rectangle of the word in page px
  kind       the box's kind (KINDS: 0 zoning code, 1 block number, 2 street name, 3 other), NO_KIND outside text
  angle      the reading direction in whole degrees, counter-clockwise from left-to-right, inside text (0 outside)
  unknown    inside another pass's box (OTHER_PASSES) or within READ_POINT_RADIUS of one of Sol's label points, but in
             no Terra box: text Terra may have missed, taught neither as text nor as not-text
  heldout    the px of the quarter held out: `lots.quarter_of` of the px's map point through the sheet's corrected
             placement
  known      the px the loss reads: not unknown and not held out
  centre_ok  the px a training crop may centre on: ink or text, not held out

The reading angle is kept in degrees, as Terra gives it; the model learns its cosine and sine.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sfzmaps import config
from sfzmaps.solver import lots, page, paths, saved
from sfzmaps.transform import SheetTransform

KINDS = {"zoning code": 0, "block number": 1, "street name": 2, "other": 3}
NO_KIND = 255
OTHER_PASSES = "experiments-2000/text-positions"   # the saved answers holding the other passes' boxes, page px
READ_POINT_RADIUS = 30            # px around one of Sol's label points treated as unknown
TYPICAL_HEIGHT = 14.0             # px, the text height used when a slanted box's own shape cannot give it
PLACEMENT_STEP = 32               # px between the page points where the placement itself is evaluated


@dataclass
class SheetTargets:
    """One sheet's page and what the detector learns on it, every array the page's shape (see the module)."""
    ink: np.ndarray          # bool
    text: np.ndarray         # uint8 0 / 1
    kind: np.ndarray         # uint8 KINDS, NO_KIND outside text
    angle: np.ndarray        # int16 degrees
    unknown: np.ndarray      # bool
    heldout: np.ndarray      # bool
    known: np.ndarray        # bool
    centre_ok: np.ndarray    # bool


def quarter_index(q: str) -> int:
    """The index of a quarter's name ("nw" 0, "ne" 1, "sw" 2, "se" 3), as `lots.quarter_of` numbers them."""
    if q not in paths.QUARTERS:
        raise ValueError(f"no quarter {q!r}: one of {', '.join(paths.QUARTERS)}")
    return paths.QUARTERS.index(q)


def word_rect(box, angle_deg: float) -> np.ndarray:
    """The rotated rectangle (4 page px corners, the first edge along the reading direction) of a word, from its axis
    box (x0, y0, x1, y1) and reading angle (degrees, counter-clockwise from left-to-right)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    a = np.deg2rad(angle_deg)
    c, s = abs(np.cos(a)), abs(np.sin(a))
    det = c * c - s * s
    if abs(det) > 0.25:
        length, height = (c * w - s * h) / det, (c * h - s * w) / det
    else:
        height = TYPICAL_HEIGHT
        length = (w - height * s) / c if c >= s else (h - height * c) / s
    length, height = max(length, 2.0), max(height, 2.0)
    v = np.array([np.cos(a), -np.sin(a)])                     # the reading direction, page coordinates (y down)
    n = np.array([-v[1], v[0]])
    ctr = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return np.array([ctr - v * length / 2 - n * height / 2, ctr + v * length / 2 - n * height / 2,
                     ctr + v * length / 2 + n * height / 2, ctr - v * length / 2 + n * height / 2])


def reading_angle(rect: np.ndarray) -> int:
    """A word rectangle's reading direction (its first edge) in whole degrees, counter-clockwise from left-to-right."""
    dx, dy = rect[1] - rect[0]
    return int(round(np.rad2deg(np.arctan2(-dy, dx))))


def _fill(raster: np.ndarray, rect: np.ndarray, value: int) -> None:
    cv2.fillPoly(raster, [np.round(rect).astype(np.int32).reshape(-1, 1, 2)], value)


def targets(ink: np.ndarray, terra: list[dict], others: list[dict], read_points, quarter: np.ndarray,
            q: str) -> SheetTargets:
    """A page's targets from its ink, Terra's boxes (each `box` and `kind`, optional `angle_deg`; later boxes drawn
    over earlier ones), the other passes' boxes, Sol's label points (x, y) and each px's quarter (0..3), with the
    quarter `q` ("nw", "ne", "sw" or "se") held out."""
    shape = ink.shape
    text = np.zeros(shape, np.uint8)
    kind = np.full(shape, NO_KIND, np.uint8)
    angle = np.zeros(shape, np.int16)
    for b in terra:
        rect = word_rect(b["box"], float(b.get("angle_deg") or 0.0))
        _fill(text, rect, 1)
        _fill(kind, rect, KINDS.get(b["kind"], KINDS["other"]))
        _fill(angle, rect, reading_angle(rect))
    other = np.zeros(shape, np.uint8)
    for b in others:
        _fill(other, word_rect(b["box"], float(b.get("angle_deg") or 0.0)), 1)
    for x, y in read_points:
        cv2.circle(other, (int(round(x)), int(round(y))), READ_POINT_RADIUS, 1, -1)
    unknown = (other > 0) & (text == 0)
    heldout = quarter == quarter_index(q)
    return SheetTargets(ink=ink, text=text, kind=kind, angle=angle, unknown=unknown, heldout=heldout,
                        known=~unknown & ~heldout, centre_ok=(ink | (text > 0)) & ~heldout)


def _between(n: int, step: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """For px 0..n-1 and samples every `step` px from 0 (`count` of them, the last at or past n - 1): each px's
    sample before it and its share of the way to the next."""
    i = np.arange(n)
    j = np.minimum(i // step, count - 2)
    return j, (i - j * step) / step


def page_world(transform, shape: tuple[int, int], step: int = PLACEMENT_STEP) -> tuple[np.ndarray, np.ndarray]:
    """The map point (x, y; EPSG:2227 ft, float64 rasters of `shape`) of every page px, px (col, row) taken at page
    coordinates (col, row): `transform.to_world` at every `step`-th px each way (from 0 to at or past the far edge),
    bilinear between."""
    h, w = shape
    cols = np.arange(0, w - 1 + step, step, dtype=float)
    rows = np.arange(0, h - 1 + step, step, dtype=float)
    gx, gy = np.meshgrid(cols, rows)
    world = transform.to_world(np.c_[gx.ravel(), gy.ravel()]).reshape(len(rows), len(cols), 2)
    jr, fr = _between(h, step, len(rows))
    jc, fc = _between(w, step, len(cols))
    out = []
    for k in (0, 1):
        by_row = (1 - fr)[:, None] * world[jr, :, k] + fr[:, None] * world[jr + 1, :, k]
        out.append((1 - fc)[None, :] * by_row[:, jc] + fc[None, :] * by_row[:, jc + 1])
    return out[0], out[1]


def page_quarters(transform, shape: tuple[int, int]) -> np.ndarray:
    """Each page px's quarter of the city (int8 raster: nw 0, ne 1, sw 2, se 3), from its map point (`page_world`)."""
    x, y = page_world(transform, shape)
    return lots.quarter_of(np.c_[x.ravel(), y.ravel()]).reshape(shape)


def load_page(sheet: str) -> np.ndarray:
    """A sheet's page (data/work/<series>/<sheet>/page.png), grayscale uint8."""
    f = config.WORK_DIR / paths.SERIES / sheet / "page.png"
    gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"can't read {f}")
    return gray


def warp_dir(sheet: str) -> Path:
    """A sheet's folder of the warp's output (its corrected placement and footprint)."""
    return config.DATA_DIR / "warp" / paths.SERIES / sheet


def load_transform(sheet: str) -> SheetTransform:
    """A sheet's corrected placement (data/warp/<series>/<sheet>/transform.json)."""
    folder = warp_dir(sheet)
    return SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)


def terra_boxes(sheet: str) -> list[dict]:
    """A sheet's Terra boxes, every kind, page px, as the prints stage decoded them (`PRINTS_DIR/terra_boxes.json`)."""
    f = paths.PRINTS_DIR / "terra_boxes.json"
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver prints`")
    boxes = json.loads(f.read_text())
    if sheet not in boxes:
        raise KeyError(f"{f} has no boxes for {sheet}: run `sfzmaps solver prints` again")
    return boxes[sheet]


def other_boxes(sheet: str) -> list[dict]:
    """The boxes the other text passes found on a sheet (every file of OTHER_PASSES, by name, in file order)."""
    files = sorted((paths.SAVED_ANSWERS / OTHER_PASSES).glob("*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"no saved answers under {paths.SAVED_ANSWERS / OTHER_PASSES}")
    return [b for f in files for b in saved.by_sheet(f"{OTHER_PASSES}/{f.name.removesuffix('.jsonl.gz')}", sheet)]


def sheet_targets(sheet: str, q: str) -> SheetTargets:
    """One sheet's targets with the quarter `q` held out (see the module)."""
    gray = load_page(sheet)
    quarter = page_quarters(load_transform(sheet), gray.shape)
    read_points = [(lb["x"], lb["y"]) for lb in saved.sol_labels(sheet)]
    return targets(gray < page.INK_BELOW, terra_boxes(sheet), other_boxes(sheet), read_points, quarter, q)
