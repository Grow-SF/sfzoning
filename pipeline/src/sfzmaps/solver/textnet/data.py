"""The text detector's data: each sheet's own page (`page.png`, the scan before any warp), with what it learns.

  ink        page < `grid.INK_BELOW` (128), the model's only input
  text       1 inside one of Terra's text boxes (every kind), drawn as the rotated rectangle of the word in page px
  kind       the box's kind (KINDS: 0 zoning code, 1 block number, 2 street name, 3 other), NO_KIND outside text
  angle      the reading direction in whole degrees, counter-clockwise from left-to-right, inside text (0 outside)
  unknown    within READ_POINT_RADIUS of one of Sol's label points or, on the 2000 sheets, inside another pass's box
             (OTHER_PASSES; the other passes read only those), but in no Terra box: text Terra may have missed, taught
             neither as text nor as not-text
  heldout    the px held out: whose map point, through the sheet's corrected placement, is in the held-out area
  known      the px the loss reads: not unknown and not held out
  centre_ok  the px a training crop may centre on: ink or text, not held out

The reading angle is kept in degrees, as Terra gives it; the model learns its cosine and sine.

The detector of every map (`load_sheets`, `crop_targets`) learns from the sheets Terra read (`paths.TEXT_SHEETS`) of
the training maps, Terra's boxes decoded from its saved answers (`boxes.terra_boxes`), and holds out VALIDATION_AREA
on every map at once.  A sheet is kept compact (`SheetData`): its ink as packed bits and its boxes, points and
placement as short arrays; the targets are drawn for each crop from them, never for a whole page.  A shape is filled
as `cv2.fillPoly` / `cv2.circle` fill it on a page large enough that none is cut at its edge, so a crop's targets are
those of the whole page, cropped.

The 2000 detectors of the four quarters (`sheet_targets`) still read whole pages, with a quarter held out, from the
prints stage's decoding of Terra's boxes (`PRINTS_DIR/terra_boxes.json`).
"""
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sfzmaps import config
from sfzmaps.solver import grid, paths, saved
from sfzmaps.solver.textnet import boxes
from sfzmaps.transform import SheetTransform

KINDS = {"zoning code": 0, "block number": 1, "street name": 2, "other": 3}
NO_KIND = 255
OTHER_PASSES = "experiments-2000/text-positions"   # the saved answers holding the other passes' boxes, page px
READ_POINT_RADIUS = 30            # px around one of Sol's label points treated as unknown
TYPICAL_HEIGHT = 14.0             # px, the text height used when a slanted box's own shape cannot give it
PLACEMENT_STEP = 32               # px between the page points where the placement itself is evaluated
# The validation area (x0, y0, x1, y1; EPSG:2227 ft): a page px is in it when its map point is (x0 <= x < x1,
# y0 <= y < y1).  Held out of training on every map at once (so the reprints of one drawing can't teach it to each
# other); the scores are computed on it.  Chosen once, by this rule: of the rectangles of 10% of the area the training
# maps' Terra sheets cover together (the union of their warped footprints, 50.2 sq mi: so 5.0 sq mi) that lie inside
# every training map's sheets (at least 99.5% inside their intersection, 38.7 sq mi; corners on a 250 ft grid, width
# to height 1:2 to 2:1), the one whose share of each training map's Terra box area is nearest 10%: in all and, at half
# weight, of each kind the map prints (5% of its boxes or more; the older maps print no block numbers).  The rule's
# best was the residential south-west (Parkside, Lakeshore, Ingleside; score 0.022), but its lettering is sparse and
# simple: an easy area can't tell training recipes apart, and the scores are to be taken on dense areas, never only the
# simple west.  So this is the runner-up (score 0.030), the central strip through the Mission, Noe Valley and Glen
# Park: 99.7% inside every training map's sheets, with 10-13% of each training map's box area (7-16% of each kind it
# prints).  `pipeline/scripts/choose_text_validation_area.py` re-derives the candidates and this choice from the data.
VALIDATION_AREA = (6000442.0, 2091727.0, 6008805.0, 2108453.0)
BOX_FIELDS = ("words", "kinds", "angles", "other", "points")    # SheetData's arrays of boxes and points
LOAD_THREADS = 8                  # sheets read at once
BAND = 512                        # page rows drawn at a time when finding where crops may centre


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


@dataclass
class SheetData:
    """One sheet of a map, compact: its ink, its boxes and points, and its placement, from which `crop_targets` draws
    a crop's targets (see the module).  The ink is kept as packed bits: one bit a px, 1/8 of a bool page (the
    training sheets' 2.4 Gpx are 0.3 GB, not 2.4), and a crop unpacks only its own rows and bytes."""
    series: str
    sheet: str
    height: int              # the page's px
    width: int
    ink_bits: np.ndarray     # uint8 (height, ceil(width / 8)): the ink, packed 8 px a byte along each row (`page_ink`)
    words: np.ndarray        # int32 (n, 4, 2): Terra's words' rectangles (`word_rect`), corners (x, y) rounded to px
    kinds: np.ndarray        # uint8 (n,): each word's KINDS
    angles: np.ndarray       # int16 (n,): each word's reading angle (`reading_angle`), degrees
    other: np.ndarray        # int32 (m, 4, 2): the other passes' rectangles, as `words` (none off the 2000 sheets)
    points: np.ndarray       # int32 (p, 2): Sol's label points (x, y), rounded to px
    world: np.ndarray        # float64 (rows, cols, 2): the map point of every PLACEMENT_STEP-th page px each way
    ft_per_px: float         # the placement's scale over the page (`SheetTransform.median_scale_ft_per_px`)
    hold_out: bool           # VALIDATION_AREA is held out: no loss there, no crop centred there


def quarter_index(q: str) -> int:
    """The index of a quarter's name ("nw" 0, "ne" 1, "sw" 2, "se" 3), as `grid.quarter_of` numbers them."""
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


def _between(i: np.ndarray, step: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """For page px `i` and samples every `step` px from 0 (`count` of them, the last at or past the page's last px):
    each px's sample before it and its share of the way to the next."""
    j = np.minimum(i // step, count - 2)
    return j, (i - j * step) / step


def placement_samples(transform, shape: tuple[int, int], step: int = PLACEMENT_STEP) -> np.ndarray:
    """The map point (x, y; EPSG:2227 ft) of every `step`-th px each way of a page of `shape`, from 0 to at or past
    its far edge: float64 (rows, cols, 2), px (col, row) taken at page coordinates (col, row)."""
    h, w = shape
    cols = np.arange(0, w - 1 + step, step, dtype=float)
    rows = np.arange(0, h - 1 + step, step, dtype=float)
    gx, gy = np.meshgrid(cols, rows)
    return transform.to_world(np.c_[gx.ravel(), gy.ravel()]).reshape(len(rows), len(cols), 2).copy()


def _world_at(world: np.ndarray, rows: np.ndarray, cols: np.ndarray,
              step: int = PLACEMENT_STEP) -> tuple[np.ndarray, np.ndarray]:
    """The map point (x, y rasters, float64) of the page px at `rows` x `cols`, bilinear between the placement's
    samples `world` (`placement_samples`).  The same numbers whichever window of the page is asked for."""
    jr, fr = _between(rows, step, world.shape[0])
    jc, fc = _between(cols, step, world.shape[1])
    out = []
    for k in (0, 1):
        by_row = (1 - fr)[:, None] * world[jr, :, k] + fr[:, None] * world[jr + 1, :, k]
        out.append((1 - fc)[None, :] * by_row[:, jc] + fc[None, :] * by_row[:, jc + 1])
    return out[0], out[1]


def page_world(transform, shape: tuple[int, int], step: int = PLACEMENT_STEP) -> tuple[np.ndarray, np.ndarray]:
    """The map point (x, y; EPSG:2227 ft, float64 rasters of `shape`) of every page px, px (col, row) taken at page
    coordinates (col, row): `transform.to_world` at every `step`-th px each way (from 0 to at or past the far edge),
    bilinear between."""
    h, w = shape
    return _world_at(placement_samples(transform, shape, step), np.arange(h), np.arange(w), step)


def page_quarters(transform, shape: tuple[int, int]) -> np.ndarray:
    """Each page px's quarter of the city (int8 raster: nw 0, ne 1, sw 2, se 3), from its map point (`page_world`)."""
    x, y = page_world(transform, shape)
    return grid.quarter_of(np.c_[x.ravel(), y.ravel()]).reshape(shape)


def load_page(sheet: str, series: str = paths.SERIES) -> np.ndarray:
    """A sheet's page (data/work/<series>/<sheet>/page.png), grayscale uint8."""
    f = config.WORK_DIR / series / sheet / "page.png"
    gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"can't read {f}")
    return gray


def warp_dir(sheet: str, series: str = paths.SERIES) -> Path:
    """A sheet's folder of the warp's output (its corrected placement and footprint)."""
    return config.DATA_DIR / "warp" / series / sheet


def load_transform(sheet: str, series: str = paths.SERIES) -> SheetTransform:
    """A sheet's corrected placement (data/warp/<series>/<sheet>/transform.json)."""
    folder = warp_dir(sheet, series)
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


def _other_files() -> list[Path]:
    files = sorted((paths.SAVED_ANSWERS / OTHER_PASSES).glob("*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"no saved answers under {paths.SAVED_ANSWERS / OTHER_PASSES}")
    return files


def other_boxes(sheet: str) -> list[dict]:
    """The boxes the other text passes found on a sheet (every file of OTHER_PASSES, by name, in file order)."""
    return [b for f in _other_files() for b in saved.by_sheet(f"{OTHER_PASSES}/{f.name.removesuffix('.jsonl.gz')}",
                                                               sheet)]


def sheet_targets(sheet: str, q: str) -> SheetTargets:
    """One sheet's targets with the quarter `q` held out (see the module)."""
    gray = load_page(sheet)
    quarter = page_quarters(load_transform(sheet), gray.shape)
    read_points = [(lb["x"], lb["y"]) for lb in saved.sol_labels(sheet)]
    return targets(gray < grid.INK_BELOW, terra_boxes(sheet), other_boxes(sheet), read_points, quarter, q)


# --- every map, compact, with the validation area held out ----------------------------------------------------------

def _corners(rects: list[np.ndarray]) -> np.ndarray:
    """Word rectangles' corners rounded to page px, int32 (n, 4, 2)."""
    return np.round(np.array(rects, float).reshape(-1, 4, 2)).astype(np.int32)


def sheet_data(series: str, sheet: str, hold_out_validation: bool, terra: list[dict] | None = None,
               others: list[dict] | None = None) -> SheetData:
    """One sheet's SheetData (see the module).  `terra`: lines of Terra's saved answers holding the sheet's (read
    from disk when None); `others`: the other passes' boxes of the sheet (read from disk when None, on a 2000 sheet;
    none on another map's)."""
    gray = load_page(sheet, series)
    transform = load_transform(sheet, series)
    found = boxes.terra_boxes(series, sheet, terra)
    rects = [word_rect(b["box"], float(b.get("angle_deg") or 0.0)) for b in found]
    if others is None:
        others = other_boxes(sheet) if series == paths.SERIES else []
    points = [(int(round(lb["x"])), int(round(lb["y"]))) for lb in saved.sol_labels(sheet, series)]
    return SheetData(
        series=series, sheet=sheet, height=gray.shape[0], width=gray.shape[1],
        ink_bits=np.packbits(gray < grid.INK_BELOW, axis=1),
        words=_corners(rects), kinds=np.array([KINDS.get(b["kind"], KINDS["other"]) for b in found], np.uint8),
        angles=np.array([reading_angle(r) for r in rects], np.int16),
        other=_corners([word_rect(b["box"], float(b.get("angle_deg") or 0.0)) for b in others]),
        points=np.array(points, np.int32).reshape(-1, 2),
        world=placement_samples(transform, gray.shape), ft_per_px=float(transform.median_scale_ft_per_px(gray.shape)),
        hold_out=bool(hold_out_validation))


def load_sheets(series_list, hold_out_validation: bool) -> list[SheetData]:
    """Every sheet Terra read (`paths.TEXT_SHEETS`) of the maps in `series_list`, map by map in that order and in the
    sheets' order, with VALIDATION_AREA held out or not; LOAD_THREADS sheets at a time.  Stops (ValueError) when the
    sheet lists differ from the saved answers (`boxes.check_text_sheets`) or a map has none."""
    unknown = [s for s in series_list if s not in paths.TEXT_SHEETS]
    if unknown:
        raise ValueError(f"no sheets of {', '.join(unknown)} in sfzmaps.solver.paths.TEXT_SHEETS")
    records = boxes.terra_records()
    boxes.check_text_sheets(records)
    by_sheet: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        by_sheet.setdefault((r["series"], r["sheet"]), []).append(r)
    others: dict[str, list[dict]] = {}
    if paths.SERIES in series_list:
        for f in _other_files():
            for b in saved.answers(f"{OTHER_PASSES}/{f.name.removesuffix('.jsonl.gz')}"):
                others.setdefault(b.get("sheet"), []).append(b)
    jobs = [(s, name) for s in series_list for name in paths.TEXT_SHEETS[s]]

    def one(job):
        s, name = job
        return sheet_data(s, name, hold_out_validation, by_sheet.get(job, []),
                          others.get(name, []) if s == paths.SERIES else [])

    with ThreadPoolExecutor(LOAD_THREADS) as pool:
        return list(pool.map(one, jobs))


def page_ink(sd: SheetData) -> np.ndarray:
    """A sheet's whole page of ink, bool (height, width)."""
    return np.unpackbits(sd.ink_bits, axis=1, count=sd.width).view(bool)


def _ink_window(sd: SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The ink of the page window of h x w px whose top-left px is (row y0, col x0), bool."""
    b0 = x0 // 8
    bits = np.unpackbits(sd.ink_bits[y0:y0 + h, b0:(x0 + w + 7) // 8], axis=1)
    return bits[:, x0 - 8 * b0:x0 - 8 * b0 + w].view(bool)


def _paint(shapes: np.ndarray, values: np.ndarray, y0: int, x0: int, h: int, w: int,
           discs: np.ndarray | None = None, radius: int = 0) -> np.ndarray:
    """A page window of h x w px (top-left px (row y0, col x0)) with each of `shapes` ((n, 4, 2) int32 page px corners)
    that reaches it filled with its `values` entry, in order (later over earlier), and a disc of `radius` around each
    of `discs` ((p, 2) int32 page px) filled with 1; 0 elsewhere.  Filled as cv2 fills them on a page large enough
    that none is cut at its edge: on a canvas holding the window and every shape that reaches it."""
    dtype = np.uint16 if len(values) == 0 or int(values.max()) < 2 ** 16 else np.int32
    lo, hi = shapes.min(axis=1), shapes.max(axis=1)
    hit = np.flatnonzero((hi[:, 0] >= x0) & (lo[:, 0] < x0 + w) & (hi[:, 1] >= y0) & (lo[:, 1] < y0 + h))
    discs = np.zeros((0, 2), np.int32) if discs is None else discs
    near = np.flatnonzero((discs[:, 0] + radius >= x0) & (discs[:, 0] - radius < x0 + w)
                          & (discs[:, 1] + radius >= y0) & (discs[:, 1] - radius < y0 + h))
    if not len(hit) and not len(near):
        return np.zeros((h, w), dtype)
    los = np.concatenate([lo[hit], discs[near] - radius, [[x0, y0]]])
    his = np.concatenate([hi[hit] + 1, discs[near] + radius + 1, [[x0 + w, y0 + h]]])
    cx0, cy0 = los.min(axis=0)
    cx1, cy1 = his.max(axis=0)
    canvas = np.zeros((cy1 - cy0, cx1 - cx0), dtype)
    corner = np.array([cx0, cy0], np.int32)
    for j in hit:
        cv2.fillPoly(canvas, [(shapes[j] - corner).reshape(-1, 1, 2)], int(values[j]))
    for x, y in discs[near] - corner:
        cv2.circle(canvas, (int(x), int(y)), radius, 1, -1)
    return canvas[y0 - cy0:y0 - cy0 + h, x0 - cx0:x0 - cx0 + w]


def in_validation(sd: SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Which px of a page window (h x w px, top-left px (row y0, col x0)) have their map point in VALIDATION_AREA,
    bool; whether or not the sheet holds it out.  A window whose placement samples all lie on one side of the area's
    edge, or all well inside it, is answered from them alone (a px's map point is a weighted mean of four samples)."""
    ax0, ay0, ax1, ay1 = VALIDATION_AREA
    step, (rows, cols) = PLACEMENT_STEP, sd.world.shape[:2]
    r0, r1 = min(y0 // step, rows - 2), min((y0 + h - 1) // step, rows - 2) + 1
    c0, c1 = min(x0 // step, cols - 2), min((x0 + w - 1) // step, cols - 2) + 1
    sx, sy = sd.world[r0:r1 + 1, c0:c1 + 1, 0], sd.world[r0:r1 + 1, c0:c1 + 1, 1]
    margin = 1.0                   # ft: far more than a rounding of the weighted mean can move it
    if (sx < ax0 - margin).all() or (sx >= ax1 + margin).all() or (sy < ay0 - margin).all() \
            or (sy >= ay1 + margin).all():
        return np.zeros((h, w), bool)
    if ((sx >= ax0 + margin) & (sx < ax1 - margin) & (sy >= ay0 + margin) & (sy < ay1 - margin)).all():
        return np.ones((h, w), bool)
    x, y = _world_at(sd.world, np.arange(y0, y0 + h), np.arange(x0, x0 + w), step)
    return (x >= ax0) & (x < ax1) & (y >= ay0) & (y < ay1)


def _held(sd: SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The held-out px of a page window: those in VALIDATION_AREA when the sheet holds it out, else none."""
    return in_validation(sd, y0, x0, h, w) if sd.hold_out else np.zeros((h, w), bool)


def _words(sd: SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The page window's words: 1 + the index of the last word drawn on each px (0 on none)."""
    return _paint(sd.words, np.arange(1, len(sd.words) + 1), y0, x0, h, w)


def _unknown(sd: SheetData, word: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The unknown px of a page window, given its words (`_words`): within READ_POINT_RADIUS of one of Sol's points
    or in another pass's box, but in no word."""
    other = _paint(sd.other, np.ones(len(sd.other), np.int32), y0, x0, h, w, sd.points, READ_POINT_RADIUS)
    return (other > 0) & (word == 0)


def _known(sd: SheetData, word: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The px of a page window the loss reads, given its words (`_words`): not unknown, not held out."""
    return ~_unknown(sd, word, y0, x0, h, w) & ~_held(sd, y0, x0, h, w)


def page_text(sd: SheetData) -> tuple[np.ndarray, np.ndarray]:
    """A sheet's whole page of text px (in one of Terra's words) and of unknown px, bool each, as the detector learns
    them (`crop_targets`), whether or not the sheet holds the area out: for its scores."""
    word = _words(sd, 0, 0, sd.height, sd.width)
    return word > 0, _unknown(sd, word, 0, 0, sd.height, sd.width)


def crop_targets(sd: SheetData, y: int, x: int, size: int, scale: float, turns: int) -> dict[str, np.ndarray]:
    """The training crop centred on page px (row y, col x), moved inside the page: size / `scale` px of page resized
    to `size` x `size` (nearest px) and turned `turns` quarter turns counter-clockwise, the reading angle with it (in
    [-180, 180) degrees on text, 0 elsewhere).  Keys: ink (bool), text (uint8), kind (uint8), angle (int16), known
    (bool); drawn from the sheet's boxes for this crop alone."""
    src = int(round(size / scale))
    y0 = int(np.clip(y - src // 2, 0, sd.height - src))
    x0 = int(np.clip(x - src // 2, 0, sd.width - src))
    word = _words(sd, y0, x0, src, src)
    parts = {"ink": _ink_window(sd, y0, x0, src, src), "word": word,
             "known": _known(sd, word, y0, x0, src, src)}
    if src != size:
        parts = {k: cv2.resize(v.view(np.uint8) if v.dtype == bool else v, (size, size),
                               interpolation=cv2.INTER_NEAREST).astype(v.dtype) for k, v in parts.items()}
    word = parts.pop("word")
    # each word's kind and turned reading angle, looked up by word (index 0: no word)
    kind = np.concatenate([[NO_KIND], sd.kinds]).astype(np.uint8)
    angle = np.concatenate([[0], (sd.angles.astype(np.int32) + 90 * turns + 180) % 360 - 180]).astype(np.int16)
    out = {**parts, "text": (word > 0).view(np.uint8), "kind": kind[word], "angle": angle[word]}
    return {k: np.rot90(v, turns) for k, v in out.items()}


def centres(sd: SheetData, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Where a training crop may centre on a sheet: every `stride`-th page px each way (rows and columns 0, stride,
    2 stride, ...) on text, and on ink, that is not held out.  Two int32 (n, 2) arrays of (row, col); the page drawn
    BAND rows at a time."""
    on_text, on_ink = [], []
    band = max(stride, BAND - BAND % stride)
    w = sd.width
    for y0 in range(0, sd.height, band):
        h = min(band, sd.height - y0)
        free = ~_held(sd, y0, 0, h, w)[::stride, ::stride]
        for pool, mask in ((on_text, _words(sd, y0, 0, h, w)[::stride, ::stride] > 0),
                           (on_ink, _ink_window(sd, y0, 0, h, w)[::stride, ::stride])):
            r, c = np.nonzero(mask & free)
            pool.append(np.c_[y0 + r * stride, c * stride].astype(np.int32))
    return np.concatenate(on_text), np.concatenate(on_ink)
