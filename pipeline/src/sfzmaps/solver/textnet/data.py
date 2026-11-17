"""The text detector's data: each sheet's own page (`page.png`, the scan before any warp), with what it learns.

  ink        page < `grid.INK_BELOW` (128), the model's only input
  text       1 inside one of Terra's text boxes (every kind), drawn as the rotated rectangle of the word in page px
  kind       the box's kind (KINDS: 0 zoning code, 1 block number, 2 street name, 3 other), NO_KIND outside text
  angle      the reading direction in whole degrees, counter-clockwise from left-to-right, inside text (0 outside)
  unknown    within the sheet's `point_radius` (READ_POINT_RADIUS, unless the sheet was read with another) of one of
             Sol's label points or, on the 2000 sheets, inside another pass's box (OTHER_PASSES; the other passes read
             only those), but in no Terra box: text Terra may have missed, taught neither as text nor as not-text
  held out   the px whose map point, through the sheet's corrected placement, is in one of the areas the sheet holds
             out (`SheetData.held`): VALIDATION_AREA for a trial (`VALIDATION`); a quarter of the city (`Quarter`) and
             the answered blocks (`GridArea`) for the code reader's detectors (`reader_stage`); none for the stage's
  known      the px the loss reads: not unknown and not held out; under the label rule "lines-unknown" also not a
             px inside a word that is a line px (`SheetData.line_bits`: the run rule's line ink grown by 1 px)
  centres    the px a training crop may centre on: ink or text, not held out

The label rule (`crop_targets(..., labels=)`, LABEL_RULES): "boxes" teaches every px of a word's rectangle as text,
the lines through it too; "lines-unknown" leaves the line px inside a word unknown (no lesson either way).  It needs
the sheet's line px, which `textnet.zones.prepare` fills before training, with the zones it keeps around Sol's points.

The word targets (`crop_targets(..., words=True)`, for the word heads `train.Recipe.heads="words"`), drawn from each
word's core (`SheetData.cores`: the word's own labelled rectangle, no shrink, which `wordread.shrink.prepare` fills
before training):
  core       1 on the px of a word's core (filled as the words are, later over earlier), 0 elsewhere
  tag_id     1 + the index of the word whose core holds the px (int32), 0 elsewhere
  geo        float32 (5, h, w), at core px only (0 elsewhere), the word's geometry with no sign and no side order:
               0, 1  dx, dy: the offset from the px to the centre of the word's whole rectangle (`SheetData.words`),
                     in the crop's px (x right, y down), divided by the word's height: free of the crop's scale;
                     a quarter turn of the crop turns it as a vector, (dx, dy) to (dy, -dx)
               2, 3  the log of the word's length and of its height in crop px (page px scaled by the crop's scale;
                     the turns leave them): its extents along its reading axis and across it, each the furthest
                     of the rectangle's corners that way, by its reading angle (`SheetData.angles`)
               4     its axis in degrees, in [0, 180): the reading angle modulo 180, turned with the crop
             All are continuous in the word's angle: a word at 179 and one at 181 degrees have nearly the same.  A
             word whose rectangle holds a held-out px has no geometry: its first four are 0 at its core px, as they
             would tell how far it reaches into the held-out area (its axis is kept: it is read off the word's px
             outside the area too)
  geo_known  1 at the core px of a word whose geometry is known (its rectangle holds no held-out px), 0 elsewhere:
             the px the loss reads geo's first four at; which words reach into a held-out area is found once a sheet
  weight     float32, only when the sheet holds word weights (`SheetData.weights`, Terra's doubt, which
             `wordread.doubt.prepare` fills before training): at a core px its word's weight, 1 elsewhere.  With none,
             every word weighs 1 and the key is absent (the targets are those without weights, byte for byte).

An area answers for map points whether each is in it (`contains`) and, from the placement's samples around a page
window, whether all of the window's px are in it or none is, when that follows from the samples alone (`decided`): a
px's map point is a weighted mean of four samples, so it lies in their bounding box.  Only where it doesn't follow is
each px's map point found (`in_area`).

The reading angle is kept in degrees, as Terra gives it; the model learns its cosine and sine.

The detector of every map (`load_sheets`, `crop_targets`) learns from the sheets Terra read (`paths.TEXT_SHEETS`) of
the training maps, Terra's boxes decoded from its saved answers (`boxes.terra_boxes`), and holds out VALIDATION_AREA
on every map at once.  A sheet is kept compact (`SheetData`): its ink as packed bits and its boxes, points and
placement as short arrays; the targets are drawn for each crop from them, never for a whole page.  A shape is filled
as `cv2.fillPoly` / `cv2.circle` fill it on a page large enough that none is cut at its edge, so a crop's targets are
those of the whole page, cropped.

A sheet's placement (`data/warp/<series>/<sheet>/transform.json`, and the correction it names) is read for the
areas it holds out and for the sheet's scale (`ft_per_px`, the scores' feet); a detector trained on the whole of every
map needs neither, and reads its sheets without it (`load_sheets(..., placement=False)`).
"""
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sfzmaps import config
from sfzmaps.solver import grid, paths, saved
from sfzmaps.solver.textnet import boxes
from sfzmaps.solver.textnet.boxes import TYPICAL_HEIGHT, word_rect  # noqa: F401  (the walls read them from boxes)
from sfzmaps.transform import SheetTransform

KINDS = {"zoning code": 0, "block number": 1, "street name": 2, "other": 3}
NO_KIND = 255
OTHER_PASSES = "experiments-2000/text-positions"   # the saved answers holding the other passes' boxes, page px
READ_POINT_RADIUS = 30            # px around one of Sol's label points treated as unknown (unless read with another)
PLACEMENT_STEP = 32               # px between the page points where the placement itself is evaluated
AREA_MARGIN = 1.0                 # ft: far more than a rounding of a px's weighted mean of samples can move its point
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
BOX_FIELDS = ("words", "kinds", "angles", "other", "points", "cores", "weights")  # SheetData's arrays of boxes, points
LOAD_THREADS = 8                  # sheets read at once
BAND = 512                        # page rows drawn at a time when finding where crops may centre
LABEL_RULES = ("boxes", "lines-unknown")   # what a word box teaches (`crop_targets`; see the module)


# --- areas of the map a sheet may hold out ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MapRect:
    """A rectangle of the map (EPSG:2227 ft): a map point (x, y) is in it when x0 <= x < x1 and y0 <= y < y1."""
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.x0) & (x < self.x1) & (y >= self.y0) & (y < self.y1)

    def decided(self, sx: np.ndarray, sy: np.ndarray) -> bool | None:
        """From the samples around a window: False when they all lie on one side of an edge, True when they all lie
        well inside, else None (see the module)."""
        m = AREA_MARGIN
        if (sx < self.x0 - m).all() or (sx >= self.x1 + m).all() or (sy < self.y0 - m).all() \
                or (sy >= self.y1 + m).all():
            return False
        if ((sx >= self.x0 + m) & (sx < self.x1 - m) & (sy >= self.y0 + m) & (sy < self.y1 - m)).all():
            return True
        return None


VALIDATION = MapRect(*VALIDATION_AREA)     # the trials' hold-out (`load_sheets(..., hold_out_validation=True)`)


@dataclass(frozen=True)
class Quarter:
    """One quarter of the city by the solver's split (`grid.quarter_of`: nw 0, ne 1, sw 2, se 3)."""
    q: int

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return grid.quarter_of(np.stack([x, y], -1)).reshape(np.shape(x)) == self.q

    def decided(self, sx: np.ndarray, sy: np.ndarray) -> bool | None:
        """From the samples around a window: False when they all lie across a split line from the quarter, True when
        they all lie well inside it, else None."""
        m = AREA_MARGIN
        sides = []
        for v, split, high in ((sx, grid.QUARTER_SPLIT[0], self.q % 2 == 1), (sy, grid.QUARTER_SPLIT[1], self.q < 2)):
            if (v >= split + m).all():
                sides.append(high)
            elif (v < split - m).all():
                sides.append(not high)
            else:
                sides.append(None)
        if False in sides:
            return False
        return True if all(sides) else None


@dataclass(frozen=True, eq=False)
class GridArea:
    """Px of the solver's grid (`grid.to_grid`; grid px (col, row) covers [col, col + 1) x [row, row + 1)): a map
    point is in the area when the grid px holding it is True in `mask`, a window of the grid whose top-left px is
    (col0, row0); no point off the window is.  One area may be shared by many sheets."""
    mask: np.ndarray        # bool (rows, cols)
    col0: int = 0
    row0: int = 0

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        g = grid.to_grid(np.stack([x, y], -1))
        col = np.floor(g[..., 0]).astype(np.int64) - self.col0
        row = np.floor(g[..., 1]).astype(np.int64) - self.row0
        h, w = self.mask.shape
        on = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        out = np.zeros(np.shape(x), bool)
        out[on] = self.mask[row[on], col[on]]
        return out

    def decided(self, sx: np.ndarray, sy: np.ndarray) -> bool | None:
        """From the samples around a window: False when no grid px under their bounding box (grown by AREA_MARGIN) is
        in the area, True when every one is, else None."""
        m = AREA_MARGIN
        (c0, r0), (c1, r1) = np.floor(grid.to_grid([[sx.min() - m, sy.max() + m], [sx.max() + m, sy.min() - m]]))
        c0, r0, c1, r1 = int(c0) - self.col0, int(r0) - self.row0, int(c1) - self.col0 + 1, int(r1) - self.row0 + 1
        h, w = self.mask.shape
        k0, j0, k1, j1 = max(c0, 0), max(r0, 0), min(c1, w), min(r1, h)
        if k0 >= k1 or j0 >= j1:
            return False
        under = self.mask[j0:j1, k0:k1]
        if not under.any():
            return False
        return True if (k0, j0, k1, j1) == (c0, r0, c1, r1) and under.all() else None


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
    world: np.ndarray        # float64 (rows, cols, 2): the map point of every PLACEMENT_STEP-th page px each way;
                             # (0, 0, 2) when read without the placement
    ft_per_px: float         # the placement's scale over the page (`SheetTransform.median_scale_ft_per_px`); nan
                             # without the placement
    held: tuple = ()         # the areas held out (MapRect, Quarter, GridArea): no loss there, no crop centred there
    point_radius: int = READ_POINT_RADIUS   # px around each of Sol's points that are unknown (outside every word)
    line_bits: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 0), np.uint8))
                             # uint8, packed as `ink_bits`: the line px (`linerules.line_px` grown by 1 px, the 8
                             # around each), for the label rule "lines-unknown"; (0, 0) until `zones.prepare` fills it
    cores: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros((0, 4, 2), np.int32))
                             # int32 (n, 4, 2): each word's core (its own labelled rectangle, no shrink, in the order
                             # of `words`), for the word targets; none until `wordread.shrink.prepare` fills it
    weights: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0, np.float32))
                             # float32 (n,): each word's weight in the word heads' loss (Terra's doubt, in the order of
                             # `words`); none (the default) is every word weighing 1; `wordread.doubt.prepare` fills it


def reading_angle(rect: np.ndarray) -> int:
    """A word rectangle's reading direction (its first edge) in whole degrees, counter-clockwise from left-to-right."""
    dx, dy = rect[1] - rect[0]
    return int(round(np.rad2deg(np.arctan2(-dy, dx))))


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


def load_page(sheet: str, series: str = paths.SERIES) -> np.ndarray:
    """A sheet's page (data/work/<series>/<sheet>/page.png), grayscale uint8.  `page.load_page` reads the same bytes for
    the walls' text grid, which must not import this module."""
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


def _other_files() -> list[Path]:
    files = sorted((paths.SAVED_ANSWERS / OTHER_PASSES).glob("*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"no saved answers under {paths.SAVED_ANSWERS / OTHER_PASSES}")
    return files


def other_boxes(sheet: str) -> list[dict]:
    """The boxes the other text passes found on a sheet (every file of OTHER_PASSES, by name, in file order)."""
    return [b for f in _other_files() for b in saved.by_sheet(f"{OTHER_PASSES}/{f.name.removesuffix('.jsonl.gz')}",
                                                               sheet)]


# --- every map, compact, with areas held out -------------------------------------------------------------------------

def _corners(rects: list[np.ndarray]) -> np.ndarray:
    """Word rectangles' corners rounded to page px, int32 (n, 4, 2)."""
    return np.round(np.array(rects, float).reshape(-1, 4, 2)).astype(np.int32)


def sheet_data(series: str, sheet: str, hold_out_validation: bool, terra: list[dict] | None = None,
               others: list[dict] | None = None, placement: bool = True,
               point_radius: int = READ_POINT_RADIUS) -> SheetData:
    """One sheet's SheetData (see the module), holding VALIDATION out or nothing.  `terra`: lines of Terra's saved
    answers holding the sheet's (read from disk when None); `others`: the other passes' boxes of the sheet (read from
    disk when None, on a 2000 sheet; none on another map's); `placement`: read the sheet's placement (needed to hold
    the validation area out: ValueError without it); `point_radius`: the px around Sol's points that are unknown."""
    if hold_out_validation and not placement:
        raise ValueError("the validation area is held out by the sheets' placements: read them (placement=True)")
    gray = load_page(sheet, series)
    if placement:
        transform = load_transform(sheet, series)
        world = placement_samples(transform, gray.shape)
        ft_per_px = float(transform.median_scale_ft_per_px(gray.shape))
    else:
        world, ft_per_px = np.zeros((0, 0, 2)), float("nan")
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
        world=world, ft_per_px=ft_per_px, held=(VALIDATION,) if hold_out_validation else (),
        point_radius=int(point_radius))


def load_sheets(series_list, hold_out_validation: bool, placement: bool = True,
                point_radius: int = READ_POINT_RADIUS) -> list[SheetData]:
    """Every sheet Terra read (`paths.TEXT_SHEETS`) of the maps in `series_list`, map by map in that order and in the
    sheets' order, with VALIDATION_AREA held out or not, with their placements or not, and with `point_radius` px
    around Sol's points unknown (`sheet_data`); LOAD_THREADS sheets at a time.  Stops (ValueError) when the sheet
    lists differ from the saved answers (`boxes.check_text_sheets`) or a map has none."""
    if hold_out_validation and not placement:
        raise ValueError("the validation area is held out by the sheets' placements: read them (placement=True)")
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
                          others.get(name, []) if s == paths.SERIES else [], placement, point_radius)

    with ThreadPoolExecutor(LOAD_THREADS) as pool:
        return list(pool.map(one, jobs))


def hold_out(sheets: list[SheetData], areas) -> list[SheetData]:
    """The sheets holding `areas` out (MapRect, Quarter, GridArea), in place of what they held; everything else, the
    arrays too, shared with `sheets`.  A sheet read without its placement can hold nothing out (ValueError)."""
    areas = tuple(areas)
    bare = [f"{sd.series} {sd.sheet}" for sd in sheets if areas and not sd.world.size]
    if bare:
        raise ValueError(f"areas are held out by the sheets' placements: {bare[0]} was read without its placement")
    return [dataclasses.replace(sd, held=areas) for sd in sheets]


def page_ink(sd: SheetData) -> np.ndarray:
    """A sheet's whole page of ink, bool (height, width)."""
    return np.unpackbits(sd.ink_bits, axis=1, count=sd.width).view(bool)


def _bits_window(packed: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The px of a page packed 8 a byte along each row (as `ink_bits`), in the window of h x w px whose top-left px is
    (row y0, col x0), bool."""
    b0 = x0 // 8
    bits = np.unpackbits(packed[y0:y0 + h, b0:(x0 + w + 7) // 8], axis=1)
    return bits[:, x0 - 8 * b0:x0 - 8 * b0 + w].view(bool)


def _ink_window(sd: SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The ink of the page window of h x w px whose top-left px is (row y0, col x0), bool."""
    return _bits_window(sd.ink_bits, y0, x0, h, w)


def line_window(sd: SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The line px (`SheetData.line_bits`) of the page window of h x w px whose top-left px is (row y0, col x0), bool.
    ValueError when the sheet holds none (`zones.prepare` fills them)."""
    if sd.line_bits.shape != sd.ink_bits.shape:
        raise ValueError(f"{sd.series} {sd.sheet}: no line px (SheetData.line_bits {sd.line_bits.shape}, the ink "
                         f"{sd.ink_bits.shape}): the label rule lines-unknown needs them (textnet.zones.prepare)")
    return _bits_window(sd.line_bits, y0, x0, h, w)


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


def _area_px(sd: SheetData, area, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Which of the page px at `rows` x `cols` (ascending px indices) have their map point in `area`, bool (len(rows),
    len(cols)): answered from the placement samples around them when that decides it (`area.decided`), else px by
    px (`area.contains` of `_world_at`)."""
    if not sd.world.size:
        raise ValueError(f"{sd.series} {sd.sheet} was read without its placement: which of its px are in a held-out "
                         "area is not known")
    if not len(rows) or not len(cols):
        return np.zeros((len(rows), len(cols)), bool)
    step, (n_rows, n_cols) = PLACEMENT_STEP, sd.world.shape[:2]
    r0, r1 = min(int(rows[0]) // step, n_rows - 2), min(int(rows[-1]) // step, n_rows - 2) + 1
    c0, c1 = min(int(cols[0]) // step, n_cols - 2), min(int(cols[-1]) // step, n_cols - 2) + 1
    decided = area.decided(sd.world[r0:r1 + 1, c0:c1 + 1, 0], sd.world[r0:r1 + 1, c0:c1 + 1, 1])
    if decided is not None:
        return np.full((len(rows), len(cols)), decided)
    x, y = _world_at(sd.world, rows, cols, step)
    return area.contains(x, y)


def in_area(sd: SheetData, area, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Which px of a page window (h x w px, top-left px (row y0, col x0)) have their map point in `area` (MapRect,
    Quarter, GridArea), bool; whether or not the sheet holds it out (see the module).  ValueError for a sheet read
    without its placement."""
    return _area_px(sd, area, np.arange(y0, y0 + h), np.arange(x0, x0 + w))


def in_validation(sd: SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """Which px of a page window (h x w px, top-left px (row y0, col x0)) have their map point in VALIDATION_AREA,
    bool; whether or not the sheet holds it out (`in_area` of VALIDATION)."""
    return in_area(sd, VALIDATION, y0, x0, h, w)


def _held_px(sd: SheetData, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Which of the page px at `rows` x `cols` are held out: in one of the areas the sheet holds out."""
    out = np.zeros((len(rows), len(cols)), bool)
    for area in sd.held:
        out |= _area_px(sd, area, rows, cols)
    return out


def _held(sd: SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The held-out px of a page window: those in one of the areas the sheet holds out (none when it holds none)."""
    return _held_px(sd, np.arange(y0, y0 + h), np.arange(x0, x0 + w))


def _words(sd: SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The page window's words: 1 + the index of the last word drawn on each px (0 on none)."""
    return _paint(sd.words, np.arange(1, len(sd.words) + 1), y0, x0, h, w)


def _unknown(sd: SheetData, word: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The unknown px of a page window, given its words (`_words`): within the sheet's `point_radius` of one of Sol's
    points or in another pass's box, but in no word."""
    other = _paint(sd.other, np.ones(len(sd.other), np.int32), y0, x0, h, w, sd.points, sd.point_radius)
    return (other > 0) & (word == 0)


def _known(sd: SheetData, word: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The px of a page window the loss reads, given its words (`_words`): not unknown, not held out."""
    return ~_unknown(sd, word, y0, x0, h, w) & ~_held(sd, y0, x0, h, w)


def page_text(sd: SheetData) -> tuple[np.ndarray, np.ndarray]:
    """A sheet's whole page of text px (in one of Terra's words) and of unknown px, bool each, as the detector learns
    them (`crop_targets`), whether or not the sheet holds the area out: for its scores."""
    word = _words(sd, 0, 0, sd.height, sd.width)
    return word > 0, _unknown(sd, word, 0, 0, sd.height, sd.width)


def _cores(sd: SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The page window's word cores: 1 + the index of the last word whose core is drawn on each px (0 on none), int32.
    ValueError when the sheet holds no core for each of its words (`wordread.shrink.prepare` fills them)."""
    if sd.cores.shape != sd.words.shape:
        raise ValueError(f"{sd.series} {sd.sheet}: no word cores (SheetData.cores {sd.cores.shape}, the words "
                         f"{sd.words.shape}): the word targets need them (wordread.shrink.prepare)")
    return _paint(sd.cores, np.arange(1, len(sd.cores) + 1), y0, x0, h, w).astype(np.int32)


def _touch_held(sd: SheetData, words: np.ndarray) -> np.ndarray:
    """Which of the sheet's `words` (indices) have a px of their rectangle (as `_paint` fills it) in an area the sheet
    holds out, bool; none when it holds none."""
    out = np.zeros(len(words), bool)
    if not sd.held:
        return out
    for n, j in enumerate(words):
        (wx0, wy0), (wx1, wy1) = np.maximum(sd.words[j].min(0), 0), np.minimum(sd.words[j].max(0) + 1,
                                                                                [sd.width, sd.height])
        if wx1 <= wx0 or wy1 <= wy0:
            continue
        h, w = int(wy1 - wy0), int(wx1 - wx0)
        rect = _paint(sd.words[j:j + 1], np.ones(1, np.int32), int(wy0), int(wx0), h, w) > 0
        out[n] = bool((_held(sd, int(wy0), int(wx0), h, w) & rect).any())
    return out


def _held_words(sd: SheetData) -> np.ndarray:
    """Which of the sheet's words reach into an area it holds out (`_touch_held` of every word), bool (n,): found once
    per sheet and kept on it beside its fields (a sheet holding other areas is another SheetData, `hold_out`, and finds
    its own; kept with the areas and words it was found for, so it is found again if either is set anew)."""
    kept = getattr(sd, "_held_words_found", None)
    if kept is None or kept[0] is not sd.held or kept[1] is not sd.words:
        kept = (sd.held, sd.words, _touch_held(sd, np.arange(len(sd.words))))
        sd._held_words_found = kept
    return kept[2]


def _geometry(sd: SheetData, tag: np.ndarray, y0: int, x0: int) -> np.ndarray:
    """At each core px of a page window (`_cores`), its word's geometry at page scale, unturned (see the module): the
    offset from the px to the word's centre over the word's height (x, y), the log of the word's length and of its
    height in page px, and its axis in degrees, in [0, 180); float32 (h, w, 5), 0 off the cores.  A word whose
    rectangle holds a held-out px (`_held_words`) has its first four 0, its axis kept."""
    out = np.zeros((*tag.shape, 5), np.float32)
    rows, cols = np.nonzero(tag)
    if not len(rows):
        return out
    j = tag[rows, cols] - 1
    theta = np.deg2rad(sd.angles[j].astype(float))
    along = np.stack([np.cos(theta), -np.sin(theta)], -1)       # the reading axis (page px, y down)
    across = np.stack([-np.sin(theta), -np.cos(theta)], -1)     # across it
    corners = sd.words[j].astype(float)                         # (m, 4, 2)
    a, b = (corners * along[:, None]).sum(-1), (corners * across[:, None]).sum(-1)
    centre = (along * ((a.max(-1) + a.min(-1)) / 2)[:, None] + across * ((b.max(-1) + b.min(-1)) / 2)[:, None])
    # a word under 1 px long or high is no word on a scan (the rectangles are at least 2 px each way): the floor
    # only keeps the log and the division defined
    length, height = np.maximum(a.max(-1) - a.min(-1), 1.0), np.maximum(b.max(-1) - b.min(-1), 1.0)
    offset = (centre - np.stack([x0 + cols, y0 + rows], -1)) / height[:, None]
    out[rows, cols, :4] = np.stack([offset[:, 0], offset[:, 1], np.log(length), np.log(height)], -1)
    out[rows, cols, 4] = np.mod(sd.angles[j], 180)
    unseen = _held_words(sd)[j]
    out[rows[unseen], cols[unseen], :4] = 0
    return out


def _turn_offsets(dx: np.ndarray, dy: np.ndarray, turns: int) -> tuple[np.ndarray, np.ndarray]:
    """An image offset (x right, y down) after the picture is turned `turns` quarter turns counter-clockwise
    (`np.rot90`): each turn takes (dx, dy) to (dy, -dx)."""
    for _ in range(turns % 4):
        dx, dy = dy, -dx
    return dx, dy


def crop_targets(sd: SheetData, y: int, x: int, size: int, scale: float, turns: int,
                 labels: str = "boxes", words: bool = False) -> dict[str, np.ndarray]:
    """The training crop centred on page px (row y, col x), moved inside the page: size / `scale` px of page resized
    to `size` x `size` (nearest px) and turned `turns` quarter turns counter-clockwise, the reading angle with it (in
    [-180, 180) degrees on text, 0 elsewhere).  Keys: ink (bool), text (uint8), kind (uint8), angle (int16), known
    (bool); drawn from the sheet's boxes for this crop alone, by the label rule `labels` (LABEL_RULES, see the
    module): under "lines-unknown" a px inside a word that is a line px is not known (ValueError when the sheet holds
    no line px).  With `words`, also the word targets core (uint8), tag_id (int32), geo (float32 (5, size, size)) and
    geo_known (uint8), from the sheet's cores (see the module; ValueError when it holds none for its words), and, when
    the sheet holds word weights, weight (float32; ValueError unless one weight per word, each a number at least 0)."""
    if labels not in LABEL_RULES:
        raise ValueError(f"labels is one of {', '.join(LABEL_RULES)}, not {labels!r}")
    if words and len(sd.weights) and (sd.weights.shape != (len(sd.words),) or not np.isfinite(sd.weights).all()
                                      or (sd.weights < 0).any()):
        raise ValueError(f"{sd.series} {sd.sheet}: SheetData.weights {sd.weights.shape} must be one number at least 0 "
                         f"per word ({len(sd.words)}; wordread.doubt.prepare)")
    src = int(round(size / scale))
    y0 = int(np.clip(y - src // 2, 0, sd.height - src))
    x0 = int(np.clip(x - src // 2, 0, sd.width - src))
    word = _words(sd, y0, x0, src, src)
    known = _known(sd, word, y0, x0, src, src)
    if labels == "lines-unknown":
        known &= ~((word > 0) & line_window(sd, y0, x0, src, src))
    parts = {"ink": _ink_window(sd, y0, x0, src, src), "word": word, "known": known}
    if words:
        parts["tag_id"] = _cores(sd, y0, x0, src, src)
        parts["geo"] = _geometry(sd, parts["tag_id"], y0, x0)
    if src != size:
        parts = {k: cv2.resize(v.view(np.uint8) if v.dtype == bool else v, (size, size),
                               interpolation=cv2.INTER_NEAREST).astype(v.dtype) for k, v in parts.items()}
    word = parts.pop("word")
    # each word's kind and turned reading angle, looked up by word (index 0: no word)
    kind = np.concatenate([[NO_KIND], sd.kinds]).astype(np.uint8)
    angle = np.concatenate([[0], (sd.angles.astype(np.int32) + 90 * turns + 180) % 360 - 180]).astype(np.int16)
    out = {**parts, "text": (word > 0).view(np.uint8), "kind": kind[word], "angle": angle[word]}
    if not words:
        return {k: np.rot90(v, turns) for k, v in out.items()}
    geo = out.pop("geo")
    on = out["tag_id"] > 0
    known_geo = np.concatenate([[0], ~_held_words(sd)]).astype(np.uint8)[out["tag_id"]]
    geo[..., 0], geo[..., 1] = _turn_offsets(geo[..., 0], geo[..., 1], turns)     # over the height: no scale
    geo[..., 2:4] += np.where(known_geo > 0, np.float32(np.log(size / src)), np.float32(0))[..., None]
    geo[..., 4] = np.where(on, np.mod(geo[..., 4] + 90 * turns, 180), 0)
    out["core"] = on.view(np.uint8)
    out["geo"] = np.moveaxis(geo, -1, 0)
    out["geo_known"] = known_geo
    if len(sd.weights):
        out["weight"] = np.concatenate([[1], sd.weights]).astype(np.float32)[out["tag_id"]]
    return {k: np.rot90(v, turns, axes=(1, 2) if k == "geo" else (0, 1)) for k, v in out.items()}


def centres(sd: SheetData, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Where a training crop may centre on a sheet: every `stride`-th page px each way (rows and columns 0, stride,
    2 stride, ...) on text, and on ink, that is not held out.  Two int32 (n, 2) arrays of (row, col); the page drawn
    BAND rows at a time."""
    on_text, on_ink = [], []
    band = max(stride, BAND - BAND % stride)
    w = sd.width
    for y0 in range(0, sd.height, band):
        h = min(band, sd.height - y0)
        free = ~_held_px(sd, np.arange(y0, y0 + h, stride), np.arange(0, w, stride))     # = _held(...)[::s, ::s]
        for pool, mask in ((on_text, _words(sd, y0, 0, h, w)[::stride, ::stride] > 0),
                           (on_ink, _ink_window(sd, y0, 0, h, w)[::stride, ::stride])):
            r, c = np.nonzero(mask & free)
            pool.append(np.c_[y0 + r * stride, c * stride].astype(np.int32))
    return np.concatenate(on_text), np.concatenate(on_ink)
