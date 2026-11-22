"""The learned reader's training data: each sheet's scan as printed, today's blocks placed on it, the solver lot under
every page px, and the lot table (the old reader's reads and the owner's answers).

Per sheet (`sheet_rasters`), in the scan's own px, turned upright:

  ink        the scan's luminance in [0, 1], 0 = ink (`load_luminance`: a grey scan's value / its full scale, a colour
             scan's Rec. 601 luma), every px exactly as scanned: nothing erased, cleaned, masked or thresholded
  block      the target of the learned street output: 1 where the px lies in today's blocks (the solver's lots with
             their enclosed holes filled: `binary_fill_holes(lot >= 0)` on the whole grid), 0 where it doesn't (a
             street, or no lot), 255 (unknown) outside the sheet's frame cut and on the area another sheet owns (R9,
             below), so each place is labelled once
  street     the street labels (`street_labels`, on the solver's grid, carried to the page as `block` is): 1 street,
             where the px is off today's blocks, within the learned half-widths of its nearest centre line and short of
             that side's block; 0 not street, on today's blocks (exactly where `block` is 1); 255 ignored, everything
             else (the Bay and ocean past the learned half-widths, land in no parcel, 1921 streets with no centre line
             today; a lake inside a parcel, as Lake Merced, is a block), and wherever `block` is 255
  street_w   float16, each label's weight in the street loss: 1 on not street; on street S(d) = P(half-width > d) of
             the street type's learned half-widths at the px's distance d from its centre line; 0 where ignored
  centre     1 on a centre line: today's centre lines (`src.streets`, data/ref/streets.parquet, every row) drawn on the
             page through the placement (`page_centres`), kept on street px of this sheet's own area
  lot        the solver lot index of the px (`lots.npz` `lot`, 4 ft grid px), -1 where none and wherever `block` is
             255 (each lot's px are pooled on one sheet only)
  crop       the frame cut: the page px the frame finder keeps (`sfzmaps frame`, `hatching.load_crop`), the map
             inside the printed frame less the legend and title blocks, compass roses and badges
  ft_per_px  the placement's median scale (`SheetTransform.median_scale_ft_per_px`): measured on the placement, never
             on the reads

The placement is `data/work/<series>/<sheet>/transform.json` (the one the reads were made through; the warp's correction
in data/warp bends straight print), read as the merged page reads a sheet: `hatching.PageWorld` (the placement every
LATTICE px, bilinear between) and `hatching.page_values`.  Which sheet owns a px (ruling R9, `owned_px`): a place inside
several sheets' frame cuts belongs to the sheet that prints it.  On one of today's blocks (`block_ids`: each
connected piece of the filled lots), each sheet whose frame cut holds some of the block is measured by its mean ink over
the block's px on its page inside its frame cut (`BlockInk`: 1 - luminance), and every px of the block goes to the sheet
with the most ink that holds it (argmax, no threshold; a tie to the sheet drawn higher); a block on one sheet keeps
it.  A px on no block (streets, water) keeps the draw-order rule: the sheet drawn highest (the published draw order,
`stitch.draw_order`) whose frame cut holds it, as `hatching.owned` says for points.  A px is carried to another sheet's
page by that sheet's inverse placement at the lattice nodes, bilinear between (the same lattice).  A sheet scanned on
its side (`hatching.paper_turn`) is turned back by its quarter turn, which is lossless (`turn_of`); every raster is
computed in the scan's own px and turned with it.

The lot table (`lot_table`), one row per solver lot:

  lot, blklot, block, quarter (nw / ne / sw / se, `paths.QUARTERS`), centre_sheet (the sheet that owns the lot's
  centre: its representative point, `shapely.point_on_surface`, by the rule the px follow (R9, `centre_owner`); ""
  none)
  read_code, read_conf, read_via, read_sheet, read_blklot, read_recovered   the pattern reader's read
            (`district_pattern.json`; its off-sheet and no-tile marks and empty codes are no reads), one per lot
            (`attach_reads`, `one_read_per_lot`):
              - a read whose blklot is a solver lot is that lot's (every row of the blklot);
              - R4: a read of a parcel the solver's April 2000 table lacks (today's blklots: condo and subdivision ids)
                lands on the solver lot under its parcel's representative point (`shapely.point_on_surface` of its
                polygon in data/ref/parcels.parquet, DataSF's parcels; `hatching.grid_values` on `lots.npz` `lot`), no
                distance allowed; a read whose parcel has no polygon there or whose point is on no solver lot stays
                unattached, and is counted (`recover_lots`);
              - several reads of a lot from one sheet: the most confident (ties: a read of its own blklot first);
              - R3': a lot read on one sheet keeps that read, whichever sheet it is; read on several, the draw order
                picks: the read of `centre_sheet` when it read the lot, else that of the sheet drawn highest.
            read_blklot is the read's own blklot (today's, for a recovered read)
  n_reads   how many sheets read the lot (before keeping one)
  answer, answer_page, answer_case, stratum, n_answers   the owner's answer (truth by `answers1921.owner_answers`:
            ok the code shown, fix the fix, unzoned 2R, canttell / unsure dropped) on the lot of its blklot, else of the
            first lot of its footprint that is a solver lot (`answers1921`'s footprint matching); stratum random or
            disagree (the random page's cases), likely or lowconf (the pages').  A lot answered twice keeps the random
            page's answer; two answers that disagree stop the run.

Writes, under data/solver/experiments/hatch-cnn/data/<series>/ (`run`):
  <sheet>.npz     ink, block, street, street_w, centre, lot, crop, ft_per_px, turn (np.rot90 k), scanned_shape, sheet,
                  series
  lots.parquet    the lot table, with px: the lot's owned page px over every sheet
  summary.json    the sheets used and not, the draw order, per-sheet and per-quarter counts, reads per code, the
                  lots read on several sheets, answers per page and stratum, the legend swatches, the street labels'
                  counts on the grid and the learned half-widths per street type (`street_grid`), and the blocks R9
                  gives to a sheet other than the one drawn highest (`ownership_counts`)

$0: no API calls.  Deterministic: the same inputs give the same bytes (the npz members are dated, as `textread.store`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely
from PIL import Image
from scipy import ndimage

from sfzmaps import config
from sfzmaps.solver import answers1921, grid, hatching, paths
from sfzmaps.solver.codes import canon
from sfzmaps.solver.textread.store import save_npz
from sfzmaps.transform import SheetTransform

EXPERIMENT = config.DATA_DIR / "solver" / "experiments" / "hatch-cnn"
LOOK = EXPERIMENT / "look"
SKIP_VIA = hatching.SKIP_VIA            # the reader's marks for a lot not on the sheet: no reads
LATTICE = hatching.LATTICE              # page px between the nodes where a placement is evaluated
UNKNOWN = 255                           # `block` where the place is not this sheet's to label
LUMA = (0.299, 0.587, 0.114)            # Rec. 601 luma (R, G, B): a colour scan's grey, as PIL and OpenCV take it
ROWS = 512                              # page rows laid at once (memory only: the result doesn't depend on it)
ANSWER_PAGES = {"1921": answers1921.PAGES}     # the owner's answer pages per series (none yet for the others)


@dataclass(frozen=True)
class Sources:
    """Where a series is read: the pages, placements and reads (work), the solver's lots (solver), the legend's
    swatch boxes, the owner's answers and their case files, today's parcels (None: reads of parcels the solver's table
    lacks are not placed); `order`: the draw order, bottom first (None: the published map's); today's street centre
    lines (None: none, so every px off today's blocks is ignored); the frame finder's cuts (`sfzmaps frame`; None:
    data/frame/<series>)."""
    series: str
    work: Path
    solver: Path
    legends: Path
    answers: Path
    cases: Path
    parcels: Path | None = None
    order: tuple[str, ...] | None = None
    streets: Path | None = None
    frame: Path | None = None

    @property
    def frames(self) -> Path:
        return self.frame if self.frame is not None else config.DATA_DIR / "frame" / self.series


def sources(series: str) -> Sources:
    return Sources(series, config.WORK_DIR / series, config.DATA_DIR / "solver" / series, hatching.LEGENDS_FILE,
                   answers1921.ANSWERS_DIR, answers1921.EXPERIMENT / "cases", config.REF_DIR / "parcels.parquet",
                   streets=config.REF_DIR / "streets.parquet")


def data_dir(series: str) -> Path:
    return EXPERIMENT / "data" / series


# --- the scan -------------------------------------------------------------------------------------------------------

def load_luminance(path: Path) -> np.ndarray:
    """A page as scanned: float32 luminance in [0, 1], 0 = ink.  A grey scan's value over its full scale (1-bit and
    8-bit 1921 scans alike); a colour scan's (1948) Rec. 601 luma (`LUMA`, in double precision, banded for memory);
    an alpha channel is not part of the print."""
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(f"can't read {path}")
    if a.dtype.kind != "u":
        raise ValueError(f"{path}: {a.dtype} px, expected unsigned integers")
    full = float(np.iinfo(a.dtype).max)
    if a.ndim == 2:
        return a.astype(np.float32) / np.float32(full)
    if a.shape[2] <= 2:                                   # grey (+ alpha)
        return a[..., 0].astype(np.float32) / np.float32(full)
    out = np.empty(a.shape[:2], np.float32)
    for y0 in range(0, a.shape[0], ROWS):
        b, g, r = (a[y0:y0 + ROWS, :, k].astype(np.float64) for k in range(3))     # OpenCV reads BGR
        out[y0:y0 + ROWS] = (LUMA[0] * r + LUMA[1] * g + LUMA[2] * b) / full
    return out


def load_transform(src: Sources, sheet: str) -> SheetTransform:
    """The sheet's placement in data/work (not data/warp)."""
    folder = src.work / sheet
    return SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)


def page_shape(src: Sources, sheet: str) -> tuple[int, int]:
    """(rows, cols) of the sheet's page.png as scanned, from its header."""
    with Image.open(src.work / sheet / "page.png") as im:
        w, h = im.size
    return h, w


def turn_of(crop: np.ndarray, transform, shape: tuple[int, int]) -> int:
    """The `np.rot90` k that turns a scan upright: 0 for a sheet scanned upright; for one scanned on its side
    (`hatching.paper_turn`: its frame cut wider than tall), the quarter turn, anticlockwise (1: its right edge comes to
    the top) or clockwise (3: its left edge), that brings the sheet's north nearer the top, as on the upright sheets
    (from its placement at the page's centre)."""
    if hatching.paper_turn(crop) == 0:
        return 0
    h, w = shape
    c = np.array([[(w - 1) / 2.0, (h - 1) / 2.0]])
    right = (transform.to_world(c + [LATTICE, 0.0]) - transform.to_world(c))[0]

    def off_north(v):
        return abs(float(np.arctan2(v[0], v[1])))

    return 1 if off_north(right) <= off_north(-right) else 3


def turn_box(box, shape: tuple[int, int], k: int) -> tuple[int, int, int, int]:
    """A box (x0, y0, x1, y1, ends exclusive) on a page of `shape` (rows, cols) as scanned, on the page turned by
    `np.rot90(page, k)`."""
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = shape
    k %= 4
    if k == 0:
        return x0, y0, x1, y1
    if k == 1:
        return y0, w - x1, y1, w - x0
    if k == 2:
        return w - x1, h - y1, w - x0, h - y0
    return h - y1, x0, h - y0, x1


def load_scan(series: str, sheet: str, src: Sources | None = None) -> np.ndarray:
    """The sheet's page as scanned (`load_luminance`), turned upright (`turn_of`)."""
    src = src or sources(series)
    lum = load_luminance(src.work / sheet / "page.png")
    k = turn_of(hatching.load_crop(src.frames, sheet, lum.shape), load_transform(src, sheet), lum.shape)
    return np.ascontiguousarray(np.rot90(lum, k))


# --- the sheets -----------------------------------------------------------------------------------------------------

@dataclass
class Sheet:
    """A sheet's geometry in its scanned px: the page shape, the placement, the frame cut and the cut's world bounds
    (ft)."""
    name: str
    shape: tuple[int, int]
    transform: SheetTransform
    crop: np.ndarray
    bbox: tuple[float, float, float, float]

    def above(self) -> hatching.Above:
        return hatching.Above(self.transform, self.crop, self.bbox)


def load_sheet(src: Sources, name: str) -> Sheet:
    shape = page_shape(src, name)
    tr = load_transform(src, name)
    crop = hatching.load_crop(src.frames, name, shape)
    return Sheet(name, shape, tr, crop, hatching.crop_bbox(tr, crop))


def sheet_names(src: Sources) -> list[str]:
    """The map sheets: a page, a placement (data/work) and a frame cut (`sfzmaps frame`), less the duplicate scans
    (`hatching.duplicate_scans`: one scan per section)."""
    skip = hatching.duplicate_scans(src.series)
    return [d.name for d in sorted(p for p in src.work.iterdir() if p.is_dir())
            if (d / "page.png").is_file() and (d / "transform.json").is_file() and _has_cut(src, d.name)
            and d.name not in skip]


def _has_cut(src: Sources, sheet: str) -> bool:
    return (src.frames / sheet / hatching.FRAME_CUT).is_file()


def not_used(src: Sources) -> dict[str, list[str]]:
    """The series' folders that are not map sheets, with what they lack."""
    out = {}
    for d in sorted(p for p in src.work.iterdir() if p.is_dir()):
        miss = [f for f in ("page.png", "transform.json") if not (d / f).is_file()]
        if not _has_cut(src, d.name):
            miss.append("frame")
        if d.name in hatching.duplicate_scans(src.series):
            miss.append("a duplicate scan")
        if miss:
            out[d.name] = miss
    return out


def draw_order(src: Sources, sheets: list[str]) -> list[str]:
    """The sheets bottom first, as the published map draws them (`stitch.draw_order`, or `src.order`); any it doesn't
    rank at the bottom, by name."""
    if src.order is not None:
        ranked = [s for s in src.order if s in sheets]
    else:
        from sfzmaps import stitch
        ranked = [s for s in stitch.draw_order(src.series, stitch.load_scores(src.series)) if s in sheets]
    return sorted(set(sheets) - set(ranked)) + ranked


def overlaps(a: Sheet, b: Sheet) -> bool:
    return a.bbox[0] <= b.bbox[2] and a.bbox[2] >= b.bbox[0] and a.bbox[1] <= b.bbox[3] and a.bbox[3] >= b.bbox[1]


def overlapping(name: str, order: list[str], sheets: dict[str, Sheet]) -> list[Sheet]:
    """The other sheets, drawn below or above `name`, whose frame cuts' bounds meet its own (bottom first)."""
    me = sheets[name]
    return [sheets[s] for s in order if s != name and s in sheets and overlaps(me, sheets[s])]


# --- which sheet owns a place (ruling R9) ---------------------------------------------------------------------------

def block_ids(blocks: np.ndarray) -> tuple[np.ndarray, int]:
    """Today's blocks numbered (int32): each connected piece of `blocks` (4-connected, `ndimage.label`) 1..n, 0 off
    them; and n."""
    bid, n = ndimage.label(np.asarray(blocks, bool))
    return bid.astype(np.int32, copy=False), int(n)


@dataclass(frozen=True)
class BlockInk:
    """R9's measure: each sheet's ink over each of today's blocks, from the block's px on its page inside its frame cut
    (`crop`): their count and the sum of their ink, 1 - luminance (paper 0, solid ink 1; every 1921 scan's paper reads
    1.0).  Rows: the sheets in `order` (the draw order, bottom first), a sheet not measured all 0; columns: block ids
    (`block_ids`), 0 off the blocks."""
    order: tuple[str, ...]
    px: np.ndarray                 # (sheets, blocks + 1) int64
    ink: np.ndarray                # (sheets, blocks + 1) float64

    def level(self, sheet: str) -> int:
        """The sheet's place in the draw order (higher: drawn above)."""
        return self.order.index(sheet)

    def key(self, sheet: str) -> np.ndarray:
        """Per block id: the sheet's mean ink over the block's px; -inf where it has none of them, and off the blocks
        (0), where only the draw order decides."""
        i = self.level(sheet)
        out = np.full(self.px.shape[1], -np.inf)
        has = self.px[i] > 0
        has[0] = False
        out[has] = self.ink[i][has] / self.px[i][has]
        return out

    def owners(self) -> np.ndarray:
        """Per block id, the level of the sheet it goes to among those with px of it (`key`'s argmax, ties to the one
        drawn highest); -1 where no sheet has any."""
        keys = np.stack([self.key(s) for s in self.order]) if self.order else np.zeros((0, self.px.shape[1]))
        out = np.full(self.px.shape[1], -1, np.int64)
        best = np.full(self.px.shape[1], -np.inf)
        for i in range(len(self.order)):                    # bottom first: a tie goes to the sheet drawn above
            take = (self.px[i] > 0) & (keys[i] >= best)
            out[take], best[take] = i, keys[i][take]
        return out


def beats(ink: BlockInk, other: str, me: str, bid: np.ndarray) -> np.ndarray:
    """Where `other` takes a place both sheets' frame cuts hold from `me` (R9), at block ids `bid`: on a block, the
    sheet with more ink over it (`BlockInk.key`); equal, and off the blocks, the one drawn higher."""
    ko, km = ink.key(other)[bid], ink.key(me)[bid]
    return (ko > km) | ((ko == km) & (ink.level(other) > ink.level(me)))


def sheet_block_ink(src: Sources, sh: Sheet, bid_cut: np.ndarray, offset, n_blocks: int) -> tuple[np.ndarray,
                                                                                                    np.ndarray]:
    """One sheet's row of `BlockInk`: per block id, its px on the page inside the frame cut (the block ids read at
    every page px through the placement, as `block` is) and their summed ink (1 - luminance, float64)."""
    lum = load_luminance(src.work / sh.name / "page.png")
    if lum.shape != tuple(sh.shape):
        raise ValueError(f"{sh.name}: page {lum.shape}, header {sh.shape}")
    (b,) = hatching.page_values(hatching.PageWorld(sh.transform, sh.shape), sh.shape, [(bid_cut, 0)], offset)
    m = np.asarray(sh.crop, bool) & (b > 0)
    v = b[m]
    return (np.bincount(v, minlength=n_blocks + 1).astype(np.int64),
            np.bincount(v, weights=1.0 - lum[m].astype(np.float64), minlength=n_blocks + 1))


def _ink_job(job):
    src, name, bid_cut, offset, n_blocks = job
    return (name, *sheet_block_ink(src, load_sheet(src, name), bid_cut, offset, n_blocks))


def block_ink(src: Sources, order: list[str], bid: np.ndarray, n_blocks: int, workers: int = 1,
              names: list[str] | None = None) -> BlockInk:
    """`BlockInk` of the sheets in `order` (or only `names` of them; the rest stay unmeasured), `bid` the block ids
    on the whole grid."""
    want = list(order) if names is None else [s for s in order if s in set(names)]
    jobs = []
    for s in want:
        r0, r1, c0, c1 = hatching._slice(load_transform(src, s), _cut(src, s), bid.shape)
        jobs.append((src, s, bid[r0:r1, c0:c1], (r0, c0), n_blocks))
    px = np.zeros((len(order), n_blocks + 1), np.int64)
    ink = np.zeros((len(order), n_blocks + 1), np.float64)
    for name, p, k in _map(_ink_job, jobs, workers):
        i = list(order).index(name)
        px[i], ink[i] = p, k
    return BlockInk(tuple(order), px, ink)


def ownership_counts(ink: BlockInk) -> dict:
    """For the summary: today's blocks, those with px on several sheets, and those R9 gives to a sheet other than the
    highest-drawn of them (per pair: the sheet it goes to < the one the draw order gave it)."""
    has = ink.px[:, 1:] > 0
    several = has.sum(axis=0) > 1
    top = np.where(has.any(axis=0), len(ink.order) - 1 - np.argmax(has[::-1], axis=0), -1)
    win = ink.owners()[1:]
    moved = several & (win != top)
    pairs: dict[str, int] = {}
    for w, t in zip(win[moved], top[moved]):
        k = f"{ink.order[w]} < {ink.order[t]}"
        pairs[k] = pairs.get(k, 0) + 1
    return {"rule": "R9: a block's px go to the sheet with the most ink over it; px in no block by the draw order",
            "blocks": int(ink.px.shape[1] - 1), "blocks_on_any_sheet": int(has.any(axis=0).sum()),
            "blocks_on_several_sheets": int(several.sum()), "blocks_not_to_the_top_sheet": int(moved.sum()),
            "blocks_not_to_the_top_sheet_by_pair": dict(sorted(pairs.items()))}


# --- the street labels ----------------------------------------------------------------------------------------------

STREET, NOT_STREET = 1, 0               # `street`'s labels; UNKNOWN (255) is ignored
SUBPIXEL_BITS = 4                       # line points go to cv2 to 1/16 px (as `lots.lot_rasters` fills the lots)
KIND_NONE = "(none)"                    # the street type of a line with none (freeways, The Embarcadero, ...)
POOLED = "(pooled)"                     # the half-widths of every street type together
CONFIDENCE = 0.95                       # a street type keeps its own half-widths only when their distribution differs
                                        # from all types' by more than the DKW band at this confidence for its number
                                        # of lines (the one setting the learning needs: the usual 95%)


@dataclass(frozen=True)
class Lines:
    """Today's street centre lines: each one's geometry (EPSG:2227 ft), street type and street."""
    geoms: np.ndarray
    kinds: np.ndarray
    streets: np.ndarray


def _type_from_name(name, known: set[str]) -> str:
    """A line with no `st_type`: its name's last word when that is a type other lines have (OCTAVIA ST -> ST), else
    KIND_NONE (freeways, THE EMBARCADERO, BROADWAY)."""
    last = name.split()[-1] if isinstance(name, str) and name.split() else ""
    return last if last in known else KIND_NONE


def centre_lines(src: Sources) -> Lines:
    """Today's street centre lines (`src.streets`: DataSF's, data/ref/streets.parquet, EPSG:2227 ft as the solver's
    grid), every row: the ones no longer active too, since 1921's streets are older than today's list.  Each one's
    street type (`st_type`; where missing, `_type_from_name`) and street (`streetname`; a line with none is its own
    street).  None when `src.streets` is None."""
    if src.streets is None:
        empty = np.empty(0, dtype=object)
        return Lines(empty, empty, empty)
    df = pd.read_parquet(src.streets)
    geoms = np.array([shapely.from_wkb(v) if isinstance(v, (bytes, bytearray)) else v for v in df["geometry"]],
                     dtype=object)
    n = len(df)
    kinds = df["st_type"].astype(object).to_numpy() if "st_type" in df else np.full(n, None, dtype=object)
    names = df["streetname"].astype(object).to_numpy() if "streetname" in df else np.full(n, None, dtype=object)
    known = {k for k in kinds if isinstance(k, str) and k}
    kinds = np.array([k if isinstance(k, str) and k else _type_from_name(nm, known) for k, nm in zip(kinds, names)],
                     dtype=object)
    streets = np.array([s if isinstance(s, str) and s else f"#{i}" for i, s in enumerate(names)], dtype=object)
    keep = ~(shapely.is_missing(geoms) | shapely.is_empty(geoms)) if n else np.zeros(0, bool)
    return Lines(geoms[keep], kinds[keep], streets[keep])


def _paths(lines) -> list[np.ndarray]:
    """The vertices (n, 2) of each line part (points and other parts left out)."""
    parts = shapely.get_parts(np.asarray(lines, dtype=object))
    parts = parts[shapely.get_type_id(parts) == 1]                     # LineString
    if not len(parts):
        return []
    xy, idx = shapely.get_coordinates(parts, return_index=True)
    return np.split(xy, np.flatnonzero(np.diff(idx)) + 1)


def _draw_paths(shape: tuple[int, int], paths: list[np.ndarray]) -> np.ndarray:
    """Paths (each (n, 2) x, y in px, px centres at integers) as 8-connected 1 px lines (cv2, to 1/16 px), bool."""
    out = np.zeros(shape, np.uint8)
    scale = 1 << SUBPIXEL_BITS
    pts = [np.rint(p * scale).astype(np.int32).reshape(-1, 1, 2) for p in paths if len(p)]
    if pts:
        cv2.polylines(out, pts, False, 1, 1, cv2.LINE_8, shift=SUBPIXEL_BITS)
    return out.astype(bool)


@dataclass(frozen=True)
class Centres:
    """Today's centre lines on the solver's grid, segment by segment (two consecutive vertices of a line), each
    8-connected and 1 px wide (cv2, to 1/16 px), a later segment over an earlier one.  Grid px (c, r) covers [c, c+1)
    x [r, r+1) and cv2 takes integer coordinates as px centres, so the vertices go in at `to_grid - 0.5`.  A line off
    the grid draws nothing."""
    seg: np.ndarray            # int32 grid: the segment of each centre px, -1 none
    direction: np.ndarray      # (n, 2): each segment's unit direction on the grid (col, row)
    line: np.ndarray           # (n,): its line (index into the lines given)
    street: np.ndarray         # (n,): its street (index into the distinct streets)
    kind: np.ndarray           # (n,): its street type (index into `kinds`)
    kinds: tuple[str, ...]


def centre_segments(lines, shape: tuple[int, int], kinds=None, streets=None) -> Centres:
    """`Centres` of `lines` (EPSG:2227 ft) on a grid of `shape` from the solver grid's corner; `kinds`: each line's
    street type (KIND_NONE when None), `streets`: each line's street (each line its own when None)."""
    lines = np.asarray(lines, dtype=object)
    n = len(lines)
    kinds = np.full(n, KIND_NONE, dtype=object) if kinds is None else np.asarray(kinds, dtype=object)
    streets = np.arange(n).astype(str) if streets is None else np.asarray(streets, dtype=object)
    kind_names, kind_of = np.unique(kinds.astype(str), return_inverse=True) if n else (np.array([KIND_NONE]),
                                                                                       np.zeros(0, np.int64))
    street_of = np.unique(streets.astype(str), return_inverse=True)[1] if n else np.zeros(0, np.int64)
    seg = np.full(shape, -1, np.int32)
    none = np.zeros(0, np.int64)
    if n:
        parts, part_line = shapely.get_parts(lines, return_index=True)
        is_line = shapely.get_type_id(parts) == 1
        parts, part_line = parts[is_line], part_line[is_line]
    if not n or not len(parts):
        return Centres(seg, np.zeros((0, 2)), none, none, none, tuple(str(k) for k in kind_names))
    xy, part = shapely.get_coordinates(parts, return_index=True)
    g = grid.to_grid(xy) - 0.5
    same = part[1:] == part[:-1]
    a, b, line = g[:-1][same], g[1:][same], part_line[part[:-1][same]]
    v = b - a
    length = np.hypot(v[:, 0], v[:, 1])
    ok = length > 0
    a, b, v, length, line = a[ok], b[ok], v[ok], length[ok], line[ok]
    scale = 1 << SUBPIXEL_BITS
    pa, pb = np.rint(a * scale).astype(np.int64), np.rint(b * scale).astype(np.int64)
    for k in range(len(a)):
        cv2.line(seg, (int(pa[k, 0]), int(pa[k, 1])), (int(pb[k, 0]), int(pb[k, 1])), int(k), 1, cv2.LINE_8,
                 SUBPIXEL_BITS)
    return Centres(seg, v / length[:, None], line, street_of[line], kind_of[line], tuple(str(k) for k in kind_names))


def data_area(src: Sources, names: list[str], shape: tuple[int, int]) -> np.ndarray:
    """Where the series' sheets hold data, on the grid: the union of their frame cuts, each cut's outlines (outer and
    inner: a deleted block inside the frame is no data) carried through its sheet's placement at every boundary px and
    filled (cv2, to 1/16 px)."""
    area = np.zeros(shape, np.uint8)
    scale = 1 << SUBPIXEL_BITS
    for s in names:
        cs, _ = cv2.findContours(_cut(src, s).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        tr = load_transform(src, s)
        rings = []
        for c in cs:
            w = grid.to_grid(tr.to_world(c.reshape(-1, 2).astype(float))) - 0.5
            w = w[np.isfinite(w).all(axis=1)]
            if len(w) >= 3:
                rings.append(np.rint(w * scale).astype(np.int32))
        if rings:
            one = np.zeros(shape, np.uint8)
            cv2.fillPoly(one, rings, 1, cv2.LINE_8, SUBPIXEL_BITS)
            area |= one
    return area.astype(bool)


def _cut(src: Sources, sheet: str) -> np.ndarray:
    """The sheet's frame cut (`hatching.load_crop`), at its page's shape."""
    return hatching.load_crop(src.frames, sheet, page_shape(src, sheet))


@dataclass(frozen=True)
class Sides:
    """The centre px off today's blocks and each one's two half-widths (`side_widths`)."""
    rows: np.ndarray
    cols: np.ndarray
    seg: np.ndarray            # each one's segment
    width: np.ndarray          # (n, 2) grid px: side 0 along the normal (-dy, dx), side 1 against it; inf: unbounded


def _normals(direction: np.ndarray) -> np.ndarray:
    return np.c_[-direction[:, 1], direction[:, 0]]


def side_widths(block: np.ndarray, centres: Centres, area: np.ndarray | None, near: np.ndarray, rows: np.ndarray,
                cols: np.ndarray) -> np.ndarray:
    """Each centre px's half-width on each side (n, 2), in grid px: a ray from the px along its segment's normal, one
    px along its major axis per step, to the first block px it meets.  The side is unbounded (inf) when the ray first
    leaves the grid or the data `area` (None: the grid), or enters px nearer another street's centre line than its own
    (`near`: the nearest centre px of every px, `distance_transform_edt` indices): past there the ray runs along or
    into another street (down a cross street at a junction, across a plaza, out over water to another street), and a
    block it meets measures that street, not this one's width."""
    h, w = block.shape
    sid = centres.seg[rows, cols]
    own = centres.street[sid]
    normal = _normals(centres.direction[sid])
    width = np.full((len(rows), 2), np.inf)
    start = np.arange(len(rows)) if area is None else np.flatnonzero(area[rows, cols])
    for side, sign in enumerate((1.0, -1.0)):
        major = np.abs(normal).max(axis=1)
        step = sign * normal / major[:, None]
        act, k = start, 0
        while act.size:
            k += 1
            x = np.rint(cols[act] + k * step[act, 0]).astype(np.int64)
            y = np.rint(rows[act] + k * step[act, 1]).astype(np.int64)
            on = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x, y = np.where(on, x, 0), np.where(on, y, 0)
            if area is not None:
                on &= area[y, x]
            hit = on & block[y, x]
            width[act[hit], side] = k / major[act[hit]]
            same = centres.street[centres.seg[near[0][y, x], near[1][y, x]]] == own[act]
            act = act[on & ~hit & same]
    return width


@dataclass(frozen=True)
class HalfWidths:
    """The half-widths learned from the centre px bounded on both sides (grid px, sorted): every type's together
    (`pooled`) and the types that keep their own; `report` per type."""
    pooled: np.ndarray
    own: dict
    report: dict

    def of(self, kind: str) -> np.ndarray:
        return self.own.get(kind, self.pooled)


def survival(samples: np.ndarray, d) -> np.ndarray:
    """S(d) = P(half-width > d): the share of `samples` (sorted) beyond d; 0 with no samples.  Strictly beyond, so a
    px at d is street as often as a street's block lies past it: a side whose block is at w labels px below w."""
    d = np.asarray(d, float)
    if not len(samples):
        return np.zeros(d.shape)
    return (len(samples) - np.searchsorted(samples, d, side="right")) / len(samples)


def _ecdf_gap(a: np.ndarray, b: np.ndarray) -> float:
    """sup |F_a - F_b| of two sorted samples (at their values, where the step functions jump)."""
    v = np.union1d(a, b)
    return float(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b)).max())


def _ft(s: np.ndarray) -> dict:
    if not len(s):
        return {"median_ft": None, "p90_ft": None, "max_ft": None}
    f = s * grid.FT_PER_PX
    return {"median_ft": float(np.quantile(f, 0.5)), "p90_ft": float(np.quantile(f, 0.9)), "max_ft": float(f[-1])}


def learn_half_widths(width: np.ndarray, sid: np.ndarray, centres: Centres) -> HalfWidths:
    """The half-widths of the centre px bounded on both sides, both sides' (the empirical distribution: nothing
    smoothed).  Per street type: its own when its two-sided px come from lines enough to tell it apart, i.e. its
    empirical CDF differs from every type's together by more than the DKW band sqrt(ln(2 / (1 - CONFIDENCE)) /
    (2 n)), n its lines (the px of one line are not independent); else the pooled ones."""
    both = np.isfinite(width).all(axis=1)
    kind, line = centres.kind[sid], centres.line[sid]
    pooled = np.sort(width[both].ravel())
    own, report = {}, {POOLED: {"px": int(both.sum()), "lines": int(len(np.unique(line[both]))), **_ft(pooled)}}
    for k, name in enumerate(centres.kinds):
        m = both & (kind == k)
        s = np.sort(width[m].ravel())
        n = int(len(np.unique(line[m])))
        band = float(np.sqrt(np.log(2.0 / (1.0 - CONFIDENCE)) / (2.0 * n))) if n else None
        gap = _ecdf_gap(s, pooled) if len(s) else None
        keep = bool(band is not None and gap > band)
        if keep:
            own[name] = s
        report[name] = {"px": int(m.sum()), "lines": n, "own": keep, "ecdf_gap": gap, "dkw_band": band, **_ft(s)}
    return HalfWidths(pooled, own, report)


@dataclass(frozen=True)
class StreetLabels:
    street: np.ndarray         # uint8: 1 street, 0 not street, 255 ignored
    weight: np.ndarray         # float16: each label's weight (1 not street, (0, 1] street, 0 ignored)
    centre: np.ndarray         # uint8: 1 on the centre px used (those off today's blocks)
    sides: Sides
    learned: HalfWidths


def street_labels(block_world: np.ndarray, centres: Centres, area: np.ndarray | None = None,
                  learned: HalfWidths | None = None) -> StreetLabels:
    """The street labels on the solver's 4 ft grid.  Each px x off today's blocks belongs to its nearest centre px c
    (`distance_transform_edt` indices), at distance d, on the side of c's segment that x lies on, with that side's
    block distance w (`side_widths`; inf with no block).  On every side x is street with weight S(d) = P(half-width >
    d) (`survival`) of c's street type's learned half-widths (`learn_half_widths`, from this grid's two-sided px unless
    `learned` is given) while d < w: the side's own block cuts it, and a far block (across water or open land) never
    makes full-weight street, since S is ~0 there.  Ignored where S(d) is 0 (beyond the largest half-width learned)
    and past the side's block.  A weight, never a 1/2 target.  Today's blocks are not street (weight 1).  A centre px
    on a block (a park road, a street since built over) is no centre: it would label nothing off the block.  No width
    is picked: block distances are measured, the weights learned."""
    block = np.asarray(block_world, bool)
    h, w = block.shape
    kept = (centres.seg >= 0) & ~block
    street = np.where(block, NOT_STREET, UNKNOWN).astype(np.uint8)
    weight = block.astype(np.float16)
    rows, cols = np.nonzero(kept)
    sid = centres.seg[rows, cols]
    if not len(rows):
        sides = Sides(rows, cols, sid, np.zeros((0, 2)))
        learned = learned if learned is not None else learn_half_widths(sides.width, sid, centres)
        return StreetLabels(street, weight, kept.astype(np.uint8), sides, learned)
    near = ndimage.distance_transform_edt(~kept, return_distances=False, return_indices=True)
    width = side_widths(block, centres, area, near, rows, cols)
    sides = Sides(rows, cols, sid, width)
    learned = learned if learned is not None else learn_half_widths(width, sid, centres)
    index = np.full(block.shape, -1, np.int32)
    index[rows, cols] = np.arange(len(rows), dtype=np.int32)
    normal = _normals(centres.direction[sid])
    kind = centres.kind[sid]
    samples = [learned.of(name) for name in centres.kinds]
    xs = np.arange(w, dtype=np.int64)
    for y0 in range(0, h, ROWS):
        band = slice(y0, min(h, y0 + ROWS))
        ri, ci = near[0][band], near[1][band]
        i = index[ri, ci]
        vy = (np.arange(band.start, band.stop, dtype=np.int64)[:, None] - ri).astype(float)
        vx = (xs - ci).astype(float)
        side = ((vx * normal[i, 0] + vy * normal[i, 1]) < 0).astype(np.int64)
        half = width[i, side]
        d = np.sqrt(vx * vx + vy * vy)
        inside = d < half
        wt = np.zeros(d.shape)
        kk = np.where(inside, kind[i], -1)
        for k in np.unique(kk[inside]):
            m = kk == k
            wt[m] = survival(samples[k], d[m])
        off = ~block[band]
        street[band][off & (wt > 0)] = STREET
        weight[band][off] = wt[off].astype(np.float16)
    return StreetLabels(street, weight, kept.astype(np.uint8), sides, learned)


def street_grid(lines: Lines, blocks: np.ndarray, area: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray,
                                                                                            dict]:
    """The street labels and weights on the whole solver grid (`street_labels` of today's blocks and `lines`, rays
    kept to the data `area`) and their counts: lines, centre px (those on blocks dropped), the centre px bounded on
    both sides, one or none, px per label, the soft street px and the street weight's sum, and the learned
    half-widths per street type."""
    centres = centre_segments(lines.geoms, blocks.shape, lines.kinds, lines.streets)
    lab = street_labels(blocks, centres, area)
    fin = np.isfinite(lab.sides.width).sum(axis=1)
    on = lab.street == STREET
    stats = {"lines": int(len(lines.geoms)), "centre_grid_px": int(len(lab.sides.rows)),
             "centre_on_blocks_px": int(np.count_nonzero((centres.seg >= 0) & blocks)),
             "centre_sides": {"two_sided_px": int((fin == 2).sum()), "one_sided_px": int((fin == 1).sum()),
                              "unbounded_px": int((fin == 0).sum())},
             "street_grid_px": int(np.count_nonzero(on)),
             "soft_street_grid_px": int(np.count_nonzero(on & (lab.weight < 1))),
             "street_weight_sum_grid": float(lab.weight[on].astype(np.float64).sum()),
             "not_street_grid_px": int(np.count_nonzero(lab.street == NOT_STREET)),
             "ignored_grid_px": int(np.count_nonzero(lab.street == UNKNOWN)),
             "half_widths": lab.learned.report}
    return lab.street, lab.weight, stats


def page_centres(lines, sh: "Sheet") -> np.ndarray:
    """The centre lines on the sheet's page as scanned (bool), drawn there rather than read off the grid: a page px is
    about 5 ft on most 1921 sheets, wider than the grid's 4 ft line, so reading the grid's lines at page px centres (as
    `block` is read) breaks them (20180102094149543, 5.2 ft/px: 16,847 pieces read, 6 drawn; 99.4% of the drawn px
    within 1 px of a read one).  Each line near the sheet (its frame cut's world bounds and two page px beyond,
    as `centre_owner` reaches) is split every page px (`median_scale_ft_per_px`), carried through the placement's
    inverse (`grid_to_pixel`, as `owned_px` carries px to another sheet), and drawn 8-connected, 1 px wide; a point
    the inverse can't place breaks its line there."""
    h, w = sh.shape
    lines = np.asarray(lines, dtype=object)
    if not len(lines):
        return np.zeros((h, w), bool)
    step = float(sh.transform.median_scale_ft_per_px(sh.shape))
    reach = 2.0 * step
    x0, y0, x1, y1 = sh.bbox
    b = shapely.bounds(lines)
    near = (b[:, 0] <= x1 + reach) & (b[:, 2] >= x0 - reach) & (b[:, 1] <= y1 + reach) & (b[:, 3] >= y0 - reach)
    if not near.any():
        return np.zeros((h, w), bool)
    cut = shapely.segmentize(shapely.clip_by_rect(lines[near], x0 - reach, y0 - reach, x1 + reach, y1 + reach), step)
    paths = _paths(cut)
    if not paths:
        return np.zeros((h, w), bool)
    n = np.array([len(p) for p in paths])
    px = sh.transform.grid_to_pixel(np.concatenate(paths), sh.shape)
    runs = []
    for p in np.split(px, np.cumsum(n)[:-1]):
        ok = np.isfinite(p).all(axis=1)
        if ok.all():
            runs.append(p)
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
        runs += [p[a:b] for a, b in zip(edges[::2], edges[1::2])]
    return _draw_paths((h, w), runs)


# --- the rasters ----------------------------------------------------------------------------------------------------

@dataclass
class SheetRasters:
    """A sheet's training rasters, turned upright (see the module)."""
    sheet: str
    ink: np.ndarray
    block: np.ndarray
    lot: np.ndarray
    crop: np.ndarray
    ft_per_px: float
    turn: int = 0
    scanned_shape: tuple[int, int] = (0, 0)
    street: np.ndarray | None = None
    centre: np.ndarray | None = None
    street_w: np.ndarray | None = None


def solver_grid(src: Sources) -> tuple[np.ndarray, np.ndarray]:
    """The solver's lot raster and today's blocks (the lots with their enclosed holes, filled on the whole grid)."""
    with np.load(src.solver / "lots" / "lots.npz") as z:
        lot = z["lot"]
    return lot, ndimage.binary_fill_holes(lot >= 0)


def _bilinear(nodes: np.ndarray, pw: hatching.PageWorld, rows: slice, w: int) -> np.ndarray:
    """Values (rows x w x 2) at page px of the lattice `nodes` (at `pw`'s node px), bilinear; NaN where any of a px's
    four nodes is."""
    yy, xx = np.mgrid[rows, 0:w].astype(float)
    fx, fy = (xx - pw.nx0) / LATTICE, (yy - pw.ny0) / LATTICE
    j = np.clip(np.floor(fx).astype(np.int64), 0, nodes.shape[1] - 2)
    i = np.clip(np.floor(fy).astype(np.int64), 0, nodes.shape[0] - 2)
    ax, ay = (fx - j)[..., None], (fy - i)[..., None]
    return ((nodes[i, j] * (1 - ax) + nodes[i, j + 1] * ax) * (1 - ay)
            + (nodes[i + 1, j] * (1 - ax) + nodes[i + 1, j + 1] * ax) * ay)


def owned_px(pw: hatching.PageWorld, shape: tuple[int, int], crop: np.ndarray, me: str, others: list[Sheet],
             ink: BlockInk, bid: np.ndarray) -> tuple[np.ndarray, int, dict]:
    """The page px a sheet labels (R9): inside its frame cut (`crop`), and taken by no other sheet whose frame cut also
    holds the px (`others`, drawn below or above).  On one of today's blocks (`bid`: the block id at every page px,
    0 off them) the sheet with more ink over the block takes it (`beats`); off the blocks, and on a tie, the sheet
    drawn higher (the draw-order rule, `hatching.owned`'s).  Each px is carried to another sheet's page by that
    sheet's inverse placement (`grid_to_pixel`) at this page's lattice nodes, bilinear between.  Nodes beyond reach of
    the other sheet's frame cut bounds (twice the lattice's longest cell diagonal) aren't carried: no px of a cell they
    bound can lie in it.  Also returns the nodes the inverse placement could not place (their cells stay this
    sheet's) and the block px R9 decides against the draw order: kept from a sheet drawn above, or yielded to one
    drawn below."""
    h, w = shape
    ok = np.asarray(crop, bool).copy()
    by_order = ok.copy()                     # the draw order alone, for the counts
    nodes = pw.nodes
    if nodes.shape[0] < 2 or nodes.shape[1] < 2:
        raise ValueError(f"a page of {shape} px has too few lattice nodes")
    diag = max(float(np.linalg.norm(nodes[1:, 1:] - nodes[:-1, :-1], axis=-1).max()),
               float(np.linalg.norm(nodes[1:, :-1] - nodes[:-1, 1:], axis=-1).max()))
    reach = 2.0 * diag
    flat = nodes.reshape(-1, 2)
    failed = 0
    for t in others:
        x0, y0, x1, y1 = t.bbox
        cand = ((flat[:, 0] >= x0 - reach) & (flat[:, 0] <= x1 + reach) &
                (flat[:, 1] >= y0 - reach) & (flat[:, 1] <= y1 + reach))
        if not cand.any():
            continue
        P = np.full(flat.shape, np.nan)
        P[cand] = t.transform.grid_to_pixel(flat[cand], t.shape)
        failed += int((cand & ~np.isfinite(P).all(axis=1)).sum())
        P = P.reshape(nodes.shape)
        above = ink.level(t.name) > ink.level(me)
        for y0b in range(0, h, ROWS):
            rows = slice(y0b, min(h, y0b + ROWS))
            xy = _bilinear(P, pw, rows, w).reshape(-1, 2)
            held = hatching._in_crop(t.crop, xy).reshape(rows.stop - rows.start, w)
            ok[rows] &= ~(held & beats(ink, t.name, me, bid[rows]))
            if above:
                by_order[rows] &= ~held
    on = bid > 0
    moved = {"block_px_kept_from_above": int(np.count_nonzero(ok & ~by_order & on)),
             "block_px_yielded_to_below": int(np.count_nonzero(~ok & by_order & on))}
    return ok, failed, moved


def _rasters(src: Sources, sh: Sheet, lot_cut: np.ndarray, block_cut: np.ndarray, street_cut: np.ndarray,
             weight_cut: np.ndarray, bid_cut: np.ndarray, offset, others: list[Sheet], ink: BlockInk, lines):
    """A sheet's rasters (`SheetRasters`) and counts, from its cut of the solver's grid (`lot_cut`, `block_cut`,
    `street_cut`, `weight_cut`, `bid_cut` (`block_ids`) from grid row and column `offset`), the sheets overlapping it
    and the blocks' ink (`owned_px`), and today's centre lines (`lines`, geometries)."""
    lum = load_luminance(src.work / sh.name / "page.png")
    if lum.shape != tuple(sh.shape):
        raise ValueError(f"{sh.name}: page {lum.shape}, header {sh.shape}")
    pw = hatching.PageWorld(sh.transform, sh.shape)
    lot_px, blk_px, st_px, w_px, bid_px = hatching.page_values(
        pw, sh.shape, [(lot_cut, -1), (block_cut.astype(np.uint8), 0), (street_cut, UNKNOWN),
                       (weight_cut.astype(np.float16), np.float16(0)), (bid_cut, 0)], offset)
    mine, failed, moved = owned_px(pw, sh.shape, sh.crop, sh.name, others, ink, bid_px)
    del bid_px
    block = np.where(mine, blk_px, UNKNOWN).astype(np.uint8)
    street = np.where(mine, st_px, UNKNOWN).astype(np.uint8)
    street_w = np.where(mine, w_px, 0).astype(np.float16)
    centre = (page_centres(lines, sh) & (street == STREET)).astype(np.uint8)
    lot = np.where(mine, lot_px, -1).astype(np.int32)
    k = turn_of(sh.crop, sh.transform, sh.shape)

    def up(a):
        return np.ascontiguousarray(np.rot90(a, k))

    r = SheetRasters(sh.name, up(lum), up(block), up(lot), up(np.asarray(sh.crop, bool)),
                     float(sh.transform.median_scale_ft_per_px(sh.shape)), int(k), tuple(int(v) for v in sh.shape),
                     up(street), up(centre), up(street_w))
    stats = {"sheet": sh.name, "scanned_shape": list(r.scanned_shape), "shape": list(r.ink.shape), "turn": r.turn,
             "ft_per_px": r.ft_per_px,
             "above": [a.name for a in others if ink.level(a.name) > ink.level(sh.name)],
             "below": [a.name for a in others if ink.level(a.name) < ink.level(sh.name)], "px": int(lum.size),
             "crop_px": int(np.count_nonzero(sh.crop)), "owned_px": int(np.count_nonzero(mine)), **moved,
             "block_px": int(np.count_nonzero(block == 1)), "street_px": int(np.count_nonzero(block == 0)),
             "unknown_px": int(np.count_nonzero(block == UNKNOWN)), "lot_px": int(np.count_nonzero(lot >= 0)),
             "lots": int(len(np.unique(lot[lot >= 0]))), "unplaced_nodes": failed,
             "labels": {"street_px": int(np.count_nonzero(street == STREET)),
                        "not_street_px": int(np.count_nonzero(street == NOT_STREET)),
                        "ignored_owned_px": int(np.count_nonzero(mine & (street == UNKNOWN))),
                        "not_block_ignored_px": int(np.count_nonzero((block == 0) & (street == UNKNOWN))),
                        "centre_px": int(np.count_nonzero(centre)),
                        "soft_street_px": int(np.count_nonzero((street == STREET) & (street_w < 1))),
                        "street_weight_sum": float(street_w[street == STREET].astype(np.float64).sum())}}
    return r, stats


def sheet_rasters(series: str, sheet: str, src: Sources | None = None) -> SheetRasters:
    """One sheet's training rasters (see the module)."""
    src = src or sources(series)
    names = sheet_names(src)
    if sheet not in names:
        raise KeyError(f"{sheet} is not a {series} map sheet (a page, a data/work placement and a frame)")
    order = draw_order(src, names)
    sheets = {s: load_sheet(src, s) for s in order}
    lot, blocks = solver_grid(src)
    bid, n_blocks = block_ids(blocks)
    lines = centre_lines(src)
    street, weight, _ = street_grid(lines, blocks, data_area(src, names, lot.shape))
    me = sheets[sheet]
    others = overlapping(sheet, order, sheets)
    ink = block_ink(src, order, bid, n_blocks, names=[sheet] + [o.name for o in others])
    r0, r1, c0, c1 = hatching._slice(me.transform, me.crop, lot.shape)
    return _rasters(src, me, lot[r0:r1, c0:c1], blocks[r0:r1, c0:c1], street[r0:r1, c0:c1], weight[r0:r1, c0:c1],
                    bid[r0:r1, c0:c1], (r0, c0), others, ink, lines.geoms)[0]


def write_rasters(f: Path, r: SheetRasters, series: str) -> None:
    f.parent.mkdir(parents=True, exist_ok=True)
    save_npz(f, ink=r.ink, block=r.block, street=r.street, street_w=r.street_w, centre=r.centre, lot=r.lot,
             crop=r.crop, ft_per_px=np.float64(r.ft_per_px), turn=np.int64(r.turn),
             scanned_shape=np.array(r.scanned_shape, np.int64), sheet=np.array(r.sheet), series=np.array(series))


def read_rasters(f: Path) -> SheetRasters:
    with np.load(f) as z:
        labels = [z[k] if k in z.files else None for k in ("street", "centre", "street_w")]
        return SheetRasters(str(z["sheet"]), z["ink"], z["block"], z["lot"], z["crop"], float(z["ft_per_px"]),
                            int(z["turn"]), tuple(int(v) for v in z["scanned_shape"]), *labels)


# --- the legend -----------------------------------------------------------------------------------------------------

def swatches(series: str, src: Sources | None = None) -> list[tuple[str, str, tuple[int, int, int, int]]]:
    """The legend's swatches (`districts/legends.json`: the outer printed rectangle of each, ends exclusive, as
    `hatching` reads them), (sheet, code, box) on the sheet's page turned upright."""
    src = src or sources(series)
    leg = json.loads(Path(src.legends).read_text()).get(series)
    if not leg:
        return []
    sheet = leg["sheet"]
    shape = page_shape(src, sheet)
    k = turn_of(hatching.load_crop(src.frames, sheet, shape), load_transform(src, sheet), shape) \
        if _has_cut(src, sheet) else 0
    return [(sheet, code, turn_box(box, shape, k)) for code, box in leg["swatches"].items()]


# --- the lot table --------------------------------------------------------------------------------------------------

def centre_owner(points: np.ndarray, sheets: list[Sheet], blocks_at: np.ndarray | None = None,
                 ink: BlockInk | None = None) -> tuple[np.ndarray, int]:
    """The sheet that owns each map point (n, 2), `sheets` bottom first, by `owned_px`'s rule (R9): of the sheets whose
    frame cut holds it (its nearest px in the crop, as `hatching.owned` reads one), the one with the most ink over the
    point's block (`blocks_at`: the block id under each point, 0 none; `ink`); off the blocks, on a tie, or with no
    `ink`, the one drawn highest; "" none.  Also how many points several sheets held."""
    pts = np.asarray(points, float)
    owner = np.full(len(pts), "", dtype=object)
    best = np.full(len(pts), -np.inf)
    held_by = np.zeros(len(pts), np.int64)
    block = np.zeros(len(pts), np.int64) if blocks_at is None else np.asarray(blocks_at, np.int64)
    finite = np.isfinite(pts).all(axis=1)
    for sh in sheets:
        x0, y0, x1, y1 = sh.bbox
        margin = 2.0 * sh.transform.median_scale_ft_per_px(sh.shape)       # the crop's nearest px reaches 1/2 px out
        cand = np.flatnonzero(finite & (pts[:, 0] >= x0 - margin) & (pts[:, 0] <= x1 + margin)
                              & (pts[:, 1] >= y0 - margin) & (pts[:, 1] <= y1 + margin))
        if not len(cand):
            continue
        held = cand[hatching._in_crop(sh.crop, np.asarray(sh.transform.grid_to_pixel(pts[cand], sh.shape), float))]
        key = ink.key(sh.name)[block[held]] if ink is not None else np.full(len(held), -np.inf)
        take = key >= best[held]                            # bottom first: a tie goes to the sheet drawn above
        owner[held[take]], best[held[take]] = sh.name, key[take]
        held_by[held] += 1
    return owner, int((held_by > 1).sum())


def reads(src: Sources, sheets: list[str]) -> pd.DataFrame:
    """The pattern reader's reads on the sheets (sheet, blklot, code in the one spelling, confidence, via), less its
    off-sheet and no-tile marks and those with no code (as `hatchfit.reads`)."""
    rows = []
    for s in sheets:
        f = src.work / s / "district_pattern.json"
        if not f.is_file():
            continue
        for p in json.loads(f.read_text())["parcels"]:
            code = p.get("code")
            if not isinstance(code, str) or not code.strip() or p.get("via") in SKIP_VIA:
                continue
            rows.append({"sheet": s, "blklot": str(p["blklot"]), "code": canon(code),
                         "confidence": float(p["confidence"]), "via": str(p.get("via", ""))})
    return pd.DataFrame(rows, columns=["sheet", "blklot", "code", "confidence", "via"])


def _stratum(page: str, case: dict, series: str) -> str:
    """random or disagree (the random page's cases); otherwise the page's kind (likely, lowconf)."""
    s = case.get("stratum")
    if s and s != "all":
        return str(s)
    return page.removeprefix(f"{series}-") if page != series else page


def answers(src: Sources) -> pd.DataFrame:
    """The owner's answers kept (`answers1921.owner_answers`: blklot, block, truth, page, case, blklots) with each
    one's stratum; attrs as owner_answers'."""
    pages = ANSWER_PAGES.get(src.series, ())
    dirs = [src.answers / f"answers-{p}" for p in pages]
    files = [f for p in pages for f in (src.cases / f"{p}.json", src.cases / f"{p}-retired.json") if f.is_file()]
    a = answers1921.owner_answers(dirs, files)
    cases: dict[str, dict[str, dict]] = {}
    for f in files:
        page = cases.setdefault(answers1921._page_of(f), {})
        for c in json.loads(f.read_text())["cases"]:
            page.setdefault(c["id"], c)
    a["stratum"] = [_stratum(p, cases[p][c], src.series) for p, c in zip(a["page"], a["case"])]
    return a


UNATTACHED = ("no_geometry", "no_solver_lot", "not_in_parcels")


def recover_lots(src: Sources, blklots, lot: np.ndarray) -> tuple[dict[str, int], dict[str, str]]:
    """R4: for parcels the solver's table lacks, the solver lot under each one's representative point
    (`shapely.point_on_surface` of its polygon in `src.parcels`, EPSG:2227 ft like the solver's grid; `lot`: the
    solver's lot raster from the grid's corner, read by `hatching.grid_values`), with no distance allowed; and why
    each of the others has none (`UNATTACHED`; with no parcels file, not_in_parcels)."""
    want = sorted(set(map(str, blklots)))
    why = {b: "not_in_parcels" for b in want}
    if not want or src.parcels is None:
        return {}, why
    p = pd.read_parquet(src.parcels, columns=["blklot", "geometry"])
    p = p[p["blklot"].astype(str).isin(want)].drop_duplicates("blklot")
    names = p["blklot"].astype(str).to_numpy()
    geoms = np.array([shapely.from_wkb(g) if isinstance(g, (bytes, bytearray)) else None
                      for g in p["geometry"].to_numpy()], dtype=object)
    has = ~(shapely.is_missing(geoms) | shapely.is_empty(geoms)) if len(geoms) else np.zeros(0, bool)
    pts = shapely.point_on_surface(geoms[has])
    under = hatching.grid_values(shapely.get_x(pts), shapely.get_y(pts), lot, fill=-1)
    found = {}
    for b in names[~has]:
        why[b] = "no_geometry"
    for b, i in zip(names[has], under):
        if i >= 0:
            found[str(b)] = int(i)
            del why[b]
        else:
            why[b] = "no_solver_lot"
    return found, why


def attach_reads(rd: pd.DataFrame, lots_of: dict[str, list[int]], src: Sources, lot: np.ndarray):
    """The reads with the solver lot each is about, one row per read and lot (`recovered`: placed through today's
    parcel, `recover_lots`); and the counts: direct (reads of a solver lot's own blklot), recovered, and the
    unattached by why."""
    direct = rd["blklot"].isin(list(lots_of)).to_numpy()
    d = rd[direct].assign(lot=rd.loc[direct, "blklot"].map(lots_of), recovered=False).explode("lot")
    u = rd[~direct]
    found, why = recover_lots(src, u["blklot"].unique(), lot)
    on = u["blklot"].isin(list(found)).to_numpy()
    r = u[on].assign(lot=u.loc[on, "blklot"].map(found), recovered=True)
    att = pd.concat([d, r], ignore_index=True)
    att["lot"] = att["lot"].astype(np.int64)
    att["recovered"] = att["recovered"].astype(bool)
    lost = u.loc[~on, "blklot"].map(why).value_counts()
    counts = {"direct": int(direct.sum()), "recovered": int(on.sum()),
              "unattached": {k: int(lost.get(k, 0)) for k in UNATTACHED}}
    return att, counts


READ_COLUMNS = ["lot", "sheet", "blklot", "code", "confidence", "via", "recovered", "n_sheets"]


def one_read_per_lot(att: pd.DataFrame, owner: np.ndarray, order: list[str]):
    """One read per lot from `attach_reads`' rows: per sheet the most confident (ties: a read of the lot's own blklot
    first, then by blklot); across sheets (R3') the read of the sheet owning the lot's centre (`owner`, per lot) when
    it read the lot, else that of the sheet drawn highest (`order`, bottom first).  Returns `READ_COLUMNS` per read
    lot (n_sheets: how many sheets read it) and the counts."""
    counts = {"lots_read": 0, "lots_several_reads_one_sheet": 0, "lots_several_reads_one_sheet_recovered": 0,
              "lots_read_on_several_sheets": 0, "lots_several_sheets_by_centre_sheet": 0}
    if not len(att):
        return pd.DataFrame(columns=READ_COLUMNS), counts
    a = att.assign(_conf=-att["confidence"], _rec=att["recovered"].astype(int)).sort_values(
        ["lot", "sheet", "_conf", "_rec", "blklot"], kind="stable")
    g = a.groupby(["lot", "sheet"], sort=False)
    size = g["blklot"].transform("size").to_numpy()
    any_rec = g["_rec"].transform("max").to_numpy()
    first = ~a.duplicated(["lot", "sheet"]).to_numpy()
    several = first & (size > 1)
    per_sheet = a[first].copy()
    rank = {s: k for k, s in enumerate(order)}
    per_sheet["_own"] = (per_sheet["sheet"].to_numpy() == owner[per_sheet["lot"].to_numpy()]).astype(int)
    per_sheet["_rank"] = per_sheet["sheet"].map(rank).fillna(-1)
    per_sheet["n_sheets"] = per_sheet.groupby("lot")["sheet"].transform("size")
    best = per_sheet.sort_values(["lot", "_own", "_rank"], ascending=[True, False, False], kind="stable")
    best = best[~best.duplicated("lot").to_numpy()]
    multi = best[best["n_sheets"] > 1]
    counts.update({"lots_read": int(len(best)),
                   "lots_several_reads_one_sheet": int(a.loc[several, "lot"].nunique()),
                   "lots_several_reads_one_sheet_recovered": int(a.loc[several & (any_rec > 0), "lot"].nunique()),
                   "lots_read_on_several_sheets": int(len(multi)),
                   "lots_several_sheets_by_centre_sheet": int(multi["_own"].sum())})
    return best[READ_COLUMNS].reset_index(drop=True), counts


def lot_table(series: str, src: Sources | None = None, lot: np.ndarray | None = None, ink: BlockInk | None = None,
              bid: np.ndarray | None = None) -> pd.DataFrame:
    """One row per solver lot (see the module); `attrs` holds the counts of how the reads and answers were kept.
    `lot`: the solver's lot raster, `bid` its blocks' ids (`block_ids`) and `ink` every sheet's `BlockInk`, when they
    are already at hand."""
    src = src or sources(series)
    table = pd.read_parquet(src.solver / "lots" / "table.parquet", columns=["blklot", "block_num", "geometry"])
    with np.load(src.solver / "lots" / "lots.npz") as z:
        quarter = z["quarter"].astype(np.int64)
    n = len(table)
    if len(quarter) != n:
        raise ValueError(f"lots.npz has {len(quarter)} quarters, the table {n} lots")
    blklot = table["blklot"].astype(str).to_numpy()
    names = np.array(paths.QUARTERS + ("",), dtype=object)
    out = pd.DataFrame({"lot": np.arange(n, dtype=np.int64), "blklot": blklot,
                        "block": table["block_num"].astype(str).to_numpy(),
                        "quarter": names[np.where(quarter >= 0, quarter, len(paths.QUARTERS))]})
    lots_of: dict[str, list[int]] = {}
    for i, b in enumerate(blklot):
        lots_of.setdefault(b, []).append(i)

    # the sheet owning each lot's centre (R9)
    used = sheet_names(src)
    order = draw_order(src, used)
    sheets = [load_sheet(src, s) for s in order]
    if lot is None:
        with np.load(src.solver / "lots" / "lots.npz") as z:
            lot = z["lot"]
    if bid is None:
        bid = block_ids(ndimage.binary_fill_holes(lot >= 0))[0]
    if ink is None:
        ink = block_ink(src, order, bid, int(bid.max()))
    geoms = shapely.from_wkb(table["geometry"].to_numpy()) if n and isinstance(
        table["geometry"].iloc[0], (bytes, bytearray)) else np.asarray(table["geometry"].to_numpy())
    pts = shapely.point_on_surface(geoms)
    centres = np.c_[shapely.get_x(pts), shapely.get_y(pts)]
    owner, several = centre_owner(centres, sheets, hatching.grid_values(centres[:, 0], centres[:, 1], bid, fill=0),
                                  ink)
    out["centre_sheet"] = owner.astype(str)

    # one read per lot (R4, then R3')
    rd = reads(src, used)
    att, read_counts = attach_reads(rd, lots_of, src, lot)
    kept, keep_counts = one_read_per_lot(att, owner, order)
    code = np.full(n, "", dtype=object)
    conf = np.full(n, np.nan)
    via = np.full(n, "", dtype=object)
    rsheet = np.full(n, "", dtype=object)
    rblklot = np.full(n, "", dtype=object)
    recovered = np.zeros(n, bool)
    n_reads = np.zeros(n, np.int64)
    k = kept["lot"].to_numpy(np.int64)
    code[k], conf[k], via[k] = kept["code"].to_numpy(), kept["confidence"].to_numpy(float), kept["via"].to_numpy()
    rsheet[k], rblklot[k] = kept["sheet"].to_numpy(), kept["blklot"].to_numpy()
    recovered[k] = kept["recovered"].to_numpy(bool)
    n_reads[k] = kept["n_sheets"].to_numpy(np.int64)
    out["read_code"], out["read_conf"], out["read_via"], out["read_sheet"] = (
        code.astype(str), conf, via.astype(str), rsheet.astype(str))
    out["read_blklot"], out["read_recovered"] = rblklot.astype(str), recovered
    out["n_reads"] = n_reads

    # the owner's answers
    ans = answers(src)
    priority = {p: k for k, p in enumerate([series] + [p for p in ANSWER_PAGES.get(series, ()) if p != series])}
    ans = ans.assign(_p=ans["page"].map(priority)).sort_values(["_p", "case"], kind="stable")
    a_code = np.full(n, "", dtype=object)
    a_page = np.full(n, "", dtype=object)
    a_case = np.full(n, "", dtype=object)
    a_strat = np.full(n, "", dtype=object)
    n_ans = np.zeros(n, np.int64)
    unattached = []
    for r in ans.itertuples(index=False):
        tried = [r.blklot] + [b for b in (r.blklots or ()) if b != r.blklot]
        hit = next((lots_of[b] for b in tried if b in lots_of), None)
        if hit is None:
            unattached.append(f"{r.page}/{r.case}")
            continue
        for i in hit:
            n_ans[i] += 1
            if a_code[i] == "":
                a_code[i], a_page[i], a_case[i], a_strat[i] = r.truth, r.page, r.case, r.stratum
            elif a_code[i] != r.truth:
                raise ValueError(f"lot {i} ({blklot[i]}): the owner answered {a_code[i]} ({a_page[i]}/{a_case[i]}) "
                                 f"and {r.truth} ({r.page}/{r.case})")
    out["answer"], out["answer_page"], out["answer_case"], out["stratum"] = (
        a_code.astype(str), a_page.astype(str), a_case.astype(str), a_strat.astype(str))
    out["n_answers"] = n_ans
    out.attrs = {
        "sheets": used, "draw_order": order, "centres_on_several_sheets": several,
        "lots_without_centre_sheet": int((owner == "").sum()),
        "reads": {"kept": int((out["read_code"] != "").sum()), "total": int(len(rd)), **read_counts, **keep_counts,
                  "kept_recovered": int(recovered.sum()),
                  "kept_not_from_centre_sheet": int(((out["read_code"] != "") &
                                                     (out["read_sheet"] != out["centre_sheet"])).sum()),
                  "by_sheet": {s: int(v) for s, v in rd["sheet"].value_counts().sort_index().items()}},
        "answers": {"kept": int(len(ans)), "dropped": int(ans.attrs.get("dropped", 0)),
                    "unattached": unattached, "lots_answered_twice": int((n_ans > 1).sum()),
                    "by_page": answers1921_pages(ans)},
    }
    return out


def answers1921_pages(ans: pd.DataFrame) -> dict:
    return {p: {"dir_exists": bool(i["dir_exists"]), "read": int(i["read"]), "kept": int(i["kept"]),
                "dropped": int(i["dropped"])} for p, i in sorted(ans.attrs.get("pages", {}).items())}


# --- the run --------------------------------------------------------------------------------------------------------

def _init_worker():
    cv2.setNumThreads(1)


def _job(job) -> dict:
    """One sheet: its rasters written to `out`/<sheet>.npz; returns its counts and its lots' px."""
    src, name, order, lot_cut, block_cut, street_cut, weight_cut, bid_cut, offset, out, n_lots, lines, ink = job
    sheets = {s: load_sheet(src, s) for s in order}
    r, stats = _rasters(src, sheets[name], lot_cut, block_cut, street_cut, weight_cut, bid_cut, offset,
                        overlapping(name, order, sheets), ink, lines)
    write_rasters(out / f"{name}.npz", r, src.series)
    stats["lot_px_per_lot"] = np.bincount(r.lot[r.lot >= 0], minlength=n_lots).astype(np.int64)
    return stats


def _map(fn, jobs, workers: int):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    with get_context("spawn").Pool(min(workers, len(jobs)), initializer=_init_worker) as pool:
        return list(pool.imap_unordered(fn, jobs))


def run(series: str, out: Path | None = None, workers: int = 4, src: Sources | None = None,
        only: list[str] | None = None) -> dict:
    """Every map sheet's rasters (or `only` those; the draw order still takes every sheet), the lot table and the
    summary, into `out` (`data_dir(series)`)."""
    src = src or sources(series)
    out = Path(out or data_dir(series))
    names = sheet_names(src)
    if not names:
        raise FileNotFoundError(f"no {series} sheets with a page, a placement and a frame under {src.work}")
    order = draw_order(src, names)
    lot, blocks = solver_grid(src)
    bid, n_blocks = block_ids(blocks)
    ink = block_ink(src, order, bid, n_blocks, workers)                    # every sheet: the lot table needs them all
    lines = centre_lines(src)
    street, weight, labels = street_grid(lines, blocks, data_area(src, names, lot.shape))
    table = lot_table(series, src, lot, ink, bid)
    n_lots = len(table)
    jobs = []
    for s in order:
        if only and s not in only:
            continue
        tr = load_transform(src, s)
        r0, r1, c0, c1 = hatching._slice(tr, _cut(src, s), lot.shape)
        jobs.append((src, s, order, lot[r0:r1, c0:c1], blocks[r0:r1, c0:c1], street[r0:r1, c0:c1],
                     weight[r0:r1, c0:c1], bid[r0:r1, c0:c1], (r0, c0), out, n_lots, lines.geoms, ink))
    jobs.sort(key=lambda j: -int(np.prod(page_shape(src, j[1]))))          # the largest first: the pool ends sooner
    out.mkdir(parents=True, exist_ok=True)
    done = sorted(_map(_job, jobs, workers), key=lambda d: d["sheet"])
    px = np.zeros(n_lots, np.int64)
    for d in done:
        px += d.pop("lot_px_per_lot")
    table["px"] = px
    attrs, table.attrs = table.attrs, {}
    table.to_parquet(out / "lots.parquet", index=False)
    summary = {"series": series, "sheets_used": names, "not_used": not_used(src), "draw_order": order,
               "sheets": {d["sheet"]: {k: v for k, v in d.items() if k != "sheet"} for d in done},
               "quarters": quarter_counts(table), "reads_per_code": reads_per_code(table),
               "answers_per_page_stratum": answers_per_page(table),
               "lot_table": attrs, "swatches": [[s, c, list(b)] for s, c, b in swatches(series, src)],
               "street_labels": labels, "ownership": ownership_counts(ink)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def quarter_counts(t: pd.DataFrame) -> dict:
    out = {}
    for q in paths.QUARTERS:
        m = (t["quarter"] == q).to_numpy()
        has = m & (t["px"].to_numpy() > 0) if "px" in t else m
        out[q] = {"lots": int(m.sum()), "lots_with_px": int(has.sum()),
                  "px": int(t["px"].to_numpy()[m].sum()) if "px" in t else 0,
                  "lots_read": int((m & (t["read_code"] != "").to_numpy()).sum()),
                  "lots_read_with_px": int((has & (t["read_code"] != "").to_numpy()).sum()),
                  "lots_answered": int((m & (t["answer"] != "").to_numpy()).sum()),
                  "lots_answered_random": int((m & (t["stratum"] == "random").to_numpy()).sum())}
    return out


def reads_per_code(t: pd.DataFrame) -> dict:
    r = t[t["read_code"] != ""]
    return {c: int(v) for c, v in r["read_code"].value_counts().sort_index().items()}


def answers_per_page(t: pd.DataFrame) -> dict:
    a = t[t["answer"] != ""]
    out: dict = {}
    for (p, s), v in a.groupby(["answer_page", "stratum"]).size().sort_index().items():
        out.setdefault(p, {})[s] = int(v)
    return out


# --- the look picture -----------------------------------------------------------------------------------------------

TINT = {1: (255, 110, 0), 0: (0, 140, 255), UNKNOWN: (128, 128, 128)}    # BGR: blocks blue, not a block orange
ALPHA = {1: 0.40, 0: 0.40, UNKNOWN: 0.55}
ZOOM = 3                       # the zoom's px per scan px (nearest: each scan px a square)
ZOOM_SIDE = 520                # the zoom's side in scan px


def _tinted(r: SheetRasters) -> np.ndarray:
    grey = np.round(r.ink * 255).astype(np.uint8)
    img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR).astype(np.float32)
    for v, bgr in TINT.items():
        m = r.block == v
        img[m] = img[m] * (1 - ALPHA[v]) + np.array(bgr, np.float32) * ALPHA[v]
    return np.round(img).astype(np.uint8)


def _banner(img: np.ndarray, lines: list[str], scale: float) -> np.ndarray:
    h = int(round(34 * scale)) * len(lines) + int(round(16 * scale))
    top = np.full((h, img.shape[1], 3), 255, np.uint8)
    for k, line in enumerate(lines):
        cv2.putText(top, line, (int(12 * scale), int(round(34 * scale)) * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * scale, (0, 0, 0), max(1, int(round(2 * scale))), cv2.LINE_AA)
    return np.vstack([top, img])


def _zoom_window(r: SheetRasters, side: int) -> tuple[int, int]:
    """The top-left of the `side` square with the most px of both blocks and not-blocks (the most to compare)."""
    h, w = r.block.shape
    side = min(side, h, w)

    def integral(m):
        return np.pad(np.cumsum(np.cumsum(m.astype(np.int64), 0), 1), ((1, 0), (1, 0)))

    a, b = integral(r.block == 1), integral(r.block == 0)
    ys = np.arange(0, h - side + 1, max(1, side // 8))
    xs = np.arange(0, w - side + 1, max(1, side // 8))
    Y, X = np.meshgrid(ys, xs, indexing="ij")

    def count(s):
        return s[Y + side, X + side] - s[Y, X + side] - s[Y + side, X] + s[Y, X]

    score = np.minimum(count(a), count(b))
    k = np.unravel_index(np.argmax(score), score.shape)
    return int(Y[k]), int(X[k])


def look_targets(series: str, sheets: list[str], data: Path | None = None, dest: Path = LOOK) -> list[Path]:
    """For the owner: each sheet's street target (today's blocks placed on the scan: blue 1, orange 0, grey 255
    unknown) tinted over the scan, whole at the scan's own px, and a zoom (ZOOM x, nearest) where blocks and streets
    meet most, the target's block edges drawn thin."""
    data = Path(data or data_dir(series))
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    legend = ["blue: today's blocks placed on the scan (target 1)   orange: not a block / street (0)",
              "grey: unknown (outside the frame cut or another sheet's area)"]
    for sheet in sheets:
        r = read_rasters(data / f"{sheet}.npz")
        img = _tinted(r)
        head = f"{series} {sheet}  turn {r.turn}  {r.ft_per_px:.2f} ft/px  street target, data/work placement"
        whole = _banner(img, [head] + legend, 1.6)
        f = dest / f"targets_{sheet}_whole.png"
        cv2.imwrite(str(f), whole)
        written.append(f)
        y0, x0 = _zoom_window(r, ZOOM_SIDE)
        s = min(ZOOM_SIDE, *r.block.shape)
        crop = np.repeat(np.repeat(img[y0:y0 + s, x0:x0 + s], ZOOM, 0), ZOOM, 1)
        blk = np.repeat(np.repeat(r.block[y0:y0 + s, x0:x0 + s] == 1, ZOOM, 0), ZOOM, 1)
        edge = blk & (cv2.erode(blk.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0)
        crop[edge] = TINT[1]
        zoom = _banner(crop, [head, f"zoom x{ZOOM}: scan px x {x0}-{x0 + s}, y {y0}-{y0 + s} (upright)",
                              legend[0], legend[1], "thin blue: the target's block edges"], 1.2)
        f = dest / f"targets_{sheet}_zoom.png"
        cv2.imwrite(str(f), zoom)
        written.append(f)
    return written


LABELS_LOOK = EXPERIMENT / "streets-v2" / "look"
LABEL_TINT = {STREET: ((0, 140, 255), 0.45), NOT_STREET: ((255, 110, 0), 0.40),      # BGR, alpha: street orange,
              UNKNOWN: ((150, 150, 150), 0.55)}                                       # not street blue, ignored grey
OTHER_AREA = ((60, 60, 60), 0.70)       # dark grey: not this sheet's to label (outside its frame cut or drawn above)
CENTRE_BGR = (0, 0, 200)                # dark red: today's centre lines
LABEL_WHOLE = ("20180102094149543", "20180102094355582",            # the plan's whole sheets (Lake Merced second),
               "20180102094212101", "20180102094308402")            # Ocean Beach and the Bay Bridge
LABEL_ZOOMS = {"20180102100525362": (2015, 1495, 520, 520),         # x0, y0, width, height (upright px): Task 2's
               "20180102094325427": (1176, 880, 604, 420)}          # south zoom; the review's Islais Creek crop


def _label_tinted(r: SheetRasters) -> np.ndarray:
    """The labels over the scan; a street px's orange opacity is LABEL_TINT's times its weight (`street_w`)."""
    grey = np.round(r.ink * 255).astype(np.uint8)
    img = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR).astype(np.float32)
    other = r.block == UNKNOWN
    weight = np.ones(r.street.shape, np.float32) if r.street_w is None else r.street_w.astype(np.float32)
    for m, (bgr, a) in [(r.street == v, LABEL_TINT[v]) for v in (STREET, NOT_STREET)] + [
            ((r.street == UNKNOWN) & ~other, LABEL_TINT[UNKNOWN]), (other, OTHER_AREA)]:
        a = a * weight[m][:, None] if bgr == LABEL_TINT[STREET][0] else a
        img[m] = img[m] * (1 - a) + np.array(bgr, np.float32) * a
    img = np.round(img).astype(np.uint8)
    img[r.centre == 1] = CENTRE_BGR
    return img


def look_labels(series: str, whole=LABEL_WHOLE, zooms=LABEL_ZOOMS, data: Path | None = None,
                dest: Path = LABELS_LOOK) -> list[Path]:
    """For the owner: the street labels tinted over the scan (street orange, its opacity its weight; not street blue,
    ignored grey, not this sheet's area dark grey, today's centre lines dark red), each `whole` sheet at the scan's own
    px and each of `zooms` ((x0, y0, width, height) upright px) at ZOOM x, nearest."""
    data = Path(data or data_dir(series))
    dest.mkdir(parents=True, exist_ok=True)
    legend = ["orange: street (1): off today's blocks, in its centre line's corridor;",
              "  opacity = weight = P(learned half-width > distance), cut at the side's block (white: ~0)",
              "blue: not street (0): today's blocks     grey: ignored (255)",
              "dark grey: not this sheet's area (255)     dark red: today's centre lines"]
    written = []
    for sheet in list(whole) + list(zooms):
        r = read_rasters(data / f"{sheet}.npz")
        img = _label_tinted(r)
        n = np.count_nonzero(r.block != UNKNOWN)
        share = {v: np.count_nonzero((r.street == v) & (r.block != UNKNOWN)) / max(n, 1)
                 for v in (STREET, NOT_STREET, UNKNOWN)}
        on = r.street == STREET
        soft = np.count_nonzero(on & (r.street_w < 1)) / max(np.count_nonzero(on), 1)
        head = [f"{series} {sheet}  {r.ft_per_px:.2f} ft/px  street labels v2 "
                "(learned widths, cut at each side's block)",
                f"of this sheet's area: street {share[STREET]:.1%} ({soft:.1%} of it soft), not street "
                f"{share[NOT_STREET]:.1%}, ignored {share[UNKNOWN]:.1%}"]
        if sheet in whole:
            f = dest / f"labels_{sheet}_whole.png"
            cv2.imwrite(str(f), _banner(img, head + legend, 1.6))
            written.append(f)
        if sheet in zooms:
            x0, y0, zw, zh = zooms[sheet]
            crop = np.repeat(np.repeat(img[y0:y0 + zh, x0:x0 + zw], ZOOM, 0), ZOOM, 1)
            f = dest / f"labels_{sheet}_zoom.png"
            zoom_line = f"zoom x{ZOOM}: scan px x {x0}-{x0 + zw}, y {y0}-{y0 + zh} (upright)"
            cv2.imwrite(str(f), _banner(crop, head + [zoom_line] + legend, 1.0))
            written.append(f)
    return written
