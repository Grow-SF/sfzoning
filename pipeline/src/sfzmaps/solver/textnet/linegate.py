"""The text-vs-line gate: measures of a text detector's chance of text on the 2000 sheets that can see it call line ink
(block, street and district lines through or touching a label) text, and the text it finds.

A chance comes from a source (`ChanceSource`: a sheet's name -> its chance x 255, uint8, the page's shape;
`folder_source` reads a folder of PNGs).  A missing chance or one of another shape stops the gate, naming the sheet:
it never scores fewer px.  Every measure is taken on the ink px (page < `grid.INK_BELOW`) whose quarter (`Geo.quarter`,
by the px's map point) is one of those asked for:

  m1a           district-border ink called text: every ink px, weighted by w(d), the share of the ink d grid px from
                the nearest district crack that the district line adds, inside the line's own reach (the cut, below).
                No ink rule and no Terra box define it.  m1a is the total; m1a_in_box, m1a_near and m1a_far split it
                by box
  m1a_context   a diagnostic, in no verdict: the same weight beyond the cut (the profile's tail: ink near district
                borders that is not the line's own), m1a_context and m1a_in_box_context (`Geo.context_weight`; None for
                a geo without it)
  m1b_*         straight-line ink called text: the run rule's px (`linerules.line_px` at the sheet's scale), split by
                box (in_box, near, far) and by the distance to the nearest of Sol's label points (m1b_by_point:
                0-30, 30-100, 100+ px)
  m1c           all ink outside every Terra word box called text, nothing excluded
  m2            text found: ink inside a Terra word box that is not a run-rule px and not within 1 px of one (its 8
                neighbours); m2_by_kind by the kind of the word drawn last on the px (as the detector learns it), and
                m2_answered on the owner's answered labels' words (`Geo.answered`; None for a geo without it)
  m2_clean_words  text found on clean words: ALL ink inside the box of a clean word (`clean_word_mask`), with no line
                rule applied inside the word, so long letter strokes (big P and M stems, bold letters joined to a
                hyphen) count; m2_clean_words_by_kind by the kind of the word drawn last on the px
  overlap_band  ink inside a word box that is a run-rule px or within 1 px of one: reported on its own, never in m2

"In box": inside a Terra word's rectangle (`data.SheetData.words`, filled as `data._paint` fills them); "near": in
none, within NEAR_PX px of one (px centres, Euclidean); "far": farther.  Line ink by the run rule is the rule's, so
`rules` gives the rule's accuracy on exact-truth cases (`linerules.rule_accuracy`) beside every number, and the M1a
weight's table goes with M1a (`m1a_weight`).

Each measure: `px` (for M1a the sum of w), `called_05` and `called_09` (the share of px whose chance is at least 0.5
or 0.9: CALLED_05 and CALLED_09 of 255), `mean` (the mean chance), `log_loss` (against the measure's TARGET: not text
for M1, text for M2 and the overlap band, which mixes line and glyph ink; the chance clipped to 1e-4 .. 1 - 1e-4) and
`ci_05`, the 95% interval of called_05 by bootstrap over blocks.  A block is a unit of (quarter, block): the zoning
table's block of the grid px holding the px's map point (`Geo.block`); px off the grid are one unit per quarter (block
-1).  The sums per block (`tallies`, `blocks.npz`) let two sets of runs be compared block by block (`paired`).

M1a's weight (`CityGeo`), as the plan's red team set it:
  district cracks  `borders.border_paths()` points whose two lots are both known and differ in the table's code
                   (`measures.Table.line`'s rule, `C.table`); the other cracks with both lots known are the baseline
  f_D(d), f_L(d)   the ink share of the cracks' profile (`prof`) at offsets +d and -d (their mean), over the district
                   cracks and over the other both-known cracks, for touching lots and across a street apart
  w(d)             max(0, f_D(d) - f_L(d)) / f_D(d) for d below the class's cut, 0 from it on (`weight_cut`)
  the cut          where the line's own excess e(d) = f_D(d) - f_L(d) ends, per class: the first d past the excess's
                   peak at which it has fallen to the class's far-field level (the median excess over the outer third
                   of d = 0 .. J, J = `borders.J`) or stops falling (e(d + 1) > e(d): another feature's ink begins,
                   as the block outlines along district streets do across a street).  Beyond it, up to J, the same
                   formula gives the context weight (m1a_context)
  d                the distance, in grid px and rounded, from a grid px to the nearest district crack, each crack drawn
                   at the px where its profile reads offset 0 (the floor of its middle), so the px of a straight border
                   d rows away on either side are at offset +d or -d; the table used is the nearest crack's (touch or
                   across a street; touch where they tie)
A page px's grid px: its map point by the sheet's placement samples (`data._world_at`), `grid.to_grid`, floor: the
path the quarter split takes.  Taken in bands of `data.BAND` rows; the block and the weight at ink px only, and only
inside the sheet's crop on the merged page (the page whose ink the weight was measured on): the margins have weight 0.
"""
import hashlib
import json
import math
from collections.abc import Callable
from functools import lru_cache, reduce
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from scipy.spatial import cKDTree

from sfzmaps.solver import borders, grid, page, paths
from sfzmaps.solver.city import city
from sfzmaps.solver.textnet import data, linerules

ChanceSource = Callable[[str], np.ndarray]

CALLED_05 = 128           # a chance of at least 0.5 (127.5 of 255): `scores.CALLED`
CALLED_09 = 230           # a chance of at least 0.9 (229.5 of 255)
NEAR_PX = 30              # page px from a word box that are "near" it
POINT_BINS = (("0-30", 30.0), ("30-100", 100.0), ("100+", np.inf))   # px to the nearest Sol point, upper ends kept
KINDS = tuple(data.KINDS)                                              # m2_by_kind, in data.KINDS' order
CLEAN_PX = 3              # a clean word has no run-rule ink within this many px outside its box
CLIP = 1e-4               # the chance is clipped to CLIP .. 1 - CLIP in the log loss
BOOT_CHUNK = 100          # bootstrap draws made at a time
FIELDS = ("px", "called_05", "called_09", "chance", "log_loss")       # the sums per block (float64)
STATS = {"called_05": "called_05", "called_09": "called_09", "mean": "chance", "log_loss": "log_loss"}
SCORES_FILE, MARKDOWN_FILE, BLOCKS_FILE = "scores.json", "scores.md", "blocks.npz"
LEARNED_RULES = paths.SOLVER_DIR / "experiments" / "textlines" / "rules.json"   # `scripts/textlines/rules.py`

M1A = ("m1a", "m1a_in_box", "m1a_near", "m1a_far")
CONTEXT = ("m1a_context", "m1a_in_box_context")          # the weight beyond the cut: a diagnostic, in no verdict
WEIGHTED = (*M1A, *CONTEXT)                               # px are the sum of a weight
M1B = ("m1b_in_box", "m1b_near", "m1b_far", *(f"m1b_by_point:{b}" for b, _ in POINT_BINS))
M2 = ("m2", *(f"m2_by_kind:{k}" for k in KINDS), "m2_answered",
      "m2_clean_words", *(f"m2_clean_words_by_kind:{k}" for k in KINDS))
MEASURES = (*M1A, *CONTEXT, *M1B, "m1c", *M2, "overlap_band")
TARGET = {m: ("text" if m.startswith("m2") or m == "overlap_band" else "not text") for m in MEASURES}
WHAT = {
    "m1a": "ink, weighted by w(d) to the nearest district crack",
    "m1a_in_box": "the same, inside a Terra word box",
    "m1a_near": "the same, within 30 px of a box",
    "m1a_far": "the same, farther",
    "m1a_context": "diagnostic: ink weighted by w(d) beyond the cut",
    "m1a_in_box_context": "diagnostic: the same, inside a box",
    "m1b_in_box": "run-rule line ink inside a box",
    "m1b_near": "run-rule line ink within 30 px of a box",
    "m1b_far": "run-rule line ink farther",
    "m1b_by_point:0-30": "run-rule line ink within 30 px of a Sol point",
    "m1b_by_point:30-100": "run-rule line ink 30-100 px from a Sol point",
    "m1b_by_point:100+": "run-rule line ink over 100 px from a Sol point",
    "m1c": "all ink outside every box",
    "m2": "in-box ink, not run-rule ink or within 1 px of it",
    "m2_by_kind:zoning code": "the same, in a zoning code's box",
    "m2_by_kind:block number": "the same, in a block number's box",
    "m2_by_kind:street name": "the same, in a street name's box",
    "m2_by_kind:other": "the same, in another word's box",
    "m2_answered": "the same, in an owner-answered label's box",
    "m2_clean_words": "all ink in the box of a clean word (no run rule applied inside it)",
    **{f"m2_clean_words_by_kind:{k}": f"the same, in a clean {k}'s box" for k in KINDS},
    "overlap_band": "in-box ink that is run-rule ink or within 1 px of it",
}
# What each measure would miss (the spec: a gate states what it would miss), one line per family, in scores.md.
MISSES = {
    "m1a, m1a_in_box, m1a_near, m1a_far": "weight 0 on the margins, the cut boxes and other series' sheets (not on "
    "the merged page) and beyond each class's cut; ink within the cut is weighted by the city's average, so label ink "
    "touching a district border carries some weight too",
    "m1a_context, m1a_in_box_context": "a diagnostic only: mostly ink near district borders that is not the line's own "
    "(labels, block outlines); in no verdict",
    "m1b_*": "lines the run rule misses (stair-step district lines: recall 0.08; dashed: 0.02); a few glyph px it calls "
    "line (the table below)",
    "m1c": "counts as not-text the ink in the other passes' boxes (`sd.other`, unknown in training), text Terra missed "
    "(near Sol's points, unknown in training too) and the margins' text",
    "m2, m2_by_kind, m2_answered": "counts as text the lines the run rule misses inside boxes (stair-step and dashed "
    "lines); leaves out glyph ink within 1 px of a run-rule line (in the overlap band); m2_answered only the answered "
    "labels with a Terra box",
    "m2_clean_words, m2_clean_words_by_kind": "words touching or crossed by lines are left out; it measures letters, "
    "including long strokes, on clean words only",
    "overlap_band": "mixes line and glyph ink, so neither target is right; its log loss is against text",
}
_PROB = np.clip(np.arange(256) / 255.0, CLIP, 1 - CLIP)
_LOSS = {"text": -np.log(_PROB), "not text": -np.log1p(-_PROB)}       # by chance x 255


# --- where the chance comes from -------------------------------------------------------------------------------------

def folder_source(folder: Path, layout: str) -> ChanceSource:
    """A source reading each sheet's chance from `folder`: `<sheet>.png` ("flat", as `paths.TEXT_CHANCE_DIR`) or
    `<sheet>/chance.png` ("nested", as `paths.READER_TEXT_DIR`).  A missing file raises FileNotFoundError naming the
    sheet; a file that is no one-channel 8-bit picture raises ValueError (the page's shape is checked by the gate)."""
    if layout not in ("flat", "nested"):
        raise ValueError(f"a folder's layout is flat or nested, not {layout!r}")
    folder = Path(folder)

    def source(sheet: str) -> np.ndarray:
        f = folder / f"{sheet}.png" if layout == "flat" else folder / sheet / "chance.png"
        if not f.is_file():
            raise FileNotFoundError(f"no chance of text for {sheet}: {f} is missing")
        chance = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if chance is None or chance.dtype != np.uint8 or chance.ndim != 2:
            raise ValueError(f"the chance of {sheet} ({f}) is not a one-channel 8-bit picture")
        return chance
    return source


def _chance(source: ChanceSource, sd: data.SheetData) -> np.ndarray:
    chance = np.asarray(source(sd.sheet))
    if chance.shape != (sd.height, sd.width):
        raise ValueError(f"the chance of {sd.sheet} is {chance.shape}, not its page's shape {(sd.height, sd.width)}: "
                         "a chance must cover the whole sheet")
    if chance.dtype != np.uint8:
        raise ValueError(f"the chance of {sd.sheet} is {chance.dtype}, not uint8 (the chance x 255)")
    return chance


# --- the map behind each page px -------------------------------------------------------------------------------------

class Geo(Protocol):
    """What the gate needs of the map behind a sheet's page, as page rasters (read at ink px only):
      quarter(sd)        uint8, the index into `paths.QUARTERS`, 255 where the px has no map point
      block(sd)          int32, the block (-1: none)
      border_weight(sd)  float32, M1a's weight (inside the cut)
    and, optionally, `context_weight(sd)` (float32, the weight beyond the cut: m1a_context), `answered(sd)` (bool, the
    px of the owner's answered labels' words), `weight_table` (the weight's table, for scores.md) and `block_key`
    (what identifies the block numbering, for `paired`)."""

    def quarter(self, sd: data.SheetData) -> np.ndarray: ...

    def block(self, sd: data.SheetData) -> np.ndarray: ...

    def border_weight(self, sd: data.SheetData) -> np.ndarray: ...


def grid_px(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map points (x, y; EPSG:2227 ft, finite) -> the grid px (col, row) holding them, int64: floor of `grid.to_grid`."""
    g = grid.to_grid(np.stack([np.asarray(x, float), np.asarray(y, float)], -1))
    return np.floor(g[..., 0]).astype(np.int64), np.floor(g[..., 1]).astype(np.int64)


def ink_shares(prof: np.ndarray, J: int) -> np.ndarray:
    """The share of ink at offsets +d and -d (their mean) of profiles `prof` (n, 2J + 1), for d = 0 .. J; NaN with no
    profiles."""
    prof = np.asarray(prof)
    if not len(prof):
        return np.full(J + 1, np.nan)
    f = prof.sum(axis=0, dtype=np.int64) / len(prof)
    return np.array([0.5 * (f[J + d] + f[J - d]) for d in range(J + 1)])


def excess_weight(f_district: np.ndarray, f_other: np.ndarray) -> np.ndarray:
    """w(d) = max(0, f_D(d) - f_L(d)) / f_D(d): the share of the ink at distance d that a district line adds over an
    ordinary border; 0 where f_D is 0 or unknown."""
    fd, fl = np.asarray(f_district, float), np.asarray(f_other, float)
    ok = np.isfinite(fd) & (fd > 0)
    out = np.zeros(len(fd))
    out[ok] = np.maximum(0.0, fd[ok] - np.nan_to_num(fl[ok])) / fd[ok]
    return out


def weight_cut(excess: np.ndarray) -> tuple[int, float]:
    """Where a district line's own excess ink ends (see the module), from its excess e(d) = f_D(d) - f_L(d), d = 0 ..
    J: (the cut, the far-field level).  The far-field level is the median of e over the outer third of d (the last
    ceil((J + 1) / 3) values); the cut is the first d past e's peak at which e is at most that level, or at which e
    stops falling (e(d + 1) > e(d)); J + 1 when neither happens.  The weight is kept for d below the cut."""
    e = np.nan_to_num(np.asarray(excess, float), nan=0.0)
    n = len(e)
    far = float(np.median(e[n - math.ceil(n / 3):]))
    for d in range(int(np.argmax(e)) + 1, n):
        if e[d] <= far or (d + 1 < n and e[d + 1] > e[d]):
            return d, far
    return n, far


def weight_table(prof: np.ndarray, touch: np.ndarray, district: np.ndarray, other: np.ndarray, J: int) -> dict:
    """M1a's table (see the module): per class ("touch", "across"), the cracks counted, f_D, f_L, the far-field level
    and the cut (`weight_cut`), w by d inside the cut (0 from it on) and the context weight beyond it (0 before)."""
    out: dict = {"J": int(J)}
    for name, cls in (("touch", True), ("across", False)):
        d_sel, o_sel = district & (touch == cls), other & (touch == cls)
        fd, fl = ink_shares(prof[d_sel], J), ink_shares(prof[o_sel], J)
        w = excess_weight(fd, fl)
        cut, far = weight_cut(fd - fl)
        out[name] = {"district_cracks": int(d_sel.sum()), "other_cracks": int(o_sel.sum()),
                     "f_district": fd.tolist(), "f_other": fl.tolist(), "far_level": far, "cut": cut,
                     "w": np.where(np.arange(J + 1) < cut, w, 0.0).tolist(),
                     "w_context": np.where(np.arange(J + 1) >= cut, w, 0.0).tolist()}
    return out


def _distance_to(shape: tuple[int, int], r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The Euclidean distance (float32) from every px of a raster of `shape` to the nearest of px (r, c); inf with
    none."""
    if not len(r):
        return np.full(shape, np.inf, np.float32)
    src = np.ones(shape, np.uint8)
    src[r, c] = 0
    return cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def district_distance(shape: tuple[int, int], my: np.ndarray, mx: np.ndarray, touch: np.ndarray,
                      J: int) -> tuple[np.ndarray, np.ndarray]:
    """Each px's distance to the nearest district crack and that crack's class, on a raster of `shape`: each crack
    (its middle (my, mx), grid rows and columns; touching lots or across a street) drawn at px (floor my, floor mx).
    Returns d (uint8, the distance rounded, J + 1 for anything farther) and touch (bool, the nearest crack touches;
    touch where the classes tie)."""
    h, w = shape
    r, c = np.floor(np.asarray(my)).astype(np.int64), np.floor(np.asarray(mx)).astype(np.int64)
    touch = np.asarray(touch, bool)
    on = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    d_touch = _distance_to(shape, r[on & touch], c[on & touch])
    d_across = _distance_to(shape, r[on & ~touch], c[on & ~touch])
    use_touch = d_touch <= d_across
    np.minimum(d_touch, d_across, out=d_touch)
    del d_across
    np.rint(d_touch, out=d_touch)
    return np.minimum(d_touch, J + 1).astype(np.uint8), use_touch


def weight_at(d: np.ndarray, touch: np.ndarray, w_touch, w_across) -> np.ndarray:
    """The weight (float32) at distances `d` (uint8, J + 1 for farther) from cracks of class `touch`: `w_touch` or
    `w_across` (w(d), d = 0 .. J) by class, 0 past J."""
    w_touch, w_across = np.asarray(w_touch, np.float32), np.asarray(w_across, np.float32)
    if len(w_touch) != len(w_across):
        raise ValueError("the two classes' tables differ in length")
    J = len(w_touch) - 1
    lut_touch, lut_across = np.zeros(J + 2, np.float32), np.zeros(J + 2, np.float32)
    lut_touch[:J + 1], lut_across[:J + 1] = w_touch, w_across
    out = lut_across[d]
    out[touch] = lut_touch[d[touch]]
    return out


def district_weight(shape: tuple[int, int], my: np.ndarray, mx: np.ndarray, touch: np.ndarray,
                    w_touch: np.ndarray, w_across: np.ndarray) -> np.ndarray:
    """A weight on a raster of `shape` (float32): each px gets the table (`w_touch` or `w_across`, w(d) for d = 0 ..
    J) of the nearest district crack at its distance (`district_distance`), 0 past J."""
    d, near_touch = district_distance(shape, my, mx, touch, len(w_touch) - 1)
    return weight_at(d, near_touch, w_touch, w_across)


class CityGeo:
    """The real Geo: each page px's map point through the sheet's placement samples (`data._world_at`, in bands of
    `data.BAND` rows), its quarter (`grid.quarter_of`, every px), and at the ink px the merged page reads (inside the
    sheet's crop: its frame quad in the page stage's frames.json minus the cut boxes, `page.crop_mask`) the grid px
    holding it (`grid_px`), its block (`C.grown_raster`'s lot -> that lot's `block_num` in lots/table.parquet,
    numbered in sorted order) and M1a's weights, inside the cut and beyond it (see the module).  Other px (the margins
    and cut boxes, whose ink is not the grid's ink the weight was measured on; px off the grid; every px of a sheet not
    on the merged page) have block -1 and weight 0.  The city's rasters and the distances are built on first use
    (`borders.border_paths()`, about 15 s); one sheet's rasters are kept at a time.  `answered`: the words (Terra's
    box and reading angle, `data.word_rect`) of the labels an owner's answer joined (`codereader.labels.load_labels`,
    `held_out`)."""

    def __init__(self):
        self._city: dict | None = None
        self._last: tuple | None = None
        self._answered: dict | None = None

    def _built(self) -> dict:
        if self._city is None:
            C, Z = city(), borders.border_paths()
            J = int(Z["J"])
            k = Z["k"].astype(np.int64)
            a, b = C.a[k], C.b[k]
            both = C.known[a] & C.known[b]
            district = both & (C.table[a] != C.table[b])
            table = weight_table(Z["prof"], Z["touch"], district, both & ~district, J)
            d, near_touch = district_distance(C.grown_raster.shape, Z["my"][district], Z["mx"][district],
                                              Z["touch"][district], J)
            names, lot_block = np.unique(C.lots["block_num"].astype(str).to_numpy(), return_inverse=True)
            self._city = {"table": table, "d": d, "touch": near_touch, "grown": C.grown_raster,
                          "lot_block": lot_block.astype(np.int32), "names": names,
                          "frames": json.loads((paths.PAGE_DIR / "frames.json").read_text()), "cuts": page.load_cuts()}
        return self._city

    @property
    def weight_table(self) -> dict:
        return self._built()["table"]

    @property
    def block_names(self) -> np.ndarray:
        """The block number of each block id."""
        return self._built()["names"]

    @property
    def block_key(self) -> str:
        """What identifies the block numbering: the count and a digest of the block numbers in id order."""
        names = self.block_names
        return f"{len(names)}:{hashlib.sha256(chr(10).join(names).encode()).hexdigest()[:16]}"

    def _rasters(self, sd: data.SheetData) -> tuple[np.ndarray, ...]:
        if not sd.world.size:
            raise ValueError(f"{sd.series} {sd.sheet} was read without its placement: its map points are not known")
        key = (sd.series, sd.sheet, sd.height, sd.width,
               hashlib.blake2b(np.ascontiguousarray(sd.world).tobytes(), digest_size=16).hexdigest())
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        B = self._built()
        table = B["table"]
        h, w = sd.height, sd.width
        quarter = np.full((h, w), 255, np.uint8)
        block = np.full((h, w), -1, np.int32)
        weight = np.zeros((h, w), np.float32)
        context = np.zeros((h, w), np.float32)
        quad = B["frames"].get(sd.sheet) if sd.series == paths.SERIES else None
        mapped = np.zeros((h, w), bool) if quad is None else \
            page.crop_mask((h, w), np.asarray(quad, float), B["cuts"].get(sd.sheet, []))
        ink = data.page_ink(sd) & mapped
        cols = np.arange(w)
        gh, gw = B["grown"].shape
        for y0 in range(0, h, data.BAND):
            rows = np.arange(y0, min(h, y0 + data.BAND))
            x, y = data._world_at(sd.world, rows, cols)
            band = slice(y0, y0 + len(rows))
            finite = np.isfinite(x) & np.isfinite(y)
            q = quarter[band]
            q[finite] = grid.quarter_of(np.stack([x[finite], y[finite]], -1))
            m = ink[band] & finite
            col, row = grid_px(x[m], y[m])
            on = (col >= 0) & (col < gw) & (row >= 0) & (row < gh)
            bl = np.full(len(col), -1, np.int32)
            bl[on] = B["lot_block"][B["grown"][row[on], col[on]]]
            dd, tt = B["d"][row[on], col[on]], B["touch"][row[on], col[on]]
            ww, cw = np.zeros(len(col), np.float32), np.zeros(len(col), np.float32)
            ww[on] = weight_at(dd, tt, table["touch"]["w"], table["across"]["w"])
            cw[on] = weight_at(dd, tt, table["touch"]["w_context"], table["across"]["w_context"])
            block[band][m] = bl
            weight[band][m] = ww
            context[band][m] = cw
        self._last = (key, (quarter, block, weight, context))
        return self._last[1]

    def quarter(self, sd: data.SheetData) -> np.ndarray:
        return self._rasters(sd)[0]

    def block(self, sd: data.SheetData) -> np.ndarray:
        return self._rasters(sd)[1]

    def border_weight(self, sd: data.SheetData) -> np.ndarray:
        return self._rasters(sd)[2]

    def context_weight(self, sd: data.SheetData) -> np.ndarray:
        return self._rasters(sd)[3]

    def answered(self, sd: data.SheetData) -> np.ndarray:
        if self._answered is None:
            from sfzmaps.solver.codereader import labels  # only the real geo reads the owner's answers
            words: dict[str, list] = {}
            for lab in labels.load_labels(crops=False):
                if lab.held_out and lab.box is not None:
                    words.setdefault(lab.sheet, []).append(data.word_rect(lab.box, lab.angle))
            self._answered = {s: np.round(np.array(r, float).reshape(-1, 4, 2)).astype(np.int32)
                              for s, r in words.items()}
        rects = self._answered.get(sd.sheet)
        if rects is None:
            return np.zeros((sd.height, sd.width), bool)
        return data._paint(rects, np.ones(len(rects), np.int32), 0, 0, sd.height, sd.width) > 0


# --- the measures ----------------------------------------------------------------------------------------------------

class _Sums:
    """One measure's sums per (quarter, block), added sheet by sheet."""

    def __init__(self, target: str):
        self.loss = _LOSS[target]
        self.parts: list[tuple[np.ndarray, np.ndarray]] = []

    def add(self, q: np.ndarray, blk: np.ndarray, chance: np.ndarray, w: np.ndarray | None = None) -> None:
        if not len(chance):
            return
        unit = (q.astype(np.int64) << 32) | (blk.astype(np.int64) + 1)
        units, inv = np.unique(unit, return_inverse=True)
        w = np.ones(len(chance)) if w is None else w.astype(np.float64)
        cols = (w, w * (chance >= CALLED_05), w * (chance >= CALLED_09), w * (chance / 255.0), w * self.loss[chance])
        self.parts.append((units, np.stack([np.bincount(inv, v, len(units)) for v in cols], 1)))

    def table(self) -> dict[str, np.ndarray]:
        if not self.parts:
            return {"quarter": np.zeros(0, np.int8), "block": np.zeros(0, np.int32),
                    **{f: np.zeros(0) for f in FIELDS}}
        units, inv = np.unique(np.concatenate([u for u, _ in self.parts]), return_inverse=True)
        v = np.concatenate([s for _, s in self.parts])
        s = {f: np.bincount(inv, v[:, j], len(units)) for j, f in enumerate(FIELDS)}
        return {"quarter": (units >> 32).astype(np.int8), "block": ((units & 0xFFFFFFFF) - 1).astype(np.int32), **s}


def _page_raster(a, sd: data.SheetData, what: str, dtype) -> np.ndarray:
    a = np.asarray(a)
    if a.shape != (sd.height, sd.width):
        raise ValueError(f"the geo's {what} of {sd.sheet} is {a.shape}, not its page's shape {(sd.height, sd.width)}")
    return a.astype(dtype, copy=False)


def clean_word_mask(sd: data.SheetData, line: np.ndarray) -> np.ndarray:
    """Which of the sheet's Terra words are clean, bool (len(sd.words),): every 8-connected piece of ink that touches
    the word's box lies wholly inside it, and no run-rule px (`line`, `linerules.line_px`) is within CLEAN_PX px outside
    the box (a line's px inside the box are in a crossing piece, caught by the first test; a long letter stroke inside
    the box is no reason to leave the word out).  The composites pick their clean words with the same test, but run it
    with the run rule's px grown by 1 px (`SheetData.line_bits`) where this gate passes the raw `line_px`, so their
    clean words are a subset of these."""
    out = np.zeros(len(sd.words), bool)
    if not len(sd.words):
        return out
    ink = data.page_ink(sd)
    n, label = cv2.connectedComponents(ink.view(np.uint8), connectivity=8)
    size = np.bincount(label.ravel(), minlength=n)
    reach = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CLEAN_PX + 1, 2 * CLEAN_PX + 1))
    for j, rect in enumerate(sd.words):
        lo = np.maximum(rect.min(axis=0) - CLEAN_PX, 0)
        hi = np.minimum(rect.max(axis=0) + CLEAN_PX + 1, [sd.width, sd.height])
        if (hi <= lo).any():
            continue
        shape = (hi[1] - lo[1], hi[0] - lo[0])
        box = np.zeros(shape, np.uint8)
        cv2.fillPoly(box, [(rect - lo).reshape(-1, 1, 2)], 1)
        window = (slice(lo[1], hi[1]), slice(lo[0], hi[0]))
        ring = (cv2.dilate(box, reach) > 0) & (box == 0)
        if line[window][ring].any():
            continue
        inside = label[window][(box > 0) & ink[window]]
        pieces, got = np.unique(inside, return_counts=True)
        out[j] = bool((size[pieces] == got).all())
    return out


def _add_sheet(sums: dict[str, _Sums], sd: data.SheetData, chance: np.ndarray, quarter_ix: np.ndarray, geo) -> None:
    """Add one sheet's px to every measure's sums (see the module)."""
    if not np.isfinite(sd.ft_per_px):
        raise ValueError(f"{sd.series} {sd.sheet} has no scale (read it with its placement): the run rule needs it")
    h, w = sd.height, sd.width
    ink = data.page_ink(sd)
    quarter = _page_raster(geo.quarter(sd), sd, "quarter", np.uint8)
    scope = ink & np.isin(quarter, quarter_ix)
    if not scope.any():
        return
    block = _page_raster(geo.block(sd), sd, "block", np.int32)
    weight = _page_raster(geo.border_weight(sd), sd, "border weight", np.float32)
    word = data._words(sd, 0, 0, h, w)
    in_box = word > 0
    if in_box.any():
        dist = cv2.distanceTransform((~in_box).view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        near = ~in_box & (dist <= NEAR_PX)
        del dist
    else:
        near = np.zeros((h, w), bool)
    line = linerules.line_px(ink, sd.ft_per_px)
    by_line = cv2.dilate(line.view(np.uint8), np.ones((3, 3), np.uint8)) > 0
    q, blk, c = quarter.ravel(), block.ravel(), chance.ravel()

    def add(name: str, mask: np.ndarray, weight: np.ndarray | None = None) -> None:
        i = np.flatnonzero(mask)
        sums[name].add(q[i], blk[i], c[i], None if weight is None else weight.ravel()[i])

    split = {"in_box": in_box, "near": near, "far": ~in_box & ~near}
    border = scope & (weight > 0)
    add("m1a", border, weight)
    for part, where in split.items():
        add(f"m1a_{part}", border & where, weight)
    if "m1a_context" in sums:
        context = _page_raster(geo.context_weight(sd), sd, "context weight", np.float32)
        beyond = scope & (context > 0)
        add("m1a_context", beyond, context)
        add("m1a_in_box_context", beyond & in_box, context)
    on_line = scope & line
    for part, where in split.items():
        add(f"m1b_{part}", on_line & where)
    i = np.flatnonzero(on_line)
    if len(sd.points):
        to_point = cKDTree(sd.points.astype(float)).query(np.c_[i % w, i // w].astype(float))[0]
    else:
        to_point = np.full(len(i), np.inf)
    lower = 0.0
    for name, upper in POINT_BINS:
        j = i[(to_point > lower) & (to_point <= upper)] if lower > 0 else i[to_point <= upper]
        sums[f"m1b_by_point:{name}"].add(q[j], blk[j], c[j])
        lower = upper
    add("m1c", scope & ~in_box)
    base = scope & in_box & ~by_line
    add("m2", base)
    kind = np.concatenate([[data.NO_KIND], sd.kinds]).astype(np.uint8)[word]
    for name in KINDS:
        add(f"m2_by_kind:{name}", base & (kind == data.KINDS[name]))
    if "m2_answered" in sums:
        add("m2_answered", base & _page_raster(geo.answered(sd), sd, "answered words", bool))
    clean = np.concatenate([[False], clean_word_mask(sd, line)])[word]
    add("m2_clean_words", scope & clean)
    for name in KINDS:
        add(f"m2_clean_words_by_kind:{name}", scope & clean & (kind == data.KINDS[name]))
    add("overlap_band", scope & in_box & by_line)


def _draws(n: int, boot: int, seed: int):
    """The bootstrap's draws of `n` units with replacement, BOOT_CHUNK draws (rows) at a time, from one generator."""
    rng = np.random.default_rng(seed)
    for i in range(0, boot, BOOT_CHUNK):
        yield rng.integers(0, n, (min(BOOT_CHUNK, boot - i), n))


def _summary(t: dict[str, np.ndarray], weighted: bool, boot: int, seed: int) -> dict:
    px = float(t["px"].sum())
    if px <= 0:
        return {"px": 0.0 if weighted else 0, "called_05": None, "called_09": None, "mean": None, "log_loss": None,
                "ci_05": None}
    ci = None
    if boot > 0:
        got = np.concatenate([t["called_05"][idx].sum(1) / t["px"][idx].sum(1)
                              for idx in _draws(len(t["px"]), boot, seed)])
        ci = [float(v) for v in np.percentile(got, [2.5, 97.5])]
    return {"px": px if weighted else round(px), "called_05": float(t["called_05"].sum() / px),
            "called_09": float(t["called_09"].sum() / px), "mean": float(t["chance"].sum() / px),
            "log_loss": float(t["log_loss"].sum() / px), "ci_05": ci}


def summarise_rules(acc: dict) -> dict:
    """`linerules.rule_accuracy`'s table, per case over its texts: line recall mean and min, glyph false-line mean and
    max (and the text giving the max; None where no glyph px was far enough from the line to count)."""
    out = {}
    for case, per in acc.items():
        rec = [v["line_recall"] for v in per.values()]
        fal = {t: v["glyph_false_line"] for t, v in per.items() if v["glyph_false_line"] is not None}
        worst = max(fal, key=fal.get) if fal else None
        out[case] = {"line_recall_mean": float(np.mean(rec)), "line_recall_min": float(np.min(rec)),
                     "glyph_false_line_mean": float(np.mean(list(fal.values()))) if fal else None,
                     "glyph_false_line_max": fal[worst] if fal else None, "worst_text": worst}
    return out


@lru_cache(maxsize=1)
def _put_text_rules() -> str:
    return json.dumps(summarise_rules(linerules.rule_accuracy(linerules.line_px)))


def rule_table() -> dict:
    """The run rule's accuracy on the exact-truth cases with cv2.putText glyphs (`summarise_rules`), found once per
    process."""
    return json.loads(_put_text_rules())


def gate_sheets(sheets: list[data.SheetData], source: ChanceSource, quarters: tuple[str, ...], geo: Geo,
                boot: int = 1000, seed: int = 0) -> tuple[dict, dict]:
    """The gate's measures (see the module) of the chance from `source` on `sheets`, at the ink px of `quarters`
    (names in `paths.QUARTERS`).  Returns (result, tallies): the result is JSON-able (each measure's px, called_05,
    called_09, mean, log_loss and ci_05, with what defines them); the tallies are each measure's sums per (quarter,
    block), for `write` and `paired`.  A missing chance, or one not of its page's shape, stops it naming the sheet."""
    quarters = tuple(quarters)
    unknown = [q for q in quarters if q not in paths.QUARTERS]
    if not quarters or unknown:
        raise ValueError(f"quarters are some of {paths.QUARTERS}, not {quarters}")
    quarter_ix = np.array([paths.QUARTERS.index(q) for q in quarters], np.uint8)
    optional = {"m2_answered": callable(getattr(geo, "answered", None)),
                **{m: callable(getattr(geo, "context_weight", None)) for m in CONTEXT}}
    sums = {m: _Sums(TARGET[m]) for m in MEASURES if optional.get(m, True)}
    for sd in sheets:
        _add_sheet(sums, sd, _chance(source, sd), quarter_ix, geo)
    tallies = {m: s.table() for m, s in sums.items()}
    summary = {m: _summary(t, m in WEIGHTED, boot, seed) for m, t in tallies.items()}
    result = {
        "quarters": list(quarters), "sheets": [sd.sheet for sd in sheets],
        "called": {"0.5": CALLED_05, "0.9": CALLED_09}, "near_px": NEAR_PX,
        "point_bins": [name for name, _ in POINT_BINS], "boot": boot, "seed": seed,
        "targets": {m: TARGET[m] for m in MEASURES}, "misses": MISSES,
    }
    for m in MEASURES:
        if ":" in m:
            group, part = m.split(":")
            result.setdefault(group, {})[part] = summary[m]
        else:
            result[m] = summary.get(m)
    result["m1a_weight"] = getattr(geo, "weight_table", None)
    result["block_key"] = getattr(geo, "block_key", None)
    result["rules"] = {"rule": "the run rule (linerules.line_px): defines m1b, m2's base, the overlap band and which words are clean",
                       "putText": rule_table()}
    return result, tallies


def gate(source: ChanceSource, quarters: tuple[str, ...], geo: Geo | None = None, boot: int = 1000,
         seed: int = 0) -> tuple[dict, dict]:
    """`gate_sheets` on the 2000 sheets (`paths.SHEETS`, with their placements, Terra's words and Sol's points;
    `data.load_sheets`), by `CityGeo` unless `geo` is given; with the run rule's accuracy on the learned font beside
    cv2.putText's when `scripts/textlines/rules.py` has written it (LEARNED_RULES).  Returns (result, tallies)."""
    paths.check_sheets()
    sheets = data.load_sheets([paths.SERIES], hold_out_validation=False)
    result, tallies = gate_sheets(sheets, source, quarters, CityGeo() if geo is None else geo, boot, seed)
    if LEARNED_RULES.is_file():
        found = json.loads(LEARNED_RULES.read_text()).get("run / font.npz")
        if found:
            result["rules"]["font.npz"] = summarise_rules(found)
    return result, tallies


# --- what the gate writes, and two sets of runs compared -------------------------------------------------------------

def _f(v, digits: int = 4) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _px(v) -> str:
    return "-" if v is None else (f"{v:,.1f}" if isinstance(v, float) else f"{v:,}")


def markdown(result: dict) -> str:
    """The gate's result as a page (scores.md): every measure, M1a's weight table and the run rule's accuracy."""
    out = ["# The text-vs-line gate", "",
           (f"Quarters: {', '.join(result['quarters'])}. Sheets: {len(result['sheets'])}. Called: chance >= 0.5 "
           f"({result['called']['0.5']} of 255) and >= 0.9 ({result['called']['0.9']} of 255). Interval: 95%, "
           f"{result['boot']} bootstrap draws over blocks (a block in a quarter), seed {result['seed']}. "
           "Every px is ink; M1a's px are the sum of its weights. m1a_context and m1a_in_box_context are "
           "diagnostics, in no verdict."), "",
           "| measure | ink | target | px | called >= 0.5 | 95% interval | called >= 0.9 | mean chance | log loss |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in MEASURES:
        r = result[m.split(":")[0]]
        if r is not None and ":" in m:
            r = r[m.split(":")[1]]
        if r is None:
            out.append(f"| {m} | {WHAT[m]} | {TARGET[m]} | not measured | | | | | |")
            continue
        ci = "-" if r["ci_05"] is None else f"[{r['ci_05'][0]:.4f}, {r['ci_05'][1]:.4f}]"
        out.append(f"| {m} | {WHAT[m]} | {TARGET[m]} | {_px(r['px'])} | {_f(r['called_05'])} | {ci} | "
                   f"{_f(r['called_09'])} | {_f(r['mean'])} | {_f(r['log_loss'])} |")
    out += ["", "## What each measure misses", ""]
    out += [f"- **{family}:** {line}." for family, line in result.get("misses", MISSES).items()]
    out += ["", "## M1a's weight w(d)", ""]
    table = result.get("m1a_weight")
    if not table:
        out.append("The geo gave no table.")
    else:
        t, a = table["touch"], table["across"]
        out += [(f"d: grid px (4 ft) from the nearest district crack. Touching lots: {t['district_cracks']:,} district "
                 f"cracks, {t['other_cracks']:,} other both-known cracks; across a street: {a['district_cracks']:,} "
                 f"and {a['other_cracks']:,}. w(d) = max(0, f_D - f_L) / f_D below the class's cut, 0 from it on; "
                 f"beyond the cut (up to d = {table['J']}) the same formula is the context weight (m1a_context)."), "",
                ("The cut, per class: the first d past the excess's peak (excess e(d) = f_D - f_L) at which e is at "
                 "most the far-field level, the median e over the outer third of d = 0 .. J, or at which e stops "
                 "falling (e(d + 1) > e(d): another feature's ink begins). "
                 f"Touching lots: far-field level {t['far_level']:+.4f}, cut at d = {t['cut']} (weight on d <= "
                 f"{t['cut'] - 1}). Across a street: far-field level {a['far_level']:+.4f}, cut at d = {a['cut']} "
                 f"(weight on d <= {a['cut'] - 1})."), "",
                "| d | touching f_D | f_L | excess | w | beyond the cut | across f_D | f_L | excess | w | beyond the cut |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in range(table["J"] + 1):
            cells = []
            for c in (t, a):
                cells += [_f(c["f_district"][d], 3), _f(c["f_other"][d], 3),
                          f"{c['f_district'][d] - c['f_other'][d]:+.4f}", _f(c["w"][d], 3), _f(c["w_context"][d], 3)]
            out.append(f"| {d} | " + " | ".join(cells) + " |")
    rules = result["rules"]
    out += ["", "## The run rule's accuracy on exact-truth cases", "", f"{rules['rule']}.", ""]
    for glyphs in ("putText", "font.npz"):
        if glyphs not in rules:
            continue
        out += [(f"Glyphs: {glyphs}. Line recall mean (min over texts); glyph false-line mean (max over texts, "
                "the text)."), "", "| case | line recall | glyph false-line |", "|---|---|---|"]
        for case, v in rules[glyphs].items():
            worst = "" if v["worst_text"] is None else f", {v['worst_text']}"
            out.append(f"| {case} | {_f(v['line_recall_mean'], 3)} ({_f(v['line_recall_min'], 3)}) | "
                       f"{_f(v['glyph_false_line_mean'], 3)} ({_f(v['glyph_false_line_max'], 3)}{worst}) |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def write(result: dict, tallies: dict, out: Path) -> None:
    """Write the gate's scores.json (the result), scores.md (`markdown`) and blocks.npz (per measure, `<measure>.quarter`,
    `<measure>.block` and the float64 sums per block `<measure>.<field>` of FIELDS; `block_key`) into `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / SCORES_FILE).write_text(json.dumps(result, indent=1) + "\n")
    (out / MARKDOWN_FILE).write_text(markdown(result))
    arrays = {f"{m}.{f}": v for m, t in tallies.items() for f, v in t.items()}
    np.savez(out / BLOCKS_FILE, block_key=np.array(result.get("block_key") or ""), **arrays)


def _load_blocks(folder: Path, measure: str) -> tuple[dict[str, np.ndarray], str]:
    f = Path(folder) / BLOCKS_FILE
    if not f.is_file():
        raise FileNotFoundError(f"no {BLOCKS_FILE} in {folder}: write the gate's result there first")
    with np.load(f) as z:
        if f"{measure}.px" not in z.files:
            raise ValueError(f"{f} holds no {measure}")
        return {k: z[f"{measure}.{k}"] for k in ("quarter", "block", *FIELDS)}, str(z["block_key"])


def paired(a: list[Path], b: list[Path], measure: str, boot: int = 1000, seed: int = 0,
           stat: str = "called_05") -> dict:
    """The difference in `measure`'s `stat` (called_05, called_09, mean or log_loss) of the runs `b` less the runs `a`
    (each a folder `write` wrote), on the blocks (quarter, block) every run of both has px in: each arm's value is the
    mean over its runs of the ratio of sums; each bootstrap draw resamples those blocks once, the same draw for both
    arms.  Returns {"diff", "lo", "hi"} (the 95% interval) with each arm's value and the blocks used.  A block's px
    do not depend on the chance, so px that differ between runs on a block both scored stop it (ValueError): the runs
    were scored on different sheets, code or geo."""
    if not a or not b:
        raise ValueError("both arms need at least one run")
    if stat not in STATS:
        raise ValueError(f"stat is one of {', '.join(STATS)}, not {stat!r}")
    runs = [_load_blocks(p, measure) for p in [*a, *b]]
    if len({k for _, k in runs}) > 1:
        raise ValueError(f"the runs number their blocks differently: {sorted({k for _, k in runs})}")

    def units(t):
        return (t["quarter"].astype(np.int64) << 32) | (t["block"].astype(np.int64) + 1)

    common = reduce(np.intersect1d, [units(t)[t["px"] > 0] for t, _ in runs])
    if not len(common):
        raise ValueError(f"no block has {measure} px in every run")
    field = STATS[stat]
    cols = []
    for t, _ in runs:
        u = units(t)
        order = np.argsort(u)
        at = order[np.searchsorted(u, common, sorter=order)]
        cols.append((t["px"][at], t[field][at]))
    folders = [*a, *b]
    for (px, _), folder in zip(cols[1:], folders[1:]):
        if not np.allclose(px, cols[0][0]):
            bad = int(np.flatnonzero(~np.isclose(px, cols[0][0]))[0])
            raise ValueError(f"{measure}'s px differ between {folders[0]} and {folder} on a block both scored "
                             f"(quarter {int(common[bad] >> 32)}, block {int((common[bad] & 0xFFFFFFFF) - 1)}): px do "
                             "not depend on the chance, so the runs were scored on different sheets, code or geo")
    n_a = len(a)

    def arms(idx=None):
        vals = [num.sum(-1) / px.sum(-1) if idx is None else num[idx].sum(-1) / px[idx].sum(-1) for px, num in cols]
        return np.mean(vals[:n_a], axis=0), np.mean(vals[n_a:], axis=0)

    va, vb = arms()
    draws = np.concatenate([np.subtract(*arms(idx)[::-1]) for idx in _draws(len(common), boot, seed)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"measure": measure, "stat": stat, "diff": float(vb - va), "lo": float(lo), "hi": float(hi),
            "a": float(va), "b": float(vb), "blocks": len(common), "boot": boot, "seed": seed}
