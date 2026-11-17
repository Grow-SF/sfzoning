"""Composites inside real crops, for the text detector's training (`train.Recipe.composites`): real clean words ORed
onto real line work and real line pieces ORed across clean words, with exact masks, so the detector learns letters
(whole, their long strokes too) apart from the line ink through and beside them.

  clean word   a Terra word whose ink pieces all lie inside its box, with no line px (`data.SheetData.line_bits`, the
               run rule's px grown by 1 px) within CLEAN_PX px outside it (`linegate.clean_word_mask`, the gate's
               m2_clean_words): its ink is an exact mask of text, with the scan's own texture
  line piece   a connected piece of line ink (the line px that are ink) more than CLEAN_PX px from every Terra word,
               other pass's box, unmatched zone and held-out px (`free_line_ink`), cut by a grid of TILE px tiles,
               whose bounding diagonal is at least PIECE_PX px (`line_pieces`)
  crossed      a word with at least CROSS_PX line px inside its box (`crossings`)

The bank (`Bank.build`) holds references only: each clean word as (sheet, rectangle, kind, angle), each line piece as
(sheet, box, a seed px of it), per map, none from an area held out (a held-out quarter, the answered blocks), and per
map and kind the share of words crossed (`cross_by_kind`), measured on its own sheets, the words whose centre is in an
area held out left out.  Its own generator (`default_rng([seed, 1])`) draws only where each sheet's tile grid starts.

A composite (`Bank.crop`, drawn from the batch thread's generator alone) starts from the real crop the batch drew, as
`data.crop_targets(..., labels="lines-unknown")` makes it, and changes it at page scale, before the resize and the
turns, one of two ways (A_SHARE of them the first):

  (a) word onto lines   a clean word of the crop's map is ORed in, as it is printed (its own size and angle).  With
                        its kind's crossing share it goes on the crop's line ink (a px of its rectangle on a line ink
                        px, and CROSS_PX line px or more inside its rectangle), else off it (no line px within
                        CLEAN_PX px of its rectangle); and only where its rectangle and CLEAN_PX px around it hold no
                        Terra word, no other pass's box, no unmatched zone and no held-out px, inside the crop.  TRIES
                        places are tried, then the next clean word is drawn; after WORDS words the crop is left as it
                        is.  Its ink is text, known, with its kind and angle (where it overlaps line ink too); the line
                        ink under its rectangle off its letters stays as the crop had it: not text, known.
  (b) line onto a word  the crop is moved to centre on a clean word of the crop's map (on its own sheet), whose ink is
                        all text and known (its long strokes too, which the label rule leaves unknown), and a line
                        piece of that map, turned a random number of quarter turns, is ORed across it (a px of the
                        piece on a px of the word's rectangle).  The piece's px that were paper become ink, not text,
                        known as the px was (never where it was held out or unknown); those on the word's ink stay
                        text (the overlap: text, with weight 1, there being no measured weight); those on other ink
                        keep their targets.

A composite's masks are exact, so its text is only ever ink: the paper inside a word box there (the pasted word's
rectangle's too) is unknown, a lesson neither of text nor of not-text (in the real crops it is text, by the box rule).

`measure` gives what the bank's rules rest on: on sheets with areas left out, the share of words crossed by kind
(`cross_by_kind`), the clean words, their texture (`texture_stats`) and the texture of straight line pieces
(`line_texture`), which drawn lines would have to match (none is drawn: real line pieces only).

No stage imports this module (`train` takes the bank as any object with `.crop`), so no stage's import closure
changes with it.
"""
import collections
import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from sfzmaps.solver.textnet import data, linegate, zones

log = logging.getLogger(__name__)

CLEAN_PX = linegate.CLEAN_PX    # px around a clean word, a pasted word and a line piece kept clear (3)
CROSS_PX = 3                    # line px inside a word's box that make it crossed
PIECE_PX = 100                  # a line piece's least bounding diagonal, px
TILE = 256                      # px: line pieces are cut by a grid of tiles this size
TRIES = 20                      # places tried for one clean word
WORDS = 8                       # clean words tried before a crop is left as it is
A_SHARE = 0.5                   # the share of composites that are (a) word onto lines; the rest (b)
GAP_PX = 1.5                    # line_texture: px between two px across a piece that make them two strokes
WIDTH_TOP = 6                   # line_texture's width classes: 1 .. WIDTH_TOP - 1 px, and WIDTH_TOP px or more
LAGS = (1, 2, 4)                # line_texture's autocorrelation lags, px along the piece
DISTRIBUTIONS = zones.TEXTLINES_DIR / "distributions.json"
KIND_NAMES = tuple(data.KINDS)                    # by kind index
_REACH = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CLEAN_PX + 1, 2 * CLEAN_PX + 1))


# --- what the bank holds ---------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Words:
    """Clean words by reference: the sheet's index in the bank, the rectangle's corners (page px, as
    `SheetData.words`), the kind and the reading angle."""
    sheet: np.ndarray       # int32 (n,)
    rect: np.ndarray        # int32 (n, 4, 2)
    kind: np.ndarray        # uint8 (n,)
    angle: np.ndarray       # int16 (n,)

    def __len__(self) -> int:
        return len(self.sheet)

    def take(self, idx) -> "Words":
        return Words(self.sheet[idx], self.rect[idx], self.kind[idx], self.angle[idx])

    @staticmethod
    def join(parts: list["Words"]) -> "Words":
        if not parts:
            return Words(np.zeros(0, np.int32), np.zeros((0, 4, 2), np.int32), np.zeros(0, np.uint8),
                         np.zeros(0, np.int16))
        return Words(*(np.concatenate([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(Words)))


@dataclasses.dataclass(frozen=True)
class Pieces:
    """Line pieces by reference: the sheet's index in the bank, the piece's box (x0, y0, x1, y1; page px, the far
    edges excluded) and one px of it (x, y), which names it among the line ink in its box (`piece_mask`)."""
    sheet: np.ndarray       # int32 (m,)
    box: np.ndarray         # int32 (m, 4)
    seed: np.ndarray        # int32 (m, 2)

    def __len__(self) -> int:
        return len(self.sheet)

    @staticmethod
    def join(parts: list["Pieces"]) -> "Pieces":
        if not parts:
            return Pieces(np.zeros(0, np.int32), np.zeros((0, 4), np.int32), np.zeros((0, 2), np.int32))
        return Pieces(*(np.concatenate([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(Pieces)))


# --- the sheets' words and line ink ----------------------------------------------------------------------------------

def _line_page(sd: data.SheetData) -> np.ndarray:
    """A sheet's whole page of line px, bool.  ValueError when the sheet holds none (`zones.prepare` fills them)."""
    if sd.line_bits.shape != sd.ink_bits.shape:
        raise ValueError(f"{sd.series} {sd.sheet}: no line px (SheetData.line_bits {sd.line_bits.shape}): composites "
                         "need them (textnet.zones.prepare)")
    return np.unpackbits(sd.line_bits, axis=1, count=sd.width).view(bool)


def _rect_in(rect: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """A word rectangle's px in a page window (h x w px, top-left px (row y0, col x0)), filled as `data._paint` fills
    it (as on a page large enough), bool."""
    return data._paint(rect[None], np.ones(1, np.int32), y0, x0, h, w) > 0


def _clipped_box(sd: data.SheetData, rect: np.ndarray) -> tuple[int, int, int, int] | None:
    """A rectangle's bounding px on the page (y0, x0, h, w), cut to it; None when none is on it."""
    lo = np.maximum(rect.min(axis=0), 0)
    hi = np.minimum(rect.max(axis=0) + 1, [sd.width, sd.height])
    if (hi <= lo).any():
        return None
    return int(lo[1]), int(lo[0]), int(hi[1] - lo[1]), int(hi[0] - lo[0])


def crossings(sd: data.SheetData) -> np.ndarray:
    """Per Terra word of the sheet, int64 (n, 2): the line px inside its box (filled as `data._paint` fills it) and
    those of them that are ink.  ValueError for a sheet without its line px."""
    _line_page(sd)
    out = np.zeros((len(sd.words), 2), np.int64)
    for j, rect in enumerate(sd.words):
        box = _clipped_box(sd, rect)
        if box is None:
            continue
        inside = _rect_in(rect, *box) & data.line_window(sd, *box)
        out[j] = inside.sum(), (inside & data._ink_window(sd, *box)).sum()
    return out


def _held_at(sd: data.SheetData, x: int, y: int) -> bool:
    return bool(data._held(sd, y, x, 1, 1)[0, 0])


def _centre(sd: data.SheetData, rect: np.ndarray) -> tuple[int, int]:
    """A word's centre px (x, y): its rectangle's corners' mean, rounded, kept on the page (as `zones.zone_radius`)."""
    x, y = np.clip(np.rint(rect.astype(float).mean(axis=0)).astype(int), 0, [sd.width - 1, sd.height - 1])
    return int(x), int(y)


def _forbidden(sd: data.SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The px of a page window where nothing may be pasted and no line piece taken: in a Terra word, another pass's
    box or an unmatched zone, or held out."""
    word = data._words(sd, y0, x0, h, w)
    return (word > 0) | data._unknown(sd, word, y0, x0, h, w) | data._held(sd, y0, x0, h, w)


def free_line_ink(sd: data.SheetData, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The line ink (line px that are ink) of a page window that is more than CLEAN_PX px from every forbidden px
    (`_forbidden`, found on the window grown by CLEAN_PX, so the same whichever window holds a px), bool."""
    c = CLEAN_PX
    py0, px0 = max(y0 - c, 0), max(x0 - c, 0)
    py1, px1 = min(y0 + h + c, sd.height), min(x0 + w + c, sd.width)
    bad = _forbidden(sd, py0, px0, py1 - py0, px1 - px0)
    near = cv2.dilate(bad.view(np.uint8), _REACH)[y0 - py0:y0 - py0 + h, x0 - px0:x0 - px0 + w] > 0
    return data.line_window(sd, y0, x0, h, w) & data._ink_window(sd, y0, x0, h, w) & ~near


def _tile_pieces(sd: data.SheetData, y0: int, x0: int, h: int, w: int):
    """The line pieces of one tile: (box (x0, y0, x1, y1), seed (x, y), mask in the box) of each 8-connected piece of
    the tile's free line ink whose bounding diagonal is at least PIECE_PX px; the seed is its first px row by row."""
    free = np.ascontiguousarray(free_line_ink(sd, y0, x0, h, w))
    if not free.any():
        return
    n, label, stats, _ = cv2.connectedComponentsWithStats(free.view(np.uint8), connectivity=8)
    diag = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = [k for k in range(1, n) if diag[k] >= PIECE_PX]
    if not keep:
        return
    found, first = np.unique(label.ravel(), return_index=True)
    first_of = dict(zip(found.tolist(), first.tolist(), strict=True))
    for k in keep:
        bx, by, bw, bh = (int(v) for v in stats[k, :4])
        sy, sx = divmod(int(first_of[k]), w)
        yield ((x0 + bx, y0 + by, x0 + bx + bw, y0 + by + bh), (x0 + sx, y0 + sy),
               label[by:by + bh, bx:bx + bw] == k)


def _tiles(sd: data.SheetData, phase=(0, 0)):
    """The tiles of TILE px the page is cut into, the grid starting `phase` (row, col) px before the page's corner:
    (y0, x0, h, w), cut to the page."""
    for ty in range(-int(phase[0]), sd.height, TILE):
        for tx in range(-int(phase[1]), sd.width, TILE):
            y0, x0 = max(ty, 0), max(tx, 0)
            y1, x1 = min(ty + TILE, sd.height), min(tx + TILE, sd.width)
            if y1 > y0 and x1 > x0:
                yield y0, x0, y1 - y0, x1 - x0


def line_pieces(sd: data.SheetData, phase=(0, 0)) -> tuple[np.ndarray, np.ndarray]:
    """A sheet's line pieces (see the module), its tile grid starting `phase` (row, col) px before the page's corner:
    boxes int32 (m, 4) (x0, y0, x1, y1) and seeds int32 (m, 2) (x, y).  The sheet's held areas give none."""
    _line_page(sd)
    boxes, seeds = [], []
    for tile in _tiles(sd, phase):
        for box, seed, _ in _tile_pieces(sd, *tile):
            boxes.append(box)
            seeds.append(seed)
    return np.array(boxes, np.int32).reshape(-1, 4), np.array(seeds, np.int32).reshape(-1, 2)


def piece_mask(sd: data.SheetData, box, seed) -> np.ndarray:
    """A line piece's px in its box (bool (y1 - y0, x1 - x0)): the 8-connected piece of the box's free line ink that
    holds the seed px; the same piece `line_pieces` found (connectivity inside a part of a tile is the tile's)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    free = np.ascontiguousarray(free_line_ink(sd, y0, x0, y1 - y0, x1 - x0))
    _, label = cv2.connectedComponents(free.view(np.uint8), connectivity=8)
    k = label[int(seed[1]) - y0, int(seed[0]) - x0]
    if k == 0:
        raise ValueError(f"{sd.series} {sd.sheet}: no line ink at the piece's seed {tuple(seed)} in {tuple(box)}")
    return label == k


# --- the measures ----------------------------------------------------------------------------------------------------

def texture_stats(masks) -> dict:
    """The texture of ink masks (bool arrays; paper beyond their edges): `isolated_share` the share of ink px with no
    ink among their 8 neighbours, `pinholes_per_1000` the paper px whose 4 neighbours are all ink per 1,000 ink px,
    `pieces_per_word` the mean count of 8-connected pieces of ink per mask; with `masks` and `ink_px` counted."""
    n = ink = isolated = pinholes = pieces = 0
    for m in masks:
        n += 1
        m = np.asarray(m, bool)
        if not m.any():
            continue
        p = np.pad(m, 1)
        four = (p[:-2, 1:-1].astype(np.uint8) + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:])
        eight = four + p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
        ink += int(m.sum())
        isolated += int((m & (eight == 0)).sum())
        pinholes += int((~m & (four == 4)).sum())
        pieces += cv2.connectedComponents(np.ascontiguousarray(m).view(np.uint8), connectivity=8)[0] - 1
    return {"masks": n, "ink_px": ink, "isolated_share": isolated / ink if ink else None,
            "pinholes_per_1000": 1000 * pinholes / ink if ink else None, "pieces_per_word": pieces / n if n else None}


def _edges(mask: np.ndarray):
    """A straight piece's width and its two edges' offsets from their straight fits, or None when the piece is not one
    straight stroke.  Its px are taken along its main axis (the px' principal direction) in steps of 1 px (each px at
    its rounded distance along); across it, at each step, the farthest px each way is an edge.  One straight stroke:
    every step holds px, none holds two runs (px more than GAP_PX apart across), and neither edge strays more than the
    width (the median of the steps' spans, plus 1) from its least-squares line."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 3:
        return None
    p = np.c_[xs, ys].astype(float)
    p -= p.mean(axis=0)
    _, vec = np.linalg.eigh(p.T @ p)
    s, t = p @ vec[:, 1], p @ vec[:, 0]
    step = np.rint(s - s.min()).astype(np.int64)
    order = np.lexsort((t, step))
    ss, ts = step[order], t[order]
    if ((ss[1:] == ss[:-1]) & (np.diff(ts) > GAP_PX)).any():
        return None
    count = np.bincount(step)
    if (count == 0).any() or len(count) < 3:
        return None
    top = np.full(len(count), -np.inf)
    np.maximum.at(top, step, t)
    bottom = np.full(len(count), np.inf)
    np.minimum.at(bottom, step, t)
    width = float(np.median(top - bottom)) + 1
    along = np.arange(len(count), dtype=float)
    res = [e - np.polyval(np.polyfit(along, e, 1), along) for e in (top, bottom)]
    if max(float(np.abs(r).max()) for r in res) > width:
        return None
    return width, res


def _width_class(width: float) -> str:
    w = int(round(width))
    return str(max(w, 1)) if w < WIDTH_TOP else f"{WIDTH_TOP}+"


def line_texture(pieces) -> dict:
    """The texture of straight line pieces (bool masks), by width class (`_width_class` of `_edges`' width): `pieces`,
    `edge_px` (the steps along their edges, two a step), `sd` the SD of each edge's offset from its straight fit
    (pooled over every edge of the class), and `ac1`, `ac2`, `ac4` its autocorrelation at lags of 1, 2 and 4 px along
    the piece (the pooled lagged products over the pooled squares, each a mean; None when the offsets are all 0);
    `not_straight` counts the pieces left out (`_edges`)."""
    sums: dict[str, dict] = {}
    skipped = 0
    for mask in pieces:
        found = _edges(mask)
        if found is None:
            skipped += 1
            continue
        width, res = found
        c = sums.setdefault(_width_class(width), {"pieces": 0, "n": 0, "sq": 0.0,
                                                  **{f"lag{k}": [0.0, 0] for k in LAGS}})
        c["pieces"] += 1
        for r in res:
            c["n"] += len(r)
            c["sq"] += float((r * r).sum())
            for k in LAGS:
                if len(r) > k:
                    c[f"lag{k}"][0] += float((r[:-k] * r[k:]).sum())
                    c[f"lag{k}"][1] += len(r) - k
    out: dict = {}
    for cls in sorted(sums, key=lambda v: int(v.rstrip("+"))):
        c = sums[cls]
        var = c["sq"] / c["n"]
        out[cls] = {"pieces": c["pieces"], "edge_px": c["n"], "sd": float(np.sqrt(var)),
                    **{f"ac{k}": (c[f"lag{k}"][0] / c[f"lag{k}"][1] / var if var > 0 and c[f"lag{k}"][1] else None)
                       for k in LAGS}}
    out["not_straight"] = skipped
    return out


def _word_ink(sd: data.SheetData, rect: np.ndarray) -> np.ndarray | None:
    box = _clipped_box(sd, rect)
    return None if box is None else data._ink_window(sd, *box) & _rect_in(rect, *box)


def _share(words: int, crossed: int) -> dict:
    return {"words": int(words), "crossed": int(crossed), "share": crossed / words if words else None}


def _sheet_counts(sd: data.SheetData, line: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per word of a sheet whose `held` are the areas left out: in a training area (its centre px not held out),
    crossed (by line px, and by line ink), and clean (`linegate.clean_word_mask` with the line px)."""
    train = np.array([not _held_at(sd, *_centre(sd, r)) for r in sd.words], bool).reshape(-1)
    cross = crossings(sd) >= CROSS_PX
    return train, cross[:, 0], cross[:, 1], linegate.clean_word_mask(sd, line)


def _sheet_measure(sd: data.SheetData) -> tuple:
    """One sheet's part of `measure` (its held areas the areas left out)."""
    line = _line_page(sd)
    train, cross, cross_ink, clean = _sheet_counts(sd, line)
    del line
    kinds = sd.kinds.astype(np.int64)
    k = len(KIND_NAMES)
    counts = [np.bincount(kinds[train & m], minlength=k) for m in (train, cross, cross_ink, clean)]
    clean_ink = [m for m in (_word_ink(sd, r) for r in sd.words[train & clean]) if m is not None]
    pieces = [mask for tile in _tiles(sd) for _, _, mask in _tile_pieces(sd, *tile)]
    return counts, clean_ink, pieces


def measure(sheets: list[data.SheetData], areas=(), threads: int = 1) -> dict:
    """What the composites rest on, on `sheets` (with their line px) with `areas` left out (the words whose centre is
    in one, the line ink in or within CLEAN_PX px of one): `cross_by_kind` per kind and over all ("all", word-weighted)
    the words, those crossed (CROSS_PX line px or more inside the box) and their share (None without words);
    `cross_by_kind_ink` the same counting only line px that are ink; `clean_words` per kind; `clean_word_texture`
    (`texture_stats` of the clean words' ink); `line_texture` of the line pieces (`line_texture`, tile grid at the
    page's corner).  `threads` sheets at a time."""
    sheets = data.hold_out(sheets, areas) if tuple(areas) else sheets
    if threads > 1:
        with ThreadPoolExecutor(threads) as ex:
            parts = list(ex.map(_sheet_measure, sheets))
    else:
        parts = [_sheet_measure(sd) for sd in sheets]
    words, crossed, crossed_ink, clean_n = (sum(p[0][i] for p in parts) for i in range(4)) if parts else \
        (np.zeros(len(KIND_NAMES), np.int64) for _ in range(4))
    clean_ink = [m for p in parts for m in p[1]]
    return {
        "cross_by_kind": {**{n: _share(words[i], crossed[i]) for i, n in enumerate(KIND_NAMES)},
                          "all": _share(words.sum(), crossed.sum())},
        "cross_by_kind_ink": {**{n: _share(words[i], crossed_ink[i]) for i, n in enumerate(KIND_NAMES)},
                              "all": _share(words.sum(), crossed_ink.sum())},
        "clean_words": {n: int(clean_n[i]) for i, n in enumerate(KIND_NAMES)},
        "clean_word_texture": texture_stats(clean_ink),
        "line_texture": line_texture(m for p in parts for m in p[2]),
        "sheets": len(sheets),
    }


# --- the bank and its composites -------------------------------------------------------------------------------------

def _sheet_bank(sd: data.SheetData, j: int, phase) -> tuple[Words, Pieces, np.ndarray, np.ndarray]:
    """One sheet's part of the bank (its held areas the bank's): its clean words wholly clear of every held-out px
    (their rectangle's bounding box and CLEAN_PX px around it on the page), its line pieces, and per kind the words in
    a training area and those crossed."""
    line = _line_page(sd)
    train, cross, _, clean = _sheet_counts(sd, line)
    del line
    keep = []
    c = CLEAN_PX
    for i in np.flatnonzero(clean):
        lo, hi = sd.words[i].min(axis=0) - c, sd.words[i].max(axis=0) + 1 + c
        if (lo < 0).any() or hi[0] > sd.width or hi[1] > sd.height:
            continue
        if data._held(sd, int(lo[1]), int(lo[0]), int(hi[1] - lo[1]), int(hi[0] - lo[0])).any():
            continue
        keep.append(i)
    keep = np.array(keep, np.int64)
    words = Words(np.full(len(keep), j, np.int32), sd.words[keep].astype(np.int32), sd.kinds[keep].astype(np.uint8),
                  sd.angles[keep].astype(np.int16))
    boxes, seeds = line_pieces(sd, phase)
    pieces = Pieces(np.full(len(boxes), j, np.int32), boxes, seeds)
    kinds = sd.kinds.astype(np.int64)
    k = len(KIND_NAMES)
    return words, pieces, np.bincount(kinds[train], minlength=k), np.bincount(kinds[train & cross], minlength=k)


@dataclasses.dataclass(frozen=True)
class _Page:
    """A crop's targets at page scale (`_page_parts`), the window it is of, its line px and which px carry a word's
    reading angle."""
    ink: np.ndarray
    text: np.ndarray
    kind: np.ndarray
    angle: np.ndarray
    known: np.ndarray
    has_angle: np.ndarray
    line: np.ndarray
    y0: int
    x0: int
    src: int


def _page_parts(sd: data.SheetData, y: int, x: int, size: int, scale: float) -> _Page:
    """The crop `data.crop_targets(sd, y, x, size, scale, turns, labels="lines-unknown")` draws, at page scale and not
    turned (its src x src px of page, src = round(size / scale)), writable, with the window's line px."""
    src = int(round(size / scale))
    y0 = int(np.clip(y - src // 2, 0, sd.height - src))
    x0 = int(np.clip(x - src // 2, 0, sd.width - src))
    t = data.crop_targets(sd, y, x, src, 1.0, 0, labels="lines-unknown")
    return _Page(ink=np.array(t["ink"]), text=np.array(t["text"]), kind=np.array(t["kind"]),
                 angle=np.array(t["angle"]), known=np.array(t["known"]), has_angle=t["text"] > 0,
                 line=data.line_window(sd, y0, x0, src, src), y0=y0, x0=x0, src=src)


def _finish(page: _Page, size: int, turns: int) -> dict[str, np.ndarray]:
    """A page-scale crop as `data.crop_targets` hands one over: resized to size x size (nearest px), turned `turns`
    quarter turns counter-clockwise, the reading angle with it (in [-180, 180) where a word's, 0 elsewhere)."""
    parts = {k: getattr(page, k) for k in ("ink", "text", "kind", "angle", "known", "has_angle")}
    if page.src != size:
        parts = {k: cv2.resize(v.view(np.uint8) if v.dtype == bool else v, (size, size),
                               interpolation=cv2.INTER_NEAREST).astype(v.dtype) for k, v in parts.items()}
    has = parts.pop("has_angle")
    parts["angle"] = np.where(has, (parts["angle"].astype(np.int32) + 90 * turns + 180) % 360 - 180,
                              0).astype(np.int16)
    return {k: np.rot90(v, turns) for k, v in parts.items()}


def _exact(page: _Page) -> None:
    """A composite's masks are exact: the paper inside a word box is neither text nor not-text (unknown)."""
    paper = (page.text > 0) & ~page.ink
    page.text[paper] = 0
    page.known[paper] = False


def _normal(angle: int) -> int:
    return (int(angle) + 180) % 360 - 180


@dataclasses.dataclass
class _Sprite:
    """A clean word ready to paste: its frame (its rectangle's bounding box and CLEAN_PX px around it), the rectangle
    and the px within CLEAN_PX px of it in that frame, its ink there, its kind and angle."""
    rect: np.ndarray
    grown: np.ndarray
    ink: np.ndarray
    kind: int
    angle: int


@dataclasses.dataclass
class Bank:
    """Clean words and line pieces by reference, per map (series), with the sheets they are on (holding the bank's
    areas out) and per map and kind the share of words crossed; `crop` makes a composite (see the module).  `tally`
    counts what the composites were; `last` says what the last one was made of (for pictures and tests)."""
    sheets: list
    words: dict
    pieces: dict
    cross: dict
    crossing: dict
    a_share: float = A_SHARE
    tally: collections.Counter = dataclasses.field(default_factory=collections.Counter, compare=False)
    last: dict = dataclasses.field(default_factory=dict, compare=False)

    @classmethod
    def build(cls, sheets: list[data.SheetData], areas, seed: int, threads: int = 1) -> "Bank":
        """The bank of `sheets` (with their line px, `zones.prepare`) with `areas` held out (see the module); its own
        generator `default_rng([seed, 1])` draws each sheet's tile grid's start; `threads` sheets at a time.
        ValueError for a sheet without its line px."""
        held = data.hold_out(sheets, tuple(areas))
        rng = np.random.default_rng([seed, 1])
        phases = rng.integers(0, TILE, (len(held), 2))
        jobs = range(len(held))
        if threads > 1:
            with ThreadPoolExecutor(threads) as ex:
                parts = list(ex.map(lambda j: _sheet_bank(held[j], j, phases[j]), jobs))
        else:
            parts = [_sheet_bank(held[j], j, phases[j]) for j in jobs]
        words, pieces, cross, crossing = {}, {}, {}, {}
        for series in dict.fromkeys(sd.series for sd in held):
            mine = [p for p, sd in zip(parts, held, strict=True) if sd.series == series]
            words[series] = Words.join([p[0] for p in mine])
            pieces[series] = Pieces.join([p[1] for p in mine])
            n = sum(p[2] for p in mine)
            c = sum(p[3] for p in mine)
            crossing[series] = {KIND_NAMES[k]: _share(n[k], c[k]) for k in range(len(KIND_NAMES))}
            cross[series] = {k: float(c[k] / n[k]) if n[k] else 0.0 for k in range(len(KIND_NAMES))}
        return cls(sheets=held, words=words, pieces=pieces, cross=cross, crossing=crossing)

    def sizes(self) -> dict:
        """Per map: the clean words per kind, the line pieces, and the crossing counts and shares by kind."""
        return {s: {"clean_words": {n: int((w.kind == k).sum()) for k, n in enumerate(KIND_NAMES)},
                    "pieces": len(self.pieces[s]), "crossing": self.crossing[s]} for s, w in self.words.items()}

    def _sprite(self, words: Words, j: int) -> _Sprite:
        sd, rect = self.sheets[int(words.sheet[j])], words.rect[j]
        lo = rect.min(axis=0) - CLEAN_PX
        h, w = (int(v) for v in (rect.max(axis=0) + 1 + CLEAN_PX - lo)[::-1])
        inside = _rect_in(rect, int(lo[1]), int(lo[0]), h, w)
        return _Sprite(rect=inside, grown=cv2.dilate(inside.view(np.uint8), _REACH) > 0,
                       ink=data._ink_window(sd, int(lo[1]), int(lo[0]), h, w) & inside, kind=int(words.kind[j]),
                       angle=_normal(words.angle[j]))

    def crop(self, rng: np.random.Generator, sd: data.SheetData, y: int, x: int, size: int, scale: float,
             turns: int) -> dict[str, np.ndarray]:
        """A composite in place of `data.crop_targets(sd, y, x, size, scale, turns)`: the keys of a crop, from `rng`
        alone (see the module)."""
        words = self.words.get(sd.series)
        if rng.random() < self.a_share:
            return self._word_onto_lines(rng, sd, y, x, size, scale, turns, words)
        return self._line_onto_word(rng, sd, y, x, size, scale, turns, words)

    def _as_it_is(self, page: _Page, size: int, turns: int, why: str) -> dict[str, np.ndarray]:
        self.tally["left_as_is"] += 1
        self.tally[f"left_as_is:{why}"] += 1
        self.last = {"mode": "none", "why": why, "window": (page.y0, page.x0, page.src)}
        return _finish(page, size, turns)

    def _word_onto_lines(self, rng, sd, y, x, size, scale, turns, words) -> dict[str, np.ndarray]:
        page = _page_parts(sd, y, x, size, scale)
        if words is None or not len(words):
            return self._as_it_is(page, size, turns, "no clean words")
        src = page.src
        forbidden = (page.text > 0) | ~page.known
        targets = np.flatnonzero(page.line & page.ink & ~forbidden)
        cross = self.cross[sd.series]
        for _ in range(WORDS):
            j = int(rng.integers(len(words)))
            sprite = self._sprite(words, j)
            on = bool(rng.random() < cross.get(sprite.kind, 0.0))
            fh, fw = sprite.rect.shape
            placed = None
            if fh <= src and fw <= src and (len(targets) or not on):
                rect_px = np.flatnonzero(sprite.rect)
                for _ in range(TRIES):
                    if on:
                        ty, tx = divmod(int(targets[rng.integers(len(targets))]), src)
                        ry, rx = divmod(int(rect_px[rng.integers(len(rect_px))]), fw)
                        oy, ox = ty - ry, tx - rx
                        if not (0 <= oy <= src - fh and 0 <= ox <= src - fw):
                            continue
                    else:
                        oy, ox = int(rng.integers(src - fh + 1)), int(rng.integers(src - fw + 1))
                    win = (slice(oy, oy + fh), slice(ox, ox + fw))
                    if (forbidden[win] & sprite.grown).any():
                        continue
                    if on and (page.line[win] & sprite.rect).sum() < CROSS_PX:
                        continue
                    if not on and (page.line[win] & sprite.grown).any():
                        continue
                    placed = (oy, ox)
                    break
            if placed is None:
                self.tally["words_failed"] += 1
                continue
            oy, ox = placed
            win = (slice(oy, oy + fh), slice(ox, ox + fw))
            ink = sprite.ink
            page.ink[win] |= ink
            page.text[win][ink] = 1
            page.kind[win][ink] = sprite.kind
            page.angle[win][ink] = sprite.angle
            page.known[win][ink] = True
            page.has_angle[win][ink] = True
            page.known[win][sprite.rect & ~page.ink[win]] = False     # the pasted box's paper: unknown
            _exact(page)
            self.tally["a"] += 1
            self.tally["a_on" if on else "a_off"] += 1
            self.last = {"mode": "a", "word": j, "on_line": on, "placed": (oy, ox, fh, fw),
                         "window": (page.y0, page.x0, src)}
            return _finish(page, size, turns)
        return self._as_it_is(page, size, turns, "no place")

    def _line_onto_word(self, rng, sd, y, x, size, scale, turns, words) -> dict[str, np.ndarray]:
        pieces = self.pieces.get(sd.series)
        if words is None or not len(words) or pieces is None or not len(pieces):
            return self._as_it_is(_page_parts(sd, y, x, size, scale), size, turns, "no clean words or pieces")
        j = int(rng.integers(len(words)))
        wsd, rect = self.sheets[int(words.sheet[j])], words.rect[j]
        cx, cy = _centre(wsd, rect)
        page = _page_parts(wsd, cy, cx, size, scale)
        src = page.src
        in_rect = _rect_in(rect, page.y0, page.x0, src, src)
        word = in_rect & page.ink
        page.text[word] = 1
        page.known[word] = True
        page.kind[word] = int(words.kind[j])
        page.angle[word] = _normal(words.angle[j])
        page.has_angle[word] = True
        k = int(rng.integers(len(pieces)))
        piece = np.rot90(piece_mask(self.sheets[int(pieces.sheet[k])], pieces.box[k], pieces.seed[k]),
                         int(rng.integers(4)))
        ph, pw = piece.shape
        on_piece = np.flatnonzero(piece)
        ay, ax = divmod(int(on_piece[rng.integers(len(on_piece))]), pw)
        rect_px = np.flatnonzero(in_rect)
        by, bx = divmod(int(rect_px[rng.integers(len(rect_px))]), src)
        oy, ox = by - ay, bx - ax
        y0, y1, x0, x1 = max(oy, 0), min(oy + ph, src), max(ox, 0), min(ox + pw, src)
        line = piece[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        win = (slice(y0, y1), slice(x0, x1))
        paper = line & ~page.ink[win]
        page.ink[win] |= line
        page.text[win][paper] = 0
        _exact(page)
        self.tally["b"] += 1
        self.last = {"mode": "b", "word": j, "sheet": int(words.sheet[j]), "centre": (cy, cx), "piece": k,
                     "placed": (oy, ox, ph, pw), "window": (page.y0, page.x0, src)}
        return _finish(page, size, turns)
