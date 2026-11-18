"""The text reader's training crops beyond Terra's boxes: crops of our own word boxes, sloppy crops of Terra's words,
and masking outside the word.  The reader reads 96% of the owner's answered labels right in Terra's tight boxes and
41% in our own (`wordread.measures`): it has never practised on boxes like ours.  Nothing here changes what a stage
does: `textread.train.fit` takes these crops as `extra` (`extra_pool`), behind its off-by-default options
(`READER_OPTIONS`).  Every function that makes or measures crops for a reader takes its held-out quarter, and stops
without one.  No cap or cutoff is set by hand: what a sloppy crop holds is drawn to match what was measured.

  Terra's words   (`terra_frame`) the 2000 sheets' words from `solver-text-crops`' rows: Terra's string as the reader
                  learns it, its rectangle as `crops.rectify` cuts it (`cx`, `cy` its centre's page px index, `length`,
                  `height`, the reading `angle`), the held-out areas its crop touches (`crops.TOUCH`), and its
                  rectangle as Task 1 matches it (`c0x`..`c3y`, `textnet.boxes.word_rect`)
  matching        Task 1's (`measures`): each of our boxes (its grouped rectangle) goes to the Terra word it meets
                  best, kept at IoU >= IOU (`matches`)
  own-box crops   (`own_box_crops`) each matched box cut as the reader reads it (`crops.box_crop`), turned a half turn
                  when Terra's word reads the other way along its axis, with Terra's string as its target (ABCNet v2's
                  training on its own detections).  Kept only when neither the crop's footprint nor Terra's word
                  touches the reader's held-out quarter or an answered block.  `extra_pool(kind_weights="terra")`
                  draws them in the kinds' shares of the training quarters' Terra words (`own_draw_weights`)
  the offsets     (`offset_distribution`) how our matched boxes sit on Terra's words, pair by pair, in Terra's word
                  heights H, in our box's own frame (its reading direction the axis turned towards Terra's): our
                  trimmed box's start and end along it, each minus Terra's (`start`, `end`; Terra's word from -L/2 to
                  L/2 about its centre), its centre's offset down the letters (`cross`), its angle minus Terra's
                  (`angle`, degrees in [-90, 90]), its height over Terra's (`height_ratio`), the largest share of
                  another Terra word's rectangle inside the box (`neighbour`, 0 for none) and that neighbour's ink
                  against the word's (`neighbour_ink` touching it, `neighbour_on` on its letters; `ink_overlaps`):
                  cutting Terra's word with one pair's offsets cuts that box's own crop.  The line ink crossing a word
                  (`line_measures`): the line px that are ink inside its rectangle, in line ink components that run
                  on outside it (a crossing line; a letter's straight stroke stays inside); a word holding any is
                  crossed (`line_share`), with that ink's
                  thickness (`line_thickness`, `thickness` on the sheet's line ink) and share of the word's ink
                  (`line_cover`); and the kinds' shares of the training quarters' Terra words (`kind_shares`)
  ink overlap     (`ink_overlaps`) touching: the share of a neighbour's ink (in its rectangle and off the word's; all
                  of it when its rectangle lies in the word's) in ink components (`read.ink_components`) that also
                  hold the word's own ink (in the word's rectangle, off the neighbour's); on: the share of the
                  neighbour's ink inside the extent of the word's own ink along and across its reading direction.  Of
                  ink, not of the loose rectangles
  sloppy crops    (`sloppy_crop`) Terra's word cut with one matched pair's offsets, its neighbour share and ink overlaps
                  with them (drawn together: their spread and their ties are the measured ones), as Weinman's
                  synthesizer makes its words.  With probability `line_share` a real line piece (`Lines`: a
                  `composite.Bank`'s pieces, `piece_patches`, drawn by `calibrate_lines`' weights so the pasted
                  pieces' thickness and their cover of the word's ink both follow the crossings' measured
                  distributions, a draw kept with its cover's chance and drawn again otherwise, so the line rate stays
                  `line_share`), turned a random number of quarter turns, is ORed through a letter px of the word.
                  When the pair held a neighbour, that real neighbour (`neighbour_pose`, a training word's) is ORed in
                  at its own place, angle and scale in the word's frame: its share inside the box and its ink on the
                  word's letters come together from the same real example (the empirical joint; nothing is fitted
                  or searched).  The page itself is never changed.  `sloppy_crops` makes them for a reader: only of
                  words touching neither its held-out quarter nor the answered blocks, and kept only when the crop's
                  footprint touches neither
  masking         (`mask_outside`; `read.kept_ink`, `crops.box_mask`) a crop is paper outside the ink components that
                  hold ink in its own rectangle (our box's, Terra's, or a sloppy crop's drawn one): ink touching the
                  word is never blanked, a neighbouring word or stray mark wholly outside is; no detector's call is in
                  it.  Terra's crops (`real_masks`, every map) and the sloppy crops are masked so too in a training
                  with the mask, and our boxes when read with it (`read.build_read(mask=True)`)

`build_offsets` measures the offsets for each held-out quarter (the pairs whose box is outside it and whose word
touches neither it nor an answered block) from today's boxes and writes OFFSETS_FILE
(`python -m sfzmaps.solver.wordread.readercrops`).  Seeded throughout; no stage imports this module.
"""
import collections
import dataclasses
import json
import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

from sfzmaps.solver import grid, paths, runs, wordboxes
from sfzmaps.solver.codereader import labels as L
from sfzmaps.solver.textnet import boxes as textboxes
from sfzmaps.solver.textnet import composite, data, reader_stage, zones
from sfzmaps.solver.textread import crops as K
from sfzmaps.solver.textread import read, store
from sfzmaps.solver.wordread import measures

log = logging.getLogger(__name__)

IOU = 0.5
OFFSETS = ("start", "end", "cross", "angle", "height_ratio")
QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
REACH = 2 * wordboxes.INK_R + 2   # px: ink farther apart than this never shares an ink component (each grows INK_R)
PROPOSALS = 20000                 # line draws simulated per round of `calibrate_lines`
ROUNDS = 4                        # rounds of `calibrate_lines`' raking of the piece weights
KIND_NAMES = tuple(data.KINDS)    # zoning code, block number, street name, other
OFFSETS_FILE = runs.EXPERIMENTS_DIR / "wordread" / "reader" / "offsets.json"
TOUCH = store.TOUCH
mask_outside = K.mask_outside


def _need_quarter(quarter, what: str) -> int:
    if quarter is None:
        raise ValueError(f"{what} needs the reader's held-out quarter: with none it would use that quarter's words")
    return int(quarter)


# --- Terra's words ----------------------------------------------------------------------------------------------------

def _rect(cx: float, cy: float, length: float, height: float, angle: float) -> np.ndarray:
    """A word rectangle's corners (4, 2), page px index coordinates: start-top, end-top, end-bottom, start-bottom."""
    (ux, uy), (vx, vy) = L._axes(angle)
    u, v = np.array([ux, uy]) * length / 2, np.array([vx, vy]) * height / 2
    c = np.array([cx, cy], float)
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


def _word_rect(t) -> np.ndarray:
    return _rect(t["cx"], t["cy"], t["length"], t["height"], t["angle"])


LINE_COLUMNS = ("line_ink", "cross_ink", "word_ink", "line_thick")


def terra_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Terra's words on the 2000 sheets with a target, from rows of `solver-text-crops`' crops.parquet (see the
    module), positions 0..n-1; LINE_COLUMNS too when the rows have them."""
    w = rows[(rows["series"] == paths.SERIES) & (rows["word"] >= 0) & rows["text"].notna()]
    w = w[w["text"].astype(str) != ""].reset_index(drop=True)
    box = w[["box_x0", "box_y0", "box_x1", "box_y1"]].to_numpy(float)
    angle = w["angle"].to_numpy(float)
    rects = np.array([textboxes.word_rect(b, a) for b, a in zip(box, angle)], float).reshape(-1, 4, 2)
    out = pd.DataFrame({"sheet": w["sheet"].astype(str).to_numpy(), "word": w["word"].astype(int).to_numpy(),
                        "text": w["text"].astype(str).to_numpy(), "kind": w["kind"].to_numpy(),
                        "quarter": w["quarter"].astype(int).to_numpy(), "angle": angle,
                        **{c: box[:, k] for k, c in enumerate(("box_x0", "box_y0", "box_x1", "box_y1"))},
                        "cx": (box[:, 0] + box[:, 2] - 1) / 2, "cy": (box[:, 1] + box[:, 3] - 1) / 2,
                        "length": w["length"].to_numpy(float), "height": w["height"].to_numpy(float),
                        **{t: w[t].astype(bool).to_numpy() for t in TOUCH},
                        **dict(zip(measures.C_COLS, rects.reshape(-1, 8).T))})
    for c in LINE_COLUMNS:
        if c in w:
            out[c] = w[c].to_numpy()
    return out


def _training_words(terra: pd.DataFrame, quarter: int) -> np.ndarray:
    """Which words of `terra` touch neither `quarter` nor an answered block, bool."""
    return ~terra["touch_blocks"].to_numpy(bool) & ~terra[f"touch_{paths.QUARTERS[quarter]}"].to_numpy(bool)


class DiskPages:
    """The 2000 pages read from disk by sheet (`pages[sheet]`), the last one kept."""

    def __init__(self):
        self._last: dict = {}

    def __getitem__(self, sheet: str) -> np.ndarray:
        if sheet not in self._last:
            self._last.clear()
            self._last[sheet] = data.load_page(sheet)
        return self._last[sheet]


def _pages(pages):
    """A page getter: `pages[sheet]`, or each 2000 page read from disk (`DiskPages`)."""
    pages = DiskPages() if pages is None else pages
    return lambda sheet: pages[sheet]


def _window(page: np.ndarray, rects, margin: int) -> tuple[np.ndarray, tuple[int, int]]:
    """The page window holding rectangles (each 4 x 2, page px index coordinates) and `margin` px around them, cut to
    the page, and its top-left px (x0, y0)."""
    allc = np.concatenate([np.asarray(r, float).reshape(-1, 2) for r in rects])
    x0, y0 = max(int(np.floor(allc[:, 0].min())) - margin, 0), max(int(np.floor(allc[:, 1].min())) - margin, 0)
    x1 = min(int(np.ceil(allc[:, 0].max())) + margin + 1, page.shape[1])
    y1 = min(int(np.ceil(allc[:, 1].max())) + margin + 1, page.shape[0])
    return page[y0:max(y0, y1), x0:max(x0, x1)], (x0, y0)


def ink_overlaps(win: np.ndarray, origin, word: np.ndarray, neighbour: np.ndarray, angle: float,
                 word_ink: np.ndarray | None = None) -> tuple[float | None, float | None]:
    """A neighbour's ink against a word's, on a grey window (see the module): `word` and `neighbour` their rectangles
    (4 x 2, page px index coordinates), `angle` the word's reading angle.  `touch`: the share of the neighbour's ink
    (in its rectangle and off the word's; all of it when its rectangle lies in the word's) in ink components
    (`read.ink_components`) that also hold the word's own ink (in the word's rectangle, off the neighbour's).  `on`:
    the share of the neighbour's ink (in its rectangle) inside the extent of the word's own ink along and across its
    reading direction: on the word's letters, not merely inside its loose rectangle.  The word's own ink is, on a real
    page, its rectangle's ink off the neighbour's (the best the one ink layer allows); for a neighbour pasted in, the
    word's ink before the paste (`word_ink`, bool, the window's shape).  (None, None) when the neighbour has no ink in
    the window."""
    ink, lab = read.ink_components(win)
    w_in = read.in_rect(win.shape, origin, word)
    n_in = read.in_rect(win.shape, origin, neighbour)
    own = n_in & ~w_in & ink
    if not own.any():
        own = n_in & ink
    if not own.any():
        return None, None
    word_ink = w_in & ~n_in & ink if word_ink is None else word_ink & ink
    mine = np.unique(lab[word_ink])
    touch = float(np.isin(lab[own], mine[mine > 0]).mean())
    if not word_ink.any():
        return touch, 0.0
    (ux, uy), (vx, vy) = L._axes(float(angle))
    cx, cy = np.asarray(word, float).mean(0) - np.asarray(origin, float)

    def frame(m):
        ys, xs = np.nonzero(m)
        return (xs - cx) * ux + (ys - cy) * uy, (xs - cx) * vx + (ys - cy) * vy

    p, q = frame(word_ink)
    np_, nq = frame(n_in & ink)
    on = ((np_ >= p.min() - 0.5) & (np_ <= p.max() + 0.5) & (nq >= q.min() - 0.5) & (nq <= q.max() + 0.5))
    return touch, float(on.mean())


def neighbour_pose(page: np.ndarray, word_geo, nb_geo) -> dict | None:
    """A real pair's neighbour in its word's own frame, to be drawn whole with that pair (`sloppy_crop`): the
    neighbour's ink (its rectangle's px off the word's; when its rectangle lies in the word's, its px less the ink
    components holding the word's own ink) sampled on a whole-px grid along the word's reading direction and down its
    letters, from the word's nearer end, and its rectangle there.  `word_geo`, `nb_geo`: (cx, cy, length, height,
    angle) of the word and of the neighbour (page px index).  {"side": +1 past the word's end or -1 before its start,
    "h": the word's height, "p0", "q0": the grid's first place (px, from the end along, from the centre across),
    "cells": [[i, j], ...] the grid's ink, "rect": the neighbour's rectangle's corners in the same frame}; None when
    the neighbour has no ink of its own, or no place of the grid falls on it."""
    cx, cy, length, height, angle = (float(v) for v in word_geo)
    word, nb = _rect(*word_geo), _rect(*nb_geo)
    win, (ox, oy) = _window(page, [word, nb], REACH)
    ink, lab = read.ink_components(win)
    w_in, n_in = read.in_rect(win.shape, (ox, oy), word), read.in_rect(win.shape, (ox, oy), nb)
    moving = n_in & ~w_in & ink
    if not moving.any():
        own = np.unique(lab[w_in & ~n_in & ink])
        moving = n_in & ink & ~np.isin(lab, own[own > 0])
    if not moving.any():
        return None
    (ux, uy), (vx, vy) = L._axes(angle)
    ys, xs = np.nonzero(moving)
    dx, dy = xs + ox - cx, ys + oy - cy
    p, q = dx * ux + dy * uy, dx * vx + dy * vy
    side = 1 if p.mean() >= 0 else -1
    end = side * length / 2
    p0, q0 = float((p - end).min()), float(q.min())
    gi = np.arange(int(np.ceil((p - end).max() - p0)) + 1)
    gj = np.arange(int(np.ceil(q.max() - q0)) + 1)
    ii, jj = np.meshgrid(gi, gj)
    pr, qq = p0 + ii + end, q0 + jj                 # each cell's centre in the word's frame (exact at a quarter turn)
    xi = np.rint(cx + pr * ux + qq * vx - ox).astype(int)
    yi = np.rint(cy + pr * uy + qq * vy - oy).astype(int)
    ok = (xi >= 0) & (yi >= 0) & (xi < win.shape[1]) & (yi < win.shape[0])
    hit = np.zeros(ii.shape, bool)
    hit[ok] = moving[yi[ok], xi[ok]]
    if not hit.any():                               # the grid's places all fall off its ink: nothing to draw
        return None
    rx, ry = nb[:, 0] - cx, nb[:, 1] - cy
    return {"side": side, "h": height, "p0": p0, "q0": q0,
            "cells": np.stack([ii[hit], jj[hit]], 1).astype(int).tolist(),
            "rect": np.stack([rx * ux + ry * uy - end, rx * vx + ry * vy], 1).tolist()}


# --- our boxes against Terra's words ----------------------------------------------------------------------------------

def matches(boxes: pd.DataFrame, terra: pd.DataFrame, iou: float = IOU) -> pd.DataFrame:
    """Our boxes matched to a Terra word at `iou` or more (Task 1's rule: each box to the word it meets best): `box`
    and `word` their positions in the frames, `iou`."""
    ti, bj, v = measures._overlaps(boxes, terra)
    word, best = measures._best_word(ti, bj, v, len(boxes))
    pos = np.flatnonzero((best >= iou) & (word >= 0))
    return pd.DataFrame({"box": pos, "word": word[pos], "iou": best[pos]})


def _turned(terra_angle: float, axis: float) -> bool:
    """Whether Terra's word reads the other way along our box's axis (more than a quarter turn from it)."""
    return math.cos(math.radians(float(terra_angle) - float(axis))) < 0


def _corners(b) -> np.ndarray:
    return np.array([[b[f"t{i}x"], b[f"t{i}y"]] for i in range(4)], float)


def _pair_offsets(b, t) -> tuple[float, ...]:
    """One matched pair's offsets (see the module): our box `b` (a boxes.parquet row), Terra's word `t`."""
    axis = float(b["axis"]) + (180.0 if _turned(t["angle"], b["axis"]) else 0.0)
    (ux, uy), (vx, vy) = L._axes(axis)
    ox, oy = _corners(b).mean(0) - 0.5
    dx, dy = ox - t["cx"], oy - t["cy"]
    along, across = dx * ux + dy * uy, dx * vx + dy * vy
    length, height = float(t["length"]), float(t["height"])
    lo, ho = float(b["t_length"]), float(b["t_height"])
    return ((along - lo / 2 + length / 2) / height, (along + lo / 2 - length / 2) / height, across / height,
            (axis - float(t["angle"]) + 180.0) % 360.0 - 180.0, ho / height)


def _held(flags, quarter: int) -> bool:
    return bool(flags["touch_blocks"]) or bool(flags[f"touch_{paths.QUARTERS[quarter]}"])


def _summary(x, values: bool = False) -> dict:
    """`n` and `quantiles` (QUANTILES) of measured values (None without any); with `values`, the values too."""
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    out = {"n": int(len(x)),
           "quantiles": dict(zip(map(str, QUANTILES), np.quantile(x, QUANTILES).tolist())) if len(x) else None}
    if values:
        out["values"] = x.tolist()
    return out


def _area(r: np.ndarray) -> float:
    return float(wordboxes.poly_area(np.asarray(r, float).reshape(1, 4, 2))[0])


def _meet(a: np.ndarray, b: np.ndarray) -> float:
    """The area two convex quadrilaterals (4, 2) share."""
    return float(cv2.intersectConvexConvex(np.asarray(a, np.float32), np.asarray(b, np.float32))[0])


def offset_distribution(boxes: pd.DataFrame, terra: pd.DataFrame, *, quarter: int, iou: float = IOU,
                        pages=None) -> dict:
    """The measured spread of our matched boxes against Terra's words for the reader holding `quarter` out (see the
    module): `n` pairs, `offsets` (each a list, pair by pair, `neighbour` and `neighbour_ink` too), the offsets'
    `mean`, `sd` and `quantiles`, `turned` (pairs read a half turn from the axis), `line_share` (with `cross_ink`),
    `line_thickness` and `line_cover` (`_summary` with the values; with `word_ink` and `line_thick`), for comparison
    `line_share_any` and `line_cover_any` (every line px in the rectangle, a word crossed at composite.CROSS_PX of
    them: the first measure), `neighbour_share`, `neighbour_part`, `neighbour_ink` (`_summary` with the values;
    measured on `pages`, sheet -> page, None for none) and `kind_shares`.  Left out: pairs whose word touches an
    answered block or the quarter, or whose box is in it.  ValueError without a quarter."""
    quarter = _need_quarter(quarter, "offset_distribution")
    m = matches(boxes, terra, iou)
    keep = [not _held(terra.iloc[w], quarter) and int(boxes.iloc[p]["quarter"]) != quarter
            for p, w in zip(m["box"], m["word"])]
    m = m[np.array(keep, bool)].reset_index(drop=True)
    words = m["word"].to_numpy()
    rows = [_pair_offsets(boxes.iloc[p], terra.iloc[w]) for p, w in zip(m["box"], words)]
    off = np.array(rows, float).reshape(-1, len(OFFSETS))
    out = {"iou": iou, "quarter": paths.QUARTERS[quarter], "n": int(len(m)),
           "offsets": {k: off[:, j].tolist() for j, k in enumerate(OFFSETS)},
           "mean": {k: float(off[:, j].mean()) if len(m) else None for j, k in enumerate(OFFSETS)},
           "sd": {k: float(off[:, j].std()) if len(m) else None for j, k in enumerate(OFFSETS)},
           "quantiles": {k: _summary(off[:, j])["quantiles"] for j, k in enumerate(OFFSETS)},
           "turned": int(sum(_turned(terra.iloc[w]["angle"], boxes.iloc[p]["axis"])
                             for p, w in zip(m["box"], words)))}
    out["line_share"], out["line_words"] = None, None
    out["line_thickness"], out["line_cover"] = _summary([], True), _summary([], True)
    if "cross_ink" in terra:
        crossed = terra["cross_ink"].to_numpy()[words] > 0
        out["line_share"] = float(crossed.mean()) if len(m) else None
        out["line_words"] = int(crossed.sum())
        mine = words[crossed]
        ink = terra["word_ink"].to_numpy(float)[mine]
        out["line_thickness"] = _summary(terra["line_thick"].to_numpy(float)[mine], True)
        out["line_cover"] = _summary(terra["cross_ink"].to_numpy(float)[mine] / np.maximum(ink, 1), True)
    if "line_ink" in terra:
        old = terra["line_ink"].to_numpy()[words] >= composite.CROSS_PX
        out["line_share_any"] = float(old.mean()) if len(m) else None
        ink = terra["word_ink"].to_numpy(float)[words[old]]
        out["line_cover_any"] = _summary(terra["line_ink"].to_numpy(float)[words[old]] / np.maximum(ink, 1))
    # each matched box's largest share of another Terra word's rectangle inside its trimmed box, and that word's ink
    # overlap with the matched word
    part, nb = np.zeros(len(m)), np.full(len(m), -1)
    sheet_of = boxes["sheet"].astype(str).to_numpy()
    tsheet = terra["sheet"].astype(str).to_numpy()
    geo = terra[["cx", "cy", "length", "height", "angle"]].to_numpy(float)
    for sheet in sorted(set(sheet_of[m["box"]])):
        mine = np.flatnonzero(sheet_of[m["box"]] == sheet)
        others = np.flatnonzero(tsheet == sheet)
        ours = np.array([_corners(boxes.iloc[p]) - 0.5 for p in m["box"].to_numpy()[mine]], float).reshape(-1, 4, 2)
        theirs = np.array([_rect(*g) for g in geo[others]], float).reshape(-1, 4, 2)
        i, j, inter = wordboxes.pairs(ours, theirs)
        share = inter / np.maximum(wordboxes.poly_area(theirs)[j], 1e-9)
        for a, b, s in zip(i, j, share):
            k = mine[a]
            if others[b] != words[k] and s > part[k]:
                part[k], nb[k] = s, others[b]
    holds = part > 0
    touch: list = [None] * len(m)
    on: list = [None] * len(m)
    pose: list = [None] * len(m)
    if pages is not None:
        for k in np.flatnonzero(holds):
            wr, nr = _rect(*geo[words[k]]), _rect(*geo[nb[k]])
            page = pages[tsheet[words[k]]]
            win, origin = _window(page, [wr, nr], REACH)
            touch[k], on[k] = ink_overlaps(win, origin, wr, nr, geo[words[k], 4])
            if not _held(terra.iloc[nb[k]], quarter):         # a neighbour in a held-out area is never drawn
                pose[k] = neighbour_pose(page, geo[words[k]], geo[nb[k]])
    out["offsets"]["neighbour"] = part.tolist()
    out["offsets"]["neighbour_ink"] = touch
    out["offsets"]["neighbour_on"] = on
    out["offsets"]["neighbour_pose"] = pose
    out["neighbour_poses"] = int(sum(v is not None for v in pose))
    out["neighbour_share"] = float(holds.mean()) if len(m) else None
    out["neighbour_part"] = part[holds].tolist()
    out["neighbour_ink"] = _summary(touch, True)
    out["neighbour_on"] = _summary(on, True)
    kinds = collections.Counter(terra.loc[_training_words(terra, quarter), "kind"])
    total = sum(kinds.values())
    out["kind_shares"] = {k: kinds.get(k, 0) / total if total else 0.0 for k in KIND_NAMES}
    return out


# --- which held-out areas a crop touches ------------------------------------------------------------------------------

class HeldAreas:
    """Which held-out areas a page window of a 2000 sheet touches, as `solver-text-crops` records its crops' (TOUCH:
    each quarter of the city, then the answered blocks; through every px's map point, the sheet's placement):
    `touch(sheet, (x0, y0, x1, y1), page shape) -> {name: bool}`."""

    def __init__(self, blocks=None):
        blocks = reader_stage.held_blocks() if blocks is None else blocks
        self.held = (*(data.Quarter(q) for q in range(len(paths.QUARTERS))), reader_stage.load_block_area(blocks))
        self._sheets: dict = {}

    def __call__(self, sheet: str, window, shape) -> dict[str, bool]:
        sd = self._sheets.get(sheet)
        if sd is None:
            world = data.placement_samples(data.load_transform(sheet), tuple(shape))
            sd = self._sheets[sheet] = SimpleNamespace(series=paths.SERIES, sheet=sheet, world=world)
        x0, y0, x1, y1 = (int(v) for v in window)
        if x1 <= x0 or y1 <= y0:
            return {k: False for k in TOUCH}
        return dict(zip(TOUCH, K._areas(sd, self.held, x0, y0, x1, y1)))


# --- crops of our own boxes -------------------------------------------------------------------------------------------

def own_box_crops(boxes: pd.DataFrame, terra: pd.DataFrame, *, quarter: int, iou: float = IOU, pages=None,
                  touch=None, mask: bool = False) -> list[tuple[np.ndarray, str, dict]]:
    """Our boxes matched to a Terra word at `iou`, each cut as `crops.box_crop` cuts it (turned a half turn when
    Terra's word reads the other way), with Terra's string: (crop, string, meta), by sheet and box, for the reader
    holding `quarter` out.  Left out: a box whose crop's footprint or whose word touches an answered block or the
    quarter.  `pages`: sheet -> page (read from disk when None); `touch`: `HeldAreas` (made when None); with `mask`,
    each meta's `mask` is the ink touching the box's rectangle (`read.kept_ink`) cut as the crop (turned with it), as
    `read.box_crops` masks it.  meta: `sheet`, `row` (the box's index label), `word` (Terra's word on the sheet),
    `iou`, `turned`, `kind`, `quarter` (the box's) and the TOUCH flags (crop or word).  ValueError without a quarter."""
    quarter = _need_quarter(quarter, "own_box_crops")
    touch = HeldAreas() if touch is None else touch
    page_of = _pages(pages)
    m = matches(boxes, terra, iou)
    sheet_of = boxes["sheet"].astype(str).to_numpy()
    out = []
    for sheet in sorted(set(sheet_of[m["box"]])):
        mine = m[sheet_of[m["box"]] == sheet].sort_values("box")
        page = page_of(sheet)
        for p, w, v in mine.itertuples(index=False):
            r = {**boxes.iloc[int(p)].to_dict(), "row": boxes.index[int(p)]}
            t = terra.iloc[w]
            corners = _corners(r)
            crop, a, shape = K.box_cut(page, corners, r["t_length"], r["t_height"], r["axis"])
            window = K._footprint(a, shape, page.shape)
            flags = touch(sheet, window, page.shape)
            flags = {k: bool(flags[k]) or bool(t[k]) for k in TOUCH}
            if _held(flags, quarter):
                continue
            turned = _turned(t["angle"], r["axis"])
            if turned:
                crop = np.ascontiguousarray(np.rot90(crop, 2))
            meta = {"sheet": sheet, "row": r["row"], "word": int(t["word"]), "iou": float(v), "turned": turned,
                    "kind": t["kind"], "quarter": int(r["quarter"]), **flags}
            if mask:
                x0, y0, x1, y1 = window
                keep = read.kept_ink(page[y0:y1, x0:x1], (x0, y0), corners - 0.5)
                mk = K.box_mask(keep, corners, r["t_length"], r["t_height"], r["axis"], (x0, y0))
                meta["mask"] = np.ascontiguousarray(np.rot90(mk, 2)) if turned else mk
            out.append((crop, str(t["text"]), meta))
    return out


def own_draw_weights(own: list, shares: dict) -> np.ndarray:
    """The chance each own-box crop (`own_box_crops`' triples) is drawn by so that their kinds come in `shares` (kind
    -> share, `offset_distribution`'s `kind_shares`): each kind its share, split evenly among its crops; the kinds
    with no crop give theirs to the others in proportion."""
    kinds = [str((meta or {}).get("kind")) for _, _, meta in own]
    count = collections.Counter(kinds)
    total = sum(float(shares.get(k, 0.0)) for k in count)
    if not own or total <= 0:
        raise ValueError(f"no own-box crop of a kind with a share ({dict(count)}; {shares})")
    p = np.array([float(shares.get(k, 0.0)) / total / count[k] for k in kinds])
    return p / p.sum()


def extra_pool(own: list, sloppy: list = (), kind_weights: str | None = None, kind_shares: dict | None = None,
               real_masks: dict | None = None) -> dict:
    """`textread.train.fit`'s `extra`: the own-box and sloppy crops; with `kind_weights="terra"` the own-box crops
    drawn in `kind_shares` (`own_draw_weights`; off by default: drawn alike); `real_masks` (`real_masks`) for a
    training with the mask."""
    out = {"own": list(own), "sloppy": list(sloppy)}
    if kind_weights == "terra":
        if kind_shares is None:
            raise ValueError("kind_weights='terra' needs the measured kind_shares")
        out["own_p"] = own_draw_weights(out["own"], kind_shares)
    elif kind_weights is not None:
        raise ValueError(f"kind_weights: None or 'terra', not {kind_weights!r}")
    if real_masks is not None:
        out["real_masks"] = real_masks
    return out


def real_masks(rows: pd.DataFrame, pages=None) -> dict:
    """The masks of `solver-text-crops`' crops (rows of crops.parquet, every map; words and non-text windows): the ink
    touching each crop's rectangle (`read.kept_ink`, within the px the crop reads), cut as the crop was
    (`crops.rectify`), keyed by the row's position; a crop with no ink in its rectangle has none.  `pages`: sheet ->
    page (each read from disk, by its series, when None)."""
    last: dict = {}

    def page_of(series, sheet):
        if pages is not None:
            return pages[sheet]
        if sheet not in last:
            last.clear()
            last[sheet] = data.load_page(sheet, series)
        return last[sheet]

    box = rows[["box_x0", "box_y0", "box_x1", "box_y1"]].to_numpy(float)
    length, height = rows["length"].to_numpy(float), rows["height"].to_numpy(float)
    angle = rows["angle"].to_numpy(float)
    out = {}
    for (series, sheet), idx in sorted(rows.groupby(["series", "sheet"], sort=True).indices.items()):
        page = page_of(series, sheet)
        for p in idx:
            centre = ((box[p, 0] + box[p, 2] - 1) / 2, (box[p, 1] + box[p, 3] - 1) / 2)
            a, shape = K.frame_of(centre, length[p], height[p], angle[p])
            x0, y0, x1, y1 = K._footprint(a, shape, page.shape)
            if x1 <= x0 or y1 <= y0:
                continue
            keep = read.kept_ink(page[y0:y1, x0:x1], (x0, y0), _rect(*centre, length[p], height[p], angle[p]))
            if keep.any():
                out[int(p)] = K.cut(keep, centre, length[p], height[p], angle[p], fill=False, origin=(x0, y0))[0]
    return out


# --- line pieces ------------------------------------------------------------------------------------------------------

def _place(dst: np.ndarray, src: np.ndarray, top: int, left: int, op) -> None:
    """dst = op(dst, src) where `src` lies on `dst` at (top, left), cut to `dst`."""
    h, w = dst.shape
    py0, px0 = max(0, -top), max(0, -left)
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(h, top + src.shape[0]), min(w, left + src.shape[1])
    if y1 > y0 and x1 > x0:
        dst[y0:y1, x0:x1] = op(dst[y0:y1, x0:x1], src[py0:py0 + y1 - y0, px0:px0 + x1 - x0])


def _inside(shape, origin, cx, cy, length, height, angle) -> np.ndarray:
    """Which px of a window (its px (0, 0) the page's px `origin`) have their centre in a word rectangle."""
    (ux, uy), (vx, vy) = L._axes(angle)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dx, dy = xx + origin[0] - cx, yy + origin[1] - cy
    return (np.abs(dx * ux + dy * uy) <= length / 2) & (np.abs(dx * vx + dy * vy) <= height / 2)


def thickness(mask: np.ndarray) -> float:
    """The thickness of ink (bool): twice the largest distance (L2) from one of its px to paper."""
    m = np.pad(np.asarray(mask, bool), 1).view(np.uint8)
    return 2.0 * float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max()) if m.any() else 0.0


@dataclasses.dataclass
class Lines:
    """Line pieces ready for `sloppy_crop`: the patches (`piece_patches`), each one's thickness (`thickness`), the
    chance each is drawn by (`p`), and per bin of cover (`cover_edges`) the chance a draw of that cover is kept
    (`accept`; None: every draw).  `calibrate_lines` sets them; `uniform` draws every piece alike and keeps all."""
    patches: list
    thick: np.ndarray
    p: np.ndarray
    cover_edges: np.ndarray | None = None
    accept: np.ndarray | None = None
    report: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def uniform(cls, patches) -> "Lines":
        patches = list(patches)
        n = len(patches)
        return cls(patches, np.array([thickness(p < 255) for p in patches], float),
                   np.full(n, 1.0 / n) if n else np.zeros(0))

    def __post_init__(self):
        if self.accept is not None and not (len(self.accept) and float(np.max(self.accept)) == 1.0):
            raise ValueError("Lines.accept must keep some cover always (its largest chance 1): a draw is drawn again "
                             "until one is kept")

    def __len__(self) -> int:
        return len(self.patches)

    def draw(self, rng) -> int:
        return int(rng.choice(len(self.p), p=self.p))

    def keep(self, cover: float) -> float:
        if self.accept is None:
            return 1.0
        k = int(np.clip(np.searchsorted(self.cover_edges, cover, side="right") - 1, 0, len(self.accept) - 1))
        return float(self.accept[k])


def _letters(win, origin, t):
    """Terra's word's rectangle in the window, and the px a line piece may be anchored on: its ink px (none when the
    rectangle holds no ink: there are no letters to cross, and the word gets no line)."""
    inside = _inside(win.shape, origin, t["cx"], t["cy"], t["length"], t["height"], t["angle"])
    ys, xs = np.nonzero(inside & (win < grid.INK_BELOW))
    return inside, ys, xs


def _propose(win, inside, ys, xs, lines: Lines, rng):
    """One line draw: a letter px, a piece (by `lines.p`), its quarter turns and a px of its ink on the letter px;
    the window with it ORed in, and its cover: the piece's px inside the word's rectangle over the ink there."""
    k = int(rng.integers(len(ys)))
    j, turns = lines.draw(rng), int(rng.integers(4))
    patch = np.rot90(lines.patches[j], turns)
    py, px = np.nonzero(patch < 255)
    m = int(rng.integers(len(py)))
    top, left = int(ys[k] - py[m]), int(xs[k] - px[m])
    trial = win.copy()
    _place(trial, patch, top, left, np.minimum)
    on = np.zeros(win.shape, bool)
    _place(on, patch < 255, top, left, np.logical_or)
    cover = float((on & inside).sum() / max(((trial < grid.INK_BELOW) & inside).sum(), 1))
    return trial, cover, j, turns, k


def _wquantiles(x, w, qs) -> list[float]:
    x, w = np.asarray(x, float), np.asarray(w, float)
    order = np.argsort(x, kind="stable")
    cw = np.cumsum(w[order])
    if not len(x) or cw[-1] <= 0:
        return [None] * len(qs)
    return [float(x[order][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)]) for q in qs]


def calibrate_lines(pieces, dist: dict, words: pd.DataFrame, pages=None, seed: int = 0,
                    proposals: int = PROPOSALS, rounds: int = ROUNDS) -> Lines:
    """`Lines` whose pasted pieces follow the measured crossings (`dist`'s `line_thickness` and `line_cover` values,
    `offset_distribution`): each piece weighted by the measured share of its thickness over the bank's (the
    thicknesses take the distance transform's own values: matched value for value; a piece of a thickness no
    crossing has is never drawn), and each draw kept with the chance that makes the kept covers follow the measured
    histogram (Freedman-Diaconis bins on [0, 1]): the measured share of its bin over the drawn share, the largest
    such ratio kept always.  Keeping draws by cover moves the thicknesses kept, so the piece weights are raked by the
    kept thicknesses' shares, `rounds` times, each round simulating `proposals` draws on `words` (Terra's words the
    crops are made of, on `pages`).  `report`: the measured and the realised (kept) quantiles of both, the mean chance
    a draw is kept, and the measured mass the bank or the draws cannot reach."""
    lines = pieces if isinstance(pieces, Lines) else Lines.uniform(pieces)
    real_t = np.round(np.asarray(dist["line_thickness"]["values"], float), 6)
    real_c = np.asarray(dist["line_cover"]["values"], float)
    values, counts = np.unique(real_t, return_counts=True)
    real_tp = counts / counts.sum()
    piece_v = np.round(lines.thick, 6)
    slot = np.array([int(np.searchsorted(values, v)) if v in set(values) else -1 for v in piece_v])
    bank_tp = np.bincount(slot[slot >= 0], minlength=len(values)) / max(len(slot), 1)
    w = np.where(slot >= 0, real_tp[np.maximum(slot, 0)] / np.maximum(bank_tp[np.maximum(slot, 0)], 1e-12), 0.0)
    if w.sum() <= 0:
        raise ValueError("no line piece has a thickness a measured crossing has")
    edges = np.histogram_bin_edges(real_c, bins="fd", range=(0.0, 1.0))
    real_cp = np.histogram(real_c, edges)[0] / len(real_c)
    page_of = _pages(pages)
    prep = []
    for _, t in words.iterrows():
        win, origin = _window(page_of(t["sheet"]), [_word_rect(t)], 0)
        if win.size:
            inside, ys, xs = _letters(win, origin, t)
            if len(ys):
                prep.append((win, inside, ys, xs))
    if not prep:
        raise ValueError("no word to simulate line draws on")
    rng = np.random.default_rng([seed, 3])
    for r in range(rounds + 1):
        lines.p = w / w.sum()
        drawn_t, drawn_c = np.zeros(proposals, int), np.zeros(proposals)
        for n in range(proposals):
            win, inside, ys, xs = prep[int(rng.integers(len(prep)))]
            _, drawn_c[n], j, _, _ = _propose(win, inside, ys, xs, lines, rng)
            drawn_t[n] = slot[j]
        cb = np.clip(np.searchsorted(edges, drawn_c, side="right") - 1, 0, len(real_cp) - 1)
        prop_cp = np.bincount(cb, minlength=len(real_cp)) / proposals
        a = np.where(prop_cp > 0, real_cp / np.maximum(prop_cp, 1e-12), 0.0)
        a = a / a.max()
        kept = a[cb]
        if r == rounds:
            break
        kept_tp = np.bincount(drawn_t, weights=kept, minlength=len(values))
        kept_tp = kept_tp / kept_tp.sum()
        ratio = np.where(kept_tp > 0, real_tp / np.maximum(kept_tp, 1e-12), 1.0)
        w = np.where(slot >= 0, w * ratio[np.maximum(slot, 0)], 0.0)
    lines.cover_edges, lines.accept = edges, a
    lines.__post_init__()
    qs = (0.1, 0.5, 0.9, 0.99)
    kept_tp = np.bincount(drawn_t, weights=kept, minlength=len(values)) / kept.sum()
    kept_cp = np.bincount(cb, weights=kept, minlength=len(real_cp)) / kept.sum()
    lines.report = {
        "thickness_tv": float(np.abs(kept_tp - real_tp).sum() / 2),
        "cover_tv": float(np.abs(kept_cp - real_cp).sum() / 2),
        "thickness_shares": {str(v): [float(m), float(k)] for v, m, k in zip(values, real_tp, kept_tp)},
        "thickness": {"measured": np.quantile(real_t, qs).tolist(), "realised": _wquantiles(values[drawn_t], kept, qs)},
        "cover": {"measured": np.quantile(real_c, qs).tolist(), "realised": _wquantiles(drawn_c, kept, qs)},
        "quantiles": list(qs), "mean_keep": float(kept.mean()), "cover_bins": int(len(real_cp)),
        "measured_cover_unreached": float(real_cp[prop_cp == 0].sum()),
        "measured_thickness_unreached": float(real_tp[np.bincount(slot[slot >= 0], minlength=len(values)) == 0].sum()),
        "pieces_drawable": int((lines.p > 0).sum()), "pieces": len(lines), "words": len(prep),
        "proposals": proposals, "rounds": rounds}
    return lines


def _line_through(win, origin, t, lines: Lines, rng) -> dict:
    """OR a line piece through a letter px of Terra's word `t` (`_propose`), each draw kept with its cover's chance
    (`Lines.keep`) and drawn again otherwise, until one is kept."""
    inside, ys, xs = _letters(win, origin, t)
    if not len(ys):
        return {}
    rejected = 0
    while True:
        trial, cover, j, turns, k = _propose(win, inside, ys, xs, lines, rng)
        if lines.accept is None or rng.random() < lines.keep(cover):
            break
        rejected += 1
    win[:] = trial
    return {"line": True, "piece": j, "turns": turns, "line_at": (int(xs[k] + origin[0]), int(ys[k] + origin[1])),
            "line_cover": cover, "line_thick": float(lines.thick[j]), "line_rejected": rejected}


# --- the neighbour ----------------------------------------------------------------------------------------------------

def _paste_neighbour(win, origin, t, pose: dict, rect) -> dict:
    """OR a real pair's neighbour (`neighbour_pose`) into the window at its place, angle and scale in the frame of
    Terra's word `t` (its height over the pair's word's; past the same end), as printed (nearest px); its share of its
    rectangle inside the crop's rectangle `rect` (centre, length, height, angle) and its ink against the word's
    (`ink_overlaps`) as they came out."""
    cx, cy, length, height, angle = (float(t[k]) for k in ("cx", "cy", "length", "height", "angle"))
    scale = height / float(pose["h"])
    end = int(pose["side"]) * length / 2
    (ux, uy), (vx, vy) = L._axes(angle)
    cells = np.asarray(pose["cells"], int).reshape(-1, 2)
    patch = np.zeros((cells[:, 1].max() + 1, cells[:, 0].max() + 1), np.uint8)
    patch[cells[:, 1], cells[:, 0]] = 255
    ox, oy = origin
    p0, q0 = float(pose["p0"]), float(pose["q0"])
    to_patch = np.array([[ux / scale, uy / scale, ((ox - cx) * ux + (oy - cy) * uy - end) / scale - p0],
                         [vx / scale, vy / scale, ((ox - cx) * vx + (oy - cy) * vy) / scale - q0]])
    ink = cv2.warpAffine(patch, to_patch, (win.shape[1], win.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    word = _word_rect(t)
    before = (win < grid.INK_BELOW) & read.in_rect(win.shape, origin, word)
    win[ink] = 0
    pr = np.asarray(pose["rect"], float) * scale
    nb = np.stack([cx + (pr[:, 0] + end) * ux + pr[:, 1] * vx, cy + (pr[:, 0] + end) * uy + pr[:, 1] * vy], 1)
    centre, clen, cheight, cangle = rect
    got_part = _meet(_rect(centre[0], centre[1], clen, cheight, cangle), nb) / max(_area(nb), 1e-9)
    got_touch, got_on = ink_overlaps(win, origin, word, nb, angle, before)
    return {"neighbour": True, "part_got": got_part, "ink_got": got_touch, "on_got": got_on,
            "side": int(pose["side"]), "scale": scale, "rect": nb.tolist()}


# --- sloppy crops of Terra's words ------------------------------------------------------------------------------------

def sloppy_crop(page: np.ndarray, terra_word, dist: dict, pieces, rng, info: dict | None = None,
                with_mask: bool = False) -> np.ndarray:
    """A sloppy crop of Terra's word `terra_word` (a row of `terra_frame`) on `page` (see the module): one pair of
    `dist` (`offset_distribution`): its offsets, and its neighbour when it held one (`neighbour_pose`); a line piece of
    `pieces` (`Lines`, or patches drawn alike and all kept) with probability `dist["line_share"]`; each from `rng`
    alone.  With `with_mask`, `info["mask"]` is the
    ink touching the drawn rectangle (`read.kept_ink`, the pasted ink too) cut as the crop.  `info` (a dict) gets what
    was drawn: the offsets, `draw`, `frame` (centre, length, height, angle: `crops.cut`'s), `window` (the page px the
    crop reads), `line` (and `piece`, `turns`, `line_at`, `line_cover`, `line_thick`, `line_rejected`), `neighbour`
    (and the pair's measured `part`, `ink` (touching) and `on`, and as pasted `part_got`, `ink_got`, `on_got`;
    `side`, `scale`, `rect`)."""
    t = terra_word
    lines = pieces if isinstance(pieces, Lines) else Lines.uniform(pieces)
    cx, cy, length0, height0, angle0 = (float(t[k]) for k in ("cx", "cy", "length", "height", "angle"))
    off = dist["offsets"]
    k = int(rng.integers(len(off["start"])))
    s, e, x, da, r = (float(off[n][k]) for n in OFFSETS)
    part = float(off["neighbour"][k]) if "neighbour" in off else 0.0
    def drawn(key):
        v = off[key][k] if key in off else None
        return None if v is None or not np.isfinite(v) else float(v)

    touch, on = drawn("neighbour_ink"), drawn("neighbour_on")
    angle = angle0 + da
    (ux, uy), (vx, vy) = L._axes(angle)
    length, height = max(length0 + (e - s) * height0, 1.0), max(r * height0, 1.0)
    along, across = (s + e) * height0 / 2, x * height0
    centre = (cx + ux * along + vx * across, cy + uy * along + vy * across)
    a, shape = K.frame_of(centre, length, height, angle)
    x0, y0, x1, y1 = K._footprint(a, shape, page.shape)
    got = {"draw": k, **dict(zip(OFFSETS, (s, e, x, da, r))), "frame": (centre, length, height, angle),
           "window": (x0, y0, x1, y1), "line": False, "line_rejected": 0, "neighbour": False}
    u_line = rng.random()
    if x1 <= x0 or y1 <= y0:                     # off the page: paper
        crop = K.cut(page, centre, length, height, angle)[0]
    else:
        win = page[y0:y1, x0:x1].copy()
        if u_line < float(dist.get("line_share") or 0.0) and len(lines):
            got.update(_line_through(win, (x0, y0), t, lines, rng))
        pose = off["neighbour_pose"][k] if "neighbour_pose" in off else None
        if pose is not None:
            got.update(part=part, ink=touch, on=on)
            got.update(_paste_neighbour(win, (x0, y0), t, pose, (centre, length, height, angle)))
        crop = K.cut(win, centre, length, height, angle, origin=(x0, y0))[0]
        if with_mask:
            keep = read.kept_ink(win, (x0, y0), _rect(centre[0], centre[1], length, height, angle))
            got["mask"] = K.cut(keep, centre, length, height, angle, fill=False, origin=(x0, y0))[0]
    if info is not None:
        info.update(got)
    return crop


def sloppy_crops(terra: pd.DataFrame, dist: dict, pieces, seed: int, *, quarter: int, copies: int = 1, pages=None,
                 touch=None, mask: bool = False) -> list[tuple[np.ndarray, str, dict]]:
    """`copies` sloppy crops of every word of `terra` touching neither the answered blocks nor `quarter`, each from its
    own generator (`default_rng([seed, sheet, word, copy])`); a crop whose footprint touches either is left out (the
    neighbours `dist` holds are training words').  (crop, string, meta: `sheet`, `word`, `copy`, `kind`, what
    was drawn, the TOUCH flags, and with `mask` its `mask`, `sloppy_crop`'s), by sheet and word.  `pieces` (`Lines`
    or patches) must hold no held-out ink (`piece_patches` checks its bank).  ValueError without a quarter."""
    quarter = _need_quarter(quarter, "sloppy_crops")
    touch = HeldAreas() if touch is None else touch
    page_of = _pages(pages)
    lines = pieces if isinstance(pieces, Lines) else Lines.uniform(pieces)
    t = terra[_training_words(terra, quarter)]
    out = []
    for sheet in sorted(set(t["sheet"])):
        page = page_of(sheet)
        for _, row in t[t["sheet"] == sheet].iterrows():
            for c in range(copies):
                rng = np.random.default_rng([seed, K.hash_name(sheet), int(row["word"]), c])
                info: dict = {}
                crop = sloppy_crop(page, row, dist, lines, rng, info, with_mask=mask)
                flags = touch(sheet, info["window"], page.shape)
                if _held(flags, quarter):
                    continue
                out.append((crop, str(row["text"]), {"sheet": sheet, "word": int(row["word"]), "copy": c,
                                                     "kind": row["kind"], **info,
                                                     **{k: bool(v) for k, v in flags.items()}}))
    return out


def _same_area(a, b) -> bool:
    return (isinstance(a, data.GridArea) and isinstance(b, data.GridArea) and a.col0 == b.col0 and a.row0 == b.row0
            and np.shape(a.mask) == np.shape(b.mask) and np.array_equal(a.mask, b.mask))


def piece_patches(bank, *, quarter: int, pages=None, piece_mask=None, block_area=None,
                  info: dict | None = None) -> list[np.ndarray]:
    """The 2000 sheets' line pieces of a `composite.Bank` as patches: each piece's box of its page, the piece's px as
    printed, paper elsewhere (`calibrate_lines` weights them).  ValueError unless every sheet of the bank holds out
    `quarter` and the answered blocks (`block_area`, read when None).  `piece_mask`: `composite.piece_mask` (for
    tests); `info` gets the count."""
    quarter = _need_quarter(quarter, "piece_patches")
    block_area = reader_stage.load_block_area(reader_stage.held_blocks()) if block_area is None else block_area
    for sd in bank.sheets:
        held = tuple(getattr(sd, "held", ()))
        if data.Quarter(quarter) not in held or not any(_same_area(a, block_area) for a in held):
            raise ValueError(f"the bank's sheet {sd.sheet} has not held out quarter {paths.QUARTERS[quarter]} and the "
                             "answered blocks: its line pieces could come from them")
    piece_mask = composite.piece_mask if piece_mask is None else piece_mask
    page_of = _pages(pages)
    p = bank.pieces[paths.SERIES]
    out = []
    for j in range(len(p)):
        sd = bank.sheets[int(p.sheet[j])]
        x0, y0, x1, y1 = (int(v) for v in p.box[j])
        mask = piece_mask(sd, p.box[j], p.seed[j])
        out.append(np.where(mask, page_of(sd.sheet)[y0:y1, x0:x1], 255).astype(np.uint8))
    if info is not None:
        info.update(pieces=len(out))
    return out


# --- the measured offsets ---------------------------------------------------------------------------------------------

def line_measures(terra: pd.DataFrame) -> pd.DataFrame:
    """Per word of `terra` (its rectangle as `data.SheetData.words`), on the sheet's line ink (the line px of
    `zones.sheet_lines`, kept in `zones.LINES_DIR`, that are ink; 8-connected): `line_ink` the line ink inside the
    rectangle (`composite.crossings`' second count), `cross_ink` that of it in line ink components running on outside
    the rectangle grown by zones.GROW_PX (the line px's own growth), `word_ink` the ink inside the rectangle, and
    `line_thick` the thickness of the crossing ink (twice the largest distance to paper of one of its px, on the
    sheet's whole line ink; 0 without)."""
    cols = {c: np.zeros(len(terra)) for c in LINE_COLUMNS}
    recs = textboxes.terra_records()
    g = zones.GROW_PX
    grow = np.ones((2 * g + 1, 2 * g + 1), np.uint8)
    for sheet in sorted(set(terra["sheet"])):
        sd = data.sheet_data(paths.SERIES, sheet, hold_out_validation=False, terra=recs, placement=True)
        sd = dataclasses.replace(sd, line_bits=zones.sheet_lines(sd, zones.LINES_DIR))
        line_ink = composite._line_page(sd) & np.unpackbits(sd.ink_bits, axis=1, count=sd.width).view(bool)
        dist = cv2.distanceTransform(np.pad(line_ink, 1).view(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
        _, lab, stats, _ = cv2.connectedComponentsWithStats(line_ink.view(np.uint8), connectivity=8)
        for k in np.flatnonzero(terra["sheet"].to_numpy() == sheet):
            j = int(terra["word"].iat[k])
            if j >= len(sd.words):
                raise ValueError(f"{sheet}: Terra's words of crops.parquet are not the sheet's ({len(sd.words)})")
            rect = sd.words[j]
            x0, y0 = max(int(rect[:, 0].min()) - g - 1, 0), max(int(rect[:, 1].min()) - g - 1, 0)
            x1, y1 = min(int(rect[:, 0].max()) + g + 2, sd.width), min(int(rect[:, 1].max()) + g + 2, sd.height)
            if x1 <= x0 or y1 <= y0:
                continue
            inside = composite._rect_in(rect, y0, x0, y1 - y0, x1 - x0)
            ink = inside & data._ink_window(sd, y0, x0, y1 - y0, x1 - x0)
            lw = lab[y0:y1, x0:x1]
            li = ink & (lw > 0)
            grown = cv2.dilate(inside.view(np.uint8), grow) > 0
            beyond = set(np.unique(lw[~grown & (lw > 0)]).tolist())
            crossing = [int(v) for v in np.unique(lw[li])
                        if v in beyond or stats[v, 0] < x0 or stats[v, 1] < y0
                        or stats[v, 0] + stats[v, 2] > x1 or stats[v, 1] + stats[v, 3] > y1]
            cross = li & np.isin(lw, crossing)
            cols["line_ink"][k], cols["cross_ink"][k], cols["word_ink"][k] = li.sum(), cross.sum(), ink.sum()
            cols["line_thick"][k] = 2.0 * float(dist[y0:y1, x0:x1][cross].max()) if cross.any() else 0.0
    return pd.DataFrame(cols, index=terra.index)


def _rounded(d, places: int = 6):
    """Floats rounded to `places` throughout (lists and dicts walked; None and other values kept)."""
    if isinstance(d, dict):
        return {k: _rounded(v, places) for k, v in d.items()}
    if isinstance(d, list):
        return [_rounded(v, places) for v in d]
    return round(d, places) if isinstance(d, float) and math.isfinite(d) else d


def build_offsets(out: Path = OFFSETS_FILE) -> dict:
    """Measure the offsets of today's boxes (`solver-boxes`) against Terra's words (`solver-text-crops`) for each
    held-out quarter and write them to `out`: {"quarters": {name: offset_distribution}, ...}."""
    t0 = time.perf_counter()
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    terra = terra_frame(pd.read_parquet(store.CROPS_DIR / "crops.parquet"))
    terra[list(LINE_COLUMNS)] = line_measures(terra)
    pages = DiskPages()
    got = {name: _rounded(offset_distribution(boxes, terra, quarter=q, pages=pages))
           for q, name in enumerate(paths.QUARTERS)}
    result = {"what": "our boxes (solver-boxes) matched to Terra's 2000 words at IoU >= 0.5, per held-out quarter "
                      "(pairs whose box is outside it and whose word touches neither it nor an answered block); "
                      "offsets in Terra's word heights (see wordread.readercrops)",
              "iou": IOU, "boxes": int(len(boxes)), "terra_words": int(len(terra)), "quarters": got}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    log.info("offsets of %s pairs written to %s in %.0f s", {q: d["n"] for q, d in got.items()}, out,
             time.perf_counter() - t0)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    build_offsets()
