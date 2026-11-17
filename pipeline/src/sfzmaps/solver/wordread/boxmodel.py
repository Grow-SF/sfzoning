"""The learned box model: a small network that reads one of our word boxes' own px and gives the number of words it
holds (a distribution over 0..K) and, per crop column, the chance that the column lies between two of its words.  The
owner: "there are things you can learn about boxes and how many words they hold ... Don't just hardcode these ...
There is a hidden function that you should be able to learn automatically."  So no hand features (ink density, the ink
centroid against the box centre, gaps): the network learns what matters from the px.  Its log-probabilities feed Task
9's box choice (`box_logp`); Task 8's arm (c) trains it (`fit_boxmodel`).  No stage imports this module.

The examples (`box_examples`), for the model that holds one quarter of the city out:
  the crop      each box cut as the reader cuts it (`textread.crops.box_cut`: its trimmed rectangle along its axis, the
                reader crops' own margin PAD_X / PAD_Y, HEIGHT px tall), one channel per layer of the sheet
                (`SheetInputs`): the ink (the page darker than `grid.INK_BELOW`, 0 or 1, area-averaged by the resize),
                the detector's per-ink text chance (`text-reader/<sheet>/chance.png`, 0..1), and when given the
                detector's word-head outputs: the core chance (0..1) and the k tag channels (as they come)
  the count     each Terra word's share of its ink inside the box's trimmed rectangle (its ink: the ink px whose
                centre is inside its rectangle, `textnet.boxes.word_rect`; a word with no ink px: the share of its
                rectangle's area), each word held or not independently with that chance, so the count is the
                Poisson-binomial of the shares (`poisson_binomial`): exact, no "mostly inside" cut
  the boundary  per crop column, the chance that the column lies between two consecutive held words: at a point x
                along the box's axis, P(some held word ends at or before x) x P(some held word starts at or after x) x
                P(no held word covers x), each word's span its rectangle's extent along the axis; integrated exactly
                over the column's span (`boundary_target`).  With every share 1 these are exactly the columns between
                consecutive words' ends, a column half in a gap counting a half
  the weight    with Task 7's per-word weights w (the chance Terra's box is right; None: 1), an example counts
                prod_i (1 - s_i (1 - w_i)) over the words it may hold (s_i its share): a word charges its doubt by the
                chance the box holds it
  joined        two real Terra word crops (the reader's own crops of the words, `crops.cut` at the words' geometry),
  neighbours    drawn at random from the training words, laid on one line at a gap drawn from the measured
                distribution of real same-line gaps (`draw_gaps`), in word heights: the first word's crop up to its
                end, paper across the gap, the second word's crop from its start (each word's outer margin real; a
                negative gap overlaps them, each px from the crop with more ink there).  Count 2 and the gap's columns,
                known by construction; turned a half turn with the measured share of real boxes' words that read
                against the box's axis (`turned`), so the joined crops read both ways as the real boxes do.  How many
                (`joined`): by default the expected number of real boxes holding two words or more (the sum of their
                P(count >= 2)): measured, and an argument Task 8 may screen
  the gaps      (`same_line_gaps`) the real same-line gaps, on the training words: from each word's end to the start
                of the next word on its line of text, along its reading direction (as the joined crops lay them), over
                their mean height.  No line rule and no distance cut: a word ahead shares the line with the chance
                `line_share` (the overlap of their letters' bands over the smaller, times |cos| of the angle between
                their readings: stacked words, a code over its block number, and words read across it count nothing),
                and it is the next word with that chance times the chance that no word ahead nearer shares the line.
                On a map the next word on a line is mostly another label far along the street (measured 11-17: median
                8-10 word heights, 95th percentile 75-85; a joined crop that long is squeezed to `crops.MAX_WIDTH` as a
                long box is); no mixture fitted to these gaps separates the words of one phrase from them.  For
                comparison (`held_pair_gaps`, reported by the run script, not drawn from): the same gaps between words
                that one of our boxes holds together are mostly negative, since a merged box is mostly a 2-D cluster (a
                code with its block number tucked beside or below it), not words side by side on a line
  held out     only training-quarter words outside the answered blocks: `box_examples` raises on a word whose crop
                touches the held quarter or an answered block (`crops.TOUCH`, as `solver-text-crops` records them), on
                a box in the held quarter and on a box whose crop touches either (`touch`).  `training_boxes` and
                `training_words` give what may go in (a box meeting a held word's rectangle is left out: its count
                would miss that word)

K, the largest count of the network's output, is measured: the largest count with any chance in the training
examples (`Examples.K`).

The network (`BoxNet`), in the style of the reader's backbone (`textread.model`, MobileNetV3 blocks): a smaller
MobileNetV3 stack whose strides take a HEIGHT-px crop to one row and its width to a quarter (STRIDE px a frame), then
dilated 1-D convolutions over the frames (each frame sees its neighbours), masked to the crop's own frames so a crop's
output never depends on the padding of its batch beyond the 2-D stack; a boundary logit per frame (each of its STRIDE
columns), and count logits over 0..K from the frames' masked mean, max and sum.  `fit_boxmodel` trains it by the
recipe (`RECIPE`; Task 8 screens it), in an exclusive GPU turn, seeded.  `box_logp` reads crops one padded width at a
time, so a crop's output is its own.
"""
import contextlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from sfzmaps import gpu
from sfzmaps.solver import grid, paths, runs, wordboxes
from sfzmaps.solver.codereader import labels as L
from sfzmaps.solver.textnet import boxes as textboxes
from sfzmaps.solver.textnet import data
from sfzmaps.solver.textread import crops as K
from sfzmaps.solver.textread import model, store

HEIGHT = K.HEIGHT
STRIDE = model.STRIDE            # crop px per frame
TOUCH = store.TOUCH
C_COLS = [f"c{i}{a}" for i in range(4) for a in "xy"]     # a Terra word's rectangle (`measures.C_COLS`)
T_COLS = [f"t{i}{a}" for i in range(4) for a in "xy"]     # a box's trimmed rectangle
SEED = 20261117
OUT_DIR = runs.EXPERIMENTS_DIR / "wordread" / "boxmodel"
GAPS_FILE = OUT_DIR / "gaps.json"
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)      # reported only
# MobileNetV3 blocks (`textread.model.InvertedResidual`): input channels, kernel, expanded, output, squeeze-excite,
# hard swish, stride.  With the stem's (2, 1) a HEIGHT-px crop comes out one row tall and a quarter as wide.
STEM = 16
LAYERS = ((16, 3, 16, 16, True, False, (2, 1)), (16, 3, 64, 24, False, False, (2, 1)),
          (24, 3, 72, 24, False, False, 1), (24, 5, 96, 40, True, True, (2, 2)),
          (40, 5, 120, 40, True, True, 1), (40, 5, 144, 48, True, True, (2, 2)))
LAST = 96                        # the frames' channels
DILATIONS = (1, 2, 4, 8)         # the context convolutions over the frames
SUM_SCALE = 16.0                 # numerical: the frames' summed features are divided by this (decides nothing)
BN_MOMENTUM = 0.1                # torch's default: the reader's 0.01 (docTR's) leaves a short fit's running statistics
#   far from its batches', and the network reads crops by them
# The training recipe's defaults (Task 8 screens them on the held-out outcome); `boundary_weight` weighs the boundary
# loss against the count's.
RECIPE = {"steps": 6000, "batch": 32, "max_lr": 2e-3, "pct_start": 0.15, "weight_decay": 1e-4, "clip": 5.0,
          "boundary_weight": 1.0, "seed": 0, "device": "cpu", "log_every": 250}


def _need_quarter(quarter, what: str) -> int:
    if quarter is None:
        raise ValueError(f"{what} needs the held-out quarter: with none it would use that quarter's words")
    return int(quarter)


# --- a sheet's layers -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SheetInputs:
    """One sheet's px the box model reads: `ink` bool (h, w); `chance` uint8 (h, w), the detector's per-ink text
    chance x 255; optional word heads: `core` float (h, w) in [0, 1] and `tag` float (k, h, w)."""
    ink: np.ndarray
    chance: np.ndarray
    core: np.ndarray | None = None
    tag: np.ndarray | None = None

    def __post_init__(self):
        shape = np.shape(self.ink)
        if np.shape(self.chance) != shape or (self.core is not None and np.shape(self.core) != shape) or (
                self.tag is not None and np.shape(self.tag)[1:] != shape):
            raise ValueError("every layer of a sheet must be the page's shape")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(np.shape(self.ink))

    @property
    def channels(self) -> tuple[str, ...]:
        out = ["ink", "chance"]
        if self.core is not None:
            out.append("core")
        if self.tag is not None:
            out += [f"tag{k}" for k in range(len(self.tag))]
        return tuple(out)

    def layers(self) -> list[tuple[np.ndarray, float]]:
        """Each channel's page array and the factor its px are multiplied by (float32 crops)."""
        out = [(self.ink, 1.0), (self.chance, 1.0 / 255.0)]
        if self.core is not None:
            out.append((self.core, 1.0))
        if self.tag is not None:
            out += [(t, 1.0) for t in self.tag]
        return out


def load_sheet(sheet: str, chance_dir: Path | None = None, heads=None) -> SheetInputs:
    """A 2000 sheet's `SheetInputs`: its ink, the fold detectors' chance (`<chance_dir>/<sheet>/chance.png`,
    `paths.READER_TEXT_DIR` when None), and `heads(sheet) -> (core, tag)` when given."""
    ink = data.load_page(sheet) < grid.INK_BELOW
    f = (chance_dir or paths.READER_TEXT_DIR) / sheet / "chance.png"
    chance = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if chance is None or chance.shape != ink.shape:
        raise FileNotFoundError(f"no chance of text the page's size at {f}")
    core, tag = heads(sheet) if heads is not None else (None, None)
    return SheetInputs(ink=ink, chance=chance, core=core, tag=tag)


def _getter(sheets):
    return sheets.__getitem__ if hasattr(sheets, "__getitem__") else sheets


def _cut(inputs: SheetInputs, centre, length: float, height: float, angle: float) -> np.ndarray:
    """Every layer of the crop `crops.cut` makes of the rectangle (centre a page px index), float32 (C, HEIGHT, W):
    each layer's window under the crop made float first, so the resize averages px exactly."""
    a, shape = K.frame_of(centre, length, height, angle)
    x0, y0, x1, y1 = L._footprint(a, shape, 2, inputs.shape)
    out = []
    for arr, scale in inputs.layers():
        win = np.asarray(arr)[y0:y1, x0:x1].astype(np.float32) * np.float32(scale)
        out.append(K.cut(win, centre, length, height, angle, 0, origin=(x0, y0))[0].astype(np.float32))
    return np.stack(out)


def layout(length: float, height: float) -> tuple[int, int]:
    """`crops.cut`'s page-px width w of the crop of a `length` x `height` rectangle, and its crop's width.  A point
    p page px along the axis from the rectangle's centre is at w / 2 + p in the page-px crop, and crop column c holds
    [c, c + 1] x w / width of that."""
    w = max(4, round(length + 2 * K.PAD_X * height))
    h = max(4, round(height * (1 + 2 * K.PAD_Y)))
    return w, min(K.MAX_WIDTH, max(4, round(w * HEIGHT / h)))


def _box_geometry(b) -> tuple[np.ndarray, float, float, float]:
    """A box's trimmed corners (4, 2), its sides as `crops.box_cut` takes them, and its axis."""
    corners = np.array([[b[f"t{i}x"], b[f"t{i}y"]] for i in range(4)], float)
    return corners, max(float(b["t_length"]), 1.0), max(float(b["t_height"]), 1.0), float(b["axis"])


def box_crop(inputs: SheetInputs, corners, length: float, height: float, axis: float) -> np.ndarray:
    """A rectangle's crop as the box model reads it (`crops.box_crop`'s cut of every layer): `corners` (4, 2) page px
    edge coordinates, `length` and `height` its sides along and across `axis` (degrees).  Task 9's candidates are cut
    by it."""
    centre = np.asarray(corners, float).reshape(4, 2).mean(0) - 0.5
    return _cut(inputs, (float(centre[0]), float(centre[1])), max(float(length), 1.0), max(float(height), 1.0),
                float(axis))


def _footprint(corners, length: float, height: float, axis: float, shape) -> tuple[int, int, int, int]:
    centre = np.asarray(corners, float).reshape(4, 2).mean(0) - 0.5
    a, crop_shape = K.frame_of((float(centre[0]), float(centre[1])), length, height, axis)
    return K._footprint(a, crop_shape, shape)


# --- the targets ------------------------------------------------------------------------------------------------------

def poisson_binomial(p) -> np.ndarray:
    """The distribution of the number of successes of independent trials with chances `p`: (len(p) + 1,)."""
    out = np.ones(1)
    for s in np.asarray(p, float).ravel():
        out = np.concatenate([out * (1.0 - s), [0.0]]) + np.concatenate([[0.0], out * s])
    return out


def _inside(quad: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Which points are inside a convex quadrilateral (4, 2) (its edges included)."""
    c = np.asarray(quad, float).reshape(4, 2)
    cross = np.stack([(c[(k + 1) % 4, 0] - c[k, 0]) * (y - c[k, 1]) - (c[(k + 1) % 4, 1] - c[k, 1]) * (x - c[k, 0])
                      for k in range(4)])
    return (cross >= 0).all(0) | (cross <= 0).all(0)


def _ink_of(rect: np.ndarray, ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The centres (edge coordinates) of the ink px whose centre is inside a rectangle (4, 2)."""
    h, w = ink.shape
    x0, y0 = np.floor(rect.min(0)).astype(int)
    x1, y1 = np.ceil(rect.max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(0), np.zeros(0)
    ys, xs = np.nonzero(ink[y0:y1, x0:x1])
    x, y = xs + x0 + 0.5, ys + y0 + 0.5
    keep = _inside(rect, x, y)
    return x[keep], y[keep]


def ink_shares(rects: np.ndarray, words: np.ndarray, ink: np.ndarray):
    """For every rectangle of `rects` (n, 4, 2) and Terra word rectangle of `words` (m, 4, 2) (page px edge
    coordinates): each word's share of its ink inside the rectangle (see the module), for the pairs with a share above
    0: (rectangle positions, word positions, shares)."""
    ri, wj, inter = wordboxes.pairs(np.asarray(rects, float).reshape(-1, 4, 2),
                                    np.asarray(words, float).reshape(-1, 4, 2))
    share = np.zeros(len(ri))
    area = wordboxes.poly_area(np.asarray(words, float).reshape(-1, 4, 2))
    for j in np.unique(wj):
        sel = np.flatnonzero(wj == j)
        x, y = _ink_of(words[j], ink)
        if not len(x):
            share[sel] = np.minimum(inter[sel] / max(area[j], 1e-12), 1.0)
            continue
        for k in sel:
            share[k] = _inside(rects[ri[k]], x, y).sum() / len(x)
    keep = share > 0
    return ri[keep], wj[keep], share[keep]


def boundary_target(spans: np.ndarray, held, w: float, out_w: int) -> np.ndarray:
    """The boundary target of each of `out_w` crop columns (see the module): `spans` (m, 2) the words' extents along
    the axis in page-px crop coordinates (0..w), `held` (m,) the chance each is held.  Column c holds [c, c + 1] x w /
    out_w; its target is the mean over it of P(x lies between two consecutive held words)."""
    spans = np.asarray(spans, float).reshape(-1, 2)
    p = np.asarray(held, float).ravel()
    out = np.zeros(out_w)
    if len(spans) < 2 or out_w <= 0:
        return out
    lo, hi = spans.min(1), spans.max(1)
    cuts = np.unique(np.concatenate([np.arange(out_w + 1) * (w / out_w), np.clip(np.r_[lo, hi], 0.0, w)]))
    mid, length = (cuts[:-1] + cuts[1:]) / 2, np.diff(cuts)
    keep = length > 0
    mid, length = mid[keep], length[keep]
    q = 1.0 - p[None, :]
    before = 1.0 - np.where(hi[None, :] <= mid[:, None], q, 1.0).prod(1)
    after = 1.0 - np.where(lo[None, :] >= mid[:, None], q, 1.0).prod(1)
    across = np.where((lo[None, :] < mid[:, None]) & (hi[None, :] > mid[:, None]), q, 1.0).prod(1)
    col = np.clip(np.floor(mid * out_w / w).astype(int), 0, out_w - 1)
    out = np.bincount(col, weights=before * after * across * length, minlength=out_w) / (w / out_w)
    return np.clip(out, 0.0, 1.0)


def _spans(rects: np.ndarray, corners: np.ndarray, axis: float, w: float) -> np.ndarray:
    """The extents along `axis` of word rectangles (m, 4, 2) in the page-px crop coordinates of the box `corners`."""
    (ux, uy), _ = L._axes(axis)
    p = (np.asarray(rects, float).reshape(-1, 4, 2) - np.asarray(corners, float).mean(0)) @ np.array([ux, uy])
    return np.stack([p.min(1), p.max(1)], 1) + w / 2


def _turned(terra_angle, axis) -> np.ndarray:
    """Whether a word reads the other way along the axis (more than a quarter turn from it)."""
    return np.cos(np.radians(np.asarray(terra_angle, float) - np.asarray(axis, float))) < 0


def box_targets(boxes: pd.DataFrame, terra: pd.DataFrame, ink: np.ndarray) -> list[dict]:
    """The targets of each of one sheet's boxes (rows of boxes.parquet) against one sheet's Terra words (C_COLS, and
    `angle`): per box `words` (positions in `terra` with a share above 0), `shares`, `count` (`poisson_binomial`),
    `turned` (whether each word reads against the box's axis), `spans` (the words' extents along the axis in the box
    crop's page-px coordinates), and the words' `rects`, reading `angles` and `heights` (page px), and the crop's `w`
    and `out_w` (`layout`)."""
    rects = boxes[T_COLS].to_numpy(float).reshape(-1, 4, 2)
    words = terra[C_COLS].to_numpy(float).reshape(-1, 4, 2)
    word_h = np.hypot(*(words[:, 3] - words[:, 0]).T) if len(words) else np.zeros(0)
    bi, wj, s = ink_shares(rects, words, ink)
    angle = terra["angle"].to_numpy(float) if "angle" in terra else np.zeros(len(terra))
    out = []
    for k, (_, b) in enumerate(boxes.iterrows()):
        corners, length, height, axis = _box_geometry(b)
        sel = np.flatnonzero(bi == k)
        sel = sel[np.argsort(wj[sel], kind="stable")]
        w, out_w = layout(length, height)
        out.append({"words": wj[sel], "shares": s[sel], "count": poisson_binomial(s[sel]),
                    "turned": _turned(angle[wj[sel]], axis), "spans": _spans(words[wj[sel]], corners, axis, w),
                    "rects": words[wj[sel]], "angles": angle[wj[sel]], "heights": word_h[wj[sel]], "w": w,
                    "out_w": out_w})
    return out


# --- what may go in ---------------------------------------------------------------------------------------------------

def _held_words(terra: pd.DataFrame, quarter: int) -> np.ndarray:
    missing = [c for c in (f"touch_{paths.QUARTERS[quarter]}", "touch_blocks") if c not in terra]
    if missing:
        raise ValueError(f"Terra's words lack {missing}: which are held out can't be checked")
    return terra[f"touch_{paths.QUARTERS[quarter]}"].to_numpy(bool) | terra["touch_blocks"].to_numpy(bool)


def training_words(terra: pd.DataFrame, quarter: int) -> pd.DataFrame:
    """Terra's words whose crop touches neither `quarter` nor an answered block."""
    q = _need_quarter(quarter, "training_words")
    return terra[~_held_words(terra, q)]


def training_boxes(boxes: pd.DataFrame, terra: pd.DataFrame, *, quarter: int, touch, shapes: dict) -> pd.DataFrame:
    """The boxes the model holding `quarter` out may train on: outside the quarter, their crop touching neither it nor
    an answered block (`touch(sheet, (x0, y0, x1, y1), shape) -> {TOUCH name: bool}`, `readercrops.HeldAreas`;
    `shapes` sheet -> page shape), and meeting no held word's rectangle (`terra`: every Terra word, with TOUCH)."""
    q = _need_quarter(quarter, "training_boxes")
    held = _held_words(terra, q)
    keep = (boxes["quarter"].to_numpy() != q)
    sheet_of = boxes["sheet"].astype(str).to_numpy()
    tsheet = terra["sheet"].astype(str).to_numpy()
    for sheet in sorted(set(sheet_of[keep])):
        mine = np.flatnonzero((sheet_of == sheet) & keep)
        theirs = np.flatnonzero((tsheet == sheet) & held)
        rects = boxes.iloc[mine][T_COLS].to_numpy(float).reshape(-1, 4, 2)
        if len(theirs):
            bi, _, _ = wordboxes.pairs(rects, terra.iloc[theirs][C_COLS].to_numpy(float).reshape(-1, 4, 2))
            keep[mine[np.unique(bi)]] = False
        for k in mine:
            if not keep[k]:
                continue
            corners, length, height, axis = _box_geometry(boxes.iloc[k])
            flags = touch(sheet, _footprint(corners, length, height, axis, shapes[sheet]), shapes[sheet])
            keep[k] = not (flags[f"touch_{paths.QUARTERS[q]}"] or flags["touch_blocks"])
    return boxes[keep]


# --- the gaps between words on one line -------------------------------------------------------------------------------

def _in_frame(rects: np.ndarray, angles: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Words (rectangles (m, 4, 2), reading angles (m,)) in the frame of a line read at `angle`: their extents along
    it and across it (m, 2) each, and |cos| of each word's angle to the line's (a quarter turn exactly 0)."""
    (ux, uy), (vx, vy) = L._axes(angle)
    rel = np.asarray(rects, float).reshape(-1, 4, 2)
    p, q = rel @ np.array([ux, uy]), rel @ np.array([vx, vy])
    d = (np.asarray(angles, float) - angle) % 180.0
    turn = np.where(d == 90.0, 0.0, np.abs(np.cos(np.radians(d))))
    return np.stack([p.min(1), p.max(1)], 1), np.stack([q.min(1), q.max(1)], 1), turn


def _shared(across: np.ndarray, turn: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """How much words a and b (index arrays, broadcast) share one line of text (see `line_share`)."""
    over = np.minimum(across[a, 1], across[b, 1]) - np.maximum(across[a, 0], across[b, 0])
    size = np.minimum(across[a, 1] - across[a, 0], across[b, 1] - across[b, 0])
    return np.clip(over / np.maximum(size, 1e-12), 0.0, 1.0) * np.minimum(turn[a], turn[b])


def line_share(rects: np.ndarray, angles: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Words (rectangles (m, 4, 2), reading angles (m,)) in the frame of a line read at `angle`: their extents along
    it (m, 2), and how much each two share one line of text (m, m) in [0, 1]: the overlap of their extents across it
    over the smaller extent, times the smaller |cos| of a word's angle to the line's (1: one word's letters wholly
    within the other's band, read the same way; 0: stacked words, or a word read across the line)."""
    span, across, turn = _in_frame(rects, angles, angle)
    m = np.arange(len(span))
    return span, _shared(across, turn, m[:, None], m[None, :])


def pair_gaps(t: dict) -> list[tuple[int, int, float, float]]:
    """The consecutive pairs on one line of text among the words one box holds (`box_targets`; see the module):
    (word, next word (positions of `terra`, in reading order), gap, weight).  Each two words are taken in the frame of
    the first's reading direction (`line_share`): the gap from the earlier one's end to the later one's start along
    it, over their mean height; the weight the chance that both are held, that they share a line (as a chance) and
    that no word between them along it is held on that line: s_i s_j l_ij prod_k (1 - s_k min(l_ik, l_kj))."""
    s, h, rects, angles = t["shares"], t["heights"], t["rects"], t["angles"]
    out = []
    for i in range(len(s)):
        span, line = line_share(rects, angles, float(angles[i]))
        c = span.mean(1)
        for j in range(i + 1, len(s)):
            a, b = (i, j) if c[i] <= c[j] else (j, i)
            between = np.flatnonzero((c > c[a]) & (c < c[b]))
            between = between[(between != i) & (between != j)]
            weight = s[i] * s[j] * line[i, j] * np.prod(1.0 - s[between] * np.minimum(line[i, between],
                                                                                        line[between, j]))
            if weight > 0:
                gap = (span[b, 0] - span[a, 1]) / ((h[i] + h[j]) / 2)
                out.append((int(t["words"][a]), int(t["words"][b]), float(gap), float(weight)))
    return out


def held_pair_gaps(boxes: pd.DataFrame, terra: pd.DataFrame, ink: np.ndarray, targets: list | None = None
                   ) -> pd.DataFrame:
    """The real same-line gaps of one sheet's boxes (see the module): one row per consecutive held pair of every box
    (`pair_gaps`): sheet, box (its index label), word and next (positions in `terra`), gap, weight.  `targets`: the
    boxes' `box_targets` against `terra` (computed when None)."""
    targets = box_targets(boxes, terra, ink) if targets is None else targets
    rows = [{"sheet": str(b["sheet"]), "box": label, "word": a, "next": c, "gap": g, "weight": w}
            for (label, b), t in zip(boxes.iterrows(), targets) for a, c, g, w in pair_gaps(t)]
    return pd.DataFrame(rows, columns=["sheet", "box", "word", "next", "gap", "weight"])


def same_line_gaps(terra: pd.DataFrame) -> pd.DataFrame:
    """The real same-line gaps (see the module): for each Terra word, every other word of its sheet whose centre lies
    ahead along its reading direction and that shares its line of text (`line_share`), with the chance that it is the
    word's next word on that line: that it shares the line and that no word ahead nearer (by centre) does,
    l_ab prod_k (1 - l_ak).  `gap`: from the word's end to the other's start along its reading direction, over their
    mean height (negative when their rectangles overlap).  Columns: sheet, word, next (Terra's word numbers), pos,
    next_pos (positions in `terra`), gap, weight."""
    rects_all = terra[C_COLS].to_numpy(float).reshape(-1, 4, 2)
    angle_all = terra["angle"].to_numpy(float)
    number = terra["word"].to_numpy() if "word" in terra else np.arange(len(terra))
    sheets = terra["sheet"].astype(str).to_numpy()
    rows = []
    for sheet in sorted(set(sheets)):
        idx = np.flatnonzero(sheets == sheet)
        r, ang = rects_all[idx], angle_all[idx]
        h = np.hypot(*(r[:, 3] - r[:, 0]).T)
        for a in range(len(idx)):
            span, across, turn = _in_frame(r, ang, float(ang[a]))
            c = span.mean(1)
            ahead = np.flatnonzero(c > c[a])
            line = _shared(across, turn, np.full(len(ahead), a), ahead)
            ahead, line = ahead[line > 0], line[line > 0]
            order = np.argsort(c[ahead], kind="stable")
            ahead, line = ahead[order], line[order]
            weight = line * np.concatenate([[1.0], np.cumprod(1.0 - line)[:-1]])
            for b, w in zip(ahead[weight > 0], weight[weight > 0]):
                rows.append({"sheet": sheet, "word": int(number[idx[a]]), "next": int(number[idx[b]]),
                             "pos": int(idx[a]), "next_pos": int(idx[b]),
                             "gap": float((span[b, 0] - span[a, 1]) / ((h[a] + h[b]) / 2)), "weight": float(w)})
    return pd.DataFrame(rows, columns=["sheet", "word", "next", "pos", "next_pos", "gap", "weight"])


def _weighted_quantiles(x: np.ndarray, w: np.ndarray, qs) -> list[float]:
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    c = np.cumsum(w)
    return [float(x[min(np.searchsorted(c, q * c[-1]), len(x) - 1)]) for q in qs]


def gap_distribution(gaps: pd.DataFrame) -> dict:
    """Measured gaps (`same_line_gaps`, or `held_pair_gaps`) as a distribution to draw from: every `gap` with its
    `weight`, `n`, the weights' sum and the weighted QUANTILES (reported only)."""
    x, w = gaps["gap"].to_numpy(float), gaps["weight"].to_numpy(float)
    out = {"gap": x.tolist(), "weight": w.tolist(), "n": int(len(x)), "weight_sum": float(w.sum())}
    out["quantiles"] = (dict(zip(map(str, QUANTILES), _weighted_quantiles(x, w, QUANTILES)))
                        if len(x) and w.sum() > 0 else None)
    return out


def draw_gaps(dist: dict, n: int, rng) -> np.ndarray:
    """`n` gaps drawn from a measured distribution (`gap_distribution`: each measured gap with the chance of its
    weight)."""
    x, w = np.asarray(dist["gap"], float), np.asarray(dist["weight"], float)
    if not len(x) or w.sum() <= 0:
        raise ValueError("no measured gap to draw from")
    return x[rng.choice(len(x), size=n, p=w / w.sum())]


# --- the examples -----------------------------------------------------------------------------------------------------

@dataclass
class Examples:
    """The box model's examples: crop i is `flat[:, :, offsets[i]:offsets[i + 1]]` (float32, (C, HEIGHT, W)) with
    channels `channels`; its count distribution `count[i]` (over 0..K), its boundary target per column
    `boundary[offsets[i]:offsets[i + 1]]`, its `weight`; `meta` one row each."""
    channels: tuple[str, ...]
    flat: np.ndarray
    offsets: np.ndarray
    count: np.ndarray
    boundary: np.ndarray
    weight: np.ndarray
    meta: pd.DataFrame

    @classmethod
    def of(cls, crops, counts, boundaries, weights, meta: pd.DataFrame, channels) -> "Examples":
        """From lists: crops (C, HEIGHT, W_i), count distributions (any length; padded to K + 1, K the largest count
        with any chance among them), boundary targets (W_i,), weights."""
        crops = [np.asarray(c, np.float32) for c in crops]
        widths = np.array([c.shape[2] for c in crops], np.int64)
        if any(np.shape(b) != (wd,) for b, wd in zip(boundaries, widths)):
            raise ValueError("a boundary target per crop column")
        top = max([int(np.flatnonzero(np.asarray(c) > 0).max()) for c in counts if np.any(np.asarray(c) > 0)],
                  default=0)
        count = np.zeros((len(counts), top + 1))
        for i, c in enumerate(counts):
            c = np.asarray(c, float)[:top + 1]
            count[i, :len(c)] = c
        flat = (np.concatenate(crops, axis=2) if crops else np.zeros((len(channels), HEIGHT, 0), np.float32))
        return cls(tuple(channels), np.ascontiguousarray(flat), np.concatenate([[0], np.cumsum(widths)]).astype(
            np.int64), count, np.concatenate([np.asarray(b, np.float32) for b in boundaries]) if len(boundaries)
            else np.zeros(0, np.float32), np.asarray(weights, float), meta.reset_index(drop=True))

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def K(self) -> int:
        return self.count.shape[1] - 1

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.offsets)

    def crop(self, i: int) -> np.ndarray:
        return self.flat[:, :, self.offsets[i]:self.offsets[i + 1]]

    def target(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.count[i], self.boundary[self.offsets[i]:self.offsets[i + 1]]

    def save(self, path: Path) -> None:
        """`<path>.npz` (fixed dates, `store.save_npz`) and `<path>.parquet` (the meta)."""
        path = Path(path)
        store.save_npz(path.with_suffix(".npz"), channels=np.array(self.channels), flat=self.flat,
                       offsets=self.offsets, count=self.count, boundary=self.boundary, weight=self.weight)
        self.meta.to_parquet(path.with_suffix(".parquet"), index=False)

    @classmethod
    def load(cls, path: Path) -> "Examples":
        path = Path(path)
        with np.load(path.with_suffix(".npz")) as z:
            return cls(tuple(str(c) for c in z["channels"]), z["flat"], z["offsets"], z["count"], z["boundary"],
                       z["weight"], pd.read_parquet(path.with_suffix(".parquet")))


def _word_crop(inputs: SheetInputs, t) -> tuple[np.ndarray, tuple[float, float], float]:
    """A Terra word's crop as the reader cuts it (`crops.rectify`'s centre, the row's length and floored height):
    every layer, its span along the line in crop columns, and its height in crop columns."""
    length, height = float(t["length"]), float(t["height"])
    centre = ((t["box_x0"] + t["box_x1"] - 1) / 2, (t["box_y0"] + t["box_y1"] - 1) / 2)
    crop = _cut(inputs, centre, length, height, float(t["angle"]))
    w, out_w = layout(length, height)
    if crop.shape[2] != out_w:
        raise AssertionError(f"crops.cut made a crop {crop.shape[2]} wide, its layout {out_w}")
    s = out_w / w
    return crop, ((w / 2 - length / 2) * s, (w / 2 + length / 2) * s), height * s


def join(a: tuple, b: tuple, gap: float, turned: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Two word crops (`_word_crop`'s three) on one line, `gap` word heights (their mean) from the first's end to the
    second's start, B placed at the nearest whole crop column: the first word's crop up to its end, paper across the
    gap, the second's from its start; where they overlap (a negative gap), each px from the crop with more ink there.
    Squeezed to `crops.MAX_WIDTH` as `crops.cut` squeezes, and turned a half turn with `turned`.  The crop, its
    boundary target (the gap's columns) and the gap realised (`gap_px`, `unit_px`: crop columns before any squeeze)."""
    (ca, (a0, a1), ua), (cb, (b0, b1), ub) = a, b
    unit = (ua + ub) / 2
    o = int(round(a1 + gap * unit - b0))          # B's crop's first column, from A's
    oa = max(0, -o)
    ob = oa + o
    sa0, sa1, sb0, sb1 = oa + a0, oa + a1, ob + b0, ob + b1
    width = max(oa + ca.shape[2], ob + cb.shape[2])
    out = np.zeros((ca.shape[0], HEIGHT, width), np.float32)
    end_a = min(oa + ca.shape[2], math.ceil(sa1))            # A's crop up to its word's end
    if end_a > oa:
        out[:, :, oa:end_a] = ca[:, :, :end_a - oa]
    start_b = max(ob, math.floor(sb0))                      # B's from its word's start
    if start_b < ob + cb.shape[2]:
        part, here = cb[:, :, start_b - ob:], out[:, :, start_b:ob + cb.shape[2]]
        out[:, :, start_b:ob + cb.shape[2]] = np.where(part[:1] > here[:1], part, here)
    spans = np.array([[sa0, sa1], [sb0, sb1]])
    out_w = min(width, K.MAX_WIDTH)
    target = boundary_target(spans, np.ones(2), width, out_w)
    if out_w < width:
        out = np.stack([cv2.resize(c, (out_w, HEIGHT), interpolation=cv2.INTER_AREA) for c in out])
    if turned:
        out, target = np.ascontiguousarray(out[:, ::-1, ::-1]), target[::-1].copy()
    return out, target, {"gap_px": float(sb0 - sa1), "unit_px": float(unit), "width": int(width)}


def box_examples(boxes: pd.DataFrame, terra: pd.DataFrame, sheets, quarter: int, weights=None, *, touch=None,
                 joined: int | None = None, gaps: dict | None = None, turned: float | None = None,
                 seed: int = SEED) -> Examples:
    """The examples for the box model holding `quarter` out (see the module).  `boxes`: rows of boxes.parquet (any
    run's, Task 9's candidates alike); `terra`: Terra's words (`terra_frame`: C_COLS, angle, the axis box, the crop's
    length and height, TOUCH), every word that meets a box among them; `sheets`: sheet -> `SheetInputs` (a mapping or a
    function); `weights`: per word of `terra` (Task 7's; None: 1); `touch` (`training_boxes`; `readercrops.HeldAreas`
    when None); `joined`: how many joined neighbours (None: the expected number of real boxes holding two words or
    more); `gaps`: the gap distribution (`gap_distribution`; None: measured from `terra`, `same_line_gaps`);
    `turned`: the share of joined crops turned a half turn (None: measured from these boxes).  ValueError on a held
    word or box (see the module)."""
    q = _need_quarter(quarter, "box_examples")
    held = _held_words(terra, q)
    if held.any():
        raise ValueError(f"{int(held.sum())} of Terra's words touch the held quarter {paths.QUARTERS[q]} or an "
                         "answered block: pass training_words(...)")
    if (boxes["quarter"].to_numpy() == q).any():
        raise ValueError(f"boxes in the held quarter {paths.QUARTERS[q]}: pass training_boxes(...)")
    terra = terra.reset_index(drop=True)
    wt = np.ones(len(terra)) if weights is None else np.asarray(weights, float).ravel()
    if len(wt) != len(terra) or (wt < 0).any() or (wt > 1).any():
        raise ValueError(f"weights: {len(wt)} in [0, 1] for {len(terra)} words")
    if touch is None:
        from sfzmaps.solver.wordread import readercrops
        touch = readercrops.HeldAreas()
    get = _getter(sheets)
    crops, counts, bounds, wts, meta = [], [], [], [], []
    channels = None
    turn_s = turn_n = 0.0
    sheet_of = boxes["sheet"].astype(str).to_numpy()
    tsheet = terra["sheet"].astype(str).to_numpy()
    for sheet in sorted(set(sheet_of)):
        inputs = get(sheet)
        if channels is None:
            channels = inputs.channels
        elif inputs.channels != channels:
            raise ValueError(f"{sheet}: channels {inputs.channels}, the others' {channels}")
        mine = boxes[sheet_of == sheet]
        twords = np.flatnonzero(tsheet == sheet)
        got = box_targets(mine, terra.iloc[twords], inputs.ink)
        for (label, b), t in zip(mine.iterrows(), got):
            corners, length, height, axis = _box_geometry(b)
            flags = touch(sheet, _footprint(corners, length, height, axis, inputs.shape), inputs.shape)
            if flags[f"touch_{paths.QUARTERS[q]}"] or flags["touch_blocks"]:
                raise ValueError(f"{sheet} box {label}: its crop touches the held quarter or an answered block")
            crop = box_crop(inputs, corners, length, height, axis)
            if crop.shape[2] != t["out_w"]:
                raise AssertionError(f"crops.box_cut made a crop {crop.shape[2]} wide, its layout {t['out_w']}")
            pos = twords[t["words"]]
            s = t["shares"]
            crops.append(crop)
            counts.append(t["count"])
            bounds.append(boundary_target(t["spans"], s, t["w"], t["out_w"]))
            wts.append(float(np.prod(1.0 - s * (1.0 - wt[pos]))))
            turn_s += float((s * t["turned"]).sum())
            turn_n += float(s.sum())
            meta.append({"source": "box", "sheet": sheet, "box": int(label) if np.issubdtype(type(label), np.integer)
                         else -1, "quarter": int(b["quarter"]), "sheet_b": "", "word_a": -1, "word_b": -1,
                         "gap": np.nan, "gap_px": np.nan, "unit_px": np.nan, "turned": False,
                         "support": int(len(s)), "expected": float(s.sum())})
    n_join = (int(round(sum(float(c[2:].sum()) for c in counts))) if joined is None else int(joined))
    if n_join > 0:
        if len(terra) < 2:
            raise ValueError("joined neighbours need two training words or more")
        dist = gap_distribution(same_line_gaps(terra)) if gaps is None else gaps
        share = (turn_s / turn_n if turn_n > 0 else 0.0) if turned is None else float(turned)
        rng = np.random.default_rng([seed, 1])
        first = rng.integers(0, len(terra), n_join)
        second = rng.integers(0, len(terra) - 1, n_join)
        second = second + (second >= first)
        gap = draw_gaps(dist, n_join, np.random.default_rng([seed, 2]))
        turn = np.random.default_rng([seed, 3]).random(n_join) < share
        need = np.unique(np.r_[first, second])
        word_crops = {}
        for sheet in sorted(set(tsheet[need])):
            inputs = get(sheet)
            if channels is None:
                channels = inputs.channels
            elif inputs.channels != channels:
                raise ValueError(f"{sheet}: channels {inputs.channels}, the others' {channels}")
            for p in need[tsheet[need] == sheet]:
                word_crops[int(p)] = _word_crop(inputs, terra.iloc[int(p)])
        for a, b, g, tn in zip(first, second, gap, turn):
            crop, target, got = join(word_crops[int(a)], word_crops[int(b)], float(g), bool(tn))
            crops.append(crop)
            counts.append(np.array([0.0, 0.0, 1.0]))
            bounds.append(target)
            wts.append(float(wt[a] * wt[b]))
            meta.append({"source": "joined", "sheet": tsheet[a], "box": -1, "quarter": int(terra["quarter"].iat[a]),
                         "sheet_b": tsheet[b], "word_a": int(a), "word_b": int(b), "gap": float(g),
                         "gap_px": got["gap_px"], "unit_px": got["unit_px"], "turned": bool(tn), "support": 2,
                         "expected": 2.0})
    if channels is None:
        raise ValueError("no box and no joined neighbour: nothing to learn from")
    return Examples.of(crops, counts, bounds, wts, pd.DataFrame(meta), channels)


def terra_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Every Terra word of the 2000 sheets (any kind and string) from rows of `solver-text-crops`' crops.parquet, as
    `box_examples` takes them: sheet, word (its number on the sheet), kind, text (Terra's), quarter, angle, the axis
    box, the crop's length and height (a slanted word's height floored, as the reader's crops), TOUCH, and C_COLS
    (`textnet.boxes.word_rect`)."""
    w = rows[(rows["series"] == paths.SERIES) & (rows["word"] >= 0)].reset_index(drop=True)
    box = w[["box_x0", "box_y0", "box_x1", "box_y1"]].to_numpy(float)
    angle = w["angle"].to_numpy(float)
    rects = np.array([textboxes.word_rect(b, a) for b, a in zip(box, angle)], float).reshape(-1, 4, 2)
    return pd.DataFrame({"sheet": w["sheet"].astype(str).to_numpy(), "word": w["word"].astype(int).to_numpy(),
                         "kind": w["kind"].to_numpy(), "text": w["terra"].astype(str).to_numpy(),
                         "quarter": w["quarter"].astype(int).to_numpy(), "angle": angle,
                         **{c: box[:, k] for k, c in enumerate(("box_x0", "box_y0", "box_x1", "box_y1"))},
                         "length": w["length"].to_numpy(float), "height": w["height"].to_numpy(float),
                         **{t: w[t].astype(bool).to_numpy() for t in TOUCH},
                         **dict(zip(C_COLS, rects.reshape(-1, 8).T))})


# --- the network ------------------------------------------------------------------------------------------------------

class BoxNet(nn.Module):
    """The box model (see the module): `forward(x, frames)` with x (n, channels, HEIGHT, W), W a multiple of STRIDE,
    and each crop's own frames; gives the count logits (n, k + 1) and a boundary logit per frame (n, W / STRIDE)."""

    def __init__(self, channels: int, k: int):
        super().__init__()
        self.channels, self.k = int(channels), int(k)
        layers = [model.ConvNormAct(self.channels, STEM, 3, (2, 1))]
        for cin, kernel, exp, cout, se, hard, stride in LAYERS:
            layers.append(model.InvertedResidual(cin, kernel, exp, cout, se, hard, stride))
        layers.append(model.ConvNormAct(LAYERS[-1][3], LAST, 1))
        self.features = nn.Sequential(*layers)
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                m.momentum = BN_MOMENTUM
        self.context = nn.ModuleList([nn.Conv1d(LAST, LAST, 3, padding=d, dilation=d) for d in DILATIONS])
        self.boundary = nn.Conv1d(LAST, 1, 1)
        self.count = nn.Sequential(nn.Linear(3 * LAST, LAST), nn.ReLU(inplace=True), nn.Linear(LAST, self.k + 1))

    def forward(self, x: torch.Tensor, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.features(x)
        if f.shape[2] != 1:
            raise ValueError(f"crops {x.shape[2]} px tall; the box model reads {HEIGHT}")
        f = f[:, :, 0]
        t = f.shape[2]
        mask = (torch.arange(t, device=f.device)[None, :] < frames[:, None]).to(f.dtype)[:, None]
        f = f * mask
        for conv in self.context:
            f = (f + F.relu(conv(f))) * mask
        boundary = self.boundary(f)[:, 0]
        n = frames.to(f.dtype)[:, None]
        mean = f.sum(2) / n
        top = f.masked_fill(mask == 0, float("-inf")).amax(2)
        total = f.sum(2) / SUM_SCALE
        return self.count(torch.cat([mean, top, total], 1)), boundary


def _cpu_convs(device: str):
    """On the CPU, NNPACK off while the body runs: it convolves a depthwise layer one group at a time (3x slower here);
    elsewhere nothing."""
    return torch.backends.nnpack.flags(enabled=False) if str(device) == "cpu" else contextlib.nullcontext()


def _batch(crops: list[np.ndarray], width: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Crops (C, HEIGHT, W_i) padded with paper (0) on the right to whole frames (at least two; to `width` when
    given) as the network's input, and each crop's own frames."""
    frames = [max(2, math.ceil(c.shape[2] / STRIDE)) for c in crops]
    w = max(width or 0, STRIDE * max(frames))
    x = np.zeros((len(crops), crops[0].shape[0], HEIGHT, w), np.float32)
    for i, c in enumerate(crops):
        x[i, :, :, :c.shape[2]] = c
    return torch.from_numpy(x), torch.tensor(frames, dtype=torch.long)


def _frame_targets(boundary: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """A crop's column targets as its frames' (the mean over each frame's own columns) and each frame's own columns."""
    w = len(boundary)
    cols = np.clip(w - STRIDE * np.arange(frames), 0, STRIDE).astype(float)
    sums = np.bincount(np.arange(w) // STRIDE, weights=boundary, minlength=frames)[:frames]
    return np.where(cols > 0, sums / np.maximum(cols, 1), 0.0), cols


def fit_boxmodel(examples: Examples, recipe: dict | None = None) -> BoxNet:
    """A `BoxNet` trained on `examples` by `recipe` (RECIPE's keys; see the module), in an exclusive GPU turn on mps,
    seeded (torch and numpy, `recipe["seed"]`).  Each step draws `batch` examples (`np.random.default_rng([seed, 0])`),
    pads them to the widest, and minimises the weighted mean over them of the count's cross-entropy against its target
    distribution plus `boundary_weight` x the mean binary cross-entropy of its columns.  AdamW on a one-cycle
    schedule, gradients clipped; a fixed number of steps.  The network comes back on the CPU in eval mode, its log as
    `fit_log`."""
    rc = {**RECIPE, **(recipe or {})}
    if not len(examples):
        raise ValueError("no examples")
    seed, device = int(rc["seed"]), str(rc["device"])
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, 0])
    net = BoxNet(len(examples.channels), examples.K)
    widths = examples.widths
    log = {"recipe": rc, "examples": len(examples), "k": examples.K, "channels": list(examples.channels),
           "torch": torch.__version__, "steps": []}
    with gpu.gpu_turn(True, device), _cpu_convs(device):
        t0 = time.perf_counter()
        net = net.to(device)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=rc["max_lr"], weight_decay=rc["weight_decay"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=rc["max_lr"], total_steps=int(rc["steps"]),
                                                    pct_start=rc["pct_start"])
        run, runs_n = 0.0, 0
        for step in range(1, int(rc["steps"]) + 1):
            idx = rng.integers(0, len(examples), int(rc["batch"]))
            idx = idx[np.argsort(widths[idx], kind="stable")]
            x, frames = _batch([examples.crop(int(i)) for i in idx])
            t = x.shape[3] // STRIDE
            tf, cf = np.zeros((len(idx), t), np.float32), np.zeros((len(idx), t), np.float32)
            for k, i in enumerate(idx):
                a, c = _frame_targets(examples.target(int(i))[1], int(frames[k]))
                tf[k, :len(a)], cf[k, :len(c)] = a, c
            count_t = torch.from_numpy(examples.count[idx].astype(np.float32)).to(device)
            w = torch.from_numpy(examples.weight[idx].astype(np.float32)).to(device)
            logits, blog = net(x.to(device), frames.to(device))
            ce = -(count_t * F.log_softmax(logits, 1)).sum(1)
            bce = F.binary_cross_entropy_with_logits(blog, torch.from_numpy(tf).to(device), reduction="none")
            cf_t = torch.from_numpy(cf).to(device)
            per = ce + rc["boundary_weight"] * (bce * cf_t).sum(1) / cf_t.sum(1)
            loss = (per * w).sum() / w.sum().clamp(min=1e-12)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), rc["clip"])
            opt.step()
            sched.step()
            run += float(loss.detach().cpu())
            runs_n += 1
            if step % int(rc["log_every"]) == 0 or step == int(rc["steps"]):
                mean = run / runs_n
                if not math.isfinite(mean):
                    raise FloatingPointError(f"box model: loss {mean} at step {step}")
                log["steps"].append({"step": step, "loss": round(mean, 5), "lr": sched.get_last_lr()[0],
                                     "seconds": round(time.perf_counter() - t0, 1)})
                run, runs_n = 0.0, 0
        if gpu.on_gpu(device):
            torch.mps.synchronize()
        log["train_seconds"] = round(time.perf_counter() - t0, 1)
        net = net.cpu()
        if gpu.on_gpu(device):
            torch.mps.empty_cache()
    net.eval()
    net.fit_log = log
    return net


def box_logp(net: BoxNet, crops: list[np.ndarray], device: str = "cpu", batch: int = 64
             ) -> tuple[np.ndarray, list[np.ndarray]]:
    """The box model's log-probabilities of crops (C, HEIGHT, W_i) (`box_crop`): the count's (n, K + 1), and per crop
    its columns' (W_i, 2): log P(no boundary), log P(boundary).  Crops are read in batches of one padded width, so a
    crop's numbers are its own.  On mps, inside an exclusive GPU turn (don't call it inside another turn)."""
    if not len(crops):
        return np.zeros((0, net.k + 1)), []
    frames = np.array([max(2, math.ceil(c.shape[2] / STRIDE)) for c in crops])
    count = np.zeros((len(crops), net.k + 1))
    bounds: list = [None] * len(crops)
    net = net.eval()
    with gpu.gpu_turn(True, device), _cpu_convs(device), torch.no_grad():
        net = net.to(device)
        for f in np.unique(frames):
            idx = np.flatnonzero(frames == f)
            for s in range(0, len(idx), batch):
                part = idx[s:s + batch]
                x, fr = _batch([np.asarray(crops[i], np.float32) for i in part], int(f) * STRIDE)
                logits, blog = net(x.to(device), fr.to(device))
                count[part] = F.log_softmax(logits.double(), 1).cpu().numpy()
                b = blog.double().cpu()
                lp = torch.stack([F.logsigmoid(-b), F.logsigmoid(b)], 2).numpy()
                for k, i in enumerate(part):
                    bounds[i] = np.repeat(lp[k], STRIDE, axis=0)[:crops[i].shape[2]]
        net.cpu()
    return count, bounds


def save_boxnet(path: Path, net: BoxNet, **about) -> None:
    """The weights (on the CPU), the channels and K, and `about` (JSON-able), in one torch file."""
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "channels": net.channels, "k": net.k, "about": json.loads(json.dumps(about, default=str))}, path)


def load_boxnet(path: Path) -> BoxNet:
    """A network written by `save_boxnet`, in eval mode on the CPU."""
    got = torch.load(path, map_location="cpu", weights_only=True)
    net = BoxNet(got["channels"], got["k"])
    net.load_state_dict(got["state_dict"], strict=True)
    return net.eval()
