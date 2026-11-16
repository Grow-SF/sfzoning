"""The code reader's alphabet: glyph templates learned from the page, and the noise that turns a rendered code into
the scan's ink (Kopec & Chou's document image decoding; Berg-Kirkpatrick et al.'s Ocular).

The font (`Font`).  One template per character: p(ink) at each px of a ROWS-high grid, SIZE px to the cap height,
in the upright frame; the lettering's lean (`slant`, degrees, leaning right) is put back when text is laid out.  Each
character has an `origin` (the template column of the pen) and an `advance` (how far the pen moves after it), as
type is set: the pen starts at 0 and each character's origin sits on it.  A gap between two pens then measures the
first character's advance alone.  The text's centre is the middle of its pen's travel (the advances' sum) at the
middle of the cap height.  The space has an advance and no template.  `extent` is the first and last column of the
first font's ink (one character's px give way to the next's halfway between their inks); `width` is a character's
inked width (its template px at p >= 0.5).

The first font (`initial_font`) is a plain sans: OpenCV's Hershey Simplex (a single-stroke sans, no font files),
drawn with a stroke INIT_STROKE of the cap height, anti-aliased at 4x and averaged down.

Rendering (`render`).  The laid-out text's shape (the template px at p >= 0.5, united over the characters), leaned
by the slant, as a signed distance from the stroke edge (template px; + on a stroke).  A `Pose` places it on a crop:
the text's centre at (x, y) in the crop, `scale` crop px per template px, turned by `angle` degrees
counter-clockwise (180: upside down).  The result is the signed distance in crop px, clipped to EDGE_PX either way.

The noise (`NoiseModel`).  Kanungo et al.'s degradation model, measured rather than parametric: p(ink) at a crop px
is a function of its signed distance d from the rendered stroke edge, linear between knots KNOT_PX apart over
[-EDGE_PX, EDGE_PX] and constant beyond, and never falling as d grows.  Off the strokes, p is the paper-to-ink flip
rate; on them, 1 - p is the ink-to-paper rate; far off them it is the background's ink (lines the masks miss,
other text).  Px are independent given the rendering.  Px on a line (`labels.line_mask`) are unobserved: no
likelihood, no fit and no template reads them, so a line through a label neither helps nor hurts any text.

Learning (`learn_font`): hard EM on the training labels' crops and their read strings (`consensus`).
  - Round 0 chooses the first font's slant (INIT_SLANTS) by the likelihood of every SLANT_EVERY-th label.
  - E-step (`align`): each string laid out with the current font and aligned by its best pose: a grid of scales and
    angles (`search_for`) and every position in the crop, with a Gaussian prior on where its drawn ink's centre is
    (about Terra's word's centre, or the point).  The likelihood of every position is two correlations of the crop
    with the rendered text (by FFT).  Round 0's prior is broad (POSITION_SD word heights); from round 1 on, a word
    label's prior has the spread measured in round 0 (the "good" part of the escape's mixture below), since a broad
    prior lets a string go to larger ink elsewhere in the crop, which a likelihood summed over px rewards.
  - The "mislabelled" escape (`label_weights`): where a label's string was found (its drawn ink's centre), measured
    from where the label says it is (Terra's word's centre, or the point), and with a word how large (its cap height
    over the word's height), is drawn from a "good" Gaussian or from "bad", any pose tried alike (a wrong string, or
    a word whose ink is all on lines, finds its best match on other ink, or swells to take in more ink).  Both are
    fitted to the round's fits, word labels and point labels apart, no sd below the pose grid's rounding; each
    label counts by its share of "good".
  - M-step: each character placed at its own best shift near its laid-out spot (`refine`; a Gaussian prior on the
    shift; the other characters' strokes unobserved meanwhile), then each template px re-estimated as the labels'
    weighted mean ink there, line px unobserved, with the first font's p counting as PRIOR_WEIGHT instances.  A
    character's px are those on its side of the middles of the gaps to its neighbours' ink (the first font's,
    upright frame).  The advances are fitted by least squares to the gaps between the shifted pens (ridge
    PRIOR_WEIGHT towards the last); each label is moved to its best spot at its scale and angle under the new
    layout, and the noise refitted there with the new templates.
  - Each round's training log-likelihood (every label at its best pose) goes into the font's `trace`.  Hard EM's
    rounds wander a little once settled (the M-step's per-character shifts are not the E-step's rigid layout), so
    the font returned is the round's (after round 0) that explained the training ink best.

Deterministic: no randomness; labels aligned on threads (scipy's FFT, OpenCV and numpy on the CPU), gathered in
label order.
"""
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import fft

from sfzmaps.solver import codes, grid
from sfzmaps.solver.codereader import labels as labels_mod

log = logging.getLogger(__name__)

SIZE = 32                  # a template's cap height, px
VPAD = 10                  # template rows above the cap height and below the baseline
HPAD = 10                  # template columns either side of the first font's ink
ROWS = SIZE + 2 * VPAD     # every template's height
MIDLINE = VPAD + (SIZE - 1) / 2    # the row at the middle of the cap height (px centres)
EDGE_PX = 6.0              # the noise's reach from a stroke edge, crop px; beyond it p(ink) is constant
KNOT_PX = 0.5              # the noise's knots, crop px apart
S_MIN = 0.25               # the smallest scale rendered (a cap 8 px tall); a smaller one is drawn at this
MARGIN = math.ceil(EDGE_PX / S_MIN) + 2    # a laid-out text's margin, template px: its signed distance reaches EDGE_PX
FAR = 1e3                  # the signed distance beyond a laid-out text's margin, template px
INIT_FACE = cv2.FONT_HERSHEY_SIMPLEX
INIT_STROKE = 0.18         # the first font's stroke, of the cap height (the 2000 lettering: about 0.2)
INIT_FLOOR = 0.02          # the first font's p(ink) is INIT_FLOOR on paper and 1 - INIT_FLOOR on a stroke
INIT_SLANTS = (0.0, 5.0, 10.0, 15.0, 20.0)   # the first font's slants tried in round 0, degrees
SLANT_EVERY = 4            # round 0 tries the slants on every 4th label
PRIOR_WEIGHT = 2.0         # the first font's p (and the last advances) count as this many instances
ITERATIONS = 6             # EM rounds
# the line rule the training crops' unobserved px come from (`labels.line_mask`): "run" alone, as "long" hides large
# letters (82-100% of a big P's or SSO's ink) and so sends their strings to other ink
RULES = "run"
RESIDUALS = (-6.0, -3.0, 0.0, 3.0, 6.0)      # degrees tried about each base angle
# cap height / Terra's word height tried with a box, steps of 7.8%: the fitted ratio's median is 1.0, and up to 1.25
# a tenth of the words sat at the top (Terra's box a little short of tall letters)
RATIOS = tuple(np.geomspace(0.6, 1.6, 14))
POSITION_SD = 1.0          # the position prior's sd with a box: this many word heights about the box's centre
POINT_SD = 2.0             # without a box: this many median word heights about the point
POINT_QUANTILES = (0.05, 0.99)               # without a box, cap heights between these word-height quantiles x RATIOS
SHIFT = 0.25               # `refine` moves a character up to this share of the cap height either way
INK_P = (0.08, 0.92)       # the noise before any fit: p(ink) far off and deep in a stroke ...
INIT_EDGE_PX = 0.8         # ... a logistic in the distance with this scale, crop px


# --- poses ------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Pose:
    """Where text sits on a crop: its centre (crop px, px centres), crop px per template px, and its turn in
    degrees counter-clockwise (0 reads left to right, 180 is upside down)."""
    x: float
    y: float
    scale: float
    angle: float


def _axes(angle: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """The reading direction and the direction down the letters on the crop (x right, y down): unit vectors, exact for
    a quarter turn."""
    turns = angle / 90
    if turns == round(turns):
        eu = ((1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0))[round(turns) % 4]
    else:
        a = math.radians(angle)
        eu = (math.cos(a), -math.sin(a))
    return eu, (-eu[1], eu[0])


def _linear(scale: float, angle: float, slant: float) -> np.ndarray:
    """The (2, 2) map from the upright frame (template px; x along the line, y down the letters) to crop px: the lean,
    then the scale and turn."""
    (ux, uy), (vx, vy) = _axes(angle)
    lean = np.array([[1.0, -math.tan(math.radians(slant))], [0.0, 1.0]])
    return max(scale, S_MIN) * np.array([[ux, vx], [uy, vy]]) @ lean


# --- the font ---------------------------------------------------------------------------------------------------------

@dataclass(eq=False)
class Laid:
    """Text laid out by a font: its shape's signed distance (template px, leaned; + on a stroke) over a canvas, and
    the canvas px (ox, oy) of the text's centre."""
    sdf: np.ndarray
    ox: float
    oy: float


@dataclass(eq=False)
class Font:
    """Glyph templates (see the module)."""
    templates: dict[str, np.ndarray]     # float32 (ROWS, width): p(ink) per px; no entry for the space
    origin: dict[str, float]             # the template column of the pen
    extent: dict[str, tuple[float, float]]   # the template columns of the first font's ink, first and last
    advance: dict[str, float]            # the pen's move after it, template px (the space's too)
    slant: float = 0.0                   # the lettering's lean, degrees (leaning right)
    size: int = SIZE                     # the cap height, template px
    counts: dict[str, float] = field(default_factory=dict)    # the weighted instances each template learned from
    spacing_sd: float = 0.0              # the gaps' spread about the advances' prediction, template px
    position_sd: float = 0.0             # the drawn text's spread about Terra's word centre, word heights (round 0)
    noise: "NoiseModel | None" = None    # the noise fitted with it
    trace: list[dict] = field(default_factory=list)          # learn_font's rounds
    _laid: dict = field(default_factory=dict, repr=False)

    @property
    def chars(self) -> str:
        """Every character it lays out."""
        return "".join(sorted(self.advance))

    def width(self, char: str) -> float:
        """A character's inked width, template px (0 for the space)."""
        t = self.templates.get(char)
        cols = np.flatnonzero((t >= 0.5).any(axis=0)) if t is not None else []
        return float(cols[-1] - cols[0] + 1) if len(cols) else 0.0

    def layout(self, text: str) -> tuple[list[str], np.ndarray]:
        """Text's characters and their pen positions along the line, template px from the text's centre."""
        chars = list(text)
        missing = sorted({c for c in chars if c not in self.advance})
        if missing:
            raise KeyError(f"the font has no {''.join(missing)!r}")
        adv = np.array([self.advance[c] for c in chars], float)
        pens = np.concatenate([[0.0], np.cumsum(adv)])
        return chars, pens[:-1] - pens[-1] / 2

    def laid(self, text: str) -> Laid:
        """Text laid out on the line (`lay`), kept for the font's life."""
        if text not in self._laid:
            chars, xs = self.layout(text)
            self._laid[text] = lay(self, chars, xs, np.zeros(len(chars)))
        return self._laid[text]

    def save(self, path) -> None:
        """Write the font (and its noise) to an .npz file."""
        chars = self.chars
        arrays = {f"t{ord(c)}": self.templates[c] for c in self.templates}
        np.savez_compressed(path, chars=np.array([ord(c) for c in chars]),
                            origin=np.array([self.origin.get(c, np.nan) for c in chars]),
                            extent=np.array([self.extent.get(c, (np.nan, np.nan)) for c in chars]),
                            advance=np.array([self.advance[c] for c in chars]),
                            counts=np.array([self.counts.get(c, 0.0) for c in chars]),
                            meta=np.array([self.slant, self.size, self.spacing_sd, self.position_sd]),
                            noise=self.noise.p if self.noise is not None else np.zeros(0), **arrays)

    @classmethod
    def load(cls, path) -> "Font":
        """Read a font written by `save`."""
        with np.load(path) as z:
            chars = [chr(int(o)) for o in z["chars"]]
            templates = {c: z[f"t{ord(c)}"].astype(np.float32) for c in chars if f"t{ord(c)}" in z}
            slant, size, spread, position = z["meta"].tolist()
            noise = NoiseModel(z["noise"]) if len(z["noise"]) else None
            return cls(templates=templates, origin={c: float(a) for c, a in zip(chars, z["origin"]) if c in templates},
                       extent={c: (float(a), float(b)) for c, (a, b) in zip(chars, z["extent"]) if c in templates},
                       advance={c: float(a) for c, a in zip(chars, z["advance"])}, slant=slant, size=int(size),
                       counts={c: float(n) for c, n in zip(chars, z["counts"])}, spacing_sd=spread,
                       position_sd=position, noise=noise)


def signed_distance(shape: np.ndarray) -> np.ndarray:
    """The signed distance (px; + inside) of every px's centre from the edge of `shape` (bool), the edge halfway
    between an inside and an outside px; -FAR everywhere when the shape is empty."""
    if not shape.any():
        return np.full(shape.shape, -FAR, np.float32)
    u = np.ascontiguousarray(shape, np.uint8)
    inside = cv2.distanceTransform(u, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    outside = cv2.distanceTransform(1 - u, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(shape, inside - 0.5, 0.5 - outside).astype(np.float32)


def lay(font: Font, chars: list[str], xs: np.ndarray, dys: np.ndarray) -> Laid:
    """Characters laid out with their pens at `xs` along the line and `dys` down from it (template px, the upright
    frame, from the text's centre), leaned by the font's slant: their shape's signed distance on a canvas with MARGIN
    template px around the templates."""
    t = math.tan(math.radians(font.slant))
    placed = [(font.templates[c], x - font.origin[c], dy) for c, x, dy in zip(chars, xs, dys) if c in font.templates]
    if not placed:
        return Laid(np.full((2 * MARGIN + 1, 2 * MARGIN + 1), -FAR, np.float32), MARGIN, MARGIN)
    # a template px (col, row) goes to the upright (X, Y) = (left + col, row - MIDLINE + dy), leaned: X - Y t
    corners = []
    for tpl, left, dy in placed:
        for col in (0, tpl.shape[1] - 1):
            for row in (0, ROWS - 1):
                y = row - MIDLINE + dy
                corners.append((left + col - y * t, y))
    corners = np.array(corners)
    ox, oy = MARGIN - math.floor(corners[:, 0].min()), MARGIN - math.floor(corners[:, 1].min())
    w, h = math.ceil(corners[:, 0].max()) + ox + MARGIN + 1, math.ceil(corners[:, 1].max()) + oy + MARGIN + 1
    prob = np.zeros((h, w), np.float32)
    for tpl, left, dy in placed:
        m = np.array([[1.0, -t, left + (MIDLINE - dy) * t + ox], [0.0, 1.0, dy - MIDLINE + oy]])
        np.maximum(prob, cv2.warpAffine(tpl, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0), out=prob)
    return Laid(signed_distance(prob >= 0.5), float(ox), float(oy))


def _warp(laid: Laid, pose: Pose, shape: tuple[int, int]) -> np.ndarray:
    """A laid-out text's signed distance on a crop of `shape` at a pose, crop px, clipped to EDGE_PX."""
    lin = _linear(pose.scale, pose.angle, 0.0)           # the canvas is leaned already
    m = np.hstack([lin, (np.array([pose.x, pose.y]) - lin @ (laid.ox, laid.oy))[:, None]])
    d = cv2.warpAffine(laid.sdf, m, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=-FAR)
    return np.clip(d * max(pose.scale, S_MIN), -EDGE_PX, EDGE_PX)


def render(font: Font, text: str, pose: Pose, shape: tuple[int, int]) -> np.ndarray:
    """Text drawn by the font at a pose on a crop of `shape`: the signed distance from its strokes' edge, crop px
    (+ on a stroke), clipped to EDGE_PX either way (float32)."""
    return _warp(font.laid(text), pose, shape)


def _patch(laid: Laid, scale: float, angle: float) -> tuple[np.ndarray, int, int, tuple[float, float]]:
    """A laid-out text's signed distance at a scale and angle on the smallest window holding every px within EDGE_PX
    of its strokes, the window px (cx, cy) its centre falls on, and its drawn ink's box's centre less that px."""
    h, w = laid.sdf.shape
    lin = _linear(scale, angle, 0.0)
    xy = lin @ (np.array([[0, w - 1, 0, w - 1], [0, 0, h - 1, h - 1]], float) - np.array([[laid.ox], [laid.oy]]))
    cx, cy = math.ceil(-xy[0].min()), math.ceil(-xy[1].min())
    shape = (math.floor(xy[1].max()) + cy + 1, math.floor(xy[0].max()) + cx + 1)
    d = _warp(laid, Pose(cx, cy, scale, angle), shape)
    rows, cols = np.flatnonzero((d > -EDGE_PX).any(axis=1)), np.flatnonzero((d > -EDGE_PX).any(axis=0))
    if not len(rows):
        return np.full((1, 1), -EDGE_PX, np.float32), 0, 0, (0.0, 0.0)
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    ir, ic = np.flatnonzero((d > 0).any(axis=1)), np.flatnonzero((d > 0).any(axis=0))
    ink = ((ic[0] + ic[-1]) / 2 - cx, (ir[0] + ir[-1]) / 2 - cy) if len(ir) else (0.0, 0.0)
    return np.ascontiguousarray(d[r0:r1, c0:c1]), cx - int(c0), cy - int(r0), ink


# --- the first font ---------------------------------------------------------------------------------------------------

UP = 4                     # the first font is drawn this many times larger, then averaged down


def _hershey(char: str, face: int, stroke: float) -> tuple[np.ndarray, float, float]:
    """A character in an OpenCV Hershey face at cap height SIZE (its strokes' outer edges), the stroke `stroke` of
    the cap height: its coverage (ROWS high; its ink and HPAD columns each side), the column of its pen and its
    advance, template px."""
    probe = np.zeros((300, 300), np.uint8)
    cv2.putText(probe, "H", (20, 250), face, 8.0, 255, 1, cv2.LINE_8)
    rows = np.flatnonzero(probe.any(axis=1))
    cap = (rows[-1] - rows[0]) / 8.0                     # the cap height at scale 1, between the stroke centres
    thick = max(1, round(stroke * SIZE * UP))
    scale = (SIZE * UP - thick) / cap
    advance = cv2.getTextSize(char, face, scale, thick)[0][0] / UP
    big = np.zeros((ROWS * UP, 3 * SIZE * UP), np.uint8)
    pen = SIZE * UP // 2
    cv2.putText(big, char, (pen, round((VPAD + SIZE) * UP - thick / 2)), face, scale, 255, thick, cv2.LINE_AA)
    small = cv2.resize(big, (big.shape[1] // UP, ROWS), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    cols = np.flatnonzero((small >= 0.5).any(axis=0))
    if not len(cols):
        return np.zeros((ROWS, 0), np.float32), 0.0, advance
    c0, c1 = max(0, cols[0] - HPAD), min(small.shape[1], cols[-1] + 1 + HPAD)
    return small[:, c0:c1], (pen + 0.5) / UP - 0.5 - c0, advance    # big px k's centre, in small px centres


def initial_font(chars: str, slant: float = 0.0, face: int = INIT_FACE, stroke: float = INIT_STROKE) -> Font:
    """The first font (see the module): every character of `chars` (and the space) in a Hershey face (by default
    Hershey Simplex, a plain single-stroke sans), leaning `slant` degrees."""
    templates, origin, extent, advance = {}, {}, {}, {}
    for c in sorted(set(chars) | {" "}):
        cover, pen, adv = _hershey(c, face, stroke)
        advance[c] = float(adv)
        if c != " " and cover.shape[1]:
            templates[c] = (INIT_FLOOR + (1 - 2 * INIT_FLOOR) * cover).astype(np.float32)
            cols = np.flatnonzero((cover >= 0.5).any(axis=0))
            origin[c], extent[c] = float(pen), (float(cols[0]), float(cols[-1]))
    return Font(templates=templates, origin=origin, extent=extent, advance=advance, slant=float(slant))


# --- the noise --------------------------------------------------------------------------------------------------------

KNOTS = np.linspace(-EDGE_PX, EDGE_PX, round(2 * EDGE_PX / KNOT_PX) + 1)


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The weighted least-squares non-decreasing fit to y (pool adjacent violators)."""
    blocks: list[list[float]] = []      # [mean, weight, count]
    for v, wt in zip(y, w):
        blocks.append([v, wt, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, n2 = blocks.pop()
            m1, w1, n1 = blocks.pop()
            wt_ = w1 + w2
            blocks.append([(m1 * w1 + m2 * w2) / wt_ if wt_ > 0 else (m1 + m2) / 2, wt_, n1 + n2])
    return np.concatenate([np.full(n, m) for m, _, n in blocks])


@dataclass(frozen=True, eq=False)
class NoiseModel:
    """p(ink) at the KNOTS (signed distance from the rendered stroke edge, crop px); see the module."""
    p: np.ndarray

    @classmethod
    def default(cls) -> "NoiseModel":
        """The noise before any fit: a logistic from INK_P[0] off the strokes to INK_P[1] on them."""
        lo, hi = INK_P
        return cls(lo + (hi - lo) / (1 + np.exp(-KNOTS / INIT_EDGE_PX)))

    @property
    def background(self) -> float:
        """p(ink) far off any stroke."""
        return float(self.p[0])

    def p_ink(self, rendered: np.ndarray) -> np.ndarray:
        """p(ink) at each px of a rendering (`render`'s signed distance)."""
        return np.interp(rendered, KNOTS, self.p).astype(np.float32)

    def flips(self) -> tuple[np.ndarray, np.ndarray]:
        """The flip rates by distance: paper -> ink at the knots off the strokes (d < 0), ink -> paper at those on
        them (d > 0); each with its knots."""
        off, on = KNOTS < 0, KNOTS > 0
        return np.vstack([KNOTS[off], self.p[off]]), np.vstack([KNOTS[on], 1 - self.p[on]])

    def log_lik(self, rendered: np.ndarray, crop: np.ndarray, observed: np.ndarray) -> float:
        """log p(the crop's ink | the rendering) over its observed px.  `crop`: its ink (bool) or the uint8 crop."""
        ink = crop if crop.dtype == bool else crop < grid.INK_BELOW
        o = np.asarray(observed, bool)
        p = np.interp(rendered[o], KNOTS, self.p)
        return float(np.where(ink[o], np.log(p), np.log1p(-p)).sum())

    @classmethod
    def fit(cls, rendered: list[np.ndarray], crops: list[np.ndarray], observed: list[np.ndarray],
            weights=None) -> "NoiseModel":
        """The noise measured on renderings and their crops' ink (bool, or uint8 crops), observed px only: each px
        shared between its two nearest knots (linearly), p = (ink + 1/2) / (px + 1) at each knot (weighted by
        `weights`, one per crop), then made non-decreasing in the distance."""
        ink_at, n_at = np.zeros(len(KNOTS)), np.zeros(len(KNOTS))
        weights = np.ones(len(rendered)) if weights is None else np.asarray(weights, float)
        for d, crop, obs, w in zip(rendered, crops, observed, weights):
            if w <= 0:
                continue
            ink = crop if crop.dtype == bool else crop < grid.INK_BELOW
            o = np.asarray(obs, bool)
            pos = (np.clip(d[o].astype(float), -EDGE_PX, EDGE_PX) + EDGE_PX) / KNOT_PX
            k = np.minimum(np.floor(pos).astype(np.int64), len(KNOTS) - 2)
            f = pos - k
            x = ink[o].astype(float)
            for kk, ff in ((k, 1 - f), (k + 1, f)):
                n_at += w * np.bincount(kk, ff, len(KNOTS))
                ink_at += w * np.bincount(kk, ff * x, len(KNOTS))
        p = (ink_at + 0.5) / (n_at + 1.0)
        return cls(np.clip(_pav(p, n_at + 1.0), 1e-4, 1 - 1e-4))


# --- the training strings and the pose search ---------------------------------------------------------------------

def consensus(label: labels_mod.Label) -> str:
    """The string a training label is learned from: Terra's text as written when its code (`codes.canon`) has the
    most reads (`label.reads`, ties to Terra); else Sol's as written when its code has; else the read with the most
    reads (the first of a tie, in order), in its own spelling."""
    count: dict[str, int] = {}
    for v in label.reads.values():
        if v:
            count[v] = count.get(v, 0) + 1
    if not count:
        return label.terra_text or label.sol_text
    most = max(count.values())
    for text in (label.terra_text, label.sol_text):
        if text and count.get(codes.canon(text), 0) == most:
            return text
    return sorted(c for c, n in count.items() if n == most)[0]


@dataclass(frozen=True)
class Search:
    """The poses `align` tries: scales and angles (every position in the crop each), and a Gaussian prior on the
    centre of the text's drawn ink (the box of its px on a stroke)."""
    centre: tuple[float, float]
    sd: float
    scales: tuple[float, ...]
    angles: tuple[float, ...]
    unit: float = 0.0          # with a word: the scale at which the cap height is the word's height; else 0


def word_heights(labels: list[labels_mod.Label]) -> np.ndarray:
    """The heights of the labels' Terra word rectangles, px (the labels with a box)."""
    return np.array([lab.word[1] for lab in labels if lab.box is not None], float)


def search_for(label: labels_mod.Label, heights: np.ndarray, residuals=RESIDUALS, ratios=RATIOS) -> Search:
    """A label's poses to try.  With Terra's box: the cap height `ratios` x the word's height, upright or upside down
    (the crop reads along Terra's angle, which may be a half turn out), `residuals` degrees about each, the position
    prior about the word's centre with POSITION_SD word heights.  Without: cap heights over the POINT_QUANTILES of
    the training words' `heights`, x ratios, at the same steps; every quarter turn; the prior about the point."""
    if label.code_rect is not None:
        u0, v0, u1, v1 = label.code_rect
        h = v1 - v0
        scales = tuple(max(S_MIN, r * h / SIZE) for r in ratios)
        bases, centre, sd = (0.0, 180.0), ((u0 + u1 - 1) / 2, (v0 + v1 - 1) / 2), POSITION_SD * h
    else:
        lo, hi = np.quantile(heights, POINT_QUANTILES)
        step = (ratios[-1] / ratios[0]) ** (1 / max(1, len(ratios) - 1))
        n = max(1, math.ceil(math.log(hi * ratios[-1] / (lo * ratios[0])) / math.log(step))) + 1
        scales = tuple(max(S_MIN, s / SIZE) for s in np.geomspace(lo * ratios[0], hi * ratios[-1], n))
        bases, centre, sd = (0.0, 90.0, 180.0, 270.0), (label.centre[0] - 0.5, label.centre[1] - 0.5), \
            POINT_SD * float(np.median(heights))
    angles = tuple(float(b + r) for b in bases for r in residuals)
    unit = (label.code_rect[3] - label.code_rect[1]) / SIZE if label.code_rect is not None else 0.0
    return Search(centre=centre, sd=float(sd), scales=scales, angles=angles, unit=unit)


@dataclass(frozen=True)
class Fit:
    """A string's best pose on a crop: its log-likelihood (observed px), that less the bare background's, the number
    of observed px within EDGE_PX of its strokes, and the centre of its drawn ink's box (crop px, px centres; as
    Terra's word box is the box of the printed ink)."""
    pose: Pose
    loglik: float
    llr: float
    near: int
    ink_centre: tuple[float, float]


FFT_BATCH = 16             # patches correlated at once


def _best_spots(xo: np.ndarray, obs: np.ndarray, patches: list[tuple], search: "Search") -> list[tuple[float, int]]:
    """For each patch (a, b, cx, cy, ink), the crop px where its centre px (cx, cy) is most likely, with the position
    prior on its drawn ink's centre (`ink` from the centre px): the score (the log-likelihood ratio against the
    background, a summed over the observed ink under the patch and b over the observed px, plus the prior) and the
    px's flat index.  Correlations by FFT (scipy, float32), every lag of the crop px with no wrap-around."""
    h, w = xo.shape
    shape = (fft.next_fast_len(h + max(p[0].shape[0] for p in patches) - 1, real=True),
             fft.next_fast_len(w + max(p[0].shape[1] for p in patches) - 1, real=True))
    fxo, fobs = fft.rfft2(xo, shape), fft.rfft2(obs, shape)
    v, u = np.indices((h, w), dtype=np.float32)
    out = []
    for k in range(0, len(patches), FFT_BATCH):
        chunk = patches[k:k + FFT_BATCH]
        a = np.zeros((len(chunk), *shape), np.float32)
        b = np.zeros((len(chunk), *shape), np.float32)
        for n, (pa, pb, *_) in enumerate(chunk):
            a[n, :pa.shape[0], :pa.shape[1]] = pa
            b[n, :pb.shape[0], :pb.shape[1]] = pb
        corr = fft.irfft2(np.conj(fft.rfft2(a)) * fxo + np.conj(fft.rfft2(b)) * fobs, shape)
        for n, (_, _, cx, cy, (ix, iy)) in enumerate(chunk):
            prior = -((u + ix - search.centre[0]) ** 2 + (v + iy - search.centre[1]) ** 2) / (2 * search.sd ** 2)
            score = np.roll(corr[n], (cy, cx), axis=(0, 1))[:h, :w] + prior
            i = int(np.argmax(score))
            out.append((float(score.flat[i]), i))
    return out


def _ab(noise: "NoiseModel", d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The per-px weights of the likelihood ratio against the background: a on ink, b on every observed px."""
    p = noise.p_ink(d).astype(np.float64)
    q = noise.background
    b = np.log1p(-p) - math.log1p(-q)
    return (np.log(p) - math.log(q) - b).astype(np.float32), b.astype(np.float32)


def align(font: Font, noise: NoiseModel, text: str, ink: np.ndarray, observed: np.ndarray, search: Search) -> Fit:
    """A string's best pose on a crop's ink (bool), `observed` px only: the most likely of every scale and angle of
    `search` at every crop px, with the position prior on its drawn ink's centre; its likelihood then summed
    exactly."""
    h, w = ink.shape
    obs = np.asarray(observed, bool)
    xo, of = (ink & obs).astype(np.float32), obs.astype(np.float32)
    laid = font.laid(text)
    poses, patches = [], []
    for s in search.scales:
        for angle in search.angles:
            d, cx, cy, ink_at = _patch(laid, s, angle)
            patches.append((*_ab(noise, d), cx, cy, ink_at))
            poses.append((float(s), float(angle)))
    spots = _best_spots(xo, of, patches, search)
    k = max(range(len(spots)), key=lambda n: spots[n][0])        # the first of a tie
    i = spots[k][1]
    pose = Pose(float(i % w), float(i // w), *poses[k])
    rendered = render(font, text, pose, (h, w))
    full = noise.log_lik(rendered, ink, obs)
    q = noise.background
    base = float(np.where(ink[obs], math.log(q), math.log1p(-q)).sum())
    rows, cols = np.flatnonzero((rendered > 0).any(axis=1)), np.flatnonzero((rendered > 0).any(axis=0))
    centre = ((cols[0] + cols[-1]) / 2, (rows[0] + rows[-1]) / 2) if len(rows) else (pose.x, pose.y)
    return Fit(pose=pose, loglik=full, llr=full - base, near=int((obs & (rendered > -EDGE_PX)).sum()),
               ink_centre=(float(centre[0]), float(centre[1])))


# --- the M-step -------------------------------------------------------------------------------------------------------

def refine(font: Font, noise: NoiseModel, text: str, fit: Fit, ink: np.ndarray, observed: np.ndarray
           ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Each character's own best spot near where the string's pose lays it: the most likely shift of the character
    alone within SHIFT of the cap height (crop px, a Gaussian prior with half that sd), as its pen along the line
    and its drop below it (template px, upright frame).  The other characters' strokes (as laid out) are unobserved
    while one is moved, so a neighbour's ink neither pulls it nor pushes it away.  Returns the characters, pens and
    drops."""
    chars, xs = font.layout(text)
    laid_xs = xs.astype(float)
    xs, dys = laid_xs.copy(), np.zeros(len(chars))
    pose = fit.pose
    lin = _linear(pose.scale, pose.angle, font.slant)
    obs = np.asarray(observed, bool)
    reach = max(2, round(SHIFT * font.size * max(pose.scale, S_MIN)))
    sd = reach / 2
    dv, du = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    prior = (-(du ** 2 + dv ** 2) / (2 * sd ** 2)).astype(np.float32)
    for i, c in enumerate(chars):
        if c not in font.templates:
            continue
        others = [k for k in range(len(chars)) if k != i]
        context = _warp(lay(font, [chars[k] for k in others], laid_xs[others], np.zeros(len(others))), pose, ink.shape)
        mine = obs & (context <= 0)
        xo, of = (ink & mine).astype(np.float32), mine.astype(np.float32)
        ex, ey = np.array([pose.x, pose.y]) + lin @ (xs[i] + font.advance[c] / 2, 0.0)   # the character alone's centre
        d, cx, cy, _ = _patch(font.laid(c), pose.scale, pose.angle)
        a, b = _ab(noise, d)
        ph, pw = d.shape
        x0, y0 = round(ex) - cx - reach, round(ey) - cy - reach
        win = [_window(arr, x0, y0, pw + 2 * reach, ph + 2 * reach) for arr in (xo, of)]
        score = cv2.matchTemplate(win[0], a, cv2.TM_CCORR) + cv2.matchTemplate(win[1], b, cv2.TM_CCORR) + prior
        j = int(np.argmax(score))
        moved = np.array([round(ex) + j % score.shape[1] - reach - ex, round(ey) + j // score.shape[1] - reach - ey])
        dx, dy = np.linalg.solve(lin, moved)
        xs[i] += dx
        dys[i] = dy
    return chars, xs, dys


def _window(arr: np.ndarray, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """arr[y0:y0+h, x0:x0+w], zero beyond its edges."""
    out = np.zeros((h, w), arr.dtype)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(arr.shape[1], x0 + w), min(arr.shape[0], y0 + h)
    if sx0 < sx1 and sy0 < sy1:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out


def _gather(font: Font, pose: Pose, chars: list[str], xs: np.ndarray, dys: np.ndarray, ink: np.ndarray,
            observed: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The crop's observed ink and observed px under each character's template with its pen at `xs` (bilinear), the
    px owned by the character only (those on its side of the middles of the gaps between its `extent` and its inked
    neighbours', upright frame): {char: (ink, px)} summed over its instances."""
    lin = _linear(pose.scale, pose.angle, font.slant)
    obs = np.asarray(observed, bool)
    xo, of = (ink & obs).astype(np.float32), obs.astype(np.float32)
    inked = [i for i, c in enumerate(chars) if c in font.templates]
    ends = {i: (xs[i] - font.origin[chars[i]] + font.extent[chars[i]][0],
                xs[i] - font.origin[chars[i]] + font.extent[chars[i]][1]) for i in inked}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for n, i in enumerate(inked):
        c = chars[i]
        tpl = font.templates[c]
        left = xs[i] - font.origin[c]
        m = np.hstack([lin, (np.array([pose.x, pose.y]) + lin @ (left, dys[i] - MIDLINE))[:, None]])
        size = (tpl.shape[1], ROWS)
        num = cv2.warpAffine(xo, m, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        den = cv2.warpAffine(of, m, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        lo = (ends[inked[n - 1]][1] + ends[i][0]) / 2 if n > 0 else -np.inf
        hi = (ends[i][1] + ends[inked[n + 1]][0]) / 2 if n + 1 < len(inked) else np.inf
        x = left + np.arange(tpl.shape[1])
        own = ((x >= lo) & (x < hi)).astype(np.float32)
        num, den = num * own, den * own
        if c in out:
            out[c] = (out[c][0] + num, out[c][1] + den)
        else:
            out[c] = (num, den)
    return out


def _advances(font: Font, placed: list[tuple[list[str], np.ndarray]], weights: np.ndarray
              ) -> tuple[dict[str, float], float]:
    """The advances best fitting the gaps between consecutive inked characters' pens (a gap is the first one's advance
    and the spaces' between), weighted least squares with a ridge of PRIOR_WEIGHT towards the font's; and the gaps'
    spread about the fit, template px."""
    order = sorted(font.advance)
    at = {c: k for k, c in enumerate(order)}
    rows, gaps, ws = [], [], []
    for (chars, xs), w in zip(placed, weights):
        inked = [i for i, c in enumerate(chars) if c in font.templates]
        for i, j in zip(inked, inked[1:]):
            r = np.zeros(len(order))
            for k in range(i, j):
                r[at[chars[k]]] += 1.0
            rows.append(r)
            gaps.append(xs[j] - xs[i])
            ws.append(w)
    old = np.array([font.advance[c] for c in order])
    if not rows:
        return dict(font.advance), font.spacing_sd
    a, g, w = np.array(rows), np.array(gaps), np.array(ws)
    lhs = (a * w[:, None]).T @ a + PRIOR_WEIGHT * np.eye(len(order))
    rhs = (a * w[:, None]).T @ g + PRIOR_WEIGHT * old
    adv = np.linalg.solve(lhs, rhs)
    spread = math.sqrt(float((w * (g - a @ adv) ** 2).sum() / max(w.sum(), 1e-9)))
    return {c: float(v) for c, v in zip(order, adv)}, spread


# --- the mislabelled escape -------------------------------------------------------------------------------------------

def label_weights(z: np.ndarray, area: np.ndarray, step=None, rounds: int = 200) -> tuple[np.ndarray, dict]:
    """Each label's share of "good" (see the module).  `z`: (n, d), where (and how large) its string was found,
    against what the label says (`_offsets`); `area`: the volume of the poses tried in those units; `step`: the pose
    grid's step on each axis (d,), whose rounding (sd step / sqrt 12) no sd goes below.  "Good": a Gaussian (a mean
    and an sd per axis); "bad": any pose tried alike.  Fitted by EM, the bad share at most a half (so "good" is the
    bulk).  Returns the shares and the fitted parameters."""
    z = np.asarray(z, float)
    z = z.reshape(len(z), -1)
    floor = np.maximum(1e-3, np.asarray(step if step is not None else np.zeros(z.shape[1]), float) / math.sqrt(12))
    log_bad = -np.log(np.asarray(area, float))
    mu = np.median(z, axis=0)
    sd = np.maximum(floor, 1.4826 * np.median(np.abs(z - mu), axis=0))
    pi = 0.05
    r = np.ones(len(z))
    for _ in range(rounds):
        lg = (math.log(1 - pi) - z.shape[1] / 2 * math.log(2 * math.pi) - np.log(sd).sum()
              - (((z - mu) / sd) ** 2).sum(axis=1) / 2)
        lb = math.log(pi) + log_bad
        r = np.exp(lg - np.logaddexp(lg, lb))
        pi = min(0.5, max(1e-6, 1 - float(r.mean())))
        mu = (r[:, None] * z).sum(axis=0) / max(r.sum(), 1e-12)
        sd = np.maximum(floor, np.sqrt((r[:, None] * (z - mu) ** 2).sum(axis=0) / max(r.sum(), 1e-12)))
    return r, {"good_mean": mu.tolist(), "good_sd": sd.tolist(), "bad_share": pi}


def _offsets(fits: list[Fit], searches: list[Search], shapes: list[tuple[int, int]]
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each fit's drawn ink's centre less its search's centre (Terra's word's, or the point), in the position prior's
    sds, and with a word (`Search.unit`) the log of its cap height over the word's height; the volume of the poses
    tried in those units (the crop's area, times the log of the scales' range); and the grid's step on each axis (a
    px; the scales' log step), the median over the labels."""
    z, area, step = [], [], []
    for f, s, (h, w) in zip(fits, searches, shapes):
        z.append([(f.ink_centre[0] - s.centre[0]) / s.sd, (f.ink_centre[1] - s.centre[1]) / s.sd])
        area.append(h * w / s.sd ** 2)
        step.append([1 / s.sd, 1 / s.sd])
        if s.unit:
            scales = sorted(s.scales)
            z[-1].append(math.log(f.pose.scale / s.unit))
            area[-1] *= max(math.log(scales[-1] / scales[0]), 1e-3)
            step[-1].append(math.log(scales[1] / scales[0]) if len(scales) > 1 else 0.0)
    z = np.array(z, float).reshape(len(z), -1)
    return z, np.array(area, float), np.median(np.array(step, float).reshape(len(z), -1), axis=0)


# --- learning ---------------------------------------------------------------------------------------------------------

def _map(fn, n: int, workers: int | None) -> list:
    """fn(0..n-1) on threads, in order."""
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 1) as ex:
        return list(ex.map(fn, range(n)))


def learn_font(labels: list[labels_mod.Label], iterations: int = ITERATIONS, *, texts: list[str] | None = None,
               observed: list[np.ndarray] | None = None, charset: str = "", rules=RULES, residuals=RESIDUALS,
               ratios=RATIOS, workers: int | None = None) -> Font:
    """The font learned from the labels' crops by hard EM (see the module).  `texts`: each label's string
    (`consensus` when None); `observed`: each crop's observed px (not on a line by `rules` when None); `charset`:
    characters the font holds besides the strings' (kept at the first font where never seen).  The font's `noise` is
    the noise fitted with it; its `trace` holds each round's training log-likelihood."""
    t0 = time.perf_counter()
    texts = [consensus(lab) for lab in labels] if texts is None else list(texts)
    observed = [~labels_mod.line_mask(lab, rules=rules) for lab in labels] if observed is None else \
        [np.asarray(o, bool) for o in observed]
    inks = [lab.ink for lab in labels]
    heights = word_heights(labels)
    if not len(heights):
        heights = np.array([SIZE], float)
    searches = [search_for(lab, heights, residuals, ratios) for lab in labels]
    chars = "".join(sorted(set(charset) | set("".join(texts))))
    noise = NoiseModel.default()

    def e_step(font: Font, noise: NoiseModel, which: list[int]) -> list[Fit]:
        return _map(lambda k: align(font, noise, texts[which[k]], inks[which[k]], observed[which[k]],
                                    searches[which[k]]), len(which), workers)

    every = list(range(len(labels)))
    tried = {s: sum(f.loglik for f in e_step(initial_font(chars, s), noise, every[::SLANT_EVERY]))
             for s in INIT_SLANTS}
    slant = max(INIT_SLANTS, key=lambda s: (tried[s], -s))
    first = initial_font(chars, slant)
    font = first
    font.noise = noise
    log.info("slant %.0f deg (log-likelihood of every %dth label by slant: %s)", slant, SLANT_EVERY,
             ", ".join(f"{s:.0f}: {v:.0f}" for s, v in tried.items()))
    trace = [{"round": "slant", "tried": tried, "slant": slant, "seconds": time.perf_counter() - t0}]
    best: tuple[float, int, Font] = (-np.inf, 0, font)
    groups = {"box": [i for i in every if labels[i].code_rect is not None],
              "point": [i for i in every if labels[i].code_rect is None]}
    for k in range(iterations + 1):
        t1 = time.perf_counter()
        fits = e_step(font, noise, every)
        weights, mix = np.ones(len(labels)), {}
        for name, group in groups.items():
            if len(group) > 1:
                w, mix[name] = label_weights(*_offsets([fits[i] for i in group], [searches[i] for i in group],
                                                       [inks[i].shape for i in group]))
                weights[group] = w
        if k == 0 and "box" in mix:
            # Terra's words sit this close to the drawn text when the string was found at its word: the position
            # prior from here on (the round's broad prior lets a string go to larger ink elsewhere, which pays more)
            tight = max(mix["box"]["good_sd"][:2])
            for i in groups["box"]:
                s = searches[i]
                searches[i] = Search(s.centre, tight * s.sd, s.scales, s.angles, s.unit)
            mix["position_sd"] = tight * POSITION_SD
        trace.append({"round": k, "loglik": sum(f.loglik for f in fits), "llr": sum(f.llr for f in fits),
                      "weighted_loglik": float(sum(w * f.loglik for w, f in zip(weights, fits))),
                      "kept": float(weights.sum()), "labels": len(fits), **mix,
                      "upside_down": sum(abs(((f.pose.angle + 45) % 360) - 225) < 45 for f in fits),
                      "seconds": time.perf_counter() - t1})
        log.info("round %d: log-likelihood %.0f (llr %.0f), labels kept %.1f of %d, %.0f s", k, trace[-1]["loglik"],
                 trace[-1]["llr"], trace[-1]["kept"], len(fits), trace[-1]["seconds"])
        if k > 0 and trace[-1]["loglik"] > best[0]:       # round 0 is the first font, under the broad prior
            best = (trace[-1]["loglik"], k, font)
        if k == iterations:
            break
        placed = _map(lambda i: refine(font, noise, texts[i], fits[i], inks[i], observed[i]), len(labels), workers)
        gathered = _map(lambda i: _gather(font, fits[i].pose, *placed[i], inks[i], observed[i]), len(labels),
                        workers)
        num = {c: np.zeros_like(t, np.float64) for c, t in font.templates.items()}
        den = {c: np.zeros_like(t, np.float64) for c, t in font.templates.items()}
        for g, w in zip(gathered, weights):
            for c, (a, b) in g.items():
                num[c] += w * a
                den[c] += w * b
        templates = {c: ((num[c] + PRIOR_WEIGHT * first.templates[c]) / (den[c] + PRIOR_WEIGHT)).astype(np.float32)
                     for c in font.templates}
        advance, spread = _advances(font, [(ch, xs) for ch, xs, _ in placed], weights)
        counts = {c: float(den[c].max()) if c in den else 0.0 for c in font.advance}
        font = Font(templates=templates, origin=dict(font.origin), extent=dict(font.extent), advance=advance,
                    slant=font.slant, counts=counts, spacing_sd=spread)
        # the new layout moves the text's centre: each label moved to its best spot at its round's scale and angle
        moved = _map(lambda i: align(font, noise, texts[i], inks[i], observed[i], Search(
            searches[i].centre, searches[i].sd, (fits[i].pose.scale,), (fits[i].pose.angle,))), len(labels), workers)
        rendered = _map(lambda i: render(font, texts[i], moved[i].pose, inks[i].shape), len(labels), workers)
        noise = NoiseModel.fit(rendered, inks, observed, weights)
        font.noise = noise
    font.noise = noise
    if iterations > 0:
        # hard EM's rounds wander a little once it has settled: the round whose font explained the training ink best
        _, k, font = best
        trace.append({"round": "chosen", "chosen": k})
        log.info("the font of round %d chosen", k)
    font.trace = trace
    font.position_sd = next((r["position_sd"] for r in trace if "position_sd" in r), POSITION_SD)
    log.info("font learned in %.0f s", time.perf_counter() - t0)
    return font
