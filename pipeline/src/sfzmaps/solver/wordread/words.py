"""Word boxes from the detector's word heads (`textnet.train.Recipe.heads="words"`, `textnet.model`): every px votes
for its word's box and carries its word's tag, weighted by its core chance, and the words are the components of a
mixture over those votes with a spatial prior.  No level on the core chance, no merge IoU, no shrink ratio, no window
that decides, and no rule drops a box: each keeps its chance and its log-odds of being a word.

`core_words(core_logit, geo, tag, ink, spreads, prior)` works on one page's arrays (the net's outputs over a page,
tiles stitched, are Task 8's):
  core_logit  float (h, w): the CORE logit of each px (its chance of being in a word, the word's whole labelled
              rectangle, `shrink.prepare`, is sigmoid(core_logit); read in log space, so no chance is 0 or 1)
  geo         float (5, h, w): the word heads' geometry at each px as `textnet.model.word_geometry` decodes it: the
              offset in px from the px to its word's centre (x right, y down), the word's length and height in px,
              and its axis in degrees in [0, 180) (counter-clockwise from left-to-right)
  tag         float (k, h, w): each px's word tag (TAG)
  ink         bool (h, w): the page's ink
A vote has no reading sign: which way a word reads is the reader's (its two readings, its classifier).

The model (one likelihood for every decision).  Every px casts a vote: the centre p + H (dx, dy) it points to (H its
height), its log length, log height and axis, and its tag.
  - A word is a centre c, a log length and a log height (a box at least a px each way: a smaller one would be a
    single px's own square, which a dead word in a fit shrinks towards forever), an axis theta (modulo 180 degrees)
    and a mean tag.  Where its px lie: in its box, each edge uncertain by `spreads.edge` px (measured), so a px at p
    is in word k's box with chance q_k(p): its box grown by a px's half width (a px is a unit square at its numbers,
    so one centred on the box's edge is in it), convolved with that Gaussian (1 inside, a half on that edge, falling
    outside), taken out to the page's reach and 0 beyond: the word's support.  The reach is where q times the most
    any vote of the page can favour a word over the background (K: its core evidence and its vote's density at a
    word's peak over the background's) falls below double precision, so what the cut leaves out is below it.
  - The core head's output is evidence of where boxes are: w = sigmoid(core_logit) is its chance that the px is in
    a word's box, at a base rate of `spreads.core_share` (measured), so the px's output is w / core_share as likely
    in a box as on average, and (1 - w) / (1 - core_share) out of every box.  A px is in some word's box with chance
    C(p) = 1 - prod_k (1 - q_k(p)), word k's among them by its weight's share a_k q_k(p) / sum_j a_j q_j(p), its vote
    then the word's; out of every box with chance 1 - C(p), its vote then the background's.  Each word's weight a_k is
    fitted (where boxes overlap, as a code printed on a street name, the votes say whose the px are).  So a box
    covering px the head calls background pays for each (a long length vote cannot stretch a box over them), far from
    every word a px is the background's, and a copy of a word gains nothing from the shares (the copies' weights
    split its own) and next to nothing from C (no higher where q is 1).
  - A word's vote, each dimension calibrated against Terra's words (measured: pred = alpha + beta true + error): its
    offset (dx, dy) in heights is alpha + beta (c - p) / H (H the word's height; alpha `spreads.centre_bias`, beta
    `spreads.centre_slope`, each way), its log length and log height alpha + beta times the word's (`length_bias`,
    `length_slope`; `height_bias`, `height_slope`), its axis the word's + `angle_bias` (modulo 180 degrees), its tag
    the word's mean tag; each plus the word's own shared error u (one per word and dimension, Gaussian about 0 of sd
    `spreads.centre_word`, `length_word`, `height_word`, `angle_word`: the votes of one word share part of their
    error) and the vote's own error, a Student-t of scale `spreads.centre`, `length`, `height`, `angle`, `tag` and
    degrees of freedom `spreads.centre_dof`, ... (heavy tails, measured; a Gaussian at infinite dof).  So the
    likelihood reads each vote as the calibrated vote (pred - alpha) / beta at scale s / |beta|, and a dimension the
    net does not know (beta near 0) carries next to no information, by its measurement, with no rule.  A word's u is
    fitted with it, its prior counted once per word (cell times in the votes' units: see below).
  - The background's vote: its offset Gaussian about 0 of sd `spreads.background_centre` each way, its log length and
    log height Gaussian of mean and sd `spreads.background_length` and `spreads.background_height`, its axis any, its
    tag Gaussian about `spreads.background_tag_mean` of sd `spreads.background_tag` (the votes as the net gives them).
  - Neighbouring px's votes are not independent (the net's outputs are smooth): n votes carry about n / `spreads.cell`
    independent votes' evidence, so the log likelihood is divided by `cell`.
  - The score of a set of words: prior.slope x (the log likelihood / cell - sum over words of u^2 / 2 sd^2) +
    prior.log_prior for each word; the slope and the log prior fitted on the decisions this module makes
    (`prior_decisions`, `fit_word_prior`).  (u is fitted, its prior's log ratio to u = 0 counted, no Occam factor: a
    word's best u gains at most about half a log of (1 + sd^2 x its votes' precision) a dimension, which the fitted
    logit takes up.)
  The score is a sum over px of log L_p - log b_p (b_p the background's density of the px's vote), which is 0 off
  every word's support; so each move's change is exact on the union of the supports of the words it changes.
All the spreads are measured on the training quarters' predictions against Terra's words (`measure_spreads`); Task 8
runs it and the prior's fit.

The fit (tractable; every decision by the score):
  1. Proposals: the basins of the votes' centres.  Each px adds its core chance at the centre its vote places (the
     point placement: the inverse calibration a + b pred, the true value's mean given the vote, from the forward one
     and the true values' spread over word px, `spreads.centre_truth`; at b near 0 a px places its word's centre about
     itself) at the page's own px; the sum is smoothed by a Gaussian of that placement's spread (in px at the votes'
     core-weighted median height);
     each px of the page climbs to its highest 8-neighbour until none is higher (steepest ascent), and the px whose
     votes land in one basin are one proposal, fitted as one word on its own votes (EM per basin, all basins at once).
     A basin is proposed as a word when its own votes alone could pay for one (its own-vote gain x slope / cell + log
     prior > 0): the search's candidates, not a decision; a basin's votes that cannot are left to the words around
     them.
  2. The search, from no words, by moves each scored as the exact change of the total score (the changed words
     refitted with every other word held): add a proposal's word, split a word (the two-word fit started at the word and
     at the vote of its support most left to the background by core chance), or join two words whose supports overlap
     (the one-word fit started at each and the better kept). The move raising the score most is made, and only one that
     raises it, so the search ends.
  3. Every connected set of words whose supports overlap (sets that do not overlap do not interact: the likelihood
     factorises over them) is refitted jointly.
  Each fit step: an E-step gives each word's votes' own update (each dimension's Gauss-Newton step on the votes'
  errors weighted by the t's weights, (dof + 1) / (dof + z^2): the axis, the log length, then the centre and log
  height alternately; the tag their weighted mean); each tag goes to its update at once, and each word's shared
  errors u to theirs given its box (both exact CM steps of the t as a scale mixture of Gaussians); the words' weights
  take a Newton step on the likelihood (exact gradient and Hessian); then each
  word in turn takes a Newton step on the likelihood in its box's parameters (the gradient exact, the Hessian by its
  finite differences, the Hessian's eigenvalues taken as their size and no less than the votes' curvature, within a
  trust region: this sees the spatial terms the votes' update does not, such as the core px ending where the word
  does against long length votes) and then moves its box towards its votes' own update.  Each move goes to the
  first point of its line, its end, half way, a quarter, ..., that raises the log likelihood of the whole model
  (every other word held, the spatial terms included), none if none does; so no step lowers the likelihood itself
  (the expected complete log likelihood is no guide to the boxes: a vote at a support's edge, given a tiny chance of
  being the word's, makes it -inf as soon as the box turns off it), and a fall beyond float noise (LL_NOISE) raises,
  as a bug.

The table (`wordboxes.COLUMNS`, the same meaning, so the reader reads these boxes unchanged; then EXTRA_COLUMNS):
  g corners       the word's box, its first edge along its axis taken in (-90, 90] degrees (the reader reads it both
                  ways), the second across it; clipped to the page (a box running off the page keeps its axes and
                  becomes the extent of its part on the page: never wrapped, never dropped)
  t corners       the trimmed box: the box's axes, the extent of the ink px inside it (each a px wide; their centres at
                  (col + 0.5, row + 0.5)), within the box; the box itself when it holds no ink
  axis            the first edge's direction, degrees counter-clockwise from left-to-right, modulo 180
  length, height  the box's sides along and across it; t_length, t_height the trimmed box's
  page_x, page_y  the box's centre
  ink_px          the ink px inside the box; p_<kind> their mean kind probabilities (`kind_p`, else 0)
  mean_chance     as p_word
  sheet, i        the sheet given; the box's number on the page (the words in the order of their first support px,
                  row by row)
  level           NaN: there is no level
  quarter, x, y, lab_lot, street   not placed: -1, NaN, NaN, -1, False.  Task 8 places the boxes as
                  `wordboxes.sheet_boxes` does (`prints._place` of page_x, page_y; `grid.quarter_of`) before reading.
  p_word          the word's share of the weighted vote mass: its votes (each vote's chance of being its) weighted by
                  core chance, over its votes
  core_mass       its votes weighted by core chance (the numerator of p_word)
  tag_spread      the spread of its votes' tags about its mean tag (root mean square per channel, votes weighted)
  word_logodds    the model's log-odds that the box is a word: prior.slope x (the log likelihood with it less without
                  it, the other words held, their shares renormalised) / cell + prior.log_prior

Coordinates: a px stands at its numbers (col, row), as the targets measure from it, so the boxes come back in the
frame of Terra's rectangles, as `wordboxes`' boxes and Terra's words are compared.

Fixed numbers here are numerical only: EM_STEPS and EM_TOL (when an EM stops), LL_NOISE (the float noise a
likelihood may fall by), FIT_STEPS (the prior's Newton steps), SPREAD_ROUNDS and SPREAD_TOL (the spreads' fixed
point), SHAPE_STEPS (the centre and height's alternations in a word's own update), LINE_TOL (the shortest share of a
step tried), FD_STEP (the gradient's finite differences' step), TRUST (a Newton step's trust region: the steps
repeat, so where they end does not depend on it), the weights' Newton step's numerical 0 (eigenvalues below
sqrt(eps) of the largest), CHUNK_PX (votes per pass: memory, the results do not depend on it), INDEX_CELL (the grid
words and moves are found by: speed, the results do not depend on it), PRECISION (2^-53:
the reach, where what a support leaves out falls below double precision), the variogram's scale bounds (1/100 px to
the longest lag), DOF_BOUNDS (the t's degrees of freedom searched between them, a reach past any tail measured: a
fit at a bound is reported, not taken), the edge's search bounds (1/1000 px to the longest box side), the lattice
sum's span (to underflow), RECT_FLOOR (a degenerate rectangle's side, to keep its log finite) and the Gaussian
smoothing's extent (OpenCV's: 4 sd).
"""
import dataclasses
import math

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from scipy.special import gammaln, log_ndtr, ndtr, ndtri_exp

from sfzmaps.solver import wordboxes

EXTRA_COLUMNS = ("p_word", "core_mass", "tag_spread", "word_logodds")
EM_STEPS = 200        # numerical only: an EM stops after this many steps ...
EM_TOL = 1e-7         # ... or once its log likelihood rises by less than this per vote
LL_NOISE = 1e-9       # numerical only: a likelihood falling by more than this share of itself is a bug
FIT_STEPS = 100       # numerical only: the prior's fit takes at most this many Newton steps
SPREAD_ROUNDS = 200   # numerical only: the spreads' fixed point raises if it has not settled after this many rounds,
SPREAD_TOL = 1e-6     # ... settled once no spread moves by this share of itself
CHUNK_PX = 1 << 20    # votes per pass (memory only: the results do not depend on it)
PRECISION = 2.0 ** -53   # double precision: a support leaves out less than this of any vote's likelihood
DOF_BOUNDS = (1e-2, 1e8)   # numerical only: the t's degrees of freedom are searched between these
RECT_FLOOR = 1e-12    # numerical only: a degenerate rectangle's side, so its log stays finite
INDEX_CELL = 256      # px: the cells words and moves are found by (speed only: the results do not depend on it)
SHAPE_STEPS = 3      # numerical only: the centre and log height's alternations in the votes' own update
LINE_TOL = 1e-6       # numerical only: a step is halved down to this share of itself
FD_STEP = float(np.sqrt(np.finfo(np.float64).eps))   # numerical only: the gradient's differences' step
TRUST = 1.0           # numerical only: a Newton step moves at most this many box lengths, log sizes and radians
_LOG2PI = math.log(2 * math.pi)
_LOG2 = math.log(2.0)
_DIMS = ("centre", "length", "height", "angle", "tag")
MEASURED: dict = {}   # the last `measure_spreads`' counts: words measured, words left out of the edge's measure


@dataclasses.dataclass(frozen=True)
class Spreads:
    """The measured spreads of the model (see the module; `measure_spreads`).  A vote's dimension is calibrated
    against Terra's words, pred = alpha (the *_bias) + beta (the *_slope) x true + the word's shared error (sd the
    *_word) + the vote's own error (a Student-t of scale the field itself and dof the *_dof); the defaults are a vote
    that is its word's value (slope 1, no shared error) with Gaussian errors, and no knowledge of the true values'
    spread (the point placements then the calibrated vote itself)."""
    centre: float               # heights: the t scale of a word vote's offset (x and y) about its calibrated value
    length: float               # the t scale of a word vote's log length about its calibrated value
    height: float               # the t scale of a word vote's log height about its calibrated value
    angle: float                # degrees: the t scale of a word vote's axis about its word's + angle_bias (mod 180)
    tag: float                  # the t scale of a word px's tag channel about its word's mean tag
    cell: float                 # px: n votes carry n / cell independent votes' evidence
    edge: float                 # px: sd of where a word's core px end about its box's edges (the spatial prior)
    core_share: float           # the share of px in words: the core head's base rate (its evidence is w / core_share)
    centre_bias: tuple          # heights: alpha of the offset (x, y)
    length_bias: float          # alpha of the log length
    height_bias: float          # alpha of the log height
    angle_bias: float           # degrees: the votes' mean axis error (modulo 180) against Terra's words
    background_centre: float    # heights: sd of a background vote's offset about 0, each way
    background_length: tuple    # (mean, sd) of a background vote's log length
    background_height: tuple    # (mean, sd) of a background vote's log height
    background_tag: float       # sd of a background px's tag channel about its mean
    background_tag_mean: tuple  # a background px's mean tag, one per channel
    centre_slope: tuple = (1.0, 1.0)    # beta of the offset (x, y)
    length_slope: float = 1.0           # beta of the log length
    height_slope: float = 1.0           # beta of the log height
    centre_dof: float = math.inf        # the t's degrees of freedom of each dimension (inf: a Gaussian)
    length_dof: float = math.inf
    height_dof: float = math.inf
    angle_dof: float = math.inf
    tag_dof: float = math.inf
    centre_word: float = 0.0            # heights: sd of a word's shared offset error (x and y; 0: none)
    length_word: float = 0.0            # sd of a word's shared log length error
    height_word: float = 0.0            # sd of a word's shared log height error
    angle_word: float = 0.0             # degrees: sd of a word's shared axis error
    centre_truth: tuple = ((0.0, math.inf), (0.0, math.inf))   # the true offsets' (x, y) mean and sd over word px
    length_truth: tuple = (0.0, math.inf)                      # the true log lengths' mean and sd over word px
    height_truth: tuple = (0.0, math.inf)                      # the true log heights' mean and sd over word px

    def check(self, channels: int | None = None) -> None:
        sds = {k: getattr(self, k) for k in ("centre", "length", "height", "angle", "tag", "cell", "edge",
                                             "background_centre", "background_tag")}
        sds["background_length sd"], sds["background_height sd"] = self.background_length[1], self.background_height[1]
        bad = {k: v for k, v in sds.items() if not (v > 0 and math.isfinite(v))}
        if bad:
            raise ValueError(f"every spread is a positive number, not {bad}")
        if not 0 < self.core_share < 1:
            raise ValueError(f"core_share is a share strictly between 0 and 1, not {self.core_share}")
        biases = (*self.centre_bias, self.length_bias, self.height_bias, self.angle_bias, *self.centre_slope,
                  self.length_slope, self.height_slope)
        if len(self.centre_bias) != 2 or len(self.centre_slope) != 2 or not all(math.isfinite(b) for b in biases):
            raise ValueError(f"the calibrations are finite numbers (centre's two each), not {biases}")
        dofs = {k: getattr(self, f"{k}_dof") for k in _DIMS}
        if not all(d > 0 for d in dofs.values()):
            raise ValueError(f"every dof is positive (inf: a Gaussian), not {dofs}")
        shared = {k: getattr(self, f"{k}_word") for k in _DIMS[:4]}
        if not all(s >= 0 and math.isfinite(s) for s in shared.values()):
            raise ValueError(f"every shared error's sd is a finite number >= 0, not {shared}")
        truths = (*self.centre_truth, self.length_truth, self.height_truth)
        if len(self.centre_truth) != 2 or not all(math.isfinite(m) and s > 0 for m, s in truths):
            raise ValueError(f"every truth is a finite mean and a positive sd (inf: unknown), not {truths}")
        slopes = (*self.centre_slope, self.length_slope, self.height_slope)
        if any(beta == 0 and math.isinf(sd) for beta, (_, sd) in zip(slopes, truths)):
            raise ValueError("a slope of 0 places nothing from a vote: its truth's sd is needed (the placement's)")
        if channels is not None and len(self.background_tag_mean) != channels:
            raise ValueError(f"background_tag_mean has {len(self.background_tag_mean)} channels, the tags {channels}")


@dataclasses.dataclass(frozen=True)
class WordPrior:
    """The fitted decision to have a word (`fit_word_prior`): its log-odds are slope x gain + log_prior, the gain the
    word's log likelihood ratio divided by cell."""
    slope: float
    log_prior: float

    def logodds(self, gain):
        return self.slope * np.asarray(gain, np.float64) + self.log_prior


# --- votes --------------------------------------------------------------------------------------------------------

def _half(x):
    """Axis differences (radians) into [-pi/2, pi/2): modulo a half turn."""
    return (np.asarray(x) + math.pi / 2) % math.pi - math.pi / 2


def _axes(theta):
    """An axis's directions in page px (y down): along (cos, -sin) and across (-sin, -cos); each (..., 2)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c, -s], -1), np.stack([-s, -c], -1)


def _box(prm: np.ndarray, grow: float = 0.0) -> np.ndarray:
    """A word's rectangle (4, 2) from its parameters (cx, cy, log length, log height, theta), each edge moved out by
    `grow`: corners in `boxes.word_rect`'s order (start-top, end-top, end-foot, start-foot), the axis taken in
    (-90, 90] degrees."""
    theta = -_half(-prm[4])                           # in (-pi/2, pi/2]
    v, up = _axes(theta)
    a, b = math.exp(prm[2]) / 2 + grow, math.exp(prm[3]) / 2 + grow
    c = prm[:2]
    return np.array([c - v * a + up * b, c + v * a + up * b, c + v * a - up * b, c - v * a - up * b])


def _sub(v: dict, sel) -> dict:
    return {key: a[sel] for key, a in v.items()}


def _t_log_norm(dof: float) -> float:
    """log of a unit-scale Student-t's density at 0 (the Gaussian's at dof inf)."""
    if math.isinf(dof):
        return -0.5 * _LOG2PI
    return float(gammaln((dof + 1) / 2) - gammaln(dof / 2) - 0.5 * math.log(dof * math.pi))


def _t_rho(z2: np.ndarray, dof: float) -> np.ndarray:
    """minus a unit-scale t's log density at z, less its value at 0: (dof + 1) / 2 log(1 + z^2 / dof) (z^2 / 2 at dof
    inf), from z^2."""
    return 0.5 * z2 if math.isinf(dof) else (dof + 1) / 2 * np.log1p(z2 / dof)


def _t_weight(z2: np.ndarray, dof: float) -> np.ndarray:
    """A t's weight at z: (dof + 1) / (dof + z^2), so d rho / dz = weight z (1 at dof inf), from z^2."""
    return np.ones_like(z2) if math.isinf(dof) else (dof + 1) / (dof + z2)


def _placement(alpha: float, beta: float, scale: float, truth: tuple) -> tuple[float, float, float]:
    """A dimension's point placement from one vote: the inverse calibration true ~ a + b pred and its sd, from the
    forward one (pred = alpha + beta true + an error of sd `scale`) and the true values' mean and sd (`truth`): the
    true value's mean given the vote (Gaussian); with no known spread of the true values (sd inf), the calibrated vote
    (pred - alpha) / beta itself."""
    mean, sd = truth
    if math.isinf(sd):
        return -alpha / beta, 1 / beta, scale / abs(beta)
    v = sd * sd
    b = v * beta / (v * beta * beta + scale * scale)
    return mean - b * (alpha + beta * mean), b, math.sqrt(v * scale * scale / (v * beta * beta + scale * scale))


def _log_interval(x: np.ndarray, lo, hi, s: float) -> np.ndarray:
    """log(Phi((x - lo) / s) - Phi((x - hi) / s)) for lo <= hi: the log chance a point at x lies in [lo, hi] when the
    ends are uncertain by sd s, computed in the tail it is in."""
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    u0, u1 = (x - lo) / s, (x - hi) / s
    out = np.empty(np.shape(u0))
    right, left = u1 > 0, u0 < 0
    mid = ~(right | left)
    with np.errstate(divide="ignore"):
        if right.any():
            a, b = log_ndtr(-u1[right]), log_ndtr(-u0[right])
            out[right] = a + np.log1p(-np.exp(b - a))
        if left.any():
            a, b = log_ndtr(u0[left]), log_ndtr(u1[left])
            out[left] = a + np.log1p(-np.exp(b - a))
        if mid.any():
            out[mid] = np.log1p(-(ndtr(-u0[mid]) + ndtr(u1[mid])))
    return out


def _log_out(x: np.ndarray, half, s: float) -> np.ndarray:
    """log(1 - interval) of `_log_interval(x, -half, half, s)`: log(Phi(-(x + half) / s) + Phi((x - half) / s)),
    exact in both tails (deep inside, where 1 - q is far below double precision of 1, too)."""
    return np.logaddexp(log_ndtr(-(x + half) / s), log_ndtr((x - half) / s))


def _log_interval_d(x: np.ndarray, half: np.ndarray, s: float):
    """`_log_interval(x, -half, half, s)` (log q) and `_log_out` (log t, t = 1 - q), and their derivatives in x and in
    half: log q, d log q / dx, d log q / d half, log t, d log t / dx, d log t / d half (each finite: a normal density
    over its tail's mass)."""
    lz = _log_interval(x, -half, half, s)
    lt = _log_out(x, half, s)
    u0, u1 = (x + half) / s, (x - half) / s
    l0, l1 = -u0 ** 2 / 2 - _LOG2PI / 2, -u1 ** 2 / 2 - _LOG2PI / 2
    p0, p1 = np.exp(l0 - lz), np.exp(l1 - lz)
    t0, t1 = np.exp(l0 - lt), np.exp(l1 - lt)
    return lz, (p0 - p1) / s, (p0 + p1) / s, lt, (t1 - t0) / s, -(t0 + t1) / s


def _lse(a: np.ndarray) -> np.ndarray:
    """log sum exp over axis 1, -inf for a row of -inf (or no columns)."""
    if a.shape[1] == 0:
        return np.full(a.shape[0], -np.inf)
    top = a.max(1)
    safe = np.where(np.isfinite(top), top, 0.0)
    with np.errstate(divide="ignore"):
        return safe + np.log(np.exp(a - safe[:, None]).sum(1))


def _between(old: np.ndarray, new: np.ndarray, s: float) -> np.ndarray:
    """The parameters a share s of the way from `old` to `new` (the axis the short way round, modulo a half turn)."""
    out = old + s * (new - old)
    out[4] = old[4] + s * _half(new[4] - old[4])
    return out


class _Model:
    """The model's spreads, as the densities read them."""

    def __init__(self, spreads: Spreads, channels: int):
        spreads.check(channels)
        self.k = channels
        self.cell = spreads.cell
        self.ac = np.asarray(spreads.centre_bias, np.float64)
        self.bc = np.asarray(spreads.centre_slope, np.float64)
        self.al, self.bl, self.ah, self.bh = (spreads.length_bias, spreads.length_slope, spreads.height_bias,
                                              spreads.height_slope)
        self.aa = math.radians(spreads.angle_bias)
        self.sc, self.sl, self.sh = spreads.centre, spreads.length, spreads.height
        self.sa, self.st = math.radians(spreads.angle), spreads.tag
        self.nc, self.nl, self.nh, self.na, self.nt = (spreads.centre_dof, spreads.length_dof, spreads.height_dof,
                                                       spreads.angle_dof, spreads.tag_dof)
        # each word's shared errors' sds (cx, cy, log length, log height, axis): 0 holds that error at 0
        self.tau = np.array([spreads.centre_word, spreads.centre_word, spreads.length_word, spreads.height_word,
                             math.radians(spreads.angle_word)])
        self.norm = (2 * (math.log(self.sc) - _t_log_norm(self.nc)) + math.log(self.sl) - _t_log_norm(self.nl)
                     + math.log(self.sh) - _t_log_norm(self.nh) + math.log(self.sa) - _t_log_norm(self.na)
                     + channels * (math.log(self.st) - _t_log_norm(self.nt)))
        self.edge = spreads.edge
        self.reach = None                     # the support's reach: a page's (`_Votes`)
        self.lpi, self.lnpi = math.log(spreads.core_share), math.log1p(-spreads.core_share)
        bl, bh = spreads.background_length, spreads.background_height
        self.gc, self.gl_mean, self.gl, self.gh_mean, self.gh = (1 / (2 * spreads.background_centre ** 2), bl[0],
                                                                 1 / (2 * bl[1] ** 2), bh[0], 1 / (2 * bh[1] ** 2))
        self.gt = 1 / (2 * spreads.background_tag ** 2)
        self.gmean = np.asarray(spreads.background_tag_mean, np.float64)
        self.gnorm = (_LOG2PI + 2 * math.log(spreads.background_centre) + 0.5 * (_LOG2PI + 2 * math.log(bl[1]))
                      + 0.5 * (_LOG2PI + 2 * math.log(bh[1])) + math.log(math.pi)
                      + channels / 2 * (_LOG2PI + 2 * math.log(spreads.background_tag)))
        # the point placements (`_placement`): (a, b, sd) of the offset x, y, the log length and the log height, a
        # vote's error the t's scale and its word's shared error together
        self.place = [_placement(self.ac[i], self.bc[i], math.hypot(self.sc, spreads.centre_word),
                                 spreads.centre_truth[i]) for i in (0, 1)] + [
            _placement(self.al, self.bl, math.hypot(self.sl, spreads.length_word), spreads.length_truth),
            _placement(self.ah, self.bh, math.hypot(self.sh, spreads.height_word), spreads.height_truth)]

    def log_background(self, v: dict) -> np.ndarray:
        """The background's log density of each vote."""
        return (-(v["o"] ** 2).sum(1) * self.gc - (v["ll"] - self.gl_mean) ** 2 * self.gl
                - (v["lh"] - self.gh_mean) ** 2 * self.gh - ((v["t"] - self.gmean) ** 2).sum(1) * self.gt
                - self.gnorm)

    def errors(self, v: dict, prm: np.ndarray, u: np.ndarray | None = None):
        """Each vote's errors under a word (`prm` (5,) or (n, 5), its shared errors `u` the same, None: 0): the
        offset's (n, 2) in heights, the log length's, the log height's, the axis's (radians, modulo a half turn); and
        the word's (c - p) / H (n, 2)."""
        n = len(v["ll"])
        prm = np.broadcast_to(prm, (n, 5))
        u = np.zeros((n, 5)) if u is None else np.broadcast_to(u, (n, 5))
        x = (prm[:, :2] - v["p"]) * np.exp(-prm[:, 3])[:, None]
        return (v["o"] - self.ac - self.bc * x - u[:, :2], v["ll"] - self.al - self.bl * prm[:, 2] - u[:, 2],
                v["lh"] - self.ah - self.bh * prm[:, 3] - u[:, 3], _half(v["th"] - self.aa - prm[:, 4] - u[:, 4]), x)

    def log_vote(self, v: dict, prm: np.ndarray, tag_mean: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """log density of each vote under a word (`prm` (5,) or (n, 5), `tag_mean` (k,) or (n, k), its shared errors
        `u` (5,) or (n, 5), None: 0)."""
        ec, el, eh, ea, _ = self.errors(v, prm, u)
        return -(_t_rho((ec / self.sc) ** 2, self.nc).sum(1) + _t_rho((el / self.sl) ** 2, self.nl)
                 + _t_rho((eh / self.sh) ** 2, self.nh) + _t_rho((ea / self.sa) ** 2, self.na)
                 + _t_rho(((v["t"] - tag_mean) / self.st) ** 2, self.nt).sum(1) + self.norm)

    def weights(self, ec, el, eh, ea):
        """The t's weights of the errors (`errors`): offset (n, 2), log length, log height, axis."""
        return (_t_weight((ec / self.sc) ** 2, self.nc), _t_weight((el / self.sl) ** 2, self.nl),
                _t_weight((eh / self.sh) ** 2, self.nh), _t_weight((ea / self.sa) ** 2, self.na))

    def log_q(self, p: np.ndarray, prm: np.ndarray) -> np.ndarray:
        """log q and log(1 - q) (see the module) of points `p`, (n, 2): the chance a px there is in the word's box and
        not, each exact (1 - q = (1 - q_a) + q_a (1 - q_b), its logs from each axis's tails)."""
        pa, pb, a, b = self._frame(p, prm)
        lqa, lqb = _log_interval(pa, -a, a, self.edge), _log_interval(pb, -b, b, self.edge)
        return np.stack([lqa + lqb, np.logaddexp(_log_out(pa, a, self.edge), lqa + _log_out(pb, b, self.edge))], -1)

    def _frame(self, p: np.ndarray, prm: np.ndarray):
        """Points in the word's frame, and the half sides of the box its px lie in: its box grown by a px's half
        width (a px is a unit square at its numbers, so one centred on the box's edge is in it)."""
        along, up = _axes(prm[4])
        rel = p - prm[:2]
        return rel @ along, rel @ up, math.exp(prm[2]) / 2 + 0.5, math.exp(prm[3]) / 2 + 0.5

    def inside(self, p: np.ndarray, prm: np.ndarray, grow: float = 0.0) -> np.ndarray:
        """Which points lie in the word's support (its box grown by the reach), its edges moved out by `grow` px."""
        pa, pb, a, b = self._frame(p, prm)
        return (np.abs(pa) <= a + self.reach + grow) & (np.abs(pb) <= b + self.reach + grow)

    def vote_curvature(self, v: dict, r: np.ndarray, prm: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        """The curvature (Gauss-Newton, the t's weights at `prm`) of the votes' log densities weighted by r in each
        of a word's parameters (cx, cy, log length, log height, theta)."""
        ec, el, eh, ea, x = self.errors(v, prm, u)
        wc, wl, wh, wa = self.weights(ec, el, eh, ea)
        ih2 = math.exp(-2 * prm[3])
        rc = r[:, None] * wc * self.bc ** 2
        return np.array([float(rc[:, 0].sum()) * ih2 / self.sc ** 2, float(rc[:, 1].sum()) * ih2 / self.sc ** 2,
                         float((r * wl).sum()) * self.bl ** 2 / self.sl ** 2,
                         float((rc * x ** 2).sum()) / self.sc ** 2 + float((r * wh).sum()) * self.bh ** 2 / self.sh ** 2,
                         float((r * wa).sum()) / self.sa ** 2])

    def own_update(self, v: dict, r: np.ndarray, prm: np.ndarray, tag: np.ndarray, u: np.ndarray | None = None):
        """The votes' own update of a word's parameters and tag, weighted by r (no spatial term; `_own_updates` of
        one word).  None when r holds no mass."""
        if not float(r.sum()) > 0:
            return None
        u = np.zeros((1, 5)) if u is None else np.asarray(u, np.float64).reshape(1, 5)
        new, tags = _own_updates(self, v, r, np.asarray(prm, np.float64)[None], np.asarray(tag, np.float64)[None],
                                 u, np.zeros(len(r), np.int64), 1)
        return new[0], tags[0]

    def shared_update(self, v: dict, r: np.ndarray, prm: np.ndarray, u: np.ndarray) -> np.ndarray:
        """A word's shared errors u given its box (`prm`), weighted by r: each dimension's (with a shared error)
        weighted mean of its votes' errors with u's prior, sum r w e / s^2 / (sum r w / s^2 + cell / tau^2) (the t's
        weights w at u: an exact CM step of the t as a scale mixture of Gaussians, the prior counted once per word,
        cell times in the votes' units)."""
        u = np.array(u, np.float64)
        ec, el, eh, ea, _ = self.errors(v, prm, u)
        wc, wl, wh, wa = self.weights(ec, el, eh, ea)
        errs = (ec[:, 0] + u[0], ec[:, 1] + u[1], el + u[2], eh + u[3], ea + u[4])     # each error without its u
        ws = (wc[:, 0], wc[:, 1], wl, wh, wa)
        scales = (self.sc, self.sc, self.sl, self.sh, self.sa)
        for d in range(5):
            if self.tau[d] > 0:
                prec = float((r * ws[d]).sum()) / scales[d] ** 2
                mean = float((r * ws[d] * errs[d]).sum()) / scales[d] ** 2
                u[d] = mean / (prec + self.cell / self.tau[d] ** 2)
        return u

    def shared_prior(self, words: list[dict]) -> float:
        """The words' shared errors' log prior against none, in the votes' units (cell times the once-per-word
        -u^2 / 2 tau^2)."""
        total = 0.0
        for w in words:
            u = w.get("u")
            if u is not None:
                on = self.tau > 0
                total -= float((np.asarray(u)[on] ** 2 / (2 * self.tau[on] ** 2)).sum())
        return self.cell * total

    def placements(self, v: dict) -> np.ndarray:
        """The parameters (n, 5) of the boxes the votes place (each dimension's point placement)."""
        (ax, bx, _), (ay, by, _), (al, bl, _), (ah, bh, _) = self.place
        lh = ah + bh * v["lh"]
        off = np.stack([ax + bx * v["o"][:, 0], ay + by * v["o"][:, 1]], -1)
        return np.c_[v["p"] + np.exp(lh)[:, None] * off, al + bl * v["ll"], lh, v["th"] - self.aa]


def _own_updates(m: _Model, v: dict, r: np.ndarray, prms: np.ndarray, tags: np.ndarray, us: np.ndarray,
                 group: np.ndarray, count: int):
    """The votes' own update of each group's word (no spatial term), its votes weighted by r: each dimension a
    Gauss-Newton step on the votes' errors weighted by r and the t's weights at the current values (the axis, the log
    length, then the centre and the log height in turn, SHAPE_STEPS; each step exact for the t's weights held); the
    tag the votes' t-weighted mean.  A group with no mass, or a parameter its votes cannot place (no curvature, or a
    step past float range), keeps its value."""
    prms, tags = np.array(prms, np.float64), np.array(tags, np.float64)
    on = np.bincount(group, r, count) > 0
    U = us[group]

    def step(num, den, i):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d = num / den
        ok = on & (den > 0) & np.isfinite(d)
        prms[ok, i] += d[ok]

    def errs():
        ec, el, eh, ea, x = m.errors(v, prms[group], U)
        return (ec, el, eh, ea, x), m.weights(ec, el, eh, ea)

    wt = r[:, None] * _t_weight(((v["t"] - tags[group]) / m.st) ** 2, m.nt)
    for ch in range(m.k):
        den = np.bincount(group, wt[:, ch], count)
        num = np.bincount(group, wt[:, ch] * v["t"][:, ch], count)
        ok = on & (den > 0)
        tags[ok, ch] = num[ok] / den[ok]
    (_, _, _, ea, _), (_, _, _, wa) = errs()
    step(np.bincount(group, r * wa * ea, count), np.bincount(group, r * wa, count), 4)
    (_, el, _, _, _), (_, wl, _, _) = errs()
    step(np.bincount(group, r * wl * el, count) * m.bl, np.bincount(group, r * wl, count) * m.bl ** 2, 2)
    for _ in range(SHAPE_STEPS):
        (ec, _, _, _, _), (wc, _, _, _) = errs()
        ih = np.exp(-prms[group, 3])
        for a in (0, 1):
            rw = r * wc[:, a] * m.bc[a]
            step(np.bincount(group, rw * ec[:, a] * ih, count), np.bincount(group, rw * m.bc[a] * ih * ih, count), a)
        (ec, _, eh, _, x), (wc, _, wh, _) = errs()
        rc = r[:, None] * wc * m.bc
        num = (np.bincount(group, -(rc * ec * x).sum(1), count) / m.sc ** 2
               + np.bincount(group, r * wh * eh, count) * m.bh / m.sh ** 2)
        den = (np.bincount(group, (rc * m.bc * x * x).sum(1), count) / m.sc ** 2
               + np.bincount(group, r * wh, count) * m.bh ** 2 / m.sh ** 2)
        step(num, den, 3)
    return prms, tags


class _Votes(_Model):
    """The votes of a page's px (or of the px in `mask`), read from the arrays when asked (`take`)."""

    def __init__(self, core_logit, geo, tag, spreads: Spreads, mask=None):
        core_logit, self.geo, self.tag = np.asarray(core_logit), np.asarray(geo), np.asarray(tag)
        super().__init__(spreads, self.tag.shape[0])
        self.shape = core_logit.shape
        h, w = self.shape
        on = np.ones(self.shape, bool) if mask is None else np.asarray(mask, bool)
        self.ys, self.xs = np.nonzero(on)
        self.index = np.full((h, w), -1, np.int64)
        self.index[self.ys, self.xs] = np.arange(len(self.ys))
        self.x = core_logit[self.ys, self.xs].astype(np.float64)
        # the reach: q at the reach times the most any vote can favour a word (its core evidence, a word's peak
        # density of its vote over the background's) is double precision, so a support leaves out less than that
        most = -math.inf
        for lo in range(0, len(self), CHUNK_PX):
            v = self.take(np.arange(lo, min(lo + CHUNK_PX, len(self))))
            most = max(most, float((v["lin"] - v["lout"] - v["lbg"]).max()))
        log_k = most - self.norm
        self.reach = (self.edge * max(-float(ndtri_exp(min(math.log(PRECISION) - log_k, 0.0))), 0.0)
                      if len(self) else 0.0)

    def __len__(self):
        return len(self.ys)

    def chance(self, idx=slice(None)) -> np.ndarray:
        return 0.5 * (1 + np.tanh(self.x[idx] / 2))

    def take(self, idx) -> dict:
        """The votes `idx` as the net gives them: p (n, 2), the offset o (n, 2) in heights, log length ll, log height
        lh, axis th (radians), tags t (n, k), the core chance w, the core head's evidence in a word's box and out of
        every one (log w / core_share and log (1 - w) / (1 - core_share), from the logit: never infinite), and the
        background's log density of each vote."""
        ys, xs = self.ys[idx], self.xs[idx]
        g = self.geo[:, ys, xs].astype(np.float64)
        x = self.x[idx]
        lw, lnw = -np.logaddexp(0, -x), -np.logaddexp(0, x)
        v = {"p": np.stack([xs, ys], -1).astype(np.float64), "o": (g[:2] / g[3]).T,
             "ll": np.log(g[2]), "lh": np.log(g[3]), "th": np.deg2rad(g[4]),
             "t": np.moveaxis(self.tag[:, ys, xs].astype(np.float64), 0, -1),
             "w": 0.5 * (1 + np.tanh(x / 2)), "lin": lw - self.lpi, "lout": lnw - self.lnpi}
        v["lbg"] = self.log_background(v)
        return v

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """The centre each vote places (n, 2) and the height it places (n,) (`placements`), CHUNK_PX votes at a
        time."""
        out, heights = np.empty((len(self), 2)), np.empty(len(self))
        (ax, bx, _), (ay, by, _), _, (ah, bh, _) = self.place
        for lo in range(0, len(self), CHUNK_PX):
            ys, xs = self.ys[lo:lo + CHUNK_PX], self.xs[lo:lo + CHUNK_PX]
            g = self.geo[:4, ys, xs].astype(np.float64)
            height = np.exp(ah + bh * np.log(g[3]))
            off = np.stack([ax + bx * g[0] / g[3], ay + by * g[1] / g[3]], -1)
            out[lo:lo + CHUNK_PX] = np.stack([xs, ys], -1) + height[:, None] * off
            heights[lo:lo + CHUNK_PX] = height
        return out, heights

    def region(self, lo, hi) -> np.ndarray:
        """The votes of the page px in [lo, hi] (x, y; clipped to the page)."""
        h, w = self.shape
        x0, y0 = max(int(math.floor(lo[0])), 0), max(int(math.floor(lo[1])), 0)
        x1, y1 = min(int(math.ceil(hi[0])) + 1, w), min(int(math.ceil(hi[1])) + 1, h)
        if x0 >= x1 or y0 >= y1:
            return np.zeros(0, np.int64)
        idx = self.index[y0:y1, x0:x1].ravel()
        return idx[idx >= 0]

    def bounds(self, prm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The word's support's axis-aligned bounds (lo, hi)."""
        box = _box(prm, self.reach + 0.5)
        return box.min(0), box.max(0)

    def support(self, prm: np.ndarray) -> np.ndarray:
        """The votes in the word's numerical support, sorted."""
        idx = self.region(*self.bounds(prm))
        p = np.stack([self.xs[idx], self.ys[idx]], -1).astype(np.float64)
        return idx[self.inside(p, prm)]


# --- the mixture with the spatial prior --------------------------------------------------------------------------

def _weights(words: list[dict]) -> np.ndarray:
    """The words' log weights (a word without one: 0)."""
    return np.array([float(w.get("a", 0.0)) for w in words], np.float64)


def _terms(V: _Votes, v: dict, words: list[dict]):
    """For votes `v` and each word: log q and log(1 - q) ((n, K, 2); -inf and 0 off its support) and the vote's log
    density (-inf off it)."""
    n, kk = len(v["w"]), len(words)
    lq = np.zeros((n, kk, 2))
    lq[..., 0] = -np.inf
    lf = np.full((n, kk), -np.inf)
    for j, word in enumerate(words):
        inside = V.inside(v["p"], word["prm"])
        if inside.any():
            sv = _sub(v, inside)
            lq[inside, j] = V.log_q(sv["p"], word["prm"])
            lf[inside, j] = V.log_vote(sv, word["prm"], word["tag"], word.get("u"))
    return lq, lf


def _log1m_exp(x: np.ndarray) -> np.ndarray:
    """log(1 - exp(x)) for x <= 0, in the form exact for each range (log(-expm1(x)) near 0, log1p(-exp(x)) below)."""
    x = np.asarray(x, np.float64)
    with np.errstate(divide="ignore"):
        return np.where(x > -_LOG2, np.log(-np.expm1(np.minimum(x, 0.0))), np.log1p(-np.exp(np.minimum(x, 0.0))))


def _cover(lq: np.ndarray):
    """Per vote, from the words' log q and log(1 - q) (n, K, 2): log C (the chance its px is in some word's box,
    1 - prod(1 - q)) and log(1 - C)."""
    s = lq[..., 1].sum(1) if lq.shape[1] else np.zeros(lq.shape[0])
    return _log1m_exp(s), s


def _shares(lq: np.ndarray, la: np.ndarray) -> np.ndarray:
    """log of each word's share of each vote's px among the words whose boxes may hold it, a_k q_k / sum a q (`la`
    the log weights a; `lq` (n, K, 2) as `_terms` gives it); -inf off its support, and for every word where none of
    them holds any weight."""
    lw = lq[..., 0] + la[None, :]
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(lw), lw - _lse(lw)[:, None], -np.inf)


def _mixture(v: dict, lq: np.ndarray, lf: np.ndarray, model: "_Model", la: np.ndarray):
    """Each vote's score term (log L less its value with no word: 0 where no word's support reaches) and its chance
    of being each word's.  See the module: in some word's box with chance C = 1 - prod(1 - q_k), word k's among them
    by its weight's share a_k q_k / sum a q (`la` the log weights), the core evidence w / core_share and the vote
    word k's; out of every box with chance 1 - C, the core evidence (1 - w) / (1 - core_share) and the background's
    vote."""
    log_c, log_rest = _cover(lq)
    share = _shares(lq, la)
    joint = v["lin"][:, None] + log_c[:, None] + share + lf
    none = v["lout"] + v["lbg"]
    num = np.logaddexp(_lse(joint), log_rest + none)
    with np.errstate(invalid="ignore"):
        resp = np.nan_to_num(np.exp(joint - num[:, None]))
    return num - none, resp


def _weight_step(v: dict, lq: np.ndarray, lf: np.ndarray, la: np.ndarray, n: int, model: "_Model") -> np.ndarray:
    """The first `n` words' log weights after a Newton step on the log likelihood in them (the others held).  Per
    vote, with r its chances of being each word's, m their sum, s the shares and c = r - m s: the gradient is sum c
    and the Hessian sum (diag(c) - c s' - s c' - c c') (both exact).  The Hessian's eigenvalues are taken as their
    size, those numerically 0 (below sqrt(eps) of the largest: a shift of weights that changes no share) left out;
    the first of the step, its half, its quarter, ... (to LINE_TOL) that raises the likelihood is taken, none if none
    does."""
    out = la[:n].copy()
    if not n:
        return out
    term, resp = _mixture(v, lq, lf, model, la)
    f0 = float(term.sum())
    shares = np.exp(_shares(lq, la))
    c = resp - resp.sum(1)[:, None] * shares
    cn, sn = c[:, :n], shares[:, :n]
    grad = cn.sum(0)
    hess = np.diag(grad) - cn.T @ sn - sn.T @ cn - cn.T @ cn
    lam, vec = np.linalg.eigh(-hess)
    size = np.abs(lam)
    keep = size > math.sqrt(np.finfo(np.float64).eps) * size.max()
    if not keep.any():
        return out
    step = vec[:, keep] @ ((vec[:, keep].T @ grad) / size[keep])
    share = 1.0
    while share >= LINE_TOL:
        trial = la.copy()
        trial[:n] = out + share * step
        if float(_mixture(v, lq, lf, model, trial)[0].sum()) > f0:
            return trial[:n]
        share /= 2
    return out


def _span(V: _Votes, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(idx):
        return np.zeros(2), -np.ones(2)
    return (np.array([V.xs[idx].min(), V.ys[idx].min()], np.float64),
            np.array([V.xs[idx].max(), V.ys[idx].max()], np.float64))


def _near(V: _Votes, words, lo, hi) -> list[dict]:
    """The words (a list, or a `_Words`) whose supports' bounds meet [lo, hi] (the others cannot touch a px there)."""
    if isinstance(words, _Words):
        return words.near(lo, hi)
    out = []
    for word in words:
        blo, bhi = V.bounds(word["prm"])
        if (blo <= hi).all() and (lo <= bhi).all():
            out.append(word)
    return out


def _supports(V: _Votes, words: list[dict]) -> np.ndarray:
    return np.unique(np.concatenate([V.support(w["prm"]) for w in words])) if words else np.zeros(0, np.int64)


def _score(V: _Votes, region: np.ndarray, words: list[dict], held: list[dict]) -> float:
    """The log likelihood of the votes of `region` (less the background's), words `words` and the held words near,
    with the words' shared errors' prior (`_Model.shared_prior`; the held words' is the same either way)."""
    v = V.take(region)
    every = words + _near(V, held, *_span(V, region))
    lq, lf = _terms(V, v, every)
    return float(_mixture(v, lq, lf, V, _weights(every))[0].sum()) + V.shared_prior(words)


def _reach_of(V: _Votes, prm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounds holding the support of a word with these parameters, at any axis (its circumscribed circle, grown by
    the support's reach)."""
    radius = 0.5 * math.hypot(math.exp(prm[2]) + 1, math.exp(prm[3]) + 1) + V.reach
    return prm[:2] - radius, prm[:2] + radius


def _lines_reach(V: _Votes, *prms) -> tuple[np.ndarray, np.ndarray]:
    """Bounds holding the support of a word at every point of the lines between `prms` (each point's centre between
    theirs and its sizes no bigger than the larger end's: the bounds of every end's `_reach_of`)."""
    spans = [_reach_of(V, prm) for prm in prms]
    return np.min([a for a, _ in spans], 0), np.max([b for _, b in spans], 0)


def _fit(V: _Votes, start: list[dict], held: list[dict], trace: list | None = None) -> list[dict]:
    """Words `start` fitted (see the module) with the words `held` held.  `trace` collects each step's gain of the
    words (the log likelihood with them less without them, the held words there either way, with their shared
    errors' prior) before and after its moves (tags, shared errors, weights and boxes): the same on any region that
    holds their supports, so one step's after is the next one's before.  RuntimeError when a step lowers it beyond
    float noise (a bug)."""
    words = [{"prm": np.array(w["prm"], np.float64), "tag": np.array(w["tag"], np.float64),
              "a": float(w.get("a", 0.0)), "u": np.array(w.get("u", np.zeros(5)), np.float64)} for w in start]
    n = len(words)
    for _ in range(EM_STEPS):
        region = _supports(V, words)
        v = V.take(region)
        every = words + _near(V, held, *_span(V, region))
        lq, lf = _terms(V, v, every)
        la = _weights(every)
        _, resp = _mixture(v, lq, lf, V, la)
        # the tags' and the shared errors' exact CM steps from this E-step (they raise the expected complete log
        # likelihood, so the likelihood), before any other move; the box's moves then hold them
        old = [(w["tag"], w["u"]) for w in words]
        for j, w in enumerate(words):
            target = V.own_update(v, resp[:, j], w["prm"], w["tag"], w["u"])
            if target is not None:
                w["tag"] = target[1]
            w["u"] = V.shared_update(v, resp[:, j], w["prm"], w["u"])
        targets = [V.own_update(v, resp[:, j], w["prm"], w["tag"], w["u"]) for j, w in enumerate(words)]
        lf[:, :n] = _terms(V, v, words)[1]
        steps = [_newton(V, v, lq, lf, la, j, w, resp[:, j]) for j, w in enumerate(words)]
        # the step's region: every point of every word's lines (to its Newton step's end, then to its votes' update)
        spans = []
        for w, target, d in zip(words, targets, steps):
            ends = [w["prm"]] + ([w["prm"] + d] if d is not None else []) + ([target[0]] if target is not None else [])
            spans.append(V.region(*_lines_reach(V, *ends)))
        region = np.union1d(region, np.concatenate(spans))
        v = V.take(region)
        every = words + _near(V, held, *_span(V, region))
        lq, lf = _terms(V, v, every)
        la = _weights(every)
        before_words = [{**w, "tag": t, "u": u} for w, (t, u) in zip(words, old)]
        base = float(_mixture(v, lq[:, n:], lf[:, n:], V, la[n:])[0].sum())          # the held words alone
        before = (float(_mixture(v, lq, np.hstack([_terms(V, v, before_words)[1], lf[:, n:]]), V, la)[0].sum())
                  - base + V.shared_prior(before_words))
        la[:n] = _weight_step(v, lq, lf, la, n, V)
        for j, w in enumerate(words):
            w["a"] = float(la[j])
        for j, (w, target, d) in enumerate(zip(words, targets, steps)):
            if d is not None:                                    # its Newton step
                _climb(V, v, lq, lf, la, j, w, w["prm"] + d, w["tag"])
            if target is not None:                               # then to the votes' own update
                _climb(V, v, lq, lf, la, j, w, target[0], w["tag"])
        after = float(_mixture(v, lq, lf, V, la)[0].sum()) - base + V.shared_prior(words)
        if trace is not None:
            trace.append((before, after))
        if after < before - LL_NOISE * max(1.0, abs(before)):
            raise RuntimeError(f"a fit step lowered the log likelihood from {before} to {after}: a bug")
        if after - before <= EM_TOL * len(region):
            break
    return words


def _near_line(V: _Votes, v: dict, *prms) -> np.ndarray:
    """The votes of `v` a word's support can touch at any point of the line between `prms` (`_lines_reach`)."""
    lo, hi = _lines_reach(V, *prms)
    return np.flatnonzero(((v["p"] >= lo) & (v["p"] <= hi)).all(1))


def _move(V: _Votes, v: dict, lq, lf, la, j: int, near: np.ndarray, u: np.ndarray | None = None):
    """value(prm, tag): the log likelihood of the votes `near` (of `v`) with word j at (prm, tag) (its shared errors
    `u` held), the other words held (lq, lf their terms on `v`, la the log weights); and word j's terms there.  Off
    those votes word j's terms are -inf at every (prm, tag) it is asked of (`_near_line`), so only theirs change.  The
    other words' parts of `_mixture` (their prod (1 - q), sum a q and sum a q f) are summed once, so each value costs
    word j's terms alone: the same sums as `_mixture`'s, in another order."""
    vn = {key: v[key][near] for key in ("p", "o", "ll", "lh", "th", "t")}
    lin, none = v["lin"][near], v["lout"][near] + v["lbg"][near]
    oq, of, oa = np.delete(lq[near], j, 1), np.delete(lf[near], j, 1), np.delete(la, j)
    rest = oq[..., 1].sum(1) if oq.shape[1] else np.zeros(len(near))
    lw = oq[..., 0] + oa[None, :]
    others = _lse(lw)
    with np.errstate(invalid="ignore"):
        others_joint = _lse(lw + of)
    aj = float(la[j])

    def value(prm, tag):
        lqj, lfj = np.zeros((len(near), 2)), np.full(len(near), -np.inf)
        lqj[:, 0] = -np.inf
        inside = V.inside(vn["p"], prm)
        if inside.any():
            sv = {key: a[inside] for key, a in vn.items()}
            lqj[inside] = V.log_q(sv["p"], prm)
            lfj[inside] = V.log_vote(sv, prm, tag, u)
        log_rest = rest + lqj[:, 1]
        log_c = _log1m_exp(log_rest)
        wj = lqj[:, 0] + aj
        total = np.logaddexp(others, wj)
        with np.errstate(invalid="ignore"):
            words_ = np.where(np.isfinite(total), np.logaddexp(others_joint, wj + lfj) - total, -np.inf)
        num = np.logaddexp(lin + log_c + words_, log_rest + none)
        return float((num - none).sum()), lqj, lfj

    def gradient(prm, tag) -> np.ndarray:
        """The log likelihood's gradient in word j's parameters (cx, cy, log length, log height, theta), exact.  Per
        vote (L its likelihood, R_t = prod (1 - q) over every word, A the words' share-weighted density, s word j's
        share, r its chance of being the vote's, m any word's): dL / L = (exp(none) - exp(lin) A) R_t / L d log(1 - q_j)
        + (r - s m) d log q_j + r d log f_j.  Where q_j < 1/2 the first term is read through d log(1 - q_j) =
        -q_j / (1 - q_j) d log q_j, else through each axis's tail: d log(1 - q) / d pa = q_b (1 - q_a) / (1 - q)
        d log(1 - q_a) / d pa (and likewise); so every factor stays within float range (a px a box must hold whose
        vote it cannot explain gives a large gradient, not an overflow).  Then through q_j's frame and f_j's terms
        (each dimension's error times its t weight over its scale squared, times the error's derivative)."""
        g = np.zeros(5)
        inside = V.inside(vn["p"], prm)
        if not inside.any():
            return g
        sv = {key: a[inside] for key, a in vn.items()}
        pa, pb, half_a, half_b = V._frame(sv["p"], prm)
        lqa, dqa_x, dqa_half, lta, dta_x, dta_half = _log_interval_d(pa, half_a, V.edge)
        lqb, dqb_x, dqb_half, ltb, dtb_x, dtb_half = _log_interval_d(pb, half_b, V.edge)
        lqj = lqa + lqb
        l1q = np.logaddexp(lta, lqa + ltb)
        lfj = V.log_vote(sv, prm, tag, u)
        rest_i, lin_i, none_i = rest[inside], lin[inside], none[inside]
        log_rest = rest_i + l1q
        log_c = _log1m_exp(log_rest)
        wj = lqj + aj
        total = np.logaddexp(others[inside], wj)
        words_ = np.logaddexp(others_joint[inside], wj + lfj) - total
        num = np.logaddexp(lin_i + log_c + words_, log_rest + none_i)
        r = np.exp(lin_i + log_c + wj + lfj - total - num)
        sm = np.exp(wj - total + lin_i + log_c + words_ - num)
        low = lqj < -_LOG2                                    # q_j < 1/2: the first term through d log q_j
        dq = r - sm                                           # on d log q_j
        lo_, hi_ = np.flatnonzero(low), np.flatnonzero(~low)
        dq[lo_] += (np.exp(lin_i[lo_] + words_[lo_] + rest_i[lo_] + lqj[lo_] - num[lo_])
                    - np.exp(none_i[lo_] + rest_i[lo_] + lqj[lo_] - num[lo_]))
        dt = np.zeros(len(lqj))                               # on d log(1 - q_j)
        dt[hi_] = (np.exp(none_i[hi_] + log_rest[hi_] - num[hi_])
                   - np.exp(lin_i[hi_] + words_[hi_] + log_rest[hi_] - num[hi_]))
        # d log(1 - q) through each axis: its share of 1 - q times its tail's log derivative
        sa_, sb_ = np.exp(lqb + lta - l1q), np.exp(lqa + ltb - l1q)
        da = dq * dqa_x + dt * sa_ * dta_x                    # d / d pa
        db = dq * dqb_x + dt * sb_ * dtb_x                    # d / d pb
        dha = dq * dqa_half + dt * sa_ * dta_half             # d / d half_a
        dhb = dq * dqb_half + dt * sb_ * dtb_half             # d / d half_b
        along, up = _axes(prm[4])
        g[:2] = -(da.sum() * along + db.sum() * up)
        g[2] = float(dha.sum()) * math.exp(prm[2]) / 2
        g[3] = float(dhb.sum()) * math.exp(prm[3]) / 2
        g[4] = float((da * pb - db * pa).sum())
        ec, el, eh, ea, x = V.errors(sv, prm, u)
        wc, wl, wh, wa = V.weights(ec, el, eh, ea)
        rc = r[:, None] * wc * V.bc
        g[:2] += math.exp(-prm[3]) * (rc * ec).sum(0) / V.sc ** 2
        g[2] += V.bl * float((r * wl * el).sum()) / V.sl ** 2
        g[3] += -float((rc * ec * x).sum()) / V.sc ** 2 + V.bh * float((r * wh * eh).sum()) / V.sh ** 2
        g[4] += float((r * wa * ea).sum()) / V.sa ** 2
        return g

    value.gradient = gradient
    return value


def _newton(V: _Votes, v: dict, lq, lf, la, j: int, w: dict, r: np.ndarray):
    """Word j's Newton step on the log likelihood in its parameters (the spatial terms included, which the votes' own
    update does not see): the gradient exact (`_move`), the Hessian by its finite differences (FD_STEP, made
    symmetric), each of the Hessian's eigenvalues taken as its size and no less than
    the votes' curvature along its eigenvector (Gauss-Newton, `r` their chances of being the word's), so the step
    climbs where the likelihood does not curve down.  None where there is no step (no gradient, or past float
    range)."""
    prm, tag = w["prm"], w["tag"]
    near = np.flatnonzero(V.inside(v["p"], prm, 1.0))     # every support within FD_STEP of prm's (< 1 px away)
    if not len(near):
        return None
    gradient = _move(V, v, lq, lf, la, j, near, w.get("u")).gradient
    grad = gradient(prm, tag)
    e = np.eye(5) * FD_STEP
    hess = np.array([(gradient(prm + e[i], tag) - grad) / FD_STEP for i in range(5)])
    hess = (hess + hess.T) / 2
    if not (np.isfinite(grad).all() and np.isfinite(hess).all()):
        return None
    lam, vec = np.linalg.eigh(-hess)
    votes = V.vote_curvature(_sub(v, near), r[near], prm, w.get("u"))
    lam = np.maximum(np.abs(lam), (vec ** 2 * votes[:, None]).sum(0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        step = vec @ np.where(lam > 0, (vec.T @ grad) / np.where(lam > 0, lam, 1.0), 0.0)
    if not (np.isfinite(step).all() and np.abs(step).sum() > 0):
        return None
    # the trust region: scaled down to move by at most TRUST box lengths, log sizes and radians
    reach = max(math.hypot(step[0], step[1]) * math.exp(-prm[2]), *np.abs(step[2:])) / TRUST
    return step / reach if reach > 1 else step


def _climb(V: _Votes, v: dict, lq, lf, la, j: int, w: dict, prm_to: np.ndarray, tag_to: np.ndarray) -> None:
    """Word j (`w`, changed in place, with its columns of lq and lf) moved to the first point of the line to (prm_to,
    tag_to), its end, half way, a quarter, ... (to LINE_TOL), that raises the log likelihood; not moved if none
    does.  Each point's box at least a px long and high (log sizes >= 0)."""
    prm, tag = w["prm"], w["tag"]
    near = _near_line(V, v, prm, prm_to)
    value = _move(V, v, lq, lf, la, j, near, w.get("u"))
    f0 = value(prm, tag)[0]
    share = 1.0
    while share >= LINE_TOL:
        at = _between(prm, prm_to, share), tag + share * (tag_to - tag)
        at[0][2:4] = np.maximum(at[0][2:4], 0.0)          # a box is at least a px long and high
        got = value(*at)
        if got[0] > f0:
            w["prm"], w["tag"] = at
            lq[near, j], lf[near, j] = got[1], got[2]
            return
        share /= 2


def _change(V: _Votes, old: list[dict], new: list[dict], held: list[dict]) -> tuple[float, np.ndarray]:
    """The log likelihood's change from words `old` to words `new`, the others held: on the union of their supports
    (off it nothing changes).  And that region."""
    region = np.union1d(_supports(V, old), _supports(V, new))
    return _score(V, region, new, held) - _score(V, region, old, held), region


def _parts(V: _Votes, region: np.ndarray, focus: list[dict], held: list[dict]) -> list[tuple]:
    """For each word of `focus`, the votes of `region` and their chances of being its (the held words near
    competing)."""
    v = V.take(region)
    every = focus + _near(V, held, *_span(V, region))
    lq, lf = _terms(V, v, every)
    _, resp = _mixture(v, lq, lf, V, _weights(every))
    return [(region, resp[:, j]) for j in range(len(focus))]


# --- proposals ------------------------------------------------------------------------------------------------------

def _basins(centres: np.ndarray, w: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Each vote's basin (see the module), numbered from 0: the core-chance-weighted sum of votes at their centres'
    px (clipped onto the page), smoothed by a Gaussian of sd `sigma`, climbed by steepest ascent over 8-neighbours."""
    h, wd = shape
    ix = np.clip(np.rint(centres[:, 0]), 0, wd - 1).astype(np.int64)
    iy = np.clip(np.rint(centres[:, 1]), 0, h - 1).astype(np.int64)
    cell = iy * wd + ix
    smooth = cv2.GaussianBlur(np.bincount(cell, w, h * wd).reshape(h, wd), (0, 0), sigma,
                              borderType=cv2.BORDER_CONSTANT)
    pad = np.pad(smooth, 1, constant_values=-np.inf)
    best = smooth.copy()
    here = np.arange(h * wd, dtype=np.int64).reshape(h, wd)
    step = here.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                nb = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + wd]
                higher = nb > best
                best = np.where(higher, nb, best)
                step = np.where(higher, here + dy * wd + dx, step)
    del pad, best, smooth
    top = step.ravel()
    while True:
        nxt = top[top]
        if np.array_equal(nxt, top):
            break
        top = nxt
    return np.unique(top[cell], return_inverse=True)[1]


@dataclasses.dataclass
class _Proposals:
    """Every basin: its votes (order[first[b]:end[b]]), its one-word fit on them (prm, tag, ll) and the background's
    log likelihood of them (ll0)."""
    basin: np.ndarray
    order: np.ndarray
    first: np.ndarray
    end: np.ndarray
    prms: np.ndarray
    tags: np.ndarray
    ll: np.ndarray
    ll0: np.ndarray

    def votes(self, b: int) -> np.ndarray:
        return np.sort(self.order[self.first[b]:self.end[b]])


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """The median of `x` weighted by `w` (the first value whose cumulative weight reaches half)."""
    order = np.argsort(x, kind="stable")
    cum = np.cumsum(w[order])
    return float(x[order][min(int(np.searchsorted(cum, cum[-1] / 2)), len(x) - 1)])


def _proposals(V: _Votes, spreads: Spreads) -> _Proposals:
    """Every basin with its one-word fit on its own votes, all at once, a pass at a time over whole basins.  The
    basins' smoothing: the centre placement's spread (heights, the root mean square of its two ways) at the placed
    heights' median weighted by core chance (the word px's)."""
    centres, heights = V.centres()
    w = V.chance()
    spread = math.sqrt((V.place[0][2] ** 2 + V.place[1][2] ** 2) / 2)
    sigma = spread * _weighted_median(heights, w) if len(w) else 1.0
    basin = _basins(centres, w, V.shape, sigma)
    order = np.lexsort((-V.x, basin))               # by basin, the highest core chance first
    bs = basin[order]
    first = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
    end = np.r_[first[1:], len(order)]
    nb = len(first)
    prms, tags = np.empty((nb, 5)), np.empty((nb, V.k))
    ll, ll0 = np.empty(nb), np.empty(nb)
    lo = 0
    while lo < nb:
        hi = lo + 1
        while hi < nb and end[hi] - first[lo] <= CHUNK_PX:
            hi += 1
        sl = slice(first[lo], end[hi - 1])
        out = _fit_basins(V, order[sl], bs[sl] - bs[first[lo]], first[lo:hi] - first[lo])
        prms[lo:hi], tags[lo:hi], ll[lo:hi], ll0[lo:hi] = out
        lo = hi
    return _Proposals(basin, order, first, end, prms, tags, ll, ll0)


def _group_step(V: _Votes, v: dict, r: np.ndarray, prms, tags, group, count):
    """The one-word-per-group M-step on the votes alone (a basin's own fit, no shared errors): each group's votes' own
    update (`_own_updates`), kept where it raises the group's expected vote log density (else no change)."""
    prms, tags = np.array(prms, np.float64), np.array(tags, np.float64)
    new, new_tags = _own_updates(V, v, r, prms, tags, np.zeros((count, 5)), group, count)
    q_old = np.bincount(group, r * V.log_vote(v, prms[group], tags[group]), count)
    with np.errstate(invalid="ignore", over="ignore"):
        q_new = np.bincount(group, r * V.log_vote(v, new[group], new_tags[group]), count)
    keep = q_new > q_old
    prms[keep], tags[keep] = new[keep], new_tags[keep]
    return prms, tags


def _fit_basins(V: _Votes, idx, group, starts):
    """One word fitted to each basin's own votes at once (EM per basin on the votes alone, the background in the
    model, no spatial term: each vote in the word's box or out of every one, by the core evidence and its vote: the
    proposals' own fit), started at the box its vote of the highest core chance places, centred at its votes' placed
    centres' mean weighted by core chance (where the offsets say little, the core px's middle); each basin's EM stops
    on its own and only the basins still moving are computed.  The words' parameters and tags, and each basin's log
    likelihood with its word and with none."""
    v = V.take(idx)
    nb, n = len(starts), len(idx)
    prms = V.placements(_sub(v, starts))
    placed = V.placements(v)[:, :2]
    mass = np.bincount(group, v["w"], nb)
    for a in (0, 1):
        mean = np.bincount(group, v["w"] * placed[:, a], nb)
        prms[mass > 0, a] = mean[mass > 0] / mass[mass > 0]
    tags = v["t"][starts].copy()
    back = v["lout"] + v["lbg"]
    count = np.bincount(group, minlength=nb)
    lp = np.empty(n)
    prev = np.full(nb, -math.inf)
    moving = np.ones(nb, bool)
    for _ in range(EM_STEPS):
        sel = np.flatnonzero(moving[group])
        if not len(sel):
            break
        vs, g = _sub(v, sel), group[sel]
        word = vs["lin"] + V.log_vote(vs, prms[g], tags[g])
        lp[sel] = np.logaddexp(word, back[sel])
        ll = np.bincount(g, lp[sel], nb)
        moving &= ll - prev > EM_TOL * count
        prev = np.where(moving | (prev == -math.inf), ll, prev)
        keep = moving[g]
        if not keep.any():
            break
        r = np.exp(word[keep] - lp[sel][keep])
        prms, tags = _group_step(V, _sub(vs, keep), r, prms, tags, g[keep], nb)
    return prms, tags, np.bincount(group, lp, nb), np.bincount(group, back, nb)


# --- the search -----------------------------------------------------------------------------------------------------

def _overlap(V: _Votes, a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two words' supports (their boxes grown by the reach) share area."""
    pa, pb = _box(a, V.reach + 0.5), _box(b, V.reach + 0.5)
    if (pa.max(0) < pb.min(0)).any() or (pb.max(0) < pa.min(0)).any():
        return False
    return cv2.intersectConvexConvex(pa.astype(np.float32), pb.astype(np.float32))[0] > 0


class _Grid:
    """Keys by the axis-aligned bounds they cover, in square cells of INDEX_CELL px (speed only: a query returns
    exactly the keys a scan of every key's bounds would)."""

    def __init__(self):
        self.cells: dict[tuple, set] = {}
        self.spans: dict = {}

    def _cells(self, lo, hi):
        x0, y0 = (math.floor(float(lo[0]) / INDEX_CELL), math.floor(float(lo[1]) / INDEX_CELL))
        x1, y1 = (math.floor(float(hi[0]) / INDEX_CELL), math.floor(float(hi[1]) / INDEX_CELL))
        return [(i, j) for i in range(x0, x1 + 1) for j in range(y0, y1 + 1)]

    def add(self, key, lo, hi) -> None:
        self.spans[key] = (np.asarray(lo, np.float64), np.asarray(hi, np.float64))
        for c in self._cells(lo, hi):
            self.cells.setdefault(c, set()).add(key)

    def remove(self, key) -> None:
        lo, hi = self.spans.pop(key)
        for c in self._cells(lo, hi):
            self.cells[c].discard(key)

    def meeting(self, lo, hi) -> set:
        """The keys whose bounds meet [lo, hi]."""
        found = set()
        for c in self._cells(lo, hi):
            found |= self.cells.get(c, set())
        return {k for k in found if (self.spans[k][0] <= hi).all() and (lo <= self.spans[k][1]).all()}


class _Words:
    """Words by id, with their supports' bounds in a `_Grid`; `without(ids)` a view leaving some out.  `_near` reads it
    as it reads a list (the same words, in the order of their ids)."""

    def __init__(self, V: _Votes):
        self.V, self.words, self.grid, self.left_out = V, {}, _Grid(), frozenset()

    def add(self, k, word) -> None:
        self.words[k] = word
        self.grid.add(k, *self.V.bounds(word["prm"]))

    def remove(self, k) -> None:
        del self.words[k]
        self.grid.remove(k)

    def without(self, ids) -> "_Words":
        view = _Words.__new__(_Words)
        view.V, view.words, view.grid, view.left_out = self.V, self.words, self.grid, frozenset(ids)
        return view

    def near(self, lo, hi) -> list[dict]:
        return [self.words[k] for k in sorted(self.grid.meeting(lo, hi) - self.left_out)]

    def ids_near(self, lo, hi) -> list:
        return sorted(self.grid.meeting(lo, hi) - self.left_out)


def _total(V: _Votes, words: list[dict], prior: WordPrior, spreads: Spreads) -> float:
    """The configuration's total score, from the words' parameters (every vote; 0 off every support)."""
    region = _supports(V, words)
    return prior.slope * _score(V, region, words, []) / spreads.cell + prior.log_prior * len(words)


def _search(V: _Votes, P: _Proposals, spreads: Spreads, prior: WordPrior, record: list | None = None,
            scores: list | None = None) -> list[dict]:
    """The words (see the module): from none, add, split and join moves while one raises the total score, the best
    first (of equals, the first weighed), each move's change exact (its words refitted with every other word held).
    A move is weighed once and kept until a move made changes the votes it was weighed on (its region meets the
    changed one) or takes one of its words; then it is weighed again.  Words never change in place (a move makes new
    ones), so a pair whose supports do not meet is tested once.  `record` collects every move weighed: (kind, gain of
    the extra word (log likelihood / cell), [(votes, chances) of each part]).  `scores` collects the total score
    recomputed from the words' parameters at the start and after every move."""
    import heapq
    scale = prior.slope / spreads.cell
    proposed = [int(b) for b in np.argsort(-(P.ll - P.ll0), kind="stable")
                if scale * (P.ll[b] - P.ll0[b]) + prior.log_prior > 0]
    open_ = set(proposed)                           # proposals not yet made words
    index = _Words(V)
    next_id = 0
    moves: dict = {}                                # key: (delta, change, region, sequence)
    spans = _Grid()                                 # the moves by their regions' bounds
    of_word: dict = {}                              # word id: its moves' keys
    heap: list = []
    sequence = 0
    to_add, to_split, to_join = set(proposed), set(), set()

    def add(b):
        H = index.without(())
        new = _fit(V, [{"prm": P.prms[b], "tag": P.tags[b]}], H)
        d, region = _change(V, [], new, H)
        if record is not None:
            record.append(("word", d / spreads.cell, _parts(V, region, new, H)))
        return scale * d + prior.log_prior, ([], new), region

    def split(k):
        H = index.without((k,))
        word = index.words[k]
        region = V.support(word["prm"])
        if len(region) < 2:
            return -math.inf, None, region
        v = V.take(region)
        every = [word] + _near(V, H, *_span(V, region))
        lq, lf = _terms(V, v, every)
        _, resp = _mixture(v, lq, lf, V, _weights(every))
        # the seed: the vote of its support most left to the background, by core chance (a px the net calls a word's
        # whose vote no word holds: this word cannot, nor any other); of equals, the one it explains worst
        left = v["w"] * np.clip(1 - resp.sum(1), 0, 1)
        j = int(np.lexsort((np.where(np.isfinite(lf[:, 0]), lf[:, 0], 0.0), -left))[0])
        seed = {"prm": V.placements(_sub(v, slice(j, j + 1)))[0], "tag": v["t"][j], "a": float(word.get("a", 0.0))}
        new = _fit(V, [word, seed], H)
        d, region = _change(V, [word], new, H)
        if record is not None:
            record.append(("split", d / spreads.cell, _parts(V, region, new, H)))
        return scale * d + prior.log_prior, ([k], new), region

    def join(a, b):
        H = index.without((a, b))
        wa, wb = index.words[a], index.words[b]
        best = None
        for start in (wa, wb):                      # the one word, from each start; the better kept
            new = _fit(V, [start], H)
            d, region = _change(V, [wa, wb], new, H)
            if best is None or d > best[0]:
                best = (d, new, region)
        d, new, region = best
        if record is not None:
            record.append(("join", -d / spreads.cell, _parts(V, region, [wa, wb], H)))
        return scale * d - prior.log_prior, ([a, b], new), region     # joining: d, and one word fewer

    def keep(key, got):
        nonlocal sequence
        delta, change, region = got
        moves[key] = (delta, change, region, sequence)
        spans.add(key, *_span(V, region))
        for k in key[1:] if key[0] != "add" else ():
            of_word.setdefault(k, set()).add(key)
        heapq.heappush(heap, (-delta, sequence, key))
        sequence += 1

    def drop(key):
        moves.pop(key)
        spans.remove(key)
        for k in key[1:] if key[0] != "add" else ():
            of_word.get(k, set()).discard(key)

    if scores is not None:
        scores.append(_total(V, [], prior, spreads))
    while True:
        for b in [b for b in proposed if b in to_add and b in open_]:
            keep(("add", b), add(b))
        for k in sorted(to_split):
            if k in index.words:
                keep(("split", k), split(k))
        for a, b in sorted(to_join):
            if a in index.words and b in index.words and _overlap(V, index.words[a]["prm"], index.words[b]["prm"]):
                keep(("join", a, b), join(a, b))
        to_add, to_split, to_join = set(), set(), set()
        best = None
        while heap:
            neg, seq, key = heap[0]
            if key in moves and moves[key][3] == seq:
                best = key
                break
            heapq.heappop(heap)
        if best is None or not moves[best][0] > 0:
            break
        delta, (old, new), region, _ = moves[best]
        if best[0] == "add":
            open_.discard(best[1])
        changed = np.union1d(region, _supports(V, [index.words[k] for k in old]))
        lo, hi = _span(V, changed)
        stale = {best} | spans.meeting(lo, hi)
        for k in old:
            stale |= of_word.pop(k, set())
            index.remove(k)
        for key in stale:
            if key in moves:
                drop(key)
            if key[0] == "add":
                to_add.add(key[1])
            elif key[0] == "split":
                to_split.add(key[1])
            else:
                to_join.add(key[1:])
        for w in new:
            index.add(next_id, w)
            to_split.add(next_id)
            for k in index.ids_near(*index.grid.spans[next_id]):
                if k != next_id:
                    to_join.add((min(k, next_id), max(k, next_id)))
            next_id += 1
        if scores is not None:
            scores.append(_total(V, list(index.words.values()), prior, spreads))
    return [index.words[k] for k in sorted(index.words)]


def _final(V: _Votes, found: list[dict]) -> list[dict]:
    """The words refitted jointly, each connected set of words whose supports overlap at once, every other word held
    as it stands (a set's fit may move a support onto another set's px)."""
    n = len(found)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    index = _Words(V)
    for i, w in enumerate(found):
        index.add(i, w)
    for i in range(n):
        for j in index.ids_near(*index.grid.spans[i]):
            if j > i and _overlap(V, found[i]["prm"], found[j]["prm"]):
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    out = list(found)
    for members in groups.values():
        fitted = _fit(V, [out[i] for i in members], index.without(members))
        for i, w in zip(members, fitted):
            out[i] = w
            index.remove(i)
            index.add(i, w)
    return out


# --- rectangles -----------------------------------------------------------------------------------------------------

def _rect(c, v, n, a0, a1, b0, b1) -> np.ndarray:
    """The rectangle of centre c, axes v (along) and n (across, towards the feet), extents [a0, a1] x [b0, b1]: its
    corners in `boxes.word_rect`'s order."""
    return np.array([c + v * a0 + n * b0, c + v * a1 + n * b0, c + v * a1 + n * b1, c + v * a0 + n * b1])


def _frame(rect: np.ndarray):
    """A rectangle's centre, axes (its first edge's direction, and that turned towards the feet) and its corners'
    extents along them: c, v, n, (a0, a1, b0, b1)."""
    e = rect[1] - rect[0]
    norm = math.hypot(*e)
    v = e / norm if norm > 0 else np.array([1.0, 0.0])
    n = np.array([-v[1], v[0]])
    c = rect.mean(0)
    a, b = (rect - c) @ v, (rect - c) @ n
    return c, v, n, (a.min(), a.max(), b.min(), b.max())


def _clip_polygon(poly: np.ndarray, width: int, height: int) -> np.ndarray:
    """A convex polygon cut to the page [0, width] x [0, height] (Sutherland-Hodgman, float64)."""
    for axis, bound, below in ((0, 0.0, False), (0, float(width), True), (1, 0.0, False), (1, float(height), True)):
        if not len(poly):
            break
        inside = poly[:, axis] <= bound if below else poly[:, axis] >= bound
        out = []
        for k in range(len(poly)):
            a, b = poly[k], poly[(k + 1) % len(poly)]
            if inside[k]:
                out.append(a)
            if inside[k] != inside[(k + 1) % len(poly)]:
                out.append(a + (bound - a[axis]) / (b[axis] - a[axis]) * (b - a))
        poly = np.array(out).reshape(-1, 2)
    return poly


def _on_page(rect: np.ndarray, width: int, height: int) -> np.ndarray:
    """The box clipped to the page (see the module); a box with nothing on the page keeps its corners moved onto
    it."""
    if (rect >= 0).all() and (rect[:, 0] <= width).all() and (rect[:, 1] <= height).all():
        return rect
    pts = _clip_polygon(rect, width, height)
    if not len(pts):
        return np.clip(rect, 0, [width, height])
    c, v, n, _ = _frame(rect)
    a, b = (pts - c) @ v, (pts - c) @ n
    return _rect(c, v, n, a.min(), a.max(), b.min(), b.max())


def _inked(rect: np.ndarray, ink: np.ndarray, kind_p: np.ndarray | None):
    """The trimmed box of a box (see the module), its ink px and their mean kind probabilities (4,)."""
    h, w = ink.shape
    x0, y0 = np.maximum(np.floor(rect.min(0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(rect.max(0)).astype(int), [w, h])
    kinds = np.zeros(4)
    if x0 >= x1 or y0 >= y1:
        return rect, 0, kinds
    yy, xx = np.nonzero(ink[y0:y1, x0:x1])
    c, v, n, (a0, a1, b0, b1) = _frame(rect)
    p = np.stack([xx + x0 + 0.5, yy + y0 + 0.5], -1)
    a, b = (p - c) @ v, (p - c) @ n
    inside = (a >= a0) & (a <= a1) & (b >= b0) & (b <= b1)
    count = int(inside.sum())
    if not count:
        return rect, 0, kinds
    a, b = a[inside], b[inside]
    if kind_p is not None:
        kinds = kind_p[:, yy[inside] + y0, xx[inside] + x0].astype(np.float64).mean(1)
    t = _rect(c, v, n, max(a0, a.min() - 0.5), min(a1, a.max() + 0.5), max(b0, b.min() - 0.5), min(b1, b.max() + 0.5))
    return t, count, kinds




# --- the words of a page -------------------------------------------------------------------------------------------

def _table(rows: list[dict]) -> pd.DataFrame:
    """The words' rows as the table (wordboxes.COLUMNS then EXTRA_COLUMNS), typed as `wordboxes.build_boxes` types
    them."""
    floats = ("level", "page_x", "page_y", "axis", "length", "height", "t_length", "t_height", "mean_chance", "x",
              "y", *EXTRA_COLUMNS, *wordboxes.CORNER_COLS, *[f"p_{k}" for k in wordboxes.KIND_NAMES])
    return pd.DataFrame(rows, columns=[*wordboxes.COLUMNS, *EXTRA_COLUMNS]).astype(
        {"i": "int32", "quarter": "int8", "lab_lot": "int32", "street": bool, "ink_px": "int64",
         **{c: "float64" for c in floats}})


def _inputs(core_logit, geo, tag) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    core_logit, geo, tag = np.asarray(core_logit), np.asarray(geo), np.asarray(tag)
    if core_logit.ndim != 2:
        raise ValueError(f"core_logit is (h, w), not {core_logit.shape}")
    h, w = core_logit.shape
    if geo.ndim != 3 or geo.shape[0] != 5 or geo.shape[1:] != core_logit.shape:
        raise ValueError(f"geo is (5, {h}, {w}) as model.word_geometry decodes it (offset x, offset y, length, height, "
                         f"axis degrees), not {geo.shape}")
    if tag.ndim != 3 or tag.shape[1:] != core_logit.shape:
        raise ValueError(f"tag is (k, {h}, {w}), not {tag.shape}")
    _check_geo(geo)
    return core_logit, geo, tag


def _check_geo(geo: np.ndarray) -> None:
    """A vote's offset and axis are finite numbers and its length and height positive finite ones (their logs are
    read)."""
    for lo in range(0, geo.shape[1], max(CHUNK_PX // max(geo.shape[2], 1), 1)):
        g = geo[:, lo:lo + max(CHUNK_PX // max(geo.shape[2], 1), 1)]
        if not (np.isfinite(g).all() and (g[2:4] > 0).all()):
            raise ValueError("geo's offsets and axis are finite and its lengths and heights positive and finite")


def core_words(core_logit, geo, tag, ink, spreads: Spreads, prior: WordPrior, *, kind_p: np.ndarray | None = None,
               sheet: str = "") -> pd.DataFrame:
    """One page's word boxes from the word heads (see the module): `core_logit` (h, w), `geo` (5, h, w), `tag`
    (k, h, w), `ink` (h, w) bool; `spreads` measured and `prior` fitted (Task 8); `kind_p` (4, h, w) the kinds'
    probabilities (optional); `sheet` written to the table.  A DataFrame of wordboxes.COLUMNS then EXTRA_COLUMNS, one
    row per word."""
    core_logit, geo, tag = _inputs(core_logit, geo, tag)
    if not (math.isfinite(prior.slope) and math.isfinite(prior.log_prior)):
        raise ValueError(f"the prior's slope and log prior are finite numbers, not {prior}")
    height, width = core_logit.shape
    ink = np.asarray(ink, bool)
    V = _Votes(core_logit, geo, tag, spreads)
    found = _final(V, _search(V, _proposals(V, spreads), spreads, prior))
    index = _Words(V)
    for k, word in enumerate(found):
        index.add(k, word)
    rows, order = [], []
    for k, word in enumerate(found):
        region = V.support(word["prm"])
        others = index.without((k,))
        (idx, r), = _parts(V, region, [word], others)
        with_k = _score(V, region, [word], others)
        without = _score(V, region, [], others)
        v = V.take(idx)
        g = _on_page(_box(word["prm"]), width, height)
        t, ink_px, kinds = _inked(g, ink, kind_p)
        _, vv, _, (a0, a1, b0, b1) = _frame(g)
        _, _, _, (ta0, ta1, tb0, tb1) = _frame(t)
        votes = float(r.sum())
        mass = float((r * v["w"]).sum())
        p_word = mass / votes if votes > 0 else 0.0
        spread = (math.sqrt(float((r * ((v["t"] - word["tag"]) ** 2).sum(1)).sum()) / (votes * V.k))
                  if votes > 0 else 0.0)
        centre = g.mean(0)
        order.append(int((V.ys[idx] * width + V.xs[idx]).min()) if len(idx) else 0)
        rows.append({"sheet": sheet, "level": math.nan, "quarter": -1,
                     "page_x": float(centre[0]), "page_y": float(centre[1]),
                     "axis": float(np.rad2deg(np.arctan2(-vv[1], vv[0])) % 180.0),
                     "length": float(a1 - a0), "height": float(b1 - b0),
                     "t_length": float(ta1 - ta0), "t_height": float(tb1 - tb0),
                     "mean_chance": p_word, "ink_px": ink_px,
                     **{f"p_{kn}": float(kinds[c]) for c, kn in enumerate(wordboxes.KIND_NAMES)},
                     "x": math.nan, "y": math.nan, "lab_lot": -1, "street": False,
                     **dict(zip(wordboxes.CORNER_COLS, [*g.reshape(8).tolist(), *t.reshape(8).tolist()])),
                     "p_word": p_word, "core_mass": mass, "tag_spread": spread,
                     "word_logodds": float(prior.logodds((with_k - without) / spreads.cell))})
    rows = [rows[j] for j in np.argsort(order, kind="stable")]
    for i, row in enumerate(rows):
        row["i"] = i
    return _table(rows)


# --- the prior: the decisions, labelled by Terra's words, and their fit (Task 8's) ---------------------------------

def _terra_shares(V: _Votes, word_id: np.ndarray, idx: np.ndarray, r: np.ndarray) -> tuple[float, dict]:
    """Of votes `idx` weighted by their chances `r` and core chance: the share inside Terra's words, and the shares
    of each Terra word among those inside."""
    w = V.chance(idx) * r
    t = word_id[V.ys[idx], V.xs[idx]]
    total = float(w.sum())
    inside = t > 0
    word_mass = float(w[inside].sum())
    if not (total > 0 and word_mass > 0):
        return 0.0, {}
    ids, at = np.unique(t[inside], return_inverse=True)
    return word_mass / total, dict(zip(ids.tolist(), (np.bincount(at, w[inside]) / word_mass).tolist()))


def _apart(V: _Votes, word_id, a: tuple, b: tuple) -> float:
    """The label of an extra word between parts a and b ((votes, chances) each): the chance both are words' and are
    different Terra words, from their votes' weighted shares (no cut)."""
    ma, sa = _terra_shares(V, word_id, *a)
    mb, sb = _terra_shares(V, word_id, *b)
    return ma * mb * (1 - sum(p * sb.get(t, 0.0) for t, p in sa.items()))


def prior_decisions(core_logit, geo, tag, spreads: Spreads, word_id, prior: WordPrior | None = None) -> pd.DataFrame:
    """The decisions `core_words` makes on a page, each an extra word's gain (its log likelihood ratio / cell) and
    its label from Terra's words (`word_id`: 1 + the Terra word whose rectangle holds the px, 0 none), a chance with
    no cut.  Kind "proposal" for every basin (its own votes as one word against none: whether it is proposed; labelled
    by the share of its votes' core-weighted mass inside Terra's words) and, with a `prior`, "word", "split" and
    "join" for every add, split and join the search weighs ("word" labelled as a proposal is, by its votes' chances of
    being its; "split" and "join" by the chance their two parts are different Terra words, `_apart`).  Task 8: the
    "proposal" decisions alone fit a first prior, with which the search's decisions are taken, and all of them fit the
    prior (`fit_word_prior`); on the training quarters only."""
    core_logit, geo, tag = _inputs(core_logit, geo, tag)
    word_id = np.asarray(word_id)
    V = _Votes(core_logit, geo, tag, spreads)
    P = _proposals(V, spreads)
    inside = (word_id[V.ys, V.xs] > 0).astype(np.float64)
    w = V.chance()
    total = np.bincount(P.basin, w, len(P.first))
    shares = np.bincount(P.basin, w * inside, len(P.first)) / np.where(total > 0, total, 1)
    out = pd.DataFrame({"kind": "proposal", "gain": (P.ll - P.ll0) / spreads.cell, "label": shares})
    if prior is not None:
        record = []
        _search(V, P, spreads, prior, record)
        rows = []
        for kind, gain, parts in record:
            label = (_terra_shares(V, word_id, *parts[0])[0] if kind == "word"
                     else _apart(V, word_id, parts[0], parts[1]))
            rows.append({"kind": kind, "gain": float(gain), "label": float(label)})
        out = pd.concat([out, pd.DataFrame(rows, columns=["kind", "gain", "label"])], ignore_index=True)
    return out


def fit_word_prior(gains, labels) -> WordPrior:
    """The decision to have a word as a fitted logit: P(the extra word is real) = sigmoid(slope x gain + log_prior),
    the gains and the labels (chances, 0 to 1: a fractional logistic regression) from `prior_decisions`; the maximum
    likelihood fit (the log likelihood is concave), by Newton steps.  ValueError when no finite fit exists (every label
    0, or every label 1, or the steps do not settle)."""
    g = np.asarray(gains, np.float64)
    y = np.asarray(labels, np.float64)
    if len(g) != len(y) or not len(g) or not np.isfinite(g).all():
        raise ValueError("gains and labels are one finite number each per decision")
    some, short = g[y > 0], g[y < 1]            # decisions that are words at all, and that are not wholly words
    if not len(some) or not len(short) or short.max() <= some.min() or some.max() <= short.min():
        raise ValueError("the labels part the gains by a threshold (or are all 0, or all 1): no finite fit")
    scale = float(np.std(g)) or 1.0
    x = np.stack([g / scale, np.ones_like(g)], -1)

    def loglik(beta):
        z = x @ beta
        return float((y * -np.logaddexp(0, -z) + (1 - y) * -np.logaddexp(0, z)).sum())

    beta = np.zeros(2)
    current = loglik(beta)
    for _ in range(FIT_STEPS):
        p = 0.5 * (1 + np.tanh((x @ beta) / 2))
        grad = x.T @ (y - p)
        hess = -(x * (p * (1 - p))[:, None]).T @ x
        try:
            step = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        while loglik(beta + t * step) < current and t > np.finfo(float).eps:
            t /= 2
        beta = beta + t * step
        new = loglik(beta)
        settled = abs(new - current) <= np.finfo(float).eps * max(1.0, abs(new)) * len(g)
        current = new
        if settled:
            if not np.isfinite(beta).all():
                break
            return WordPrior(slope=float(beta[0] / scale), log_prior=float(beta[1]))
    raise ValueError("the fit does not settle: no finite slope and log prior fit these decisions")




# --- measuring the spreads (Task 8's, on the training quarters) ----------------------------------------------------

def _lattice(ell: float) -> np.ndarray:
    """exp(-k^2 / (2 ell^2)) over the integers k, out to where it underflows (the span numerical: beyond, every term
    is 0 in double precision)."""
    span = int(math.ceil(ell * math.sqrt(2 * 745.0))) + 1
    k = np.arange(-span, span + 1, dtype=np.float64)
    return np.exp(-k ** 2 / (2 * ell ** 2))


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """sum_p a(p) b(p + lag) for every lag (2h - 1, 2w - 1), lag 0 at the centre."""
    return fftconvolve(a[::-1, ::-1], b, mode="full")


def _rect_params(rect: np.ndarray) -> np.ndarray:
    """A rectangle's parameters (cx, cy, log length, log height, theta): its centre, its first edge's length and
    direction (modulo a half turn), its second edge's length (a degenerate side RECT_FLOOR, numerical)."""
    e1, e2 = rect[1] - rect[0], rect[3] - rect[0]
    theta = math.atan2(-e1[1], e1[0])
    return np.r_[rect.mean(0), math.log(max(math.hypot(*e1), RECT_FLOOR)), math.log(max(math.hypot(*e2), RECT_FLOOR)),
                 theta]


def _t_fit(dev: np.ndarray, dof_count: float, scale: float, dof: float):
    """A Student-t's scale and dof by maximum likelihood on deviations `dev` (each about its word's location),
    from (scale, dof): one EM pass of the scale (the t's weights at the current values; the deviations' degrees of
    freedom `dof_count`, not their number, as each word's location took some), then the dof at that scale: its
    search on its log between DOF_BOUNDS, or the Gaussian (inf) where that is likelier (the supremum past the
    bound).  The scale, the dof, the weights and the log likelihood."""
    w = _t_weight((dev / scale) ** 2, dof)
    scale = math.sqrt(float((w * dev ** 2).sum()) / dof_count)
    z2 = (dev / scale) ** 2

    def nll(d):
        return -(len(dev) * (_t_log_norm(d) - math.log(scale)) - float(_t_rho(z2, d).sum()))

    lo, hi = math.log(DOF_BOUNDS[0]), math.log(DOF_BOUNDS[1])
    dof = math.exp(minimize_scalar(lambda x: nll(math.exp(x)), bounds=(lo, hi), method="bounded").x)
    if nll(math.inf) <= nll(dof):
        dof = math.inf
    return scale, dof, _t_weight(z2, dof), -nll(dof)


def _locations(e: np.ndarray, w: np.ndarray, word: np.ndarray, count: int, axis: bool = False) -> np.ndarray:
    """Each word's location of its errors `e`, weighted by `w` (the axis's taken as a turn's half, about each word's
    first error)."""
    if axis:
        ref = np.zeros(count)
        first = np.unique(word, return_index=True)[1]
        ref[word[first]] = e[first]
        d = _half(e - ref[word])
        return ref + np.bincount(word, w * d, count) / np.bincount(word, w, count)
    return np.bincount(word, w * e, count) / np.bincount(word, w, count)


def _edge_terms(p: np.ndarray, prm: np.ndarray, edge: float, a_in: np.ndarray, a_out: np.ndarray):
    """A word's core px's log likelihood about its box (`prm`) at edge spread `edge`: each px in the box with chance
    q, its core evidence a_in = log w / core_share in it and `a_out` out of it, so log(q e^a_in + (1 - q) e^a_out);
    and its gradient in the box's parameters.  (`_measure_edge` gives as a_out the evidence of a px out of this box,
    which may be in another at the base rate.)"""
    along, up = _axes(prm[4])
    rel = p - prm[:2]
    pa, pb = rel @ along, rel @ up
    half_a, half_b = math.exp(prm[2]) / 2 + 0.5, math.exp(prm[3]) / 2 + 0.5
    lqa, dqa_x, dqa_half, lta, dta_x, dta_half = _log_interval_d(pa, half_a, edge)
    lqb, dqb_x, dqb_half, ltb, dtb_x, dtb_half = _log_interval_d(pb, half_b, edge)
    lq = lqa + lqb
    l1q = np.logaddexp(lta, lqa + ltb)
    num = np.logaddexp(lq + a_in, l1q + a_out)
    value = float(num.sum())
    # dL / L = q e^a_in / L d log q + (1 - q) e^a_out / L d log(1 - q), the second through each axis's tail
    wq, wt = np.exp(lq + a_in - num), np.exp(l1q + a_out - num)
    sa_, sb_ = np.exp(lqb + lta - l1q), np.exp(lqa + ltb - l1q)
    da = wq * dqa_x + wt * sa_ * dta_x
    db = wq * dqb_x + wt * sb_ * dtb_x
    g = np.empty(5)
    g[:2] = -(da.sum() * along + db.sum() * up)
    g[2] = float((wq * dqa_half + wt * sa_ * dta_half).sum()) * math.exp(prm[2]) / 2
    g[3] = float((wq * dqb_half + wt * sb_ * dtb_half).sum()) * math.exp(prm[3]) / 2
    g[4] = float((da * pb - db * pa).sum())
    return value, g


def _measure_edge(items: list[tuple], core_share: float) -> tuple[float, int]:
    """The edge spread (see `measure_spreads`): sqrt(1/12 + b^2), a px's own extent (a unit square, its position
    uniform within it: 1/12 px^2 a side) and b, the spread of the core px's ends about the boxes beyond it.  Each word's
    box fitted to its core px at the edge spread (L-BFGS-B on `_edge_terms`, from Terra's rectangle), then b to the
    most likelihood over every word at those boxes (its log searched between 1/1000 px and the longest box side), in
    turn until the likelihood moves by no more than EM_TOL a px (RuntimeError after SPREAD_ROUNDS).  Each word's px
    are those of Terra's rectangle grown by its height that are in the mask and in no other Terra word, read as
    `core_words` reads them (in the box, or in none).  A word whose reach at the spread found (where q times the most
    any of its px not its own can favour its box, as the px past them may, falls below double precision) passes the
    px gathered is left out, as a word cut by the mask is (its evidence not wholly read), and the spread found again
    without it.  The spread and the number of words left out."""
    from scipy.optimize import minimize
    lpi, lnpi = math.log(core_share), math.log1p(-core_share)
    prepared = []
    for p, x, own, start, margin in items:
        a_in = -np.logaddexp(0, -x) - lpi
        a_out = -np.logaddexp(0, x) - lnpi
        # the most a px about the word (not its own) can favour a box: what the px past the margin may do
        with np.errstate(invalid="ignore"):
            k = a_in[~own] - a_out[~own]
        most = float(np.nanmax(k)) if len(k) and np.isfinite(k).any() else -math.inf
        prepared.append((p, a_in, a_out, np.array(start, np.float64), margin, most))
    left_out = 0
    while True:
        boxes = [s for _, _, _, s, _, _ in prepared]
        longest = max(math.exp(max(b[2], b[3])) for b in boxes)
        n_px = sum(len(p) for p, *_ in prepared)

        def total(edge):
            return sum(_edge_terms(p, boxes[k], edge, a_in, a_out)[0]
                       for k, (p, a_in, a_out, *_) in enumerate(prepared))

        edge = math.sqrt(1 / 12)
        ll = -math.inf
        for _ in range(SPREAD_ROUNDS):
            for k, (p, a_in, a_out, *_) in enumerate(prepared):
                got = minimize(lambda prm: tuple(-x for x in _edge_terms(p, prm, edge, a_in, a_out)), boxes[k],
                               jac=True, method="L-BFGS-B")
                if got.fun <= -_edge_terms(p, boxes[k], edge, a_in, a_out)[0]:
                    boxes[k] = got.x
            log_b = minimize_scalar(lambda lb: -total(math.sqrt(1 / 12 + math.exp(2 * lb))),
                                    bounds=(math.log(1e-3), math.log(longest)), method="bounded").x
            edge = math.sqrt(1 / 12 + math.exp(2 * log_b))
            new = total(edge)
            settled = abs(new - ll) <= EM_TOL * n_px
            ll = new
            if settled:
                break
        else:
            raise RuntimeError(f"the edge spread did not settle in {SPREAD_ROUNDS} rounds (last {edge})")
        whole = [item for item in prepared
                 if not (math.isfinite(item[5])
                         and edge * -float(ndtri_exp(min(math.log(PRECISION) - item[5], 0.0))) > item[4])]
        if len(whole) == len(prepared):
            return edge, left_out
        if not whole:
            raise ValueError("every word's reach passes the px gathered about it: no edge to measure")
        left_out += len(prepared) - len(whole)
        prepared = whole


def measure_spreads(pages) -> Spreads:
    """The spreads (see the module), measured on predictions against Terra's words, never the owner's answers.
    `pages`: dicts of geo, tag (as `core_words` takes them: geo as `model.word_geometry` decodes it), word_id (int
    (h, w): 1 + the Terra word whose rectangle holds the px, later over earlier, 0 on none: the targets' `tag_id`),
    words (float (n, 4, 2): Terra's rectangles) and, optionally, mask (bool (h, w): the px that may be read; none,
    every px) and core (float (h, w): the core logit, for the edge; none, a px's core chance is 1 in a word, 0 out).
    Only the mask's px are read: a word with a px outside it is left out whole (a part of a word is no word's
    measure), the background is the px of the mask in no word, and the core share is counted on the mask's px.
      core_share  the share of the pages' px in Terra's words (the core head's base rate)
      calibration each of the offset x and y (in heights: (c - p) / H true), the log length and the log height
                  against Terra's words over every word px: the forward fit pred = alpha + beta true by least squares
                  (`*_bias`, `*_slope`), and the true values' mean and sd (`*_truth`, the point placements); the axis
                  its mean error modulo 180 (`angle_bias`).  ValueError where the true values do not vary (no slope).
      Each vote's error e = pred - alpha - beta true (the axis's modulo a half turn; the tag's the tag itself) is a
      word's shared error plus the vote's own: each word's location of its errors (the t-weighted mean) and the
      deviations about it, per dimension:
      scale, dof  the Student-t of the deviations by maximum likelihood (`_t_fit`), each word's degrees of freedom n
                  less its correlated share (the sum of rho over its px pairs / n); the locations and the t in turn
                  until their log likelihood moves by no more than EM_TOL a deviation (RuntimeError after
                  SPREAD_ROUNDS): `centre`
                  (x and y pooled), `length`, `height`, `angle`, `tag` (channels pooled) and their `*_dof`
      *_word      the shared errors' sd: the words' locations' mean square less their noise's (the t location's
                  variance s^2 (dof + 3) / (dof + 1) times the word's sum of rho / n^2), at least 0; not for the tag
                  (a word's mean tag is its own)
      cell        from the variogram of the offset's deviations within words, over every lag: half the mean squared
                  difference at lag (dy, dx) is sigma^2 (1 - rho), rho Gaussian in the lag of scale ell, fitted by
                  least squares weighted by the pairs (ell searched between numerical bounds: 1/100 px and the
                  longest lag); cell = the sum of rho over the lattice (independent votes give 1)
      edge        where a word's core px end about its box (`_measure_edge`): the core head's evidence about each
                  word's box fitted to it, as `core_words` reads it (a px in the box with chance q, convolved with the
                  edge spread), the spread that makes the core px likeliest: a px's own extent (1/12 px^2) and the
                  core px's ends' spread beyond it, in quadrature
      background_centre, background_length, background_height, background_tag_mean, background_tag   a background
                  px's offsets' root mean square about 0; its log length's and log height's mean and sd; its tags'
                  mean and root mean square about it (the votes as the net gives them)"""
    per_word = []                      # (v, Terra's parameters, mask, its px within the mask, pair counts)
    edge_items = []                    # (px, core logit, the word's own px, Terra's parameters, margin)
    acc = {"o2": 0.0, "n": 0, "ll": 0.0, "ll2": 0.0, "lh": 0.0, "lh2": 0.0, "t": None, "t2": None}
    n_in = n_all = 0
    channels = None
    for page in pages:
        geo, tag = np.asarray(page["geo"]), np.asarray(page["tag"])
        word_id = np.asarray(page["word_id"])
        on = page.get("mask")
        on = np.ones(word_id.shape, bool) if on is None else np.asarray(on, bool)
        if on.shape != word_id.shape:
            raise ValueError(f"a page's mask is the page's shape {word_id.shape}, not {on.shape}")
        core = page.get("core")
        if core is not None and np.shape(core) != word_id.shape:
            raise ValueError(f"a page's core is the page's shape {word_id.shape}, not {np.shape(core)}")
        rects = np.asarray(page["words"], np.float64).reshape(-1, 4, 2)
        channels = tag.shape[0]
        n_all += int(on.sum())
        n_in += int((on & (word_id > 0)).sum())
        h_page, w_page = word_id.shape
        # the words, each whole: a word with a px outside the mask is left out with all its px (each word's px in
        # row order, as np.nonzero gives them)
        wy, wx = np.nonzero(word_id > 0)
        ids = word_id[wy, wx]
        whole = ~np.isin(ids, np.unique(word_id[~on & (word_id > 0)]))
        order = np.argsort(ids[whole], kind="stable")
        wy, wx, ids = wy[whole][order], wx[whole][order], ids[whole][order]
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]) if len(ids) else np.zeros(0, np.int64)
        for j, lo, hi in zip(ids[starts], starts, np.r_[starts[1:], len(ids)]):
            ys, xs = wy[lo:hi], wx[lo:hi]
            g = geo[:, ys, xs].astype(np.float64)
            if not (np.isfinite(g).all() and (g[2:4] > 0).all()):
                raise ValueError("geo's offsets and axis are finite and its lengths and heights positive and finite")
            v = {"p": np.stack([xs, ys], -1).astype(np.float64), "o": (g[:2] / g[3]).T, "ll": np.log(g[2]),
                 "lh": np.log(g[3]), "th": np.deg2rad(g[4]),
                 "t": np.moveaxis(tag[:, ys, xs].astype(np.float64), 0, -1)}
            truth = _rect_params(rects[j - 1])
            y0, x0 = ys.min(), xs.min()
            m = np.zeros((ys.max() - y0 + 1, xs.max() - x0 + 1))
            m[ys - y0, xs - x0] = 1
            per_word.append((v, truth, m, (ys - y0, xs - x0), np.rint(_corr(m, m))))
            # its core px for the edge: Terra's rectangle grown by its height, in the mask, in no other word
            margin = math.exp(truth[3])
            corners = _box(truth, margin + 0.5)
            bx0, by0 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
            bx1, by1 = np.minimum(np.ceil(corners.max(0)).astype(int) + 1, [w_page, h_page])
            yy, xx = np.mgrid[by0:by1, bx0:bx1]
            near = on[by0:by1, bx0:bx1] & np.isin(word_id[by0:by1, bx0:bx1], (0, j))
            yy, xx = yy[near], xx[near]
            own = word_id[yy, xx] == j
            x = np.asarray(core)[yy, xx].astype(np.float64) if core is not None else np.where(own, math.inf, -math.inf)
            edge_items.append((np.stack([xx, yy], -1).astype(np.float64), x, own, truth, margin))
        by, bx = np.nonzero((word_id == 0) & on)
        for lo in range(0, len(by), CHUNK_PX):
            yy, xx = by[lo:lo + CHUNK_PX], bx[lo:lo + CHUNK_PX]
            g = geo[:4, yy, xx].astype(np.float64)
            tt = tag[:, yy, xx].astype(np.float64)
            o, ll, lh = g[:2] / g[3], np.log(g[2]), np.log(g[3])
            acc["o2"] += float((o ** 2).sum())
            acc["n"] += len(yy)
            acc["ll"] += float(ll.sum())
            acc["ll2"] += float((ll ** 2).sum())
            acc["lh"] += float(lh.sum())
            acc["lh2"] += float((lh ** 2).sum())
            acc["t"] = tt.sum(1) if acc["t"] is None else acc["t"] + tt.sum(1)
            acc["t2"] = (tt ** 2).sum(1) if acc["t2"] is None else acc["t2"] + (tt ** 2).sum(1)
    n = acc["n"]
    if not per_word or not n:
        raise ValueError("the pages hold no whole word or no background px to measure")

    def mean_sd(total, squares):
        mean = total / n
        return (mean, math.sqrt(max(squares / n - mean ** 2, 0.0)))

    tmean = acc["t"] / n
    background = {"background_centre": math.sqrt(acc["o2"] / (2 * n)),
                  "background_length": mean_sd(acc["ll"], acc["ll2"]),
                  "background_height": mean_sd(acc["lh"], acc["lh2"]),
                  "background_tag": math.sqrt(float((acc["t2"] / n - tmean ** 2).sum()) / channels),
                  "background_tag_mean": tuple(float(x) for x in tmean)}
    core_share = n_in / n_all

    # the calibration: every word px's prediction against its Terra word's true value
    word = np.concatenate([np.full(len(v["ll"]), k) for k, (v, *_) in enumerate(per_word)])
    nw = len(per_word)
    cat = {key: np.concatenate([v[key] for v, *_ in per_word]) for key in ("p", "o", "ll", "lh", "th", "t")}
    truths = np.array([t for _, t, *_ in per_word])
    T = truths[word]
    true = {"x": (T[:, 0] - cat["p"][:, 0]) * np.exp(-T[:, 3]), "y": (T[:, 1] - cat["p"][:, 1]) * np.exp(-T[:, 3]),
            "ll": T[:, 2], "lh": T[:, 3]}
    pred = {"x": cat["o"][:, 0], "y": cat["o"][:, 1], "ll": cat["ll"], "lh": cat["lh"]}
    cal = {}
    for key in ("x", "y", "ll", "lh"):
        t, q = true[key], pred[key]
        vt = float(np.mean((t - t.mean()) ** 2))
        if not vt > 0:
            raise ValueError(f"the words' true {key} do not vary: no calibration slope")
        beta = float(np.mean((t - t.mean()) * (q - q.mean()))) / vt
        cal[key] = (float(q.mean()) - beta * float(t.mean()), beta, (float(t.mean()), math.sqrt(vt)))
    angle_bias = float(np.mean(_half(cat["th"] - T[:, 4])))
    errs = {key: pred[key] - cal[key][0] - cal[key][1] * true[key] for key in cal}
    errs["th"] = _half(cat["th"] - angle_bias - T[:, 4])

    # the variogram of the offset's deviations within words (each word's location cancels in its differences)
    loc0 = {key: _locations(errs[key], np.ones(len(word)), word, nw) for key in ("x", "y")}
    lag_n, lag_d = {}, {}
    starts = np.r_[0, np.cumsum([len(v["ll"]) for v, *_ in per_word])]
    for k, (v, _, m, (ry, rx), n_lag) in enumerate(per_word):
        sl = slice(starts[k], starts[k + 1])
        d_lag = np.zeros_like(n_lag)
        for key in ("x", "y"):
            f = np.zeros_like(m)
            f[ry, rx] = errs[key][sl] - loc0[key][k]
            d_lag += _corr(m, m * f * f) + _corr(m * f * f, m) - 2 * _corr(m * f, m * f)
        lag_n[n_lag.shape] = lag_n.get(n_lag.shape, 0) + n_lag
        lag_d[n_lag.shape] = lag_d.get(n_lag.shape, 0) + d_lag
    hh = max(key[0] for key in lag_n) // 2
    ww = max(key[1] for key in lag_n) // 2
    pairs = np.zeros((2 * hh + 1, 2 * ww + 1))
    sq = np.zeros_like(pairs)
    for (h, w), nn in lag_n.items():
        sl2 = (slice(hh - h // 2, hh + h // 2 + 1), slice(ww - w // 2, ww + w // 2 + 1))
        pairs[sl2] += nn
        sq[sl2] += lag_d[(h, w)]
    pairs[hh, ww] = 0                                    # lag 0: no difference
    dy, dx = np.nonzero(pairs > 0)
    if not len(dy):
        raise ValueError("the pages' words hold no pair of px: no variogram")
    weight = pairs[dy, dx]
    gamma = sq[dy, dx] / (2 * 2 * weight)
    dist2 = (dy - hh) ** 2.0 + (dx - ww) ** 2.0

    def fitted(log_ell):
        f = 1 - np.exp(-dist2 / (2 * math.exp(2 * log_ell)))
        sill = float((weight * gamma * f).sum() / (weight * f * f).sum())
        return sill, float((weight * (gamma - sill * f) ** 2).sum())

    extent = math.log(max(math.sqrt(float(dist2.max())), 1.0))
    log_ell = minimize_scalar(lambda le: fitted(le)[1], bounds=(math.log(1e-2), extent), method="bounded").x
    ell = math.exp(log_ell)
    cell = float(_lattice(ell).sum() ** 2)
    # each word's correlated share: the sum of rho over its px pairs (its n_lag) / n
    rho_sum = np.empty(nw)
    for k, (_, _, _, _, n_lag) in enumerate(per_word):
        h, w = n_lag.shape
        yy, xx = np.mgrid[-(h // 2):h // 2 + 1, -(w // 2):w // 2 + 1]
        rho_sum[k] = float((n_lag * np.exp(-(yy ** 2 + xx ** 2) / (2 * ell ** 2))).sum())
    n_w = np.bincount(word, minlength=nw).astype(np.float64)
    dof_words = float(np.maximum(n_w - rho_sum / n_w, 0.0).sum())       # n less its correlated share, each word

    # the t of each dimension's deviations about its words' locations, and the locations, to a fixed point
    groups = {"centre": ("x", "y"), "length": ("ll",), "height": ("lh",), "angle": ("th",)}
    tags = cat["t"]
    fits = {}
    for dim, keys in [*groups.items(), ("tag", None)]:
        if dim == "tag":
            es = [tags[:, c] for c in range(channels)]
        else:
            es = [errs[key] for key in keys]
        count = dof_words * len(es)
        scale = math.sqrt(float(np.mean(np.concatenate(es) ** 2))) or 1.0
        dof, ws, ll = math.inf, [np.ones(len(word)) for _ in es], -math.inf
        for _ in range(SPREAD_ROUNDS):
            locs = [_locations(e, w, word, nw, axis=dim == "angle") for e, w in zip(es, ws)]
            devs = [(_half(e - loc[word]) if dim == "angle" else e - loc[word]) for e, loc in zip(es, locs)]
            dev = np.concatenate(devs)
            scale, dof, w_all, new_ll = _t_fit(dev, count, scale, dof)
            ws = np.split(w_all, len(es))
            settled = abs(new_ll - ll) <= EM_TOL * len(dev)
            ll = new_ll
            if settled:
                break
        else:
            raise RuntimeError(f"the {dim} spread did not settle in {SPREAD_ROUNDS} rounds (last {scale}, dof {dof})")
        fits[dim] = (scale, dof, locs)
    shared = {}
    for dim in groups:
        scale, dof, locs = fits[dim]
        noise = scale ** 2 * (dof + 3) / (dof + 1) if math.isfinite(dof) else scale ** 2
        var_loc = noise * rho_sum / n_w ** 2
        mean_sq = float(np.mean(np.concatenate([loc ** 2 for loc in locs])))
        shared[dim] = math.sqrt(max(mean_sq - float(np.mean(var_loc)), 0.0))
    edge, edge_left_out = _measure_edge(edge_items, core_share)
    MEASURED.update(words=len(per_word), edge_left_out=edge_left_out)
    deg = math.degrees
    return Spreads(centre=fits["centre"][0], length=fits["length"][0], height=fits["height"][0],
                   angle=deg(fits["angle"][0]), tag=fits["tag"][0], cell=cell, edge=edge, core_share=core_share,
                   centre_bias=(cal["x"][0], cal["y"][0]), length_bias=cal["ll"][0], height_bias=cal["lh"][0],
                   angle_bias=deg(angle_bias), **background,
                   centre_slope=(cal["x"][1], cal["y"][1]), length_slope=cal["ll"][1], height_slope=cal["lh"][1],
                   centre_dof=fits["centre"][1], length_dof=fits["length"][1], height_dof=fits["height"][1],
                   angle_dof=fits["angle"][1], tag_dof=fits["tag"][1],
                   centre_word=shared["centre"], length_word=shared["length"], height_word=shared["height"],
                   angle_word=deg(shared["angle"]),
                   centre_truth=(cal["x"][2], cal["y"][2]), length_truth=cal["ll"][2], height_truth=cal["lh"][2])


def calibration_report(spreads: Spreads) -> dict:
    """Each calibrated dimension's forward fit (alpha, beta; s the t's scale, its dof, and the words' shared error's
    sd), the inverse one from it and the true values' spread (a, b, sigma: the point placement, `_placement`), and the
    information share of one vote, R^2 = beta^2 var(true) / (beta^2 var(true) + s^2 + shared^2) (the t's scale taken
    as its sd; the inverse fit's b^2 var(pred) / (b^2 var(pred) + sigma^2) is the same number)."""
    out = {}
    for name, alpha, beta, s, dof, shared, truth in (
            ("centre x", spreads.centre_bias[0], spreads.centre_slope[0], spreads.centre, spreads.centre_dof,
             spreads.centre_word, spreads.centre_truth[0]),
            ("centre y", spreads.centre_bias[1], spreads.centre_slope[1], spreads.centre, spreads.centre_dof,
             spreads.centre_word, spreads.centre_truth[1]),
            ("log length", spreads.length_bias, spreads.length_slope, spreads.length, spreads.length_dof,
             spreads.length_word, spreads.length_truth),
            ("log height", spreads.height_bias, spreads.height_slope, spreads.height, spreads.height_dof,
             spreads.height_word, spreads.height_truth)):
        a, b, sigma = _placement(alpha, beta, math.hypot(s, shared), truth)
        vt = truth[1] ** 2
        out[name] = {"alpha": alpha, "beta": beta, "s": s, "dof": dof, "shared": shared, "a": a, "b": b,
                     "sigma": sigma,
                     "r2": (beta * beta * vt / (beta * beta * vt + s * s + shared * shared) if math.isfinite(vt)
                            else 1.0)}
    return out
