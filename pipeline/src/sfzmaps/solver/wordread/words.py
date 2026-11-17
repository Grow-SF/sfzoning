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
  - A word is a centre c, a log length, a log height, an axis theta (modulo 180 degrees) and a mean tag.  Where its
    px lie: in its box, each edge uncertain by `spreads.edge` px (measured), so a px at p is in word k's box with
    chance q_k(p): its box grown by a px's half width (a px is a unit square at its numbers, so one centred on the
    box's edge is in it), convolved with that Gaussian (1 inside, a half on that edge, falling outside), the Gaussian
    taken to EDGE_SDS spreads (where it is below double precision of its peak) and 0 beyond: the word's support.
  - The core head's output is evidence of where boxes are: w = sigmoid(core_logit) is its chance that the px is in
    a word's box, at a base rate of `spreads.core_share` (measured), so the px's output is w / core_share as likely
    in a box as on average, and (1 - w) / (1 - core_share) out of every box.  A px is in some word's box with chance
    C(p) = 1 - prod_k (1 - q_k(p)), word k's among them by its weight's share a_k q_k(p) / sum_j a_j q_j(p), its vote
    then the word's; out of every box with chance 1 - C(p), its vote then the background's.  Each word's weight a_k is
    fitted (where boxes overlap, as a code printed on a street name, the votes say whose the px are).  So a box
    covering px the head calls background pays for each (a long length vote cannot stretch a box over them), far from
    every word a px is the background's, and a copy of a word gains nothing from the shares (the copies' weights
    split its own) and next to nothing from C (no higher where q is 1).
  - A word px's vote is its word's with chance 1 - `spreads.outlier`, else an outlier with the background's
    density (measured: the net's votes have heavy tails), so a box may hold a few votes it cannot explain.
  - Each vote dimension is first corrected by its measured mean bias against Terra's words (`spreads.centre_bias`,
    `length_bias`, `height_bias`, `angle_bias`).
  - A word's vote: its offset (dx, dy) differs from (c - p) / H (H the word's height) by Gaussian errors of sd
    `spreads.centre` heights each way, its log length and log height from the word's by sd `spreads.length` and
    `spreads.height`, its axis from the word's (the difference taken modulo 180 degrees) by sd `spreads.angle`
    degrees, its tag from the word's by sd `spreads.tag` each channel.
  - The background's vote: its offset Gaussian about 0 of sd `spreads.background_centre` each way, its log length and
    log height Gaussian of mean and sd `spreads.background_length` and `spreads.background_height`, its axis any, its
    tag Gaussian about `spreads.background_tag_mean` of sd `spreads.background_tag`.
  - Neighbouring px's votes are not independent (the net's outputs are smooth): n votes carry about n / `spreads.cell`
    independent votes' evidence, so the log likelihood is divided by `cell`.
  - The score of a set of words: prior.slope x (the log likelihood / cell) + prior.log_prior for each word; the
    slope and the log prior fitted on the decisions this module makes (`prior_decisions`, `fit_word_prior`).
  The score is a sum over px of log L_p - log b_p (b_p the background's density of the px's vote), which is 0 off
  every word's support; so each move's change is exact on the union of the supports of the words it changes.
All the spreads are measured on the training quarters' predictions against Terra's words (`measure_spreads`); Task 8
runs it and the prior's fit.

The fit (tractable; every decision by the score):
  1. Proposals: the basins of the votes' centres.  Each px adds its core chance at the centre its vote points to (at
     the page's own px); the sum is smoothed by a Gaussian of the centre spread (in px at the votes' median height);
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
  Each fit step: the words' weights take a Newton step on the likelihood (exact gradient and Hessian); an E-step
  gives each word's votes' own update (the axis by the votes' mean error, the log length by their mean, the centre
  and log height alternately, the tag by its mean); then each word in turn takes a Newton step on the likelihood in
  its parameters (gradient and Hessian by finite differences, the Hessian's eigenvalues taken as their size and no
  less than the votes' curvature, within a trust region: this sees the spatial terms the votes' update does not,
  such as the core px ending where the word does against long length votes) and then moves towards its votes' own
  update.  Each move goes to the first point of its line, its end, half way, a quarter, ..., that raises the log
  likelihood of the whole model (every other word held, the spatial terms included), none if none does; so no step
  lowers the likelihood itself (the expected complete log likelihood is no guide to the boxes: a vote at a
  support's edge, given a tiny chance of being the word's, makes it -inf as soon as the box turns off it), and a
  fall beyond float noise (LL_NOISE) raises, as a bug.

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
step tried), FD_STEP (the finite differences' step), TRUST (a Newton step's trust region: the steps repeat, so where
they end does not depend on it), the weights' Newton step's numerical 0 (eigenvalues below sqrt(eps) of the
largest), CHUNK_PX (votes per pass: memory, the results do not depend on it), EDGE_SDS (where a word's spatial term
falls below double precision of its value inside), the variogram's scale bounds (1/100 px to the longest lag), the
lattice sum's span (to underflow) and the Gaussian smoothing's extent (OpenCV's: 4 sd).
"""
import dataclasses
import math

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from scipy.special import log_ndtr, ndtr, ndtri

from sfzmaps.solver import wordboxes

EXTRA_COLUMNS = ("p_word", "core_mass", "tag_spread", "word_logodds")
EM_STEPS = 200        # numerical only: an EM stops after this many steps ...
EM_TOL = 1e-7         # ... or once its log likelihood rises by less than this per vote
LL_NOISE = 1e-9       # numerical only: a likelihood falling by more than this share of itself is a bug
FIT_STEPS = 100       # numerical only: the prior's fit takes at most this many Newton steps
SPREAD_ROUNDS = 50    # numerical only: the spreads' fixed point stops after this many rounds ...
SPREAD_TOL = 1e-6     # ... or once no spread moves by this share of itself
CHUNK_PX = 1 << 20    # votes per pass (memory only: the results do not depend on it)
EDGE_SDS = float(-ndtri(2.0 ** -53))   # 8.21: beyond, a word's spatial term < double precision of its value inside
SHAPE_STEPS = 3       # numerical only: the centre and log height's alternations in the votes' own update
LINE_TOL = 1e-6       # numerical only: a step is halved down to this share of itself
FD_STEP = float(np.finfo(np.float64).eps ** 0.25)   # numerical only: the finite differences' step (px, log, rad)
TRUST = 1.0           # numerical only: a Newton step moves at most this many box lengths, log sizes and radians
_LOG2PI = math.log(2 * math.pi)
_LOG2 = math.log(2.0)


@dataclasses.dataclass(frozen=True)
class Spreads:
    """The measured spreads of the model (see the module; `measure_spreads`)."""
    centre: float               # heights: sd of a word vote's offset to its word's centre, each way
    length: float               # sd of a word vote's log length about its word's
    height: float               # sd of a word vote's log height about its word's
    angle: float                # degrees: sd of a word vote's axis about its word's (modulo 180)
    tag: float                  # sd of a word px's tag channel about its word's mean tag
    cell: float                 # px: n votes carry n / cell independent votes' evidence
    edge: float                 # px: sd of where a word's px end about its box's edges (the spatial prior)
    core_share: float           # the share of px in words: the core head's base rate (its evidence is w / core_share)
    outlier: float              # the share of a word's px whose vote is no vote for it (the background's density)
    centre_bias: tuple          # heights: the votes' mean offset error (x, y) against Terra's words, taken off
    length_bias: float          # the votes' mean log length error against Terra's words, taken off
    height_bias: float          # the votes' mean log height error against Terra's words, taken off
    angle_bias: float           # degrees: the votes' mean axis error (modulo 180) against Terra's words, taken off
    background_centre: float    # heights: sd of a background vote's offset about 0, each way
    background_length: tuple    # (mean, sd) of a background vote's log length
    background_height: tuple    # (mean, sd) of a background vote's log height
    background_tag: float       # sd of a background px's tag channel about its mean
    background_tag_mean: tuple  # a background px's mean tag, one per channel

    def check(self, channels: int | None = None) -> None:
        sds = {k: getattr(self, k) for k in ("centre", "length", "height", "angle", "tag", "cell", "edge",
                                             "background_centre", "background_tag")}
        sds["background_length sd"], sds["background_height sd"] = self.background_length[1], self.background_height[1]
        bad = {k: v for k, v in sds.items() if not (v > 0 and math.isfinite(v))}
        if bad:
            raise ValueError(f"every spread is a positive number, not {bad}")
        for name in ("core_share", "outlier"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} is a share strictly between 0 and 1, not {getattr(self, name)}")
        biases = (*self.centre_bias, self.length_bias, self.height_bias, self.angle_bias)
        if len(self.centre_bias) != 2 or not all(math.isfinite(b) for b in biases):
            raise ValueError(f"the biases are finite numbers (centre_bias two), not {biases}")
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


def _own(v: dict) -> np.ndarray:
    """The parameters (n, 5) of the boxes the votes give."""
    return np.c_[v["p"] + np.exp(v["lh"])[:, None] * v["o"], v["ll"], v["lh"], v["th"]]


def _sub(v: dict, sel) -> dict:
    return {key: a[sel] for key, a in v.items()}


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
        sa = math.radians(spreads.angle)
        self.ic, self.il, self.ih = (1 / (2 * spreads.centre ** 2), 1 / (2 * spreads.length ** 2),
                                     1 / (2 * spreads.height ** 2))
        self.ia, self.it = 1 / (2 * sa ** 2), 1 / (2 * spreads.tag ** 2)
        self.norm = (_LOG2PI + 2 * math.log(spreads.centre) + 0.5 * (_LOG2PI + 2 * math.log(spreads.length))
                     + 0.5 * (_LOG2PI + 2 * math.log(spreads.height)) + 0.5 * (_LOG2PI + 2 * math.log(sa))
                     + channels / 2 * (_LOG2PI + 2 * math.log(spreads.tag)))
        self.edge = spreads.edge
        self.reach = EDGE_SDS * spreads.edge
        self.lpi, self.lnpi = math.log(spreads.core_share), math.log1p(-spreads.core_share)
        self.leps, self.lkeep = math.log(spreads.outlier), math.log1p(-spreads.outlier)
        self.bias_c = np.asarray(spreads.centre_bias, np.float64)
        self.bias_l, self.bias_h = spreads.length_bias, spreads.height_bias
        self.bias_a = math.radians(spreads.angle_bias)
        bl, bh = spreads.background_length, spreads.background_height
        self.bc, self.bl_mean, self.bl, self.bh_mean, self.bh = (1 / (2 * spreads.background_centre ** 2), bl[0],
                                                                 1 / (2 * bl[1] ** 2), bh[0], 1 / (2 * bh[1] ** 2))
        self.bt = 1 / (2 * spreads.background_tag ** 2)
        self.bmean = np.asarray(spreads.background_tag_mean, np.float64)
        self.bnorm = (_LOG2PI + 2 * math.log(spreads.background_centre) + 0.5 * (_LOG2PI + 2 * math.log(bl[1]))
                      + 0.5 * (_LOG2PI + 2 * math.log(bh[1])) + math.log(math.pi)
                      + channels / 2 * (_LOG2PI + 2 * math.log(spreads.background_tag)))

    def log_background(self, v: dict) -> np.ndarray:
        """The background's log density of each vote."""
        return (-(v["o"] ** 2).sum(1) * self.bc - (v["ll"] - self.bl_mean) ** 2 * self.bl
                - (v["lh"] - self.bh_mean) ** 2 * self.bh - ((v["t"] - self.bmean) ** 2).sum(1) * self.bt
                - self.bnorm)

    def log_vote(self, v: dict, prm: np.ndarray, tag_mean: np.ndarray) -> np.ndarray:
        """log density of each vote under a word (`prm` (5,) or (n, 5), `tag_mean` (k,) or (n, k))."""
        prm = np.broadcast_to(prm, (len(v["ll"]), 5))
        off = v["o"] - (prm[:, :2] - v["p"]) * np.exp(-prm[:, 3])[:, None]
        return (-(off ** 2).sum(1) * self.ic - (v["ll"] - prm[:, 2]) ** 2 * self.il
                - (v["lh"] - prm[:, 3]) ** 2 * self.ih - _half(v["th"] - prm[:, 4]) ** 2 * self.ia
                - ((v["t"] - tag_mean) ** 2).sum(1) * self.it - self.norm)

    def _frame(self, p: np.ndarray, prm: np.ndarray):
        """Points in the word's frame, and the half sides of the box its px lie in: its box grown by a px's half
        width (a px is a unit square at its numbers, so one centred on the box's edge is in it)."""
        along, up = _axes(prm[4])
        rel = p - prm[:2]
        return rel @ along, rel @ up, math.exp(prm[2]) / 2 + 0.5, math.exp(prm[3]) / 2 + 0.5

    def inside(self, p: np.ndarray, prm: np.ndarray, grow: float = 0.0) -> np.ndarray:
        """Which points lie in the word's numerical support (its box grown by EDGE_SDS edge spreads), its edges moved
        out by `grow` px."""
        pa, pb, a, b = self._frame(p, prm)
        return (np.abs(pa) <= a + self.reach + grow) & (np.abs(pb) <= b + self.reach + grow)

    def log_q(self, p: np.ndarray, prm: np.ndarray) -> np.ndarray:
        """log q (see the module) of points `p`: the chance a px there is in the word's box."""
        pa, pb, a, b = self._frame(p, prm)
        return _log_interval(pa, -a, a, self.edge) + _log_interval(pb, -b, b, self.edge)

    def vote_curvature(self, v: dict, r: np.ndarray, prm: np.ndarray) -> np.ndarray:
        """The curvature (Gauss-Newton: minus the second derivative) of the votes' log densities weighted by r in each
        of a word's parameters (cx, cy, log length, log height, theta)."""
        mass = float(r.sum())
        u = (prm[:2] - v["p"]) * math.exp(-prm[3])
        ih2 = math.exp(-2 * prm[3])
        return np.array([2 * self.ic * mass * ih2, 2 * self.ic * mass * ih2, 2 * self.il * mass,
                         2 * self.ic * float((r * (u ** 2).sum(1)).sum()) + 2 * self.ih * mass, 2 * self.ia * mass])

    def own_update(self, v: dict, r: np.ndarray, prm: np.ndarray):
        """The votes' own update of a word's parameters and tag, weighted by r (no spatial term): the axis moved by
        the votes' mean error (modulo a half turn), the log length their mean, the centre (the mean of the centres
        the votes point to at the word's height) and the log height (a Gauss-Newton step on its two terms) in turn
        (SHAPE_STEPS), the tag their mean.  None when r holds no mass."""
        mass = float(r.sum())
        if not mass > 0:
            return None
        new = np.array(prm, np.float64)
        new[4] = prm[4] + float((r * _half(v["th"] - prm[4])).sum()) / mass
        new[2] = float((r * v["ll"]).sum()) / mass
        for _ in range(SHAPE_STEPS):
            height = math.exp(new[3])
            new[:2] = (r[:, None] * (v["p"] + height * v["o"])).sum(0) / mass
            u = (new[:2] - v["p"]) / height                     # (c - p) / H: d/d log H is -u
            res = v["o"] - u
            grad = (2 * self.ic * float((r * (res * u).sum(1)).sum())
                    - 2 * self.ih * float((r * (v["lh"] - new[3])).sum()))
            curve = 2 * self.ic * float((r * (u ** 2).sum(1)).sum()) + 2 * self.ih * mass
            new[3] = new[3] - grad / curve
        return new, (r[:, None] * v["t"]).sum(0) / mass


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

    def __len__(self):
        return len(self.ys)

    def chance(self, idx=slice(None)) -> np.ndarray:
        return 0.5 * (1 + np.tanh(self.x[idx] / 2))

    def take(self, idx) -> dict:
        """The votes `idx`, each dimension less its measured bias: p (n, 2), the offset o (n, 2) in heights, log
        length ll, log height lh, axis th (radians, modulo a half turn), tags t (n, k), the core chance w, the core
        head's evidence in a word's box and out of every one (log w / core_share and log (1 - w) / (1 - core_share),
        from the logit: never infinite), and the background's log density of each vote."""
        ys, xs = self.ys[idx], self.xs[idx]
        g = self.geo[:, ys, xs].astype(np.float64)
        x = self.x[idx]
        lw, lnw = -np.logaddexp(0, -x), -np.logaddexp(0, x)
        v = {"p": np.stack([xs, ys], -1).astype(np.float64), "o": (g[:2] / g[3]).T - self.bias_c,
             "ll": np.log(g[2]) - self.bias_l, "lh": np.log(g[3]) - self.bias_h, "th": np.deg2rad(g[4]) - self.bias_a,
             "t": np.moveaxis(self.tag[:, ys, xs].astype(np.float64), 0, -1),
             "w": 0.5 * (1 + np.tanh(x / 2)), "lin": lw - self.lpi, "lout": lnw - self.lnpi}
        v["lbg"] = self.log_background(v)
        return v

    def centres(self) -> np.ndarray:
        """The centre each vote points to (n, 2), its biases taken off, CHUNK_PX votes at a time."""
        out = np.empty((len(self), 2))
        for lo in range(0, len(self), CHUNK_PX):
            ys, xs = self.ys[lo:lo + CHUNK_PX], self.xs[lo:lo + CHUNK_PX]
            g = self.geo[:4, ys, xs].astype(np.float64)
            height = g[3] * math.exp(-self.bias_h)
            out[lo:lo + CHUNK_PX] = np.stack([xs, ys], -1) + height[:, None] * ((g[:2] / g[3]).T - self.bias_c)
        return out

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
    """For votes `v` and each word: log q (-inf off its support) and the vote's log density (-inf off it)."""
    n, kk = len(v["w"]), len(words)
    lq = np.full((n, kk), -np.inf)
    lf = np.full((n, kk), -np.inf)
    for j, word in enumerate(words):
        inside = V.inside(v["p"], word["prm"])
        if inside.any():
            sv = _sub(v, inside)
            lq[inside, j] = V.log_q(sv["p"], word["prm"])
            lf[inside, j] = V.log_vote(sv, word["prm"], word["tag"])
    return lq, lf


def _log1m_exp(x: np.ndarray) -> np.ndarray:
    """log(1 - exp(x)) for x <= 0, in the form exact for each range (log(-expm1(x)) near 0, log1p(-exp(x)) below)."""
    x = np.asarray(x, np.float64)
    with np.errstate(divide="ignore"):
        return np.where(x > -_LOG2, np.log(-np.expm1(np.minimum(x, 0.0))), np.log1p(-np.exp(np.minimum(x, 0.0))))


def _cover(lq: np.ndarray):
    """Per vote, from the words' log q (n, K): log C (the chance its px is in some word's box, 1 - prod(1 - q)) and
    log(1 - C)."""
    s = _log1m_exp(lq).sum(1) if lq.shape[1] else np.zeros(lq.shape[0])
    return _log1m_exp(s), s


def _shares(lq: np.ndarray, la: np.ndarray) -> np.ndarray:
    """log of each word's share of each vote's px among the words whose boxes may hold it, a_k q_k / sum a q (`la`
    the log weights a); -inf off its support, and for every word where none of them holds any weight."""
    lw = lq + la[None, :]
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(lw), lw - _lse(lw)[:, None], -np.inf)


def _mixture(v: dict, lq: np.ndarray, lf: np.ndarray, model: "_Model", la: np.ndarray):
    """Each vote's score term (log L less its value with no word: 0 where no word's support reaches) and its chance
    of being each word's.  See the module: in some word's box with chance C = 1 - prod(1 - q_k), word k's among them
    by its weight's share a_k q_k / sum a q (`la` the log weights), the core evidence w / core_share, the vote word
    k's with chance 1 - outlier or the background's density's with chance outlier; out of every box with chance
    1 - C, the core evidence (1 - w) / (1 - core_share) and the background's vote."""
    log_c, log_rest = _cover(lq)
    share = _shares(lq, la)
    joint = v["lin"][:, None] + log_c[:, None] + share + model.lkeep + lf
    stray = v["lin"] + log_c + model.leps + v["lbg"]           # in a box, its vote an outlier
    none = v["lout"] + v["lbg"]
    num = np.logaddexp(np.logaddexp(_lse(joint), stray), log_rest + none)
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


def _near(V: _Votes, words: list[dict], lo, hi) -> list[dict]:
    """The words whose supports' bounds meet [lo, hi] (the others cannot touch a px there)."""
    out = []
    for word in words:
        blo, bhi = V.bounds(word["prm"])
        if (blo <= hi).all() and (lo <= bhi).all():
            out.append(word)
    return out


def _supports(V: _Votes, words: list[dict]) -> np.ndarray:
    return np.unique(np.concatenate([V.support(w["prm"]) for w in words])) if words else np.zeros(0, np.int64)


def _score(V: _Votes, region: np.ndarray, words: list[dict], held: list[dict]) -> float:
    """The log likelihood of the votes of `region` (less the background's), words `words` and the held words near."""
    v = V.take(region)
    every = words + _near(V, held, *_span(V, region))
    lq, lf = _terms(V, v, every)
    return float(_mixture(v, lq, lf, V, _weights(every))[0].sum())


def _reach_of(V: _Votes, prm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounds holding the support of a word with these parameters and of any word on a line from them to others of
    no bigger box (its circumscribed circle, grown by the support's reach)."""
    radius = 0.5 * math.hypot(math.exp(prm[2]) + 1, math.exp(prm[3]) + 1) + V.reach
    return prm[:2] - radius, prm[:2] + radius


def _fit(V: _Votes, start: list[dict], held: list[dict], trace: list | None = None) -> list[dict]:
    """Words `start` fitted (see the module) with the words `held` held.  `trace` collects each step's log likelihood
    before and after its weight and line steps, on the step's region.  RuntimeError when a step lowers it beyond
    float noise (a bug)."""
    words = [{"prm": np.array(w["prm"], np.float64), "tag": np.array(w["tag"], np.float64),
              "a": float(w.get("a", 0.0))} for w in start]
    n = len(words)
    for _ in range(EM_STEPS):
        region = _supports(V, words)
        v = V.take(region)
        every = words + _near(V, held, *_span(V, region))
        lq, lf = _terms(V, v, every)
        la = _weights(every)
        _, resp = _mixture(v, lq, lf, V, la)
        targets = [V.own_update(v, resp[:, j], w["prm"]) for j, w in enumerate(words)]
        steps = [_newton(V, v, lq, lf, la, j, w, resp[:, j]) for j, w in enumerate(words)]
        # the step's region: every point of every word's lines (its Newton step's and to its votes' update)
        ends = ([w["prm"] for w in words] + [t[0] for t in targets if t is not None]
                + [w["prm"] + d for w, d in zip(words, steps) if d is not None])
        region = np.union1d(region, np.concatenate([V.region(*_reach_of(V, e)) for e in ends]))
        v = V.take(region)
        every = words + _near(V, held, *_span(V, region))
        lq, lf = _terms(V, v, every)
        la = _weights(every)
        term, resp = _mixture(v, lq, lf, V, la)
        before = float(term.sum())
        la[:n] = _weight_step(v, lq, lf, la, n, V)
        for j, w in enumerate(words):
            w["a"] = float(la[j])
        for j, (w, target, d) in enumerate(zip(words, targets, steps)):
            if d is not None:                                    # its Newton step
                _climb(V, v, lq, lf, la, j, w, w["prm"] + d, w["tag"])
            if target is not None:                               # then to the votes' own update
                _climb(V, v, lq, lf, la, j, w, *target)
        after = float(_mixture(v, lq, lf, V, la)[0].sum())
        if trace is not None:
            trace.append((before, after))
        if after < before - LL_NOISE * max(1.0, abs(before)):
            raise RuntimeError(f"a fit step lowered the log likelihood from {before} to {after}: a bug")
        if after - before <= EM_TOL * len(region):
            break
    return words


def _near_line(V: _Votes, v: dict, *prms) -> np.ndarray:
    """The votes of `v` a word's support can touch at any point of the line between `prms` (every end's reach)."""
    spans = [_reach_of(V, prm) for prm in prms]
    lo, hi = np.min([a for a, _ in spans], 0), np.max([b for _, b in spans], 0)
    return np.flatnonzero(((v["p"] >= lo) & (v["p"] <= hi)).all(1))


def _move(V: _Votes, v: dict, lq, lf, la, j: int, near: np.ndarray):
    """value(prm, tag): the log likelihood of the votes `near` (of `v`) with word j at (prm, tag), the other words
    held (lq, lf their terms on `v`, la the log weights); and word j's terms there.  Off those votes word j's terms
    are -inf at every (prm, tag) it is asked of (`_near_line`), so only theirs change."""
    vn = _sub(v, near)
    nq, nf = lq[near], lf[near]

    def value(prm, tag):
        lqj, lfj = np.full(len(near), -np.inf), np.full(len(near), -np.inf)
        inside = V.inside(vn["p"], prm)
        if inside.any():
            sv = _sub(vn, inside)
            lqj[inside] = V.log_q(sv["p"], prm)
            lfj[inside] = V.log_vote(sv, prm, tag)
        cq, cf = nq.copy(), nf.copy()
        cq[:, j], cf[:, j] = lqj, lfj
        return float(_mixture(vn, cq, cf, V, la)[0].sum()), lqj, lfj

    return value


def _newton(V: _Votes, v: dict, lq, lf, la, j: int, w: dict, r: np.ndarray):
    """Word j's Newton step on the log likelihood in its parameters (the spatial terms included, which the votes' own
    update does not see): the gradient and the Hessian by finite differences (FD_STEP; central for the gradient and
    the diagonal, forward for the cross terms), each of the Hessian's eigenvalues taken as its size and no less than
    the votes' curvature along its eigenvector (Gauss-Newton, `r` their chances of being the word's), so the step
    climbs where the likelihood does not curve down.  None where there is no step (no gradient, or past float
    range)."""
    prm, tag = w["prm"], w["tag"]
    near = np.flatnonzero(V.inside(v["p"], prm, 1.0))     # every support within FD_STEP of prm's (< 1 px away)
    if not len(near):
        return None
    value = _move(V, v, lq, lf, la, j, near)
    f0 = value(prm, tag)[0]
    h = FD_STEP
    e = np.eye(5) * h
    up = np.array([value(prm + e[i], tag)[0] for i in range(5)])
    down = np.array([value(prm - e[i], tag)[0] for i in range(5)])
    grad = (up - down) / (2 * h)
    hess = np.diag((up - 2 * f0 + down) / h ** 2)
    for i in range(5):
        for k in range(i + 1, 5):
            hess[i, k] = hess[k, i] = (value(prm + e[i] + e[k], tag)[0] - up[i] - up[k] + f0) / h ** 2
    if not (np.isfinite(grad).all() and np.isfinite(hess).all()):
        return None
    lam, vec = np.linalg.eigh(-hess)
    votes = V.vote_curvature(_sub(v, near), r[near], prm)
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
    does."""
    prm, tag = w["prm"], w["tag"]
    near = _near_line(V, v, prm, prm_to)
    value = _move(V, v, lq, lf, la, j, near)
    f0 = value(prm, tag)[0]
    share = 1.0
    while share >= LINE_TOL:
        at = _between(prm, prm_to, share), tag + share * (tag_to - tag)
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


def _proposals(V: _Votes, spreads: Spreads) -> _Proposals:
    """Every basin with its one-word fit on its own votes, all at once, a pass at a time over whole basins."""
    sigma = spreads.centre * float(np.median(V.geo[3]))              # the centre spread at the median height
    basin = _basins(V.centres(), V.chance(), V.shape, sigma)
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
    """The one-word-per-group M-step on the votes alone (a basin's own fit): each group's votes' own update
    (`_Model.own_update`, vectorised), kept where it raises the group's expected vote log density (else no change)."""
    prms, tags = np.array(prms, np.float64), np.array(tags, np.float64)
    mass = np.bincount(group, r, count)
    on = mass > 0
    safe = np.where(on, mass, 1.0)
    new, new_tags = prms.copy(), tags.copy()
    for ch in range(V.k):
        new_tags[on, ch] = np.bincount(group, r * v["t"][:, ch], count)[on] / mass[on]
    new[on, 4] = prms[on, 4] + (np.bincount(group, r * _half(v["th"] - prms[group, 4]), count) / safe)[on]
    new[on, 2] = (np.bincount(group, r * v["ll"], count) / safe)[on]
    for _ in range(SHAPE_STEPS):
        height = np.exp(new[group, 3])
        for a in (0, 1):
            new[on, a] = (np.bincount(group, r * (v["p"][:, a] + height * v["o"][:, a]), count) / safe)[on]
        u = (new[group, :2] - v["p"]) / height[:, None]
        res = v["o"] - u
        grad = (2 * V.ic * np.bincount(group, r * (res * u).sum(1), count)
                - 2 * V.ih * np.bincount(group, r * (v["lh"] - new[group, 3]), count))
        curve = 2 * V.ic * np.bincount(group, r * (u ** 2).sum(1), count) + 2 * V.ih * mass
        new[on, 3] = new[on, 3] - (grad / np.where(on, curve, 1.0))[on]
    q_old = np.bincount(group, r * V.log_vote(v, prms[group], tags[group]), count)
    q_new = np.bincount(group, r * V.log_vote(v, new[group], new_tags[group]), count)
    keep = on & (q_new > q_old)
    prms[keep], tags[keep] = new[keep], new_tags[keep]
    return prms, tags


def _fit_basins(V: _Votes, idx, group, starts):
    """One word fitted to each basin's own votes at once (EM per basin on the votes alone, the background in the
    model, no spatial term: each vote in the word's box or out of every one, by the core evidence and its vote: the
    proposals' own fit), started at its vote of the highest core chance; each basin's EM
    stops on its own and only the basins still moving are computed.  The words' parameters and tags, and each basin's
    log likelihood with its word and with none."""
    v = V.take(idx)
    nb, n = len(starts), len(idx)
    prms = _own(_sub(v, starts))
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
        word = vs["lin"] + np.logaddexp(V.lkeep + V.log_vote(vs, prms[g], tags[g]), V.leps + vs["lbg"])
        lp[sel] = np.logaddexp(word, back[sel])
        ll = np.bincount(g, lp[sel], nb)
        moving &= ll - prev > EM_TOL * count
        prev = np.where(moving | (prev == -math.inf), ll, prev)
        keep = moving[g]
        if not keep.any():
            break
        ln = V.log_vote(_sub(vs, keep), prms[g[keep]], tags[g[keep]])      # not an outlier: the word's own vote
        r = np.exp(vs["lin"][keep] + V.lkeep + ln - lp[sel][keep])
        prms, tags = _group_step(V, _sub(vs, keep), r, prms, tags, g[keep], nb)
    return prms, tags, np.bincount(group, lp, nb), np.bincount(group, back, nb)


# --- the search -----------------------------------------------------------------------------------------------------

def _overlap(V: _Votes, a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two words' supports (their boxes grown by EDGE_SDS edge spreads) share area."""
    pa, pb = _box(a, V.reach + 0.5), _box(b, V.reach + 0.5)
    if (pa.max(0) < pb.min(0)).any() or (pb.max(0) < pa.min(0)).any():
        return False
    return cv2.intersectConvexConvex(pa.astype(np.float32), pb.astype(np.float32))[0] > 0


def _total(V: _Votes, words: list[dict], prior: WordPrior, spreads: Spreads) -> float:
    """The configuration's total score, from the words' parameters (every vote; 0 off every support)."""
    region = _supports(V, words)
    return prior.slope * _score(V, region, words, []) / spreads.cell + prior.log_prior * len(words)


def _search(V: _Votes, P: _Proposals, spreads: Spreads, prior: WordPrior, record: list | None = None,
            scores: list | None = None) -> list[dict]:
    """The words (see the module): from none, add, split and join moves while one raises the total score, the best
    first, each move's change exact (its words refitted with every other word held).  `record` collects every move
    weighed: (kind, gain of the extra word (log likelihood / cell), [(votes, chances) of each part]).  `scores` collects
    the total score recomputed from the words' parameters at the start and after every move."""
    scale = prior.slope / spreads.cell
    proposed = [int(b) for b in np.argsort(-(P.ll - P.ll0), kind="stable")
                if scale * (P.ll[b] - P.ll0[b]) + prior.log_prior > 0]
    words: dict[int, dict] = {}
    next_id = 0
    moves: dict = {}

    def held(exclude=()) -> list[dict]:
        return [w for k, w in words.items() if k not in exclude]

    def add(b):
        H = held()
        new = _fit(V, [{"prm": P.prms[b], "tag": P.tags[b]}], H)
        d, region = _change(V, [], new, H)
        if record is not None:
            record.append(("word", d / spreads.cell, _parts(V, region, new, H)))
        return scale * d + prior.log_prior, ([], new), region

    def split(k):
        H = held((k,))
        word = words[k]
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
        seed = {"prm": _own(_sub(v, slice(j, j + 1)))[0], "tag": v["t"][j], "a": float(word.get("a", 0.0))}
        new = _fit(V, [word, seed], H)
        d, region = _change(V, [word], new, H)
        if record is not None:
            record.append(("split", d / spreads.cell, _parts(V, region, new, H)))
        return scale * d + prior.log_prior, ([k], new), region

    def join(a, b):
        H = held((a, b))
        best = None
        for start in (words[a], words[b]):          # the one word, from each start; the better kept
            new = _fit(V, [start], H)
            d, region = _change(V, [words[a], words[b]], new, H)
            if best is None or d > best[0]:
                best = (d, new, region)
        d, new, region = best
        if record is not None:
            record.append(("join", -d / spreads.cell, _parts(V, region, [words[a], words[b]], H)))
        return scale * d - prior.log_prior, ([a, b], new), region     # joining: d, and one word fewer

    if scores is not None:
        scores.append(_total(V, [], prior, spreads))
    while True:
        for b in proposed:
            if ("add", b) not in moves:
                moves[("add", b)] = add(b)
        ids = sorted(words)
        for k in ids:
            if ("split", k) not in moves:
                moves[("split", k)] = split(k)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if ("join", a, b) not in moves and _overlap(V, words[a]["prm"], words[b]["prm"]):
                    moves[("join", a, b)] = join(a, b)
        if not moves:
            break
        key, (delta, change, region) = max(moves.items(), key=lambda kv: kv[1][0])   # the first of equals
        if not delta > 0:
            break
        old, new = change
        if key[0] == "add":
            proposed.remove(key[1])
        changed = np.union1d(region, _supports(V, [words[k] for k in old]))
        lo, hi = _span(V, changed)
        for k in old:
            del words[k]
        for w in new:
            words[next_id] = w
            next_id += 1

        def stale(mkey, mval):
            if any(k in old for k in mkey[1:]) and mkey[0] != "add":
                return True
            mlo, mhi = _span(V, mval[2])
            return bool((mlo <= hi).all() and (lo <= mhi).all())

        moves = {m: val for m, val in moves.items() if m != key and not stale(m, val)}
        if scores is not None:
            scores.append(_total(V, list(words.values()), prior, spreads))
    return [words[k] for k in sorted(words)]


def _final(V: _Votes, found: list[dict]) -> list[dict]:
    """The words refitted jointly, each connected set of words whose supports overlap at once (the likelihood
    factorises over the sets)."""
    n = len(found)
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _overlap(V, found[i]["prm"], found[j]["prm"]):
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    out = list(found)
    for members in groups.values():
        fitted = _fit(V, [found[i] for i in members], [])
        for i, w in zip(members, fitted):
            out[i] = w
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
    return core_logit, geo, tag


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
    rows, order = [], []
    for k, word in enumerate(found):
        region = V.support(word["prm"])
        others = found[:k] + found[k + 1:]
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
    direction (modulo a half turn), its second edge's length."""
    e1, e2 = rect[1] - rect[0], rect[3] - rect[0]
    theta = math.atan2(-e1[1], e1[0])
    return np.r_[rect.mean(0), math.log(max(math.hypot(*e1), 1e-12)), math.log(max(math.hypot(*e2), 1e-12)), theta]


def _own_fit(v: dict, rect: np.ndarray, model: _Model | None):
    """A Terra word's own fit on its votes (all of them a word's px): with `model`, by EM over each vote's chance of
    being the word's own rather than an outlier (the background's density at the share `outlier`), each step along
    the line from where it is to the votes' own update to the most log likelihood (the model's M-step without the
    spatial terms); without, the votes' means (the first round).  The parameters, the tag and each vote's chance of
    being the word's own."""
    prm = _rect_params(rect)
    if model is None:
        prm[4] = prm[4] + float(_half(v["th"] - prm[4]).mean())
        prm[2], prm[3] = float(v["ll"].mean()), float(v["lh"].mean())
        prm[:2] = (v["p"] + math.exp(prm[3]) * v["o"]).mean(0)
        return prm, v["t"].mean(0), np.ones(len(v["ll"]))
    tag = v["t"].mean(0)
    stray = model.leps + model.log_background(v)
    prev = -math.inf
    for _ in range(EM_STEPS):
        own = model.lkeep + model.log_vote(v, prm, tag)
        r = np.exp(own - np.logaddexp(own, stray))
        target = model.own_update(v, r, prm)
        if target is None:
            break

        def value(share, prm=prm, tag=tag, target=target):
            ln = model.lkeep + model.log_vote(v, _between(prm, target[0], share), tag + share * (target[1] - tag))
            return float(np.logaddexp(ln, stray).sum())

        inner = minimize_scalar(lambda x: -value(x), bounds=(0.0, 1.0), method="bounded").x
        share = max((0.0, 1.0, inner), key=value)
        prm, tag = _between(prm, target[0], share), tag + share * (target[1] - tag)
        now = value(0.0, prm, tag, target)
        if now - prev <= EM_TOL * len(r):
            break
        prev = now
    own = model.lkeep + model.log_vote(v, prm, tag)
    return prm, tag, np.exp(own - np.logaddexp(own, stray))


def measure_spreads(pages) -> Spreads:
    """The spreads (see the module), measured on predictions against Terra's words, never the owner's answers.
    `pages`: dicts of geo, tag (as `core_words` takes them: geo as `model.word_geometry` decodes it), word_id (int
    (h, w): 1 + the Terra word whose rectangle holds the px, later over earlier, 0 on none: the targets' `tag_id`) and
    words (float (n, 4, 2): Terra's rectangles).
      core_share  the share of the pages' px in Terra's words (the core head's base rate)
      centre_bias, length_bias, height_bias, angle_bias   each vote dimension's mean error against its Terra word over
                  every word px: its offset in heights less (c - p) / H, its log length and log height less the
                  word's, its axis less the word's (modulo 180); taken off every vote (word and background) before
                  anything below, as `core_words` takes them off
      Each Terra word's own fit is the one-word model's on its votes (all its px a word's), as `core_words` fits it
      (its M-step without the spatial terms, each vote the word's own or an outlier by EM); the spreads it reads with
      are the ones being measured, to a fixed point (SPREAD_ROUNDS, SPREAD_TOL), the first round on the votes' means.
      A vote's residuals: its offset less (c - p) / H, its log length, log height, axis (modulo a half turn) and tag
      less the fit's; each weighted by its chance of being the word's own below.
      outlier     the share of word px whose vote is an outlier: the mean of that chance's complement
      centre, cell    from the variogram of the offset residuals within words, over every lag (differences within one
                  word): half the mean squared difference at lag (dy, dx) is sigma^2 (1 - rho), rho Gaussian in the
                  lag of scale ell, fitted by least squares weighted by the pairs (ell searched between numerical
                  bounds: 1/100 px and the longest lag); centre = sigma, cell = the sum of rho over the lattice
                  (independent votes give 1)
      length, height, angle, tag   the root mean square of those residuals, each word's degrees of freedom n less its
                  correlated share (the sum of rho over its px pairs / n)
      edge        where a word's px end about its box: the root mean square, over the edges of every word, of its
                  own fit's edge less Terra's rectangle's (both along the fit's axes), with a px's own extent (1/12
                  px^2 a side, its position uniform within it) added in quadrature
      background_centre, background_length, background_height, background_tag_mean, background_tag   a background
                  px's offsets' root mean square about 0; its log length's and log height's mean and sd; its tags'
                  mean and root mean square about it"""
    per_word = []                      # (v, Terra's rectangle, mask, its px within the mask, pair counts)
    err = np.zeros(5)                  # summed over word px: offset x, offset y, log length, log height, axis
    acc = {"o": np.zeros(2), "o2": np.zeros(2), "n": 0, "ll": 0.0, "ll2": 0.0, "lh": 0.0, "lh2": 0.0, "t": None,
           "t2": None}
    n_word = n_all = 0
    channels = None
    for page in pages:
        geo, tag = np.asarray(page["geo"]), np.asarray(page["tag"])
        word_id = np.asarray(page["word_id"])
        rects = np.asarray(page["words"], np.float64).reshape(-1, 4, 2)
        channels = tag.shape[0]
        n_all += word_id.size
        for j in np.unique(word_id[word_id > 0]):
            ys, xs = np.nonzero(word_id == j)
            g = geo[:, ys, xs].astype(np.float64)
            v = {"p": np.stack([xs, ys], -1).astype(np.float64), "o": (g[:2] / g[3]).T, "ll": np.log(g[2]),
                 "lh": np.log(g[3]), "th": np.deg2rad(g[4]),
                 "t": np.moveaxis(tag[:, ys, xs].astype(np.float64), 0, -1)}
            truth = _rect_params(rects[j - 1])
            err[:2] += (v["o"] - (truth[:2] - v["p"]) / math.exp(truth[3])).sum(0)
            err[2] += float((v["ll"] - truth[2]).sum())
            err[3] += float((v["lh"] - truth[3]).sum())
            err[4] += float(_half(v["th"] - truth[4]).sum())
            n_word += len(ys)
            y0, x0 = ys.min(), xs.min()
            m = np.zeros((ys.max() - y0 + 1, xs.max() - x0 + 1))
            m[ys - y0, xs - x0] = 1
            per_word.append((v, rects[j - 1], m, (ys - y0, xs - x0), np.rint(_corr(m, m))))
        by, bx = np.nonzero(word_id == 0)
        for lo in range(0, len(by), CHUNK_PX):
            yy, xx = by[lo:lo + CHUNK_PX], bx[lo:lo + CHUNK_PX]
            g = geo[:4, yy, xx].astype(np.float64)
            tt = tag[:, yy, xx].astype(np.float64)
            o, ll, lh = g[:2] / g[3], np.log(g[2]), np.log(g[3])
            acc["o"] += o.sum(1)
            acc["o2"] += (o ** 2).sum(1)
            acc["n"] += len(yy)
            acc["ll"] += float(ll.sum())
            acc["ll2"] += float((ll ** 2).sum())
            acc["lh"] += float(lh.sum())
            acc["lh2"] += float((lh ** 2).sum())
            acc["t"] = tt.sum(1) if acc["t"] is None else acc["t"] + tt.sum(1)
            acc["t2"] = (tt ** 2).sum(1) if acc["t2"] is None else acc["t2"] + (tt ** 2).sum(1)
    n = acc["n"]
    if not per_word or not n:
        raise ValueError("the pages hold no word px or no background px to measure")
    bias = err / n_word
    for v, *_ in per_word:                                  # the biases taken off, as core_words takes them off
        v["o"] = v["o"] - bias[:2]
        v["ll"], v["lh"], v["th"] = v["ll"] - bias[2], v["lh"] - bias[3], v["th"] - bias[4]

    def mean_sd(total, squares, shift):
        mean = total / n
        return (mean - shift, math.sqrt(max(squares / n - mean ** 2, 0.0)))

    o_mean = acc["o"] / n                                   # the background's offsets about 0, the bias taken off
    o_sq = acc["o2"] / n - 2 * bias[:2] * o_mean + bias[:2] ** 2
    tmean = acc["t"] / n
    fixed = {"core_share": n_word / n_all, "centre_bias": (float(bias[0]), float(bias[1])),
             "length_bias": float(bias[2]), "height_bias": float(bias[3]), "angle_bias": math.degrees(bias[4]),
             "background_centre": math.sqrt(float(o_sq.sum()) / 2),
             "background_length": mean_sd(acc["ll"], acc["ll2"], bias[2]),
             "background_height": mean_sd(acc["lh"], acc["lh2"], bias[3]),
             "background_tag": math.sqrt(float((acc["t2"] / n - tmean ** 2).sum()) / channels),
             "background_tag_mean": tuple(float(x) for x in tmean)}
    spreads = got = None
    fields = ("centre", "length", "height", "angle", "tag", "cell", "edge", "outlier")
    for _ in range(SPREAD_ROUNDS):
        got = _measure_round(per_word, channels, spreads, fixed)
        if spreads is not None and all(abs(getattr(got, f) - getattr(spreads, f)) <= SPREAD_TOL * getattr(spreads, f)
                                       for f in fields):
            break
        spreads = got
    return got


def _measure_round(per_word, channels: int, spreads: Spreads | None, fixed: dict) -> Spreads:
    """One round of `measure_spreads`: each word's own fit with `spreads` (None: the votes' means, the first round),
    then every spread from the residuals."""
    model = _Model(spreads, channels) if spreads is not None else None
    lag_n, lag_d = {}, {}
    fits, offsets = [], []             # (n, residual sums of squares: log length, log height, axis, tag; pairs)
    strays = votes = 0.0
    for v, rect, m, (ry, rx), _ in per_word:
        prm, tag, r = _own_fit(v, rect, model)
        strays += float((1 - r).sum())
        votes += len(r)
        res = v["o"] - (prm[:2] - v["p"]) * math.exp(-prm[3])
        mw = np.zeros_like(m)                  # each px weighted by its chance of being the word's own vote
        mw[ry, rx] = r
        n_lag = _corr(mw, mw)
        d_lag = np.zeros_like(n_lag)
        for c in range(2):
            f = np.zeros_like(m)
            f[ry, rx] = res[:, c]
            d_lag += _corr(mw, mw * f * f) + _corr(mw * f * f, mw) - 2 * _corr(mw * f, mw * f)
        key = n_lag.shape
        lag_n[key] = lag_n.get(key, 0) + n_lag
        lag_d[key] = lag_d.get(key, 0) + d_lag
        fits.append((float(r.sum()), float((r * (v["ll"] - prm[2]) ** 2).sum()),
                     float((r * (v["lh"] - prm[3]) ** 2).sum()), float((r * _half(v["th"] - prm[4]) ** 2).sum()),
                     float((r[:, None] * (v["t"] - tag) ** 2).sum()), n_lag))
        along, up = _axes(prm[4])
        a, b = (rect - prm[:2]) @ along, (rect - prm[:2]) @ up
        half_l, half_h = math.exp(prm[2]) / 2, math.exp(prm[3]) / 2
        offsets += [-half_l - a.min(), half_l - a.max(), -half_h - b.min(), half_h - b.max()]
    # the variogram's lags, pooled over words of every shape on one canvas, lag (0, 0) at its centre
    hh = max(key[0] for key in lag_n) // 2
    ww = max(key[1] for key in lag_n) // 2
    pairs = np.zeros((2 * hh + 1, 2 * ww + 1))
    sq = np.zeros_like(pairs)
    for (h, w), nn in lag_n.items():
        sl = (slice(hh - h // 2, hh + h // 2 + 1), slice(ww - w // 2, ww + w // 2 + 1))
        pairs[sl] += nn
        sq[sl] += lag_d[(h, w)]
    pairs[hh, ww] = 0                                    # lag 0: no difference
    dy, dx = np.nonzero(pairs > 0)
    weight = pairs[dy, dx]
    gamma = sq[dy, dx] / (2 * 2 * weight)
    dist2 = (dy - hh) ** 2.0 + (dx - ww) ** 2.0

    def fitted(log_ell):
        f = 1 - np.exp(-dist2 / (2 * math.exp(2 * log_ell)))
        sill = float((weight * gamma * f).sum() / (weight * f * f).sum())
        return sill, float((weight * (gamma - sill * f) ** 2).sum())

    extent = math.log(max(math.sqrt(float(dist2.max())), 1.0))
    log_ell = minimize_scalar(lambda le: fitted(le)[1], bounds=(math.log(1e-2), extent), method="bounded").x
    sill, _ = fitted(log_ell)
    ell = math.exp(log_ell)
    cell = float(_lattice(ell).sum() ** 2)
    dof = 0.0
    sums = np.zeros(4)
    for n, sl, sh, sa, st, n_lag in fits:
        h, w = n_lag.shape
        yy, xx = np.mgrid[-(h // 2):h // 2 + 1, -(w // 2):w // 2 + 1]
        if n > 0:
            dof += n - float((n_lag * np.exp(-(yy ** 2 + xx ** 2) / (2 * ell ** 2))).sum()) / n
        sums += (sl, sh, sa, st)
    dof = max(dof, np.finfo(float).tiny)
    outlier = strays / votes if model is not None else 0.5     # the first round reads no outliers: a start only
    outlier = min(max(outlier, np.finfo(float).tiny), 1 - np.finfo(float).eps)   # numerical: a share in (0, 1)
    return Spreads(outlier=outlier, centre=math.sqrt(sill), length=math.sqrt(sums[0] / dof),
                   height=math.sqrt(sums[1] / dof),
                   angle=math.degrees(math.sqrt(sums[2] / dof)), tag=math.sqrt(sums[3] / (channels * dof)), cell=cell,
                   edge=math.sqrt(float(np.mean(np.square(offsets))) + 1 / 12), **fixed)
