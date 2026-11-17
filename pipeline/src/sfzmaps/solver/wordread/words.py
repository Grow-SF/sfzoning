"""Word boxes from the detector's word heads (`textnet.train.Recipe.heads="words"`, `textnet.model` CORE, GEO, TAG):
every px votes for its word's box and carries its word's tag, weighted by its core chance, and the words are the
components of a mixture over those votes.  No level on the core chance, no merge IoU, no shrink ratio, and no rule
drops a box: each keeps its chance.

`core_words(core, geo, tag, ink, spreads, word_prior)` works on one page's arrays (the net's outputs over a page,
tiles stitched, are Task 8's):
  core   float (h, w): each px's chance of being in a word (the CORE logit's sigmoid; the core is the word's whole
         labelled rectangle, `shrink.prepare`), in [0, 1]
  geo    float (5, h, w) or (6, h, w): the distances in px from a px to its word's top, bottom, start and end edges
         (`textnet.model.distances` of GEO), then its reading direction: the angle in degrees counter-clockwise from
         left-to-right (as `data.crop_targets(..., words=True)`'s geo), or its cosine and sine (the net's channels 5-6,
         any length)
  tag    float (k, h, w): each px's word tag (TAG)
  ink    bool (h, w): the page's ink

The model (one likelihood for every decision).  Each px with core chance w > 0 casts a vote: its four distances, its
reading direction and its tag, as the net gives them (a vote's box is the rectangle they give from the px's (col,
row), `data._geometry` inverted exactly).
  - With chance w the px is a word px, and its vote is one word's (the words share the votes by their mixture
    weights).  A word is a reading direction phi, its four edges along phi's axes, and a mean tag.  At px p the word
    implies the distances from p to its edges along its own axes, and its direction; the vote's distances differ from
    those by Gaussian errors of sd `spreads.distance` px each, its direction by a Gaussian error of sd `spreads.angle`
    degrees, its tag from the word's by sd `spreads.tag` each channel.  A vote may read the word the other way round
    (its start at the word's end, its top at the word's foot): either reading, a half each.
  - Otherwise the px is background: each distance is half-normal of sd `spreads.background_distance`, the direction
    is any, and the tag is Gaussian about 0 of sd `spreads.background_tag`.
  - Neighbouring px's votes are not independent (the net's outputs are smooth): a word's n votes carry about
    n / `spreads.cell` independent votes' evidence, so the log likelihood is divided by `cell`.
  - The score of a set of words: that log likelihood plus `word_prior` (a fitted log prior) for each word.
  The errors are modelled on the distances and the direction, as the net makes them, not on the box's corners: a
  vote's 8 corner coordinates come from its 4 distances and 1 direction, so they are not 8 independent errors (a
  corner model of them gains likelihood from splitting one word, more the bigger the word).
All six spreads are measured on the training quarters' predictions against Terra's words (`measure_spreads`), and
`word_prior` is fitted on the same decisions this module makes, one word over none and two over one (`word_gain`,
`extra_word_gain`, `fit_word_prior`); Task 8 runs both.

The fit (tractable, every decision by the score):
  1. Proposals: the basins of the votes' centres.  Each px adds its core chance at its vote's box's centre (at the
     page's own px); the sum is smoothed by a Gaussian of the distance spread; each px of the page climbs to its
     highest 8-neighbour until none is higher (steepest ascent), and the px whose votes land in one basin are one
     proposal.  A word's votes share a centre, so a word is a basin, or a few when its votes are noisy; touching
     words are separate basins; the background's votes, about their own px, make many small basins.
  2. Each proposal is fitted as one word (EM, background in the model) and is a word when that raises the score
     (its gain in log likelihood / cell + word_prior > 0).
  3. Then, while some move raises the score, the best one is made: two words whose boxes' bounds meet are joined
     (one word against the two-word mixture on their px), or a word is split (the two-word mixture, started at the
     word and at its vote it explains worst, against one word).  Proposals that never paid stay background.
  The search is greedy (its moves are local); the likelihood decides each one.

The table (`wordboxes.COLUMNS`, the same meaning, so the reader reads these boxes unchanged; then EXTRA_COLUMNS):
  g corners       the word's box, its first edge along its reading direction (the side nearest the chance-weighted
                  sum of its px's directions), the second towards the letters' feet; clipped to the page (a box
                  running off the page keeps its axes and becomes the extent of its part on the page: never wrapped,
                  never dropped)
  t corners       the trimmed box: the box's axes, the extent of the ink px inside it (each a px wide; their centres at
                  (col + 0.5, row + 0.5)), within the box; the box itself when it holds no ink
  axis            the first edge's direction, degrees counter-clockwise from left-to-right, modulo 180
  length, height  the box's sides along and across it; t_length, t_height the trimmed box's
  page_x, page_y  the box's centre
  ink_px          the ink px inside the box; p_<kind> their mean kind probabilities (`kind_p`, else 0)
  mean_chance     as p_word
  sheet, i        the sheet given; the box's number on the page (the words in the order of their first px, row by row)
  level           NaN: there is no level
  quarter, x, y, lab_lot, street   not placed: -1, NaN, NaN, -1, False.  Task 8 places the boxes as
                  `wordboxes.sheet_boxes` does (`prints._place` of page_x, page_y; `grid.quarter_of`) before reading.
  p_word          the word's share of the weighted vote mass: its votes (each px's posterior of voting for it) weighted
                  by core chance, over the votes; a weak word keeps its low chance, no rule drops it
  core_mass       its votes weighted by core chance (the numerator of p_word)
  tag_spread      the spread of its px's tags about its mean tag (root mean square per channel, votes weighted)

Coordinates: the distances were measured from a px's numbers (col, row) to the corners of Terra's rectangles
(`data._geometry`), so the boxes come back in the frame of Terra's rectangles, as `wordboxes`' boxes and Terra's words
are compared.

Fixed numbers here are numerical only: EM_STEPS and EM_TOL (when an EM stops), CHUNK_PX (px per pass of the
proposals' fits: memory, the results do not depend on it) and the Gaussian smoothing's extent (OpenCV's: 4 sd).
"""
import dataclasses
import math

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from sfzmaps.solver import wordboxes

EXTRA_COLUMNS = ("p_word", "core_mass", "tag_spread")
EM_STEPS = 200        # numerical only: an EM stops after this many steps ...
EM_TOL = 1e-7         # ... or once its log likelihood rises by less than this per vote
CHUNK_PX = 1 << 20    # votes per pass of the proposals' fits (memory only: the results do not depend on it)
_LOG2 = math.log(2.0)


@dataclasses.dataclass(frozen=True)
class Spreads:
    """The measured spreads of the model (see the module; `measure_spreads`)."""
    distance: float             # px: sd of a word vote's distance to an edge about the distance its word implies
    angle: float                # degrees: sd of a word vote's reading direction about its word's
    tag: float                  # sd of a word px's tag channel about its word's mean tag
    cell: float                 # px: a word's n votes carry n / cell independent votes' evidence
    background_distance: float  # px: sd of a background px's (half-normal) distances
    background_tag: float       # sd of a background px's tag channel about 0

    def check(self) -> None:
        bad = {k: v for k, v in dataclasses.asdict(self).items() if not (v > 0 and math.isfinite(v))}
        if bad:
            raise ValueError(f"every spread is a positive number, not {bad}")


# --- votes --------------------------------------------------------------------------------------------------------

def _wrap(x):
    """Angles (radians) into [-pi, pi)."""
    return (np.asarray(x) + math.pi) % (2 * math.pi) - math.pi


def _axes(phi):
    """A reading direction's axes in page px (y down): along (cos, -sin) and up, towards the letters' tops (-sin,
    -cos); each (..., 2)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c, -s], -1), np.stack([-s, -c], -1)


def _psi(geo: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Each px's reading direction (radians, counter-clockwise from left-to-right), from geo's angle or its cosine
    and sine."""
    if geo.shape[0] == 5:
        return np.deg2rad(geo[4, ys, xs].astype(np.float64))
    return np.arctan2(geo[5, ys, xs].astype(np.float64), geo[4, ys, xs].astype(np.float64))


def _box(prm: np.ndarray) -> np.ndarray:
    """A word's rectangle (4, 2) from its parameters (phi, a0, a1, b0, b1): corners in `boxes.word_rect`'s order
    (start-top, end-top, end-foot, start-foot)."""
    v, up = _axes(prm[0])
    a0, a1, b0, b1 = prm[1:]
    return np.array([a0 * v + b1 * up, a1 * v + b1 * up, a1 * v + b0 * up, a0 * v + b0 * up])


def _own(p: np.ndarray, d: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """The parameters (n, 5) of the boxes the votes give along their own directions."""
    v, up = _axes(psi)
    pa, pb = (p * v).sum(-1), (p * up).sum(-1)
    t, b, s, e = d.T
    return np.stack([psi, pa - s, pa + e, pb - b, pb + t], -1)


def _implied(v: dict, prm: np.ndarray):
    """For words `prm` (5,) or one per vote (n, 5): the edges each vote (p, d, psi) implies along the word's axes,
    read as is and the other way round ((n, 4) each, as (a0, a1, b0, b1)), and its direction's error each way
    (radians)."""
    prm = np.broadcast_to(prm, (len(v["psi"]), 5))
    along, up = _axes(prm[:, 0])
    pa, pb = (v["p"] * along).sum(-1), (v["p"] * up).sum(-1)
    t, b, s, e = v["d"].T
    same = np.stack([pa - s, pa + e, pb - b, pb + t], -1)
    other = np.stack([pa - e, pa + s, pb - t, pb + b], -1)
    return same, other, _wrap(v["psi"] - prm[:, 0]), _wrap(v["psi"] + math.pi - prm[:, 0])


class _Votes:
    """The votes of the px with core chance above 0, read from the arrays when asked (`take`)."""

    def __init__(self, core, geo, tag, spreads: Spreads, mask=None):
        self.geo, self.tag = geo, tag
        on = core > 0 if mask is None else (core > 0) & mask
        self.ys, self.xs = np.nonzero(on)
        self.w = core[self.ys, self.xs].astype(np.float64)
        self.k = tag.shape[0]
        sa = math.radians(spreads.angle)
        self.id, self.ia, self.it = 1 / (2 * spreads.distance ** 2), 1 / (2 * sa ** 2), 1 / (2 * spreads.tag ** 2)
        self.norm = (2 * math.log(2 * math.pi * spreads.distance ** 2) + 0.5 * math.log(2 * math.pi * sa ** 2)
                     + self.k / 2 * math.log(2 * math.pi * spreads.tag ** 2) + _LOG2)
        self.bd, self.bt = 1 / (2 * spreads.background_distance ** 2), 1 / (2 * spreads.background_tag ** 2)
        self.bnorm = (4 * (0.5 * math.log(2 * math.pi * spreads.background_distance ** 2) - _LOG2)
                      + math.log(2 * math.pi) + self.k / 2 * math.log(2 * math.pi * spreads.background_tag ** 2))

    def __len__(self):
        return len(self.ys)

    def take(self, idx) -> dict:
        """The votes of px `idx`: p (n, 2), d (n, 4) (top, bottom, start, end), psi (n,), tags t (n, k), w, log w,
        log(1 - w), and the background's log density of each vote."""
        ys, xs = self.ys[idx], self.xs[idx]
        d = np.moveaxis(self.geo[:4, ys, xs].astype(np.float64), 0, -1)
        t = np.moveaxis(self.tag[:, ys, xs].astype(np.float64), 0, -1)
        w = self.w[idx]
        with np.errstate(divide="ignore"):
            lnw = np.log1p(-w)
        lbg = -(d ** 2).sum(1) * self.bd - (t ** 2).sum(1) * self.bt - self.bnorm
        return {"p": np.stack([xs, ys], -1).astype(np.float64), "d": d, "psi": _psi(self.geo, ys, xs), "t": t,
                "w": w, "lw": np.log(w), "lnw": lnw, "lbg": lbg}

    def centres(self) -> np.ndarray:
        """Each vote's box's centre (n, 2), CHUNK_PX votes at a time."""
        out = np.empty((len(self), 2))
        for lo in range(0, len(self), CHUNK_PX):
            v = self.take(slice(lo, lo + CHUNK_PX))
            own = _own(v["p"], v["d"], v["psi"])
            along, up = _axes(own[:, 0])
            out[lo:lo + CHUNK_PX] = (along * ((own[:, 1] + own[:, 2]) / 2)[:, None]
                                     + up * ((own[:, 3] + own[:, 4]) / 2)[:, None])
        return out

    def log_word(self, v: dict, prm: np.ndarray, tag_mean: np.ndarray):
        """log N(vote | a word) of each vote (`prm` (5,) or (n, 5), `tag_mean` (k,) or (n, k)), and whether it reads
        the word the other way round (the likelier reading)."""
        same, other, ds, do = _implied(v, prm)
        edges = np.broadcast_to(prm, (len(v["w"]), 5))[:, 1:]
        ls = -((same - edges) ** 2).sum(1) * self.id - ds ** 2 * self.ia
        lo = -((other - edges) ** 2).sum(1) * self.id - do ** 2 * self.ia
        return np.logaddexp(ls, lo) - ((v["t"] - tag_mean) ** 2).sum(1) * self.it - self.norm, lo > ls

    def updated(self, v: dict, r: np.ndarray, prm: np.ndarray, flip: np.ndarray, group=None, count=1):
        """The words' parameters refitted to votes weighted by r (each vote in its likelier reading `flip`): the
        direction moved by the weighted mean error, then the edges the weighted means of the edges the votes imply
        along it.  `group` (n,) and `count`: one word per group (prm (count, 5)); else one word (prm (5,))."""
        g = np.zeros(len(r), np.int64) if group is None else group
        prm = np.atleast_2d(np.array(prm, np.float64))
        mass = np.bincount(g, r, count)
        on = mass > 0
        _, _, ds, do = _implied(v, prm[g])
        err = np.where(flip, do, ds)
        prm[on, 0] = prm[on, 0] + np.bincount(g, r * err, count)[on] / mass[on]
        same, other, _, _ = _implied(v, prm[g])
        edges = np.where(flip[:, None], other, same)
        for j in range(4):
            prm[on, 1 + j] = np.bincount(g, r * edges[:, j], count)[on] / mass[on]
        return prm if group is not None else prm[0]


# --- the mixture ----------------------------------------------------------------------------------------------------

def _em(V: _Votes, v: dict, prms: np.ndarray, tags: np.ndarray) -> dict:
    """The mixture of len(prms) words and the background over votes `v`, fitted by EM from `prms` (K, 5) and `tags`
    (K, k): its log likelihood (not yet divided by cell), the words' parameters and tags, and each vote's posterior
    of being each word's (n, K)."""
    prms, tags = np.array(prms, np.float64), np.array(tags, np.float64)
    n, kk = len(v["w"]), len(prms)
    log_pi = np.full(kk, -math.log(kk))
    back = v["lnw"] + v["lbg"]
    prev = -math.inf
    for _ in range(EM_STEPS):
        lj = np.empty((n, kk))
        flips = np.empty((n, kk), bool)
        for j in range(kk):
            ln, flips[:, j] = V.log_word(v, prms[j], tags[j])
            lj[:, j] = v["lw"] + log_pi[j] + ln
        lp = np.logaddexp(np.logaddexp.reduce(lj, axis=1), back)
        ll = float(lp.sum())
        resp = np.exp(lj - lp[:, None])
        if ll - prev <= EM_TOL * n:
            break
        prev = ll
        mass = resp.sum(0)
        for j in np.flatnonzero(mass > 0):
            prms[j] = V.updated(v, resp[:, j], prms[j], flips[:, j])
            tags[j] = (resp[:, j, None] * v["t"]).sum(0) / mass[j]
        with np.errstate(divide="ignore"):
            log_pi = np.log(mass / mass.sum())
    return {"ll": ll, "prms": prms, "tags": tags, "resp": resp}


def _worst(V: _Votes, v: dict, one: dict) -> tuple[np.ndarray, np.ndarray]:
    """The vote a one-word fit explains worst, as a word of its own (its box and tag): the most of (its posterior of
    being the word's) x (minus its log density under the word)."""
    ln, _ = V.log_word(v, one["prms"][0], one["tags"][0])
    j = int(np.argmax(one["resp"][:, 0] * (-ln)))
    return _own(v["p"][j:j + 1], v["d"][j:j + 1], v["psi"][j:j + 1])[0], v["t"][j]


def _two(V: _Votes, v: dict, one: dict) -> dict:
    """The two-word mixture over votes `v`, started at the one-word fit `one` and at the vote it explains worst."""
    prm, tag = _worst(V, v, one)
    return _em(V, v, np.stack([one["prms"][0], prm]), np.stack([one["tags"][0], tag]))


def _start(v: dict) -> tuple[np.ndarray, np.ndarray]:
    """A one-word fit's start: the vote of the px with the highest core chance (the first of equals)."""
    j = int(np.argmax(v["w"]))
    return _own(v["p"][j:j + 1], v["d"][j:j + 1], v["psi"][j:j + 1]), v["t"][j:j + 1]


def word_gain(core, geo, tag, spreads: Spreads, mask) -> float:
    """The gain in log likelihood (divided by cell) of one word over none (every vote the background's) on the votes
    of the px in `mask`: the decision a proposal is a word by, with `word_prior` added (infinite when a px is surely a
    word px, core chance 1).  Task 8 computes it on the training quarters' Terra words (a real word) and on the
    proposals away from them (none) for `fit_word_prior`, beside `extra_word_gain`."""
    spreads.check()
    V = _Votes(np.asarray(core), np.asarray(geo), np.asarray(tag), spreads, np.asarray(mask, bool))
    if not len(V):
        return 0.0
    v = V.take(slice(None))
    return (_em(V, v, *_start(v))["ll"] - float((v["lnw"] + v["lbg"]).sum())) / spreads.cell


def extra_word_gain(core, geo, tag, spreads: Spreads, mask) -> float:
    """The gain in log likelihood (divided by cell) of two words over one on the votes of the px in `mask`: the
    decision `core_words` makes by adding `word_prior` (split when it is above 0).  The two-word mixture holds the
    one-word fit, so the gain is at least 0.  Task 8 computes it on the training quarters' Terra words (one word each:
    no real extra word) and pairs of touching words (a real one) for `fit_word_prior`."""
    spreads.check()
    V = _Votes(np.asarray(core), np.asarray(geo), np.asarray(tag), spreads, np.asarray(mask, bool))
    if len(V) < 2:
        return 0.0
    v = V.take(slice(None))
    one = _em(V, v, *_start(v))
    return max(_two(V, v, one)["ll"] - one["ll"], 0.0) / spreads.cell


def fit_word_prior(gains, real) -> float:
    """The log prior of a word, fitted on decisions whose answer is known: each extra word's gain (`word_gain`: one
    word over none; `extra_word_gain`: two over one) and whether it was real (Terra holds a word there; Terra holds
    two, not one).  The decision `core_words` makes, as a
    probability: P(real) = sigmoid(gain + word_prior); the maximum-likelihood word_prior (the log likelihood is
    concave in it).  ValueError when every decision is of one kind (no finite prior)."""
    g = np.asarray(gains, np.float64)
    y = np.asarray(real, bool)
    if len(g) != len(y) or not len(g):
        raise ValueError(f"gains and real are one per decision, not {len(g)} and {len(y)}")
    if y.all() or not y.any():
        raise ValueError("every decision is of one kind: no finite word prior fits them")

    def slope(rho):                     # d log L / d rho: falls with rho, above 0 far below, below 0 far above
        return float((y - 0.5 * (1 + np.tanh((g + rho) / 2))).sum())

    lo, hi = -1.0, 1.0
    while slope(lo) <= 0:
        lo *= 2
    while slope(hi) >= 0:
        hi *= 2
    return float(brentq(slope, lo, hi, xtol=1e-12, rtol=4 * np.finfo(float).eps))


# --- proposals ------------------------------------------------------------------------------------------------------

def _basins(centres: np.ndarray, w: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Each vote's basin (see the module), numbered from 0: the core-chance-weighted sum of votes at their centres'
    px (clipped onto the page), smoothed by a Gaussian of sd `sigma`, climbed by steepest ascent over 8-neighbours."""
    h, wd = shape
    ix = np.clip(np.rint(centres[:, 0]), 0, wd - 1).astype(np.int64)
    iy = np.clip(np.rint(centres[:, 1]), 0, h - 1).astype(np.int64)
    cell = iy * wd + ix
    acc = np.bincount(cell, w, h * wd).reshape(h, wd)
    smooth = cv2.GaussianBlur(acc, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
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
    top = step.ravel()
    while True:
        nxt = top[top]
        if np.array_equal(nxt, top):
            break
        top = nxt
    return np.unique(top[cell], return_inverse=True)[1]


def _proposals(V: _Votes, shape, spreads: Spreads, word_prior: float) -> list[dict]:
    """The basins that pay as a word (see the module): per basin its votes' px indices and its one-word fit.  Fitted
    all at once, a pass at a time over whole basins."""
    basin = _basins(V.centres(), V.w, shape, spreads.distance)
    order = np.lexsort((-V.w, basin))               # by basin, the highest core chance first
    bs = basin[order]
    first = np.flatnonzero(np.r_[True, bs[1:] != bs[:-1]])
    ends = np.r_[first[1:], len(order)]
    out = []
    lo = 0
    while lo < len(first):
        hi = lo + 1
        while hi < len(first) and ends[hi] - first[lo] <= CHUNK_PX:
            hi += 1
        idx = order[first[lo]:ends[hi - 1]]
        out += _fit_basins(V, idx, bs[first[lo]:ends[hi - 1]] - bs[first[lo]], first[lo:hi] - first[lo], spreads,
                           word_prior)
        lo = hi
    return out


def _fit_basins(V: _Votes, idx, group, starts, spreads: Spreads, word_prior: float) -> list[dict]:
    """One word fitted to each basin's votes at once (EM per basin, background in the model), started at its vote of
    the highest core chance; the basins whose word raises the score.  Each basin's EM stops on its own (EM_STEPS,
    EM_TOL), and only the basins still moving are computed."""
    v = V.take(idx)
    nb, n = len(starts), len(idx)
    prms = _own(v["p"][starts], v["d"][starts], v["psi"][starts])
    tags = v["t"][starts].copy()
    back = v["lnw"] + v["lbg"]
    count = np.bincount(group, minlength=nb)
    lp = np.empty(n)
    prev = np.full(nb, -math.inf)
    moving = np.ones(nb, bool)
    for _ in range(EM_STEPS):
        sel = np.flatnonzero(moving[group])
        if not len(sel):
            break
        vs, g = {key: a[sel] for key, a in v.items()}, group[sel]
        ln, flip = V.log_word(vs, prms[g], tags[g])
        word = vs["lw"] + ln
        lp[sel] = np.logaddexp(word, back[sel])
        ll = np.bincount(g, lp[sel], nb)
        moving &= ll - prev > EM_TOL * count
        prev = np.where(moving | (prev == -math.inf), ll, prev)
        keep = moving[g]
        if not keep.any():
            break
        vk, gk = {key: a[keep] for key, a in vs.items()}, g[keep]
        r = np.exp(word[keep] - lp[sel][keep])
        prms = V.updated(vk, r, prms, flip[keep], gk, nb)
        mass = np.bincount(gk, r, nb)
        on = mass > 0
        for ch in range(V.k):
            tags[on, ch] = np.bincount(gk, r * vk["t"][:, ch], nb)[on] / mass[on]
    ll_b = np.bincount(group, lp, nb)
    gain = ll_b - np.bincount(group, back, nb)
    ends = np.r_[starts[1:], n]
    return [{"idx": np.sort(idx[starts[b]:ends[b]]), "prms": prms[b:b + 1].copy(), "tags": tags[b:b + 1].copy(),
             "ll": float(ll_b[b])} for b in np.flatnonzero(gain / spreads.cell + word_prior > 0)]


# --- the search -----------------------------------------------------------------------------------------------------

def _search(V: _Votes, found: list[dict], spreads: Spreads, word_prior: float) -> list[dict]:
    """Join and split words while a move raises the score, the best move first (see the module)."""
    words = dict(enumerate(found))
    next_id = len(words)
    splits, joins = {}, {}

    def split(k):
        v = V.take(words[k]["idx"])
        if len(v["w"]) < 2:
            return -math.inf, None
        one = _em(V, v, words[k]["prms"], words[k]["tags"])
        two = _two(V, v, one)
        return (two["ll"] - one["ll"]) / spreads.cell + word_prior, two

    def join(a, b):
        idx = np.union1d(words[a]["idx"], words[b]["idx"])
        v = V.take(idx)
        bigger = a if len(words[a]["idx"]) >= len(words[b]["idx"]) else b
        one = _em(V, v, words[bigger]["prms"], words[bigger]["tags"])
        two = _em(V, v, np.concatenate([words[a]["prms"], words[b]["prms"]]),
                  np.concatenate([words[a]["tags"], words[b]["tags"]]))
        return (one["ll"] - two["ll"]) / spreads.cell - word_prior, (idx, one)

    while True:
        ids = sorted(words)
        for k in ids:
            if k not in splits:
                splits[k] = split(k)
        live = []
        if len(ids) > 1:
            boxes = np.array([_box(words[k]["prms"][0]) for k in ids])
            lo, hi = boxes.min(1), boxes.max(1)
            meet = (lo[:, None] <= hi[None, :]).all(-1) & (lo[None, :] <= hi[:, None]).all(-1)
            live = [(ids[i], ids[j]) for i, j in zip(*np.nonzero(np.triu(meet, 1)))]
            for key in live:
                if key not in joins:
                    joins[key] = join(*key)
        moves = [(d, 0, k) for k, (d, _) in splits.items()] + [(joins[key][0], 1, key) for key in live]
        if not moves:
            break
        best, kind, key = max(moves, key=lambda m: (m[0], -m[1]))
        if not best > 0:
            break
        if kind == 0:
            two = splits[key][1]
            mine = two["resp"].argmax(1)
            parts = [words[key]["idx"][mine == j] for j in (0, 1)]
            if not all(len(p) for p in parts):
                splits[key] = (-math.inf, None)
                continue
            old, new = [key], []
            for j, p in enumerate(parts):
                fit = _em(V, V.take(p), two["prms"][j:j + 1], two["tags"][j:j + 1])
                new.append({"idx": p, "prms": fit["prms"], "tags": fit["tags"], "ll": fit["ll"]})
        else:
            idx, one = joins[key][1]
            old = list(key)
            new = [{"idx": idx, "prms": one["prms"], "tags": one["tags"], "ll": one["ll"]}]
        for k in old:
            del words[k]
            splits.pop(k, None)
        joins = {pair: j for pair, j in joins.items() if not set(pair) & set(old)}
        for w in new:
            words[next_id] = w
            next_id += 1
    return [words[k] for k in sorted(words)]


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
              "y", "p_word", "core_mass", "tag_spread", *wordboxes.CORNER_COLS,
              *[f"p_{k}" for k in wordboxes.KIND_NAMES])
    return pd.DataFrame(rows, columns=[*wordboxes.COLUMNS, *EXTRA_COLUMNS]).astype(
        {"i": "int32", "quarter": "int8", "lab_lot": "int32", "street": bool, "ink_px": "int64",
         **{c: "float64" for c in floats}})


def core_words(core, geo, tag, ink, spreads: Spreads, word_prior: float, *, kind_p: np.ndarray | None = None,
               sheet: str = "") -> pd.DataFrame:
    """One page's word boxes from the word heads (see the module): `core` (h, w), `geo` (5 or 6, h, w), `tag` (k, h,
    w), `ink` (h, w) bool; `spreads` and `word_prior` measured and fitted (Task 8); `kind_p` (4, h, w) the kinds'
    probabilities (optional); `sheet` written to the table.  A DataFrame of wordboxes.COLUMNS then EXTRA_COLUMNS, one
    row per word."""
    spreads.check()
    if not math.isfinite(word_prior):
        raise ValueError(f"word_prior is a finite log prior, not {word_prior}")
    core = np.asarray(core)
    height, width = core.shape
    geo, tag = np.asarray(geo), np.asarray(tag)
    if geo.ndim != 3 or geo.shape[0] not in (5, 6) or geo.shape[1:] != core.shape:
        raise ValueError(f"geo is (5 or 6, {height}, {width}), not {geo.shape}")
    if tag.ndim != 3 or tag.shape[1:] != core.shape:
        raise ValueError(f"tag is (k, {height}, {width}), not {tag.shape}")
    ink = np.asarray(ink, bool)
    V = _Votes(core, geo, tag, spreads)
    if not len(V):
        return _table([])
    found = _search(V, _proposals(V, core.shape, spreads, word_prior), spreads, word_prior)

    rows = []
    for word in sorted(found, key=lambda wd: int((V.ys[wd["idx"]] * width + V.xs[wd["idx"]]).min())):
        v = V.take(word["idx"])
        fit = _em(V, v, word["prms"], word["tags"])
        r = fit["resp"][:, 0]
        box = _box(fit["prms"][0])
        along, _ = _axes(v["psi"])
        if (((r * v["w"])[:, None] * along).sum(0) @ (box[1] - box[0])) < 0:
            box = np.roll(box, 2, 0)                # the votes, weighted, read it the other way round
        g = _on_page(box, width, height)
        t, ink_px, kinds = _inked(g, ink, kind_p)
        _, vv, _, (a0, a1, b0, b1) = _frame(g)
        _, _, _, (ta0, ta1, tb0, tb1) = _frame(t)
        votes = float(r.sum())
        mass = float((r * v["w"]).sum())
        p_word = mass / votes if votes > 0 else 0.0
        spread = (math.sqrt(float((r * ((v["t"] - fit["tags"][0]) ** 2).sum(1)).sum()) / (votes * V.k))
                  if votes > 0 else 0.0)
        centre = g.mean(0)
        rows.append({"sheet": sheet, "i": len(rows), "level": math.nan, "quarter": -1,
                     "page_x": float(centre[0]), "page_y": float(centre[1]),
                     "axis": float(np.rad2deg(np.arctan2(-vv[1], vv[0])) % 180.0),
                     "length": float(a1 - a0), "height": float(b1 - b0),
                     "t_length": float(ta1 - ta0), "t_height": float(tb1 - tb0),
                     "mean_chance": p_word, "ink_px": ink_px,
                     **{f"p_{k}": float(kinds[c]) for c, k in enumerate(wordboxes.KIND_NAMES)},
                     "x": math.nan, "y": math.nan, "lab_lot": -1, "street": False,
                     **dict(zip(wordboxes.CORNER_COLS, [*g.reshape(8).tolist(), *t.reshape(8).tolist()])),
                     "p_word": p_word, "core_mass": mass, "tag_spread": spread})
    return _table(rows)


# --- measuring the spreads (Task 8's, on the training quarters) ----------------------------------------------------

def measure_spreads(pages) -> Spreads:
    """The spreads (see the module) measured on predictions against Terra's words, never the owner's answers.
    `pages`: dicts of core, geo, tag (as `core_words` takes them), word_id (int (h, w): 1 + the Terra word whose
    rectangle holds the px, later over earlier, 0 on none: the targets' `tag_id`) and words (float (n, 4, 2): Terra's
    rectangles, the first edge along the reading direction).  Each word px's vote is read the way nearer its word's
    direction (as the model reads it).
      distance    the root mean square, over every word px's four distances, of its distance minus the one its word
                  implies at the px (`data._geometry`'s)
      angle       the root mean square of a word px's direction minus its word's (degrees)
      tag         the root mean square, over every channel of every word px, of its tag minus its word's mean tag
      cell        how many px make one independent vote: the mean over words of n |mean distance error|^2 / (4
                  distance^2) (votes of independent errors give 1; votes alike in blocks of m px give m)
      background_distance, background_tag   the root mean square of a background px's distances, and of its tags"""
    sq_d = n_d = sq_a = n_a = sq_t = n_t = sq_bd = n_bd = sq_bt = n_bt = 0.0
    per_word = []                       # (n, |mean distance error|^2)
    for page in pages:
        geo, tag = np.asarray(page["geo"]), np.asarray(page["tag"])
        word_id = np.asarray(page["word_id"])
        rects = np.asarray(page["words"], np.float64).reshape(-1, 4, 2)
        phi = np.arctan2(-(rects[:, 1, 1] - rects[:, 0, 1]), rects[:, 1, 0] - rects[:, 0, 0])
        along, up = _axes(phi)
        a, b = np.einsum("nkc,nc->nk", rects, along), np.einsum("nkc,nc->nk", rects, up)
        truth = np.stack([phi, a.min(1), a.max(1), b.min(1), b.max(1)], -1)
        ys, xs = np.nonzero(word_id > 0)
        if len(ys):
            j = word_id[ys, xs].astype(np.int64) - 1
            v = {"p": np.stack([xs, ys], -1).astype(np.float64),
                 "d": np.moveaxis(geo[:4, ys, xs].astype(np.float64), 0, -1), "psi": _psi(geo, ys, xs)}
            same, other, ds, do = _implied(v, truth[j])
            flip = np.abs(do) < np.abs(ds)
            err = np.where(flip[:, None], other, same) - truth[j, 1:]
            sq_d += float((err ** 2).sum())
            n_d += err.size
            ang = np.where(flip, do, ds)
            sq_a += float((np.rad2deg(ang) ** 2).sum())
            n_a += len(ang)
            t = np.moveaxis(tag[:, ys, xs].astype(np.float64), 0, -1)
            for word in np.unique(j):
                mine = j == word
                dev = t[mine] - t[mine].mean(0)
                sq_t += float((dev ** 2).sum())
                n_t += dev.size
                per_word.append((int(mine.sum()), float((err[mine].mean(0) ** 2).sum())))
        rows = max(1, CHUNK_PX // max(1, word_id.shape[1]))
        for y0 in range(0, word_id.shape[0], rows):
            by, bx = np.nonzero(word_id[y0:y0 + rows] == 0)
            by = by + y0
            sq_bd += float((geo[:4, by, bx].astype(np.float64) ** 2).sum())
            n_bd += 4 * len(by)
            sq_bt += float((tag[:, by, bx].astype(np.float64) ** 2).sum())
            n_bt += tag.shape[0] * len(by)
    if not (n_d and n_t and n_bd and per_word):
        raise ValueError("the pages hold no word px or no background px to measure")
    distance = math.sqrt(sq_d / n_d)
    n, e2 = np.array(per_word, np.float64).T
    return Spreads(distance=distance, angle=math.sqrt(sq_a / n_a), tag=math.sqrt(sq_t / n_t),
                   cell=float((n * e2).mean() / (4 * distance ** 2)),
                   background_distance=math.sqrt(sq_bd / n_bd), background_tag=math.sqrt(sq_bt / n_bt))
