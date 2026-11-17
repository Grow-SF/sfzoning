"""Word boxes from the detector's word heads (`textnet.train.Recipe.heads="words"`, `textnet.model` CORE, GEO, TAG):
every px votes for its word's box and carries its word's tag, weighted by its core chance, and the words are the
components of a mixture over those votes.  No level on the core chance, no merge IoU, no shrink ratio, and no rule
drops a box: each keeps its chance and its log-odds of being a word.

`core_words(core_logit, geo, tag, ink, spreads, prior)` works on one page's arrays (the net's outputs over a page,
tiles stitched, are Task 8's):
  core_logit  float (h, w): the CORE logit of each px (its chance of being in a word, the word's whole labelled
              rectangle, `shrink.prepare`, is sigmoid(core_logit); read in log space, so no chance is 0 or 1)
  geo         float (5, h, w) or (6, h, w): the distances in px from a px to its word's top, bottom, start and end
              edges (`textnet.model.distances` of GEO), then its reading direction: the angle in degrees
              counter-clockwise from left-to-right (as `data.crop_targets(..., words=True)`'s geo), or its cosine and
              sine (the net's channels 5-6, any length)
  tag         float (k, h, w): each px's word tag (TAG)
  ink         bool (h, w): the page's ink

The model (one likelihood for every decision).  Every px casts a vote: its four distances, its reading direction and
its tag, as the net gives them (a vote's box is the rectangle they give from the px's (col, row), `data._geometry`
inverted exactly).
  - With chance w = sigmoid(core_logit) the px is a word px, and its vote is one word's (the words a vote may belong
    to share it by their masses).  A word is a reading direction phi, its four edges along phi's axes, and a mean tag.
    At px p the word implies the distances from p to its edges along its own axes, and its direction; the vote's
    distances differ from those by Gaussian errors of sd `spreads.distance` px each, its direction by a Gaussian
    error of sd `spreads.angle` degrees, its tag from the word's by sd `spreads.tag` each channel.  A vote reads its
    word the other way round (its start at the word's end, its top at the word's foot) with probability
    `spreads.reversed`, as measured; the likelier reading is decided inside the likelihood.
  - Otherwise the px is background: each distance is half-normal of sd `spreads.background_distance`, the direction
    any, and the tag Gaussian about `spreads.background_tag_mean` of sd `spreads.background_tag`.
  - Neighbouring px's votes are not independent (the net's outputs are smooth): n votes carry about n / `spreads.cell`
    independent votes' evidence, so the log likelihood is divided by `cell`.
  - The score of a set of words: prior.slope x (the log likelihood / cell) + prior.log_prior for each word.  The slope
    and the log prior are fitted on the decisions this module makes, against Terra's words (`prior_decisions`,
    `fit_word_prior`); the decision to have a word is a fitted logit of its gain.
  The errors are modelled on the distances and the direction, as the net makes them, not on the box's corners: a
  vote's 8 corner coordinates come from its 4 distances and 1 direction, so they are not 8 independent errors.
All the spreads are measured on the training quarters' predictions against Terra's words (`measure_spreads`); Task 8
runs it and the prior's fit.

The fit (tractable, every decision by the score):
  1. Proposals: the basins of the votes' centres.  Each px adds its core chance at its vote's box's centre (at the
     page's own px); the sum is smoothed by a Gaussian of the distance spread; each px of the page climbs to its
     highest 8-neighbour until none is higher (steepest ascent), and the px whose votes land in one basin are one
     proposal.  Each is fitted as one word (EM, background in the model) and is a word when that raises the score.
  2. The search: from those words and the background, while some move raises the configuration's total score (each
     word's log likelihood on its own votes, the background's on the rest, the prior per word), the move raising it
     most is made, and only a move that raises it: split a word (its votes parted by the two-word mixture, each part
     refitted), or join two words whose boxes' bounds meet.  Each move's change is the exact change of the tracked
     score, so the search ends.  Absorbing a proposal that was not a word into a word never lowers the score (a word
     only adds to what the background explains), so every word, when made, takes the free proposals with votes in its
     bounds, each by the word it gains most under (one batch of absorb moves, booked exactly).
  3. One soft EM over all the words and the background on all the votes near them (a word's votes counted within
     PRECISION_SDS distance spreads of its box's bounds: farther, its density is below double precision of its
     peak).  Each vote's chance of being each word's comes from it.
  Each EM's M-step is generalised: the direction takes a Gauss-Newton step on both of its terms (the votes' implied
  edges turn with it, as well as the direction errors), with the edges refitted at it; of that, the edges refitted at
  the old direction, and no change, it keeps the one with the most expected complete log likelihood, so no step lowers
  the likelihood.

The table (`wordboxes.COLUMNS`, the same meaning, so the reader reads these boxes unchanged; then EXTRA_COLUMNS):
  g corners       the word's box, its first edge along its reading direction (the side nearest the chance-weighted
                  sum of its votes' directions), the second towards the letters' feet; clipped to the page (a box
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
  p_word          the word's share of the weighted vote mass: its votes (each vote's chance of being its) weighted by
                  core chance, over its votes
  core_mass       its votes weighted by core chance (the numerator of p_word)
  tag_spread      the spread of its votes' tags about its mean tag (root mean square per channel, votes weighted)
  word_logodds    the model's log-odds that the box is a word: prior.slope x (the log likelihood with it, less
                  without it, the other words held) / cell + prior.log_prior

Coordinates: the distances were measured from a px's numbers (col, row) to the corners of Terra's rectangles
(`data._geometry`), so the boxes come back in the frame of Terra's rectangles, as `wordboxes`' boxes and Terra's words
are compared.

Fixed numbers here are numerical only: EM_STEPS and EM_TOL (when an EM stops), FIT_STEPS (the prior's Newton steps),
CHUNK_PX (votes per pass: memory, the results do not depend on it), PRECISION_SDS (where a word's density falls below
double precision of its peak) and the Gaussian smoothing's extent (OpenCV's: 4 sd).
"""
import dataclasses
import math

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve

from sfzmaps.solver import wordboxes

EXTRA_COLUMNS = ("p_word", "core_mass", "tag_spread", "word_logodds")
EM_STEPS = 200        # numerical only: an EM stops after this many steps ...
EM_TOL = 1e-7         # ... or once its log likelihood rises by less than this per vote
FIT_STEPS = 100       # numerical only: the prior's fit takes at most this many Newton steps
CHUNK_PX = 1 << 20    # votes per pass (memory only: the results do not depend on it)
PRECISION_SDS = math.sqrt(2 * 53 * math.log(2))    # 8.57: beyond, a word's density < double precision of its peak
_LOG2 = math.log(2.0)


@dataclasses.dataclass(frozen=True)
class Spreads:
    """The measured spreads of the model (see the module; `measure_spreads`)."""
    distance: float             # px: sd of a word vote's distance to an edge about the distance its word implies
    angle: float                # degrees: sd of a word vote's reading direction about its word's
    tag: float                  # sd of a word px's tag channel about its word's mean tag
    reversed: float             # the share of word votes that read their word the other way round
    cell: float                 # px: n votes carry n / cell independent votes' evidence
    background_distance: float  # px: sd of a background px's (half-normal) distances
    background_tag: float       # sd of a background px's tag channel about its mean
    background_tag_mean: tuple  # a background px's mean tag, one per channel

    def check(self, channels: int | None = None) -> None:
        sds = {k: getattr(self, k) for k in ("distance", "angle", "tag", "cell", "background_distance",
                                             "background_tag")}
        bad = {k: v for k, v in sds.items() if not (v > 0 and math.isfinite(v))}
        if bad:
            raise ValueError(f"every spread is a positive number, not {bad}")
        if not 0 < self.reversed < 1:
            raise ValueError(f"reversed is a share strictly between 0 and 1, not {self.reversed}")
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


def _sub(v: dict, sel) -> dict:
    return {key: a[sel] for key, a in v.items()}


class _Votes:
    """The votes of a page's px (or of the px in `mask`), read from the arrays when asked (`take`)."""

    def __init__(self, core_logit, geo, tag, spreads: Spreads, mask=None):
        core_logit, self.geo, self.tag = np.asarray(core_logit), np.asarray(geo), np.asarray(tag)
        self.shape = core_logit.shape
        h, w = self.shape
        on = np.ones(self.shape, bool) if mask is None else np.asarray(mask, bool)
        self.ys, self.xs = np.nonzero(on)
        self.index = np.full((h, w), -1, np.int64)
        self.index[self.ys, self.xs] = np.arange(len(self.ys))
        self.x = core_logit[self.ys, self.xs].astype(np.float64)
        self.k = self.tag.shape[0]
        spreads.check(self.k)
        sa = math.radians(spreads.angle)
        self.id, self.ia, self.it = 1 / (2 * spreads.distance ** 2), 1 / (2 * sa ** 2), 1 / (2 * spreads.tag ** 2)
        self.norm = (2 * math.log(2 * math.pi * spreads.distance ** 2) + 0.5 * math.log(2 * math.pi * sa ** 2)
                     + self.k / 2 * math.log(2 * math.pi * spreads.tag ** 2))
        self.lsame, self.lrev = math.log1p(-spreads.reversed), math.log(spreads.reversed)
        self.bd, self.bt = 1 / (2 * spreads.background_distance ** 2), 1 / (2 * spreads.background_tag ** 2)
        self.bmean = np.asarray(spreads.background_tag_mean, np.float64)
        self.bnorm = (4 * (0.5 * math.log(2 * math.pi * spreads.background_distance ** 2) - _LOG2)
                      + math.log(2 * math.pi) + self.k / 2 * math.log(2 * math.pi * spreads.background_tag ** 2))

    def __len__(self):
        return len(self.ys)

    def chance(self, idx=slice(None)) -> np.ndarray:
        return 0.5 * (1 + np.tanh(self.x[idx] / 2))

    def take(self, idx) -> dict:
        """The votes `idx`: p (n, 2), d (n, 4) (top, bottom, start, end), psi (n,), tags t (n, k), w, log w,
        log(1 - w) (from the logit: never infinite), and the background's log density of each vote."""
        ys, xs = self.ys[idx], self.xs[idx]
        d = np.moveaxis(self.geo[:4, ys, xs].astype(np.float64), 0, -1)
        t = np.moveaxis(self.tag[:, ys, xs].astype(np.float64), 0, -1)
        x = self.x[idx]
        lbg = -(d ** 2).sum(1) * self.bd - ((t - self.bmean) ** 2).sum(1) * self.bt - self.bnorm
        return {"p": np.stack([xs, ys], -1).astype(np.float64), "d": d, "psi": _psi(self.geo, ys, xs), "t": t,
                "w": 0.5 * (1 + np.tanh(x / 2)), "lw": -np.logaddexp(0, -x), "lnw": -np.logaddexp(0, x),
                "lbg": lbg}

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
        edges = np.broadcast_to(prm, (len(v["psi"]), 5))[:, 1:]
        ls = self.lsame - ((same - edges) ** 2).sum(1) * self.id - ds ** 2 * self.ia
        lo = self.lrev - ((other - edges) ** 2).sum(1) * self.id - do ** 2 * self.ia
        return np.logaddexp(ls, lo) - ((v["t"] - tag_mean) ** 2).sum(1) * self.it - self.norm, lo > ls

    def mstep(self, v: dict, r: np.ndarray, prm, tags, flip, group=None, count=1):
        """The words' parameters and tags refitted to votes weighted by r (each vote in its likelier reading `flip`):
        the generalised M-step (see the module).  `group` (n,) and `count`: one word per group (prm (count, 5), tags
        (count, k)); else one word."""
        g = np.zeros(len(r), np.int64) if group is None else group
        prm, tags = np.atleast_2d(np.array(prm, np.float64)), np.atleast_2d(np.array(tags, np.float64))
        mass = np.bincount(g, r, count)
        on = mass > 0
        for ch in range(self.k):
            tags[on, ch] = np.bincount(g, r * v["t"][:, ch], count)[on] / mass[on]

        def edges_at(phi):
            cand = prm.copy()
            cand[on, 0] = phi[on]
            same, other, _, _ = _implied(v, cand[g])
            edges = np.where(flip[:, None], other, same)
            for j in range(4):
                cand[on, 1 + j] = np.bincount(g, r * edges[:, j], count)[on] / mass[on]
            return cand

        # a Gauss-Newton step in phi on id x (the votes' implied edges about their means)^2 + ia x (direction
        # errors)^2: the implied edges turn with phi (d pa / d phi = pb, d pb / d phi = -pa), so both terms move it
        same, other, ds, do = _implied(v, prm[g])
        dev = np.where(flip[:, None], other, same)
        along, up = _axes(prm[g, 0])
        pa, pb = (v["p"] * along).sum(-1), (v["p"] * up).sum(-1)
        slope = np.stack([pb, pb, -pa, -pa], -1)
        safe = np.where(on, mass, 1)
        for x in (dev, slope):
            for j in range(4):
                x[:, j] -= (np.bincount(g, r * x[:, j], count) / safe)[g]
        err = np.where(flip, do, ds)
        grad = (2 * self.id * np.bincount(g, r * (dev * slope).sum(1), count)
                - 2 * self.ia * np.bincount(g, r * err, count))
        curve = 2 * self.id * np.bincount(g, r * (slope ** 2).sum(1), count) + 2 * self.ia * mass
        step = prm[:, 0] - np.where(on, grad / np.where(on, curve, 1), 0)
        cands = [prm, edges_at(prm[:, 0]), edges_at(step)]
        q = np.stack([np.bincount(g, r * self.log_word(v, c[g], tags[g])[0], count) for c in cands])
        best = np.argmax(q, 0)                           # the first of equals: no change before a change
        out = np.stack(cands)[best, np.arange(count)]
        return (out, tags) if group is not None else (out[0], tags[0])


# --- the mixture ----------------------------------------------------------------------------------------------------

def _em(V: _Votes, v: dict, prms: np.ndarray, tags: np.ndarray, trace: list | None = None) -> dict:
    """The mixture of len(prms) words and the background over votes `v` (their masses fitted), by generalised EM from
    `prms` (K, 5) and `tags` (K, k): its log likelihood (not yet divided by cell), the words' parameters and tags,
    and each vote's chance of being each word's (n, K).  `trace` collects the log likelihood of every step."""
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
        if trace is not None:
            trace.append(ll)
        resp = np.exp(lj - lp[:, None])
        if ll - prev <= EM_TOL * n:
            break
        prev = ll
        mass = resp.sum(0)
        for j in np.flatnonzero(mass > 0):
            prms[j], tags[j] = V.mstep(v, resp[:, j], prms[j], tags[j], flips[:, j])
        with np.errstate(divide="ignore"):
            log_pi = np.log(mass / mass.sum())
    return {"ll": ll, "prms": prms, "tags": tags, "resp": resp}


def _one(V: _Votes, v: dict, prm=None, tag=None) -> dict:
    """One word fitted to votes `v`, started at (prm, tag) or at the vote of the highest core chance."""
    if prm is None:
        j = int(np.argmax(v["lw"]))
        prm, tag = _own(v["p"][j:j + 1], v["d"][j:j + 1], v["psi"][j:j + 1]), v["t"][j:j + 1]
    return _em(V, v, np.reshape(prm, (1, 5)), np.reshape(tag, (1, -1)))


def _two(V: _Votes, v: dict, one: dict) -> dict:
    """The two-word mixture over votes `v`, started at the one-word fit `one` and at the vote it explains worst (the
    most of its chance of being the word's x minus its log density under it)."""
    ln, _ = V.log_word(v, one["prms"][0], one["tags"][0])
    j = int(np.argmax(one["resp"][:, 0] * (-ln)))
    prm = _own(v["p"][j:j + 1], v["d"][j:j + 1], v["psi"][j:j + 1])[0]
    return _em(V, v, np.stack([one["prms"][0], prm]), np.stack([one["tags"][0], v["t"][j]]))


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
    """Every basin: its votes (order[first[b]:end[b]]), its one-word fit (prm, tag, ll) and the background's log
    likelihood of its votes (ll0)."""
    basin: np.ndarray       # (n,) each vote's basin
    order: np.ndarray       # vote indices by basin
    first: np.ndarray
    end: np.ndarray
    prms: np.ndarray
    tags: np.ndarray
    ll: np.ndarray
    ll0: np.ndarray

    def votes(self, b: int) -> np.ndarray:
        return np.sort(self.order[self.first[b]:self.end[b]])


def _proposals(V: _Votes, spreads: Spreads) -> _Proposals:
    """Every basin with its one-word fit, all at once, a pass at a time over whole basins."""
    basin = _basins(V.centres(), V.chance(), V.shape, spreads.distance)
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


def _fit_basins(V: _Votes, idx, group, starts):
    """One word fitted to each basin's votes at once (generalised EM per basin, background in the model), started at
    its vote of the highest core chance; each basin's EM stops on its own and only the basins still moving are
    computed.  The words' parameters and tags, and each basin's log likelihood with its word and with none."""
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
        vs, g = _sub(v, sel), group[sel]
        ln, flip = V.log_word(vs, prms[g], tags[g])
        word = vs["lw"] + ln
        lp[sel] = np.logaddexp(word, back[sel])
        ll = np.bincount(g, lp[sel], nb)
        moving &= ll - prev > EM_TOL * count
        prev = np.where(moving | (prev == -math.inf), ll, prev)
        keep = moving[g]
        if not keep.any():
            break
        r = np.exp(word[keep] - lp[sel][keep])
        prms, tags = V.mstep(_sub(vs, keep), r, prms, tags, flip[keep], g[keep], nb)
    return prms, tags, np.bincount(group, lp, nb), np.bincount(group, back, nb)


# --- the search -----------------------------------------------------------------------------------------------------

def _region(V: _Votes, lo, hi) -> np.ndarray:
    """The votes of the page px in [lo, hi) (x, y; clipped to the page)."""
    h, w = V.shape
    x0, y0 = max(int(math.floor(lo[0])), 0), max(int(math.floor(lo[1])), 0)
    x1, y1 = min(int(math.ceil(hi[0])) + 1, w), min(int(math.ceil(hi[1])) + 1, h)
    if x0 >= x1 or y0 >= y1:
        return np.zeros(0, np.int64)
    idx = V.index[y0:y1, x0:x1].ravel()
    return idx[idx >= 0]


def _search(V: _Votes, P: _Proposals, spreads: Spreads, prior: WordPrior, record: list | None = None,
            scores: list | None = None) -> list[dict]:
    """The words (see the module): the basins that pay as words, each taking the free basins it explains; then split
    and join moves while one raises the total score, the best first.  `record` collects every split and join weighed:
    (kind, gain, votes of the one part, votes of the other), the gain that of the extra word.  `scores` collects the
    configuration's total score (slope x (the words' log likelihoods on their votes + the background's on the free
    basins' votes) / cell + the log prior per word), recomputed from the configuration at the start and after every
    move."""
    scale = prior.slope / spreads.cell
    accepted = scale * (P.ll - P.ll0) + prior.log_prior > 0
    words = {int(b): {"idx": P.votes(b), "prms": P.prms[b:b + 1].copy(), "tags": P.tags[b:b + 1].copy(),
                      "ll": float(P.ll[b])} for b in np.flatnonzero(accepted)}
    free = ~accepted                                     # basins whose votes are the background's
    next_id = len(P.first)
    moves = {}

    def fit(idx, start):
        v = V.take(idx)
        return v, _one(V, v, *start)

    def claim(ks):
        """The absorb moves of words `ks`, all at once: a free basin with votes in a word's bounds goes to the word
        under which its votes gain most over the background (the gain, the sum over its votes of log(w N + (1 - w)
        bg) - log((1 - w) bg), is never below 0: a word only adds to what the background explains; a basin no word
        gains on stays free).  Each word is then refitted to its votes, and keeps the refit only where it raises the
        likelihood above the one booked (its own plus the basins' at the word as it was), so no claim lowers the
        score."""
        owners = {}
        for k in ks:
            box = _box(words[k]["prms"][0])
            for b in np.unique(P.basin[_region(V, box.min(0), box.max(0))]):
                if free[b]:
                    owners.setdefault(int(b), []).append(k)
        taken = {}
        for b, ks_b in sorted(owners.items()):
            v = V.take(P.votes(b))
            back = v["lnw"] + v["lbg"]
            best = (0.0, None)
            for k in ks_b:
                ln, _ = V.log_word(v, words[k]["prms"][0], words[k]["tags"][0])
                gain = float((np.logaddexp(v["lw"] + ln, back) - back).sum())
                if gain > best[0]:
                    best = (gain, k)
            if best[1] is not None:
                taken.setdefault(best[1], []).append((b, best[0]))
                free[b] = False
        for k, got in taken.items():
            word = words[k]
            idx = np.union1d(word["idx"], np.concatenate([P.votes(b) for b, _ in got]))
            booked = word["ll"] + sum(float(P.ll0[b]) + gain for b, gain in got)
            _, one = fit(idx, (word["prms"], word["tags"]))
            if one["ll"] > booked:
                words[k] = {"idx": idx, "prms": one["prms"], "tags": one["tags"], "ll": one["ll"]}
            else:
                words[k] = {**word, "idx": idx, "ll": booked}

    def split(k):
        word = words[k]
        if len(word["idx"]) < 2:
            return -math.inf, None
        v, one = fit(word["idx"], (word["prms"], word["tags"]))
        two = _two(V, v, one)
        mine = two["resp"].argmax(1)
        parts = [word["idx"][mine == j] for j in (0, 1)]
        if not all(len(p) for p in parts):
            return -math.inf, None
        new = []
        for j, p in enumerate(parts):
            _, f = fit(p, (two["prms"][j], two["tags"][j]))
            new.append({"idx": p, "prms": f["prms"], "tags": f["tags"], "ll": f["ll"]})
        gain = (new[0]["ll"] + new[1]["ll"] - word["ll"]) / spreads.cell
        if record is not None:
            record.append(("split", gain, parts[0], parts[1]))
        return prior.slope * gain + prior.log_prior, ([k], new)

    def join(a, b):
        idx = np.union1d(words[a]["idx"], words[b]["idx"])
        big = a if len(words[a]["idx"]) >= len(words[b]["idx"]) else b
        _, one = fit(idx, (words[big]["prms"], words[big]["tags"]))
        gain = (words[a]["ll"] + words[b]["ll"] - one["ll"]) / spreads.cell      # the extra word's
        if record is not None:
            record.append(("join", gain, words[a]["idx"], words[b]["idx"]))
        return -(prior.slope * gain + prior.log_prior), ([a, b], [{"idx": idx, "prms": one["prms"],
                                                                  "tags": one["tags"], "ll": one["ll"]}])

    def total():
        ll = sum(w["ll"] for w in words.values()) + float(P.ll0[free].sum())
        return prior.slope * ll / spreads.cell + prior.log_prior * len(words)

    claim(sorted(words))
    if scores is not None:
        scores.append(total())
    while True:
        ids = sorted(words)
        for k in ids:
            if ("split", k) not in moves:
                moves[("split", k)] = split(k)
        if len(ids) > 1:
            boxes = np.array([_box(words[k]["prms"][0]) for k in ids])
            lo, hi = boxes.min(1), boxes.max(1)
            meet = (lo[:, None] <= hi[None, :]).all(-1) & (lo[None, :] <= hi[:, None]).all(-1)
            for i, j in zip(*np.nonzero(np.triu(meet, 1))):
                if ("join", ids[i], ids[j]) not in moves:
                    moves[("join", ids[i], ids[j])] = join(ids[i], ids[j])
        if not moves:
            break
        delta, key = max(((m[0], key) for key, m in moves.items()), key=lambda m: m[0])   # the first of equals
        if not delta > 0:
            break
        old, new = moves[key][1]
        for k in old:
            del words[k]
        gone = set(old)
        moves = {m: val for m, val in moves.items() if not (m[1] in gone or (m[0] == "join" and m[2] in gone))}
        made = []
        for w in new:
            words[next_id] = w
            made.append(next_id)
            next_id += 1
        claim(made)
        if scores is not None:
            scores.append(total())
    return [words[k] for k in sorted(words)]


# --- the soft fit over every word -----------------------------------------------------------------------------------

def _global(V: _Votes, found: list[dict], spreads: Spreads) -> dict:
    """One soft EM over all the words and the background (see the module): each word's votes are those within
    PRECISION_SDS distance spreads of its box's bounds; a vote shares itself among the words it may belong to by their
    masses.  The words' parameters and tags, and per (vote, word) pair its vote, word, the vote's chance of being the
    word's and the votes' data."""
    margin = PRECISION_SDS * spreads.distance
    pv, pw = [], []
    for k, word in enumerate(found):
        box = _box(word["prms"][0])
        idx = np.union1d(_region(V, box.min(0) - margin, box.max(0) + margin), word["idx"])
        pv.append(idx)
        pw.append(np.full(len(idx), k, np.int64))
    pv, pw = np.concatenate(pv), np.concatenate(pw)
    order = np.argsort(pv, kind="stable")
    pv, pw = pv[order], pw[order]
    votes, at = np.unique(pv, return_inverse=True)
    starts = np.flatnonzero(np.r_[True, pv[1:] != pv[:-1]])
    v = V.take(votes)
    vp = _sub(v, at)                                    # the votes' data per pair
    kk = len(found)
    prms = np.concatenate([w["prms"] for w in found])
    tags = np.concatenate([w["tags"] for w in found])
    mass = np.array([len(w["idx"]) for w in found], np.float64)
    back = v["lnw"] + v["lbg"]
    prev = -math.inf
    for _ in range(EM_STEPS):
        with np.errstate(divide="ignore"):
            share = np.log(mass[pw]) - np.log(np.add.reduceat(mass[pw], starts))[at]
        ln, flip = V.log_word(vp, prms[pw], tags[pw])
        lj = vp["lw"] + share + ln
        top = np.maximum.reduceat(lj, starts)
        word = top + np.log(np.add.reduceat(np.exp(lj - top[at]), starts))
        lp = np.logaddexp(word, back)
        ll = float(lp.sum())
        r = np.exp(lj - lp[at])
        if ll - prev <= EM_TOL * len(votes):
            break
        prev = ll
        mass = np.bincount(pw, r, kk)
        prms, tags = V.mstep(vp, r, prms, tags, flip, pw, kk)
    # log(1 - r) of each pair: the vote's other words and the background, summed without the pair's own term
    at_top = np.where(lj == top[at], np.arange(len(pv)), len(pv))
    first_top = np.zeros(len(pv), bool)
    first_top[np.minimum.reduceat(at_top, starts)] = True       # each vote's top pair (its first, of equals)
    rest = np.where(first_top, -np.inf, lj)
    top2 = np.maximum.reduceat(rest, starts)
    shift = np.where(np.isfinite(top2), top2, 0.0)
    with np.errstate(divide="ignore"):
        others_top = np.where(np.isfinite(top2), shift + np.log(np.add.reduceat(np.exp(rest - shift[at]), starts)),
                              -np.inf)                           # the top pair's others
        total = np.add.reduceat(np.exp(lj - top[at]), starts)    # >= 1: it holds the top pair's own 1
        others = np.where(first_top, others_top[at], top[at] + np.log(total[at] - np.exp(lj - top[at])))
    log1m_r = np.logaddexp(others, back[at]) - lp[at]
    return {"prms": prms, "tags": tags, "pair_word": pw, "r": r, "log1m_r": log1m_r, "votes": vp}


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
    if geo.ndim != 3 or geo.shape[0] not in (5, 6) or geo.shape[1:] != core_logit.shape:
        raise ValueError(f"geo is (5 or 6, {h}, {w}), not {geo.shape}")
    if tag.ndim != 3 or tag.shape[1:] != core_logit.shape:
        raise ValueError(f"tag is (k, {h}, {w}), not {tag.shape}")
    return core_logit, geo, tag


def core_words(core_logit, geo, tag, ink, spreads: Spreads, prior: WordPrior, *, kind_p: np.ndarray | None = None,
               sheet: str = "") -> pd.DataFrame:
    """One page's word boxes from the word heads (see the module): `core_logit` (h, w), `geo` (5 or 6, h, w), `tag`
    (k, h, w), `ink` (h, w) bool; `spreads` measured and `prior` fitted (Task 8); `kind_p` (4, h, w) the kinds'
    probabilities (optional); `sheet` written to the table.  A DataFrame of wordboxes.COLUMNS then EXTRA_COLUMNS, one
    row per word."""
    core_logit, geo, tag = _inputs(core_logit, geo, tag)
    if not (math.isfinite(prior.slope) and math.isfinite(prior.log_prior)):
        raise ValueError(f"the prior's slope and log prior are finite numbers, not {prior}")
    height, width = core_logit.shape
    ink = np.asarray(ink, bool)
    V = _Votes(core_logit, geo, tag, spreads)
    found = _search(V, _proposals(V, spreads), spreads, prior)
    if not found:
        return _table([])
    fit = _global(V, found, spreads)
    pw, r, vp = fit["pair_word"], fit["r"], fit["votes"]
    first = [int((V.ys[w["idx"]] * width + V.xs[w["idx"]]).min()) for w in found]
    rows = []
    for k in sorted(range(len(found)), key=lambda j: first[j]):
        mine = pw == k
        rk, wk = r[mine], vp["w"][mine]
        box = _box(fit["prms"][k])
        along, _ = _axes(vp["psi"][mine])
        if ((rk * wk)[:, None] * along).sum(0) @ (box[1] - box[0]) < 0:
            box = np.roll(box, 2, 0)                # the votes, weighted, read it the other way round
        g = _on_page(box, width, height)
        t, ink_px, kinds = _inked(g, ink, kind_p)
        _, vv, _, (a0, a1, b0, b1) = _frame(g)
        _, _, _, (ta0, ta1, tb0, tb1) = _frame(t)
        votes = float(rk.sum())
        mass = float((rk * wk).sum())
        p_word = mass / votes if votes > 0 else 0.0
        spread = (math.sqrt(float((rk * ((vp["t"][mine] - fit["tags"][k]) ** 2).sum(1)).sum()) / (votes * V.k))
                  if votes > 0 else 0.0)
        gain = -float(fit["log1m_r"][mine].sum()) / spreads.cell
        centre = g.mean(0)
        rows.append({"sheet": sheet, "i": len(rows), "level": math.nan, "quarter": -1,
                     "page_x": float(centre[0]), "page_y": float(centre[1]),
                     "axis": float(np.rad2deg(np.arctan2(-vv[1], vv[0])) % 180.0),
                     "length": float(a1 - a0), "height": float(b1 - b0),
                     "t_length": float(ta1 - ta0), "t_height": float(tb1 - tb0),
                     "mean_chance": p_word, "ink_px": ink_px,
                     **{f"p_{kn}": float(kinds[c]) for c, kn in enumerate(wordboxes.KIND_NAMES)},
                     "x": math.nan, "y": math.nan, "lab_lot": -1, "street": False,
                     **dict(zip(wordboxes.CORNER_COLS, [*g.reshape(8).tolist(), *t.reshape(8).tolist()])),
                     "p_word": p_word, "core_mass": mass, "tag_spread": spread,
                     "word_logodds": float(prior.logodds(gain))})
    return _table(rows)


# --- the prior: the decisions, labelled by Terra's words, and their fit (Task 8's) ---------------------------------

def _terra_shares(V: _Votes, word_id: np.ndarray, idx: np.ndarray) -> tuple[float, dict]:
    """Of votes `idx`, weighted by core chance: the share inside Terra's words, and the shares of each Terra word
    among those inside."""
    w = V.chance(idx)
    t = word_id[V.ys[idx], V.xs[idx]]
    total = float(w.sum())
    inside = t > 0
    word_mass = float(w[inside].sum())
    if word_mass <= 0:
        return 0.0, {}
    ids, at = np.unique(t[inside], return_inverse=True)
    return word_mass / total, dict(zip(ids.tolist(), (np.bincount(at, w[inside]) / word_mass).tolist()))


def _apart(V: _Votes, word_id, a: np.ndarray, b: np.ndarray) -> float:
    """The label of an extra word between parts a and b: the chance both are words' and are different Terra words,
    from their votes' core-chance-weighted shares (no cut)."""
    ma, sa = _terra_shares(V, word_id, a)
    mb, sb = _terra_shares(V, word_id, b)
    return ma * mb * (1 - sum(p * sb.get(t, 0.0) for t, p in sa.items()))


def prior_decisions(core_logit, geo, tag, spreads: Spreads, word_id, prior: WordPrior | None = None) -> pd.DataFrame:
    """The decisions `core_words` makes on a page, each an extra word's gain (log likelihood ratio / cell) and its
    label from Terra's words (`word_id`: 1 + the Terra word whose rectangle holds the px, 0 none), a chance with no
    cut: kind "word" for every proposal (one word against none; labelled by the share of its votes' core-weighted
    mass inside Terra's words), and, with a `prior`, "split" and "join" for every split and join the search weighs
    (labelled by the chance its two parts are different Terra words, `_apart`).  Task 8: decisions of "word" alone
    fit a first prior, with which the search's decisions are taken, and all of them fit the prior (`fit_word_prior`);
    on the training quarters only."""
    core_logit, geo, tag = _inputs(core_logit, geo, tag)
    word_id = np.asarray(word_id)
    V = _Votes(core_logit, geo, tag, spreads)
    P = _proposals(V, spreads)
    inside = (word_id[V.ys, V.xs] > 0).astype(np.float64)
    w = V.chance()
    total = np.bincount(P.basin, w, len(P.first))
    shares = np.bincount(P.basin, w * inside, len(P.first)) / np.where(total > 0, total, 1)
    out = pd.DataFrame({"kind": "word", "gain": (P.ll - P.ll0) / spreads.cell, "label": shares})
    if prior is not None:
        record = []
        _search(V, P, spreads, prior, record)
        moves = pd.DataFrame([{"kind": kind, "gain": float(gain), "label": _apart(V, word_id, a, b)}
                              for kind, gain, a, b in record], columns=["kind", "gain", "label"])
        out = pd.concat([out, moves], ignore_index=True)
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
    """exp(-k^2 / (2 ell^2)) for the integers k out to where it underflows."""
    span = int(math.ceil(ell * math.sqrt(2 * 745.0))) + 1
    k = np.arange(-span, span + 1, dtype=np.float64)
    return np.exp(-k ** 2 / (2 * ell ** 2))


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """sum_p a(p) b(p + lag) for every lag (2h - 1, 2w - 1), lag 0 at the centre."""
    return fftconvolve(a[::-1, ::-1], b, mode="full")


def measure_spreads(pages) -> Spreads:
    """The spreads (see the module), measured on predictions against Terra's words, never the owner's answers, each
    from residuals about each word's own fit, so a word's offset from Terra's box is not counted as spread.
    `pages`: dicts of geo, tag (as `core_words` takes them), word_id (int (h, w): 1 + the Terra word whose rectangle
    holds the px, later over earlier, 0 on none: the targets' `tag_id`) and words (float (n, 4, 2): Terra's
    rectangles, the first edge along the reading direction; their directions start each word's own fit).
      Each Terra word's own fit: its direction (the votes' mean direction, each read the way nearer it), the mean of
      the edges its votes imply along it, and its mean tag; a vote's residuals are its implied edges, its direction
      and its tag less those.
      reversed    the share of word votes read the other way round, (count + 1/2) / (votes + 1)
      distance, cell   from the variogram of the implied edges within words, over every lag (pairs of px of one word:
                  differences, so each word's own offset cancels): half the mean squared difference at lag (dy, dx)
                  is sigma^2 (1 - rho), rho Gaussian in the lag of scale ell, fitted by least squares weighted by the
                  pairs; distance = sigma, cell = the sum of rho over the lattice (independent votes give 1)
      angle, tag  the root mean square of the direction and tag residuals, each word's degrees of freedom n less its
                  correlated share (the sum of rho over its px pairs / n)
      background_distance, background_tag_mean, background_tag   the root mean square of a background px's distances;
                  the mean of its tags and their root mean square about it"""
    per_word = []                      # (n, sum sq angle residual, sum sq tag residual, mask, pair counts)
    lag_n, lag_d = {}, {}              # summed over words, keyed by the lag array's shape (h, w)
    reversed_n = votes_n = 0
    bd_sq = bd_n = 0.0
    bt_sum = bt_sq = None
    bt_n = 0
    channels = None
    for page in pages:
        geo, tag = np.asarray(page["geo"]), np.asarray(page["tag"])
        word_id = np.asarray(page["word_id"])
        rects = np.asarray(page["words"], np.float64).reshape(-1, 4, 2)
        channels = tag.shape[0]
        for j in np.unique(word_id[word_id > 0]):
            ys, xs = np.nonzero(word_id == j)
            v = {"p": np.stack([xs, ys], -1).astype(np.float64),
                 "d": np.moveaxis(geo[:4, ys, xs].astype(np.float64), 0, -1), "psi": _psi(geo, ys, xs)}
            r0 = rects[j - 1]
            phi = math.atan2(-(r0[1, 1] - r0[0, 1]), r0[1, 0] - r0[0, 0])
            for _ in range(EM_STEPS):
                _, _, ds, do = _implied(v, np.array([phi, 0, 0, 0, 0]))
                flip = np.abs(do) < np.abs(ds)
                move = float(np.where(flip, do, ds).mean())
                phi += move
                if abs(move) <= EM_TOL:
                    break
            same, other, ds, do = _implied(v, np.array([phi, 0, 0, 0, 0]))
            flip = np.abs(do) < np.abs(ds)
            edges = np.where(flip[:, None], other, same)
            res = edges - edges.mean(0)
            ang = np.where(flip, do, ds)
            ang = ang - ang.mean()
            t = np.moveaxis(tag[:, ys, xs].astype(np.float64), 0, -1)
            reversed_n += int(flip.sum())
            votes_n += len(ys)
            y0, x0 = ys.min(), xs.min()
            m = np.zeros((ys.max() - y0 + 1, xs.max() - x0 + 1))
            m[ys - y0, xs - x0] = 1
            n_lag = np.rint(_corr(m, m))
            d_lag = np.zeros_like(n_lag)
            for c in range(4):
                f = np.zeros_like(m)
                f[ys - y0, xs - x0] = res[:, c]
                d_lag += _corr(m, m * f * f) + _corr(m * f * f, m) - 2 * _corr(f, f)
            key = n_lag.shape
            lag_n[key] = lag_n.get(key, 0) + n_lag
            lag_d[key] = lag_d.get(key, 0) + d_lag
            per_word.append((len(ys), float((ang ** 2).sum()), float(((t - t.mean(0)) ** 2).sum()), n_lag))
        by, bx = np.nonzero(word_id == 0)
        for lo in range(0, len(by), CHUNK_PX):
            yy, xx = by[lo:lo + CHUNK_PX], bx[lo:lo + CHUNK_PX]
            bd_sq += float((geo[:4, yy, xx].astype(np.float64) ** 2).sum())
            bd_n += 4 * len(yy)
            tt = tag[:, yy, xx].astype(np.float64)
            bt_sum = tt.sum(1) if bt_sum is None else bt_sum + tt.sum(1)
            bt_sq = (tt ** 2).sum(1) if bt_sq is None else bt_sq + (tt ** 2).sum(1)
            bt_n += len(yy)
    if not per_word or not bt_n:
        raise ValueError("the pages hold no word px or no background px to measure")
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
    lags = np.stack([dy - hh, dx - ww], -1).astype(np.float64)
    weight = pairs[dy, dx]
    gamma = sq[dy, dx] / (2 * 4 * weight)
    dist2 = (lags ** 2).sum(1)

    def fitted(log_ell):
        f = 1 - np.exp(-dist2 / (2 * math.exp(2 * log_ell)))
        sill = float((weight * gamma * f).sum() / (weight * f * f).sum())
        return sill, float((weight * (gamma - sill * f) ** 2).sum())

    extent = math.log(max(math.sqrt(float(dist2.max())), 1.0))
    log_ell = minimize_scalar(lambda le: fitted(le)[1], bounds=(math.log(1e-2), extent), method="bounded").x
    sill, _ = fitted(log_ell)
    ell = math.exp(log_ell)
    row = _lattice(ell)
    cell = float(row.sum() ** 2)
    dof = 0.0
    sq_a = sq_t = 0.0
    for n, a2, t2, n_lag in per_word:
        h, w = n_lag.shape
        dy, dx = np.mgrid[-(h // 2):h // 2 + 1, -(w // 2):w // 2 + 1]
        rho = np.exp(-(dy ** 2 + dx ** 2) / (2 * ell ** 2))
        dof += n - float((n_lag * rho).sum()) / n
        sq_a += a2
        sq_t += t2
    dof = max(dof, np.finfo(float).tiny)
    mean = bt_sum / bt_n
    return Spreads(distance=math.sqrt(sill), angle=math.degrees(math.sqrt(sq_a / dof)),
                   tag=math.sqrt(sq_t / (channels * dof)), reversed=(reversed_n + 0.5) / (votes_n + 1), cell=cell,
                   background_distance=math.sqrt(bd_sq / bd_n),
                   background_tag=math.sqrt(float((bt_sq / bt_n - mean ** 2).sum()) / channels),
                   background_tag_mean=tuple(float(m) for m in mean))
