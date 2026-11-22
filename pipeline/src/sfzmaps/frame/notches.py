"""Notches: ruled boxes at the margin cut from the kept region (design spec, amendment 2).

A notch is a rectangle anchored at a side, turned by `psi` degrees (within NOTCH_ANGLE of the frame's angle; the
owner: "Rectangles, but they can be rotated").  In the frame turned by psi about the page centre it deletes the
pixels with "along" in [a0, a1) and inward distance e < d (`geometry.e_along`), out to the canvas edge.

**A notch must be a ruled box** (label rule 3): its gain is the sum of -D (`geometry.cost_field`'s D > 0 means
"cheaper kept") over what it deletes of the kept region, plus beta x the ink in the STRIP px just inside its three
inner sides, per px of side (measured in the first trial of the shape decision).  A cut that follows nothing printed gets no such support.  Notches are
added greedily while the best one's gain exceeds lambda_notch, and **only if its area term alone (the sum of -D) is
positive** (controller ruling C5): side ink is evidence that a box is ruled; it can't stand in for the region being
outside (on 1948/14, beta = 3 paid for 9-px slivers up to 2,800 px deep along pairs of lines, their sums of -D -2,146,
-1,894 and +78).  The
search stages propose only such notches too.  beta above COST_MAP can be gamed (the spike: 1-3 px
slivers along street lines, paid for by beta x a line's ink counted on both sides), so a notch is at least
2 x STRIP + 1 px wide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sfzmaps.frame.geometry import SIDES, Canvas, Frame, box, e_along, half, pool, side_view, uv_of

if TYPE_CHECKING:
    from sfzmaps.frame.shape import Params

STRIP = 4  # px: the strip just inside a notch's inner sides whose ink supports it (measured in the first trial of the shape decision)
NOTCH_MIN_PX = 2 * STRIP + 1  # px: a notch is at least this wide (the same trial)
NOTCH_CAP = 6  # notches per page (computational; the labels have at most 3)
NOTCH_ANGLE = 2.46  # degrees: a notch's angle around phi (the labels' 1.64 x 1.5)
NOTCH_REACH = 0.5  # a notch reaches at most this share of the page inward (computational: to the page's middle)
# Computational settings of the search (not fitted):
COARSE = 8  # px: the global stage's cells
COARSE_ANGLE_STEP = 0.25  # degrees, at most, between the global stage's angles
STAGES = (2, 1)  # px: the cells of the refining stages after it (1: full resolution, forward binning)
FINE_ANGLE = 0.25  # degrees: a refining stage's angle window, at least, either side of the angle before ...
FINE_ANGLE_STEP = 0.05  # degrees: ... and its step
FINE_CELLS = 2  # a refining stage searches the ends and the depth this many of the previous stage's cells either side
AREA_TOP = 64  # the best (depth, end) pairs a stage checks for a positive area term (computational)


@dataclass
class Notch:
    """A notch at `side`, turned by `psi` degrees about the page centre: along in [a0, a1) and e < d there."""

    side: str
    psi: float
    a0: float
    a1: float
    d: float
    gain: float = 0.0  # sum of -D over what it deletes + beta x `ink`, before lambda_notch
    ink: float = 0.0  # the ink in the STRIP px inside its three inner sides, per px of side

    def copy(self) -> Notch:
        return Notch(self.side, self.psi, self.a0, self.a1, self.d, self.gain, self.ink)


def notch_box(n: Notch, h: int, w: int, pad: float = 2.0) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1): the page px the notch can cover (its ends and depth widened by `pad`)."""
    cv = Canvas.page(n.psi, h, w)
    E, _ = cv.EA(n.side)
    pts = [cv.xy(*uv_of(n.side, e, al)) for al in (n.a0 - pad, n.a1 + pad) for e in (-E - 2.0, n.d + pad)]
    xs, ys = zip(*pts)
    return box(xs, ys, h, w, 2)


def notch_crop(n: Notch, h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Bool on the page crop [y0, y1) x [x0, x1): the notch's px (pixel-centre rule)."""
    cv = Canvas.page(n.psi, h, w)
    y, x = np.mgrid[y0:y1, x0:x1]
    e, al = e_along(n.side, *cv.uv(x, y))
    return (al >= n.a0) & (al < n.a1) & (e < n.d)


def notch_mask(n: Notch, shape: tuple[int, int]) -> np.ndarray:
    """Bool on the page: the notch's px."""
    h, w = shape
    out = np.zeros((h, w), bool)
    y0, y1, x0, x1 = notch_box(n, h, w)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = notch_crop(n, h, w, y0, y1, x0, x1)
    return out


def notches_mask(ns, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, bool)
    for n in ns:
        y0, y1, x0, x1 = notch_box(n, *shape)
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] |= notch_crop(n, *shape, y0, y1, x0, x1)
    return out


def _value(Dv, Iv, bw: float, s: int, kc: int, a0: int, a1: int, d: int) -> tuple[float, float]:
    """(value, area term) of the notch rows [0, d) x columns [a0, a1) of a side view, as `_kadane` scores it."""
    n_a = Dv.shape[1]
    area = float(Dv[:d, a0:a1].sum(dtype=np.float64))
    ink = (Iv[d - kc : d, a0:a1].sum() + Iv[:d, a0 : min(a0 + kc, n_a)].sum() + Iv[:d, max(a1 - kc, 0) : a1].sum())
    return area + bw * float(ink) / (kc * s), area


def _kadane(Dv, Iv, bw: float, s: int, kc: int, d_lo: int = 1, d_hi: int | None = None,
            area_positive: bool = False):
    """The best notch on a side view (rows from the canvas edge inward, columns along the side, cells of s px): rows
    [0, d) x columns [a0, a1), at least NOTCH_MIN_PX wide.  Its value: the sum of the gain Dv over it + bw x the ink
    along its three inner sides, each the ink in the strip of kc cells just inside it per px of side.  For every depth
    d the best span is a maximum-sum subarray with end terms (Kadane, with a running maximum of the start term).
    Returns (value, a0, a1, d); value -inf when there is none.

    `area_positive`: only notches whose area term (the sum of Dv over them, without the side ink) is positive, so
    side ink alone can't pay for a notch (controller ruling C5).  Not exact: the best of the AREA_TOP best
    (depth, end) pairs whose best start has a positive area term, and the best notch by area alone."""
    n_e, n_a = Dv.shape
    d_hi = n_e if d_hi is None else min(d_hi, n_e)
    d_lo = max(d_lo, kc)
    if d_hi < d_lo or n_a == 0:
        return -np.inf, 0, 0, 0
    G = np.zeros((n_e + 1, n_a))
    G[1:] = np.cumsum(Dv, 0, dtype=np.float64)
    Q = np.zeros((n_e + 1, n_a))
    Q[1:] = np.cumsum(Iv, 0, dtype=np.float64)
    ds = np.arange(d_lo, d_hi + 1)
    norm = float(kc * s)
    col = G[ds] + bw * (Q[ds] - Q[ds - kc]) / norm  # the gain down to depth d, + the parallel side's strip
    P = np.cumsum(col, 1)
    Pm1 = np.zeros_like(P)
    Pm1[:, 1:] = P[:, :-1]
    if bw:
        Qc = np.zeros((len(ds), n_a + 1))
        Qc[:, 1:] = np.cumsum(Q[ds], 1)
        idx = np.arange(n_a)
        pL = (Qc[:, np.minimum(idx + kc, n_a)] - Qc[:, idx]) / norm  # the strip of kc columns starting at a0
        pR = (Qc[:, idx + 1] - Qc[:, np.maximum(idx + 1 - kc, 0)]) / norm  # ... ending at a1 - 1
        M = bw * pL - Pm1
        end = bw * pR
    else:
        M = -Pm1
        end = 0.0
    wmin = int(math.ceil(NOTCH_MIN_PX / s))  # spans of at least wmin columns: a0 <= a1 - wmin + 1
    Mx = np.full_like(M, -np.inf)
    if wmin <= n_a:
        Mx[:, wmin - 1 :] = np.maximum.accumulate(M, 1)[:, : n_a - wmin + 1]
    val = P + end + Mx
    if area_positive:
        return _area_positive(Dv, Iv, bw, s, kc, d_lo, d_hi, val, M, G, ds, wmin)
    di, a1 = np.unravel_index(int(np.argmax(val)), val.shape)
    if not np.isfinite(val[di, a1]):
        return -np.inf, 0, 0, 0
    a0 = int(np.argmax(M[di, : a1 - wmin + 2]))
    return float(val[di, a1]), a0, int(a1) + 1, int(ds[di])


def _area_positive(Dv, Iv, bw, s, kc, d_lo, d_hi, val, M, G, ds, wmin):
    """`_kadane`'s `area_positive` choice (see there)."""
    best = (-np.inf, 0, 0, 0)
    if bw:
        PA = np.cumsum(G[ds], 1)  # the area term's running sums along the side, per depth
        flat = val.ravel()
        k = min(AREA_TOP, int(np.isfinite(flat).sum()))
        if k > 0:
            top = np.argpartition(-flat, k - 1)[:k]
            for i in top[np.argsort(-flat[top], kind="stable")]:
                di, a1 = np.unravel_index(int(i), val.shape)
                a0 = int(np.argmax(M[di, : a1 - wmin + 2]))
                area = PA[di, a1] - (PA[di, a0 - 1] if a0 > 0 else 0.0)
                if area > 0:
                    best = (float(flat[i]), a0, int(a1) + 1, int(ds[di]))
                    break
    plain = _kadane(Dv, Iv, 0.0, s, kc, d_lo, d_hi)  # the best by area alone
    if np.isfinite(plain[0]) and plain[0] > 0:
        v, _ = _value(Dv, Iv, bw, s, kc, plain[1], plain[2], plain[3])
        if v > best[0]:
            best = (v, plain[1], plain[2], plain[3])
    return best


class _Gains:
    """The gain GK (-D on the kept region, 0 elsewhere) and the ink IK on it, at full resolution and pooled to each
    stage's cells, kept up to date as notches are cut (only the cells a notch touches are pooled again)."""

    def __init__(self, D: np.ndarray, ink: np.ndarray, K: np.ndarray):
        self.GK = np.where(K, -D, np.float32(0)).astype(np.float32)
        self.IK = np.where(K, ink, np.float32(0)).astype(np.float32)
        self.pooled = {s: (pool(self.GK, s), pool(self.IK, s)) for s in (COARSE, *STAGES) if s > 1}

    def at(self, s: int) -> tuple[np.ndarray, np.ndarray]:
        return (self.GK, self.IK) if s == 1 else self.pooled[s]

    def cut(self, y0: int, y1: int, x0: int, x1: int, m: np.ndarray) -> None:
        """Zero the px `m` of the page box [y0, y1) x [x0, x1)."""
        self.GK[y0:y1, x0:x1][m] = 0
        self.IK[y0:y1, x0:x1][m] = 0
        for s, (Gp, Ip) in self.pooled.items():
            ya, yb, xa, xb = y0 // s, -(-y1 // s), x0 // s, -(-x1 // s)
            Gp[ya:yb, xa:xb] = pool(self.GK[ya * s : yb * s, xa * s : xb * s], s)
            Ip[ya:yb, xa:xb] = pool(self.IK[ya * s : yb * s, xa * s : xb * s], s)


def _best_notch(g: _Gains, phi: float, bw: float) -> Notch | None:
    """The best single notch for the gains `g`.  First on cells of COARSE px over every side and every angle within
    phi +-NOTCH_ANGLE; then on cells of each of STAGES, each over the angles within the previous stage's uncertainty
    (`_angle_window`) and FINE_CELLS of its cells around its ends and depth.  (A box a few hundred px across turned
    1 degree moves its edges by half a coarse cell: on cells of 8 its score is flat to 2% over the whole angle
    range.)"""
    h, w = g.GK.shape
    s = COARSE
    Gp, Ip = g.at(s)
    n_ang = int(math.ceil(2 * NOTCH_ANGLE / COARSE_ANGLE_STEP)) + 1
    best = (-np.inf,)
    for dpsi in np.linspace(-NOTCH_ANGLE, NOTCH_ANGLE, n_ang):
        psi = phi + float(dpsi)
        cv = Canvas.page(psi, h, w)
        Gw, Iw = cv.warp(Gp, s, s), cv.warp(Ip, s, s)
        for side in SIDES:
            E, A = cv.EA(side)
            hp = half(side, h, w)
            ne = int((E - hp + NOTCH_REACH * 2 * hp) // s)
            r = _kadane(side_view(Gw, side)[:ne], side_view(Iw, side)[:ne], bw, s, 1, area_positive=True)
            if r[0] > best[0]:
                best = (r[0], Notch(side, psi, float(r[1] * s - A), float(r[2] * s - A), float(r[3] * s - E), r[0]))
    if not np.isfinite(best[0]):
        return None
    n, prev = best[1], s
    for s in STAGES:
        n = _refine_notch(n, *g.at(s), phi, bw, s, prev, h, w)
        if n is None:
            return None
        prev = s
    return n


def _angle_window(n: Notch, cell: int, h: int, w: int) -> float:
    """How far the previous stage's angle can be off: the angle that moves the far end of the notch's longer extent
    (its width, or its depth from the page edge) by FINE_CELLS of that stage's cells; at least FINE_ANGLE."""
    extent = max(n.a1 - n.a0, n.d + half(n.side, h, w), 1.0)
    return max(FINE_ANGLE, math.degrees(math.atan(FINE_CELLS * cell / extent)))


def _refine_notch(n: Notch, G, I, phi, bw, s, prev, h, w) -> Notch | None:
    """The best notch on cells of s (G and I the page arrays pooled by s) at `n`'s side, around `n` (see
    `_best_notch`)."""
    width = _angle_window(n, prev, h, w)
    lo, hi = max(n.psi - width, phi - NOTCH_ANGLE), min(n.psi + width, phi + NOTCH_ANGLE)
    psis = n.psi + FINE_ANGLE_STEP * np.arange(math.ceil((lo - n.psi) / FINE_ANGLE_STEP - 1e-9),
                                               math.floor((hi - n.psi) / FINE_ANGLE_STEP + 1e-9) + 1)
    psis = psis[np.argsort(np.abs(psis - n.psi), kind="stable")]  # the previous angle first: ties keep it
    pad = FINE_CELLS * prev
    best = (-np.inf,)
    for psi in psis:
        cv = Canvas.page(float(psi), h, w)
        E, A = cv.EA(n.side)
        n_e, n_a = cv.cells(s) if n.side in ("top", "bottom") else cv.cells(s)[::-1]
        e1 = min(n_e, int(math.ceil((n.d + E + pad) / s)))
        a0 = max(0, int(math.floor((n.a0 + A - pad) / s)))
        a1 = min(n_a, int(math.ceil((n.a1 + A + pad) / s)))
        if e1 <= 0 or a1 <= a0:
            continue
        Gv = cv.side_window(n.side, G, 0, e1, a0, a1, s)
        Iv = cv.side_window(n.side, I, 0, e1, a0, a1, s)
        d_lo = max(1, int(math.floor((n.d + E - pad) / s)))
        r = _kadane(Gv, Iv, bw, s, max(1, STRIP // s), d_lo=d_lo, d_hi=e1, area_positive=True)
        if r[0] > best[0]:
            best = (r[0], Notch(n.side, float(psi), float((r[1] + a0) * s - A), float((r[2] + a0) * s - A),
                                   float(r[3] * s - E), r[0]))
    return best[1] if np.isfinite(best[0]) else None


def side_ink(n: Notch, IK: np.ndarray) -> float:
    """The ink IK in the STRIP px just inside the notch's three inner sides, per px of side (page px, centre rule)."""
    h, w = IK.shape
    y0, y1, x0, x1 = notch_box(n, h, w)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    cv = Canvas.page(n.psi, h, w)
    y, x = np.mgrid[y0:y1, x0:x1]
    e, al = e_along(n.side, *cv.uv(x, y))
    m = (al >= n.a0) & (al < n.a1) & (e < n.d)
    strip = m & ((e >= n.d - STRIP) | (al < n.a0 + STRIP) | (al >= n.a1 - STRIP))
    return float(IK[y0:y1, x0:x1][strip].sum(dtype=np.float64)) / STRIP


def greedy_notches(D: np.ndarray, ink: np.ndarray, frame: Frame, params: Params, kept=None) -> list[Notch]:
    """Notches one at a time, each the best given the ones before, while its gain exceeds `params.lam_notch`: the
    gain is the sum of -D over the px it deletes from the kept region plus `params.beta` x its side ink (see the
    module).  The kept region starts as `kept` (default the frame's mask).  The search runs on warped and binned
    grids; the gain that decides is the exact one on the page px."""
    D = np.asarray(D, np.float32)
    ink = np.asarray(ink, np.float32)
    K = (frame.mask(D.shape) if kept is None else kept).copy()
    h, w = D.shape
    g = _Gains(D, ink, K)
    out = []
    for _ in range(NOTCH_CAP):
        n = _best_notch(g, frame.phi, params.beta)
        if n is None:
            break
        y0, y1, x0, x1 = notch_box(n, h, w)
        m = np.zeros((y1 - y0, x1 - x0), bool) if y1 <= y0 or x1 <= x0 else notch_crop(n, h, w, y0, y1, x0, x1)
        m &= K[y0:y1, x0:x1]
        if not m.any():
            break
        n.ink = side_ink(n, g.IK)
        area = float(g.GK[y0:y1, x0:x1][m].sum(dtype=np.float64))
        n.gain = area + params.beta * n.ink
        if area <= 0 or n.gain <= params.lam_notch:  # side ink supports a box; it can't make a region outside
            break
        out.append(n)
        K[y0:y1, x0:x1] &= ~m
        g.cut(y0, y1, x0, x1, m)
    return out
