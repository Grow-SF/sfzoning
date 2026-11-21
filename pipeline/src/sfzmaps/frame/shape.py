"""The cut traces straight lines: the frame of a page as one rotated rectangle with small per-side tilts and bows and
filleted corners (design spec, amendment 2 and amendment 3; ported from the measured spike
`scripts/spikes/issue71_shape.py`, whose report is .superpowers/sdd/2026-11-20-frame-crop/spike-shape-report.md).

**Sign.** `cost_field` gives, per pixel, D = ink x (COST_MAP p0 - COST_FRAME (p1 + p2)): the expected cost of deleting
the pixel less the expected cost of keeping it.  So D > 0 means "cheaper kept" (map ink), D < 0 "cheaper deleted"
(frame or block ink), and paper (no ink) costs nothing either way.  Every search here keeps the region K that
maximises the sum of D over K; what it deletes collects the negative D.

**Geometry.**  The frame is turned by `phi` degrees about the page centre.  With (u, v) the turned coordinates of a
pixel's centre, each side measures e, the distance inward from it (top: v, bottom: -v, left: u, right: -u), and
"along", the coordinate along it.  A side keeps the pixels with e >= its line, beta + tilt x (along - m) + bow x
(1 - t^2), where m is the middle of the side between its neighbours and t runs from -1 to 1 along it.  A side's
`offset` is its line's distance inward from the page edge's midpoint (beta = offset - cos(phi) x half the page across
it).  **An offset of 0 means the side has no frame: it is the page's own edge** (label rule 5), not turned by phi,
tilted or bowed, and it cuts nothing.  So phi is shared by the framed sides only; with none framed, phi = 0.  (With
turned page-edge sides, 1948/8 and 1948/9 came out at +2.56 and -2.10 degrees and lost their corners.)

**Ties** go to the simplest shape: zero tilt, zero bow, the angle nearest 0 and the outermost offset (on 1-bit pages
blank paper has D = 0 exactly, so a side over a paper gap ties over every tilt and bow).

The search, in order (each exact given what it holds fixed):
1. `frame_search`: every angle within +-ANGLE_MAX and all four offsets on a GRID-px grid.  For fixed (top, bottom) the
   best (left, right) separates exactly, since every left candidate lies left of every right one.  Each side's
   candidates are the page edge (offset 0) and turned lines whose middle lies inside the page (from GRID px in on
   the coarse grid, 1 px in the refinements); the turned lines that only clip a page corner, which a page-edge side
   would otherwise become, are not candidates.  Then refined to 1 px, and each side settled outward over ties (see
   `_settle_outward`).  The notch-aware variant scores each rectangle with its best single notch per side.
2. `refine_sides`: per side, tilt and bow within their bounds and the offset re-optimised, from running sums along
   the side's normal, leaving out the columns within a fitted fillet's radius.
3. `fit_arcs`: per corner of two framed sides, the fillet radius from one histogram.
`fit_sides_arcs` runs 2 and 3 as coordinate descent: fillets, sides, fillets, and while any fillet is cut, sides and
fillets once more.  The full-resolution steps bin every page pixel into the canvas cell its centre falls in (exact,
and the same centre rule as `Frame.mask`); the coarse steps warp pooled sums bilinearly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import minimum_filter1d

from sfzmaps.frame.decide import COST_FRAME, COST_MAP

ANGLE_MAX = 2.5  # degrees: the labels' 95% frame tilt 1.56, widened by half and rounded
TILT_MAX = 0.45  # degrees: per-side deviation (labels' 80% parallelism 0.28 x 1.5)
BOW_MAX = 22  # px: per-side bow at the middle (labels' 95% bow 15 x 1.5)
ARC_MAX = 332  # px: the largest fillet radius (the largest labelled, 221 px, x 1.5)
NOTCH_SPAN_MAX = 0.5  # a credited notch covers at most this share of its side (labelled notches: 23-29%)
NOTCH_REACH = 0.5  # a credited notch reaches at most this share of the page inward
ANGLE_STEP = 0.1  # degrees, the plain search's angle grid
NOTCH_ANGLE_STEP = 0.2  # degrees, the notch-aware search's angle grid
GRID = 16  # px, the offsets' grid
REACH = 0.4  # a side's offset runs up to this share of the page inward
END_MARGIN = 1  # px: a side is scored without its neighbours' first kept row (a sliver of their rule can sit there)
ARC_MARGIN = 8  # px: and without the columns this far beyond a fitted fillet's radius

SIDES = ("top", "bottom", "left", "right")
CORNERS = (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"))
_END_CORNERS = {
    "top": (("top", "left"), ("top", "right")),
    "bottom": (("bottom", "left"), ("bottom", "right")),
    "left": (("top", "left"), ("bottom", "left")),
    "right": (("top", "right"), ("bottom", "right")),
}


def cost_field(p: np.ndarray, lum: np.ndarray) -> np.ndarray:
    """D = ink x (COST_MAP p0 - COST_FRAME (p1 + p2)), float32, ink = 1 - lum.  D > 0: cheaper kept."""
    p = np.asarray(p, np.float32)
    ink = 1.0 - np.asarray(lum, np.float32)
    return (ink * (np.float32(COST_MAP) * p[0] - np.float32(COST_FRAME) * (p[1] + p[2]))).astype(np.float32)


def pool(a: np.ndarray, s: int) -> np.ndarray:
    """Sums over s x s cells (the page padded with zeros to a multiple of s)."""
    if s == 1:
        return np.ascontiguousarray(a, np.float32)
    h, w = a.shape
    H, W = -(-h // s) * s, -(-w // s) * s
    b = np.zeros((H, W), np.float32)
    b[:h, :w] = a
    return b.reshape(H // s, s, W // s, s).sum((1, 3), dtype=np.float32)


def _half(side: str, h: int, w: int) -> float:
    return h / 2.0 if side in ("top", "bottom") else w / 2.0


def e_along(side, u, v):
    """(e, along) of frame coordinates (u, v) seen from `side`."""
    return {"top": (v, u), "bottom": (-v, u), "left": (u, v), "right": (-u, v)}[side]


def side_view(A: np.ndarray, side: str) -> np.ndarray:
    """A whole canvas array seen from `side`: rows inward from that edge, columns along it."""
    return {"top": A, "bottom": A[::-1], "left": A.T, "right": A[:, ::-1].T}[side]


class Canvas:
    """Page coordinates -> the frame turned by `phi` degrees about (cx, cy): u = c dx + s dy, v = -s dx + c dy, with
    (dx, dy) from (cx, cy) to a pixel's centre.  Canvas pixel (j, i) covers u in [i - U, i - U + 1) and v in
    [j - V, j - V + 1); U and V are multiples of 8 and exceed the turned page's half extents, so cells of s | 8 line up
    and the canvas edge lies outside the page."""

    def __init__(self, phi: float, cx: float, cy: float, hu: float, hv: float):
        self.phi, self.cx, self.cy = float(phi), float(cx), float(cy)
        t = math.radians(phi)
        self.c, self.sn = math.cos(t), math.sin(t)
        self.U = int(math.ceil((hu + 2) / 8.0) * 8)
        self.V = int(math.ceil((hv + 2) / 8.0) * 8)

    @classmethod
    def page(cls, phi: float, h: int, w: int) -> Canvas:
        t = math.radians(phi)
        c, s = abs(math.cos(t)), abs(math.sin(t))
        return cls(phi, w / 2.0, h / 2.0, c * w / 2 + s * h / 2, s * w / 2 + c * h / 2)

    def uv(self, x, y):
        """(u, v) of the centres of page pixels (x, y)."""
        dx = np.asarray(x, np.float64) + 0.5 - self.cx
        dy = np.asarray(y, np.float64) + 0.5 - self.cy
        return self.c * dx + self.sn * dy, -self.sn * dx + self.c * dy

    def xy(self, u, v):
        """Page coordinates (continuous) of (u, v)."""
        return self.cx + self.c * u - self.sn * v, self.cy + self.sn * u + self.c * v

    def cells(self, s: int) -> tuple[int, int]:
        return 2 * self.V // s, 2 * self.U // s

    def EA(self, side: str) -> tuple[int, int]:
        """(E, A): seen from `side`, row k covers e in [k - E, k - E + 1), column a covers along in [a - A, ...)."""
        return (self.V, self.U) if side in ("top", "bottom") else (self.U, self.V)

    def warp(self, src: np.ndarray, s_src: int, s: int) -> np.ndarray:
        """`src` (page pixels pooled by s_src) drawn on the whole canvas at cells of s px, bilinear."""
        nj, ni = self.cells(s)
        c, sn = self.c, self.sn
        ox, oy = s_src / 2.0 - self.cx, s_src / 2.0 - self.cy
        M = np.array(
            [
                [c * s_src / s, sn * s_src / s, (c * ox + sn * oy + self.U - s / 2.0) / s],
                [-sn * s_src / s, c * s_src / s, (-sn * ox + c * oy + self.V - s / 2.0) / s],
            ]
        )
        return cv2.warpAffine(
            np.ascontiguousarray(src, np.float32),
            M,
            (ni, nj),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    def bin(self, src: np.ndarray, window=None) -> np.ndarray:
        """Forward binning at full resolution: every page pixel of `src` added to the canvas pixel its centre falls
        in, over window (j0, j1, i0, i1) (default the whole canvas).  Exact sums, and the same centre rule as the
        masks (bilinear warping blurs rule ink into the first map row, which the 3:1 cost then cuts)."""
        h, w = src.shape
        nj, ni = self.cells(1)
        j0, j1, i0, i1 = window or (0, nj, 0, ni)
        out = np.zeros((j1 - j0, i1 - i0), np.float64)
        us = np.array([i0, i1]) - self.U
        vs = np.array([j0, j1]) - self.V
        xs, ys = zip(*[self.xy(u, v) for u in us for v in vs])
        x0, x1 = max(0, int(math.floor(min(xs))) - 2), min(w, int(math.ceil(max(xs))) + 2)
        y0, y1 = max(0, int(math.floor(min(ys))) - 2), min(h, int(math.ceil(max(ys))) + 2)
        if x1 <= x0 or y1 <= y0:
            return out.astype(np.float32)
        step = max(1, 4_000_000 // (x1 - x0))
        xx = (np.arange(x0, x1) + 0.5 - self.cx)[None, :]
        for ya in range(y0, y1, step):
            yb = min(y1, ya + step)
            yy = (np.arange(ya, yb) + 0.5 - self.cy)[:, None]
            i = np.floor(self.c * xx + self.sn * yy + self.U).astype(np.int64) - i0
            j = np.floor(-self.sn * xx + self.c * yy + self.V).astype(np.int64) - j0
            ok = (i >= 0) & (i < i1 - i0) & (j >= 0) & (j < j1 - j0)
            out += np.bincount(
                j[ok] * (i1 - i0) + i[ok], weights=src[ya:yb, x0:x1][ok].astype(np.float64), minlength=out.size
            ).reshape(out.shape)
        return out.astype(np.float32)


@dataclass
class Frame:
    """The frame: turned by `phi` degrees; per side the offset (px inward of the page edge's midpoint; 0 = the page's
    own edge), a tilt deviation (degrees) and a bow (px at the middle, + = inward); per corner a fillet radius (px)."""

    phi: float = 0.0
    offset: dict = field(default_factory=lambda: dict.fromkeys(SIDES, 0.0))
    tilt: dict = field(default_factory=lambda: dict.fromkeys(SIDES, 0.0))
    bow: dict = field(default_factory=lambda: dict.fromkeys(SIDES, 0.0))
    arc: dict = field(default_factory=lambda: dict.fromkeys(CORNERS, 0.0))

    def framed(self, side: str) -> bool:
        return self.offset[side] > 0

    def copy(self) -> Frame:
        return Frame(self.phi, dict(self.offset), dict(self.tilt), dict(self.bow), dict(self.arc))

    def mask(self, shape: tuple[int, int], chunk: int = 1024) -> np.ndarray:
        """Bool, True = kept by the frame (before notches and blocks), by the pixel-centre rule."""
        h, w = shape
        g = _Lines(self, h, w)
        K = np.zeros((h, w), bool)
        for y0 in range(0, h, chunk):
            y1 = min(h, y0 + chunk)
            y, x = np.mgrid[y0:y1, 0:w]
            K[y0:y1] = g.kept(*g.cv.uv(x, y))
        return K


class _Lines:
    """A frame's lines on a page of h x w: the canvas, each side's beta (page-edge sides: the canvas edge, which cuts
    nothing), the tilts as slopes, and the bound of each side along it."""

    def __init__(self, fr: Frame, h: int, w: int):
        self.fr, self.h, self.w = fr, h, w
        self.cv = Canvas.page(fr.phi, h, w)
        c = math.cos(math.radians(fr.phi))
        self.beta = {
            s: fr.offset[s] - c * _half(s, h, w) if fr.framed(s) else -float(self.cv.EA(s)[0]) for s in SIDES
        }
        self.slope = {s: math.tan(math.radians(fr.tilt[s])) if fr.framed(s) else 0.0 for s in SIDES}
        self.bow = {s: fr.bow[s] if fr.framed(s) else 0.0 for s in SIDES}

    def offset_of(self, side: str, beta: float) -> float:
        return beta + math.cos(math.radians(self.fr.phi)) * _half(side, self.h, self.w)

    def ends(self, side: str) -> tuple[float, float]:
        b = self.beta
        return (b["left"], -b["right"]) if side in ("top", "bottom") else (b["top"], -b["bottom"])

    def bound(self, side: str, along):
        lo, hi = self.ends(side)
        m, hw = (lo + hi) / 2.0, max((hi - lo) / 2.0, 1.0)
        t = (along - m) / hw
        return self.beta[side] + self.slope[side] * (along - m) + self.bow[side] * (1.0 - t * t)

    def kept(self, u, v) -> np.ndarray:
        keep = np.ones(np.shape(u), bool)
        dist = {}
        for side in SIDES:
            if not self.fr.framed(side):
                continue
            e, al = e_along(side, u, v)
            dist[side] = e - self.bound(side, al)
            keep &= dist[side] >= 0
        for c in CORNERS:
            r = self.fr.arc[c]
            if r <= 0 or c[0] not in dist or c[1] not in dist:
                continue
            a, b = dist[c[0]], dist[c[1]]
            near = keep & (a < r) & (b < r)
            if near.any():
                an, bn = a[near], b[near]
                idx = np.flatnonzero(near.ravel())[an + bn + np.sqrt(2.0 * an * bn) < r]
                keep.ravel()[idx] = False
        return keep


# ---------------------------------------------------------------------------------------------------------------
# 1. The rectangle
# ---------------------------------------------------------------------------------------------------------------


def _sat_rows(A: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """S[k, i] = sum of A[:rows[k], :i] for each of the sorted `rows`: (len(rows), ni + 1) float64."""
    out = np.zeros((len(rows), A.shape[1] + 1))
    cum = np.zeros(A.shape[1])
    prev = 0
    for k, r in enumerate(rows):
        if r > prev:
            cum += A[prev:r].sum(0, dtype=np.float64)
            prev = r
        out[k, 1:] = np.cumsum(cum)
    return out


def _rect_best(A: np.ndarray, KT, KB, KL, KR):
    """The rectangle of canvas array A keeping rows [kt, nj - kb) and columns [kl, ni - kr) with the largest sum, over
    the candidate offsets (each sorted outermost first): (cut, kt, kb, kl, kr) with cut = total - kept sum.  For fixed
    (kt, kb) the best (kl, kr) separates exactly (every left candidate lies left of every right one):
    max_{l, r} R(r) - R(l) = max_r R(r) - min_l R(l), R the band's running sum.  Ties: the outermost (first)."""
    nj, ni = A.shape
    KT, KB, KL, KR = (np.asarray(k, np.int64) for k in (KT, KB, KL, KR))
    JT, JB, IL, IR = KT, nj - KB, KL, ni - KR
    rows = np.unique(np.concatenate([JT, JB, [nj]]))
    Sr = _sat_rows(A, rows)
    St, Sb = Sr[np.searchsorted(rows, JT)], Sr[np.searchsorted(rows, JB)]
    total = Sr[-1, ni]
    RR = Sb[:, None, IR] - St[None, :, IR]  # (nB, nT, nR)
    LL = Sb[:, None, IL] - St[None, :, IL]
    val = RR.max(-1) - LL.min(-1)
    b, t = np.unravel_index(np.argmax(val), val.shape)
    kl, kr = int(KL[LL[b, t].argmin()]), int(KR[RR[b, t].argmax()])
    return float(total - val[b, t]), int(KT[t]), int(KB[b]), kl, kr


def _candidates(cv: Canvas, side: str, s: int, h: int, w: int, phi: float, d_lo: float, d_hi: float, step: int = 1):
    """Cell offsets from the canvas edge for `side` at cells of s: 0 (the page edge) and the turned lines whose offset
    from the page edge's midpoint lies in [max(1, d_lo), d_hi], every `step` cells (outermost first)."""
    E = cv.EA(side)[0]
    n = cv.cells(s)[0 if side in ("top", "bottom") else 1]
    base = E - math.cos(math.radians(phi)) * _half(side, h, w)
    k0 = math.ceil((base + max(1.0, d_lo)) / s - 1e-9)
    k1 = min(math.floor((base + d_hi) / s + 1e-9), n // 2 - 1)
    return np.concatenate([[0], np.arange(max(1, k0), k1 + 1, step)]).astype(np.int64)


def _offsets(cv: Canvas, s: int, h: int, w: int, phi: float, ks: dict) -> dict:
    """Side -> offset from the page edge's midpoint of cell offsets `ks` (0: the page edge, offset 0)."""
    c = math.cos(math.radians(phi))
    return {sd: 0.0 if ks[sd] == 0 else float(s * ks[sd] - cv.EA(sd)[0] + c * _half(sd, h, w)) for sd in SIDES}


def _nearest_zero_first(phis) -> np.ndarray:
    phis = np.round(np.asarray(phis, float), 6)
    return phis[np.argsort(np.abs(phis), kind="stable")]


def _better(val: float, best) -> bool:
    return best is None or val > best[0] + 1e-7 * abs(best[0]) + 1e-6


def _plain_search(D4: np.ndarray, h: int, w: int, T: float):
    """The best rectangle over every angle and the GRID-px offset grid, on the canvas at cells of 4: (value, phi,
    offsets)."""
    best = None
    for phi in _nearest_zero_first(np.arange(-ANGLE_MAX, ANGLE_MAX + 1e-9, ANGLE_STEP)):
        cv = Canvas.page(phi, h, w)
        A = cv.warp(D4, 4, 4)
        K = {sd: _candidates(cv, sd, 4, h, w, phi, GRID, REACH * 2 * _half(sd, h, w), GRID // 4) for sd in SIDES}
        _, kt, kb, kl, kr = _rect_best(A, K["top"], K["bottom"], K["left"], K["right"])
        # every 4 px within a grid step of it: the grid's phase alone would otherwise pick the angle (a side 8 px off
        # its rule costs as much as a wrong angle on a small page)
        off = _offsets(cv, 4, h, w, phi, {"top": kt, "bottom": kb, "left": kl, "right": kr})
        K = {sd: _candidates(cv, sd, 4, h, w, phi, off[sd] - GRID, off[sd] + GRID) for sd in SIDES}
        cut, kt, kb, kl, kr = _rect_best(A, K["top"], K["bottom"], K["left"], K["right"])
        if _better(T - cut, best):
            ks = {"top": kt, "bottom": kb, "left": kl, "right": kr}
            best = (T - cut, float(phi), _offsets(cv, 4, h, w, phi, ks))
    return best


def _notch_credit(Av: np.ndarray, ks, d_step: int, d_max: int, a_lo: int, a_hi: int) -> np.ndarray:
    """g[k]: the gain (sum of -D) of the best single notch at the frame's angle from cell offset ks[k] of a side view
    Av (rows [ks[k], d) for d up to ks[k] + d_max every d_step, a span within columns [a_lo, a_hi) no longer than
    NOTCH_SPAN_MAX of it), clipped at 0.  A length-bounded maximum-sum subarray per depth (a sliding minimum of the
    running sums).  The bound keeps a credited notch from standing in for the side itself."""
    n = Av.shape[0]
    L = a_hi - a_lo
    out = np.zeros(len(ks))
    if L <= 2:
        return out
    G = np.zeros((n + 1, L))
    G[1:] = np.cumsum(-Av[:, a_lo:a_hi], 0, dtype=np.float64)
    Lmax = max(1, int(NOTCH_SPAN_MAX * L))
    Lmax -= 1 - Lmax % 2  # odd, for the filter's origin
    for k, t in enumerate(ks):
        ds = np.arange(t + d_step, min(n, t + d_max) + 1, d_step)
        if len(ds) == 0:
            continue
        P = np.zeros((len(ds), L + 1))
        P[:, 1:] = np.cumsum(G[ds] - G[t], 1)
        f = minimum_filter1d(P, size=Lmax, axis=1, mode="nearest", origin=Lmax // 2)
        out[k] = max(0.0, float((P[:, 1:] - f[:, :-1]).max()))
    return out


def _notch_aware_search(D16: np.ndarray, h: int, w: int, T: float, lam: float):
    """Every angle every NOTCH_ANGLE_STEP, cells of 16: each rectangle scored with, per side, its best single notch
    from that side's line (span within the rectangle's own side), credited max(0, gain - lam).  Spans tied to the
    other pair of sides break separability, so per angle it is coordinate descent from the plain best: (top, bottom)
    given (left, right) is exactly separable, and vice versa; tb, lr, tb.  (Without the credit a large notch plus the
    map between it and the side pulls the whole side in.)  Returns (value, phi, offsets)."""
    s = 16
    best = None
    for phi in _nearest_zero_first(np.arange(-ANGLE_MAX, ANGLE_MAX + 1e-9, NOTCH_ANGLE_STEP)):
        cv = Canvas.page(phi, h, w)
        A = cv.warp(D16, s, s).astype(np.float64)
        nj, ni = A.shape
        S = np.zeros((nj + 1, ni + 1))
        S[1:, 1:] = A.cumsum(0).cumsum(1)
        KT = _candidates(cv, "top", s, h, w, phi, GRID, REACH * h)
        KL = _candidates(cv, "left", s, h, w, phi, GRID, REACH * w)
        _, kt, kb, kl, kr = _rect_best(A, KT, KT, KL, KL)
        cache = {}

        def cred(side, lo, hi):
            if (side, lo, hi) not in cache:
                K = KT if side in ("top", "bottom") else KL
                reach = int(NOTCH_REACH * 2 * _half(side, h, w) / s) + 1
                g = _notch_credit(side_view(A, side), K, 2, reach, lo, hi)
                cache[(side, lo, hi)] = np.maximum(0.0, g - lam - 1e-6)
            return cache[(side, lo, hi)]

        for rnd in range(3):
            if rnd % 2 == 0:  # (top, bottom) given (left, right): credits over exactly the kept columns
                lo, hi = kl + 1, ni - kr - 1
                f = S[:, ni - kr] - S[:, kl]
                kb = int(KT[np.argmax(f[nj - KT] + cred("bottom", lo, hi))])
                kt = int(KT[np.argmin(f[KT] - cred("top", lo, hi))])
            else:  # (left, right) given (top, bottom)
                lo, hi = kt + 1, nj - kb - 1
                g = S[nj - kb] - S[kt]
                kr = int(KL[np.argmax(g[ni - KL] + cred("right", lo, hi))])
                kl = int(KL[np.argmin(g[KL] - cred("left", lo, hi))])
        rect = S[nj - kb, ni - kr] - S[kt, ni - kr] - S[nj - kb, kl] + S[kt, kl]
        it = {k: i for i, k in enumerate(KT)}
        il = {k: i for i, k in enumerate(KL)}
        tb_lo, tb_hi, lr_lo, lr_hi = kl + 1, ni - kr - 1, kt + 1, nj - kb - 1
        credit = (
            cred("top", tb_lo, tb_hi)[it[kt]]
            + cred("bottom", tb_lo, tb_hi)[it[kb]]
            + cred("left", lr_lo, lr_hi)[il[kl]]
            + cred("right", lr_lo, lr_hi)[il[kr]]
        )
        val = T - (S[nj, ni] - rect) + credit
        if _better(val, best):
            ks = {"top": kt, "bottom": kb, "left": kl, "right": kr}
            best = (float(val), float(phi), _offsets(cv, s, h, w, phi, ks))
    return best


def _refine_rect(D: np.ndarray, D2, phis, s: int, half_px: float, start: dict, T: float):
    """The best rectangle over angles `phis` with each side's offset within +-half_px of `start` (or the page edge),
    cells of s (s = 1: forward binning; s = 2: the pooled D2 warped): (value, phi, offsets, the canvas at phi)."""
    h, w = D.shape
    best = None
    for phi in _nearest_zero_first(phis):
        cv = Canvas.page(phi, h, w)
        A = cv.bin(D) if s == 1 else cv.warp(D2, 2, 2)
        K = {sd: _candidates(cv, sd, s, h, w, phi, start[sd] - half_px, start[sd] + half_px) for sd in SIDES}
        cut, kt, kb, kl, kr = _rect_best(A, K["top"], K["bottom"], K["left"], K["right"])
        if _better(T - cut, best):
            ks = {"top": kt, "bottom": kb, "left": kl, "right": kr}
            best = (T - cut, float(phi), _offsets(cv, s, h, w, phi, ks), A)
    return best


def _angle_window(step: float, cell: float, h: int, w: int) -> float:
    """How far the coarse angle can be off: half its step, plus the angle that moves a side's end by one coarse
    cell (on a small page a cell's worth of offset trades against a wide angle).  At least 0.1 degrees."""
    return max(0.1, step / 2 + math.degrees(math.atan(cell / (max(h, w) / 2.0))))


def _settle_outward(fr: Frame, A1: np.ndarray, h: int, w: int, rounds: int = 4) -> Frame:
    """In place: each framed side, given the others, moves outward to the outermost offset (the page edge
    included) whose cut is no worse than its own, scored on its own span: END_MARGIN px from either end are left out,
    where the neighbouring lines' sub-pixel slivers live.  Over blank paper (D = 0) every position ties, and the side
    goes back to the rule that bounds the paper; without this, each grid's rounding (a sliver of a neighbour's rule
    left in its first kept row) pays a side to cut ever deeper into a free strip.  Only outward: an inward move beyond the refinement's
    window is what the notch-aware start guards against."""
    for _ in range(rounds):
        moved = False
        for side in SIDES:
            if not fr.framed(side):
                continue
            g = _Lines(fr, h, w)
            E, Aoff = g.cv.EA(side)
            lo, hi = g.ends(side)
            view = side_view(A1, side)
            a0 = max(0, int(math.ceil(lo + END_MARGIN + Aoff - 0.5)))
            a1 = min(view.shape[1], int(math.ceil(hi - END_MARGIN + Aoff - 0.5)))
            kc = int(round(g.beta[side] + E))
            if a1 <= a0 or kc <= 0:
                continue
            cut = np.zeros(kc + 1)
            cut[1:] = np.cumsum(view[:kc, a0:a1].sum(1, dtype=np.float64))  # cut[k]: D cut by keeping rows >= k
            ks = _candidates(g.cv, side, 1, h, w, fr.phi, 1.0, fr.offset[side])
            ks = ks[ks <= kc]
            k = int(ks[np.argmin(cut[ks])])  # outermost first: ties go outward
            if k != kc:
                fr.offset[side] = 0.0 if k == 0 else g.offset_of(side, float(k - E))
                moved = True
        if not moved:
            break
    return fr


def _angles(centre: float, half: float, step: float) -> np.ndarray:
    phis = centre + np.arange(-half, half + 1e-9, step)
    return phis[np.abs(phis) <= ANGLE_MAX + 1e-9]


def frame_search(D: np.ndarray, notch_aware: bool = False, lam_notch: float = 0.0) -> Frame:
    """The frame rectangle maximising the sum of D (see the module) over what it keeps.

    1. Every angle within +-ANGLE_MAX (every ANGLE_STEP) and all four offsets on a GRID-px grid, cells of 4 px: exact
       per angle.  `notch_aware`: every NOTCH_ANGLE_STEP at cells of 16 px, each rectangle scored with its best
       single notch per side (span at most NOTCH_SPAN_MAX of the side, credited above `lam_notch`).
    2. Refined: angle +-0.1 degrees every 0.02 with offsets +-24 px at 2 px, then +-0.02 every 0.01 with offsets
       +-4 px at 1 px (forward binning).
    A side whose best offset is 0 is the page's own edge; with no framed side, phi = 0."""
    D = np.ascontiguousarray(D, np.float32)
    h, w = D.shape
    T = float(D.sum(dtype=np.float64))
    if notch_aware:
        _, phi, off = _notch_aware_search(pool(D, 16), h, w, T, lam_notch)
        win = _angle_window(NOTCH_ANGLE_STEP, 16, h, w)
    else:
        _, phi, off = _plain_search(pool(D, 4), h, w, T)
        win = _angle_window(ANGLE_STEP, 4, h, w)
    D2 = pool(D, 2)
    _, phi, off, _ = _refine_rect(D, D2, _angles(phi, math.ceil(win / 0.05 - 1e-9) * 0.05, 0.05), 2, 24, off, T)
    _, phi, off, _ = _refine_rect(D, D2, _angles(phi, 0.04, 0.02), 2, 8, off, T)
    _, phi, off, A1 = _refine_rect(D, None, _angles(phi, 0.02, 0.01), 1, 4, off, T)
    fr = _settle_outward(Frame(phi=float(phi), offset=off), A1, h, w)
    if not any(fr.framed(s) for s in SIDES):
        fr.phi = 0.0
    return fr


# ---------------------------------------------------------------------------------------------------------------
# 2. Per-side tilt and bow; 3. corner fillets
# ---------------------------------------------------------------------------------------------------------------


def _refine_sides(fr: Frame, A1: np.ndarray, h: int, w: int, rounds: int = 2) -> Frame:
    """In place on `fr`: each framed side's tilt (+-TILT_MAX), bow (+-BOW_MAX) and offset (+-24 px, staying > 0)
    minimising the sum of D it cuts, from running sums along its normal on the full-resolution canvas A1 at the
    frame's angle.  The neighbours' first kept row (END_MARGIN), or a fitted fillet's radius plus ARC_MARGIN, at
    either end is left out: the neighbour and the fillet decide there.  Ties: zero tilt, zero bow, the outermost offset."""
    tmax = math.tan(math.radians(TILT_MAX))
    for _ in range(rounds):
        for side in SIDES:
            if not fr.framed(side):
                continue
            g = _Lines(fr, h, w)
            E, Aoff = g.cv.EA(side)
            lo, hi = g.ends(side)
            m, hw = (lo + hi) / 2.0, max((hi - lo) / 2.0, 1.0)
            view = side_view(A1, side)
            c_lo, c_hi = _END_CORNERS[side]
            f_lo = lo + (fr.arc[c_lo] + ARC_MARGIN if fr.arc[c_lo] > 0 else END_MARGIN)
            f_hi = hi - (fr.arc[c_hi] + ARC_MARGIN if fr.arc[c_hi] > 0 else END_MARGIN)
            a0 = max(0, int(math.ceil(f_lo + Aoff - 0.5)))
            a1 = min(view.shape[1], int(math.ceil(f_hi + Aoff - 0.5)))
            if a1 - a0 < 2:
                continue
            R = 24 + int(math.ceil(tmax * hw)) + BOW_MAX + 4
            beta0, d0 = g.beta[side], fr.offset[side]
            base = int(math.floor(beta0 + E))
            e0, e1 = max(0, base - R), min(view.shape[0], base + R)
            C = np.zeros((e1 - e0 + 1, a1 - a0))
            C[1:] = np.cumsum(view[e0:e1, a0:a1], 0, dtype=np.float64)
            al = np.arange(a0, a1) - Aoff + 0.5
            x = (al - m) / hw

            def score(taus, bows, offs, step):
                """Least cut over the grid, simplest first so that a strict comparison keeps it on ties."""
                offs = np.asarray([o for o in offs if d0 + o >= 1.0], float)
                best = (np.inf, 0.0, 0.0, 0.0)
                if len(offs) == 0:
                    return best
                cols = np.arange(0, a1 - a0, step)
                for tau in sorted(taus, key=abs):
                    for bw in sorted(bows, key=abs):
                        line = tau * (al[cols] - m) + bw * (1 - x[cols] ** 2)
                        kk = np.ceil(beta0 + offs[:, None] + line[None, :] + E - 0.5).astype(np.int64) - e0
                        np.clip(kk, 0, C.shape[0] - 1, out=kk)
                        tot = C[kk, cols[None, :]].sum(1)
                        i = int(np.argmin(tot))  # offs ascending: ties go outermost
                        if tot[i] < best[0] - 1e-6:
                            best = (float(tot[i]), float(tau), float(bw), float(offs[i]))
                return best

            b = score(
                np.tan(np.radians(np.linspace(-TILT_MAX, TILT_MAX, 7))),
                np.arange(-BOW_MAX, BOW_MAX + 1, 2),
                np.arange(-24, 25),
                2,
            )
            if not np.isfinite(b[0]):
                continue
            dt = math.tan(math.radians(TILT_MAX / 3))
            taus = np.unique(np.clip(np.r_[b[1] + np.linspace(-dt, dt, 13), 0.0], -tmax, tmax))
            bows = np.unique(np.clip(np.r_[b[2] + np.arange(-2, 2.01, 0.5), 0.0], -BOW_MAX, BOW_MAX))
            b = score(taus, bows, b[3] + np.arange(-3, 4), 1)
            fr.offset[side] = d0 + b[3]
            fr.tilt[side] = math.degrees(math.atan(b[1]))
            fr.bow[side] = b[2]
    return fr


def _fit_arcs(fr: Frame, A1: np.ndarray, h: int, w: int) -> Frame:
    """In place on `fr`: per corner of two framed sides, the fillet radius r in 0..ARC_MAX minimising the sum of D it
    cuts.  The fillet is a circle of radius r tangent to both sides; a pixel at inward distances (a, b) from them is
    cut iff a + b + sqrt(2ab) < r, so the cuts nest and one histogram gives every r.  The sides' first kept rows
    (END_MARGIN) are left out of the score, as in `_settle_outward`: otherwise a fillet over blank paper grows to cut
    a neighbour's sliver (seen: 289 px over a blank strip).  Ties: the smaller radius."""
    g = _Lines(fr, h, w)
    cv = g.cv
    nj, ni = A1.shape
    for c in CORNERS:
        s1, s2 = c
        if not (fr.framed(s1) and fr.framed(s2)):
            fr.arc[c] = 0.0
            continue
        uc = g.beta["left"] if s2 == "left" else -g.beta["right"]
        vc = g.beta["top"] if s1 == "top" else -g.beta["bottom"]
        du = 1 if s2 == "left" else -1
        dv = 1 if s1 == "top" else -1
        span = ARC_MAX + 48
        ua, ub = sorted((uc - 48 * du, uc + span * du))
        va, vb = sorted((vc - 48 * dv, vc + span * dv))
        i0, i1 = max(0, int(ua + cv.U)), min(ni, int(ub + cv.U) + 1)
        j0, j1 = max(0, int(va + cv.V)), min(nj, int(vb + cv.V) + 1)
        if i1 <= i0 or j1 <= j0:
            fr.arc[c] = 0.0
            continue
        jj, ii = np.mgrid[j0:j1, i0:i1]
        u, v = ii - cv.U + 0.5, jj - cv.V + 0.5
        e1, al1 = e_along(s1, u, v)
        e2, al2 = e_along(s2, u, v)
        a = e1 - g.bound(s1, al1)
        b = e2 - g.bound(s2, al2)
        ok = (a >= END_MARGIN) & (b >= END_MARGIN) & (a < ARC_MAX) & (b < ARC_MAX)
        rs = a[ok] + b[ok] + np.sqrt(2 * a[ok] * b[ok])
        sel = rs < ARC_MAX
        hist = np.bincount(
            np.floor(rs[sel]).astype(np.int64),
            weights=A1[j0:j1, i0:i1][ok][sel].astype(np.float64),
            minlength=ARC_MAX + 1,
        )[: ARC_MAX + 1]
        cut = np.concatenate([[0.0], np.cumsum(hist)[:ARC_MAX]])  # cut[r] = sum of D over the fillet of radius r
        fr.arc[c] = float(np.argmin(cut))
    return fr


def refine_sides(frame: Frame, D: np.ndarray) -> Frame:
    """A copy of `frame` with each framed side's tilt, bow and offset fitted (see `_refine_sides`)."""
    D = np.asarray(D, np.float32)
    fr = frame.copy()
    return _refine_sides(fr, Canvas.page(fr.phi, *D.shape).bin(D), *D.shape)


def fit_arcs(frame: Frame, D: np.ndarray) -> Frame:
    """A copy of `frame` with each corner's fillet radius fitted (see `_fit_arcs`)."""
    D = np.asarray(D, np.float32)
    fr = frame.copy()
    return _fit_arcs(fr, Canvas.page(fr.phi, *D.shape).bin(D), *D.shape)


def fit_sides_arcs(frame: Frame, D: np.ndarray) -> Frame:
    """Sides and fillets by coordinate descent on one binned canvas: fillets first, then the sides (two rounds) with
    the columns within each fillet's radius left out, the fillets again, and if any is cut, the sides and fillets once
    more.  Fitted before the fillets, the sides' bows absorb them (the spike: sides 3 px in, radii 111-115 instead of
    120; on a 900 x 1200 page, bows of -17 px and radii 86-99)."""
    D = np.asarray(D, np.float32)
    h, w = D.shape
    fr = frame.copy()
    A1 = Canvas.page(fr.phi, h, w).bin(D)
    _fit_arcs(fr, A1, h, w)
    _refine_sides(fr, A1, h, w)
    _fit_arcs(fr, A1, h, w)
    if any(r > 0 for r in fr.arc.values()):
        _refine_sides(fr, A1, h, w, rounds=1)
        _fit_arcs(fr, A1, h, w)
    return fr
