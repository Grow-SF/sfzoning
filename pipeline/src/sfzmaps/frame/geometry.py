"""What the shape decision's parts share: the cost field, the turned frame's coordinates (`Canvas`), the frame itself
(`Frame`, drawn by `Lines`) and a few array helpers.  `shape` fits the frame; `notches`, `blocks` and `paper` add the
notches, the removed blocks and the on-paper slide.  See `shape` for the sign of D and the frame's geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from sfzmaps.frame.decide import COST_FRAME, COST_MAP

SIDES = ("top", "bottom", "left", "right")
CORNERS = (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"))
END_CORNERS = {
    "top": (("top", "left"), ("top", "right")),
    "bottom": (("bottom", "left"), ("bottom", "right")),
    "left": (("top", "left"), ("bottom", "left")),
    "right": (("top", "right"), ("bottom", "right")),
}
TILT_MAX = 0.45  # degrees: a side's tilt deviation from phi (labels' 80% parallelism 0.28 x 1.5)
ARC_MARGIN = 8  # px: a side is fitted without the columns this far beyond a fitted fillet's radius (computational)


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


def half(side: str, h: int, w: int) -> float:
    """Half the page across `side`."""
    return h / 2.0 if side in ("top", "bottom") else w / 2.0


def e_along(side, u, v):
    """(e, along) of frame coordinates (u, v) seen from `side`."""
    return {"top": (v, u), "bottom": (-v, u), "left": (u, v), "right": (-u, v)}[side]


def uv_of(side, e, along):
    """The inverse of `e_along`: (u, v) of (e, along) seen from `side`."""
    return {"top": (along, e), "bottom": (along, -e), "left": (e, along), "right": (-e, along)}[side]


def side_view(A: np.ndarray, side: str) -> np.ndarray:
    """A whole canvas array seen from `side`: rows inward from that edge, columns along it."""
    return {"top": A, "bottom": A[::-1], "left": A.T, "right": A[:, ::-1].T}[side]


def box(xs, ys, h: int, w: int, pad: float = 2.0) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1): the page px covering the points (xs, ys) widened by `pad`, clipped to the page."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    x0, x1 = max(0, int(math.floor(xs.min() - pad))), min(w, int(math.ceil(xs.max() + pad)) + 1)
    y0, y1 = max(0, int(math.floor(ys.min() - pad))), min(h, int(math.ceil(ys.max() + pad)) + 1)
    return y0, max(y0, y1), x0, max(x0, x1)


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

    def warp(self, src: np.ndarray, s_src: int, s: int, window=None, x_off: float = 0.0, y_off: float = 0.0):
        """`src` (page pixels pooled by s_src, its pixel (0, 0) at page (x_off, y_off)) drawn on the canvas at cells of
        s px, bilinear, over window (j0, j1, i0, i1) in cells (default the whole canvas)."""
        nj, ni = self.cells(s)
        j0, j1, i0, i1 = window or (0, nj, 0, ni)
        c, sn = self.c, self.sn
        ox, oy = x_off + s_src / 2.0 - self.cx, y_off + s_src / 2.0 - self.cy
        M = np.array(
            [
                [c * s_src / s, sn * s_src / s, (c * ox + sn * oy + self.U - s / 2.0) / s - i0],
                [-sn * s_src / s, c * s_src / s, (-sn * ox + c * oy + self.V - s / 2.0) / s - j0],
            ]
        )
        return cv2.warpAffine(
            np.ascontiguousarray(src, np.float32),
            M,
            (int(i1 - i0), int(j1 - j0)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    def bin(self, src: np.ndarray, window=None) -> np.ndarray:
        """Forward binning at full resolution: every page pixel of `src` added to the canvas pixel its centre falls
        in, over window (j0, j1, i0, i1) (default the whole canvas).  Exact sums, and the same centre rule as the
        masks (bilinear warping blurs rule ink into the first map row, which the cut then takes)."""
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

    def side_window(self, side: str, src: np.ndarray, e0: int, e1: int, a0: int, a1: int, s: int = 1) -> np.ndarray:
        """`src` seen from `side` at cells of s px: rows e0..e1 inward from the canvas edge, columns a0..a1 along it
        (in cells).  s = 1: the page array `src` binned at full resolution (`bin`); s > 1: `src` pooled by s, warped."""
        nj, ni = self.cells(s)
        get = (lambda win: self.bin(src, win)) if s == 1 else (lambda win: self.warp(src, s, s, win))
        if side == "top":
            out = get((e0, e1, a0, a1))
        elif side == "bottom":
            out = get((nj - e1, nj - e0, a0, a1))[::-1]
        elif side == "left":
            out = get((a0, a1, e0, e1)).T
        else:
            out = get((a0, a1, ni - e1, ni - e0))[:, ::-1].T
        return np.ascontiguousarray(out)


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
        g = Lines(self, h, w)
        K = np.zeros((h, w), bool)
        for y0 in range(0, h, chunk):
            y1 = min(h, y0 + chunk)
            y, x = np.mgrid[y0:y1, 0:w]
            K[y0:y1] = g.kept(*g.cv.uv(x, y))
        return K


class Lines:
    """A frame's lines on a page of h x w: the canvas, each side's beta (page-edge sides: the canvas edge, which cuts
    nothing), the tilts as slopes, and the bound of each side along it."""

    def __init__(self, fr: Frame, h: int, w: int):
        self.fr, self.h, self.w = fr, h, w
        self.cv = Canvas.page(fr.phi, h, w)
        c = math.cos(math.radians(fr.phi))
        self.beta = {s: fr.offset[s] - c * half(s, h, w) if fr.framed(s) else -float(self.cv.EA(s)[0]) for s in SIDES}
        self.slope = {s: math.tan(math.radians(fr.tilt[s])) if fr.framed(s) else 0.0 for s in SIDES}
        self.bow = {s: fr.bow[s] if fr.framed(s) else 0.0 for s in SIDES}

    def offset_of(self, side: str, beta: float) -> float:
        return beta + math.cos(math.radians(self.fr.phi)) * half(side, self.h, self.w)

    def ends(self, side: str) -> tuple[float, float]:
        b = self.beta
        return (b["left"], -b["right"]) if side in ("top", "bottom") else (b["top"], -b["bottom"])

    def bound(self, side: str, along):
        lo, hi = self.ends(side)
        m, hw = (lo + hi) / 2.0, max((hi - lo) / 2.0, 1.0)
        t = (along - m) / hw
        return self.beta[side] + self.slope[side] * (along - m) + self.bow[side] * (1.0 - t * t)

    def kept(self, u, v, skip_side: str | None = None, skip_arc: tuple | None = None) -> np.ndarray:
        """Kept by the frame at frame coordinates (u, v); `skip_side` leaves that side (and the fillets at its
        corners) out, `skip_arc` that corner's fillet."""
        keep = np.ones(np.shape(u), bool)
        dist = {}
        for side in SIDES:
            if not self.fr.framed(side):
                continue
            e, al = e_along(side, u, v)
            dist[side] = e - self.bound(side, al)
            if side != skip_side:
                keep &= dist[side] >= 0
        for c in CORNERS:
            r = self.fr.arc[c]
            if r <= 0 or c[0] not in dist or c[1] not in dist or skip_side in c or c == skip_arc:
                continue
            a, b = dist[c[0]], dist[c[1]]
            near = keep & (a >= 0) & (b >= 0) & (a < r) & (b < r)
            if near.any():
                an, bn = a[near], b[near]
                idx = np.flatnonzero(near.ravel())[an + bn + np.sqrt(2.0 * an * bn) < r]
                keep.ravel()[idx] = False
        return keep

    def kept_box(self, y0: int, y1: int, x0: int, x1: int, **skip) -> np.ndarray:
        """`kept` on the page crop [y0, y1) x [x0, x1)."""
        y, x = np.mgrid[y0:y1, x0:x1]
        return self.kept(*self.cv.uv(x, y), **skip)
