"""Removed blocks as shapes: legend and title blocks as rotated rectangles, SECTION badges and compass roses as
ellipses (design spec, amendment 2, edit 3, and amendment 3).

The candidates are the 8-connected pieces of p2 > `decide.cut_level()` inside the kept region.  Each gets the better
of the best rectangle and the best ellipse, both by local search, scored by the gain of deleting it: the sum of -D (`geometry.cost_field`'s D > 0 means "cheaper kept") over the shape within the kept
region.  **Shapes are bounded by the labelled blocks' sizes, widened by half** (`BlockBounds`): paper is free, so
unbounded ellipses ballooned over blank water in the spike.  A block is kept while its gain, re-scored within the
kept region less the blocks already kept, exceeds lambda_block.  A map patch wrongly called "removed" has a
map-shaped outline, so no rectangle or ellipse takes it without the map around it.

**A rectangle is square to the frame** (design spec amendment 4, the note "after the straight-line test"; owner
2026-11-21): it tries only angles within FRAME_ANGLES of the frame's (the page's own when no side is framed).  The
printed title and legend boxes are ruled square to the map's frame; around the piece's own moment angle they scored
within about 1% of the frame's on real boxes, so the piece's coarse shape tilted them by up to 22 degrees.  An
ellipse still starts from the piece's own moments: a compass needle points where it points.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy import ndimage as ndi
from shapely import affinity
from shapely.geometry import Point, Polygon

from sfzmaps.frame import data
from sfzmaps.frame.decide import cut_level
from sfzmaps.frame.geometry import Canvas, Frame, pool

if TYPE_CHECKING:
    from sfzmaps.frame.shape import Params

WIDEN = 1.5  # the labels' largest block, widened by half (the design's pattern for every bound)
BLOCK_MIN_PX = 50  # px: smaller pieces of p2 > level are not candidates (noise specks; computational)
# Computational settings of the local searches (not fitted):
FRAME_ANGLES = (-0.5, -0.25, 0, 0.25, 0.5)  # degrees around the frame's angle, for rectangles
CROP_MARGIN = (24, 0.3)  # px and share: the search crop around a piece, at least 24 px or 30% of its larger side
ELLIPSE_CELLS = 300  # the ellipse's coarse pass works on cells making the crop at most this many across
_EIGHT = ndi.generate_binary_structure(2, 2)


@dataclass
class Block:
    """A removed block about page point (cx, cy), turned by `theta` degrees.  With (u, v) a px centre's turned
    coordinates from (cx, cy): "rect" removes u in [p[0], p[1]) and v in [p[2], p[3]); "ellipse" removes
    (u / p[0])^2 + (v / p[1])^2 <= 1."""

    kind: str
    cx: float
    cy: float
    theta: float
    p: tuple
    gain: float = 0.0  # the sum of -D over it when it was fitted

    def copy(self) -> Block:
        return Block(self.kind, self.cx, self.cy, self.theta, tuple(self.p), self.gain)

    def extent(self) -> tuple[float, float]:
        """(longer, shorter) full size: the rectangle's sides or the ellipse's axes."""
        if self.kind == "rect":
            a, b = self.p[1] - self.p[0], self.p[3] - self.p[2]
        else:
            a, b = 2 * self.p[0], 2 * self.p[1]
        return max(a, b), min(a, b)


@dataclass(frozen=True)
class BlockBounds:
    """The largest rectangle (long side, short side) and ellipse (major, minor full axis) a block may take, px."""

    rect_max: tuple[float, float]
    ellipse_max: tuple[float, float]


# ---------------------------------------------------------------------------------------------------------------
# The bounds, from the labels
# ---------------------------------------------------------------------------------------------------------------


def _ellipse_of(P: np.ndarray):
    """The ellipse with the polygon's centroid and second moments: (shapely polygon, full major, full minor)."""
    m = cv2.moments(P.astype(np.float32))
    if abs(m["m00"]) < 1e-9:
        return None, 0.0, 0.0
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    cov = np.array([[m["mu20"], m["mu11"]], [m["mu11"], m["mu02"]]]) / m["m00"]
    ev, evec = np.linalg.eigh(cov)
    a, b = 2 * math.sqrt(max(ev[1], 0.0)), 2 * math.sqrt(max(ev[0], 0.0))
    th = math.degrees(math.atan2(evec[1, 1], evec[0, 1]))
    e = affinity.scale(Point(0, 0).buffer(1.0, quad_segs=64), a, b)
    return affinity.translate(affinity.rotate(e, th, origin=(0, 0)), cx, cy), 2 * a, 2 * b


def label_kind(poly) -> tuple[str, float, float]:
    """A labelled block's kind and size: ("ellipse", major, minor) when the ellipse with its centroid and second
    moments overlaps it better (intersection over union) than its minimum-area rectangle does, else ("rect", long
    side, short side)."""
    P = np.asarray(poly, np.float64)
    shp = Polygon(P).buffer(0)
    (_, _), (a, b), _ = cv2.minAreaRect(P.astype(np.float32))
    rect = Polygon(cv2.boxPoints(cv2.minAreaRect(P.astype(np.float32)))).buffer(0)
    ell, major, minor = _ellipse_of(P)

    def iou(q):
        if q is None or shp.is_empty:
            return 0.0
        u = shp.union(q).area
        return shp.intersection(q).area / u if u > 0 else 0.0

    if iou(ell) > iou(rect):
        return "ellipse", major, minor
    return "rect", max(a, b), min(a, b)


def block_bounds(labelled: Iterable[data.SheetId], root: Path = data.LABELS) -> BlockBounds:
    """The labelled blocks' (every `remove` polygon of the `labelled` sheets) largest rectangle sides and ellipse
    axes, widened by WIDEN.  Each polygon is assigned to the family whose shape fits it better (`label_kind`: the
    larger intersection over union), and bounds that family only; the labels are the authority (controller ruling
    C7: a compass rose with its long needle is legitimately a long thin ellipse; on the 21 labels this gives
    rectangles <= 1212 x 994 px and ellipses <= 1419 x 484).  A family with no labelled block gets (0, 0): no such
    shape."""
    rect, ell = [0.0, 0.0], [0.0, 0.0]
    for sid in labelled:
        for poly in data.load_remove(*sid, root=root):
            if len(poly) < 3:
                continue
            kind, a, b = label_kind(poly)
            m = ell if kind == "ellipse" else rect
            m[0], m[1] = max(m[0], a), max(m[1], b)
    return BlockBounds((WIDEN * rect[0], WIDEN * rect[1]), (WIDEN * ell[0], WIDEN * ell[1]))


# ---------------------------------------------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------------------------------------------


def block_box(b: Block, h: int, w: int, pad: float = 2.0) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1): the page px the block can cover, widened by `pad`."""
    t = math.radians(b.theta)
    c, sn = math.cos(t), math.sin(t)
    if b.kind == "rect":
        u0, u1, v0, v1 = b.p
        pts = [(u, v) for u in (u0 - pad, u1 + pad) for v in (v0 - pad, v1 + pad)]
    else:
        a, bb = b.p
        pts = [(u, v) for u in (-a - pad, a + pad) for v in (-bb - pad, bb + pad)]
    xs = [b.cx + c * u - sn * v for u, v in pts]
    ys = [b.cy + sn * u + c * v for u, v in pts]
    x0, x1 = max(0, int(math.floor(min(xs))) - 2), min(w, int(math.ceil(max(xs))) + 2)
    y0, y1 = max(0, int(math.floor(min(ys))) - 2), min(h, int(math.ceil(max(ys))) + 2)
    return y0, max(y0, y1), x0, max(x0, x1)


def block_uv(b: Block, y0: int, y1: int, x0: int, x1: int):
    """(u, v) of the px centres of the page crop [y0, y1) x [x0, x1) about the block."""
    t = math.radians(b.theta)
    c, sn = math.cos(t), math.sin(t)
    y, x = np.mgrid[y0:y1, x0:x1]
    dx, dy = x + 0.5 - b.cx, y + 0.5 - b.cy
    return c * dx + sn * dy, -sn * dx + c * dy


def block_crop(b: Block, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Bool on the page crop: the block's px (pixel-centre rule)."""
    u, v = block_uv(b, y0, y1, x0, x1)
    if b.kind == "rect":
        u0, u1, v0, v1 = b.p
        return (u >= u0) & (u < u1) & (v >= v0) & (v < v1)
    a, bb = b.p
    return (u / a) ** 2 + (v / bb) ** 2 <= 1.0


def blocks_mask(bs: Iterable[Block], shape: tuple[int, int]) -> np.ndarray:
    """Bool on the page: every block's px."""
    h, w = shape
    out = np.zeros((h, w), bool)
    for b in bs:
        y0, y1, x0, x1 = block_box(b, h, w)
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] |= block_crop(b, y0, y1, x0, x1)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------------------------------------------


def _rect_on(Gc, x0, y0, cx, cy, theta, init, lu, lv):
    """The best rectangle (largest sum of the gain Gc, the page crop with its px (0, 0) at page (x0, y0)) at angle
    theta about (cx, cy), its u-extent at most lu and its v-extent at most lv px: coordinate descent over its four
    edges, each an exact bounded 1-D scan of the summed-area table, from `init` = (u0, u1, v0, v1)."""
    hc, wc = Gc.shape
    hw = math.hypot(hc, wc) / 2 + 4
    cv = Canvas(theta, cx, cy, hw + abs(cx - (x0 + wc / 2)), hw + abs(cy - (y0 + hc / 2)))
    A = cv.warp(Gc, 1, 1, x_off=x0, y_off=y0)
    S = np.zeros((A.shape[0] + 1, A.shape[1] + 1))
    S[1:, 1:] = A.cumsum(0, dtype=np.float64).cumsum(1)
    nj, ni = A.shape
    lu, lv = max(1, int(lu)), max(1, int(lv))
    il = int(np.clip(math.floor(init[0] + cv.U), 0, ni - 1))
    ir = int(np.clip(math.ceil(init[1] + cv.U), il + 1, ni))
    jt = int(np.clip(math.floor(init[2] + cv.V), 0, nj - 1))
    jb = int(np.clip(math.ceil(init[3] + cv.V), jt + 1, nj))
    if ir - il > lu:
        m = (il + ir) // 2
        il = max(0, m - lu // 2)
        ir = min(ni, il + lu)
    if jb - jt > lv:
        m = (jt + jb) // 2
        jt = max(0, m - lv // 2)
        jb = min(nj, jt + lv)
    for _ in range(20):
        old = (il, ir, jt, jb)
        f = S[:, ir] - S[:, il]
        lo = max(0, jb - lv)
        jt = lo + int(np.argmin(f[lo:jb]))
        hi = min(nj, jt + lv)
        jb = jt + 1 + int(np.argmax(f[jt + 1 : hi + 1]))
        g = S[jb] - S[jt]
        lo = max(0, ir - lu)
        il = lo + int(np.argmin(g[lo:ir]))
        hi = min(ni, il + lu)
        ir = il + 1 + int(np.argmax(g[il + 1 : hi + 1]))
        if (il, ir, jt, jb) == old:
            break
    gain = S[jb, ir] - S[jt, ir] - S[jb, il] + S[jt, il]
    return float(gain), Block("rect", cx, cy, float(theta), (il - cv.U, ir - cv.U, jt - cv.V, jb - cv.V), float(gain))


def _ellipse_gain(G, X, Y, cx, cy, a, b, th):
    t = math.radians(th)
    c, s = math.cos(t), math.sin(t)
    dx, dy = X - cx, Y - cy
    u, v = c * dx + s * dy, -s * dx + c * dy
    return float(G[(u / a) ** 2 + (v / b) ** 2 <= 1.0].sum(dtype=np.float64))


def _ellipse_on(Gc, x0, y0, cx, cy, a, b, th, amax, bmax):
    """Pattern search over (cx, cy, a, b, theta) for the largest sum of the gain Gc over the ellipse, its longer full
    axis at most amax and its shorter at most bmax; coarse on pooled cells for big crops, then at full resolution."""
    hc, wc = Gc.shape
    f = max(1, int(math.ceil(max(hc, wc) / ELLIPSE_CELLS)))
    a, b = max(a, 2.0), max(b, 2.0)
    if a >= b:
        a, b = min(a, amax / 2), min(b, bmax / 2)
    else:
        a, b = min(a, bmax / 2), min(b, amax / 2)
    p = np.array([cx, cy, max(a, 2.0), max(b, 2.0), th])

    def ok(q):
        lo, hi = sorted((q[2], q[3]))
        return lo >= 2 and 2 * hi <= amax and 2 * lo <= bmax

    def run(arr, f, p, steps, minsteps):
        Y, X = np.mgrid[0 : arr.shape[0], 0 : arr.shape[1]]
        X = x0 + f * X + f / 2.0
        Y = y0 + f * Y + f / 2.0
        cur = _ellipse_gain(arr, X, Y, *p)
        steps = np.array(steps, float)
        while np.any(steps >= minsteps):
            improved = False
            for k in range(5):
                if steps[k] < minsteps[k]:
                    continue
                for sgn in (1, -1):
                    q = p.copy()
                    q[k] += sgn * steps[k]
                    if not ok(q):
                        continue
                    g = _ellipse_gain(arr, X, Y, *q)
                    if g > cur:
                        cur, p, improved = g, q, True
            if not improved:
                steps = steps / 2
        return cur, p

    if f > 1:
        _, p = run(pool(Gc, f), f, p, [8 * f, 8 * f, 8 * f, 8 * f, 8.0], [f, f, f, f, 1.0])
        st = [2 * f] * 4 + [1.0]
    else:
        st = [max(2.0, p[2] / 8)] * 2 + [max(2.0, p[2] / 8), max(2.0, p[3] / 8), 8.0]
    g, p = run(Gc, 1, p, st, [1, 1, 1, 1, 0.25])
    return g, Block("ellipse", float(p[0]), float(p[1]), float(p[4]), (float(p[2]), float(p[3])), g)


def candidates(p: np.ndarray, D: np.ndarray, kept: np.ndarray, frame: Frame, bounds: BlockBounds) -> list[Block]:
    """Per 8-connected piece of p2 > `cut_level()` inside `kept` (at least BLOCK_MIN_PX px): the better of the best
    rectangle and the best ellipse within `bounds`, by gain within `kept`; highest gain first.  The rectangle is square
    to the frame (`frame.phi` + FRAME_ANGLES; with no framed side phi is 0, square to the page); the ellipse starts
    from the piece's own moments."""
    D = np.asarray(D, np.float32)
    h, w = D.shape
    L, S = bounds.rect_max
    amax, bmax = bounds.ellipse_max
    lab, n = ndi.label((np.asarray(p[2]) > cut_level()) & kept, structure=_EIGHT)
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or sizes[k] < BLOCK_MIN_PX:
            continue
        ys, xs = np.nonzero(lab[sl] == k)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        bh, bw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        mg = max(CROP_MARGIN[0], int(CROP_MARGIN[1] * max(bh, bw)))
        y0, y1 = max(0, sl[0].start - mg), min(h, sl[0].stop + mg)
        x0, x1 = max(0, sl[1].start - mg), min(w, sl[1].stop + mg)
        Gc = np.where(kept[y0:y1, x0:x1], -D[y0:y1, x0:x1], np.float32(0))
        cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
        cov = np.cov(np.stack([xs, ys])) if len(xs) > 2 else np.eye(2)
        ev, evec = np.linalg.eigh(cov)
        thm = math.degrees(math.atan2(evec[1, 1], evec[0, 1]))
        best = None
        if L > 0 and S > 0:
            angles = sorted({round(frame.phi + d, 3) for d in FRAME_ANGLES})
            for th in angles:
                t = math.radians(th)
                c, s = math.cos(t), math.sin(t)
                pu = c * (xs + 0.5 - cx) + s * (ys + 0.5 - cy)
                pv = -s * (xs + 0.5 - cx) + c * (ys + 0.5 - cy)
                init = (pu.min(), pu.max() + 1, pv.min(), pv.max() + 1)
                for lu, lv in ((L, S), (S, L)):
                    g, b = _rect_on(Gc, x0, y0, cx, cy, th, init, lu, lv)
                    if best is None or g > best.gain:
                        best = b
        if amax > 0 and bmax > 0:
            a, b = 2 * math.sqrt(max(ev[1], 1.0)), 2 * math.sqrt(max(ev[0], 1.0))
            g, e = _ellipse_on(Gc, x0, y0, cx, cy, a, b, thm, amax, bmax)
            if best is None or g > best.gain:
                best = e
        if best is not None:
            out.append(best)
    out.sort(key=lambda b: -b.gain)
    return out


def accept(bs: list[Block], D: np.ndarray, kept: np.ndarray, lam_block: float) -> tuple[np.ndarray, list[Block]]:
    """The blocks kept, in the given order (highest fitted gain first): each re-scored as the sum of -D over its px
    within `kept` less the blocks already kept, and kept if that exceeds `lam_block`.  (removed mask, kept blocks:
    the same objects as in `bs`.)"""
    h, w = D.shape
    removed = np.zeros((h, w), bool)
    out = []
    for b in bs:
        if b.gain <= lam_block:
            break
        y0, y1, x0, x1 = block_box(b, h, w)
        if y1 <= y0 or x1 <= x0:
            continue
        m = block_crop(b, y0, y1, x0, x1) & kept[y0:y1, x0:x1] & ~removed[y0:y1, x0:x1]
        if -float(D[y0:y1, x0:x1][m].sum(dtype=np.float64)) > lam_block:
            removed[y0:y1, x0:x1] |= m
            out.append(b)
    return removed, out


def block_shapes(p: np.ndarray, D: np.ndarray, kept: np.ndarray, frame: Frame, bounds: BlockBounds,
                 params: Params) -> list[Block]:
    """The removed blocks: `candidates`, kept by `accept` at `params.lam_block`."""
    return accept(candidates(p, D, kept, frame, bounds), np.asarray(D, np.float32), kept, params.lam_block)[1]
