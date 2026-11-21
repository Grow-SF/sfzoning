"""Issue #71 frame crop, amendment 2: the straight-line shape decision, measured on the fold-0 "fold 16" net's held-out
chances against the per-pixel cut (`decide.decide`).  Throwaway measurement (brief:
.superpowers/sdd/2026-11-20-frame-crop/spike-shape-brief.md).

With D(x) = COST_FRAME (p1 + p2) ink - COST_MAP p0 ink (ink = 1 - lum), keeping a set K costs const + sum_{x in K} D(x),
so the best shape minimises sum_K D.  D > 0 is "cheaper deleted" (frame or block ink), D < 0 "cheaper kept" (map ink),
paper costs ~0.  The shape, searched in this order:

1. the frame: one rectangle rotated by phi, searched globally (phi over +-2.5 deg at 0.1 deg; four offsets on a 16 px
   grid from the canvas edge to 40% of the page inward).  For fixed (top, bottom) the best (left, right) separates
   exactly: the left offsets all lie left of every right offset, so min_{l<r} R(r) - R(l) = min_r R(r) - max_l R(l)
   with R the band's cumulative sum along u.  Then phi and the offsets are refined at 1 px.
2. per side, a tilt deviation (+-0.45 deg) and a quadratic bow (+-22 px sag), the offset re-optimised, from running
   sums along the side's normal;
3. per corner, an arc of the form the labels show (checked by `labels`), radius by a 1-D search;
4. notches, greedily: rotated rectangles anchored at a side (angle within +-NOTCH_DEG of phi), the best per (side,
   angle, depth) by a Kadane pass over column sums, while the best gain exceeds lambda_notch (cap 6);
5. removed blocks: per 8-connected piece of p2 > level inside K, the best rotated rectangle and the best ellipse,
   kept if its gain exceeds lambda_block.

    uv run python scripts/spikes/issue71_shape.py labels      # the corner-arc form and the notch angles on the labels
    uv run python scripts/spikes/issue71_shape.py synth       # self-check on synthetic pages
    uv run python scripts/spikes/issue71_shape.py run         # the fold-0 measurement, the fit, the pictures
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, optimize

from sfzmaps.frame import data, decide

log = logging.getLogger("shape")
Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(4)

OUT = data.DATA_DIR / "frame-shape"
_EIGHT = ndimage.generate_binary_structure(2, 2)


# ----------------------------------------------------------------------------------------------------------------
# The labels: corner-arc form and notch angles
# ----------------------------------------------------------------------------------------------------------------

def _fit_line(x, y, tol=4.0, tries=400, seed=0):
    """y = a x + b: the line through two of the points with the most points within `tol` px (RANSAC), refitted by
    least squares on those, so a notch or a stub in the middle can't drag it."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(tries):
        i, j = rng.choice(len(x), 2, replace=False)
        if abs(x[j] - x[i]) < 50:
            continue
        a = (y[j] - y[i]) / (x[j] - x[i])
        b = y[i] - a * x[i]
        n = int((np.abs(y - (a * x + b)) <= tol).sum())
        if n > best_n:
            best, best_n = (a, b), n
    a, b = best
    keep = np.abs(y - (a * x + b)) <= tol
    a, b = np.polyfit(x[keep], y[keep], 1)
    return a, b


def _densify(P, step=5.0):
    pts = []
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]
        n = max(1, int(np.hypot(*(q - p)) // step))
        t = np.arange(n)[:, None] / n
        pts.append(p + t * (q - p))
    return np.concatenate(pts)


def label_sides(P, shape_hw):
    """The four side lines of a label polygon, fitted to the middle 70% of each side: dict side -> (a, b) with
    top/bottom y = a x + b and left/right x = a y + b."""
    D = _densify(P)
    x0, y0 = P.min(0)
    x1, y1 = P.max(0)
    W, H = x1 - x0, y1 - y0
    sides = {}
    mx = (D[:, 0] > x0 + 0.15 * W) & (D[:, 0] < x1 - 0.15 * W)
    my = (D[:, 1] > y0 + 0.15 * H) & (D[:, 1] < y1 - 0.15 * H)
    for name, sel, u, v in (("top", mx & (D[:, 1] < y0 + 0.1 * H), 0, 1),
                            ("bottom", mx & (D[:, 1] > y1 - 0.1 * H), 0, 1),
                            ("left", my & (D[:, 0] < x0 + 0.1 * W), 1, 0),
                            ("right", my & (D[:, 0] > x1 - 0.1 * W), 1, 0)):
        pts = D[sel]
        # the outermost dominant line: a notch is a minority of the middle
        sides[name] = _fit_line(pts[:, u], pts[:, v])
    return sides


def _line_point(side, a, b, t):
    return np.array([t, a * t + b]) if side in ("top", "bottom") else np.array([a * t + b, t])


def _intersect(top_like, left_like):
    (a1, b1), (a2, b2) = top_like, left_like      # y = a1 x + b1 ; x = a2 y + b2
    y = (a1 * b2 + b1) / (1 - a1 * a2)
    return np.array([a2 * y + b2, y])


def _inward(side, ab, pts):
    """Signed distance of pts inward from the side line (positive inside the frame)."""
    a, b = ab
    if side == "top":
        return (pts[:, 1] - (a * pts[:, 0] + b)) / math.hypot(1, a)
    if side == "bottom":
        return ((a * pts[:, 0] + b) - pts[:, 1]) / math.hypot(1, a)
    if side == "left":
        return (pts[:, 0] - (a * pts[:, 1] + b)) / math.hypot(1, a)
    return ((a * pts[:, 1] + b) - pts[:, 0]) / math.hypot(1, a)


def fit_circle(pts, trim=5.0):
    """Least-squares circle (Kasa start, geometric refinement), refitted once without points more than `trim` px
    off it: (cx, cy, r, rms, n used)."""
    cx, cy, r, rms = _fit_circle(pts)
    keep = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) <= trim
    if 5 <= keep.sum() < len(pts):
        cx, cy, r, rms = _fit_circle(pts[keep])
    return cx, cy, r, rms, int(keep.sum())


def _fit_circle(pts):
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    c, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = c[0], c[1]
    r = math.sqrt(c[2] + cx * cx + cy * cy)
    res = optimize.least_squares(lambda q: np.hypot(x - q[0], y - q[1]) - q[2], [cx, cy, r])
    cx, cy, r = res.x
    return cx, cy, r, float(np.sqrt(np.mean(res.fun ** 2)))


CORNERS = (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"))


def corner_arcs(P, sides, reach=400.0, off=4.0):
    """Per corner: the polygon's vertices within `reach` of the corner and more than `off` px inside both side lines
    (the arc), their circle, and how it sits against the two sides."""
    out = []
    for s1, s2 in CORNERS:
        c = _intersect(sides[s1], sides[s2])
        d = np.hypot(*(P - c).T)
        i1, i2 = _inward(s1, sides[s1], P), _inward(s2, sides[s2], P)
        arc = P[(d < reach) & (i1 > off) & (i2 > off)]
        row = {"corner": f"{s1}-{s2}", "points": int(len(arc))}
        if len(arc) >= 5:
            cx, cy, r, rms, used = fit_circle(arc)
            cen = np.array([[cx, cy]])
            d1 = float(_inward(s1, sides[s1], cen)[0])
            d2 = float(_inward(s2, sides[s2], cen)[0])
            row.update(r=r, rms=rms, used=used, d1=d1, d2=d2, centre_inside=bool(d1 > 0 and d2 > 0),
                       centre_to_corner=float(np.hypot(cx - c[0], cy - c[1])),
                       span=float(np.hypot(*(arc.max(0) - arc.min(0)))))
        out.append(row)
    return out


def _wrap90(deg):
    return (deg + 45.0) % 90.0 - 45.0


def page_edge_sides(sides, shape_hw, tol=3.0):
    """The sides lying on the page's edge (no frame there): within `tol` px of it at both ends."""
    h, w = shape_hw
    out = set()
    for s, (a, b) in sides.items():
        n = w if s in ("top", "bottom") else h
        ends = np.array([b, a * (n - 1) + b])
        edge = 0.0 if s in ("top", "left") else (h - 1 if s == "bottom" else w - 1)
        if s in ("bottom", "right"):
            edge = (h if s == "bottom" else w)
        if np.all(np.abs(ends - edge) <= tol) or (s in ("bottom", "right") and np.all(np.abs(ends - (edge - 1)) <= tol)):
            out.add(s)
    return out


def frame_angle(sides, skip=()):
    """The frame's rotation (deg) from its side lines, the page-edge sides `skip` left out: top/bottom slope atan(a),
    left/right -atan(a)."""
    angs = {s: (math.degrees(math.atan(sides[s][0])) if s in ("top", "bottom") else
                -math.degrees(math.atan(sides[s][0]))) for s in sides}
    use = [v for s, v in angs.items() if s not in skip]
    return (float(np.mean(use)) if use else 0.0), angs


def notch_edges(P, sides, phi, min_len=150.0, off=15.0, tol=2.0):
    """Edges of the polygon (simplified at `tol` px) at least `min_len` long whose two ends both lie more than `off`
    px inside every side line: the notches' inner sides."""
    from shapely.geometry import LinearRing
    ring = LinearRing(P).simplify(tol)
    Q = np.asarray(ring.coords)[:-1]
    ins = np.min(np.stack([_inward(s, sides[s], Q) for s in sides]), axis=0)
    edges = []
    for i in range(len(Q)):
        p, q = Q[i], Q[(i + 1) % len(Q)]
        L = float(np.hypot(*(q - p)))
        if L < min_len:
            continue
        # an inner edge: at least one end well inside every side (a notch edge runs from a side inward, or along the
        # notch's bottom)
        if max(ins[i], ins[(i + 1) % len(Q)]) <= off:
            continue
        th = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
        edges.append({"from": [round(float(v)) for v in p], "to": [round(float(v)) for v in q], "length": L,
                      "dev_deg": _wrap90(th - phi), "precision_deg": math.degrees(2 * 1.5 / L)})
    return edges


def labels_check() -> dict:
    """The corner arcs of every labelled sheet (the 1921 ones decide the form) and the notch angles."""
    res = {"corners": {}, "notches": {}, "frame_angles": {}}
    for sid in data.labelled_sheets():
        polys = data.load_label(*sid)
        with Image.open(data.page_path(*sid)) as im:
            w, h = im.size
        for k, poly in enumerate(polys):
            P = np.asarray(poly, float)
            if len(P) <= 4:
                continue        # the whole page, or a plain quadrilateral: no arcs, no notches
            sides = label_sides(P, (h, w))
            key = f"{sid[0]}/{sid[1]}" + (f"#{k}" if len(polys) > 1 else "")
            edge = page_edge_sides(sides, (h, w))
            phi, angs = frame_angle(sides, edge)
            res["frame_angles"][key] = {"phi": phi, "sides": angs, "page_edge": sorted(edge)}
            res["corners"][key] = corner_arcs(P, sides)
            res["notches"][key] = notch_edges(P, sides, phi)
    return res


def cmd_labels():
    res = labels_check()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "labels_check.json").write_text(json.dumps(res, indent=1))
    for key, rows in res["corners"].items():
        for r in rows:
            if "r" in r:
                print(f"{key:40s} {r['corner']:13s} n={r['points']:3d}/{r['used']:3d} r={r['r']:7.1f} rms={r['rms']:5.2f} "
                      f"d1-r={r['d1'] - r['r']:+7.1f} d2-r={r['d2'] - r['r']:+7.1f} inside={r['centre_inside']} "
                      f"centre-corner={r['centre_to_corner']:7.1f}")
            else:
                print(f"{key:40s} {r['corner']:13s} n={r['points']:3d} (no arc)")
    for key, edges in res["notches"].items():
        for e in edges:
            print(f"notch edge {key:40s} {e['from']} -> {e['to']} L={e['length']:6.0f} dev={e['dev_deg']:+.3f} "
                  f"(+-{e['precision_deg']:.3f})")
    for key, f in res["frame_angles"].items():
        print(f"frame {key:40s} phi={f['phi']:+.3f} sides={ {k: round(v, 3) for k, v in f['sides'].items()} } "
              f"page edge={f['page_edge']}")




# ----------------------------------------------------------------------------------------------------------------
# The numbers of the shape (from the labels, see the brief and `labels`)
# ----------------------------------------------------------------------------------------------------------------

PHI_RANGE = 2.5        # deg: the labels' frames tilt up to ~1.6 deg (95%)
PHI_STEP = 0.1         # deg, the global grid
GRID = 16              # px, the global grid of the four offsets
REACH = 0.4            # offsets run from the canvas edge to this share of the page inward
TILT_MAX = 0.45        # deg, per-side tilt deviation (labels' 80% parallelism 0.28 deg x 1.5, rounded)
SAG_MAX = 22           # px, per-side bow (labels' 95% bow 15 px x 1.5)
ARC_MAX = 332          # px: the largest labelled arc radius (221 px, 1935 p08; 1921: 149) x 1.5 (`labels`)
NOTCH_DEG = 2.46       # deg: the labels' notch edges lie up to 1.64 deg off the frame (1980-1990 p01) x 1.5
NOTCH_CAP = 6
NOTCH_REACH = 0.5      # a notch reaches at most to the page's middle
STRIP = 4              # px: the strip just inside a notch's inner side whose ink is "ink along the side"
BLOCK_MIN_PX = 50      # px: smaller pieces of p2 > level are not block candidates (noise specks)
SIDES = ("top", "bottom", "left", "right")


def pool(a: np.ndarray, s: int) -> np.ndarray:
    """Sum over s x s cells (the page padded with zeros to a multiple of s)."""
    if s == 1:
        return np.ascontiguousarray(a, np.float32)
    h, w = a.shape
    H, W = -(-h // s) * s, -(-w // s) * s
    b = np.zeros((H, W), np.float32)
    b[:h, :w] = a
    return b.reshape(H // s, s, W // s, s).sum((1, 3), dtype=np.float32)


class Canvas:
    """Page coordinates -> the frame of angle `phi` (deg) about (cx, cy): u = c dx + s dy, v = -s dx + c dy with
    (dx, dy) from (cx, cy) to a pixel's centre.  Full-res canvas px (i, j) covers u in [i - U, i - U + 1) and v in
    [j - V, j - V + 1); U and V are multiples of 8, so cells of s | 8 px line up and flip exactly."""

    def __init__(self, phi, cx, cy, hu, hv):
        self.phi, self.cx, self.cy = float(phi), float(cx), float(cy)
        t = math.radians(phi)
        self.c, self.sn = math.cos(t), math.sin(t)
        self.U = int(math.ceil((hu + 2) / 8.0) * 8)
        self.V = int(math.ceil((hv + 2) / 8.0) * 8)

    @classmethod
    def page(cls, phi, h, w):
        t = math.radians(phi)
        c, s = abs(math.cos(t)), abs(math.sin(t))
        return cls(phi, w / 2.0, h / 2.0, c * w / 2 + s * h / 2, s * w / 2 + c * h / 2)

    def uv(self, x, y):
        """(u, v) of the centres of page px (x, y) (arrays, page px indices)."""
        dx = np.asarray(x, np.float64) + 0.5 - self.cx
        dy = np.asarray(y, np.float64) + 0.5 - self.cy
        return self.c * dx + self.sn * dy, -self.sn * dx + self.c * dy

    def xy(self, u, v):
        """Page coordinates (continuous, not px indices) of (u, v)."""
        return self.cx + self.c * u - self.sn * v, self.cy + self.sn * u + self.c * v

    def cells(self, s):
        return 2 * self.V // s, 2 * self.U // s

    def warp(self, src, s_src, s, window=None, x_off=0.0, y_off=0.0):
        """`src` (page px pooled by s_src, its px (0, 0) at page (x_off, y_off)) drawn on this canvas at cells of s
        px, bilinear; window (j0, j1, i0, i1) in cells, default the whole canvas."""
        nj, ni = self.cells(s)
        j0, j1, i0, i1 = window or (0, nj, 0, ni)
        c, sn = self.c, self.sn
        ox, oy = x_off + s_src / 2.0 - self.cx, y_off + s_src / 2.0 - self.cy
        M = np.array([[c * s_src / s, sn * s_src / s, (c * ox + sn * oy + self.U - s / 2.0) / s - i0],
                      [-sn * s_src / s, c * s_src / s, (-sn * ox + c * oy + self.V - s / 2.0) / s - j0]])
        return cv2.warpAffine(np.ascontiguousarray(src, np.float32), M, (int(i1 - i0), int(j1 - j0)),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    def bin(self, src, window=None):
        """Forward binning at full res: every page px of `src` (page-sized) added to the canvas px its centre falls
        in, over window (j0, j1, i0, i1) (default the whole canvas).  Exact sums, and the same centre rule as the
        masks, so a boundary on the canvas is the same boundary on the page (bilinear warping blurs rule ink into
        the first map row, which the 3:1 cost then cuts)."""
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
            out += np.bincount((j[ok] * (i1 - i0) + i[ok]), weights=src[ya:yb, x0:x1][ok].astype(np.float64),
                               minlength=out.size).reshape(out.shape)
        return out.astype(np.float32)

    def EA(self, side):
        """(E, A): side-view row eps covers e in [eps - E, eps - E + 1), column alpha covers along in [alpha - A, ...)."""
        return (self.V, self.U) if side in ("top", "bottom") else (self.U, self.V)

    def side_window(self, side, src, s_src, s, e0, e1, a0, a1, binned=False):
        """The canvas at cells of s seen from `side`: rows e0..e1 inward from the canvas edge, columns a0..a1 along
        (`binned`: forward binning at full res, s_src = s = 1)."""
        nj, ni = self.cells(s)
        get = (lambda win: self.bin(src, win)) if binned else (lambda win: self.warp(src, s_src, s, win))
        if side == "top":
            out = get((e0, e1, a0, a1))
        elif side == "bottom":
            out = get((nj - e1, nj - e0, a0, a1))[::-1]
        elif side == "left":
            out = get((a0, a1, e0, e1)).T
        else:
            out = get((a0, a1, ni - e1, ni - e0))[:, ::-1].T
        return np.ascontiguousarray(out)


def side_view(A, side):
    """A whole canvas array seen from `side` (rows inward from that edge, columns along it)."""
    return {"top": A, "bottom": A[::-1], "left": A.T, "right": A[:, ::-1].T}[side]


def e_along(side, u, v):
    return {"top": (v, u), "bottom": (-v, u), "left": (u, v), "right": (-u, v)}[side]


CORNER_SIDES = (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"))


@dataclass
class Notch:
    side: str
    psi: float              # deg
    a0: float               # along in [a0, a1), e < d, in the psi frame about the page centre
    a1: float
    d: float
    gain: float = 0.0       # sum D over it (within K) + beta x side ink, before lambda
    ink: float = 0.0        # the ink along its inner sides (per px of side)


@dataclass
class Block:
    kind: str               # "rect" | "ellipse"
    cx: float
    cy: float
    theta: float            # deg
    p: tuple                # rect: (u0, u1, v0, v1); ellipse: (a, b)
    gain: float = 0.0


@dataclass
class Frame:
    h: int
    w: int
    phi: float
    beta: dict              # side -> e of the boundary at the side's middle (kept iff e >= bound)
    tilt: dict = field(default_factory=lambda: {s: 0.0 for s in SIDES})   # tan of the deviation
    sag: dict = field(default_factory=lambda: {s: 0.0 for s in SIDES})    # px, + = bowed inward
    arc: dict = field(default_factory=lambda: {c: 0.0 for c in CORNER_SIDES})
    notches: list = field(default_factory=list)

    @property
    def canvas(self):
        return Canvas.page(self.phi, self.h, self.w)

    def copy(self):
        return Frame(self.h, self.w, self.phi, dict(self.beta), dict(self.tilt), dict(self.sag), dict(self.arc),
                     list(self.notches))

    def ends(self, side):
        b = self.beta
        return (b["left"], -b["right"]) if side in ("top", "bottom") else (b["top"], -b["bottom"])

    def bound(self, side, along):
        lo, hi = self.ends(side)
        m, hw = (lo + hi) / 2.0, max((hi - lo) / 2.0, 1.0)
        t = (along - m) / hw
        return self.beta[side] + self.tilt[side] * (along - m) + self.sag[side] * (1.0 - t * t)

    def kept_rows(self, y0, y1, x0=0, x1=None, arcs=True):
        """K (frame, arcs, no notches) on page rows y0..y1 (cols x0..x1)."""
        x1 = self.w if x1 is None else x1
        cv = self.canvas
        y, x = np.mgrid[y0:y1, x0:x1]
        u, v = cv.uv(x, y)
        keep = np.ones(u.shape, bool)
        ea = {}
        for side in SIDES:
            e, al = e_along(side, u, v)
            dist = e - self.bound(side, al)
            ea[side] = dist
            keep &= dist >= 0
        if arcs:
            for c in CORNER_SIDES:
                r = self.arc[c]
                if r <= 0:
                    continue
                a, b = ea[c[0]], ea[c[1]]
                near = (a >= 0) & (b >= 0) & (a < r) & (b < r)
                if near.any():
                    an, bn = a[near], b[near]
                    cut = an + bn + np.sqrt(2.0 * an * bn) < r
                    idx = np.flatnonzero(near.ravel())[cut]
                    keep.ravel()[idx] = False
        return keep

    def kept(self, chunk=1024):
        K = np.zeros((self.h, self.w), bool)
        for y0 in range(0, self.h, chunk):
            K[y0:y0 + chunk] = self.kept_rows(y0, min(self.h, y0 + chunk))
        for n in self.notches:
            K &= ~notch_mask(n, self.h, self.w)
        return K


def notch_mask(n: Notch, h, w):
    """The page px of the notch (centre rule)."""
    cv = Canvas.page(n.psi, h, w)
    E, A = cv.EA(n.side)
    # its corners in the psi frame: along a0..a1, e from the canvas edge (-E) to d
    corners = []
    for al in (n.a0, n.a1):
        for e in (-E - 2.0, n.d):
            u, v = {"top": (al, e), "bottom": (al, -e), "left": (e, al), "right": (-e, al)}[n.side]
            corners.append(cv.xy(u, v))
    xs, ys = zip(*corners)
    x0, x1 = max(0, int(math.floor(min(xs))) - 2), min(w, int(math.ceil(max(xs))) + 2)
    y0, y1 = max(0, int(math.floor(min(ys))) - 2), min(h, int(math.ceil(max(ys))) + 2)
    out = np.zeros((h, w), bool)
    if x1 <= x0 or y1 <= y0:
        return out
    y, x = np.mgrid[y0:y1, x0:x1]
    u, v = cv.uv(x, y)
    e, al = e_along(n.side, u, v)
    out[y0:y1, x0:x1] = (al >= n.a0) & (al < n.a1) & (e < n.d)
    return out


def block_mask_box(b: Block, h, w):
    """(y0, y1, x0, x1, mask) of the block's px (centre rule) within its bounding box."""
    t = math.radians(b.theta)
    c, sn = math.cos(t), math.sin(t)
    if b.kind == "rect":
        u0, u1, v0, v1 = b.p
        pts = [(u, v) for u in (u0, u1) for v in (v0, v1)]
    else:
        a, bb = b.p
        pts = [(u, v) for u in (-a, a) for v in (-bb, bb)]
    xs = [b.cx + c * u - sn * v for u, v in pts]
    ys = [b.cy + sn * u + c * v for u, v in pts]
    x0, x1 = max(0, int(math.floor(min(xs))) - 2), min(w, int(math.ceil(max(xs))) + 2)
    y0, y1 = max(0, int(math.floor(min(ys))) - 2), min(h, int(math.ceil(max(ys))) + 2)
    if x1 <= x0 or y1 <= y0:
        return y0, y0, x0, x0, np.zeros((0, 0), bool)
    y, x = np.mgrid[y0:y1, x0:x1]
    dx, dy = x + 0.5 - b.cx, y + 0.5 - b.cy
    u, v = c * dx + sn * dy, -sn * dx + c * dy
    if b.kind == "rect":
        m = (u >= u0) & (u < u1) & (v >= v0) & (v < v1)
    else:
        m = (u / a) ** 2 + (v / bb) ** 2 <= 1.0
    return y0, y1, x0, x1, m


# ----------------------------------------------------------------------------------------------------------------
# 1. The frame rectangle, searched globally
# ----------------------------------------------------------------------------------------------------------------

def _sat_rows(A, rows):
    """S[r, :] = sum of A[:r, :i] for each r of `rows` (sorted), as (len(rows), ni + 1) float64."""
    out = np.zeros((len(rows), A.shape[1] + 1))
    cum = np.zeros(A.shape[1])
    prev = 0
    for k, r in enumerate(rows):
        if r > prev:
            cum += A[prev:r].sum(0, dtype=np.float64)
            prev = r
        out[k, 1:] = np.cumsum(cum)
    return out


def _rect_best(A, s, KT, KB, KL, KR, full_sat=None, credit=None):
    """The best rectangle on the canvas array A (cells of s): kept rows [kt, nj - kb), cols [kl, ni - kr) over the
    candidate cell offsets, minimising sum A over it less `credit[side][k]` (a per-side, per-offset credit: the best
    single notch from that offset).  For fixed (kt, kb) the best (kl, kr) separates exactly, since every left offset
    lies left of every right one: min_{l, r} R(r) - c_r(r) - R(l) - c_l(l) = min_r (R(r) - c_r(r)) - max_l (R(l) +
    c_l(l)), R the band's cumulative sum."""
    nj, ni = A.shape
    JT, JB = np.asarray(KT), nj - np.asarray(KB)
    IL, IR = np.asarray(KL), ni - np.asarray(KR)
    if full_sat is not None:
        St, Sb = full_sat[JT], full_sat[JB]
    else:
        rows = np.unique(np.concatenate([JT, JB]))
        Sr = _sat_rows(A, rows)
        pos = {r: k for k, r in enumerate(rows)}
        St, Sb = Sr[[pos[r] for r in JT]], Sr[[pos[r] for r in JB]]
    RR = Sb[:, None, IR] - St[None, :, IR]          # (nB, nT, nR)
    LL = Sb[:, None, IL] - St[None, :, IL]
    if credit is not None:
        RR = RR - credit["right"][None, None, :]
        LL = LL + credit["left"][None, None, :]
    rmin, rarg = RR.min(-1), RR.argmin(-1)
    lmax, larg = LL.max(-1), LL.argmax(-1)
    val = rmin - lmax
    if credit is not None:
        val = val - credit["top"][None, :] - credit["bottom"][:, None]
    b, t = np.unravel_index(np.argmin(val), val.shape)
    return float(val[b, t]), int(KT[t]), int(KB[b]), int(KL[larg[b, t]]), int(KR[rarg[b, t]])


def _offset_range(V, half_page, s, step_cells):
    """Cell offsets from the canvas edge to REACH of the page inward, every step_cells."""
    kmax = int((V - half_page + REACH * 2 * half_page) // s)
    return np.arange(0, kmax + 1, step_cells)


CREDIT_SPAN = 0.5      # a credited notch covers at most this share of its side (labelled notches: 23-29%, x 1.5)


def _notch_credit(Av, T, d_step, d_max, a_lo, a_hi):
    """g[k]: the gain of the best single notch at the frame's angle from offset T[k] (rows [T[k], d) for d up to
    T[k] + d_max, every d_step, x a span within columns [a_lo, a_hi) no longer than CREDIT_SPAN of it) of a side view
    Av: a length-bounded maximum-sum subarray per depth (sliding-window minimum of the prefix sums).  The bound keeps
    a credited notch from standing in for the side itself (frame rule included), which double-counts at the corners
    (seen on 1921/355582)."""
    from scipy.ndimage import minimum_filter1d
    n = Av.shape[0]
    L = a_hi - a_lo
    if L <= 2:
        return np.zeros(len(T))
    G = np.zeros((n + 1, L))
    G[1:] = np.cumsum(Av[:, a_lo:a_hi], 0, dtype=np.float64)
    Lmax = max(1, int(CREDIT_SPAN * L))
    Lmax -= 1 - Lmax % 2                     # odd, for the filter's origin
    out = np.zeros(len(T))
    for k, t in enumerate(T):
        ds = np.arange(t + d_step, min(n, t + d_max) + 1, d_step)
        if len(ds) == 0:
            continue
        Pz = np.zeros((len(ds), L + 1))
        Pz[:, 1:] = np.cumsum(G[ds] - G[t], 1)
        # Mn[:, b] = min(Pz[:, b - Lmax : b]) (b >= 1): the best start for a span ending at b
        f = minimum_filter1d(Pz, size=Lmax, axis=1, mode="nearest", origin=Lmax // 2)
        best = Pz[:, 1:] - f[:, :-1]
        out[k] = max(0.0, float(best.max()))
    return out


def _centre_first(phis, centre=None):
    """The angles nearest `centre` (default the middle of `phis`) first: with a strict comparison, ties go to it (on
    1-bit pages blank paper has D = 0 exactly, so a frameless side ties over every angle)."""
    phis = np.asarray(phis, float)
    c = phis[len(phis) // 2] if centre is None else centre
    return phis[np.argsort(np.abs(phis - c), kind="stable")]


def _refine_rect(D, pools, phis, s, half_px, beta0):
    """The best rectangle by sum D over angles `phis` and offsets within +-half_px of beta0 (cells of s; s = 1 by
    forward binning): (value, phi, beta)."""
    h, w = D.shape
    out = None
    for phi in _centre_first(phis):
        cv = Canvas.page(phi, h, w)
        A = cv.bin(D) if s == 1 else cv.warp(pools[s], s, s)
        offs = np.arange(-half_px // s, half_px // s + 1)
        k = {sd: int(round((beta0[sd] + cv.EA(sd)[0]) / s)) for sd in SIDES}
        nj, ni = A.shape
        KT = np.clip(k["top"] + offs, 0, nj // 2 - 1)
        KB = np.clip(k["bottom"] + offs, 0, nj // 2 - 1)
        KL = np.clip(k["left"] + offs, 0, ni // 2 - 1)
        KR = np.clip(k["right"] + offs, 0, ni // 2 - 1)
        r = _rect_best(A, s, KT, KB, KL, KR)
        if out is None or r[0] < out[0] - 1e-7 * abs(out[0]) - 1e-6:
            out = (r[0], float(phi), {"top": s * r[1] - cv.V, "bottom": s * r[2] - cv.V,
                                      "left": s * r[3] - cv.U, "right": s * r[4] - cv.U})
    return out


def rect_given(rect, Dz):
    """Coordinate descent: the rectangle re-fitted by sum D with the notches' and blocks' px zeroed (Dz), angle
    +-0.5 deg at 0.05 and offsets +-32 px (cells of 2), then +-0.02 at 0.01 and +-4 px at 1 px.  (The notch-aware
    start's credit is scored at the frame's angle, so a notch turned off the frame pulls the angle: synthetic page
    "A" came out 0.28 deg off, every side's tilt making up for it.)"""
    pools = {2: pool(Dz, 2)}
    b2 = _refine_rect(Dz, pools, rect.phi + np.arange(-0.5, 0.5 + 1e-9, 0.05), 2, 32, rect.beta)
    b1 = _refine_rect(Dz, pools, b2[1] + np.arange(-0.02, 0.02 + 1e-9, 0.01), 1, 4, b2[2])
    return Frame(rect.h, rect.w, b1[1], {k: float(v) for k, v in b1[2].items()})


def frame_search(D, lams=(0.0,), timing=None, notch_aware=True):
    """The frame rectangle for each lambda_notch of `lams`, and the plain one under "plain": {key: (Frame, value)}
(equal rectangles share one Frame).

    1. Global: phi over +-PHI_RANGE at PHI_STEP and the four offsets on a GRID px grid (cells of 4 px), by sum D.
    2. Notch-aware, over the same range every 0.2 deg (cells of 16 px): each rectangle is scored with, per side, the
       best single notch from its offset at the frame's angle (its span within the rectangle's own side, 16 px in from
       the corners, at most CREDIT_SPAN of it), credited max(0, gain - lam).  Spans that depend on the other pair of
       sides break separability, so per angle it is coordinate descent from the plain best rectangle: (top, bottom)
       given (left, right) is exactly separable with the top/bottom credits over the kept columns, and vice versa;
       tb, lr, tb, then the total.  (A span fixed by the plain rectangle double-counted a corner notch on 1980-1990
       p01: the top side cut the inset's upper part while the right side's credit still counted all of it.)  (First run searched only +-0.5 deg around
       the plain angle: on 1980-1990 p01 the plain search tilts the whole frame to -2.6 deg to clip the index inset,
       and the label's region had a lower sum D than either start's result.)  Without it a large notch (an index inset, a
       margin legend) plus the map between pulls a whole side inward: the synthetic page "A" lost a 400 px band of
       map that way.  Still exactly separable.
    3. Refined by sum D: phi +-0.1 at 0.02 with offsets +-24 px at 2 px (cells of 2), then phi +-0.02 at 0.01 with
       offsets +-4 px at 1 px (forward binning)."""
    h, w = D.shape
    t0 = time.perf_counter()
    pools = {s: pool(D, s) for s in (16, 4, 2)}
    pools[1] = np.ascontiguousarray(D, np.float32)
    best = None
    for phi in _centre_first(np.round(np.arange(-PHI_RANGE, PHI_RANGE + 1e-9, PHI_STEP), 4), 0.0):
        cv = Canvas.page(phi, h, w)
        A = cv.warp(pools[4], 4, 4)
        S = np.zeros((A.shape[0] + 1, A.shape[1] + 1))
        S[1:, 1:] = A.cumsum(0, dtype=np.float64).cumsum(1)
        g = GRID // 4
        KT = _offset_range(cv.V, h / 2, 4, g)
        KL = _offset_range(cv.U, w / 2, 4, g)
        r = _rect_best(A, 4, KT, KT, KL, KL, full_sat=S)
        if best is None or r[0] < best[0] - 1e-7 * abs(best[0]) - 1e-6:
            best = (r[0], phi, {"top": 4 * r[1] - cv.V, "bottom": 4 * r[2] - cv.V,
                                "left": 4 * r[3] - cv.U, "right": 4 * r[4] - cv.U})
    t1 = time.perf_counter()

    # 2. notch-aware, cells of 16 px, every 0.2 deg (the later refinements set the angle)
    s = 16
    found = {lam: None for lam in lams}
    for phi in _centre_first(np.round(np.arange(-PHI_RANGE, PHI_RANGE + 1e-9, 0.2), 4), 0.0):
        cv = Canvas.page(phi, h, w)
        A = cv.warp(pools[s], s, s)
        nj, ni = A.shape
        S = np.zeros((nj + 1, ni + 1))
        S[1:, 1:] = A.cumsum(0, dtype=np.float64).cumsum(1)
        KT = _offset_range(cv.V, h / 2, s, 1)
        KL = _offset_range(cv.U, w / 2, s, 1)
        _, kt0, kb0, kl0, kr0 = _rect_best(A, s, KT, KT, KL, KL, full_sat=S)
        cache = {}

        def gains(side, lo, hi):
            if (side, lo, hi) not in cache:
                K = KT if side in ("top", "bottom") else KL
                half = (h if side in ("top", "bottom") else w) / 2.0
                cache[(side, lo, hi)] = _notch_credit(side_view(A, side), K, 2,
                                                      int(NOTCH_REACH * 2 * half / s) + 1, lo, hi)
            return cache[(side, lo, hi)]

        for lam in found:
            kt, kb, kl, kr = kt0, kb0, kl0, kr0

            def cred(side, lo, hi):
                return np.maximum(0.0, gains(side, lo, hi) - lam - 1e-6) * notch_aware

            for rnd in range(3):
                if rnd % 2 == 0:     # (top, bottom) given (left, right): credits over exactly the kept columns
                    lo, hi = kl + 1, ni - kr - 1
                    f = S[:, ni - kr] - S[:, kl]
                    vb = f[nj - KT] - cred("bottom", lo, hi)
                    vt = f[KT] + cred("top", lo, hi)
                    kb, kt = int(KT[np.argmin(vb)]), int(KT[np.argmax(vt)])
                else:                # (left, right) given (top, bottom)
                    lo, hi = kt + 1, nj - kb - 1
                    g = S[nj - kb] - S[kt]
                    vr = g[ni - KL] - cred("right", lo, hi)
                    vl = g[KL] + cred("left", lo, hi)
                    kr, kl = int(KL[np.argmin(vr)]), int(KL[np.argmax(vl)])
            ik = {k: i for i, k in enumerate(KT)}
            il = {k: i for i, k in enumerate(KL)}
            rect = S[nj - kb, ni - kr] - S[kt, ni - kr] - S[nj - kb, kl] + S[kt, kl]
            val = (rect - cred("top", kl + 1, ni - kr - 1)[ik[kt]] - cred("bottom", kl + 1, ni - kr - 1)[ik[kb]]
                   - cred("left", kt + 1, nj - kb - 1)[il[kl]] - cred("right", kt + 1, nj - kb - 1)[il[kr]])
            if found[lam] is None or val < found[lam][0] - 1e-7 * abs(found[lam][0]) - 1e-6:
                found[lam] = (float(val), float(phi), {"top": s * kt - cv.V, "bottom": s * kb - cv.V,
                                                        "left": s * kl - cv.U, "right": s * kr - cv.U})
    t2 = time.perf_counter()

    def refine(phis, s, half_px, beta0):
        return _refine_rect(D, pools, phis, s, half_px, beta0)

    res, done = {}, {}
    found["plain"] = best
    for lam, (v, phi, beta) in found.items():
        key = (phi, tuple(sorted(beta.items())))
        if key not in done:
            b2 = refine(phi + np.arange(-0.1, 0.1 + 1e-9, 0.02), 2, 24, beta)
            b1 = refine(b2[1] + np.arange(-0.02, 0.02 + 1e-9, 0.01), 1, 4, b2[2])
            done[key] = (Frame(h, w, b1[1], {k: float(v) for k, v in b1[2].items()}), b1[0])
        res[lam] = done[key]
    if timing is not None:
        timing["frame_global"] = t1 - t0
        timing["frame_notch_aware"] = t2 - t1
        timing["frame_refine"] = time.perf_counter() - t2
    return res


# ----------------------------------------------------------------------------------------------------------------
# 2. Per-side tilt and bow
# ----------------------------------------------------------------------------------------------------------------

END_CORNERS = {"top": (("top", "left"), ("top", "right")), "bottom": (("bottom", "left"), ("bottom", "right")),
               "left": (("top", "left"), ("bottom", "left")), "right": (("top", "right"), ("bottom", "right"))}


def refine_sides(fr: Frame, A1, rounds=2):
    """Each side's tilt (within +-TILT_MAX deg) and bow (sag within +-SAG_MAX px), its offset re-optimised (+-24 px),
    from the running sums along the side's normal on the full-res canvas A1 of the frame's angle.  The columns
    within a corner's arc radius (+8 px) of either end are left out: the arc decides there."""
    cv = fr.canvas
    for _ in range(rounds):
        for side in SIDES:
            E, Aoff = cv.EA(side)
            lo, hi = fr.ends(side)
            m, hw = (lo + hi) / 2.0, max((hi - lo) / 2.0, 1.0)
            view = side_view(A1, side)
            c_lo, c_hi = END_CORNERS[side]
            f_lo = lo + (fr.arc[c_lo] + 8 if fr.arc[c_lo] > 0 else 0)
            f_hi = hi - (fr.arc[c_hi] + 8 if fr.arc[c_hi] > 0 else 0)
            a0 = max(0, int(math.ceil(f_lo + Aoff - 0.5)))
            a1 = min(view.shape[1], int(math.ceil(f_hi + Aoff - 0.5)))
            R = 24 + int(math.ceil(math.tan(math.radians(TILT_MAX)) * hw)) + SAG_MAX + 4
            base = int(math.floor(fr.beta[side] + E))
            e0, e1 = max(0, base - R), min(view.shape[0], base + R)
            band = view[e0:e1, a0:a1].astype(np.float64)
            C = np.zeros((band.shape[0] + 1, band.shape[1]))
            C[1:] = np.cumsum(band, 0)
            al = np.arange(a0, a1) - Aoff + 0.5
            x = (al - m) / hw
            cols = np.arange(a1 - a0)
            beta0 = fr.beta[side]

            def score(taus, sags, offs, step=1):
                best = (-np.inf, 0.0, 0.0, 0.0)
                cc = cols[::step]
                Cs = C[:, ::step]
                # simplest first: on 1-bit pages a side over blank paper ties for every tilt and bow, and the
                # strict comparison then keeps zero tilt and zero bow (first run: every side at -0.45 deg, -22 px)
                for tau in sorted(taus, key=abs):
                    for sg in sorted(sags, key=abs):
                        g = tau * (al[::step] - m) + sg * (1 - x[::step] ** 2)
                        kk = np.ceil(beta0 + np.asarray(offs)[:, None] + g[None, :] + E - 0.5).astype(np.int64) - e0
                        np.clip(kk, 0, C.shape[0] - 1, out=kk)
                        tot = Cs[kk, np.arange(len(cc))[None, :]].sum(1)
                        i = int(np.argmax(tot))
                        if tot[i] > best[0] + 1e-6:
                            best = (float(tot[i]), float(tau), float(sg), float(offs[i]))
                return best

            tmax = math.tan(math.radians(TILT_MAX))
            b = score(np.tan(np.radians(np.linspace(-TILT_MAX, TILT_MAX, 7))), np.arange(-SAG_MAX, SAG_MAX + 1, 2),
                      np.arange(-24, 25), step=2)
            dt = math.tan(math.radians(TILT_MAX / 3))
            taus = np.unique(np.clip(np.r_[b[1] + np.linspace(-dt, dt, 13), 0.0], -tmax, tmax))
            sags = np.unique(np.clip(np.r_[b[2] + np.arange(-2, 2.01, 0.5), 0.0], -SAG_MAX, SAG_MAX))
            b = score(taus, sags, b[3] + np.arange(-3, 4), step=1)
            fr.beta[side] = beta0 + b[3]
            fr.tilt[side], fr.sag[side] = b[1], b[2]
    return fr


# ----------------------------------------------------------------------------------------------------------------
# 3. Corner arcs: a fillet tangent to both sides (the labels' form), radius by a 1-D search
# ----------------------------------------------------------------------------------------------------------------

def fit_arcs(fr: Frame, A1, rmax=ARC_MAX):
    """Per corner the radius r in 0..rmax maximising the gain sum D over the cut (the corner outside a circle of
    radius r tangent to both sides): a px at inward distances (a, b) from the two sides is cut iff
    a + b + sqrt(2ab) < r, so the cuts nest and one histogram gives every r."""
    cv = fr.canvas
    nj, ni = A1.shape
    for c in CORNER_SIDES:
        s1, s2 = c
        uc = fr.beta["left"] if s2 == "left" else -fr.beta["right"]
        vc = fr.beta["top"] if s1 == "top" else -fr.beta["bottom"]
        du = 1 if s2 == "left" else -1
        dv = 1 if s1 == "top" else -1
        span = rmax + 48
        ua, ub = sorted((uc - 48 * du, uc + span * du))
        va, vb = sorted((vc - 48 * dv, vc + span * dv))
        i0, i1 = max(0, int(ua + cv.U)), min(ni, int(ub + cv.U) + 1)
        j0, j1 = max(0, int(va + cv.V)), min(nj, int(vb + cv.V) + 1)
        jj, ii = np.mgrid[j0:j1, i0:i1]
        u, v = ii - cv.U + 0.5, jj - cv.V + 0.5
        e1, al1 = e_along(s1, u, v)
        e2, al2 = e_along(s2, u, v)
        a = e1 - fr.bound(s1, al1)
        b = e2 - fr.bound(s2, al2)
        ok = (a >= 0) & (b >= 0) & (a < rmax) & (b < rmax)
        rs = a[ok] + b[ok] + np.sqrt(2 * a[ok] * b[ok])
        sel = rs < rmax
        hist = np.bincount(np.floor(rs[sel]).astype(np.int64), weights=A1[j0:j1, i0:i1][ok][sel].astype(np.float64),
                           minlength=rmax + 1)[:rmax + 1]
        gain = np.concatenate([[0.0], np.cumsum(hist)[:rmax]])      # gain[r] = sum over r* < r
        fr.arc[c] = float(np.argmax(gain))
    return fr


# ----------------------------------------------------------------------------------------------------------------
# 4. Notches: rotated rectangles anchored at a side, greedily
# ----------------------------------------------------------------------------------------------------------------

NOTCH_MIN_PX = 2 * STRIP + 1   # a notch narrower than its two side strips is not a box (first run: 1-3 px slivers
                               # along street lines, paid for by beta x the line's ink counted on both sides)


def kadane(Dv, Iv, bw, s, kc, d_lo=1, d_hi=None):
    """The best notch on a side view (rows from the canvas edge inward, columns along the side, cells of s px): rows
    [0, d) x columns [a0, a1).  Its value: the sum of Dv over it + bw x the ink along its three inner sides, each the
    ink in the strip of kc cells just inside it per px of side.  For every depth d the best span is a maximum-sum
    subarray with end terms (Kadane, with a running maximum of the start term).  Returns (value, a0, a1, d)."""
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
    col = G[ds] + bw * (Q[ds] - Q[ds - kc]) / norm          # D down to depth d, + the parallel side's strip
    P = np.cumsum(col, 1)
    Pm1 = np.zeros_like(P)
    Pm1[:, 1:] = P[:, :-1]
    if bw:
        Qc = np.zeros((len(ds), n_a + 1))
        Qc[:, 1:] = np.cumsum(Q[ds], 1)
        idx = np.arange(n_a)
        pL = (Qc[:, np.minimum(idx + kc, n_a)] - Qc[:, idx]) / norm        # strip of kc columns starting at a0
        pR = (Qc[:, idx + 1] - Qc[:, np.maximum(idx + 1 - kc, 0)]) / norm  # ... ending at a1 - 1
        M = bw * pL - Pm1
        end = bw * pR
    else:
        M = -Pm1
        end = 0.0
    wmin = int(math.ceil(NOTCH_MIN_PX / s))           # spans of at least wmin columns: a0 <= a1 - wmin + 1
    Mx = np.full_like(M, -np.inf)
    if wmin <= n_a:
        Mx[:, wmin - 1:] = np.maximum.accumulate(M, 1)[:, :n_a - wmin + 1]
    val = P + end + Mx
    di, a1 = np.unravel_index(int(np.argmax(val)), val.shape)
    if not np.isfinite(val[di, a1]):
        return -np.inf, 0, 0, 0
    a0 = int(np.argmax(M[di, :a1 - wmin + 2]))
    return float(val[di, a1]), a0, int(a1) + 1, int(ds[di])


def best_notch(DK, IK, fr: Frame, bw, coarse=None):
    """The best single notch for K (DK = D on K, IK = ink on K): coarse on cells of 8 over every side and angle in
    phi +-NOTCH_DEG (steps <= 0.25 deg), then refined at full res (angle +-0.25 at 0.05 deg, ends and depth +-16 px)."""
    h, w = DK.shape
    s = 8
    DKp, IKp = pool(DK, s), pool(IK, s)
    n_ang = int(math.ceil(2 * NOTCH_DEG / 0.25)) + 1
    best = (-np.inf,)
    for dpsi in np.linspace(-NOTCH_DEG, NOTCH_DEG, n_ang):
        psi = fr.phi + float(dpsi)
        cv = Canvas.page(psi, h, w)
        Dw, Iw = cv.warp(DKp, s, s), cv.warp(IKp, s, s)
        for side in SIDES:
            E, _ = cv.EA(side)
            half = (h if side in ("top", "bottom") else w) / 2.0
            ne = int((E - half + NOTCH_REACH * 2 * half) // s)
            r = kadane(side_view(Dw, side)[:ne], side_view(Iw, side)[:ne], bw, s, 1)
            if r[0] > best[0]:
                best = (r[0], side, psi, r[1], r[2], r[3])
    if not np.isfinite(best[0]):
        return None
    _, side, psi_c, a0c, a1c, dc = best
    fine = (-np.inf,)
    for psi in psi_c + np.arange(-0.25, 0.2501, 0.05):
        cv = Canvas.page(psi, h, w)
        E, A = cv.EA(side)
        n_e, n_a = (cv.cells(1) if side in ("top", "bottom") else cv.cells(1)[::-1])
        e1 = min(n_e, s * dc + 2 * s)
        al0, al1 = max(0, s * a0c - 2 * s), min(n_a, s * a1c + 2 * s)
        Dv = cv.side_window(side, DK, 1, 1, 0, e1, al0, al1, binned=True)
        Iv = cv.side_window(side, IK, 1, 1, 0, e1, al0, al1, binned=True)
        r = kadane(Dv, Iv, bw, 1, STRIP, d_lo=max(1, s * dc - 2 * s), d_hi=e1)
        if r[0] > fine[0]:
            fine = (r[0], psi, r[1] + al0, r[2] + al0, r[3], E, A)
    val, psi, a0, a1, d, E, A = fine
    return Notch(side, float(psi), float(a0 - A), float(a1 - A), float(d - E), float(val))


def greedy_notches(D, ink, K0, fr: Frame, bw, cap=NOTCH_CAP, lam=0.0):
    """Notches one at a time while the best one's value exceeds lam: (notches, K after each)."""
    K = K0.copy()
    out = []
    for _ in range(cap):
        IK = ink * K
        n = best_notch(D * K, IK, fr, bw)
        if n is None:
            break
        m = notch_mask(n, *K.shape) & K
        if not m.any():
            break
        # the exact value on the page px (the search ran on warped grids); it decides, so a grid over lam only
        # takes prefixes of the lam = 0 sequence
        n.ink = _side_ink(n, IK, K.shape)
        n.gain = float(D[m].sum(dtype=np.float64)) + bw * n.ink
        if n.gain <= lam:
            break
        out.append(n)
        K &= ~m
    return out


def _side_ink(n: Notch, IK, shape):
    """The ink in the STRIP px just inside the notch's three inner sides, per px of side (page px, centre rule)."""
    h, w = shape
    cv = Canvas.page(n.psi, h, w)
    m = notch_mask(n, h, w)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return 0.0
    u, v = cv.uv(xs, ys)
    e, al = e_along(n.side, u, v)
    strip = (e >= n.d - STRIP) | (al < n.a0 + STRIP) | (al >= n.a1 - STRIP)
    return float(IK[ys[strip], xs[strip]].sum(dtype=np.float64)) / STRIP


# ----------------------------------------------------------------------------------------------------------------
# 5. Removed blocks: rotated rectangles and ellipses
# ----------------------------------------------------------------------------------------------------------------

def _wrap90(deg):
    return (deg + 45.0) % 90.0 - 45.0


def _rect_on(DKc, x0, y0, cx, cy, theta, init):
    """The best rectangle at angle theta about (cx, cy) on the crop DKc (page px (x0, y0) at its (0, 0)): coordinate
    descent over its four edges, each an exact 1-D scan on the summed-area table, from `init` = (u0, u1, v0, v1)."""
    hc, wc = DKc.shape
    half = math.hypot(hc, wc) / 2 + 4
    cv = Canvas(theta, cx, cy, half + abs(cx - (x0 + wc / 2)), half + abs(cy - (y0 + hc / 2)))
    A = cv.warp(DKc, 1, 1, x_off=x0, y_off=y0)
    S = np.zeros((A.shape[0] + 1, A.shape[1] + 1))
    S[1:, 1:] = A.cumsum(0, dtype=np.float64).cumsum(1)
    nj, ni = A.shape
    il = int(np.clip(math.floor(init[0] + cv.U), 0, ni - 1)); ir = int(np.clip(math.ceil(init[1] + cv.U), il + 1, ni))
    jt = int(np.clip(math.floor(init[2] + cv.V), 0, nj - 1)); jb = int(np.clip(math.ceil(init[3] + cv.V), jt + 1, nj))
    for _ in range(20):
        old = (il, ir, jt, jb)
        f = S[:, ir] - S[:, il]
        jt = int(np.argmin(f[:jb])); jb = jt + 1 + int(np.argmax(f[jt + 1:]))
        g = S[jb] - S[jt]
        il = int(np.argmin(g[:ir])); ir = il + 1 + int(np.argmax(g[il + 1:]))
        if (il, ir, jt, jb) == old:
            break
    gain = S[jb, ir] - S[jt, ir] - S[jb, il] + S[jt, il]
    return float(gain), Block("rect", cx, cy, float(theta), (il - cv.U, ir - cv.U, jt - cv.V, jb - cv.V), float(gain))


def _ellipse_gain(DKc, X, Y, cx, cy, a, b, th):
    t = math.radians(th)
    c, s = math.cos(t), math.sin(t)
    dx, dy = X - cx, Y - cy
    u, v = c * dx + s * dy, -s * dx + c * dy
    return float(DKc[(u / a) ** 2 + (v / b) ** 2 <= 1.0].sum(dtype=np.float64))


def _ellipse_on(DKc, x0, y0, cx, cy, a, b, th):
    """Pattern search over (cx, cy, a, b, theta) maximising sum D over the ellipse; coarse on cells of f px for big
    crops, then at full res."""
    hc, wc = DKc.shape
    f = max(1, int(math.ceil(max(hc, wc) / 300)))
    p = np.array([cx, cy, max(a, 2.0), max(b, 2.0), th])

    def run(arr, f, p, steps, minsteps):
        Y, X = np.mgrid[0:arr.shape[0], 0:arr.shape[1]]
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
                    if q[2] < 2 or q[3] < 2:
                        continue
                    g = _ellipse_gain(arr, X, Y, *q)
                    if g > cur:
                        cur, p, improved = g, q, True
            if not improved:
                steps = steps / 2
        return cur, p

    if f > 1:
        _, p = run(pool(DKc, f), f, p, [8 * f, 8 * f, 8 * f, 8 * f, 8.0], [f, f, f, f, 1.0])
        st = [2 * f] * 4 + [1.0]
    else:
        st = [max(2.0, p[2] / 8)] * 2 + [max(2.0, p[2] / 8), max(2.0, p[3] / 8), 8.0]
    g, p = run(DKc, 1, p, st, [1, 1, 1, 1, 0.25])
    return g, Block("ellipse", float(p[0]), float(p[1]), float(p[4]), (float(p[2]), float(p[3])), g)


def block_shapes(p, D, K, phi, level=None):
    """Per 8-connected piece of p2 > level inside K (>= BLOCK_MIN_PX px): the better of the best rotated rectangle and
    the best ellipse, both found by local search from the piece's own moments, scored by sum D over the shape within
    K (the gain of deleting it)."""
    level = decide.cut_level() if level is None else level
    h, w = D.shape
    DK = np.where(K, D, 0).astype(np.float32)
    lab, n = ndimage.label((p[2] > level) & K, structure=_EIGHT)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None or sizes[k] < BLOCK_MIN_PX:
            continue
        ys, xs = np.nonzero(lab[sl] == k)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        bh, bw_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        mg = max(24, int(0.3 * max(bh, bw_)))
        y0, y1 = max(0, sl[0].start - mg), min(h, sl[0].stop + mg)
        x0, x1 = max(0, sl[1].start - mg), min(w, sl[1].stop + mg)
        DKc = DK[y0:y1, x0:x1]
        cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
        cov = np.cov(np.stack([xs, ys])) if len(xs) > 2 else np.eye(2)
        ev, evec = np.linalg.eigh(cov)
        thm = math.degrees(math.atan2(evec[1, 1], evec[0, 1]))
        best = None
        angles = sorted({round(_wrap90(thm) + d, 3) for d in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3)} |
                        {round(phi + d, 3) for d in (-0.5, -0.25, 0, 0.25, 0.5)})
        for th in angles:
            t = math.radians(th)
            c, s = math.cos(t), math.sin(t)
            pu = c * (xs + 0.5 - cx) + s * (ys + 0.5 - cy)
            pv = -s * (xs + 0.5 - cx) + c * (ys + 0.5 - cy)
            g, blk = _rect_on(DKc, x0, y0, cx, cy, th, (pu.min(), pu.max() + 1, pv.min(), pv.max() + 1))
            if best is None or g > best.gain:
                best = blk
        a, b = 2 * math.sqrt(max(ev[1], 1.0)), 2 * math.sqrt(max(ev[0], 1.0))
        g, blk = _ellipse_on(DKc, x0, y0, cx, cy, a, b, thm)
        if g > best.gain:
            best = blk
        out.append(best)
    out.sort(key=lambda b: -b.gain)
    return out


def accept_blocks(blocks, D, K, lam):
    """The blocks kept, in order of gain, each re-scored within K less the blocks already kept: removed mask."""
    h, w = D.shape
    removed = np.zeros((h, w), bool)
    kept = []
    for b in blocks:
        if b.gain <= lam:
            break
        y0, y1, x0, x1, m = block_mask_box(b, h, w)
        m = m & K[y0:y1, x0:x1] & ~removed[y0:y1, x0:x1]
        g = float(D[y0:y1, x0:x1][m].sum(dtype=np.float64))
        if g > lam:
            removed[y0:y1, x0:x1] |= m
            kept.append(b)
    return removed, kept


# ----------------------------------------------------------------------------------------------------------------
# The whole decision
# ----------------------------------------------------------------------------------------------------------------

def cost_map(p, lum):
    ink = (1.0 - np.asarray(lum, np.float32))
    D = ink * (np.float32(decide.COST_FRAME) * (p[1] + p[2]) - np.float32(decide.COST_MAP) * p[0])
    return D.astype(np.float32), ink


def notches_mask(notches, h, w):
    m = np.zeros((h, w), bool)
    for n in notches:
        m |= notch_mask(n, h, w)
    return m


def fit_sides_arcs(rect: Frame, Dz, notches):
    """Steps 2-3 given the notches and blocks (Dz: D with the notch and block px zeroed, as they are deleted
    whatever the sides do): a copy of the rectangle with each side's tilt, bow and offset and each corner's arc."""
    fr = rect.copy()
    fr.notches = list(notches)
    A1 = fr.canvas.bin(Dz)
    refine_sides(fr, A1)
    fit_arcs(fr, A1)
    if any(r > 0 for r in fr.arc.values()):      # coordinate descent: the sides again without the arcs' ends
        refine_sides(fr, A1, rounds=1)
        fit_arcs(fr, A1)
    return fr


def objective(D, K, removed, notches, kept_blocks, lam_n, bw, lam_b):
    """The joint objective the shape minimises: sum D over the kept px + lambda_notch per notch + lambda_block per
    block - beta x the notches' side ink."""
    return (float(D[K & ~removed].sum(dtype=np.float64)) + lam_n * len(notches) + lam_b * len(kept_blocks)
            - bw * sum(n.ink for n in notches))


def pipeline(rect, p, D, ink, lam_n, bw, lam_b, timing=None):
    """Steps 2-5 from a rectangle (see `shape_decision`): (K, removed, Frame, kept blocks, objective)."""
    h, w = D.shape
    t0 = time.perf_counter()
    K0 = rect.kept()
    shapes = block_shapes(p, D, K0, rect.phi)
    B0, _ = accept_blocks(shapes, D, K0, lam_b)
    t1 = time.perf_counter()
    Dz = np.where(B0, 0, D).astype(np.float32)
    inkz = np.where(B0, 0, ink).astype(np.float32)
    notches = greedy_notches(Dz, inkz, K0, rect, bw, lam=lam_n)
    t2 = time.perf_counter()
    Nm = notches_mask(notches, h, w)
    Dzz = np.where(Nm, 0, Dz).astype(np.float32)
    fr = fit_sides_arcs(rect_given(rect, Dzz), Dzz, notches)
    K = fr.kept()
    removed, kept = accept_blocks(shapes, D, K, lam_b)
    t3 = time.perf_counter()
    if timing is not None:
        for k, v in (("blocks", t1 - t0), ("notches", t2 - t1), ("sides_arcs_final", t3 - t2)):
            timing[k] = timing.get(k, 0.0) + v
    return K, removed & K, fr, kept, objective(D, K, removed & K, notches, kept, lam_n, bw, lam_b)


def shape_decision(p, lum, lam_notch, bw, lam_block, timing=None, notch_aware=True):
    """The whole shape decision at one set of numbers: (Decision, Frame with its notches, kept blocks).  One sweep of
    coordinate descent, each piece exact (or greedy, for the notches) given the ones before:
    1. the frame rectangle (global; see `frame_search`);
    2. the block shapes inside it, kept by lam_block;
    3. the notches, greedily, with the blocks' px zeroed (a greedy notch otherwise swallows a block near a side
       together with the map between: the synthetic page's badge);
    4. the rectangle again (angle +-0.5 deg, offsets +-32 px), then each side's tilt, bow and offset and the corner
       arcs, all with the notches' and blocks' px zeroed (fitted before them, a side bows into a notch: the synthetic
       page's bottom side);
    5. the blocks re-scored within the final K.
    Run from two starts, the plain best rectangle and the notch-aware one; the lower joint objective wins (each
    start is a local search; the objective is the one being minimised).  `notch_aware` False: the plain start only."""
    timing = {} if timing is None else timing
    t0 = time.perf_counter()
    D, ink = cost_map(p, lum)
    rects = frame_search(D, (lam_notch,), timing)
    starts = {"plain": rects["plain"][0]}
    if notch_aware and rects[lam_notch][0] is not rects["plain"][0]:
        starts["notch-aware"] = rects[lam_notch][0]
    best = None
    for name, rect in starts.items():
        r = pipeline(rect, p, D, ink, lam_notch, bw, lam_block, timing)
        if best is None or r[4] < best[1][4]:
            best = (name, r)
    name, (K, removed, fr, kept, J) = best
    timing["total"] = time.perf_counter() - t0
    timing["start"] = name
    timing["starts"] = len(starts)
    return to_decision(K, removed, ink), fr, kept


def to_decision(K, removed, ink):
    outside = ~K
    rem = removed & K
    return decide.Decision(outside, rem, 0.0, 0.0, float(ink[rem].sum(dtype=np.float64)))


# ----------------------------------------------------------------------------------------------------------------
# Self-check on synthetic pages (truth geometry written out independently of Frame / Notch)
# ----------------------------------------------------------------------------------------------------------------

def _rot(phi, X, Y, cx, cy):
    t = math.radians(phi)
    dx, dy = X + 0.5 - cx, Y + 0.5 - cy
    return math.cos(t) * dx + math.sin(t) * dy, -math.sin(t) * dx + math.cos(t) * dy


def make_synth(kind, seed=0):
    """A synthetic page: (lum, p, truth).  "A": frame at 0.7 deg, top side bowed 10 px inward, fillet arcs of 120 px
    at every corner, a ruled notch on the bottom side turned 1.0 deg off the frame, a ruled legend (rectangle) and a
    badge (circle) inside.  "B": frame at -1.3 deg on three sides, the map running off the right page edge."""
    rng = np.random.default_rng(seed)
    if kind == "A":
        h, w, phi = 2400, 3200, 0.7
        beta = {"top": -1000.0, "bottom": -1010.0, "left": -1400.0, "right": -1420.0}
        sag_top, arc_r = 10.0, 120.0
        notch = dict(psi=phi + 1.0, a0=300.0, a1=1000.0, d=-620.0)
        right_edge = False
    else:
        h, w, phi = 2000, 2600, -1.3
        beta = {"top": -880.0, "bottom": -890.0, "left": -1200.0, "right": -5000.0}
        sag_top, arc_r, notch, right_edge = 0.0, 0.0, None, True
    cx, cy = w / 2.0, h / 2.0
    Y, X = np.mgrid[0:h, 0:w]
    u, v = _rot(phi, X, Y, cx, cy)
    # inward distances from each side (top bowed)
    lo, hi = beta["left"], -beta["right"]
    m, hw = (lo + hi) / 2, (hi - lo) / 2
    dist = {"top": v - (beta["top"] + sag_top * (1 - ((u - m) / hw) ** 2)), "bottom": -v - beta["bottom"],
            "left": u - beta["left"], "right": -u - beta["right"]}
    inside = np.ones((h, w), bool)
    for d in dist.values():
        inside &= d >= 0
    ink = np.zeros((h, w), np.float32)
    # rule (5 px) and bar (15 px) just outside each frame side
    for s, d in dist.items():
        if s == "right" and right_edge:
            continue
        others = [dist[o] for o in dist if o != s and not (o == "right" and right_edge)]
        span = np.ones((h, w), bool)
        for o in others:
            span &= o >= -30
        ink[span & (d < 0) & (d >= -5)] = 1
        ink[span & (d < -15) & (d >= -30)] = 1
    cut_arc = np.zeros((h, w), bool)
    if arc_r > 0:
        for s1, s2 in CORNER_SIDES:
            a, b = dist[s1], dist[s2]
            near = (a >= 0) & (b >= 0) & (a < arc_r) & (b < arc_r)
            delta = np.hypot(a - arc_r, b - arc_r)
            cut_arc |= near & (delta > arc_r)
            ink[near & (delta > arc_r) & (delta < arc_r + 4)] = 1
    K = inside & ~cut_arc
    N = np.zeros((h, w), bool)
    if notch:
        un, vn = _rot(notch["psi"], X, Y, cx, cy)
        e, al = -vn, un
        N = (al >= notch["a0"]) & (al < notch["a1"]) & (e < notch["d"]) & K
        strip = N & ((e >= notch["d"] - 4) | (al < notch["a0"] + 4) | (al >= notch["a1"] - 4))
        ink[strip] = 1
        ink[N & ~strip & (rng.random((h, w)) < 0.15)] = 1
        K &= ~N
    blocks = np.zeros((h, w), bool)
    btruth = []
    if kind == "A":
        r = (u >= -1100) & (u < -700) & (v >= 500) & (v < 800)
        rb = r & ~((u >= -1097) & (u < -703) & (v >= 503) & (v < 797))
        ink[rb] = 1
        ink[r & ~rb & (rng.random((h, w)) < 0.25)] = 1
        e_ = np.hypot(u - 900, v + 700)
        c = e_ <= 70
        ink[c & (e_ > 66)] = 1
        ink[c & (e_ <= 66) & (rng.random((h, w)) < 0.25)] = 1
        blocks = (r | c) & K
        btruth = [r & K, c & K]
    mapink = K & ~blocks
    ink[mapink] = (rng.random((h, w)) < 0.2)[mapink]
    cls = np.where(K & ~blocks, 0, 1).astype(np.uint8)
    cls[blocks] = 2
    base = np.array([[0.95, 0.04, 0.01], [0.04, 0.95, 0.01], [0.08, 0.02, 0.90]], np.float32)
    p = base[cls].transpose(2, 0, 1) * rng.uniform(0.8, 1.2, (3, h, w)).astype(np.float32)
    p /= p.sum(0, keepdims=True)
    lum = 1.0 - ink
    truth = dict(phi=phi, beta=beta, sag_top=sag_top, arc=arc_r, notch=notch, K=K, blocks=btruth, cls=cls)
    return lum.astype(np.float32), p.astype(np.float32), truth


def cmd_synth():
    rows = []
    for kind in ("A", "B"):
        lum, p, tr = make_synth(kind)
        timing = {}
        dec, fr, kept = shape_decision(p, lum, lam_notch=1e3, bw=3.0, lam_block=1e3, timing=timing)
        K = ~dec.outside
        Kt = tr["K"]
        xor = K ^ Kt
        # the farthest wrong px from the true boundary
        bd = Kt & ~ndimage.binary_erosion(Kt, border_value=1)
        far = float(ndimage.distance_transform_edt(~bd)[xor].max()) if xor.any() else 0.0
        r = {"kind": kind, "phi_err_deg": fr.phi - tr["phi"],
             "beta_err_px": {s: fr.beta[s] - tr["beta"][s] for s in SIDES if not (kind == "B" and s == "right")},
             "right_beta": fr.beta["right"], "sag_top": fr.sag["top"], "sag_top_true": tr["sag_top"],
             "tilt_deg": {s: math.degrees(math.atan(fr.tilt[s])) for s in SIDES},
             "arcs": {f"{a}-{b}": fr.arc[(a, b)] for a, b in CORNER_SIDES}, "arc_true": tr["arc"],
             "notches": [dict(side=n.side, psi_err_deg=n.psi - (tr["notch"]["psi"] if tr["notch"] else 0),
                              a0=n.a0, a1=n.a1, d=n.d, gain=n.gain) for n in fr.notches],
             "notch_true": tr["notch"],
             "kept_xor_px": int(xor.sum()), "kept_xor_farthest_px": far, "timing_s": timing}
        rem = dec.removed
        r["blocks"] = [{"kind": b.kind, "gain": b.gain} for b in kept]
        r["block_iou"] = [float((rem & t).sum() / max(1, (rem & t).sum() + (t & ~rem).sum() +
                                                     (rem & ~t & ndimage.binary_dilation(t, iterations=40)).sum()))
                          for t in tr["blocks"]]
        rows.append(r)
        print(json.dumps(r, indent=1, default=float))
        img = np.stack([lum * 255] * 3, -1).astype(np.uint8)
        img[xor] = (255, 0, 0)
        edge = K & ~ndimage.binary_erosion(K, border_value=1)
        img[ndimage.binary_dilation(edge)] = (255, 0, 255)
        img[rem & ~ndimage.binary_erosion(rem)] = (0, 160, 255)
        Image.fromarray(img).save(OUT / f"synth_{kind}.png")
    (OUT / "synth.json").write_text(json.dumps(rows, indent=1, default=float))


# ----------------------------------------------------------------------------------------------------------------
# The fold-0 measurement
# ----------------------------------------------------------------------------------------------------------------

ARCH = (16, 5, 32)
SRC = data.DATA_DIR / "frame-arch-fold16" / "fold-0"
CACHE = OUT / "cache"
DEV = "mps"
LAMS_NOTCH = (0.0, 1e3, 1e4, 1e5)
BETAS = (0.0, 1.0, 3.0, 10.0)
LAMS_BLOCK = (0.0, 1e3, 1e4)
COMPARE = [("1948", "14"), ("1948", "8"), ("1948", "12"), ("1921", "20180102094355582")]
WORKERS = 3
ALT = (1e4, 0.0, 1e3)          # the best grid point with beta <= COST_MAP (third grid run)
ALT_DIR = "shape-beta0"


def _setup_model():
    from sfzmaps.frame import model
    model.FOLD, model.LEVELS, model.WIDTH = ARCH
    model.BUCKET = 256          # the padding the fold-0 net was trained with
    return model


def _held():
    from sfzmaps.frame import run as fr_run
    res = json.loads((SRC / fr_run.FOLDS_FILE).read_text())
    return float(res["temperature"]), [(s["series"], s["sheet"]) for s in res["sheets"]]


def _net():
    from sfzmaps.frame import run as fr_run
    return fr_run.load_snapshot(fr_run.snapshot_path(SRC, 0, 0, 4000))


def _cache(sid):
    return CACHE / f"{sid[0]}__{sid[1]}.npy"


def _labels(sid, shape):
    return data.class_mask(data.load_label(*sid), data.load_remove(*sid), shape)


def fast_score(deleted, cls, lum):
    ink = 1.0 - lum.astype(np.float64)
    fk = float(ink[(cls != 0) & ~deleted].sum())
    mc = float(ink[(cls == 0) & deleted].sum())
    return {"frame_kept": fk, "map_cut": mc, "cost": decide.COST_FRAME * fk + decide.COST_MAP * mc}


def _frame_json(fr: Frame, kept_blocks=()):
    return {"phi": fr.phi, "beta": fr.beta, "tilt_deg": {k: math.degrees(math.atan(v)) for k, v in fr.tilt.items()},
            "sag": fr.sag, "arcs": {f"{a}-{b}": r for (a, b), r in fr.arc.items()},
            "notches": [n.__dict__ for n in fr.notches],
            "blocks": [{"kind": b.kind, "cx": b.cx, "cy": b.cy, "theta": b.theta, "p": list(b.p), "gain": b.gain}
                       for b in kept_blocks]}


def grid_sheet(sid):
    """Per-pixel and shape decisions of one held sheet over the grid (lambda_notch x beta x lambda_block), sharing
    what doesn't depend on a number: the rectangles (plain, and notch-aware per lambda_notch), the block shapes per
    rectangle, one greedy notch run per (rectangle, beta, lambda_block) whose prefixes give every lambda_notch.  Per
    combination, the start with the lower joint objective is scored (as `shape_decision` does)."""
    cv2.setNumThreads(2)
    from sfzmaps.frame import scores
    lum = data.load_luminance(data.page_path(*sid))
    cls = _labels(sid, lum.shape)
    p = np.load(_cache(sid))
    counts = data.map_counts(data.PAGE_KINDS)
    pix = decide.decide(p, lum, counts.get(sid, 1))
    pix_sc = scores.sheet_scores(p, pix.deleted, cls, lum)
    t0 = time.perf_counter()
    D, ink = cost_map(p, lum)
    h, w = D.shape
    rects = frame_search(D, LAMS_NOTCH)
    plain = rects["plain"][0]
    # per combination: candidates (objective, score, start) from each start
    cand = {f"{ln:g}|{bw:g}|{lb:g}": [] for ln in LAMS_NOTCH for bw in BETAS for lb in LAMS_BLOCK}
    fits, info = {}, {}
    for rect in {id(r): r for r, _ in rects.values()}.values():
        # the lambdas this rectangle serves: as the plain start (all), as the notch-aware start (its own)
        serves = [(ln, "plain" if rect is plain else "notch-aware") for ln in LAMS_NOTCH
                  if rect is plain or rects[ln][0] is rect]
        if rect is plain:
            serves += [(ln, "notch-aware") for ln in LAMS_NOTCH if rects[ln][0] is plain]
        K0 = rect.kept()
        shapes = block_shapes(p, D, K0, rect.phi)
        for lam_b in LAMS_BLOCK:
            B0, _ = accept_blocks(shapes, D, K0, lam_b)
            Dz = np.where(B0, 0, D).astype(np.float32)
            inkz = np.where(B0, 0, ink).astype(np.float32)
            for bw in BETAS:
                seq = greedy_notches(Dz, inkz, K0, rect, bw, lam=0.0)
                info[f"{id(rect) == id(plain) and 'plain' or 'aware'} beta={bw:g} lam_b={lam_b:g}"] = \
                    [(n.side, round(n.gain), round(n.ink)) for n in seq]
                for lam_n, start in serves:
                    k = 0
                    while k < len(seq) and seq[k].gain > lam_n:
                        k += 1
                    notches = seq[:k]
                    key = (id(rect), lam_b, tuple((n.side, n.psi, n.a0, n.a1, n.d) for n in notches))
                    if key not in fits:
                        Nm = notches_mask(notches, h, w)
                        Dzz = np.where(Nm, 0, Dz).astype(np.float32)
                        fr = fit_sides_arcs(rect_given(rect, Dzz), Dzz, notches)
                        K = fr.kept()
                        removed, kept = accept_blocks(shapes, D, K, lam_b)
                        removed &= K
                        fits[key] = (K, removed, kept, fast_score(~K | removed, cls, lum))
                    K, removed, kept, sc = fits[key]
                    J = objective(D, K, removed, notches, kept, lam_n, bw, lam_b)
                    cand[f"{lam_n:g}|{bw:g}|{lam_b:g}"].append(
                        (J, {**sc, "sumD": float(D[K & ~removed].sum(dtype=np.float64))}, start))
        fits = {k: (None, None, v[2], v[3]) if k[0] != id(rect) else v for k, v in fits.items()}
    table = {}
    for key, cs in cand.items():
        J, sc, start = min(cs, key=lambda c: c[0])
        table[key] = {**sc, "start": start, "J": J,
                      "plain_cost": min((c for c in cs if c[2] == "plain"), key=lambda c: c[0])[1]["cost"]}
    return {"sid": list(sid), "pixel": {k: pix_sc[k] for k in ("frame_kept", "map_cut", "cost")},
            "pixel_fast": fast_score(pix.deleted, cls, lum), "table": table, "seqs": info,
            "label_sumD": float(D[cls == 0].sum(dtype=np.float64)),
            "pixel_sumD": float(D[~pix.deleted].sum(dtype=np.float64)),
            "rects": {str(k): {"phi": r.phi, "beta": r.beta} for k, (r, _) in rects.items()},
            "grid_s": time.perf_counter() - t0}


def final_page(sid, lam_n, bw, lam_b, p_path=None, held=False, plain_rect=False):
    """Both decisions of one page at the fitted numbers, written under OUT/pixel and OUT/shape; returns the scores
    (held sheets), the shape's timing and its pieces."""
    cv2.setNumThreads(2)
    from sfzmaps.frame import run as fr_run, scores
    lum = data.load_luminance(data.page_path(*sid))
    path = Path(p_path) if p_path else _cache(sid)
    p = np.load(path)
    maps = data.map_counts(data.PAGE_KINDS).get(sid, 1)
    pix = decide.decide(p, lum, maps)
    timing = {}
    if maps > 1:
        shp, fr, kept = pix, None, []          # several maps: the per-pixel decision, listed
    else:
        shp, fr, kept = shape_decision(p, lum, lam_n, bw, lam_b, timing)
    out = {"sid": list(sid), "maps": maps, "timing": timing,
           "shape": _frame_json(fr, kept) if fr is not None else None}
    if held:
        cls = _labels(sid, lum.shape)
        keys = ("frame_kept", "map_cut", "cost", "frame_kept_share", "map_cut_share", "hd95_px")
        out["pixel"] = {k: v for k, v in scores.sheet_scores(p, pix.deleted, cls, lum).items() if k in keys}
        out["shape_scores"] = {k: v for k, v in scores.sheet_scores(p, shp.deleted, cls, lum).items() if k in keys}
        if plain_rect:
            d2, _, _ = shape_decision(p, lum, lam_n, bw, lam_b, None, notch_aware=False)
            out["plain_rect"] = fast_score(d2.deleted, cls, lum)
    fr_run.write_outputs(OUT / "pixel", sid, p, pix, maps)
    fr_run.write_outputs(OUT / "shape", sid, p, shp, maps)
    if ALT is not None and maps == 1:
        # the second shape arm: the best grid point with beta <= COST_MAP (see the report)
        tm2 = {}
        d2, fr2, kept2 = shape_decision(p, lum, *ALT, tm2)
        out["alt"] = {"numbers": list(ALT), "timing": tm2, "shape": _frame_json(fr2, kept2)}
        if held:
            out["alt"]["scores"] = {k: v for k, v in scores.sheet_scores(p, d2.deleted, cls, lum).items() if k in keys}
        fr_run.write_outputs(OUT / ALT_DIR, sid, p, d2, maps)
    if p_path:
        path.unlink(missing_ok=True)
    return out


def _pool():
    import concurrent.futures
    import multiprocessing
    return concurrent.futures.ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("spawn"))


def cmd_chances():
    """The held sheets' chances, computed once (one GPU turn) and cached as float32."""
    from sfzmaps import gpu
    model = _setup_model()
    T, held = _held()
    net = _net()
    CACHE.mkdir(parents=True, exist_ok=True)
    with gpu.gpu_turn(True, DEV):
        for sid in held:
            if _cache(sid).exists():
                continue
            lum = data.load_luminance(data.page_path(*sid))
            np.save(_cache(sid), model.prob([net], lum, DEV, T))
            log.info("chances %s/%s", *sid)


def cmd_grid():
    _, held = _held()
    with _pool() as ex:
        res = list(ex.map(grid_sheet, held))
    pooled_pix = sum(r["pixel"]["cost"] for r in res)
    log.info("per-pixel pooled cost %.0f (the earlier spikes: 381,666)", pooled_pix)
    keys = list(res[0]["table"])
    pooled = {k: sum(r["table"][k]["cost"] for r in res) for k in keys}
    best = min(keys, key=lambda k: pooled[k])
    log.info("best %s pooled %.0f", best, pooled[best])
    (OUT / "grid.json").write_text(json.dumps({"pixel_pooled": pooled_pix, "pooled": pooled, "best": best,
                                               "sheets": res}, indent=1))


def _write_folds(out, rows, key):
    (out / "folds.json").write_text(json.dumps({"sheets": [
        {"series": r["sid"][0], "sheet": r["sid"][1], "scores": r[key]} for r in rows if key in r]}, indent=1))


def cmd_final():
    """Both decisions at the fitted numbers: the held sheets (scored, timed), then the picture pages (the 1948 map
    pages and the close-ups' pages; their chances one GPU turn each), then the pictures."""
    from sfzmaps import gpu
    from sfzmaps.frame import look
    g = json.loads((OUT / "grid.json").read_text())
    lam_n, bw, lam_b = (float(x) for x in g["best"].split("|"))
    model = _setup_model()
    T, held = _held()
    sids = sorted({sid for sid in data.map_pages(data.PAGE_KINDS, data.WORK) if sid[0] == "1948"}
                  | {(s, h) for s, h, _, _ in look.CLOSEUPS} | set(held))
    rest = [s for s in sids if s not in held]
    net = _net()
    tmp = OUT / "tmp-chances"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    with _pool() as ex:
        futs = [ex.submit(final_page, sid, lam_n, bw, lam_b, None, True, False) for sid in held]
        for sid in rest:
            while sum(not f.done() for f in futs) > WORKERS + 1:   # few chances on disk at once
                time.sleep(2)
            lum = data.load_luminance(data.page_path(*sid))
            with gpu.gpu_turn(True, DEV):
                p = model.prob([net], lum, DEV, T)
            path = tmp / f"{sid[0]}__{sid[1]}.npy"
            np.save(path, p)
            del p
            futs.append(ex.submit(final_page, sid, lam_n, bw, lam_b, str(path), False, False))
        rows = [f.result() for f in futs]
    (OUT / "final.json").write_text(json.dumps({"numbers": {"lam_notch": lam_n, "beta": bw, "lam_block": lam_b},
                                                "pages": rows}, indent=1, default=float))
    _write_folds(OUT / "pixel", rows, "pixel")
    _write_folds(OUT / "shape", rows, "shape_scores")
    (OUT / ALT_DIR / "folds.json").write_text(json.dumps({"sheets": [
        {"series": r["sid"][0], "sheet": r["sid"][1], "scores": r["alt"]["scores"]}
        for r in rows if "alt" in r and "scores" in r["alt"]]}, indent=1))
    cmd_pictures()


def _pictures_one(out):
    from sfzmaps.frame import look
    look.series_mosaics(out)
    look.closeups(out)
    return str(out)


def cmd_pictures():
    from sfzmaps.frame import look
    with _pool() as ex:
        list(ex.map(_pictures_one, [OUT / "pixel", OUT / "shape", OUT / ALT_DIR]))
    for arms, name in ((("pixel", "shape"), "compare.png"), (("pixel", ALT_DIR), "compare_beta0.png")):
        _compare(arms, name)


def _compare(arms, name):
    from sfzmaps.frame import look
    font = look._font()
    rows = []
    for sid in COMPARE:
        lum = data.load_luminance(data.page_path(*sid))
        tiles = []
        for arm in arms:
            q, deleted = look._load_outputs(OUT / arm, sid)
            tiles.append(look.sheet_picture(lum, q, deleted, 1100))
        rows.append((sid, tiles))
    W = sum(t.width for t in rows[0][1]) + 3 * look.GAP_PX
    W = max(sum(t.width for t in r[1]) + 3 * look.GAP_PX for r in rows)
    H = sum(max(t.height for t in r[1]) + look.CAPTION_PX + look.GAP_PX for r in rows) + look.GAP_PX
    canvas = Image.new("RGB", (W, H), look.BACKGROUND_RGB)
    d = ImageDraw.Draw(canvas)
    y = look.GAP_PX
    for sid, tiles in rows:
        d.text((look.GAP_PX, y + 2), f"{sid[0]}/{sid[1]}: per-pixel cut (left), {arms[1]} cut (right)",
               fill=(0, 0, 0), font=font)
        y += look.CAPTION_PX
        x = look.GAP_PX
        for t in tiles:
            canvas.paste(t, (x, y))
            x += t.width + look.GAP_PX
        y += max(t.height for t in tiles) + look.GAP_PX
    canvas.save(OUT / name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    {"labels": cmd_labels, "synth": cmd_synth, "chances": cmd_chances, "grid": cmd_grid, "final": cmd_final,
     "pictures": cmd_pictures}[sys.argv[1]]()
