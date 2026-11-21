"""The cut lies on paper (label rule 1; design spec, amendment 3, "the cut lies on paper"; ported from the measured
spike `scripts/spikes/issue71_on_paper.py`, its "sum + tilt" rule, report
.superpowers/sdd/2026-11-21-frame-crop-combined/on-paper-report.md).

The layout sees patches of about 50 px, so a fitted line can sit a few px off the printed rule.  Each fitted line
slides along its normal, within one patch width R = max(page sides) / `layout.LONG_PATCHES` of where the fit put it,
to the position t minimising

    J(t) = E(t) + COST_FRAME x (the ink in the first kept row)

over the px the line alone decides (kept or deleted by no other part of the shape), binned in rows by their signed
distance s from the line (s > 0 on the kept side; the line at t deletes the rows below t and row t is the first kept
row, the row "under the line").  E(t) is the expected ink cost of the window's rows the line keeps, the sum of -D
(`geometry.cost_field`'s D > 0 means "cheaper kept") over them.  The ink under the line is charged at the frame-left-in
price: rule 1 says the outline lies on paper, so ink there is ink the true outline has outside it, kept.  No number is
fitted.  (Measured on fold 0's six held sheets: frame/legend ink left in 1.19% -> 0.81%, map ink cut 0.136% ->
0.139%.  The layout's cost alone moved almost nothing; the lightest row first lands on the blank margin outside a
frame bar; ink at the map price did less.)

- A frame side's two ends move independently within +-R (a shift plus a tilt, the side's tilt deviation staying
  within +-TILT_MAX, controller ruling C6), its cost summed over ALONG_BINS bins along it; fillet radii, notch sides and block sides only shift.
- Ties stay where the fit put the line (paper rows of a 1-bit scan tie exactly).  A side at the page edge doesn't
  move.  A notch's or a block's side never slides past the opposite one: a notch stays at least NOTCH_MIN_PX wide and
  a rectangle 1 px (in the spike a side's window ran past the opposite side, and on 1948/14 a notch came out
  inverted).  The lines go in turn, one pass: the frame's sides, its fillets, each notch's inner sides, each
  block's sides.
"""

from __future__ import annotations

import math

import numpy as np

from sfzmaps.frame.blocks import Block, block_box, block_crop, block_uv
from sfzmaps.frame.decide import COST_FRAME
from sfzmaps.frame.geometry import (
    ARC_MARGIN,
    CORNERS,
    END_CORNERS,
    SIDES,
    TILT_MAX,
    Canvas,
    Frame,
    Lines,
    box,
    cost_field,
    e_along,
    uv_of,
)
from sfzmaps.frame.layout import LONG_PATCHES
from sfzmaps.frame.notches import NOTCH_MIN_PX, Notch, notch_box, notch_crop

ALONG_BINS = 64  # a frame side's columns are binned along it for its two ends' move (computational)


def reach(shape: tuple[int, int]) -> int:
    """One patch width, px: the page's long side / LONG_PATCHES, rounded up."""
    return int(math.ceil(max(shape) / LONG_PATCHES))


class _State:
    """The shape being slid: the frame, the notches and the blocks (copies, changed in place)."""

    def __init__(self, fr: Frame, ns: list[Notch], bs: list[Block], h: int, w: int):
        self.fr, self.ns, self.bs, self.h, self.w = fr, ns, bs, h, w

    def kept(self, y0, y1, x0, x1, skip_side=None, skip_arc=None, skip_notch=None, skip_block=None) -> np.ndarray:
        """Kept on the page crop with one part of the shape left out."""
        K = Lines(self.fr, self.h, self.w).kept_box(y0, y1, x0, x1, skip_side=skip_side, skip_arc=skip_arc)
        for k, n in enumerate(self.ns):
            if k != skip_notch:
                K &= ~_clip(lambda *b, n=n: notch_crop(n, self.h, self.w, *b), notch_box(n, self.h, self.w),
                            (y0, y1, x0, x1))
        for k, b in enumerate(self.bs):
            if k != skip_block:
                K &= ~_clip(lambda *c, b=b: block_crop(b, *c), block_box(b, self.h, self.w), (y0, y1, x0, x1))
        return K


def _clip(make, part_box, crop) -> np.ndarray:
    """A part's mask (`make` on a page box) drawn on the crop, computed only where its box meets the crop."""
    y0, y1, x0, x1 = crop
    out = np.zeros((y1 - y0, x1 - x0), bool)
    ya, yb = max(y0, part_box[0]), min(y1, part_box[1])
    xa, xb = max(x0, part_box[2]), min(x1, part_box[3])
    if ya < yb and xa < xb:
        out[ya - y0 : yb - y0, xa - x0 : xb - x0] = make(ya, yb, xa, xb)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Choosing a line's position
# ---------------------------------------------------------------------------------------------------------------


def _rows(s, zone, ink_c, D_c, R):
    """Per row r in [-R, R) (s in [r, r + 1)) over the line's zone: (ink, D) sums."""
    r = np.floor(s).astype(np.int64)
    sel = zone & (r >= -R) & (r < R)
    idx = r[sel] + R
    ik = np.bincount(idx, weights=ink_c[sel].astype(np.float64), minlength=2 * R)
    dd = np.bincount(idx, weights=D_c[sel].astype(np.float64), minlength=2 * R)
    return ik, dd


def _choose(ik, dd, R, lo: float = -np.inf) -> int:
    """The shift t in [-R, R), and t >= lo, minimising J (see the module); ties nearest 0 (then the outer one)."""
    ts = np.arange(-R, R)
    E = -np.cumsum(dd[::-1])[::-1]  # E[i]: -(sum of D over rows >= i - R)
    J = np.where(ts >= min(lo, 0), E + COST_FRAME * ik, np.inf)
    return int(ts[np.lexsort((np.abs(ts), np.round(J, 6)))[0]])


def _choose_ends(x, s, zone, ink_c, D_c, R, dmin=-np.inf, dmax=np.inf) -> tuple[int, int]:
    """A frame side's two ends each moved within [-R, R) (x: the px's place along the side, 0 at one end and 1 at
    the other), the far end's move less the near end's within [dmin, dmax], minimising J summed over ALONG_BINS bins
    along it; ties: the least total move."""
    r = np.floor(s).astype(np.int64)
    sel = zone & (r >= -R) & (r < R)
    nb, n = ALONG_BINS, 2 * R
    b = np.clip((x[sel] * nb).astype(np.int64), 0, nb - 1)
    idx = b * n + r[sel] + R
    ik = np.bincount(idx, weights=ink_c[sel].astype(np.float64), minlength=nb * n).reshape(nb, n)
    dd = np.bincount(idx, weights=D_c[sel].astype(np.float64), minlength=nb * n).reshape(nb, n)
    J1 = -np.cumsum(dd[:, ::-1], 1)[:, ::-1] + COST_FRAME * ik  # per bin, the line's first kept row at i - R
    xc = (np.arange(nb) + 0.5) / nb
    ts = np.arange(-R, R)
    cols = np.arange(nb)[None, :]
    best = None
    for tl in ts:
        k = np.clip(np.round(tl + (ts[:, None] - tl) * xc[None, :]).astype(np.int64) + R, 0, n - 1)  # (th, bin)
        J = np.where((ts - tl >= dmin) & (ts - tl <= dmax), J1[cols, k].sum(1), np.inf)
        for th, j in zip(ts, J):
            cand = (round(float(j), 6), abs(int(tl)) + abs(int(th)), int(tl), int(th))
            if best is None or cand < best:
                best = cand
    return best[2], best[3]


# ---------------------------------------------------------------------------------------------------------------
# The lines
# ---------------------------------------------------------------------------------------------------------------


def _slide_side(st: _State, side: str, R: int, ink, D) -> None:
    fr, h, w = st.fr, st.h, st.w
    g = Lines(fr, h, w)
    lo, hi = g.ends(side)
    c_lo, c_hi = END_CORNERS[side]
    f_lo = lo + (fr.arc[c_lo] + ARC_MARGIN if fr.arc[c_lo] > 0 else 0)
    f_hi = hi - (fr.arc[c_hi] + ARC_MARGIN if fr.arc[c_hi] > 0 else 0)
    if f_hi <= f_lo:
        return
    al = np.linspace(f_lo, f_hi, 64)
    pts = [g.cv.xy(*uv_of(side, g.bound(side, al) + de, al)) for de in (-R - 2, R + 2)]
    y0, y1, x0, x1 = box(np.concatenate([q[0] for q in pts]), np.concatenate([q[1] for q in pts]), h, w)
    if y1 <= y0 or x1 <= x0:
        return
    y, x = np.mgrid[y0:y1, x0:x1]
    e, a = e_along(side, *g.cv.uv(x, y))
    s = e - g.bound(side, a)
    zone = (a >= f_lo) & (a < f_hi) & st.kept(y0, y1, x0, x1, skip_side=side)
    # the added tilt keeps the side's tilt deviation within +-TILT_MAX (controller ruling C6)
    t0, tmax = math.tan(math.radians(fr.tilt[side])), math.tan(math.radians(TILT_MAX))
    dmin, dmax = (-tmax - t0) * (f_hi - f_lo), (tmax - t0) * (f_hi - f_lo)
    t_lo, t_hi = _choose_ends((a - f_lo) / (f_hi - f_lo), s, zone, ink[y0:y1, x0:x1], D[y0:y1, x0:x1], R, dmin, dmax)
    if t_lo == 0 and t_hi == 0:
        return
    m = (lo + hi) / 2.0
    slope = (t_hi - t_lo) / (f_hi - f_lo)
    fr.offset[side] += t_lo + slope * (m - f_lo)
    fr.tilt[side] = math.degrees(math.atan(math.tan(math.radians(fr.tilt[side])) + slope))
    if fr.offset[side] < 1.0:  # slid off the page at its middle: the page's own edge (offset 0 means that)
        fr.offset[side], fr.tilt[side], fr.bow[side] = 0.0, 0.0, 0.0


def _slide_arc(st: _State, c: tuple, R: int, ink, D) -> None:
    fr, h, w = st.fr, st.h, st.w
    g = Lines(fr, h, w)
    s1, s2 = c
    uc = g.beta["left"] if s2 == "left" else -g.beta["right"]
    vc = g.beta["top"] if s1 == "top" else -g.beta["bottom"]
    du = 1 if s2 == "left" else -1
    dv = 1 if s1 == "top" else -1
    span = fr.arc[c] + 2 * R + 48
    xs, ys = zip(*[g.cv.xy(uc + a * du, vc + b * dv) for a in (-8, span) for b in (-8, span)])
    y0, y1, x0, x1 = box(xs, ys, h, w)
    if y1 <= y0 or x1 <= x0:
        return
    y, x = np.mgrid[y0:y1, x0:x1]
    u, v = g.cv.uv(x, y)
    e1, al1 = e_along(s1, u, v)
    e2, al2 = e_along(s2, u, v)
    a, b = e1 - g.bound(s1, al1), e2 - g.bound(s2, al2)
    ok = (a >= 0) & (b >= 0)
    rho = np.where(ok, a + b + np.sqrt(2 * np.maximum(a, 0) * np.maximum(b, 0)), -1e9)
    zone = ok & st.kept(y0, y1, x0, x1, skip_arc=c)
    ik, dd = _rows(rho - fr.arc[c], zone, ink[y0:y1, x0:x1], D[y0:y1, x0:x1], R)
    t = _choose(ik, dd, R)
    if t:
        fr.arc[c] = max(0.0, fr.arc[c] + t)


def _slide_notch(st: _State, k: int, R: int, ink, D) -> None:
    h, w = st.h, st.w
    for which in ("d", "a0", "a1"):
        n = st.ns[k]
        cv = Canvas.page(n.psi, h, w)
        E, _ = cv.EA(n.side)
        pts = [cv.xy(*uv_of(n.side, e, a)) for a in (n.a0 - R - 2, n.a1 + R + 2) for e in (-E - 2.0, n.d + R + 2)]
        xs, ys = zip(*pts)
        y0, y1, x0, x1 = box(xs, ys, h, w)
        if y1 <= y0 or x1 <= x0:
            return
        y, x = np.mgrid[y0:y1, x0:x1]
        e, al = e_along(n.side, *cv.uv(x, y))
        if which == "d":
            s, zone = e - n.d, (al >= n.a0) & (al < n.a1)
        elif which == "a0":  # the px it alone decides: not past the other side
            s, zone = n.a0 - al, (e < n.d) & (al < n.a1)
        else:
            s, zone = al - n.a1, (e < n.d) & (al >= n.a0)
        zone &= st.kept(y0, y1, x0, x1, skip_notch=k)
        lo = -np.inf if which == "d" else NOTCH_MIN_PX - (n.a1 - n.a0)  # a side never crosses the other one
        t = _choose(*_rows(s, zone, ink[y0:y1, x0:x1], D[y0:y1, x0:x1], R), R, lo)
        if which == "d":
            n.d += t
        elif which == "a0":
            n.a0 -= t
        else:
            n.a1 += t


def _ellipse_s(u, v, a, b):
    """Per px, the m whose ellipse of semi-axes (a + m, b + m) passes through it (bisection): > 0 outside the block's
    ellipse (kept), < 0 inside."""
    lo = np.full(u.shape, -min(a, b) + 0.5)
    hi = np.full(u.shape, 4.0 * max(a, b) + 10)
    for _ in range(40):
        m = (lo + hi) / 2
        inside = (u / (a + m)) ** 2 + (v / (b + m)) ** 2 <= 1.0
        hi = np.where(inside, m, hi)
        lo = np.where(inside, lo, m)
    return (lo + hi) / 2


def _slide_block(st: _State, k: int, R: int, ink, D) -> None:
    h, w = st.h, st.w
    whiches = ("u0", "u1", "v0", "v1") if st.bs[k].kind == "rect" else ("ring",)
    for which in whiches:
        b = st.bs[k]
        y0, y1, x0, x1 = block_box(b, h, w, pad=R + 2)
        if y1 <= y0 or x1 <= x0:
            return
        u, v = block_uv(b, y0, y1, x0, x1)
        if b.kind == "rect":
            u0, u1, v0, v1 = b.p
            inu, inv = (u >= u0) & (u < u1), (v >= v0) & (v < v1)
            s, zone = {"u0": (u0 - u, inv & (u < u1)), "u1": (u - u1, inv & (u >= u0)),
                       "v0": (v0 - v, inu & (v < v1)), "v1": (v - v1, inu & (v >= v0))}[which]
        else:
            s, zone = _ellipse_s(u, v, *b.p), np.ones(u.shape, bool)
        zone = zone & st.kept(y0, y1, x0, x1, skip_block=k)
        lo = -np.inf
        if b.kind == "rect":  # a side never crosses the opposite one (the block keeps at least 1 px)
            lo = 1 - (b.p[1] - b.p[0] if which in ("u0", "u1") else b.p[3] - b.p[2])
        t = _choose(*_rows(s, zone, ink[y0:y1, x0:x1], D[y0:y1, x0:x1], R), R, lo)
        if not t:
            continue
        if b.kind == "rect":
            u0, u1, v0, v1 = b.p
            b.p = {"u0": (u0 - t, u1, v0, v1), "u1": (u0, u1 + t, v0, v1), "v0": (u0, u1, v0 - t, v1),
                   "v1": (u0, u1, v0, v1 + t)}[which]
        else:
            b.p = (max(2.0, b.p[0] + t), max(2.0, b.p[1] + t))


def on_paper(frame: Frame, notches: list[Notch], blocks: list[Block], p: np.ndarray, lum: np.ndarray,
             D: np.ndarray | None = None) -> tuple[Frame, list[Notch], list[Block]]:
    """Each fitted line slid onto paper (see the module): copies of the frame, the notches and the blocks.  `D` is
    `cost_field(p, lum)`, computed when not given."""
    lum = np.asarray(lum, np.float32)
    h, w = lum.shape
    D = cost_field(p, lum) if D is None else np.asarray(D, np.float32)
    ink = 1.0 - lum
    R = reach((h, w))
    st = _State(frame.copy(), [n.copy() for n in notches], [b.copy() for b in blocks], h, w)
    for side in SIDES:
        if st.fr.framed(side):
            _slide_side(st, side, R, ink, D)
    for c in CORNERS:
        if st.fr.framed(c[0]) and st.fr.framed(c[1]):
            _slide_arc(st, c, R, ink, D)
    for k in range(len(st.ns)):
        _slide_notch(st, k, R, ink, D)
    for k in range(len(st.bs)):
        _slide_block(st, k, R, ink, D)
    return st.fr, st.ns, st.bs
