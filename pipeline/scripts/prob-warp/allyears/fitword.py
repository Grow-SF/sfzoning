"""Fit a known word into Terra's box, whole word (the owner's approach A).

The letters are the pen font's strokes, kept continuous at any size (every page pixel they pass).
Two layouts:
  straight: the word at Terra's angle (as before); searched: shift, stretch, height, skew (+ angle if
    slanted).
  street:   the word's baseline follows the street centreline through the box (page px polyline), each
    stroke pixel placed at its own distance along the line, so a name bends with the road; searched:
    position along the line, offset across it, stretch, height; no skew.
Start: the word spans Terra's box, its height the box's short side.  Coordinate descent: step each
setting, keep any step that raises the score, halve the steps when a pass finds none.

Score (the one measure, used to place the fit and to judge it):
  on  = share of the letters' stroke pixels on ink   (stroke on white is penalised)
  cov = share of the ink in Terra's box within a pen radius of a stroke   (dark is rewarded)
  score = on + cov"""
import functools
import math
import os

import cv2

cv2.setNumThreads(0)      # OpenCV's own threads deadlock inside forked worker processes
import numpy as np
import shapely

import textfit

FS = 4.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
_th = cv2.getTextSize("H", FONT, FS, 1)[0][1]
_im = np.zeros((_th + 60, _th * 2), np.uint8)
cv2.putText(_im, "H", (20, _th + 20), FONT, FS, 255, 1, cv2.LINE_8)
TH0 = int(np.ptp(np.nonzero(_im)[0])) + 1          # cap height: the drawn H's own height, px at FS


@functools.lru_cache(maxsize=None)
def glyph(ch):
    """One letter's stroke pixels at font size FS (cap height TH0 px; baseline y=0, left x=0), and its
    advance.  Dense: at page sizes several of these fall in each page pixel, so strokes stay unbroken."""
    ch = textfit.GLYPH.get(ch, ch)
    (tw, th), base = cv2.getTextSize(ch, FONT, FS, 1)
    img = np.zeros((TH0 + base + 40, tw + 40), np.uint8)
    cv2.putText(img, ch, (20, TH0 + 20), FONT, FS, 255, 1, cv2.LINE_8)
    ys, xs = np.nonzero(img)
    return np.c_[xs - 20, ys - (TH0 + 20)].astype(float), float(tw)


def word(text, p):
    """The word's stroke points in its own frame: x along (px), y down from the text's middle (px),
    centred along; and each point's letter index.  (speed E5: the same word at the same size, stretch and slant is
    built once; the arrays are read-only, so a cached copy can't be changed by a caller)"""
    return _word(text, p["h"], p["sx"], p.get("it", 0.0))


@functools.lru_cache(maxsize=4096)
def _word(text, h, sx, it):
    p = {"h": h, "sx": sx, "it": it}
    P, L = _word_build(text, p)
    P.setflags(write=False)
    L.setflags(write=False)
    return P, L


def _word_build(text, p):
    k = p["h"] / TH0
    pts, lis, x = [], [], 0.0
    for i, ch in enumerate(text):
        g, adv = glyph(ch)
        pts.append(np.c_[(g[:, 0] + x) * k * p["sx"], g[:, 1] * k + p["h"] / 2])
        lis.append(np.full(len(g), i))
        x += adv
    P, L = np.vstack(pts), np.concatenate(lis)
    P[:, 0] -= P[:, 1] * math.tan(math.radians(p.get("it", 0.0)))
    P[:, 0] -= (P[:, 0].min() + P[:, 0].max()) / 2
    return P, L


def render(text, p, curve=None):
    """Stroke points in page px (straight layout, or along `curve` = (pts, cumulative s, tangents))."""
    P, L = word(text, p)
    if curve is None:
        a = math.radians(p["ang"])
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return P @ R.T @ textfit.SKEW.T + [p["cx"], p["cy"]], L, np.full(len(P), p["ang"])
    C, S, T = curve
    s = np.clip(p["s0"] + P[:, 0], S[0], S[-1])
    i = np.clip(np.searchsorted(S, s) - 1, 0, len(S) - 2)
    f = ((s - S[i]) / np.maximum(1e-9, S[i + 1] - S[i]))[:, None]
    base = C[i] * (1 - f) + C[i + 1] * f
    t = T[i]
    up = np.c_[t[:, 1], -t[:, 0]]                       # left of the reading direction (image y down)
    return base + up * (p["d"] - P[:, 1:2]), L, np.degrees(np.arctan2(t[:, 1], t[:, 0]))


def pixels(P):
    """The page pixels a stroke passes (each once)."""
    return np.unique(np.round(P).astype(int), axis=0)


# white bridging two pieces of ink: the kth px of a run's length costs 0.2 (k = 1), then (k + 1)^2 / 10
WALL_COST = 3.0      # a stroke pixel on or past a wall running along the text costs this much (a miss costs 0, a hit gains 1)
# trials, off unless set: A = white runs measured in stroke widths; B = walls ignored inside Terra's box
WHITE_UNIT = os.environ.get("FITWORD_WHITE_UNIT", "px")
WALL_INSIDE = os.environ.get("FITWORD_WALL_INSIDE", "keep")
PW_FROM = os.environ.get("FITWORD_PW", "box")


def _clip_inside(segs, box, across):
    """Trial B: the wall segments with the part crossing the inside of Terra's box removed (the box shrunk by
    a fifth of its short side each way: a wall on the box's edge stays).  Terra's box is tight to the letters,
    so a line through its inside is not the edge of the label's strip."""
    if not len(segs):
        return segs
    a, b, c, e = box
    s_ = max(2.0, 0.2 * across)
    if c - a <= 2 * s_ or e - b <= 2 * s_:
        return segs
    inner = shapely.box(a + s_, b + s_, c - s_, e - s_)
    out = []
    for x0_, y0_, x1_, y1_ in segs:
        g = shapely.LineString([(x0_, y0_), (x1_, y1_)]).difference(inner)
        for p in getattr(g, "geoms", [g]):
            if not p.is_empty and p.length > 0.5:
                (u0, v0), (u1, v1) = p.coords[0], p.coords[-1]
                out.append([u0, v0, u1, v1])
    return np.asarray(out, float).reshape(-1, 4)


def _beyond(Pabs, walls, mid):
    """Which points lie past a wall: on the far side of a wall segment from the label's middle, within that
    segment's own length (or on it).  walls: (M, 4) segments x0, y0, x1, y1 (same px frame)."""
    if walls is None or not len(walls):
        return np.zeros(len(Pabs), bool)
    p0, p1 = walls[:, :2], walls[:, 2:]
    dv = p1 - p0
    L2 = np.maximum(1e-9, (dv ** 2).sum(1))
    rel = Pabs[:, None, :] - p0[None]
    tpar = (rel * dv[None]).sum(2) / L2[None]
    side = dv[None, :, 0] * rel[:, :, 1] - dv[None, :, 1] * rel[:, :, 0]
    rm = np.asarray(mid, float) - p0
    mside = dv[:, 0] * rm[:, 1] - dv[:, 1] * rm[:, 0]
    dist = np.abs(side) / np.sqrt(L2)[None]
    past = (np.sign(side) != np.sign(mside)[None]) | (dist < 0.5)
    return ((tpar >= 0) & (tpar <= 1) & past & (np.abs(mside)[None] > 1e-6)).any(1)


def _crop(box, shape):
    """score_pts' crop round the box: (x0, y0, x1, y1)."""
    a, b, c, e = box
    H, W = shape
    pad = int(max(c - a, e - b) * 0.3) + 10
    return max(0, a - pad), max(0, b - pad), min(W, c + pad), min(H, e + pad)


def past_mask(walls, mid, box, shape, rows=16):
    """For every pixel of score_pts' crop: is it past a wall (as _beyond)?  Built a few rows at a time so
    memory stays small; depends only on the label (walls, middle, box), not on the fit."""
    x0, y0, x1, y1 = _crop(box, shape)
    m = np.zeros((y1 - y0, x1 - x0), bool)
    if walls is None or not len(walls):
        return m
    xs = np.arange(x0, x1)
    rows = max(1, int(2_000_000 // max(1, (x1 - x0) * len(walls))))   # each block: about 2M pixel-wall pairs
    for r0 in range(0, y1 - y0, rows):
        yy = np.arange(y0 + r0, min(y1, y0 + r0 + rows))
        gx, gy = np.meshgrid(xs, yy)
        m[r0:r0 + len(yy)] = _beyond(np.c_[gx.ravel(), gy.ravel()].astype(float), walls, mid).reshape(len(yy), len(xs))
    return m


def _unique_rows(q):
    """np.unique(q, axis=0) for an (n, 2) int array, fast (speed E5): each row packed into one int64 key whose order
    is the rows' order (x first, then y), the keys made unique and unpacked.  Same rows, same order."""
    if len(q) == 0:
        return np.unique(q, axis=0)
    xmin, ymin = q[:, 0].min(), q[:, 1].min()
    span = int(q[:, 1].max() - ymin) + 1
    k = np.unique((q[:, 0] - xmin) * span + (q[:, 1] - ymin))
    return np.stack([k // span + xmin, k % span + ymin], axis=1)


def score_pts(P, ink, box, pw, D=None, lines=None):
    """(score, on, cov) of stroke points P on the ink image, against the ink inside box [a,b,c,e].
    lines = (walls, mid): the block-outline segments running along the text near the label, and the label's
    middle.  A stroke pixel past a wall (far side from the middle, within the wall's length) costs WALL_COST.
    on = (stroke px on ink, not past a wall - WALL_COST x stroke px past a wall - the stacked cost of white
    runs) / stroke px.  cov = share of the box's ink (not past a wall) within a pen radius of a stroke."""
    a, b, c, e = box
    H, W = ink.shape
    pad = int(max(c - a, e - b) * 0.3) + 10
    x0, y0, x1, y1 = max(0, a - pad), max(0, b - pad), min(W, c + pad), min(H, e + pad)
    I = ink[y0:y1, x0:x1] > 0
    q = _unique_rows(np.round(P).astype(int)) - [x0, y0]
    inside = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < x1 - x0) & (q[:, 1] < y1 - y0)
    qi = q[inside]
    hit = I[qi[:, 1], qi[:, 0]]
    bi = np.zeros(I.shape, bool)
    bi[max(0, b - y0):e - y0 + 1, max(0, a - x0):c - x0 + 1] = True
    box_ink = bi & I
    if lines is not None and len(lines) > 2 and lines[2] is not None:
        pm = lines[2]                                   # past-a-wall area over this crop, computed once
        out = pm[qi[:, 1], qi[:, 0]]
        on = ((hit & ~out).sum() - WALL_COST * out.sum() - _bridging(qi, hit, I, pw)) / max(1, len(q))
        box_ink &= ~pm
    else:
        on = (hit.sum() - _bridging(qi, hit, I, pw)) / max(1, len(q))
    S = np.zeros(I.shape, np.uint8)
    S[qi[:, 1], qi[:, 0]] = 1
    r = max(1, int(round(pw / 2)))
    S = cv2.dilate(S, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    cov = (S & box_ink).sum() / max(1, box_ink.sum())
    return on + cov, float(on), float(cov)


def _bridging(qi, hit, I, pw):
    """Cost of the stroke's white: the stroke's white pixels, grouped into connected runs (any run, whether
    or not it touches ink).  A run's length along the stroke = its pixels / the stroke width; its cost is the
    sum over its length of the stacked per-px cost (_stack_cost), over the stroke's width."""
    S = np.zeros(I.shape, np.uint8)
    S[qi[:, 1], qi[:, 0]] = 1
    white = (S > 0) & ~I
    if not white.any():
        return 0.0
    nw, lw = cv2.connectedComponents(white.astype(np.uint8), connectivity=8)
    width = max(1.0, pw)
    sizes = np.bincount(lw.ravel())[1:]
    if WHITE_UNIT == "grow":
        # trial A': a run's length counted in units that start at 1 px on thin text and grow with the stroke,
        # more slowly than it: unit = max(1, stroke width / 3)  (2 px stroke -> 1 px, 8 px -> 2.7 px)
        u = max(1.0, width / 3.0)
        return float(sum(_stack_cost(max(1, int(round(n_ / width / u)))) * width * u for n_ in sizes))
    if WHITE_UNIT == "stroke":
        # trial A: a run's length counted in stroke widths, not px; the stacked cost is per stroke width of
        # length, over that stretch of stroke (width x width px), so the same gap costs the same at any size
        return float(sum(_stack_cost(max(1, int(round(n_ / width / width)))) * width * width for n_ in sizes))
    return float(sum(_stack_cost(max(1, int(round(n_ / width)))) * width for n_ in sizes))


@functools.lru_cache(maxsize=None)
def _stack_cost(n):
    """Total cost of a bridging run n px long: the 1st px 0.2, then the kth px (k >= 2) (k + 1)^2 / 10:
    0.2, 0.9, 1.6, 2.5, 3.6, 4.9, ..."""
    return sum(0.2 if k == 1 else (k + 1) ** 2 / 10 for k in range(1, n + 1))


def make_curve(line_px, k, extend=200.0):
    """The centreline (page px polyline) resampled every 0.5 px and oriented in the reading direction
    Terra gives: left-to-right / slanted -> x increasing; bottom-to-top -> y decreasing; top-to-bottom ->
    y increasing.  Returns (pts, cumulative s, unit tangents) or None."""
    L = np.asarray(line_px, float)
    if len(L) < 2:
        return None
    seg = np.linalg.norm(np.diff(L, axis=0), axis=1)
    S = np.r_[0, np.cumsum(seg)]
    if S[-1] < 2:
        return None
    s = np.arange(0, S[-1], 0.5)
    C = np.c_[np.interp(s, S, L[:, 0]), np.interp(s, S, L[:, 1])]
    d = C[-1] - C[0]
    dr = k.get("direction", "left-to-right")
    flip = (d[1] > 0) if dr == "bottom-to-top" else (d[1] < 0) if dr == "top-to-bottom" else (d[0] < 0)
    if flip:
        C = C[::-1]
    T = np.diff(C, axis=0)
    T = T / np.maximum(1e-9, np.linalg.norm(T, axis=1))[:, None]
    T = np.vstack([T, T[-1:]])
    ext = np.arange(0.5, extend + 0.5, 0.5)[:, None]          # the line runs on straight past both ends
    C = np.vstack([C[0] - T[0] * ext[::-1], C, C[-1] + T[-1] * ext])
    T = np.vstack([np.repeat(T[:1], len(ext), 0), T, np.repeat(T[-1:], len(ext), 0)])
    S = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
    return C, S, T


def _pw(p, pw):
    """The pen width the score uses: from Terra's box (pw), or, with FITWORD_PW=height, from the drawn
    letters' own height (0.2 x h, at least 2 px): a diagonal word's box is far bigger than its letters."""
    return max(2.0, 0.2 * p["h"]) if PW_FROM == "height" else pw


def _descend(text, p, steps, ink, box, pw, curve, limits, lines=None, min_frac=0.1):
    R_ = render(text, p, curve)
    best = score_pts(R_[0], ink, box, _pw(p, pw), R_[2], lines)
    st = dict(steps)
    for _ in range(400):
        if not any(st[n] >= steps[n] * min_frac for n in st):
            break
        moved = False
        for n in st:
            if st[n] < steps[n] * min_frac:
                continue
            for sgn in (1, -1):
                q = dict(p)
                q[n] += sgn * st[n]
                if not limits(q):
                    continue
                R_ = render(text, q, curve)
                s = score_pts(R_[0], ink, box, _pw(q, pw), R_[2], lines)
                if s[0] > best[0] + 1e-9:
                    p, best, moved = q, s, True
                    break
        if not moved:
            st = {n: v / 2 for n, v in st.items()}
    return p, best


def _near_walls(segs, box, text_deg):
    """The wall segments near the box (within its long side) running within 45 deg of the text."""
    a, b, c, e = box
    m = max(c - a, e - b)
    s = np.asarray(segs, float)
    mx, my = (s[:, 0] + s[:, 2]) / 2, (s[:, 1] + s[:, 3]) / 2
    near = (np.maximum(s[:, 0], s[:, 2]) >= a - m) & (np.minimum(s[:, 0], s[:, 2]) <= c + m) & \
           (np.maximum(s[:, 1], s[:, 3]) >= b - m) & (np.minimum(s[:, 1], s[:, 3]) <= e + m)
    s = s[near]
    if not len(s):
        return s
    dd = np.abs((np.degrees(np.arctan2(s[:, 3] - s[:, 1], s[:, 2] - s[:, 0])) - text_deg) % 180)
    return s[np.minimum(dd, 180 - dd) < 45]


def _heights(across):
    """Starting letter heights: every whole px from half the box's short side to the full side (min 4)."""
    return list(range(max(4, int(round(across / 2))), max(4, int(round(across))) + 1))


def fit(text, k, ink, pw, line_px=None, lines=None):
    """Best whole-word fit of `text` in Terra's label k (k['box'] in ink's px).  line_px: the street
    centreline through the box (page px, same frame), for street names.  Returns a dict or None."""
    if not text:
        return None
    a, b, c, e = k["box"]
    bw, bh = c - a, e - b
    slanted = k.get("direction") == "slanted" and 5 <= abs(k["angle_deg"]) <= 85
    across = (k.get("_kind_h") or min(bw, bh) * 0.5) if slanted else min(bw, bh)
    nat = sum(glyph(ch)[1] for ch in text) * across / TH0
    lim = lambda q: q["h"] >= 4 and q["sx"] > 0.2
    curve = make_curve(line_px, k) if line_px is not None else None
    if curve is not None:
        C, S, T = curve
        corners = np.array([[a, b], [c, b], [c, e], [a, e]], float)
        # where the box sits on the line: each corner's nearest point along it
        j = [int(np.argmin(np.linalg.norm(C - q, axis=1))) for q in corners]
        jc = int(np.argmin(np.linalg.norm(C - [(a + c) / 2, (b + e) / 2], axis=1)))
        th = math.atan2(T[jc, 1], T[jc, 0])
        span = S[max(j)] - S[min(j)]
        along = max(across, span - across * abs(math.sin(2 * th)))
        up = np.array([T[jc, 1], -T[jc, 0]])
        d0 = float(np.dot([(a + c) / 2 - C[jc, 0], (b + e) / 2 - C[jc, 1]], up))
        starts = [dict(s0=float(S[jc]), d=d0, h=float(h0), sx=along / max(1.0, nat * h0 / across)) for h0 in _heights(across)]
        lines_ = None
        if lines is not None:
            wl_ = _near_walls(lines, (a, b, c, e), math.degrees(th))
            if WALL_INSIDE == "ignore":
                wl_ = _clip_inside(wl_, (a, b, c, e), across)
            lines_ = (wl_, tuple(C[jc]), past_mask(wl_, tuple(C[jc]), (a, b, c, e), ink.shape))   # middle: on the street centreline
        p, best = max((_descend(text, p0, {"s0": 2.0, "d": 2.0, "sx": 0.1, "h": 2.0}, ink, (a, b, c, e), pw, curve, lim, lines_)
                       for p0 in starts), key=lambda r: r[1][0])
        layout, ang = "street", math.degrees(th)
    else:
        if slanted:
            th = math.radians(abs(k["angle_deg"]))
            cs, sn = math.cos(th), math.sin(th)
            along = (bw - across * sn) / cs if cs >= sn else (bh - across * cs) / sn
            along = max(along, across)
            ang = -k["angle_deg"]
        else:
            vertical = (bh > bw * 1.3 and len(text) > 1) or abs(k["angle_deg"]) >= 45 \
                or k.get("direction") in ("bottom-to-top", "top-to-bottom")
            ang = (90.0 if k.get("direction") == "top-to-bottom" else -90.0) if vertical else 0.0
            along = bh if vertical else bw
        starts = [dict(cx=(a + c) / 2, cy=(b + e) / 2, h=float(h0), ang=ang, it=0.0, sx=along / max(1.0, nat * h0 / across)) for h0 in _heights(across)]
        steps = {"cx": 2.0, "cy": 2.0, "sx": 0.1, "h": 2.0, "it": 6.0, **({"ang": 3.0} if slanted else {})}
        lines_ = None
        if lines is not None:
            wl_ = _near_walls(lines, (a, b, c, e), ang)
            if WALL_INSIDE == "ignore":
                wl_ = _clip_inside(wl_, (a, b, c, e), across)
            mid_ = ((a + c) / 2, (b + e) / 2)
            lines_ = (wl_, mid_, past_mask(wl_, mid_, (a, b, c, e), ink.shape))
        p, best = max((_descend(text, p0, steps, ink, (a, b, c, e), pw, None, lim, lines_) for p0 in starts),
                      key=lambda r: r[1][0])
        layout, ang = "straight", p["ang"]
    P, L, _ = render(text, p, curve)
    return {"pts": P, "li": L, "score": best[0], "on": best[1], "cov": best[2], "h": p["h"], "sx": p["sx"],
            "angle": ang, "italic": p.get("it", 0.0), "across": across, "pw": _pw(p, pw), "layout": layout,
            **({"curve": curve[0][::4].round(1).tolist()} if curve is not None else {})}
