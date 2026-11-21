"""Synthetic map pages with a known frame, for the shape decision's tests (ported from the issue #71 spike's
`make_synth`, scaled down so a page fits in about a second on the cpu).

The truth geometry is written out here independently of `sfzmaps.frame.shape`: the frame is turned by `phi` degrees
about the page centre; with (u, v) the turned coordinates of a pixel's centre, each framed side keeps the pixels whose
inward distance e from its line is >= 0 (top: e = v - beta, bottom: -v - beta, left: u - beta, right: -u - beta).
A side's offset d is its line's inward distance from the page edge's midpoint, so beta = d - cos(phi) * half the page
across that side.  A side given as None has no frame: the map runs to the page's own edge there.

Each framed side has a 5-px rule just outside the kept region and a 15-px bar 10 px further out; the map inside is
20% ink; the chances are the true class with noise.
"""
from __future__ import annotations

import math

import numpy as np

SIDES = ("top", "bottom", "left", "right")
CORNERS = (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"))
_BASE = np.array([[0.95, 0.04, 0.01], [0.04, 0.95, 0.01], [0.08, 0.02, 0.90]], np.float32)


def _turn(phi, X, Y, cx, cy):
    t = math.radians(phi)
    dx, dy = X + 0.5 - cx, Y + 0.5 - cy
    return math.cos(t) * dx + math.sin(t) * dy, -math.sin(t) * dx + math.cos(t) * dy


def make_page(h=900, w=1200, phi=1.2, offset=None, bow_top=0.0, arc=0.0, notch=None, blocks=False,
              blank_strip=0, gap=0, seed=0):
    """(lum, p, truth) of a synthetic page.

    offset: side -> px inward of the page edge (None: no frame on that side); default a frame on all four sides.
    bow_top: the top side's inward bow at its middle (a parabola, zero where it meets the left and right sides).
    arc: the fillet radius at every corner between two framed sides (tangent to both).
    notch: dict(side="bottom", psi, a0, a1, d): a ruled box cut from the kept region at angle psi, along in [a0, a1)
        and inward distance e < d in the psi-turned frame (the spike's form); `ruled=False` leaves out the rule
        along its three inner sides.
    blocks: a ruled legend (rectangle) and a badge (circle) inside, called "removed block".
    blank_strip: a strip this wide just inside the left side with no ink, which the chances call "outside".
    gap: blank paper this wide (called map) just inside every framed side, between the rule and the map's ink.
    truth: phi, offset, beta, bow_top, arc, K (the kept region), cls (0 map, 1 outside, 2 removed block)."""
    rng = np.random.default_rng(seed)
    if offset is None:
        offset = {"top": 80.0, "bottom": 70.0, "left": 90.0, "right": 75.0}
    framed = {s: offset.get(s) is not None for s in SIDES}
    c = math.cos(math.radians(phi))
    half = {"top": h / 2, "bottom": h / 2, "left": w / 2, "right": w / 2}
    beta = {s: offset[s] - c * half[s] for s in SIDES if framed[s]}
    cx, cy = w / 2.0, h / 2.0
    Y, X = np.mgrid[0:h, 0:w]
    u, v = _turn(phi, X, Y, cx, cy)
    inf = np.full((h, w), np.inf)
    dist = {s: inf for s in SIDES}
    if framed["top"]:
        bow = 0.0
        if bow_top:
            lo, hi = beta["left"], -beta["right"]
            m, hw = (lo + hi) / 2, (hi - lo) / 2
            bow = bow_top * (1 - ((u - m) / hw) ** 2)
        dist["top"] = v - (beta["top"] + bow)
    if framed["bottom"]:
        dist["bottom"] = -v - beta["bottom"]
    if framed["left"]:
        dist["left"] = u - beta["left"]
    if framed["right"]:
        dist["right"] = -u - beta["right"]
    inside = np.ones((h, w), bool)
    for d in dist.values():
        inside &= d >= 0
    ink = np.zeros((h, w), np.float32)
    for s in SIDES:
        if not framed[s]:
            continue
        span = np.ones((h, w), bool)
        for o in SIDES:
            if o != s:
                span &= dist[o] >= -30
        d = dist[s]
        ink[span & (d < 0) & (d >= -5)] = 1
        ink[span & (d < -15) & (d >= -30)] = 1
    cut_arc = np.zeros((h, w), bool)
    if arc > 0:
        for s1, s2 in CORNERS:
            if not (framed[s1] and framed[s2]):
                continue
            a, b = dist[s1], dist[s2]
            near = (a >= 0) & (b >= 0) & (a < arc) & (b < arc)
            delta = np.hypot(a - arc, b - arc)
            cut_arc |= near & (delta > arc)
            ink[near & (delta > arc) & (delta < arc + 4)] = 1
    K = inside & ~cut_arc
    if notch:
        un, vn = _turn(notch["psi"], X, Y, cx, cy)
        e, al = {"top": (vn, un), "bottom": (-vn, un), "left": (un, vn), "right": (-un, vn)}[notch["side"]]
        N = (al >= notch["a0"]) & (al < notch["a1"]) & (e < notch["d"]) & K
        strip = N & ((e >= notch["d"] - 4) | (al < notch["a0"] + 4) | (al >= notch["a1"] - 4))
        if not notch.get("ruled", True):  # no rule: the strip gets the box's own 15% ink
            strip = np.zeros((h, w), bool)
        ink[strip] = 1
        ink[N & ~strip & (rng.random((h, w)) < 0.15)] = 1
        K &= ~N
    B = np.zeros((h, w), bool)
    if blocks:
        r = (u >= -410) & (u < -260) & (v >= 190) & (v < 300)
        rb = r & ~((u >= -407) & (u < -263) & (v >= 193) & (v < 297))
        ink[rb] = 1
        ink[r & ~rb & (rng.random((h, w)) < 0.25)] = 1
        e_ = np.hypot(u - 340, v + 260)
        circ = e_ <= 30
        ink[circ & (e_ > 26)] = 1
        ink[circ & (e_ <= 26) & (rng.random((h, w)) < 0.25)] = 1
        B = (r | circ) & K
    strip = np.zeros((h, w), bool)
    if blank_strip:
        strip = K & (dist["left"] < blank_strip)
    if gap:
        for s in SIDES:
            if framed[s]:
                strip |= K & (dist[s] < gap)
    mapink = K & ~B & ~strip
    ink[mapink] = (rng.random((h, w)) < 0.2)[mapink]
    cls = np.where(K & ~B, 0, 1).astype(np.uint8)
    cls[B] = 2
    called = cls.copy()
    if blank_strip:
        called[K & (dist["left"] < blank_strip)] = 1
    p = _BASE[called].transpose(2, 0, 1) * rng.uniform(0.8, 1.2, (3, h, w)).astype(np.float32)
    p /= p.sum(0, keepdims=True)
    lum = 1.0 - ink
    truth = dict(phi=phi, offset={s: (offset[s] if framed[s] else 0.0) for s in SIDES}, beta=beta,
                 bow_top=bow_top, arc=arc, K=K, cls=cls)
    return lum.astype(np.float32), p.astype(np.float32), truth
