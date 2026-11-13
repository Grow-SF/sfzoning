"""Fit known text to the scan: render each string as single strokes (Hershey), find the position,
size, angle, slant and width that put those strokes on the ink, refine letter by letter, and say
which ink pixels the fitted letters cover.

Score: mean distance (px) from the rendered stroke points to the nearest ink (chamfer); lower is
better.  Search over translation is one cv2.matchTemplate per (height, angle, slant, width).
"""
import math

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
CLIP = 8.0                  # distances clipped here, so one stray letter doesn't dominate


GLYPH = {"1": "I"}          # the map's 1 is a plain vertical stroke; the font's I is exactly that
SKEW = np.eye(2)            # the sheet's local page distortion (aspect, shear), set per window


def _points(text, italic, h, sx, angle_deg, step=1.5):
    """Stroke points of `text` (cap height h px, width factor sx, slanted `italic` degrees to the
    right, rotated so the baseline runs at angle_deg in image coords, then distorted by the
    sheet's SKEW), centred on (0, 0); also each point's letter index."""
    fs = 4.0
    shown = "".join(GLYPH.get(ch, ch) for ch in text)
    (tw, th), base = cv2.getTextSize(shown, FONT, fs, 1)
    img = np.zeros((th + base + 40, tw + 40), np.uint8)
    lab = np.zeros_like(img, np.int16)
    x = 20
    for i, ch in enumerate(shown):
        tmp = np.zeros_like(img)
        cv2.putText(tmp, ch, (x, th + 20), FONT, fs, 255, 1, cv2.LINE_8)
        lab[tmp > 0] = i + 1
        img |= tmp
        x += cv2.getTextSize(ch, FONT, fs, 1)[0][0]
    ys, xs = np.nonzero(img)
    if not len(xs):
        return np.zeros((0, 2)), np.zeros(0, int)
    k = h / th
    p = np.c_[(xs - xs.mean()) * k * sx, (ys - ys.mean()) * k]
    p[:, 0] -= p[:, 1] * math.tan(math.radians(float(italic)))     # lean right (y grows downward)
    li = lab[ys, xs] - 1
    # thin to roughly one point per `step` px
    q = np.round(p / step).astype(int)
    _, keep = np.unique(q, axis=0, return_index=True)
    p, li = p[keep], li[keep]
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return p @ R.T @ SKEW.T, li


def _best_shift(dist, pts, cx, cy, radius):
    """Best translation of pts (centred) near (cx, cy): (mean distance, x, y)."""
    H, W = dist.shape
    mn = np.floor(pts.min(0)).astype(int)
    mx = np.ceil(pts.max(0)).astype(int)
    tw, th = mx - mn + 1
    tmpl = np.zeros((th, tw), np.float32)
    q = np.round(pts - mn).astype(int)
    tmpl[q[:, 1], q[:, 0]] = 1
    n = tmpl.sum()
    x0 = int(cx + mn[0] - radius); y0 = int(cy + mn[1] - radius)
    x1 = int(cx + mx[0] + radius + 1); y1 = int(cy + mx[1] + radius + 1)
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H or x1 - x0 < tw or y1 - y0 < th:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 < tw or y1 - y0 < th:
            return math.inf, cx, cy
    res = cv2.matchTemplate(dist[y0:y1, x0:x1], tmpl, cv2.TM_CCORR)
    _, _, loc, _ = cv2.minMaxLoc(res)      # lowest summed distance
    v = res[loc[1], loc[0]] / n
    return float(v), x0 + loc[0] - mn[0], y0 + loc[1] - mn[1]


def _best_ncc(ink, pts, pw, cx, cy, radius):
    """Best translation near (cx, cy) of the letters drawn with pen width pw, by normalised
    correlation with the ink over the letters' box plus a margin of paper: strokes must be on
    ink AND the paper around and between them white.  (ncc, x, y)."""
    H, W = ink.shape
    m = int(math.ceil(pw / 2 + 3))
    mn = np.floor(pts.min(0)).astype(int) - m
    mx = np.ceil(pts.max(0)).astype(int) + m
    tw, th = mx - mn + 1
    tmpl = np.zeros((th, tw), np.uint8)
    for x, y in np.round(pts - mn).astype(int):
        cv2.circle(tmpl, (int(x), int(y)), max(0, int(round(pw / 2))), 1, -1)
    if tmpl.min() == tmpl.max():
        return -1.0, cx, cy
    x0 = max(0, int(cx + mn[0] - radius)); y0 = max(0, int(cy + mn[1] - radius))
    x1 = min(W, int(cx + mx[0] + radius + 1)); y1 = min(H, int(cy + mx[1] + radius + 1))
    if x1 - x0 < tw or y1 - y0 < th:
        return -1.0, cx, cy
    res = cv2.matchTemplate(ink[y0:y1, x0:x1], tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, v, _, loc = cv2.minMaxLoc(res)
    return float(v), x0 + loc[0] - mn[0], y0 + loc[1] - mn[1]


def fit(ink, text, cx, cy, *, heights, angles, radius, italics=(True, False), widths=(0.8, 1.0, 1.25),
        pens=(0.1, 0.16, 0.22)):
    """Best fit of `text` near (cx, cy) on the float ink image (1 = ink)."""
    best = None
    for h in heights:
        for a in angles:
            for it in italics:
                for sx in widths:
                    pts, li = _points(text, it, h, sx, a)
                    if not len(pts):
                        continue
                    for pf in pens:
                        pw = max(1.0, h * pf)
                        v, x, y = _best_ncc(ink, pts, pw, cx, cy, radius)
                        if best is None or v > best["score"]:
                            best = {"text": text, "score": v, "x": x, "y": y, "h": float(h), "angle": a,
                                    "italic": it, "sx": sx, "pw": pw}
    if best is None:
        return None
    b = best
    for h in np.linspace(b["h"] * 0.92, b["h"] * 1.08, 5):
        for a in (b["angle"] - 3, b["angle"], b["angle"] + 3):
            for sx in np.linspace(b["sx"] * 0.9, b["sx"] * 1.1, 3):
                pts, li = _points(text, b["italic"], h, sx, a)
                v, x, y = _best_ncc(ink, pts, b["pw"], b["x"], b["y"], 4)
                if v > best["score"]:
                    best = {**best, "score": v, "x": x, "y": y, "h": float(h), "angle": a, "sx": float(sx)}
    # letter by letter: each letter may move a few px
    pts, li = _points(text, best["italic"], best["h"], best["sx"], best["angle"])
    pts = pts + [best["x"], best["y"]]
    out, per = [], []
    for i in range(len(text)):
        m = li == i
        if not m.any():
            continue
        c = pts[m].mean(0)
        v, x, y = _best_ncc(ink, pts[m] - c, best["pw"], c[0], c[1], 4)
        out.append(pts[m] - c + [x, y] if v > -1 else pts[m])
        per.append(round(v, 2))
    best["pts"] = np.concatenate(out) if out else pts
    best["letters"] = per
    return best


ONES = "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE".split()
ORD = {1: "FIRST", 2: "SECOND", 3: "THIRD", 4: "FOURTH", 5: "FIFTH", 6: "SIXTH", 7: "SEVENTH", 8: "EIGHTH",
       9: "NINTH", 10: "TENTH", 11: "ELEVENTH", 12: "TWELFTH", 13: "THIRTEENTH", 14: "FOURTEENTH",
       15: "FIFTEENTH", 16: "SIXTEENTH", 17: "SEVENTEENTH", 18: "EIGHTEENTH", 19: "NINETEENTH"}
TENS = {2: "TWENTY", 3: "THIRTY", 4: "FORTY"}
TENS_ORD = {2: "TWENTIETH", 3: "THIRTIETH", 4: "FORTIETH"}


def street_spellings(street):
    """How the map might print a street's name: as recorded, and numbered streets spelled out."""
    s = str(street).upper().strip()
    out = [s]
    import re
    m = re.fullmatch(r"(\d+)(ST|ND|RD|TH)", s)
    if m:
        n = int(m.group(1))
        if n in ORD:
            out.append(ORD[n])
        elif n % 10 == 0 and n // 10 in TENS_ORD:
            out.append(TENS_ORD[n // 10])
        elif n // 10 in TENS:
            out.append(f"{TENS[n // 10]}-{ORD[n % 10]}")
        out.append(m.group(1) + m.group(2).lower())
    return out
