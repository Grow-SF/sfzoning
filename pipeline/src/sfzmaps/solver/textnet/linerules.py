"""Line rules for the text detector's measures, and the exact-truth cases that check them.

A rule maps a page's ink and its scale to the ink px it calls line (block, street and district lines).  Each is
scored on drawn cases where the answer is known: a word's glyphs drawn with `cv2.putText`, and lines of known
width drawn through, along or touching them.  `rule_accuracy` gives, per case and text, the share of the line ink
the rule finds (`line_recall`) and the share of glyph ink, more than 1 px from the line, it calls line
(`glyph_false_line`).

`run_lines`, `run_px` and `line_px` copy `codereader.labels`' run rule; it is copied, not imported, so this module
imports nothing from `sfzmaps.solver` (an import would be circular through `textnet.scores`).  `chain_lines` is
measured and is not part of `line_px`: a chain of straight runs follows a letter's strokes as well as a line.
"""
import math

import cv2
import numpy as np

RUN_ANGLES = 32          # the directions a straight run is looked for in, over a half turn
LONG_LINE_FT = 200.0     # `textnet.scores.LONG_LINE_FT`, the run rule's length in feet
FT_PER_PX = 3.98         # the 2000 maps' scale, for the cases
JOIN_PX = 2              # a chain's runs join where their ends are this close


def run_px(ft_per_px: float) -> int:
    """The run rule's length in px: LONG_LINE_FT at the sheet's scale."""
    return round(LONG_LINE_FT / ft_per_px)


def _line_kernel(length: int, angle: float) -> np.ndarray:
    """A one px line `length` px long through the centre of an odd square, at `angle` degrees."""
    k = length | 1
    m = np.zeros((k, k), np.uint8)
    c = k // 2
    dx, dy = math.cos(math.radians(angle)) * (length - 1) / 2, -math.sin(math.radians(angle)) * (length - 1) / 2
    cv2.line(m, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1, 1)
    return m


def run_lines(ink: np.ndarray, length: int) -> np.ndarray:
    """The ink px on a straight run of ink at least `length` px long in one of RUN_ANGLES directions over a half
    turn (an opening by a one px line).  Junctions don't cut a run, and a letter's strokes are shorter than one."""
    u = np.ascontiguousarray(np.asarray(ink, bool)).view(np.uint8)
    out = np.zeros(u.shape, bool)
    for i in range(RUN_ANGLES):
        out |= cv2.morphologyEx(u, cv2.MORPH_OPEN, _line_kernel(length, 180 * i / RUN_ANGLES)) > 0
    return out


def line_px(ink: np.ndarray, ft_per_px: float) -> np.ndarray:
    """Line ink by the run rule (bool), at the sheet's scale."""
    return run_lines(ink, run_px(ft_per_px))


def chain_lines(ink: np.ndarray, length: int, run: int) -> np.ndarray:
    """Ink on chains of straight runs of at least `run` px, joined where their ends meet within JOIN_PX, whose
    end-to-end extent (the chain's bounding diagonal) is at least `length` px.  Measured, not part of `line_px`."""
    runs = run_lines(ink, run)
    grown = cv2.dilate(runs.view(np.uint8), np.ones((JOIN_PX + 1, JOIN_PX + 1), np.uint8))
    _, label, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    extent = np.hypot(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = extent >= length
    keep[0] = False
    return runs & keep[label]


TRUTH_CASES = ("line through the middle", "line along the baseline", "block corner",
               "two street lines, word between", "vertical line through the 2nd letter",
               "stair-step border", "dashed line", "hatching")
SIZE = 400               # the cases' square canvas, px


def _draw_glyphs(text: str, cap_px: int, rng: np.random.Generator) -> np.ndarray:
    """The word in Hershey simplex, `cap_px` tall, near the canvas centre (bool)."""
    face = cv2.FONT_HERSHEY_SIMPLEX
    cap1 = cv2.getTextSize("H", face, 1.0, 1)[0][1]
    scale = cap_px / cap1
    thick = max(1, round(cap_px / 8))
    (w, h), _ = cv2.getTextSize(text, face, scale, thick)
    jx, jy = rng.integers(-15, 16, 2)
    org = (int(SIZE // 2 - w // 2 + jx), int(SIZE // 2 + h // 2 + jy))
    canvas = np.zeros((SIZE, SIZE), np.uint8)
    cv2.putText(canvas, text, org, face, scale, 1, thick, cv2.LINE_8)
    return canvas.view(bool)


def truth_case(name: str, text: str, cap_px: int, seed: int, draw=None) -> dict:
    """One exact-truth case: `ink` (glyph | line), `glyph`, `line` (bool, SIZE x SIZE).  The line is 4 px or 6 px by
    seed.  `draw(text, cap_px, rng)` may supply the glyph mask instead (another font)."""
    if name not in TRUTH_CASES:
        raise ValueError(f"a case is one of {TRUTH_CASES}, not {name!r}")
    rng = np.random.default_rng([seed, cap_px])
    glyph = (draw or _draw_glyphs)(text, cap_px, rng)
    ys, xs = np.nonzero(glyph)
    top, bot, left, right = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    w = (4, 6)[seed % 2]
    mid = (top + bot) // 2
    line = np.zeros((SIZE, SIZE), np.uint8)

    def seg(p, q, width=w):
        cv2.line(line, p, q, 1, width)

    last = SIZE - 1
    if name == "line through the middle":
        seg((0, mid), (last, mid))
    elif name == "line along the baseline":
        seg((0, bot + 1), (last, bot + 1), 4)
    elif name == "block corner":
        seg((0, bot + 2), (right + 3, bot + 2), 4)
        seg((right + 3, 0), (right + 3, bot + 2), 4)
    elif name == "two street lines, word between":
        seg((0, top - 3), (last, top - 3), 4)
        seg((0, bot + 3), (last, bot + 3), 4)
    elif name == "vertical line through the 2nd letter":
        x = left + int(cap_px * 0.9)
        seg((x, 0), (x, last))
    elif name == "stair-step border":
        # lot jogs of 20-40 px: a monotone staircase down, placed so its step at the word's centre column runs
        # through the word's middle
        cx = (left + right) // 2
        steps, x, y = [], 0, 0
        while x < last:
            nx = min(x + int(rng.integers(20, 41)), last)
            ny = y + int(rng.integers(20, 41))
            steps.append((x, nx, y, ny))
            x, y = nx, ny
        y0 = mid - next(a_y for a_x, b_x, a_y, _ in steps if a_x <= cx <= b_x)
        for a_x, b_x, a_y, b_y in steps:
            seg((a_x, a_y + y0), (b_x, a_y + y0))
            seg((b_x, a_y + y0), (b_x, b_y + y0))
    elif name == "dashed line":
        for x in range(0, SIZE, 20):
            seg((x, mid), (min(x + 11, last), mid))          # 12 on, 8 off
    elif name == "hatching":
        for k in range(-SIZE, SIZE, 14):                      # 45 degree lines, 10 px apart (14 along x)
            seg((k, 0), (k + SIZE, SIZE), 4)
    line = line.view(bool)
    return {"ink": glyph | line, "glyph": glyph, "line": line}


def rule_accuracy(rule, texts=("RM-2", "RH-1(D)", "NC-3", "C-3-O", "RM-4", "P", "NC-1", "M-1", "RTO-M"),
                  caps=(20, 27), seeds=range(4), draw=None) -> dict:
    """`rule(ink, ft_per_px)` -> line ink, scored on every truth case: {case: {text: {line_recall,
    glyph_false_line}}}, the means over caps and seeds.  Glyph px within 1 px of the line aren't counted; a case
    with none to count gives None."""
    k = np.ones((3, 3), np.uint8)
    out: dict = {}
    for name in TRUTH_CASES:
        out[name] = {}
        for text in texts:
            rec, fal = [], []
            for cap in caps:
                for seed in seeds:
                    c = truth_case(name, text, cap, seed, draw)
                    found = np.asarray(rule(c["ink"], FT_PER_PX), bool)
                    line = c["line"] & ~c["glyph"]
                    far = c["glyph"] & ~(cv2.dilate(c["line"].view(np.uint8), k) > 0)
                    rec.append((found & line).sum() / max(line.sum(), 1))
                    if far.sum():
                        fal.append((found & far).sum() / far.sum())
            out[name][text] = {"line_recall": float(np.mean(rec)),
                               "glyph_false_line": float(np.mean(fal)) if fal else None}
    return out
