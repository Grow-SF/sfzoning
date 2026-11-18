"""The picture panel: fixed windows of the 2000 sheets, a row per window and a column per detector.  Ink the detector
calls text (chance >= 0.5) is red, other ink grey.  What it is for: seeing at a glance whether a detector colours line
ink (block, street and district lines, dashes, hatching) red, and whether the letters stay red.  The same windows
every time, so two detectors' panels can be set side by side.

Windows are (sheet, row, col, size): the page's row and column of the window's CENTRE and its side, in page px.  The
three rm2 windows are the red team's RM-2 zooms; the rest were chosen once by looking at the scans:
  downtown     p01, C-3-S / M-1 / SLI labels over a diagonal street grid
  residential  p02, RH-2 and UNION NCD, thick district outlines through small blocks
  dashed       p02, the dashed pier-head line across the water
  hatched      p01, lots filled with hatching and small lettering
  sw_dense     p12, curving hill streets with RH-1(D) lettering
  sw_street    p12, a regular block grid with a thick district line and lot numbers
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "1")
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

P = "2000-digital-zoning-p"
WINDOWS: dict[str, tuple[str, int, int, int]] = {
    "rm2a": (f"{P}02", 1850, 2505, 240),
    "rm2b": (f"{P}02", 1745, 2175, 240),
    "rm2c": (f"{P}02", 1868, 2218, 240),
    "downtown": (f"{P}01", 3330, 1730, 240),
    "residential": (f"{P}02", 1750, 1680, 240),
    "dashed": (f"{P}02", 890, 2050, 240),
    "hatched": (f"{P}01", 3430, 1690, 240),
    "sw_dense": (f"{P}12", 1500, 1470, 240),
    "sw_street": (f"{P}12", 2770, 1170, 240),
}
CALLED = 128                      # chance >= 0.5 of 255
RED, GREY, PAPER, OUTSIDE = (0, 0, 255), (150, 150, 150), (255, 255, 255), (225, 225, 225)
ZOOM = 2                          # each window px is drawn ZOOM x ZOOM px
HEAD = 22                         # px of column heading
SIDE = 96                         # px of row names


def page_ink(sheet: str) -> np.ndarray:
    """A sheet's whole page of ink, bool."""
    from sfzmaps.solver import paths
    from sfzmaps.solver.textnet import data
    return data.page_ink(data.sheet_data(paths.SERIES, sheet, hold_out_validation=False, placement=False))


def _window(a: np.ndarray, row: int, col: int, size: int, fill) -> np.ndarray:
    """`a`'s `size` x `size` window centred on (row, col), `fill` outside the page."""
    h = size // 2
    out = np.full((size, size), fill, a.dtype)
    y0, x0 = max(row - h, 0), max(col - h, 0)
    sub = a[y0:row + h, x0:col + h]
    oy, ox = y0 - (row - h), x0 - (col - h)
    out[oy:oy + sub.shape[0], ox:ox + sub.shape[1]] = sub
    return out


def _tile(ink: np.ndarray, chance: np.ndarray, row: int, col: int, size: int) -> np.ndarray:
    i = _window(ink, row, col, size, False)
    c = _window(chance, row, col, size, 0)
    inside = _window(np.ones(ink.shape, bool), row, col, size, False)
    t = np.empty((size, size, 3), np.uint8)
    t[:] = OUTSIDE
    t[inside] = PAPER
    t[i] = GREY
    t[i & (c >= CALLED)] = RED
    return cv2.resize(t, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)


def render(sources: dict[str, Callable[[str], np.ndarray]], out: Path, title: str) -> Path:
    """Write the panel to `out`: a row per window of WINDOWS, a column per source (its name above), `title` on top.
    Red: ink with a chance of text >= 0.5; grey: other ink."""
    out = Path(out)
    ink_cache: dict[str, np.ndarray] = {}
    rows = []
    cell = 0
    for name, (sheet, row, col, size) in WINDOWS.items():
        if sheet not in ink_cache:
            ink_cache[sheet] = page_ink(sheet)
        tiles = [np.pad(_tile(ink_cache[sheet], np.asarray(src(sheet)), row, col, size),
                        ((2, 2), (2, 2), (0, 0)), constant_values=60) for src in sources.values()]
        cell = tiles[0].shape[1]
        label = np.full((tiles[0].shape[0], SIDE, 3), 255, np.uint8)
        for k, (text, shade) in enumerate([(name, 0), (sheet[-3:], 90), (f"{row},{col}", 90), (f"{size}px", 90)]):
            cv2.putText(label, text, (4, 18 + 17 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45 if k == 0 else 0.4,
                        (shade,) * 3, 1)
        rows.append(np.hstack([label, *tiles]))
    width = rows[0].shape[1]
    head = np.full((HEAD, width, 3), 255, np.uint8)
    for k, n in enumerate(sources):
        cv2.putText(head, n, (SIDE + k * cell + 4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    top = np.full((26, width, 3), 255, np.uint8)
    cv2.putText(top, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([top, head, *rows]))
    return out
