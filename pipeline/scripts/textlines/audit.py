"""The missed-text audit: letter-sized ink pieces a detector gives a chance of text below 0.5, outside every word box,
drawn at random on a contact sheet to be judged by eye.  What it is for: counting how much of the text a detector
misses that the Terra boxes do not cover.  A piece that is plainly a letter or digit is text the detector missed;
a piece that is a stub of line or a smudge is not.

"Letter-sized" is `scripts/spikes/redteam_text_vs_line/redteam_missed.py`'s: after taking out the run rule's line ink
(`linerules.line_px`, and the 3 x 3 px around it), an 8-connected piece of ink with height 8 to 40 px, width 3 to 40
px, 15 or more px and at most 70% of its bounding box, whose centre is more than 10 px from every Terra word box and
every other pass's box.  "Missed": the piece's mean chance is below 0.5 (a chance of 128 or more of 255 is called).
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)

CALLED = 128                       # chance >= 0.5 of 255
FAR = 10                           # px from every box
HEIGHT, WIDTH, MIN_AREA, FILL = (8, 40), (3, 40), 15, 0.7
R = 40                             # a tile shows 2 R px square of the page, at 2 x


def pieces(ink: np.ndarray, boxes: np.ndarray, run_line: np.ndarray) -> list[dict]:
    """The letter-sized pieces of `ink` (see the module) outside `boxes` (bool, the union of every word box) and
    `run_line` (bool, the run rule's px): dicts sheet-less, with the centre (x, y) and the bounding box (x0, y0, w, h)."""
    rest = ink & ~(cv2.dilate(run_line.view(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    far = (cv2.distanceTransform((~boxes).view(np.uint8), cv2.DIST_L2, 3) if boxes.any()
           else np.full(ink.shape, np.inf, np.float32))
    n, lab, st, cen = cv2.connectedComponentsWithStats(rest.view(np.uint8), connectivity=8)
    h, w, a = st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_AREA]
    ok = (h >= HEIGHT[0]) & (h <= HEIGHT[1]) & (w >= WIDTH[0]) & (w <= WIDTH[1]) & (a >= MIN_AREA) & (a <= FILL * h * w)
    ok[0] = False
    cx = np.clip(cen[:, 0].round().astype(int), 0, ink.shape[1] - 1)
    cy = np.clip(cen[:, 1].round().astype(int), 0, ink.shape[0] - 1)
    ok &= far[cy, cx] > FAR
    out = []
    for j in np.flatnonzero(ok):
        x0, y0 = int(st[j, 0]), int(st[j, 1])
        out.append({"x": int(cx[j]), "y": int(cy[j]), "x0": x0, "y0": y0, "w": int(w[j]), "h": int(h[j]),
                    "mask": lab[y0:y0 + h[j], x0:x0 + w[j]] == j})
    return out


def missed(found: list[dict], chance: np.ndarray) -> list[dict]:
    """The pieces whose mean chance over their own px is below 0.5, each with `chance` (the mean, of 255)."""
    out = []
    for p in found:
        c = float(chance[p["y0"]:p["y0"] + p["h"], p["x0"]:p["x0"] + p["w"]][p["mask"]].mean())
        if c < CALLED:
            out.append({k: v for k, v in p.items() if k != "mask"} | {"chance": c})
    return out


def candidates(sheets: list, source: Callable[[str], np.ndarray], quarters: tuple[str, ...], geo) -> tuple[list, int]:
    """Every missed piece (`missed`) of `sheets` (`data.SheetData`) whose centre lies in one of `quarters`, with a
    `sheet` key, and the count of all letter-sized pieces there (missed or not).  `geo.quarter(sd)` gives the quarters."""
    from sfzmaps.solver import paths
    from sfzmaps.solver.textnet import data, linerules
    ix = np.array([paths.QUARTERS.index(q) for q in quarters], np.uint8)
    out, total = [], 0
    for sd in sheets:
        ink = data.page_ink(sd)
        chance = np.asarray(source(sd.sheet))
        if chance.shape != ink.shape:
            raise ValueError(f"the chance of {sd.sheet} is {chance.shape}, not its page's shape {ink.shape}")
        boxes = (data._words(sd, 0, 0, sd.height, sd.width) > 0) | (
            data._paint(sd.other, np.ones(len(sd.other), np.int32), 0, 0, sd.height, sd.width) > 0)
        found = pieces(ink, boxes, linerules.line_px(ink, sd.ft_per_px))
        quarter = geo.quarter(sd)
        found = [p for p in found if np.isin(quarter[p["y"], p["x"]], ix)]
        total += len(found)
        out += [p | {"sheet": sd.sheet} for p in missed(found, chance)]
    return out, total


def draw(cands: list[dict], n: int = 64, seed: int = 0) -> list[dict]:
    """`n` of `cands` at random, without replacement (all of them when fewer), by a generator seeded with `seed`."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cands), min(n, len(cands)), replace=False)
    return [cands[i] for i in sorted(pick)]


def sample(source: Callable[[str], np.ndarray], quarters: tuple[str, ...], n: int = 64, seed: int = 0,
           geo=None, sheets: list | None = None) -> list[dict]:
    """`n` of the missed letter-sized pieces of the 2000 sheets (in `quarters`) at random: each a dict with sheet,
    centre x, y, box x0, y0, w, h and chance.  `geo` (default `CityGeo`) and `sheets` (default every 2000 sheet) can be
    given to share them between calls."""
    from sfzmaps.solver import paths
    from sfzmaps.solver.textnet import data, linegate
    if sheets is None:
        paths.check_sheets()
        sheets = data.load_sheets([paths.SERIES], hold_out_validation=False)
    cands, _ = candidates(sheets, source, quarters, geo or linegate.CityGeo())
    return draw(cands, n, seed)


def sheet(items: list[dict], out: Path, ink_of: Callable[[str], np.ndarray], columns: int = 8) -> Path:
    """The contact sheet of `items` (each drawn on a 2 R px window of its page, at 2 x, the piece boxed in red and
    numbered), `columns` to a row, written to `out`.  `ink_of(sheet)` gives a sheet's page of ink."""
    cache: dict[str, np.ndarray] = {}
    tiles = []
    for i, it in enumerate(items):
        if it["sheet"] not in cache:
            cache[it["sheet"]] = ink_of(it["sheet"])
        ink = cache[it["sheet"]]
        y0, x0 = max(it["y"] - R, 0), max(it["x"] - R, 0)
        win = np.zeros((2 * R, 2 * R), bool)
        sub = ink[y0:it["y"] + R, x0:it["x"] + R]
        win[:sub.shape[0], :sub.shape[1]] = sub
        t = np.full((2 * R, 2 * R, 3), 255, np.uint8)
        t[win] = (150, 150, 150)
        bx, by = it["x0"] - x0, it["y0"] - y0
        cv2.rectangle(t, (bx - 1, by - 1), (bx + it["w"], by + it["h"]), (0, 0, 230), 1)
        t = cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, str(i), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
        tiles.append(np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=0))
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack((tiles[r:r + columns] + [blank] * columns)[:columns]) for r in range(0, len(tiles), columns)]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    return out
