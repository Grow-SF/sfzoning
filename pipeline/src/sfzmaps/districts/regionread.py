"""Reader B, per-region read: a vision call names the district of each region that carries no
label of its own (ruling R26).

The 1990s-2000s sheets draw heavy lines around most blocks, so a district's printed code sits
in one small region and its neighbours, cut off by heavy block outlines that aren't district
boundaries, carry none.  Inheriting a neighbour's code was right only ~47% of the time on
2000-digital-zoning.  Here each unlabelled region of at least MIN_AREA_PX is shown in context
(a window CONTEXT region-widths beyond it each side, at least MIN_WIN_PX), outlined in
magenta, with the candidate codes: the labels inside the window first, then the other codes
read on the sheet.  The model answers a code from that list or "none".  Regions go to the
model MAX_TILES at a time as a numbered mosaic (as in pattern.py); calls are disk-cached by
llm.ask_json.  Inheritance (labels.assign_regions) stays the fallback for regions answered
"none", below MIN_CONF, or not read.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from pydantic import BaseModel
from shapely.geometry import Polygon

from sfzmaps import llm

TILE_PX = 384          # tile edge in the mosaic (the smallest that still reads the codes; see notes)
COLS = 4
MAX_TILES = 12         # 4 x 3 tiles: 1556 x 1222 px, ~2.2k input tokens
GAP = 6
NUM_H = 18
CONTEXT = 1.5          # context beyond the region on each side, in region widths (its bbox's long side)
MIN_WIN_PX = 1100      # the window is at least this many page px (nearby codes stay in view) ...
MAX_WIN_PX = 2400      # ... and at most this (codes stay legible), unless the region is bigger
MIN_AREA_PX = 1500     # smaller regions (letter counters, specks) only inherit ...
MIN_THICK_PX = 12      # ... as do strips thinner than this (2 x area / perimeter): street slivers
MIN_CONF = 0.5         # an answer below this confidence is ignored (inheritance decides)
OUTLINE = (255, 0, 255)  # RGB magenta
OUTLINE_PX = 3
WORKERS = 8
ENABLED = True         # labels.assign_parcels reads unlabelled regions (tests switch it off)


class RegionItem(BaseModel):
    index: int
    code: str
    confidence: float


class RegionReads(BaseModel):
    items: list[RegionItem]


def window(region: Polygon, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Square page-px window (x0, y0, x1, y1) around the region, clipped to the page: its
    bbox plus CONTEXT region widths (sqrt of the area, so long strips don't zoom out) each
    side, within [MIN_WIN_PX, MAX_WIN_PX] unless the region itself is bigger."""
    x0, y0, x1, y1 = region.bounds
    s = max(x1 - x0, y1 - y0)
    side = max(MIN_WIN_PX, min(MAX_WIN_PX, s + 2 * CONTEXT * region.area ** 0.5), 1.1 * s)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = shape[:2]
    return (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
            int(min(w, cx + side / 2)), int(min(h, cy + side / 2)))


def tile(page: np.ndarray, region: Polygon, win: tuple[int, int, int, int],
         px: int = TILE_PX) -> np.ndarray:
    """RGB px x px tile: the window (aspect kept, padded white), the region outlined."""
    x0, y0, x1, y1 = win
    crop = page[y0:y1, x0:x1]
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    s = px / max(crop.shape[:2])
    small = cv2.resize(crop, (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    out = np.full((px, px, 3), 255, np.uint8)
    out[:small.shape[0], :small.shape[1]] = small[:px, :px]
    for ring in [region.exterior, *region.interiors]:
        pts = (np.asarray(ring.coords)[:, :2] - (x0, y0)) * s
        cv2.polylines(out, [np.round(pts).astype(np.int32)], True, OUTLINE, OUTLINE_PX, cv2.LINE_AA)
    return out


def tile_boxes(n: int, px: int = TILE_PX) -> list[tuple[int, int, int, int]]:
    out = []
    for i in range(n):
        r, c = divmod(i, COLS)
        x0 = GAP + c * (px + GAP)
        y0 = r * (px + NUM_H + GAP) + NUM_H
        out.append((x0, y0, x0 + px, y0 + px))
    return out


def build_mosaic(tiles: list[np.ndarray], px: int = TILE_PX) -> np.ndarray:
    """RGB mosaic of numbered tiles (1..n), COLS per row."""
    if len(tiles) > MAX_TILES:
        raise ValueError(f"{len(tiles)} tiles > {MAX_TILES} per mosaic")
    cols = min(COLS, max(1, len(tiles)))
    rows = max(1, -(-len(tiles) // COLS))
    img = np.full((rows * (px + NUM_H + GAP), GAP + cols * (px + GAP), 3), 255, np.uint8)
    for i, (t, (x0, y0, x1, y1)) in enumerate(zip(tiles, tile_boxes(len(tiles), px))):
        img[y0:y1, x0:x1] = t
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x1, y1), (0, 0, 0), 1)
        cv2.putText(img, str(i + 1), (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
    return img


def prompt(cands: list[list[str]], kind: str, sheet_codes: list[str]) -> str:
    what = ("zoning use-district codes (such as R-1, RH-2, RM-1, NC-1, C-2, M-1, P)" if kind == "use"
            else "height-and-bulk district codes (such as 40-X, 65-A, 105-E, OS)")
    lines = "\n".join(f"{i + 1}: {', '.join(c) or '-'}" for i, c in enumerate(cands))
    return (
        f"This image holds {len(cands)} numbered tiles, each cut from a scanned San Francisco "
        f"zoning map that prints {what}. Districts are areas bounded by heavy black lines, with "
        "the district's code printed somewhere inside it, usually in large letters. Thin lines "
        "are block and lot outlines, and some maps also draw heavy lines around single blocks "
        "inside a district. In each tile one area is outlined in magenta. Decide which "
        "district code applies to the area inside the magenta outline: a code printed inside "
        "it; otherwise the code printed in the same district, following the district's heavy "
        "boundary across streets and block outlines (a code printed in the street beside the "
        "area, or on the neighbouring blocks it is not separated from by a district boundary). "
        "Ignore block numbers, street names and lot numbers. Answer with one of the codes "
        f"printed on this sheet ({', '.join(sheet_codes)}); the codes printed in each tile "
        "are listed below, nearest the outlined area first. Answer \"none\" when no code "
        "can be told for the area.\n"
        f"Codes printed in each tile:\n{lines}\n"
        "Return one item per tile: index = the tile number, code = the chosen code or \"none\", "
        "confidence = 0 to 1.")


def _canon(c: str) -> str:
    return "".join(ch for ch in c.upper() if ch.isalnum())


def candidates(win, labs) -> list[str]:
    """Codes of the labels inside the window, nearest the window centre first."""
    x0, y0, x1, y1 = win
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    near = sorted((l for l in labs if x0 <= l.x <= x1 and y0 <= l.y <= y1),
                  key=lambda l: (l.x - cx) ** 2 + (l.y - cy) ** 2)
    out: list[str] = []
    for c in [l.code for l in near]:
        if c not in out:
            out.append(c)
    return out


def read_regions(page: np.ndarray, regs: list[Polygon], todo: list[int], labs, kind: str, *,
                 client: llm.LLMClient | None = None, workers: int = WORKERS
                 ) -> dict[int, tuple[str, float]]:
    """{region index: (code, confidence)} for the regions in `todo` (unlabelled) that the model
    names with confidence >= MIN_CONF.  labs: the sheet's labels of this kind (page px)."""
    if page.ndim == 2:
        page = cv2.cvtColor(page, cv2.COLOR_GRAY2RGB)
    todo = [i for i in todo if regs[i].area >= MIN_AREA_PX
            and 2 * regs[i].area / max(regs[i].length, 1) >= MIN_THICK_PX]
    if not todo or not labs:
        return {}
    freq: dict[str, int] = {}
    for l in labs:
        freq[l.code] = freq.get(l.code, 0) + 1
    sheet_codes = sorted(freq, key=lambda c: -freq[c])
    items = []
    for i in todo:
        win = window(regs[i], page.shape)
        items.append((i, tile(page, regs[i], win), candidates(win, labs)))
    batches = [items[b:b + MAX_TILES] for b in range(0, len(items), MAX_TILES)]
    asks = [(cv2.cvtColor(build_mosaic([t for _, t, _ in b]), cv2.COLOR_RGB2BGR),
             prompt([c for _, _, c in b], kind, sheet_codes)) for b in batches]
    if client is None and not all(llm.is_cached(im, pr, RegionReads) for im, pr in asks):
        client = llm.OpenAIClient()

    def ask(a):
        return llm.ask_json(a[0], a[1], RegionReads, client=client)

    out: dict[int, tuple[str, float]] = {}
    with ThreadPoolExecutor(max(1, min(workers, len(batches)))) as ex:
        for batch, res in zip(batches, ex.map(ask, asks)):
            by = {it.index: it for it in res.items}
            for n, (i, _, cands) in enumerate(batch, 1):
                it = by.get(n)
                if it is None:
                    continue
                code = next((c for c in cands + sheet_codes if _canon(c) == _canon(it.code)), None)
                conf = float(np.clip(it.confidence, 0, 1))
                if code is not None and conf >= MIN_CONF:
                    out[i] = (code, conf)
    return out
