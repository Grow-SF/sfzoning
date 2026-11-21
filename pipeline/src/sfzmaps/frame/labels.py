"""Check of the hand-drawn frame labels: rank the ink lying under each outline.

A label outline must lie on paper, except where map ink truly touches the frame rule (design spec, measurement 2 and
part 5).  Ink under an outline therefore marks a place to look at, and a person fixes the label there.  The outline is
cut into stretches of `STRETCH_PX` px and listed by the ink under it, highest first; `qa_sheet` draws a crop per stretch.
Nothing here decides anything: it only orders what a person looks at.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString

STRETCH_PX = 50   # a presentation unit for listing; it decides nothing
CROP_MARGIN = 40  # px of paper shown around a stretch in its crop
MAGENTA = (255, 0, 255)


@dataclass(frozen=True)
class Stretch:
    series: str
    sheet: str
    start: tuple[int, int]  # (x, y) page px
    end: tuple[int, int]
    ink: float              # sum of (1 - lum) over the stretch's px


def _ring_pixels(poly) -> list[tuple[int, int]]:
    """The closed ring sampled at 1 px steps and rounded to px, consecutive repeats removed."""
    ring = LineString(list(poly) + [poly[0]])
    pts = [ring.interpolate(float(d)) for d in range(int(ring.length) + 1)]
    px: list[tuple[int, int]] = []
    for p in pts + [ring.interpolate(ring.length)]:
        q = (int(round(p.x)), int(round(p.y)))
        if not px or px[-1] != q:
            px.append(q)
    return px


def outline_stretches(lum: np.ndarray, polygons, series: str, sheet: str) -> list[Stretch]:
    """Every polygon's outline in runs of `STRETCH_PX` px, highest ink first.  Px on the page's outermost row or
    column are dropped: a frameless side follows the page edge, and the edge is not a frame rule."""
    h, w = lum.shape
    out: list[Stretch] = []
    for poly in polygons:
        runs: list[list[tuple[int, int]]] = [[]]
        for x, y in _ring_pixels(poly):
            if 0 < x < w - 1 and 0 < y < h - 1:
                runs[-1].append((x, y))
            elif runs[-1]:
                runs.append([])
        for run in runs:
            for i in range(0, len(run), STRETCH_PX):
                part = run[i:i + STRETCH_PX]
                ink = float(sum(1.0 - lum[y, x] for x, y in part))
                out.append(Stretch(series, sheet, part[0], part[-1], ink))
    return sorted(out, key=lambda s: -s.ink)


def qa_sheet(lum: np.ndarray, polygons, stretches, path: Path) -> Path:
    """One crop per stretch (its bounding box plus `CROP_MARGIN` px each way, at the page's own resolution) with the
    outline drawn 1 px in magenta and the stretch's ink as a caption, stacked vertically, saved at `path`."""
    h, w = lum.shape
    base = Image.fromarray((np.clip(lum, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    tiles = []
    for s in stretches:
        x0 = max(0, min(s.start[0], s.end[0]) - CROP_MARGIN)
        x1 = min(w, max(s.start[0], s.end[0]) + CROP_MARGIN + 1)
        y0 = max(0, min(s.start[1], s.end[1]) - CROP_MARGIN)
        y1 = min(h, max(s.start[1], s.end[1]) + CROP_MARGIN + 1)
        tile = base.crop((x0, y0, x1, y1))
        d = ImageDraw.Draw(tile)
        for poly in polygons:
            d.line([(x - x0, y - y0) for x, y in list(poly) + [poly[0]]], fill=MAGENTA, width=1)
        tiles.append((tile, f"{s.series}/{s.sheet}  ({s.start[0]},{s.start[1]})-({s.end[0]},{s.end[1]})  ink {s.ink:.1f}"))
    cap = 14
    width = max([t.width for t, _ in tiles] + [200])
    sheet = Image.new("RGB", (width, sum(t.height + cap for t, _ in tiles) or 1), (255, 255, 255))
    y = 0
    d = ImageDraw.Draw(sheet)
    for tile, text in tiles:
        d.text((2, y + 1), text, fill=(0, 0, 0))
        sheet.paste(tile, (0, y + cap))
        y += tile.height + cap
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path
