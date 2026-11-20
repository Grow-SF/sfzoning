"""Pages, hand-drawn map-area labels and their masks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sfzmaps.config import DATA_DIR, INPUTS_DIR

SheetId = tuple[str, str]  # (series, sheet)

LABELS = INPUTS_DIR / "frame-labels"
PAGE_KINDS = INPUTS_DIR / "page_kinds.csv"
WORK = DATA_DIR / "work"
OUT = DATA_DIR / "frame"


def load_luminance(path: Path) -> np.ndarray:
    """Float32 in [0, 1], 1 = paper. Grey is value/255; colour is Rec. 601 luma."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def ink_weight(lum: np.ndarray) -> np.ndarray:
    return 1.0 - lum


Polygon = list[tuple[float, float]]


def load_label(series: str, sheet: str, root: Path = LABELS) -> tuple[list[Polygon], list[Polygon]]:
    """(polygons, holes) of a sheet's label, in page px; `holes` (blocks printed over the map that are not the city,
    see `inputs/frame-labels/README.md` rule 4) is empty when the file has none."""
    data = json.loads((root / series / f"{sheet}.json").read_text())

    def polys(key: str) -> list[Polygon]:
        return [[(float(x), float(y)) for x, y in poly] for poly in data.get(key, [])]
    return polys("polygons"), polys("holes")


def labelled_sheets(root: Path = LABELS) -> list[SheetId]:
    return sorted((p.parent.name, p.stem) for p in root.glob("*/*.json"))


def map_mask(polygons, shape: tuple[int, int], scale: float = 1.0, holes=()) -> np.ndarray:
    """Uint8 mask, 1 inside any polygon, then 0 inside any hole (outlines included both times).  Polygons and holes
    are in page px, drawn times `scale`."""
    img = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        draw.polygon([(x * scale, y * scale) for x, y in poly], fill=1, outline=1)
    for hole in holes:
        draw.polygon([(x * scale, y * scale) for x, y in hole], fill=0, outline=0)
    return np.asarray(img, dtype=np.uint8)


EDGE_MAX = 0.5      # the most `position_planes`' edge plane says (half the shorter side: the middle of a page)


def position_planes(shape: tuple[int, int],
                    window: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(valid, edge), float32: what the frame finder's nets read beside the page to know where the scan is (issue #71
    Task 14; a net gets absolute position almost only from its zero padding, which the paper padding hides).  `valid`
    is 1 on the page and 0 off it; `edge` is each page px's distance to the nearest edge of the page, 0 on the edge
    px themselves (in px between px centres), as a fraction of the shorter side's span (`min(h, w) - 1` px, so the
    middle of an odd shorter side is exactly 1/2), at most EDGE_MAX, and 0 off the page.  Both are scalar fields of
    the page's own shape, so a quarter turn or mirror flip of the planes is the planes of the moved page.

    With `window` (top, left, height, width), the planes over that window of the page's frame (it may leave the page),
    without computing the whole page's."""
    h, w = shape
    top, left, wh, ww = window if window is not None else (0, 0, h, w)
    rows, cols = np.arange(top, top + wh), np.arange(left, left + ww)
    dist = np.minimum(np.minimum(rows, h - 1 - rows)[:, None], np.minimum(cols, w - 1 - cols)[None])
    valid = dist >= 0
    edge = np.where(valid, np.minimum(dist / max(min(h, w) - 1, 1), EDGE_MAX), 0.0)
    return valid.astype(np.float32), edge.astype(np.float32)


def page_path(series: str, sheet: str, work: Path = WORK) -> Path:
    return work / series / sheet / "page.png"


def map_pages(kinds_csv: Path = PAGE_KINDS, work: Path = WORK) -> list[SheetId]:
    """Every sheet under `work` whose kind is `map`; KeyError if a page has no row."""
    with open(kinds_csv, newline="") as f:
        kinds = {(r["series"], r["sheet"]): r["kind"] for r in csv.DictReader(f)}
    pages = sorted((p.parent.parent.name, p.parent.name) for p in work.glob("*/*/page.png"))
    missing = [f"{s}/{h}" for s, h in pages if (s, h) not in kinds]
    if missing:
        raise KeyError(f"pages with no row in {kinds_csv}: {', '.join(missing)}")
    return [sid for sid in pages if kinds[sid] == "map"]
