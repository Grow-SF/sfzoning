"""Pages, hand-drawn map-area labels and their masks.

A label file (`inputs/frame-labels/<series>/<sheet>.json`, see its README; the test set's, never trained on, in
`inputs/frame-labels-test/`, the same format) has `polygons`, the map area, and may have
`remove`, the printed blocks inside the frame that are deleted wherever they sit (legend and title blocks, compass
roses, "SECTION N" badges).  `class_mask` turns both into the layout's three answers per px: 0 map, 1 outside the
frame, 2 removed block (`layout.CLASSES`)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sfzmaps.config import DATA_DIR, INPUTS_DIR

SheetId = tuple[str, str]  # (series, sheet)

LABELS = INPUTS_DIR / "frame-labels"
TEST_LABELS = INPUTS_DIR / "frame-labels-test"
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


def _label_file(series: str, sheet: str, root: Path) -> dict:
    return json.loads((root / series / f"{sheet}.json").read_text())


def _polys(data: dict, key: str) -> list[Polygon]:
    return [[(float(x), float(y)) for x, y in poly] for poly in data.get(key, [])]


def load_label(series: str, sheet: str, root: Path = LABELS) -> list[Polygon]:
    """A sheet's label polygons, in page px (x, y)."""
    label = _label_file(series, sheet, root)
    if "holes" in label:
        raise ValueError(f"{root / series / f'{sheet}.json'} has `holes`; labels have no holes (use `remove`)")
    return _polys(label, "polygons")


def load_remove(series: str, sheet: str, root: Path = LABELS) -> list[Polygon]:
    """A sheet's removed blocks (the file's `remove`), in page px (x, y); empty when it has none."""
    return _polys(_label_file(series, sheet, root), "remove")


def labelled_sheets(root: Path = LABELS) -> list[SheetId]:
    return sorted((p.parent.name, p.stem) for p in root.glob("*/*.json"))


def test_sheets(root: Path | None = None) -> list[SheetId]:
    """The test set's sheets (`TEST_LABELS` by default): scored, never trained on."""
    return labelled_sheets(TEST_LABELS if root is None else root)


test_sheets.__test__ = False   # a data function, not a test


def check_split(train, test) -> None:
    """Raise ValueError naming every sheet that is both a training sheet (`LABELS`) and a test sheet (`TEST_LABELS`):
    a test sheet must never be trained on."""
    both = sorted(set(map(tuple, train)) & set(map(tuple, test)))
    if both:
        names = ", ".join(f"{s}/{h}" for s, h in both)
        raise ValueError(f"{len(both)} sheet(s) labelled both for training ({LABELS}) and for testing ({TEST_LABELS}): "
                         f"{names}; a test sheet is never trained on: remove one of the two labels")


def map_mask(polygons, shape: tuple[int, int]) -> np.ndarray:
    """Uint8 mask, 1 inside any polygon (outlines included), in page px."""
    img = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        draw.polygon([(x, y) for x, y in poly], fill=1, outline=1)
    return np.asarray(img, dtype=np.uint8)


def class_mask(polygons, remove, shape: tuple[int, int]) -> np.ndarray:
    """Uint8 class per px: 1 (outside the frame) everywhere, then 0 (map) inside any of `polygons`, then 2 (removed
    block) inside any of `remove`, outlines included each time; `remove` is drawn last, so it wins where they overlap."""
    img = Image.new("L", (shape[1], shape[0]), 1)
    draw = ImageDraw.Draw(img)
    for poly in polygons:
        draw.polygon(list(poly), fill=0, outline=0)
    for block in remove:
        draw.polygon(list(block), fill=2, outline=2)
    return np.asarray(img, dtype=np.uint8)


def map_counts(kinds_csv: Path = PAGE_KINDS) -> dict[SheetId, int]:
    """How many maps each page holds, from the column `maps` of the page-kinds file; blank means 1."""
    with open(kinds_csv, newline="") as f:
        return {(r["series"], r["sheet"]): int(r["maps"] or 1) for r in csv.DictReader(f)}


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
