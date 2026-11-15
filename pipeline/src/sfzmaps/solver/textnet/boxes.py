"""Terra's text boxes on any sheet it read, and the sheets it read, from its saved answers (`text-on-every-sheet`).

Terra was shown each sheet's page in tiles (its script: `saved-reads/scripts_as_they_ran/textboxes.py`); each saved
answer is one tile (`page_box`, page px) of one sheet (`series`, `sheet`: the sheet's folder in data/work/<series>),
with every piece of text Terra saw in it, its box in the tile's px.  `terra_boxes` decodes a sheet's tiles into page
px as that script did; the prints stage (`prints.terra_boxes`) and the text detector both read them through it.  Nothing
here asks a model.
"""
import re
from typing import Literal

import numpy as np
from pydantic import BaseModel

from sfzmaps.solver import paths, saved

TERRA_READ = "text-on-every-sheet"
SAME_TEXT_IOU = 0.3       # a box overlapping a kept box by more is the same text, read again from another tile
PART_OF_LABEL = 0.7       # a word with more of its box than this inside a longer label holding it is part of it


# The answer format, copied from the script that asked.  A docstring would enter the JSON schema (as its
# "description") and so the cache key: these classes keep the script's names and fields, and no docstrings.
class TextItem(BaseModel):    # one piece of text in Terra's answer
    text: str
    kind: Literal["zoning code", "block number", "street name", "other"]
    x0: int
    y0: int
    x1: int
    y1: int
    direction: Literal["left-to-right", "bottom-to-top", "top-to-bottom", "upside-down", "slanted"]
    angle_deg: int


class TextItems(BaseModel):
    items: list[TextItem]


def iou(a, b) -> float:
    """Intersection over union of two boxes (x0, y0, x1, y1)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def terra_records() -> list[dict]:
    """Every saved answer of TERRA_READ, in file order."""
    return saved.answers(TERRA_READ)


def terra_boxes(series: str, sheet: str, records: list[dict] | None = None) -> list[dict]:
    """A sheet's Terra text boxes, every kind, in page px: each tile's items moved by the tile's `page_box`, sorted by
    distance from their tile's edge (farthest first; tiles in the order they were read, row by row), a box
    overlapping a kept one by IoU > SAME_TEXT_IOU dropped as the same text, a word inside a longer label that holds
    it dropped, and boxes outside the window the script was asked about dropped.  Each item: Terra's own fields
    (x0..y1 in its tile), `box` (page px), `edge` and `tile` (its tile's top-left corner).  `records`: the lines of
    TERRA_READ (read from disk when None); only those of `series` and `sheet` are used."""
    recs = [r for r in (terra_records() if records is None else records)
            if r.get("series") == series and r.get("sheet") == sheet]
    if not recs:
        return []
    windows = {r["script"]["args"][3] for r in recs}
    if len(windows) != 1:
        raise ValueError(f"{series} {sheet}: Terra's tiles come from {len(windows)} windows; one run per sheet is "
                         "expected")
    wx0, wy0, wx1, wy1 = map(int, windows.pop().split(","))
    items = []
    for r in sorted(recs, key=lambda r: (r["page_box"][1], r["page_box"][0])):
        tx, ty, ex, ey = r["page_box"]
        tw, th = ex - tx, ey - ty
        for it in TextItems.model_validate(r["answer"]).items:
            items.append({**it.model_dump(), "box": [it.x0 + tx, it.y0 + ty, it.x1 + tx, it.y1 + ty],
                          "edge": min(it.x0, it.y0, tw - it.x1, th - it.y1), "tile": [tx, ty]})
    items.sort(key=lambda i: -i["edge"])
    # one box's overlap with many at once, for the IoU (`iou`'s arithmetic) and the share of the box inside another
    # (its overlap / max(1, its area)), in float64: exact on these integer px
    every = np.array([it["box"] for it in items], float).reshape(-1, 4)
    area = (every[:, 2] - every[:, 0]) * (every[:, 3] - every[:, 1])

    def overlap(a: int, others: np.ndarray) -> np.ndarray:
        b = every[others]
        ix = np.maximum(0, np.minimum(every[a, 2], b[:, 2]) - np.maximum(every[a, 0], b[:, 0]))
        iy = np.maximum(0, np.minimum(every[a, 3], b[:, 3]) - np.maximum(every[a, 1], b[:, 1]))
        return ix * iy

    kept: list[int] = []
    for a in range(len(items)):
        i = overlap(a, np.array(kept, int))
        u = area[a] + area[kept] - i
        if not (np.where(u > 0, i / np.where(u > 0, u, 1), 0) > SAME_TEXT_IOU).any():
            kept.append(a)
    plain = [items[a]["text"].replace(" ", "") for a in kept]
    whole = np.array(kept, int)
    dropped = set()
    for n, a in enumerate(kept):
        share = overlap(a, whole) / max(1, area[a])
        if any(m != n and len(items[kept[m]]["text"]) > len(items[a]["text"]) and plain[n] in plain[m]
               for m in np.flatnonzero(share > PART_OF_LABEL)):
            dropped.add(a)
    out = [items[a] for a in kept if a not in dropped]
    return [k for k in out if k["box"][2] > wx0 and k["box"][3] > wy0 and k["box"][0] < wx1 and k["box"][1] < wy1]


def sheet_order(name: str) -> tuple:
    """The order of a series' sheets: by name, the numbers in it compared as numbers ("2" before "10")."""
    return tuple((1, int(t), "") if t.isdigit() else (0, 0, t) for t in re.split(r"(\d+)", name) if t)


def text_sheets(records: list[dict] | None = None) -> dict[str, tuple[str, ...]]:
    """The sheets Terra read, by series (in the order the series first appear), each once, in `sheet_order`.
    `records`: the lines of TERRA_READ (read from disk when None)."""
    found: dict[str, set[str]] = {}
    for r in terra_records() if records is None else records:
        found.setdefault(r["series"], set()).add(r["sheet"])
    return {s: tuple(sorted(names, key=sheet_order)) for s, names in found.items()}


def check_text_sheets(records: list[dict] | None = None) -> None:
    """Stop (ValueError) unless `paths.TEXT_SHEETS` names exactly the sheets Terra read (`text_sheets`): every series
    and every sheet, in order.  The fixed lists in code then read every sheet with boxes, and none without."""
    found, listed = text_sheets(records), paths.TEXT_SHEETS
    problems = []
    for s in [*listed, *(s for s in found if s not in listed)]:
        have, want = listed.get(s, ()), found.get(s, ())
        if have == want:
            continue
        extra, missing = [n for n in have if n not in want], [n for n in want if n not in have]
        if s not in found:
            problems.append(f"{s}: listed, but Terra read no sheet of it")
        elif s not in listed:
            problems.append(f"{s}: Terra read {len(want)} sheets of it, none listed")
        elif extra or missing:
            problems.append(f"{s}: listed but not read: {', '.join(extra) or 'none'}; read but not listed: "
                            f"{', '.join(missing) or 'none'}")
        else:
            problems.append(f"{s}: listed out of order (`sheet_order`)")
    if problems:
        raise ValueError(f"sfzmaps.solver.paths.TEXT_SHEETS differs from the sheets of the saved answers of "
                         f"{TERRA_READ}: {'; '.join(problems)}.  Change TEXT_SHEETS and `sfzmaps build "
                         "--write-dvc-yaml`")
