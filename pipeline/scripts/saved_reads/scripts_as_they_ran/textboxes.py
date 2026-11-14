"""Every piece of text in a scan window, at full resolution, from the vision model: its text, kind,
a tight box and reading direction.  Draws the boxes on the window.

usage: textboxes.py MODEL SERIES SHEET x0,y0,x1,y1 OUTDIR
"""
import json
import math
import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel

from sfzmaps import config, llm

model, series, sheet, box, outdir = sys.argv[1:6]
llm.VISION_MODEL = model
x0, y0, x1, y1 = map(int, box.split(","))
O = Path(outdir)
O.mkdir(parents=True, exist_ok=True)
import os  # noqa: E402
# TB_PAGE: read a different image of the sheet (e.g. the 4a image, block lines deleted); same px frame
page = cv2.imread(os.environ.get("TB_PAGE") or str(config.WORK_DIR / series / sheet / "page.png"))
win = page[y0:y1, x0:x1]
# read a margin round the window too, so a word crossing its border is seen whole
MARGIN = 200
bx0, by0 = max(0, x0 - MARGIN), max(0, y0 - MARGIN)
bx1, by1 = min(page.shape[1], x1 + MARGIN), min(page.shape[0], y1 + MARGIN)
big = page[by0:by1, bx0:bx1]
H, W = big.shape[:2]
TILE, OVER = 768, 160


class TextItem(BaseModel):
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


from sfzmaps.districts import read as _read  # noqa: E402
MAPYEAR = int(_read.series_year(series))


def prompt(w, h):
    return f"""This image is a crop of a scanned {MAPYEAR} San Francisco zoning map. It is {w} pixels wide and {h} pixels high (origin top-left, x to the right, y down).

List EVERY piece of printed text in it, including text that is partly cut off at the image edge:
- zoning codes (e.g. RH-1, RH-2, RM-1, NC-1, NC-3, P, C-M, "VALENCIA NCD"), written large inside districts, sometimes vertically or touching lines;
- block numbers (4-digit numbers like 3604, small, inside blocks);
- street names and their "ST." / "AVE." (small capitals along streets, sometimes vertical);
- anything else (kind "other").

For each one give:
- text: exactly as printed, letter for letter. One item per whole label: all the words printed together as one name on one line are ONE item with one box, e.g. "24TH-NOE VALLEY NCD", "VALENCIA NCD", "CASTRO NCD", "TWENTY-FIRST ST." when ST. follows the name on the same line. A street's "ST." printed elsewhere, away from its name, is its own item;
- kind;
- x0, y0, x1, y1: the tightest box around the printed letters themselves, in pixels of this image; not around nearby lines;
- direction: the way the text reads;
- angle_deg: angle of the text's baseline counterclockwise from horizontal (0 = normal, 90 = reads bottom to top, -90 = reads top to bottom).

Do not invent text. Do not include lines, arrows or leader lines."""


def starts(n):
    k = max(1, math.ceil((n - OVER) / (TILE - OVER)))
    return sorted({int(round(v)) for v in np.linspace(0, max(0, n - TILE), k)})


tiles = []
for ty in starts(H):
    for tx in starts(W):
        tiles.append((tx, ty, np.ascontiguousarray(big[ty:ty + TILE, tx:tx + TILE])))
client = None if all(llm.is_cached(t[2], prompt(t[2].shape[1], t[2].shape[0]), TextItems) for t in tiles) else llm.OpenAIClient()
res = llm.pmap(lambda t: llm.ask_json(t[2], prompt(t[2].shape[1], t[2].shape[0]), TextItems, client=client), tiles, 4)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


items = []
for (tx, ty, t), r in zip(tiles, res):
    th, tw = t.shape[:2]
    for it in r.items:
        bx = [it.x0 + tx, it.y0 + ty, it.x1 + tx, it.y1 + ty]
        edge = min(it.x0, it.y0, tw - it.x1, th - it.y1)       # distance from the tile edge
        items.append({**it.model_dump(), "box": bx, "edge": edge, "tile": [tx, ty]})
items.sort(key=lambda i: -i["edge"])
def inside(a, b):
    """Share of box a's area inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1, (a[2] - a[0]) * (a[3] - a[1]))


kept = []
for it in items:
    if any(iou(it["box"], k["box"]) > 0.3 for k in kept):
        continue
    kept.append(it)
# a word whose box lies inside a longer label that contains it is part of that label
kept = [k for k in kept if not any(o is not k and len(o["text"]) > len(k["text"]) and k["text"].replace(" ", "") in o["text"].replace(" ", "")
                                   and inside(k["box"], o["box"]) > 0.7 for o in kept)]
# to window coordinates; keep what overlaps the window
ox, oy = x0 - bx0, y0 - by0
for k in kept:
    k["box"] = [k["box"][0] - ox, k["box"][1] - oy, k["box"][2] - ox, k["box"][3] - oy]
    k["tile"] = [k["tile"][0] - ox, k["tile"][1] - oy]
kept = [k for k in kept if k["box"][2] > 0 and k["box"][3] > 0 and k["box"][0] < x1 - x0 and k["box"][1] < y1 - y0]
(O / f"textboxes_{model}.json").write_text(json.dumps(kept, indent=0))

COL = {"zoning code": (0, 0, 230), "block number": (0, 140, 0), "street name": (200, 0, 160), "other": (0, 150, 220)}
img = cv2.addWeighted(win, 0.75, np.full_like(win, 255), 0.25, 0)
for k in kept:
    a, b, c, d = k["box"]
    col = COL[k["kind"]]
    cv2.rectangle(img, (a, b), (c, d), col, 2)
    # reading direction: an arrow along the baseline through the box centre
    cx, cy, ang = (a + c) / 2, (b + d) / 2, math.radians(k["angle_deg"])
    L = max(8, 0.35 * max(c - a, d - b))
    p0 = (int(cx - L * math.cos(ang)), int(cy + L * math.sin(ang)))
    p1 = (int(cx + L * math.cos(ang)), int(cy - L * math.sin(ang)))
    cv2.arrowedLine(img, p0, p1, col, 1, tipLength=0.25)
    cv2.putText(img, k["text"], (a, max(10, b - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
cv2.imwrite(str(O / f"s2a_{model}.png"), img)
by = {}
for k in kept:
    by[k["kind"]] = by.get(k["kind"], 0) + 1
print(model, len(tiles), "tiles;", by, f"live calls {llm.CallCounter.live}, metered ~${llm.CallCounter.usd():.2f} (at Sol prices)")
