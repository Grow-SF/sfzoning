"""A second pass of the text finder over every sheet (paid, capped by SFZMAPS_MAX_USD), each tile shown another way:
   cw      the tile turned a quarter clockwise (text printed bottom-to-top then reads upright)
   big     tiles of half the size shown at twice the size
   big_cw  both
Same prompt, model and schema as the first pass (textboxes.py).  Boxes are mapped back to page px.
Writes k8/p<nn>/textboxes2_<variant>.json.  usage: textpass.py VARIANT [count]   (count: only say how many calls are not cached)"""
import json
import math
import sys
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel

from lotwalls import K, SERIES, SHEETS, config
from sfzmaps import llm

VAR = sys.argv[1]
llm.VISION_MODEL = "gpt-6.1-terra"
W = config.WORK_DIR / SERIES
BIG = VAR.startswith("big")
TILE, OVER = (384, 80) if BIG else (768, 160)


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


def prompt(w, h):
    return f"""This image is a crop of a scanned 2000 San Francisco zoning map. It is {w} pixels wide and {h} pixels high (origin top-left, x to the right, y down).

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


def show(win):
    if BIG:
        win = cv2.resize(win, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if VAR.endswith("cw"):
        win = cv2.rotate(win, cv2.ROTATE_90_CLOCKWISE)
    return np.ascontiguousarray(win)


def back(it, th, tw):
    """a box in the shown image -> the tile's own px (th, tw: the tile's height and width)"""
    x0, y0, x1, y1 = it.x0, it.y0, it.x1, it.y1
    f = 2 if BIG else 1
    if VAR.endswith("cw"):                      # shown (x, y) = (th*f - 1 - wy, wx)
        x0, y0, x1, y1 = y0, th * f - 1 - x1, y1, th * f - 1 - x0
    return [x0 / f, y0 / f, x1 / f, y1 / f]


jobs = []
for nn in SHEETS:
    page = cv2.imread(str(W / f"{SERIES}-p{nn}" / "page.png"))
    H, Wd = page.shape[:2]
    for ty in starts(H):
        for tx in starts(Wd):
            t = page[ty:ty + TILE, tx:tx + TILE]
            jobs.append((nn, tx, ty, t.shape[0], t.shape[1], show(t)))
todo = sum(not llm.is_cached(j[5], prompt(j[5].shape[1], j[5].shape[0]), TextItems) for j in jobs)
print(f"{VAR}: {len(jobs)} tiles, {todo} not yet read", flush=True)
if len(sys.argv) > 2:
    sys.exit()


def one(j):
    try:
        return llm.ask_json(j[5], prompt(j[5].shape[1], j[5].shape[0]), TextItems)
    except llm.FATAL:
        raise
    except Exception as e:
        print("failed", j[0], j[1], j[2], type(e).__name__, flush=True)
        return None


stopped = False
try:
    out = llm.pmap(one, jobs, 8)
except llm.FATAL as e:
    stopped = True
    print("stopped:", type(e).__name__, str(e)[:120], flush=True)
    out = [llm.ask_json(j[5], prompt(j[5].shape[1], j[5].shape[0]), TextItems) if llm.is_cached(j[5], prompt(j[5].shape[1], j[5].shape[0]), TextItems) else None for j in jobs]
C = llm.CallCounter
print(f"calls: live {C.live}, from cache {C.hits}; tokens in {C.in_tokens:,} out {C.out_tokens:,}; cost ${C.usd():.2f}; tiles without an answer {sum(o is None for o in out)}")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


by = {}
for j, r in zip(jobs, out):
    if r is None:
        continue
    nn, tx, ty, th, tw, _ = j
    for it in r.items:
        b = back(it, th, tw)
        edge = min(b[0], b[1], tw - b[2], th - b[3])
        by.setdefault(nn, []).append({"text": it.text, "kind": it.kind, "direction": it.direction, "angle_deg": it.angle_deg,
                                      "box": [b[0] + tx, b[1] + ty, b[2] + tx, b[3] + ty], "edge": edge, "tile": [tx, ty]})
if not stopped:
    for nn, items in by.items():
        items.sort(key=lambda i: -i["edge"])
        kept = []
        for it in items:
            if not any(iou(it["box"], k["box"]) > 0.3 for k in kept):
                kept.append(it)
        (K / f"p{nn}" / f"textboxes2_{VAR}.json").write_text(json.dumps(kept, indent=0))
        print(nn, len(kept), "boxes;", sum(k["kind"] == "zoning code" for k in kept), "zoning codes")
