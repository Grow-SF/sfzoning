"""Pilot (paid, capped by SFZMAPS_MAX_USD): at table districts where no print was found, run the text finder again on one
window around the district, shown five ways: as it is, turned a quarter each way, and at twice the size (upright and turned).
Reports which ways find a zoning code on the district.  The table is used only to pick the places and to score.
usage: rotpilot.py FIRST LAST OUT_JSON"""
import json
import sys
from typing import Literal

import cv2
import numpy as np
import shapely
from pydantic import BaseModel

import lotwalls
import truth
import yard
from lotcode_lib import canon
from lotwalls import SERIES, SHEETS, SheetTransform, config
from sfzmaps import llm

llm.VISION_MODEL = "gpt-6.1-terra"
RES = lotwalls.RES
W = config.WORK_DIR / SERIES
B = lotwalls.build("city5")
g, lot, grown, pairs, ins = truth.load(B)
code, _ = yard.truth_codes(g)
bl = g["blklot"].astype(str).values
cen = shapely.get_coordinates(shapely.centroid(g.geometry.values))
DP = np.load(truth.D / "distprint.npz")
dist, cls, wr, d_top = DP["dist"], DP["cls"].astype(str), DP["wr"], DP["d_top"]
nd = d_top.size
wd = np.bincount(dist[wr & ~d_top[dist]], minlength=nd)
order = np.argsort(-wd)


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


sheets = {}
for nn in SHEETS:
    d = W / f"{SERIES}-p{nn}"
    sheets[nn] = (SheetTransform.from_json(json.loads((d / "transform.json").read_text())),
                  shapely.union_all(shapely.get_parts(shapely.from_geojson((d / "warped.footprint.geojson").read_text()))))
pages = {}


def page(nn):
    if nn not in pages:
        pages[nn] = cv2.imread(str(W / f"{SERIES}-p{nn}" / "page.png"))
    return pages[nn]


def back(variant, x, y, n):
    """a point in the shown image -> the 768 (or 384) window, px"""
    if variant in ("cw", "big_cw"):       # shown = window turned clockwise: shown (x, y) = (n-1-wy, wx)
        x, y = y, n - 1 - x
    elif variant == "ccw":                # shown (x, y) = (wy, n-1-wx)
        x, y = n - 1 - y, x
    if variant.startswith("big"):
        x, y = x / 2, y / 2
    return x, y


jobs = []
for rank in range(int(sys.argv[1]), int(sys.argv[2])):
    d = order[rank]
    m = dist == d
    c = cen[m].mean(0)
    best, pick = -1, None
    for nn, (t, drawn) in sheets.items():
        if not shapely.contains(drawn, shapely.Point(c)):
            continue
        dd = shapely.distance(shapely.boundary(drawn), shapely.Point(c))
        if dd > best:
            best, pick = dd, nn
    if pick is None:
        continue
    t = sheets[pick][0]
    px, py = t.to_pixel(c[None])[0]
    pg = page(pick)
    for variant in ("asis", "cw", "ccw", "big", "big_cw"):
        half = 192 if variant.startswith("big") else 384
        x0, y0 = int(np.clip(px - half, 0, pg.shape[1] - 2 * half)), int(np.clip(py - half, 0, pg.shape[0] - 2 * half))
        win = pg[y0:y0 + 2 * half, x0:x0 + 2 * half]
        if variant.startswith("big"):
            win = cv2.resize(win, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        if variant in ("cw", "big_cw"):
            win = cv2.rotate(win, cv2.ROTATE_90_CLOCKWISE)
        elif variant == "ccw":
            win = cv2.rotate(win, cv2.ROTATE_90_COUNTERCLOCKWISE)
        jobs.append((rank, int(d), pick, variant, x0, y0, np.ascontiguousarray(win)))


def one(j):
    try:
        return llm.ask_json(j[6], prompt(j[6].shape[1], j[6].shape[0]), TextItems)
    except llm.FATAL:
        raise
    except Exception as e:
        print("failed", j[0], j[3], type(e).__name__, flush=True)
        return None


try:
    out = llm.pmap(one, jobs, 8)
except llm.FATAL as e:
    print("stopped:", type(e).__name__, str(e)[:120])
    out = [llm.ask_json(j[6], prompt(j[6].shape[1], j[6].shape[0]), TextItems) if llm.is_cached(j[6], prompt(j[6].shape[1], j[6].shape[0]), TextItems) else None for j in jobs]
C = llm.CallCounter
print(f"calls: live {C.live}, from cache {C.hits}; tokens in {C.in_tokens:,} out {C.out_tokens:,}; cost ${C.usd():.2f}")
res = []
hits = {}
for j, a in zip(jobs, out):
    rank, d, nn, variant, x0, y0, win = j
    if a is None:
        continue
    t = sheets[nn][0]
    found, anyc = [], []
    for it in a.items:
        if it.kind != "zoning code":
            continue
        wx, wy = back(variant, (it.x0 + it.x1) / 2, (it.y0 + it.y1) / 2, win.shape[0])
        wc = t.to_world(np.array([[x0 + wx, y0 + wy]], float))[0]
        sc, sr = int(round((wc[0] - B["x0"]) / RES)), int(round((wc[1] - B["y0"]) / RES))
        on = 0 <= sr < grown.shape[0] and 0 <= sc < grown.shape[1] and dist[grown[sr, sc]] == d
        tx = canon(" ".join(it.text.upper().split()))
        anyc.append(tx)
        if on:
            found.append(tx)
        res.append(dict(rank=rank, variant=variant, sheet=nn, text=tx, x=float(wc[0]), y=float(wc[1]), on=bool(on), direction=it.direction))
    i = np.flatnonzero((dist == d) & wr)[0]
    hits.setdefault(rank, {})[variant] = (code[i], found, len(anyc))
json.dump(res, open(sys.argv[3], "w"))
tot = {}
for rank in sorted(hits):
    line = []
    for v, (cd, found, n) in hits[rank].items():
        ok = cd in found
        tot.setdefault(v, [0, 0, 0])
        tot[v][0] += ok
        tot[v][1] += bool(found)
        tot[v][2] += 1
        line.append(f"{v}: {'YES' if ok else '-'} {found} ({n})")
    print(rank, cd, int(wd[order[rank]]), " | ".join(line))
for v, s in tot.items():
    print(f"{v:<8} districts {s[2]}: its code found on it {s[0]}, some code found on it {s[1]}")
