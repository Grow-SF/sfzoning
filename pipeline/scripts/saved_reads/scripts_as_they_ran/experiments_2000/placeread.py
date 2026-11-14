"""Look again, closer, where no print reaches (paid, capped by SFZMAPS_MAX_USD).
From a solve (RUN.npz): how likely a lot is wrong given the cost of the cheapest path to any print (fitted on the west half);
the lots more likely wrong than right, joined to their neighbours, are the places to look again.  Each place's piece of the
scan is read again by the text finder (same prompt and model as textpass.py), at twice the size, upright and turned a quarter.
Writes k8/p<nn>/textboxes2_pl.json and textboxes2_cwpl.json (the turned pass plus these).
usage: placeread.py RUN.npz NMAX [count]   (NMAX: the largest NMAX places; count: only say how many calls are not cached)"""
import json
import math
import sys
from typing import Literal

import cv2
import numpy as np
import shapely
from pydantic import BaseModel
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import lotwalls
import truth
import yard
from lotwalls import K, SERIES, SHEETS, SheetTransform, config
from sfzmaps import llm

llm.VISION_MODEL = "gpt-6.1-terra"
W = config.WORK_DIR / SERIES
TILE, OVER = 384, 80
NMAX = int(sys.argv[2])


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
- text: exactly as printed, letter for letter. One item per whole label: all the words printed together as one name on one line are ONE item with one box, e.g. "24TH-NOE VALLEY NCD", "VALENCIA NCD", "CASTRO NCD", "TWENTY-FIRST ST." when ST. follows the name on the same line. A street's "ST." printed elsewhere, away from its name
- kind;
- x0, y0, x1, y1: the tightest box around the printed letters themselves, in pixels of this image; not around nearby lines;
- direction: the way the text reads;
- angle_deg: angle of the text's baseline counterclockwise from horizontal (0 = normal, 90 = reads bottom to top, -90 = reads top to bottom).

Do not invent text. Do not include lines, arrows or leader lines."""


def starts(lo_, hi_, n):
    """tile starts covering [lo_, hi_) inside a page side of n px"""
    lo_, hi_ = max(0, lo_), min(n, hi_)
    if hi_ - lo_ <= TILE:
        return [int(min(max(0, (lo_ + hi_) // 2 - TILE // 2), max(0, n - TILE)))]
    k = math.ceil((hi_ - lo_ - OVER) / (TILE - OVER))
    return sorted({int(round(v)) for v in np.linspace(lo_, hi_ - TILE, k)})


def show(win, cw):
    win = cv2.resize(win, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(cv2.rotate(win, cv2.ROTATE_90_CLOCKWISE) if cw else win)


def back(it, th, cw):
    x0, y0, x1, y1 = it.x0, it.y0, it.x1, it.y1
    if cw:
        x0, y0, x1, y1 = y0, th * 2 - 1 - x1, y1, th * 2 - 1 - x0
    return [x0 / 2, y0 / 2, x1 / 2, y1 / 2]


B = lotwalls.build("city5")
g, lot, grown, pairs, ins = truth.load(B)
code, _ = yard.truth_codes(g)
known = g["known"].values
scored = np.load(truth.D / "lotcode_run.npz")["scored"]
geo = g.geometry.values
cen = shapely.get_coordinates(shapely.centroid(geo))
west = cen[:, 0] < np.median(cen[scored, 0])
P = np.load(truth.D / sys.argv[1])
pred = P["names"].astype(str)[P["pred"]]
sm = P["supmin"].astype(float)
m = scored & known & west
X = np.c_[np.ones(len(sm)), sm]
y = (pred != code).astype(float)
be = np.zeros(2)
for _ in range(100):
    p = 1 / (1 + np.exp(-X[m] @ be))
    st = np.linalg.solve((X[m] * (p * (1 - p))[:, None]).T @ X[m] + 1e-6 * np.eye(2), X[m].T @ (y[m] - p))
    be += st
    if np.abs(st).max() < 1e-9:
        break
far = (X @ be > 0) & scored
print(f"wrong given the cost to the nearest print (fitted west): more likely wrong than right from cost {-be[0] / be[1]:.2f}; such lots {int(far.sum()):,}", flush=True)
a, b = pairs.T[:2]
k = far[a] & far[b]
nc, lab = connected_components(coo_matrix((np.ones(k.sum()), (a[k], b[k])), shape=(len(sm), len(sm))), directed=False)
lab = np.where(far, lab, -1)
ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
order = ids[np.argsort(-cnt)][:NMAX]
print(f"places: {len(ids):,}; taking the largest {len(order)} (lots {int(np.sort(cnt)[::-1][:NMAX].sum()):,} of {int(cnt.sum()):,})", flush=True)

sheets = {}
for nn in SHEETS:
    sh = f"{SERIES}-p{nn}"
    sheets[nn] = (SheetTransform.from_json(json.loads((W / sh / "transform.json").read_text())),
                  shapely.from_geojson((W / sh / "warped.footprint.geojson").read_text()), cv2.imread(str(W / sh / "page.png")))
tiles = set()
nos = 0
for pid in order:
    u = shapely.union_all(geo[lab == pid])
    c = shapely.centroid(u)
    best = None
    for nn, (t, fp, page) in sheets.items():
        if shapely.contains(fp, c):
            d = shapely.distance(shapely.boundary(fp), c)
            if best is None or d > best[0]:
                best = (d, nn)
    if best is None:
        nos += 1
        continue
    nn = best[1]
    t, fp, page = sheets[nn]
    x0, y0, x1, y1 = shapely.bounds(u)
    px = t.to_pixel(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))
    H, Wd = page.shape[:2]
    for ty in starts(int(px[:, 1].min()) - OVER, int(px[:, 1].max()) + OVER, H):
        for tx in starts(int(px[:, 0].min()) - OVER, int(px[:, 0].max()) + OVER, Wd):
            tiles.add((nn, tx, ty))
jobs = []
for nn, tx, ty in sorted(tiles):
    win = sheets[nn][2][ty:ty + TILE, tx:tx + TILE]
    for cw in (False, True):
        jobs.append((nn, tx, ty, win.shape[0], win.shape[1], show(win, cw), cw))
todo = sum(not llm.is_cached(j[5], prompt(j[5].shape[1], j[5].shape[0]), TextItems) for j in jobs)
print(f"places on no sheet {nos}; tiles {len(tiles)}; reads {len(jobs)}, {todo} not yet read", flush=True)
if len(sys.argv) > 3:
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
    out = [None] * len(jobs)
C = llm.CallCounter
print(f"calls: live {C.live}, from cache {C.hits}; tokens in {C.in_tokens:,} out {C.out_tokens:,}; cost ${C.usd():.2f}; reads without an answer {sum(o is None for o in out)}")
if stopped:
    sys.exit()


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0]))
    iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
    i = ix * iy
    un = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i
    return i / un if un > 0 else 0


by = {}
for j, r in zip(jobs, out):
    if r is None:
        continue
    nn, tx, ty, th, tw, _, cw = j
    for it in r.items:
        bb = back(it, th, cw)
        edge = min(bb[0], bb[1], tw - bb[2], th - bb[3])
        by.setdefault(nn, []).append({"text": it.text, "kind": it.kind, "direction": it.direction, "angle_deg": it.angle_deg,
                                      "box": [bb[0] + tx, bb[1] + ty, bb[2] + tx, bb[3] + ty], "edge": edge, "tile": [tx, ty], "turned": cw})
for nn in SHEETS:
    items = sorted(by.get(nn, []), key=lambda i: -i["edge"])
    kept = []
    for it in items:
        if not any(iou(it["box"], q["box"]) > 0.3 for q in kept):
            kept.append(it)
    (K / f"p{nn}" / "textboxes2_pl.json").write_text(json.dumps(kept, indent=0))
    f = K / f"p{nn}" / "textboxes2_cw.json"
    cwb = json.load(open(f)) if f.exists() else []
    (K / f"p{nn}" / "textboxes2_cwpl.json").write_text(json.dumps(cwb + kept, indent=0))
    print(nn, len(kept), "boxes;", sum(q["kind"] == "zoning code" for q in kept), "zoning codes")
