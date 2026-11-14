"""Ask the reader directly about each place no print reaches (paid, capped by SFZMAPS_MAX_USD).
Places as in placeread.py (lots more likely wrong than right given the cost to the nearest print, fitted west, joined to their
neighbours).  For each place: the piece of the scan around it, enlarged, the place's lots tinted yellow, shown upright and
turned a quarter; the question is which zoning codes are printed on or right beside the tinted lots.
Writes k8/p<nn>/textboxes2_pa.json and textboxes2_cwpa.json (turned pass + place reads + these), and says how often the
answer is the table's code of the place.
usage: placeask.py RUN.npz FIRST LAST [count]   (places FIRST..LAST-1 by size, largest first)"""
import json
import sys
from collections import Counter
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
from lotcode_lib import canon
from lotwalls import K, SERIES, SHEETS, SheetTransform, config
from sfzmaps import llm

llm.VISION_MODEL = "gpt-6.1-terra"
W = config.WORK_DIR / SERIES
import os
SPLIT = bool(os.environ.get("SPLIT"))
OUTV = "pb" if SPLIT else "pa"
PAD, SIDE, ZMAX = 70, 1400, (4.0 if SPLIT else 3.0)
FIRST, LAST = int(sys.argv[2]), int(sys.argv[3])


class Code(BaseModel):
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    direction: Literal["left-to-right", "bottom-to-top", "top-to-bottom", "upside-down", "slanted"]
    where: Literal["on the tinted lots", "beside the tinted lots"]
    legible: Literal["clear", "partly hidden", "a guess"]


class Codes(BaseModel):
    items: list[Code]


def prompt(w, h):
    return f"""This image is an enlarged crop of a scanned 2000 San Francisco zoning map. It is {w} pixels wide and {h} pixels high (origin top-left, x to the right, y down).

Heavy black lines are zoning district borders; thin lines are block and lot lines. Each district has its zoning code printed in it, or right beside it when the district is small, in large slanted hand lettering: for example RH-1, RH-1(D), RH-2, RH-3, RM-1, RM-2, RM-3, RC-4, NC-1, NC-2, NC-3, C-2, C-M, M-1, M-2, P, or a name such as "VALENCIA NCD". A code can run vertically, be crossed by lines, or lie over a block number.
Small 3- or 4-digit numbers are block numbers and small capitals are street names: they are not zoning codes.

Some lots are tinted yellow. List every zoning code that is printed on the tinted lots, or printed right beside them and meant for them (no heavy line between the code and the tinted lots, or the code overlaps the heavy line around them).

For each one give:
- text: the code exactly as printed;
- x0, y0, x1, y1: the tightest box around its letters, in pixels of this image;
- direction: the way the text reads;
- where: on the tinted lots, or beside them;
- legible: clear, partly hidden, or a guess.

Do not report codes of neighbouring districts. Do not invent a code: if no zoning code is printed for the tinted lots, return an empty list."""


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
a, b = pairs.T[:2]
k = far[a] & far[b]
nc, lab = connected_components(coo_matrix((np.ones(k.sum()), (a[k], b[k])), shape=(len(sm), len(sm))), directed=False)
lab = np.where(far, lab, -1)
if SPLIT:  # each place cut into its blocks: a closer look at each
    blk = np.array([s_[:4] for s_ in g["blklot"].astype(str).values])
    _, lab2 = np.unique(np.char.add(np.char.add(lab.astype(str), "_"), blk), return_inverse=True)
    big = np.bincount(lab[lab >= 0])
    place_size = np.where(lab >= 0, big[np.maximum(lab, 0)], 0)
    lab = np.where(lab >= 0, lab2, -1)
ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
if SPLIT:
    psz = np.array([place_size[lab == i].max() for i in ids])
    order = ids[np.lexsort((-cnt, -psz))][FIRST:LAST]
else:
    order = ids[np.argsort(-cnt, kind="stable")][FIRST:LAST]
print(f"places {len(ids):,} (lots {int(cnt.sum()):,}); asking about places {FIRST}..{FIRST + len(order) - 1} by size (lots {int(sum((lab == i).sum() for i in order)):,})", flush=True)

sheets = {}
for nn in SHEETS:
    sh = f"{SERIES}-p{nn}"
    sheets[nn] = (SheetTransform.from_json(json.loads((W / sh / "transform.json").read_text())),
                  shapely.from_geojson((W / sh / "warped.footprint.geojson").read_text()), cv2.imread(str(W / sh / "page.png")))
jobs = []
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
        continue
    nn = best[1]
    t, fp, page = sheets[nn]
    H, Wd = page.shape[:2]
    polys = [t.to_pixel(np.array(q.exterior.coords, float)) for gg in geo[lab == pid] for q in (gg.geoms if hasattr(gg, "geoms") else [gg])]
    allp = np.vstack(polys)
    x0, y0 = int(max(0, allp[:, 0].min() - PAD)), int(max(0, allp[:, 1].min() - PAD))
    x1, y1 = int(min(Wd, allp[:, 0].max() + PAD)), int(min(H, allp[:, 1].max() + PAD))
    if x1 - x0 < 16 or y1 - y0 < 16:
        continue
    z = min(ZMAX, SIDE / max(x1 - x0, y1 - y0))
    win = cv2.resize(page[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    mask = np.zeros(win.shape[:2], np.uint8)
    cv2.fillPoly(mask, [((q - [x0, y0]) * z).astype(np.int32) for q in polys], 1)
    tint = win.copy()
    light = (mask > 0) & (win.min(2) > 128)
    tint[light] = (150, 245, 255)
    for cw in (False, True):
        img = np.ascontiguousarray(cv2.rotate(tint, cv2.ROTATE_90_CLOCKWISE) if cw else tint)
        jobs.append((pid, nn, x0, y0, z, win.shape[0], img, cw))
todo = sum(not llm.is_cached(j[6], prompt(j[6].shape[1], j[6].shape[0]), Codes) for j in jobs)
print(f"reads {len(jobs)}, {todo} not yet read", flush=True)
if len(sys.argv) > 4:
    if sys.argv[4] == "show":
        for j in jobs[:6]:
            cv2.imwrite(str(K / f"placeask_{j[0]}_{int(j[7])}.png"), j[6])
    sys.exit()


def one(j):
    try:
        return llm.ask_json(j[6], prompt(j[6].shape[1], j[6].shape[0]), Codes)
    except llm.FATAL:
        raise
    except Exception as e:
        print("failed", j[0], type(e).__name__, flush=True)
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

by = {}
seen = {}
tally = Counter()
per = {}
for j, r in zip(jobs, out):
    if r is None:
        continue
    pid, nn, x0, y0, z, hh, _, cw = j
    top = Counter(code[(lab == pid) & known]).most_common(1)
    top = top[0][0] if top else ""
    per.setdefault(pid, [top, west[lab == pid].mean() > 0.5, set()])
    for it in r.items:
        bx = [it.x0, it.y0, it.x1, it.y1]
        if cw:
            bx = [it.y0, hh - 1 - it.x1, it.y1, hh - 1 - it.x0]
        box = [bx[0] / z + x0, bx[1] / z + y0, bx[2] / z + x0, bx[3] / z + y0]
        cc = canon(" ".join(it.text.upper().split()))
        per[pid][2].add(cc)
        tally[("west" if per[pid][1] else "east", it.where, it.legible, cc == top)] += 1
        ctr = sheets[nn][0].to_world(np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], float))[0]
        li = np.flatnonzero(lab == pid)
        l0 = li[np.argmin(shapely.distance(geo[li], shapely.Point(ctr)))]
        at = shapely.get_coordinates(shapely.point_on_surface(geo[l0]))[0]
        rank = ["a guess", "partly hidden", "clear"].index(it.legible)
        e = seen.get((int(pid), cc))
        if e is None:
            e = {"text": it.text, "kind": "zoning code", "direction": it.direction, "angle_deg": 0, "box": box, "edge": 0.0,
                 "turned": [cw], "where": it.where, "rank": rank, "place": int(pid), "at": [float(at[0]), float(at[1])],
                 "nlots": int(len(li))}
            seen[(int(pid), cc)] = e
            by.setdefault(nn, []).append(e)
        else:
            e["turned"] = sorted(set(e["turned"]) | {cw})
            e["rank"] = max(e["rank"], rank)
            if it.where == "on the tinted lots":
                e["where"] = it.where
for kk in sorted(tally, key=str):
    print("  ", kk, tally[kk])
for side in (True, False):
    v = [q for q in per.values() if q[1] == side and q[0]]
    print(f"{'west' if side else 'east'} places with a table code {len(v)}: no answer {sum(not q[2] for q in v)}; the table's code among the answers {sum(q[0] in q[2] for q in v)}; only other codes {sum(bool(q[2]) and q[0] not in q[2] for q in v)}")
for nn in SHEETS:
    new = by.get(nn, [])
    old = json.load(open(K / f"p{nn}" / f"textboxes2_{OUTV}.json")) if (K / f"p{nn}" / f"textboxes2_{OUTV}.json").exists() else []
    keep = [o for o in old if o.get("place") not in {q["place"] for q in new}] + new
    (K / f"p{nn}" / f"textboxes2_{OUTV}.json").write_text(json.dumps(keep, indent=0))
