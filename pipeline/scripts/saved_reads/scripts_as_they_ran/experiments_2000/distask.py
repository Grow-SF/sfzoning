"""Ask the reader about each closed district of ours that holds no print (paid, capped by SFZMAPS_MAX_USD).
A district = an area of the map no line of ours crosses (the open areas between the walls of the chain's last step).
Chosen without the table: districts that hold lots, where no print we use lies (by its own point or by its lot).
For each: the piece of the scan around it at the scan's own size or enlarged, the district tinted yellow, shown upright and
turned a quarter; the question is which zoning code the map gives the tinted area: printed inside it (any way up), or outside
with a leader line into it, or right beside it with no heavy line between.
Writes k8/p<nn>/textboxes2_pd.json (as placeask.py's files, plus "district" and the three-way "where").
usage: distask.py TAG [show N | count]"""
import json
import sys
from collections import Counter
from typing import Literal

import cv2
import numpy as np
import shapely
from pydantic import BaseModel

import lotwalls
import truth
import yard
from lotcode_lib import canon
from lotwalls import K, SERIES, SHEETS, SheetTransform, config
from sfzmaps import llm

llm.VISION_MODEL = "gpt-6.1-terra"
W = config.WORK_DIR / SERIES
RES = lotwalls.RES
TAG = sys.argv[1]
OUTV = "pd"
PAD, SIDE, ZMAX = 90, 1400, 4.0


class Code(BaseModel):
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    direction: Literal["left-to-right", "bottom-to-top", "top-to-bottom", "upside-down", "slanted"]
    where: Literal["inside the tinted area", "outside, a leader line or arrow runs from it into the tinted area", "outside, right beside the tinted area with no heavy line between"]
    legible: Literal["clear", "partly hidden", "a guess"]


class Codes(BaseModel):
    items: list[Code]


def prompt(w, h):
    return f"""This image is an enlarged crop of a scanned 2000 San Francisco zoning map. It is {w} pixels wide and {h} pixels high (origin top-left, x to the right, y down).

Heavy black lines are zoning district borders; thin lines are block and lot lines. Each district has its zoning code printed in it in large slanted hand lettering: for example RH-1, RH-1(D), RH-2, RH-3, RM-1, RM-2, RM-3, RC-4, NC-1, NC-2, NC-3, C-2, C-M, M-1, M-2, P, or a name such as "VALENCIA NCD". A code can run vertically or slanted, be crossed by lines, or lie over a block number. When a district is too small to hold its code, the code is printed outside it, in a street or a neighbouring block, often with a thin leader line or arrow running from the code into the district.
Small 3- or 4-digit numbers are block numbers and small capitals are street names: they are not zoning codes.

One area is tinted yellow. Heavy lines close it all round: it is one zoning district. Which zoning code does the map give the tinted area? List each code that is meant for it:
- printed inside the tinted area, whichever way it reads;
- or printed outside it with a leader line or arrow that runs from the code into the tinted area;
- or printed outside it, right beside it, with no heavy line between the code and the tinted area (or overlapping the heavy line around it).

For each one give:
- text: the code exactly as printed;
- x0, y0, x1, y1: the tightest box around its letters, in pixels of this image;
- direction: the way the text reads;
- where: one of the three cases above;
- legible: clear, partly hidden, or a guess.

Do not report a code that belongs to a neighbouring district (it sits inside that district, across a heavy line, with no leader into the tinted area). Do not invent a code: if the map prints no zoning code for the tinted area, return an empty list."""


B = lotwalls.build("city5")
g, lot, grown, pairs, ins = truth.load(B)
N = len(g)
code, _ = yard.truth_codes(g)
known = g["known"].values
scored = np.load(truth.D / "lotcode_run.npz")["scored"]
geo = g.geometry.values
cen = shapely.get_coordinates(shapely.centroid(geo))
west = cen[:, 0] < np.median(cen[scored, 0])
n0, lab, stats, _ = cv2.connectedComponentsWithStats((B["wall"] == 0).astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
Hc, Wc = lab.shape
ul, ua, cm, tot = truth.lot_area(lot, lab)
reg = np.full(N, -1, np.int64)
reg[ul] = ua
LR = np.load(truth.D / f"labread2_{TAG}.npz")
A6 = np.load(truth.D / f"lotclf6_{TAG}.npz")
L = len(LR["x"])
hasc = np.zeros(L, bool)
hasc[A6["cj"][A6["cj"] < L]] = True
pc_ = np.round((LR["x"] - B["x0"]) / RES).astype(int)
pr_ = np.round((LR["y"] - B["y0"]) / RES).astype(int)
inb = (pr_ >= 0) & (pr_ < Hc) & (pc_ >= 0) & (pc_ < Wc) & hasc
held = np.zeros(n0, bool)
held[lab[pr_[inb], pc_[inb]]] = True
ll = LR["lab_lot"]
q = hasc & (ll >= 0)
held[reg[ll[q]][reg[ll[q]] >= 0]] = True
nlots = np.bincount(reg[(reg >= 0) & scored], minlength=n0)
order = np.flatnonzero((nlots > 0) & ~held)
order = order[np.argsort(-nlots[order], kind="stable")]
print(f"closed districts holding scored lots {int((nlots > 0).sum()):,}; with no print {len(order)} (lots {int(nlots[order].sum()):,})", flush=True)

sheets = {}
for nn in SHEETS:
    sh = f"{SERIES}-p{nn}"
    sheets[nn] = (SheetTransform.from_json(json.loads((W / sh / "transform.json").read_text())),
                  shapely.from_geojson((W / sh / "warped.footprint.geojson").read_text()), cv2.imread(str(W / sh / "page.png")))
jobs = []
small = 0
for d in order:
    x, y, w, h = stats[d, :4]
    msk = (lab[y:y + h, x:x + w] == d).astype(np.uint8)
    cs, hier = cv2.findContours(msk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    rings = [(c.reshape(-1, 2).astype(float) + (x, y)) * RES + (B["x0"], B["y0"]) for c in cs]        # world
    outer = [i for i in range(len(cs)) if hier[0][i][3] < 0]
    big = max(outer, key=lambda i: len(rings[i]))
    c = shapely.Point(rings[big].mean(0)) if len(rings[big]) < 4 else shapely.point_on_surface(shapely.Polygon(rings[big]).buffer(0))
    best = None
    for nn, (t, fp, page) in sheets.items():
        if shapely.contains(fp, c):
            dd = shapely.distance(shapely.boundary(fp), c)
            if best is None or dd > best[0]:
                best = (dd, nn)
    if best is None:
        continue
    nn = best[1]
    t, fp, page = sheets[nn]
    H, Wd = page.shape[:2]
    pr = [t.to_pixel(r) for r in rings]
    allp = np.vstack(pr)
    x0, y0 = int(max(0, allp[:, 0].min() - PAD)), int(max(0, allp[:, 1].min() - PAD))
    x1, y1 = int(min(Wd, allp[:, 0].max() + PAD)), int(min(H, allp[:, 1].max() + PAD))
    if x1 - x0 < 16 or y1 - y0 < 16:
        continue
    z = min(ZMAX, max(1.0, SIDE / max(x1 - x0, y1 - y0)))            # never smaller than the scan's own size
    small += max(x1 - x0, y1 - y0) > SIDE
    win = cv2.resize(page[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC) if z != 1 else page[y0:y1, x0:x1].copy()
    mask = np.zeros(win.shape[:2], np.uint8)
    for i in outer:
        cv2.fillPoly(mask, [((pr[i] - [x0, y0]) * z).astype(np.int32)], 1)
    for i in range(len(cs)):
        if hier[0][i][3] >= 0:
            cv2.fillPoly(mask, [((pr[i] - [x0, y0]) * z).astype(np.int32)], 0)
    tint = win.copy()
    light = (mask > 0) & (win.min(2) > 128)
    tint[light] = (150, 245, 255)
    for cw in (False, True):
        img = np.ascontiguousarray(cv2.rotate(tint, cv2.ROTATE_90_CLOCKWISE) if cw else tint)
        jobs.append((int(d), nn, x0, y0, z, win.shape[0], img, cw))
todo = sum(not llm.is_cached(j[6], prompt(j[6].shape[1], j[6].shape[0]), Codes) for j in jobs)
print(f"reads {len(jobs)}, {todo} not yet read; crops larger than {SIDE} px kept at the scan's own size: {small}; largest side {max(max(j[6].shape[:2]) for j in jobs)} px", flush=True)
if len(sys.argv) > 2:
    if sys.argv[2] == "show":
        n = int(sys.argv[3])
        pick = np.linspace(0, len(jobs) // 2 - 1, n).astype(int) * 2
        (K / "distask_show").mkdir(exist_ok=True)
        for i in pick:
            j = jobs[i]
            cv2.imwrite(str(K / "distask_show" / f"d{j[0]}_p{j[1]}_{int(nlots[j[0]])}lots.png"), j[6])
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

by, seen, tally, per = {}, {}, Counter(), {}
SHORT = {"inside the tinted area": "inside", "outside, a leader line or arrow runs from it into the tinted area": "leader", "outside, right beside the tinted area with no heavy line between": "beside"}
for j, r in zip(jobs, out):
    if r is None:
        continue
    d, nn, x0, y0, z, hh, _, cw = j
    li = np.flatnonzero(reg == d)
    top = Counter(code[li][known[li] & scored[li]]).most_common(1)
    top = top[0][0] if top else ""
    per.setdefault(d, [top, west[li].mean() > 0.5, set(), int(nlots[d])])
    for it in r.items:
        bx = [it.x0, it.y0, it.x1, it.y1]
        if cw:
            bx = [it.y0, hh - 1 - it.x1, it.y1, hh - 1 - it.x0]
        box = [bx[0] / z + x0, bx[1] / z + y0, bx[2] / z + x0, bx[3] / z + y0]
        cc = canon(" ".join(it.text.upper().split()))
        per[d][2].add(cc)
        wh = SHORT[it.where]
        tally[("west" if per[d][1] else "east", wh, it.legible, cc == top)] += 1
        ctr = sheets[nn][0].to_world(np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]], float))[0]
        l0 = li[np.argmin(shapely.distance(geo[li], shapely.Point(ctr)))]
        at = shapely.get_coordinates(shapely.point_on_surface(geo[l0]))[0]
        rank = ["a guess", "partly hidden", "clear"].index(it.legible)
        e = seen.get((d, cc))
        if e is None:
            e = {"text": it.text, "kind": "zoning code", "direction": it.direction, "angle_deg": 0, "box": box, "edge": 0.0,
                 "turned": [cw], "where": wh, "rank": rank, "district": d, "at": [float(at[0]), float(at[1])], "nlots": int(len(li))}
            seen[(d, cc)] = e
            by.setdefault(nn, []).append(e)
        else:
            e["turned"] = sorted(set(e["turned"]) | {cw})
            e["rank"] = max(e["rank"], rank)
            if wh == "inside" or (wh == "leader" and e["where"] == "beside"):
                e["where"] = wh
for kk in sorted(tally, key=str):
    print("  ", kk, tally[kk])
for side in (True, False):
    v = [q_ for q_ in per.values() if q_[1] == side and q_[0]]
    print(f"{'west' if side else 'east'} districts with a table code {len(v)} (lots {sum(q_[3] for q_ in v)}): no answer {sum(not q_[2] for q_ in v)} (lots {sum(q_[3] for q_ in v if not q_[2])}); "
          f"the table's code among the answers {sum(q_[0] in q_[2] for q_ in v)} (lots {sum(q_[3] for q_ in v if q_[0] in q_[2])}); "
          f"only other codes {sum(bool(q_[2]) and q_[0] not in q_[2] for q_ in v)} (lots {sum(q_[3] for q_ in v if q_[2] and q_[0] not in q_[2])})")
for nn in SHEETS:
    (K / f"p{nn}" / f"textboxes2_{OUTV}.json").write_text(json.dumps(by.get(nn, []), indent=0))
