"""A third read of every printed code (paid, capped by SFZMAPS_MAX_USD; answers cached in the scratch data dir): a close
crop of the scan around the code, asking only 'what code is printed here, and does a leader line point it elsewhere'.
Saves read3.npz (r3, c3, leader, done), in labread2's order.
usage: read3.py [LIMIT]"""
import json
import sys

import cv2
import numpy as np
from pydantic import BaseModel

import truth
from lotcode_lib import canon
from lotwalls import K, SERIES, SHEETS, SheetTransform, config
from sfzmaps import llm


class CodeRead(BaseModel):
    code: str
    confidence: float
    leader: bool


PROMPT = """This image is a close crop of a scanned San Francisco zoning district map. It is {w} pixels wide and {h} pixels high. A zoning district code is printed in the middle of the image, within about the rectangle from ({x0}, {y0}) to ({x1}, {y1}) (pixels, origin at the top-left). Read that one code exactly as printed, character by character; take particular care with the digit after the hyphen and with any suffix in brackets. Codes look like RH-1, RH-1(D), RH-1(S), RH-2, RH-3, RM-1, RM-2, RM-3, RM-4, RC-3, RC-4, NC-1, NC-2, NC-3, NC-S, C-1, C-2, C-3-O, C-3-R, C-3-G, C-3-S, C-M, M-1, M-2, P, RED, SLR, SLI, SSO, RSD, SPD; some districts are printed as a name instead (for example NORTH BEACH). Street names, block numbers and other lettering are not zoning codes. Return: code = the code as printed (an empty string if what is printed there is not a zoning district code); confidence = from 0 to 1, how sure you are of every character; leader = true only if a drawn leader line or arrow runs from this code to an area other than where the code itself is printed."""

W = config.WORK_DIR / SERIES
LR = np.load(truth.D / "labread2.npz")
L = len(LR["x"])
rows = []                                              # sheet, world x, y, page box
for nn in SHEETS:
    sheet = f"{SERIES}-p{nn}"
    f = W / sheet / "district_labels.json"
    pipe = [l for l in (json.load(open(f))["labels"] if f.exists() else []) if l.get("kind", "use") == "use"]
    t = SheetTransform.from_json(json.loads((W / sheet / "transform.json").read_text()))
    pl = np.array([[l["x"], l["y"]] for l in pipe], float).reshape(-1, 2)
    used = np.zeros(len(pipe), bool)
    for b in json.load(open(K / f"p{nn}" / "textboxes_gpt-6.1-terra.json")):
        if b["kind"] != "zoning code":
            continue
        bx0, by0, bx1, by1 = b["box"]
        ctr = t.to_world(np.array([[(bx0 + bx1) / 2, (by0 + by1) / 2]], float))[0]
        if len(pl):
            used[(pl[:, 0] >= bx0) & (pl[:, 0] <= bx1) & (pl[:, 1] >= by0) & (pl[:, 1] <= by1)] = True
        rows.append((nn, ctr[0], ctr[1], (bx0, by0, bx1, by1)))
    for j in np.flatnonzero(~used):
        ctr = t.to_world(pl[j:j + 1])[0]
        rows.append((nn, ctr[0], ctr[1], (pl[j, 0], pl[j, 1], pl[j, 0], pl[j, 1])))
key = {(r[0], round(r[1], 1), round(r[2], 1)): r[3] for r in rows}
boxes = [key[(str(s), round(float(x), 1), round(float(y), 1))] for s, x, y in zip(LR["sheet"], LR["x"], LR["y"])]
sizes = np.array([max(b[2] - b[0], b[3] - b[1]) for b in boxes], float)
med = np.median(sizes[sizes > 0])
pages = {}
items = []
order = np.argsort(LR["sheet"], kind="stable")
lim = int(sys.argv[1]) if len(sys.argv) > 1 else L
pick = np.random.default_rng(1).permutation(L)[:lim]
for j in pick:
    nn = str(LR["sheet"][j])
    if nn not in pages:
        pages[nn] = cv2.imread(str(W / f"{SERIES}-p{nn}" / "page.png"), cv2.IMREAD_GRAYSCALE)
    pg = pages[nn]
    bx0, by0, bx1, by1 = boxes[j]
    sz = max(bx1 - bx0, by1 - by0)
    if sz <= 0:                                        # a point from the label reader, no box: a box of the usual size
        bx0, by0, bx1, by1 = bx0 - med / 2, by0 - med / 4, bx1 + med / 2, by1 + med / 4
        sz = med
    cx, cy, half = (bx0 + bx1) / 2, (by0 + by1) / 2, sz * 1.5
    r0, r1 = int(max(0, cy - half)), int(min(pg.shape[0], cy + half))
    c0, c1 = int(max(0, cx - half)), int(min(pg.shape[1], cx + half))
    crop = pg[r0:r1, c0:c1]
    z = max(1, int(np.ceil(640 / max(crop.shape))))
    z = min(z, max(1, llm.MAX_EDGE // max(crop.shape)))
    if z > 1:
        crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
    prompt = PROMPT.format(w=crop.shape[1], h=crop.shape[0], x0=int((bx0 - c0) * z), y0=int((by0 - r0) * z),
                           x1=int((bx1 - c0) * z), y1=int((by1 - r0) * z))
    items.append((j, crop, prompt))
if len(sys.argv) > 2:                                  # dump a few crops to look at, no calls
    for j, crop, prompt in items[:6]:
        cv2.imwrite(f"{sys.argv[2]}/r3_{j}.png", crop)
        print(j, LR["pc"][j], LR["tx"][j], prompt[100:230])
    sys.exit()


def one(it):
    try:
        return llm.ask_json(it[1], it[2], CodeRead)
    except llm.FATAL:
        raise
    except Exception as e:                             # one bad answer must not lose the rest
        print("failed", it[0], type(e).__name__, flush=True)
        return None


try:
    out = llm.pmap(one, items, 8)
except llm.FATAL as e:
    print("stopped:", type(e).__name__, str(e)[:120])
    out = []
    for it in items:                                   # keep what was answered before the stop
        out.append(llm.ask_json(it[1], it[2], CodeRead) if llm.is_cached(it[1], it[2], CodeRead) else None)
C = llm.CallCounter
print(f"calls: live {C.live}, from cache {C.hits}; tokens in {C.in_tokens:,} out {C.out_tokens:,}; cost ${C.usd():.2f}")
r3 = np.array([""] * L, dtype=object)
c3 = np.full(L, np.nan)
leader = np.zeros(L, bool)
done = np.zeros(L, bool)
for (j, _, _), a in zip(items, out):
    if a is None:
        continue
    t_ = " ".join(a.code.upper().split())
    r3[j], c3[j], leader[j], done[j] = (canon(t_) if t_ else ""), a.confidence, a.leader, True
np.savez_compressed(truth.D / ("read3.npz" if lim >= L else "read3_pilot.npz"), r3=r3.astype(str), c3=c3, leader=leader, done=done)
pc, tx = LR["pc"], LR["tx"]
d = done
print(f"read {int(d.sum())}: same as label reader {(r3[d & (pc != '')] == pc[d & (pc != '')]).mean():.1%}, same as text-box reader {(r3[d & (tx != '')] == tx[d & (tx != '')]).mean():.1%}, empty {(r3[d] == '').mean():.1%}, leader {leader[d].mean():.1%}")
