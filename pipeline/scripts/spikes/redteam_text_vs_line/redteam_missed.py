"""Letter-sized ink pieces outside every Terra word and other pass's box (>10 px), on the 2000 sheets: how many, and a
contact sheet of 64 drawn at random for judging by eye which are text. Run-rule line ink is removed first."""
import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(k, "1")
import json
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
from sfzmaps.solver import paths, saved  # noqa: E402
from sfzmaps.solver.textnet import data  # noqa: E402

data.LOAD_THREADS = 1
OUT = Path(__file__).parent / "redteam-out"
rng = np.random.default_rng(0)
cands, inside_n, words_n = [], 0, 0
pages = {}
for name in paths.SHEETS:
    sd = data.sheet_data(paths.SERIES, name, hold_out_validation=False)
    ink = data.page_ink(sd)
    run = np.unpackbits(np.load(OUT / f"run-{name}.npz")["run"], axis=1, count=sd.width).view(bool)
    rest = ink & ~cv2.dilate(run.view(np.uint8), np.ones((3, 3), np.uint8)).view(bool)
    word = data._words(sd, 0, 0, sd.height, sd.width) > 0
    other = data._paint(sd.other, np.ones(len(sd.other), np.int32), 0, 0, sd.height, sd.width) > 0
    near = cv2.distanceTransform((~(word | other)).view(np.uint8), cv2.DIST_L2, 3)
    n, lab, st, cen = cv2.connectedComponentsWithStats(rest.view(np.uint8), connectivity=8)
    h, w, a = st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_AREA]
    ok = (h >= 8) & (h <= 40) & (w >= 3) & (w <= 40) & (a >= 15) & (a <= 0.7 * h * w)
    ok[0] = False
    cx, cy = cen[:, 0].round().astype(int), cen[:, 1].round().astype(int)
    out_ = ok & (near[np.clip(cy, 0, sd.height - 1), np.clip(cx, 0, sd.width - 1)] > 10)
    in_ = ok & word[np.clip(cy, 0, sd.height - 1), np.clip(cx, 0, sd.width - 1)]
    inside_n += int(in_.sum())
    words_n += len(sd.words)
    for j in np.flatnonzero(out_):
        cands.append((name, int(cx[j]), int(cy[j]), int(st[j, 0]), int(st[j, 1]), int(w[j]), int(h[j])))
    pages[name] = ink
    print(name, int(out_.sum()), int(in_.sum()), flush=True)
pick = rng.choice(len(cands), 64, replace=False)
tiles = []
for i in pick:
    name, cx, cy, x0, y0, w, h = cands[i]
    ink = pages[name]
    R = 40
    win = np.zeros((2 * R, 2 * R), bool)
    ys, xs = slice(max(cy - R, 0), cy + R), slice(max(cx - R, 0), cx + R)
    sub = ink[ys, xs]
    win[:sub.shape[0], :sub.shape[1]] = sub
    t = np.full((2 * R, 2 * R, 3), 255, np.uint8)
    t[win] = (150, 150, 150)
    bx0, by0 = x0 - max(cx - R, 0), y0 - max(cy - R, 0)
    cv2.rectangle(t, (bx0 - 1, by0 - 1), (bx0 + w, by0 + h), (0, 0, 230), 1)
    t = cv2.resize(t, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.putText(t, str(len(tiles)), (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
    tiles.append(np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=0))
rows = [np.hstack(tiles[r * 8:(r + 1) * 8]) for r in range(8)]
cv2.imwrite(str(OUT / "missed-candidates.png"), np.vstack(rows))
res = {"letter-sized pieces >10 px from every box": len(cands), "same-size pieces centred in Terra words": inside_n,
       "terra words": words_n}
print(json.dumps(res))
(OUT / "missed.json").write_text(json.dumps(res))
