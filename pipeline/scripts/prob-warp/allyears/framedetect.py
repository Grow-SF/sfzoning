"""The map frame's bottom rule on one sheet (legend and title lie below it), as on p07: a long horizontal rule
in the lower 40% of the page (a row with ink across more than half the width); its top edge fitted as a line
across the columns.  Writes frame_bottom.json {k, b} (y = k x + b, page px) and cutline.png; if no rule is
found, b is far below the page (nothing is cut) and found=false.
usage: framedetect.py SHEET OUTDIR"""
import json
import sys

import cv2
import numpy as np

from sfzmaps import config

sheet, out = sys.argv[1:3]
import os  # noqa: E402
g = cv2.imread(str(config.WORK_DIR / os.environ["SERIES"] / sheet / "page.png"), cv2.IMREAD_GRAYSCALE)
H, W = g.shape
ink = g < 128
rows = ink.mean(1)
cand = [y for y in range(int(H * 0.6), H) if rows[y] > 0.5]
res = {"k": 0.0, "b": float(H * 10), "found": False}
for y_top in [y for i, y in enumerate(cand) if i == 0 or cand[i - 1] != y - 1]:
    xs, ys = [], []
    for x in range(0, W, 10):
        col = ink[y_top - 30:y_top + 30, x]
        if col.any():
            xs.append(x)
            ys.append(y_top - 30 + int(np.nonzero(col)[0].min()))
    xs, ys = np.array(xs), np.array(ys)
    k, b = np.polyfit(xs, ys, 1)
    keep = np.abs(ys - (k * xs + b)) < 4
    if keep.sum() > 0.6 * len(xs) and keep.sum() > 0.5 * len(range(0, W, 10)):
        k, b = np.polyfit(xs[keep], ys[keep], 1)
        res = {"k": float(k), "b": float(b), "found": True, "row": int(y_top), "columns_used": int(keep.sum())}
        break
json.dump(res, open(f"{out}/frame_bottom.json", "w"))
im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
if res["found"]:
    over = im.copy()
    cv2.fillPoly(over, [np.array([[0, res["b"]], [W, res["k"] * W + res["b"]], [W, H], [0, H]], np.int32)], (180, 180, 255))
    im = cv2.addWeighted(im, 0.6, over, 0.4, 0)
    cv2.line(im, (0, int(res["b"])), (W, int(res["k"] * W + res["b"])), (0, 0, 230), 4)
small = cv2.resize(im[int(H * 0.55):], None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
st = np.full((40, small.shape[1], 3), 235, np.uint8)
cv2.putText(st, f"{sheet}: legend cut {'found' if res['found'] else 'NOT found (nothing cut)'}", (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 160), 2, cv2.LINE_AA)
cv2.imwrite(f"{out}/cutline.png", np.vstack([st, small]))
print(sheet, res)
