"""Today's control points and how far the correction map TAG moves them, over the raw page: small green dot =
under 0.5 ft; orange ring = 0.5-2 ft; red ring + label = over 2 ft; a line shows the move (x10).  Crop of the
drawn map.  usage: gcpmoved.py NN TAG"""
import json
import sys

import cv2
import numpy as np

from sfzmaps import config
from sfzmaps.transform import SheetTransform

TAG = sys.argv[1]
from ay import *  # noqa: F403,E402
sys.path.insert(0, S)
from composed import Composed  # noqa: E402

T0 = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
Z = np.load(S + f"confit_{NN}_{TAG}.npz")
C = Composed(T0, Z["Gc"], Z["spacing"], Z["origin"])
g = np.array([[x.px, x.py] for x in T0.gcps if x.enabled])
u = C.u(g)
ft = np.linalg.norm(u, axis=1) * ftpx()
import os
fr = np.load(S + f"gcfree_{NN}.npy") if os.path.exists(S + f"gcfree_{NN}.npy") else np.zeros(len(g), bool)
page = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(np.where(page < 128, 185, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
for (x, y), v, f, isfree in zip(g, u, ft, fr):
    p = (int(round(x)), int(round(y)))
    if isfree:
        cv2.circle(img, p, 20, (230, 120, 0), 4)
    if f < 0.5:
        cv2.circle(img, p, 4, (0, 160, 0), -1)
        continue
    col = (0, 140, 255) if f <= 2 else (0, 0, 230)
    cv2.circle(img, p, 14, col, 4)
    cv2.line(img, p, (int(round(x + 10 * v[0])), int(round(y + 10 * v[1]))), col, 3)
    if f > 2:
        cv2.putText(img, f"{f:.1f} ft", (p[0] + 16, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 3, cv2.LINE_AA)
print(f"free control points {fr.sum()} (blue ring), moved ft: {np.round(np.sort(ft[fr]), 1).tolist()}; HELD ones moved > 0.5 ft: {((ft > 0.5) & ~fr).sum()}, worst {ft[~fr].max():.2f} ft")
print(f"control points {len(g)}: under 0.5 ft {(ft < 0.5).sum()}, 0.5-2 ft {((ft >= 0.5) & (ft <= 2)).sum()}, over 2 ft {(ft > 2).sum()}")
R2 = np.load(S + f'reg2_{NN}.npz'); _o = R2['origin']; _h, _w = R2['targets'].shape[1:]
crop = img[int(_o[1]):int(_o[1]) + _h, int(_o[0]):int(_o[0]) + _w]
ban = np.full((60, crop.shape[1], 3), (235, 235, 235), np.uint8)
cv2.putText(ban, f"p{NN} today's control points under the correction ({TAG}): green dot < 0.5 ft; orange ring 0.5-2 ft; red ring > 2 ft (labelled); line = the move x10; big blue ring = free (a side at its corner is free)",
            (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 160), 2, cv2.LINE_AA)
cv2.imwrite(S + f"gcpmoved_{NN}_{TAG}.png", np.vstack([ban, crop]))
print(S + f"gcpmoved_{NN}_{TAG}.png")
