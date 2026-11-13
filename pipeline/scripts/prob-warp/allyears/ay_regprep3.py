"""regprep3.py for any series/sheet (env SERIES, SHEET; ay.py): block-line centres weighted by their probability of
being a block line at their stroke width (the sheet's own widths fitted as two groups), on reg2's crop.  Text out where
the years chain erased it (allyears/chain/SERIES/SHEET: noblock.png ink not in SHEET_c1/clean.png: the zoning codes
its letter fit drew); other text stays and is weighed by its width like any ink.  -> S/regw_NN.npz"""
import json

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from ay import *  # noqa: F403

page = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE) < 128
CH = Path(os.environ.get("PW_CHAIN", AY.parents[2] / "data" / "prob-warp" / "chain")) / SERIES   # years chain
nb, cl = CH / NN / "noblock.png", CH / f"{NN}_c1" / "clean.png"
text_src = "none"
if nb.exists() and cl.exists():
    noblock = cv2.imread(str(nb), cv2.IMREAD_GRAYSCALE) < 128
    clean = cv2.imread(str(cl), cv2.IMREAD_GRAYSCALE) < 128
    if noblock.shape == page.shape == clean.shape:
        page = page & ~(noblock & ~clean)
        text_src = "years chain (codes erased)"
R2 = np.load(S + f"reg2_{NN}.npz")
x0, y0 = (int(v) for v in R2["origin"])
B, h, w = R2["targets"].shape
sub = page[y0:y0 + h, x0:x0 + w]
half = ndimage.distance_transform_edt(sub)
sk = skeletonize(sub)
wid = 2.0 * half[sk]
m = np.array([np.percentile(wid, 25), np.percentile(wid, 95)])
s = np.array([1.0, 1.0])
pi = np.array([0.7, 0.3])
for _ in range(200):
    lk = pi[None] * np.exp(-(wid[:, None] - m[None]) ** 2 / (2 * s[None] ** 2)) / (np.sqrt(2 * np.pi) * s[None])
    r = lk / np.maximum(lk.sum(1, keepdims=True), 1e-300)
    pi = r.mean(0)
    m = (r * wid[:, None]).sum(0) / r.sum(0)
    s = np.sqrt((r * (wid[:, None] - m[None]) ** 2).sum(0) / r.sum(0)) + 1e-3
thin = int(np.argmin(m))
weight = np.zeros(sub.shape, np.float32)
weight[sk] = r[:, thin]
f = ndimage.gaussian_filter(sub.astype(np.float32), 1.0)
gx, gy = ndimage.sobel(f, 1), ndimage.sobel(f, 0)
Jxx, Jxy, Jyy = (ndimage.gaussian_filter(v, 2.0) for v in (gx * gx, gx * gy, gy * gy))
cdir = (0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy)) + 90.0) % 180
W8 = np.zeros((B, h, w), np.float16)
for i, b in enumerate(np.arange(0, 180, 10.0)):
    dd = np.abs((cdir - b + 90) % 180 - 90)
    W8[i] = (weight * (dd <= 20.0)).astype(np.float16)
np.savez_compressed(S + f"regw_{NN}.npz", weights=W8, origin=np.array([x0, y0]))
print(json.dumps({"series": SERIES, "sheet": NN, "text_out": text_src,
                  "block_line_px": round(float(m[thin]), 2), "heavy_px": round(float(m[1 - thin]), 2),
                  "block_share": round(float(pi[thin]), 2)}))
