"""How likely each inked centre pixel is to be a block line, from its stroke width: the sheet's stroke widths on the
crop of fit_targets.npz, fitted as two groups (thin block lines and heavier ink); each centre pixel's weight is its
chance of being in the thin group.  All ink counts as printed, text included (its width speaks for it).  Kept per
direction (18, every 10 degrees: the ink's direction within 20 degrees), for the probabilistic fit.
-> block_line_weights.npz (weights: 18 x h x w)."""
import json

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from warp_common import D, S, SERIES, SHEET

page = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE) < 128
FT = np.load(S + "fit_targets.npz")
x0, y0 = (int(v) for v in FT["origin"])
B, h, w = FT["targets"].shape
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
np.savez_compressed(S + "block_line_weights.npz", weights=W8)
print(json.dumps({"series": SERIES, "sheet": SHEET,
                  "block_line_px": round(float(m[thin]), 2), "heavy_px": round(float(m[1 - thin]), 2),
                  "block_share": round(float(pi[thin]), 2)}))
