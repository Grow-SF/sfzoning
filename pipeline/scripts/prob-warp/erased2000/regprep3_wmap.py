"""Weighted block-line centres for the probabilistic correction (probfit.py, WMAP), sheet NN, on reg2_NN.npz's crop.
  text out  the scan's ink less the letters run k4's fit drew and erased (k4/pNN_c1: noblock.png ink not in
            clean.png), so block lines stay and fitted text goes
  width     each centreline (skeleton) pixel's stroke width = 2 x its distance to paper; the sheet's widths fitted
            as two groups, block lines and heavy zoning lines (two Gaussians, EM); a pixel's weight = its
            probability of being in the block-line group at its width (no cutoff: heavy ink keeps a small weight)
  direction as regprep2.py: the ink's local direction (structure tensor); one weight map per 10 degree bin, the
            pixels running within 20 degrees of it
-> regw_NN.npz (weights float16 B x h x w, same crop/origin as reg2).  usage: regprep3.py NN"""
import json
import sys

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from sfzmaps import config

NN = sys.argv[1]
S = "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/"
D = config.WORK_DIR / "2000-digital-zoning" / f"2000-digital-zoning-p{NN}"
page = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE) < 128
noblock = cv2.imread(S + f"stepview/sheetsK/p{NN}/noblock.png", cv2.IMREAD_GRAYSCALE) < 128
clean = cv2.imread(S + f"k4/p{NN}_c1/clean.png", cv2.IMREAD_GRAYSCALE) < 128
text = noblock & ~clean
ink = page & ~text
R2 = np.load(S + f"reg2_{NN}.npz")
x0, y0 = (int(v) for v in R2["origin"])
B, h, w = R2["targets"].shape
sub = ink[y0:y0 + h, x0:x0 + w]
half = ndimage.distance_transform_edt(sub)
sk = skeletonize(sub)
wid = 2.0 * half[sk]
# two Gaussians on the stroke widths of every centreline pixel
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
pw = r[:, thin]
weight = np.zeros(sub.shape, np.float32)
weight[sk] = pw
f = ndimage.gaussian_filter(sub.astype(np.float32), 1.0)
gx, gy = ndimage.sobel(f, 1), ndimage.sobel(f, 0)
Jxx, Jxy, Jyy = (ndimage.gaussian_filter(v, 2.0) for v in (gx * gx, gx * gy, gy * gy))
cdir = (0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy)) + 90.0) % 180
W8 = np.zeros((B, h, w), np.float16)
for i, b in enumerate(np.arange(0, 180, 10.0)):
    dd = np.abs((cdir - b + 90) % 180 - 90)
    W8[i] = (weight * (dd <= 20.0)).astype(np.float16)
np.savez_compressed(S + f"regw_{NN}.npz", weights=W8, origin=np.array([x0, y0]))
widths = {f"{v:.1f}": round(float(pw[np.argmin(np.abs(wid - v))]), 3) for v in (2, 3, 4, 5, 6, 7, 8)}
print(json.dumps({"sheet": NN, "text_px_removed": int(text[y0:y0 + h, x0:x0 + w].sum()), "centre_px": int(sk.sum()),
                  "groups": {"block line": {"width_px": round(float(m[thin]), 2), "spread": round(float(s[thin]), 2), "share": round(float(pi[thin]), 2)},
                             "heavy": {"width_px": round(float(m[1 - thin]), 2), "spread": round(float(s[1 - thin]), 2), "share": round(float(pi[1 - thin]), 2)}},
                  "p_block_line_at_width": widths, "old_rule_kept": int((sk & (half <= 2.0)).sum())}))
