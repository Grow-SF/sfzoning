"""Street / interior / block-material classes and territories around a pixel: uv run python scripts/tuning/terr.py 1948 1 3230 2847"""
import sys, numpy as np, cv2
from pathlib import Path
from skimage.segmentation import watershed
from _common import blocks_corners, out_dir
from sfzmaps import config, linework
D = out_dir()
series, sheet, cx, cy = sys.argv[1], sys.argv[2], int(float(sys.argv[3])), int(float(sys.argv[4]))
img = cv2.imread(str(config.WORK_DIR / series / sheet / "page.png"), cv2.IMREAD_UNCHANGED)
H = 500; sub = img[cy-H:cy+H, cx-H:cx+H]
ink = linework.binarize(sub)
outside = np.zeros(ink.shape, bool)
street, interior = linework.classify_white(ink, outside, linework.is_photo(sub))
mat, anc = linework.block_material(ink, outside, linework.is_photo(sub))
terr = watershed(np.zeros(ink.shape, np.uint8), markers=anc, mask=mat, connectivity=1)
S = 80; c = slice(H-S, H+S)
v = np.zeros((2*S, 2*S, 3), np.uint8)
v[street[c, c]] = (200, 200, 200)
v[interior[c, c]] = (230, 255, 230)
v[mat[c, c] & ink[c, c]] = (0, 0, 150)
v[mat[c, c] & ~ink[c, c] & ~interior[c, c]] = (150, 0, 150)
v[ink[c, c] & ~mat[c, c]] = (0, 200, 255)
t = terr[c, c]; edge = (np.diff(t, axis=0, prepend=t[:1]) != 0) | (np.diff(t, axis=1, prepend=t[:, :1]) != 0)
v[edge] = (255, 0, 0)
v = cv2.resize(v, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
cv2.circle(v, (4*S, 4*S), 6, (0, 255, 0), 2)
cv2.imwrite(str(D / f"terr_{cx}_{cy}.png"), v); print("ok")
