"""Correlation of stroke offsets of nearby / facing edges (needs edges.py output): uv run python scripts/tuning/xstreet.py 1948 1"""
import sys, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from _common import out_dir
D = out_dir()
R = np.load(D / f"edges_{sys.argv[1]}_{sys.argv[2]}.npy")
mx, my, n, cen, sd, out, inn, ex, fpp = R.T
ok = np.abs(cen) < 15
xy = np.c_[mx, my][ok]; c = cen[ok]
tr = cKDTree(xy)
pairs = tr.query_pairs(110, output_type="ndarray")
d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
pp = pairs[(d > 30)]
a, b = c[pp[:, 0]], c[pp[:, 1]]
print("edge pairs 30-110 ft apart (across street or same block):", len(pp), " corr", np.corrcoef(a, b)[0, 1].round(2))
# nearest-other-edge correlation by distance band
for lo, hi in ((0, 300), (300, 600), (600, 1200), (1200, 3000)):
    pr = tr.query_pairs(hi, output_type="ndarray"); dd = np.linalg.norm(xy[pr[:, 0]] - xy[pr[:, 1]], axis=1)
    pr = pr[dd > lo]
    print(f" pairs {lo}-{hi} ft: corr of stroke offset {np.corrcoef(c[pr[:,0]], c[pr[:,1]])[0,1]:.2f}")
print(" fraction |offset|<=3ft", np.mean(np.abs(cen) <= 3).round(2), "<=6", np.mean(np.abs(cen) <= 6).round(2))
