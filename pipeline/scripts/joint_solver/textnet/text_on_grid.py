"""The detector's text chance on the 4-ft lot grid, out of fold: every pixel takes the chance from the model that held
out its own quarter, so no pixel's chance comes from a model trained on that quarter's terra boxes.
Grid row r, column c <-> merged page row row0 - r, column c - col0.  Saves zon2000/ztextchance_oof.npz (chance, uint8).
usage: text_on_grid.py"""
import os
import sys
from pathlib import Path

import numpy as np

S = Path("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad")
os.environ.setdefault("SFZ_RND", str(S))
sys.path.insert(0, str(S / "pkg"))
from joint_solver.data import city  # noqa: E402

QUARTERS = ("nw", "ne", "sw", "se")
D = np.load(S / "textnet" / "data.npz")
quarter = D["quarter"]
page = np.zeros(quarter.shape, np.uint8)
for k, q in enumerate(QUARTERS):
    p = np.load(S / "textnet" / f"text_pred_{q}.npz")["prob"]
    m = quarter == k
    page[m] = p[m]
C = city()
_, row0, col0 = C.scan()
H, W = np.load(S / "zon2000" / "truth_grid.npz")["lot"].shape
r, c = np.arange(H), np.arange(W)
pr, pc = row0 - r, c - col0
okr, okc = (pr >= 0) & (pr < page.shape[0]), (pc >= 0) & (pc < page.shape[1])
grid = np.zeros((H, W), np.uint8)
grid[np.ix_(np.flatnonzero(okr), np.flatnonzero(okc))] = page[np.ix_(pr[okr], pc[okc])]
np.savez_compressed(S / "zon2000" / "ztextchance_oof.npz", chance=grid)
print(f"saved ztextchance_oof.npz: grid {H} x {W}; share called text (>= 1/2) {(grid >= 128).mean():.2%}")
