"""In-sample vs held-out error of the final warp as a function of TPS smoothing.

Takes a sheet's final inliers (refine.json) and, for each smoothing, refits the warp on
all of them (in-sample) and runs the production block-held-out k-fold
(sfzmaps.score.holdout_residuals: modern blocks in 5 seeded folds).

    uv run python scripts/tuning/cv.py 1948 1
"""
import json
import sys
import warnings

import numpy as np

from sfzmaps import config, score
from sfzmaps.runner import _result_load
from sfzmaps.sources import datasf
from sfzmaps.transform import GCP, SheetTransform

warnings.filterwarnings("ignore")
series, sheet = sys.argv[1], sys.argv[2]
ref = datasf.load_reference()
res = _result_load(json.loads((config.WORK_DIR / series / sheet / "refine.json").read_text()), ref)
inl = [m for m in res.matches if m.label == "inlier"]
sc = [m for m in res.matches if m.label in ("inlier", "outlier")]
P = np.array([[m.px, m.py] for m in sc])
W = np.array([[m.wx, m.wy] for m in sc])
print(f"{series}/{sheet}: {len(inl)} inliers, {len(sc)} scored")
print("smooth   in:f6   f8  p50  p95 | held-out:f6   f8  p50  p95")
for s in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
    t = SheetTransform([GCP(m.px, m.py, m.wx, m.wy, "auto") for m in inl], s)
    r_in = np.linalg.norm(t.to_world(P) - W, axis=1)
    res.transform = t
    h = score.holdout_residuals(res)
    h = h[np.isfinite(h)]
    print(f"{s:6g}  {np.mean(r_in <= 6):.3f} {np.mean(r_in <= 8):.3f} {np.median(r_in):4.1f} {np.quantile(r_in, .95):4.1f} |"
          f"       {np.mean(h <= 6):.3f} {np.mean(h <= 8):.3f} {np.median(h):4.1f} {np.quantile(h, .95):4.1f}")
