"""How many anchor candidates does the owner's answer key agree with (within 150/300 ft)?

    uv run python scripts/tuning/anchor_check.py 1948 10
"""
import sys
import warnings

import numpy as np

from sfzmaps import anchors, config, pages, reading, runner
from sfzmaps.sources import datasf
from sfzmaps.transform import SheetTransform

warnings.filterwarnings("ignore")
series, sheet = sys.argv[1], sys.argv[2]
page = [p for p in pages.discover() if p.series == series and p.sheet_id == sheet][0]
d = config.WORK_DIR / series / sheet
ref = datasf.load_reference()
cands = anchors.match_labels(reading.load_labels(d / "labels.json"), ref)
key = SheetTransform(runner._answer_key_gcps(page, d / "page.png"), config.COARSE_TPS_SMOOTHING)
P = np.array([[c.label.x, c.label.y] for c in cands], float)
W = np.array([c.world for c in cands])
r = np.linalg.norm(key.to_world(P) - W, axis=1)
kind = np.array([c.kind for c in cands])
for k in ("block", "street_pair"):
    s = kind == k
    if s.any():
        # street pairs: one label pair may map to several intersections; count labels with any hit
        print(f"{k}: {s.sum()} candidates; within 150 ft {np.sum(r[s] <= 150)}, 300 ft {np.sum(r[s] <= 300)}; median {np.median(r[s]):.0f}")
p0 = key._enabled()
J = key.to_world(np.array([[3000., 3000.], [3010., 3000.]]))
print("key scale ft/px", np.hypot(*(J[1] - J[0])) / 10)
