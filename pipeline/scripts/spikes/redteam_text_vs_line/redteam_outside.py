import os

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(k, "1")
import json
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)
from sfzmaps.solver import paths, runs  # noqa: E402
from sfzmaps.solver.textnet import data  # noqa: E402

data.LOAD_THREADS = 1
OUT = Path(__file__).parent / "redteam-out"
LINES = runs.EXPERIMENTS_DIR / "detector-overcall" / "lines" / paths.SERIES
tot = np.zeros(6, np.int64)
for name in paths.SHEETS:
    sd = data.sheet_data(paths.SERIES, name, hold_out_validation=False)
    ink = data.page_ink(sd)
    with np.load(LINES / f"{name}.npz") as z:
        long = np.unpackbits(z["long"], axis=1, count=sd.width).view(bool)
    run = np.unpackbits(np.load(OUT / f"run-{name}.npz")["run"], axis=1, count=sd.width).view(bool)
    word = data._words(sd, 0, 0, sd.height, sd.width) > 0
    other = data._paint(sd.other, np.ones(len(sd.other), np.int32), 0, 0, sd.height, sd.width) > 0
    away = cv2.distanceTransform((~(word | other)).view(np.uint8), cv2.DIST_L2, 3) > 10
    o = ink & away
    half = cv2.distanceTransform(ink.view(np.uint8), cv2.DIST_L2, 3)
    thick = cv2.dilate((half >= 2.5).view(np.uint8), np.ones((5, 5), np.uint8)).view(bool) & ink
    r = [o.sum(), (o & long).sum(), (o & run).sum(), (o & (long | run)).sum(), (o & thick).sum(),
         (o & thick & ~(long | run)).sum()]
    tot += np.array(r, np.int64)
    print(name, r, flush=True)
n = tot[0]
print(json.dumps({"ink >10px from every box": int(n), "stroke rule": round(tot[1] / n, 3),
                  "run rule": round(tot[2] / n, 3), "either": round(tot[3] / n, 3),
                  "neither": round(1 - tot[3] / n, 3), "thick (>=5px) ink": int(tot[4]),
                  "thick ink on neither rule": round(tot[5] / tot[4], 3),
                  "thick ink on stroke rule": None}))
