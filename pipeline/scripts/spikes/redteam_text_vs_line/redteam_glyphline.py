"""In Terra words that hold no run-rule line ink within 3 px (so their ink is glyphs, specks, or curved/stair lines),
the share of ink the stroke rule (M1's base) calls long line, by kind."""
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
kinds = {v: k for k, v in data.KINDS.items()}
acc = {k: [0, 0, 0, 0] for k in data.KINDS}       # words, words with any stroke-long, ink, stroke-long ink
for name in paths.SHEETS:
    sd = data.sheet_data(paths.SERIES, name, hold_out_validation=False)
    ink = data.page_ink(sd)
    with np.load(LINES / f"{name}.npz") as z:
        long = np.unpackbits(z["long"], axis=1, count=sd.width).view(bool)
    run = np.unpackbits(np.load(OUT / f"run-{name}.npz")["run"], axis=1, count=sd.width).view(bool)
    near_run = cv2.dilate(run.view(np.uint8), np.ones((7, 7), np.uint8)).view(bool)
    lo, hi = sd.words.min(axis=1), sd.words.max(axis=1) + 1
    for j in range(len(sd.words)):
        (x0, y0), (x1, y1) = np.maximum(lo[j], 0), np.minimum(hi[j], [sd.width, sd.height])
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        rect = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(rect, [(sd.words[j] - [x0, y0]).reshape(-1, 1, 2)], 1)
        m = rect.view(bool) & ink[y0:y1, x0:x1]
        if not m.any() or (m & near_run[y0:y1, x0:x1]).any():
            continue
        a = acc[kinds[int(sd.kinds[j])]]
        nl = int((m & long[y0:y1, x0:x1]).sum())
        a[0] += 1
        a[1] += nl > 0
        a[2] += int(m.sum())
        a[3] += nl
res = {k: {"words with no run line within 3 px": a[0], "of them with stroke-long ink": a[1],
           "share of their ink stroke-long": round(a[3] / max(a[2], 1), 4)} for k, a in acc.items()}
print(json.dumps(res, indent=1))
(OUT / "glyphline.json").write_text(json.dumps(res, indent=1))
