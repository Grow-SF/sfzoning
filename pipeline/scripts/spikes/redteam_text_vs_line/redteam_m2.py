"""How the existing r30 detectors call in-box ink, split by the line rules: what M2 (in-box ink not on M1's stroke
rule) would lose if a detector stopped calling in-box straight-line ink text."""
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
EXP = runs.EXPERIMENTS_DIR / "detector-overcall"
LINES = EXP / "lines" / paths.SERIES
sets = ("run_not_long", "long_not_run", "both", "neither")
acc = {d: {s: [0, 0] for s in sets} for d in ("trial", "all")}
for name in paths.SHEETS:
    sd = data.sheet_data(paths.SERIES, name, hold_out_validation=False, placement=False)
    ink = data.page_ink(sd)
    with np.load(LINES / f"{name}.npz") as z:
        long = np.unpackbits(z["long"], axis=1, count=sd.width).view(bool)
    run = np.unpackbits(np.load(OUT / f"run-{name}.npz")["run"], axis=1, count=sd.width).view(bool)
    inbox = ink & (data._words(sd, 0, 0, sd.height, sd.width) > 0)
    m = {"run_not_long": inbox & run & ~long, "long_not_run": inbox & long & ~run, "both": inbox & long & run,
         "neither": inbox & ~long & ~run}
    ch = {"trial": cv2.imread(str(EXP / "chance" / "trial" / f"{name}.png"), cv2.IMREAD_GRAYSCALE),
          "all": cv2.imread(str(paths.TEXT_CHANCE_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)}
    for d, c in ch.items():
        for s in sets:
            acc[d][s][0] += int(m[s].sum())
            acc[d][s][1] += int((c[m[s]] >= 128).sum())
res = {d: {s: {"px": a[0], "called_text": round(a[1] / max(a[0], 1), 3)} for s, a in v.items()} for d, v in acc.items()}
tot = sum(acc["trial"][s][0] for s in sets)
res["m2_base_share_run_not_long"] = round(acc["trial"]["run_not_long"][0] / (tot - acc["trial"]["both"][0]
                                                                           - acc["trial"]["long_not_run"][0]), 3)
print(json.dumps(res, indent=1))
(OUT / "m2.json").write_text(json.dumps(res, indent=1))
