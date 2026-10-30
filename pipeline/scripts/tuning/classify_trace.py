"""Trace classify_changes rounds on a sheet's refine_raw.json: real-change count per round.

    uv run python scripts/tuning/classify_trace.py 1948 1
"""
import json
import sys
import warnings

import numpy as np

from sfzmaps import config, refine
from sfzmaps.runner import _result_load
from sfzmaps.sources import datasf

warnings.filterwarnings("ignore")
series, sheet = sys.argv[1], sys.argv[2]
ref = datasf.load_reference()
res = _result_load(json.loads((config.WORK_DIR / series / sheet / "refine_raw.json").read_text()), ref)
orig = refine._real_change
hist = []


def traced(*a, **k):
    rc, cp = orig(*a, **k)
    hist.append(rc.copy())
    return rc, cp


refine._real_change = traced
out = refine.classify_changes(res)
for i, h in enumerate(hist):
    diff = int((h != hist[i - 1]).sum()) if i else int(h.sum())
    print(f"round {i}: real_change {int(h.sum())}, changed vs previous {diff}")
print("classify_converged", out.classify_converged)
