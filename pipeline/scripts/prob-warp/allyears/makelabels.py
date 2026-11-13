"""The zoning-code labels for graph7 (labels.json, map feet), from a codes-only run's fits: every zoning code
Terra read on the map part, erased or not (reading a code doesn't depend on erasing it).  The code is the
reading: a code fitted without its dash keeps its dash (RM4 printed -> RM-4).
usage: makelabels.py RUN_DIR SHEET   -> RUN_DIR/graph/labels.json"""
import json
import sys
from pathlib import Path

import numpy as np

from sfzmaps import config
from sfzmaps.transform import SheetTransform

R, SHEET = Path(sys.argv[1]), sys.argv[2]
M = 80
import os  # noqa: E402
t = SheetTransform.from_json(json.loads((config.WORK_DIR / os.environ["SERIES"] / SHEET / "transform.json").read_text()))
FT = json.load(open(R / "graph" / "graph.json"))["ft_per_px"]
import re  # noqa: E402

import pandas as pd  # noqa: E402

# the map's own codes (as published now), to write a code read without its dash the way the map lists it
VOCAB = set(pd.read_parquet(config.WORK_DIR / os.environ["SERIES"] / "districts_raw.parquet").code.dropna())


def canon(c):
    """A code as the map lists it: 'RH2' -> 'RH-2', 'RH1(D)' -> 'RH-1(D)', 'C3O(SD)' -> 'C-3-O(SD)' when that's a
    listed code; anything else as read."""
    if c in VOCAB:
        return c
    for v in VOCAB:
        if v.replace("-", "") == c.replace("-", ""):
            return v
    return c


out = []
for f in json.load(open(R / "fits3.json")):
    if f["kind"] != "zoning code":
        continue
    code = canon(f["candidates"][0] if f.get("source") == "reading without its dash" else f["text"])
    a, b, c, e = [v - M for v in f["box"]]
    poly = t.to_world(np.array([[a, b], [c, b], [c, e], [a, e]], float))
    out.append({"text": code, "poly": np.round(poly, 2).tolist(), "height_ft": round(min(c - a, e - b) * FT, 1),
                "erased": bool(f["ok"])})
json.dump(out, open(R / "graph" / "labels.json", "w"))
print(f"{SHEET}: {len(out)} code labels, {sum(l['erased'] for l in out)} of them erased")
