"""Issue #71 spike: agreement a_b and chance n_b at chosen FINE px of one sheet, per sibling.

    uv run python scripts/spikes/issue71_probe.py SERIES SHEET x,y [x,y ...]
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import issue71_shared_ink as m  # noqa: E402

series, sa = sys.argv[1:3]
pts = [tuple(int(v) for v in p.split(",")) for p in sys.argv[3:]]
al = json.load(open(m.OUT / series / f"{sa}.align.json"))
sheets = sorted(p.parent.name for p in (m.WORK / series).glob("*/page.png"))
inks = {s: m.load_ink(m.WORK / series / s / "page.png", m.FINE) for s in sheets}
A = inks[sa]
k = 2 * m.NULL_HALF + 1
print("A ink at pts", [round(float(A[y, x]), 2) for x, y in pts])
for sb in [s for s in sheets if s != sa]:
    D = m.agreement(sa, sb, np.array(al[sb]["M"]), inks)
    N = cv2.blur(D, (k, k))
    print(sb[-5:], " ".join(f"{D[y, x]:.2f}/{N[y, x]:.2f}" for x, y in pts))
