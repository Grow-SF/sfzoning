"""How far apart two runs' prob6 fields are, per sheet: the correction displacement (Gc, page px, on the fit's grid)
of run A minus run B: largest, 99th percentile and median over grid points.
usage: field_diff.py RUN_A RUN_B SERIES"""
import sys
from pathlib import Path

import numpy as np

a, b, ser = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
print(f"{'sheet':26s} {'max':>6s} {'p99':>6s} {'median':>7s}  (px)")
for d in sorted((a / ser).iterdir()):
    fa, fb = d / f"confit_{d.name}_prob6.npz", b / ser / d.name / f"confit_{d.name}_prob6.npz"
    if not (fa.exists() and fb.exists()):
        continue
    ga, gb = np.load(fa)["Gc"], np.load(fb)["Gc"]
    m = np.hypot(*(ga - gb))
    print(f"{d.name:26s} {m.max():6.2f} {np.percentile(m, 99):6.2f} {np.median(m):7.3f}")
