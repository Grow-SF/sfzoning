"""Issue #71 spike debugging: agreement values for one sheet against a few siblings."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import issue71_shared_ink as m  # noqa: E402

ser = sys.argv[1] if len(sys.argv) > 1 else "2000-digital-zoning"
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 4
sheets = sorted(p.parent.name for p in (m.WORK / ser).glob("*/page.png"))
ic = {s: m.load_ink(m.WORK / ser / s / "page.png", m.COARSE) for s in sheets}
iff = {s: m.load_ink(m.WORK / ser / s / "page.png", m.FINE) for s in sheets}
sa = sheets[idx]
A = iff[sa]
mask = A > 0.25
print(sa, "ink frac", mask.mean())
ags = []
for sb in [s for s in sheets if s != sa][:4]:
    M, info = m.pair_transform(sa, sb, ic, iff)
    print(sb, M.round(3).tolist(), info)
    for r in (0, 1, 2):
        m.AGREE_R = r
        ag = m.agreement(sa, sb, M, iff)[mask]
        print("  r", r, "agree mean", round(float(ag.mean()), 3), np.quantile(ag, [0.1, 0.5, 0.9]).round(3))
    ags.append(ag)
mn = np.stack(ags, 1).mean(1)
print("mean-agree quantiles .5 .9 .95 .99", np.quantile(mn, [0.5, 0.9, 0.95, 0.99]).round(3))
