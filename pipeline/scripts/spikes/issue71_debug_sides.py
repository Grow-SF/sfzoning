"""Issue #71 debugging: why does the shared-print mixture find no frame on the early editions?

For one sheet A:
  1. Locate A's frame lines (debug only: rows/columns where most of the row/column is ink).
  2. Per sibling, the mean agreement a and chance n on each side's frame px, and on city px (ink far from the frame).
  3. The mixture's log-likelihood at its fitted answer vs at "frame = the located frame px" (rates refitted).
     Higher at the located frame => the fit is stuck (optimisation); lower => the model itself prefers no frame.

    uv run python scripts/spikes/issue71_debug_sides.py SERIES SHEET
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import issue71_shared_ink as m  # noqa: E402

series, sa = sys.argv[1:3]
al = json.load(open(m.OUT / series / f"{sa}.align.json"))
sheets = sorted(p.parent.name for p in (m.WORK / series).glob("*/page.png"))
kinds = {}
for line in open(m.ROOT / "inputs" / "page_kinds.csv").read().splitlines()[1:]:
    s, sh, k, _ = line.split(",", 3)
    if s == series:
        kinds[sh] = k
inks = {s: m.load_ink(m.WORK / series / s / "page.png", m.FINE) for s in sheets}
A = inks[sa]
H, W = A.shape
ink = A > 0.25
rowf = ink.mean(1)
colf = ink.mean(0)
rows = np.where(rowf > 0.6)[0]
cols = np.where(colf > 0.6)[0]
print("frame rows", rows.tolist()[:20], "...", "frame cols", cols.tolist()[:20])
F = np.zeros_like(ink)
side = np.full(A.shape, "", dtype=object)
for r in rows:
    F[r, :] |= ink[r, :]
    side[r, ink[r, :]] = "top" if r < H / 2 else "bottom"
for c in cols:
    F[:, c] |= ink[:, c]
    side[ink[:, c], c] = np.where(side[ink[:, c], c] == "", "left" if c < W / 2 else "right", side[ink[:, c], c])
near = cv2.dilate(F.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
C = ink & ~near
print(f"frame px {F.sum()}, city px {C.sum()}, ink px {ink.sum()}")

k = 2 * m.NULL_HALF + 1
sibs = [s for s in sheets if s != sa]
AG, NL = [], []
print(f"{'sibling':>8} kind   " + "  ".join(f"{n:>13}" for n in ("top a/n", "bottom a/n", "left a/n", "right a/n", "city a/n")))
for sb in sibs:
    D = m.agreement(sa, sb, np.array(al[sb]["M"]), inks)
    N = cv2.blur(D, (k, k))
    AG.append(D[ink])
    NL.append(N[ink])
    cells = []
    for nm in ("top", "bottom", "left", "right"):
        sel = F & (side == nm)
        cells.append(f"{D[sel].mean():.2f}/{N[sel].mean():.2f}" if sel.any() else "   -   ")
    cells.append(f"{D[C].mean():.2f}/{N[C].mean():.2f}")
    print(f"{sb[-6:]:>8} {kinds.get(sb, '?')[:5]:5}  " + "  ".join(f"{c:>13}" for c in cells))

ag = np.stack(AG, -1)
nl = np.clip(np.stack(NL, -1), 1e-4, 1 - 1e-4)
w = A[ink]
isF = F[ink]


def loglik(r):
    eps = 1e-4
    wr, wq = w * r, w * (1 - r)
    f = np.clip((wr[:, None] * ag).sum(0) / (wr.sum() + 1e-9), eps, 1 - eps)
    d = np.clip(((wq[:, None] * (ag - nl)).sum(0)) / ((wq[:, None] * (1 - nl)).sum(0) + 1e-9), 0, 1 - eps)
    pc = np.clip(1 - (1 - nl) * (1 - d), eps, 1 - eps)
    pi = np.clip(wr.sum() / w.sum(), eps, 1 - eps)
    lf = np.log(pi) + (ag * np.log(f) + (1 - ag) * np.log(1 - f)).sum(1)
    lc = np.log(1 - pi) + (ag * np.log(pc) + (1 - ag) * np.log(1 - pc)).sum(1)
    return float((w * np.logaddexp(lf, lc)).sum()), f, d


r_fit, f_fit, d_fit, pi_fit = m.fit_mixture(ag, nl, w)
L_fit, _, _ = loglik(r_fit)
L_hand, f_h, d_h = loglik(isF.astype(float))
print(f"fitted: pi {pi_fit:.4f}  loglik {L_fit:.1f}")
print(f"frame=located px: loglik {L_hand:.1f}  (f per sibling {np.round(f_h, 2).tolist()})")
print("=> fit stuck (optimisation)" if L_hand > L_fit else "=> the model prefers no frame (model/evidence)")
