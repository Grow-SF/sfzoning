"""Issue #71: measure every map-area label against the scan.

Every STEP px along each polygon edge, walk outward along the edge's normal and record the distance to the first ink
px (luminance < 128). 0 = the edge sits on ink (on the rule, or cutting map ink); a large distance = the edge is far
inside the rule (or that side has no rule: page edge, notch). Prints a summary per sheet and lists the worst points.

    uv run python scripts/spikes/issue71_label_qa.py
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "data" / "work"
LABELS = ROOT / "inputs" / "frame-labels"
STEP = 150
MAXD = 40


def signed_area(poly):
    x, y = np.array(poly, float).T
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


for j in sorted(LABELS.glob("*/*.json")):
    series, sheet = j.parent.name, j.stem
    lab = json.load(open(j))
    ink = np.asarray(Image.open(WORK / series / sheet / "page.png").convert("L")) < 128
    H, W = ink.shape
    dists = []
    for poly in lab["polygons"]:
        ccw = signed_area(poly) > 0  # image coords (y down): positive area = clockwise on screen
        n = len(poly)
        for i in range(n):
            a, b = np.array(poly[i], float), np.array(poly[(i + 1) % n], float)
            L = np.linalg.norm(b - a)
            t = (b - a) / L
            nrm = np.array([t[1], -t[0]]) if ccw else np.array([-t[1], t[0]])  # outward
            for s in np.arange(STEP / 2, L, STEP):
                p = a + t * s
                d = MAXD
                for k in range(MAXD):
                    x, y = np.round(p + nrm * k).astype(int)
                    if not (0 <= x < W and 0 <= y < H):
                        d = -1  # reached the page edge: no rule on this side
                        break
                    if ink[y, x]:
                        d = k
                        break
                dists.append((d, i, int(p[0]), int(p[1])))
    ds = np.array([d for d, *_ in dists])
    ruled = ds[(ds >= 0) & (ds < MAXD)]
    on = int((ds == 0).sum())
    far = int((ds >= 9).sum() - (ds == MAXD).sum())
    print(f"{series}/{sheet}: {len(ds)} pts; to first ink: median {np.median(ruled) if len(ruled) else -1:.0f} px; "
          f"on ink {on}; 9-39 px {far}; none within {MAXD} {int((ds == MAXD).sum())}; page edge {int((ds == -1).sum())}")
    bad = [x for x in dists if x[0] == 0 or 9 <= x[0] < MAXD]
    for d, i, x, y in bad[:6]:
        print(f"    edge {i} at ({x},{y}): {d} px")
