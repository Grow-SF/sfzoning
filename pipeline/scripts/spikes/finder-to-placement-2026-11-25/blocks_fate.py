# One-off check, not pipeline code: how many of the year's blocks on a trial sheet the finder gives alone, joins to others, or loses at the rim.
"""Scratch check (not pipeline code): of the year's blocks that fall on a trial sheet (one point each, from the lots),
how many the finder gives as a block of their own, how many are joined to others, how many are lost at the rim."""
import json, sys
from collections import Counter
import cv2, numpy as np
from sfzmaps.find import areas, pages, read, wholesheet, placeblocks, trial

run = pages.OUT / "runs" / "all-pieces"
root = trial.arm_root("all-pieces", "finder")
for s in sys.argv[1:]:
    series, sheet = s.split("/")
    sid = (series, sheet)
    work = root / "work" / series / sheet
    img = cv2.imread(str(work / "page.png"), cv2.IMREAD_UNCHANGED)
    ch = read.load(wholesheet.paths(run, sid)[0])
    keep = pages.load_keep(sid, img.shape[:2])
    lab = areas.areas(ch[0], keep)
    cut = {int(g) for g in placeblocks.edge_cut(lab, keep)}
    d = json.loads((work / "blocks_px.geojson").read_text())
    kept = {int(f["properties"]["area"]) for f in d["features"]}
    pts = np.asarray(placeblocks.lot_block_points(sid, lab.shape), float).reshape(-1, 2)
    xi = np.rint(pts[:, 0]).astype(int); yi = np.rint(pts[:, 1]).astype(int)
    on = (xi >= 0) & (xi < lab.shape[1]) & (yi >= 0) & (yi < lab.shape[0])
    xi, yi = xi[on], yi[on]
    inside = keep[yi, xi]
    g = lab[yi, xi]
    per = Counter(int(v) for v in g if v > 0)
    fate = Counter()
    for v, k in zip(g, inside):
        v = int(v)
        if not k:
            fate["outside the frame cut"] += 1
        elif v == 0:
            fate["point falls on ground read as street"] += 1
        elif v in cut:
            fate["in an area cut at the rim"] += 1
        elif v in kept:
            fate["a block of its own" if per[v] == 1 else "joined with other blocks in one outline"] += 1
        else:
            fate["in an area dropped by cleaning"] += 1
    n = int(inside.sum())
    print(f"\n{s}: the year's blocks with their point inside the frame: {n}")
    for k, v in fate.most_common():
        print(f"  {k:42s} {v:4d}  {100 * v / max(n, 1):5.1f}%")
    big = sorted(((c, a) for a, c in per.items() if c > 1 and a in kept), reverse=True)[:5]
    print("  largest joined outlines kept (blocks in each):", [c for c, _ in big],
          "| joined outlines kept:", sum(1 for a, c in per.items() if c > 1 and a in kept))
    # good corners by kind of outline
    ncorner = len(d["corners"])
    print("  finder good corners:", ncorner)
