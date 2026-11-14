"""The area each 2000 sheet is measured on when it is fitted as the approved run fitted it (AS_APPROVED_2000,
warp_common.py PART()): each sheet keeps the part of its drawing (the largest polygon of its warped.footprint.geojson,
buffer 0) where no other sheet whose drawing also covers the point has its drawing's centre nearer.  From the
pipeline's footprints only.  usage: measured_area_2000.py OUT.json"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import shapely
from shapely.ops import unary_union

W = Path(os.environ.get("WARP_WORK", Path(__file__).resolve().parents[3] / "data" / "work")) / "2000-digital-zoning"
fp = {}
for i in range(1, 14):
    nn = f"{i:02d}"
    g = shapely.from_geojson((W / f"2000-digital-zoning-p{nn}" / "warped.footprint.geojson").read_text())
    fp[nn] = max(shapely.get_parts(g), key=lambda p: p.area).buffer(0)
cen = {nn: np.array(p.centroid.coords[0]) for nn, p in fp.items()}
big = unary_union(list(fp.values())).envelope.buffer(1e5)


def closer(j, i):
    """The half of the plane nearer centre j than centre i (as mergecity.py)."""
    ci, cj = cen[i], cen[j]
    m, d = (ci + cj) / 2, cj - ci
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    L = 1e6
    return shapely.Polygon([m + n * L, m + n * L + d / np.linalg.norm(d) * L, m - n * L + d / np.linalg.norm(d) * L, m - n * L])


part = {}
for i in fp:
    own = fp[i]
    for j in fp:
        if j != i:
            own = own.difference(fp[j].intersection(closer(j, i)))
    part[i] = own
json.dump({k: v.wkt for k, v in part.items()}, open(sys.argv[1], "w"))
print("parts:", {k: round(v.area / 1e6, 1) for k, v in part.items()}, "M sqft")
