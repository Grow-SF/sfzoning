"""Regenerate the 2000 run's parts.json (each sheet's drawing: the largest polygon of its warped.footprint.geojson,
buffer 0, as overlapmerge.py writes it) from the pipeline's footprints, and compare with surviving parts.json copies.
usage: make_parts.py OUT.json"""
import json
import os
import sys
from pathlib import Path

import shapely

W = Path(__file__).resolve().parents[3] / "data" / "work" / "2000-digital-zoning"
S = Path(os.environ.get("PW_SCRATCH_OLD", "/nonexistent"))   # an old scratch folder holding k*/city*/parts.json
parts = {}
for i in range(1, 14):
    nn = f"{i:02d}"
    g = shapely.from_geojson((W / f"2000-digital-zoning-p{nn}" / "warped.footprint.geojson").read_text())
    parts[nn] = max(shapely.get_parts(g), key=lambda p: p.area).buffer(0).wkt
json.dump(parts, open(sys.argv[1], "w"))
for f in sorted(S.glob("k*/city*/parts.json")):
    other = json.load(open(f))
    same = [k for k in parts if k in other and other[k] == parts[k]]
    close = [k for k in parts if k in other and shapely.from_wkt(other[k]).equals_exact(shapely.from_wkt(parts[k]), 1e-6)]
    print(f"{f.relative_to(S)}: sheets {len(other)}, identical text {len(same)}, equal shape {len(close)}")
