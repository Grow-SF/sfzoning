"""For the changed areas of two published versions of a series (as changed_areas.py finds them): what the April 2000
table says on the changed lots, so a before/after picture can be read against it.

    uv run python scripts/incidents/changed_vs_table.py BEFORE.fgb AFTER.fgb X0 Y0 X1 Y1

Prints, for the lots in the box whose code changed, counts of (before, after, table) by lot ID.  Reads only."""
import sys
from collections import Counter

import geopandas as gpd
import pandas as pd

from sfzmaps.solver import paths
from sfzmaps.solver.codes import canon

before, after = sys.argv[1:3]
bbox = tuple(float(v) for v in sys.argv[3:7])
b = gpd.read_file(before, bbox=bbox)[["blklot", "code"]]
n = gpd.read_file(after, bbox=bbox)[["blklot", "code"]]
m = b.merge(n, on="blklot", suffixes=("_before", "_after"))
m = m[m["code_before"].fillna("") != m["code_after"].fillna("")]
t = pd.read_parquet(paths.LOTS_DIR / "table.parquet")
col = next(c for c in ("code", "zoning", "districtname", "zoning_sim") if c in t.columns)
table = dict(zip(t["blklot"].astype(str), t[col].astype(str)))
rows = Counter((r.code_before, r.code_after, canon(table.get(str(r.blklot), "")) or "-") for r in m.itertuples())
for (p, q, tc), k in rows.most_common(8):
    print(f"{k:5d}  {p} -> {q}   table {tc}")
