"""The connected areas where two published versions of a series' zoning differ (parcels.fgb, by lot ID), largest
first: each with its lot count, its (before, after) codes and its box (EPSG:2227 ft), for release_check.py.

    uv run python scripts/incidents/changed_areas.py BEFORE.fgb AFTER.fgb [--top N] [--gap FT]

Lots whose code differs (blank counts as a code) and lie within --gap ft of each other form one area.  Each area also
counts its lots whose code the April 2000 table (data/solver/lots/table.parquet, by lot ID, in the one spelling) backs
before and after: a fair yardstick for the 1996-2000 scan, the same drawing.  Prints JSON lines; writes nothing.
"""
import argparse
import json
from collections import Counter

import geopandas as gpd
import pandas as pd

from sfzmaps.solver import paths
from sfzmaps.solver.codes import canon


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--gap", type=float, default=60.0)
    a = ap.parse_args()
    b = gpd.read_file(a.before)[["blklot", "code", "geometry"]]
    n = gpd.read_file(a.after)[["blklot", "code"]]
    m = b.merge(n, on="blklot", suffixes=("_before", "_after"))
    cb, ca = m["code_before"].fillna(""), m["code_after"].fillna("")
    changed = m[cb != ca].copy()
    t = pd.read_parquet(paths.LOTS_DIR / "table.parquet", columns=["blklot", "zoning"])
    table = {str(k): canon(str(v)) for k, v in zip(t["blklot"], t["zoning"])}
    tc = changed["blklot"].astype(str).map(table)
    changed["backs_before"] = changed["code_before"].fillna("").map(canon) == tc
    changed["backs_after"] = changed["code_after"].fillna("").map(canon) == tc
    print(json.dumps({"lots": len(m), "changed": len(changed),
                      "only_before": int((~b["blklot"].isin(n["blklot"])).sum()),
                      "only_after": int((~n["blklot"].isin(b["blklot"])).sum()),
                      "table_backs_before": int(changed["backs_before"].sum()),
                      "table_backs_after": int(changed["backs_after"].sum())}))
    if not len(changed):
        return
    blobs = gpd.GeoDataFrame(geometry=list(changed.geometry.buffer(a.gap / 2).union_all().geoms
                                           if changed.geometry.buffer(a.gap / 2).union_all().geom_type == "MultiPolygon"
                                           else [changed.geometry.buffer(a.gap / 2).union_all()]), crs=b.crs)
    blobs["area_id"] = range(len(blobs))
    hit = gpd.sjoin(changed.set_geometry(changed.geometry.representative_point()), blobs, predicate="within")
    rows = []
    for aid, g in hit.groupby("area_id"):
        pairs = Counter(zip(g["code_before"].fillna("-"), g["code_after"].fillna("-"))).most_common(3)
        x0, y0, x1, y1 = changed.loc[g.index].total_bounds
        rows.append({"lots": len(g), "pairs": [[p, q, k] for (p, q), k in pairs],
                     "table_backs_before": int(changed.loc[g.index, "backs_before"].sum()),
                     "table_backs_after": int(changed.loc[g.index, "backs_after"].sum()),
                     "box": [round(v, 1) for v in (x0, y0, x1, y1)]})
    for r in sorted(rows, key=lambda r: -r["lots"])[:a.top]:
        print(json.dumps(r))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
