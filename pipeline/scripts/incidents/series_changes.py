"""Per series, how a rebuild changed its published map: the lots whose code or category differs between two copies of
the published outputs (BEFORE_OUT/<series>/districts/parcels.fgb against data/out's), and whether the districts and
unknown codes are byte for byte the same.

    uv run python scripts/incidents/series_changes.py BEFORE_OUT [SERIES ...]

Prints one line per series; reads only."""
import sys
from pathlib import Path

import geopandas as gpd

from sfzmaps import config
from sfzmaps.districts.series import SERIES

before = Path(sys.argv[1])
names = sys.argv[2:] or [s for s, i in SERIES.items() if i.role == "use"] + ["today"]
for s in names:
    a, b = before / s / "districts", config.OUT_DIR / s / "districts"
    if not (a / "parcels.fgb").exists() or not (b / "parcels.fgb").exists():
        print(f"{s:22s} missing")
        continue
    same = {f: (a / f).read_bytes() == (b / f).read_bytes() for f in ("parcels.fgb", "districts.fgb",
                                                                       "unknown_codes.json")}
    if same["parcels.fgb"]:
        print(f"{s:22s} unchanged (parcels, districts and unknown codes byte for byte: {all(same.values())})")
        continue
    pa = gpd.read_file(a / "parcels.fgb", ignore_geometry=True)[["blklot", "code", "category"]]
    pb = gpd.read_file(b / "parcels.fgb", ignore_geometry=True)[["blklot", "code", "category"]]
    m = pa.merge(pb, on="blklot", how="outer", suffixes=("_b", "_a"), indicator=True)
    code = (m["code_b"].fillna("") != m["code_a"].fillna("")).sum()
    cat = (m["category_b"].fillna("") != m["category_a"].fillna("")).sum()
    print(f"{s:22s} changed: {code:,} lots' code, {cat:,} lots' category of {len(pb):,} "
          f"(only before {int((m['_merge'] == 'left_only').sum())}, only after {int((m['_merge'] == 'right_only').sum())})")
