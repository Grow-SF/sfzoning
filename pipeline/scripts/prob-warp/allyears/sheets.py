"""The sheets of a series the warp corrects, one per line, from the pipeline's data only: every fitted sheet (its
transform.json, warped.tif, blocks_px.geojson and refine.json exist) that is a zoning-use sheet.  The pattern-map
series (reader A, 1921-1948) print no codes, so all their fitted sheets; the others keep the sheets the pipeline's
label reader classes as "use" (districts.labels.sheet_roles: not height-and-bulk, special-use or blank sheets).  This
gives the approved run's 174 sheets.  No model is asked: the labels come from the answer cache or not at all.
usage: sheets.py SERIES"""
import sys

from sfzmaps import config, llm
from sfzmaps.districts import labels
from sfzmaps.districts.series import SERIES

llm.CACHE_ONLY = True
series = sys.argv[1]
need = ("transform.json", "warped.tif", "blocks_px.geojson", "refine.json")
fitted = sorted(d.name for d in (config.WORK_DIR / series).iterdir()
                if d.is_dir() and all((d / f).exists() for f in need))
if SERIES[series].reader != "A":
    roles = labels.sheet_roles(series, fitted)
    fitted = [sh for sh in fitted if roles[sh] == "use"]
print("\n".join(fitted))
