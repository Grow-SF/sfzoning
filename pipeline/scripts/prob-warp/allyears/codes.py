"""The code check for a series that has no person-checked codes file (2000's codes_final.json): every use-district
code the pipeline read on each sheet (district_labels.json, the cached reads, no API), with the series'
label_fixes.csv applied, as rows {sid, x, y, code, final}.  usage: codes.py SERIES OUT.json"""
import json
import sys

from sfzmaps import config
from sfzmaps.districts import labels as L

series, out = sys.argv[1:3]
rows = []
for d in sorted(p for p in (config.WORK_DIR / series).iterdir() if (p / "district_labels.json").exists()):
    labs = L.load_labels(d / "district_labels.json")
    fx = L.label_fixes(series, d.name)
    if fx:
        labs = L.apply_label_fixes(labs, fx)
    sid = d.name.rsplit("-", 1)[-1]
    n = 0
    for lb in labs:
        if lb.kind != "use":
            continue
        rows.append({"sid": sid, "x": lb.x, "y": lb.y, "code": lb.code, "final": lb.code, "how": "pipeline read"})
        n += 1
    print(f"{d.name}: {n} use codes ({len(fx)} label fixes)")
json.dump(rows, open(out, "w"), indent=0)
print(len(rows), "codes ->", out)
