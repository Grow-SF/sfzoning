"""Make ($0): per sheet, textboxes2_<OUT>.json = the turned pass the chain uses (textboxes2_cw.json) plus the zoning-code
boxes of the other turned passes (cwpa, cwpl, pa, pb, pd, pl) whose text is one of the legend's codes, that lie on the
map, that overlap no cw box and hold no print we already have (the codes found but lost, zlost.py); one box per place.
usage: zmerge.py OUT"""
import json
import re
import sys

import numpy as np
import shapely

import truth
from sfzmaps import config
from sfzmaps.transform import SheetTransform

LEGEND = {"C-2", "C-3-G", "C-3-O", "C-3-R", "C-3-S", "C-M", "CCB", "CRNC", "CVR", "M-1", "M-2", "MB-O", "NC-1", "NC-2", "NC-3", "NC-S",
          "NCD", "P", "RC-3", "RC-4", "RED", "RH-1", "RH-1(D)", "RH-1(S)", "RH-2", "RH-3", "RM-1", "RM-2", "RM-3", "RM-4", "RSD", "SLI",
          "SLR", "SPD", "SSO"}
OUT = sys.argv[1]
LR = np.load(truth.D / "labread2_cwfwxne.npz")
added = 0
for nn in [f"{i:02d}" for i in range(1, 14)]:
    d = truth.K / f"p{nn}"
    W = config.WORK_DIR / "2000-digital-zoning" / f"2000-digital-zoning-p{nn}"
    t = SheetTransform.from_json(json.loads((W / "transform.json").read_text()))
    drawn = shapely.from_geojson((W / "warped.footprint.geojson").read_text())
    m = LR["sheet"] == nn
    px = t.to_pixel(np.c_[LR["x"][m], LR["y"][m]]) if m.any() else np.zeros((0, 2))
    cw = json.load(open(d / "textboxes2_cw.json")) if (d / "textboxes2_cw.json").exists() else []
    first = json.load(open(d / "textboxes_gpt-6.1-terra.json"))
    taken = [b["box"] for b in cw + first if b.get("kind") == "zoning code" and re.sub(r"\s+", "", b.get("text", "").upper()) in LEGEND]
    out = list(cw)
    for v in ("cwpa", "cwpl", "pa", "pb", "pd", "pl"):
        f = d / f"textboxes2_{v}.json"
        for b in (json.load(open(f)) if f.exists() else []):
            c = re.sub(r"\s+", "", b.get("text", "").upper())
            if b.get("kind") != "zoning code" or c not in LEGEND or "box" not in b:
                continue
            x0, y0, x1, y1 = b["box"]
            wc = t.to_world(np.array([[(x0 + x1) / 2, (y0 + y1) / 2]], float))[0]
            if not shapely.contains(drawn, shapely.Point(wc)):
                continue
            if len(px) and ((px[:, 0] >= x0 - 4) & (px[:, 0] <= x1 + 4) & (px[:, 1] >= y0 - 4) & (px[:, 1] <= y1 + 4)).any():
                continue
            if any(not (x1 < u0 or x0 > u1 or y1 < v0 or y0 > v1) for u0, v0, u1, v1 in taken):
                continue
            nb = dict(b, text=c, edge=b.get("edge", 0), direction=b.get("direction", ""))
            out.append(nb)
            taken.append(b["box"])
            added += 1
    json.dump(out, open(d / f"textboxes2_{OUT}.json", "w"))
print(f"added {added} boxes (found by a pass the chain does not use, lost before) -> textboxes2_{OUT}.json on every sheet")
