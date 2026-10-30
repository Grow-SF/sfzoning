"""Diagnose rough placement per sheet: labels, anchor candidates, rough-fit inliers and the
distance of the rough / final transforms to the owner's answer key.

    uv run python scripts/tuning/misplace_diag.py 1948 1960
"""
import json
import sys
import warnings
from collections import Counter

import numpy as np

from sfzmaps import anchors, config, pages, reading, runner
from sfzmaps.evaluate import answer_key_distances
from sfzmaps.sources import datasf
from sfzmaps.transform import SheetTransform

warnings.filterwarnings("ignore")
ref = datasf.load_reference()
allp = pages.discover()
for series in sys.argv[1:]:
    for page in [p for p in allp if p.series == series]:
        d = config.WORK_DIR / series / page.sheet_id
        if not (d / "labels.json").exists():
            continue
        labels = reading.load_labels(d / "labels.json")
        kinds = Counter(lb.kind for lb in labels)
        cands = anchors.match_labels(labels, ref)
        ck = Counter(c.kind for c in cands)
        fit = anchors.rough_fit(cands)
        from shapely.geometry import shape
        from sfzmaps import linework
        polys = [shape(f["geometry"]) for f in json.loads((d / "blocks_px.geojson").read_text())["features"]]
        hint = linework.STREET_FT / linework.estimate_street_px(polys)
        fit2 = anchors.rough_fit(cands, scale_hint=hint)
        fit3 = anchors.street_snap(fit2, labels, ref) if fit2 else None
        key = runner._answer_key_gcps(page, d / "page.png") if runner.answer_key_path(page) else None
        row = [f"{series}/{page.sheet_id:8s}", f"labels {dict(kinds)}", f"cands b{ck['block']} s{ck['street_pair']}"]
        if fit is None:
            row.append("rough: none")
        else:
            ik = Counter(c.kind for c in fit.inliers)
            row.append(f"rough inl b{ik['block']} s{ik['street_pair']} scale {fit.scale_ft_per_px:.2f} rot {fit.rotation_deg:.1f}")
            if key:
                g = [x for x in fit.transform.gcps]
                for sm in (1.0, config.COARSE_TPS_SMOOTHING):
                    t = SheetTransform(g, sm)
                    row.append(f"key@rough{sm:g} {np.median(answer_key_distances(t, key)):.0f}")
        for nm, f in (("hint", fit2), ("hint+snap", fit3)):
            if f is None:
                row.append(f"{nm}: none")
            elif key:
                row.append(f"key@{nm} {np.median(answer_key_distances(f.transform, key)):.0f} (scale {f.scale_ft_per_px:.2f})")
        tj = d / "transform.json"
        if key and tj.exists():
            row.append(f"key@final {np.median(answer_key_distances(SheetTransform.from_json(json.loads(tj.read_text())), key)):.1f}")
        print("  ".join(row))
