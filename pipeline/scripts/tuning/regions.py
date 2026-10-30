"""Share within 6/8 ft and real-change / no-equivalent counts by neighbourhood box: uv run python scripts/tuning/regions.py"""
import json, numpy as np
from pyproj import Transformer
from sfzmaps import config
from sfzmaps.transform import SheetTransform
T = Transformer.from_crs('EPSG:2227', 'EPSG:4326', always_xy=True)
R = {"Downtown/FiDi": (37.787, 37.797, -122.410, -122.398), "Russian Hill": (37.797, 37.805, -122.424, -122.412),
     "North Beach": (37.797, 37.806, -122.412, -122.403), "Marina": (37.799, 37.806, -122.444, -122.430),
     "Pacific Hts": (37.788, 37.796, -122.444, -122.425), "Western Addition": (37.776, 37.787, -122.440, -122.420),
     "Van Ness corridor": (37.780, 37.805, -122.4248, -122.4205), "Embarcadero/waterfront": (37.790, 37.808, -122.401, -122.385),
     "Richmond": (37.772, 37.787, -122.51, -122.46)}
for s, sheets in (("1948", ["1", "2-and-3", "3", "4"]), ("1960", ["1", "2", "9"])):
    for sh in sheets:
        d = config.WORK_DIR / s / sh
        if not (d / "refine.json").exists():
            continue
        ms = json.loads((d / "refine.json").read_text())["matches"]
        t = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
        P = np.array([[m["px"] if m["px"] is not None else np.nan, m["py"] if m["py"] is not None else np.nan] for m in ms])
        ok = np.isfinite(P).all(1); Wd = np.full_like(P, np.nan); Wd[ok] = t.to_world(P[ok])
        lo, la = T.transform(Wd[:, 0], Wd[:, 1])
        lab = np.array([m["label"] for m in ms]); dist = np.array([m["dist_ft"] if m["dist_ft"] is not None else np.inf for m in ms])
        for name, (a, b, c, e) in R.items():
            sel = ok & (la >= a) & (la < b) & (lo >= c) & (lo < e)
            sc = sel & np.isin(lab, ["inlier", "outlier"])
            if sel.sum() < 15: continue
            print(f"{s}/{sh:8s} {name:22s} scored {sc.sum():4d} f6 {np.mean(dist[sc] <= 6) if sc.any() else float('nan'):.2f} f8 {np.mean(dist[sc] <= 8) if sc.any() else float('nan'):.2f}  real_change {np.sum(sel & (lab == 'real_change')):3d}  no_eq {np.sum(sel & (lab == 'no_equivalent')):4d}")
