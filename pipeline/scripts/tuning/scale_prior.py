"""Compare the street-width scale guess (STREET_FT / street px from the extracted blocks)
with the answer key's ft/px, per sheet.

    uv run python scripts/tuning/scale_prior.py 1948 1960
"""
import json
import sys
import warnings

import numpy as np
from shapely.geometry import shape

from sfzmaps import config, linework, pages, runner
from sfzmaps.transform import SheetTransform

warnings.filterwarnings("ignore")
for series in sys.argv[1:]:
    for page in [p for p in pages.discover() if p.series == series]:
        d = config.WORK_DIR / series / page.sheet_id
        if not (d / "blocks_px.geojson").exists():
            continue
        polys = [shape(f["geometry"]) for f in json.loads((d / "blocks_px.geojson").read_text())["features"]]
        guess = linework.STREET_FT / linework.estimate_street_px(polys)[0]
        key = runner._answer_key_gcps(page, d / "page.png") if runner.answer_key_path(page) else None
        ks = float("nan")
        if key:
            t = SheetTransform(key, config.COARSE_TPS_SMOOTHING)
            c = np.median([[g.px, g.py] for g in key], 0)
            J = t.to_world(np.array([c, c + [10, 0]]))
            ks = np.hypot(*(J[1] - J[0])) / 10
        print(f"{series}/{page.sheet_id:8s} guess {guess:.2f} ft/px  key {ks:.2f}  ratio {guess / ks:.2f}")
