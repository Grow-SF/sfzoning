"""Issue #71: score each held-out sheet with the page pass alone ("coarse") and with both passes ("map"), with the
same scores — does the pixel pass help or hurt?

    uv run python scripts/spikes/issue71_compare_passes.py
"""

import numpy as np

from sfzmaps.frame import data, scores

KEYS = ("log_loss_frame_band", "log_loss_map_band", "log_loss_all_ink", "frame_called_map", "map_called_frame",
        "edge_median_px", "edge_hd95_px")

print(f"{'sheet':40s} {'pass':6s} " + " ".join(f"{k[:16]:>16s}" for k in KEYS))
for sid in data.labelled_sheets():
    f = data.OUT / sid[0] / sid[1] / "map_prob.npz"
    if not f.exists():
        continue
    z = np.load(f)
    lum = data.load_luminance(data.page_path(*sid))
    polygons, holes = data.load_label(*sid)
    mask = data.map_mask(polygons, lum.shape, holes=holes)
    for name in ("coarse", "map"):
        s = scores.sheet_scores(z[name].astype(np.float32), mask, lum)
        print(f"{sid[0] + '/' + sid[1][-12:]:40s} {name:6s} " + " ".join(f"{s[k]:16.3f}" for k in KEYS))
