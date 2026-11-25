# One-off check, not pipeline code: does the rough placement of 1921/094348509 fail because of the scale hint the finder's blocks give? Run from pipeline/.
"""Scratch test: does the rough placement of 1921/094348509 go wrong because of the scale hint the finder's blocks give?
Same labels, same reference; only the hint changes."""
import json
from pathlib import Path
from sfzmaps import anchors, reading, linework
from sfzmaps.sources import datasf
from shapely.geometry import shape
T = Path("data/find/trial/all-pieces"); s = "1921/20180102094348509"
ref = datasf.load_reference()
labels = reading.load_labels(T / "traced/work" / s / "labels.json")
cands = anchors.match_labels(labels, ref)
print("labels", len(labels), "candidates", len(cands), "accepted scale factor", anchors.SCALE_HINT_FACTOR)
for arm in ("traced", "finder"):
    polys = [shape(f["geometry"]) for f in json.loads((T / arm / "work" / s / "blocks_px.geojson").read_text())["features"]]
    px, measured = linework.estimate_street_px(polys)
    print(arm, "street width %.2f px" % px, "measured", measured, "hint %.2f ft/px" % (linework.STREET_FT / px))
for hint in (None, 3.0, 3.81, 4.2, 4.5, 4.70, 4.82, 5.2):
    fit = anchors.rough_fit(cands, scale_hint=hint)
    if fit is None:
        print("hint", hint, "-> no fit"); continue
    print("hint", hint, "-> scale %.2f ft/px, rotation %.1f deg, inliers %d" % (fit.scale_ft_per_px, fit.rotation_deg, len(fit.inliers)))
