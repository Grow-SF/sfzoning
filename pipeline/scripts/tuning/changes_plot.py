"""changes_plot.py SERIES SHEET [TAG]: modern blocks, scored corners, real-change offsets (x5), no-equivalent hulls."""
import sys, json, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from shapely.geometry import shape, box
from pathlib import Path
from _common import blocks_corners, out_dir
from sfzmaps import config
from sfzmaps.transform import SheetTransform
D = out_dir()
series, sheet = sys.argv[1], sys.argv[2]; tag = sys.argv[3] if len(sys.argv) > 3 else ""
d = config.WORK_DIR / series / sheet
blocks, corners = blocks_corners()
ch = json.loads((d / "changes.geojson").read_text())
rf = json.loads((d / "refine.json").read_text())
t = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
sc = [m for m in rf["matches"] if m["label"] in ("inlier", "outlier")]
W = np.array([[m["wx"], m["wy"]] for m in sc]); dist = np.array([m["dist_ft"] for m in sc])
lo, hi = W.min(0) - 1500, W.max(0) + 1500
fig, ax = plt.subplots(figsize=(14, 14))
for g in blocks[blocks.intersects(box(*lo, *hi))].geometry:
    for p in getattr(g, "geoms", [g]):
        x, y = p.exterior.xy; ax.plot(x, y, color="0.75", lw=0.4)
for f in ch["features"]:
    if f["properties"]["kind"] == "no_equivalent":
        g = shape(f["geometry"]); x, y = g.exterior.xy; ax.fill(x, y, color="orange", alpha=0.25, lw=0)
s = ax.scatter(W[:, 0], W[:, 1], c=dist, s=5, cmap="RdYlGn_r", vmin=0, vmax=12)
rc = [f for f in ch["features"] if f["properties"]["kind"] == "real_change"]
if rc:
    M = np.array([[f["properties"]["modern_x"], f["properties"]["modern_y"], f["properties"]["dx_ft"], f["properties"]["dy_ft"]] for f in rc])
    ax.quiver(M[:, 0], M[:, 1], M[:, 2], M[:, 3], color="magenta", angles="xy", scale_units="xy", scale=1/5, width=0.002)
    ax.scatter(M[:, 0], M[:, 1], s=14, facecolors="none", edgecolors="magenta")
plt.colorbar(s, ax=ax, shrink=0.6, label="residual ft (scored)")
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
ax.set_title(f"{series}/{sheet} {tag}: real change (magenta, offset x5), no-equivalent (orange), scored corners by residual")
fn = D / f"changes_{series}_{sheet}{('_' + tag) if tag else ''}.png"; fig.tight_layout(); fig.savefig(fn, dpi=60); print(fn, len(rc))
