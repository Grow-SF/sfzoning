"""Warp one 2000 sheet with the corrected fit (today's fit + correction map) into the run k1 mirror folder:
warped.tif and warped_paper.tif (+ footprints).  Run with the patch on (SFZ_CORR=1, PYTHONPATH=corrpatch,
SFZMAPS_DATA_DIR=data_k1).

How: the corrected fit sends map point w to page pixel p0 + u(p0), p0 = today's to_pixel(w).  So the scan is
first resampled by the correction map alone (pixel p takes the scan at p + u(p); where u is 0 this is an exact
copy), then warped exactly as today (warp.warp_sheet, today's GCP grid, gdalwarp -tps).  The footprint is traced
on the original page and taken to the map with the corrected to_world.
usage: warpK.py NN [plain|paper|both]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from sfzmaps import config, reading, warp
from sfzmaps.transform import SheetTransform

NN = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else "both"
import os  # noqa: E402
D = config.WORK_DIR / os.environ["SERIES"] / NN
assert "data_ay" in str(D), D
t = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
assert hasattr(t, "_corr"), "patch not loaded or no correction"
G, sp, o = t._corr


class Proxy:
    """today's fit for the GCP grid and the output resolution; the corrected fit for the footprint."""
    def dense_grid(self, shape, step=150):
        return t._base.dense_grid(shape, step)

    def median_scale_ft_per_px(self, shape, step=150):
        return t._base.median_scale_ft_per_px(shape, step)

    def to_world(self, px):
        return t.to_world(px)


_make = warp.make_rgba
interp = {}


def make_rgba_corrected(*a, **k):
    rgba = _make(*a, **k)
    h, w = rgba.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ux = map_coordinates(G[0], [(ys - o[1]) / sp, (xs - o[0]) / sp], order=1, mode="constant", cval=0.0).astype(np.float32)
    uy = map_coordinates(G[1], [(ys - o[1]) / sp, (xs - o[0]) / sp], order=1, mode="constant", cval=0.0).astype(np.float32)
    moved = np.hypot(ux, uy) > 0
    print(f"p{NN}: pixels with a correction {moved.mean():.1%}, max {np.hypot(ux, uy).max():.1f} px", flush=True)
    out = cv2.remap(rgba, xs + ux, ys + uy, interp["mode"], borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    same = np.array_equal(out[~moved], rgba[~moved])
    print(f"p{NN}: uncorrected pixels copied exactly: {same}", flush=True)
    return out


warp.make_rgba = make_rgba_corrected
png = D / "page.png"
boxes = reading.load_page(D / "page.json").mask_boxes
for kind in (["plain", "paper"] if which == "both" else [which]):
    out = D / ("warped_paper.tif" if kind == "paper" else "warped.tif")
    fp = out.with_name(out.stem + ".footprint.geojson")
    for f in (out, fp):
        if f.is_symlink():
            f.unlink()          # the mirror's link to the real file, never the real file
    interp["mode"] = cv2.INTER_CUBIC if kind == "paper" else cv2.INTER_LINEAR
    warp.warp_sheet(png, Proxy(), boxes, out, paper=(kind == "paper"))
    print(f"p{NN}: wrote {out.name}", flush=True)
