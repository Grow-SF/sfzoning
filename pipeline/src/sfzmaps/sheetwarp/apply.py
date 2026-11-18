"""Warp a sheet with its correction."""
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from sfzmaps import reading, warp
from sfzmaps.sheetwarp.sheet import Sheet
from sfzmaps.transform import Correction, SheetTransform

WARP_FILES = ("warped.tif", "warped.footprint.geojson", "warped_paper.tif", "warped_paper.footprint.geojson")


class _PlacementForTheWarp:
    """Today's placement for the control-point grid and the output resolution (so the warp is exactly today's), and
    the corrected placement for the footprint."""

    def __init__(self, today: SheetTransform, corrected: SheetTransform):
        self.today, self.corrected = today, corrected

    def dense_grid(self, shape, step=150):
        return self.today.dense_grid(shape, step)

    def median_scale_ft_per_px(self, shape, step=150):
        return self.today.median_scale_ft_per_px(shape, step)

    def to_world(self, px):
        return self.corrected.to_world(px)


def _resampled_by(correction: Correction, interpolation: int):
    """The corrected placement sends map point w to page pixel p0 + u(p0), p0 = today's to_pixel(w).  So pixel p
    takes the scan at p + u(p): the scan resampled by the correction alone, then warped as today.  Where u is 0 this
    is an exact copy."""
    G, sp, o = correction.grid, float(correction.spacing), np.asarray(correction.origin, float)

    def resample(rgba: np.ndarray) -> np.ndarray:
        h, w = rgba.shape[:2]
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        coords = [(ys - o[1]) / sp, (xs - o[0]) / sp]
        ux, uy = (map_coordinates(G[k], coords, order=1, mode="constant", cval=0.0).astype(np.float32) for k in (0, 1))
        return cv2.remap(rgba, xs + ux, ys + uy, interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return resample


def warp_with_correction(sheet: Sheet, correction: Correction | None, out_dir: Path) -> None:
    """Write warped.tif and warped_paper.tif, with their footprints, into `out_dir`: the scan resampled by the
    correction (linear for the plain warp, cubic for paper) and warped as today, the footprint taken to the map
    with the corrected placement.  Without a correction, the sheet's four warp files are copied as they are
    (today's placement, already warped)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if correction is None:
        for f in WARP_FILES:
            shutil.copyfile(sheet.folder / f, out_dir / f)
        return
    today = sheet.transform
    corrected = SheetTransform(today.gcps, today.smoothing, correction)
    placement = _PlacementForTheWarp(today, corrected)
    boxes = reading.load_page(sheet.folder / "page.json").mask_boxes
    for paper in (False, True):
        warp.warp_sheet(sheet.folder / "page.png", placement, boxes,
                        out_dir / ("warped_paper.tif" if paper else "warped.tif"), paper=paper,
                        resample=_resampled_by(correction, cv2.INTER_CUBIC if paper else cv2.INTER_LINEAR))
