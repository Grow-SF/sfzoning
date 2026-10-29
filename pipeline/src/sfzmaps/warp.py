"""Warp a scanned sheet into a transparent-paper RGBA GeoTIFF in EPSG:2227.

Pipeline: PNG -> RGBA (alpha 0 for paper, margins and mask boxes) -> intermediate GeoTIFF
carrying a dense GCP grid sampled from our SheetTransform -> gdalwarp -tps.  Because the
GCPs are a dense sampling (every 150 px) of the *smooth* transform, GDAL's thin-plate
spline only interpolates our field; it never re-fits raw, noisy matches.

Pixel convention: SheetTransform pixels are pixel-centre indices (cv2 style, (0, 0) is the
centre of the top-left pixel); GDAL GCPs are measured from the top-left corner of the
raster, so every GCP pixel coordinate is shifted by +0.5.

Output resolution: the median local scale of the transform in ft/px, taken from the
world distance between adjacent dense-grid points divided by their pixel distance.  It is
passed to gdalwarp with -tr (otherwise GDAL picks its own, usually coarser, size) so
source detail is preserved, no more, no less.  (1948 pages: about 0.3-0.5 ft/px.)

Output format: tiled, DEFLATE-compressed 8-bit RGBA (grayscale scans replicate gray into
R, G, B; RGBA is the simplest form every downstream tiler accepts).

Paper transparency (alpha):
  * gray / 1-bit pages: alpha ramps with darkness: 0 for gray >= PAPER_GRAY (200), 255 for
    gray <= PAPER_GRAY - RAMP_GRAY (150).  1-bit pages are just 0/255; anti-aliased scans get
    soft edges.
  * colour pages (1948 photos of a bound atlas): paper = luminance above the Otsu threshold
    (computed inside the neatline, so blue cloth and gutter do not skew it) AND HSV
    saturation < 0.15.  alpha = max(darkness ramp below Otsu over RAMP_COLOR levels, colour
    ramp from saturation 0.15 to 0.25) so coloured ink stays opaque.
  * Margins outside the neatline and reading.PageInfo mask boxes are alpha 0 (shared with
    reading.exclusion_mask).
  * Halo: GDAL's bilinear resampling weights samples by the source alpha, and in addition
    RGB of every partially transparent pixel is pre-multiplied towards black
    (rgb * alpha / 255), and fully transparent pixels are black.  So white paper can
    never bleed into the edge of an inked line; edges darken instead of whitening.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.enums import ColorInterp

from sfzmaps import config
from sfzmaps.reading import Box, PageInfo, exclusion_mask
from sfzmaps.transform import SheetTransform

PAPER_GRAY = 200
RAMP_GRAY = 50
RAMP_COLOR = 40
SAT_PAPER = 0.15
SAT_RAMP = 0.10
GRID_STEP = 150


def _lum(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def paper_alpha(img: np.ndarray, region: np.ndarray | None = None) -> np.ndarray:
    """uint8 alpha (0 = paper, 255 = ink/colour) from a BGR or gray uint8 page.

    `region` (bool, True = inside the map) restricts the colour Otsu statistics.
    """
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lum = _lum(img).astype(np.float32)
    if img.ndim == 2:
        a = np.clip((PAPER_GRAY - lum) / RAMP_GRAY, 0, 1)
        return np.round(a * 255).astype(np.uint8)
    sample = _lum(img)
    if region is not None and region.any():
        sample = sample[region]
    otsu, _ = cv2.threshold(sample.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32) / 255.0
    dark = np.clip((otsu - lum) / RAMP_COLOR, 0, 1)
    colour = np.clip((sat - SAT_PAPER) / SAT_RAMP, 0, 1)
    return np.round(np.maximum(dark, colour) * 255).astype(np.uint8)


def make_rgba(img: np.ndarray, mask_boxes: list[Box]) -> np.ndarray:
    """(h, w, 4) uint8 RGBA, alpha 0 for paper, outside the neatline and in mask boxes."""
    excl = exclusion_mask(img, PageInfo(kind="map", mask_boxes=mask_boxes))
    alpha = paper_alpha(img, ~excl if (~excl).any() else None)
    alpha[excl] = 0
    rgb = img[..., ::-1] if img.ndim == 3 else np.repeat(img[..., None], 3, axis=2)
    rgb = np.ascontiguousarray(rgb).astype(np.float32)
    rgb *= (alpha.astype(np.float32) / 255.0)[..., None]  # towards black as alpha falls
    return np.dstack([np.round(rgb).astype(np.uint8), alpha])


def median_scale_ft_per_px(t: SheetTransform, shape: tuple[int, int], step: int = GRID_STEP) -> float:
    """Median world distance (ft) per pixel between adjacent dense-grid GCPs."""
    gs = t.dense_grid(shape, step)
    h, w = shape
    nx = len({round(g.px) for g in gs})
    P = np.array([[g.px, g.py] for g in gs]).reshape(-1, nx, 2)
    W = np.array([[g.wx, g.wy] for g in gs]).reshape(-1, nx, 2)
    ratios = []
    for ax in (0, 1):
        dp = np.linalg.norm(np.diff(P, axis=ax), axis=2)
        dw = np.linalg.norm(np.diff(W, axis=ax), axis=2)
        ratios.append((dw / np.maximum(dp, 1e-9)).ravel())
    return float(np.median(np.concatenate(ratios)))


def warp_sheet(png: Path, t: SheetTransform, mask_boxes: list[Box], out: Path) -> Path:
    """Write the transparent-paper RGBA GeoTIFF for `png` to `out` (atomically)."""
    png, out = Path(png), Path(out)
    if shutil.which("gdalwarp") is None:
        raise RuntimeError("gdalwarp not found on PATH; install GDAL (e.g. `brew install gdal`)")
    img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {png}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:  # e.g. 16-bit PNG
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = img.shape[:2]
    res = median_scale_ft_per_px(t, (h, w))
    rgba = make_rgba(img, mask_boxes)
    gcps = [GroundControlPoint(row=g.py + 0.5, col=g.px + 0.5, x=g.wx, y=g.wy)
            for g in t.dense_grid((h, w), GRID_STEP)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sfzwarp_") as td:
        src = Path(td) / "gcp.tif"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            ds = rasterio.open(
                src, "w", driver="GTiff", width=w, height=h, count=4, dtype="uint8",
                tiled=True, blockxsize=256, blockysize=256, compress="deflate",
                BIGTIFF="IF_SAFER")
        with ds:
            ds.write(np.moveaxis(rgba, 2, 0))
            ds.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
            ds.gcps = (gcps, rasterio.crs.CRS.from_string(config.WORK_CRS))
        fd, tmp_name = tempfile.mkstemp(suffix=".tif", prefix=out.stem + ".", dir=out.parent)
        os.close(fd)
        try:
            subprocess.run(
                ["gdalwarp", "-overwrite", "-tps", "-r", "bilinear", "-t_srs", config.WORK_CRS,
                 "-tr", repr(res), repr(res), "-srcalpha", "-dstalpha", "-multi",
                 "-wo", "NUM_THREADS=ALL_CPUS", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
                 "-co", "BIGTIFF=IF_SAFER", "-q", str(src), tmp_name],
                check=True, capture_output=True, text=True)
            os.replace(tmp_name, out)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"gdalwarp failed: {e.stderr}") from e
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    return out
