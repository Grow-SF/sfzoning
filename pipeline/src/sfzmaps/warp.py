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
  * Neatline stroke: the detected neatline mask (pages.neatline_mask) includes the border
    stroke itself, which would show as a black frame in the stitched layer.  The interior is
    therefore eroded by NEATLINE_INSET_PX (Euclidean distance from the mask edge) before the
    alpha is built.  Measured on 1948/1 and 1960/3, /11 (300 dpi, 5-6k px wide): the stroke
    band lies 5-45 px inside the mask edge (double rules and the downscaled-contour jitter
    included), so 60 px (about 0.5 in, about 25-40 ft on the ground) clears it.  A page with no
    single neatline falls back to pages.neatline_frames_mask (union of several framed maps,
    e.g. the 1948 "2-and-3" two-page photo); one with no frame at all (1948/7) is not eroded.
  * Footprint: warp_sheet also writes `<out stem>.footprint.geojson`, the sheet's content
    (ink closed and hole-filled, inside the neatline interior and outside mask boxes) as
    EPSG:2227 polygons.  stitch uses it for per-sheet cutlines.
  * Halo: GDAL's bilinear resampling weights samples by the source alpha, and in addition
    RGB of every partially transparent pixel is pre-multiplied towards black
    (rgb * alpha / 255), and fully transparent pixels are black.  So white paper can
    never bleed into the edge of an inked line; edges darken instead of whitening.
"""
from __future__ import annotations

import logging
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
from scipy.ndimage import distance_transform_edt

from sfzmaps import config
from sfzmaps.pages import neatline_frames_mask, neatline_mask
from sfzmaps.reading import Box
from sfzmaps.transform import SheetTransform

log = logging.getLogger(__name__)

PAPER_GRAY = 200
RAMP_GRAY = 50
RAMP_COLOR = 40
SAT_PAPER = 0.15
SAT_RAMP = 0.10
GRID_STEP = 150
NEATLINE_INSET_PX = 60
FOOTPRINT_CLOSE_PX = 41   # closing kernel for the content footprint (about 0.7 in on a 6000 px scan)
FOOTPRINT_MAX_PX = 1000   # footprint contours are traced on a copy at most this many px wide


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


def inset_exclusion(img: np.ndarray, mask_boxes: list[Box], inset: int | None = None) -> np.ndarray:
    """True where the sheet must be transparent: outside the neatline interior (the border
    stroke is inside the detected mask, so the interior is eroded by `inset` px), or in a
    mask box."""
    inset = NEATLINE_INSET_PX if inset is None else inset
    nl = neatline_mask(img)
    if nl.all():   # no single frame: maybe a spread with several framed maps
        nl = neatline_frames_mask(img)
    excl = ~nl
    for b in mask_boxes:
        excl[max(0, b.y0):min(excl.shape[0], b.y1), max(0, b.x0):min(excl.shape[1], b.x1)] = True
    if inset > 0 and not nl.all():
        padded = np.pad(nl, 1, constant_values=True)   # the page edge is not a border
        interior = distance_transform_edt(padded)[1:-1, 1:-1] > inset
        excl = excl | ~interior
    return excl


def footprint_polygons(excl: np.ndarray, alpha: np.ndarray, t: SheetTransform):
    """Shapely geometry (EPSG:2227) of the sheet's content: the ink (alpha > 0), closed with a
    FOOTPRINT_CLOSE_PX kernel (at FOOTPRINT_MAX_PX width) and with enclosed holes filled, inside
    the non-excluded area.  Not simply the neatline interior: a framed page often has blank
    corners beyond the drawn map, and a cutline there would hide a neighbour's real content."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    h, w = excl.shape
    f = min(1.0, FOOTPRINT_MAX_PX / max(h, w))
    ink = (alpha > 0).astype(np.uint8)
    ok0 = (~excl).astype(np.uint8)
    if f < 1:
        size = (max(1, round(w * f)), max(1, round(h * f)))
        ink = cv2.resize(ink, size, interpolation=cv2.INTER_AREA)
        ink = (ink > 0).astype(np.uint8)
        ok0 = cv2.resize(ok0, size, interpolation=cv2.INTER_NEAREST)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FOOTPRINT_CLOSE_PX, FOOTPRINT_CLOSE_PX))
    closed = cv2.morphologyEx(np.pad(ink, FOOTPRINT_CLOSE_PX), cv2.MORPH_CLOSE, k)
    closed = closed[FOOTPRINT_CLOSE_PX:-FOOTPRINT_CLOSE_PX, FOOTPRINT_CLOSE_PX:-FOOTPRINT_CLOSE_PX]
    ff = np.pad(closed, 1).copy()
    cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 2)
    filled = (ff[1:-1, 1:-1] != 2).astype(np.uint8)   # content plus enclosed holes
    ok = filled & ok0
    ok = np.pad(ok, 1)
    cs, hier = cv2.findContours(ok, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = []
    for c in cs:
        c = cv2.approxPolyDP(c, 1.0, True)[:, 0, :].astype(float)
        if len(c) < 3:
            continue
        px = (c - 1) / f   # undo padding and scaling: full-resolution pixel indices
        rings.append(Polygon(t.to_world(px)))
    if not rings:
        return Polygon()
    # RETR_CCOMP alternates outer/hole rings; nesting is resolved by even-odd symmetric difference.
    # Rings from approxPolyDP can self-intersect (bow-ties), so repair each one first.
    try:
        return _combine_rings(rings)
    except Exception as e:  # GEOSException etc: never fail the sheet over its cutline
        log.warning("footprint union failed (%s); falling back to neatline footprint", e)
        return _neatline_footprint(excl, t)


def _valid_poly(g):
    import shapely

    g = shapely.make_valid(g)
    if not g.is_valid:
        g = g.buffer(0)
    return g


def _combine_rings(rings):
    from shapely.ops import unary_union

    rings = [_valid_poly(r) for r in rings]
    geom = rings[0]
    for r in rings[1:]:
        try:
            geom = geom.symmetric_difference(r)
        except Exception:
            geom = geom.buffer(0).symmetric_difference(r.buffer(0))
    return unary_union([_valid_poly(geom)])


def _neatline_footprint(excl: np.ndarray, t: SheetTransform):
    """Plain footprint: bounding box of the non-excluded (inside-neatline) pixels."""
    from shapely.geometry import Polygon

    ys, xs = np.nonzero(~excl)
    if len(xs) == 0:
        return Polygon()
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    return _valid_poly(Polygon(t.to_world(px)))


def make_rgba(img: np.ndarray, mask_boxes: list[Box], excl: np.ndarray | None = None) -> np.ndarray:
    """(h, w, 4) uint8 RGBA, alpha 0 for paper, outside the neatline (stroke included) and
    in mask boxes."""
    if excl is None:
        excl = inset_exclusion(img, mask_boxes)
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


def _write_footprint(path: Path, geom) -> None:
    import json
    from shapely.geometry import mapping

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::2227"}},
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(geom)}]}))
    tmp.replace(path)


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
    excl = inset_exclusion(img, mask_boxes)
    rgba = make_rgba(img, mask_boxes, excl)
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
            _write_footprint(out.with_name(out.stem + ".footprint.geojson"), footprint_polygons(excl, rgba[..., 3], t))
            os.replace(tmp_name, out)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"gdalwarp failed: {e.stderr}") from e
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
    return out
