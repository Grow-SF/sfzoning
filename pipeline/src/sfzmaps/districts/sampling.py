"""Parcels <-> warped sheets: which sheet covers a parcel, its pixel crop, and texture stats."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from sfzmaps import stitch

MIN_COVER = 0.5


def sheet_order(series: str) -> list[str]:
    """Sheet ids with a warp, highest priority first (the VRT draw order, reversed: passed or
    reviewed sheets before needs_review, then lower held-out p50)."""
    return stitch.draw_order(series, stitch.load_scores(series))[::-1]


def assign_sheets(parcels, series: str, sheets: list[str] | None = None):
    """Series blklot -> sheet id: the highest-priority sheet (of `sheets`, default all) whose
    footprint covers >= 50% of the parcel's area; None where no sheet does."""
    import pandas as pd
    from shapely import STRtree

    order = sheet_order(series) if sheets is None else list(sheets)
    geoms = list(parcels.geometry)
    out: list[str | None] = [None] * len(geoms)
    for sid in order:
        todo = [i for i, o in enumerate(out) if o is None]
        if not todo:
            break
        fp = stitch.footprint(series, sid)
        tree = STRtree([geoms[i] for i in todo])
        for j in tree.query(fp, predicate="intersects"):
            i = todo[j]
            g = geoms[i]
            if g.area > 0 and g.intersection(fp).area >= MIN_COVER * g.area:
                out[i] = sid
    return pd.Series(out, index=parcels["blklot"].to_numpy(), dtype=object, name="sheet")


def fit_square(img: np.ndarray, px: int) -> np.ndarray:
    """Resize so the long edge is px (aspect kept), centred on a transparent px x px canvas."""
    h, w = img.shape[:2]
    s = px / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST
    small = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.zeros((px, px) + img.shape[2:], img.dtype)
    y0, x0 = (px - nh) // 2, (px - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


@lru_cache(maxsize=64)
def _footprint(warped_tif: str):
    """The sheet footprint polygon next to warped.tif (warped.footprint.geojson), or None."""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    fp = Path(warped_tif).with_suffix(".footprint.geojson")
    if not fp.exists():
        return None
    feats = json.loads(fp.read_text())["features"]
    return unary_union([shape(f["geometry"]) for f in feats]) if feats else None


def read_window(warped_tif: Path, bounds) -> tuple[np.ndarray, np.ndarray, object]:
    """(RGB uint8 on white, on-sheet bool mask, window transform) for world bounds.

    The warp makes paper transparent with RGB premultiplied toward black, so the scan as
    printed is rgb + (255 - alpha).  On-sheet is the sheet footprint where one was written
    (paper is alpha 0 but still on the sheet), else alpha > 0."""
    import rasterio
    from rasterio import features, windows

    x0, y0, x1, y1 = bounds
    with rasterio.open(warped_tif) as ds:
        win = windows.from_bounds(x0, y0, x1, y1, ds.transform)
        win = windows.Window(np.floor(win.col_off), np.floor(win.row_off),
                             max(1, np.ceil(win.width) + 1), max(1, np.ceil(win.height) + 1))
        data = ds.read(window=win, boundless=True, fill_value=0)
        tr = ds.window_transform(win)
    rgb = np.clip(data[:3].astype(np.int16) + (255 - data[3].astype(np.int16)), 0, 255)
    rgb = np.moveaxis(rgb.astype(np.uint8), 0, -1).copy()
    fp = _footprint(str(warped_tif))
    if fp is None:
        on = data[3] > 0
    else:
        on = features.rasterize([(fp, 1)], out_shape=data.shape[1:], transform=tr, fill=0,
                                dtype="uint8") > 0
    return rgb, on, tr


def parcel_crop(warped_tif: Path, geom, *, inset_ft: float = 3.0, px: int | None = 64,
                keep_aspect: bool = False) -> np.ndarray | None:
    """RGBA uint8 of the parcel shrunk by inset_ft as printed (paper white), alpha 255 inside
    the parcel and on the sheet, everything else transparent black.

    px x px (squashed to a square); with keep_aspect the long edge is px and the rest is
    transparent padding, so hatch angles survive; px=None keeps the native raster
    resolution.  None when under 50% of the parcel's pixels are on the sheet."""
    from rasterio import features

    g = geom.buffer(-inset_ft)
    if g.is_empty:
        g = geom
    rgb, on, tr = read_window(warped_tif, g.bounds)
    mask = features.rasterize([(g, 1)], out_shape=on.shape, transform=tr, fill=0, dtype="uint8") > 0
    if not mask.any() or on[mask].mean() < MIN_COVER:
        return None
    mask &= on
    rgba = np.dstack([rgb, np.where(mask, 255, 0).astype(np.uint8)])
    rgba[~mask] = 0
    if px is None:
        return rgba
    if keep_aspect:
        return fit_square(rgba, px)
    interp = cv2.INTER_AREA if min(rgba.shape[:2]) > px else cv2.INTER_NEAREST
    return cv2.resize(rgba, (px, px), interpolation=interp)


def texture_stats(crop: np.ndarray) -> np.ndarray:
    """[ink fraction, 4 gradient-orientation bins (magnitude-weighted, sum 1), FFT peak ratio]
    over the crop's alpha > 0 pixels.  Ink = dark (luminance < 128); the FFT peak ratio is the
    strongest non-DC spectral peak over the mean non-DC magnitude (0 when flat)."""
    gray = cv2.cvtColor(np.ascontiguousarray(crop[..., :3]), cv2.COLOR_RGB2GRAY).astype(np.float32)
    valid = crop[..., 3] > 0
    if not valid.any():
        return np.zeros(6)
    ink = float(((gray < 128) & valid).sum() / valid.sum())
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    ev = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0   # ignore the mask edge
    mag = np.hypot(gx, gy) * ev
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    bins = (ang / (np.pi / 4) + 0.5).astype(int) % 4   # 0, 45, 90, 135 degrees
    hist = np.bincount(bins.ravel(), weights=mag.ravel(), minlength=4)[:4]
    hist = hist / hist.sum() if hist.sum() > 0 else np.zeros(4)
    g = np.where(valid, gray, gray[valid].mean())
    g = (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    f = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    cy, cx = f.shape[0] // 2, f.shape[1] // 2
    f[cy - 1:cy + 2, cx - 1:cx + 2] = 0
    rest = f[f > 0]
    ratio = float(f.max() / rest.mean()) if rest.size and f.max() > 1e-6 else 0.0
    return np.array([ink, *hist, ratio])


def tone(crop: np.ndarray, k: int = 9) -> float:
    """Median luminance (0-255) of the crop's alpha > 0 pixels after a k x k box blur that
    averages only valid pixels: a robust 'how dark is the fill' on which white, hatch,
    cross-hatch and solid separate, and which a printed label covering under half the parcel
    cannot move (texture_stats' ink fraction can: measured on 1948/1, labels push white lots
    to ink 0.4).  255 when the crop is empty."""
    valid = (crop[..., 3] > 0).astype(np.float32)
    if not valid.any():
        return 255.0
    g = cv2.cvtColor(np.ascontiguousarray(crop[..., :3]), cv2.COLOR_RGB2GRAY).astype(np.float32)
    num, den = cv2.blur(g * valid, (k, k)), cv2.blur(valid, (k, k))
    v = valid > 0
    return float(np.median(num[v] / np.maximum(den[v], 1e-6)))
