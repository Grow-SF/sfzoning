"""Parcels <-> warped sheets: which sheet covers a parcel, its pixel crop, and texture stats."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sfzmaps import stitch

MIN_COVER = 0.5


def sheet_order(series: str) -> list[str]:
    """Sheet ids with a warp, highest priority first (the VRT draw order, reversed: passed or
    reviewed sheets before needs_review, then lower held-out p50)."""
    return stitch.draw_order(series, stitch.load_scores(series))[::-1]


def assign_sheets(parcels, series: str):
    """Series blklot -> sheet id: the highest-priority sheet whose footprint covers >= 50% of the
    parcel's area; None where no sheet does."""
    import pandas as pd
    from shapely import STRtree

    order = sheet_order(series)
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


def parcel_crop(warped_tif: Path, geom, *, inset_ft: float = 3.0, px: int = 64) -> np.ndarray | None:
    """RGBA uint8 (px x px x 4) of the parcel shrunk by inset_ft, everything outside it
    transparent.  None when under 50% of the parcel's pixels have alpha."""
    import rasterio
    from rasterio import features, windows

    g = geom.buffer(-inset_ft)
    if g.is_empty:
        g = geom
    with rasterio.open(warped_tif) as ds:
        x0, y0, x1, y1 = g.bounds
        win = windows.from_bounds(x0, y0, x1, y1, ds.transform)
        win = windows.Window(np.floor(win.col_off), np.floor(win.row_off),
                             max(1, np.ceil(win.width) + 1), max(1, np.ceil(win.height) + 1))
        data = ds.read(window=win, boundless=True, fill_value=0)
        tr = ds.window_transform(win)
    mask = features.rasterize([(g, 1)], out_shape=data.shape[1:], transform=tr, fill=0, dtype="uint8") > 0
    if not mask.any() or (data[3][mask] > 0).mean() < MIN_COVER:
        return None
    rgba = np.moveaxis(data, 0, -1).copy()
    rgba[~mask] = 0
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
