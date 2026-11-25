"""A sheet's page px and the solver's grid: the placement carried between them as the merged page reads a sheet.

`PageWorld` takes page px to EPSG:2227 ft (the placement evaluated every LATTICE px, bilinear between);
`page_values` reads grid rasters at every px of a page through it; `grid_slice` is the part of the grid a sheet's
frame cut can reach.  Moved here unchanged from the 1921 hatching reader (`solver.hatching`, branch
`hatch-experiments-archive`).  Reads and writes nothing."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from sfzmaps.solver import grid, page

LATTICE = page.STEP                       # page px between the points where a placement is evaluated


def crop_outline(crop: np.ndarray) -> np.ndarray:
    """Every boundary px (x, y) of a crop's outer outlines, page px."""
    cs, _ = cv2.findContours(np.asarray(crop, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return np.concatenate([c.reshape(-1, 2) for c in cs]) if cs else np.zeros((0, 2), np.int32)


class PageWorld:
    """A sheet's page px -> EPSG:2227 ft: its placement evaluated every LATTICE px (as the merged page reads a
    sheet), bilinear between.  `box` (x0, y0, x1, y1): only the nodes that cover it; the values are the whole page's."""

    def __init__(self, transform, shape: tuple[int, int], box=None):
        h, w = shape
        x0, y0, x1, y1 = box if box is not None else (0, 0, w, h)
        self.nx0, self.ny0 = x0 // LATTICE * LATTICE, y0 // LATTICE * LATTICE
        xs = np.arange(self.nx0, max(x1 - 1, x0) + LATTICE, LATTICE, dtype=float)
        ys = np.arange(self.ny0, max(y1 - 1, y0) + LATTICE, LATTICE, dtype=float)
        gx, gy = np.meshgrid(xs, ys)
        self.nodes = np.asarray(transform.to_world(np.c_[gx.ravel(), gy.ravel()]), float).reshape(gy.shape + (2,))

    def at(self, xy: np.ndarray) -> np.ndarray:
        """World ft (n, 2) of page px (n, 2) (x, y; px centres at integers)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        rc = [(xy[:, 1] - self.ny0) / LATTICE, (xy[:, 0] - self.nx0) / LATTICE]
        return np.stack([map_coordinates(self.nodes[..., k], rc, order=1, mode="nearest") for k in (0, 1)], -1)

    def block(self, x0: int, y0: int, x1: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
        """World x and y (each (y1 - y0, x1 - x0)) of the px of a box."""
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = self.at(np.c_[xx.ravel(), yy.ravel()])
        return w[:, 0].reshape(yy.shape), w[:, 1].reshape(yy.shape)


def grid_values(X: np.ndarray, Y: np.ndarray, raster: np.ndarray, offset=(0, 0), fill=-1) -> np.ndarray:
    """The value of a grid raster (rows and columns from `offset` (row, col) of the solver's grid) at world points;
    `fill` off it."""
    c = np.floor((X - grid.X0) / grid.FT_PER_PX)
    r = np.floor((grid.Y1 - Y) / grid.FT_PER_PX)
    ok = np.isfinite(c) & np.isfinite(r)
    ci = np.where(ok, c, -1).astype(np.int64) - offset[1]
    ri = np.where(ok, r, -1).astype(np.int64) - offset[0]
    ok &= (ci >= 0) & (ci < raster.shape[1]) & (ri >= 0) & (ri < raster.shape[0])
    out = np.full(X.shape, fill, raster.dtype)
    out[ok] = raster[ri[ok], ci[ok]]
    return out


def page_values(pw: PageWorld, shape: tuple[int, int], rasters: list[tuple[np.ndarray, int]], offset=(0, 0),
                rows: int = 512) -> list[np.ndarray]:
    """Grid rasters (each with its fill off it) read at every px of a page (bands of `rows` rows at once: memory
    only)."""
    h, w = shape
    outs = [np.full(shape, fill, r.dtype) for r, fill in rasters]
    for y0 in range(0, h, rows):
        X, Y = pw.block(0, y0, w, min(h, y0 + rows))
        for o, (r, fill) in zip(outs, rasters):
            o[y0:y0 + rows] = grid_values(X, Y, r, offset, fill)
    return outs


def grid_slice(tr, crop: np.ndarray, shape) -> tuple[int, int, int, int]:
    """The grid rows and columns (r0, r1, c0, c1, ends exclusive) a sheet's frame cut can reach: every boundary px of
    its outer outlines carried to the grid by its placement, `page.STEP` px wider each way, inside a grid of
    `shape`."""
    cr = grid.to_grid(tr.to_world(crop_outline(crop).astype(float)))
    cr = cr[np.isfinite(cr).all(axis=1)]
    if not len(cr):
        return 0, 0, 0, 0
    rows, cols = shape
    r0 = max(0, int(np.floor(cr[:, 1].min())) - page.STEP)
    r1 = min(rows, int(np.ceil(cr[:, 1].max())) + page.STEP)
    c0 = max(0, int(np.floor(cr[:, 0].min())) - page.STEP)
    c1 = min(cols, int(np.ceil(cr[:, 0].max())) + page.STEP)
    return r0, r1, c0, c1
