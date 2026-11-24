"""A tiny world for the finder's tests, in the corner of the solver's grid: four lots in two blocks either side of a
16 ft street, today's centre line down the street, and a flat placement of a page over it.

The grid is SHAPE (60 rows x 80 columns of 4 ft px from the grid's top-left corner, `grid.X0`, `grid.Y1`).  The west
block is columns 20-35, the street columns 36-39 (16 ft), the east block columns 40-55; both blocks run rows 10-49,
each cut into two lots at row 30.  The centre line runs down the middle of the street (x = 152 ft from X0), the
grid's whole height."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import shapely

from sfzmaps.solver import grid
from sfzmaps.solver.hatchnet import data

SHAPE = (60, 80)                          # grid rows, columns
WEST, STREET, EAST = (20, 36), (36, 40), (40, 56)     # grid columns (ends exclusive)
ROWS = (10, 50)                           # the blocks' grid rows (ends exclusive)
SPLIT = 30                                # the grid row that cuts each block into two lots
CENTRE_FT = 152.0                         # the centre line's x, ft east of X0: the street's middle
INSET_FT = 1.0                            # each lot's edges this far inside its px: an edge on a px boundary is halfway
                                          # between px centres, where the lot painter (cv2) fills the px past it too


def _box(c0: int, c1: int, r0: int, r1: int) -> shapely.Polygon:
    """Grid columns c0..c1 and rows r0..r1 (ends exclusive) as a lot in EPSG:2227 ft, `INSET_FT` inside them."""
    return shapely.box(grid.X0 + 4 * c0 + INSET_FT, grid.Y1 - 4 * r1 + INSET_FT, grid.X0 + 4 * c1 - INSET_FT,
                       grid.Y1 - 4 * r0 - INSET_FT)


def lots_world() -> gpd.GeoDataFrame:
    """The four lots (blklot, geometry in EPSG:2227 ft), two in each block."""
    boxes = [_box(*cols, *rows) for cols in (WEST, EAST) for rows in ((ROWS[0], SPLIT), (SPLIT, ROWS[1]))]
    return gpd.GeoDataFrame({"blklot": ["0001001", "0001002", "0002001", "0002002"]}, geometry=boxes,
                            crs="EPSG:2227")


def lines_world() -> data.Lines:
    """Today's one centre line, down the street the grid's whole height."""
    line = shapely.LineString([(grid.X0 + CENTRE_FT, grid.Y1), (grid.X0 + CENTRE_FT, grid.Y1 - 4 * SHAPE[0])])
    return data.Lines(np.array([line], dtype=object), np.array(["ST"], dtype=object), np.array(["A ST"], dtype=object))


class Flat:
    """A flat placement: page px (x, y) (px centres at integers) at (X0 + (x - dx) ft_per_px, Y1 - (y - dy)
    ft_per_px), so with no shift page px (0, 0) is the grid's top-left corner, and y runs down the page as it runs down
    the grid; with a shift the grid's corner is page px (dx, dy)."""

    correction = None                     # no warp correction laid over it, as a SheetTransform without one

    def __init__(self, ft_per_px: float, dx: float = 0.0, dy: float = 0.0):
        self.ft_per_px = float(ft_per_px)
        self.dx, self.dy = float(dx), float(dy)

    def to_world(self, px) -> np.ndarray:
        px = np.asarray(px, float).reshape(-1, 2)
        return np.c_[grid.X0 + (px[:, 0] - self.dx) * self.ft_per_px, grid.Y1 - (px[:, 1] - self.dy) * self.ft_per_px]

    def to_pixel(self, world) -> np.ndarray:
        w = np.asarray(world, float).reshape(-1, 2)
        return np.c_[(w[:, 0] - grid.X0) / self.ft_per_px + self.dx, (grid.Y1 - w[:, 1]) / self.ft_per_px + self.dy]

    def grid_to_pixel(self, world, page_shape) -> np.ndarray:
        """`to_pixel` for a large batch, as SheetTransform.grid_to_pixel."""
        return self.to_pixel(world)


GRID_COLS = [(10 + 12 * i, 18 + 12 * i) for i in range(5)]     # the grid columns of `blocks_world`'s blocks
GRID_ROWS = [(6 + 12 * j, 14 + 12 * j) for j in range(4)]      # and their grid rows (ends exclusive)


def blocks_world() -> tuple[gpd.GeoDataFrame, data.Lines]:
    """A 5 x 4 grid of one-lot blocks (8 x 8 grid px, streets 4 grid px wide between), lots column by column (lot
    4 i + j in column i, row j); and a centre line down the middle of every street between them."""
    boxes = [_box(*cols, *rows) for cols in GRID_COLS for rows in GRID_ROWS]
    lots = gpd.GeoDataFrame({"blklot": [f"{i:07d}" for i in range(len(boxes))]}, geometry=boxes, crs="EPSG:2227")
    xs = [grid.X0 + 4 * (a[1] + b[0]) / 2 for a, b in zip(GRID_COLS, GRID_COLS[1:])]
    ys = [grid.Y1 - 4 * (a[1] + b[0]) / 2 for a, b in zip(GRID_ROWS, GRID_ROWS[1:])]
    width, height = 4 * SHAPE[1], 4 * SHAPE[0]
    found = [shapely.LineString([(x, grid.Y1), (x, grid.Y1 - height)]) for x in xs] + \
            [shapely.LineString([(grid.X0, y), (grid.X0 + width, y)]) for y in ys]
    n = len(found)
    return lots, data.Lines(np.array(found, dtype=object), np.array(["ST"] * n, dtype=object),
                            np.array([f"{i} ST" for i in range(n)], dtype=object))
