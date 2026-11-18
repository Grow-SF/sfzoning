"""The solver's grid: one raster over the city that the merged page, the lots and the walls share.  EPSG:2227 feet,
4 ft per px, north up, top-left corner at (X0, Y1).  Grid px (col, row) covers x from X0 + 4 col to X0 + 4 (col + 1)
and y from Y1 - 4 (row + 1) to Y1 - 4 row; its centre is `to_map((col + 0.5, row + 0.5))`.

Also what every part of the solver reads the same way: which page px are ink (INK_BELOW), and the split of the city
into the four quarters that the fits hold out in turn (QUARTER_SPLIT, `quarter_of`)."""
import numpy as np

X0 = 5979300.0      # x of the grid's left edge, ft
Y1 = 2124100.0      # y of the grid's top edge, ft
FT_PER_PX = 4.0
WIDTH = 11900       # columns
HEIGHT = 9825       # rows

INK_BELOW = 128     # a page px darker than this is ink (the frame, the merged page and the text detector alike)
# The split of the city into four quarters, EPSG:2227 ft: the median centroid (x, y) of the scored lots of the first
# 2000 run, kept so held-out numbers compare with it.  East is x >= split, north is y >= split.
QUARTER_SPLIT = (6000180.818639896, 2101727.4853113447)


def to_grid(xy) -> np.ndarray:
    """Map points (..., 2) in EPSG:2227 ft -> grid (col, row), as floats: col = (x - X0) / 4, row = (Y1 - y) / 4."""
    xy = np.asarray(xy, float)
    return np.stack([(xy[..., 0] - X0) / FT_PER_PX, (Y1 - xy[..., 1]) / FT_PER_PX], -1)


def to_map(colrow) -> np.ndarray:
    """Grid (col, row) (..., 2) -> map points in EPSG:2227 ft; the inverse of to_grid."""
    cr = np.asarray(colrow, float)
    return np.stack([X0 + cr[..., 0] * FT_PER_PX, Y1 - cr[..., 1] * FT_PER_PX], -1)


def quarter_of(xy: np.ndarray) -> np.ndarray:
    """The quarter of each map point (n, 2): nw 0, ne 1, sw 2, se 3 (int8)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    east = xy[:, 0] >= QUARTER_SPLIT[0]
    north = xy[:, 1] >= QUARTER_SPLIT[1]
    return (np.where(north, 0, 2) + east).astype(np.int8)
