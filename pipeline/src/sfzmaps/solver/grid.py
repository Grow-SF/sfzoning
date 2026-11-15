"""The solver's grid: one raster over the city that the merged page, the lots and the walls share.  EPSG:2227 feet,
4 ft per px, north up, top-left corner at (X0, Y1).  Grid px (col, row) covers x from X0 + 4 col to X0 + 4 (col + 1)
and y from Y1 - 4 (row + 1) to Y1 - 4 row; its centre is `to_map((col + 0.5, row + 0.5))`."""
import numpy as np

X0 = 5979300.0      # x of the grid's left edge, ft
Y1 = 2124100.0      # y of the grid's top edge, ft
FT_PER_PX = 4.0
WIDTH = 11900       # columns
HEIGHT = 9825       # rows


def to_grid(xy) -> np.ndarray:
    """Map points (..., 2) in EPSG:2227 ft -> grid (col, row), as floats: col = (x - X0) / 4, row = (Y1 - y) / 4."""
    xy = np.asarray(xy, float)
    return np.stack([(xy[..., 0] - X0) / FT_PER_PX, (Y1 - xy[..., 1]) / FT_PER_PX], -1)


def to_map(colrow) -> np.ndarray:
    """Grid (col, row) (..., 2) -> map points in EPSG:2227 ft; the inverse of to_grid."""
    cr = np.asarray(colrow, float)
    return np.stack([X0 + cr[..., 0] * FT_PER_PX, Y1 - cr[..., 1] * FT_PER_PX], -1)
