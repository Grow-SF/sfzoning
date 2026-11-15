"""The lots on the solver's grid: the April 2000 table's lots painted on the merged page's grid, which lots touch or
face each other across a street, which lots are scored, and which quarter of the city each is in.

Writes, under data/solver/lots/:
  lots.npz       lot (int32 raster, the lot index at each px, -1 outside lots), grown (int32, each px's nearest lot),
                 pairs (int64 (n, 4): a, b, touching px, street px), scored (bool per lot), known (bool per lot),
                 quarter (int8 per lot: nw 0, ne 1, sw 2, se 3), table (each lot's code, "" where not known)
  table.parquet  the loaded table (EPSG:2227), in the order of the lot index
"""
import time
from dataclasses import dataclass

import cv2
import geopandas as gpd
import numpy as np
import shapely
from scipy import ndimage

from sfzmaps.solver import grid, page, paths, table
from sfzmaps.solver.grid import QUARTER_SPLIT, quarter_of  # noqa: F401 - the quarters are the grid's; kept here too

SUBPIXEL_BITS = 4           # polygon corners are filled to 1/16 px


def _polygons(geom) -> list:
    """The polygons of a geometry (none for an empty or non-areal one)."""
    return [g for g in shapely.get_parts(geom) if g.geom_type == "Polygon" and not g.is_empty]


def lot_rasters(lots: gpd.GeoDataFrame, shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`lot`: the lot index at each grid px (int32, -1 outside every lot), each polygon's exterior filled largest lot
    first so a lot inside another stays visible.  Grid px (c, r) covers [c, c+1) x [r, r+1) and fillPoly takes integer
    coordinates as px centres, so the corners go in at `to_grid - 0.5`, to 1/16 px: the px filled are those whose
    centres are inside the lot.  `grown`: each px's nearest lot (int32), so lots facing across a street are
    neighbours."""
    shape = shape or (grid.HEIGHT, grid.WIDTH)
    lot = np.full(shape, -1, np.int32)
    scale = 1 << SUBPIXEL_BITS
    for i in np.argsort(-lots.geometry.area.values, kind="stable"):
        for poly in _polygons(lots.geometry.values[i]):
            xy = grid.to_grid(shapely.get_coordinates(poly.exterior)) - 0.5
            pts = np.rint(xy * scale).astype(np.int32)
            cv2.fillPoly(lot, [pts], int(i), shift=SUBPIXEL_BITS)
    idx = ndimage.distance_transform_edt(lot < 0, return_distances=False, return_indices=True)
    grown = lot[idx[0], idx[1]]
    return lot, grown


def neighbour_pairs(lot: np.ndarray, grown: np.ndarray) -> np.ndarray:
    """Every two lots whose regions in `grown` meet, as rows (a, b, touching px, street px) with a < b, sorted by
    (a, b) (int64).  Each pair of 4-neighbour px with different `grown` counts once for its lots: touching when both
    px are inside lots, street otherwise."""
    n = int(grown.max()) + 1 if grown.size else 1
    keys, touching = [], []
    for ga, gb, la, lb in ((grown[:, :-1], grown[:, 1:], lot[:, :-1], lot[:, 1:]),
                           (grown[:-1], grown[1:], lot[:-1], lot[1:])):
        m = ga != gb
        lo = np.minimum(ga[m], gb[m]).astype(np.int64)
        hi = np.maximum(ga[m], gb[m]).astype(np.int64)
        keys.append(lo * n + hi)
        touching.append((la[m] >= 0) & (lb[m] >= 0))
    keys, touching = np.concatenate(keys), np.concatenate(touching)
    uniq, inverse = np.unique(keys, return_inverse=True)
    return np.c_[uniq // n, uniq % n, np.bincount(inverse, touching), np.bincount(inverse, ~touching)].astype(np.int64)


def scored_lots(lot: np.ndarray, mapped: np.ndarray, known: np.ndarray) -> np.ndarray:
    """The lots the solver is scored on (bool per lot): a known code, and more than half of the lot's px on the
    mapped area."""
    n = len(known)
    inside = lot >= 0
    total = np.bincount(lot[inside], minlength=n)[:n]
    on_map = np.bincount(lot[inside & mapped], minlength=n)[:n]
    return known & (2 * on_map > total)


def lot_centroids(geometries) -> np.ndarray:
    """Each lot's centroid (x, y) (float64 (n, 2), one row per lot in lot order; NaN for an empty geometry, whose
    quarter is then sw, and which has no px to be scored)."""
    c = shapely.centroid(np.asarray(geometries, dtype=object))
    has = ~shapely.is_empty(c)
    out = np.full((len(c), 2), np.nan)
    out[has] = shapely.get_coordinates(c[has])
    return out


def build_lots() -> dict:
    """Paint the lots on the grid and write lots.npz and table.parquet into LOTS_DIR.  Returns a summary: lots,
    scored lots, pairs, the median centroid of the scored lots, seconds."""
    t0 = time.perf_counter()
    lots = table.load_table()
    known, codes = table.known_and_codes(lots)
    lot, grown = lot_rasters(lots)
    pairs = neighbour_pairs(lot, grown)
    mapped = page.load_sheets() > 0
    scored = scored_lots(lot, mapped, known)
    centroids = lot_centroids(lots.geometry.values)
    paths.LOTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths.LOTS_DIR / "lots.npz", lot=lot, grown=grown, pairs=pairs, scored=scored, known=known,
             quarter=quarter_of(centroids), table=codes)
    lots.to_parquet(paths.LOTS_DIR / "table.parquet")
    median = np.median(centroids[scored], axis=0) if scored.any() else np.full(2, np.nan)
    return {"lots": len(lots), "scored": int(scored.sum()), "pairs": len(pairs),
            "median_x": float(median[0]), "median_y": float(median[1]), "seconds": round(time.perf_counter() - t0, 1)}


@dataclass
class Lots:
    """The built lots: `lot` and `grown` rasters, the neighbour pairs (`a`, `b`, `touching`, `street` px and
    `border_ft`, the shared border in feet), and per lot `scored`, `known`, `quarter` and `codes`."""
    lot: np.ndarray
    grown: np.ndarray
    a: np.ndarray
    b: np.ndarray
    touching: np.ndarray
    street: np.ndarray
    scored: np.ndarray
    known: np.ndarray
    quarter: np.ndarray
    codes: np.ndarray

    @property
    def n(self) -> int:
        """The number of lots."""
        return len(self.known)

    @property
    def border_ft(self) -> np.ndarray:
        """Each pair's shared border length, ft: (touching + street) px of FT_PER_PX each."""
        return (self.touching + self.street) * grid.FT_PER_PX


def load_lots() -> Lots:
    """Read lots.npz."""
    f = paths.LOTS_DIR / "lots.npz"
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver lots`")
    z = np.load(f)
    p = z["pairs"]
    return Lots(z["lot"], z["grown"], p[:, 0], p[:, 1], p[:, 2], p[:, 3], z["scored"], z["known"], z["quarter"],
                z["table"])
