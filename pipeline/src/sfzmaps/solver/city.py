"""The city the solver reads: the lots, their neighbours and the table's codes (the `solver-lots` stage's lots.npz
and table.parquet), the page's ink on the same grid, the wall odds (the `solver-walls` stage's walls_{q}.npz), and
the runs the solver writes (data/solver/solve/runs/)."""
from functools import cached_property, lru_cache
from pathlib import Path

import numpy as np

from sfzmaps.solver import grid, page, paths, table
from sfzmaps.solver.codes import canon

UNREAD_ODDS = 10.0          # the wall odds of a pair the wall reader gave none: a line


def wall_odds(q: str, walls_dir: Path | None = None) -> np.ndarray:
    """Per pair of neighbouring lots (the rows of lots.npz `pairs`): the log odds that a printed district line runs
    between them, fitted without quarter q, from `walls_dir/walls_{q}.npz` (default WALLS_DIR).  A pair with no
    odds (NaN) gets UNREAD_ODDS."""
    f = Path(walls_dir or paths.WALLS_DIR) / f"walls_{q}.npz"
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver walls`")
    return np.nan_to_num(np.load(f)["lo_ab"].astype(float), nan=UNREAD_ODDS)


class City:
    """The lots, in the order of the lot index, with:
      n, a, b, border_ft   the number of lots; each neighbour pair's lots and shared border (ft)
      known                the table gives the lot one district
      table                the table's code per lot with the by-eye corrections (`table.corrected_codes`): what
                           the scores count against ("" where not known)
      table_raw            the table's code as downloaded (lots.npz `table`): what every fit learns from
      scored, quarter      the lots the scores count, and each lot's quarter (nw 0, ne 1, sw 2, se 3)
      centroid             each lot's centroid, map ft
      lots                 the table itself (table.parquet: blklot, zoning, geometry)
    The rasters (`lot_raster`, `grown_raster`) and the page's ink (`scan()`) load on use; all three share the grid."""

    def __init__(self):
        import geopandas as gpd
        import shapely
        f = paths.LOTS_DIR / "lots.npz"
        if not f.is_file():
            raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver lots`")
        z = np.load(f)
        self.lots = gpd.read_parquet(paths.LOTS_DIR / "table.parquet")
        self.n = len(self.lots)
        pairs = z["pairs"]
        self.a, self.b = pairs[:, 0].astype(int), pairs[:, 1].astype(int)
        self.border_ft = (pairs[:, 2] + pairs[:, 3]) * grid.FT_PER_PX
        self.known = z["known"]
        self.table_raw = z["table"].astype(str)
        if not np.array_equal(table.known_and_codes(self.lots)[1], self.table_raw):
            raise ValueError(f"{f} and table.parquet give the lots different codes: run `sfzmaps solver lots`")
        self.table = table.corrected_codes(self.lots)
        self.scored = z["scored"]
        self.quarter = z["quarter"].astype(int)
        self.centroid = shapely.get_coordinates(shapely.centroid(self.lots.geometry.values))

    @cached_property
    def lot_raster(self) -> np.ndarray:
        """The lot index at each grid px (int32, -1 outside every lot)."""
        return np.load(paths.LOTS_DIR / "lots.npz")["lot"]

    @cached_property
    def grown_raster(self) -> np.ndarray:
        """Each grid px's nearest lot (int32)."""
        return np.load(paths.LOTS_DIR / "lots.npz")["grown"]

    @staticmethod
    def px(xy) -> np.ndarray:
        """Map points (..., 2) -> the grid px (col, row) holding them (int64): floor of `grid.to_grid`."""
        return np.floor(grid.to_grid(xy)).astype(np.int64)

    @staticmethod
    def scan() -> np.ndarray:
        """The merged page's ink (bool, True = ink) on the grid."""
        return page.load_ink()

    def lot_ink(self) -> np.ndarray:
        """The share of each lot's own px that are inked on the page."""
        lot = self.lot_raster
        inside = lot >= 0
        ids = lot[inside]
        inked = np.bincount(ids, self.scan()[inside], minlength=self.n)
        return inked / np.maximum(np.bincount(ids, minlength=self.n), 1)


@lru_cache(maxsize=1)
def city() -> City:
    """The city, read once per process."""
    return City()


def run_file(name: str) -> Path:
    """Where the run `name` is kept: SOLVE_DIR/runs/{name}.npz."""
    return paths.SOLVE_DIR / "runs" / f"{name}.npz"


def load_run(name: str) -> np.ndarray:
    """A run's code per lot, in the one spelling."""
    R = np.load(run_file(name), allow_pickle=True)
    return np.array([canon(x) for x in np.asarray(R["names"]).astype(str)])[R["pred"]]


def save_run(path: Path, codes) -> None:
    """Write a code per lot as a run: `names` (the codes used) and `pred` (each lot's index into them)."""
    names, pred = np.unique(np.asarray(codes).astype(str), return_inverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pred=pred, names=names)
