"""Street-or-not labels from a series' lots: which ground is street and which is block, on the solver's 4 ft grid, and
the same read onto a scan's page through its sheet's placement.

Only a series whose `lots` flag is set in inputs/find/label_sources.csv is labelled from lots (`series_lots`): the three
`2000-digital` series from the April 2000 table (`solver.table.load_table`), so the lot index lines up with the table's
rows; every other such series from the lots of its year (`districts.read.load_parcels`).  On the grid (`Ground`): the
lots painted (`solver.lots.lot_rasters`), today's blocks the lots with their enclosed holes filled, and the street
labels and weights of `hatchnet.data.street_grid` from the blocks and today's centre lines (data/ref/streets.parquet),
every street type with the pooled half-widths: 1 street, 0 not street (a block), 255 no label.

Reads data/ref/parcels.parquet or the April 2000 table, data/ref/streets.parquet, and a sheet's placement
(data/warp/<series>/<sheet>/transform.json with its correction, or data/work/<series>/<sheet>/transform.json).
Writes data/find/labels/<series>/grid.npz (`ground`), the grid labels kept so they are made once per series."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy import ndimage

from sfzmaps import config
from sfzmaps.districts.read import load_parcels, series_year
from sfzmaps.find import pages
from sfzmaps.find.pages import OUT, SheetId
from sfzmaps.solver import hatching
from sfzmaps.solver.hatchnet.data import UNKNOWN, Lines, lines_from, street_grid
from sfzmaps.solver.lots import lot_rasters
from sfzmaps.solver.table import load_table
from sfzmaps.solver.textread.store import save_npz
from sfzmaps.transform import SheetTransform

TABLE_BORDERS = ("zoning", "height_lim", "sud")     # the April 2000 table's columns a series' zoning borders can be
PLACEMENTS = {"warp": config.DATA_DIR / "warp", "work": config.WORK_DIR}
GRID = "grid.npz"


def series_lots(series: str) -> gpd.GeoDataFrame:
    """The lots that label a series: the April 2000 table when its zoning borders are one of the table's columns
    (`TABLE_BORDERS`), else the lots of its year.  ValueError for a series whose `lots` flag is not set."""
    src = pages.source(series)
    if not src.lots:
        raise ValueError(f"series {series}: its lots do not label it (no `lots` in {pages.SOURCES})")
    if src.borders in TABLE_BORDERS:
        return load_table()
    return load_parcels(series_year(series))


@dataclass(frozen=True)
class Ground:
    """The street-or-not labels on the solver's grid: the lot index (-1 none), today's blocks, the street labels (1
    street, 0 not street, 255 no label) and their weights (1 not street, (0, 1] street, 0 no label)."""
    lot: np.ndarray            # int32
    block: np.ndarray          # bool
    street: np.ndarray         # uint8
    weight: np.ndarray         # float16


def build_ground(lots: gpd.GeoDataFrame, lines: Lines, shape: tuple[int, int] | None = None) -> Ground:
    """`Ground` of `lots` (EPSG:2227 ft) and today's centre `lines` on a grid of `shape` from the solver grid's corner
    (None: the whole grid)."""
    lot, _ = lot_rasters(lots, shape)
    block = ndimage.binary_fill_holes(lot >= 0)
    street, weight, _ = street_grid(lines, block, per_type=False)
    return Ground(lot, block, street, weight)


def grid_path(series: str) -> Path:
    return OUT / "labels" / series / GRID


def ground(series: str) -> Ground:
    """The series' `Ground` on the whole grid: read from data/find/labels/<series>/grid.npz when there, else made from
    `series_lots` and today's centre lines and written there."""
    f = grid_path(series)
    if f.is_file():
        with np.load(f) as z:
            return Ground(z["lot"], z["block"], z["street"], z["weight"])
    g = build_ground(series_lots(series), lines_from(config.REF_DIR / "streets.parquet"))
    f.parent.mkdir(parents=True, exist_ok=True)
    part = f.with_name(f".{GRID}.part")
    save_npz(part, lot=g.lot, block=g.block, street=g.street, weight=g.weight)
    part.replace(f)
    return g


def placement(sid: SheetId, which: str = "warp") -> SheetTransform:
    """The sheet's placement: the warp's (data/warp/<series>/<sheet>/transform.json, with its correction) or, for
    "work", the one the reads were made through (data/work/<series>/<sheet>/transform.json)."""
    if which not in PLACEMENTS:
        raise ValueError(f"placement {which!r}: expected one of {', '.join(PLACEMENTS)}")
    folder = PLACEMENTS[which] / sid[0] / sid[1]
    return SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)


@dataclass(frozen=True)
class PageGround:
    """The street-or-not labels at every page px: 1 street, 0 block, 255 no label; and their weights (0 no label)."""
    street: np.ndarray         # uint8 (h, w)
    weight: np.ndarray         # float32 (h, w)


def page_ground(g: Ground, tr, shape: tuple[int, int], keep: np.ndarray) -> PageGround:
    """The grid's labels read at every px of a page of `shape` through its placement `tr` (`hatching.PageWorld`,
    `hatching.page_values`); 255 with weight 0 off the grid and off `keep` (the frame cut's map area)."""
    keep = np.asarray(keep, bool)
    if keep.shape != tuple(shape):
        raise ValueError(f"the frame cut is {keep.shape}, the page {tuple(shape)}")
    street, weight = hatching.page_values(hatching.PageWorld(tr, shape), shape, [(g.street, UNKNOWN), (g.weight, 0)])
    weight = weight.astype(np.float32)
    street[~keep] = UNKNOWN
    weight[~keep] = 0
    return PageGround(street, weight)
