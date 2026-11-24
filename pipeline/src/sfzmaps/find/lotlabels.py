"""Street-or-not labels from a series' lots: which ground is street and which is block, on the solver's 4 ft grid, and
the same read onto a scan's page through its sheet's placement.

Only a series whose `lots` flag is set in inputs/find/label_sources.csv is labelled from lots (`series_lots`): the three
`2000-digital` series from the April 2000 table (`solver.table.load_table`), so the lot index lines up with the table's
rows; every other such series from the lots of its year (`districts.read.load_parcels`).  On the grid (`Ground`): the
lots painted (`solver.lots.lot_rasters`), today's blocks the lots with their enclosed holes filled, and the street
labels and weights of `hatchnet.data.street_grid` from the blocks and today's centre lines (data/ref/streets.parquet),
every street type with the pooled half-widths: 1 street, 0 not street (a block), 255 no label.

A sheet's labels (`sheet_labels`): the street-or-not labels read onto its page, and the cut statements of
`find.cuts`: block edges for a series labelled from lots, and zoning borders where the series' zoning borders are a
column of the April 2000 table (`series_codes`).  A series whose zoning borders are `none` (the hatched maps) has no
zoning border anywhere on its map.

Reads data/ref/parcels.parquet or the April 2000 table, data/ref/streets.parquet, a sheet's page, frame cut and
placement (data/warp/<series>/<sheet>/transform.json with its correction, or data/work/<series>/<sheet>/transform.json)
and data/warp/<series>/<sheet>/scores.json.  Writes data/find/labels/<series>/grid.npz (`make_ground`, made by
`sfzmaps find labels`, read by everything else) and data/find/labels/<series>/<sheet>.npz (`sheet_labels`)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image
from scipy import ndimage

from sfzmaps import config
from sfzmaps.districts.read import load_parcels, series_year
from sfzmaps.find import pages
from sfzmaps.find.pages import SheetId
from sfzmaps.solver import hatching
from sfzmaps.frame import data as frame_data
from sfzmaps.solver.hatchnet.data import UNKNOWN, Lines, lines_from, street_grid
from sfzmaps.solver.lots import lot_rasters
from sfzmaps.solver.table import load_table
from sfzmaps.solver.textread.store import save_npz
from sfzmaps.transform import SheetTransform

TABLE_BORDERS = ("zoning", "height_lim", "sud")     # the April 2000 table's columns a series' zoning borders can be
# What a lot with no value in a table column is: unknown ("", no statement), except in `sud`, where the table leaves a
# lot in no special-use district blank; that is known plain ground, and the special-use map draws a zoning border
# between a district and it.
MISSING_CODE = {"sud": "(none)"}
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
    return pages.OUT / "labels" / series / GRID


def make_ground(series: str) -> Ground:
    """The series' `Ground` on the whole grid, made from `series_lots` and today's centre lines and written to
    data/find/labels/<series>/grid.npz (`sfzmaps find labels` makes it every time it runs)."""
    g = build_ground(series_lots(series), lines_from(config.REF_DIR / "streets.parquet"))
    f = grid_path(series)
    f.parent.mkdir(parents=True, exist_ok=True)
    part = f.with_name(f".{GRID}.part")
    save_npz(part, lot=g.lot, block=g.block, street=g.street, weight=g.weight)
    part.replace(f)
    return g


def ground(series: str) -> Ground:
    """The series' `Ground` as `make_ground` last wrote it; FileNotFoundError naming the command that makes it."""
    f = grid_path(series)
    if not f.is_file():
        raise FileNotFoundError(f"{f}: no grid labels; make them with `sfzmaps find labels --series {series}`")
    with np.load(f) as z:
        return Ground(z["lot"], z["block"], z["street"], z["weight"])


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


def page_shape(sid: SheetId) -> tuple[int, int]:
    """The page's (h, w), from its file's header."""
    with Image.open(frame_data.page_path(*sid, work=frame_data.WORK)) as im:
        return im.height, im.width


def series_codes(series: str) -> np.ndarray | None:
    """Each table lot's code in the table column the series' zoning borders are (`TABLE_BORDERS`), as downloaded and
    stripped, in lot-index order; a missing value is `MISSING_CODE` for its column, else "" (unknown).  None for a
    series whose zoning borders are not a table column."""
    col = pages.source(series).borders
    if col not in TABLE_BORDERS:
        return None
    return load_table()[col].fillna(MISSING_CODE.get(col, "")).astype(str).str.strip().to_numpy(dtype=str)


def sheet_path(sid: SheetId) -> Path:
    return pages.OUT / "labels" / sid[0] / f"{sid[1]}.npz"


def sheet_labels(sid: SheetId, which: str = "warp") -> dict:
    """A sheet's labels through its placement `which`, written to data/find/labels/<series>/<sheet>.npz and returned:
    `street` and `weight` (`page_ground`; none for a series not labelled from lots), `cuts` (`find.cuts.Cuts`: edge
    cuts for a series labelled from lots, then zoning-border cuts where its zoning borders are a table column; in the
    file as cuts_px, cuts_cut, cuts_prior, cuts_answer, cuts_says, cuts_weight), `edge_none` (px with a street-or-not
    label on no edge cut), `border_none` (px inside a lot with a known code on no zoning-border cut; for a series whose
    zoning borders are `none`, every px of the frame cut), `sigma` (`cuts.spread_px`) and `placement`.

    Labels count only where the sheet draws the blocks they are about (`find.drawn`): `drawn`, each block's chance of
    being drawn, from how many of its edge cuts cross ink (`drawn_fit`: pi, q_drawn, q_blank; `drawn_groups`; the
    blocks are `cuts.blocks` of the grid window `blocks_window`).  A px's street-or-not weight is multiplied by the
    chance of its block (or, off the blocks, its nearest block); each cut's weight is `Cuts.weighted`."""
    from sfzmaps.find import cuts, drawn

    series, _ = sid
    src = pages.source(series)
    lum = pages.load_scan(sid)
    shape = lum.shape
    keep = pages.load_keep(sid, shape)
    tr = placement(sid, which)
    sigma = cuts.spread_px(sid, which)
    found = cuts.Cuts.empty()
    street, weight = np.full(shape, UNKNOWN, np.uint8), np.zeros(shape, np.float32)
    chance, fit = drawn.drawn_chance(np.zeros(0), np.zeros(0))
    window = (0, 0, 0, 0)
    g = ground(series) if src.lots else None
    if g is not None:
        pg = page_ground(g, tr, shape, keep)
        street, weight = pg.street, pg.weight
        found = cuts.edge_cuts(g, tr, shape, keep, sigma)
        bl = cuts.blocks(g.block, tr, keep)
        window = bl.window
        inked = np.bincount(found.cut, lum.ravel()[found.px] < drawn.INK_BELOW, minlength=len(found.answer)) > 0
        owner = found.blocks[:, 0]
        chance, fit = drawn.drawn_chance(np.bincount(owner, inked, minlength=bl.count).astype(np.int64),
                                         np.bincount(owner, minlength=bl.count))
        found = found.weighted(chance)
        [near] = hatching.page_values(hatching.PageWorld(tr, shape), shape, [(bl.id, -1)], window[::2])
        weight = (weight * np.r_[chance, 0.0][near]).astype(np.float32)       # off the window (-1): nothing
    edge_none = (street != UNKNOWN) & ~found.on(shape)
    codes = series_codes(series)
    if codes is not None:
        border = cuts.border_cuts(g.lot, codes, tr, shape, keep, sigma, block=g.block).weighted(chance)
        found = cuts.Cuts.joined(found, border)
        r0, r1, c0, c1 = hatching._slice(tr, keep, g.lot.shape)
        [lot] = hatching.page_values(hatching.PageWorld(tr, shape), shape, [(g.lot[r0:r1, c0:c1], -1)], (r0, c0))
        known = np.r_[codes != "", False][lot]                  # lot -1 (none) reads the appended False
        border_none = known & keep & ~border.on(shape)
    else:
        border_none = keep.copy() if src.borders == "none" else np.zeros(shape, bool)
    out = {"street": street, "weight": weight, "cuts": found, "edge_none": edge_none, "border_none": border_none,
           "sigma": float(sigma), "placement": which, "drawn": chance, "drawn_fit": fit, "blocks_window": window}
    f = sheet_path(sid)
    f.parent.mkdir(parents=True, exist_ok=True)
    part = f.with_name(f".{f.name}.part")
    save_npz(part, street=street, weight=weight, cuts_px=found.px, cuts_cut=found.cut, cuts_prior=found.prior,
             cuts_answer=found.answer, cuts_says=found.says, cuts_weight=found.weight, edge_none=edge_none,
             border_none=border_none, sigma=np.float64(sigma), placement=np.array(which),
             drawn=chance.astype(np.float32), drawn_fit=np.array([fit["pi"], fit["q_drawn"], fit["q_blank"]]),
             drawn_groups=np.int64(fit["groups"]), blocks_window=np.array(window, np.int64))
    part.replace(f)
    return out
