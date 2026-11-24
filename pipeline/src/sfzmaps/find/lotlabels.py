"""Street-or-not labels from a series' lots: which ground is street and which is block, on the solver's 4 ft grid, and
the same read onto a scan's page through its sheet's placement.

Only a series whose `lots` is set in inputs/find/label_sources.csv is labelled from lots (`series_lots`): "table" from
the lots of the April 2000 table (`solver.table.load_table`, the three `2000-digital` series), "year" from the lots of
its year (`districts.read.load_parcels`).  On the grid (`Ground`): the lots painted (`solver.lots.lot_rasters`),
today's blocks the lots with their enclosed holes filled, and the street labels and weights of
`hatchnet.data.street_grid` from the blocks and today's centre lines (data/ref/streets.parquet), every street type with
the pooled half-widths: 1 street, 0 not street (a block), 255 no label.

A sheet's labels (`sheet_labels`): the street-or-not labels read onto its page, the px said to carry no block edge, and
the block-edge cut statements of `find.cuts`.  The finder makes no label about zoning borders.

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

PLACEMENTS = {"warp": config.DATA_DIR / "warp", "work": config.WORK_DIR}
GRID = "grid.npz"


def series_lots(series: str) -> gpd.GeoDataFrame:
    """The lots that label a series: the April 2000 table's for `lots` "table", else the lots of its year.  ValueError
    for a series whose `lots` is not set."""
    src = pages.source(series)
    if not src.lots:
        raise ValueError(f"series {series}: its lots do not label it (no `lots` in {pages.SOURCES})")
    if src.lots == "table":
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


def sheet_path(sid: SheetId) -> Path:
    return pages.OUT / "labels" / sid[0] / f"{sid[1]}.npz"


def load_sheet_labels(sid: SheetId, which: str = "warp", shape: tuple[int, int] | None = None) -> dict:
    """The arrays of the sheet's label file (data/find/labels/<series>/<sheet>.npz, as `sheet_labels` wrote it), every
    reader's way in.  FileNotFoundError naming the command that makes a missing file; ValueError naming the command
    that makes it again when the file was made through another placement than `which`, or (`shape` given) does not
    fit a page of that shape."""
    f = sheet_path(sid)
    again = f"`sfzmaps find labels --series {sid[0]} --sheet {sid[1]} --placement {which}`"
    if not f.is_file():
        raise FileNotFoundError(f"{f}: no labels; make them with {again}")
    with np.load(f) as z:
        lab = {k: z[k] for k in z.files}
    if str(lab["placement"]) != which:
        raise ValueError(f"{f}: made through the {lab['placement']} placement, not {which}; make it again with {again}")
    if shape is not None and lab["street"].shape != tuple(shape):
        raise ValueError(f"{f}: labels are {lab['street'].shape}, the page {tuple(shape)}; make them again with "
                         f"{again}")
    return lab


OLD_EDGE = 0                # in a label file made before 2026-11-23, cuts_answer of a block-edge cut (1 was a
                            # zoning-border cut, which the finder no longer reads)


def file_cuts(lab: dict):
    """The block-edge cut statements of a label file's arrays (`load_sheet_labels`) as `find.cuts.Cuts` (the file
    holds no `inside`: -1).  A file made before 2026-11-23 also holds zoning-border cuts (`cuts_answer`): only its
    block-edge cuts are read."""
    from sfzmaps.find.cuts import Cuts
    n = len(lab["cuts_weight"])
    c = Cuts(lab["cuts_px"].astype(np.int64), lab["cuts_cut"].astype(np.int32), lab["cuts_prior"].astype(np.float32),
             lab["cuts_weight"].astype(np.float32), np.full(n, -1, np.int32), np.zeros(0, np.int64))
    return c.only(lab["cuts_answer"] == OLD_EDGE) if "cuts_answer" in lab else c


def sheet_labels(sid: SheetId, which: str = "warp") -> dict:
    """A sheet's labels through its placement `which`, written to data/find/labels/<series>/<sheet>.npz and returned:
    `street` and `weight` (`page_ground`), `cuts` (`find.cuts.edge_cuts`; in the file as cuts_px, cuts_cut, cuts_prior
    and cuts_weight), `edge_none` (px with a street-or-not label on no block-edge cut), `sigma` (`cuts.spread_px`),
    `placement` and `counts`.  ValueError for a series whose lots do not label it.

    Labels count only where the sheet draws ink (`find.drawn`): a cut not used has weight 0, and a px whose nearest
    block outline px is not inked has weight 0 and no `edge_none`.  A px on a cut the lots call for, kept or dropped at
    the frame cut, is not in `edge_none`.  `counts` (in the file as count_<name>): the cuts used and dropped
    (`drawn.use`), and the labelled px (weight > 0) before and after."""
    from sfzmaps.find import cuts, drawn

    series, _ = sid
    if not pages.source(series).lots:
        raise ValueError(f"series {series}: its lots do not label it (no `lots` in {pages.SOURCES})")
    lum = pages.load_scan(sid)
    shape = lum.shape
    keep = pages.load_keep(sid, shape)
    tr = placement(sid, which)
    sigma = cuts.spread_px(sid, which)
    g = ground(series)
    pg = page_ground(g, tr, shape, keep)
    street, weight = pg.street, pg.weight
    found = cuts.edge_cuts(g, tr, shape, keep, sigma)
    edge_none = (street != UNKNOWN) & ~found.on(shape, dropped=True)
    found, counts = drawn.use(found, lum)
    r0, r1, c0, c1 = cuts.window(tr, keep, g.block.shape)
    near_win = drawn.near_inked(g.block[r0:r1, c0:c1], found.inside[found.weight > 0])
    [near] = hatching.page_values(hatching.PageWorld(tr, shape), shape, [(near_win, False)], (r0, c0))
    before = int((weight > 0).sum())
    weight = (weight * near).astype(np.float32)
    edge_none &= near
    counts.update(labelled_px_before=before, labelled_px_after=int((weight > 0).sum()))
    out = {"street": street, "weight": weight, "cuts": found, "edge_none": edge_none, "sigma": float(sigma),
           "placement": which, "counts": counts}
    f = sheet_path(sid)
    f.parent.mkdir(parents=True, exist_ok=True)
    part = f.with_name(f".{f.name}.part")
    save_npz(part, street=street, weight=weight, cuts_px=found.px, cuts_cut=found.cut, cuts_prior=found.prior,
             cuts_weight=found.weight, edge_none=edge_none, sigma=np.float64(sigma), placement=np.array(which),
             **{f"count_{k}": np.int64(v) for k, v in counts.items()})
    part.replace(f)
    return out
