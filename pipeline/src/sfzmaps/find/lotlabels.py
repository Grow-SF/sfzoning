"""Street-or-not labels from a series' lots: which ground is street and which is block, on the solver's 4 ft grid, and
the same read onto a scan's page through its sheet's placement.

Only a series whose `lots` is set in inputs/find/label_sources.csv is labelled from lots (`series_lots`): "table" from
the lots of the April 2000 table (`solver.table.load_table`, the three `2000-digital` series), "year" from the lots of
its year (`districts.read.load_parcels`).  On the grid (`Ground`): the lots painted (`solver.lots.lot_rasters`),
today's blocks the lots with their enclosed holes filled, and the street labels and weights of
`hatchnet.data.street_grid` from the blocks and today's centre lines (data/ref/streets.parquet), every street type with
the pooled half-widths: 1 street, 0 not street (a block), 255 no label.

Street where a centre line runs (`centre_line_street`, on the grid, before anything is read onto a page): every line
of the streets reference data, active or not, drawn on the grid (`hatchnet.data.centre_segments`).  A centre-line px
reaches as far as its distance to the nearest px that has a lot: a street is as wide on one side of its centre line as
on the other, and its near side is the block face.  A px with no lot is street when it lies within the reach of some
centre-line px (`within_reach`); any other px with no lot gets no label (255, weight 0), so it carries no "no block
edge" statement either.  A street label that stays keeps its weight; one the rule adds where the lots' labels had none
weighs 1 (nothing measured says how sure it is less than that).  No distance, width or size picked by a person decides
anything; the rule is exact up to the grid's own rounding: the lines are drawn to grid px (4 ft) and every distance is
between px centres.  What the rule cannot do, and does not guess at:
- where the centre line is not in the middle of the street, a sliver along the far block face gets no label;
- ground with no lot and no centre line near it (a plaza, a strip nobody named) gets no label;
- a road inside a lot (a park that is one lot: its drives) is still called block, because lots come first;
- where a centre line runs through ground with no lot far from any block (a bridge over the water, a road across open
  land), its reach is long and the rule calls a wide strip street.

Block px and the block-edge cuts (`find.cuts`, from the blocks alone) are not touched.

A sheet's labels (`sheet_labels`): the street-or-not labels read onto its page, the px said to carry no block edge, and
the block-edge cut statements of `find.cuts`.  The finder makes no label about zoning borders.  `ft_per_px`: the scale
of a box of a page through its placement (the trainer's items and print sizes, `find.data`; the test crops' feet,
`find.counts`).

Reads data/ref/parcels.parquet or the April 2000 table, data/ref/streets.parquet, a sheet's page, frame cut and
placement (data/warp/<series>/<sheet>/transform.json with its correction, or data/work/<series>/<sheet>/transform.json)
and data/warp/<series>/<sheet>/scores.json.  Writes data/find/labels/<series>/grid.npz (`make_ground`, with `off`,
made by `sfzmaps find labels`, read by everything else) and data/find/labels/<series>/<sheet>.npz (`sheet_labels`)."""
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
from sfzmaps.solver.hatchnet.data import STREET, UNKNOWN, Lines, centre_segments, lines_from, street_grid
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
    street, 0 not street, 255 no label) and their weights (1 not street, (0, 1] street, 0 no label); `off`: the px with
    no lot that the centre lines do not call street (`centre_line_street`)."""
    lot: np.ndarray            # int32
    block: np.ndarray          # bool
    street: np.ndarray         # uint8
    weight: np.ndarray         # float16
    off: np.ndarray            # bool


ENTRIES = 1 << 24           # engineering setting (memory only): the row runs laid down at once in `within_reach`


def within_reach(centre: np.ndarray, has_lot: np.ndarray) -> np.ndarray:
    """Bool (h, w): the px within the reach of some centre-line px of `centre`, a centre-line px's reach being its
    distance to the nearest px of `has_lot` (between px centres; 0 for one inside a lot; with no lot on the grid, no
    reach at all).  Exact on the grid, and the same whatever order the centre-line px come in: a px x is within
    reach when |x - p|^2 <= r_p^2 for some centre-line px p, all in whole numbers (r_p^2 is a squared distance between
    px centres, a whole number).  Taken in two passes, as a squared distance transform is: along each row, every
    centre-line px p lays down on its row the run of columns x with (x - px)^2 <= r_p^2, each keeping the lowest
    g = (x - px)^2 - r_p^2 any p gives it (<= 0); then down each column, a px (y', x) with g <= 0 covers the rows y with
    (y - y')^2 <= -g.  The two passes give the same set as testing every px against every centre-line px, with no cap
    and no step."""
    centre, has_lot = np.asarray(centre, bool), np.asarray(has_lot, bool)
    h, w = centre.shape
    out = np.zeros((h, w), bool)
    if not has_lot.any() or not centre.any():
        return out
    ys, xs = np.nonzero(centre)
    r2 = np.rint(ndimage.distance_transform_edt(~has_lot)[ys, xs] ** 2).astype(np.int64)
    half = np.floor(np.sqrt(r2)).astype(np.int64)                 # exact: r2 is a whole number far below 2**52
    runs = 2 * half + 1
    lowest = np.ones(h * w, np.int64)                             # 1: no centre-line px reaches the px along its row
    ends = np.cumsum(runs)
    first = 0
    while first < len(ys):
        last = max(first + 1, int(np.searchsorted(ends, ends[first] - runs[first] + ENTRIES, side="right")))
        n = runs[first:last]
        start = np.repeat(np.cumsum(n) - n, n)
        dx = np.arange(int(n.sum())) - start - np.repeat(half[first:last], n)
        x = np.repeat(xs[first:last], n) + dx
        ok = (x >= 0) & (x < w)
        np.minimum.at(lowest, (np.repeat(ys[first:last], n) * w + x)[ok], (dx * dx - np.repeat(r2[first:last], n))[ok])
        first = last
    at = np.flatnonzero(lowest <= 0)
    up = np.floor(np.sqrt(-lowest[at])).astype(np.int64)
    row, col = at // w, at % w
    edges = np.zeros((h + 1) * w, np.int64)
    np.add.at(edges, np.maximum(row - up, 0) * w + col, 1)
    np.add.at(edges, np.minimum(row + up + 1, h) * w + col, -1)
    return np.cumsum(edges.reshape(h + 1, w), axis=0)[:h] > 0


def centre_line_street(street: np.ndarray, weight: np.ndarray, block: np.ndarray, has_lot: np.ndarray,
                       centre: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The lots' street labels `street` and weights `weight` with the centre lines' rule (see the module): a px off
    `block` is street when it lies within the reach of some centre-line px (`within_reach`), keeping its weight where
    it was street already and weighing 1 where it was not; any other px off `block` has no label (255, weight 0).
    Block px are as they were.  (street, weight, off), `off` the px off `block` the rule does not call street."""
    block = np.asarray(block, bool)
    called = within_reach(centre, has_lot) & ~block
    was = np.asarray(street) == STREET
    street, weight = np.array(street, copy=True), np.array(weight, copy=True)
    off = ~block & ~called
    street[called] = STREET
    weight[called & ~was] = 1
    street[off] = UNKNOWN
    weight[off] = 0
    return street, weight, off


def build_ground(lots: gpd.GeoDataFrame, lines: Lines, shape: tuple[int, int] | None = None) -> Ground:
    """`Ground` of `lots` (EPSG:2227 ft) and today's centre `lines` on a grid of `shape` from the solver grid's corner
    (None: the whole grid): the lots' labels with the street label given by the centre lines (`centre_line_street`)."""
    lot, _ = lot_rasters(lots, shape)
    block = ndimage.binary_fill_holes(lot >= 0)
    street, weight, _ = street_grid(lines, block, per_type=False)
    centre = centre_segments(lines.geoms, block.shape).seg >= 0
    street, weight, off = centre_line_street(street, weight, block, lot >= 0, centre)
    return Ground(lot, block, street, weight, off)


def grid_path(series: str) -> Path:
    return pages.OUT / "labels" / series / GRID


def make_ground(series: str) -> Ground:
    """The series' `Ground` on the whole grid, made from `series_lots` and today's centre lines and written to
    data/find/labels/<series>/grid.npz (`sfzmaps find labels` makes it every time it runs)."""
    g = build_ground(series_lots(series), lines_from(config.REF_DIR / "streets.parquet"))
    f = grid_path(series)
    f.parent.mkdir(parents=True, exist_ok=True)
    part = f.with_name(f".{GRID}.part")
    save_npz(part, lot=g.lot, block=g.block, street=g.street, weight=g.weight, off=g.off)
    part.replace(f)
    return g


def ground(series: str) -> Ground:
    """The series' `Ground` as `make_ground` last wrote it; FileNotFoundError naming the command that makes it, also for
    a file made before the centre lines gave the street label (it has no `off`)."""
    f = grid_path(series)
    again = f"make them with `sfzmaps find labels --series {series}`"
    if not f.is_file():
        raise FileNotFoundError(f"{f}: no grid labels; {again}")
    with np.load(f) as z:
        if "off" not in z.files:
            raise FileNotFoundError(f"{f}: made before the centre lines gave the street label; {again}")
        return Ground(z["lot"], z["block"], z["street"], z["weight"], z["off"])


def placement(sid: SheetId, which: str = "warp") -> SheetTransform:
    """The sheet's placement: the warp's (data/warp/<series>/<sheet>/transform.json, with its correction) or, for
    "work", the one the reads were made through (data/work/<series>/<sheet>/transform.json)."""
    if which not in PLACEMENTS:
        raise ValueError(f"placement {which!r}: expected one of {', '.join(PLACEMENTS)}")
    folder = PLACEMENTS[which] / sid[0] / sid[1]
    return SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)


def ft_per_px(tr, box: tuple[int, int, int, int]) -> float:
    """The feet per page px at the box (x0, y0, w, h) through the placement `tr` (`to_world`, ft): the mean over its
    four sides of the side's length in feet over its length in px."""
    x0, y0, w, h = box
    corners = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], float)
    world = np.asarray(tr.to_world(corners), float)
    feet = np.linalg.norm(world - np.roll(world, -1, axis=0), axis=1)
    return float(np.mean(feet / np.array([w, h, w, h], float)))


def placement_check(sid: SheetId) -> dict | None:
    """The warp's own verdict on the sheet's placement, as it records it (`placement_check` in
    data/warp/<series>/<sheet>/scores.json: `passed`, `reason`); None when it recorded none."""
    f = PLACEMENTS["warp"] / sid[0] / sid[1] / "scores.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text()).get("placement_check")


def page_ft_per_px(tr, shape: tuple[int, int]) -> float:
    """`ft_per_px` over the whole page of `shape` (its box, centred on the page's centre): a sheet's print size."""
    return ft_per_px(tr, (0, 0, shape[1], shape[0]))


@dataclass(frozen=True)
class PageGround:
    """The street-or-not labels at every page px: 1 street, 0 block, 255 no label; their weights (0 no label); and, on
    the frame cut, the px with no lot the centre lines do not call street (`Ground.off`)."""
    street: np.ndarray         # uint8 (h, w)
    weight: np.ndarray         # float32 (h, w)
    off: np.ndarray            # bool (h, w)


def page_ground(g: Ground, tr, shape: tuple[int, int], keep: np.ndarray) -> PageGround:
    """The grid's labels read at every px of a page of `shape` through its placement `tr` (`hatching.PageWorld`,
    `hatching.page_values`); 255 with weight 0 off the grid and off `keep` (the frame cut's map area), and no px off
    there."""
    keep = np.asarray(keep, bool)
    if keep.shape != tuple(shape):
        raise ValueError(f"the frame cut is {keep.shape}, the page {tuple(shape)}")
    street, weight, off = hatching.page_values(hatching.PageWorld(tr, shape), shape,
                                               [(g.street, UNKNOWN), (g.weight, 0), (g.off, False)])
    weight = weight.astype(np.float32)
    street[~keep] = UNKNOWN
    weight[~keep] = 0
    return PageGround(street, weight, off & keep)


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
    (`drawn.use`), the labelled px (weight > 0) before and after; `street_left_off`: the px on the frame cut with no lot
    that the centre lines do not call street (the street labels the rule leaves off compared with "every px with no
    lot is street")."""
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
    counts.update(labelled_px_before=before, labelled_px_after=int((weight > 0).sum()),
                  street_left_off=int(pg.off.sum()))
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
