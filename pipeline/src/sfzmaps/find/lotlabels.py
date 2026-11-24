"""Street-or-not labels from a series' lots: which ground is street and which is block, on the solver's 4 ft grid, and
the same read onto a scan's page through its sheet's placement.

Only a series whose `lots` is set in inputs/find/label_sources.csv is labelled from lots (`series_lots`): "table" from
the lots of the April 2000 table (`solver.table.load_table`, the three `2000-digital` series), "year" from the lots of
its year (`districts.read.load_parcels`).  On the grid (`Ground`): the lots painted (`solver.lots.lot_rasters`),
today's blocks the lots with their enclosed holes filled, and the street labels and weights of
`hatchnet.data.street_grid` from the blocks and today's centre lines (data/ref/streets.parquet), every street type with
the pooled half-widths: 1 street, 0 not street (a block), 255 no label.

Street only between blocks (`street_between`, on the grid, before anything is read onto a page): a px with no lot keeps
its street label only where it lies between blocks: going straight away from its nearest block px, the ray reaches a
block px before the grid ends.  Anywhere else it gets no label (255, weight 0), so it carries no "no block edge"
statement either: nothing is known about what is printed there (the beach and the ocean beyond the last blocks, open
land past a block's outer side).  No distance, width or size decides it.  "Reaches" is numerics only: the ray reaches
a block when one of its samples falls on a block px.  Far from blocks the samples are the distance to the nearest
block less root two apart, and then no point between two samples can fall on a block px; near a block they are held
one grid step apart, and then a ray can graze a single corner px of a block between samples, or slip between two
block px that touch only at a corner, without a sample falling on either.  That miss only ever takes a label away
(measured against a dense reference on random grids: 36 of 59,694 px differed, all towards "no label").
What the rule cannot do, and does not guess at:
- a street with lots on one side only (a road along the shore) gets no label;
- a lot-free expanse enclosed by blocks on every side (a plaza, or a park that has no lot) is still called street;
- where a px's ray runs lengthwise down a street (the square of a T-junction under the stem's mouth, or a crossing
  whose two halves are offset), the px is called street only if some block closes that street before the grid ends;
  where the street runs off the edge of the lots with nothing beyond, that patch gets no label.  On the real sheets
  this happens at the city's edge only: 2000 zoning sheet 7, height sheet 7 and 1980-1990 sheet 7 lost no street
  label at all; the labels lost are at the ocean (zoning sheet 5) and the county line (zoning sheet 11);
- the open inner corner of an L-shaped block gets no label (every ray from it runs alongside the other arm, out into
  the open); a notch cut into a block is street between its sides, but at its closed end, where nothing lies beyond
  the mouth, the px whose nearest block px is the end get no label.
Where the label stays, it and its weight are exactly as before; block px and the block-edge
cuts (`find.cuts`, from the blocks alone) are not touched: a block's side facing the ocean still has its cuts.

A sheet's labels (`sheet_labels`): the street-or-not labels read onto its page, the px said to carry no block edge, and
the block-edge cut statements of `find.cuts`.  The finder makes no label about zoning borders.  `ft_per_px`: the scale
of a box of a page through its placement (the trainer's items and print sizes, `find.data`; the test crops' feet,
`find.counts`).

Reads data/ref/parcels.parquet or the April 2000 table, data/ref/streets.parquet, a sheet's page, frame cut and
placement (data/warp/<series>/<sheet>/transform.json with its correction, or data/work/<series>/<sheet>/transform.json)
and data/warp/<series>/<sheet>/scores.json.  Writes data/find/labels/<series>/grid.npz (`make_ground`, with `lost`,
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
from sfzmaps.solver.hatchnet.data import STREET, UNKNOWN, Lines, lines_from, street_grid
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
    street, 0 not street, 255 no label) and their weights (1 not street, (0, 1] street, 0 no label), and the px whose
    street label `street_between` took away (`lost`)."""
    lot: np.ndarray            # int32
    block: np.ndarray          # bool
    street: np.ndarray         # uint8
    weight: np.ndarray         # float16
    lost: np.ndarray           # bool


RAYS = 1 << 20              # engineering setting (memory only): the rays stepped together


def between_blocks(block: np.ndarray, which: np.ndarray | None = None) -> np.ndarray:
    """Bool (h, w): the px of `which` (None: every px off `block`) that lie between blocks: going straight away from
    the px's nearest block px, the ray reaches a block px before the grid ends (see the module).  Each ray starts at its
    px and is sampled at distances t = 0, t1, ...; a sample at q looks at the px r nearest q (rounded) and its distance
    D to the nearest block px, and the next sample is max(1, D - sqrt 2) further on (numerics, not a setting).  While
    the step is D - sqrt 2, no point before the next sample can round to a block px; where it is held at one grid step
    (D under 1 + sqrt 2, near a block), the ray can graze a single corner px of a block between samples, or slip
    between two block px touching only at a corner, and miss them (only ever taking a label away).  A block px a
    sample falls on is reached; a sample off the grid ends the ray unreached.  With no block on the grid, no px lies between blocks."""
    block = np.asarray(block, bool)
    h, w = block.shape
    which = ~block if which is None else np.asarray(which, bool) & ~block
    out = np.zeros(block.shape, bool)
    if not block.any() or not which.any():
        return out
    dist, (ny, nx) = ndimage.distance_transform_edt(~block, return_indices=True)
    ys, xs = np.nonzero(which)
    hit = np.zeros(len(ys), bool)
    for c0 in range(0, len(ys), RAYS):
        y0, x0 = ys[c0:c0 + RAYS].astype(np.float64), xs[c0:c0 + RAYS].astype(np.float64)
        uy, ux = y0 - ny[ys[c0:c0 + RAYS], xs[c0:c0 + RAYS]], x0 - nx[ys[c0:c0 + RAYS], xs[c0:c0 + RAYS]]
        norm = np.hypot(uy, ux)
        uy, ux = uy / norm, ux / norm
        t = np.zeros(len(y0))
        live = np.arange(len(y0))
        reached = np.zeros(len(y0), bool)
        while len(live):
            ry = np.rint(y0[live] + t[live] * uy[live]).astype(np.int64)
            rx = np.rint(x0[live] + t[live] * ux[live]).astype(np.int64)
            on = (ry >= 0) & (ry < h) & (rx >= 0) & (rx < w)
            live, ry, rx = live[on], ry[on], rx[on]
            d = dist[ry, rx]
            landed = d == 0
            reached[live[landed]] = True
            live, d = live[~landed], d[~landed]
            t[live] += np.maximum(1.0, d - np.sqrt(2))
        hit[c0:c0 + RAYS] = reached
    out[ys, xs] = hit
    return out


def street_between(street: np.ndarray, weight: np.ndarray, block: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The street labels `street` and weights `weight` with the street label kept only between blocks
    (`between_blocks`; see the module): (street, weight, lost), `lost` the px whose street label was taken away (now
    255, weight 0).  Every other px is as it was."""
    on = np.asarray(street) == STREET
    lost = on & ~between_blocks(block, on)
    street, weight = np.array(street, copy=True), np.array(weight, copy=True)
    street[lost] = UNKNOWN
    weight[lost] = 0
    return street, weight, lost


def build_ground(lots: gpd.GeoDataFrame, lines: Lines, shape: tuple[int, int] | None = None) -> Ground:
    """`Ground` of `lots` (EPSG:2227 ft) and today's centre `lines` on a grid of `shape` from the solver grid's corner
    (None: the whole grid), the street label kept only between blocks (`street_between`)."""
    lot, _ = lot_rasters(lots, shape)
    block = ndimage.binary_fill_holes(lot >= 0)
    street, weight, _ = street_grid(lines, block, per_type=False)
    street, weight, lost = street_between(street, weight, block)
    return Ground(lot, block, street, weight, lost)


def grid_path(series: str) -> Path:
    return pages.OUT / "labels" / series / GRID


def make_ground(series: str) -> Ground:
    """The series' `Ground` on the whole grid, made from `series_lots` and today's centre lines and written to
    data/find/labels/<series>/grid.npz (`sfzmaps find labels` makes it every time it runs)."""
    g = build_ground(series_lots(series), lines_from(config.REF_DIR / "streets.parquet"))
    f = grid_path(series)
    f.parent.mkdir(parents=True, exist_ok=True)
    part = f.with_name(f".{GRID}.part")
    save_npz(part, lot=g.lot, block=g.block, street=g.street, weight=g.weight, lost=g.lost)
    part.replace(f)
    return g


def ground(series: str) -> Ground:
    """The series' `Ground` as `make_ground` last wrote it; FileNotFoundError naming the command that makes it, also for
    a file made before the street label was kept only between blocks (it has no `lost`)."""
    f = grid_path(series)
    again = f"make them with `sfzmaps find labels --series {series}`"
    if not f.is_file():
        raise FileNotFoundError(f"{f}: no grid labels; {again}")
    with np.load(f) as z:
        if "lost" not in z.files:
            raise FileNotFoundError(f"{f}: made before the street label was kept only between blocks; {again}")
        return Ground(z["lot"], z["block"], z["street"], z["weight"], z["lost"])


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
    """The street-or-not labels at every page px: 1 street, 0 block, 255 no label; their weights (0 no label); and the
    px on the frame cut whose street label the grid's `street_between` took away (`Ground.lost`)."""
    street: np.ndarray         # uint8 (h, w)
    weight: np.ndarray         # float32 (h, w)
    lost: np.ndarray           # bool (h, w)


def page_ground(g: Ground, tr, shape: tuple[int, int], keep: np.ndarray) -> PageGround:
    """The grid's labels read at every px of a page of `shape` through its placement `tr` (`hatching.PageWorld`,
    `hatching.page_values`); 255 with weight 0 off the grid and off `keep` (the frame cut's map area), and no px lost
    there."""
    keep = np.asarray(keep, bool)
    if keep.shape != tuple(shape):
        raise ValueError(f"the frame cut is {keep.shape}, the page {tuple(shape)}")
    street, weight, lost = hatching.page_values(hatching.PageWorld(tr, shape), shape,
                                                [(g.street, UNKNOWN), (g.weight, 0), (g.lost, False)])
    weight = weight.astype(np.float32)
    street[~keep] = UNKNOWN
    weight[~keep] = 0
    return PageGround(street, weight, lost & keep)


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
    (`drawn.use`), the labelled px (weight > 0) before and after, and `street_label_lost`: the px on the frame cut
    whose street label `street_between` took away (not between blocks)."""
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
                  street_label_lost=int(pg.lost.sum()))
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
