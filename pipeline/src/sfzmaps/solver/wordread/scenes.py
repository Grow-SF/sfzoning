"""Map-like synthetic scenes for the text detector's training (item 8 of the map-text plan; `train.Recipe.scenes`):
DataSF's geometry drawn as the maps draw it, at a real sheet's scale and in its frame, with real words pasted where
the maps print labels, so every head's masks are exact.  No number here decides by hand: every width, presence,
offset, rate and choice of word is drawn from what the training quarters show (`measure`).

The geometry (`Geometry`, EPSG:2227 ft): the Assessor's lots (`lots/table.parquet`: `block_num` and the outline only,
never a code), the blocks (the union of each block's lots) and DataSF's street centrelines
(`context.sf_streets`).  Three kinds of line (LINE_KINDS), each kept by item:
  block    a block's outline: the street edges
  lot      a block's lot lines (its lots' outlines less its own)
  street   a centreline (the maps draw few: district lines down a street)
On a page they are cut into EDGES (`page_edges`): straight runs between the vertices the page can show (DataSF puts a
vertex at every lot corner along a block's straight side, half of them within 0.003 ft of the line; a vertex within
SIMPLIFY_PX of the line through its neighbours is dropped: less than half a px off the line, it can't make a visible
corner at the page's resolution, a physical limit).

What is held out (`usable`): the areas (a quarter, the answered blocks) as shapes on the map; a block is usable when
its outline (and so every lot in it) meets none, a centreline when it meets none.  An answered-blocks area
(`data.GridArea`) is the grid px of the mask grown by one px, traced through their centres, so a line that only
clips a held px's corner still meets it.  A word is usable when no px of its rectangle is held (`data._touch_held`).
Nothing of a held-out area is drawn, pasted or measured.

A pasteable word (`pasteable`): a Terra word wholly on its page whose every 8-connected ink piece touching its
rectangle lies wholly inside it (the structural half of `linegate.clean_word_mask`, with no ring of px around it): its
own ink is those pieces, an exact mask.  A word whose touching ink runs outside its rectangle (joined to a line) can't
be cut out exactly and is never pasted.  Each is kept with its text (Terra's, `norm`), kind, reading angle and
rectangle (page px).

What is measured (by `SceneBank.build`, `measure_map`), per map, on each sheet's training px of its printed map
only (its footprint, `load_footprint`, less the held-out areas):
  lines    the edges of every kind on the page (through the inverse placement, `_ToPage`), drawn 1 px wide; each
           training px's nearest edge and its distance d (`_sheet_lines`).  Over the skeleton px of the sheet's line
           ink (its ink outside every Terra word's and other pass's rectangle; the run rule's line px,
           `SheetData.line_bits`, miss a block's short edges), d is either the nearest edge's kind's own ink (the
           printed line about the projected one, a Gaussian of spread sigma: g(d) = exp(-d^2 / 2 sigma^2)) or anything
           else, each weighed by n_b(d), the count of the page's training px nearest that kind at that distance (the
           geometry of the px around a line): P(d | own) ~ n_b(d) g(d), P(d | other) ~ n_b(d); a share a per kind
           and ONE sigma for every kind (the placement that moves the lines is one), by EM (`fit_near`).  (Tried first:
           an offset mu and a sigma per kind.  The offset came out 0 for block and lot lines on every map; and a
           centreline's own spread, fitted alone, covered the whole street on the 1980-2000 maps (sigma 17-21 px
           against 0.8-1.0 px for the blocks and lots): the street's untranscribed text, not a line.)  Each edge's
           INKED FRACTION: its own px (the responsibilities of the skeleton px nearest it) out of its px outside
           every word's box (at most 1: a line covers no more than itself); the kind's measured distribution of it,
           one value an edge, is what scenes draw from (the length-weighted share reported beside it).  The thickness: the own px' thickness (2
           x the distance to paper at the skeleton - 1), explained as a mixture of the thicknesses `cv2.line` draws
           (`thickness_shares`; each thickness's profile measured by drawing it at every angle)
  codes    each zoning-code word's centre less the centroid of the lot nearest it (ft); its rate per lot
  blocks   each block-number word that lies in the block its text names (`context.block_forms`): its centre less
           the block's centroid (ft); its rate per block
  streets  each street-name word, with the centreline nearest it among those whose street prints its text
           (`context.street_forms`): its signed offset across the centreline (ft) and the angle between its axis
           and the centreline's (0-90 degrees).  The offsets are a mixture of the street's own (half-normal) and
           misreads naming a street elsewhere (exponential), by EM from every decile start, the likeliest kept
           (`fit_far`); the near rate per ft of named centreline is the near responsibilities over the length; the
           angle's density a reflected Gaussian kernel density of the near words' angles, its bandwidth by
           leave-one-out likelihood (`context.loo_bandwidth`)
Every rate counts per sheet (a lot on two sheets counts twice, as its words can).  A word is measured when usable and
its centre is on the printed map; a lot, block or centreline when usable and its centroid (a centreline's middle) is
on the printed map at a training px.  The pasteable words are those whose centre is on the printed map (not the
legend's).

A scene (`SceneBank.crop(rng, size, scale, turns)`), from the generator alone:
  - where: a centre as the real crops draw theirs (`train.TEXT_SHARE` on text, the rest on ink; `data.centres` of
    the sheets with the areas held out, on the printed map), its window of size / scale page px (moved inside the
    page), and the sheet's placement over the window as one affine map (least squares on its samples there,
    `window_affine`)
  - lines: the edges of every usable item reaching the window (`page_edges`), each inked over a fraction drawn from
    its kind's measured per-edge inked fractions, as one contiguous run at a uniform position along the edge (a kind
    whose fractions sit at 0 and 1 draws whole edges or none), at a thickness drawn from its kind's thickness shares
  - labels: each usable block with the block rate (at its centroid plus a measured block offset), each lot of a
    usable block with the code rate (at its centroid plus a measured code offset), and along each usable named
    centreline a Poisson count at the near street rate x its length (uniform along it, offset across it by a
    measured near offset).  Each label is a pasteable word of its map: of the block's number; any code word; a word
    its street prints, drawn by the angle density at its angle to the centreline.  With no word of that string, a
    same-kind stand-in: a pasteable word of the label's kind (block number, street name), drawn alike (a street
    name's by the same angle density), recorded as one (`standin`); the detector's targets never read a label's
    string, only the reader does, and the reader doesn't train on scenes.  No word of the kind either: skipped.
    The labels are placed in a random order, each at its point (its rectangle's centre moved by whole px, so its
    ink is the page's, unresampled); one whose rectangle would meet a placed one is skipped (two words never share
    a core px)
  - targets: `data.crop_targets(..., words=True)` of the scene as a sheet (its words the pasted rectangles, their
    cores the same: `shrink.prepare`'s rule), so scenes and real crops share every target builder; then made exact:
    text is the pasted words' own ink alone (`text` ⊆ ink), a line px inside a word's rectangle is known not-text,
    and the paper inside a word's rectangle is unknown (as the composites' `_exact`: a lesson neither of text nor of
    not-text)
`tally` counts what the scenes held; `last` says what the last one was made of (for pictures, measures and tests).

No stage imports this module (`train` takes the bank as any object with `.scene`), so no stage's import closure
changes with it.  `SceneBank.build` draws nothing (its `seed` is kept for the run's record).
"""
import collections
import dataclasses
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, lru_cache

import cv2
import numpy as np
import shapely
from scipy import optimize
from scipy.interpolate import LinearNDInterpolator
from skimage.morphology import skeletonize

from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import boxes, data
from sfzmaps.solver.wordread import context

log = logging.getLogger(__name__)

LINE_KINDS = ("block", "lot", "street")      # a px as near two kinds at once is the first's
CODE, BLOCK_NUMBER, STREET_NAME = data.KINDS["zoning code"], data.KINDS["block number"], data.KINDS["street name"]
SHIFT = 4                    # cv2 drawing's fractional bits (numerical: a line's ends at 1/16 px)
CALIBRATION_ANGLES = np.arange(0.0, 180.0, 1.0)    # the angles a thickness is drawn at to measure it (numerical)
CALIBRATION_LENGTH = 160     # px, the length of each calibration line (numerical)
EM_TOL = 1e-10               # an EM stops when its log-likelihood rises by less (relative) ...
EM_MAX_ITER = 5000           # ... or after this many iterations (reported when reached)
ANGLE_GRID = np.linspace(0.0, 90.0, 9001)   # the street angle density's table (0.01 degree; numerical)
DECILES = np.arange(1, 10) / 10              # fit_far's starts
BIG = 1e9                    # ft: a quarter's shape reaches this far past the split (beyond the city; numerical)
SIMPLIFY_PX = 0.5            # px: a vertex less than half a px off the straight line through its neighbours can't
#                              make a visible corner at the page's resolution (the px grid): a physical limit of the
#                              drawing, not a decision about the data


def norm(text) -> str:
    """A word's text as the forms are spelled: upper case, one space between words."""
    return " ".join(str(text or "").upper().split())


# --- the geometry --------------------------------------------------------------------------------------------------

def _segments(geoms) -> tuple[np.ndarray, np.ndarray]:
    """The straight segments of line geometries (any (multi)line, ring or collection): float64 (n, 2, 2) and the
    index of the geometry each came from."""
    geoms = np.asarray(geoms, object)
    parts, owner = shapely.get_parts(geoms, return_index=True)
    lines = np.isin(shapely.get_type_id(parts), (1, 2))          # lines and rings (an overlay's stray points dropped)
    parts, owner = parts[lines], owner[lines]
    xy, idx = shapely.get_coordinates(parts, return_index=True)
    same = idx[1:] == idx[:-1]
    seg = np.stack([xy[:-1][same], xy[1:][same]], 1)
    keep = (seg[:, 0] != seg[:, 1]).any(1)
    return seg[keep], owner[idx[:-1][same]][keep]


@dataclasses.dataclass
class Geometry:
    """DataSF's geometry (see the module): the blocks and lots, each block's outline and lot lines, the centrelines."""
    block_poly: np.ndarray     # shapely (b,): each block's outline (the union of its lots)
    block_number: np.ndarray   # str (b,): the Assessor's number
    lot_poly: np.ndarray       # shapely (l,)
    lot_block: np.ndarray      # int64 (l,): each lot's block
    lot_lines: np.ndarray      # shapely (b,): each block's lot lines (its lots' outlines less its own)
    lines: np.ndarray          # shapely LineStrings (c,): the centrelines
    line_forms: tuple          # (c,) tuples: the strings each centreline's street prints (`context.street_forms`)

    @classmethod
    def from_frames(cls, lots, streets) -> "Geometry":
        """From the lots (`block_num`, geometry) and the centrelines (`street`, `type`, geometry; `context.sf_streets`'
        columns), both in EPSG:2227 ft."""
        lots = lots[lots["block_num"].notna() & lots.geometry.notna() & ~lots.geometry.is_empty]
        poly = shapely.make_valid(np.asarray(lots.geometry.values, object))
        numbers, lot_block = np.unique(lots["block_num"].astype(str).to_numpy(), return_inverse=True)
        order = np.argsort(lot_block, kind="stable")
        groups = np.split(order, np.flatnonzero(np.diff(lot_block[order])) + 1) if len(order) else []
        blocks = np.array([shapely.union_all(poly[g]) for g in groups], object)
        inner = np.array([shapely.difference(shapely.union_all(shapely.boundary(poly[g])), shapely.boundary(b))
                          for g, b in zip(groups, blocks, strict=True)], object)
        forms = tuple(tuple(context.street_forms(s, t)) for s, t in zip(streets["street"], streets["type"],
                                                                         strict=True))
        return cls(block_poly=blocks, block_number=numbers.astype(str), lot_poly=poly,
                   lot_block=lot_block.astype(np.int64), lot_lines=inner,
                   lines=np.asarray(streets.geometry.values, object), line_forms=forms)

    @classmethod
    def load(cls) -> "Geometry":
        """San Francisco's: the lots stage's table (its block numbers and outlines) and DataSF's centrelines."""
        import geopandas as gpd
        lots = gpd.read_parquet(paths.LOTS_DIR / "table.parquet", columns=["block_num", "geometry"])
        return cls.from_frames(lots.to_crs(context.CRS), context.sf_streets())

    @cached_property
    def outlines(self) -> np.ndarray:
        return shapely.boundary(self.block_poly)

    def items(self, kind: int) -> np.ndarray:
        """The geometries of a LINE_KINDS kind: each block's outline, each block's lot lines, each centreline."""
        return (self.outlines, self.lot_lines, self.lines)[kind]

    def item_tree(self, kind: int) -> shapely.STRtree:
        """The tree of a kind's items: the blocks (outlines and lot lines are inside them) or the centrelines."""
        return self.block_poly_tree if kind < 2 else self.line_tree

    @cached_property
    def block_poly_tree(self) -> shapely.STRtree:
        return shapely.STRtree(self.block_poly)

    @cached_property
    def named(self) -> np.ndarray:
        """Whether each centreline's street prints anything (`context.street_forms`)."""
        return np.array([bool(f) for f in self.line_forms], bool)

    @cached_property
    def block_centre(self) -> np.ndarray:
        return shapely.get_coordinates(shapely.centroid(self.block_poly)).reshape(-1, 2)

    @cached_property
    def lot_centre(self) -> np.ndarray:
        return shapely.get_coordinates(shapely.centroid(self.lot_poly)).reshape(-1, 2)

    @cached_property
    def block_tree(self) -> shapely.STRtree:
        return shapely.STRtree(shapely.points(self.block_centre))

    @cached_property
    def lot_tree(self) -> shapely.STRtree:
        return shapely.STRtree(self.lot_poly)

    @cached_property
    def lot_centre_tree(self) -> shapely.STRtree:
        return shapely.STRtree(shapely.points(self.lot_centre))

    @cached_property
    def line_tree(self) -> shapely.STRtree:
        return shapely.STRtree(self.lines)

    @cached_property
    def line_length(self) -> np.ndarray:
        return shapely.length(self.lines)

    @cached_property
    def form_lines(self) -> dict[str, np.ndarray]:
        """Each printed form: the centrelines whose street prints it."""
        out = collections.defaultdict(list)
        for c, forms in enumerate(self.line_forms):
            for f in forms:
                out[f].append(c)
        return {f: np.array(v, np.int64) for f, v in out.items()}

    @cached_property
    def block_of_form(self) -> dict[str, np.ndarray]:
        """Each printed form: the blocks whose number prints it."""
        out = collections.defaultdict(list)
        for b, n in enumerate(self.block_number):
            for f in context.block_forms(n):
                out[f].append(b)
        return {f: np.array(v, np.int64) for f, v in out.items()}

    @cached_property
    def _form_trees(self) -> dict:
        return {}

    def form_tree(self, form: str) -> shapely.STRtree:
        """The centrelines whose street prints `form`, as a tree (made once a form)."""
        if form not in self._form_trees:
            self._form_trees[form] = shapely.STRtree(self.lines[self.form_lines[form]])
        return self._form_trees[form]


# --- what is held out ----------------------------------------------------------------------------------------------

def _quadrant(q: int) -> shapely.Polygon:
    sx, sy = grid.QUARTER_SPLIT
    x0, x1 = (sx, sx + BIG) if q % 2 == 1 else (sx - BIG, sx)
    y0, y1 = (sy, sy + BIG) if q < 2 else (sy - BIG, sy)
    return shapely.box(x0, y0, x1, y1)


def _grid_shape(area: data.GridArea):
    """The grid px of an answered-blocks area grown by one px (the 8 around each), traced through their centres, on
    the map: it covers every held px wholly (see the module)."""
    if not area.mask.size or not area.mask.any():
        return None
    grown = cv2.dilate(np.pad(area.mask, 2).view(np.uint8), np.ones((3, 3), np.uint8))
    contours, hierarchy = cv2.findContours(grown, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    to_map = [grid.to_map(c.reshape(-1, 2) + 0.5 + [area.col0 - 2, area.row0 - 2]) for c in contours]
    polys = []
    for i, (_, _, child, parent) in enumerate(hierarchy[0]):
        if parent >= 0:
            continue
        holes, k = [], child
        while k >= 0:
            if len(to_map[k]) >= 3:
                holes.append(to_map[k])
            k = hierarchy[0][k][0]
        ring = to_map[i]
        polys.append(shapely.Polygon(ring, holes) if len(ring) >= 3 else shapely.Point(ring[0]).buffer(1.0))
    return shapely.union_all(shapely.make_valid(np.array(polys, object)))


def held_shapes(areas) -> list:
    """The held-out areas as shapes on the map (ft): a rectangle, a quarter, the answered blocks (see the module)."""
    out = []
    for a in areas:
        if isinstance(a, data.MapRect):
            out.append(shapely.box(a.x0, a.y0, a.x1, a.y1))
        elif isinstance(a, data.Quarter):
            out.append(_quadrant(a.q))
        elif isinstance(a, data.GridArea):
            shape = _grid_shape(a)
            if shape is not None:
                out.append(shape)
        else:
            raise TypeError(f"no shape for a held-out area of type {type(a).__name__}")
    return out


def usable(geo: Geometry, areas) -> tuple[np.ndarray, np.ndarray]:
    """Which blocks and which centrelines meet no held-out area, bool each."""
    block_ok = np.ones(len(geo.block_poly), bool)
    line_ok = np.ones(len(geo.lines), bool)
    for shape in held_shapes(areas):
        block_ok &= ~shapely.intersects(geo.block_poly, shape)
        line_ok &= ~shapely.intersects(geo.lines, shape)
    return block_ok, line_ok


def item_ok(kind: int, block_ok: np.ndarray, line_ok: np.ndarray) -> np.ndarray:
    """Which items of a kind are usable: their block's (outlines, lot lines) or the centreline's."""
    return block_ok if kind < 2 else line_ok


def page_edges(geoms, to_page) -> tuple[np.ndarray, np.ndarray]:
    """Geometries' straight edges on a page (px): each mapped by `to_page` ((n, 2) map ft -> page px), its vertices
    the page can't show dropped (`shapely.simplify` by SIMPLIFY_PX), and cut into edges between the vertices left;
    float64 (n, 2, 2) and the index of the geometry each came from.  DataSF's outlines carry a vertex at every lot
    corner along a straight side (half of them within 0.003 ft of the line): on the page they are one edge."""
    geoms = np.asarray(geoms, object)
    if not len(geoms):
        return np.zeros((0, 2, 2)), np.zeros(0, np.int64)
    page = shapely.transform(geoms, to_page)
    return _segments(shapely.simplify(page, SIMPLIFY_PX, preserve_topology=False))


# --- a sheet's placement -------------------------------------------------------------------------------------------

def page_to_world(sd: data.SheetData, xy) -> np.ndarray:
    """The map point (ft) of page points (x, y; float px), bilinear between the placement's samples (as `data._world_at`
    does for whole px)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    step = data.PLACEMENT_STEP
    rows, cols = sd.world.shape[:2]
    jc = np.clip(np.floor(xy[:, 0] / step).astype(np.int64), 0, cols - 2)
    jr = np.clip(np.floor(xy[:, 1] / step).astype(np.int64), 0, rows - 2)
    fc, fr = (xy[:, 0] / step - jc)[:, None], (xy[:, 1] / step - jr)[:, None]
    w = sd.world
    return ((1 - fr) * ((1 - fc) * w[jr, jc] + fc * w[jr, jc + 1]) + fr * ((1 - fc) * w[jr + 1, jc]
                                                                            + fc * w[jr + 1, jc + 1]))


def window_affine(sd: data.SheetData, y0: int, x0: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """The sheet's placement over a page window as one affine map, fitted by least squares on its samples there (those
    of the sample cells the window covers): page (x, y, 1) -> map, (2, 3), and its inverse, map (x, y, 1) -> page."""
    step = data.PLACEMENT_STEP
    rows, cols = sd.world.shape[:2]
    r0, r1 = max(y0 // step, 0), min(-(-(y0 + h) // step), rows - 1)
    c0, c1 = max(x0 // step, 0), min(-(-(x0 + w) // step), cols - 1)
    r0, c0 = min(r0, rows - 2), min(c0, cols - 2)
    r1, c1 = max(r1, r0 + 1), max(c1, c0 + 1)
    gr, gc = np.mgrid[r0:r1 + 1, c0:c1 + 1]
    page = np.c_[gc.ravel() * step, gr.ravel() * step, np.ones(gr.size)].astype(float)
    world = sd.world[gr.ravel(), gc.ravel()]
    fwd = np.linalg.lstsq(page, world, rcond=None)[0].T                 # (2, 3)
    lin = np.linalg.inv(fwd[:, :2])
    return fwd, np.c_[lin, -lin @ fwd[:, 2]]


def _apply(m: np.ndarray, xy) -> np.ndarray:
    xy = np.asarray(xy, float)
    return xy @ m[:, :2].T + m[:, 2]


class _ToPage:
    """The inverse of a sheet's placement: map (ft) -> page (x, y) px, piecewise linear over its samples and, past
    them, the affine map fitted to them all (least squares)."""

    def __init__(self, sd: data.SheetData):
        rows, cols = sd.world.shape[:2]
        gr, gc = np.mgrid[0:rows, 0:cols]
        page = np.c_[gc.ravel(), gr.ravel()].astype(float) * data.PLACEMENT_STEP
        world = sd.world.reshape(-1, 2)
        self.linear = LinearNDInterpolator(world, page)
        self.affine = np.linalg.lstsq(np.c_[world, np.ones(len(world))], page, rcond=None)[0].T

    def __call__(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        out = self.linear(xy).reshape(-1, 2)
        off = ~np.isfinite(out).all(1)
        out[off] = _apply(self.affine, xy[off])
        return out


def page_from_world(sd: data.SheetData) -> _ToPage:
    """The inverse of the sheet's placement (`_ToPage`)."""
    return _ToPage(sd)


def _footprint(sd: data.SheetData):
    return shapely.convex_hull(shapely.multipoints(sd.world.reshape(-1, 2)))


def _held_page(sd: data.SheetData) -> np.ndarray:
    """The sheet's held-out px, bool (height, width), found data.BAND rows at a time."""
    out = np.zeros((sd.height, sd.width), bool)
    for y0 in range(0, sd.height, data.BAND):
        h = min(data.BAND, sd.height - y0)
        out[y0:y0 + h] = data._held(sd, y0, 0, h, sd.width)
    return out


def _held_points(sd: data.SheetData, xy: np.ndarray) -> np.ndarray:
    """Which map points are in an area the sheet holds out."""
    out = np.zeros(len(xy), bool)
    for a in sd.held:
        out |= a.contains(xy[:, 0], xy[:, 1])
    return out


def load_footprint(sd: data.SheetData):
    """The sheet's printed map on the map (ft): the warp's content footprint (`warped.footprint.geojson` beside its
    placement: the neatline's interior less the legend and mask boxes); None when there is none."""
    import json
    f = data.warp_dir(sd.sheet, sd.series) / "warped.footprint.geojson"
    if not f.is_file():
        return None
    from shapely.geometry import shape
    return shapely.union_all([shape(x["geometry"]) for x in json.loads(f.read_text())["features"]])


def content_mask(sd: data.SheetData, footprint, inv: _ToPage) -> np.ndarray:
    """The page px of the sheet's printed map (bool (height, width)): its footprint (`load_footprint`) drawn on the
    page through the inverse placement; every px when there is none."""
    if footprint is None:
        return np.ones((sd.height, sd.width), bool)
    out = np.zeros((sd.height, sd.width), np.uint8)
    clipped = shapely.intersection(footprint, _footprint(sd))
    for part in shapely.get_parts(clipped):
        if shapely.get_type_id(part) != 3:              # polygons only (an overlay's stray lines or points: no area)
            continue
        for value, rings in ((1, [part.exterior]), (0, list(part.interiors))):
            pts = [np.rint(inv(np.asarray(r.coords)) * 2 ** SHIFT).astype(np.int32).reshape(-1, 1, 2) for r in rings]
            pts = [q for q in pts if len(q) >= 3]
            if pts:
                cv2.fillPoly(out, pts, value, cv2.LINE_8, SHIFT)
    return out > 0


def _at(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """The mask at page points (x, y; float px, rounded, kept on the page)."""
    h, w = mask.shape
    c = np.clip(np.rint(xy[:, 0]).astype(np.int64), 0, w - 1)
    r = np.clip(np.rint(xy[:, 1]).astype(np.int64), 0, h - 1)
    return mask[r, c]


# --- the words -----------------------------------------------------------------------------------------------------

def terra_texts(sheets, records=None) -> dict[tuple[str, str], list[str]]:
    """Each sheet's Terra words' texts (`norm`), in the order of its `words` (`boxes.terra_boxes`, as `data.sheet_data`
    reads them).  ValueError when a sheet's count differs."""
    records = boxes.terra_records() if records is None else records
    by_sheet = collections.defaultdict(list)
    for r in records:
        by_sheet[r["series"], r["sheet"]].append(r)
    out = {}
    for sd in sheets:
        found = boxes.terra_boxes(sd.series, sd.sheet, by_sheet.get((sd.series, sd.sheet), []))
        if len(found) != len(sd.words):
            raise ValueError(f"{sd.series} {sd.sheet}: {len(found)} Terra words, the sheet holds {len(sd.words)}")
        out[sd.series, sd.sheet] = [norm(b["text"]) for b in found]
    return out


def pasteable(sd: data.SheetData, ink: np.ndarray | None = None) -> list[tuple[int, int, int, np.ndarray]]:
    """The sheet's pasteable words (see the module): (word index, x0, y0, own ink (bool, the rectangle's bounding box
    whose top-left px is (x0, y0))), in word order.  The sheet's held areas' words are not."""
    ink = data.page_ink(sd) if ink is None else ink
    if not len(sd.words):
        return []
    _, label, stats, _ = cv2.connectedComponentsWithStats(ink.view(np.uint8), connectivity=8)
    size = stats[:, cv2.CC_STAT_AREA]
    held = data._touch_held(sd, np.arange(len(sd.words)))
    out = []
    for j, rect in enumerate(sd.words):
        lo, hi = rect.min(axis=0), rect.max(axis=0) + 1
        if held[j] or (lo < 0).any() or hi[0] > sd.width or hi[1] > sd.height:
            continue
        box = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
        cv2.fillPoly(box, [(rect - lo).reshape(-1, 1, 2)], 1)
        window = (slice(lo[1], hi[1]), slice(lo[0], hi[0]))
        own = (box > 0) & ink[window]
        if not own.any():
            continue
        pieces, got = np.unique(label[window][own], return_counts=True)
        if (size[pieces] == got).all():
            out.append((j, int(lo[0]), int(lo[1]), own))
    return out


@dataclasses.dataclass(frozen=True)
class Words:
    """A map's pasteable words: the bank's sheet index and the word's index there, its rectangle (page px), kind,
    reading angle and text, and its own ink (packed 8 px a byte along each row) with the top-left px and size of
    its box."""
    sheet: np.ndarray       # int32 (n,)
    index: np.ndarray       # int32 (n,)
    rect: np.ndarray        # int32 (n, 4, 2)
    kind: np.ndarray        # uint8 (n,)
    angle: np.ndarray       # int16 (n,)
    text: np.ndarray        # str (n,)
    origin: np.ndarray      # int32 (n, 2): (x0, y0)
    shape: np.ndarray       # int32 (n, 2): (h, w)
    bits: tuple             # (n,) uint8 arrays

    def __len__(self) -> int:
        return len(self.sheet)

    def ink(self, i: int) -> np.ndarray:
        h, w = (int(v) for v in self.shape[i])
        return np.unpackbits(self.bits[i], axis=1, count=w).view(bool).reshape(h, w)

    @cached_property
    def by_text(self) -> dict[str, np.ndarray]:
        out = collections.defaultdict(list)
        for i, t in enumerate(self.text):
            out[str(t)].append(i)
        return {t: np.array(v, np.int64) for t, v in out.items()}

    @cached_property
    def of_kind(self) -> dict[int, np.ndarray]:
        return {int(k): np.flatnonzero(self.kind == k) for k in np.unique(self.kind)}

    @cached_property
    def axis(self) -> np.ndarray:
        """Each word's reading direction on its page, a unit (x, y) vector (y down)."""
        a = np.deg2rad(self.angle.astype(float))
        return np.c_[np.cos(a), -np.sin(a)]

    @cached_property
    def reach(self) -> float:
        """The farthest a word's rectangle reaches from its centre, px."""
        if not len(self):
            return 0.0
        c = self.rect.mean(axis=1, keepdims=True)
        return float(np.hypot(*(self.rect - c).transpose(2, 0, 1)).max()) + 1.0

    @staticmethod
    def join(parts: list[dict]) -> "Words":
        def cat(key, dtype, shape):
            got = [np.asarray(p[key], dtype).reshape(shape) for p in parts]
            return np.concatenate(got) if got else np.zeros(shape, dtype)
        return Words(sheet=cat("sheet", np.int32, (-1,)), index=cat("index", np.int32, (-1,)),
                     rect=cat("rect", np.int32, (-1, 4, 2)), kind=cat("kind", np.uint8, (-1,)),
                     angle=cat("angle", np.int16, (-1,)), text=cat("text", str, (-1,)).astype(str),
                     origin=cat("origin", np.int32, (-1, 2)), shape=cat("shape", np.int32, (-1, 2)),
                     bits=tuple(b for p in parts for b in p["bits"]))


# --- the measures --------------------------------------------------------------------------------------------------

def _thickness_key(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The skeleton px of ink and, at each, the squared distance to paper (whole px; 2 x its root - 1 is the stroke's
    thickness there)."""
    skel = skeletonize(ink)
    dt = cv2.distanceTransform(ink.view(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return skel, np.rint(dt.astype(np.float64) ** 2).astype(np.int64)


@lru_cache(maxsize=None)
def thickness_profile(t: int) -> tuple[np.ndarray, np.ndarray]:
    """What a `cv2.line` of thickness t looks like to the thickness measure: the squared distances to paper at its
    skeleton px (`_thickness_key`) and their shares, pooled over CALIBRATION_ANGLES."""
    pad = t + 4
    n = CALIBRATION_LENGTH + 2 * pad
    keys = []
    for a in np.deg2rad(CALIBRATION_ANGLES):
        canvas = np.zeros((n, n), np.uint8)
        c = np.array([n / 2, n / 2])
        d = np.array([math.cos(a), -math.sin(a)]) * CALIBRATION_LENGTH / 2
        p, q = (np.rint((c - d) * 2 ** SHIFT)).astype(int), (np.rint((c + d) * 2 ** SHIFT)).astype(int)
        cv2.line(canvas, tuple(int(v) for v in p), tuple(int(v) for v in q), 1, int(t), cv2.LINE_8, SHIFT)
        skel, key = _thickness_key(canvas > 0)
        keys.append(key[skel])
    k, n_k = np.unique(np.concatenate(keys), return_counts=True)
    return k, n_k / n_k.sum()


@lru_cache(maxsize=None)
def _same_drawing(a: int, b: int) -> bool:
    """Whether `cv2.line` draws thicknesses a and b alike at every calibration angle."""
    pad = max(a, b) + 4
    n = CALIBRATION_LENGTH + 2 * pad
    for ang in np.deg2rad(CALIBRATION_ANGLES):
        c = np.array([n / 2, n / 2])
        d = np.array([math.cos(ang), -math.sin(ang)]) * CALIBRATION_LENGTH / 2
        p, q = (tuple(int(v) for v in np.rint(e * 2 ** SHIFT)) for e in (c - d, c + d))
        one, two = np.zeros((n, n), np.uint8), np.zeros((n, n), np.uint8)
        cv2.line(one, p, q, 1, int(a), cv2.LINE_8, SHIFT)
        cv2.line(two, p, q, 1, int(b), cv2.LINE_8, SHIFT)
        if not np.array_equal(one, two):
            return False
    return True


def thickness_shares(keys: np.ndarray, weights: np.ndarray) -> dict:
    """The shares q_t of `cv2.line` thicknesses t = 1, 2, ... whose mixture of measured profiles (`thickness_profile`)
    best explains the weighted measured keys (EM for the shares, the profiles fixed; thicknesses up to the first whose
    profile reaches the largest key carrying weight).  A thickness cv2 draws exactly as the one before (3 as 2 + 1:
    4 as 3, 6 as 5, ...) is the same drawing and is not a candidate of its own.  Keys no thickness can make are left
    out and their weight reported."""
    keys, weights = np.asarray(keys, np.int64), np.asarray(weights, float)
    on = weights > 0
    keys, weights = keys[on], weights[on]
    if not len(keys):
        return {"thickness": [], "share": [], "unexplained": 0.0, "iterations": 0}
    top, profiles, kept = int(keys.max()), [], []
    t = 0
    while True:
        t += 1
        k, sh = thickness_profile(t)
        if profiles and _same_drawing(t - 1, t):
            continue
        profiles.append((k, sh))
        kept.append(t)
        if k.max() >= top:
            break
    h = np.zeros((len(profiles), len(keys)))
    for i, (k, s) in enumerate(profiles):
        pos = np.searchsorted(k, keys)
        hit = (pos < len(k)) & (k[np.minimum(pos, len(k) - 1)] == keys)
        h[i, hit] = s[pos[hit]]
    made = h.sum(0) > 0
    w = weights[made] / weights[made].sum() if made.any() else weights[made]
    h = h[:, made]
    q = np.full(len(profiles), 1.0 / len(profiles))
    last, it = -np.inf, 0
    for it in range(1, EM_MAX_ITER + 1):
        mix = q @ h
        ll = float((w * np.log(mix)).sum())
        q = q * (h @ (w / mix))
        if abs(ll - last) <= EM_TOL * max(1.0, abs(ll)):
            break
        last = ll
    return {"thickness": kept, "share": [float(v) for v in q],
            "unexplained": float(weights[~made].sum() / weights.sum()), "iterations": it}


def fit_near(kinds: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    """The line-ink mixture (see the module), every kind of line at once: per kind, distances `d` (px, each distinct
    value once) with its skeleton px `n_s` and the page's training px `n_b` at each.  Fitted: per kind the share a_k
    of its skeleton px that are its lines' own, and one spread sigma (px) of the printed lines about the projected
    ones for every kind (the placement is one: it does not know which kind of line it moves), with each distance's
    responsibility r per kind; EM from every decile split of the skeleton's distances (the same quantile in each
    kind) and from every px its kind's own, the likeliest kept (a share that reaches 0 or 1 stays there: a fixed
    point)."""
    kinds = [tuple(np.asarray(v, float) for v in k) for k in kinds]
    live = [i for i, (_, n_s, _) in enumerate(kinds) if n_s.sum() > 0]
    out = {"a": [0.0] * len(kinds), "sigma": math.nan, "r": [np.zeros(len(k[0])) for k in kinds], "loglik": 0.0,
           "iterations": 0, "start": None}
    if not live:
        return out
    every_d = np.concatenate([kinds[i][0] for i in live])
    smallest = float(every_d[every_d > 0].min()) if (every_d > 0).any() else 1.0  # a start only: the least distance

    def log_z(i, lg):
        d, _, n_b = kinds[i]
        on = n_b > 0
        v = np.log(n_b[on]) + lg[on]
        top = float(v.max())
        return top + math.log(float(np.exp(v - top).sum()))

    def mstep(rs, start):
        def nll(u):
            total = 0.0
            for i in live:
                d, n_s, _ = kinds[i]
                w = n_s * rs[i]
                lg = -d * d / (2 * math.exp(2 * u))
                total += -float((w * lg).sum()) + float(w.sum()) * log_z(i, lg)
            return total
        u0 = math.log(start)
        return math.exp(optimize.minimize_scalar(nll, bracket=(u0 - 0.5, u0)).x)

    def estep(a, sigma):
        rs, ll = [np.zeros(len(k[0])) for k in kinds], 0.0
        for i in live:
            d, n_s, n_b = kinds[i]
            lg = -d * d / (2 * sigma * sigma)
            near = a[i] * np.exp(lg - log_z(i, lg)) if a[i] > 0 else np.zeros(len(d))
            tot = near + (1 - a[i]) / n_b.sum()
            rs[i] = np.divide(near, tot, out=np.zeros(len(d)), where=tot > 0)
            seen = n_s > 0
            with np.errstate(divide="ignore"):
                ll += float((n_s[seen] * np.log(n_b[seen] * tot[seen])).sum())
        return rs, ll

    def spread(rs):
        w = sum(float((kinds[i][1] * rs[i]).sum()) for i in live)
        v = sum(float((kinds[i][1] * rs[i] * kinds[i][0] ** 2).sum()) for i in live) / w if w > 0 else 0.0
        return math.sqrt(v) if v > 0 else smallest

    starts = [("own", [np.ones(len(k[0])) for k in kinds])]
    for q in DECILES:
        rs = []
        for d, n_s, _ in kinds:
            order = np.argsort(d)
            cum = np.cumsum(n_s[order]) / max(n_s.sum(), 1.0)
            cut = d[order][min(int(np.searchsorted(cum, q)), len(d) - 1)] if len(d) else 0.0
            rs.append((d <= cut).astype(float))
        starts.append((float(q), rs))
    best = None
    for name, rs in starts:
        a = [float((kinds[i][1] * rs[i]).sum() / kinds[i][1].sum()) if i in live else 0.0
             for i in range(len(kinds))]
        if not any(a[i] > 0 for i in live):
            continue
        sigma, last, ll, it = spread(rs), -np.inf, -np.inf, 0
        for it in range(1, EM_MAX_ITER + 1):
            sigma = mstep(rs, sigma)
            rs, ll = estep(a, sigma)
            a = [float((kinds[i][1] * rs[i]).sum() / kinds[i][1].sum()) if i in live else 0.0
                 for i in range(len(kinds))]
            if ll - last <= EM_TOL * max(1.0, abs(ll)):
                break
            last = ll
        if it == EM_MAX_ITER:
            log.warning("fit_near: %d iterations from the %s start", EM_MAX_ITER, name)
        if math.isfinite(ll) and (best is None or ll > best["loglik"]):
            best = {"a": a, "sigma": sigma, "r": rs, "loglik": ll, "iterations": it, "start": name}
    return best or out


def fit_far(d: np.ndarray) -> dict:
    """The street words' offsets: |d| (ft) a mixture of a half-normal (the street's own label, share a, spread sigma)
    and an exponential (a misread naming a street elsewhere, mean `far_mean`), by EM from every decile split, the
    likeliest kept; with each word's near responsibility r."""
    d = np.abs(np.asarray(d, float))
    if len(d) < 2 or not (d > 0).any():
        return {"a": 1.0, "sigma": math.nan, "far_mean": math.nan, "r": np.ones(len(d)), "loglik": 0.0}

    def logs(a, sigma, mean):
        near = math.log(a) + math.log(2.0) - math.log(sigma * math.sqrt(2 * math.pi)) - d * d / (2 * sigma * sigma)
        far = math.log(1 - a) - math.log(mean) - d / mean
        top = np.maximum(near, far)
        tot = top + np.log(np.exp(near - top) + np.exp(far - top))
        return np.exp(near - tot), float(tot.sum())

    best = None
    for q in DECILES:
        r = (d <= np.quantile(d, q)).astype(float)
        last, ll, it, a, sigma, mean = -np.inf, -np.inf, 0, 0.0, 0.0, 0.0
        for it in range(1, EM_MAX_ITER + 1):
            a = float(r.mean())
            if not 0 < a < 1:
                break
            sigma = math.sqrt(float((r * d * d).sum() / r.sum()))
            mean = float(((1 - r) * d).sum() / (1 - r).sum())
            if not sigma > 0 or not mean > 0:
                break
            r, ll = logs(a, sigma, mean)
            if ll - last <= EM_TOL * max(1.0, abs(ll)):
                break
            last = ll
        if it == EM_MAX_ITER:
            log.warning("fit_far: %d iterations from the %.1f split", EM_MAX_ITER, q)
        if 0 < a < 1 and sigma > 0 and mean > 0 and math.isfinite(ll) and (best is None or ll > best["loglik"]):
            best = {"a": a, "sigma": sigma, "far_mean": mean, "r": r, "loglik": ll, "iterations": it,
                    "start": float(q)}
    if best is None:
        return {"a": 1.0, "sigma": math.nan, "far_mean": math.nan, "r": np.ones(len(d)), "loglik": 0.0}
    return best


def inked_fractions(own: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Each edge's inked fraction: its own ink px (soft counts) over its px, at most 1 (a line covers no more than
    itself); edges with no px measured are left out."""
    own, length = np.asarray(own, float), np.asarray(length, float)
    on = length > 0
    return np.minimum(own[on] / length[on], 1.0)


def edge_runs(edges: np.ndarray, fraction: np.ndarray, start: np.ndarray) -> np.ndarray:
    """The inked run of each edge ((n, 2, 2) ends): the `fraction` of it beginning at `start` x (1 - fraction) of the
    way along (start uniform in [0, 1): a run placed uniformly along the edge)."""
    a = (start * (1 - fraction))[:, None]
    d = edges[:, 1] - edges[:, 0]
    return np.stack([edges[:, 0] + a * d, edges[:, 0] + (a + fraction[:, None]) * d], 1)


def _line_frame(lines, at) -> tuple[np.ndarray, np.ndarray]:
    """Each centreline's point and unit direction (its straight piece there) at distance `at` along it."""
    pts, dirs = np.zeros((len(lines), 2)), np.zeros((len(lines), 2))
    for i, (line, s) in enumerate(zip(lines, np.asarray(at, float), strict=True)):
        xy = shapely.get_coordinates(line)
        step = np.diff(xy, axis=0)
        ln = np.hypot(step[:, 0], step[:, 1])
        ends = np.cumsum(ln)
        k = min(int(np.searchsorted(ends, s, side="left")), len(step) - 1)
        u = step[k] / ln[k] if ln[k] > 0 else np.array([1.0, 0.0])
        pts[i] = xy[k] + u * (s - (ends[k] - ln[k]))
        dirs[i] = u
    return pts, dirs


def _axis_gap(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The angle between two axes given as vectors (rows), 0 to 90 degrees."""
    a = np.rad2deg(np.arctan2(u[:, 1], u[:, 0])) - np.rad2deg(np.arctan2(v[:, 1], v[:, 0]))
    a = np.mod(a, 180.0)
    return np.minimum(a, 180.0 - a)


def _word_map(sd: data.SheetData, rects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Word rectangles' centres on the map (ft) and their reading axes there (vectors: the mean of the long edges)."""
    w = page_to_world(sd, rects.reshape(-1, 2).astype(float)).reshape(-1, 4, 2)
    return w.mean(axis=1), (w[:, 1] - w[:, 0] + w[:, 2] - w[:, 3]) / 2


def _counts(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.unique(keys, return_counts=True)


def _sheet_edges(sd: data.SheetData, geo: Geometry, inv: _ToPage) -> tuple[np.ndarray, np.ndarray]:
    """Every edge of every kind reaching the sheet's placement, on its page (`page_edges`): (n, 2, 2) px and each
    edge's kind."""
    hull = _footprint(sd)
    geoms, kinds = [], []
    for k in range(len(LINE_KINDS)):
        idx = np.sort(geo.item_tree(k).query(hull))
        geoms.append(geo.items(k)[idx])
        kinds.append(np.full(len(idx), k, np.uint8))
    edges, owner = page_edges(np.concatenate(geoms), inv)
    return edges, np.concatenate(kinds)[owner]


def _sheet_lines(sd: data.SheetData, geo: Geometry, inv, train: np.ndarray) -> dict:
    """One sheet's part of the line measures, on its training px outside every Terra word's and other pass's box (the
    box itself; `train` less the boxes): the edges (`_sheet_edges`) drawn 1 px wide, each px its edge's (a block
    outline's over a lot line's over a centreline's where they meet); each such px's nearest edge and distance; per
    kind, the distinct squared distances (px) of those px (background) and, over the line ink's skeleton, the counts
    per (edge, squared distance, thickness key); and each edge's kind and px there."""
    boxes_ = np.concatenate([sd.words, sd.other]).reshape(-1, 4, 2)
    train = train & ~(data._paint(boxes_, np.ones(len(boxes_), np.int32), 0, 0, sd.height, sd.width) > 0)
    line = data.page_ink(sd) & train            # every word's box (Terra's, the other passes'), the box itself, out
    skel, tkey = _thickness_key(line)
    del line
    edges, kind = _sheet_edges(sd, geo, inv)
    drawn = np.zeros((sd.height, sd.width), np.int32)
    for e in np.argsort(-kind.astype(np.int64), kind="stable"):        # centrelines, then lot lines, outlines last
        p, q = np.rint(edges[e] * 2 ** SHIFT).astype(np.int64)
        cv2.line(drawn, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), int(e) + 1, 1, cv2.LINE_8, SHIFT)
    out = {"edge_kind": kind, "edge_len": np.bincount(drawn[train & (drawn > 0)] - 1, minlength=len(kind)),
           "background": [], "skeleton": (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.int64),
                                          np.zeros(0, np.int64))}
    if not len(kind) or not (drawn > 0).any():
        out["background"] = [(np.zeros(0, np.int64), np.zeros(0, np.int64)) for _ in LINE_KINDS]
        return out
    paper = (drawn == 0).view(np.uint8)
    dist = cv2.distanceTransform(paper, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, label = cv2.distanceTransformWithLabels(paper, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    flat = drawn.ravel()
    nearest = (flat[np.flatnonzero(flat)] - 1)[label.ravel() - 1].reshape(drawn.shape)   # each px's nearest edge
    del label, paper
    d2 = np.rint(dist.astype(np.float64) ** 2).astype(np.int64)
    del dist
    near_kind = kind[nearest]
    for k in range(len(LINE_KINDS)):
        out["background"].append(_counts(d2[train & (near_kind == k)]))
    at = skel & train
    u, n = np.unique(np.stack([nearest[at].astype(np.int64), d2[at], tkey[at]], 1), axis=0, return_counts=True)
    u = u.reshape(-1, 3)
    out["skeleton"] = (u[:, 0].astype(np.int32), u[:, 1], u[:, 2], n.astype(np.int64))
    return out


def _sheet_labels(sd: data.SheetData, geo: Geometry, texts: list[str], inv, block_ok, line_ok,
                  content: np.ndarray) -> dict:
    """One sheet's part of the label measures (see the module), on the printed map's px (`content`)."""
    out = {"code": np.zeros((0, 2)), "block": np.zeros((0, 2)), "street": np.zeros((0, 2)), "lots": 0, "blocks": 0,
           "street_ft": 0.0, "words": 0, "pasteable_words": 0}
    if len(sd.words):
        ok = ~data._touch_held(sd, np.arange(len(sd.words))) & _at(content, sd.words.mean(axis=1))
        centre, axis = _word_map(sd, sd.words)
        kinds = sd.kinds
        code = np.flatnonzero(ok & (kinds == CODE))
        if len(code):
            at, lot = geo.lot_tree.query_nearest(shapely.points(centre[code]), all_matches=False)
            keep = block_ok[geo.lot_block[lot]]
            out["code"] = centre[code[at[keep]]] - geo.lot_centre[lot[keep]]
        found = []
        for j in np.flatnonzero(ok & (kinds == BLOCK_NUMBER)):
            for b in geo.block_of_form.get(texts[j], ()):
                if block_ok[b] and shapely.contains_xy(geo.block_poly[b], *centre[j]):
                    found.append(centre[j] - geo.block_centre[b])
        out["block"] = np.array(found, float).reshape(-1, 2)
        rows = []
        for j in np.flatnonzero(ok & (kinds == STREET_NAME)):
            cand = geo.form_lines.get(texts[j])
            if cand is None:
                continue
            pt = shapely.points(centre[j])
            c = int(cand[int(geo.form_tree(texts[j]).query_nearest(pt, all_matches=False)[0])])
            if not line_ok[c]:
                continue
            s = shapely.line_locate_point(geo.lines[c], pt)
            foot, u = _line_frame(geo.lines[c:c + 1], [s])
            rel = centre[j] - foot[0]
            rows.append((float(u[0, 0] * rel[1] - u[0, 1] * rel[0]), float(_axis_gap(axis[j:j + 1], u)[0])))
        out["street"] = np.array(rows, float).reshape(-1, 2)
        out["words"] = int(ok.sum())
    on_page = _on_page(sd, inv, content)
    out["lots"] = int(on_page(geo.lot_centre, block_ok[geo.lot_block]).sum())
    out["blocks"] = int(on_page(geo.block_centre, block_ok).sum())
    named = line_ok & np.array([bool(f) for f in geo.line_forms], bool)
    mids = shapely.get_coordinates(shapely.line_interpolate_point(geo.lines, 0.5, normalized=True)).reshape(-1, 2)
    out["street_ft"] = float(geo.line_length[on_page(mids, named)].sum())
    return out


def _on_page(sd: data.SheetData, inv, content: np.ndarray):
    """A test of map points: on the sheet's printed map (`content`) at a training px (and `ok`)."""
    def test(xy: np.ndarray, ok: np.ndarray) -> np.ndarray:
        out = np.zeros(len(xy), bool)
        idx = np.flatnonzero(ok)
        if not len(idx):
            return out
        p = inv(xy[idx])
        on = np.isfinite(p).all(1)
        on[on] = (p[on, 0] >= 0) & (p[on, 0] < sd.width) & (p[on, 1] >= 0) & (p[on, 1] < sd.height)
        on[on] = _at(content, p[on]) & ~_held_points(sd, xy[idx[on]])
        out[idx[on]] = True
        return out
    return test


def _merge(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if not parts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    k = np.concatenate([p[0] for p in parts])
    n = np.concatenate([p[1] for p in parts])
    u, inv = np.unique(k, return_inverse=True)
    return u, np.bincount(inv, weights=n).astype(np.int64)


def _quantiles(x, w=None) -> dict | None:
    x = np.asarray(x, float)
    if not len(x):
        return None
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    if not w.sum() > 0:
        return None
    order = np.argsort(x)
    cum = np.cumsum(w[order]) / w.sum()
    return {f"q{int(round(100 * q))}": round(float(x[order][min(int(np.searchsorted(cum, q)), len(x) - 1)]), 3)
            for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}


@dataclasses.dataclass
class Measured:
    """What one map's scenes are drawn from (see the module)."""
    inked: list                  # per LINE_KINDS kind: the measured inked fraction of each edge (one value an edge)
    thickness: list              # per kind: (thicknesses (int), shares)
    code_rate: float             # code words per lot
    code_offsets: np.ndarray     # (n, 2) ft
    block_rate: float            # block-number words per block
    block_offsets: np.ndarray    # (n, 2) ft
    street_rate: float           # near street-name words per ft of centreline
    street_offsets: np.ndarray   # (n,) signed ft across the centreline
    street_weights: np.ndarray   # (n,) each one's near responsibility
    street_angles: np.ndarray    # (n,) degrees, 0-90
    angle_bandwidth: float
    summary: dict

    @cached_property
    def angle_table(self) -> np.ndarray:
        """The street angle density on ANGLE_GRID (zeros without a bandwidth)."""
        return context.Density(self.street_angles, self.street_weights, self.angle_bandwidth, 0.0, 90.0)(ANGLE_GRID)

    @cached_property
    def street_cum(self) -> np.ndarray:
        return np.cumsum(self.street_weights)

    @cached_property
    def reach(self) -> float:
        """The farthest a label's point lies from its block, lot or centreline (ft)."""
        parts = [np.hypot(*self.code_offsets.T), np.hypot(*self.block_offsets.T),
                 np.abs(self.street_offsets[self.street_weights > 0])]
        return float(max((p.max() for p in parts if len(p)), default=0.0))


def measure_map(lines: list[dict], labels: list[dict]) -> Measured:
    """One map's measures from its sheets' parts (`_sheet_lines`, `_sheet_labels`)."""
    inked, thick, line_summary = [], [], {}
    data_k = []
    for k in range(len(LINE_KINDS)):
        bkey, bn = _merge([p["background"][k] for p in lines])
        n_s = np.zeros(len(bkey))
        for p in lines:
            e, d2, _, n = p["skeleton"]
            mine = p["edge_kind"][e] == k
            np.add.at(n_s, np.searchsorted(bkey, d2[mine]), n[mine])
        data_k.append((bkey, bn, n_s))
    fit = fit_near([(np.sqrt(b.astype(float)), n_s, bn) for b, bn, n_s in data_k])
    own, length, ekind, tkeys, tweights = [], [], [], [], [[] for _ in LINE_KINDS]
    for p in lines:
        e, d2, tk, n = p["skeleton"]
        kind = p["edge_kind"]
        r = np.zeros(len(e))
        for k in range(len(LINE_KINDS)):
            mine = kind[e] == k
            if mine.any():
                r[mine] = fit["r"][k][np.searchsorted(data_k[k][0], d2[mine])]
        own.append(np.bincount(e, weights=r * n, minlength=len(kind)))
        length.append(p["edge_len"])
        ekind.append(kind)
        tkeys.append((tk, kind[e], r * n))
    own, length, ekind = (np.concatenate(v) if v else np.zeros(0) for v in (own, length, ekind))
    for k, name in enumerate(LINE_KINDS):
        on = (ekind == k) & (length > 0)
        f = inked_fractions(own[on], length[on])
        inked.append(f)
        tk = np.concatenate([t[0][t[1] == k] for t in tkeys]) if tkeys else np.zeros(0, np.int64)
        tw = np.concatenate([t[2][t[1] == k] for t in tkeys]) if tkeys else np.zeros(0)
        keys, inv_ = np.unique(tk, return_inverse=True)
        tw = np.bincount(inv_, weights=tw, minlength=len(keys)) if len(keys) else np.zeros(0)
        shares = thickness_shares(keys, tw)
        thick.append((np.array(shares["thickness"], np.int64), np.array(shares["share"], float)))
        line_summary[name] = {
            "skeleton_px": int(data_k[k][2].sum()), "own_share_a": round(fit["a"][k], 4),
            "sigma_px (every kind's)": _round(fit["sigma"]), "own_px": round(float(own[ekind == k].sum()), 1),
            "length_px": int(length[ekind == k].sum()),
            "inked_share_by_length": round(float(np.minimum(own[on], length[on]).sum() / length[on].sum())
                                           if on.any() else 0.0, 4),
            "edges": int(len(f)), "inked_fraction_mean_by_edge": round(float(f.mean()), 4) if len(f) else None,
            "inked_fraction_by_edge": _quantiles(f), "edges_at_0": round(float((f == 0).mean()), 4) if len(f) else None,
            "edges_at_1": round(float((f == 1).mean()), 4) if len(f) else None,
            "inked_fraction_histogram": [int(v) for v in np.histogram(f, bins=10, range=(0.0, 1.0))[0]],
            "em_iterations": fit["iterations"],
            "thickness_measured_px": _quantiles(2 * np.sqrt(keys) - 1, tw),
            "cv2_thickness_shares": {int(t): round(s, 4) for t, s in zip(*thick[-1], strict=True) if round(s, 4)},
            "thickness_unexplained": round(shares["unexplained"], 5)}
    code = np.concatenate([p["code"] for p in labels]).reshape(-1, 2)
    block = np.concatenate([p["block"] for p in labels]).reshape(-1, 2)
    street = np.concatenate([p["street"] for p in labels]).reshape(-1, 2)
    lots, blocks = sum(p["lots"] for p in labels), sum(p["blocks"] for p in labels)
    street_ft = sum(p["street_ft"] for p in labels)
    far = fit_far(street[:, 0])
    r = np.asarray(far["r"], float)
    h = context.loo_bandwidth(street[:, 1], r, 0.0, 90.0) if len(street) else math.nan
    m = Measured(
        inked=inked, thickness=thick,
        code_rate=len(code) / lots if lots else 0.0, code_offsets=code,
        block_rate=len(block) / blocks if blocks else 0.0, block_offsets=block,
        street_rate=float(r.sum()) / street_ft if street_ft else 0.0, street_offsets=street[:, 0],
        street_weights=r, street_angles=street[:, 1], angle_bandwidth=h, summary={})
    m.summary = {
        "lines": line_summary,
        "codes": {"words": len(code), "lots": lots, "per_lot": round(m.code_rate, 5),
                  "offset_ft": _quantiles(np.hypot(*code.T)), "dx_ft": _quantiles(code[:, 0]),
                  "dy_ft": _quantiles(code[:, 1])},
        "block_numbers": {"words": len(block), "blocks": blocks, "per_block": round(m.block_rate, 4),
                          "offset_ft": _quantiles(np.hypot(*block.T))},
        "street_names": {"words": len(street), "street_ft": round(street_ft, 1),
                         "near_share": round(far["a"], 4), "near_sigma_ft": _round(far["sigma"]),
                         "far_mean_ft": _round(far["far_mean"]), "near_per_1000_ft": round(1000 * m.street_rate, 4),
                         "offset_ft": _quantiles(street[:, 0], r), "angle_deg": _quantiles(street[:, 1], r),
                         "angle_bandwidth_deg": _round(h)},
        "words": {"usable": sum(p["words"] for p in labels), "pasteable": sum(p["pasteable_words"] for p in labels)},
    }
    return m


def _round(v, digits=3):
    return round(float(v), digits) if v is not None and math.isfinite(float(v)) else None


# --- the bank and its scenes ---------------------------------------------------------------------------------------

def _sheet_part(sd: data.SheetData, j: int, geo: Geometry, texts: list[str], block_ok, line_ok,
                stride: int, footprint) -> dict:
    """One sheet's part of the bank (its held areas the bank's), on its printed map's px (`content_mask` of
    `footprint`): its pasteable words, measures and crop centres."""
    inv = page_from_world(sd)
    content = content_mask(sd, footprint, inv)
    lines = _sheet_lines(sd, geo, inv, ~_held_page(sd) & content)
    labels = _sheet_labels(sd, geo, texts, inv, block_ok, line_ok, content)
    found = [f for f in pasteable(sd) if _at(content, sd.words[f[0]].mean(axis=0)[None])[0]]
    idx = np.array([f[0] for f in found], np.int64)
    words = {"sheet": np.full(len(found), j), "index": idx, "rect": sd.words[idx], "kind": sd.kinds[idx],
             "angle": sd.angles[idx], "text": np.array([texts[i] for i in idx], str),
             "origin": [(f[1], f[2]) for f in found], "shape": [f[3].shape for f in found],
             "bits": [np.packbits(f[3], axis=1) for f in found]}
    labels["pasteable_words"] = len(found)
    centres = tuple(c[content[c[:, 0], c[:, 1]]] for c in data.centres(sd, stride))
    return {"lines": lines, "labels": labels, "words": words, "centres": centres}


@dataclasses.dataclass
class SceneBank:
    """Everything a scene is drawn from (see the module): the sheets (holding the bank's areas out), the geometry with
    what of it is usable, each map's pasteable words and measures, and where scenes centre.  `crop` (or `scene`)
    makes one; `tally` counts what the scenes held; `last` says what the last one was made of."""
    sheets: list
    geometry: Geometry
    block_ok: np.ndarray
    line_ok: np.ndarray
    words: dict                  # series -> Words
    measured: dict               # series -> Measured
    centres: tuple               # (on text, on ink): per sheet (n, 2) (row, col)
    text_share: float
    seed: int = 0
    tally: collections.Counter = dataclasses.field(default_factory=collections.Counter, compare=False)
    last: dict = dataclasses.field(default_factory=dict, compare=False)

    @classmethod
    def build(cls, sheets: list[data.SheetData], areas, seed: int, geometry: Geometry | None = None,
              texts: dict | None = None, threads: int = 1, text_share: float | None = None,
              stride: int | None = None, footprints: dict | None = None) -> "SceneBank":
        """The bank of `sheets` (with their placements) with `areas` held out (see the
        module): `geometry` (default `Geometry.load()`), `texts` (each sheet's words' texts, default `terra_texts`),
        `threads` sheets at a time; crops centre as the real ones do (`text_share`, `stride`: default
        `train.TEXT_SHARE`, `train.CENTRE_STRIDE`); each sheet's printed map is its footprint (`footprints`, by
        (series, sheet), a missing one the whole page; default `load_footprint`).  ValueError for a sheet without
        its placement."""
        if text_share is None or stride is None:
            from sfzmaps.solver.textnet import train
            text_share = train.TEXT_SHARE if text_share is None else text_share
            stride = train.CENTRE_STRIDE if stride is None else stride
        bare = [f"{sd.series} {sd.sheet}" for sd in sheets if not sd.world.size]
        if bare:
            raise ValueError(f"scenes are drawn in a sheet's placement: {bare[0]} was read without it")
        held = data.hold_out(sheets, tuple(areas))
        geometry = Geometry.load() if geometry is None else geometry
        texts = terra_texts(held) if texts is None else texts
        block_ok, line_ok = usable(geometry, areas)

        def one(j):
            sd = held[j]
            footprint = load_footprint(sd) if footprints is None else footprints.get((sd.series, sd.sheet))
            return _sheet_part(sd, j, geometry, texts[sd.series, sd.sheet], block_ok, line_ok, stride, footprint)
        if threads > 1:
            with ThreadPoolExecutor(threads) as ex:
                parts = list(ex.map(one, range(len(held))))
        else:
            parts = [one(j) for j in range(len(held))]
        words, measured = {}, {}
        for series in dict.fromkeys(sd.series for sd in held):
            mine = [p for p, sd in zip(parts, held, strict=True) if sd.series == series]
            words[series] = Words.join([p["words"] for p in mine])
            measured[series] = measure_map([p["lines"] for p in mine], [p["labels"] for p in mine])
        centres = ([p["centres"][0] for p in parts], [p["centres"][1] for p in parts])
        if not sum(map(len, centres[0])) or not sum(map(len, centres[1])):
            raise ValueError("scenes centre as the real crops do: no text or no ink px to centre on outside the areas "
                             "held out")
        return cls(sheets=held, geometry=geometry, block_ok=block_ok, line_ok=line_ok, words=words,
                   measured=measured, centres=centres, text_share=float(text_share), seed=int(seed))

    @cached_property
    def _pools(self):
        from sfzmaps.solver.textnet import train
        return train.Pool(self.centres[0]), train.Pool(self.centres[1])

    def sizes(self) -> dict:
        """Per map: the pasteable words per kind and the measures' summary."""
        names = tuple(data.KINDS)
        return {s: {"pasteable_words": {n: int((w.kind == k).sum()) for k, n in enumerate(names)},
                    "measured": self.measured[s].summary} for s, w in self.words.items()}

    def scene(self, rng: np.random.Generator, size: int, scale: float, turns: int) -> dict[str, np.ndarray]:
        """`crop`, by the name `train.draw` calls."""
        return self.crop(rng, size, scale, turns)

    def crop(self, rng: np.random.Generator, size: int, scale: float, turns: int) -> dict[str, np.ndarray]:
        """A scene: the keys of `data.crop_targets(..., words=True)` (no weight), size x size px, `scale` and
        `turns` as a real crop's, from `rng` alone (see the module)."""
        pools = self._pools
        pool = pools[0] if rng.random() < self.text_share else pools[1]
        j, cy, cx = pool[rng.integers(len(pool))]
        sd = self.sheets[j]
        src = int(round(size / scale))
        y0 = int(np.clip(cy - src // 2, 0, sd.height - src))
        x0 = int(np.clip(cx - src // 2, 0, sd.width - src))
        fwd, back = window_affine(sd, y0, x0, src, src)
        m = self.measured[sd.series]
        words = self.words[sd.series]
        corners = _apply(fwd, [[x0, y0], [x0 + src, y0], [x0, y0 + src], [x0 + src, y0 + src]])
        ft = math.sqrt(abs(np.linalg.det(fwd[:, :2])))
        ink = np.zeros((src, src), bool)
        segs = self._draw_lines(rng, ink, corners, ft, back, x0, y0, m, sd.series)
        props = self._propose(rng, corners, ft, fwd, m, words)
        placed = self._place(rng, ink, props, back, x0, y0, words)
        out = self._targets(sd, ink, placed, words, size, scale, turns, src)
        self.tally["scenes"] += 1
        for p in placed:
            self.tally[f"label:{p['what']}"] += 1
            how = "any code word" if p["what"] == "code" else "stand-in" if p["standin"] else "its own string"
            self.tally[f"label:{p['what']}:{how}"] += 1
        self.last = {"sheet": j, "series": sd.series, "window": (y0, x0, src), "segments": segs, "labels": placed,
                     "proposed": len(props), "own": out.pop("_own")}
        return out

    # the parts of a scene

    def _draw_lines(self, rng, ink, corners, ft, back, x0, y0, m: Measured, series: str) -> dict:
        """Every usable item reaching the window as edges on the page (`page_edges` through the window's affine map),
        each inked over a fraction drawn from its kind's measured per-edge fractions as one run at a uniform position
        along it (`edge_runs`), at a thickness drawn from its kind's shares."""
        top = max((int(t.max()) for t, _ in m.thickness if len(t)), default=1)
        grow = (top / 2 + 2) * ft
        (lx, ly), (hx, hy) = corners.min(0) - grow, corners.max(0) + grow
        box = shapely.box(lx, ly, hx, hy)
        geoms, kinds, items = [], [], []
        for k in range(len(LINE_KINDS)):
            idx = np.sort(self.geometry.item_tree(k).query(box))
            idx = idx[item_ok(k, self.block_ok, self.line_ok)[idx]]
            geoms.append(self.geometry.items(k)[idx])
            kinds.append(np.full(len(idx), k, np.uint8))
            items.append(idx)
        edges, owner = page_edges(np.concatenate(geoms), lambda xy: _apply(back, xy) - [x0, y0])
        kind, item = np.concatenate(kinds)[owner], np.concatenate(items)[owner]
        pick, start, u = rng.random(len(edges)), rng.random(len(edges)), rng.random(len(edges))
        fraction = np.zeros(len(edges))
        for k in range(len(LINE_KINDS)):
            here, f = kind == k, m.inked[k]
            if here.any() and len(f):
                fraction[here] = f[np.minimum((pick[here] * len(f)).astype(np.int64), len(f) - 1)]
        length = np.hypot(*(edges[:, 1] - edges[:, 0]).T)
        for k in range(len(LINE_KINDS)):
            here = kind == k
            self.tally[f"edge_px:{series}:{LINE_KINDS[k]}"] += float(length[here].sum())
            self.tally[f"inked_px:{series}:{LINE_KINDS[k]}"] += float((fraction * length)[here].sum())
        on = fraction > 0
        edges = edge_runs(edges[on], fraction[on], start[on])
        kind, item, u, fraction = kind[on], item[on], u[on], fraction[on]
        thick = np.ones(len(edges), np.int64)
        for k in range(len(LINE_KINDS)):
            t, q = m.thickness[k]
            here = kind == k
            if here.any() and len(t):
                thick[here] = t[np.minimum(np.searchsorted(np.cumsum(q) / q.sum(), u[here], side="right"),
                                           len(t) - 1)]
        canvas = np.zeros(ink.shape, np.uint8)
        for (p, q), t in zip(np.rint(edges * 2 ** SHIFT).astype(np.int64), thick, strict=True):
            cv2.line(canvas, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), 1, int(t), cv2.LINE_8, SHIFT)
        ink |= canvas > 0
        for k in kind:
            self.tally[f"edges inked:{LINE_KINDS[int(k)]}"] += 1
        return {"edges": edges, "kind": kind, "item": item, "thickness": thick, "fraction": fraction}

    def _propose(self, rng, corners, ft, fwd, m: Measured, words: Words) -> list[dict]:
        """The labels the scene would print, before placing: (what, the map point, the word)."""
        geo = self.geometry
        grow = words.reach * ft + m.reach
        (lx, ly), (hx, hy) = corners.min(0) - grow, corners.max(0) + grow
        box = shapely.box(lx, ly, hx, hy)
        out = []
        # block numbers at their blocks' centroids
        b = np.sort(geo.block_tree.query(box))
        b = b[self.block_ok[b]]
        b = b[rng.random(len(b)) < m.block_rate]
        blocks = words.of_kind.get(BLOCK_NUMBER, np.zeros(0, np.int64))
        for i in b:
            off = m.block_offsets[rng.integers(len(m.block_offsets))] if len(m.block_offsets) else np.zeros(2)
            cand = [w for f in context.block_forms(geo.block_number[i]) for w in words.by_text.get(f, ())]
            self._choose(rng, out, "block", i, geo.block_centre[i] + off, np.array(cand, np.int64), standin=blocks)
        # codes in lots
        lots = np.sort(geo.lot_centre_tree.query(box))
        lots = lots[self.block_ok[geo.lot_block[lots]]]
        lots = lots[rng.random(len(lots)) < m.code_rate]
        codes = words.of_kind.get(CODE, np.zeros(0, np.int64))
        for i in lots:
            off = m.code_offsets[rng.integers(len(m.code_offsets))] if len(m.code_offsets) else np.zeros(2)
            self._choose(rng, out, "code", i, geo.lot_centre[i] + off, codes)
        # street names along centrelines
        c = np.sort(geo.line_tree.query(box))
        c = c[self.line_ok[c] & geo.named[c]]
        streets = words.of_kind.get(STREET_NAME, np.zeros(0, np.int64))
        if len(m.street_offsets) and m.street_cum[-1] > 0:
            n = rng.poisson(m.street_rate * geo.line_length[c])
            for i, k in zip(c, n, strict=True):
                forms = geo.line_forms[i]
                for _ in range(int(k)):
                    s = rng.random() * geo.line_length[i]
                    d = m.street_offsets[min(int(np.searchsorted(m.street_cum, rng.random() * m.street_cum[-1],
                                                                 side="right")), len(m.street_cum) - 1)]
                    foot, u = _line_frame(geo.lines[i:i + 1], [s])
                    at = foot[0] + d * np.array([-u[0, 1], u[0, 0]])
                    cand = np.array([w for f in forms for w in words.by_text.get(f, ())], np.int64)

                    def angle_weight(c, u=u):
                        axes = words.axis[c] @ fwd[:, :2].T           # each word's axis on the map, in this frame
                        return np.interp(_axis_gap(axes, np.repeat(u, len(c), 0)), ANGLE_GRID, m.angle_table)
                    self._choose(rng, out, "street", i, at, cand, angle_weight, standin=streets)
        return out

    def _choose(self, rng, out: list, what: str, item: int, at, cand: np.ndarray, weigh=None,
                standin: np.ndarray | None = None) -> None:
        """A word for the label of `item` (a block, lot or centreline): one of its string (`cand`), drawn alike or by
        `weigh(candidates)`; when none has weight, a same-kind stand-in (`standin`: the map's pasteable words of the
        label's kind, by the same weights), recorded as such (the detector's targets never read a label's string);
        none at all: the label is skipped."""
        used = False
        for pool, is_standin in ((cand, False), (standin, True)):
            if pool is None or not len(pool):
                continue
            weight = None if weigh is None else weigh(pool)
            if weight is not None:
                keep = weight > 0
                pool, weight = pool[keep], weight[keep]
            if not len(pool):
                continue
            if weight is None:
                w = int(pool[rng.integers(len(pool))])
            else:
                cum = np.cumsum(weight)
                w = int(pool[min(int(np.searchsorted(cum, rng.random() * cum[-1], side="right")), len(pool) - 1)])
            out.append({"what": what, "item": int(item), "at": np.asarray(at, float), "word": w,
                        "standin": is_standin})
            how = "any code word" if what == "code" else "stand-in" if is_standin else "its own string"
            self.tally[f"proposed:{what}:{how}"] += 1
            used = True
            break
        if not used:
            self.tally[f"skipped:{what}:no word"] += 1

    def _place(self, rng, ink, props, back, x0, y0, words: Words) -> list[dict]:
        """The proposals placed in a random order, each at its point unless its rectangle meets a placed one."""
        src = ink.shape[0]
        pad = int(math.ceil(2 * words.reach)) + 2
        taken = np.zeros((src + 2 * pad, src + 2 * pad), np.uint8)
        placed = []
        for i in rng.permutation(len(props)):
            p = props[i]
            w = p["word"]
            rect = words.rect[w]
            pt = _apply(back, p["at"][None])[0] - [x0, y0]
            shift = np.rint(pt - rect.mean(axis=0)).astype(np.int64)
            moved = (rect + shift).astype(np.int32)
            lo, hi = moved.min(0), moved.max(0)
            if (hi < 0).any() or (lo >= src).any():
                self.tally[f"skipped:{p['what']}:off the scene"] += 1
                continue
            mine = np.zeros_like(taken)
            cv2.fillPoly(mine, [(moved + pad).reshape(-1, 1, 2)], 1)
            if (mine & taken).any():
                self.tally[f"skipped:{p['what']}:meets a placed word"] += 1
                continue
            taken |= mine
            ox, oy = words.origin[w] + shift
            own = words.ink(w)
            placed.append({**p, "rect": moved, "origin": (int(ox), int(oy)), "own": own})
            h, wd = own.shape
            ya, yb, xa, xb = max(oy, 0), min(oy + h, src), max(ox, 0), min(ox + wd, src)
            if ya < yb and xa < xb:
                ink[ya:yb, xa:xb] |= own[ya - oy:yb - oy, xa - ox:xb - ox]
        return placed

    def _targets(self, sd, ink, placed, words: Words, size, scale, turns, src) -> dict:
        """The scene's targets (see the module), with its own ink at page scale under `_own`."""
        own = np.zeros((src, src), bool)
        for p in placed:
            ox, oy = p["origin"]
            h, w = p["own"].shape
            ya, yb, xa, xb = max(oy, 0), min(oy + h, src), max(ox, 0), min(ox + w, src)
            if ya < yb and xa < xb:
                own[ya:yb, xa:xb] |= p["own"][ya - oy:yb - oy, xa - ox:xb - ox]
        rects = np.array([p["rect"] for p in placed], np.int32).reshape(-1, 4, 2)
        idx = np.array([p["word"] for p in placed], np.int64)
        none = np.zeros((0, 4, 2), np.int32)

        def sheet(page, rects):
            n = len(rects)
            return data.SheetData(
                series=sd.series, sheet="scene", height=src, width=src, ink_bits=np.packbits(page, axis=1),
                words=rects, kinds=words.kind[idx] if n else np.zeros(0, np.uint8),
                angles=words.angle[idx] if n else np.zeros(0, np.int16), other=none,
                points=np.zeros((0, 2), np.int32), world=np.zeros((0, 0, 2)), ft_per_px=sd.ft_per_px, cores=rects)
        c = src // 2
        t = data.crop_targets(sheet(ink, rects), c, c, size, scale, turns, words=True)
        mine = data.crop_targets(sheet(own, none), c, c, size, scale, turns)["ink"]
        rect = t["text"] > 0
        text = rect & mine
        out = dict(t)
        out["text"] = text.view(np.uint8)
        out["kind"] = np.where(text, t["kind"], data.NO_KIND).astype(np.uint8)
        out["angle"] = np.where(text, t["angle"], 0).astype(np.int16)
        out["known"] = t["known"] & ~(rect & ~t["ink"])
        out["_own"] = own
        return out


@dataclasses.dataclass
class Banks:
    """Composites and scenes from one bank object (`train.fit`'s `bank`): `crop` is the composites' (`composite.Bank`),
    `scene` the scenes' (`SceneBank`)."""
    composites: object
    scenes: SceneBank

    def crop(self, rng, sd, y, x, size, scale, turns):
        return self.composites.crop(rng, sd, y, x, size, scale, turns)

    def scene(self, rng, size, scale, turns):
        return self.scenes.crop(rng, size, scale, turns)

    def sizes(self) -> dict:
        return {"composites": self.composites.sizes(), "scenes": self.scenes.sizes()}

    @property
    def tally(self) -> collections.Counter:
        return self.composites.tally + self.scenes.tally
