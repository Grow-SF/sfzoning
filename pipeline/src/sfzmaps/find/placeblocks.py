"""The finder's areas as placement's blocks: outlines and graded corners in the form placement reads
(`blocks_px.geojson`), with each block's read confidence kept.

An area (`find.areas`) is connected ground the finder calls not street: the street chance at most one half, the more
likely answer.  That is the only place a chance becomes a yes/no; nothing here takes another cut on a chance.

- `edge_cut`: the areas touching the page's edge or a px off the frame cut.  They are only parts of blocks, and their
  cut sides would give false corners, so they get no outline (the tracing step drops its pieces there too).
- `outlines`: each other area's outline, made by the tracing step's own per-block outline code (smooth the piece,
  take its contour, straighten its sides and restore its corners, decide whether the block is outlined or solid and
  move an outlined one's sides in by half the sheet's measured line width, drop boxed-off lettering), mirrored from
  `linework._extract` with linework's own helpers and numbers.  The finder's area of an outlined (mostly white) block
  ends at the outer edge of its printed line and that of a solid block at the ink's edge, as the tracing step's pieces
  do, so the same rule puts the outline where the tracing step puts it: on the line's centre, or on the ink's edge.
  The vertices sit on px centres, as the tracing step's do.  An area holding two printed blocks (the finder broke the
  street between them) stays one outline here; it grades few good corners.
- `anchors`: the tracing step's seed px (block interiors and solid fills, `linework.block_material`), which its
  white-or-dark rule counts: a block is outlined (mostly white) when more of its seed px are paper than ink.
- `outline_confidence`: for each block, how sure the read was: the street chance just outside its rim (is there a
  street parting it from the next block?) and the block-edge chance on its rim.
- `joined`: for the report only, how many areas hold two or more of the year's blocks (`lot_block_points`: one point
  per block of the year's lots, placed on the page by today's placement).
- `offsets`, `outline_offset`: for the report only, where the finder's areas end against today's traced outlines.  At
  each px along a traced outline: how far the finder's area reaches past it (towards the street), in px; negative
  where the area stops short of it.  An area ending at the outer edge of an outlined block's line measures half the
  measured line width plus half a px: the tracing step takes the block's contour through the centres of its outer ink
  px (half a px inside the line's outer edge) and moves it in by half the measured line width.  For an odd line width
  the measured width is the line's, so the traced outline sits half a px inside the line's true centre.
- `one_to_one`, `traced_in_areas`, `mostly_ink`: for the reports: which traced blocks lie in a finder area holding no
  other traced block, how many traced blocks share an area or lie on ground the finder calls street, and which traced
  blocks are dark by the tracing step's own rule (solid or hatched: the tracing step leaves those outlines on the
  outer edge of their ink, not on a line's centre).
- `blocks_file`: today's blocks file (a FeatureCollection of the outlines, the good corners as
  `linework.good_corners_px` grades them), each block's area and confidence in its properties, and the read it came
  from under `finder`.
- `write_sheet`: makes a sheet's blocks file from its stored read and writes it into a trial root (never today's
  data), leaving that file's freshness record as it was, so the placement steps run there take the blocks as current.

Only `write_sheet` and `lot_block_points` read files, and only `write_sheet` writes one: the stored read, the frame
cut, the root's page, today's traced blocks (copied in the `traced` arm root), the year's lots and today's placement,
read only; the root's blocks file, written."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from scipy import ndimage as ndi
from shapely.affinity import translate
import shapely
from shapely.geometry import Polygon, mapping

from sfzmaps.linework import (CLEAN_PX, CLOSE_STREET_FRAC, MIN_BLOCK_AREA, MIN_BLOCK_SOLIDITY, POLY_EPS_STREET_FRAC,
                              SIMPLIFY_PX, SMALL_EPS_FRAC, _disk, _drop_lettering, _odd, _polygon, block_material,
                              estimate_street_px, good_corners_px, sheet_scale, street_width, stroke_width)

NOT_STREET = "street chance <= 0.5 (find.areas)"
_CROSS = ndi.generate_binary_structure(2, 1)          # a px and its four side neighbours


def edge_cut(lab: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Sorted ids of the areas in `lab` with a px on the page's edge or beside (at a side, not only a corner) a px off
    the frame cut `keep`."""
    lab = np.asarray(lab)
    touch = ndi.binary_dilation(~np.asarray(keep, bool), _CROSS)
    touch[0, :] = touch[-1, :] = touch[:, 0] = touch[:, -1] = True
    ids = np.unique(lab[touch])
    return ids[ids > 0]


def anchors(ink: np.ndarray, keep: np.ndarray, *, photo: bool, scale: float) -> np.ndarray:
    """Int32 (h, w): the tracing step's seed px at `scale` (block interiors and solid fills; 0 none), as
    `linework._extract` makes them (`linework.block_material`, with the stroke given only below scale 1), the px off
    the frame cut `keep` playing the part of its excluded area.  `photo`: whether the page is a photo
    (`linework.is_photo` of the page)."""
    ink = np.asarray(ink, bool)
    stroke = stroke_width(ink) if scale < 1.0 else 0.0
    return block_material(ink, ~np.asarray(keep, bool), photo=photo, scale=scale, stroke_px=stroke)[1]


def _white(counts_white: np.ndarray, counts_dark: np.ndarray) -> np.ndarray:
    """The tracing step's white-or-dark rule (`linework._extract`): outlined (mostly white) when more seed px are
    paper than ink."""
    return counts_white > counts_dark


class Outlines(NamedTuple):
    polys: list[Polygon]       # page px, x right / y down, px centres at integers
    ids: list[int]             # the area of each
    moved_in: list[bool]       # whether each was moved in by half the line width (an outlined block)
    stroke_px: float           # the sheet's measured line width (`linework.stroke_width` of its ink)


def _pass(lab: np.ndarray, drop: set[int], ink: np.ndarray, keep: np.ndarray, photo: bool, sw: float, stroke: float,
          scale: float) -> Outlines:
    """One pass of the tracing step's per-piece outline code at `scale`, an area of `lab` playing the part of a group.

    Mirrors `linework._extract` (linework.py as of commit 50c1100): the scaled sizes (lines 613-614, 627, 632), the
    seed px and the white-or-dark counts (lines 616-617, 633-634), the half stroke (line 626) and the loop over the
    groups (lines 642-671), then `_drop_lettering` (line 672).  One thing differs: the pieces to skip are `drop` rather
    than linework's `bad` (computed at lines 636-640 from the page's edge and the excluded area).  Keep the two in step
    if linework's loop changes."""
    clean_px = _odd(CLEAN_PX * scale)
    min_area = MIN_BLOCK_AREA * scale ** 2
    seeds = anchors(ink, keep, photo=photo, scale=scale) > 0
    half_stroke = stroke / 2.0
    eps = max(SIMPLIFY_PX, POLY_EPS_STREET_FRAC * sw)
    close_px = max(clean_px, int(round(CLOSE_STREET_FRAC * sw)) | 1)
    is_white = np.bincount(lab[seeds & ~ink].ravel(), minlength=lab.max() + 1)
    is_dark = np.bincount(lab[seeds & ink].ravel(), minlength=lab.max() + 1)
    h, w = lab.shape
    polys: list[Polygon] = []
    ids: list[int] = []
    moved: list[bool] = []
    for g, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None or g in drop:
            continue
        y0, y1 = max(0, sl[0].start - 1), min(h, sl[0].stop + 1)
        x0, x1 = max(0, sl[1].start - 1), min(w, sl[1].stop + 1)
        pad = close_px + 1
        sub = np.zeros((y1 - y0 + 2 * pad, x1 - x0 + 2 * pad), np.uint8)
        sub[pad:-pad, pad:-pad] = lab[y0:y1, x0:x1] == g
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, _disk(close_px))
        sub = cv2.morphologyEx(sub, cv2.MORPH_OPEN, _disk(clean_px))
        if int(sub.sum()) < min_area:
            continue
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        if area < min_area or hull <= 0 or area / hull < MIN_BLOCK_SOLIDITY:
            continue
        poly = _polygon(c, min(eps, SMALL_EPS_FRAC * math.sqrt(area)), clean_px)
        if poly is None or poly.area < min_area:
            continue
        white = bool(_white(is_white[g], is_dark[g]))
        if white:
            poly = poly.buffer(-half_stroke, join_style="mitre")
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
        polys.append(translate(poly, x0 - pad, y0 - pad))
        ids.append(g)
        moved.append(white)
    kept = {id(p) for p in _drop_lettering(polys)}       # linework's own filter; it returns the same objects
    on = [id(p) in kept for p in polys]
    return Outlines([p for p, k in zip(polys, on) if k], [g for g, k in zip(ids, on) if k],
                    [m for m, k in zip(moved, on) if k], float(stroke))


def outlines(lab: np.ndarray, drop: np.ndarray, ink: np.ndarray, *, keep: np.ndarray, photo: bool) -> Outlines:
    """The outline of every area of `lab` not in `drop`, by the tracing step's outline code (`_pass`): (the polygons
    in page px, x right / y down, px centres at integers; the area id of each; whether each was moved in; the measured
    line width).  An area is moved in by half the sheet's measured line width (`linework.stroke_width` of `ink`) when
    it is outlined by the tracing step's own rule (more of its seed px, `anchors`, are paper than ink), and left where
    it is when it is solid, as the tracing step does.  `photo`: whether the page is a photo (`linework.is_photo`).

    The scale is set as `linework.extract_blocks` sets it: a first pass at scale 1, then, when the street width
    measured from that pass's outlines (`estimate_street_px`) is a measurement, `sheet_scale` of it, and a second pass
    if that is not 1.  The street width that sizes the smoothing and the straightening is `street_width` of the paper
    (not `ink`) inside the frame cut `keep`, as the tracing step measures it on the paper inside its excluded area."""
    lab = np.asarray(lab, np.int32)
    ink = np.asarray(ink, bool)
    keep = np.asarray(keep, bool)
    drop_set = {int(g) for g in np.asarray(drop).ravel()}
    sw = street_width(~ink & keep)
    stroke = stroke_width(ink)
    got = _pass(lab, drop_set, ink, keep, photo, sw, stroke, 1.0)
    est, measured = estimate_street_px(got.polys)
    scale = sheet_scale(est) if measured else 1.0
    return got if scale == 1.0 else _pass(lab, drop_set, ink, keep, photo, sw, stroke, scale)


def tracing_scale(polys: list[Polygon]) -> float:
    """The scale the tracing step's second pass takes from block outlines `polys` (`sheet_scale` of their measured
    street width; 1 when it is not a measurement): for the reports on today's traced blocks, whose first pass is not
    kept."""
    est, measured = estimate_street_px(polys)
    return sheet_scale(est) if measured else 1.0


def _pct(v: np.ndarray, q: float) -> float:
    return float(np.percentile(v, q)) if len(v) else float("nan")


def outline_confidence(lab: np.ndarray, ids: list[int], chances: np.ndarray, *, keep: np.ndarray) -> list[dict]:
    """For each area id, how sure the read was about its outline (a dict per id, in order):

    - `outline_px`: the number of the area's rim px (its own px beside, at a side, a px outside it or the page's edge);
    - `outline_street_p10`, `outline_street_p50`: the 10th and 50th percentile of the street chance on the px just
      outside the rim (beside a rim px at a side), inside the frame cut `keep` (a read holds 0 off the cut);
    - `outline_edge_p50`: the median block-edge chance on the rim px.

    `chances` is a read as `find.read.load` gives it: (2, h, w), street then block edge.  A percentile of no px is
    NaN."""
    lab = np.asarray(lab)
    keep = np.asarray(keep, bool)
    street, edge = np.asarray(chances[0]), np.asarray(chances[1])
    h, w = lab.shape
    boxes = ndi.find_objects(lab)
    out = []
    for g in ids:
        sl = boxes[g - 1]
        y0, y1 = max(0, sl[0].start - 1), min(h, sl[0].stop + 1)
        x0, x1 = max(0, sl[1].start - 1), min(w, sl[1].stop + 1)
        m = lab[y0:y1, x0:x1] == g
        rim = m & ~ndi.binary_erosion(m, _CROSS, border_value=0)     # off the page counts as outside
        ring = ndi.binary_dilation(m, _CROSS) & ~m & keep[y0:y1, x0:x1]
        s = street[y0:y1, x0:x1][ring]
        out.append({"outline_px": int(rim.sum()),
                    "outline_street_p10": _pct(s, 10), "outline_street_p50": _pct(s, 50),
                    "outline_edge_p50": _pct(edge[y0:y1, x0:x1][rim], 50)})
    return out


def joined(lab: np.ndarray, points_px: np.ndarray) -> dict:
    """For the report only: how many areas hold two or more of `points_px` (N x 2, x, y in page px, px centres at
    integers; one point per block of the year's lots), and how many points those areas hold.  Points off the page or
    on no area are not counted."""
    lab = np.asarray(lab)
    p = np.rint(np.asarray(points_px, float).reshape(-1, 2)).astype(int)
    on = (p[:, 0] >= 0) & (p[:, 0] < lab.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < lab.shape[0])
    a = lab[p[on, 1], p[on, 0]]
    n = np.bincount(a[a > 0])
    many = n >= 2
    return {"areas_with_two_or_more": int(many.sum()), "blocks_in_them": int(n[many].sum())}


def _num(x):
    """`x` as a float, or None when it is not finite (as `runner._num`)."""
    x = float(x)
    return x if math.isfinite(x) else None


def blocks_file(polys: list[Polygon], ids: list[int], confidence: list[dict], finder: dict) -> dict:
    """The blocks file placement reads (`blocks_px.geojson`, as `runner.do_blocks` writes it): a FeatureCollection of
    the outlines with `{"area": id, **confidence}` as each one's properties (a value that is not finite as null), the
    corners `linework.good_corners_px` grades good (null for a value that is not finite), and under `finder` the
    read's details as given (the read, the run, the measured line width, how many blocks were moved in) and how not
    street was taken."""
    corners = np.asarray(good_corners_px(polys), float).reshape(-1, 2)

    def props(g, c):
        return {"area": int(g), **{k: (_num(v) if isinstance(v, float) else v) for k, v in c.items()}}

    return {"type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": props(g, c), "geometry": mapping(p)}
                         for p, g, c in zip(polys, ids, confidence)],
            "corners": [[_num(x), _num(y)] for x, y in corners],
            "finder": {**finder, "not_street": NOT_STREET}}


def offsets(street: np.ndarray, keep: np.ndarray, traced: list[Polygon]) -> np.ndarray:
    """Float (n,): at each px along each traced outline's exterior, how far the finder's area reaches past the outline,
    in px: positive where the area runs on past it (towards the street), negative where it stops short of it.

    The area here is `areas`' ground: the px inside the frame cut `keep` whose street chance is at most one half.  Its
    edge lies half a px outside its outer px, so a px of ground is (its distance to the nearest px that is not ground)
    less half a px inside it, and a px that is not ground is (its distance to the nearest px of ground) less half a px
    outside it.  The outlines (page px, px centres at integers, as `blocks_px.geojson` holds them) are walked from
    their first vertex in steps of one px, and that distance is read at each point between the four px round it.  A
    point whose nearest px is off the page or off the frame cut is not measured: there the read says nothing."""
    keep = np.asarray(keep, bool)
    ground = (np.asarray(street) <= 0.5) & keep
    if not ground.any() or ground.all():
        return np.zeros(0)
    signed = np.where(ground, ndi.distance_transform_edt(ground) - 0.5,
                      0.5 - ndi.distance_transform_edt(~ground)).astype(np.float32)
    h, w = ground.shape
    out = []
    for p in traced:
        ring = p.exterior
        if ring.length <= 0:
            continue
        xy = shapely.get_coordinates(shapely.line_interpolate_point(ring, np.arange(0.0, ring.length, 1.0)))
        ix, iy = np.rint(xy[:, 0]).astype(int), np.rint(xy[:, 1]).astype(int)
        on = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        on[on] = keep[iy[on], ix[on]]
        if on.any():
            out.append(ndi.map_coordinates(signed, [xy[on, 1], xy[on, 0]], order=1, mode="nearest"))
    return np.concatenate(out) if out else np.zeros(0)


def outline_offset(street: np.ndarray, keep: np.ndarray, traced: list[Polygon]) -> dict:
    """How far the finder's area reaches past the traced outlines (`offsets`), summed up: `median_px`, `p25_px`,
    `p75_px` (NaN when no point is measured) and `points`, the number of outline points measured."""
    v = offsets(street, keep, traced)
    return {"median_px": _pct(v, 50), "p25_px": _pct(v, 25), "p75_px": _pct(v, 75), "points": int(len(v))}


def _inside(p: Polygon, shape: tuple[int, int]) -> tuple[tuple[slice, slice], np.ndarray]:
    """(the px box, a bool mask over it): the px of the page inside `p`'s exterior, its rim included."""
    h, w = shape
    x0, y0, x1, y1 = p.bounds
    xa, ya = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    xb, yb = min(w, int(math.ceil(x1)) + 1), min(h, int(math.ceil(y1)) + 1)
    if xb <= xa or yb <= ya:
        return (slice(0, 0), slice(0, 0)), np.zeros((0, 0), bool)
    m = np.zeros((yb - ya, xb - xa), np.uint8)
    pts = np.rint((np.asarray(p.exterior.coords) - (xa, ya)) * 16).astype(np.int32)   # 1/16 px: cv2's shift=4
    cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_8, shift=4)
    return (slice(ya, yb), slice(xa, xb)), m > 0


def _held(lab: np.ndarray, traced: list[Polygon]) -> np.ndarray:
    """Int (n,): for each traced block, the finder area holding the larger part of its px (`_inside`), street (no
    area, 0) counting as one more part; 0 when street is the larger part (a tie goes to street) or the block has no
    px on the page."""
    lab = np.asarray(lab)
    held = []
    for p in traced:
        sl, m = _inside(p, lab.shape)
        a = lab[sl][m]
        held.append(int(np.bincount(a).argmax()) if len(a) else 0)
    return np.asarray(held, int)


def one_to_one(lab: np.ndarray, traced: list[Polygon]) -> list[bool]:
    """For the report: for each traced block, whether the finder area holding the larger part of it (`_held`) holds no
    other traced block.  A block mostly on street (no area) is not one to one."""
    held = _held(lab, traced)
    n = np.bincount(held, minlength=1)
    return [bool(g > 0 and n[g] == 1) for g in held]


def traced_in_areas(lab: np.ndarray, traced: list[Polygon]) -> dict:
    """For the report: today's traced blocks against the finder's areas, each block in the area holding the larger part
    of it (`_held`): `one_to_one`, the blocks alone in their area; `sharing`, the blocks sharing an area with other
    traced blocks, in `sharing_areas` areas; `on_street`, the blocks mostly on ground the finder calls street (or off
    the frame cut: no area)."""
    held = _held(lab, traced)
    n = np.bincount(held, minlength=1)
    n[0] = 0
    on = held > 0
    return {"one_to_one": int((on & (n[held] == 1)).sum()), "sharing": int((on & (n[held] >= 2)).sum()),
            "sharing_areas": int((n >= 2).sum()), "on_street": int((~on).sum())}


def mostly_ink(ink: np.ndarray, traced: list[Polygon], *, anchors: np.ndarray) -> list[bool]:
    """For the report: for each traced block, whether it is dark by the tracing step's own rule: not more of the seed
    px `anchors` (`placeblocks.anchors`) inside it (`_inside`) paper than ink.  The tracing step moves the outline of
    an outlined (white) block in to its printed line's centre and leaves a dark block (solid or hatched) on its ink's
    outer edge; it counts the seed px of its own piece, so this is its question asked of the traced outline."""
    ink = np.asarray(ink, bool)
    seeds = np.asarray(anchors) > 0
    out = []
    for p in traced:
        sl, m = _inside(p, ink.shape)
        on = m & seeds[sl]
        out.append(not bool(_white(int((on & ~ink[sl]).sum()), int((on & ink[sl]).sum()))))
    return out


def _today(root: Path) -> str | None:
    """Why `root` may not be written: it is today's data root or lies in its work/, warp/ or out/; None when it is
    a separate folder."""
    from sfzmaps import config
    from sfzmaps.find import trial
    r, s = Path(root).resolve(), config.DATA_DIR.resolve()
    if r == s or any(r == (s / x).resolve() or (s / x).resolve() in r.parents for x in trial.TODAY):
        return f"{root}: is in today's data; the finder's blocks are written only into a trial root"
    return None


def _polygons(f: Path) -> tuple[list[Polygon], int]:
    """The polygons of a blocks file and its number of good corners."""
    from shapely.geometry import shape
    d = json.loads(Path(f).read_text())
    return [shape(ft["geometry"]) for ft in d["features"]], len(d.get("corners") or [])


def lot_block_points(sid: tuple[str, str], shape: tuple[int, int]) -> np.ndarray:
    """Float (n, 2), page px (x, y; px centres at integers): one point per block of the year's lots near the page (the
    April 2000 table's lots for a series labelled from it, else the lots of the series' year), each block's lots
    joined and its representative point (a point inside it) placed on the page of `shape` by today's placement
    (data/warp/<series>/<sheet>/transform.json with its correction).  Only the lots within the page's footprint's
    bounds are joined, so a block cut by those bounds gives a point of its part there.  For `joined`; read only."""
    import shapely as shp
    from sfzmaps.districts.read import load_parcels, series_year
    from sfzmaps.find import lotlabels, pages
    from sfzmaps.solver.table import load_table
    series = sid[0]
    lots = load_table() if pages.source(series).lots == "table" else load_parcels(series_year(series))
    tr = lotlabels.placement(sid, "warp")
    h, w = shape
    t = np.linspace(0.0, 1.0, 33)
    rim = np.concatenate([np.c_[t * (w - 1), 0 * t], np.c_[t * (w - 1), 0 * t + h - 1],
                          np.c_[0 * t, t * (h - 1)], np.c_[0 * t + w - 1, t * (h - 1)]])
    world = np.asarray(tr.to_world(rim), float)
    world = world[np.isfinite(world).all(axis=1)]
    x0, y0 = world.min(axis=0)
    x1, y1 = world.max(axis=0)
    near = lots.cx[x0:x1, y0:y1]
    if not len(near):
        return np.zeros((0, 2))
    blocks = near[["block_num", "geometry"]].dissolve(by="block_num")
    pts = shp.get_coordinates(blocks.geometry.representative_point().values)
    px = np.asarray(tr.to_pixel(pts), float).reshape(-1, 2)
    return px[np.isfinite(px).all(axis=1)]


def write_sheet(sid: tuple[str, str], run_folder: Path, root: Path, traced_file: Path, points_px: np.ndarray,
                run: str | None = None) -> dict:
    """Make the sheet's blocks file from its stored read and write it as root/work/<series>/<sheet>/blocks_px.geojson,
    leaving the freshness record beside it (`.inputs.json`) as it is, so the runner takes the blocks as current and
    does not trace them again.  ValueError, reading and writing nothing, when `root` is today's data root or lies in
    its work/, warp/ or out/.

    Reads the run's stored read of the sheet (`wholesheet.paths(run_folder, sid)`), its frame cut (`pages.load_keep`),
    the root's page.png as the runner reads it (`cv2.IMREAD_UNCHANGED`; ink by `linework.binarize`, photo by
    `linework.is_photo`) and today's traced blocks `traced_file` (the `traced` arm's copy).  The areas
    (`areas.areas`) less `edge_cut` are outlined (`outlines`), graded (`outline_confidence`) and written
    (`blocks_file`).  `points_px`: one point per block of the year's lots on the page, for `joined`.

    Returns the counts: `areas`, `edge_cut`, `dropped_by_cleaning` (areas neither cut nor outlined: too small, too
    ragged or lettering), `blocks`, `good_corners`, `moved_in`, `stroke_px`, `areas_with_two_or_more`,
    `blocks_in_them`, `lot_blocks_on_page` (the points on the page), today's `traced_blocks` and
    `traced_good_corners`, `traced_one_to_one`, `traced_sharing`, `traced_sharing_areas`, `traced_on_street`
    (`traced_in_areas`), and `sha256` of the file written."""
    from sfzmaps import linework
    from sfzmaps.find import areas, pages, read, wholesheet
    why = _today(root)
    if why:
        raise ValueError(why)
    series, sheet = sid
    work = Path(root) / "work" / series / sheet
    f_read = wholesheet.paths(run_folder, sid)[0]
    img = cv2.imread(str(work / "page.png"), cv2.IMREAD_UNCHANGED)            # as runner reads it
    if img is None:
        raise FileNotFoundError(f"{work / 'page.png'}: no page")
    chances = read.load(f_read)
    keep = pages.load_keep(sid, img.shape[:2])
    if chances.shape[1:] != keep.shape:
        raise ValueError(f"{series}/{sheet}: its read is {chances.shape[1:]}, its page {keep.shape}; read it again")
    ink = linework.binarize(img)
    lab = areas.areas(chances[0], keep)
    cut = edge_cut(lab, keep)
    got = outlines(lab, cut, ink, keep=keep, photo=linework.is_photo(img))
    conf = outline_confidence(lab, got.ids, chances, keep=keep)
    traced, traced_corners = _polygons(traced_file)
    d = blocks_file(got.polys, got.ids, conf, {"read": str(f_read), "run": run or Path(run_folder).name,
                                               "stroke_px": got.stroke_px, "moved_in": int(sum(got.moved_in))})
    text = json.dumps(d)
    f = work / "blocks_px.geojson"
    part = f.with_name(f".{f.name}.part")
    part.write_text(text)
    part.replace(f)
    pts = np.asarray(points_px, float).reshape(-1, 2)
    on_page = int(((pts[:, 0] >= -0.5) & (pts[:, 0] < lab.shape[1] - 0.5)
                   & (pts[:, 1] >= -0.5) & (pts[:, 1] < lab.shape[0] - 0.5)).sum())
    t = traced_in_areas(lab, traced)
    n_areas = int(lab.max())
    return {"areas": n_areas, "edge_cut": int(len(cut)), "dropped_by_cleaning": n_areas - int(len(cut)) - len(got.ids),
            "blocks": len(got.polys), "good_corners": len(d["corners"]), "moved_in": int(sum(got.moved_in)),
            "stroke_px": got.stroke_px, **joined(lab, pts), "lot_blocks_on_page": on_page,
            "traced_blocks": len(traced), "traced_good_corners": traced_corners,
            **{f"traced_{k}": v for k, v in t.items()},
            "sha256": hashlib.sha256(text.encode()).hexdigest()}
