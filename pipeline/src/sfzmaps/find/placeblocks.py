"""The finder's areas as placement's blocks: outlines and graded corners in the form placement reads
(`blocks_px.geojson`), with each block's read confidence kept.

An area (`find.areas`) is connected ground the finder calls not street: the street chance at most one half, the more
likely answer.  That is the only place a chance becomes a yes/no; nothing here takes another cut on a chance.

- `edge_cut`: the areas touching the page's edge or a px off the frame cut.  They are only parts of blocks, and their
  cut sides would give false corners, so they get no outline (the tracing step drops its pieces there too).
- `outlines`: each other area's outline, made by the tracing step's own per-block outline code (smooth the piece,
  take its contour, straighten its sides and restore its corners, drop boxed-off lettering), mirrored from
  `linework._extract` with linework's own helpers and numbers, and then moved in by `inset_px` on every side.  The
  vertices sit on px centres, as the tracing step's do.  An area holding two printed blocks (the finder broke the
  street between them) stays one outline here; it grades few good corners.
- `outline_confidence`: for each block, how sure the read was: the street chance just outside its rim (is there a
  street parting it from the next block?) and the block-edge chance on its rim.
- `joined`: for the report only, how many areas hold two or more of the year's blocks.
- `offsets`, `outline_offset`: where the finder's areas end against today's traced outlines, for choosing the inset.
  At each px along a traced outline: how far the finder's area reaches past it (towards the street), in px; negative
  where the area stops short of it.  `inset_px` takes the nearer of the design's two answers: no move (0), or half the
  printed line's width (the tracing step puts an outlined block's outline on the line's centre, so an area reaching
  half a line's width past it ends at the line's outer edge).
- `one_to_one`, `mostly_ink`: for the report on that measurement: which traced blocks lie in a finder area holding no
  other traced block, and which are mostly ink inside (solid or hatched: the tracing step leaves those outlines on the
  outer edge of their ink, not on a line's centre).
- `blocks_file`: today's blocks file (a FeatureCollection of the outlines, the good corners as
  `linework.good_corners_px` grades them), each block's area and confidence in its properties, and the read it came
  from under `finder`.

Reads and writes nothing."""
from __future__ import annotations

import math

import cv2
import numpy as np
from scipy import ndimage as ndi
from shapely.affinity import translate
import shapely
from shapely.geometry import Polygon, mapping

from sfzmaps.linework import (CLEAN_PX, CLOSE_STREET_FRAC, MIN_BLOCK_AREA, MIN_BLOCK_SOLIDITY, POLY_EPS_STREET_FRAC,
                              SIMPLIFY_PX, SMALL_EPS_FRAC, _disk, _drop_lettering, _odd, _polygon,
                              estimate_street_px, good_corners_px, sheet_scale, street_width)

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


def _pass(lab: np.ndarray, drop: set[int], sw: float, scale: float, inset_px: float
          ) -> tuple[list[Polygon], list[int]]:
    """One pass of the tracing step's per-piece outline code at `scale`, an area of `lab` playing the part of a group.

    Mirrors `linework._extract` (linework.py as of commit a3f11b4): the scaled sizes (lines 613-614, 627, 632) and the
    loop over the groups (lines 642-671), then `_drop_lettering` (line 672).  Two things differ: the pieces to skip are
    `drop` rather than linework's `bad` (computed at lines 636-640 from the page's edge and the excluded area), and the
    move of each side by half the outline's stroke for mostly white blocks (lines 667-670) is replaced by a move in of
    `inset_px` for every block.  Keep the two in step if linework's loop changes."""
    clean_px = _odd(CLEAN_PX * scale)
    min_area = MIN_BLOCK_AREA * scale ** 2
    eps = max(SIMPLIFY_PX, POLY_EPS_STREET_FRAC * sw)
    close_px = max(clean_px, int(round(CLOSE_STREET_FRAC * sw)) | 1)
    h, w = lab.shape
    polys: list[Polygon] = []
    ids: list[int] = []
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
        if inset_px:                    # no move at 0: the polygon stays exactly as traced (a buffer reorders it)
            poly = poly.buffer(-inset_px, join_style="mitre")
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
        polys.append(translate(poly, x0 - pad, y0 - pad))
        ids.append(g)
    kept = {id(p) for p in _drop_lettering(polys)}       # linework's own filter; it returns the same objects
    return [p for p in polys if id(p) in kept], [g for p, g in zip(polys, ids) if id(p) in kept]


def outlines(lab: np.ndarray, drop: np.ndarray, ink: np.ndarray, inset_px: float, *, keep: np.ndarray
             ) -> tuple[list[Polygon], list[int]]:
    """(polygons in page px, x right / y down, px centres at integers; the area id of each): the outline of every area
    of `lab` not in `drop`, by the tracing step's outline code (`_pass`), each side moved in by `inset_px`.

    The scale is set as `linework.extract_blocks` sets it: a first pass at scale 1, then, when the street width
    measured from that pass's outlines (`estimate_street_px`) is a measurement, `sheet_scale` of it, and a second pass
    if that is not 1.  The street width that sizes the smoothing and the straightening is `street_width` of the paper
    (not `ink`) inside the frame cut `keep`, as the tracing step measures it on the paper inside its excluded area."""
    lab = np.asarray(lab, np.int32)
    drop_set = {int(g) for g in np.asarray(drop).ravel()}
    sw = street_width(~np.asarray(ink, bool) & np.asarray(keep, bool))
    polys, ids = _pass(lab, drop_set, sw, 1.0, inset_px)
    est, measured = estimate_street_px(polys)
    scale = sheet_scale(est) if measured else 1.0
    return (polys, ids) if scale == 1.0 else _pass(lab, drop_set, sw, scale, inset_px)


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
    read's details as given (the read, the run, `inset_px`) and how not street was taken."""
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


def inset_px(offset_px: float, half_stroke_px: float) -> float:
    """The move in for the finder's outlines: of the design's two answers, no move (0) and half the printed line's
    width (`half_stroke_px`), the one nearer the measured offset.  A tie keeps 0."""
    return float(half_stroke_px) if abs(offset_px - half_stroke_px) < abs(offset_px) else 0.0


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


def one_to_one(lab: np.ndarray, traced: list[Polygon]) -> list[bool]:
    """For the report: for each traced block, whether the finder area holding most of its px (an area of `lab`, 0 for
    none) holds no other traced block.  A block on no area is not one to one."""
    lab = np.asarray(lab)
    held = []
    for p in traced:
        sl, m = _inside(p, lab.shape)
        a = lab[sl][m]
        a = a[a > 0]
        held.append(int(np.bincount(a).argmax()) if len(a) else 0)
    n = np.bincount(np.asarray(held, int), minlength=1)
    return [g > 0 and n[g] == 1 for g in held]


def mostly_ink(ink: np.ndarray, traced: list[Polygon]) -> list[bool]:
    """For the report: for each traced block, whether more than half its px (`_inside`) are ink.  The tracing step
    moves the outline of a mostly white block in to its printed line's centre and leaves a block of ink (solid or
    hatched) on its ink's outer edge; it decides on its own seed px, so this is the same question asked of the whole
    block, not its exact answer."""
    ink = np.asarray(ink, bool)
    out = []
    for p in traced:
        sl, m = _inside(p, ink.shape)
        out.append(bool(m.any()) and float(ink[sl][m].mean()) > 0.5)
    return out
