"""The finder's areas as placement's blocks (`find.placeblocks`): an area's outline is made by the tracing step's own
per-block outline code (vertices on px centres, as cv2 contours of the traced pieces are), an area touching the page's
edge or a px off the frame cut is cut, the inset moves every side in, two blocks joined by a gap in the street answer
stay one outline (with fewer good corners), each block keeps the chances its read gave round and on its rim, joined
areas are counted for the report, and the blocks file is read as placement reads today's.  The outlines match the
tracing step's own outlines of a page it can trace, vertex for vertex."""
import json

import cv2
import numpy as np
from shapely.geometry import shape

from sfzmaps import linework
from sfzmaps.find import areas, placeblocks
from sfzmaps.find.read import LEVELS

H, W = 200, 524
BLOCK1 = (slice(40, 160), slice(50, 250))      # 200 x 120 px, cols 50..249, rows 40..159
BLOCK2 = (slice(40, 160), slice(274, 474))     # the same, past a 24 px street (cols 250..273)
STROKE = 2


def _outline(ink, sl):
    """Draw a STROKE px outline round the inside of the box `sl`."""
    ys, xs = sl
    ink[ys.start:ys.start + STROKE, xs] = ink[ys.stop - STROKE:ys.stop, xs] = True
    ink[ys, xs.start:xs.start + STROKE] = ink[ys, xs.stop - STROKE:xs.stop] = True


def _page(neck=False):
    """(lab, ink, keep): two outlined blocks on white paper and their areas (the blocks' px, outline included); with
    `neck`, a 10 px neck of ground across the street joins them (rows 95..104)."""
    ink = np.zeros((H, W), bool)
    ground = np.zeros((H, W), bool)
    for sl in (BLOCK1, BLOCK2):
        _outline(ink, sl)
        ground[sl] = True
    if neck:
        ground[95:105, 250:274] = True
    keep = np.ones((H, W), bool)
    lab = areas.areas(np.where(ground, 0.0, 1.0), keep)
    return lab, ink, keep


def _vertices(p):
    return np.asarray(p.exterior.coords)[:-1]


def _near(v, want, tol=1.0):
    """Every point of `want` has a vertex of `v` within `tol` px, and `v` has no other vertex."""
    d = np.hypot(*(v[:, None, :] - np.asarray(want, float)[None]).transpose(2, 0, 1))
    return len(v) == len(want) and bool((d.min(0) <= tol).all()) and bool((d.min(1) <= tol).all())


def test_corners_sit_at_pixel_centres():
    lab, ink, keep = _page()
    polys, ids = placeblocks.outlines(lab, placeblocks.edge_cut(lab, keep), ink, 0.0, keep=keep)
    assert ids == [1, 2]
    v = _vertices(polys[0])
    assert _near(v, [(50, 40), (249, 40), (249, 159), (50, 159)])
    # integer = px centre: the area's last px is col 249 / row 159, so no vertex reaches half a px past it
    assert v[:, 0].min() > 49.5 and v[:, 0].max() < 249.5 and v[:, 1].min() > 39.5 and v[:, 1].max() < 159.5


def test_an_area_on_the_rim_or_the_page_edge_is_cut():
    lab = np.zeros((40, 80), np.int32)
    keep = np.ones((40, 80), bool)
    keep[:, :5] = False
    lab[10:20, 5:15] = 1          # its col 5 is beside col 4, off the frame cut
    lab[10:20, 25:35] = 2         # all round it is street inside the cut
    lab[0:6, 40:50] = 3           # on the page's top row
    keep[25, 55] = False
    lab[26:36, 56:66] = 4         # off the cut only at a corner (diagonal), never at a side: not cut
    np.testing.assert_array_equal(placeblocks.edge_cut(lab, keep), [1, 3])


def test_inset_moves_every_side_in():
    lab, ink, keep = _page()
    polys, ids = placeblocks.outlines(lab, placeblocks.edge_cut(lab, keep), ink, 2.0, keep=keep)
    assert ids == [1, 2]
    assert _near(_vertices(polys[0]), [(52, 42), (247, 42), (247, 157), (52, 157)])
    assert _near(_vertices(polys[1]), [(276, 42), (471, 42), (471, 157), (276, 157)])


def test_two_blocks_joined_by_a_neck_are_one_outline_with_fewer_good_corners():
    lab, ink, keep = _page()
    polys, _ = placeblocks.outlines(lab, placeblocks.edge_cut(lab, keep), ink, 0.0, keep=keep)
    assert len(polys) == 2 and len(linework.good_corners_px(polys)) == 8
    lab, ink, keep = _page(neck=True)
    assert lab.max() == 1
    polys, ids = placeblocks.outlines(lab, placeblocks.edge_cut(lab, keep), ink, 0.0, keep=keep)
    assert ids == [1] and len(polys) == 1
    assert len(linework.good_corners_px(polys)) < 8


def _levels(p):
    return np.round(np.asarray(p, float) * LEVELS) / LEVELS          # as a stored read holds it


def test_the_confidence_carries_the_chance():
    ground = np.zeros((H, W), bool)
    ground[BLOCK1] = ground[BLOCK2] = True
    street = np.where(ground, 0.0, np.where(np.arange(W)[None, :] < 262, 0.9, 0.6))   # 0.9 round 1, 0.6 round 2
    keep = np.ones((H, W), bool)
    keep[:, 474] = False          # off the frame cut beside area 2's right side: a read holds 0 there, not counted
    street[:, 474] = 0.0
    lab = areas.areas(street, keep)
    rim = ground & ~(np.pad(ground, 1)[2:, 1:-1] & np.pad(ground, 1)[:-2, 1:-1]
                     & np.pad(ground, 1)[1:-1, 2:] & np.pad(ground, 1)[1:-1, :-2])
    edge = np.where(rim, np.where(np.arange(W)[None, :] < 262, 0.8, 0.3), 0.1)
    chances = np.stack([_levels(street), _levels(edge)]).astype(np.float32)
    conf = placeblocks.outline_confidence(lab, [1, 2], chances, keep=keep)
    assert [c["outline_px"] for c in conf] == [2 * 200 + 2 * 118] * 2
    for c, s, e in zip(conf, (0.9, 0.6), (0.8, 0.3)):
        assert abs(c["outline_street_p10"] - s) <= 1 / LEVELS and abs(c["outline_street_p50"] - s) <= 1 / LEVELS
        assert abs(c["outline_edge_p50"] - e) <= 1 / LEVELS


def test_joined_counts_areas_holding_two_or_more_blocks():
    lab, _, _ = _page()
    pts = np.array([[100.0, 80.0], [200.0, 120.0], [300.0, 100.0], [262.0, 100.0]])    # x, y; the last on street
    assert placeblocks.joined(lab, pts) == {"areas_with_two_or_more": 1, "blocks_in_them": 2}


def test_the_blocks_file_reads_as_placement_reads_it():
    lab, ink, keep = _page()
    polys, ids = placeblocks.outlines(lab, placeblocks.edge_cut(lab, keep), ink, 1.0, keep=keep)
    chances = np.zeros((2, H, W), np.float32)
    chances[0][lab == 0] = 1.0
    conf = placeblocks.outline_confidence(lab, ids, chances, keep=keep)
    d = json.loads(json.dumps(placeblocks.blocks_file(polys, ids, conf, {"read": "r.npz", "run": "run1",
                                                                         "inset_px": 1.0})))
    assert d["type"] == "FeatureCollection"
    c = np.array([[np.nan if v is None else v for v in c] for c in d["corners"]], float).reshape(-1, 2)  # as do_refine
    np.testing.assert_allclose(c, np.asarray(linework.good_corners_px(polys), float).reshape(-1, 2))
    assert len(c) == 8
    assert [shape(f["geometry"]).equals(p) for f, p in zip(d["features"], polys)] == [True] * len(polys)
    assert [f["properties"] for f in d["features"]] == [{"area": i, **k} for i, k in zip(ids, conf)]
    assert d["finder"]["not_street"].startswith("street chance <= 0.5")
    assert (d["finder"]["read"], d["finder"]["run"], d["finder"]["inset_px"]) == ("r.npz", "run1", 1.0)


def _traceable_page():
    """A page the tracing step can trace: a 2 x 3 grid of 200 x 120 px blocks, a triangle and a trapezoid (slanted
    sides), all outlined in 2 px black ink on white paper, 24 px streets in the grid and 60 px of paper round it.
    Returns (img, ground): ground is the blocks' px, outline included (the areas a perfect read would give)."""
    h, w = 2 * 120 + 24 + 120, 3 * 200 + 2 * 24 + 120 + 520
    ink = np.zeros((h, w), bool)
    ground = np.zeros((h, w), bool)
    for r in range(2):
        for c in range(3):
            sl = (slice(60 + r * 144, 60 + r * 144 + 120), slice(60 + c * 224, 60 + c * 224 + 200))
            _outline(ink, sl)
            ground[sl] = True
    x = 3 * 200 + 2 * 24 + 120
    for pts in ([[x + 20, 60], [x + 230, 60], [x + 20, 300]], [[x + 270, 60], [x + 460, 60], [x + 420, 300],
                                                              [x + 300, 300]]):
        pts = np.array(pts, np.int32)
        fill, line = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
        cv2.fillPoly(fill, [pts], 1)
        cv2.polylines(line, [pts], True, 1, STROKE)
        ink |= line > 0
        ground |= (fill > 0) | (line > 0)
    img = cv2.cvtColor(np.where(ink, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return img, ground


def test_the_outlines_are_the_tracing_steps_own():
    img, ground = _traceable_page()
    traced = linework.extract_blocks(img)
    keep = np.ones(ground.shape, bool)
    lab = areas.areas(np.where(ground, 0.0, 1.0), keep)
    ink = linework.binarize(img)
    polys, ids = placeblocks.outlines(lab, placeblocks.edge_cut(lab, keep), ink, 0.0, keep=keep)
    assert len(polys) == len(traced) == 8 and sorted(ids) == list(range(1, 9))
    # The tracing step's piece is the block with its outline stroke, as an area is; it then moves each outlined block's
    # sides in by half its measured stroke (the line's centre).  Allowing for that, the polygons are the same.
    half = linework.stroke_width(ink) / 2.0
    for p in polys:
        t = min(traced, key=lambda q: q.centroid.distance(p.centroid))
        mine = p.buffer(-half, join_style="mitre")
        assert _near(_vertices(mine), _vertices(t), tol=1.0)
        assert mine.symmetric_difference(t).area <= 1.0
