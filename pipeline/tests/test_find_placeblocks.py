"""The finder's areas as placement's blocks (`find.placeblocks`): an area's outline is made by the tracing step's own
per-block outline code (vertices on px centres, as cv2 contours of the traced pieces are), an area touching the page's
edge or a px off the frame cut is cut, an outlined (mostly white) block is moved in by half the measured line width and
a solid one is not (the tracing step's own rule), two blocks joined by a gap in the street answer stay one outline
(with fewer good corners), each block keeps the chances its read gave round and on its rim, joined areas and today's
traced blocks against the areas are counted for the report, and the blocks file is read as placement reads today's.
The outlines match the tracing step's own outlines of a page it can trace, outlined and solid blocks alike, vertex for
vertex.  `write_sheet` replaces only a trial root's blocks file and leaves its freshness record as it was; the
outlines picture shows today's traced blocks and the finder's side by side; `sfzmaps find blocks` refuses an arm root
that was not made."""
import hashlib
import json

import cv2
import numpy as np
import pytest
from shapely.geometry import mapping, shape

from sfzmaps import linework
from sfzmaps.find import areas, placeblocks
from sfzmaps.find.read import LEVELS

H, W = 200, 524
BLOCK1 = (slice(40, 160), slice(50, 250))      # 200 x 120 px, cols 50..249, rows 40..159
BLOCK2 = (slice(40, 160), slice(274, 474))     # the same, past a 24 px street (cols 250..273)
STROKE = 2


def _outline(ink, sl, stroke=STROKE):
    """Draw a `stroke` px outline round the inside of the box `sl`."""
    ys, xs = sl
    ink[ys.start:ys.start + stroke, xs] = ink[ys.stop - stroke:ys.stop, xs] = True
    ink[ys, xs.start:xs.start + stroke] = ink[ys, xs.stop - stroke:xs.stop] = True


def _page(neck=False, solid=(), stroke=STROKE):
    """(lab, ink, keep): two outlined blocks on white paper and their areas (the blocks' px, outline included); the
    blocks numbered in `solid` (1, 2) are solid ink instead; with `neck`, a 10 px neck of ground across the street
    joins them (rows 95..104)."""
    ink = np.zeros((H, W), bool)
    ground = np.zeros((H, W), bool)
    for i, sl in enumerate((BLOCK1, BLOCK2), start=1):
        if i in solid:
            ink[sl] = True
        else:
            _outline(ink, sl, stroke)
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


def _outlines(lab, ink, keep):
    return placeblocks.outlines(lab, placeblocks.edge_cut(lab, keep), ink, keep=keep, photo=False)


def test_corners_sit_at_pixel_centres():
    lab, ink, keep = _page(solid=(1, 2))           # solid blocks: not moved in
    polys, ids, moved, _ = _outlines(lab, ink, keep)
    assert ids == [1, 2] and moved == [False, False]
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


def test_an_outlined_block_is_moved_in_by_half_the_line_width_and_a_solid_block_is_not():
    lab, ink, keep = _page(solid=(2,), stroke=3)    # block 1 outlined in 3 px ink, block 2 solid ink
    got = _outlines(lab, ink, keep)
    assert got.stroke_px == 3.0 == linework.stroke_width(ink)
    assert got.ids == [1, 2] and got.moved_in == [True, False]
    # the outlined block's sides 1.5 px in from its outer px; the solid block's on its outer px
    assert _near(_vertices(got.polys[0]), [(51.5, 41.5), (247.5, 41.5), (247.5, 157.5), (51.5, 157.5)], tol=0.01)
    assert _near(_vertices(got.polys[1]), [(274, 40), (473, 40), (473, 159), (274, 159)], tol=0.01)


def test_two_blocks_joined_by_a_neck_are_one_outline_with_fewer_good_corners():
    lab, ink, keep = _page()
    polys = _outlines(lab, ink, keep).polys
    assert len(polys) == 2 and len(linework.good_corners_px(polys)) == 8
    lab, ink, keep = _page(neck=True)
    assert lab.max() == 1
    polys, ids, _, _ = _outlines(lab, ink, keep)
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
    polys, ids, _, _ = _outlines(lab, ink, keep)
    chances = np.zeros((2, H, W), np.float32)
    chances[0][lab == 0] = 1.0
    conf = placeblocks.outline_confidence(lab, ids, chances, keep=keep)
    d = json.loads(json.dumps(placeblocks.blocks_file(polys, ids, conf, {"read": "r.npz", "run": "run1",
                                                                         "stroke_px": 1.0, "moved_in": 2})))
    assert d["type"] == "FeatureCollection"
    c = np.array([[np.nan if v is None else v for v in c] for c in d["corners"]], float).reshape(-1, 2)  # as do_refine
    np.testing.assert_allclose(c, np.asarray(linework.good_corners_px(polys), float).reshape(-1, 2))
    assert len(c) == 8
    assert [shape(f["geometry"]).equals(p) for f, p in zip(d["features"], polys)] == [True] * len(polys)
    assert [f["properties"] for f in d["features"]] == [{"area": i, **k} for i, k in zip(ids, conf)]
    assert d["finder"]["not_street"].startswith("street chance <= 0.5")
    assert (d["finder"]["read"], d["finder"]["run"], d["finder"]["stroke_px"], d["finder"]["moved_in"]) == \
        ("r.npz", "run1", 1.0, 2)


def _traceable_page():
    """A page the tracing step can trace: a 2 x 3 grid of 200 x 120 px blocks, a triangle and a trapezoid (slanted
    sides), all outlined in 3 px black ink on white paper but the grid's middle top block and the trapezoid, which are
    solid ink, 24 px streets in the grid and 60 px of paper round it.  Returns (img, ground): ground is the blocks' px,
    outline included (the areas a perfect read would give)."""
    stroke = 3
    h, w = 2 * 120 + 24 + 120, 3 * 200 + 2 * 24 + 120 + 520
    ink = np.zeros((h, w), bool)
    ground = np.zeros((h, w), bool)
    for r in range(2):
        for c in range(3):
            sl = (slice(60 + r * 144, 60 + r * 144 + 120), slice(60 + c * 224, 60 + c * 224 + 200))
            if (r, c) == (0, 1):
                ink[sl] = True
            else:
                _outline(ink, sl, stroke)
            ground[sl] = True
    x = 3 * 200 + 2 * 24 + 120
    for pts, solid in (([[x + 20, 60], [x + 230, 60], [x + 20, 300]], False),
                       ([[x + 270, 60], [x + 460, 60], [x + 420, 300], [x + 300, 300]], True)):
        pts = np.array(pts, np.int32)
        fill, line = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
        cv2.fillPoly(fill, [pts], 1)
        cv2.polylines(line, [pts], True, 1, stroke)
        ink |= (line > 0) | ((fill > 0) if solid else False)
        ground |= (fill > 0) | (line > 0)
    img = cv2.cvtColor(np.where(ink, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return img, ground


def test_the_outlines_are_the_tracing_steps_own():
    img, ground = _traceable_page()
    traced = linework.extract_blocks(img)
    keep = np.ones(ground.shape, bool)
    lab = areas.areas(np.where(ground, 0.0, 1.0), keep)
    got = placeblocks.outlines(lab, placeblocks.edge_cut(lab, keep), linework.binarize(img), keep=keep,
                               photo=linework.is_photo(img))
    assert len(got.polys) == len(traced) == 8 and sorted(got.ids) == list(range(1, 9))
    assert sum(got.moved_in) == 6                   # the six outlined blocks are moved in, the two solid ones not
    # No shift of our own: the tracing step's own white-or-dark rule and half-stroke move give the same polygons
    for p in got.polys:
        t = min(traced, key=lambda q: q.centroid.distance(p.centroid))
        assert _near(_vertices(p), _vertices(t), tol=1.0)


def _square_read(n=180, lo=40, hi=139):
    """(street, keep): a read whose ground (street chance 0) is the px lo..hi square, street (chance 1) round it.  The
    ground's edge lies half a px outside its outer px: at lo - 0.5 and hi + 0.5."""
    street = np.ones((n, n), np.float32)
    street[lo:hi + 1, lo:hi + 1] = 0.0
    return street, np.ones((n, n), bool)


def _box(x0, y0, x1, y1):
    from shapely.geometry import box
    return box(x0, y0, x1, y1)


def test_an_outline_inside_the_area_has_a_positive_offset():
    street, keep = _square_read()
    got = placeblocks.outline_offset(street, keep, [_box(43, 43, 136, 136)])
    # 3 px in from the outer px, 3.5 px in from the ground's edge: the finder's area reaches 3.5 px past the outline
    assert abs(got["median_px"] - 3.5) <= 0.05 and got["p25_px"] <= got["median_px"] <= got["p75_px"]
    assert got["points"] == 4 * 93                  # the exterior, 372 px long, sampled every px


def test_the_same_outline_has_no_offset():
    street, keep = _square_read()
    got = placeblocks.outline_offset(street, keep, [_box(40, 40, 139, 139)])
    assert abs(got["median_px"] - 0.5) <= 0.05      # its outer px on the outline: the area's edge half a px past it


def test_an_outline_outside_the_area_is_negative():
    street, keep = _square_read()
    # The line on px 37 lies 2.5 px outside the ground's edge at 39.5 (the brief's -3.5 counted from px 40's centre)
    got = placeblocks.outline_offset(street, keep, [_box(37, 37, 142, 142)])
    assert abs(got["median_px"] + 2.5) <= 0.05
    # the mirror of the first test's outline about the ground's edge: 3.5 px outside it
    got = placeblocks.outline_offset(street, keep, [_box(36, 36, 143, 143)])
    assert abs(got["median_px"] + 3.5) <= 0.05


def test_the_offset_counts_only_the_points_in_the_frame_cut():
    street, keep = _square_read()
    keep[:, :44] = False          # the outline's left side (col 43) lies off the frame cut: it is not measured
    street[:, :44] = 0.0          # (a read holds 0 there)
    got = placeblocks.outline_offset(street, keep, [_box(43, 43, 136, 136), _box(150, 150, 150, 150)])
    assert got["points"] == 4 * 93 - 94             # less the 94 px of col 43, rows 43..136; an empty outline adds none
    assert got["p25_px"] > 0


def test_a_traced_block_alone_in_its_area_is_one_to_one():
    lab, _, _ = _page(neck=True)                    # one area holding both blocks
    lab = lab.copy()
    lab[180:195, 10:30] = 2                         # an area holding one block
    traced = [_box(55, 45, 245, 155), _box(280, 45, 470, 155), _box(12, 182, 28, 193), _box(490, 180, 510, 195)]
    assert placeblocks.one_to_one(lab, traced) == [False, False, True, False]      # the last lies on no area


def test_traced_blocks_are_counted_against_the_areas():
    lab, _, _ = _page(neck=True)                    # area 1 holds blocks 1 and 2
    lab = lab.copy()
    lab[180:195, 10:30] = 2                         # area 2 holds one block
    lab[170:200, 300:340] = 3                       # area 3 holds two more
    traced = [_box(55, 45, 245, 155), _box(280, 45, 470, 155), _box(12, 182, 28, 193),
              _box(302, 172, 318, 197), _box(322, 172, 338, 197),
              _box(490, 180, 510, 195),             # on no area
              _box(330, 162, 345, 175)]             # 60 of its 224 px on area 3, the larger part on street
    got = placeblocks.traced_in_areas(lab, traced)
    assert got == {"one_to_one": 1, "sharing": 4, "sharing_areas": 2, "on_street": 2}


def test_a_traced_block_of_ink_is_counted_as_ink():
    lab, ink, keep = _page(solid=(1,), stroke=3)    # block 1 solid ink, block 2 outlined
    anchors = placeblocks.anchors(ink, keep, photo=False, scale=1.0)
    traced = [_box(50, 40, 249, 159), _box(275.5, 41.5, 471.5, 157.5)]
    assert placeblocks.mostly_ink(ink, traced, anchors=anchors) == [True, False]
    # the tracing step's rule counts its seed px, not every px: a block whose outline is drawn so thick that most of
    # its px are ink is still white inside, and not counted as ink
    thick = np.zeros((H, W), bool)
    _outline(thick, BLOCK2, stroke=22)            # too thin for the tracing step to take as a solid fill
    assert thick[BLOCK2].mean() > 0.5
    anchors = placeblocks.anchors(thick, keep, photo=False, scale=1.0)
    assert placeblocks.mostly_ink(thick, [_box(274, 40, 473, 159)], anchors=anchors) == [False]


def test_outline_offset_refuses_a_sheet_with_no_read(tmp_path, monkeypatch):
    from find_fixtures import invoke
    from sfzmaps.find import pages, trial
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    monkeypatch.setattr(trial, "TRIAL", tmp_path / "find" / "trial")
    r = invoke(["find", "outline-offset", "--run", "R", "--trial", "T", "--sheet", "1921/x"])
    assert r.exit_code == 1
    assert "sfzmaps find read --run R --sheet 1921/x" in r.output
    assert not (tmp_path / "find" / "trial").exists()


def test_the_outlines_picture_shows_each_side_on_its_own_panel():
    from sfzmaps.find import look
    lum = np.ones((H, W), np.float32)
    traced, found = [_box(274, 40, 473, 159)], [_box(50, 40, 249, 159), _box(274, 40, 473, 159)]
    pic = look.outlines_picture(lum, traced, np.zeros((0, 2)), found, np.array([[50.0, 40.0]]),
                                "RESULT (debugging picture): two blocks", found_sure=[1.0, 0.0])
    band = pic.shape[0] - H
    assert pic.shape == (band + H, 2 * W + look.GAP, 3)
    right = W + look.GAP
    assert tuple(pic[band + 100, right + 50]) == look.FOUND          # on the found outline, 1 px wide
    assert tuple(pic[band + 100, right + 51]) == (255, 255, 255)
    assert tuple(pic[band + 100, 50]) == (255, 255, 255)             # no traced polygon there: the left panel blank
    assert tuple(pic[band + 100, 274]) == look.TRACED                # the traced outline, on the left only
    assert tuple(pic[band + 100, right + 274]) == look.UNSURE        # a found block whose read was unsure of its street
    assert tuple(pic[band + 40, right + 50]) == look.CORNER          # a good corner, as a dot
    with pytest.raises(ValueError):
        look.outlines_picture(lum, traced, np.zeros((0, 2)), found, np.zeros((0, 2)), "two blocks")


def _sheet(tmp_path, monkeypatch):
    """A trial root holding a traceable page (`_traceable_page`) as sheet s/a, with a blocks file and its freshness
    record; its frame cut (all map), a read of it that calls its blocks' ground not street, and today's traced blocks
    of it.  Returns (run folder, root, traced blocks file, the record's bytes)."""
    from sfzmaps.find import read, wholesheet
    from sfzmaps.frame import data as frame_data
    img, ground = _traceable_page()
    monkeypatch.setattr(frame_data, "OUT", tmp_path / "frame")
    (tmp_path / "frame/s/a").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "frame/s/a/outside.png"), np.zeros(ground.shape, np.uint8))
    root = tmp_path / "trial/T/finder"
    work = root / "work/s/a"
    work.mkdir(parents=True)
    cv2.imwrite(str(work / "page.png"), img)
    (work / "blocks_px.geojson").write_text('{"type": "FeatureCollection", "features": [], "corners": []}')
    record = b'{"inputs": {"page.png": "abc", "page.json": "def", "linework.py": "ghi"}}'
    (work / "blocks_px.geojson.inputs.json").write_bytes(record)
    run = tmp_path / "runs/R"
    read.save(wholesheet.paths(run, ("s", "a"))[0], np.stack([np.where(ground, 0.0, 1.0), np.zeros(ground.shape)]))
    traced = linework.extract_blocks(img)
    f_traced = tmp_path / "trial/T/traced/work/s/a/blocks_px.geojson"
    f_traced.parent.mkdir(parents=True)
    f_traced.write_text(json.dumps({"type": "FeatureCollection", "corners": linework.good_corners_px(traced).tolist(),
                                    "features": [{"type": "Feature", "properties": {}, "geometry": mapping(p)}
                                                 for p in traced]}))
    return run, root, f_traced, record


def test_write_sheet_keeps_the_blocks_sidecar(tmp_path, monkeypatch):
    run, root, f_traced, record = _sheet(tmp_path, monkeypatch)
    pts = np.array([[160.0, 120.0], [384.0, 120.0], [608.0, 264.0], [5.0, 5.0]])     # three blocks, one on street
    got = placeblocks.write_sheet(("s", "a"), run, root, f_traced, pts)
    f = root / "work/s/a/blocks_px.geojson"
    assert (root / "work/s/a/blocks_px.geojson.inputs.json").read_bytes() == record
    assert got["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    d = json.loads(f.read_text())
    assert len(d["features"]) == got["blocks"] == 8 and len(d["corners"]) == got["good_corners"]
    assert (got["areas"], got["edge_cut"], got["dropped_by_cleaning"]) == (8, 0, 0)
    assert (got["moved_in"], got["stroke_px"]) == (6, 3.0) == (d["finder"]["moved_in"], d["finder"]["stroke_px"])
    assert (got["areas_with_two_or_more"], got["blocks_in_them"]) == (0, 0)
    assert (got["traced_blocks"], got["traced_one_to_one"], got["traced_sharing"], got["traced_on_street"]) == \
        (8, 8, 0, 0)
    assert d["finder"]["run"] == "R"


def test_write_sheet_refuses_todays_data():
    from sfzmaps import config
    with pytest.raises(ValueError, match="today's data"):
        placeblocks.write_sheet(("s", "a"), config.DATA_DIR / "find/runs/R", config.DATA_DIR,
                                config.DATA_DIR / "x.geojson", np.zeros((0, 2)))


def test_blocks_refuses_an_unprepared_arm(tmp_path, monkeypatch):
    from find_fixtures import invoke
    from sfzmaps.find import pages, trial
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    monkeypatch.setattr(trial, "TRIAL", tmp_path / "find" / "trial")
    r = invoke(["find", "blocks", "--run", "R", "--trial", "T", "--sheet", "1921/x"])
    assert r.exit_code == 1
    assert "sfzmaps find trial root --trial T --arm finder --sheet 1921/x" in r.output
    assert "sfzmaps find read --run R --sheet 1921/x" in r.output
    assert not (tmp_path / "find" / "trial").exists()


def test_the_overview_keeps_every_drawn_line():
    from sfzmaps.find import look
    img = np.full((1000, 4000, 3), 200, np.uint8)            # a grey scan
    img[500, 100:3900] = look.FOUND                          # a 1 px outline
    img[100:900, 2001] = look.TRACED
    img[300:305, 1000:1005] = look.CORNER                     # a dot on a line
    img[302, 900:1100] = look.FOUND
    small = look.reduced(img, side=1000)
    assert small.shape == (250, 1000, 3)
    assert (small[125, 25:475] == look.FOUND).all() and (small[125, 525:975] == look.FOUND).all()   # not faint grey-blue
    assert (small[25:225, 500] == look.TRACED).all()
    assert tuple(small[75, 250]) == look.CORNER              # a corner's dot drawn over the line it sits on
    assert tuple(small[10, 10]) == (200, 200, 200)
