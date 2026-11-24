"""Hand crops: their files, the labels made from their seeds, strokes and limits, and the ground reserved under the
test and choosing crops.  Crops are drawn with numpy: white paper, 2 px black lines."""
import json
from pathlib import Path

import numpy as np
import pytest

from find_fixtures import Flat
from sfzmaps.find import handlabels

BOX = (100, 200, 60, 40)          # the crops' box on their page: x0, y0, w, h (page px)


def _page(h=40, w=60):
    return np.ones((h, w), np.float32)


def _h(img, y, x0=0, x1=None):
    """A 2 px horizontal line, rows y and y + 1, columns x0..x1 (end exclusive; None: to the side)."""
    img[y:y + 2, x0:x1] = 0


def _v(img, x, y0=0, y1=None):
    img[y0:y1, x:x + 2] = 0


def _rect(img, x0, y0, x1, y1):
    """A 2 px outline whose outer px run x0..x1, y0..y1 (inclusive)."""
    _h(img, y0, x0, x1 + 1)
    _h(img, y1 - 1, x0, x1 + 1)
    _v(img, x0, y0, y1 + 1)
    _v(img, x1 - 1, y0, y1 + 1)


def _pg(points):
    """Crop px (x, y) as page px."""
    return tuple((x + BOX[0], y + BOX[1]) for x, y in points)


def _crop(street=(), blocks=(), open=(), strokes=(), limits=(), openings=()):
    """A crop at BOX, every mark given in crop px."""
    extra = {"openings": tuple(_pg(s) for s in openings)} if openings else {}
    return handlabels.Crop("train", "s", "p", BOX, _pg(street), _pg(blocks), _pg(open),
                           tuple(_pg(s) for s in strokes), tuple(_pg(s) for s in limits),
                           Path("train/s/p__100_200_60_40.json"), **extra)


def _street_between_two_lines():
    img = _page()
    _h(img, 10)
    _h(img, 28)
    return img


def test_the_fill_stops_at_a_line_closing_a_street():
    img = _street_between_two_lines()
    _v(img, 40, 10, 30)                   # the line closing the street's east end
    lab = handlabels.make_labels(img, _crop(street=[(20, 20)], blocks=[(50, 20)]))
    assert (lab.street[12:28, :40] == 1).all()
    assert (lab.street[12:28, 42:] == 0).all()
    # beyond the line the land is the block the dead end pokes into, as are both sides
    assert lab.area[20, 50] == lab.area[5, 20] == lab.area[35, 20] == 1


def test_letters_in_the_street_are_street():
    img = _street_between_two_lines()
    img[19:22, 30:33] = 0                 # a letter in the street
    lab = handlabels.make_labels(img, _crop(street=[(10, 20)], blocks=[(20, 4), (20, 35)]))
    assert (lab.street[19:22, 30:33] == 1).all()
    assert (lab.area[19:22, 30:33] == 0).all()
    assert not [w for w in lab.warnings if w.startswith("possible")]     # smaller than either block


def test_a_stroke_closes_a_gap_and_is_an_edge():
    img = _page()
    _rect(img, 10, 10, 40, 30)
    img[10:12, 24:28] = 1                 # a 4 px gap in the top side
    marks = dict(street=[(5, 5)], blocks=[(25, 20)])
    with pytest.raises(ValueError, match=r"p__100_200_60_40\.json.*125, 220.*stroke"):
        handlabels.make_labels(img, _crop(**marks))
    lab = handlabels.make_labels(img, _crop(**marks, strokes=[[(22, 10), (30, 10)]]))
    assert lab.kinds == {1: "block"}
    assert (lab.area[13:28, 13:38] == 1).all()
    assert lab.street[5, 5] == 1 and lab.area[5, 5] == 0
    assert (lab.edge[9:12, 22:31] == 1).all()          # the stroke, 3 px wide
    assert (lab.street[9:12, 22:31] == 255).all()


def test_a_limit_stops_the_fill_and_labels_no_edge():
    img = _page()
    _h(img, 10, 0, 30)
    _h(img, 28, 0, 30)                    # a street that ends in open land at x = 30
    without = handlabels.make_labels(img, _crop(street=[(10, 20)]))
    assert (without.street[12:28, 33:] == 1).all()
    lab = handlabels.make_labels(img, _crop(street=[(10, 20)], open=[(45, 20)], limits=[[(31, 10), (31, 29)]]))
    assert (lab.street[12:28, :30] == 1).all()
    assert (lab.street[12:28, 33:] == 0).all()
    assert lab.kinds == {1: "open"}
    assert (lab.edge[9:31, 30:33] == 255).all()        # the limit
    assert (lab.edge[12:28, 29] == 255).all() and (lab.edge[12:28, 33] == 255).all()    # beside it
    assert (lab.edge[12:28, 34] == 0).all()


def test_two_blocks_sharing_a_line_need_a_stroke():
    img = _page()
    _rect(img, 10, 10, 50, 30)
    _v(img, 30, 10, 31)                   # the line the two blocks share
    marks = dict(street=[(5, 5)], blocks=[(20, 20), (40, 20)])
    with pytest.raises(ValueError, match="stroke"):
        handlabels.make_labels(img, _crop(**marks))
    lab = handlabels.make_labels(img, _crop(**marks, strokes=[[(30, 10), (30, 31)]]))
    assert lab.kinds == {1: "block", 2: "block"}
    assert lab.area[20, 20] == 1 and lab.area[20, 40] == 2
    assert (lab.edge[15:25, 29:32] == 1).all()


def test_open_land_is_its_own_kind():
    img = _street_between_two_lines()
    lab = handlabels.make_labels(img, _crop(street=[(10, 20)], blocks=[(20, 4)], open=[(20, 35)]))
    assert lab.kinds == {lab.area[4, 20]: "block", lab.area[35, 20]: "open"}
    assert lab.area[4, 20] == 1 and lab.area[35, 20] == 2


def test_the_crop_border_is_not_an_edge():
    img = _page()
    _h(img, 10, 10)
    _h(img, 29, 10)
    _v(img, 10, 10, 31)                   # a block cut by the crop's east side
    lab = handlabels.make_labels(img, _crop(street=[(5, 5)], blocks=[(40, 20)]))
    assert (lab.area[12:29, 12:] == 1).all()
    assert (lab.edge[:, 59] == 255).all() and (lab.edge[0] == 255).all()
    assert (lab.edge[12:29, 10] == 1).all()            # its printed west side
    assert (lab.edge[10, 11:59] == 1).all()            # and north side
    assert (lab.edge[15:25, 58] == 0).all()            # inside, beside the crop's side


def test_a_seed_must_lie_where_its_kind_can_be():
    img = _street_between_two_lines()
    with pytest.raises(ValueError, match=r"street seed \(100, 210\).*paper"):
        handlabels.make_labels(img, _crop(street=[(0, 10)], blocks=[(20, 4)]))
    with pytest.raises(ValueError, match=r"block seed \(120, 220\).*stroke or limit"):
        handlabels.make_labels(img, _crop(street=[(10, 20)], blocks=[(20, 20)], limits=[[(15, 20), (25, 20)]]))


def test_an_unseeded_block_is_warned_in_page_coordinates():
    img = _page()
    _h(img, 14)
    _h(img, 28)                           # an unseeded block above the street, larger than the seeded one below
    lab = handlabels.make_labels(img, _crop(street=[(10, 20)], blocks=[(20, 35)]))
    possible = [w for w in lab.warnings if w.startswith("possible missed block at ")]
    assert len(possible) == 1
    x, y = (int(v) for v in possible[0].split(" at ")[1].split(", "))
    assert BOX[0] <= x < BOX[0] + 60 and BOX[1] <= y < BOX[1] + 14    # in the unseeded block, page px
    assert (lab.street[:14] == 1).all()                                # which is street until it is seeded
    img[19:22, 30:33] = 0                 # a letter: smaller than the seeded block, no "possible missed block"
    lab = handlabels.make_labels(img, _crop(street=[(10, 20)], blocks=[(20, 35)]))
    assert len([w for w in lab.warnings if w.startswith("possible")]) == 1


def _write(root, set_, sheet, box, **marks):
    d = root / set_ / "s"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{sheet}__{'_'.join(str(v) for v in box)}.json"
    if not marks:                         # a crop needs marks: a street seed in its corner
        marks = {"street": [list(box[:2])]}
    f.write_text(json.dumps({"box": list(box), "notes": "", "drawn_by": "", **marks}))
    return f


def test_crop_files_are_read_and_checked(tmp_path):
    _write(tmp_path, "test", "B", (5, 6, 7, 8), street=[[6, 7]])
    _write(tmp_path, "train", "A", (1, 2, 3, 4))
    found = handlabels.crops(root=tmp_path)
    assert [(c.set, c.series, c.sheet, c.box) for c in found] == [("test", "s", "B", (5, 6, 7, 8)),
                                                                  ("train", "s", "A", (1, 2, 3, 4))]
    assert found[0].street == ((6, 7),)
    assert [c.sheet for c in handlabels.crops(("train",), root=tmp_path)] == ["A"]
    f = _write(tmp_path, "choose", "C", (1, 2, 3, 4))
    f.rename(f.with_name("C__1_2_3_5.json"))
    with pytest.raises(ValueError, match=r"C__1_2_3_5\.json"):
        handlabels.crops(root=tmp_path)


def test_reserved_ground_reaches_other_scans(tmp_path):
    a, b = ("s", "A"), ("s", "B")
    _write(tmp_path, "test", "A", (200, 300, 50, 40))
    _write(tmp_path, "train", "A", (500, 500, 50, 50))
    # B's page shows the same ground 100 px further east: A's px (x, y) is B's px (x + 100, y)
    flats = {a: Flat(4.0), b: Flat(4.0, dx=100)}
    margin = 3
    got = handlabels.reserved(b, (800, 900), margin, root=tmp_path, placements=flats.__getitem__)
    want = np.zeros((800, 900), bool)
    want[300 - margin:340 + margin, 300 - margin:350 + margin] = True
    assert np.array_equal(got, want)
    own = handlabels.reserved(a, (800, 900), 0, root=tmp_path, placements=flats.__getitem__)
    assert own.sum() == 50 * 40 and own[300:340, 200:250].all()


def test_reserved_ground_needs_placed_sheets(tmp_path):
    _write(tmp_path, "choose", "A", (200, 300, 50, 40))

    def only_b(sid):                      # A, the crop's sheet, has no placement file
        if sid != ("s", "B"):
            raise FileNotFoundError(sid)
        return Flat(4.0)
    with pytest.raises(FileNotFoundError, match=r"A__200_300_50_40\.json.*placed"):
        handlabels.reserved(("s", "B"), (800, 900), 3, root=tmp_path, placements=only_b)


def test_the_crops_command_reports_a_bad_crop_and_goes_on(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import config
    from sfzmaps.cli import app
    from sfzmaps.find import pages
    root = tmp_path / "inputs" / "find-labels"
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    r = CliRunner().invoke(app, ["find", "crops"])
    assert r.exit_code == 0 and "no crops" in r.output, r.output
    page = np.ones((300, 300), np.float32)
    page[110:112] = page[128:130] = 0     # a street between two lines, crop rows 10 and 28 of a crop at y0 = 100
    monkeypatch.setattr(pages, "load_scan", lambda sid: page)
    _write(root, "test", "A", (100, 100, 60, 40), street=[[110, 120]], blocks=[[120, 104], [120, 135]])
    _write(root, "train", "A", (100, 100, 60, 40), street=[[110, 110]])          # on the line's ink
    r = CliRunner().invoke(app, ["find", "crops", "--look"])
    assert r.exit_code == 1, r.output
    assert "test/s/A__100_100_60_40.json: block 2, open 0" in r.output
    assert "train/s/A__100_100_60_40.json: FAILED" in r.output and "not on paper" in r.output
    assert (tmp_path / "find" / "look" / "crops" / "test" / "s__A__100_100_60_40.png").is_file()
    assert not (tmp_path / "find" / "look" / "crops" / "train").exists()


def _outlined(polygons=(), open_polygons=()):
    """An outline crop at BOX, its corners given in crop px."""
    return handlabels.Crop("test", "s", "p", BOX, (), (), (), (), (), Path("test/s/p__100_200_60_40.json"),
                           tuple(_pg(p) for p in polygons), tuple(_pg(p) for p in open_polygons))


def test_an_outline_crop_gives_areas_street_and_edges():
    west, east = [(5, 5), (20, 5), (20, 30), (5, 30)], [(30, 5), (45, 5), (45, 30), (30, 30)]
    lab = handlabels.make_labels(_page(), _outlined([west, east], [[(50, 5), (55, 5), (55, 30)]]))
    assert lab.kinds == {1: "block", 2: "block", 3: "open"}
    assert (lab.area[5:31, 5:21] == 1).all() and (lab.area[5:31, 30:46] == 2).all()
    assert lab.area[20, 54] == 3 and lab.area[29, 51] == 0             # the triangle, filled
    assert (lab.street[:, 21:30] == 1).all() and (lab.street[5:31, 5:21] == 0).all()
    for x0, x1 in ((5, 20), (30, 45)):
        ring = np.zeros(lab.edge.shape, bool)
        ring[5:31, x0:x1 + 1] = True
        ring[6:30, x0 + 1:x1] = False
        assert (lab.edge[ring] == 1).all()                              # each rectangle's boundary px
        assert (lab.edge[6:30, x0 + 1:x1] == 0).all()                   # inside
    assert (lab.edge[1:-1, 21:30] == 0).all()                           # the street between
    assert lab.warnings == []


def test_overlapping_outlines_are_refused():
    with pytest.raises(ValueError, match=r"polygons\[0\] and open_polygons\[0\] overlap"):
        handlabels.make_labels(_page(), _outlined([[(5, 5), (20, 5), (20, 30)]], [[(20, 5), (40, 5), (40, 30)]]))


def test_marks_of_both_kinds_are_refused(tmp_path):
    square = [[[105, 205], [120, 205], [120, 220]]]
    _write(tmp_path, "test", "A", BOX, street=[[110, 210]], polygons=square)
    with pytest.raises(ValueError, match=r"A__100_200_60_40\.json: marks of both kinds"):
        handlabels.crops(root=tmp_path)
    _write(tmp_path, "test", "A", BOX, street=[], polygons=[])
    with pytest.raises(ValueError, match=r"A__100_200_60_40\.json: no marks"):
        handlabels.crops(root=tmp_path)
    _write(tmp_path, "test", "A", BOX, polygons=square)
    assert handlabels.crops(root=tmp_path)[0].outlined


def test_an_outline_crop_cannot_train(tmp_path):
    _write(tmp_path, "train", "A", BOX, polygons=[[[105, 205], [120, 205], [120, 220]]])
    with pytest.raises(ValueError, match="only be in choose or test"):
        handlabels.crops(root=tmp_path)


def test_a_block_cut_by_the_crop_side_is_clipped_and_its_cut_side_is_no_edge():
    block = [(10, 10), (61, 10), (61, 30), (10, 30)]                    # 2 px past the last column, 59
    lab = handlabels.make_labels(_page(), _outlined([block]))
    assert (lab.area[10:31, 10:] == 1).all()
    assert (lab.edge[:, 59] == 255).all()
    assert (lab.edge[10:31, 10] == 1).all() and (lab.edge[10, 10:59] == 1).all() and (lab.edge[30, 10:59] == 1).all()
    assert (lab.edge[11:30, 11:59] == 0).all()
    with pytest.raises(ValueError, match=r"polygons\[0\] has a corner \(162, 210\)"):
        handlabels.make_labels(_page(), _outlined([[(10, 10), (62, 10), (62, 30), (10, 30)]]))


def test_the_largest_unseeded_islands_are_always_listed():
    img = _page()
    for x0, x1 in ((5, 15), (22, 36), (42, 56)):                        # three blocks, 11, 15 and 15 px wide
        _rect(img, x0, 10, x1, 30)
    lab = handlabels.make_labels(img, _crop(street=[(2, 2)]))
    assert lab.kinds == {} and (lab.street == 1).all()
    found = [w for w in lab.warnings if w.startswith("unseeded island of ")]
    assert len(found) == len(lab.warnings) == 3
    assert [int(w.split()[3]) for w in found] == [15 * 21, 15 * 21, 11 * 21]      # largest first, sizes in px
    xs = sorted(int(w.split(" at ")[1].split(",")[0]) - BOX[0] for w in found)
    assert 5 < xs[0] < 15 and 22 < xs[1] < 36 and 42 < xs[2] < 56
    assert all(w.endswith("a block left unseeded is labelled street") for w in found)


def test_bad_crop_files_are_named_and_the_command_goes_on(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import config
    from sfzmaps.cli import app
    from sfzmaps.find import pages
    root = tmp_path / "inputs" / "find-labels"
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    page = np.ones((300, 300), np.float32)
    page[110:112] = page[128:130] = 0     # a street between two lines
    monkeypatch.setattr(pages, "load_scan", lambda sid: page)
    _write(root, "test", "B", (100, 100, 60, 40), street=[[110, 120]])
    broken = _write(root, "test", "A", (100, 100, 60, 40), street=[[110, 120]])
    broken.write_text('{"box": [100, 100, 60, 40], "street": [[110, 120]')         # a JSON syntax error
    misnamed = _write(root, "test", "C", (100, 100, 60, 40), street=[[110, 120]])
    misnamed = misnamed.rename(misnamed.with_name("C__100_100_60_41.json"))
    with pytest.raises(ValueError, match=r"A__100_100_60_40\.json: not JSON"):
        handlabels.crops(root=root)
    r = CliRunner().invoke(app, ["find", "crops", "--look"])
    assert r.exit_code == 1, r.output
    assert f"{broken}: not JSON" in r.output and f"{misnamed}: its name says box" in r.output
    assert "test/s/B__100_100_60_40.json: block 0, open 0" in r.output
    assert (tmp_path / "find" / "look" / "crops" / "test" / "s__B__100_100_60_40.png").is_file()


class _Blind:
    """A placement that cannot place ground east of x_ft (to_pixel gives NaN there)."""

    def __init__(self, flat, x_ft):
        self.flat, self.x_ft = flat, x_ft

    def to_world(self, px):
        return self.flat.to_world(px)

    def to_pixel(self, world):
        out = self.flat.to_pixel(world)
        out[np.asarray(world, float).reshape(-1, 2)[:, 0] > self.x_ft] = np.nan
        return out


def test_reserved_ground_the_placement_cannot_place_in_part_is_warned(tmp_path, caplog):
    import logging

    from sfzmaps.solver import grid
    _write(tmp_path, "test", "A", (200, 300, 80, 40))
    a, b = Flat(4.0), Flat(4.0, dx=100)                # A's px (x, y) is B's (x + 100, y)
    blind = _Blind(b, grid.X0 + 4.0 * 240)             # B cannot place A's px east of x = 240 (B's 340)
    flats = {("s", "A"): a, ("s", "B"): blind}
    with caplog.at_level(logging.WARNING, logger="sfzmaps.find.handlabels"):
        got = handlabels.reserved(("s", "B"), (800, 900), 0, root=tmp_path, placements=flats.__getitem__)
    assert got[305:335, 305:335].all()                 # the part B can place
    assert not got[:, 350:].any() and not got[:, :300].any()
    # 30 points round the 80 x 40 box every 8 px; those east of A's x = 240: 4 on the top, 5 on the east side, 5 on
    # the bottom
    assert len(caplog.records) == 1 and "14 of 30 points" in caplog.records[0].getMessage()
    assert "A__200_300_80_40.json" in caplog.records[0].getMessage() and "s/B" in caplog.records[0].getMessage()
    caplog.clear()
    flats[("s", "B")] = _Blind(b, grid.X0)             # places none of it: the crop's ground is not on B
    with caplog.at_level(logging.WARNING, logger="sfzmaps.find.handlabels"):
        got = handlabels.reserved(("s", "B"), (800, 900), 3, root=tmp_path, placements=flats.__getitem__)
    assert not got.any() and caplog.records == []


def test_a_ruled_tile_shows_its_part_of_the_box():
    from sfzmaps.find import look
    lum = np.ones((100, 100), np.float32)
    lum[70, 80] = 0                                     # page px (1080, 2070)
    tiles = look.ruled_tiles(lum, (1000, 2000, 100, 100), 4, 2)
    top, left = look.RULER
    t = tiles[(1, 1)]                                   # page px 1050..1099, 2050..2099
    assert t.shape == (top + 50 * 4, left + 50 * 4, 3)
    assert (t[top + 20 * 4:top + 21 * 4, left + 30 * 4:left + 31 * 4] == 0).all()
    assert (t[top:, left:] == 0).all(axis=2).sum() == 16          # only that px is black
    assert all((tiles[k][top:, left:] > 0).all() for k in ((0, 0), (0, 1), (1, 0)))


def _block_with_a_dead_end_road():
    """A block (outer px x 10..50, y 5..35) holding a dead-end road up from its bottom side: side lines at x 26-27
    and 34-35, its end at y 15-16, its paper x 28..33, y 17..33; the block's bottom line (y 34-35) runs across its
    mouth."""
    img = _page()
    _rect(img, 10, 5, 50, 35)
    _v(img, 26, 15, 36)
    _v(img, 34, 15, 36)
    _h(img, 15, 26, 36)
    return img


MOUTH = [(29, 35), (32, 35)]              # drawn 3 px wide: x 28..33, y 34..36, between the road's side lines


def test_an_opening_lets_the_street_through_a_line():
    img = _block_with_a_dead_end_road()
    marks = dict(street=[(5, 38)], blocks=[(15, 20)])
    shut = handlabels.make_labels(img, _crop(**marks))
    assert (shut.street[17:34, 28:34] == 0).all() and (shut.area[17:34, 28:34] == 1).all()
    lab = handlabels.make_labels(img, _crop(**marks, openings=[MOUTH]))
    assert (lab.street[17:34, 28:34] == 1).all()                       # the road, joined to the street below
    assert (lab.street[34:37, 28:34] == 1).all()                       # the opening itself
    assert lab.kinds == {1: "block"} and lab.area[20, 15] == lab.area[20, 45] == 1
    assert (lab.edge[18:33, 27] == 1).all() and (lab.edge[18:33, 34] == 1).all()     # the road's sides


def test_a_stroke_over_an_opening_wins():
    img = _block_with_a_dead_end_road()
    lab = handlabels.make_labels(img, _crop(street=[(5, 38)], blocks=[(15, 20)], openings=[MOUTH], strokes=[MOUTH]))
    assert (lab.street[17:34, 28:34] == 0).all()
    assert (lab.street[34:37, 28:34] == 255).all()                     # the stroke


def test_an_outline_crop_cannot_carry_openings(tmp_path):
    _write(tmp_path, "test", "A", BOX, polygons=[[[105, 205], [120, 205], [120, 220]]],
           openings=[[[110, 210], [115, 210]]])
    with pytest.raises(ValueError, match=r"A__100_200_60_40\.json: marks of both kinds"):
        handlabels.crops(root=tmp_path)


def _at(box, street=(), blocks=(), strokes=()):
    """A seed crop at `box`, its marks given in crop px."""
    pg = lambda pts: tuple((x + box[0], y + box[1]) for x, y in pts)                 # noqa: E731
    return handlabels.Crop("train", "s", "p", tuple(box), pg(street), pg(blocks), (), tuple(pg(s) for s in strokes),
                           (), Path(f"train/s/p__{'_'.join(str(v) for v in box)}.json"))


BIG = (100, 200, 100, 80)                 # w 100, h 80


def _big_block():
    """A block whose 2 px outline's outer px run x 10..90, y 10..70."""
    img = np.ones((80, 100), np.float32)
    _rect(img, 10, 10, 90, 70)
    return img


def _cell_behind_a_pinhole(img):
    """A 6 x 6 cell (x 30..35, y 12..17) just inside `_big_block`'s top side, walled by hatching, open to the street
    by a 1 px hole in the outline at x 32."""
    _h(img, 18, 28, 38)
    _v(img, 28, 12, 20)
    _v(img, 36, 12, 20)
    img[10:12, 32] = 1


def _alley(img, x0, depth, width=6):
    """A dead end `width` px wide (paper x0..x0 + width - 1) up from `_big_block`'s bottom side, open to the street
    at its full width, its paper rows 71 - depth..70, walled by 2 px lines."""
    top = 71 - depth
    img[top:71, x0:x0 + width] = 1
    _v(img, x0 - 2, top - 2, 71)
    _v(img, x0 + width, top - 2, 71)
    _h(img, top - 2, x0 - 2, x0 + width + 2)


def test_a_cell_behind_a_pinhole_joins_its_block_and_an_alley_stays():
    img = _big_block()
    _cell_behind_a_pinhole(img)
    _alley(img, 60, 41)
    crop = _at(BIG, street=[(5, 5)], blocks=[(20, 40)])
    before = handlabels.make_labels(img, crop, smooth=False)
    assert (before.street[12:18, 30:36] == 1).all()
    lab = handlabels.make_labels(img, crop)
    assert (lab.area[12:18, 30:36] == 1).all() and (lab.area[10:12, 32] == 1).all()
    assert (lab.edge[10, 12:89] == 1).all()                            # straight along the outline, across the hole
    assert (lab.edge[12:18, 30:36] == 0).all()
    assert (lab.street[30:80, 60:66] == 1).all()                       # every px of the alley


def test_a_whole_strip_behind_a_pinhole_joins_its_block():
    img = np.ones((140, 60), np.float32)
    _rect(img, 5, 10, 54, 130)
    _v(img, 18, 10, 131)
    _v(img, 40, 10, 131)                  # two hatch lines: a strip x 20..39 (20 px), y 12..128
    img[10:12, 28:30] = 1                 # a 2 px gap in the outline into it
    crop = _at((0, 0, 60, 140), street=[(2, 2)], blocks=[(10, 70)])
    assert (handlabels.make_labels(img, crop, smooth=False).street[12:129, 20:40] == 1).all()
    lab = handlabels.make_labels(img, crop)
    assert (lab.area[10:129, 20:40] == 1).all() and lab.kinds == {1: "block"}


def test_a_pocket_with_a_street_seed_stays_street():
    img = _big_block()
    _alley(img, 40, 30, width=2)          # a dead end 2 px wide (x 40..41, y 41..70): narrower than PINHOLE
    unmarked = handlabels.make_labels(img, _at(BIG, street=[(5, 5)], blocks=[(20, 40)]))
    assert (unmarked.area[41:71, 40:42] == 1).all()                    # a pocket in one block: it joins it
    lab = handlabels.make_labels(img, _at(BIG, street=[(5, 5), (40, 55)], blocks=[(20, 40)]))
    assert (lab.street[41:71, 40:42] == 1).all()                       # marked with a street seed: it stays


def test_a_narrow_lane_between_two_blocks_stays_street():
    img = np.ones((60, 90), np.float32)
    _rect(img, 5, 10, 42, 50)
    _rect(img, 45, 10, 80, 50)            # a lane 2 px wide, x 43..44, open to the street at both ends
    crop = _at((0, 0, 90, 60), street=[(2, 2)], blocks=[(20, 30), (60, 30)])
    lab = handlabels.make_labels(img, crop)
    assert (lab.street[:, 43:45] == 1).all()
    assert (lab.street == handlabels.make_labels(img, crop, smooth=False).street).all()


def test_inner_corners_are_not_touched():
    img = np.ones((100, 120), np.float32)
    _rect(img, 5, 5, 50, 90)              # an L: a rectangle with its top-right quarter cut away
    img[5:46, 31:51] = 1
    _h(img, 46, 29, 51)
    _v(img, 29, 5, 48)
    _rect(img, 60, 5, 110, 90)            # a U: a wide dead end (x 75..94) down into it from the top
    img[5:7, 75:95] = 1
    _v(img, 73, 5, 61)
    _v(img, 95, 5, 61)
    _h(img, 60, 73, 97)
    crop = _at((0, 0, 120, 100), street=[(2, 2), (85, 20)], blocks=[(15, 70), (65, 80)])
    a, b = handlabels.make_labels(img, crop, smooth=False), handlabels.make_labels(img, crop)
    for name in ("street", "area", "edge"):
        assert (getattr(a, name) == getattr(b, name)).all(), name
    assert a.kinds == b.kinds == {1: "block", 2: "block"}


def test_the_largest_pockets_are_warned_of():
    img = np.ones((80, 200), np.float32)
    _rect(img, 5, 10, 195, 70)
    sizes = []
    for i, w in enumerate((4, 6, 8, 10, 12, 14)):                     # six cells, each behind a 1 px pin-hole
        x0 = 12 + 30 * i
        _h(img, 22, x0 - 2, x0 + w + 2)
        _v(img, x0 - 2, 12, 24)
        _v(img, x0 + w, 12, 24)
        img[10:12, x0 + 1] = 1
        sizes.append(w * 10 + 2)                                       # the cell (y 12..21) and its hole
    crop = _at((0, 0, 200, 80), street=[(2, 2)], blocks=[(100, 50)])
    lab = handlabels.make_labels(img, crop)
    found = [w for w in lab.warnings if w.startswith("pocket of ")]
    assert [int(w.split()[2]) for w in found] == sorted(sizes, reverse=True)[:5]
    assert all("joined its block" in w and "street seed" in w for w in found)
    x, y = (int(v) for v in found[0].split(" at ")[1].split(" joined")[0].split(", "))
    assert 12 + 150 <= x < 12 + 150 + 14 and 10 <= y < 22              # inside the largest cell, page px
    assert not [w for w in handlabels.make_labels(img, crop, smooth=False).warnings if w.startswith("pocket")]


def test_smoothing_changes_only_street_px_into_area_px():
    rng = np.random.default_rng(7)
    img = np.ones((120, 120), np.float32)
    seeds = []
    for x0, y0 in ((10, 10), (65, 10), (10, 65), (65, 65)):
        x1, y1 = x0 + 45, y0 + 45
        for v in range(x0 + 7, x1 - 2, 6):                             # cross-hatching: 5 x 5 px cells
            img[y0:y1 + 1, v] = 0
        for v in range(y0 + 7, y1 - 2, 6):
            img[v, x0:x1 + 1] = 0
        _rect(img, x0, y0, x1, y1)
        for _ in range(12):                                            # pin-holes through the outline
            t = int(rng.integers(3, 43))
            side = int(rng.integers(4))
            if side == 0:
                img[y0:y0 + 2, x0 + t] = 1
            elif side == 1:
                img[y1 - 1:y1 + 1, x0 + t] = 1
            elif side == 2:
                img[y0 + t, x0:x0 + 2] = 1
            else:
                img[y0 + t, x1 - 1:x1 + 1] = 1
        seeds.append((x0 + 7, y0 + 7))                                 # a hatching crossing: ink, never street
    for _ in range(40):                                                # specks in the streets
        x, y = (int(v) for v in rng.integers(0, 120, 2))
        if img[max(0, y - 2):y + 3, max(0, x - 2):x + 3].all():
            img[y, x] = 0
    crop = _at((0, 0, 120, 120), street=[(1, 1)], blocks=seeds)
    a, b = handlabels.make_labels(img, crop, smooth=False), handlabels.make_labels(img, crop)
    moved = a.area != b.area
    assert moved.any()
    assert (a.area[moved] == 0).all() and (b.area[moved] > 0).all()   # street px into an area, never the reverse
    assert (a.street[moved] == 1).all() and (b.street[moved] == 0).all()
    assert not ((a.street != b.street) & ~moved).any()
    assert a.kinds == b.kinds and b.warnings[:len(a.warnings)] == a.warnings
