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


def _crop(street=(), blocks=(), open=(), strokes=(), limits=()):
    """A crop at BOX, every mark given in crop px."""
    return handlabels.Crop("train", "s", "p", BOX, _pg(street), _pg(blocks), _pg(open),
                           tuple(_pg(s) for s in strokes), tuple(_pg(s) for s in limits),
                           Path("train/s/p__100_200_60_40.json"))


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


def _at(box, street=(), blocks=(), strokes=()):
    """A seed crop at `box`, its marks given in crop px."""
    pg = lambda pts: tuple((x + box[0], y + box[1]) for x, y in pts)                 # noqa: E731
    return handlabels.Crop("train", "s", "p", tuple(box), pg(street), pg(blocks), (), tuple(pg(s) for s in strokes),
                           (), Path(f"train/s/p__{'_'.join(str(v) for v in box)}.json"))


def _toothed(img, x0, y0, x1, y1, rng, keep=5):
    """A solid black block x0..x1, y0..y1 (inclusive) whose sides carry 1 and 2 px teeth (ink out) and pits (paper
    in), a single px wide, as many teeth as pits of each depth, none within `keep` px of a corner or beside another."""
    img[y0:y1 + 1, x0:x1 + 1] = 0
    for lo, hi, side in ((x0, x1, "top"), (x0, x1, "bottom"), (y0, y1, "left"), (y0, y1, "right")):
        at = rng.permutation(np.arange(lo + keep, hi - keep + 1, 3))[:12]
        for t, d in zip(at, (1, 1, 1, 2, 2, 2, -1, -1, -1, -2, -2, -2)):
            t = int(t)
            if side == "top":
                img[y0 - d:y0, t] = 0 if d > 0 else 1
                if d < 0:
                    img[y0:y0 - d, t] = 1
            elif side == "bottom":
                if d > 0:
                    img[y1 + 1:y1 + 1 + d, t] = 0
                else:
                    img[y1 + d + 1:y1 + 1, t] = 1
            elif side == "left":
                if d > 0:
                    img[t, x0 - d:x0] = 0
                else:
                    img[t, x0:x0 - d] = 1
            else:
                if d > 0:
                    img[t, x1 + 1:x1 + 1 + d] = 0
                else:
                    img[t, x1 + d + 1:x1 + 1] = 1


def test_teeth_along_a_side_are_straightened():
    img = np.ones((80, 100), np.float32)
    _toothed(img, 15, 15, 84, 64, np.random.default_rng(3))
    crop = _at((0, 0, 100, 80), street=[(2, 2)], blocks=[(50, 40)])
    ragged = handlabels.make_labels(img, crop, straighten=False)
    rect = np.zeros((80, 100), bool)
    rect[15:65, 15:85] = True
    assert ((ragged.area == 1) != rect).sum() > 50                    # the teeth are there
    lab = handlabels.make_labels(img, crop)
    assert ((lab.area == 1) == rect).all()                             # the rectangle, its sides where they are
    for sl in (np.s_[:, 20:80], np.s_[20:60, :]):                       # each side's edge px on one row or column
        ys, xs = np.nonzero(lab.edge[sl] == 1)
        v = ys if sl[0] == slice(None) else xs
        lo, hi = v[v < v.mean()], v[v >= v.mean()]
        assert np.ptp(lo) == 0 and np.ptp(hi) == 0
    gained = int(((ragged.area == 0) & (lab.area == 1)).sum())
    lost = int(((ragged.area == 1) & (lab.area == 0)).sum())
    assert abs(gained - lost) < (gained + lost) / 3                    # not sitting inside the ragged edge


def test_a_clean_rectangle_and_an_l_are_unchanged():
    img = np.ones((80, 100), np.float32)
    img[10:40, 10:50] = 0                                              # a rectangle
    img[10:70, 60:90] = 0
    img[10:40, 75:90] = 1                                              # an L
    crop = _at((0, 0, 100, 80), street=[(2, 2)], blocks=[(20, 20), (65, 60)])
    a, b = handlabels.make_labels(img, crop, straighten=False), handlabels.make_labels(img, crop)
    assert (a.area == b.area).all() and (a.edge == b.edge).all()


def _block_with_alleys():
    """A solid black block (x 10..89, y 10..69) with two dead ends up from its bottom side, 40 px deep: one 4 px wide
    (x 30..33), one 3 px wide (x 60..62)."""
    img = np.ones((80, 100), np.float32)
    img[10:70, 10:90] = 0
    img[30:70, 30:34] = 1
    img[30:70, 60:63] = 1
    return img


def test_alleys_survive_straightening():
    img = _block_with_alleys()
    crop = _at((0, 0, 100, 80), street=[(2, 2)], blocks=[(15, 40)])
    a, b = handlabels.make_labels(img, crop, straighten=False), handlabels.make_labels(img, crop)
    assert (b.street[30:70, 30:34] == 1).all() and (b.street[30:70, 60:63] == 1).all()     # every alley px
    assert (a.area == b.area).all()                                    # and the walls where they were, to the px


def test_a_bite_is_not_filled():
    img = np.ones((80, 100), np.float32)
    img[10:70, 10:90] = 0
    img[10:17, 40:48] = 1                 # a bite 8 px wide, 7 deep, in the top side: this is no bite filler
    crop = _at((0, 0, 100, 80), street=[(2, 2)], blocks=[(50, 40)])
    assert (handlabels.make_labels(img, crop).street[10:17, 40:48] == 1).all()


def test_straightening_takes_nothing_from_another_area_or_a_barrier():
    rng = np.random.default_rng(5)
    img = np.ones((80, 100), np.float32)
    img[10:70, 10:45] = 0
    img[10:70, 49:90] = 0                 # two blocks 3 px apart (x 46..48), their facing sides ragged by 1 px
    for y in range(15, 65):
        if rng.random() < 0.4:
            img[y, 45] = 0
        if rng.random() < 0.4:
            img[y, 48] = 0
    stroke = [(30, 72), (60, 72)]
    crop = _at((0, 0, 100, 80), street=[(2, 2)], blocks=[(20, 40), (70, 40)], strokes=[stroke])
    a, b = handlabels.make_labels(img, crop, straighten=False), handlabels.make_labels(img, crop)
    assert ((b.street[10:70, 45:49] == 1).sum(axis=1) >= 1).all()      # the street between keeps a px on every row
    assert not ((a.area > 0) & (b.area > 0) & (a.area != b.area)).any()
    assert (b.street[a.street == 255] == 255).all() and (a.street[b.street == 255] == 255).all()


def test_a_curved_side_moves_no_further_than_the_setting():
    img = np.ones((80, 80), np.float32)
    yy, xx = np.mgrid[:80, :80]
    img[(yy - 40) ** 2 + (xx - 40) ** 2 <= 25 ** 2] = 0
    crop = _at((0, 0, 80, 80), street=[(2, 2)], blocks=[(40, 40)])
    a, b = handlabels.make_labels(img, crop, straighten=False), handlabels.make_labels(img, crop)
    changed = (a.area == 1) != (b.area == 1)
    assert changed.any()
    inside = a.area == 1
    d = handlabels.ndimage.distance_transform_edt(~(inside ^ handlabels.ndimage.binary_erosion(inside)))
    assert d[changed].max() <= handlabels.STRAIGHT + 1


def test_thin_areas_are_kept_as_the_fill_made_them():
    img = np.ones((60, 100), np.float32)
    img[10:50, 0] = 0                     # a strip 1 px wide along the crop's side
    img[10:50, 20:22] = 0                 # 2 px wide
    img[10:50, 40:43] = 0                 # 3 px wide
    crop = _at((0, 0, 100, 60), street=[(60, 5)], blocks=[(0, 30), (20, 30), (41, 30)])
    a, b = handlabels.make_labels(img, crop, straighten=False), handlabels.make_labels(img, crop)
    assert (a.area == b.area).all()
    assert b.kinds == {1: "block", 2: "block", 3: "block"} and all((b.area == k).any() for k in b.kinds)


def test_a_spur_thins_and_the_block_is_straightened():
    img = np.ones((60, 80), np.float32)
    _toothed(img, 15, 20, 64, 49, np.random.default_rng(11))
    img[8:20, 40:42] = 0                  # a spur 2 px wide and 12 long on the top side
    crop = _at((0, 0, 80, 60), street=[(2, 2)], blocks=[(30, 35)])
    lab = handlabels.make_labels(img, crop)
    rect = np.zeros((60, 80), bool)
    rect[20:50, 15:65] = True
    spur = np.zeros((60, 80), bool)
    spur[8:20, 40:42] = True
    assert ((lab.area == 1) == rect)[~spur].all()                      # the block straightened, its teeth gone
    left = (lab.area == 1) & spur         # the spur, no wider than the tolerance, is thinned to a wedge (1 px at its
    assert left.any() and left.sum() < spur.sum() and left[8:12].sum(axis=1).max() == 1      # tip), not kept whole


def test_a_choosing_crop_on_test_ground_is_found_on_another_sheet(tmp_path):
    a, b = ("s", "A"), ("s", "B")
    # B's page shows the same ground 100 px further east: A's px (x, y) is B's px (x + 100, y)
    flats = {a: Flat(4.0), b: Flat(4.0, dx=100)}
    _write(tmp_path, "test", "B", (300, 300, 50, 40))                  # A's ground x 200-249
    on = handlabels.read_crop(_write(tmp_path, "choose", "A", (225, 310, 50, 20)), "choose")      # half on it
    beside = handlabels.read_crop(_write(tmp_path, "choose", "A", (250, 300, 50, 40)), "choose")  # touching only
    test = handlabels.crops(("test",), root=tmp_path)
    got = handlabels.choose_on_test([on, beside], test, flats.__getitem__)
    assert [(c.box, t.sheet) for c, t, _ in got] == [((225, 310, 50, 20), "B")]
    assert got[0][2] == pytest.approx(0.5)
    with pytest.raises(ValueError, match=r"A__225_310_50_20\.json.*B__300_300_50_40\.json \(0\.500 of"):
        handlabels.refuse_choose_on_test([on, beside], test, flats.__getitem__)
    handlabels.refuse_choose_on_test([beside], test, flats.__getitem__)          # nothing to refuse


def test_the_trace_scorer_and_count_refuse_a_choosing_crop_on_test_ground(tmp_path):
    from sfzmaps.find import counts
    flats = {("s", "A"): Flat(4.0)}
    choose = [handlabels.read_crop(_write(tmp_path, "choose", "A", (100, 100, 20, 20)), "choose")]
    test = [handlabels.read_crop(_write(tmp_path, "test", "A", (110, 100, 20, 20)), "test")]
    with pytest.raises(ValueError, match=r"choose/s/A__100_100_20_20\.json.*test/s/A__110_100_20_20\.json"):
        counts.trace_scorer(choose, "cpu", test, flats.__getitem__)
    with pytest.raises(ValueError, match=r"0\.500 of the choosing crop's area"):
        counts.count({}, choose, test, "cpu", tmp_path / "run", placements=flats.__getitem__)


def test_the_crops_command_prints_a_choosing_crop_on_test_ground(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from sfzmaps import config
    from sfzmaps.cli import app
    from sfzmaps.find import lotlabels, pages
    root = tmp_path / "inputs" / "find-labels"
    monkeypatch.setattr(config, "INPUTS_DIR", tmp_path / "inputs")
    page = np.ones((300, 300), np.float32)
    page[::10] = 0                        # lines every 10 rows: each crop has ink and paper
    monkeypatch.setattr(pages, "load_scan", lambda sid: page)
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(4.0))
    _write(root, "test", "A", (100, 100, 40, 40), street=[[101, 105]])
    _write(root, "choose", "A", (200, 100, 40, 40), street=[[201, 105]])
    r = CliRunner().invoke(app, ["find", "crops"])
    assert r.exit_code == 0 and "overlap" not in r.output, r.output
    _write(root, "choose", "A", (130, 135, 40, 40), street=[[131, 145]])
    r = CliRunner().invoke(app, ["find", "crops", "--set", "train"])
    assert r.exit_code == 1, r.output
    assert ("choose-against-test ground overlap: choose/s/A__130_135_40_40.json lies on the ground of the test crop "
            "test/s/A__100_100_40_40.json (0.031 of the choosing crop's area)") in r.output, r.output


def test_a_choosing_crop_the_other_placement_cannot_place_in_part_is_warned(tmp_path, caplog):
    import logging

    from sfzmaps.solver import grid
    a = handlabels.read_crop(_write(tmp_path, "choose", "A", (200, 300, 80, 40)), "choose")
    b = handlabels.read_crop(_write(tmp_path, "test", "B", (300, 300, 50, 40)), "test")
    blind = _Blind(Flat(4.0, dx=100), grid.X0 + 4.0 * 240)    # B cannot place A's px east of x = 240
    with caplog.at_level(logging.WARNING, logger="sfzmaps.find.handlabels"):
        share = handlabels.ground_share(a, b, Flat(4.0), blind)
    assert share > 0                                          # the part B can place (A's x 200-240) lies on b's box
    [rec] = caplog.records
    msg = rec.getMessage()
    assert "14 of 30 points" in msg and "A__200_300_80_40.json" in msg and "s/B" in msg
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="sfzmaps.find.handlabels"):  # places none: not on B, as `reserved`
        assert handlabels.ground_share(a, b, Flat(4.0), _Blind(Flat(4.0, dx=100), grid.X0)) == 0.0
    assert caplog.records == []
