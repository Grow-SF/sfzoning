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
    assert lab.warnings == []             # smaller than either block: no warning


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
    assert len(lab.warnings) == 1 and lab.warnings[0].startswith("possible missed block at ")
    x, y = (int(v) for v in lab.warnings[0].split(" at ")[1].split(", "))
    assert BOX[0] <= x < BOX[0] + 60 and BOX[1] <= y < BOX[1] + 14    # in the unseeded block, page px
    assert (lab.street[:14] == 1).all()                                # which is street until it is seeded
    img[19:22, 30:33] = 0                 # a letter: smaller than the seeded block, no warning
    assert len(handlabels.make_labels(img, _crop(street=[(10, 20)], blocks=[(20, 35)])).warnings) == 1


def _write(root, set_, sheet, box, **marks):
    d = root / set_ / "s"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{sheet}__{'_'.join(str(v) for v in box)}.json"
    f.write_text(json.dumps({"box": list(box), "street": [], "blocks": [], "open": [], "strokes": [], "limits": [],
                             "notes": "", "drawn_by": "", **marks}))
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
