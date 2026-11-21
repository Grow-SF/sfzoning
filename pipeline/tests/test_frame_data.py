import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sfzmaps.frame import data
from sfzmaps.frame.data import (
    inside_mask,
    labelled_sheets,
    load_label,
    load_luminance,
    load_remove,
    map_counts,
    map_mask,
    map_pages,
    page_path,
)


def test_load_luminance_rgb_and_grey(tmp_path):
    Image.new("L", (3, 2), 51).save(tmp_path / "g.png")
    Image.new("RGB", (3, 2), (255, 0, 0)).save(tmp_path / "c.png")
    assert np.allclose(load_luminance(tmp_path / "g.png"), 0.2)
    assert np.allclose(load_luminance(tmp_path / "c.png"), 76 / 255, atol=1 / 255)


def test_map_mask_notch_and_two_pieces():
    notch = [(0, 0), (6, 0), (6, 2), (10, 2), (10, 10), (0, 10)]
    m = map_mask([notch, [(12, 0), (14, 0), (14, 2), (12, 2)]], (11, 16))
    assert m[1, 8] == 0 and m[5, 8] == 1 and m[1, 3] == 1 and m[1, 13] == 1 and m[5, 13] == 0
    assert map_mask([notch], (6, 8), scale=0.5)[2, 2] == 1


def test_labelled_sheets_and_load_label(tmp_path):
    (tmp_path / "1921").mkdir()
    (tmp_path / "1921" / "a.json").write_text(
        '{"polygons": [[[0,0],[4,0],[4,4]]], "notes": "", "drawn_by": ""}'
    )
    (tmp_path / "README.md").write_text("not a label")
    assert labelled_sheets(tmp_path) == [("1921", "a")]
    assert load_label("1921", "a", tmp_path) == [[(0, 0), (4, 0), (4, 4)]]
    assert load_remove("1921", "a", tmp_path) == []


def test_load_remove_reads_the_removed_blocks(tmp_path):
    (tmp_path / "1948").mkdir()
    (tmp_path / "1948" / "c.json").write_text(
        '{"polygons": [[[0,0],[9,0],[9,9],[0,9]]], "remove": [[[2,2],[4,2],[4,4],[2,4]]], "notes": ""}'
    )
    assert load_label("1948", "c", tmp_path) == [[(0, 0), (9, 0), (9, 9), (0, 9)]]
    assert load_remove("1948", "c", tmp_path) == [[(2, 2), (4, 2), (4, 4), (2, 4)]]


def test_load_label_refuses_holes(tmp_path):
    (tmp_path / "2000").mkdir()
    (tmp_path / "2000" / "b.json").write_text(
        '{"polygons": [[[0,0],[9,0],[9,9],[0,9]]], "holes": [[[2,2],[4,2],[4,4],[2,4]]], "notes": ""}'
    )
    with pytest.raises(ValueError, match="b.json"):
        load_label("2000", "b", tmp_path)


REPO_LABELS = Path(__file__).parents[1] / "inputs" / "frame-labels"


def test_no_label_has_holes():
    sheets = data.labelled_sheets(REPO_LABELS)
    assert sheets
    for series, sheet in sheets:
        label = json.loads((REPO_LABELS / series / f"{sheet}.json").read_text())
        assert "holes" not in label, (series, sheet)
        for block in label.get("remove", []):
            assert len(block) >= 3, (series, sheet)


def test_map_mask_clears_holes():
    square = [(0, 0), (19, 0), (19, 19), (0, 19)]
    hole = [(5, 5), (10, 5), (10, 10), (5, 10)]
    m = map_mask([square], (20, 20), holes=[hole])
    assert m[7, 7] == 0 and m[5, 5] == 0 and m[10, 10] == 0     # the hole, its outline included, is not map
    assert m[2, 2] == 1 and m[15, 15] == 1 and m[7, 12] == 1
    assert int(map_mask([square], (20, 20)).sum()) - int(m.sum()) == 36   # exactly the hole's 6 x 6 px are cleared
    small = map_mask([square], (10, 10), scale=0.5, holes=[hole])        # holes scale with the polygons
    assert small[3, 3] == 0 and small[1, 1] == 1 and small[7, 7] == 1


def _pages(work, *sheets):
    for s, h in sheets:
        d = work / s / h
        d.mkdir(parents=True)
        Image.new("L", (2, 2), 255).save(d / "page.png")
    return work


def test_map_pages_unknown_sheet_raises(tmp_path):
    work = _pages(tmp_path / "work", ("1921", "a"), ("1921", "b"))
    csv_path = tmp_path / "kinds.csv"
    csv_path.write_text("series,sheet,kind,checked\n1921,a,map,owner\n")
    with pytest.raises(KeyError, match="1921/b"):
        map_pages(csv_path, work)


def test_map_pages_keeps_only_maps(tmp_path):
    work = _pages(tmp_path / "work", ("1921", "a"), ("1921", "b"))
    csv_path = tmp_path / "kinds.csv"
    csv_path.write_text('series,sheet,kind,checked\n1921,a,map,"x, y"\n1921,b,text,o\n')
    assert map_pages(csv_path, work) == [("1921", "a")]
    assert page_path("1921", "a", work) == work / "1921" / "a" / "page.png"


def test_map_counts_blank_means_one(tmp_path):
    p = tmp_path / "k.csv"
    p.write_text("series,sheet,kind,checked,maps\nA,1,map,x,\nA,2,map,x,2\nA,3,text,x,\n")
    assert map_counts(p) == {("A", "1"): 1, ("A", "2"): 2, ("A", "3"): 1}


def test_inside_mask_is_map_mask_equal_one():
    m = inside_mask([[(2, 2), (8, 2), (8, 8), (2, 8)]], (12, 12))
    assert m.dtype == bool and m[5, 5] and m[2, 2] and not m[0, 0]


def test_class_mask_remove_wins():
    m = data.class_mask([[(0, 0), (9, 0), (9, 9), (0, 9)]], [[(3, 3), (6, 3), (6, 6), (3, 6)]], (20, 20))
    assert m[1, 1] == 0 and m[4, 4] == 2 and m[15, 15] == 1


def test_class_mask_includes_outlines_and_is_uint8():
    m = data.class_mask([[(0, 0), (9, 0), (9, 9), (0, 9)]], [[(3, 3), (6, 3), (6, 6), (3, 6)]], (20, 20))
    assert m.dtype == np.uint8 and m.shape == (20, 20)
    assert m[9, 9] == 0 and m[10, 10] == 1 and m[3, 3] == 2 and m[6, 6] == 2 and m[7, 7] == 0
    assert (m == 2).sum() == 16 and (m == 0).sum() == 100 - 16
