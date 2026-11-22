import pytest
from PIL import Image

from sfzmaps.frame import data
from sfzmaps.frame.data import labelled_sheets, load_label, load_remove, map_counts, map_pages, page_path


def test_labelled_sheets_and_load_label(tmp_path):
    (tmp_path / "1921").mkdir()
    (tmp_path / "1921" / "a.json").write_text(
        '{"polygons": [[[0,0],[4,0],[4,4]]], "notes": "", "drawn_by": ""}'
    )
    (tmp_path / "1948").mkdir()
    (tmp_path / "1948" / "c.json").write_text(
        '{"polygons": [[[0,0],[9,0],[9,9],[0,9]]], "remove": [[[2,2],[4,2],[4,4],[2,4]]], "notes": ""}'
    )
    (tmp_path / "README.md").write_text("not a label")
    assert labelled_sheets(tmp_path) == [("1921", "a"), ("1948", "c")]
    assert load_label("1921", "a", tmp_path) == [[(0, 0), (4, 0), (4, 4)]]
    assert load_remove("1921", "a", tmp_path) == []
    assert load_label("1948", "c", tmp_path) == [[(0, 0), (9, 0), (9, 9), (0, 9)]]
    assert load_remove("1948", "c", tmp_path) == [[(2, 2), (4, 2), (4, 4), (2, 4)]]


def test_load_label_refuses_holes(tmp_path):
    (tmp_path / "2000").mkdir()
    (tmp_path / "2000" / "b.json").write_text(
        '{"polygons": [[[0,0],[9,0],[9,9],[0,9]]], "holes": [[[2,2],[4,2],[4,4],[2,4]]], "notes": ""}'
    )
    with pytest.raises(ValueError, match="b.json"):
        load_label("2000", "b", tmp_path)


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


def test_class_mask_includes_outlines_and_remove_wins():
    m = data.class_mask([[(0, 0), (9, 0), (9, 9), (0, 9)]], [[(3, 3), (6, 3), (6, 6), (3, 6)]], (20, 20))
    assert m[9, 9] == 0 and m[10, 10] == 1 and m[3, 3] == 2 and m[6, 6] == 2 and m[7, 7] == 0
    assert (m == 2).sum() == 16 and (m == 0).sum() == 100 - 16


def test_split_overlap_names_every_shared_sheet_and_both_folders():
    with pytest.raises(ValueError) as e:
        data.check_split([("1960", "6"), ("1921", "a"), ("1935", "p1")], [("1921", "a"), ("1960", "6"), ("1935", "p2")])
    msg = str(e.value)
    assert "1921/a" in msg and "1960/6" in msg and "1935" not in msg
    assert str(data.LABELS) in msg and str(data.TEST_LABELS) in msg
    data.check_split([("1960", "6")], [("1960", "1")])                 # apart: no error
