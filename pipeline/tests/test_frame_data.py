import numpy as np
import pytest
from PIL import Image

from sfzmaps.frame.data import (
    labelled_sheets,
    load_label,
    load_luminance,
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
