import itertools

import numpy as np
import pytest
from PIL import Image

from sfzmaps.frame.data import (
    EDGE_MAX,
    POSITION_CHANNELS,
    labelled_sheets,
    load_label,
    load_luminance,
    map_mask,
    map_pages,
    page_path,
    position_planes,
)


def test_position_channels_default_to_the_scan_mask_alone():
    """Task 15: the edge distance taught the nets 'near the edge = frame' (a district bar along 1929 p09's Ocean
    Beach); the default is the scan mask alone."""
    assert POSITION_CHANNELS == ("valid",)


def test_position_planes_values(position_channels):
    """The planes in POSITION_CHANNELS' order.  `valid` is 1 on the page; `edge` is 0 on the page's edge px (the
    corners too), rises by 1 / (shorter side's span) per px, and is 1/2 at the centre of the shorter side, which the
    clip to EDGE_MAX keeps."""
    h, w = 101, 151
    got = position_planes((h, w))
    assert isinstance(got, tuple) and len(got) == len(position_channels)
    planes = dict(zip(position_channels, got))
    assert all(p.dtype == np.float32 and p.shape == (h, w) for p in got)
    assert (planes["valid"] == 1).all() and EDGE_MAX == 0.5
    if "edge" not in planes:
        return
    edge = planes["edge"]
    for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        assert edge[r, c] == 0
    assert (edge[0] == 0).all() and (edge[:, -1] == 0).all()
    assert edge[h // 2, w // 2] == pytest.approx(min(min(h, w) / 2 / min(h, w), 0.5), abs=1e-7)
    assert edge[10, 75] == pytest.approx(10 / 100) and edge[50, 3] == pytest.approx(3 / 100)
    assert edge.max() <= EDGE_MAX and edge.min() >= 0


def test_position_planes_turn_and_flip_with_the_page(position_channels):
    """Each of the eight quarter-turn / mirror-flip maps of a page's planes is the planes of the moved page."""
    shape = (13, 22)
    planes = np.stack(position_planes(shape))
    for k, f in itertools.product(range(4), (0, 1)):
        moved = np.rot90(np.flip(planes, 2) if f else planes, k, (1, 2))
        assert np.array_equal(moved, np.stack(position_planes(moved.shape[1:])))


def test_position_planes_over_a_window(position_channels):
    """A window (top, left, height, width) of the page's frame, off the page included: the whole page's planes where
    it is on the page, 0 off it."""
    shape = (30, 40)
    whole = np.stack(position_planes(shape))
    for top, left, hh, ww in ((0, 0, 30, 40), (-5, 7, 20, 50), (25, -3, 12, 9), (40, 50, 5, 5)):
        got = np.stack(position_planes(shape, (top, left, hh, ww)))
        want = np.zeros((len(position_channels), hh, ww), np.float32)
        r0, c0, r1, c1 = max(top, 0), max(left, 0), min(top + hh, 30), min(left + ww, 40)
        if r1 > r0 and c1 > c0:
            want[:, r0 - top:r1 - top, c0 - left:c1 - left] = whole[:, r0:r1, c0:c1]
        assert got.dtype == np.float32 and np.array_equal(got, want)


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
    assert load_label("1921", "a", tmp_path) == ([[(0, 0), (4, 0), (4, 4)]], [])


def test_load_label_reads_holes(tmp_path):
    (tmp_path / "2000").mkdir()
    (tmp_path / "2000" / "b.json").write_text(
        '{"polygons": [[[0,0],[9,0],[9,9],[0,9]]], "holes": [[[2,2],[4,2],[4,4],[2,4]]], "notes": ""}'
    )
    polygons, holes = load_label("2000", "b", tmp_path)
    assert polygons == [[(0, 0), (9, 0), (9, 9), (0, 9)]]
    assert holes == [[(2, 2), (4, 2), (4, 4), (2, 4)]]


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
