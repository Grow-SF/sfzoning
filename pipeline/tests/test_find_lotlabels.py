"""The finder's street-or-not labels from a series' lots (`sfzmaps.find.lotlabels`): today's blocks are not street,
ground with no lot within the reach of a street's centre line is, and nothing else is labelled; read onto a page
through its placement, and nothing off the frame cut."""
from pathlib import Path

import numpy as np
import pytest

from find_fixtures import EAST, ROWS, SHAPE, STREET, WEST, Flat, lines_world, lots_world
from sfzmaps.find import lotlabels, pages
from sfzmaps.solver import grid
from sfzmaps.solver.hatchnet.data import NOT_STREET, UNKNOWN
from sfzmaps.solver.hatchnet.data import STREET as ST

CHECKED_IN = Path(__file__).parents[1] / "inputs" / "find" / "label_sources.csv"
FT_PER_PX = 2.0                                     # two page px to a grid px each way
PAGE = (SHAPE[0] * 2, SHAPE[1] * 2)                 # the page covers the whole fixture grid


def grid_cols_rows() -> tuple[np.ndarray, np.ndarray]:
    """The grid column of each page column and the grid row of each page row, through `Flat(FT_PER_PX)`."""
    return (np.floor(np.arange(PAGE[1]) * FT_PER_PX / grid.FT_PER_PX).astype(int),
            np.floor(np.arange(PAGE[0]) * FT_PER_PX / grid.FT_PER_PX).astype(int))


def page_mask(cols: tuple[int, int], rows: tuple[int, int]) -> np.ndarray:
    """The page px over grid columns and rows (ends exclusive)."""
    c, r = grid_cols_rows()
    return ((r >= rows[0]) & (r < rows[1]))[:, None] & ((c >= cols[0]) & (c < cols[1]))[None, :]


@pytest.fixture(scope="module")
def ground():
    return lotlabels.build_ground(lots_world(), lines_world(), SHAPE)


def test_block_px_are_not_street_and_the_gap_is(ground):
    assert ground.lot.dtype == np.int32 and ground.block.dtype == bool
    assert ground.street.dtype == np.uint8 and ground.weight.dtype == np.float16
    assert set(np.unique(ground.lot)) == {-1, 0, 1, 2, 3}
    pg = lotlabels.page_ground(ground, Flat(FT_PER_PX), PAGE, np.ones(PAGE, bool))
    assert pg.street.shape == PAGE and pg.street.dtype == np.uint8 and pg.weight.dtype == np.float32
    blocks = page_mask(WEST, ROWS) | page_mask(EAST, ROWS)
    assert (pg.street[blocks] == 0).all() and (pg.weight[blocks] == 1).all()
    gap = page_mask(STREET, ROWS)
    assert (pg.street[gap] == 1).all()
    assert (pg.weight[gap] > 0).all() and (pg.weight[gap] <= 1).all()
    beyond = page_mask((0, WEST[0]), ROWS) | page_mask((EAST[1], SHAPE[1]), ROWS)
    assert (pg.street[beyond] == 255).all() and (pg.weight[beyond] == 0).all()


def test_off_the_frame_cut_is_unlabelled(ground):
    keep = np.ones(PAGE, bool)
    keep[:, :PAGE[1] // 2] = False
    pg = lotlabels.page_ground(ground, Flat(FT_PER_PX), PAGE, keep)
    assert (pg.street[~keep] == 255).all() and (pg.weight[~keep] == 0).all()
    east = page_mask(EAST, ROWS)
    assert east[keep].any() and (pg.street[east] == 0).all() and (pg.weight[east] == 1).all()
    west = page_mask(WEST, ROWS)
    assert (pg.street[west] == 255).all()


def test_a_series_without_lot_labels_is_refused(monkeypatch):
    monkeypatch.setattr(pages, "SOURCES", CHECKED_IN)
    with pytest.raises(ValueError, match="1921"):
        lotlabels.series_lots("1921")


def test_the_lots_come_from_the_table_or_the_series_year(tmp_path, monkeypatch):
    src = tmp_path / "label_sources.csv"
    src.write_text("series,era,lots\n2000-digital-zoning,lined-1bit,table\n1980-1990-zoning,lined-1bit,year\n")
    monkeypatch.setattr(pages, "SOURCES", src)
    monkeypatch.setattr(lotlabels, "load_table", lambda: "the table")
    monkeypatch.setattr(lotlabels, "load_parcels", lambda year: f"the lots of {year:g}")
    assert lotlabels.series_lots("2000-digital-zoning") == "the table"
    assert lotlabels.series_lots("1980-1990-zoning") == "the lots of 1980"


def _label_file(placement: str, shape=(20, 30)) -> None:
    f = lotlabels.sheet_path(("t", "A"))
    f.parent.mkdir(parents=True, exist_ok=True)
    e = np.zeros(0)
    np.savez(f, street=np.zeros(shape, np.uint8), weight=np.ones(shape, np.float32), cuts_px=e.astype(np.int64),
             cuts_cut=e.astype(np.int32), cuts_prior=e.astype(np.float32), cuts_weight=e.astype(np.float32),
             edge_none=np.ones(shape, bool), sigma=np.float64(2.0), placement=np.array(placement))


def test_every_reader_of_a_label_file_checks_its_placement(tmp_path, monkeypatch):
    from sfzmaps.find import data
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    monkeypatch.setattr(pages, "load_scan", lambda sid: np.ones((20, 30), np.float32))
    again = r"made through the work placement, not warp; make it again with `sfzmaps find labels --series t " \
            r"--sheet A --placement warp`"
    with pytest.raises(FileNotFoundError, match=r"no labels; make them with `sfzmaps find labels --series t"):
        lotlabels.load_sheet_labels(("t", "A"))
    _label_file("work")
    for read in (lambda: lotlabels.load_sheet_labels(("t", "A")), lambda: data.sheet_labelled(("t", "A"))):
        with pytest.raises(ValueError, match=again):
            read()
    assert lotlabels.load_sheet_labels(("t", "A"), "work")["street"].shape == (20, 30)
    with pytest.raises(ValueError, match=r"labels are \(20, 30\), the page \(20, 31\)"):
        lotlabels.load_sheet_labels(("t", "A"), "work", (20, 31))


def test_a_label_file_made_before_gives_its_block_edge_cuts_only(tmp_path, monkeypatch):
    """A label file made before the finder dropped the zoning border holds zoning-border cuts among its block-edge
    cuts (cuts_answer 1, "a zoning border crosses" or "none does") and a `border_none` label: they are not read."""
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    shape = (20, 30)
    f = lotlabels.sheet_path(("t", "A"))
    f.parent.mkdir(parents=True, exist_ok=True)
    px = np.array([3, 4, 5, 33, 34, 63, 64, 65, 66, 93], np.int64)
    np.savez(f, street=np.zeros(shape, np.uint8), weight=np.ones(shape, np.float32), cuts_px=px,
             cuts_cut=np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3], np.int32),
             cuts_prior=np.linspace(0.1, 1, 10).astype(np.float32), cuts_answer=np.array([0, 1, 0, 1], np.uint8),
             cuts_says=np.array([True, True, True, False]), cuts_weight=np.array([1, 1, 0, 1], np.float32),
             edge_none=np.ones(shape, bool), border_none=np.ones(shape, bool), sigma=np.float64(2.0),
             placement=np.array("warp"))
    c = lotlabels.file_cuts(lotlabels.load_sheet_labels(("t", "A"), "warp", shape))
    assert list(c.px) == [3, 4, 5, 63, 64, 65, 66] and list(c.cut) == [0, 0, 0, 1, 1, 1, 1]
    assert np.array_equal(c.prior, np.linspace(0.1, 1, 10).astype(np.float32)[[0, 1, 2, 5, 6, 7, 8]])
    assert list(c.weight) == [1, 0] and list(c.inside) == [-1, -1]


# --- street where a centre line runs ----------------------------------------------------------------------------------

def _boxes(shape, *boxes) -> np.ndarray:
    m = np.zeros(shape, bool)
    for r0, r1, c0, c1 in boxes:
        m[r0:r1, c0:c1] = True
    return m


def _ruled(lot: np.ndarray, centre: np.ndarray, street: np.ndarray | None = None):
    """The rule applied to labels that call every px with no lot street (`street` None) or to the labels `street`, with
    weights that vary px to px; checks what holds whatever the case: block px and their weights untouched, a street
    label that stays keeps its weight, a street label the rule adds weighs 1, every px with no lot the rule does not
    call street has no label, and `off` is exactly those px.  The street labels after, and `off`."""
    block = lot
    if street is None:
        street = np.where(block, NOT_STREET, ST).astype(np.uint8)
    weight = np.where(block, 1, np.where(street == ST, 0.25 + 0.25 * (np.arange(lot.size).reshape(lot.shape) % 4),
                                         0)).astype(np.float16)
    s, w, off = lotlabels.centre_line_street(street, weight, block, lot, centre)
    assert (s[block] == NOT_STREET).all() and (w[block] == 1).all()
    stays, added = (s == ST) & (street == ST), (s == ST) & (street != ST)
    assert (w[stays] == weight[stays]).all() and (w[added] == 1).all()
    assert np.array_equal(off, ~block & (s != ST)) and (s[off] == UNKNOWN).all() and (w[off] == 0).all()
    return s, off


def _reach_by_every_point(centre: np.ndarray, lot: np.ndarray) -> np.ndarray:
    """Every px tested against every centre-line point: within the point's distance to the nearest lot px."""
    ly, lx = np.nonzero(lot)
    out = np.zeros(lot.shape, bool)
    yy, xx = np.mgrid[:lot.shape[0], :lot.shape[1]]
    for y, x in zip(*np.nonzero(centre)):
        r2 = ((ly - y) ** 2 + (lx - x) ** 2).min() if len(ly) else np.inf
        out |= (yy - y) ** 2 + (xx - x) ** 2 <= r2
    return out


def test_the_reach_is_what_testing_every_px_against_every_centre_line_point_gives():
    rng = np.random.default_rng(5)
    for trial in range(12):
        shape = (int(rng.integers(8, 40)), int(rng.integers(8, 40)))
        lot = rng.random(shape) < rng.uniform(0.02, 0.4)
        centre = rng.random(shape) < rng.uniform(0.01, 0.1)
        if trial == 0:
            lot[:] = False                                  # no lot at all: no centre-line point reaches anywhere
        np.testing.assert_array_equal(lotlabels.within_reach(centre, lot), _reach_by_every_point(centre, lot) &
                                      (lot.any() | ~centre.any()), err_msg=f"trial {trial}")


@pytest.mark.parametrize("gap", [5, 401])
def test_two_blocks_with_the_centre_line_midway_the_whole_gap_is_street_whatever_its_width(gap):
    west, east = (20, 40), (40 + gap, 60 + gap)
    shape = (100, east[1] + 20)
    lot = _boxes(shape, (10, 90, *west), (10, 90, *east))
    centre = _boxes(shape, (10, 90, 40 + gap // 2, 41 + gap // 2))
    s, _ = _ruled(lot, centre)
    assert (s[10:90, west[1]:east[0]] == ST).all()


@pytest.mark.parametrize("k", [1, 3])
def test_a_centre_line_off_the_middle_leaves_a_sliver_along_the_far_face(k):
    gap = 21                                                # columns 40-60; the lots' faces at columns 39 and 61
    lot = _boxes((100, 100), (10, 90, 20, 40), (10, 90, 61, 81))
    line = 50 - k                                           # the middle is column 50
    s, off = _ruled(lot, _boxes((100, 100), (10, 90, line, line + 1)))
    rows = slice(30, 70)                                    # away from the blocks' ends
    reach = line - 39
    assert (s[rows, 40:line + reach + 1] == ST).all()       # from the near face to as far beyond the line
    assert off[rows, line + reach + 1:61].all()             # the sliver along the far face: no label
    assert 61 - (line + reach + 1) == 2 * k - 1 and gap == 61 - 40


def test_a_road_with_a_block_on_one_side_only_is_street_as_far_beyond_its_line_as_the_face_is_before_it():
    lot = _boxes((60, 80), (10, 50, 40, 70))               # the block; open ground west of it to the grid's edge
    s, off = _ruled(lot, _boxes((60, 80), (10, 50, 33, 34)))   # the road's centre line, 7 px from the face's lots
    rows = slice(20, 40)
    assert (s[rows, 26:40] == ST).all() and off[rows, :26].all()


def test_two_streets_crossing_between_four_blocks_are_street_across_the_whole_square():
    a, b = (10, 40), (49, 79)                              # blocks 30 px, streets 9 px (40-48)
    lot = _boxes((89, 89), (*a, *a), (*a, *b), (*b, *a), (*b, *b))
    centre = _boxes((89, 89), (0, 89, 44, 45)) | _boxes((89, 89), (44, 45, 0, 89))
    s, _ = _ruled(lot, centre)
    assert (s[40:49, 40:49] == ST).all() and (s[10:40, 40:49] == ST).all() and (s[40:49, 10:40] == ST).all()


def test_ground_with_no_lot_and_no_centre_line_has_no_label():
    lot = _boxes((40, 40), (5, 15, 5, 15))
    s, off = _ruled(lot, np.zeros((40, 40), bool))
    assert not (s == ST).any() and off.sum() == 40 * 40 - 100


def test_a_centre_line_inside_a_lot_calls_nothing_street():
    lot = _boxes((40, 60), (10, 30, 10, 50))
    s, off = _ruled(lot, _boxes((40, 60), (20, 21, 15, 45)))   # a drive through a park that is a lot
    assert not (s == ST).any() and off.sum() == 40 * 60 - 20 * 40


def test_the_rule_adds_a_street_label_where_the_lots_labels_had_none():
    lot = _boxes((40, 40), (0, 40, 0, 10), (0, 40, 21, 40))
    centre = _boxes((40, 40), (0, 40, 15, 16))
    s, _ = _ruled(lot, centre, street=np.where(lot, NOT_STREET, UNKNOWN).astype(np.uint8))
    assert (s[:, 10:21] == ST).all()


def test_the_centre_lines_order_changes_nothing():
    import shapely

    from sfzmaps.solver.hatchnet.data import Lines
    base = lines_world()
    across = shapely.LineString([(grid.X0, grid.Y1 - 4 * 30.5), (grid.X0 + 4 * SHAPE[1], grid.Y1 - 4 * 30.5)])
    road = shapely.LineString([(grid.X0 + 66.0, grid.Y1), (grid.X0 + 66.0, grid.Y1 - 4 * SHAPE[0])])
    geoms = [*base.geoms, across, road]
    one = Lines(np.array(geoms, dtype=object), np.array(["ST"] * 3, dtype=object), np.array(["A", "B", "C"],
                                                                                           dtype=object))
    two = Lines(one.geoms[::-1].copy(), one.kinds[::-1].copy(), one.streets[::-1].copy())
    g1, g2 = (lotlabels.build_ground(lots_world(), lines, SHAPE) for lines in (one, two))
    assert np.array_equal(g1.street == ST, g2.street == ST) and np.array_equal(g1.off, g2.off)
    assert (g1.street == ST).any() and g1.off.any()


def test_the_labels_picture_tints_only_px_whose_label_counts():
    from sfzmaps.find import look
    lum = np.full((10, 12), 0.5, np.float32)
    street = np.full((10, 12), UNKNOWN, np.uint8)
    street[:, :4], street[:, 4:8] = ST, NOT_STREET
    weight = np.ones((10, 12), np.float32)
    weight[0, :] = 0                                    # a label that does not count: untinted
    weight[:, 8:] = 0
    img = look.beside(lum, street, weight, (1, 0, 11, 10), "t")
    h = 10
    left, right = img[-h:, :10], img[-h:, 10 + look.GAP:]
    assert img.shape[1] == 2 * 10 + look.GAP and (left == 128).all()
    assert (right[0] == 128).all() and (right[:, 7:] == 128).all()          # no label, or one that does not count
    assert (right[1:, :3, 2] > right[1:, :3, 0]).all()                      # street: orange (more red than blue)
    assert (right[1:, 3:7, 0] > right[1:, 3:7, 2]).all()                    # block: blue
    keep = np.ones((10, 12), bool)
    keep[:, 10:] = False
    big = look.beside(lum, street, weight, (1, 0, 11, 10), "t", keep=keep, zoom=2)
    right = big[-2 * h:, 2 * (10 + look.GAP):]
    assert big.shape[1] == 2 * (2 * 10 + look.GAP) and (right[:, :18] == np.repeat(img[-h:, 10 + look.GAP:][:, :9],
                                                                                   2, axis=0).repeat(2, axis=1)).all()
    out = right[:, 18:]                                                     # page column 10: outside the frame cut
    assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all() and (out != 128).all()
