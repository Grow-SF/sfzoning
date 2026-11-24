"""The finder's street-or-not labels from a series' lots (`sfzmaps.find.lotlabels`): today's blocks are not street,
the street between them is, and nothing else is labelled; read onto a page through its placement, and nothing off the
frame cut."""
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


# --- street only between blocks ---------------------------------------------------------------------------------------

def _ruled(block: np.ndarray):
    """The rule applied to labels calling every px off `block` street (as the lots' labels did by open ground before),
    with weights that vary px to px: (street, weight) before, (street, weight) after, and the px that lost the label."""
    street = np.where(block, NOT_STREET, ST).astype(np.uint8)
    weight = np.where(block, 1, 0.25 + 0.25 * (np.arange(block.size).reshape(block.shape) % 4)).astype(np.float16)
    s, w, lost = lotlabels.street_between(street, weight, block)
    assert (s[block] == NOT_STREET).all() and (w[block] == 1).all()                 # block px are untouched
    kept = s == ST
    assert (w[kept] == weight[kept]).all()                                          # kept street: its weight as it was
    assert np.array_equal(lost, (street == ST) & (s == UNKNOWN)) and (w[lost] == 0).all()
    assert not ((s == UNKNOWN) & ~lost).any()
    return s, lost


def _blocks(shape, *boxes) -> np.ndarray:
    block = np.zeros(shape, bool)
    for r0, r1, c0, c1 in boxes:
        block[r0:r1, c0:c1] = True
    return block


@pytest.mark.parametrize("gap", [3, 400])
def test_the_gap_between_two_facing_blocks_is_street_whatever_its_width(gap):
    west, east = (20, 40), (40 + gap, 60 + gap)                     # the blocks' columns, rows 20-59
    shape = (80, east[1] + 20)
    s, lost = _ruled(_blocks(shape, (20, 60, *west), (20, 60, *east)))
    assert (s[20:60, west[1]:east[0]] == ST).all()              # the gap between the faces
    for beyond in (s[20:60, :west[0]], s[20:60, east[1]:]):         # beyond the outer faces, open to the grid's edge
        assert (beyond == UNKNOWN).all()
    for mouth in (s[:20, west[1]:east[0]], s[60:, west[1]:east[0]]):  # the gap's mouths beyond the blocks' faces
        assert (mouth == UNKNOWN).all()
    assert lost.sum() == (s == UNKNOWN).sum()


def test_the_square_where_two_streets_cross_between_four_blocks_is_street():
    a, b = (10, 40), (48, 78)                                      # the blocks' rows and columns: 30 px, streets 8 px
    s, _ = _ruled(_blocks((88, 88), (*a, *a), (*a, *b), (*b, *a), (*b, *b)))
    assert (s[40:48, 40:48] == ST).all()                        # the crossing
    assert (s[10:40, 40:48] == ST).all() and (s[48:78, 40:48] == ST).all()      # the streets between
    assert (s[40:48, 10:40] == ST).all() and (s[40:48, 48:78] == ST).all()
    assert (s[:10] == UNKNOWN).all() and (s[78:] == UNKNOWN).all()  # round the outside: open to the grid's edge
    assert (s[:, :10] == UNKNOWN).all() and (s[:, 78:] == UNKNOWN).all()


def test_a_notch_cut_into_a_block_is_street_between_its_sides():
    # a block 40 x 40 with a notch 8 px wide cut 30 px deep into it from the north (its sides columns 36-43)
    block = _blocks((60, 80), (10, 50, 20, 60))
    block[10:40, 36:44] = False
    s, _ = _ruled(block)
    rows, cols = np.mgrid[10:40, 36:44]
    by_side = np.minimum(cols - 35, 44 - cols) < 40 - rows          # nearer a side of the notch than its end
    assert (s[10:40, 36:44][by_side] == ST).all()
    # the rule as it is: px whose nearest block px is the notch's end look out of its mouth, and get no label
    assert (s[39, 38:42] == UNKNOWN).all()


def test_a_strip_beside_a_single_block_has_no_label():
    s, lost = _ruled(_blocks((60, 60), (20, 40, 20, 40)))
    assert not (s == ST).any() and lost.sum() == 60 * 60 - 20 * 20


def test_a_lot_free_square_enclosed_by_blocks_is_still_street():
    # four blocks round a square (rows and columns 20-39), streets leaving it at its corners: the limit the rule admits
    s, _ = _ruled(_blocks((60, 60), (10, 20, 14, 46), (40, 50, 14, 46), (24, 36, 10, 20), (24, 36, 40, 50)))
    assert (s[20:40, 20:40] == ST).all()


def test_no_block_on_the_grid_labels_no_street():
    s, lost = _ruled(np.zeros((30, 40), bool))
    assert (s == UNKNOWN).all() and lost.all()


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
