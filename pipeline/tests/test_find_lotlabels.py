"""The finder's street-or-not labels from a series' lots (`sfzmaps.find.lotlabels`): today's blocks are not street,
the street between them is, and nothing else is labelled; read onto a page through its placement, and nothing off the
frame cut."""
from pathlib import Path

import numpy as np
import pytest

from find_fixtures import EAST, ROWS, SHAPE, STREET, WEST, Flat, lines_world, lots_world
from sfzmaps.find import lotlabels, pages
from sfzmaps.solver import grid

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


def _label_file(placement: str, shape=(20, 30), ids=lotlabels.listed_ids()) -> None:
    f = lotlabels.sheet_path(("t", "A"))
    f.parent.mkdir(parents=True, exist_ok=True)
    e = np.zeros(0)
    np.savez(f, street=np.zeros(shape, np.uint8), weight=np.ones(shape, np.float32), cuts_px=e.astype(np.int64),
             cuts_cut=e.astype(np.int32), cuts_prior=e.astype(np.float32), cuts_weight=e.astype(np.float32),
             edge_none=np.ones(shape, bool), sigma=np.float64(2.0), placement=np.array(placement),
             **({} if ids is None else {"unlabelled_ids": np.array(ids, dtype=str)}))


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
             placement=np.array("warp"), unlabelled_ids=np.array(lotlabels.listed_ids(), dtype=str))
    c = lotlabels.file_cuts(lotlabels.load_sheet_labels(("t", "A"), "warp", shape))
    assert list(c.px) == [3, 4, 5, 63, 64, 65, 66] and list(c.cut) == [0, 0, 0, 1, 1, 1, 1]
    assert np.array_equal(c.prior, np.linspace(0.1, 1, 10).astype(np.float32)[[0, 1, 2, 5, 6, 7, 8]])
    assert list(c.weight) == [1, 0] and list(c.inside) == [-1, -1]


def test_street_labels_are_counted_by_px_and_by_weight():
    street = np.array([[1, 1, 1, 0], [255, 1, 0, 255]], np.uint8)
    weight = np.array([[1.0, 0.25, 0.0, 1.0], [0.0, 0.5, 1.0, 0.0]], np.float32)
    px, weighs = lotlabels.street_totals(street, weight)
    assert px == 3 and weighs == 1.75                    # the weightless street px and the block px count for nothing


def test_the_labels_picture_tints_street_as_strongly_as_the_label_weighs():
    from sfzmaps.find import look
    from sfzmaps.find.streetlabels import NOT_STREET, STREET, UNKNOWN
    lum = np.full((8, 12), 0.5, np.float32)
    street = np.full((8, 12), UNKNOWN, np.uint8)
    street[:, :6], street[:, 6:9] = STREET, NOT_STREET
    weight = np.zeros((8, 12), np.float32)
    weight[:, 0:2], weight[:, 2:4], weight[:, 4:6], weight[:, 6:9] = 1.0, 0.25, 0.0, 1.0
    img = look.beside(lum, street, weight, (0, 0, 12, 8), "t")
    right = img[-8:, 12 + look.GAP:].astype(float)
    shift = np.abs(right - 128).sum(axis=2)                                 # how far each px is tinted from the scan
    assert (img[-8:, :12] == 128).all()                                     # the scan on the left untinted
    assert shift[:, 0:2].min() > 0 and shift[:, 4:6].max() == 0             # weight 0: no tint
    assert np.allclose(shift[:, 2:4], shift[:, 0:2] / 4, atol=2)            # weight 0.25: a quarter of weight 1's tint
    assert (right[:, 0:2, 2] > right[:, 0:2, 0]).all()                      # street orange
    assert (right[:, 6:9, 0] > right[:, 6:9, 2]).all()                      # block blue
    assert (right[:, 9:] == 128).all()                                      # no label: untinted
    keep = np.ones((8, 12), bool)
    keep[:, 10:] = False
    big = look.beside(lum, street, weight, (0, 0, 12, 8), "t", keep=keep, zoom=2)
    out = big[-16:, 2 * (12 + look.GAP) + 20:]                              # page columns 10-11: off the frame cut
    assert big.shape[1] == 2 * (2 * 12 + look.GAP)
    assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all() and (out != 128).all()


# --- lots listed as carrying no label (inputs/find/unlabelled_lots.csv) ----------------------------------------------

def test_a_listed_lot_has_no_label_and_its_neighbours_and_cuts_are_as_they_were():
    from sfzmaps.find import cuts
    plain = lotlabels.build_ground(lots_world(), lines_world(), SHAPE)
    g = lotlabels.build_ground(lots_world(), lines_world(), SHAPE, unlabelled=["0002001"])
    listed = plain.lot == 2                                          # 0002001: the east block's north lot
    assert listed.any() and np.array_equal(g.unlabelled, listed)
    assert (g.street[listed] == 255).all() and (g.weight[listed] == 0).all()
    assert np.array_equal(g.street[~listed], plain.street[~listed])  # the street beside it, every other lot: as were
    assert np.array_equal(g.weight[~listed], plain.weight[~listed])
    assert (g.street[ROWS[0]:30, STREET[0]:STREET[1]] == 1).all()
    assert np.array_equal(g.block, plain.block) and np.array_equal(g.lot, plain.lot)
    a, b = (cuts.edge_cuts(x, Flat(2.0), (2 * SHAPE[0], 2 * SHAPE[1]), np.ones((2 * SHAPE[0], 2 * SHAPE[1]), bool),
                           1.5) for x in (plain, g))
    for k in ("px", "cut", "prior", "weight", "inside"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k       # the listed lot's sides keep their cuts
    pg = lotlabels.page_ground(g, Flat(2.0), (2 * SHAPE[0], 2 * SHAPE[1]), np.ones((2 * SHAPE[0], 2 * SHAPE[1]), bool))
    assert pg.unlabelled.sum() == 4 * listed.sum() and (pg.street[pg.unlabelled] == 255).all()


def test_a_listed_id_not_among_a_series_lots_is_refused(tmp_path, monkeypatch):
    f = tmp_path / "unlabelled_lots.csv"
    f.write_text("blklot,name,why\n0002001,a lot,test\n1700001,a park,test\n")
    monkeypatch.setattr(lotlabels, "UNLABELLED", f)
    monkeypatch.setattr(lotlabels, "series_lots", lambda series: lots_world())
    monkeypatch.setattr(lotlabels, "lines_from", lambda path: lines_world())
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    with pytest.raises(ValueError, match=r"1700001 \(a park\).*not among the lots of series t"):
        lotlabels.make_ground("t", SHAPE)
    assert not (tmp_path / "find").exists()
    f.write_text("blklot,name,why\n0002001,a lot,test\n")
    g = lotlabels.make_ground("t", SHAPE)
    assert g.unlabelled.sum() > 0 and lotlabels.ground("t").unlabelled.sum() == g.unlabelled.sum()


def test_the_checked_in_list_holds_the_two_parks():
    rows = lotlabels.unlabelled_lots(Path(__file__).parents[1] / "inputs" / "find" / "unlabelled_lots.csv")
    assert [r[0] for r in rows] == ["1300001", "1700001"]


def test_a_label_file_must_record_the_listed_lots_it_was_made_under(tmp_path, monkeypatch):
    """A sheet's label file records the ids of the lots listed as carrying no label when it was made; a file that
    records none (made before the list) or another list (the list changed since) is refused, naming the command that
    makes it again; a file made under the list as it is now loads."""
    from sfzmaps.find import data
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    monkeypatch.setattr(pages, "load_scan", lambda sid: np.ones((20, 30), np.float32))
    f = tmp_path / "unlabelled_lots.csv"
    f.write_text("blklot,name,why\n1300001,the Presidio,test\n1700001,Golden Gate Park,test\n")
    monkeypatch.setattr(lotlabels, "UNLABELLED", f)
    again = r"make it again with `sfzmaps find labels --series t --sheet A --placement warp`"
    _label_file("warp", ids=None)
    for read in (lambda: lotlabels.load_sheet_labels(("t", "A")), lambda: data.sheet_labelled(("t", "A"))):
        with pytest.raises(ValueError, match=r"records no list of lots carrying no label.*" + again):
            read()
    _label_file("warp", ids=["1700001", "1300001"])          # the same lots, in another order
    assert lotlabels.load_sheet_labels(("t", "A"))["street"].shape == (20, 30)
    f.write_text("blklot,name,why\n1700001,Golden Gate Park,test\n")
    with pytest.raises(ValueError, match=r"made with the lots 1300001, 1700001 carrying no label; the list now "
                                         r"holds 1700001.*" + again):
        lotlabels.load_sheet_labels(("t", "A"))


def test_a_grid_must_record_the_listed_lots_it_was_made_under(tmp_path, monkeypatch):
    f = tmp_path / "unlabelled_lots.csv"
    f.write_text("blklot,name,why\n0002001,a lot,test\n")
    monkeypatch.setattr(lotlabels, "UNLABELLED", f)
    monkeypatch.setattr(lotlabels, "series_lots", lambda series: lots_world())
    monkeypatch.setattr(lotlabels, "lines_from", lambda path: lines_world())
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    assert lotlabels.make_ground("t", SHAPE).unlabelled_ids == ("0002001",)
    assert lotlabels.ground("t").unlabelled_ids == ("0002001",)
    f.write_text("blklot,name,why\n")
    with pytest.raises(FileNotFoundError, match=r"made with the lots 0002001 carrying no label; the list now holds "
                                                r"none.*sfzmaps find labels --series t"):
        lotlabels.ground("t")
