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
