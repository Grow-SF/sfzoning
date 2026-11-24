"""The finder's cut statements (`sfzmaps.find.cuts`): short runs of page px across each block side the lots give, each
saying "a block edge crosses this cut"; nothing off the page or off the frame cut is in a cut.  A sheet's labels
(`lotlabels.sheet_labels`) are its street-or-not labels, its "no block edge" px and its block-edge cuts, kept only where
the sheet draws ink (`find.drawn`)."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from find_fixtures import (EAST, GRID_COLS, GRID_ROWS, ROWS, SHAPE, SPLIT, STREET, WEST, Flat, blocks_world,
                           lines_world, lots_world)
from sfzmaps.find import cuts, drawn, lotlabels, pages
from sfzmaps.frame import data as fdata
from sfzmaps.solver import grid

CHECKED_IN = Path(__file__).parents[1] / "inputs" / "find" / "label_sources.csv"
FT_PER_PX = 2.0                                     # two page px to a grid px each way
PAGE = (SHAPE[0] * 2, SHAPE[1] * 2)                 # the page covers the whole fixture grid
SIGMA = 1.5
R = 6                                               # ceil(4 * SIGMA)


@pytest.fixture(scope="module")
def ground():
    return lotlabels.build_ground(lots_world(), lines_world(), SHAPE)


def every(c: cuts.Cuts):
    """Each cut's (rows, cols, prior) in page px."""
    rows, cols = np.divmod(c.px, PAGE[1])
    for i in range(len(c.weight)):
        on = c.cut == i
        yield rows[on], cols[on], c.prior[on]


def test_an_edge_cut_crosses_each_block_side(ground):
    c = cuts.edge_cuts(ground, Flat(FT_PER_PX), PAGE, np.ones(PAGE, bool), SIGMA)
    sides_x = {2 * WEST[0], 2 * WEST[1], 2 * EAST[0], 2 * EAST[1]}        # page x of the blocks' west and east sides
    sides_y = {2 * ROWS[0], 2 * ROWS[1]}                                  # page y of their north and south sides
    assert len(c.weight) == 2 * 2 * ((WEST[1] - WEST[0]) + (ROWS[1] - ROWS[0]))    # one per grid px along each side
    assert c.px.dtype == np.int64 and c.cut.dtype == np.int32 and c.prior.dtype == np.float32
    assert c.inside.shape == c.weight.shape and (c.inside >= 0).all()
    for rows, cols, prior in every(c):
        assert len(rows) == 2 * R + 1
        assert np.argmax(prior) == R and prior[R] == 1 and prior[0] < 1e-3 and prior[-1] < 1e-3
        if len(set(rows)) == 1:            # across a west or east side: along x, centred on the side
            assert np.array_equal(np.abs(np.diff(cols)), np.ones(2 * R, int)) and cols[R] in sides_x
        else:                              # across a north or south side: along y
            assert len(set(cols)) == 1 and np.array_equal(np.abs(np.diff(rows)), np.ones(2 * R, int))
            assert rows[R] in sides_y


def test_a_cut_leaving_the_frame_cut_is_dropped(ground):
    keep = np.ones(PAGE, bool)
    keep[:, 2 * STREET[0] + 2:2 * STREET[1] - 2] = False        # a strip down the middle of the street
    c = cuts.edge_cuts(ground, Flat(FT_PER_PX), PAGE, keep, SIGMA)
    assert keep.ravel()[c.px].all()
    assert len(c.weight) == 2 * 2 * ((WEST[1] - WEST[0]) + (ROWS[1] - ROWS[0])) - 2 * 40     # both sides by the strip
    for rows, cols, prior in every(c):          # dropped whole: no cut is cut short
        assert np.argmax(prior) >= R and np.argmax(prior[::-1]) >= R


def test_doubling_the_cut_length_changes_nothing_that_matters(ground):
    keep = np.ones(PAGE, bool)
    one, two = (cuts.edge_cuts(ground, Flat(FT_PER_PX), PAGE, keep, SIGMA, r) for r in (R, 2 * R))
    assert len(one.weight) == len(two.weight) and np.array_equal(one.inside, two.inside)
    s1 = np.bincount(one.cut, one.prior.astype(float))
    s2 = np.bincount(two.cut, two.prior.astype(float))
    assert (np.abs(s2 - s1) < 1e-3 * s1).all()


def ink_outline(page: np.ndarray, cols: tuple[int, int], rows: tuple[int, int], sides: str = "wnes") -> None:
    """Draw a block's outline on `page` (grid columns and rows, ends exclusive, through `Flat(FT_PER_PX)`): a 1 px line
    just outside each of its `sides` (w, n, e, s), where the block-edge cuts across that side cross."""
    (c0, c1), (r0, r1) = cols, rows
    if "w" in sides:
        page[2 * r0 - 1:2 * r1 + 1, 2 * c0 - 1] = 0
    if "e" in sides:
        page[2 * r0 - 1:2 * r1 + 1, 2 * c1] = 0
    if "n" in sides:
        page[2 * r0 - 1, 2 * c0 - 1:2 * c1 + 1] = 0
    if "s" in sides:
        page[2 * r1, 2 * c0 - 1:2 * c1 + 1] = 0


def _sheet_world(tmp_path, monkeypatch, sources: str, page: np.ndarray | None = None, margin: int = 10,
                 keep: np.ndarray | None = None):
    """A one-sheet series `s` on disk: the page `page` (None: white) of PAGE, its frame cut keeping `keep` (None: all
    but a `margin` px strip round the page), the label sources `sources`, a flat placement and a spread of SIGMA."""
    work, frame = tmp_path / "work", tmp_path / "frame"
    (work / "s" / "s-p01").mkdir(parents=True)
    page = np.full(PAGE, 255, np.uint8) if page is None else page
    Image.fromarray(page, "L").save(work / "s" / "s-p01" / "page.png")
    cut = np.full(PAGE, 255, np.uint8)
    cut[margin:-margin, margin:-margin] = 0
    if keep is not None:
        cut = np.where(keep, 0, 255).astype(np.uint8)
    (frame / "s" / "s-p01").mkdir(parents=True)
    Image.fromarray(cut, "L").save(frame / "s" / "s-p01" / "outside.png")
    (tmp_path / "label_sources.csv").write_text("series,era,lots\n" + sources)
    monkeypatch.setattr(fdata, "WORK", work)
    monkeypatch.setattr(fdata, "OUT", frame)
    monkeypatch.setattr(pages, "SOURCES", tmp_path / "label_sources.csv")
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(FT_PER_PX))
    monkeypatch.setattr(cuts, "spread_px", lambda sid, which="warp": SIGMA)
    return cut == 0


def test_a_series_not_labelled_from_lots_has_no_sheet_labels(tmp_path, monkeypatch):
    _sheet_world(tmp_path, monkeypatch, "s,hatched-grey,\n")
    with pytest.raises(ValueError, match="series s: its lots do not label it"):
        lotlabels.sheet_labels(("s", "s-p01"))
    assert not (tmp_path / "find" / "labels").exists()


def test_a_sheets_labels_are_street_no_block_edge_and_block_edge_cuts(tmp_path, monkeypatch, ground):
    page = np.full(PAGE, 255, np.uint8)
    for cols in (WEST, EAST):
        ink_outline(page, cols, ROWS)
    _sheet_world(tmp_path, monkeypatch, "s,lined-1bit,table\n", page)
    monkeypatch.setattr(lotlabels, "ground", lambda series: ground)
    lab = lotlabels.sheet_labels(("s", "s-p01"))
    c = lab["cuts"]
    assert len(c.weight) == 2 * 2 * ((WEST[1] - WEST[0]) + (ROWS[1] - ROWS[0])) and (c.weight == 1).all()
    on_edge = np.zeros(PAGE[0] * PAGE[1], bool)
    on_edge[c.px] = True
    labelled = (lab["street"] != 255).ravel()
    assert np.array_equal(lab["edge_none"].ravel(), labelled & ~on_edge)
    with np.load(tmp_path / "find" / "labels" / "s" / "s-p01.npz") as z:
        assert set(z.files) == {"street", "weight", "cuts_px", "cuts_cut", "cuts_prior", "cuts_weight", "edge_none",
                                "sigma", "placement", "count_edge_used", "count_edge_dropped",
                                "count_labelled_px_before", "count_labelled_px_after", "count_street_label_lost"}
        assert str(z["placement"]) == "warp" and float(z["sigma"]) == SIGMA


def test_spread_is_the_placement_miss_and_the_grid_in_page_px(tmp_path, monkeypatch):
    (tmp_path / "work" / "s" / "s-p01").mkdir(parents=True)
    Image.fromarray(np.full(PAGE, 255, np.uint8), "L").save(tmp_path / "work" / "s" / "s-p01" / "page.png")
    warp = tmp_path / "warp"
    (warp / "s" / "s-p01").mkdir(parents=True)
    (warp / "s" / "s-p01" / "scores.json").write_text(json.dumps(
        {"chosen": "probabilistic", "control_miss_p50_today": 2.0, "control_miss_p50_first": 9.0,
         "control_miss_p50_probabilistic": 1.0}))

    class Placed(Flat):
        def median_scale_ft_per_px(self, shape):
            assert tuple(shape) == PAGE
            return self.ft_per_px

    monkeypatch.setattr(fdata, "WORK", tmp_path / "work")
    monkeypatch.setattr(lotlabels, "PLACEMENTS", {"warp": warp, "work": tmp_path / "work"})
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Placed(FT_PER_PX))
    grid_px = grid.FT_PER_PX / np.sqrt(12) / FT_PER_PX
    assert cuts.spread_px(("s", "s-p01"), "warp") == pytest.approx(np.hypot(1.0 / 1.1774, grid_px))
    assert cuts.spread_px(("s", "s-p01"), "work") == pytest.approx(np.hypot(2.0 / 1.1774, grid_px))


def test_ground_is_made_only_by_the_labels_command(tmp_path, monkeypatch):
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    with pytest.raises(FileNotFoundError, match="sfzmaps find labels --series 2000-digital-zoning"):
        lotlabels.ground("2000-digital-zoning")


def test_a_series_without_lot_labels_is_a_usage_error(monkeypatch):
    from sfzmaps.cli import app
    monkeypatch.setattr(pages, "SOURCES", CHECKED_IN)
    for series in ("1921", "no-such-series"):
        r = CliRunner().invoke(app, ["find", "labels", "--series", series])
        assert r.exit_code == 2 and "--series" in r.output, r.output


def test_the_close_up_is_the_window_with_the_most_cut_px():
    from sfzmaps.find import look
    shape = (300, 400)
    px = np.r_[np.arange(10) * shape[1] + 5,                       # 10 cut px near the top-left corner
               (200 + np.arange(30)) * shape[1] + 350,             # 30 down a column near the right
               (50 + np.arange(60)) * shape[1] + 150]              # and 60 of a cut not used
    cut = np.r_[np.zeros(40, np.int32), np.ones(60, np.int32)]
    c = cuts.Cuts(px.astype(np.int64), cut, np.ones(len(px), np.float32), np.array([1, 0], np.float32),
                  np.zeros(2, np.int32), np.zeros(0, np.int64))
    x0, y0, x1, y1 = look.busiest({"cuts": c}, shape, 100)
    assert (x1 - x0, y1 - y0) == (100, 100)
    assert x0 <= 350 < x1 and y0 <= 200 and 229 < y1
    lum = np.ones(shape, np.float32)
    lab = {"cuts": c, "street": np.full(shape, 255, np.uint8), "weight": np.zeros(shape, np.float32),
           "placement": "warp", "sigma": 1.0}
    whole, close = look.labels(("s", "s-p01"), lum, lab), look.labels(("s", "s-p01"), lum, lab, (x0, y0, x1, y1))
    assert whole.shape[1] == shape[1] and close.shape[1] == 2 * 100       # the close-up at 2x
    painted = (close[-200:].reshape(-1, 3) != 255).any(axis=1)
    assert painted.sum() == 4 * 30                                        # the 30 cut px in it, each 2 x 2


def test_an_edge_cut_without_ink_is_not_used(ground):
    c = cuts.edge_cuts(ground, Flat(FT_PER_PX), PAGE, np.ones(PAGE, bool), SIGMA)
    lum = np.ones(PAGE, np.float32)
    lum[:, 2 * WEST[0] - 1] = 0                                   # only the west block's west side is drawn
    used, counts = drawn.use(c, lum)
    west_side = np.bincount(c.cut, c.px % PAGE[1] == 2 * WEST[0]) > 0
    assert (used.weight[~west_side] == 0).all() and counts["edge_dropped"] == int((~west_side).sum())


def test_an_edge_cut_with_ink_is_used(ground):
    c = cuts.edge_cuts(ground, Flat(FT_PER_PX), PAGE, np.ones(PAGE, bool), SIGMA)
    page = np.full(PAGE, 255, np.uint8)
    for cols in (WEST, EAST):
        ink_outline(page, cols, ROWS)
    used, counts = drawn.use(c, page.astype(np.float32) / 255)
    assert (used.weight == 1).all() and counts == {"edge_used": len(c.weight), "edge_dropped": 0}


def _labelled(tmp_path, monkeypatch, ground, page, keep=None):
    """`sheet_labels` of a lot-labelled series with lots `ground` and page `page`, the frame cut `keep` (None: room for
    the block-edge cuts of `blocks_world`'s top row); and the street-or-not weights before the rule."""
    keep = _sheet_world(tmp_path, monkeypatch, "s,lined-1bit,year\n", page, margin=2, keep=keep)
    monkeypatch.setattr(lotlabels, "ground", lambda series: ground)
    before = lotlabels.page_ground(ground, Flat(FT_PER_PX), PAGE, keep).weight
    return lotlabels.sheet_labels(("s", "s-p01")), before


def test_ground_by_a_bare_block_side_has_no_label(tmp_path, monkeypatch, ground):
    page = np.full(PAGE, 255, np.uint8)
    ink_outline(page, WEST, ROWS, "wn")                          # the west block: its west and north sides only
    ink_outline(page, EAST, ROWS)
    lab, before = _labelled(tmp_path, monkeypatch, ground, page)
    rows = slice(2 * ROWS[0] + 20, 2 * ROWS[1] - 20)             # well away from the north and south sides
    by_ink = (rows, slice(2 * WEST[0] + 1, 2 * WEST[0] + 7))     # block px by the inked west side
    by_bare = (rows, slice(2 * WEST[1] - 7, 2 * STREET[0] + 3))  # block and street px by the bare east side
    by_east = (rows, slice(2 * STREET[1] - 3, 2 * STREET[1]))    # street px by the east block's inked west side
    for box in (by_ink, by_east):
        assert (before[box] > 0).all() and np.array_equal(lab["weight"][box], before[box])
    assert (before[by_bare] > 0).all() and (lab["weight"][by_bare] == 0).all()
    assert not lab["edge_none"][by_bare].any()
    inner = (slice(2 * ROWS[0] + 12, 2 * SPLIT - 12), slice(2 * WEST[0] + 8, 2 * WEST[0] + 14))
    assert lab["edge_none"][inner].all()                         # nearest the inked west side: still labelled
    c = lab["cuts"]
    mean_x = np.bincount(c.cut, c.px % PAGE[1]) / np.bincount(c.cut)
    east_side = np.abs(mean_x - 2 * WEST[1]) < 1
    assert east_side.any() and (c.weight[east_side] == 0).all()


def test_no_label_where_the_sheet_draws_nothing(tmp_path, monkeypatch):
    lots, lines = blocks_world()
    g = lotlabels.build_ground(lots, lines, SHAPE)
    drawn_cols = 3                                          # the sheet draws the blocks of columns 0-2, not 3-4
    page = np.full(PAGE, 255, np.uint8)
    for cols in GRID_COLS[:drawn_cols]:
        for rows in GRID_ROWS:
            ink_outline(page, cols, rows)
    lab, before = _labelled(tmp_path, monkeypatch, g, page)

    split = GRID_COLS[drawn_cols - 1][1] + GRID_COLS[drawn_cols][0]        # page x of the street's middle between
    x = np.arange(PAGE[1])[None, :].repeat(PAGE[0], 0)
    labelled = before > 0
    bare, inked = labelled & (x > split), labelled & (x < split)
    assert bare.any() and inked.any()
    assert (lab["weight"][bare] == 0).all() and np.array_equal(lab["weight"][inked], before[inked])
    assert not lab["edge_none"][x > split].any() and lab["edge_none"][inked].any()

    c = lab["cuts"]
    mean_x = np.bincount(c.cut, c.px % PAGE[1]) / np.bincount(c.cut)
    assert (mean_x > split).any() and (c.weight[mean_x > split] == 0).all()
    assert (c.weight[mean_x < split] == 1).all()


def test_px_on_a_cut_dropped_at_the_frame_cut_are_not_none(tmp_path, monkeypatch, ground):
    """A px on a block-edge cut the lots call for is not labelled "no block edge", though the cut was dropped for
    leaving the frame cut."""
    page = np.full(PAGE, 255, np.uint8)
    for cols in (WEST, EAST):
        ink_outline(page, cols, ROWS)
    keep = np.ones(PAGE, bool)
    keep[:, 2 * WEST[0] - R] = False          # the tip of every cut across the west block's west side
    lab, _ = _labelled(tmp_path, monkeypatch, ground, page, keep)
    top = 2 * ROWS[0] + 1                     # the page row of the cut across the west side's top grid px
    for x in (2 * WEST[0] + 2, 2 * WEST[0] + 4):                 # on that dropped cut, by the inked north side
        assert lab["weight"][top, x] > 0 and not lab["edge_none"][top, x]
    mid = 2 * 20 + 1                          # a page row mid-block
    assert lab["edge_none"][mid, 2 * WEST[1] - 12]               # on no cut, by the inked east side


def test_open_ground_past_the_blocks_loses_its_street_label_and_nothing_else_changes(tmp_path, monkeypatch):
    """A road along the west block's open side (lots on one side only, like a road along the shore) was labelled street
    by its centre line; past the blocks it is not between blocks, so it gets no label and no "no block edge"
    statement.  The street between the blocks, its weights, the block px and every block-edge cut (the west block's
    open side keeps its cuts) are as they were before the rule."""
    import dataclasses

    import shapely

    from sfzmaps.solver.hatchnet.data import Lines, street_grid
    base = lines_world()
    road = shapely.LineString([(grid.X0 + 42.0, grid.Y1), (grid.X0 + 42.0, grid.Y1 - 4 * SHAPE[0])])   # column 10
    lines = Lines(np.array([*base.geoms, road], dtype=object), np.array([*base.kinds, "ST"], dtype=object),
                  np.array([*base.streets, "SHORE RD"], dtype=object))
    new = lotlabels.build_ground(lots_world(), lines, SHAPE)
    street, weight, _ = street_grid(lines, new.block, per_type=False)
    old = dataclasses.replace(new, street=street, weight=weight, lost=np.zeros_like(new.lost))
    page = np.full(PAGE, 255, np.uint8)
    for cols in (WEST, EAST):
        ink_outline(page, cols, ROWS)
    before, _ = _labelled(tmp_path, monkeypatch, old, page)
    monkeypatch.setattr(lotlabels, "ground", lambda series: new)
    after = lotlabels.sheet_labels(("s", "s-p01"))
    by_road = (slice(2 * ROWS[0], 2 * ROWS[1]), slice(2 * 10 - 2, 2 * 10 + 4))
    assert (before["street"][by_road] == 1).all()                             # the road was labelled street
    assert (after["street"][by_road] == 255).all() and (after["weight"][by_road] == 0).all()
    assert not after["edge_none"][by_road].any()
    lost = (before["street"] == 1) & (after["street"] == 255)
    assert after["counts"]["street_label_lost"] == lost.sum() > 0
    assert (after["street"] == 1).sum() + lost.sum() == (before["street"] == 1).sum()
    kept = after["street"] == 1
    assert (before["street"][kept] == 1).all() and np.array_equal(after["weight"][kept], before["weight"][kept])
    gap = (slice(2 * ROWS[0], 2 * ROWS[1]), slice(2 * STREET[0], 2 * STREET[1]))
    assert (after["street"][gap] == 1).all()
    assert np.array_equal(after["street"] == 0, before["street"] == 0)
    assert np.array_equal(after["edge_none"], before["edge_none"] & ~lost)
    for k in ("px", "cut", "prior", "weight"):
        assert np.array_equal(getattr(after["cuts"], k), getattr(before["cuts"], k)), k
    west_side = np.bincount(after["cuts"].cut, after["cuts"].px % PAGE[1]) / np.bincount(after["cuts"].cut)
    assert ((np.abs(west_side - 2 * WEST[0]) < 1) & (after["cuts"].weight == 1)).any()     # the open side's cuts
    with np.load(tmp_path / "find" / "labels" / "s" / "s-p01.npz") as z:
        assert int(z["count_street_label_lost"]) == lost.sum()
