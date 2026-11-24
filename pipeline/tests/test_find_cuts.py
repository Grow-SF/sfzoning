"""The finder's cut statements (`sfzmaps.find.cuts`): short runs of page px across the place where the lots say a line
is, each saying "a line crosses this cut" or "none does": block sides, the lines between lots of different and of the
same zoning, and the ground between lots across a street; nothing off the page or off the frame cut is in a cut."""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import shapely
from PIL import Image
from typer.testing import CliRunner

from find_fixtures import (EAST, GRID_COLS, GRID_ROWS, ROWS, SHAPE, SPLIT, STREET, WEST, Flat, blocks_world,
                           lines_world, lots_world)
from sfzmaps.find import cuts, drawn, lotlabels, pages
from sfzmaps.frame import data as fdata
from sfzmaps.solver import grid
from sfzmaps.solver.hatchnet import data

CHECKED_IN = Path(__file__).parents[1] / "inputs" / "find" / "label_sources.csv"
FT_PER_PX = 2.0                                     # two page px to a grid px each way
PAGE = (SHAPE[0] * 2, SHAPE[1] * 2)                 # the page covers the whole fixture grid
SIGMA = 1.5
R = 6                                               # ceil(4 * SIGMA)
FIRST = np.array(["A", "A", "B", "B"])              # lots 0, 1 the west block, 2, 3 the east: the blocks differ
VARIANT = np.array(["A", "B", "B", "B"])            # the west block's two lots differ too


@pytest.fixture(scope="module")
def ground():
    return lotlabels.build_ground(lots_world(), lines_world(), SHAPE)


def every(c: cuts.Cuts):
    """Each cut's (rows, cols, prior) in page px."""
    rows, cols = np.divmod(c.px, PAGE[1])
    for i in range(len(c.answer)):
        on = c.cut == i
        yield rows[on], cols[on], c.prior[on]


def page_box(cols: tuple[int, int], rows: tuple[int, int]) -> np.ndarray:
    """The page px over grid columns and rows (ends exclusive)."""
    m = np.zeros(PAGE, bool)
    m[rows[0] * 2:rows[1] * 2, cols[0] * 2:cols[1] * 2] = True
    return m


def test_an_edge_cut_crosses_each_block_side(ground):
    c = cuts.edge_cuts(ground, Flat(FT_PER_PX), PAGE, np.ones(PAGE, bool), SIGMA)
    sides_x = {2 * WEST[0], 2 * WEST[1], 2 * EAST[0], 2 * EAST[1]}        # page x of the blocks' west and east sides
    sides_y = {2 * ROWS[0], 2 * ROWS[1]}                                  # page y of their north and south sides
    assert len(c.answer) == 2 * 2 * ((WEST[1] - WEST[0]) + (ROWS[1] - ROWS[0]))    # one per grid px along each side
    assert (c.answer == cuts.EDGE).all() and c.says.all()
    assert c.px.dtype == np.int64 and c.cut.dtype == np.int32 and c.prior.dtype == np.float32
    for rows, cols, prior in every(c):
        assert len(rows) == 2 * R + 1
        assert np.argmax(prior) == R and prior[R] == 1 and prior[0] < 1e-3 and prior[-1] < 1e-3
        if len(set(rows)) == 1:            # across a west or east side: along x, centred on the side
            assert np.array_equal(np.abs(np.diff(cols)), np.ones(2 * R, int)) and cols[R] in sides_x
        else:                              # across a north or south side: along y
            assert len(set(cols)) == 1 and np.array_equal(np.abs(np.diff(rows)), np.ones(2 * R, int))
            assert rows[R] in sides_y


def _across_the_lot_line(c: cuts.Cuts, cols: tuple[int, int]) -> np.ndarray:
    """The says of the cuts centred on the line between a block's two lots (page y 2 * SPLIT) within grid `cols`."""
    out = []
    for i, (rows, cc, prior) in enumerate(every(c)):
        mid = np.argmax(prior)
        if len(set(cc)) == 1 and rows[mid] == 2 * SPLIT and cols[0] * 2 <= cc[0] < cols[1] * 2 \
                and len(rows) == 2 * R + 1:
            out.append(c.says[i])
    return np.array(out)


def test_touching_lots_same_code_say_none_and_different_say_one(ground):
    keep = np.ones(PAGE, bool)
    first = cuts.border_cuts(ground.lot, FIRST, Flat(FT_PER_PX), PAGE, keep, SIGMA)
    variant = cuts.border_cuts(ground.lot, VARIANT, Flat(FT_PER_PX), PAGE, keep, SIGMA)
    assert (first.answer == cuts.BORDER).all()
    west_first, west_variant = _across_the_lot_line(first, WEST), _across_the_lot_line(variant, WEST)
    assert len(west_first) == WEST[1] - WEST[0] and not west_first.any()
    assert len(west_variant) == WEST[1] - WEST[0] and west_variant.all()
    east = _across_the_lot_line(variant, EAST)          # B and B in both
    assert len(east) == EAST[1] - EAST[0] and not east.any()


def test_a_street_cut_spans_the_street(ground):
    c = cuts.border_cuts(ground.lot, FIRST, Flat(FT_PER_PX), PAGE, np.ones(PAGE, bool), SIGMA)
    street = np.arange(2 * STREET[0], 2 * STREET[1])        # the street's page columns
    found = 0
    for i, (rows, cols, prior) in enumerate(every(c)):
        if len(set(rows)) != 1 or not (2 * ROWS[0] <= rows[0] < 2 * ROWS[1]):
            continue
        if np.isin(street, cols).all():
            found += 1
            assert (prior[np.isin(cols, street)] == 1).all()
            assert c.says[i]                                # A on the west, B on the east
            assert prior[0] < 1e-3 and prior[-1] < 1e-3     # the Gaussian fall-off past both lots
    assert found == ROWS[1] - ROWS[0]                       # one along each grid row of the blocks
    same = cuts.border_cuts(ground.lot, np.array(["A"] * 4), Flat(FT_PER_PX), PAGE, np.ones(PAGE, bool), SIGMA)
    assert len(same.answer) and not same.says.any()


def test_unknown_code_makes_no_statement(ground):
    codes = FIRST.copy()
    codes[0] = ""
    c = cuts.border_cuts(ground.lot, codes, Flat(FT_PER_PX), PAGE, np.ones(PAGE, bool), SIGMA)
    lot0 = page_box(WEST, (ROWS[0], SPLIT))
    assert len(c.answer) and c.says.any()
    assert not lot0.ravel()[c.px].any()


def test_a_cut_leaving_the_frame_cut_is_dropped(ground):
    keep = np.ones(PAGE, bool)
    keep[:, 2 * STREET[0] + 2:2 * STREET[1] - 2] = False        # a strip down the middle of the street
    c = cuts.Cuts.joined(cuts.edge_cuts(ground, Flat(FT_PER_PX), PAGE, keep, SIGMA),
                         cuts.border_cuts(ground.lot, FIRST, Flat(FT_PER_PX), PAGE, keep, SIGMA))
    assert keep.ravel()[c.px].all()
    assert (c.answer == cuts.EDGE).sum() == 2 * 2 * ((WEST[1] - WEST[0]) + (ROWS[1] - ROWS[0])) - 2 * 40
    for rows, cols, prior in every(c):          # dropped whole: no cut is cut short
        assert np.argmax(prior) >= R and np.argmax(prior[::-1]) >= R


def test_doubling_the_cut_length_changes_nothing_that_matters(ground):
    keep = np.ones(PAGE, bool)
    for make in (lambda r: cuts.edge_cuts(ground, Flat(FT_PER_PX), PAGE, keep, SIGMA, r),
                 lambda r: cuts.border_cuts(ground.lot, FIRST, Flat(FT_PER_PX), PAGE, keep, SIGMA, r)):
        one, two = make(R), make(2 * R)
        assert len(one.answer) == len(two.answer) and np.array_equal(one.says, two.says)
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
    but a `margin` px border), the label sources `sources`, a flat placement and a spread of SIGMA."""
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
    (tmp_path / "label_sources.csv").write_text("series,era,lots,zoning_borders\n" + sources)
    monkeypatch.setattr(fdata, "WORK", work)
    monkeypatch.setattr(fdata, "OUT", frame)
    monkeypatch.setattr(pages, "SOURCES", tmp_path / "label_sources.csv")
    monkeypatch.setattr(pages, "OUT", tmp_path / "find")
    monkeypatch.setattr(lotlabels, "placement", lambda sid, which="warp": Flat(FT_PER_PX))
    monkeypatch.setattr(cuts, "spread_px", lambda sid, which="warp": SIGMA)
    return cut == 0


def test_hatched_series_say_no_zoning_border_everywhere(tmp_path, monkeypatch):
    keep = _sheet_world(tmp_path, monkeypatch, "s,hatched-grey,,none\n")
    lab = lotlabels.sheet_labels(("s", "s-p01"))
    assert np.array_equal(lab["border_none"], keep) and keep.any() and (~keep).any()
    assert not (lab["cuts"].answer == cuts.BORDER).any()
    assert not lab["edge_none"].any() and (lab["street"] == 255).all()
    with np.load(tmp_path / "find" / "labels" / "s" / "s-p01.npz") as z:
        assert np.array_equal(z["border_none"], keep) and str(z["placement"]) == "warp" and float(z["sigma"]) == SIGMA


def test_table_series_label_edges_and_zoning_borders(tmp_path, monkeypatch, ground):
    page = np.full(PAGE, 255, np.uint8)
    for cols in (WEST, EAST):
        ink_outline(page, cols, ROWS)
    keep = _sheet_world(tmp_path, monkeypatch, "s,lined-1bit,year,zoning\n", page)
    monkeypatch.setattr(lotlabels, "ground", lambda series: ground)
    monkeypatch.setattr(lotlabels, "series_codes", lambda series: FIRST)
    lab = lotlabels.sheet_labels(("s", "s-p01"))
    c = lab["cuts"]
    edge, border = c.answer == cuts.EDGE, c.answer == cuts.BORDER
    assert edge.any() and border.any() and c.says[border].any() and not c.says[border].all()
    on_edge = np.zeros(PAGE[0] * PAGE[1], bool)
    on_edge[c.px[edge[c.cut]]] = True
    on_border = np.zeros(PAGE[0] * PAGE[1], bool)
    on_border[c.px[border[c.cut]]] = True
    labelled = (lab["street"] != 255).ravel()
    assert np.array_equal(lab["edge_none"].ravel(), labelled & ~on_edge)
    lots = (page_box(WEST, ROWS) | page_box(EAST, ROWS)).ravel() & keep.ravel()
    assert np.array_equal(lab["border_none"].ravel(), lots & ~on_border)


def test_missing_special_use_is_plain_ground_and_missing_height_unknown(monkeypatch):
    table = gpd.GeoDataFrame({"zoning": ["RH-1", ""], "height_lim": [" 40-X ", None], "sud": [None, " BERNAL"]},
                             geometry=[shapely.box(0, 0, 1, 1)] * 2, crs="EPSG:2227")
    monkeypatch.setattr(lotlabels, "load_table", lambda: table)
    monkeypatch.setattr(pages, "SOURCES", CHECKED_IN)
    assert list(lotlabels.series_codes("2000-digital-height")) == ["40-X", ""]
    assert list(lotlabels.series_codes("2000-digital-su")) == ["(none)", "BERNAL"]
    assert list(lotlabels.series_codes("2000-digital-zoning")) == ["RH-1", ""]
    assert lotlabels.series_codes("1980-1990-zoning") is None


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
    c = cuts.Cuts(px.astype(np.int64), cut, np.ones(len(px), np.float32), np.zeros(2, np.uint8), np.ones(2, bool),
                  np.array([1, 0], np.float32), np.zeros((2, 2), np.int32), np.zeros(0, np.int64))
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
    used, counts = drawn.use(c, lum, np.ones(SHAPE, bool), np.full(PAGE, 255, np.uint8), np.zeros(PAGE, np.float32))
    west_side = np.bincount(c.cut, c.px % PAGE[1] == 2 * WEST[0]) > 0
    assert (used.weight[~west_side] == 0).all() and counts["edge_dropped"] == int((~west_side).sum())


def test_an_edge_cut_with_ink_is_used(ground):
    c = cuts.edge_cuts(ground, Flat(FT_PER_PX), PAGE, np.ones(PAGE, bool), SIGMA)
    page = np.full(PAGE, 255, np.uint8)
    for cols in (WEST, EAST):
        ink_outline(page, cols, ROWS)
    used, counts = drawn.use(c, page.astype(np.float32) / 255, np.ones(SHAPE, bool), np.full(PAGE, 255, np.uint8),
                             np.zeros(PAGE, np.float32))
    assert (used.weight == 1).all() and counts["edge_used"] == len(c.answer) and counts["edge_dropped"] == 0


def _labelled(tmp_path, monkeypatch, ground, page, codes, keep=None):
    """`sheet_labels` of a zoning-table series with lots `ground`, page `page` and lot codes `codes`, the frame cut
    `keep` (None: room for the block-edge cuts of `blocks_world`'s top row); and the street-or-not weights before the
    rule."""
    keep = _sheet_world(tmp_path, monkeypatch, "s,lined-1bit,year,zoning\n", page, margin=2, keep=keep)
    monkeypatch.setattr(lotlabels, "ground", lambda series: ground)
    monkeypatch.setattr(lotlabels, "series_codes", lambda series: codes)
    before = lotlabels.page_ground(ground, Flat(FT_PER_PX), PAGE, keep).weight
    return lotlabels.sheet_labels(("s", "s-p01")), before


def test_ground_by_a_bare_block_side_has_no_label(tmp_path, monkeypatch, ground):
    page = np.full(PAGE, 255, np.uint8)
    ink_outline(page, WEST, ROWS, "wn")                          # the west block: its west and north sides only
    ink_outline(page, EAST, ROWS)
    lab, before = _labelled(tmp_path, monkeypatch, ground, page, FIRST)
    rows = slice(2 * ROWS[0] + 20, 2 * ROWS[1] - 20)             # well away from the north and south sides
    by_ink = (rows, slice(2 * WEST[0] + 1, 2 * WEST[0] + 7))     # block px by the inked west side
    by_bare = (rows, slice(2 * WEST[1] - 7, 2 * STREET[0] + 3))  # block and street px by the bare east side
    by_east = (rows, slice(2 * STREET[1] - 3, 2 * STREET[1]))    # street px by the east block's inked west side
    for box in (by_ink, by_east):
        assert (before[box] > 0).all() and np.array_equal(lab["weight"][box], before[box])
    assert (before[by_bare] > 0).all() and (lab["weight"][by_bare] == 0).all()
    assert not lab["edge_none"][by_bare].any() and not lab["border_none"][by_bare].any()
    inner = (slice(2 * ROWS[0] + 12, 2 * SPLIT - 12), slice(2 * WEST[0] + 6, 2 * WEST[0] + 14))
    assert lab["border_none"][inner].all()                       # nearest the inked west side: still labelled
    c = lab["cuts"]
    mean_x = np.bincount(c.cut, c.px % PAGE[1]) / np.bincount(c.cut)
    edge = c.answer == cuts.EDGE
    east_side = edge & (np.abs(mean_x - 2 * WEST[1]) < 1)
    assert east_side.any() and (c.weight[east_side] == 0).all()


def test_a_crossing_cut_needs_ink_on_it(tmp_path, monkeypatch, ground):
    page = np.full(PAGE, 255, np.uint8)
    for cols in (WEST, EAST):
        ink_outline(page, cols, ROWS)
    weights = []
    for line in (True, False):
        p = page.copy()
        if line:
            p[2 * SPLIT, 2 * WEST[0]:2 * WEST[1]] = 0            # the lot line between the west block's two lots
        lab, _ = _labelled(tmp_path / str(line), monkeypatch, ground, p, VARIANT)
        c = lab["cuts"]
        rows, cols = np.divmod(c.px, PAGE[1])
        n = len(c.answer)
        length = np.bincount(c.cut, minlength=n)
        mid_y = np.bincount(c.cut, rows, minlength=n) / length
        mid_x = np.bincount(c.cut, cols, minlength=n) / length
        on_line = (c.answer == cuts.BORDER) & (length == 2 * R + 1) & (np.abs(mid_y - 2 * SPLIT) < 1)
        west = on_line & (mid_x < 2 * WEST[1]) & c.says           # A beside B: a zoning border crosses
        east = on_line & (mid_x > 2 * EAST[0]) & ~c.says          # B beside B: none does
        assert west.sum() == WEST[1] - WEST[0] and east.sum() == EAST[1] - EAST[0]
        weights.append((c.weight[west], c.weight[east]))
    (west_line, east_line), (west_bare, east_bare) = weights
    assert (west_line == 1).all() and (west_bare == 0).all()
    assert (east_line == 1).all() and (east_bare == 1).all()


def test_no_label_where_the_sheet_draws_nothing(tmp_path, monkeypatch):
    lots, lines = blocks_world()
    g = lotlabels.build_ground(lots, lines, SHAPE)
    drawn_cols = 3                                          # the sheet draws the blocks of columns 0-2, not 3-4
    page = np.full(PAGE, 255, np.uint8)
    for cols in GRID_COLS[:drawn_cols]:
        for rows in GRID_ROWS:
            ink_outline(page, cols, rows)
    codes = np.array(["A"] * 4 * drawn_cols + ["B"] * 4 * (len(GRID_COLS) - drawn_cols))
    lab, before = _labelled(tmp_path, monkeypatch, g, page, codes)

    split = GRID_COLS[drawn_cols - 1][1] + GRID_COLS[drawn_cols][0]        # page x of the street's middle between
    x = np.arange(PAGE[1])[None, :].repeat(PAGE[0], 0)
    labelled = before > 0
    bare, inked = labelled & (x > split), labelled & (x < split)
    assert bare.any() and inked.any()
    assert (lab["weight"][bare] == 0).all() and np.array_equal(lab["weight"][inked], before[inked])
    assert not lab["edge_none"][x > split].any() and not lab["border_none"][x > split].any()
    assert lab["edge_none"][inked].any() and lab["border_none"][inked].any()

    c = lab["cuts"]
    mean_x = np.bincount(c.cut, c.px % PAGE[1]) / np.bincount(c.cut)
    edge, border = c.answer == cuts.EDGE, c.answer == cuts.BORDER
    assert (edge & (mean_x > split)).any() and (c.weight[edge & (mean_x > split)] == 0).all()
    assert (c.weight[edge & (mean_x < split)] == 1).all()
    touching_bare = border & (c.says | (mean_x > split))   # A beside B, and B beside B: each touches a bare block
    assert touching_bare.any() and (c.weight[touching_bare] == 0).all()
    assert (c.weight[border & ~c.says & (mean_x < split)] == 1).all()


def test_a_street_cut_needs_street_between_its_lots(tmp_path, monkeypatch):
    lots, lines = blocks_world()
    one = data.Lines(lines.geoms[:1], lines.kinds[:1], lines.streets[:1])     # the street between columns 0 and 1 only
    g = lotlabels.build_ground(lots, one, SHAPE)
    page = np.full(PAGE, 255, np.uint8)
    for cols in GRID_COLS:
        for rows in GRID_ROWS:
            ink_outline(page, cols, rows)
    lab, _ = _labelled(tmp_path, monkeypatch, g, page, np.array(["A"] * len(lots)))
    c = lab["cuts"]
    rows, cols = np.divmod(c.px, PAGE[1])
    start = np.r_[0, np.cumsum(np.bincount(c.cut))[:-1]]                  # each cut's first px
    x0, x1 = np.minimum.reduceat(cols, start), np.maximum.reduceat(cols, start)
    across = (c.answer == cuts.BORDER) & (np.minimum.reduceat(rows, start) == np.maximum.reduceat(rows, start))
    lined = across & (x0 < 2 * GRID_COLS[0][1]) & (x1 >= 2 * GRID_COLS[1][0])       # across the street with a line
    bare = across & (x0 >= 2 * GRID_COLS[1][0]) & (x0 < 2 * GRID_COLS[1][1]) & (x1 >= 2 * GRID_COLS[2][0])
    assert lined.any() and bare.any()                                                # (bare: across the gap)
    assert (c.weight[lined] == 1).all() and (c.weight[bare] == 0).all()
    assert lab["counts"]["none_dropped_no_street_between"] >= int(bare.sum())


def test_px_on_a_cut_dropped_at_the_frame_cut_are_not_none(tmp_path, monkeypatch, ground):
    page = np.full(PAGE, 255, np.uint8)
    for cols in (WEST, EAST):
        ink_outline(page, cols, ROWS)
    keep = np.ones(PAGE, bool)
    keep[:, 2 * WEST[0] - R] = False          # the tip of every cut across the west block's west side
    keep[:, 2 * EAST[0] + 1 + R] = False      # and of every street cut between the blocks
    lab, _ = _labelled(tmp_path, monkeypatch, ground, page, FIRST, keep)
    top = 2 * ROWS[0] + 1                     # the page row of the cut across the west side's top grid px
    for x in (2 * WEST[0] + 2, 2 * WEST[0] + 4):                 # on that dropped cut, by the inked north side
        assert lab["weight"][top, x] > 0 and not lab["edge_none"][top, x]
    mid = 2 * 20 + 1                          # a page row of a dropped street cut, mid-block
    for x in range(2 * WEST[1] - 6, 2 * WEST[1] - 2):            # on it, inside the west block by its east side
        assert lab["weight"][mid, x] > 0 and not lab["border_none"][mid, x]
    assert lab["edge_none"][mid, 2 * WEST[1] - 12] and lab["border_none"][mid, 2 * WEST[1] - 12]   # on no cut
