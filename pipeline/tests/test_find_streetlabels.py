"""The street labels on the solver's grid (`sfzmaps.find.streetlabels`): corridors between blocks weighted by the
learned half-widths, open sides, centre lines rasterised on the grid, street types.  Moved with the module from the
1921 hatching reader's tests (`test_hatchnet_data.py`, branch `hatch-experiments-archive`)."""
import cv2
import numpy as np
import pandas as pd
import shapely

from sfzmaps.find import streetlabels
from sfzmaps.solver import grid


def gap_blocks(width: int, *spans) -> np.ndarray:
    """A grid (20 rows) whose blocks are the column spans given (ends exclusive), every row: a 1-D map."""
    block = np.zeros((20, width), bool)
    for c0, c1 in spans:
        block[:, c0:c1] = True
    return block


def wline(*rc) -> shapely.LineString:
    """A centre line (EPSG:2227 ft) through the centres of grid px (row, col)."""
    return shapely.LineString([(grid.X0 + 4 * (c + 0.5), grid.Y1 - 4 * (r + 0.5)) for r, c in rc])


def centres_of(shape, *lines) -> "streetlabels.Centres":
    return streetlabels.centre_segments(np.array(lines, dtype=object), shape)


def test_corridor_between_two_blocks():
    # blocks at columns 0-9 and 20-29, a 10 px street between them, its centre line at column 14 (5 px from the west
    # block, 6 from the east): every px between the blocks is street, weighted by the learned S(d) = P(half-width > d)
    # and cut at each side's block; the half-widths learned are 5 and 6 (20 px each), so S is 1 below 5 px and 1/2 at
    # 5 px (column 19, beside the east block).  The blocks are not street; the px past the east block are ignored
    block = gap_blocks(40, (0, 10), (20, 30))
    lab = streetlabels.street_labels(block, centres_of(block.shape, wline((0, 14), (19, 14))))
    assert lab.street.dtype == np.uint8 and lab.weight.dtype == np.float16 and lab.centre.dtype == np.uint8
    assert (lab.street[:, 10:20] == 1).all()
    assert (lab.weight[:, 10:19] == 1).all() and (lab.weight[:, 19] == 0.5).all()
    assert (lab.street[block] == 0).all() and (lab.weight[block] == 1).all()
    assert (lab.street[:, 30:] == streetlabels.UNKNOWN).all() and (lab.weight[:, 30:] == 0).all()
    assert lab.centre[:, 14].all() and lab.centre.sum() == 20
    assert set(np.unique(lab.street)) <= {0, 1, streetlabels.UNKNOWN}


def shore_grid():
    """Rows 0-9 and 20-39 are blocks.  A street between them (centre row 14: 5 px to the block above, 6 below), and a
    shoreline street (centre row 42: 3 px below the block) with open water below it to the grid's edge."""
    block = np.zeros((60, 80), bool)
    block[0:10] = True
    block[20:40] = True
    return block, centres_of(block.shape, wline((14, 0), (14, 79)), wline((42, 0), (42, 79)))


def test_shoreline_open_side_is_soft():
    # every side is weighted by the learned survival S(d) = P(half-width > d): the half-widths learned are the
    # two-sided street's 5 and 6 (80 px each), so S is 1 below 5 px, 1/2 at 5 px and 0 from 6 px.  The shoreline's
    # block side is cut at its block (rows 40-41, S = 1 there); its open side follows S out to where it reaches 0, and
    # the water beyond is ignored.  An open side labels what a bounded side would
    block, centres = shore_grid()
    lab = streetlabels.street_labels(block, centres)
    assert (lab.street[40:42] == 1).all() and (lab.weight[40:42] == 1).all()
    assert (lab.street[10:20] == 1).all()
    assert (lab.weight[10:19] == 1).all() and (lab.weight[19] == 0.5).all()        # row 19: 5 px above the block
    for row in range(43, 60):
        d = row - 42
        expect = streetlabels.survival(lab.learned.pooled, np.full(80, d))
        assert np.allclose(lab.weight[row].astype(float), expect), row
    assert (lab.street[43:48] == 1).all() and (lab.weight[43:47] == 1).all() and (lab.weight[47] == 0.5).all()
    assert (lab.street[48:] == streetlabels.UNKNOWN).all() and (lab.weight[48:] == 0).all()


def test_bridge_gets_only_soft_weights():
    # a centre line with no block on either side: only the learned survival weights, out to the largest learned
    # half-width, then ignored; nothing is not-street
    learned = streetlabels.street_labels(*shore_grid()).learned
    block = np.zeros((40, 40), bool)
    lab = streetlabels.street_labels(block, centres_of(block.shape, wline((0, 20), (39, 20))), learned=learned)
    d = np.abs(np.arange(40) - 20)
    for col in range(40):
        expect = streetlabels.survival(learned.pooled, np.full(40, d[col]))
        assert np.allclose(lab.weight[:, col].astype(float), expect), col
    assert (lab.street[:, d <= 5] == 1).all() and (lab.street[:, d > 5] == streetlabels.UNKNOWN).all()
    assert not (lab.street == 0).any() and (lab.weight[:, d == 5] == 0.5).all()


def test_far_shore_parcel_does_not_make_full_corridor():
    # a centre line, open water, and a parcel 400 ft (100 px) away: the side is bounded, but its weights are still the
    # learned S(d), not 1; at 200 ft (50 px) about S(50 px), and they stop at the parcel
    learned = streetlabels.HalfWidths(pooled=np.arange(1.0, 121.0), own={}, report={})      # half-widths 1-120 px
    block = gap_blocks(130, (110, 130))
    lab = streetlabels.street_labels(block, centres_of(block.shape, wline((0, 10), (19, 10))), learned=learned)
    w = lab.weight.astype(float)
    assert np.allclose(w[:, 60], streetlabels.survival(learned.pooled, 50.0), atol=1e-3)
    assert np.allclose(streetlabels.survival(learned.pooled, 50.0), 70 / 120) and (w[:, 60] < 1).all()
    assert np.allclose(w[:, 109], streetlabels.survival(learned.pooled, 99.0), atol=1e-3)
    assert (lab.street[:, 11:110] == 1).all() and (lab.street[:, 110:] == 0).all()


def test_half_widths_learned_from_two_sided_px():
    # the distribution comes only from px bounded on both sides: not the shoreline's one side (3), and not a side whose
    # ray runs into a cross street's area before it meets a block (here up a cross street that ends 12 px away at a
    # block band), which measures the cross street's length, not this street's width
    block, centres = shore_grid()
    lab = streetlabels.street_labels(block, centres)
    assert np.array_equal(np.sort(lab.learned.pooled), np.r_[np.full(80, 5.0), np.full(80, 6.0)])
    block = np.zeros((30, 40), bool)
    block[0:3] = True                                         # a block band across the top
    for r0, r1 in ((3, 10), (20, 30)):
        block[r0:r1, 0:10] = True
        block[r0:r1, 20:40] = True
    lab = streetlabels.street_labels(block, centres_of(block.shape, wline((14, 0), (14, 39)), wline((3, 14), (29, 14))))
    assert lab.learned.pooled.size and set(np.unique(lab.learned.pooled)) <= {5.0, 6.0}
    i = int(np.flatnonzero((lab.sides.rows == 14) & (lab.sides.cols == 12))[0])
    assert np.isinf(lab.sides.width[i]).any()                 # its up side runs along the cross street


def test_no_centre_line_is_ignored():
    # a street between two blocks with no centre line today (columns 30-39) is ignored, not labelled not-street; the
    # one with a centre line (columns 10-19) is street.  With no centre line at all, every px off the blocks is ignored
    block = gap_blocks(60, (0, 10), (20, 30), (40, 50))
    lab = streetlabels.street_labels(block, centres_of(block.shape, wline((0, 14), (19, 14))))
    assert (lab.street[:, 10:20] == 1).all()
    assert (lab.street[:, 30:40] == streetlabels.UNKNOWN).all() and (lab.weight[:, 30:40] == 0).all()
    assert (lab.street[block] == 0).all()
    lab = streetlabels.street_labels(block, centres_of(block.shape))
    assert (lab.street[block] == 0).all() and (lab.street[~block] == streetlabels.UNKNOWN).all()
    assert not lab.centre.any()
    assert (lab.weight[~block] == 0).all()


def test_centre_on_a_block_is_not_street():
    # a centre line crossing today's block (a park road, a street built over): the block stays not-street and the line
    # is no centre there, so it labels nothing off the block
    block = gap_blocks(40, (0, 20))
    lab = streetlabels.street_labels(block, centres_of(block.shape, wline((0, 10), (19, 10))))
    assert (lab.street[block] == 0).all() and (lab.street[~block] == streetlabels.UNKNOWN).all()
    assert not lab.centre.any()


def test_kind_pooled_unless_its_widths_differ():
    # each street type gets its own learned half-widths only when they differ from all types' by more than the DKW 95%
    # band for its number of lines; a type seen on one line is pooled
    block, _ = shore_grid()
    lines = np.array([wline((14, 0), (14, 79)), wline((42, 0), (42, 79))], dtype=object)
    centres = streetlabels.centre_segments(lines, block.shape, kinds=np.array(["AVE", "ST"]))
    lab = streetlabels.street_labels(block, centres)
    rep = lab.learned.report
    assert rep["AVE"]["own"] is False and rep["AVE"]["lines"] == 1 and rep["AVE"]["px"] == 80
    assert rep["ST"]["own"] is False and rep["ST"]["px"] == 0 and not lab.learned.own


def test_pooled_half_widths_have_no_type_test():
    # two street types of 20 lines each, AVE's half-widths all 2 px and ST's all 8: by type, each differs from the
    # pooled ones by far more than the DKW band and keeps its own; pooled (per_type=False), neither does, and every
    # type gets the pooled half-widths
    n = 40
    kind = (np.arange(n) >= n // 2).astype(np.int64)
    centres = streetlabels.Centres(np.full((1, 1), -1, np.int32), np.tile([1.0, 0.0], (n, 1)), np.arange(n),
                                   np.arange(n), kind, ("AVE", "ST"))
    width = np.where(kind == 0, 2.0, 8.0)[:, None].repeat(2, axis=1)
    sid = np.arange(n)
    assert set(streetlabels.learn_half_widths(width, sid, centres).own) == {"AVE", "ST"}
    learned = streetlabels.learn_half_widths(width, sid, centres, per_type=False)
    assert learned.own == {}
    assert np.array_equal(learned.pooled, np.r_[np.full(n, 2.0), np.full(n, 8.0)])
    for k in centres.kinds:
        assert np.array_equal(learned.of(k), learned.pooled)


def test_centre_lines_rasterised_on_the_grid():
    # today's centre lines (EPSG:2227 ft) on the solver's 4 ft grid, segment by segment, 8-connected, one px wide,
    # wherever they run, each px knowing its segment's direction; a line off the grid is left out
    lines = [shapely.LineString([(grid.X0 + 2, grid.Y1 - 42), (grid.X0 + 158, grid.Y1 - 42)]),       # row 10
             shapely.LineString([(grid.X0 + 30, grid.Y1 - 2), (grid.X0 + 70, grid.Y1 - 158)]),        # diagonal
             shapely.LineString([(grid.X0 - 500, grid.Y1 + 500), (grid.X0 - 100, grid.Y1 + 100)])]    # off the grid
    centres = streetlabels.centre_segments(np.array(lines, dtype=object), (40, 40))
    c = centres.seg >= 0
    assert c[10].all()
    assert np.allclose(centres.direction[centres.seg[10, 0]], [1.0, 0.0])
    assert np.allclose(centres.direction[centres.seg[39, 17]], np.array([40.0, 156.0]) / np.hypot(40, 156))
    n, _ = cv2.connectedComponents(c.astype(np.uint8), connectivity=8)
    assert n == 2                                                  # one piece (and the background)
    assert c[39, 17]                                               # the diagonal reaches the bottom row at x 70 ft


def test_street_type_from_name(tmp_path):
    # a line with no st_type takes its name's last word when that is a type other lines have (OCTAVIA ST -> ST); the
    # rest (freeways, THE EMBARCADERO) stay untyped
    f = tmp_path / "streets.parquet"
    line = shapely.to_wkb(shapely.LineString([(grid.X0, grid.Y1), (grid.X0 + 40, grid.Y1)]))
    pd.DataFrame({"st_type": ["ST", None, None, None], "geometry": [line] * 4,
                  "streetname": ["A ST", "OCTAVIA ST", "HWY 101 SOUTHBOUND", "THE EMBARCADERO"]}).to_parquet(f)
    lines = streetlabels.lines_from(f)
    assert list(lines.kinds) == ["ST", "ST", streetlabels.KIND_NONE, streetlabels.KIND_NONE]
