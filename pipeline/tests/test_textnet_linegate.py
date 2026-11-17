"""The text-vs-line gate (`textnet.linegate`): its measures on a drawn word crossed by a line, with the page's quarter,
blocks and district-border weight given by a fake geo; the weight's own maths on drawn cracks; and, where the data
are, the page -> grid path checked against the merged page's grid -> page path."""
import dataclasses

import cv2
import numpy as np
import pytest

from sfzmaps import config
from sfzmaps.solver import grid, page, paths
from sfzmaps.solver.textnet import data, linegate, linerules

H, W = 160, 320
FT_PER_PX = 4.0           # the run rule's length is then 50 px
TEXT, ORG = "RH-1", (110, 92)
LINE_ROW = 10             # the line's row below the glyphs' top: there the run rule's px are exactly the line's,
                          # as far as M2's base can tell (the fixture checks it)
NE = paths.QUARTERS.index("ne")
SW = paths.QUARTERS.index("sw")
REAL_SHEET = paths.SHEETS[1]
REAL_DATA = (data.warp_dir(REAL_SHEET) / "transform.json").is_file() and \
    (config.WORK_DIR / paths.SERIES / REAL_SHEET / "page.png").is_file()


@dataclasses.dataclass
class FakeGeo:
    """A geo of fixed page rasters: the quarter (index into paths.QUARTERS), the block, the border weight, the
    answered labels' px and the context weight (the border weight beyond the cut)."""
    quarters: np.ndarray
    blocks: np.ndarray
    weight: np.ndarray
    answered_px: np.ndarray
    context: np.ndarray

    def quarter(self, sd):
        return self.quarters

    def block(self, sd):
        return self.blocks

    def border_weight(self, sd):
        return self.weight

    def answered(self, sd):
        return self.answered_px

    def context_weight(self, sd):
        return self.context


@dataclasses.dataclass
class Crossed:
    sd: data.SheetData
    geo: FakeGeo
    ink: np.ndarray
    glyph: np.ndarray
    line: np.ndarray
    box: np.ndarray
    glyph_far_from_line_px: int

    def all_box_text(self, sheet: str) -> np.ndarray:
        """A detector calling every px of every word box text, and nothing else."""
        return np.where(self.box, 255, 0).astype(np.uint8)


@pytest.fixture
def crossed_word() -> Crossed:
    """One zoning code drawn with cv2.putText, its word box (the glyphs' bounds and 3 px), and a 4 px line across the
    whole page through the letters; one Sol point on the word.  The geo: every px in NE, the page's left and right
    halves two blocks, border weight 1 on the line's px (a district line) and 0 elsewhere, the word answered."""
    g = np.zeros((H, W), np.uint8)
    cv2.putText(g, TEXT, ORG, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1, 2, cv2.LINE_8)
    glyph = g.view(bool)
    ys, xs = np.nonzero(glyph)
    row = int(ys.min()) + LINE_ROW
    ln = np.zeros((H, W), np.uint8)
    cv2.line(ln, (0, row), (W - 1, row), 1, 4)
    line = ln.view(bool)
    ink = glyph | line
    x0, y0, x1, y1 = int(xs.min()) - 3, int(ys.min()) - 3, int(xs.max()) + 3, int(ys.max()) + 3
    word = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.int32)
    b = np.zeros((H, W), np.uint8)
    cv2.fillPoly(b, [word[0].reshape(-1, 1, 2)], 1)
    box = b.view(bool)
    near_line = cv2.dilate(ln, np.ones((3, 3), np.uint8)) > 0
    rule_near = cv2.dilate(linerules.line_px(ink, FT_PER_PX).view(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert np.array_equal(ink & box & ~near_line, ink & box & ~rule_near), "the run rule no longer finds this line alone"
    sd = data.SheetData(series=paths.SERIES, sheet="2000-digital-zoning-p99", height=H, width=W,
                        ink_bits=np.packbits(ink, axis=1), words=word, kinds=np.array([0], np.uint8),
                        angles=np.array([0], np.int16), other=np.zeros((0, 4, 2), np.int32),
                        points=np.array([[x0 + 5, y0 + 5]], np.int32), world=np.zeros((0, 0, 2)),
                        ft_per_px=FT_PER_PX)
    blocks = np.zeros((H, W), np.int32)
    blocks[:, W // 2:] = 1
    geo = FakeGeo(np.full((H, W), NE, np.uint8), blocks, line.astype(np.float32), box.copy(),
                  np.zeros((H, W), np.float32))
    return Crossed(sd, geo, ink, glyph, line, box, int((ink & box & ~near_line).sum()))


def test_line_ink_crossing_a_word_counts_against_a_detector_that_calls_it_text(crossed_word):
    r, _ = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), crossed_word.geo, boot=10)
    assert r["m1b_in_box"]["called_05"] == 1.0
    assert r["m1b_far"]["called_05"] == 0.0 and r["m1b_far"]["px"] > 0
    assert r["m2"]["px"] == crossed_word.glyph_far_from_line_px     # counted from the fixture's own masks
    assert r["m2"]["called_05"] == 1.0
    assert r["m2_by_kind"]["zoning code"]["px"] == r["m2"]["px"] == r["m2_answered"]["px"]
    assert r["m2_by_kind"]["street name"]["px"] == 0
    assert r["overlap_band"]["px"] > 0
    assert r["overlap_band"]["px"] + r["m2"]["px"] == int((crossed_word.ink & crossed_word.box).sum())
    assert r["m1c"]["px"] == int((crossed_word.ink & ~crossed_word.box).sum())
    assert r["m1c"]["called_05"] == 0.0


def test_border_ink_inside_a_box_is_split_out(crossed_word):
    r, _ = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), crossed_word.geo, boot=10)
    assert r["m1a_in_box"]["px"] > 0 and r["m1a"]["px"] >= r["m1a_in_box"]["px"]
    assert r["m1a_in_box"]["px"] == float((crossed_word.line & crossed_word.box).sum())
    assert r["m1a"]["px"] == pytest.approx(r["m1a_in_box"]["px"] + r["m1a_near"]["px"] + r["m1a_far"]["px"])
    assert r["m1a_in_box"]["called_05"] == 1.0 and r["m1a_far"]["called_05"] == 0.0


def test_the_weight_beyond_the_cut_is_reported_apart(crossed_word):
    # the profile's tail weighting the letters (label ink near a district border) goes to the context measures alone
    letters = crossed_word.glyph & ~crossed_word.line
    geo = dataclasses.replace(crossed_word.geo, context=np.where(letters, 0.5, 0.0).astype(np.float32))
    r, _ = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), geo, boot=10)
    base, _ = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), crossed_word.geo, boot=10)
    assert all(r[m] == base[m] for m in linegate.M1A)
    assert r["m1a_in_box_context"]["px"] == pytest.approx(0.5 * (letters & crossed_word.box).sum())
    assert r["m1a_context"]["called_05"] == 1.0
    assert base["m1a_context"]["px"] == 0.0 and base["m1a_context"]["called_05"] is None


def test_m1a_weighs_each_px_by_the_border_weight(crossed_word):
    geo = dataclasses.replace(crossed_word.geo, weight=np.where(crossed_word.line, 0.25, 0.0).astype(np.float32))
    r, _ = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), geo, boot=10)
    assert r["m1a"]["px"] == pytest.approx(0.25 * crossed_word.line.sum())
    in_box = (crossed_word.line & crossed_word.box).sum()
    assert r["m1a"]["called_05"] == pytest.approx(in_box / crossed_word.line.sum())


def test_only_the_asked_quarters_are_scored(crossed_word):
    quarters = np.full((H, W), NE, np.uint8)
    quarters[:, W // 2:] = SW
    geo = dataclasses.replace(crossed_word.geo, quarters=quarters)
    outside = crossed_word.ink & ~crossed_word.box
    r, _ = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), geo, boot=10)
    assert r["m1c"]["px"] == int(outside[:, :W // 2].sum())
    r, _ = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne", "sw"), geo, boot=10)
    assert r["m1c"]["px"] == int(outside.sum())


def test_a_missing_chance_file_stops_the_gate(tmp_path, crossed_word):
    with pytest.raises(FileNotFoundError, match=crossed_word.sd.sheet):
        linegate.gate_sheets([crossed_word.sd], linegate.folder_source(tmp_path, "flat"), ("ne",), crossed_word.geo)


def test_a_chance_covering_part_of_the_sheet_stops_the_gate(tmp_path, crossed_word):
    sheet = crossed_word.sd.sheet
    (tmp_path / sheet).mkdir()
    cv2.imwrite(str(tmp_path / sheet / "chance.png"), np.zeros((H, W - 8), np.uint8))
    with pytest.raises(ValueError, match=sheet):
        linegate.gate_sheets([crossed_word.sd], linegate.folder_source(tmp_path, "nested"), ("ne",), crossed_word.geo)


def test_paired_difference_of_identical_runs_is_zero(tmp_path, crossed_word):
    r, t = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), crossed_word.geo, boot=10)
    linegate.write(r, t, tmp_path / "a")
    linegate.write(r, t, tmp_path / "b")
    p = linegate.paired([tmp_path / "a"], [tmp_path / "b"], "m1b_in_box", boot=50)
    assert p["diff"] == 0 and p["lo"] == 0 and p["hi"] == 0


def test_paired_difference_is_the_second_arm_less_the_first(tmp_path, crossed_word):
    r, t = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), crossed_word.geo, boot=10)
    linegate.write(r, t, tmp_path / "calls")
    r, t = linegate.gate_sheets([crossed_word.sd], lambda s: np.zeros((H, W), np.uint8), ("ne",), crossed_word.geo,
                                boot=10)
    linegate.write(r, t, tmp_path / "none")
    p = linegate.paired([tmp_path / "calls"], [tmp_path / "none"], "m1b_in_box", boot=50)
    assert p["diff"] == -1.0 and p["lo"] == -1.0 and p["hi"] == -1.0
    assert (tmp_path / "none" / "scores.md").read_text().count("m1b_in_box") >= 1


def test_paired_stops_on_runs_scored_on_different_px(tmp_path, crossed_word):
    r, t = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), crossed_word.geo, boot=10)
    linegate.write(r, t, tmp_path / "a")
    geo = dataclasses.replace(crossed_word.geo, weight=np.where(crossed_word.line, 0.25, 0.0).astype(np.float32))
    r, t = linegate.gate_sheets([crossed_word.sd], crossed_word.all_box_text, ("ne",), geo, boot=10)
    linegate.write(r, t, tmp_path / "b")
    with pytest.raises(ValueError, match="px differ"):
        linegate.paired([tmp_path / "a"], [tmp_path / "b"], "m1a", boot=10)
    assert linegate.paired([tmp_path / "a"], [tmp_path / "b"], "m1b_in_box", boot=10)["diff"] == 0


def test_the_weight_ends_where_the_lines_own_excess_ends():
    # falling to the far-field level (the median of the outer third, d = 10 .. 14): ends at the first d at or below it
    falls = [0.74, 0.72, 0.64, 0.46, 0.27, 0.13, 0.06, 0.03, 0.019, 0.016, 0.013, 0.011, 0.024, 0.025, 0.028]
    assert linegate.weight_cut(np.array(falls)) == (8, 0.024)
    # falling to a dip, then a second feature's bump (block outlines across a street): ends at the dip, though the
    # far-field level (-0.01) is reached only past the bump
    bump = [0.60, 0.59, 0.53, 0.37, 0.17, 0.03, 0.001, 0.013, 0.056, 0.10, 0.07, 0.04, -0.01, -0.011, -0.012]
    assert linegate.weight_cut(np.array(bump)) == (6, -0.01)
    # the table: w below the cut, the same formula beyond it as context; a 3 px line and, at offset +3 on every crack,
    # another feature's ink
    J = 4
    prof = np.zeros((8, 2 * J + 1), bool)
    prof[:4, J - 1:J + 2] = True
    prof[:4, J + 3] = True
    touch, district = np.ones(8, bool), np.r_[np.ones(4, bool), np.zeros(4, bool)]
    t = linegate.weight_table(prof, touch, district, ~district, J)["touch"]
    assert t["cut"] == 2 and t["w"] == [1.0, 1.0, 0.0, 0.0, 0.0] and t["w_context"] == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_the_border_weight_is_the_nearest_district_cracks_excess_ink_by_distance():
    J = 4
    # ink at offsets 0 and +1 on every district crack, at -1 on half of them; a quarter of the other cracks inked
    district = np.zeros((4, 2 * J + 1), bool)
    district[:, J] = district[:, J + 1] = True
    district[:2, J - 1] = True
    other = np.zeros((4, 2 * J + 1), bool)
    other[0] = True
    shares = linegate.ink_shares(district, J)
    assert shares.tolist() == [1.0, 0.75, 0.0, 0.0, 0.0]
    w = linegate.excess_weight(shares, linegate.ink_shares(other, J))
    assert w.tolist() == pytest.approx([0.75, 2 / 3, 0.0, 0.0, 0.0])
    # a touching district border down column 10 (cracks between columns 9 and 10) and one across a street down
    # column 30: each px takes the table of the border nearest it, by its distance in px, and 0 past J
    rows = np.arange(20)
    my = np.r_[rows + 0.5, rows + 0.5]
    mx = np.r_[np.full(20, 10.0), np.full(20, 30.0)]
    touch = np.r_[np.ones(20, bool), np.zeros(20, bool)]
    w_touch, w_across = np.array([1.0, 0.8, 0.6, 0.4, 0.2]), np.array([0.9, 0.5, 0.1, 0.0, 0.0])
    got = linegate.district_weight((20, 40), my, mx, touch, w_touch, w_across)
    want = np.zeros(40)
    for d in range(J + 1):
        want[[10 - d, 10 + d]] = w_touch[d]
        want[[30 - d, 30 + d]] = w_across[d]
    assert got.dtype == np.float32 and np.allclose(got, want[None, :])


@pytest.mark.skipif(not REAL_DATA, reason="needs pipeline/data")
def test_page_to_grid_round_trip_within_one_px():
    """10,000 random ink px of one 2000 sheet: page -> grid by CityGeo's path, then the grid px back to its page px
    by the merged page's own path (`page.sheet_px`, `to_pixel`): within 1 px each way."""
    transform = data.load_transform(REAL_SHEET)
    gray = data.load_page(REAL_SHEET)
    world = data.placement_samples(transform, gray.shape)
    ys, xs = np.nonzero(gray < grid.INK_BELOW)
    pick = np.random.default_rng(0).choice(len(ys), 20000, replace=False)
    ys, xs = ys[pick], xs[pick]
    cols, rows = np.empty(len(ys), np.int64), np.empty(len(ys), np.int64)
    for i, (y, x) in enumerate(zip(ys, xs)):
        px, py = data._world_at(world, np.array([y]), np.array([x]))
        c, r = linegate.grid_px(px, py)
        cols[i], rows[i] = c[0, 0], r[0, 0]
    on = (cols >= 0) & (cols < grid.WIDTH) & (rows >= 0) & (rows < grid.HEIGHT)
    ys, xs, cols, rows = ys[on][:10000], xs[on][:10000], cols[on][:10000], rows[on][:10000]
    assert len(ys) == 10000
    back_y, back_x = np.zeros(len(ys), np.int64), np.zeros(len(ys), np.int64)
    seen = np.zeros(len(ys), bool)
    box = (int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1)
    for rs, cs, _, yi, xi in page.sheet_px(transform, np.ones(gray.shape, bool), (grid.HEIGHT, grid.WIDTH), box):
        mine = (rows >= rs.start) & (rows < rs.stop)
        r, c = rows[mine] - rs.start, cols[mine] - cs.start
        back_y[mine], back_x[mine] = yi[r, c], xi[r, c]
        seen |= mine
    assert seen.all()
    assert np.abs(back_y - ys).max() <= 1 and np.abs(back_x - xs).max() <= 1
