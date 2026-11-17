"""The learned box model (`wordread.boxmodel`): its targets, its held-out rule, its joined neighbours and a short CPU
fit, on synthetic pages."""
import itertools

import numpy as np
import pandas as pd
import pytest
import torch

from sfzmaps.solver.codereader import labels as L
from sfzmaps.solver.textnet import boxes as textboxes
from sfzmaps.solver.textread import crops as K
from sfzmaps.solver.wordread import boxmodel as B

SHEET = "s1"
NO_TOUCH = {t: False for t in B.TOUCH}


def never_held(sheet, window, shape):
    return dict(NO_TOUCH)


def word(k, x0, y0, x1, y1, angle=0.0, quarter=0, sheet=SHEET, **touch):
    """A Terra word as `terra_frame` gives it: its axis box, rectangle (c corners), crop geometry and held flags."""
    rect = textboxes.word_rect((x0, y0, x1, y1), angle)
    row = {"sheet": sheet, "word": k, "kind": "zoning code", "text": f"W{k}", "quarter": quarter, "angle": angle,
           "box_x0": x0, "box_y0": y0, "box_x1": x1, "box_y1": y1, "length": float(x1 - x0),
           "height": float(y1 - y0), **NO_TOUCH, **touch}
    row.update(dict(zip(B.C_COLS, rect.reshape(8).tolist())))
    return row


def box(x0, y0, x1, y1, quarter=0, sheet=SHEET):
    """One of our boxes at axis 0 (boxes.parquet's columns that matter): trimmed and grouped corners alike."""
    row = {"sheet": sheet, "i": 0, "quarter": quarter, "axis": 0.0, "length": x1 - x0, "height": y1 - y0,
           "t_length": x1 - x0, "t_height": y1 - y0, "page_x": (x0 + x1) / 2, "page_y": (y0 + y1) / 2}
    for i, (x, y) in enumerate([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]):
        for k in "gt":
            row[f"{k}{i}x"], row[f"{k}{i}y"] = float(x), float(y)
    return row


def page(*rects, shape=(200, 400)):
    ink = np.zeros(shape, bool)
    for x0, y0, x1, y1 in rects:
        ink[y0:y1, x0:x1] = True
    return ink


def inputs(ink, **heads):
    return B.SheetInputs(ink=ink, chance=(ink * 200).astype(np.uint8), **heads)


# --- the count target -------------------------------------------------------------------------------------------------

def test_poisson_binomial_is_exact():
    assert np.allclose(B.poisson_binomial([0.2, 0.7]), [0.8 * 0.3, 0.2 * 0.3 + 0.8 * 0.7, 0.2 * 0.7])
    p = np.array([0.1, 0.55, 0.9, 0.3])
    brute = np.zeros(5)
    for held in itertools.product([0, 1], repeat=4):
        brute[sum(held)] += np.prod(np.where(held, p, 1 - p))
    assert np.allclose(B.poisson_binomial(p), brute)
    assert np.allclose(B.poisson_binomial([]), [1.0])


def test_the_count_is_the_poisson_binomial_of_each_words_share_of_ink_in_the_box():
    # three words of solid ink; the box holds all of word 0, half of word 1's ink, none of word 2
    ink = page((20, 40, 60, 60), (70, 40, 110, 60), (150, 40, 190, 60))
    terra = pd.DataFrame([word(0, 20, 40, 60, 60), word(1, 70, 40, 110, 60), word(2, 150, 40, 190, 60)])
    boxes = pd.DataFrame([box(20, 40, 90, 60)])
    ex = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, touch=never_held, joined=0)
    assert len(ex) == 1
    assert ex.K == 2
    assert np.allclose(ex.count[0], [0.0, 0.5, 0.5])
    # a word's share counts its ink, not its rectangle: word 1 with ink in its left half only is wholly inside
    ink2 = page((20, 40, 60, 60), (70, 40, 90, 60))
    ex2 = B.box_examples(boxes, terra, {SHEET: inputs(ink2)}, quarter=1, touch=never_held, joined=0)
    assert np.allclose(ex2.count[0], [0.0, 0.0, 1.0])


def test_doubt_weights_scale_an_example_by_the_words_it_may_hold():
    ink = page((20, 40, 60, 60), (70, 40, 110, 60))
    terra = pd.DataFrame([word(0, 20, 40, 60, 60), word(1, 70, 40, 110, 60)])
    boxes = pd.DataFrame([box(20, 40, 90, 60)])
    sheets = {SHEET: inputs(ink)}
    plain = B.box_examples(boxes, terra, sheets, quarter=1, touch=never_held, joined=0)
    assert np.allclose(plain.weight, 1.0)
    got = B.box_examples(boxes, terra, sheets, quarter=1, weights=[0.8, 0.4], touch=never_held, joined=0)
    # word 0 held for sure (share 1), word 1 with chance 0.5: (1 - 1 x 0.2) (1 - 0.5 x 0.6)
    assert got.weight[0] == pytest.approx(0.8 * 0.7)


# --- the boundary target ----------------------------------------------------------------------------------------------

def _between(ink_row):
    """The columns from the first column wholly of ink to the last (the words' outer columns, partly paper, are not
    between them)."""
    full = np.flatnonzero(ink_row > 1 - 1e-6)
    out = np.zeros(len(ink_row), bool)
    out[full[0]:full[-1] + 1] = True
    return out


def test_boundary_columns_are_exactly_the_gap_between_two_held_words():
    # two words of solid ink and a box over both: in the crop (shrunk with INTER_AREA, so a column's ink is its exact
    # share of ink) every column between them is gap where it is not ink, and no column outside them is
    ink = page((50, 100, 110, 130), (130, 100, 170, 130))
    terra = pd.DataFrame([word(0, 50, 100, 110, 130), word(1, 130, 100, 170, 130)])
    boxes = pd.DataFrame([box(50, 100, 170, 130)])
    ex = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, touch=never_held, joined=0)
    crop, (count, gap) = ex.crop(0), ex.target(0)
    assert np.allclose(count, [0, 0, 1])
    row = crop[0, K.HEIGHT // 2]
    inside = _between(row)
    assert inside.any() and (gap[inside] > 0.99).sum() >= 1
    assert np.allclose(gap[inside], 1.0 - row[inside], atol=1e-5)
    assert np.allclose(gap[~inside], 0.0)


def test_a_boundary_counts_only_as_much_as_the_words_on_both_sides_are_held():
    # word 1 half inside the box: the gap is a boundary with chance 0.5
    ink = page((50, 100, 110, 130), (130, 100, 170, 130))
    terra = pd.DataFrame([word(0, 50, 100, 110, 130), word(1, 130, 100, 170, 130)])
    boxes = pd.DataFrame([box(50, 100, 150, 130)])
    ex = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, touch=never_held, joined=0)
    crop, (count, gap) = ex.crop(0), ex.target(0)
    assert np.allclose(count, [0, 0.5, 0.5])
    row = crop[0, K.HEIGHT // 2]
    inside = _between(row)
    assert np.allclose(gap[inside], 0.5 * (1.0 - row[inside]), atol=1e-5)
    assert np.allclose(gap[~inside], 0.0)


def test_boundary_target_integrates_the_held_chances_exactly():
    # three words along one line in raw px, the middle one held with chance 0.5: between 0 and 1 the gap needs 1 held
    # (0.5); between 1 and 2 it is a gap when 1 is held (0.5) and also when it is not (0 and 2 are both held): 1
    spans = np.array([[0.0, 10.0], [20.0, 30.0], [40.0, 50.0]])
    got = B.boundary_target(spans, np.array([1.0, 0.5, 1.0]), w=50, out_w=50)
    want = np.zeros(50)
    want[10:20] = 1.0                       # a gap from 0 to 2 whether or not 1 is held
    want[20:30] = 0.5                       # inside word 1: a gap only when it is not held
    want[30:40] = 1.0
    assert np.allclose(got, want)
    # a column half in a gap is half a boundary
    half = B.boundary_target(np.array([[0.0, 10.5], [20.0, 30.0]]), np.ones(2), w=30, out_w=30)
    assert half[10] == pytest.approx(0.5) and half[11] == 1.0


# --- the channels -----------------------------------------------------------------------------------------------------

def test_crops_carry_ink_chance_and_the_word_heads_when_present():
    ink = page((50, 100, 110, 130))
    terra = pd.DataFrame([word(0, 50, 100, 110, 130)])
    boxes = pd.DataFrame([box(50, 100, 110, 130)])
    core = np.zeros(ink.shape, np.float32)
    core[100:130, 50:110] = 0.75
    tag = np.stack([np.full(ink.shape, 2.0, np.float32), np.full(ink.shape, -1.0, np.float32)])
    plain = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, touch=never_held, joined=0)
    assert plain.channels == ("ink", "chance")
    heads = B.box_examples(boxes, terra, {SHEET: inputs(ink, core=core, tag=tag)}, quarter=1, touch=never_held,
                           joined=0)
    assert heads.channels == ("ink", "chance", "core", "tag0", "tag1")
    c = heads.crop(0)
    mid = c[:, K.HEIGHT // 2, c.shape[2] // 2]
    assert mid[0] == pytest.approx(1.0) and mid[1] == pytest.approx(200 / 255) and mid[2] == pytest.approx(0.75)
    assert mid[3] == pytest.approx(2.0) and mid[4] == pytest.approx(-1.0)


# --- held out ---------------------------------------------------------------------------------------------------------

def test_box_examples_raises_on_held_quarter_or_answered_block_words_and_boxes():
    ink = page((50, 100, 110, 130))
    boxes = pd.DataFrame([box(50, 100, 110, 130)])
    sheets = {SHEET: inputs(ink)}
    for flags in ({"touch_ne": True}, {"touch_blocks": True}):
        terra = pd.DataFrame([word(0, 50, 100, 110, 130, **flags)])
        with pytest.raises(ValueError):
            B.box_examples(boxes, terra, sheets, quarter=1, touch=never_held, joined=0)
    terra = pd.DataFrame([word(0, 50, 100, 110, 130)])
    with pytest.raises(ValueError):          # a box in the held quarter
        B.box_examples(pd.DataFrame([box(50, 100, 110, 130, quarter=1)]), terra, sheets, quarter=1,
                       touch=never_held, joined=0)
    with pytest.raises(ValueError):          # a box whose crop touches an answered block
        B.box_examples(boxes, terra, sheets, quarter=1, joined=0,
                       touch=lambda s, w, sh: {**NO_TOUCH, "touch_blocks": True})
    with pytest.raises(ValueError):          # no quarter: it would train on every quarter's words
        B.box_examples(boxes, terra, sheets, quarter=None, touch=never_held, joined=0)
    with pytest.raises(ValueError):          # words without their held flags can't be checked
        B.box_examples(boxes, terra.drop(columns=["touch_blocks"]), sheets, quarter=1, touch=never_held, joined=0)


def test_training_boxes_leave_out_boxes_meeting_a_held_word_or_touching_a_held_area():
    terra = pd.DataFrame([word(0, 50, 100, 110, 130), word(1, 200, 100, 260, 130, touch_ne=True),
                          word(2, 300, 100, 340, 130)])
    boxes = pd.DataFrame([box(50, 100, 110, 130), box(240, 100, 280, 130), box(300, 100, 340, 130),
                          box(10, 10, 40, 30, quarter=1)])

    def touch(sheet, window, shape):          # the crop of the box at x 300.. touches an answered block
        return {**NO_TOUCH, "touch_blocks": window[0] > 280}
    keep = B.training_boxes(boxes, terra, quarter=1, touch=touch, shapes={SHEET: (200, 400)})
    assert list(keep.index) == [0]
    words = B.training_words(terra, quarter=1)
    assert list(words["word"]) == [0, 2]


# --- joined neighbours ------------------------------------------------------------------------------------------------

def test_draws_follow_the_measured_gap_distribution():
    dist = {"gap": [-0.1, 0.3, 0.8], "weight": [1.0, 2.0, 1.0]}
    got = B.draw_gaps(dist, 8000, np.random.default_rng([0, 1]))
    assert set(np.unique(got)) <= {-0.1, 0.3, 0.8}
    share = np.array([(got == g).mean() for g in dist["gap"]])
    assert np.allclose(share, [0.25, 0.5, 0.25], atol=0.02)


def test_same_line_gaps_measure_the_space_to_the_next_word_on_the_line():
    # three words on one line (gaps 10 and 25 px, heights 20) and one word read across the line, ahead of the last
    terra = pd.DataFrame([word(0, 10, 100, 60, 120), word(1, 70, 100, 100, 120), word(2, 125, 100, 175, 120),
                          word(3, 200, 60, 220, 160, angle=90.0)])
    got = B.same_line_gaps(terra)
    by = {(r.word, r.next): r for r in got.itertuples()}
    assert set(by) == {(0, 1), (1, 2)}               # word 2 is never next to 0: word 1 is surely between
    assert by[(0, 1)].gap == pytest.approx(10 / 20) and by[(0, 1)].weight == pytest.approx(1.0)
    assert by[(1, 2)].gap == pytest.approx(25 / 20) and by[(1, 2)].weight == pytest.approx(1.0)
    # a small word partly on word 0's line (8 of its 12 px), between it and word 1: next with chance 2/3, else word 1
    terra = pd.concat([terra, pd.DataFrame([word(4, 62, 112, 68, 124)])], ignore_index=True)
    got = B.same_line_gaps(terra)
    by = {(r.word, r.next): r for r in got.itertuples()}
    assert by[(0, 4)].weight == pytest.approx(8 / 12) and by[(0, 4)].gap == pytest.approx(2 / 16)
    assert by[(0, 1)].weight == pytest.approx(1 - 8 / 12)
    assert (0, 2) not in by
    dist = B.gap_distribution(got)
    assert sorted(dist["gap"]) == sorted(got["gap"].tolist()) and dist["n"] == len(got)


def test_the_gaps_are_between_consecutive_words_on_one_line_of_a_box():
    # a box (rows 95..125, its centre line at 110) over three words along its axis (gaps 10 and 15 px; heights 20,
    # 40, 30), word 1 sticking out of it across the line so that the box holds 25 of its 40 rows of ink.  Across the
    # line the words span [-10, 10], [-30, 10] and [-15, 15]: lines shared 1 (0, 1), 25/30 (1, 2), 1 (0, 2)
    ink = page((10, 100, 60, 120), (70, 80, 110, 120), (125, 95, 175, 125))
    terra = pd.DataFrame([word(0, 10, 100, 60, 120), word(1, 70, 80, 110, 120), word(2, 125, 95, 175, 125)])
    got = B.held_pair_gaps(pd.DataFrame([box(10, 95, 175, 125)]), terra, ink)
    s1, l12 = (120 - 95) / 40, 25 / 30
    by = {(r.word, r.next): r for r in got.itertuples()}
    assert set(by) == {(0, 1), (1, 2), (0, 2)}
    assert by[(0, 1)].gap == pytest.approx(10 / 30) and by[(0, 1)].weight == pytest.approx(s1)
    assert by[(1, 2)].gap == pytest.approx(15 / 35) and by[(1, 2)].weight == pytest.approx(s1 * l12)
    # 0 and 2 are consecutive on their line unless word 1 is held and on it
    assert by[(0, 2)].gap == pytest.approx(65 / 25) and by[(0, 2)].weight == pytest.approx(1 - s1 * min(1.0, l12))
    dist = B.gap_distribution(got)
    assert sorted(dist["gap"]) == sorted(got["gap"].tolist()) and dist["n"] == 3
    # two words stacked in one box (a code over its block number) are not on one line: no gap
    ink2 = page((10, 60, 70, 80), (20, 90, 50, 105))
    terra2 = pd.DataFrame([word(0, 10, 60, 70, 80), word(1, 20, 90, 50, 105)])
    assert B.held_pair_gaps(pd.DataFrame([box(10, 60, 70, 105)]), terra2, ink2).empty


def test_the_gap_is_along_the_words_reading_direction_whatever_the_boxs_axis():
    # two level words side by side (a gap of 10 px, heights 20) in a box grouped at 30 degrees that holds them both
    ink = page((100, 100, 150, 120), (160, 100, 190, 120))
    terra = pd.DataFrame([word(0, 100, 100, 150, 120), word(1, 160, 100, 190, 120)])
    (ux, uy), (vx, vy) = L._axes(30.0)
    u, v = np.array([ux, uy]) * 60, np.array([vx, vy]) * 45
    c = np.array([145.0, 110.0])
    tilted = box(0, 0, 1, 1)
    for i, corner in enumerate([c - u - v, c + u - v, c + u + v, c - u + v]):
        tilted[f"t{i}x"], tilted[f"t{i}y"] = corner
    tilted.update(axis=30.0, t_length=120.0, t_height=90.0)
    got = B.held_pair_gaps(pd.DataFrame([tilted]), terra, ink)
    assert len(got) == 1
    assert got["gap"].iloc[0] == pytest.approx(10 / 20) and got["weight"].iloc[0] == pytest.approx(1.0)
    # a word read across the line shares no line of text with it
    terra["angle"] = [0.0, 90.0]
    assert B.held_pair_gaps(pd.DataFrame([tilted]), terra, ink).empty


def test_overlapping_joined_words_keep_both_words_ink():
    a = np.zeros((2, K.HEIGHT, 40), np.float32)
    a[:, 10:22, 6:34] = 1.0                       # the word spans columns 6..34
    b = np.zeros((2, K.HEIGHT, 30), np.float32)
    b[:, 12:20, 6:24] = 0.5
    crop, target, got = B.join((a, (6.0, 34.0), 20.0), (b, (6.0, 24.0), 20.0), gap=-0.2)
    assert got["gap_px"] == pytest.approx(-4.0)
    assert np.allclose(target, 0.0)              # no column between them
    assert crop.shape[2] == 6 + 28 - 4 + 18 + 6
    # A's ink is all there, and B's wherever A has none
    assert np.allclose(crop[:, 10:22, 6:34], 1.0)
    assert np.allclose(crop[:, 12:20, 34:48], 0.5)
    spaced, target, got = B.join((a, (6.0, 34.0), 20.0), (b, (6.0, 24.0), 20.0), gap=0.5)
    assert got["gap_px"] == pytest.approx(10.0) and target.sum() == pytest.approx(10.0)
    assert np.allclose(spaced[:, :, 34:44], 0.0)  # paper across the gap


def test_joined_neighbours_take_their_gaps_from_the_distribution_and_know_their_targets():
    ink = page((20, 40, 80, 60), (120, 40, 150, 60), (20, 120, 60, 140), (200, 120, 260, 140), shape=(200, 300))
    terra = pd.DataFrame([word(0, 20, 40, 80, 60), word(1, 120, 40, 150, 60), word(2, 20, 120, 60, 140),
                          word(3, 200, 120, 260, 140)])
    boxes = pd.DataFrame([box(20, 40, 80, 60)])
    gaps = {"gap": [0.5, 1.5], "weight": [1.0, 1.0]}
    ex = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, touch=never_held, joined=40, gaps=gaps,
                        turned=0.0, seed=3)
    j = np.flatnonzero(ex.meta["source"].to_numpy() == "joined")
    assert len(j) == 40
    drawn = ex.meta["gap"].to_numpy()[j]
    assert set(np.unique(drawn)) <= {0.5, 1.5} and len(np.unique(drawn)) == 2
    for i in j:
        count, gap = ex.target(i)
        assert np.allclose(count, [0, 0, 1])
        m = ex.meta.iloc[i]
        # the boundary columns are the realised gap, in crop px; it is the drawn gap to the rounding of a px
        assert gap.sum() == pytest.approx(m["gap_px"], abs=1e-4)      # float32 targets
        assert m["gap_px"] / m["unit_px"] == pytest.approx(m["gap"], abs=0.5 / m["unit_px"] + 1e-9)
        # the gap is paper: the ink in a boundary column is the part of it outside the gap
        row = ex.crop(i)[0, K.HEIGHT // 2]
        assert np.allclose(row[gap >= 1.0], 0.0)
    # the joined words are training words, two different ones
    assert (ex.meta["word_a"].to_numpy()[j] != ex.meta["word_b"].to_numpy()[j]).all()
    # the same seed, the same examples
    again = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, touch=never_held, joined=40, gaps=gaps,
                           turned=0.0, seed=3)
    assert np.array_equal(again.flat, ex.flat) and again.meta.equals(ex.meta)


# --- the network ------------------------------------------------------------------------------------------------------

def _bars(rng, words: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic (2, HEIGHT, width) crop of `words` bars of ink across the middle rows, and its gap columns."""
    crop = np.zeros((2, K.HEIGHT, width), np.float32)
    gap = np.zeros(width, np.float32)
    lo, hi = 8, width - 8
    if words == 1:
        crop[:, 8:24, lo:hi] = 1.0
    else:
        g0 = int(rng.integers(lo + 10, hi - 18))
        g1 = g0 + int(rng.integers(5, 9))
        crop[:, 8:24, lo:g0] = 1.0
        crop[:, 8:24, g1:hi] = 1.0
        gap[g0:g1] = 1.0
    # broken strokes inside each word: the letters' own gaps, narrower than a word gap
    for x in rng.integers(lo, hi, 6):
        if gap[x] == 0:
            crop[:, 8:24, x] = 0.0
    return crop, gap


def _synthetic(n: int, seed: int) -> B.Examples:
    rng = np.random.default_rng([seed, 9])
    crops, counts, gaps = [], [], []
    for k in range(n):
        words = 1 + k % 2
        c, g = _bars(rng, words, int(rng.integers(56, 96)))
        crops.append(c)
        gaps.append(g)
        counts.append(np.eye(3)[words])
    meta = pd.DataFrame({"source": "synthetic", "words": [1 + k % 2 for k in range(n)]})
    return B.Examples.of(crops, counts, gaps, np.ones(n), meta, ("ink", "chance"))


@pytest.fixture
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_a_short_cpu_fit_tells_one_word_from_two(one_thread):
    train, test = _synthetic(96, 0), _synthetic(64, 1)
    recipe = {**B.RECIPE, "steps": 100, "batch": 8, "max_lr": 3e-3, "seed": 0, "device": "cpu"}
    net = B.fit_boxmodel(train, recipe)
    assert net.k == 2 and net.channels == 2
    count_lp, gap_lp = B.box_logp(net, [test.crop(i) for i in range(len(test))])
    assert count_lp.shape == (len(test), 3)
    assert np.allclose(np.exp(count_lp).sum(1), 1.0, atol=1e-5)
    words = test.meta["words"].to_numpy()
    right = (count_lp.argmax(1) == words).mean()
    assert right >= 0.9                      # chance is 0.5
    # the boundary is likelier in the gap than in the words
    two = np.flatnonzero(words == 2)
    in_gap = np.mean([np.exp(gap_lp[i][test.target(i)[1] > 0.5, 1]).mean() for i in two])
    off_gap = np.mean([np.exp(gap_lp[i][test.target(i)[1] == 0, 1]).mean() for i in two])
    assert in_gap > off_gap + 0.2


def test_box_logp_reads_each_crop_alone(one_thread):
    ex = _synthetic(6, 2)
    net = B.BoxNet(2, 2).eval()
    alone = B.box_logp(net, [ex.crop(0)])
    batch = B.box_logp(net, [ex.crop(i) for i in range(len(ex))])
    assert np.allclose(alone[0][0], batch[0][0], atol=1e-6)
    assert np.allclose(alone[1][0], batch[1][0], atol=1e-6)
    assert batch[1][0].shape == (ex.crop(0).shape[2], 2)
    assert np.allclose(np.exp(batch[1][0]).sum(1), 1.0, atol=1e-6)


def test_a_saved_network_reads_as_before(tmp_path, one_thread):
    ex = _synthetic(4, 5)
    net = B.BoxNet(2, 3).eval()
    B.save_boxnet(tmp_path / "net.pt", net, quarter="ne")
    got = B.load_boxnet(tmp_path / "net.pt")
    assert got.k == 3 and got.channels == 2
    crops = [ex.crop(i) for i in range(len(ex))]
    assert np.allclose(B.box_logp(net, crops)[0], B.box_logp(got, crops)[0])


def test_examples_save_and_load(tmp_path):
    ex = _synthetic(5, 4)
    ex.save(tmp_path / "ex")
    got = B.Examples.load(tmp_path / "ex")
    assert got.channels == ex.channels and np.array_equal(got.flat, ex.flat)
    assert np.array_equal(got.count, ex.count) and np.array_equal(got.boundary, ex.boundary)
    assert got.meta.equals(ex.meta)
