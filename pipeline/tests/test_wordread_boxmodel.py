"""The learned box model (`wordread.boxmodel`): its targets, its held-out rule, its joined neighbours and a short CPU
fit, on synthetic pages."""
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch

from sfzmaps.solver.textnet import boxes as textboxes
from sfzmaps.solver.textread import crops as K
from sfzmaps.solver.wordread import boxmodel as B

SHEET = "s1"
NO_TOUCH = {t: False for t in B.TOUCH}
NO_HELD = pd.DataFrame(columns=["sheet", *B.C_COLS])         # no held-out word on the page


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
    ex = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, held_words=NO_HELD, touch=never_held, joined=0)
    assert len(ex) == 1
    assert ex.K == 2
    assert np.allclose(ex.count[0], [0.0, 0.5, 0.5])
    # a word's share counts its ink, not its rectangle: word 1 with ink in its left half only is wholly inside
    ink2 = page((20, 40, 60, 60), (70, 40, 90, 60))
    ex2 = B.box_examples(boxes, terra, {SHEET: inputs(ink2)}, quarter=1, held_words=NO_HELD, touch=never_held, joined=0)
    assert np.allclose(ex2.count[0], [0.0, 0.0, 1.0])


def test_doubt_weights_scale_an_example_by_the_words_it_may_hold():
    ink = page((20, 40, 60, 60), (70, 40, 110, 60))
    terra = pd.DataFrame([word(0, 20, 40, 60, 60), word(1, 70, 40, 110, 60)])
    boxes = pd.DataFrame([box(20, 40, 90, 60)])
    sheets = {SHEET: inputs(ink)}
    plain = B.box_examples(boxes, terra, sheets, quarter=1, held_words=NO_HELD, touch=never_held, joined=0)
    assert np.allclose(plain.weight, 1.0)
    got = B.box_examples(boxes, terra, sheets, quarter=1, weights=[0.8, 0.4], held_words=NO_HELD,
                         touch=never_held, joined=0)
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
    ex = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, held_words=NO_HELD, touch=never_held, joined=0)
    crop, (count, gap) = ex.crop(0), ex.target(0)
    assert np.allclose(count, np.eye(len(count))[2])
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
    ex = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, held_words=NO_HELD, touch=never_held, joined=0)
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
    plain = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, held_words=NO_HELD,
                           touch=never_held, joined=0)
    assert plain.channels == ("ink", "chance")
    heads = B.box_examples(boxes, terra, {SHEET: inputs(ink, core=core, tag=tag)}, quarter=1,
                           held_words=NO_HELD, touch=never_held,
                           joined=0)
    assert heads.channels == ("ink", "chance", "core", "tag0", "tag1")
    c = heads.crop(0)
    mid = c[:, K.HEIGHT // 2, c.shape[2] // 2]
    assert mid[0] == pytest.approx(1.0) and mid[1] == pytest.approx(200 / 255) and mid[2] == pytest.approx(0.75)
    assert mid[3] == pytest.approx(2.0) and mid[4] == pytest.approx(-1.0)


def test_the_axis_channels_are_twice_its_angle_to_the_crops_axis():
    ink = page((50, 100, 110, 130))
    level = box(50, 100, 110, 130)
    upright = {**level, "axis": 90.0, "t_length": 30.0, "t_height": 60.0}     # the same rectangle, read upward
    for px_axis, crop_axis, want in ((0.0, level, (1.0, 0.0)), (45.0, level, (0.0, 1.0)), (90.0, level, (-1.0, 0.0)),
                                     (90.0, upright, (1.0, 0.0)), (135.0, upright, (0.0, 1.0))):
        inputs_ = B.SheetInputs(ink=ink, chance=np.zeros(ink.shape, np.uint8), axis=np.full(ink.shape, px_axis))
        assert inputs_.channels == ("ink", "chance", "axis_cos", "axis_sin")
        corners, length, height, axis = B._box_geometry(crop_axis)
        c = B.box_crop(inputs_, corners, length, height, axis)
        mid = c[:, K.HEIGHT // 2, c.shape[2] // 2]
        assert mid[2:] == pytest.approx(want, abs=1e-5)


# --- held out ---------------------------------------------------------------------------------------------------------

def test_box_examples_raises_on_held_quarter_or_answered_block_words_and_boxes():
    ink = page((50, 100, 110, 130))
    boxes = pd.DataFrame([box(50, 100, 110, 130)])
    sheets = {SHEET: inputs(ink)}
    for flags in ({"touch_ne": True}, {"touch_blocks": True}):
        terra = pd.DataFrame([word(0, 50, 100, 110, 130, **flags)])
        with pytest.raises(ValueError):
            B.box_examples(boxes, terra, sheets, quarter=1, held_words=NO_HELD, touch=never_held, joined=0)
    terra = pd.DataFrame([word(0, 50, 100, 110, 130)])
    with pytest.raises(ValueError):          # a box in the held quarter
        B.box_examples(pd.DataFrame([box(50, 100, 110, 130, quarter=1)]), terra, sheets, quarter=1,
                       held_words=NO_HELD, touch=never_held, joined=0)
    with pytest.raises(ValueError):          # a box whose crop touches an answered block
        B.box_examples(boxes, terra, sheets, quarter=1, joined=0, held_words=NO_HELD,
                       touch=lambda s, w, sh: {**NO_TOUCH, "touch_blocks": True})
    with pytest.raises(ValueError):          # no quarter: it would train on every quarter's words
        B.box_examples(boxes, terra, sheets, quarter=None, held_words=NO_HELD, touch=never_held, joined=0)
    with pytest.raises(ValueError):          # words without their held flags can't be checked
        B.box_examples(boxes, terra.drop(columns=["touch_blocks"]), sheets, quarter=1, held_words=NO_HELD,
                       touch=never_held, joined=0)
    with pytest.raises(ValueError):          # the held words are required: no box may meet one
        B.box_examples(boxes, terra, sheets, quarter=1, touch=never_held, joined=0)
    with pytest.raises(ValueError):          # a box meeting a held word's rectangle
        B.box_examples(boxes, terra, sheets, quarter=1, touch=never_held, joined=0,
                       held_words=pd.DataFrame([word(9, 100, 110, 130, 120, touch_ne=True)]))


def test_held_out_examples_score_a_quarter_as_the_training_examples_are_made():
    ink = page((20, 40, 60, 60), (70, 40, 110, 60))
    terra = pd.DataFrame([word(0, 20, 40, 60, 60), word(1, 70, 40, 110, 60)])
    sheets = {SHEET: inputs(ink)}
    train = B.box_examples(pd.DataFrame([box(20, 40, 90, 60)]), terra, sheets, quarter=1, held_words=NO_HELD,
                           touch=never_held, joined=0)
    scored = B.held_out_examples(pd.DataFrame([box(20, 40, 90, 60, quarter=1)]), terra, sheets)   # in quarter 1
    assert np.array_equal(scored.flat, train.flat) and np.array_equal(scored.count, train.count)
    assert np.array_equal(scored.boundary, train.boundary)
    assert scored.meta["quarter"].tolist() == [1] and not scored.meta["squeezed"].any()


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


# --- the gaps record --------------------------------------------------------------------------------------------------

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


# --- the merges our boxes make ----------------------------------------------------------------------------------------

def test_pair_geometry_is_the_space_between_two_words_in_the_first_ones_frame():
    terra = pd.DataFrame([word(0, 10, 100, 60, 120), word(1, 70, 100, 100, 120),       # side by side, 10 px apart
                          word(2, 10, 140, 70, 160), word(3, 20, 170, 50, 185),        # stacked, 10 px apart
                          word(4, 200, 60, 220, 160, angle=90.0)])                     # read across
    g = B.pair_geometry(terra)
    assert len(g) == 10 and (g["a"] < g["b"]).all()
    by = {(r.a, r.b): r for r in g.itertuples()}
    side = by[(0, 1)]
    assert side.along == pytest.approx(10 / 20) and side.across == pytest.approx(-20 / 20)
    assert side.along_gap == pytest.approx(10 / 20) and side.across_gap == 0.0
    assert side.along_overlap == 0.0 and side.across_overlap == pytest.approx(1.0)
    assert side.along_overlap_h == 0.0 and side.across_overlap_h == pytest.approx(1.0)
    assert side.angle == 0.0 and side.height == pytest.approx(0.0)
    stack = by[(2, 3)]                 # along the line word 3 (30 px) lies wholly within word 2's extent (60 px)
    assert stack.along == pytest.approx(-30 / 17.5) and stack.across == pytest.approx(10 / 17.5)
    assert stack.along_overlap == pytest.approx(1.0) and stack.along_overlap_h == pytest.approx(30 / 17.5)
    assert stack.across_gap == pytest.approx(10 / 17.5) and stack.across_overlap == 0.0
    assert stack.height == pytest.approx(abs(np.log(15 / 20)))
    assert by[(0, 4)].angle == 1.0


def test_merge_labels_are_the_chance_one_box_holds_both():
    pairs = pd.DataFrame({"a": [0, 0, 1, 0], "b": [1, 2, 2, 3]})
    held = [(np.array([0, 1]), np.array([1.0, 0.5])), (np.array([2, 0, 1]), np.array([1.0, 0.25, 0.4]))]
    got = B.merge_labels(pairs, held, 4)
    assert got == pytest.approx([1 - (1 - 0.5) * (1 - 0.25 * 0.4), 0.25, 0.4, 0.0])
    assert B.merge_labels(pairs, [(np.array([1, 0]), np.array([1.0, 1.0]))], 4)[0] == 1.0


def test_fit_merge_recovers_a_known_logistic():
    rng = np.random.default_rng([0, 5])
    n = 4000
    pairs = pd.DataFrame({"a": np.arange(n), "b": np.arange(n) + n,
                          **{k: rng.uniform(0, 3, n) for k in B.MERGE_FEATURES}})
    true = dict(zip(B.MERGE_FEATURES, (-1.5, -2.0, 1.0, 0.5, -0.3, -0.2, -0.5, 0.4)))
    z = 1.0 + sum(c * pairs[k].to_numpy() for k, c in true.items())
    label = 1 / (1 + np.exp(-z))                 # the exact chances: the likelihood is greatest at the truth
    got = B.fit_merge(pairs, label)
    assert got["converged"] and got["intercept"] == pytest.approx(1.0, abs=1e-2)
    for k, c in true.items():
        assert got["coef"][k] == pytest.approx(c, abs=1e-2)
    assert B.merge_chance(got, pairs) == pytest.approx(label, abs=1e-3)
    # scored on pairs it never saw, the fit beats a model off the truth
    rng2 = np.random.default_rng([1, 5])
    other = pd.DataFrame({k: rng2.uniform(0, 3, n) for k in B.MERGE_FEATURES})
    y2 = 1 / (1 + np.exp(-(1.0 + sum(c * other[k].to_numpy() for k, c in true.items()))))
    off = {**got, "coef": {**got["coef"], "along_gap": got["coef"]["along_gap"] + 0.5}}
    assert B.merge_nll(got, other, y2) < B.merge_nll(off, other, y2)
    # the plain-offset form fits by the same code
    plain = pd.DataFrame({k: rng.uniform(-2, 8, n) for k in B.OFFSET_FEATURES})
    assert B.fit_merge(plain, label, features=B.OFFSET_FEATURES)["features"] == list(B.OFFSET_FEATURES)


def test_squeezed_crops_are_those_cut_wider_than_max_width():
    assert not B.squeezed(100.0, 20.0)
    assert B.squeezed(2000.0, 20.0)                 # 2016 x 28 page px would be 2304 crop px wide
    w, out_w = B.layout(2000.0, 20.0)
    assert out_w == K.MAX_WIDTH


def test_union_box_is_the_box_grouping_makes_of_two_merged_words():
    ink = page((50, 100, 110, 120), (120, 100, 150, 120))
    r0, r1 = (textboxes.word_rect(b, 0.0) for b in ((50, 100, 110, 120), (120, 100, 150, 120)))
    row = B.union_box(ink, (ink * 200).astype(np.uint8), r0, r1)
    assert row["axis"] % 180 == pytest.approx(0.0, abs=1e-9)
    assert row["t_length"] == pytest.approx(100.0) and row["t_height"] == pytest.approx(20.0)
    xs, ys = [row[f"t{i}x"] for i in range(4)], [row[f"t{i}y"] for i in range(4)]
    assert min(xs) == pytest.approx(50) and max(xs) == pytest.approx(150)
    assert min(ys) == pytest.approx(100) and max(ys) == pytest.approx(120)
    assert B.union_box(ink, ink.astype(np.uint8), r0, textboxes.word_rect((300, 10, 340, 30), 0.0)) is None


def _merge_page():
    """Words side by side (0, 1; 15 px apart), a code over its block number (2, 3), and a word far away (4)."""
    ink = page((20, 40, 80, 70), (95, 40, 125, 70), (20, 150, 120, 170), (40, 175, 90, 190), (300, 150, 360, 180),
               shape=(300, 400))
    terra = pd.DataFrame([word(0, 20, 40, 80, 70), word(1, 95, 40, 125, 70), word(2, 20, 150, 120, 170),
                          word(3, 40, 175, 90, 190), word(4, 300, 150, 360, 180)])
    return ink, terra


MERGE = {"features": list(B.MERGE_FEATURES), "intercept": 0.0,
         "coef": {k: -1.0 if k in ("along_gap", "across_gap") else 0.0 for k in B.MERGE_FEATURES}}


def test_joined_pairs_are_drawn_by_their_merge_chance_at_their_real_place():
    ink, terra = _merge_page()
    boxes = pd.DataFrame([box(20, 40, 80, 70)])
    sheets = {SHEET: inputs(ink)}
    ex = B.box_examples(boxes, terra, sheets, quarter=1, held_words=NO_HELD, touch=never_held, joined=2000,
                        merge=MERGE, seed=3)
    j = np.flatnonzero(ex.meta["source"].to_numpy() == "joined")
    assert len(j) == 2000 and ex.info["drawn"] == 2000 and ex.info["merge"] == MERGE
    pairs = B.pair_geometry(terra)
    want = B.merge_chance(MERGE, pairs)
    want = want / want.sum()
    drawn = Counter(zip(ex.meta["word_a"].to_numpy()[j], ex.meta["word_b"].to_numpy()[j]))
    for a, b, w in zip(pairs["a"], pairs["b"], want):
        assert drawn.get((a, b), 0) / len(j) == pytest.approx(w, abs=0.04)
    for i in j:
        assert ex.count[i][2:].sum() == pytest.approx(1.0)      # both words are wholly in their box
    first = lambda a, b: j[(ex.meta["word_a"].to_numpy()[j] == a) & (ex.meta["word_b"].to_numpy()[j] == b)][0]  # noqa
    # side by side: count 2, the boundary between them where it is not ink (as for a real box)
    i = first(0, 1)
    count, gap = ex.target(i)
    assert np.allclose(count, np.eye(len(count))[2])
    row = ex.crop(i)[0, K.HEIGHT // 2]
    inside = _between(row)
    assert (gap[inside] > 0.99).any() and np.allclose(gap[inside], 1.0 - row[inside], atol=1e-5)
    assert np.allclose(gap[~inside], 0.0)
    # a code over its block number: their extents along the box's axis overlap, so no boundary, and still two words
    count, gap = ex.target(first(2, 3))
    assert np.allclose(count, np.eye(len(count))[2]) and np.allclose(gap, 0.0)
    again = B.box_examples(boxes, terra, sheets, quarter=1, held_words=NO_HELD, touch=never_held, joined=2000,
                           merge=MERGE, seed=3)
    assert np.array_equal(again.flat, ex.flat) and again.meta.equals(ex.meta)


def test_a_joined_box_never_touches_a_held_area_or_word():
    ink, terra = _merge_page()
    boxes = pd.DataFrame([box(20, 40, 80, 70)])

    def touch(sheet, window, shape):              # the far word's place is held
        return {**NO_TOUCH, "touch_blocks": window[2] > 250}
    held = pd.DataFrame([word(9, 100, 160, 110, 165)])           # a held word inside the code's box
    ex = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, touch=touch, joined=300, merge=MERGE,
                        held_words=held, seed=4)
    j = np.flatnonzero(ex.meta["source"].to_numpy() == "joined")
    words = set(ex.meta["word_a"].to_numpy()[j]) | set(ex.meta["word_b"].to_numpy()[j])
    assert len(j) == 300 and 4 not in words and 2 not in words
    assert ex.info["tried"] > 300


def test_box_examples_fits_the_merge_model_on_its_own_boxes():
    ink, terra = _merge_page()
    boxes = pd.DataFrame([box(20, 40, 125, 70)])                # one box holding words 0 and 1
    ex = B.box_examples(boxes, terra, {SHEET: inputs(ink)}, quarter=1, held_words=NO_HELD, touch=never_held, seed=5)
    assert ex.info["merged_pairs"] == pytest.approx(1.0) and ex.info["merge"]["merged"] == pytest.approx(1.0)
    assert ex.info["pairs"] == 10
    # by default as many joined examples as the real boxes are expected to hold two words or more: one
    assert (ex.meta["source"] == "joined").sum() == 1


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
    recipe = {**B.RECIPE, "steps": 100, "batch": 8, "max_lr": 3e-2, "seed": 0, "device": "cpu"}
    net = B.fit_boxmodel(train, recipe)
    count_lp, gap_lp = B.box_logp(net, [test.crop(i) for i in range(len(test))])
    assert count_lp.shape == (len(test), 2 + 2)       # 0..K and above K
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


def test_the_count_is_ordinal_with_the_mass_above_k_a_column(one_thread):
    ex = _synthetic(6, 6)
    net = B.BoxNet(2, 4).eval()
    with torch.no_grad():
        net.theta_steps.copy_(torch.tensor([0.3, -1.0, 2.0, 0.1]))
        net.theta_1.fill_(-0.7)
    crops = [ex.crop(i) for i in range(len(ex))]
    lp, _ = B.box_logp(net, crops)
    assert lp.shape == (len(ex), 4 + 2) and np.allclose(np.exp(lp).sum(1), 1.0)
    with torch.no_grad():
        theta = net.thresholds().double().numpy()
        score = np.concatenate([net(*B._batch([c]))[0].double().numpy() for c in crops])
    assert (np.diff(theta) > 0).all()
    # P(count >= j) is the sigmoid of the one score less the j-th threshold, the column above K included
    at_least = np.exp(lp)[:, ::-1].cumsum(1)[:, ::-1][:, 1:]
    assert np.allclose(at_least, 1 / (1 + np.exp(-(score[:, None] - theta[None, :]))), atol=1e-6)


def test_models_of_different_k_are_compared_at_the_smallest_k(one_thread):
    ex = _synthetic(5, 7)
    crops = [ex.crop(i) for i in range(len(ex))]
    big, small = B.box_logp(B.BoxNet(2, 5).eval(), crops)[0], B.box_logp(B.BoxNet(2, 2).eval(), crops)[0]
    k = min(big.shape[1], small.shape[1]) - 2
    a, b = B.collapse_counts(big, k), B.collapse_counts(small, k)
    assert a.shape == b.shape == (len(ex), k + 2)
    assert np.allclose(np.exp(a).sum(1), 1.0) and np.allclose(a[:, :k + 1], big[:, :k + 1])
    assert np.allclose(np.exp(a[:, -1]), np.exp(big[:, k + 1:]).sum(1))
    assert np.allclose(B.collapse_counts(small, 2), small)          # read at its own K: as it is
    with pytest.raises(ValueError):                                # a model can't be read above its K
        B.collapse_counts(small, 3)


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
