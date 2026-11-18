"""Words from the detector's word heads (`wordread.words`): every px votes for its word's box (the centre it points
to, the word's log length and log height, its axis modulo 180 degrees) and carries its tag, weighted by its core
chance; the words are a mixture over those votes with a spatial prior, measured spreads and a fitted logit per word,
with no level, no merge IoU, no unclip and no window that decides.

The maps are synthetic.  `_maps` draws what a net would give: inside each word (a float rectangle at a float angle),
each px's exact offset to the word's centre in the word's heights, the word's log length and log height and its axis,
optionally with noise and a per-word offset, the word's tag, and a core logit; outside, a low core logit with the
background's votes (offsets about 0, log sizes about their means, any axis, tags about a mean)."""
import dataclasses
import math

import cv2
import numpy as np
import pandas as pd
import pytest

from sfzmaps.solver import grid, wordboxes
from sfzmaps.solver.textnet import data
from sfzmaps.solver.wordread import words

SIZE = 240
BG_MEAN = (0.5, 0.0, -0.5, 0.0)
BG_LENGTH, BG_HEIGHT = (math.log(20.0), 0.5), (math.log(10.0), 0.5)


def _spreads(**kw) -> words.Spreads:
    """Test values of the measured spreads (Task 8 measures them on the training quarters' predictions)."""
    base = {"centre": 0.05, "length": 0.02, "height": 0.02, "angle": 1.0, "tag": 0.3, "cell": 1.0, "edge": 0.5,
            "core_share": 0.05, "outlier": 0.01, "centre_bias": (0.0, 0.0), "length_bias": 0.0, "height_bias": 0.0,
            "angle_bias": 0.0,
            "background_centre": 1.0, "background_length": BG_LENGTH, "background_height": BG_HEIGHT,
            "background_tag": 1.0, "background_tag_mean": BG_MEAN}
    return words.Spreads(**{**base, **kw})


EXACT = _spreads()
NOISY = _spreads(centre=0.1, length=0.05, height=0.05, angle=3.0, edge=0.3)   # edge: as measure_spreads finds it here
# A test value of the fitted logit.  A background vote far in its tail is explained best as a word of its own: its
# gain is the core evidence logit(w) - logit(core_share) plus (the word's peak log density - the background's log
# density at the vote), the background's log density -(chi2_8 / 2) less its normaliser (chi2_8 over its 2 offsets,
# 2 log sizes and 4 tags).  At EXACT's spreads the peak is +14.4, the normaliser 7.1 and logit(0.05) -2.9, so the gain
# is logit(w) + 24.4 + chi2_8 / 2: at core chance 1e-3 (logit -6.9) a vote needs chi2_8 > 85 to pay a log prior of -60
# (chance about 5e-15 a vote), at 1e-6 more.  Task 8 fits the prior on decisions where the background's proposals are
# labelled not words; real words' gains here are thousands.
PRIOR = words.WordPrior(slope=1.0, log_prior=-60.0)
# Exact votes give a box exactly; its edges then sit where the whole likelihood peaks, which the px (whole squares at
# their numbers, a row of them straddling a rotated edge) and a neighbour's px inside a box's soft edge move by
# hundredths of a px.  So boxes from exact votes are checked to a tenth of the edge spread.
EDGE_TOL = EXACT.edge / 10


def _logit(w: float) -> float:
    return float(np.log(w) - np.log1p(-w))


def _rect(cx, cy, length, height, angle=0.0) -> np.ndarray:
    """A word's rectangle (4 corners, the first edge along the axis, as `boxes.word_rect` lays them)."""
    a = np.deg2rad(angle)
    v = np.array([np.cos(a), -np.sin(a)])
    n = np.array([-v[1], v[0]])
    c = np.array([cx, cy], float)
    return np.array([c - v * length / 2 - n * height / 2, c + v * length / 2 - n * height / 2,
                     c + v * length / 2 + n * height / 2, c - v * length / 2 + n * height / 2])


def _maps(rects, angles, tags=None, noise=0.0, size_noise=0.0, angle_noise=0.0, tag_noise=0.0, bias=0.0,
          length_scale=1.0, w_in=1 - 1e-9, w_out=1e-6, block=1, lines=(), seed=0, size=SIZE):
    """core_logit, geo (5, h, w) as `model.word_geometry` decodes it, tag (4, h, w), ink, word_id for words `rects`
    at axes `angles` (degrees).  Inside a word (its px (col, row) in the rectangle; later words over earlier): core
    chance `w_in`; the offset to its centre in its heights plus noise of sd `noise` heights; its log length and log
    height plus noise of sd `size_noise`; its axis plus noise of sd `angle_noise` degrees (noise drawn once per `block`
    x `block` px: correlated votes); a per-word offset of its votes' centre offsets of sd `bias` heights; its tag
    (`tags[k]`, else 10 k on the first channel) plus noise of sd `tag_noise`.  Outside: core chance `w_out` and the
    background's votes.  Ink: a block inside each word and `lines`.  The offset is decoded as the net's is: its offset
    in heights times its own height.  `length_scale` multiplies every word's length votes (a biased head)."""
    rng = np.random.default_rng([seed, 3])
    h = w = size
    rows, cols = np.mgrid[0:h, 0:w]
    p = np.stack([cols, rows], -1).astype(float)
    word_id = np.zeros((h, w), np.int32)
    for k, r in enumerate(rects):
        r = np.asarray(r, float)
        v, n = (r[1] - r[0]) / np.linalg.norm(r[1] - r[0]), (r[3] - r[0]) / np.linalg.norm(r[3] - r[0])
        a, b = (p - r[0]) @ v, (p - r[0]) @ n
        inside = ((a >= -1e-9) & (a <= np.linalg.norm(r[1] - r[0]) + 1e-9)
                  & (b >= -1e-9) & (b <= np.linalg.norm(r[3] - r[0]) + 1e-9))
        word_id[inside] = k + 1
    core = np.where(word_id > 0, _logit(w_in), _logit(w_out))
    geo = np.zeros((5, h, w), np.float64)
    tag = np.asarray(BG_MEAN)[:, None, None] + rng.normal(0, 1.0, (4, h, w))
    n_words = len(rects)
    vecs = np.asarray(tags, float) if tags is not None else np.c_[10.0 * np.arange(n_words), np.zeros((n_words, 3))]
    nb = (h + block - 1) // block, (w + block - 1) // block

    def field(k):
        return rng.normal(0, 1.0, (k, *nb)).repeat(block, 1).repeat(block, 2)[:, :h, :w]

    dn, sn, an, tn = field(2), field(2), field(1)[0], field(4)
    for k, r in enumerate(rects):
        on = word_id == k + 1
        r = np.asarray(r, float)
        length, height = np.linalg.norm(r[1] - r[0]), np.linalg.norm(r[3] - r[0])
        off = rng.normal(0, 1.0, 2) * bias
        dxy = ((r.mean(0) - p[on]) / height).T + dn[:, on] * noise + off[:, None]
        geo[2, on] = length * length_scale * np.exp(sn[0, on] * size_noise)
        geo[3, on] = height * np.exp(sn[1, on] * size_noise)
        geo[0:2, on] = dxy * geo[3, on]
        geo[4, on] = (angles[k] + an[on] * angle_noise) % 180.0
        tag[:, on] = vecs[k][:, None] + tn[:, on] * tag_noise
    off = word_id == 0
    m = int(off.sum())
    geo[2, off] = np.exp(rng.normal(*BG_LENGTH, m))
    geo[3, off] = np.exp(rng.normal(*BG_HEIGHT, m))
    geo[0:2, off] = rng.normal(0, 1.0, (2, m)) * geo[3, off]
    geo[4, off] = rng.uniform(0, 180.0, m)
    gray = np.full((h, w), 255, np.uint8)
    for r in rects:
        c = np.asarray(r).mean(0)
        v, n = r[1] - r[0], r[3] - r[0]
        hl, hh = np.linalg.norm(v) / 2 - 3, np.linalg.norm(n) * 0.3
        v, n = v / np.linalg.norm(v), n / np.linalg.norm(n)
        cv2.fillPoly(gray, [np.round([c - v * hl - n * hh, c + v * hl - n * hh, c + v * hl + n * hh,
                                      c - v * hl + n * hh]).astype(np.int32)], 0)
    for a, b, thick in lines:
        cv2.line(gray, a, b, 0, thick)
    return core, geo, tag.astype(np.float32), gray < grid.INK_BELOW, word_id


def _g(df: pd.DataFrame, k: int, key="g") -> np.ndarray:
    return df[[f"{key}{i}{a}" for i in range(4) for a in "xy"]].to_numpy(float)[k].reshape(4, 2)


def _iou(a, b) -> float:
    inter = cv2.intersectConvexConvex(np.float32(a), np.float32(b))[0]
    area = wordboxes.poly_area(np.array([a, b], float))
    return float(inter / (area.sum() - inter))


def _box_of(df, rect) -> int:
    """The row of the box meeting `rect` best."""
    return int(np.argmax([_iou(_g(df, k), rect) for k in range(len(df))]))


def _same_box(df, rect, tol):
    """`rect`'s box: its corners within `tol` px of `rect`'s, in its order or a half turn round (a box has an axis,
    no reading sign)."""
    k = _box_of(df, rect)
    g = _g(df, k)
    assert min(np.abs(g - rect).max(), np.abs(g - np.roll(rect, 2, 0)).max()) < tol
    return k


# --- one failure kind each ------------------------------------------------------------------------------------------

def test_two_touching_words_become_two_boxes():
    rects = [_rect(80, 80, 60, 20), _rect(140, 80, 60, 20)]     # side by side, the same tags: the votes part them
    core, geo, tag, ink, _ = _maps(rects, [0, 0], tags=[[0, 0, 0, 0]] * 2)
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    assert len(df) == 2
    for r in rects:
        _same_box(df, r, EDGE_TOL)


def test_three_noisy_touching_words_become_three_boxes():
    """The chain that once came apart into 51 pieces: votes and tags noisy on every px."""
    rects = [_rect(55, 80, 50, 20), _rect(105, 80, 50, 20), _rect(155, 80, 50, 20)]
    core, geo, tag, ink, _ = _maps(rects, [0, 0, 0], tags=[[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]], noise=0.1,
                                   size_noise=0.05, angle_noise=3.0, tag_noise=0.3)
    df = words.core_words(core, geo, tag, ink, NOISY, PRIOR)
    assert len(df) == 3
    for r in rects:
        _same_box(df, r, 0.5)


def test_noisy_touching_words_sharing_a_tag_below_full_core_chance_among_the_background_become_two_boxes():
    """Two touching words, one tag, votes noisy, their px at core chance 0.9 and the background's all around at
    0.001: the votes part them, and each keeps its chance."""
    rects = [_rect(90, 100, 70, 22), _rect(160, 100, 70, 22)]
    core, geo, tag, ink, _ = _maps(rects, [0, 0], tags=[[0, 0, 0, 0]] * 2, noise=0.1, size_noise=0.05,
                                   angle_noise=3.0, tag_noise=0.3, w_in=0.9, w_out=1e-3)
    df = words.core_words(core, geo, tag, ink, NOISY, PRIOR)
    assert len(df) == 2
    for r in rects:
        k = _same_box(df, r, 0.5)
        assert df["p_word"].iloc[k] == pytest.approx(0.9, abs=0.02)
        assert df["word_logodds"].iloc[k] > 100


def test_a_noisy_single_word_stays_one_word():
    """Votes and tags noisy on every px at core chance 0.9, and the background's own votes all around at 0.001."""
    rect = _rect(120, 120, 100, 24)
    core, geo, tag, ink, _ = _maps([rect], [0], noise=0.1, size_noise=0.05, angle_noise=3.0, tag_noise=0.3, w_in=0.9,
                                   w_out=1e-3)
    df = words.core_words(core, geo, tag, ink, NOISY, PRIOR)
    assert len(df) == 1
    _same_box(df, rect, 0.5)
    assert df["p_word"].iloc[0] == pytest.approx(0.9, abs=0.02)


def test_a_word_whose_votes_are_alike_in_blocks_stays_one_word_by_its_measured_cell():
    """Votes alike over 4 x 4 px blocks (16 px a draw), read at the cell they have (16): one word."""
    rect = _rect(120, 120, 160, 30)
    core, geo, tag, ink, _ = _maps([rect], [0], noise=0.1, size_noise=0.05, angle_noise=3.0, tag_noise=0.3, block=4)
    df = words.core_words(core, geo, tag, ink, _spreads(**{**NOISY.__dict__, "cell": 16.0}), PRIOR)
    assert len(df) == 1
    _same_box(df, rect, 1.0)


def test_an_axis_on_the_half_turn_stays_one_word():
    """A word upright (90 degrees), its px's axes noisy about it, so half of them read nearer -90: modulo a half turn
    they are one axis, and one word."""
    rect = _rect(120, 120, 120, 24, 90)
    core, geo, tag, ink, _ = _maps([rect], [90], noise=0.1, size_noise=0.05, angle_noise=3.0, tag_noise=0.3)
    df = words.core_words(core, geo, tag, ink, NOISY, PRIOR)
    assert len(df) == 1
    _same_box(df, rect, 0.5)
    assert wordboxes.axis_error(df["axis"].iloc[0], 90.0) < 1.0


@pytest.mark.parametrize("corrected", [False, True])
def test_length_votes_running_long_do_not_stretch_the_box_past_its_core_px(corrected):
    """Every length vote 1.4 times the word's (as Task 8's detector's run long), noisy (sd 0.3 in log length and log
    height, as the spreads say): the core px end where the word does, and a box over px the core head calls background
    pays for each, so the box keeps the painted word's extent; with the bias measured and taken off (`length_bias`),
    the votes agree with it too."""
    rect = _rect(120, 120, 100, 24)
    core, geo, tag, ink, _ = _maps([rect], [0], noise=0.1, size_noise=0.3, angle_noise=3.0, tag_noise=0.3,
                                   length_scale=1.4)
    spreads = _spreads(**{**NOISY.__dict__, "length": 0.3, "height": 0.3,
                          "length_bias": math.log(1.4) if corrected else 0.0})
    df = words.core_words(core, geo, tag, ink, spreads, PRIOR)
    assert len(df) == 1
    assert df["length"].iloc[0] == pytest.approx(100, abs=3.0)
    assert _iou(_g(df, 0), rect) > 0.95


def test_a_line_touching_a_word_does_not_extend_its_box():
    """A line runs on from the word's foot; its px have core 0.6 and vote for one thin box along the line."""
    rect = _rect(100, 100, 80, 24)
    core, geo, tag, ink, word_id = _maps([rect], [0], lines=[((140, 110), (230, 110), 2)])
    on_line = np.zeros_like(core, bool)
    on_line[109:112, 140:231] = True
    on_line &= word_id == 0
    core[on_line] = _logit(0.6)
    ys, xs = np.nonzero(on_line)
    geo[0, on_line], geo[1, on_line] = 185.0 - xs, 110.0 - ys      # each votes for the line's box, 90 x 2 px
    geo[2, on_line], geo[3, on_line] = 90.0, 2.0
    geo[4, on_line] = 0.0
    tag[:, on_line] = 0
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    k = _same_box(df, rect, EDGE_TOL)
    t = _g(df, k, "t")
    assert t[:, 0].max() <= rect[:, 0].max() + EDGE_TOL             # the trimmed box stops at the word too
    others = df.drop(index=df.index[k])
    assert len(others) == 1                                          # the line's votes: a box of their own ...
    assert others["p_word"].iloc[0] == pytest.approx(0.6, abs=0.01)  # ... kept with its chance


def test_a_strong_minority_cluster_of_votes_becomes_its_own_box():
    """A code printed on a street name, centred on it: its px (15% of the name's) vote for the code's own box, the
    rest for the name's; the two have one centre and one tag."""
    name, code = _rect(120, 120, 120, 30), _rect(120, 120, 40, 14)
    core, geo, tag, ink, _ = _maps([name, code], [0, 0], tags=[[0, 0, 0, 0]] * 2)
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    assert len(df) == 2
    _same_box(df, name, EDGE_TOL)
    _same_box(df, code, EDGE_TOL)


@pytest.mark.parametrize("where", ["left", "bottom"])
def test_a_core_at_the_page_edge_gives_a_clipped_box(where):
    rect = _rect(10, 70, 60, 20) if where == "left" else _rect(120, SIZE - 2, 60, 20)     # part off the page
    core, geo, tag, ink, _ = _maps([rect], [0])
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    assert len(df) == 1                                              # not dropped
    g, t = _g(df, 0), _g(df, 0, "t")
    for box in (g, t):                                               # never past the page, never wrapped
        assert (box >= -1e-9).all() and (box <= SIZE + 1e-9).all()
    if where == "left":
        assert g[:, 0].min() == pytest.approx(0.0, abs=1e-9)
        assert g[:, 0].max() == pytest.approx(rect[:, 0].max(), abs=EDGE_TOL)
    else:
        assert g[:, 1].max() == pytest.approx(SIZE, abs=1e-9)
        assert g[:, 1].min() == pytest.approx(rect[:, 1].min(), abs=EDGE_TOL)
    assert df["ink_px"].iloc[0] > 0


@pytest.mark.parametrize("angle", [30, 90, 150, -45, 180])
def test_rotated_words_come_back_from_unrounded_votes(angle):
    rect = _rect(120, 120, 100, 24, angle)
    core, geo, tag, ink, _ = _maps([rect], [angle])
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    assert len(df) == 1
    _same_box(df, rect, EDGE_TOL)                                    # the same corners, either way round
    assert wordboxes.axis_error(df["axis"].iloc[0], angle) < math.degrees(EDGE_TOL / 50)    # its ends within it


# --- the fit itself ----------------------------------------------------------------------------------------------

def test_no_ecm_step_lowers_the_likelihood_when_the_votes_and_the_px_disagree():
    """A long word whose px's axes are all 10 degrees off while its px lie along 0 degrees: the votes pull the axis
    one way and the spatial prior (where the px are) the other.  No step's log likelihood falls (each step's before and
    after, on its region), and the fitted axis settles strictly between the two, where the likelihood peaks along it
    (no higher a tenth of a degree either side)."""
    rect = _rect(120, 120, 200, 24)
    core, geo, tag, ink, word_id = _maps([rect], [10], noise=0.1, size_noise=0.05, angle_noise=3.0, tag_noise=0.3,
                                         w_in=0.9)
    V = words._Votes(core, geo, tag, NOISY)
    ys, xs = np.nonzero(word_id > 0)
    v = V.take(V.index[ys, xs])
    start = {"prm": words._own(words._sub(v, slice(0, 1)))[0], "tag": v["t"][0]}
    start["prm"][4] = 0.0
    trace = []
    fit, = words._fit(V, [start], [], trace)
    assert len(trace) > 1
    assert all(after >= before - 1e-9 * abs(before) for before, after in trace)
    theta = float(fit["prm"][4])
    assert 0 < theta < math.radians(10)

    def turned(by):
        prm = fit["prm"].copy()
        prm[4] = theta + math.radians(by)
        return {**fit, "prm": prm}

    region = words._supports(V, [turned(-0.1), fit, turned(0.1)])
    peak = words._score(V, region, [fit], [])
    assert all(words._score(V, region, [turned(by)], []) <= peak for by in (-0.1, 0.1))


def test_every_move_of_the_search_raises_the_configurations_total_score():
    """The configuration's total score, recomputed from the words' parameters after each move: it rises with every
    move, so the search ends; here two touching words and a lone one."""
    rects = [_rect(70, 70, 70, 22), _rect(140, 70, 70, 22), _rect(120, 170, 90, 22)]
    core, geo, tag, ink, _ = _maps(rects, [0, 0, 0], tags=[[0, 0, 0, 0]] * 3, noise=0.1, size_noise=0.05,
                                   angle_noise=3.0, tag_noise=0.3)
    V = words._Votes(core, geo, tag, NOISY)
    scores = []
    found = words._search(V, words._proposals(V, NOISY), NOISY, PRIOR, scores=scores)
    assert len(found) == 3
    assert len(scores) > 1
    assert (np.diff(scores) > 0).all()


def test_a_words_logodds_is_the_score_without_it_against_with_it():
    """Leave one out: a word's log-odds is the prior's logit of the log likelihood with it less without it (the other
    words held, their shares renormalised); a word sharing its px with a near copy of itself gains little."""
    rects = [_rect(80, 80, 60, 20), _rect(160, 160, 80, 24)]
    core, geo, tag, ink, _ = _maps(rects, [0, 0])
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    V = words._Votes(core, geo, tag, EXACT)
    found = words._final(V, words._search(V, words._proposals(V, EXACT), EXACT, PRIOR))
    for k, word in enumerate(found):
        region = V.support(word["prm"])
        others = found[:k] + found[k + 1:]
        gain = (words._score(V, region, [word], others) - words._score(V, region, [], others)) / EXACT.cell
        row = _box_of(df, words._box(word["prm"]))
        assert df["word_logodds"].iloc[row] == pytest.approx(float(PRIOR.logodds(gain)))
    copy = dict(found[0])
    region = V.support(copy["prm"])
    alone = words._score(V, region, [found[0]], found[1:]) - words._score(V, region, [], found[1:])
    shared = words._score(V, region, [copy, found[0]], found[1:]) - words._score(V, region, [found[0]], found[1:])
    assert shared < 0.01 * alone


def test_a_copy_of_a_word_adds_no_chance_of_being_a_word():
    """At core chance 0.6, a second word on the first's box: the chance a px is a word's (w C, C the chance it is in
    some word's box) does not grow, so the copy gains almost nothing (its votes are only shared); the first word's own
    gain over none is thousands of times as much."""
    rect = _rect(120, 120, 100, 24)
    core, geo, tag, ink, word_id = _maps([rect], [0], w_in=0.6)
    V = words._Votes(core, geo, tag, EXACT)
    word = {"prm": np.array([120.0, 120.0, np.log(100.0), np.log(24.0), 0.0]), "tag": np.zeros(4)}
    region = V.support(word["prm"])
    alone = words._score(V, region, [word], []) - words._score(V, region, [], [])
    copy = words._score(V, region, [word, dict(word)], []) - words._score(V, region, [word], [])
    assert alone > 1000 and copy < 1e-3 * alone


def test_a_px_centred_on_its_words_box_edge_lies_in_its_box():
    """A px is a unit square at its numbers: one centred on the word's box edge is in the box its px lie in (q near
    1 at the measured edge spread), the next one out is not."""
    V = words._Votes(np.zeros((40, 80)), np.ones((5, 40, 80)), np.zeros((4, 40, 80)), NOISY)
    prm = np.array([40.0, 20.0, np.log(40.0), np.log(10.0), 0.0])       # x 20 to 60, y 15 to 25
    q = np.exp(V.log_q(np.array([[60.0, 20.0], [61.0, 20.0], [40.0, 25.0], [40.0, 26.0]]), prm))
    assert q[0] > 0.9 and q[2] > 0.9
    assert q[1] < 0.1 and q[3] < 0.1


# --- the inputs and the table ---------------------------------------------------------------------------------------

def test_the_boxes_table_has_the_word_boxes_columns_with_their_meaning():
    rects = [_rect(70, 70, 60, 20), _rect(160, 160, 100, 26, 90)]
    core, geo, tag, ink, word_id = _maps(rects, [0, 90], w_in=0.8)
    kind_p = np.zeros((4, SIZE, SIZE), np.float32)
    kind_p[0], kind_p[2] = 0.75, 0.25
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR, kind_p=kind_p, sheet="p07")
    assert list(df.columns) == [*wordboxes.COLUMNS, *words.EXTRA_COLUMNS]
    assert words.EXTRA_COLUMNS == ("p_word", "core_mass", "tag_spread", "word_logodds")
    assert df["sheet"].tolist() == ["p07", "p07"] and df["i"].tolist() == [0, 1]
    # not placed on the map yet: Task 8 places the boxes (`prints._place`, as `wordboxes.sheet_boxes` does)
    assert (df["quarter"] == -1).all() and df["x"].isna().all() and (df["lab_lot"] == -1).all()
    assert not df["street"].any() and df["level"].isna().all()
    for k, r in enumerate(rects):
        row = _box_of(df, r)
        g, t = _g(df, row), _g(df, row, "t")
        assert (df["page_x"].iloc[row], df["page_y"].iloc[row]) == pytest.approx(tuple(g.mean(0)))
        assert df["length"].iloc[row] == pytest.approx(np.linalg.norm(g[1] - g[0]))
        assert df["height"].iloc[row] == pytest.approx(np.linalg.norm(g[3] - g[0]))
        assert df["t_length"].iloc[row] == pytest.approx(np.linalg.norm(t[1] - t[0]))
        assert df["t_length"].iloc[row] < df["length"].iloc[row]          # the glyphs, trimmed inside the box
        assert wordboxes.axis_error(df["axis"].iloc[row], data.reading_angle(g)) < 1e-6
        # every px of the word votes for it, each at core chance 0.8: its share of the weighted mass, and the mass
        assert df["p_word"].iloc[row] == df["mean_chance"].iloc[row] == pytest.approx(0.8, abs=1e-6)
        assert df["core_mass"].iloc[row] == pytest.approx(0.8 * (word_id == k + 1).sum(), rel=1e-6)
        assert df["tag_spread"].iloc[row] == pytest.approx(0.0, abs=1e-5)
        assert np.isfinite(df["word_logodds"].iloc[row]) and df["word_logodds"].iloc[row] > 100
        assert (df["p_zoning"].iloc[row], df["p_street"].iloc[row]) == pytest.approx((0.75, 0.25))


def test_bad_inputs_are_refused_and_a_page_of_background_gives_no_box():
    core, geo, tag, ink, _ = _maps([], [], w_out=1e-6)
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    assert len(df) == 0 and list(df.columns) == [*wordboxes.COLUMNS, *words.EXTRA_COLUMNS]
    with pytest.raises(ValueError, match="geo"):
        words.core_words(core, geo[:4], tag, ink, EXACT, PRIOR)
    with pytest.raises(ValueError, match="spread"):
        words.core_words(core, geo, tag, ink, _spreads(centre=0.0), PRIOR)
    with pytest.raises(ValueError, match="channels"):
        words.core_words(core, geo, tag, ink, _spreads(background_tag_mean=(0.0,)), PRIOR)


# --- measuring the spreads and fitting the prior (Task 8's, on the training quarters) --------------------------------

def _pages(block=1, bias=0.0, angle_shift=0.0, length_scale=1.0, seeds=range(3)):
    rects = [_rect(60 + 60 * (k % 3), 50 + 50 * (k // 3), 50, 18) for k in range(12)]
    pages = []
    for seed in seeds:
        core, geo, tag, ink, word_id = _maps(rects, [angle_shift] * 12, noise=0.1, size_noise=0.05, angle_noise=2.0,
                                             tag_noise=0.3, bias=bias, length_scale=length_scale, w_out=1e-3,
                                             block=block, seed=seed)
        pages.append({"geo": geo, "tag": tag, "word_id": word_id, "words": np.array(rects)})
    return pages


def test_measure_spreads_finds_the_spreads_about_each_words_own_fit_not_its_offset_from_terra():
    """Noise of sd 0.1 heights on the offsets, 0.05 on the log sizes, 2 degrees on the axis and 0.3 on every tag, once
    per px, and each word's votes offset as a whole by sd 0.5 heights: the spreads are the noise's, the votes
    independent (cell 1), the offsets no spread but the edges'."""
    got = words.measure_spreads(_pages(bias=0.5))
    assert got.centre == pytest.approx(0.1, rel=0.05)
    assert got.length == pytest.approx(0.05, rel=0.05)
    assert got.height == pytest.approx(0.05, rel=0.05)
    assert got.angle == pytest.approx(2.0, rel=0.05)
    assert got.tag == pytest.approx(0.3, rel=0.05)
    assert got.cell == pytest.approx(1.0, abs=0.25)
    assert got.edge > 2.0                                              # 0.5 heights of 18 px moves the edges ~9 px
    assert got.background_centre == pytest.approx(1.0, rel=0.05)
    assert got.background_length == pytest.approx(BG_LENGTH, rel=0.05)
    assert got.background_height == pytest.approx(BG_HEIGHT, rel=0.05)
    assert got.background_tag_mean == pytest.approx(BG_MEAN, abs=0.02)
    assert got.background_tag == pytest.approx(1.0, rel=0.05)


def test_measure_spreads_finds_each_votes_bias_against_terra_and_takes_it_off():
    """Length votes running 1.4 times long: the length bias is log 1.4, taken off before the rest is measured, so the
    edge (where the px end about the fitted box) is not stretched by it; the core share is the words' px share."""
    pages = _pages(length_scale=1.4)
    got = words.measure_spreads(pages)
    assert got.length_bias == pytest.approx(math.log(1.4), abs=0.01)
    assert got.height_bias == pytest.approx(0.0, abs=0.01)
    assert got.centre_bias == pytest.approx((0.0, 0.0), abs=0.01)
    assert got.angle_bias == pytest.approx(0.0, abs=0.2)
    assert got.edge == pytest.approx(math.sqrt(1 / 12), abs=0.15)
    share = sum(int((p["word_id"] > 0).sum()) for p in pages) / sum(p["word_id"].size for p in pages)
    assert got.core_share == pytest.approx(share)


def test_measure_spreads_finds_the_px_quantisation_alone_as_the_edge_of_exact_words():
    got = words.measure_spreads(_pages())
    assert got.edge == pytest.approx(math.sqrt(1 / 12), abs=0.15)


def test_measure_spreads_does_not_count_an_axis_offset_as_offset_or_cell():
    """Every word's px's axes 10 degrees off Terra's while the offsets and sizes agree with it: the offsets' spread and
    the cell are the noise's; the axis is fitted where the votes put it."""
    got = words.measure_spreads(_pages(angle_shift=10.0))
    assert got.centre == pytest.approx(0.1, rel=0.1)
    assert got.cell == pytest.approx(1.0, abs=0.3)
    assert got.angle == pytest.approx(2.0, rel=0.1)


def test_measure_spreads_finds_the_cell_of_votes_alike_in_blocks():
    got = words.measure_spreads(_pages(block=3))
    assert got.cell == pytest.approx(9.0, rel=0.35)
    assert got.centre == pytest.approx(0.1, rel=0.1)


def _masked(pages, keep):
    """`pages` with a mask each: `keep(page)`, the px that may be read."""
    return [{**p, "mask": keep(p)} for p in pages]


def _left(cols):
    def keep(p):
        m = np.zeros(p["word_id"].shape, bool)
        m[:, :cols] = True
        return m
    return keep


def test_measure_spreads_on_a_masked_page_is_the_page_without_the_masked_out_words():
    """A mask over the pages' top left, its edges between words (four words whole in it, the rest wholly out): the
    spreads are those of that part of the pages alone, its words and its background, the core share counted on it."""
    pages = _pages()

    def keep(p):
        m = np.zeros(p["word_id"].shape, bool)
        m[:125, :150] = True
        return m

    cropped = [{"geo": p["geo"][:, :125, :150], "tag": p["tag"][:, :125, :150], "word_id": p["word_id"][:125, :150],
                "words": p["words"]} for p in pages]
    assert np.unique(cropped[0]["word_id"]).tolist() == [0, 1, 2, 4, 5]          # four words, none cut
    assert words.measure_spreads(_masked(pages, keep)) == words.measure_spreads(cropped)


def test_a_word_with_a_px_outside_the_mask_is_left_out_whole():
    """A mask whose edge cuts the second column of words (x 95 to 145, cut at 120): each is left out with all its px,
    no part of a word fitted; its px inside the mask are no background either, but are word px in the core share.  So
    the spreads are those with those words wholly outside the mask, but for the core share; and not those of the page
    cut there (its parts of words fitted as words)."""
    pages = _pages()
    cut = np.intersect1d(np.unique(pages[0]["word_id"][:, :120]), np.unique(pages[0]["word_id"][:, 120:]))
    cut = cut[cut > 0]
    assert len(cut) == 4

    def without_cut(p):
        return _left(120)(p) & ~np.isin(p["word_id"], cut)

    got, apart = words.measure_spreads(_masked(pages, _left(120))), words.measure_spreads(_masked(pages, without_cut))
    assert dataclasses.replace(got, core_share=apart.core_share) == apart
    masks = [_left(120)(p) for p in pages]
    share = (sum(int((m & (p["word_id"] > 0)).sum()) for m, p in zip(masks, pages))
             / sum(int(m.sum()) for m in masks))
    assert got.core_share == share and got.core_share > apart.core_share
    parts = [{"geo": p["geo"][:, :, :120], "tag": p["tag"][:, :, :120], "word_id": p["word_id"][:, :120],
              "words": p["words"]} for p in pages]
    assert words.measure_spreads(parts) != got


def test_measure_spreads_with_a_mask_of_every_px_is_the_page_without_one():
    pages = _pages()
    assert (words.measure_spreads(_masked(pages, lambda p: np.ones(p["word_id"].shape, bool)))
            == words.measure_spreads(pages))
    with pytest.raises(ValueError, match="mask"):
        words.measure_spreads(_masked(pages, lambda p: np.ones((3, 3), bool)))


def test_a_one_px_fragment_whose_vote_cannot_place_its_height_keeps_it():
    """A 1 px word (a fragment, as a window's edge leaves one) whose vote points at its own px (no offset: the centre
    puts no curvature on the height) and whose chance of being the word's own is the least double: the height's
    curvature underflows to 0, so the vote cannot place it and it keeps its height (once a division by 0)."""
    model = words._Model(_spreads(height=2.0), 4)
    v = {"p": np.array([[10.0, 20.0]]), "o": np.zeros((1, 2)), "ll": np.array([math.log(30.0)]),
         "lh": np.array([math.log(12.0)]), "th": np.array([0.1]), "t": np.ones((1, 4))}
    prm = np.array([10.0, 20.0, math.log(30.0), math.log(10.0), 0.0])
    new, tag = model.own_update(v, np.array([np.nextafter(0.0, 1.0)]), prm)
    assert new[3] == prm[3] and np.isfinite(new).all() and (tag == 1.0).all()
    new, _ = model.own_update(v, np.ones(1), prm)                 # with its whole mass, the vote places it
    assert new[3] == pytest.approx(math.log(12.0))


def test_fit_word_prior_recovers_the_logit_the_labels_were_drawn_with():
    """Decisions with gains and fractional labels (each the chance the extra word is real, from the logit
    0.4 x gain - 6): the fitted slope and log prior are those."""
    rng = np.random.default_rng([0, 5])
    gains = rng.exponential(20.0, 4000)
    labels = 1 / (1 + np.exp(-(0.4 * gains - 6.0)))
    got = words.fit_word_prior(gains, labels)
    assert (got.slope, got.log_prior) == pytest.approx((0.4, -6.0), rel=1e-6)
    hard = rng.random(4000) < labels                                   # and from labels drawn 0 or 1 by those chances
    got = words.fit_word_prior(gains, hard)
    assert got.slope == pytest.approx(0.4, rel=0.15) and got.log_prior == pytest.approx(-6.0, rel=0.15)


def test_fit_word_prior_refuses_labels_a_threshold_parts():
    with pytest.raises(ValueError, match="no finite fit"):
        words.fit_word_prior([1.0, 2.0, 30.0, 40.0], [0, 0, 1, 1])
    with pytest.raises(ValueError, match="no finite fit"):
        words.fit_word_prior([1.0, 5.0, 30.0], [1, 1, 1])


def test_prior_decisions_label_the_proposals_adds_splits_and_joins_by_terras_words():
    """Two touching words (one tag) and a lone word, noisy, among the background: the proposals on the words are
    labelled words, the background's not; the adds of words are words; each word's split is labelled not real, the
    touching pair's join (two words kept apart) real."""
    rects = [_rect(70, 70, 70, 22), _rect(140, 70, 70, 22), _rect(120, 170, 90, 22)]
    core, geo, tag, ink, word_id = _maps(rects, [0, 0, 0], tags=[[0, 0, 0, 0]] * 3, noise=0.1, size_noise=0.05,
                                         angle_noise=3.0, tag_noise=0.3, w_in=0.9, w_out=1e-3)
    first = words.prior_decisions(core, geo, tag, NOISY, word_id)
    assert set(first["kind"]) == {"proposal"}
    # the words' main proposals stand far above any of the background's (small fragments of a word, a few votes
    # landing apart, sit among the background's tail: the fitted logit weighs them)
    assert ((first["label"] > 0.5) & (first["gain"] > 1000)).sum() >= 3
    assert first.loc[first["label"] < 0.5, "gain"].max() < 100
    every = words.prior_decisions(core, geo, tag, NOISY, word_id, PRIOR)
    adds, splits, joins = (every[every["kind"] == k] for k in ("word", "split", "join"))
    assert (adds["label"] > 0.9).sum() >= 3
    # every split and join weighed is labelled by Terra as its gain says: the real extra words (different Terra words,
    # labelled 1) gain far more than the rest (labelled 0); the touching pair kept apart is real
    for d in (splits, joins):
        real, not_real = d[d["label"] > 0.5], d[d["label"] < 0.5]
        if len(real) and len(not_real):
            assert real["gain"].min() > not_real["gain"].max()
    assert len(splits) >= 3 and (splits["label"] < 0.05).sum() >= 3
    assert len(joins) >= 1 and joins["label"].max() > 0.95
