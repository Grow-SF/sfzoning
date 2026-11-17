"""Words from the detector's word heads (`wordread.words`): every px votes for its word's box (its four distances
along its reading direction) and carries its tag, weighted by its core chance; the words are a mixture over those
votes with measured spreads and a fitted log prior per word, with no level, no merge IoU and no unclip.

The maps are synthetic.  `_maps` draws what a net would give: inside each word (a float rectangle at a float angle),
the exact distances from each px to the word's edges (`data._geometry`'s, from the px's (col, row)), optionally with
noise, the word's tag, and a core chance; outside, a lower core chance with votes about the px itself (the background
the model expects).  One test reads Task 5's own targets (`data.crop_targets(..., words=True)`)."""
import cv2
import numpy as np
import pandas as pd
import pytest

from sfzmaps.solver import grid, paths, wordboxes
from sfzmaps.solver.textnet import data
from sfzmaps.solver.wordread import shrink, words

SIZE = 240
# test values of the measured parameters (Task 8 measures them on the training quarters' predictions)
EXACT = words.Spreads(distance=0.5, angle=1.0, tag=0.3, cell=1.0, background_distance=8.0, background_tag=1.0)
PRIOR = -20.0


def _rect(cx, cy, length, height, angle=0.0) -> np.ndarray:
    """A word's rectangle (4 corners, the first edge along the reading direction, as `boxes.word_rect` lays them)."""
    a = np.deg2rad(angle)
    v = np.array([np.cos(a), -np.sin(a)])
    n = np.array([-v[1], v[0]])
    c = np.array([cx, cy], float)
    return np.array([c - v * length / 2 - n * height / 2, c + v * length / 2 - n * height / 2,
                     c + v * length / 2 + n * height / 2, c - v * length / 2 + n * height / 2])


def _maps(rects, angles, tags=None, noise=0.0, tag_noise=0.0, angle_noise=0.0, w_in=1.0, w_out=0.0, block=1,
          lines=(), seed=0, size=SIZE):
    """core, geo (5, h, w), tag (4, h, w), ink, word_id for words `rects` read at `angles` (degrees).  Inside a word
    (its px (col, row) in the rectangle; later words over earlier): core `w_in`, the exact distances to its edges plus
    Gaussian noise of sd `noise` px, its angle plus noise of sd `angle_noise` degrees (one draw per `block` x `block`
    px: correlated votes), its tag (`tags[k]`, else 10 k on the first channel) plus noise of sd `tag_noise`.  Outside:
    core `w_out`, half-normal distances of sd 8 px, any direction and tags of sd 1 (the background's votes).  Ink: a
    block inside each word and `lines`.  geo is float64: the votes unrounded."""
    rng = np.random.default_rng([seed, 3])
    h = w = size
    rows, cols = np.mgrid[0:h, 0:w]
    p = np.stack([cols, rows], -1).astype(float)
    word_id = np.zeros((h, w), np.int32)
    for k, r in enumerate(rects):
        inside = cv2.pointPolygonTest
        poly = np.asarray(r, np.float32).reshape(-1, 1, 2)
        y0, x0 = np.maximum(np.floor(np.asarray(r).min(0)[::-1]).astype(int) - 1, 0)
        y1, x1 = np.minimum(np.ceil(np.asarray(r).max(0)[::-1]).astype(int) + 2, [h, w])
        for y in range(y0, y1):
            for x in range(x0, x1):
                if inside(poly, (float(x), float(y)), False) >= 0:
                    word_id[y, x] = k + 1
    core = np.where(word_id > 0, w_in, w_out).astype(np.float32)
    geo = np.zeros((5, h, w), np.float64)
    tag = rng.normal(0, 1.0, (4, h, w))
    n_words = len(rects)
    vecs = np.asarray(tags, float) if tags is not None else np.c_[10.0 * np.arange(n_words), np.zeros((n_words, 3))]
    nb = (h + block - 1) // block, (w + block - 1) // block
    dn = rng.normal(0, 1.0, (4, *nb)).repeat(block, 1).repeat(block, 2)[:, :h, :w] * noise
    tn = rng.normal(0, 1.0, (4, *nb)).repeat(block, 1).repeat(block, 2)[:, :h, :w] * tag_noise
    an = rng.normal(0, 1.0, nb).repeat(block, 0).repeat(block, 1)[:h, :w] * angle_noise
    for k, r in enumerate(rects):
        on = word_id == k + 1
        a = np.deg2rad(angles[k])
        along, up = np.array([np.cos(a), -np.sin(a)]), np.array([-np.sin(a), -np.cos(a)])
        ca, cb = np.asarray(r) @ along, np.asarray(r) @ up
        pa, pb = p[on] @ along, p[on] @ up
        geo[:4, on] = np.stack([cb.max() - pb, pb - cb.min(), pa - ca.min(), ca.max() - pa]) + dn[:, on]
        geo[4, on] = angles[k] + an[on]
        tag[:, on] = vecs[k][:, None] + tn[:, on]
    off = word_id == 0
    geo[:4, off] = np.abs(rng.normal(0, 8.0, (4, int(off.sum()))))
    geo[4, off] = rng.uniform(-180.0, 180.0, int(off.sum()))
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


def _turn(a, b) -> float:
    """The angle between two reading directions (degrees), in [0, 180]."""
    return abs((a - b + 180) % 360 - 180)


def _box_of(df, rect) -> int:
    """The row of the box meeting `rect` best."""
    return int(np.argmax([_iou(_g(df, k), rect) for k in range(len(df))]))


def _same_box(df, rect, tol):
    """`rect`'s box, its corners within `tol` px of `rect`'s (the same corner order)."""
    k = _box_of(df, rect)
    assert np.abs(_g(df, k) - rect).max() < tol
    return k


# --- one failure kind each ------------------------------------------------------------------------------------------

def test_two_touching_words_become_two_boxes():
    rects = [_rect(80, 80, 60, 20), _rect(140, 80, 60, 20)]     # side by side, the same tags: the votes part them
    core, geo, tag, ink, _ = _maps(rects, [0, 0], tags=[[0, 0, 0, 0]] * 2)
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    assert len(df) == 2
    for r in rects:
        _same_box(df, r, 1e-6)


def test_three_noisy_touching_words_become_three_boxes():
    """The chain that once came apart into 51 pieces: votes and tags noisy on every px."""
    rects = [_rect(55, 80, 50, 20), _rect(105, 80, 50, 20), _rect(155, 80, 50, 20)]
    spreads = words.Spreads(distance=1.0, angle=2.0, tag=0.3, cell=1.0, background_distance=8.0, background_tag=1.0)
    core, geo, tag, ink, _ = _maps(rects, [0, 0, 0], tags=[[0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]], noise=1.0,
                                   tag_noise=0.3, angle_noise=2.0)
    df = words.core_words(core, geo, tag, ink, spreads, PRIOR)
    assert len(df) == 3
    for r in rects:
        _same_box(df, r, 0.5)


def test_a_noisy_single_word_stays_one_word():
    """Votes and tags noisy on every px, and the background's own votes all around at core chance 0.001."""
    rect = _rect(120, 120, 100, 24)
    spreads = words.Spreads(distance=1.5, angle=2.0, tag=0.3, cell=1.0, background_distance=8.0, background_tag=1.0)
    core, geo, tag, ink, _ = _maps([rect], [0], noise=1.5, tag_noise=0.3, angle_noise=2.0, w_out=1e-3)
    df = words.core_words(core, geo, tag, ink, spreads, PRIOR)
    assert len(df) == 1
    _same_box(df, rect, 0.5)
    assert df["p_word"].iloc[0] == pytest.approx(1.0, abs=1e-3)


def test_a_long_word_whose_votes_scatter_comes_back_as_one_box():
    """A long word, its px's directions off by 5 degrees (sd): an end px's box turns about the px, so the votes'
    centres scatter and the proposals are several basins (12 here); joined by the likelihood, one box."""
    rect = _rect(120, 120, 220, 24)
    spreads = words.Spreads(distance=1.0, angle=5.0, tag=0.3, cell=1.0, background_distance=8.0, background_tag=1.0)
    core, geo, tag, ink, _ = _maps([rect], [0], noise=1.0, tag_noise=0.3, angle_noise=5.0)
    assert len(words._proposals(words._Votes(core, geo, tag, spreads), core.shape, spreads, PRIOR)) > 1
    df = words.core_words(core, geo, tag, ink, spreads, PRIOR)
    assert len(df) == 1
    assert _iou(_g(df, 0), rect) > 0.99


def test_a_word_whose_votes_are_alike_in_blocks_stays_one_word_by_its_measured_cell():
    """Votes alike over 4 x 4 px blocks (16 px a draw): taken as independent (cell 1) the word's likelihood grows with
    its px and it comes apart; at the cell its votes have (16) it is one word."""
    rect = _rect(120, 120, 160, 30)
    core, geo, tag, ink, _ = _maps([rect], [0], noise=1.5, tag_noise=0.3, angle_noise=2.0, block=4)
    measured = words.Spreads(distance=1.5, angle=2.0, tag=0.3, cell=16.0, background_distance=8.0, background_tag=1.0)
    df = words.core_words(core, geo, tag, ink, measured, PRIOR)
    assert len(df) == 1
    _same_box(df, rect, 1.0)
    independent = words.Spreads(**{**measured.__dict__, "cell": 1.0})
    assert len(words.core_words(core, geo, tag, ink, independent, PRIOR)) > 1


def test_votes_reading_the_word_the_other_way_round_stay_in_it():
    """A third of the word's px read it backwards (their direction turned a half turn, so their start is the word's
    end and their top its foot): one word, read the way most of its px read it."""
    rect = _rect(120, 120, 100, 24)
    core, geo, tag, ink, word_id = _maps([rect], [0])
    ys, xs = np.nonzero(word_id > 0)
    back = np.random.default_rng([0, 9]).random(len(ys)) < 1 / 3
    ys, xs = ys[back], xs[back]
    t, b, s, e = geo[:4, ys, xs]
    geo[:4, ys, xs] = np.stack([b, t, e, s])
    geo[4, ys, xs] = 180.0
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    assert len(df) == 1
    _same_box(df, rect, 1e-6)
    assert data.reading_angle(_g(df, 0)) == 0


def test_a_line_touching_a_word_does_not_extend_its_box():
    """A line runs on from the word's foot; its px have core 0.6 and vote for thin boxes along the line."""
    rect = _rect(100, 100, 80, 24)
    core, geo, tag, ink, word_id = _maps([rect], [0], lines=[((140, 110), (230, 110), 2)])
    on_line = np.zeros_like(core, bool)
    on_line[109:112, 140:231] = True
    on_line &= word_id == 0
    core[on_line] = 0.6
    ys, xs = np.nonzero(on_line)
    geo[0, on_line], geo[1, on_line] = ys - 109.0, 111.0 - ys     # each votes for one thin box, 90 x 2 px, the line's
    geo[2, on_line], geo[3, on_line] = xs - 140.0, 230.0 - xs
    geo[4, on_line] = 0
    tag[:, on_line] = 0
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    k = _same_box(df, rect, 1e-6)
    t = _g(df, k, "t")
    assert t[:, 0].max() <= rect[:, 0].max() + 1e-6                 # the trimmed box stops at the word too
    others = df.drop(index=df.index[k])
    assert len(others) == 1                                          # the line's votes: a box of their own ...
    assert others["p_word"].iloc[0] == pytest.approx(0.6, abs=1e-3)  # ... kept with its chance


def test_a_strong_minority_cluster_of_votes_becomes_its_own_box():
    """A code printed on a street name, centred on it: its px (15% of the name's) vote for the code's own box, the
    rest for the name's; the two have one centre and one tag."""
    name, code = _rect(120, 120, 120, 30), _rect(120, 120, 40, 14)
    core, geo, tag, ink, _ = _maps([name, code], [0, 0], tags=[[0, 0, 0, 0]] * 2)
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    assert len(df) == 2
    _same_box(df, name, 1e-6)
    _same_box(df, code, 1e-6)


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
        assert g[:, 0].max() == pytest.approx(rect[:, 0].max(), abs=1e-6)
    else:
        assert g[:, 1].max() == pytest.approx(SIZE, abs=1e-9)
        assert g[:, 1].min() == pytest.approx(rect[:, 1].min(), abs=1e-6)
    assert df["ink_px"].iloc[0] > 0
    assert data.reading_angle(g) == 0


@pytest.mark.parametrize("angle", [30, 90, 150, -45, 180])
def test_rotated_words_come_back_exactly_from_unrounded_votes(angle):
    rect = _rect(120, 120, 100, 24, angle)
    core, geo, tag, ink, _ = _maps([rect], [angle])
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR)
    assert len(df) == 1
    g = _g(df, 0)
    assert np.abs(g - rect).max() < 1e-6                             # the same corners, in the same order
    assert _turn(data.reading_angle(g), angle) <= 0.5
    assert wordboxes.axis_error(df["axis"].iloc[0], angle) < 1e-6


# --- the targets, the inputs and the table ---------------------------------------------------------------------------

def test_task_5s_targets_come_back_as_their_rectangles():
    """Votes painted by `data.crop_targets(..., words=True)` from rounded rectangles (cores the words' own,
    `shrink.prepare`), two words touching: each rectangle comes back exactly."""
    rects = [_rect(80, 80, 60, 20), _rect(140, 80, 60, 20), _rect(170, 170, 70, 22, 90)]
    rounded = np.round(np.array(rects)).astype(np.int32)
    sd = data.SheetData(series=paths.SERIES, sheet="s", height=SIZE, width=SIZE,
                        ink_bits=np.packbits(np.zeros((SIZE, SIZE), bool), axis=1), words=rounded,
                        kinds=np.zeros(3, np.uint8),
                        angles=np.array([data.reading_angle(r) for r in rounded], np.int16),
                        other=np.zeros((0, 4, 2), np.int32), points=np.zeros((0, 2), np.int32),
                        world=np.zeros((0, 0, 2)), ft_per_px=float("nan"))
    sd, = shrink.prepare([sd])
    t = data.crop_targets(sd, SIZE // 2, SIZE // 2, SIZE, 1.0, 0, words=True)
    vecs = np.zeros((4, 4), np.float32)
    vecs[1:, 0] = [0, 0, 10]
    tag = np.moveaxis(vecs[t["tag_id"]], -1, 0)
    df = words.core_words(t["core"].astype(np.float32), t["geo"], tag, t["ink"], EXACT, PRIOR)
    assert len(df) == 3
    for r in rounded.astype(float):
        _same_box(df, r, 1e-6)


def test_the_direction_given_as_cos_and_sin_reads_the_same_as_degrees():
    rects = [_rect(80, 80, 90, 22, 30), _rect(170, 170, 70, 20, -60)]
    core, geo, tag, ink, _ = _maps(rects, [30, -60], noise=0.5, tag_noise=0.1)
    a = np.deg2rad(geo[4].astype(np.float64))
    six = np.concatenate([geo[:4], 3 * np.cos(a)[None], 3 * np.sin(a)[None]])     # any length: the net's channels 5-6
    spreads = words.Spreads(distance=0.5, angle=1.0, tag=0.1, cell=1.0, background_distance=8.0, background_tag=1.0)
    pd.testing.assert_frame_equal(words.core_words(core, geo, tag, ink, spreads, PRIOR),
                                  words.core_words(core, six, tag, ink, spreads, PRIOR), atol=1e-5)


def test_the_boxes_table_has_the_word_boxes_columns_with_their_meaning():
    rects = [_rect(70, 70, 60, 20), _rect(160, 160, 100, 26, 90)]
    core, geo, tag, ink, word_id = _maps(rects, [0, 90], w_in=0.8)
    kind_p = np.zeros((4, SIZE, SIZE), np.float32)
    kind_p[0], kind_p[2] = 0.75, 0.25
    df = words.core_words(core, geo, tag, ink, EXACT, PRIOR, kind_p=kind_p, sheet="p07")
    assert list(df.columns) == [*wordboxes.COLUMNS, *words.EXTRA_COLUMNS]
    assert words.EXTRA_COLUMNS == ("p_word", "core_mass", "tag_spread")
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
        assert df["p_word"].iloc[row] == df["mean_chance"].iloc[row] == pytest.approx(0.8)
        assert df["core_mass"].iloc[row] == pytest.approx(0.8 * (word_id == k + 1).sum(), rel=1e-6)
        assert df["tag_spread"].iloc[row] == pytest.approx(0.0, abs=1e-6)
        assert (df["p_zoning"].iloc[row], df["p_street"].iloc[row]) == pytest.approx((0.75, 0.25))


def test_bad_inputs_are_refused_and_no_core_gives_an_empty_table():
    z = np.zeros((SIZE, SIZE), np.float32)
    df = words.core_words(z, np.zeros((5, SIZE, SIZE), np.float32), np.zeros((4, SIZE, SIZE), np.float32), z > 0,
                          EXACT, PRIOR)
    assert len(df) == 0 and list(df.columns) == [*wordboxes.COLUMNS, *words.EXTRA_COLUMNS]
    with pytest.raises(ValueError, match="geo"):
        words.core_words(z, np.zeros((4, SIZE, SIZE)), np.zeros((4, SIZE, SIZE)), z > 0, EXACT, PRIOR)
    with pytest.raises(ValueError, match="spread"):
        words.core_words(z, np.zeros((5, SIZE, SIZE)), np.zeros((4, SIZE, SIZE)), z > 0,
                         words.Spreads(0.0, 1.0, 0.3, 1.0, 8.0, 1.0), PRIOR)


# --- measuring the spreads and fitting the prior (Task 8's, on the training quarters) --------------------------------

@pytest.mark.parametrize("block", [1, 3])
def test_measure_spreads_finds_the_spreads_the_votes_were_drawn_with(block):
    """Noise of sd 1.5 px on every distance, 2 degrees on the direction and 0.3 on every tag; drawn once per px, or
    once per 3 x 3 px (9 px a draw: a word's votes are worth n / 9 independent ones)."""
    rects = [_rect(60 + 60 * (k % 3), 50 + 50 * (k // 3), 50, 18) for k in range(12)]
    pages = []
    for seed in range(3):
        core, geo, tag, ink, word_id = _maps(rects, [0] * 12, noise=1.5, tag_noise=0.3, angle_noise=2.0, w_out=1e-3,
                                             block=block, seed=seed)
        pages.append({"core": core, "geo": geo, "tag": tag, "word_id": word_id, "words": np.array(rects)})
    got = words.measure_spreads(pages)
    assert got.distance == pytest.approx(1.5, rel=0.05)
    assert got.angle == pytest.approx(2.0, rel=0.05)
    assert got.tag == pytest.approx(0.3, rel=0.05)
    assert got.cell == pytest.approx(block * block, rel=0.35)
    assert got.background_distance == pytest.approx(8.0, rel=0.05)    # half-normal of sd 8: root mean square 8
    assert got.background_tag == pytest.approx(1.0, rel=0.05)


def test_fit_word_prior_recovers_the_prior_the_decisions_were_drawn_with():
    """Each extra word's gain in log likelihood, and whether it was real, drawn from the decision the words are made
    by: P(real) = sigmoid(gain + prior)."""
    rng = np.random.default_rng([0, 5])
    true = -6.0
    gains = rng.exponential(8.0, 3000)
    real = rng.random(3000) < 1 / (1 + np.exp(-(gains + true)))
    assert words.fit_word_prior(gains, real) == pytest.approx(true, abs=0.4)


def test_fit_word_prior_refuses_decisions_of_one_kind():
    with pytest.raises(ValueError, match="finite"):
        words.fit_word_prior(np.array([1.0, 5.0, 30.0]), np.array([True, True, True]))


def test_an_extra_words_gain_is_large_for_two_words_and_small_for_one():
    """`extra_word_gain`, what Task 8 fits the prior on: the px of two touching words (a real extra word) against a
    noisy single word's (none)."""
    spreads = words.Spreads(distance=1.0, angle=2.0, tag=0.3, cell=1.0, background_distance=8.0, background_tag=1.0)
    two = [_rect(80, 80, 60, 20), _rect(140, 80, 60, 20)]
    core, geo, tag, ink, word_id = _maps(two, [0, 0], noise=1.0, tag_noise=0.3, angle_noise=2.0)
    real = words.extra_word_gain(core, geo, tag, spreads, word_id > 0)
    core, geo, tag, ink, word_id = _maps([_rect(110, 80, 120, 20)], [0], noise=1.0, tag_noise=0.3, angle_noise=2.0)
    none = words.extra_word_gain(core, geo, tag, spreads, word_id > 0)
    assert real > 1000 and 0 <= none < -PRIOR


def test_a_words_gain_over_none_is_large_on_a_word_and_small_on_the_background():
    """`word_gain`, the other decision the prior is fitted on: a word's px (core 0.9) against background px only
    (core 0.001, their votes the background's)."""
    spreads = words.Spreads(distance=1.0, angle=2.0, tag=0.3, cell=1.0, background_distance=8.0, background_tag=1.0)
    core, geo, tag, ink, word_id = _maps([_rect(110, 80, 120, 20)], [0], noise=1.0, tag_noise=0.3, angle_noise=2.0,
                                         w_in=0.9, w_out=1e-3)
    assert words.word_gain(core, geo, tag, spreads, word_id > 0) > 1000
    away = np.zeros_like(word_id, bool)
    away[150:200, 20:70] = True                                       # 2,500 background px, far from the word
    assert words.word_gain(core, geo, tag, spreads, away) < -PRIOR

