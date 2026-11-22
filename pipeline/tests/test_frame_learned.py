"""The learned chances (issue #71, design spec amendment 6): how often a whole printed rule, and other ink, really is
frame or box given the model's reading, fitted on the training sheets' dev chances and applied to a page's."""
import numpy as np
import pytest

from sfzmaps.frame import learned

# a fit like the measured one (amendment 6): rules sigmoid(-6.03 + 2.17 logit(m) + 1.46 log(L)); other ink even at 0.46
MEASURED = learned.Chances(rule=(-6.03, 2.17, 1.46), ink=(0.16, 1.0))


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(x):
    return np.log(x / (1 - x))


def _ruled_page(h=200, w=1000, x0=100, x1=900, row=100):
    """A blank page with one 3-px rule from x0 to x1 (800 px long), its chances all map but for the rule."""
    lum = np.ones((h, w), np.float32)
    lum[row:row + 3, x0:x1] = 0.0
    p = np.zeros((3, h, w), np.float32)
    p[0] = 0.98
    p[1] = 0.02
    return lum, p


def test_fit_recovers_a_known_rule_curve():
    rng = np.random.default_rng(0)
    n = 100_000
    m = rng.uniform(0.01, 0.9, n)
    length = np.exp(rng.uniform(np.log(200), np.log(4000), n))
    ink = rng.uniform(100, 3000, n)
    a, b, c = -6.03, 2.17, 1.46
    y = rng.random(n) < _sig(a + b * _logit(m) + c * np.log(length))           # each rule wholly frame or map
    rules_ = np.stack([ink, m, y.astype(float), length], 1)
    bins = np.stack([np.full(learned.INK_BINS, 10.0), np.linspace(0.5, 9.5, learned.INK_BINS),
                     np.full(learned.INK_BINS, 5.0)], 1)
    got = learned.fit(rules_, bins).rule
    assert got == pytest.approx((a, b, c), rel=0.05)


def test_fit_recovers_a_known_ink_curve():
    rng = np.random.default_rng(1)
    i0, i1 = 0.16, 1.3
    ink = rng.uniform(1e3, 1e5, learned.INK_BINS)
    mean = (np.arange(learned.INK_BINS) + rng.uniform(0.2, 0.8, learned.INK_BINS)) / learned.INK_BINS
    bins = np.stack([ink, ink * mean, ink * _sig(i0 + i1 * _logit(mean))], 1)
    bins[40] = 0.0                                                             # an empty bin is dropped
    rules_ = np.array([[100.0, 0.3, 0.9, 1000.0], [100.0, 0.05, 0.1, 300.0], [50.0, 0.2, 0.5, 600.0]])
    got = learned.fit(rules_, bins).ink
    assert got == pytest.approx((i0, i1), rel=0.05)


def test_rows_of_a_page():
    # an 800-px rule read 0.3 on its left half and 0.1 on its right, three-quarters of it labelled frame; other ink
    # read 0.625 (labelled box) and 0.004 (map)
    lum, p = _ruled_page()
    p[0, 100:103, 100:500], p[1, 100:103, 100:500] = 0.7, 0.3
    p[0, 100:103, 500:900], p[1, 100:103, 500:900] = 0.9, 0.1
    cls = np.zeros(lum.shape, np.uint8)
    cls[100:103, 100:700] = 1
    lum[20:24, 20:24] = 0.5                                                    # ink 0.5 x 16 px
    p[:, 20:24, 20:24] = np.array([0.375, 0.3, 0.325], np.float32)[:, None, None]
    cls[20:24, 20:24] = 2
    lum[150:152, 30:32] = 0.0                                                  # ink 1 x 4 px
    p[:, 150:152, 30:32] = np.array([0.996, 0.004, 0.0], np.float32)[:, None, None]
    r = learned.rule_rows(p, lum, cls)
    assert r.shape == (1, 4)
    assert r[0] == pytest.approx((2400.0, 0.2, 0.75, 800.0), rel=1e-5)
    b = learned.ink_rows(p, lum, cls)
    assert b.shape == (learned.INK_BINS, 3)
    assert b[int(0.625 * learned.INK_BINS)] == pytest.approx((8.0, 8.0 * 0.625, 8.0), rel=1e-5)
    assert b[0] == pytest.approx((4.0, 4.0 * 0.004, 0.0), rel=1e-4)
    assert b[:, 0].sum() == pytest.approx(12.0)                               # the rule's ink is in no bin


def test_apply_gives_a_whole_rule_one_learned_chance():
    # a long rule read 0.3 on half and 0.1 on half: under the measured fit, an 800-px rule read 0.2 is frame
    lum, p = _ruled_page()
    p[0, 100:103, 100:500], p[1, 100:103, 100:500] = 0.7, 0.3
    p[0, 100:103, 500:900], p[1, 100:103, 500:900] = 0.9, 0.1
    q = learned.apply(p, lum, MEASURED)
    on = lum < 0.5
    c = q[1][on] + q[2][on]
    want = _sig(-6.03 + 2.17 * _logit(0.2) + 1.46 * np.log(800))
    assert want > 0.5
    assert c == pytest.approx(np.full(c.shape, want), abs=1e-5)
    assert q[0][on] == pytest.approx(1 - c, abs=1e-6)


def test_apply_keeps_the_models_split_between_outside_and_box():
    lum, p = _ruled_page()
    p[:, 100:103, 100:900] = np.array([0.5, 0.3, 0.2], np.float32)[:, None, None]   # the rule: outside 3 : box 2
    p[:, 10:20, 10:20] = np.array([0.4, 0.15, 0.45], np.float32)[:, None, None]     # other ink: outside 1 : box 3
    lum[10:20, 10:20] = 0.2
    p[:, 30:40, 30:40] = np.array([1.0, 0.0, 0.0], np.float32)[:, None, None]       # read as plain map
    lum[30:40, 30:40] = 0.2
    q = learned.apply(p, lum, MEASURED)
    rule = q[:, 100:103, 100:900]
    assert rule[2] == pytest.approx(rule[1] * 2 / 3, rel=1e-4)
    other = q[:, 10:20, 10:20]
    assert other[2] == pytest.approx(other[1] * 3, rel=1e-4)
    plain = q[:, 30:40, 30:40]
    assert (plain[2] == 0).all() and (plain[1] > 0).all()                      # no reading of a box: class 1
    assert q.sum(0) == pytest.approx(np.ones(lum.shape), abs=1e-5)


def test_apply_leaves_other_ink_to_the_ink_curve():
    lum, p = _ruled_page()
    lum[50:60, 50:60] = 0.4
    p[:, 50:60, 50:60] = np.array([0.6, 0.25, 0.15], np.float32)[:, None, None]
    q = learned.apply(p, lum, MEASURED)
    want = _sig(0.16 + 1.0 * _logit(0.4))
    assert q[1, 50:60, 50:60] + q[2, 50:60, 50:60] == pytest.approx(np.full((10, 10), want), abs=1e-6)
    assert q.dtype == np.float32 and q.shape == p.shape


def test_apply_does_not_modify_its_input():
    lum, p = _ruled_page()
    p[:, 100:103, 100:500] = np.array([0.7, 0.3, 0.0], np.float32)[:, None, None]
    before = p.copy()
    lum_before = lum.copy()
    learned.apply(p, lum, MEASURED)
    np.testing.assert_array_equal(p, before)
    np.testing.assert_array_equal(lum, lum_before)


def test_apply_takes_the_rule_labels_given():
    # the caller's labels name no rule: the printed rule's ink then follows the ink curve, not the rule curve
    lum, p = _ruled_page()
    none = (np.zeros(lum.shape, np.int32), 0)
    q = learned.apply(p, lum, MEASURED, none)
    on = lum < 0.5
    want = _sig(0.16 + 1.0 * _logit(0.02))
    assert q[1][on] + q[2][on] == pytest.approx(np.full(int(on.sum()), want), abs=1e-6)


def test_chances_round_trip_through_json():
    import json

    c = learned.Chances(rule=(-6.0, 2.1, 1.4), ink=(0.2, 1.1))
    assert learned.Chances.from_json(json.loads(json.dumps(c.to_json()))) == c
