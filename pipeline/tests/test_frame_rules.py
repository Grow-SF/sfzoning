"""A printed rule is one thing (issue #71, design spec amendment 5): `rules.rule_labels` finds the long thin straight
strokes and `rules.share` gives each one the ink-weighted mean of the model's chances along it."""
import math

import numpy as np

from sfzmaps.frame import rules

MAP = np.array([0.9, 0.07, 0.03], np.float32)  # the chances on paper and everything not a rule


def _blank(h=400, w=1000):
    lum = np.ones((h, w), np.float32)
    p = np.broadcast_to(MAP[:, None, None], (3, h, w)).astype(np.float32)
    return lum, p


def _labels_on(lab, mask):
    return set(np.unique(lab[mask]).tolist())


def test_a_rule_read_two_ways_takes_one_reading():
    # a 6 px rule across the page; the chances call its left 60% "outside" (0.95) and its right 40% map (outside
    # 0.2), and its right part is fainter ink (0.7), so the mean is weighted by ink
    lum, p = _blank()
    x0, x1 = 50, 950
    split = x0 + int(0.6 * (x1 - x0))
    rule = np.zeros(lum.shape, bool)
    rule[100:106, x0:x1] = True
    lum[100:106, x0:split] = 0.0
    lum[100:106, split:x1] = 0.3
    left, right = np.array([0.04, 0.95, 0.01], np.float32), np.array([0.79, 0.2, 0.01], np.float32)
    p[:, 100:106, x0:split] = left[:, None, None]
    p[:, 100:106, split:x1] = right[:, None, None]
    q = rules.share(p, lum)
    assert q.dtype == np.float32 and q.shape == p.shape
    got = q[:, rule]
    assert np.allclose(got, got[:, :1], atol=1e-6)  # every px of the rule has the same chances
    wl, wr = 1.0 * (split - x0), 0.7 * (x1 - split)
    want = (wl * left + wr * right) / (wl + wr)
    assert np.allclose(got[:, 0], want, atol=1e-5)
    assert (q[:, ~rule] == p[:, ~rule]).all()  # nothing else moves


def test_share_keeps_each_rules_expected_ink_per_class():
    # two rules (one horizontal, one vertical) with random chances and random ink above the stroke threshold
    rng = np.random.default_rng(0)
    lum, _ = _blank(600, 900)
    lum[80:85, 40:860] = rng.uniform(0.0, 0.45, (5, 820)).astype(np.float32)
    lum[150:560, 700:708] = rng.uniform(0.0, 0.45, (410, 8)).astype(np.float32)
    p = rng.dirichlet((1.0, 1.0, 1.0), lum.shape).transpose(2, 0, 1).astype(np.float32)
    lab, n = rules.rule_labels(lum)
    assert n == 2
    q = rules.share(p, lum, lab)
    ink = 1.0 - lum.astype(np.float64)
    for r in (1, 2):
        m = lab == r
        for k in range(3):
            before = (ink[m] * p[k][m]).sum()
            after = (ink[m] * q[k][m]).sum()
            assert math.isclose(before, after, rel_tol=1e-5), (r, k, before, after)
    assert (q[:, lab == 0] == p[:, lab == 0]).all()


def test_a_rule_tilted_two_degrees_is_one_rule():
    lum, _ = _blank(500, 1100)
    Y, X = np.mgrid[0:500, 0:1100]
    line = np.abs(Y - (200 + math.tan(math.radians(2.0)) * (X - 550))) < 3
    line &= (X >= 60) & (X < 1040)
    lum[line] = 0.0
    lab, n = rules.rule_labels(lum)
    assert n == 1
    assert (lab[line] == 1).mean() > 0.95  # all but the rotations' rounding at the stroke's edges


def test_a_double_frame_is_two_rules():
    # a 15 px bar and a 5 px rule 12 px inside it, as on a framed page's side
    lum, _ = _blank()
    bar, thin = np.zeros(lum.shape, bool), np.zeros(lum.shape, bool)
    bar[100:115, 60:940] = True
    thin[127:132, 60:940] = True
    lum[bar | thin] = 0.0
    lab, n = rules.rule_labels(lum)
    assert n == 2
    assert _labels_on(lab, bar) | _labels_on(lab, thin) == {1, 2}
    assert len(_labels_on(lab, bar)) == 1 and len(_labels_on(lab, thin)) == 1
    assert _labels_on(lab, bar) != _labels_on(lab, thin)


def test_a_solid_block_is_not_a_rule():
    lum, p = _blank(600, 900)
    block = np.zeros(lum.shape, bool)
    block[150:350, 200:500] = True  # 300 wide, 200 tall
    lum[block] = 0.0
    p = p.copy()
    p[:, 150:250] = np.array([0.1, 0.8, 0.1], np.float32)[:, None, None]  # a split reading over the block
    lab, n = rules.rule_labels(lum)
    assert n == 0 and not lab.any()
    q = rules.share(p, lum)
    assert (q[:, block] == p[:, block]).all()


def test_short_strokes_hatching_and_arcs_are_not_rules():
    lum, _ = _blank(900, 1200)
    Y, X = np.mgrid[0:900, 0:1200]
    lum[100:105, 100:100 + rules.RUN - 51] = 0.0  # a straight stroke 50 px short of a rule
    lum[200:200 + rules.RUN - 51, 100:105] = 0.0
    hatch = (np.abs(((X - Y) % 20) - 10) < 1.5) & (X >= 300) & (X < 700) & (Y >= 300) & (Y < 700)  # 45 degrees
    lum[hatch] = 0.0
    r = np.hypot(X - 1150, Y - 850)
    arc = (np.abs(r - 250) < 2) & (X < 1150) & (Y < 850)  # a corner's quarter circle, radius 250
    lum[arc] = 0.0
    lab, n = rules.rule_labels(lum)
    assert n == 0 and not lab.any()


def test_share_does_not_modify_its_input():
    lum, p = _blank()
    lum[100:106, 50:950] = 0.0
    p = p.copy()
    p[:, 100:106, 50:500] = np.array([0.04, 0.95, 0.01], np.float32)[:, None, None]
    p0, lum0 = p.copy(), lum.copy()
    q = rules.share(p, lum)
    assert (p == p0).all() and (lum == lum0).all()
    assert not np.shares_memory(q, p)
    assert not (q == p).all()  # it did change the rule, in its own copy
