import math

import numpy as np

from sfzmaps import score
from sfzmaps.refine import Match, RefineResult


def _result(dists, labels=None, converged=True):
    labels = labels or ["inlier"] * len(dists)
    ms = [Match(float(i), 0.0, float(i), 0.0, float(d), lb)
          for i, (d, lb) in enumerate(zip(dists, labels))]
    return RefineResult(transform=None, matches=ms, iterations=3, converged=converged)


def test_score_threshold_boundary():
    # 0.95 exactly passes
    s = score.score(_result([1.0] * 95 + [20.0] * 5))
    assert s.n_scored == 100 and s.frac_within == 0.95
    assert s.passed is True and s.reason is None
    # 0.949 fails
    s = score.score(_result([1.0] * 949 + [20.0] * 51))
    assert s.frac_within == 0.949
    assert s.passed is False and s.reason == "below_threshold"


def test_score_tolerance_is_inclusive():
    s = score.score(_result([6.0] * 50))
    assert s.frac_within == 1.0 and s.passed


def test_score_too_few_matches():
    s = score.score(_result([1.0] * 39))
    assert s.n_scored == 39 and s.frac_within == 1.0
    assert s.passed is False and s.reason == "too_few_matches"
    s = score.score(_result([1.0] * 40))
    assert s.passed is True


def test_score_excludes_real_change_and_no_equivalent():
    labels = ["inlier"] * 45 + ["outlier"] * 2 + ["real_change"] * 10 + ["no_equivalent"] * 7
    dists = [1.0] * 45 + [3.0, 30.0] + [25.0] * 10 + [math.inf] * 7
    s = score.score(_result(dists, labels))
    assert s.n_scored == 47
    assert s.n_real_change == 10 and s.n_no_equivalent == 7
    assert s.frac_within == 46 / 47
    assert s.max_ft == 30.0
    assert s.p50_ft == 1.0
    assert s.p95_ft == float(np.quantile(dists[:47], 0.95))
    assert s.passed is True


def test_score_not_converged_takes_precedence():
    s = score.score(_result([1.0] * 100, converged=False))
    assert s.passed is False and s.reason == "not_converged"
    s = score.score(_result([1.0] * 3, converged=False))
    assert s.reason == "not_converged"


def test_score_insufficient_anchors():
    s = score.score(None)
    assert s.passed is False and s.reason == "insufficient_anchors"
    assert s.n_scored == 0


def test_score_empty_result():
    s = score.score(_result([]))
    assert s.n_scored == 0 and s.frac_within == 0.0
    assert math.isnan(s.p50_ft) and math.isnan(s.max_ft)
    assert s.reason == "too_few_matches"
