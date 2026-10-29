import math

import numpy as np

from sfzmaps import config, score
from sfzmaps.refine import Match, RefineResult


def _result(dists, labels=None, converged=True, vote_status="ok", classify_converged=True):
    labels = labels or ["inlier"] * len(dists)
    ms = [Match(float(i), 0.0, float(i), 0.0, float(d), lb)
          for i, (d, lb) in enumerate(zip(dists, labels))]
    return RefineResult(transform=None, matches=ms, iterations=3, converged=converged,
                        vote_status=vote_status, classify_converged=classify_converged)


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
    labels = ["inlier"] * 45 + ["outlier"] * 2 + ["real_change"] * 8 + ["no_equivalent"] * 7
    dists = [1.0] * 45 + [3.0, 30.0] + [25.0] * 8 + [math.inf] * 7
    s = score.score(_result(dists, labels))
    assert s.n_scored == 47
    assert s.n_real_change == 8 and s.n_no_equivalent == 7
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


def _with_real_change(n_scored, n_rc):
    return _result([1.0] * n_scored + [30.0] * n_rc,
                   ["inlier"] * n_scored + ["real_change"] * n_rc)


def test_score_real_change_fraction_boundary():
    assert config.MAX_REAL_CHANGE_FRAC == 0.15
    s = score.score(_with_real_change(85, 15))      # exactly 15% passes
    assert s.n_real_change == 15 and s.passed and s.reason is None
    s = score.score(_with_real_change(84, 16))      # 16% fails
    assert not s.passed and s.reason == "excess_real_change"
    s = score.score(_with_real_change(850, 151))    # 15.08% fails
    assert s.reason == "excess_real_change"


def test_score_ambiguous_placement():
    for st in ("ambiguous", "edge"):
        s = score.score(_result([1.0] * 100, vote_status=st))
        assert not s.passed and s.reason == "ambiguous_placement"
    assert score.score(_result([1.0] * 100, vote_status="none")).passed


def test_score_reason_precedence():
    # not_converged > ambiguous_placement > too_few_matches > excess > below_threshold
    assert score.score(_result([1.0] * 3, converged=False, vote_status="edge")).reason == "not_converged"
    assert score.score(_result([1.0] * 100, classify_converged=False)).reason == "not_converged"
    assert score.score(_result([1.0] * 3, vote_status="ambiguous")).reason == "ambiguous_placement"
    assert score.score(_with_real_change(30, 20)).reason == "too_few_matches"
    r = _result([20.0] * 50 + [30.0] * 50, ["inlier"] * 50 + ["real_change"] * 50)
    assert score.score(r).reason == "excess_real_change"
