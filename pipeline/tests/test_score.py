import math

import numpy as np
import pytest

from sfzmaps import config, score
from sfzmaps.placement import Agreement
from sfzmaps.refine import Match, RefineResult
from sfzmaps.transform import GCP, SheetTransform

# transform=None: holdout_residuals falls back to dist_ft, so these tests set the held-out
# distances directly through dist_ft.


def _result(dists, labels=None, converged=True, vote_status="ok", classify_converged=True):
    labels = labels or ["inlier"] * len(dists)
    ms = [Match(float(i), 0.0, float(i), 0.0, float(d), lb)
          for i, (d, lb) in enumerate(zip(dists, labels))]
    return RefineResult(transform=None, matches=ms, iterations=3, converged=converged,
                        vote_status=vote_status, classify_converged=classify_converged)


def _dists(median, p95, n=100):
    """n distances with exactly this median and np.quantile(.., 0.95)."""
    d = np.full(n, median, float)
    d[: n // 2 - 1] = median / 2
    d[-6:] = p95          # indices 94..99: quantile(0.95) = d[94] + 0.05 (d[95] - d[94]) = p95
    return list(d)


def test_pass_rule_boundaries():
    assert (config.PASS_MEDIAN_FT, config.PASS_P95_FT) == (5.0, 12.0)
    s = score.score(_result(_dists(5.0, 12.0)))
    assert s.holdout_p50_ft == 5.0 and s.holdout_p95_ft == 12.0
    assert s.passed and s.reason is None
    s = score.score(_result(_dists(5.01, 12.0)))
    assert not s.passed and s.reason == "holdout_median"
    s = score.score(_result(_dists(5.0, 12.01)))
    assert not s.passed and s.reason == "holdout_p95"


def test_insample_fields_are_display_only():
    # 95% within 6 ft no longer decides anything
    s = score.score(_result([1.0] * 60 + [11.0] * 40))
    assert s.insample_frac_within == 0.6 and s.passed
    s = score.score(_result([6.0] * 50))
    assert s.insample_frac_within == 1.0 and not s.passed and s.reason == "holdout_median"


def test_score_too_few_matches():
    s = score.score(_result([1.0] * 39))
    assert s.n_scored == 39 and s.passed is False and s.reason == "too_few_matches"
    assert score.score(_result([1.0] * 40)).passed is True


def test_score_excludes_real_change_and_no_equivalent():
    labels = ["inlier"] * 45 + ["outlier"] * 2 + ["real_change"] * 8 + ["no_equivalent"] * 7
    dists = [1.0] * 45 + [3.0, 30.0] + [25.0] * 8 + [math.inf] * 7
    s = score.score(_result(dists, labels))
    assert s.n_scored == 47 and s.n_holdout == 47
    assert s.n_real_change == 8 and s.n_no_equivalent == 7
    assert s.insample_frac_within == 46 / 47
    assert s.insample_max_ft == 30.0 and s.insample_p50_ft == 1.0
    assert s.insample_p95_ft == float(np.quantile(dists[:47], 0.95))
    assert s.passed is True


def test_score_insufficient_anchors():
    s = score.score(None)
    assert s.passed is False and s.reason == "insufficient_anchors" and s.n_scored == 0


def test_score_empty_result():
    s = score.score(_result([]))
    assert s.n_scored == 0 and s.insample_frac_within == 0.0
    assert math.isnan(s.insample_p50_ft) and math.isnan(s.holdout_p50_ft)
    assert s.reason == "too_few_matches"


def _with_real_change(n_scored, n_rc):
    return _result([1.0] * n_scored + [30.0] * n_rc, ["inlier"] * n_scored + ["real_change"] * n_rc)


def test_score_real_change_fraction_boundary():
    assert config.MAX_REAL_CHANGE_FRAC == 0.15
    assert score.score(_with_real_change(85, 15)).passed
    assert score.score(_with_real_change(84, 16)).reason == "excess_real_change"
    assert score.score(_with_real_change(850, 151)).reason == "excess_real_change"


def test_score_ambiguous_placement():
    for st in ("ambiguous", "edge"):
        assert score.score(_result([1.0] * 100, vote_status=st)).reason == "ambiguous_placement"
    assert score.score(_result([1.0] * 100, vote_status="none")).passed


def _inlier_share(frac, n=100):
    """n inliers among n / frac finite corners (the rest no_equivalent)."""
    total = round(n / frac)
    return _result([1.0] * total, ["inlier"] * n + ["no_equivalent"] * (total - n))


def test_misplaced_by_label_agreement_needs_weak_corners():
    # overall label failure with weak corners (10% inliers): misplaced
    s = score.score(_inlier_share(0.10), agreement=Agreement(0.19, 20, 20, 0))
    assert not s.passed and s.reason == "misplaced" and s.label_agreement == 0.19
    assert score.score(_inlier_share(0.10), agreement=Agreement(0.2, 20, 20, 0)).passed
    # too few labels to judge: not misplaced on that evidence
    assert score.score(_inlier_share(0.10), agreement=Agreement(0.0, 19, 19, 0)).passed
    # the same label failure with healthy corners is renumbering/renaming (1921), not
    # misplacement: config.MISPLACED_WEAK_INLIER_FRAC is the boundary
    assert config.MISPLACED_WEAK_INLIER_FRAC == 0.15
    assert score.score(_inlier_share(0.15, 150), agreement=Agreement(0.1, 200, 200, 0, 0.9, 30)).passed
    assert score.score(_inlier_share(0.149, 149), agreement=Agreement(0.1, 200, 200, 0, 0.9, 30)).reason == "misplaced"
    assert score.score(_result([1.0] * 100), agreement=Agreement(0.0, 50, 50, 0)).passed
    # no corners at all: nothing speaks against the labels
    assert score.is_misplaced(Agreement(0.1, 50, 50, 0), float("nan"))


def test_misplaced_by_local_label_agreement():
    # labels landing near (within MISPLACED_LOCAL_FT of) their own block but not on it: a
    # shift of a few blocks, misplaced however good the corners look
    s = score.score(_result([1.0] * 100), agreement=Agreement(0.5, 100, 100, 0, 0.19, 20))
    assert s.reason == "misplaced" and s.label_agreement_local == 0.19
    assert score.score(_result([1.0] * 100), agreement=Agreement(0.5, 100, 100, 0, 0.2, 20)).passed
    assert score.score(_result([1.0] * 100), agreement=Agreement(0.5, 100, 100, 0, 0.0, 19)).passed


def test_misplacement_rule_on_task13c_sheets():
    # (overall agreement, n, local agreement, n_local, inlier share) from the real work dirs
    cases = {
        "1921/094212101": ((0.136, 264, 0.973, 37, 0.406), False),  # overlay: right place
        "1921/094156908": ((0.162, 74, 0.600, 20, 0.306), False),
        "1921/094254223": ((0.181, 127, 0.383, 60, 0.300), False),
        "1921/094333266": ((0.094, 64, 0.750, 8, 0.032), True),     # overlay: sheared, wrong
        "1921/094348509": ((0.105, 105, 0.367, 30, 0.062), True),   # weak fit, old street names
        "1948/10": ((0.018, 333, 0.545, 11, 0.019), True),
        "1948/14": ((0.621, 153, 0.699, 136, 0.095), False),
    }
    for sheet, ((f, n, lf, nl, inl), want) in cases.items():
        assert score.is_misplaced(Agreement(f, n, n, 0, lf, nl), inl) is want, sheet


def test_misplaced_by_inlier_fraction():
    # 40 inliers among 800 finite corners = 5%: passes; 39/800 < 5%: misplaced
    ok = _result([1.0] * 40 + [math.inf] * 760, ["inlier"] * 40 + ["no_equivalent"] * 760)
    assert score.score(ok).inlier_frac == 0.05 and score.score(ok).passed
    bad = _result([1.0] * 40 + [math.inf] * 761, ["inlier"] * 40 + ["no_equivalent"] * 761)
    assert score.score(bad).reason == "misplaced"


def test_score_reason_precedence():
    # misplaced > not_converged > ambiguous_placement > too_few_matches > excess > holdout
    bad = Agreement(0.0, 50, 50, 0, 0.0, 50)
    assert score.score(_result([1.0] * 3, converged=False), agreement=bad).reason == "misplaced"
    assert score.score(_result([1.0] * 3, converged=False, vote_status="edge")).reason == "not_converged"
    assert score.score(_result([1.0] * 100, classify_converged=False)).reason == "not_converged"
    assert score.score(_result([1.0] * 3, vote_status="ambiguous")).reason == "ambiguous_placement"
    assert score.score(_with_real_change(30, 20)).reason == "too_few_matches"
    r = _result([20.0] * 50 + [30.0] * 50, ["inlier"] * 50 + ["real_change"] * 50)
    assert score.score(r).reason == "excess_real_change"


# --- held-out residuals on a synthetic sheet -------------------------------------------

def _truth(p):
    p = np.asarray(p, float)
    u = p / 6000.0
    return np.c_[6_000_000 + 2.2 * p[:, 0] + 25 * np.sin(3 * u[:, 1]),
                 2_100_000 - 2.2 * p[:, 1] + 20 * np.cos(2 * u[:, 0])]


def _synthetic(sigma=3.0, seed=0):
    """Blocks of 4 corners on a 6000 px sheet, modern corners = truth + iid noise per corner."""
    rng = np.random.default_rng(seed)
    P, blk = [], []
    for bx in range(18):
        for by in range(18):
            x0, y0 = 200 + bx * 320, 200 + by * 320
            for dx, dy in ((0, 0), (250, 0), (0, 250), (250, 250)):
                P.append((x0 + dx, y0 + dy))
                blk.append(f"{bx:02d}{by:02d}")
    P = np.array(P, float)
    W = _truth(P) + rng.normal(0, sigma, P.shape)
    t = SheetTransform([GCP(p[0], p[1], w[0], w[1], "auto") for p, w in zip(P, W)], 0.1)
    fw = t.to_world(P)
    ms = [Match(p[0], p[1], w[0], w[1], float(np.hypot(*(f - w))), "inlier", i)
          for i, (p, w, f) in enumerate(zip(P, W, fw))]
    res = RefineResult(t, ms, 5, True, (0.0, 0.0), "ok", 15.0, W, np.array(blk))
    # independent fresh corners: the true error of this warp on blocks it never saw
    Q = rng.uniform(200, 5800, (2000, 2))
    true = np.linalg.norm(t.to_world(Q) - (_truth(Q) + rng.normal(0, sigma, Q.shape)), axis=1)
    return res, true


def test_holdout_is_pessimistic_vs_insample_and_close_to_truth():
    res, true = _synthetic()
    h = score.holdout_residuals(res)
    ins = np.array([m.dist_ft for m in res.matches])
    assert np.isfinite(h).all()
    assert np.median(h) >= np.median(ins)
    assert np.quantile(h, 0.95) >= np.quantile(ins, 0.95)
    assert np.median(h) == pytest.approx(np.median(true), rel=0.15)
    assert np.quantile(h, 0.95) == pytest.approx(np.quantile(true, 0.95), rel=0.15)


def test_holdout_deterministic_and_whole_blocks():
    res, _ = _synthetic()
    assert np.array_equal(score.holdout_residuals(res), score.holdout_residuals(res))
    ms = res.matches
    grp = score._groups(res, ms)
    # all corners of one block share a fold: refitting without the fold leaves none of them
    assert len(set(grp)) == 18 * 18


def test_holdout_uses_direct_distances_for_manual_transform():
    res, _ = _synthetic()
    manual = SheetTransform([GCP(g.px, g.py, g.wx, g.wy, "manual") for g in res.transform.gcps[::40]], 1.0)
    res.transform = manual
    h = score.holdout_residuals(res)
    P = np.array([[m.px, m.py] for m in res.matches])
    W = np.array([[m.wx, m.wy] for m in res.matches])
    assert np.allclose(h, np.linalg.norm(manual.to_world(P) - W, axis=1))
