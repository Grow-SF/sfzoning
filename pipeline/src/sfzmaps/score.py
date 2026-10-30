"""Per-sheet alignment score: does the sheet pass, or go to human review?

Pass rule (owner, 2026-10-29, after Task 13): measured on HELD-OUT blocks, the median
corner error is <= config.PASS_MEDIAN_FT and the 95th percentile is <= config.PASS_P95_FT,
and the sheet is not misplaced (plus the fit-health guards below).

Scale-aware limits (owner, 2026-10-29): both limits are multiplied by
k = max(1, ft_per_px / config.PASS_REF_FT_PER_PX), because corner error is roughly constant
in scan pixels and the 1980+ office-copier scans are coarser. ft_per_px is the final
transform's median local scale over the page (sheet_ft_per_px), computed once per score and
stored with the applied limits (pass_limits).

Scored corners are refine matches labelled inlier or outlier. real_change (the historic
geometry really differs from today's; kept on purpose) and no_equivalent (nothing modern
to compare against) are counted but never scored. dist_ft is the distance from the scan
corner, mapped by the sheet transform, to its modern corner.

In-sample vs held-out. The automatic warp is fit on the inlier corners themselves, so
their dist_ft is optimistic (Task 13: 5-15 percentage points at 6 ft). holdout_residuals
splits the modern blocks into HOLDOUT_FOLDS folds (seeded, deterministic), refits the
warp with the same smoothing on the inliers of the other folds, and measures each fold's
scored corners against it: the error of the warp on blocks it never saw. When the
transform has no automatic GCPs (a human review), the corners were not used to fit it, so
their distances are already held-out and are used as is. The insample_* fields are kept
for display only.

reason, first that applies (this order is the precedence); passed exactly when None:
  * "insufficient_anchors" - no RefineResult at all (score(None)): the rough fit from label
                             anchors failed, so the sheet was never refined. All counts
                             are zero; no other reason is considered;
  * "misplaced"            - the sheet sits in the wrong place (is_misplaced): almost none
                             of its corners found a modern partner (inlier share below
                             config.MISPLACED_MIN_INLIER_FRAC); or its labels near their
                             named block/street miss it (sfzmaps.placement local agreement
                             below config.MISPLACED_MAX_LABEL_AGREEMENT over at least
                             MISPLACED_MIN_LABELS labels); or its labels disagree overall
                             AND its corners are weak (inlier share below
                             MISPLACED_WEAK_INLIER_FRAC): old block numbers and street
                             names disagree on well placed 1921 sheets too (Task 13c).
                             Checked before the fit-health reasons because it is the most
                             important thing to say;
  * "not_converged"        - refine used up max_iter before the final radius settled, or
                             classify_changes' labels were still changing after its last
                             round: the fit is suspect even if its numbers look good;
  * "ambiguous_placement"  - the translation vote refused to shift (vote_status "edge" or
                             "ambiguous", ruling R17);
  * "too_few_matches"      - n_scored < min_scored, or fewer than min_scored held-out
                             corners could be measured;
  * "excess_real_change"   - n_real_change > config.MAX_REAL_CHANGE_FRAC x
                             (n_scored + n_real_change): a systematic misfit looks like
                             real change;
  * "holdout_median"       - held-out median > config.PASS_MEDIAN_FT x k;
  * "holdout_p95"          - held-out 95th percentile > config.PASS_P95_FT x k.
Percentiles are NaN when nothing is scored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sfzmaps import config
from sfzmaps.refine import RefineResult
from sfzmaps.transform import GCP, SheetTransform

HOLDOUT_FOLDS = 5
HOLDOUT_SEED = 20261029
HOLDOUT_TILE_FT = 300.0   # group key when the modern block of a corner is unknown


@dataclass
class SheetScore:
    n_scored: int
    insample_frac_within: float     # share of scored corners within config.TOLERANCE_FT (display)
    insample_p50_ft: float
    insample_p95_ft: float
    insample_max_ft: float
    n_real_change: int
    n_no_equivalent: int
    passed: bool
    reason: str | None
    holdout_p50_ft: float = float("nan")
    holdout_p95_ft: float = float("nan")
    n_holdout: int = 0
    label_agreement: float = float("nan")   # sfzmaps.placement share (NaN: not checked)
    inlier_frac: float = float("nan")       # inliers / finite scan corners
    label_agreement_local: float = float("nan")   # placement local_fraction (NaN: not checked)
    ft_per_px: float = float("nan")         # final transform's median local scale (NaN: unknown)
    pass_scale_k: float = 1.0               # max(1, ft_per_px / PASS_REF_FT_PER_PX)
    pass_median_limit_ft: float = config.PASS_MEDIAN_FT   # the limits actually applied
    pass_p95_limit_ft: float = config.PASS_P95_FT

    @classmethod
    def empty(cls, reason: str) -> "SheetScore":
        nan = float("nan")
        return cls(0, 0.0, nan, nan, nan, 0, 0, False, reason)


def _groups(result: RefineResult, ms) -> np.ndarray:
    """Group key per match: its modern block number, else a HOLDOUT_TILE_FT tile."""
    keys = []
    for m in ms:
        if result.ref_block is not None and m.ref_idx >= 0:
            keys.append("b" + str(result.ref_block[m.ref_idx]))
        else:
            keys.append("t%d_%d" % (np.floor(m.wx / HOLDOUT_TILE_FT), np.floor(m.wy / HOLDOUT_TILE_FT)))
    return np.array(keys)


def holdout_residuals(result: RefineResult, folds: int = HOLDOUT_FOLDS,
                      seed: int = HOLDOUT_SEED) -> np.ndarray:
    """Held-out distance (ft) per scored corner (inlier/outlier, in match order); NaN where
    it could not be measured. See the module docstring."""
    ms = [m for m in result.matches if m.label in ("inlier", "outlier")]
    if not ms:
        return np.empty(0)
    P = np.array([[m.px, m.py] for m in ms], float)
    W = np.array([[m.wx, m.wy] for m in ms], float)
    t = result.transform
    if t is None:
        return np.array([m.dist_ft for m in ms], float)
    if not any(g.enabled and g.source == "auto" for g in t.gcps):
        return np.linalg.norm(t.to_world(P) - W, axis=1)    # corners not used in the fit
    grp = _groups(result, ms)
    uniq = np.unique(grp)
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = dict(zip(uniq[order], np.arange(len(uniq)) % folds))
    fold = np.array([fold_of[g] for g in grp])
    inl = np.array([m.label == "inlier" for m in ms])
    out = np.full(len(ms), np.nan)
    for f in range(folds):
        test = fold == f
        train = inl & ~test
        if not test.any() or train.sum() < 10:
            continue
        gcps = [GCP(p[0], p[1], w[0], w[1], "auto") for p, w in zip(P[train], W[train])]
        try:
            tf = SheetTransform(gcps, smoothing=t.smoothing)
            out[test] = np.linalg.norm(tf.to_world(P[test]) - W[test], axis=1)
        except ValueError:
            continue
    return out


def is_misplaced(agreement, inlier_frac: float) -> bool:
    """The misplacement rule (config.MISPLACED_*): corners alone (inlier share below
    MISPLACED_MIN_INLIER_FRAC); or labels near their named feature but not on it (local
    agreement); or labels that disagree overall, believed only when the corners are weak
    too (inlier share below MISPLACED_WEAK_INLIER_FRAC, or no corners at all), because
    renumbered blocks and renamed streets disagree on correctly placed sheets as well.
    agreement: a placement.Agreement or None; inlier_frac: NaN when there are no corners."""
    def fails(frac, n):
        return n >= config.MISPLACED_MIN_LABELS and frac < config.MISPLACED_MAX_LABEL_AGREEMENT

    no_corners = not np.isfinite(inlier_frac)
    if not no_corners and inlier_frac < config.MISPLACED_MIN_INLIER_FRAC:
        return True
    if agreement is None:
        return False
    if fails(agreement.local_fraction, agreement.n_local):
        return True
    weak = no_corners or inlier_frac < config.MISPLACED_WEAK_INLIER_FRAC
    return bool(weak and fails(agreement.fraction, agreement.n))


def pass_limits(ft_per_px: float) -> tuple[float, float, float]:
    """(k, median limit ft, p95 limit ft) for a sheet at ft_per_px; k = 1 when unknown."""
    k = max(1.0, ft_per_px / config.PASS_REF_FT_PER_PX) if np.isfinite(ft_per_px) else 1.0
    return k, config.PASS_MEDIAN_FT * k, config.PASS_P95_FT * k


def sheet_ft_per_px(t: SheetTransform | None, shape: tuple[int, int] | None = None) -> float:
    """Median local scale (ft/px) of transform t over the page (h, w); without shape, over
    the box from the origin to its furthest enabled GCP. NaN when it cannot be measured."""
    if t is None:
        return float("nan")
    try:
        if shape is None:
            P = np.array([[g.px, g.py] for g in t.gcps if g.enabled], float)
            if len(P) < 3:
                return float("nan")
            shape = (int(P[:, 1].max()) + 1, int(P[:, 0].max()) + 1)
        return t.median_scale_ft_per_px(shape)
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")


def _q(d: np.ndarray, q: float) -> float:
    return float(np.quantile(d, q)) if len(d) else float("nan")


def score(result: RefineResult | None, tol_ft: float | None = None, min_scored: int = 40,
          agreement=None, ft_per_px: float | None = None,
          shape: tuple[int, int] | None = None) -> SheetScore:
    """agreement: an sfzmaps.placement.Agreement for the final transform, or None (not
    checked). tol_ft only sets the display share insample_frac_within. ft_per_px: the sheet
    scale for the pass limits; when None it is measured from result.transform over the page
    shape (h, w) (sheet_ft_per_px)."""
    if result is None:
        return SheetScore.empty("insufficient_anchors")
    if ft_per_px is None:
        ft_per_px = sheet_ft_per_px(result.transform, shape)
    k, lim50, lim95 = pass_limits(ft_per_px)
    tol = config.TOLERANCE_FT if tol_ft is None else tol_ft
    labels = [m.label for m in result.matches]
    d = np.array([m.dist_ft for m in result.matches if m.label in ("inlier", "outlier")], float)
    n = len(d)
    n_rc = labels.count("real_change")
    finite = sum(np.isfinite([m.px, m.py]).all() for m in result.matches)
    inl_frac = labels.count("inlier") / finite if finite else float("nan")
    frac = float(np.count_nonzero(d <= tol)) / n if n else 0.0
    h = holdout_residuals(result) if n else np.empty(0)
    h = h[np.isfinite(h)]
    hp50, hp95 = _q(h, 0.5), _q(h, 0.95)
    agree = float("nan") if agreement is None else float(agreement.fraction)
    misplaced = is_misplaced(agreement, inl_frac if finite else float("nan"))
    if misplaced:
        reason = "misplaced"
    elif not (result.converged and result.classify_converged):
        reason = "not_converged"
    elif result.vote_status in ("edge", "ambiguous"):
        reason = "ambiguous_placement"
    elif n < min_scored or len(h) < min_scored:
        reason = "too_few_matches"
    elif n_rc > config.MAX_REAL_CHANGE_FRAC * (n + n_rc):
        reason = "excess_real_change"
    elif hp50 > lim50:
        reason = "holdout_median"
    elif hp95 > lim95:
        reason = "holdout_p95"
    else:
        reason = None
    return SheetScore(n, frac, _q(d, 0.5), _q(d, 0.95), float(d.max()) if n else float("nan"),
                      n_rc, labels.count("no_equivalent"), reason is None, reason,
                      hp50, hp95, len(h), agree, inl_frac,
                      float("nan") if agreement is None else float(agreement.local_fraction),
                      float(ft_per_px), k, lim50, lim95)
