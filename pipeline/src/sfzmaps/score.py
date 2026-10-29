"""Per-sheet alignment score: does the sheet pass, or go to human review?

Scored corners are refine matches labelled inlier or outlier. real_change (the historic
geometry really differs from today's; kept on purpose) and no_equivalent (nothing modern
to compare against) are counted but never scored. dist_ft is the distance from the scan
corner, mapped by the sheet transform, to its modern corner.

passed = converged and n_scored >= min_scored and frac_within >= pass_fraction, with
frac_within = (scored corners with dist_ft <= tol_ft) / n_scored.

reason (None when passed), first that applies:
  * "insufficient_anchors" - no RefineResult at all (score(None)): the rough fit from
                             label anchors failed, so the sheet was never refined;
  * "not_converged"        - refine used up max_iter before the final radius settled
                             (RefineResult.converged is False); the fit is suspect even if
                             its numbers look good;
  * "too_few_matches"      - n_scored < min_scored;
  * "below_threshold"      - frac_within < pass_fraction.
Percentiles and max are NaN when nothing is scored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sfzmaps.config import PASS_FRACTION, TOLERANCE_FT
from sfzmaps.refine import RefineResult


@dataclass
class SheetScore:
    n_scored: int
    frac_within: float
    p50_ft: float
    p95_ft: float
    max_ft: float
    n_real_change: int
    n_no_equivalent: int
    passed: bool
    reason: str | None


def score(result: RefineResult | None, tol_ft: float = TOLERANCE_FT,
          pass_fraction: float = PASS_FRACTION, min_scored: int = 40) -> SheetScore:
    if result is None:
        nan = float("nan")
        return SheetScore(0, 0.0, nan, nan, nan, 0, 0, False, "insufficient_anchors")
    labels = [m.label for m in result.matches]
    d = np.array([m.dist_ft for m in result.matches if m.label in ("inlier", "outlier")], float)
    n = len(d)
    if n:
        frac = float(np.count_nonzero(d <= tol_ft)) / n
        p50, p95, mx = (float(np.quantile(d, 0.5)), float(np.quantile(d, 0.95)), float(d.max()))
    else:
        frac, p50, p95, mx = 0.0, float("nan"), float("nan"), float("nan")
    if not result.converged:
        reason = "not_converged"
    elif n < min_scored:
        reason = "too_few_matches"
    elif frac < pass_fraction:
        reason = "below_threshold"
    else:
        reason = None
    return SheetScore(n, frac, p50, p95, mx, labels.count("real_change"),
                      labels.count("no_equivalent"), reason is None, reason)
