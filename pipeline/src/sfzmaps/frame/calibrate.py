"""Temperature scaling and reliability for the frame finder's chances.

The seeds' logits are averaged as chances (`combine`); a single temperature T divides every logit first.  T is
the one number fitted to held-out labels, by minimising the multiclass log-loss, a proper scoring rule (Gneiting &
Raftery 2007), as in temperature scaling (Guo et al. 2017).  The sheet is the unit of resampling for its interval
(Efron & Tibshirani 1993).  Pixels are drawn in proportion to their ink, since ink amounts are what the scores count.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from sfzmaps.frame.layout import combine

log = logging.getLogger(__name__)

N_CAL = 200_000    # computational: px per sheet; its sampling error is far below the spread between sheets
CAL_BOOT = 200     # computational: sheet resamples for the temperature interval
LOG_T_BOUNDS = (-3.0, 3.0)  # numerical search bounds on log T
EPS = 1e-6         # numerical clip for log-loss


def sample(zs: Sequence[np.ndarray], cls: np.ndarray, lum, rng, n: int = N_CAL) -> tuple[np.ndarray, np.ndarray]:
    """(z (seeds, 3, n) float32, y (n,) uint8): n px drawn with replacement, probability proportional to 1 - lum."""
    w = (1.0 - np.asarray(lum, np.float64)).ravel()
    total = w.sum()
    idx = rng.choice(w.size, size=n, replace=True, p=w / total if total > 0 else None)
    z = np.stack([np.asarray(a).reshape(a.shape[0], -1)[:, idx] for a in zs]).astype(np.float32)
    return z, np.asarray(cls).ravel()[idx].astype(np.uint8)


def _loss(samples, temperature: float) -> float:
    total = count = 0
    for z, y in samples:
        p = np.clip(combine(list(z), temperature).astype(np.float64), EPS, 1.0)
        total += float(-np.log(np.take_along_axis(p, y[None].astype(np.int64), axis=0)).sum())
        count += y.size
    return total / count


def fit_temperature(samples: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """The T minimising the pooled mean log-loss of combine(list(z), T) against y."""
    lo, hi = LOG_T_BOUNDS
    res = minimize_scalar(lambda t: _loss(samples, float(np.exp(t))), bounds=(lo, hi), method="bounded")
    if min(res.x - lo, hi - res.x) < 1e-3:
        log.warning("temperature fit hit a search bound (log T = %.3f)", res.x)
    return float(np.exp(res.x))


def temperature_interval(samples, n: int = CAL_BOOT, seed: int = 0) -> tuple[float, float]:
    """5th/95th percentiles of the temperature refitted on sheets resampled with replacement."""
    rng = np.random.default_rng(seed)
    ts = [fit_temperature([samples[i] for i in rng.integers(0, len(samples), len(samples))]) for _ in range(n)]
    lo, hi = np.percentile(ts, (5, 95))
    return float(lo), float(hi)


def reliability(samples, temperature: float, bins: int = 10) -> list[dict]:
    """One row per equal-width bin of the chance of deletion (outside + removed): lo, hi, mean_q, share_outside, n."""
    q = np.concatenate([(lambda p: p[1] + p[2])(combine(list(z), temperature)) for z, _ in samples])
    gone = np.concatenate([y != 0 for _, y in samples])
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(q, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = which == b
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "mean_q": float(q[m].mean()) if m.any() else float("nan"),
                     "share_outside": float(gone[m].mean()) if m.any() else float("nan"), "n": int(m.sum())})
    return rows
