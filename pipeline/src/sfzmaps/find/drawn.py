"""Is a block drawn on this sheet?  A sheet prints only its own part of the city; the rest of its frame is blank paper
over which the lots would still put labels.  For each block of the sheet: n, its edge cuts on the page, and k, how many
of them cross ink (`INK_BELOW`).  Over the sheet's blocks k is a two-part binomial mixture,
P(k | n) = pi Bin(k; n, q_drawn) + (1 - pi) Bin(k; n, q_blank), q_drawn the larger rate, fitted by EM; a block's chance
of being drawn is its posterior.

A sheet with only one kind of block has one group, not two: the two-part fit is kept only when its gain in
log-likelihood beats the cost of its two extra numbers (BIC, log of the number of blocks); with one group every block is
drawn.  A block with no edge cut on the page has nothing of it on the page: chance 0, and it does not enter the fit.

Reads and writes nothing."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp, xlog1py, xlogy

INK_BELOW = 0.5                 # a px is ink where ink is the more likely reading: luminance under half (exact on the
                                # 1-bit scans the lots label)
START = (0.5, 0.75, 0.25)       # engineering: the EM's starting (pi, q_drawn, q_blank); the fit does not depend on it
STEP_TOL = 1e-13                # engineering: the EM stops when no number moves more than this in a step
MAX_STEPS = 100_000             # engineering: or after this many steps


def _log_bin(k: np.ndarray, n: np.ndarray, q: float) -> np.ndarray:
    """log Bin(k; n, q), 0 log 0 taken as 0."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1) + xlogy(k, q) + xlog1py(n - k, -q)


def _parts(k, n, pi, qd, qb) -> np.ndarray:
    """(m, 2): log pi Bin(k; n, q_drawn) and log (1 - pi) Bin(k; n, q_blank)."""
    with np.errstate(divide="ignore"):
        return np.c_[np.log(pi) + _log_bin(k, n, qd), np.log1p(-pi) + _log_bin(k, n, qb)]


def _posterior(parts: np.ndarray, pi: float) -> np.ndarray:
    """P(drawn) of each block; a block neither part can make (both log chances -inf) keeps the prior pi."""
    total = logsumexp(parts, axis=1)
    with np.errstate(invalid="ignore"):
        p = np.exp(parts[:, 0] - total)
    return np.where(np.isfinite(total), p, pi)


def _em(k: np.ndarray, n: np.ndarray, start) -> tuple[float, float, float]:
    pi, qd, qb = (float(v) for v in start)
    for _ in range(MAX_STEPS):
        r = _posterior(_parts(k, n, pi, qd, qb), pi)
        new = (float(r.mean()), float((r * k).sum() / max((r * n).sum(), 1e-300)),       # engineering: the floor
               float(((1 - r) * k).sum() / max(((1 - r) * n).sum(), 1e-300)))        # only keeps 0 / 0 at 0
        moved = max(abs(a - b) for a, b in zip(new, (pi, qd, qb)))
        pi, qd, qb = new
        if moved <= STEP_TOL:
            break
    if qd < qb:
        pi, qd, qb = 1 - pi, qb, qd
    return pi, qd, qb


def drawn_chance(k: np.ndarray, n: np.ndarray, start=START) -> tuple[np.ndarray, dict]:
    """Each block's chance of being drawn (float64), from k of its n edge cuts crossing ink, and the fit: `pi`,
    `q_drawn`, `q_blank` (nan where there is no such group) and `groups` (0 no block on the page, 1 or 2)."""
    k, n = np.asarray(k, np.int64), np.asarray(n, np.int64)
    p = np.zeros(len(n))
    on = n > 0
    if not on.any():
        return p, {"pi": float("nan"), "q_drawn": float("nan"), "q_blank": float("nan"), "groups": 0}
    k1, n1 = k[on], n[on]
    q1 = float(k1.sum() / n1.sum())
    one = _log_bin(k1, n1, q1).sum()
    pi, qd, qb = _em(k1, n1, start)
    parts = _parts(k1, n1, pi, qd, qb)
    two = logsumexp(parts, axis=1).sum()
    if two - one > np.log(on.sum()):          # BIC: two extra numbers, each costing half the log of the block count
        p[on] = _posterior(parts, pi)
        return p, {"pi": pi, "q_drawn": qd, "q_blank": qb, "groups": 2}
    p[on] = 1.0
    return p, {"pi": 1.0, "q_drawn": q1, "q_blank": float("nan"), "groups": 1}
