"""The learned chances (design spec, amendment 6; ported from the trial scripts `scripts/spikes/issue71_dev.py` and
`scripts/spikes/issue71_learned.py`, behaviour unchanged).

The model under-reads whole printed rules (on training sheets read by heads that did not train on them, a rule read
as 30% frame or box is frame or box almost nine times in ten) and reads other ink about right.  So the chance of
deletion the cut runs on is learned from those sheets, not taken from the model as it is:
- **rules** (`rules.rule_labels`): the chance that a whole rule is frame or box, given the model's reading along it
  (the ink-weighted mean of p1 + p2) and its length (the longer side of its bounding box, px): a logistic fit on
  1, logit(reading) and log(length), each rule weighted by its ink, its target the share of its ink the label calls
  frame or box;
- **other ink**: the chance that ink on no rule is frame or box, given the model's reading p1 + p2: a logistic fit on
  1 and logit(reading), from the ink summed in `INK_BINS` equal bins of the reading, each bin at its ink-weighted
  mean reading, weighted by its ink, its target the share of its ink the label calls frame or box.
`fit` makes both from the rows of every training sheet (`rule_rows`, `ink_rows`); `apply` puts them on a page's
chances.  The split between "outside" and "label box" within the chance of deletion stays as the model gave it.
Nothing here is set by hand: the two fits come from the labels and are refitted whenever the labels change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

from sfzmaps.frame import rules

EPS = 1e-4  # numerical: readings are clipped to [EPS, 1 - EPS] before the logit
INK_BINS = 100  # computational: ink on no rule is summed in this many equal bins of the reading


@dataclass(frozen=True)
class Chances:
    """The two fitted curves: `rule` (intercept, coefficient of logit(reading), coefficient of log(length px)) and
    `ink` (intercept, coefficient of logit(reading))."""

    rule: tuple[float, float, float]
    ink: tuple[float, float]

    def to_json(self) -> dict:
        return {"rule": [float(v) for v in self.rule], "ink": [float(v) for v in self.ink]}

    @classmethod
    def from_json(cls, d: dict) -> Chances:
        return cls(rule=tuple(float(v) for v in d["rule"]), ink=tuple(float(v) for v in d["ink"]))


def _logit(x):
    x = np.clip(np.asarray(x, np.float64), EPS, 1 - EPS)
    return np.log(x / (1 - x))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _labels(lum: np.ndarray, labels) -> tuple[np.ndarray, int]:
    if labels is None:
        labels = rules.rule_labels(lum)
    lab = labels[0] if isinstance(labels, tuple) else labels
    return lab, int(lab.max())


def _per_rule(p: np.ndarray, lum: np.ndarray, lab: np.ndarray, n: int):
    """Per rule label 0..n: its ink, its reading (ink-weighted mean of p1 + p2) and its length (the longer side of its
    bounding box, px; 1 where it has none)."""
    ink = (1.0 - np.asarray(lum, np.float32)).astype(np.float64).ravel()
    flat = lab.ravel()
    q = (np.asarray(p[1], np.float64) + np.asarray(p[2], np.float64)).ravel()
    tot = np.bincount(flat, weights=ink, minlength=n + 1)
    reading = np.bincount(flat, weights=ink * q, minlength=n + 1) / np.maximum(tot, 1e-9)
    length = np.ones(n + 1)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is not None:
            length[i] = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
    return tot, reading, length


def rule_rows(p: np.ndarray, lum: np.ndarray, cls: np.ndarray, labels=None) -> np.ndarray:
    """(n, 4) float64, one row per rule with ink: its ink, its reading, the share of its ink the label `cls` calls
    frame or box (cls != 0), its length (px).  `labels` is `rules.rule_labels(lum)` (the pair or the array) when
    given."""
    lab, n = _labels(lum, labels)
    if n == 0:
        return np.zeros((0, 4))
    tot, reading, length = _per_rule(p, lum, lab, n)
    ink = (1.0 - np.asarray(lum, np.float32)).astype(np.float64).ravel()
    really = np.bincount(lab.ravel(), weights=ink * (np.asarray(cls).ravel() != 0), minlength=n + 1)
    i = np.flatnonzero(tot[1:] > 0) + 1
    return np.stack([tot[i], reading[i], really[i] / tot[i], length[i]], 1)


def ink_rows(p: np.ndarray, lum: np.ndarray, cls: np.ndarray, labels=None) -> np.ndarray:
    """(INK_BINS, 3) float64: for ink on no rule, per equal bin of the reading p1 + p2, its ink, the ink-weighted sum
    of the reading, and the ink the label `cls` calls frame or box."""
    lab, _ = _labels(lum, labels)
    off = lab.ravel() == 0
    ink = (1.0 - np.asarray(lum, np.float32)).astype(np.float64).ravel()[off]
    q = (np.asarray(p[1], np.float64) + np.asarray(p[2], np.float64)).ravel()[off]
    b = np.clip((q * INK_BINS).astype(np.int64), 0, INK_BINS - 1)
    really = np.asarray(cls).ravel()[off] != 0
    return np.stack([np.bincount(b, weights=ink, minlength=INK_BINS),
                     np.bincount(b, weights=ink * q, minlength=INK_BINS),
                     np.bincount(b, weights=ink * really, minlength=INK_BINS)], 1)


def _fit_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted logistic regression with soft targets y in [0, 1] (the trial's): the coefficients, from zeros by
    BFGS."""
    X, y = np.asarray(X, np.float64), np.asarray(y, np.float64)
    w = np.asarray(w, np.float64) / np.sum(w)

    def nll(b):
        z = X @ b
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)))

    return minimize(nll, np.zeros(X.shape[1]), method="BFGS").x


def fit(rule_rows: np.ndarray, ink_rows: np.ndarray) -> Chances:
    """The two curves (see the module) from every training sheet's `rule_rows` stacked and `ink_rows` summed (or
    stacked: bins are pooled by their ink-weighted mean reading either way).  Refuses (ValueError) no rules or no
    ink."""
    r = np.asarray(rule_rows, np.float64).reshape(-1, 4)
    r = r[r[:, 0] > 0]
    b = np.asarray(ink_rows, np.float64).reshape(-1, 3)
    b = b[b[:, 0] > 0]
    if not len(r) or not len(b):
        raise ValueError(f"the learned chances need rules and other ink to fit on ({len(r)} rules, {len(b)} bins)")
    br = _fit_logistic(np.stack([np.ones(len(r)), _logit(r[:, 1]), np.log(r[:, 3])], 1), r[:, 2], r[:, 0])
    bo = _fit_logistic(np.stack([np.ones(len(b)), _logit(b[:, 1] / b[:, 0])], 1), b[:, 2] / b[:, 0], b[:, 0])
    return Chances(rule=tuple(float(v) for v in br), ink=tuple(float(v) for v in bo))


def apply(p: np.ndarray, lum: np.ndarray, chances: Chances, labels=None) -> np.ndarray:
    """The page's chances `p` (3, h, w) after the two learned curves, float32 (3, h, w): the chance of deletion c is
    the ink curve's on every px on no rule and its rule's on every px of a rule (from the rule's reading and length);
    then p0 = 1 - c, and c is split between classes 1 and 2 as the model split p1 + p2 (class 1 where p1 + p2 is ~0).
    `labels` is `rules.rule_labels(lum)` when given.  Does not modify `p`."""
    p = np.asarray(p)
    q = np.asarray(p[1], np.float64) + np.asarray(p[2], np.float64)
    i0, i1 = chances.ink
    c = _sigmoid(i0 + i1 * _logit(q))
    lab, n = _labels(lum, labels)
    if n:
        _, reading, length = _per_rule(p, lum, lab, n)
        a, b, g = chances.rule
        cr = _sigmoid(a + b * _logit(reading) + g * np.log(length))
        on = lab > 0
        c[on] = cr[lab[on]]
    share2 = np.where(q > 1e-6, np.asarray(p[2], np.float64) / np.maximum(q, 1e-6), 0.0)
    return np.stack([1.0 - c, c * (1.0 - share2), c * share2]).astype(np.float32)
