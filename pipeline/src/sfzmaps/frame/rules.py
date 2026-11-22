"""A printed rule is one thing (design spec, amendment 5; ported from the trial `scripts/spikes/issue71_rules.py`,
mode "one").

A *rule* is a connected stroke of ink that runs straight for a long way along the page (within the frame search's
angle range) and is thin across: frame rules and legend bars are rules; hatching, lettering, corner arcs and solid
blocks are not.  Every px of a rule shares one set of chances, the ink-weighted mean of the model's chances along it,
so a stroke the model read two ways (on 2000-digital-su p11, half of the top frame rule as map) is decided whole by
the straight-line cut.  The mean keeps each rule's expected ink in each class as the model gave it, so the cut's cost
accounting is unchanged; a stroke read the same way along its length is unchanged too.

None of the settings below is fitted; they are the trial's.  Measured on the 22 test sheets (amendment 5): halving
RUN (101 px) gives the same pooled result (frame ink left in 0.10%, map ink cut 0.328%); doubling it (401 px) loses
1929 p14's gain (0.18% / 0.322%) and keeps the three fixes (2000-digital-su p11, 1964 p14, 1980-1990-su p01).
"""

from __future__ import annotations

import cv2
import numpy as np

RUN = 201  # px: ink must run at least this far along a rule (trial setting; 101 measured the same, 401 lost a gain)
ACROSS = 61  # px: ... and be thinner than this across it (trial setting; the thickest labelled bars are ~30 px)
ANGLE_STEP = 0.5  # degrees: the directions tried, every this much over +-shape.ANGLE_MAX (trial setting)
INK = 0.5  # ink (1 - lum) above this is a stroke (trial setting)


def _angles() -> np.ndarray:
    from sfzmaps.frame.shape import ANGLE_MAX  # here, not at the top: shape imports this module

    return np.arange(-ANGLE_MAX, ANGLE_MAX + 1e-9, ANGLE_STEP)


def rule_labels(lum: np.ndarray) -> tuple[np.ndarray, int]:
    """(labels, n): int32, every px of a rule carries its rule's number (1..n), other px 0.

    Horizontal-ish and vertical-ish rules are found separately: the stroke mask is turned by each angle, opened with
    a 1 x RUN (or RUN x 1) line, and turned back, and the results are united.  A horizontal stroke then loses the px
    that also lie on a vertical run of ACROSS or more (a solid block, or a crossing vertical rule), and the other way
    round; the rest falls into 8-connected pieces.  A px in both families keeps the horizontal rule's number."""
    ink = ((1.0 - np.asarray(lum, np.float32)) > INK).astype(np.uint8)
    h, w = ink.shape
    c = ((w - 1) / 2, (h - 1) / 2)
    H, V = np.zeros_like(ink), np.zeros_like(ink)
    along, down = np.ones((1, RUN), np.uint8), np.ones((RUN, 1), np.uint8)
    for a in _angles():
        M = cv2.getRotationMatrix2D(c, float(a), 1.0)
        R = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_NEAREST)
        Mi = cv2.invertAffineTransform(M)
        H |= cv2.warpAffine(cv2.morphologyEx(R, cv2.MORPH_OPEN, along), Mi, (w, h), flags=cv2.INTER_NEAREST)
        V |= cv2.warpAffine(cv2.morphologyEx(R, cv2.MORPH_OPEN, down), Mi, (w, h), flags=cv2.INTER_NEAREST)
    tall = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((ACROSS, 1), np.uint8))
    wide = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, ACROSS), np.uint8))
    nh, lh = cv2.connectedComponents(H & ink & (1 - tall), connectivity=8)
    nv, lv = cv2.connectedComponents(V & ink & (1 - wide), connectivity=8)
    lab = lh.astype(np.int32)
    only_v = (lv > 0) & (lab == 0)
    lab[only_v] = lv[only_v] + (nh - 1)
    return lab, nh - 1 + nv - 1


def share(p: np.ndarray, lum: np.ndarray, labels: tuple[np.ndarray, int] | np.ndarray | None = None) -> np.ndarray:
    """p (3, h, w) with every rule's px set to the rule's ink-weighted mean chances (ink = 1 - lum), float32; other px
    untouched.  `labels` is `rule_labels(lum)` (the pair, or the label array alone) when given.  Does not modify p."""
    if labels is None:
        labels = rule_labels(lum)
    lab = labels[0] if isinstance(labels, tuple) else labels
    n = int(lab.max())
    q = np.array(p, np.float32, copy=True)
    if n == 0:
        return q
    ink = (1.0 - np.asarray(lum, np.float32)).astype(np.float64).ravel()
    flat = lab.ravel()
    tot = np.bincount(flat, weights=ink, minlength=n + 1)
    means = np.stack([np.bincount(flat, weights=ink * q[k].ravel(), minlength=n + 1) / np.maximum(tot, 1e-9)
                      for k in range(3)])
    m = lab > 0
    for k in range(3):
        q[k][m] = means[k][lab[m]]
    return q
