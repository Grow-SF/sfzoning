"""Held sides under the plain measure: a side is held if every point is within 1 px of its printed line, leaving out
points where the printed block line is hidden under heavy ink (a heavy line crossing: ink there is more than 2 px
from paper within 3 px of the point) and points within 2 px of the side's ends (the corner, where the line
running its way stops).  -> dm_held_NN.npz (held per side; per point: counted or left out).  usage: heldrule.py NN"""
import sys

import cv2
import numpy as np
from scipy import ndimage

from sfzmaps import config

from ay import *  # noqa: F403
Z = np.load(S + f"dm_{NN}_today.npz")
d, sid, p = Z["d"], Z["sid"], Z["px"]
page = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE)
ink = page < 128
heavy = ndimage.maximum_filter((ndimage.distance_transform_edt(ink) > half()).astype(np.uint8), size=7) > 0   # heavy ink within 3 px
H, W = ink.shape
ok = np.isfinite(p).all(1)
xi = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
on_heavy = ok & heavy[yi, xi]
# position along the side (points are every 2 ft = ~0.5 px apart, in order)
n = sid.max() + 1
first = np.full(n, len(sid)); last = np.zeros(n, int)
idx = np.arange(len(sid))
np.minimum.at(first, sid, idx); np.maximum.at(last, sid, idx)
ft_per_px = ftpx()
near_end = ((idx - first[sid]) * 2.0 / ft_per_px <= 2.0) | ((last[sid] - idx) * 2.0 / ft_per_px <= 2.0)
counted = ok & ~on_heavy & ~near_end
bad = np.zeros(n, bool)
np.logical_or.at(bad, sid, counted & (d > 1.0))
has = np.zeros(n, bool)
np.logical_or.at(has, sid, counted)
held = ~bad                      # a side with no measurable point lies under heavy ink: on the ink, held
L = np.bincount(sid, minlength=n) * 2.0
print(f"p{NN}: sides {n}; points left out: on heavy ink {on_heavy.mean():.1%}, near a side's end {near_end.mean():.1%}; "
      f"held sides {held.sum()} ({100 * L[held].sum() / L.sum():.1f}% of the length); free {(~held).sum()}")
np.savez(S + f"dm_held_{NN}.npz", held=held, counted=counted)
