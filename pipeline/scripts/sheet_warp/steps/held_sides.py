"""The sides the print already holds: a block side is held if every one of its points is within 1 px of its printed
line under today's placement (measure_today.npz), leaving out points where the printed line is hidden under heavy ink
(ink wider than a block line within 3 px) and points within 2 px of the side's ends (the corner, where the line
running its way stops).  The first correction keeps held sides in place.  -> held_sides.npz (held: per side)."""
import cv2
import numpy as np
from scipy import ndimage

from warp_common import D, S, SHEET, ftpx, half

Z = np.load(S + "measure_today.npz")
d, sid, p = Z["d"], Z["sid"], Z["px"]
ink = cv2.imread(str(D / "page.png"), cv2.IMREAD_GRAYSCALE) < 128
heavy = ndimage.maximum_filter((ndimage.distance_transform_edt(ink) > half(ink)).astype(np.uint8), size=7) > 0
H, W = ink.shape
ok = np.isfinite(p).all(1)
xi = np.clip(np.round(p[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.round(p[:, 1]).astype(int), 0, H - 1)
on_heavy = ok & heavy[yi, xi]
n = sid.max() + 1                     # position along the side: the points are every 2 ft, in order
first = np.full(n, len(sid)); last = np.zeros(n, int)
idx = np.arange(len(sid))
np.minimum.at(first, sid, idx); np.maximum.at(last, sid, idx)
ft_per_px = ftpx()
near_end = ((idx - first[sid]) * 2.0 / ft_per_px <= 2.0) | ((last[sid] - idx) * 2.0 / ft_per_px <= 2.0)
counted = ok & ~on_heavy & ~near_end
bad = np.zeros(n, bool)
np.logical_or.at(bad, sid, counted & (d > 1.0))
held = ~bad                           # a side with no measurable point lies under heavy ink: on the ink, held
L = np.bincount(sid, minlength=n) * 2.0
print(f"{SHEET}: sides {n}; points left out: on heavy ink {on_heavy.mean():.1%}, near a side's end {near_end.mean():.1%}; "
      f"held sides {held.sum()} ({100 * L[held].sum() / L.sum():.1f}% of the length); free {(~held).sum()}")
np.savez(S + "held_sides.npz", held=held)
