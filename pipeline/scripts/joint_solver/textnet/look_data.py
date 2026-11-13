"""Picture (for my own check): windows of the merged page with the training text regions tinted by kind (code red, block
number blue, street green, other orange) and unknown areas grey; one window per reading angle class.
usage: look_data.py OUT.png"""
import sys

import cv2
import numpy as np

D = np.load("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/textnet/data.npz")
ink, text, kind, angle, unknown = D["ink"], D["text"], D["kind"], D["angle"], D["unknown"]
COL = {1: (40, 40, 230), 2: (230, 120, 30), 3: (40, 170, 40), 4: (0, 150, 255)}
rng = np.random.default_rng(3)
panels = []
for lo, hi, n in ((-5, 5, 2), (80, 100, 2), (-100, -80, 1), (10, 70, 2), (-70, -10, 1)):
    ys, xs = np.nonzero((angle >= lo) & (angle <= hi) & (kind == 1)) if lo < -50 or lo > 50 else np.nonzero((angle >= lo) & (angle <= hi))
    if len(ys) == 0:
        continue
    for i in rng.choice(len(ys), n, replace=False):
        y, x = ys[i], xs[i]
        y0, x0 = max(y - 100, 0), max(x - 100, 0)
        sl = (slice(y0, y0 + 200), slice(x0, x0 + 200))
        im = np.where(ink[sl], 0, 255).astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR).astype(float)
        for k, c in COL.items():
            m = kind[sl] == k
            im[m] = 0.55 * im[m] + 0.45 * np.array(c)
        u = unknown[sl]
        im[u] = 0.7 * im[u] + 0.3 * np.array([128, 128, 128])
        im = cv2.resize(im.astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.putText(im, f"angle {angle[y, x]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        panels.append(im)
while len(panels) % 4:
    panels.append(np.full_like(panels[0], 255))
img = np.vstack([np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)])
cv2.imwrite(sys.argv[1], img)
print("wrote", sys.argv[1], img.shape)
