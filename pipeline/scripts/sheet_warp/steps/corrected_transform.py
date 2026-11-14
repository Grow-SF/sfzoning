"""Today's fit with a correction map laid on top, nothing re-fitted.
to_pixel(w) = p + u(p), p = today's to_pixel(w);  to_world(q) = today's to_world(p) with p solving p + u(p) = q
(fixed-point; u is small and smooth).  u: confit's field on a page-px grid (bilinear)."""
import numpy as np
from scipy.ndimage import map_coordinates


class Composed:
    def __init__(self, base, G, spacing, origin=(0.0, 0.0)):
        """G: 2 x gh x gw field, node (i, j) at page px origin + (j, i) * spacing (the crop grid the fit used)."""
        self.base, self.G, self.sp = base, G, float(spacing)
        self.o = np.asarray(origin, float)
        self.gcps = base.gcps

    def u(self, p):
        p = np.asarray(p, float) - self.o
        c = [p[:, 1] / self.sp, p[:, 0] / self.sp]
        return np.stack([map_coordinates(self.G[k], c, order=1, mode="constant", cval=0.0) for k in (0, 1)], -1)

    def to_pixel(self, w):
        p = self.base.to_pixel(np.asarray(w, float))
        return p + self.u(p)

    def to_world(self, q):
        q = np.asarray(q, float)
        p = q.copy()
        for _ in range(20):
            p = q - self.u(p)
        return self.base.to_world(p)
