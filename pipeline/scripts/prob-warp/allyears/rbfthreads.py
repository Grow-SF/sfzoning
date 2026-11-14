"""Threaded evaluation of scipy's RBFInterpolator (speed E7), bit-identical to the serial call: the same chunks scipy's
own _chunk_evaluator would make for the whole call (its chunk size depends on the call's size), each computed by
scipy's own compute_interpolation, several at once on threads.  Only the no-neighbors case (ours) is changed.
install() patches RBFInterpolator.__call__; env RBF_THREADS (default 1 = off)."""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import _rbfinterp as R

_orig = R.RBFInterpolator.__call__
_POOL = None


def _call(self, x):
    if self.neighbors is not None:
        return _orig(self, x)
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if x.ndim != 2 or x.shape[1] != self.y.shape[1]:
        return _orig(self, x)
    nx = x.shape[0]
    memory_budget = max(x.size + self.y.size + self.d.size, 1_000_000)        # as __call__
    chunksize = memory_budget // (self.powers.shape[0] + self.y.shape[0]) + 1  # as _chunk_evaluator
    if chunksize > nx:
        return _orig(self, x)
    backend = R._get_backend(self._xp)
    starts = list(range(0, nx, chunksize))

    def one(i):
        return backend.compute_interpolation(x[i:i + chunksize, :], self.y, self.kernel, self.epsilon, self.powers,
                                             self._shift, self._scale, self._coeffs, self._xp)
    out = np.empty((nx, self.d.shape[1]), dtype=np.float64)
    for i, chunk in zip(starts, _POOL.map(one, starts)):
        out[i:i + chunksize] = chunk
    out = out.view(self.d_dtype)                                              # as __call__
    return np.reshape(out, (nx,) + self.d_shape)


def install():
    global _POOL
    n = int(os.environ.get("RBF_THREADS", "1"))
    if n > 1 and _POOL is None:
        _POOL = ThreadPoolExecutor(n)
        R.RBFInterpolator.__call__ = _call
