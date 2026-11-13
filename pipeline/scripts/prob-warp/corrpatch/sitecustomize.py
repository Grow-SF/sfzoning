"""Loaded only for the corrected-fit runs (PYTHONPATH=corrpatch, SFZ_CORR=1).  A sheet's transform.json may carry
"correction": {"file": <confit npz>}: today's fit with that correction map laid on top, nothing re-fitted.
SheetTransform.to_world(q) then returns today's to_world(p), p solving p + u(p) = q (u: the map, bilinear between
its grid points); to_pixel inverts to_world as before, so it returns today's to_pixel(w) + u."""
import os

if int(os.environ.get("RBF_THREADS", "1")) > 1:      # speed E7: the thin-plate spline on threads, bit-identical
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "allyears"))
    import rbfthreads
    rbfthreads.install()

if os.environ.get("SFZ_CORR"):
    import numpy as np
    from scipy.ndimage import map_coordinates

    import sfzmaps.transform as _tr

    _from = _tr.SheetTransform.from_json.__func__
    _tw = _tr.SheetTransform.to_world

    def _from_json(cls, d):
        t = _from(cls, d)
        c = d.get("correction")
        if c:
            Z = np.load(c["file"])
            t._corr = (Z["Gc"], float(Z["spacing"]), np.asarray(Z["origin"], float))
            t._base = _from(cls, d)          # today's fit untouched (no _corr): its to_pixel inverts today's to_world
        return t

    def _u(corr, p):
        G, sp, o = corr
        q = p - o
        c = [q[:, 1] / sp, q[:, 0] / sp]
        return np.stack([map_coordinates(G[k], c, order=1, mode="constant", cval=0.0) for k in (0, 1)], -1)

    _tp = _tr.SheetTransform.to_pixel

    def _to_pixel(self, w):
        """map -> page: today's to_pixel plus the correction there (the correction's own direction: no inverting)."""
        corr = getattr(self, "_corr", None)
        if corr is None:
            return _tp(self, w)
        p = _tp(self._base, w)
        ok = np.isfinite(p).all(1)
        out = p.copy()
        out[ok] = p[ok] + _u(corr, p[ok])
        return out

    def _to_world(self, px):
        """page -> map: today's to_world(p), p solving p + u(p) = q.  Damped fixed-point steps; keeps the best p
        found (where the correction changes sharply between grid points this is the closest it gets)."""
        corr = getattr(self, "_corr", None)
        if corr is None:
            return _tw(self, px)
        q = np.atleast_2d(np.asarray(px, float))
        p = q - _u(corr, q)
        best, err = p.copy(), np.linalg.norm(p + _u(corr, p) - q, axis=1)
        for _ in range(60):
            p = p + 0.5 * (q - _u(corr, p) - p)
            e = np.linalg.norm(p + _u(corr, p) - q, axis=1)
            better = e < err
            best[better], err[better] = p[better], e[better]
            if err.max() < 1e-4:
                break
        return _tw(self, best)

    _tr.SheetTransform.from_json = classmethod(_from_json)
    _tr.SheetTransform.to_world = _to_world
    _tr.SheetTransform.to_pixel = _to_pixel
