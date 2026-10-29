"""Smooth, regularized pixel -> EPSG:2227 mapping fit from control points.

Model: world = affine(px) + thin-plate-spline(px), the TPS fit by least squares with
smoothing to the residuals left after the affine. The affine carries scale, rotation and
shear; the TPS corrects paper and drafting distortion smoothly over areas. Smoothing is
never zero, so the warp never interpolates every control point and a single bad point
cannot drag it (and individual historic edges are not snapped onto modern ones).

Units of `smoothing`: pixel inputs are centered and divided by their max half-extent, so
they lie in about [-1, 1]; the TPS targets are the affine residuals in world feet.
scipy adds `smoothing` to the diagonal of the TPS kernel matrix (kernel r^2 log r, r in
those normalized units), so it is a ridge penalty on the residual field in ft^2 per
unit-normalized-kernel. Measured on a 10x10 GCP grid over a 7000 px sheet (distortion
about +-100 ft), one GCP moved by 40 ft shifts the fit at that pixel by:
    smoothing 0.1 -> 16.2 ft   1 -> 5.5 ft   5 -> 2.6 ft   50 -> 0.85 ft   (0.01 -> 32 ft)
and the max error against the true distortion field (60 ft peak affine misfit) is
    0.1 -> 3.3 ft   1 -> 15 ft   5 -> 31 ft   50 -> 53 ft (essentially affine only).
That sparse grid was the Task 7 basis for a default of 1.0. Task 13 retuned it on real
sheets, whose final warp has 600-2500 corner GCPs ~100-300 ft apart: there 1.0 was too
stiff (block-held-out cross-validation improves down to 0.01-0.1) and config.TPS_SMOOTHING
is now 0.1, where a lone 40 ft GCP moves the fit about 3-4 ft and a whole block moved
15 ft about 5 ft (measured on 1948/1, 1960/1; see the tuning notes). Effect scales with GCP
density (denser grids resist outliers more).

Fewer than MIN_GCPS enabled GCPs is an error. With MIN_GCPS..TPS_MIN_GCPS-1 (3-9) the
fit is affine only: a TPS on so few points is unstable.

Conditioning: EPSG:2227 coordinates are ~6e6/2e6 ft. Both pixel and world coordinates
are centered internally (world by its mean, pixels also scaled) before any fit.

Smoothing is ignored with fewer than 10 GCPs (affine only). Its effect depends on GCP
count and density, so a value tuned on one sheet does not transfer to another.

Fit state: fit() returns self. to_world/to_pixel fit lazily and keep a signature of
(enabled GCP tuples, smoothing) taken at fit time; if GCPs or smoothing change afterwards
(including from_json, which does not fit) the next evaluation refits automatically.
Collinear or degenerate GCPs raise ValueError.

to_pixel: rather than fit an independent reverse TPS (which cannot guarantee a round
trip), invert the forward model numerically: start from the inverse of the affine part and
run fixed-point iterations p += A^-1 (w - to_world(p)). For gentle warps this converges to
far below 0.5 px in a few steps and is exactly consistent with to_world. Points whose
residual stays above 0.1 px afterwards get damped Newton steps (numerical Jacobian); any
still unconverged are returned as NaN with one warnings.warn naming the count (never
raises, never returns a wrong value).
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import RBFInterpolator

MIN_GCPS = 3
TPS_MIN_GCPS = 10
_INV_ITERS = 25
_INV_TOL_PX = 1e-6
_CONVERGED_PX = 0.1
_MIN_COND = 1e-6


@dataclass
class GCP:
    px: float
    py: float
    wx: float
    wy: float
    source: Literal["anchor", "auto", "manual", "answer_key"]
    enabled: bool = True


@dataclass
class SheetTransform:
    gcps: list[GCP]
    smoothing: float  # TPS ridge in ft^2 (see module docstring); 0 is forbidden

    def __post_init__(self):
        self._fitted = False
        self._rbf = None
        self._sig = None

    def _signature(self):
        return (tuple((g.px, g.py, g.wx, g.wy) for g in self._enabled()), self.smoothing)

    def _ensure_fit(self):
        if not self._fitted or self._sig != self._signature():
            self.fit()

    # -- fitting -----------------------------------------------------------
    def _enabled(self) -> list[GCP]:
        return [g for g in self.gcps if g.enabled]

    def fit(self) -> "SheetTransform":
        if not self.smoothing > 0:
            raise ValueError(f"smoothing must be > 0 (got {self.smoothing}); "
                             "0 would interpolate every control point exactly")
        gs = self._enabled()
        if len(gs) < MIN_GCPS:
            raise ValueError(f"need at least {MIN_GCPS} enabled GCPs, got {len(gs)}")
        P = np.array([[g.px, g.py] for g in gs], float)
        W = np.array([[g.wx, g.wy] for g in gs], float)
        self._pm = P.mean(0)
        self._ps = float(np.abs(P - self._pm).max()) or 1.0
        self._wm = W.mean(0)
        Pn = (P - self._pm) / self._ps
        sv = np.linalg.svd(Pn, compute_uv=False)
        if len(sv) < 2 or sv[0] == 0 or sv[1] / sv[0] < _MIN_COND:
            raise ValueError("GCPs are collinear/degenerate")
        A = np.c_[Pn, np.ones(len(Pn))]
        coef, *_ = np.linalg.lstsq(A, W - self._wm, rcond=None)
        self._coef = coef  # (3,2): ft per normalized unit
        resid = W - self._wm - A @ coef
        self._rbf = None
        if len(gs) >= TPS_MIN_GCPS:
            self._rbf = RBFInterpolator(Pn, resid, kernel="thin_plate_spline",
                                        smoothing=self.smoothing)
        # d(world)/d(px) of the affine part, for inversion
        self._lin = coef[:2].T / self._ps
        self._lin_inv = np.linalg.inv(self._lin)
        self._sig = self._signature()
        self._fitted = True
        return self

    # -- evaluation --------------------------------------------------------
    def to_world(self, px: np.ndarray) -> np.ndarray:
        self._ensure_fit()
        p = np.atleast_2d(np.asarray(px, float))
        pn = (p - self._pm) / self._ps
        out = np.c_[pn, np.ones(len(pn))] @ self._coef
        if self._rbf is not None:
            out = out + self._rbf(pn)
        return out + self._wm

    def to_pixel(self, w: np.ndarray) -> np.ndarray:
        self._ensure_fit()
        w = np.atleast_2d(np.asarray(w, float))
        p = self._pm + (w - self.to_world(self._pm[None])) @ self._lin_inv.T
        for _ in range(_INV_ITERS):
            step = (w - self.to_world(p)) @ self._lin_inv.T
            p = p + step
            if np.abs(step).max() < _INV_TOL_PX:
                break

        def err(q, ww):
            return np.linalg.norm((ww - self.to_world(q)) @ self._lin_inv.T, axis=1)

        e = err(p, w)
        bad = ~(e <= _CONVERGED_PX)
        if bad.any():
            p = p.copy()
            p[bad] = self._newton(p[bad], w[bad])
            e = err(np.nan_to_num(p), w)
            bad = ~(e <= _CONVERGED_PX) | ~np.isfinite(p).all(axis=1)
            if bad.any():
                p[bad] = np.nan
                warnings.warn(f"to_pixel: {int(bad.sum())} point(s) did not converge; "
                              "returned NaN", RuntimeWarning, stacklevel=2)
        return p

    def _newton(self, p, w, iters=60):
        """Damped Newton with numerical Jacobian, per point (vectorized over points)."""
        p = p.copy()
        h = 0.5  # px
        for _ in range(iters):
            f0 = self.to_world(p)
            r = w - f0
            ex = (self.to_world(p + [h, 0]) - f0) / h
            ey = (self.to_world(p + [0, h]) - f0) / h
            det = ex[:, 0] * ey[:, 1] - ey[:, 0] * ex[:, 1]
            det = np.where(np.abs(det) < 1e-12, np.nan, det)
            dx = (r[:, 0] * ey[:, 1] - ey[:, 0] * r[:, 1]) / det
            dy = (ex[:, 0] * r[:, 1] - r[:, 0] * ex[:, 1]) / det
            d = np.c_[dx, dy]
            d = np.where(np.isfinite(d), d, 0.0)
            r0 = np.linalg.norm(r, axis=1)
            t = np.ones(len(p))
            for _ in range(12):  # backtracking line search
                cand = p + d * t[:, None]
                ok = np.linalg.norm(w - self.to_world(cand), axis=1) < r0
                if ok.all():
                    break
                t = np.where(ok, t, t * 0.5)
            cand = p + d * t[:, None]
            ok = np.linalg.norm(w - self.to_world(cand), axis=1) < r0
            p = np.where(ok[:, None], cand, p)
            if np.abs(d * t[:, None]).max() < _INV_TOL_PX:
                break
        return p

    def residuals_ft(self) -> np.ndarray:
        """Distance in ft between fitted and given world position, per enabled GCP."""
        gs = self._enabled()
        P = np.array([[g.px, g.py] for g in gs], float)
        W = np.array([[g.wx, g.wy] for g in gs], float)
        return np.linalg.norm(self.to_world(P) - W, axis=1)

    def dense_grid(self, shape: tuple[int, int], step: int = 150) -> list[GCP]:
        """Auto GCPs on a regular pixel grid over (h, w), including the last row/column."""
        h, w = shape
        xs = sorted(set(range(0, w, step)) | {w - 1})
        ys = sorted(set(range(0, h, step)) | {h - 1})
        P = np.array([[x, y] for y in ys for x in xs], float)
        W = self.to_world(P)
        return [GCP(float(p[0]), float(p[1]), float(q[0]), float(q[1]), "auto")
                for p, q in zip(P, W)]

    # -- serialization -----------------------------------------------------
    def to_json(self) -> dict:
        return {"smoothing": self.smoothing, "gcps": [asdict(g) for g in self.gcps]}

    @classmethod
    def from_json(cls, d: dict) -> "SheetTransform":
        return cls([GCP(**g) for g in d["gcps"]], float(d["smoothing"]))
