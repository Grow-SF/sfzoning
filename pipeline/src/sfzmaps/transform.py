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
Hence the default (config.TPS_SMOOTHING = 1.0): a lone 40 ft outlier moves the fit about
5.5 ft while real distortion is still largely corrected. Effect scales with GCP density
(denser grids resist outliers more); Task 13 tunes against the answer key.

Fewer than MIN_GCPS enabled GCPs is an error. With MIN_GCPS..TPS_MIN_GCPS-1 (3-9) the
fit is affine only: a TPS on so few points is unstable.

Conditioning: EPSG:2227 coordinates are ~6e6/2e6 ft. Both pixel and world coordinates
are centered internally (world by its mean, pixels also scaled) before any fit.

to_pixel: rather than fit an independent reverse TPS (which cannot guarantee a round
trip), invert the forward model numerically: start from the inverse of the affine part and
run fixed-point iterations p += A^-1 (w - to_world(p)). The TPS part is small and smooth,
so this converges to far below 0.5 px in a few steps and is exactly consistent with
to_world.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import RBFInterpolator

MIN_GCPS = 3
TPS_MIN_GCPS = 10
_INV_ITERS = 25
_INV_TOL_PX = 1e-6


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
        self._fitted = True
        return self

    # -- evaluation --------------------------------------------------------
    def to_world(self, px: np.ndarray) -> np.ndarray:
        if not self._fitted:
            self.fit()
        p = np.atleast_2d(np.asarray(px, float))
        pn = (p - self._pm) / self._ps
        out = np.c_[pn, np.ones(len(pn))] @ self._coef
        if self._rbf is not None:
            out = out + self._rbf(pn)
        return out + self._wm

    def to_pixel(self, w: np.ndarray) -> np.ndarray:
        if not self._fitted:
            self.fit()
        w = np.atleast_2d(np.asarray(w, float))
        p = self._pm + (w - self.to_world(self._pm[None])) @ self._lin_inv.T
        for _ in range(_INV_ITERS):
            step = (w - self.to_world(p)) @ self._lin_inv.T
            p = p + step
            if np.abs(step).max() < _INV_TOL_PX:
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
