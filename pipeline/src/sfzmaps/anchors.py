"""Rough placement of a scan from text anchors (block numbers, street-name pairs).

match_labels turns vision-model labels into candidate pixel -> world pairs:
  * block_number: text normalized exactly like datasf (digits only, zero-pad to 4, drop
    suffix letters); labels matching no modern block are dropped. World point is the block
    centroid; for a MultiPolygon block, the representative point of its largest part.
  * street_pair: two street-name labels within STREET_PAIR_PX whose normalized names both
    appear in one modern intersection. Pixel point is the midpoint of the two labels (the
    Candidate's synthetic Label carries it), weight 0.5. Every qualifying intersection is
    emitted; RANSAC picks.

rough_fit runs RANSAC over a similarity transform with any rotation and uniform scale.
Pixels are y-down and world is y-up, so pixel y is negated first; the model is then a
proper (orientation preserving) complex similarity w = a*z + b, z = x - i*y.
  * Deterministic: seeded RNG (SEED); all pairs are tried when there are few enough.
  * Score: sum of candidate weights over inliers (residual <= tol), so a street-pair counts
    half a block label. Ties go to the lower weighted residual.
  * Refit: weights are used in a weighted least-squares similarity refit that re-selects
    inliers (3 rounds). The returned SheetTransform is fitted by SheetTransform.fit, which
    is unweighted affine on the inlier GCPs (weights do not enter that final fit).
rotation_deg is arg(a): the world-frame angle of the pixel +x axis, in (-180, 180].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon

from sfzmaps import config
from sfzmaps.reading import Label
from sfzmaps.sources.datasf import Reference, _norm_block, normalize_street
from sfzmaps.transform import GCP, SheetTransform

SEED = 20241029
STREET_PAIR_PX = 400
STREET_PAIR_WEIGHT = 0.5
MAX_ITERS = 3000
MIN_SAMPLE_PX = 50.0
SCALE_RANGE = (0.3, 10.0)
REFIT_ROUNDS = 3


@dataclass
class Candidate:
    label: Label
    world: tuple[float, float]
    kind: str  # "block" or "street_pair"
    weight: float


@dataclass
class RoughFit:
    transform: SheetTransform
    inliers: list[Candidate]
    scale_ft_per_px: float
    rotation_deg: float


def _block_point(g) -> tuple[float, float]:
    if isinstance(g, MultiPolygon):
        g = max(g.geoms, key=lambda p: p.area)
        p = g.representative_point()
    else:
        p = g.centroid
    return (p.x, p.y)


def match_labels(labels: list[Label], ref: Reference) -> list[Candidate]:
    pts: dict[str, tuple[float, float]] = {}
    for num, g in zip(ref.blocks["block_num"], ref.blocks.geometry):
        k = str(num)
        # duplicate rows for one number: keep the one with the larger area
        if g is None or g.is_empty or (k in pts and pts[k][2] >= g.area):
            continue
        pts[k] = (*_block_point(g), g.area)
    out: list[Candidate] = []
    for lb in labels:
        if lb.kind != "block_number" or not any(c.isdigit() for c in lb.text):
            continue
        p = pts.get(_norm_block(lb.text))
        if p is not None:
            out.append(Candidate(lb, (p[0], p[1]), "block", 1.0))

    by_street: dict[str, set[int]] = {}
    for i, sts in enumerate(ref.intersections["streets"]):
        for s in sts:
            by_street.setdefault(s, set()).add(i)
    xy = [(g.x, g.y) for g in ref.intersections.geometry]
    streets = [(lb, normalize_street(lb.text)) for lb in labels if lb.kind == "street_name"]
    for i, (a, na) in enumerate(streets):
        for b, nb in streets[i + 1:]:
            if na == nb or np.hypot(a.x - b.x, a.y - b.y) > STREET_PAIR_PX:
                continue
            hits = by_street.get(na, set()) & by_street.get(nb, set())
            if not hits:
                continue
            mid = Label(kind="street_name", text=f"{a.text} / {b.text}",
                        x=int(round((a.x + b.x) / 2)), y=int(round((a.y + b.y) / 2)),
                        confidence=min(a.confidence, b.confidence))
            for h in sorted(hits):
                out.append(Candidate(mid, (float(xy[h][0]), float(xy[h][1])),
                                     "street_pair", STREET_PAIR_WEIGHT))
    return out


def _z(a: np.ndarray) -> np.ndarray:
    return a[:, 0] - 1j * a[:, 1]


def _wls(z, w, wt):
    """Weighted LS complex similarity w = a z + b."""
    sw = wt.sum()
    zm, wm = (wt * z).sum() / sw, (wt * w).sum() / sw
    dz = z - zm
    den = (wt * np.abs(dz) ** 2).sum()
    if den == 0:
        return None
    a = (wt * np.conj(dz) * (w - wm)).sum() / den
    return a, wm - a * zm


def rough_fit(cands: list[Candidate], *, min_inliers: int = 6,
              ransac_tol_ft: float = 150) -> RoughFit | None:
    if len(cands) < min_inliers:
        return None
    z = _z(np.array([[c.label.x, c.label.y] for c in cands], float))
    w = np.array([complex(*c.world) for c in cands])
    wt = np.array([c.weight for c in cands], float)
    n = len(cands)

    def evaluate(a, b):
        r = np.abs(a * z + b - w)
        inl = r <= ransac_tol_ft
        return inl, float(wt[inl].sum()), float((wt * r)[inl].sum())

    def sample_pairs():
        if n * (n - 1) // 2 <= MAX_ITERS:
            yield from ((i, j) for i in range(n) for j in range(i + 1, n))
        else:
            rng = np.random.default_rng(SEED)
            for _ in range(MAX_ITERS):
                i, j = rng.choice(n, 2, replace=False)
                yield int(i), int(j)

    best, best_key = None, (-1.0, 0.0)
    for i, j in sample_pairs():
        dz, dw = z[j] - z[i], w[j] - w[i]
        if abs(dz) < MIN_SAMPLE_PX or abs(dw) == 0:
            continue
        a = dw / dz
        if not SCALE_RANGE[0] <= abs(a) <= SCALE_RANGE[1]:
            continue
        inl, score, resid = evaluate(a, w[i] - a * z[i])
        key = (score, -resid)
        if key > best_key:
            best, best_key = (a, w[i] - a * z[i]), key
    if best is None:
        return None

    a, b = best
    inl, _, _ = evaluate(a, b)
    for _ in range(REFIT_ROUNDS):
        if inl.sum() < 2:
            return None
        fit = _wls(z[inl], w[inl], wt[inl])
        if fit is None:
            return None
        a, b = fit
        new, _, _ = evaluate(a, b)
        if (new == inl).all():
            break
        inl = new
    scale = float(abs(a))
    if inl.sum() < min_inliers or not SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]:
        return None

    inliers = [c for c, k in zip(cands, inl) if k]
    gcps = [GCP(c.label.x, c.label.y, c.world[0], c.world[1], "anchor") for c in inliers]
    try:
        tr = SheetTransform(gcps, smoothing=config.TPS_SMOOTHING).fit()
    except ValueError:
        return None
    return RoughFit(tr, inliers, scale, float(np.degrees(np.angle(a))))
