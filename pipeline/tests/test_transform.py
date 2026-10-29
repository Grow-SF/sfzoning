import numpy as np
import pytest

from sfzmaps import config
from sfzmaps.transform import GCP, SheetTransform

W0 = np.array([6_000_000.0, 2_000_000.0])


def _affine(p):
    p = np.asarray(p, float)
    return W0 + np.c_[3.0 * p[:, 0] + 0.2 * p[:, 1], 0.1 * p[:, 0] - 3.0 * p[:, 1]]


def _warped(p):
    p = np.asarray(p, float)
    u = p / 7000.0
    d = np.c_[30 * np.sin(3 * u[:, 0]) * u[:, 1], 20 * np.cos(2 * u[:, 1]) * u[:, 0]]
    return _affine(p) + 5 * d


def _grid(fn, n=10, lo=300.0, hi=6700.0, source="auto"):
    g = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(g, g)
    P = np.c_[X.ravel(), Y.ravel()]
    W = fn(P)
    return [GCP(a, b, c, d, source) for (a, b), (c, d) in zip(P, W)]


def test_affine_recovered_exactly():
    t = SheetTransform(_grid(_affine), smoothing=config.TPS_SMOOTHING).fit()
    assert np.abs(t.residuals_ft()).max() < 0.01
    q = np.array([[1234.5, 5678.9], [10.0, 20.0]])
    assert np.abs(t.to_world(q) - _affine(q)).max() < 0.01


def test_smooth_warp_does_not_chase_outlier():
    gcps = _grid(_warped)
    base = SheetTransform(list(gcps), smoothing=config.TPS_SMOOTHING).fit()
    k = 55
    bad = list(gcps)
    g = gcps[k]
    bad[k] = GCP(g.px, g.py, g.wx + 40.0, g.wy, g.source)
    t = SheetTransform(bad, smoothing=config.TPS_SMOOTHING).fit()
    p = np.array([[g.px, g.py]])
    assert np.linalg.norm(t.to_world(p) - base.to_world(p)) < 10.0


def test_roundtrip_realistic_magnitudes():
    t = SheetTransform(_grid(_warped), smoothing=config.TPS_SMOOTHING).fit()
    q = np.random.default_rng(1).uniform(300, 6700, (200, 2))
    assert np.abs(t.to_pixel(t.to_world(q)) - q).max() < 0.5
    # accuracy of to_world itself at 6e6/2e6 ft magnitudes: smooth truth, bounded error
    assert np.abs(t.to_world(q) - _warped(q)).max() < 30.0


def test_roundtrip():
    t = SheetTransform(_grid(_warped), smoothing=config.TPS_SMOOTHING).fit()
    p = np.array([[100.0, 100.0], [3500.0, 3500.0], [6900.0, 200.0]])
    assert np.abs(t.to_pixel(t.to_world(p)) - p).max() < 0.5


def test_json_roundtrip():
    gcps = _grid(_warped, n=4)
    gcps[0].enabled = False
    t = SheetTransform(gcps, smoothing=2.5).fit()
    t2 = SheetTransform.from_json(t.to_json()).fit()
    assert t2.smoothing == 2.5
    assert t2.gcps == t.gcps and t2.gcps[0].enabled is False
    q = np.array([[1000.0, 2000.0]])
    assert np.allclose(t.to_world(q), t2.to_world(q))


def test_zero_smoothing_rejected():
    with pytest.raises(ValueError, match="smoothing"):
        SheetTransform(_grid(_affine), smoothing=0.0).fit()
    with pytest.raises(ValueError, match="smoothing"):
        SheetTransform(_grid(_affine), smoothing=-1.0).fit()


def test_too_few_gcps():
    gcps = _grid(_affine, n=4)
    gcps = gcps[:2] + [GCP(1, 1, 2, 2, "auto", enabled=False)] * 5
    with pytest.raises(ValueError, match="at least 3"):
        SheetTransform(gcps, smoothing=1.0).fit()


def test_few_gcps_affine_only():
    t = SheetTransform(_grid(_affine, n=3)[:5], smoothing=1.0).fit()
    assert t._rbf is None
    q = np.array([[500.0, 900.0]])
    assert np.abs(t.to_world(q) - _affine(q)).max() < 0.01


def test_disabled_excluded_from_fit_and_residuals():
    gcps = _grid(_affine, n=4)
    gcps.append(GCP(3000, 3000, W0[0] + 99999, W0[1], "manual", enabled=False))
    t = SheetTransform(gcps, smoothing=1.0).fit()
    assert t.residuals_ft().shape == (16,)
    assert np.abs(t.to_world(np.array([[3000.0, 3000.0]])) - _affine([[3000, 3000]])).max() < 0.01
    assert len(t.to_json()["gcps"]) == 17


def test_dense_grid_covers_image_including_edges():
    t = SheetTransform(_grid(_warped), smoothing=1.0).fit()
    h, w = 1000, 1301
    g = t.dense_grid((h, w), step=150)
    assert all(x.source == "auto" and x.enabled for x in g)
    xs = sorted({x.px for x in g})
    ys = sorted({x.py for x in g})
    assert xs[0] == 0 and xs[-1] == w - 1 and ys[0] == 0 and ys[-1] == h - 1
    assert len(g) == len(xs) * len(ys)
    p = np.array([[x.px, x.py] for x in g])
    assert np.allclose(t.to_world(p), [[x.wx, x.wy] for x in g])
