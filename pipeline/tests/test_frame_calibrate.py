import numpy as np
import pytest

from sfzmaps.frame import calibrate, layout


def _softmax(z):
    e = np.exp(z - z.max(0)); return e / e.sum(0)


def test_temperature_is_recovered():
    rng = np.random.default_rng(0)
    n = 200_000
    z0 = rng.normal(0, 4, (3, n)).astype(np.float32)
    z = np.stack([z0, z0])                                   # two identical "seeds"
    p = _softmax(z0 / 2)
    u = rng.random(n)
    y = (u > p[0]).astype(np.uint8) + (u > p[0] + p[1]).astype(np.uint8)
    assert calibrate.fit_temperature([(z, y)]) == pytest.approx(2.0, rel=0.05)


def test_temperature_interval_brackets_the_fit():
    rng = np.random.default_rng(3)
    samples = []
    for _ in range(6):
        z0 = rng.normal(0, 3, (3, 20_000)).astype(np.float32)
        p = _softmax(z0 / 2); u = rng.random(20_000)
        y = (u > p[0]).astype(np.uint8) + (u > p[0] + p[1]).astype(np.uint8)
        samples.append((z0[None], y))
    t = calibrate.fit_temperature(samples)
    lo, hi = calibrate.temperature_interval(samples, n=20)
    assert lo <= t * 1.1 and hi >= t * 0.9 and lo < hi


def test_reliability_of_calibrated_chances():
    rng = np.random.default_rng(1)
    n = 100_000
    z = rng.normal(0, 2, (1, 3, n)).astype(np.float32)
    p = layout.combine(list(z), 1.0)
    u = rng.random(n)
    y = (u > p[0]).astype(np.uint8) + (u > p[0] + p[1]).astype(np.uint8)
    rows = calibrate.reliability([(z, y)], 1.0)
    assert sum(r["n"] for r in rows) == n
    for row in rows:
        if row["n"] > 2000:
            assert abs(row["mean_q"] - row["share_outside"]) < 0.03


def test_sample_draws_ink_pixels_only():
    rng = np.random.default_rng(0)
    h = w = 20
    zs = [rng.normal(size=(3, h, w)).astype(np.float32)]
    cls = np.zeros((h, w), np.uint8); cls[:, 10:] = 1
    lum = np.ones((h, w)); lum[0, 0] = 0.0
    z, y = calibrate.sample(zs, cls, lum, rng, n=500)
    assert z.shape == (1, 3, 500) and y.shape == (500,) and y.dtype == np.uint8
    assert z.dtype == np.float32
    assert set(np.unique(y)) <= {0, 1}


def test_temperature_interval_refits_as_a_fresh_search_does():
    rng = np.random.default_rng(5)
    samples = []
    for k in range(8):
        z0 = rng.normal(0, 3, (3, 5_000)).astype(np.float32)
        p = _softmax(z0 / (1.2 + 0.2 * k)); u = rng.random(5_000)           # sheets that differ: a real spread
        y = (u > p[0]).astype(np.uint8) + (u > p[0] + p[1]).astype(np.uint8)
        samples.append((np.stack([z0, z0 + 0.1]), y))
    n = 30
    pick = np.random.default_rng(0)
    fresh = [calibrate.fit_temperature([samples[i] for i in pick.integers(0, 8, 8)]) for _ in range(n)]
    want = np.percentile(fresh, (5, 95))
    got = calibrate.temperature_interval(samples, n=n)
    assert got == pytest.approx(tuple(want), rel=2e-3)
    assert want[1] / want[0] > 1.05                                         # the interval is not a point
