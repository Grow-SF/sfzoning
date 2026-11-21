import math

import numpy as np
import pytest
from synthetic_frame import make_page

from sfzmaps.frame import notches
from sfzmaps.frame.geometry import Frame, cost_field
from sfzmaps.frame.shape import Params

PHI = 1.2
C = math.cos(math.radians(PHI))
# a ruled box at the bottom side, 250 px deep (the spike's synthetic page A, scaled to 900 x 1200)
BOX = dict(side="bottom", a0=112.0, a1=375.0, d=70 - C * 450 + 250)


def _true_frame(tr):
    return Frame(phi=tr["phi"], offset=dict(tr["offset"]))


def _page(psi, ruled=True, seed=0):
    return make_page(phi=PHI, notch=dict(BOX, psi=psi, ruled=ruled), seed=seed)


def test_ruled_notch_found_and_unruled_blob_is_not():
    params = Params(lam_notch=0.0, beta=3.0, lam_block=0.0)
    found = {}
    for ruled in (True, False):
        lum, p, tr = _page(PHI, ruled)
        got = notches.greedy_notches(cost_field(p, lum), 1 - lum, _true_frame(tr), params)
        found[ruled] = got[0]
    assert found[True].gain > found[False].gain
    lam = (found[True].gain + found[False].gain) / 2
    params = Params(lam_notch=lam, beta=3.0, lam_block=0.0)

    lum, p, tr = _page(PHI, ruled=True)
    got = notches.greedy_notches(cost_field(p, lum), 1 - lum, _true_frame(tr), params)
    assert len(got) == 1
    n = got[0]
    assert n.side == "bottom"
    assert abs(n.a0 - BOX["a0"]) <= 2 and abs(n.a1 - BOX["a1"]) <= 2 and abs(n.d - BOX["d"]) <= 2, n
    assert n.gain > lam

    lum, p, tr = _page(PHI, ruled=False)
    assert notches.greedy_notches(cost_field(p, lum), 1 - lum, _true_frame(tr), params) == []


def test_notch_may_be_rotated():
    psi = PHI + 1.5
    lum, p, tr = _page(psi)
    got = notches.greedy_notches(cost_field(p, lum), 1 - lum, _true_frame(tr), Params(2e4, 3.0, 0.0))
    assert len(got) == 1
    n = got[0]
    assert abs(n.psi - psi) <= 0.2, n.psi
    assert abs(n.a0 - BOX["a0"]) <= 3 and abs(n.a1 - BOX["a1"]) <= 3 and abs(n.d - BOX["d"]) <= 3, n


def test_notch_is_at_least_two_strips_wide():
    # a 6-px wide ruled slot: narrower than 2 x STRIP + 1, so it is no notch however much it gains
    h, w = 400, 600
    lum = np.ones((h, w), np.float32)
    lum[300:, 200:206] = 0.0
    p = np.zeros((3, h, w), np.float32)
    p[0] = 1.0
    p[:, 300:, 200:206] = np.array([0.0, 1.0, 0.0])[:, None, None]
    fr = Frame(phi=0.0, offset={"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0})
    got = notches.greedy_notches(cost_field(p, lum), 1 - lum, fr, Params(0.0, 3.0, 0.0))
    assert all(n.a1 - n.a0 >= 2 * notches.STRIP + 1 for n in got)


@pytest.mark.parametrize("seed", range(40))
def test_kadane_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n_e, n_a = rng.integers(3, 9), rng.integers(1, 12)
    Dv = rng.normal(0, 1, (n_e, n_a))
    Iv = rng.random((n_e, n_a))
    bw, s, kc = float(rng.choice([0.0, 1.0, 3.0])), int(rng.choice([1, 8])), int(rng.integers(1, 3))
    d_lo, d_hi = 1, int(rng.integers(1, n_e + 1))
    got = notches._kadane(Dv, Iv, bw, s, kc, d_lo, d_hi)
    norm = kc * s
    wmin = math.ceil((2 * notches.STRIP + 1) / s)
    best = -np.inf
    for d in range(max(d_lo, kc), d_hi + 1):
        for a0 in range(n_a):
            for a1 in range(a0 + wmin, n_a + 1):
                v = Dv[:d, a0:a1].sum()
                v += bw * Iv[d - kc:d, a0:a1].sum() / norm
                v += bw * Iv[:d, a0:min(a0 + kc, n_a)].sum() / norm
                v += bw * Iv[:d, max(a1 - kc, 0):a1].sum() / norm
                best = max(best, v)
    if not np.isfinite(best):
        assert not np.isfinite(got[0])
        return
    assert got[0] == pytest.approx(best)
    v, a0, a1, d = got
    check = Dv[:d, a0:a1].sum() + bw * (
        Iv[d - kc:d, a0:a1].sum() + Iv[:d, a0:min(a0 + kc, n_a)].sum() + Iv[:d, max(a1 - kc, 0):a1].sum()
    ) / norm
    assert check == pytest.approx(v)
