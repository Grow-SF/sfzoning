import itertools

import numpy as np
import pytest
import torch
from scipy import ndimage

from sfzmaps.frame import predict
from sfzmaps.frame.predict import predict_coarse, predict_tiles
from sfzmaps.frame.train import (OUTLINE_REACH, Page, Recipe, _coarse_arrays, _coarse_batch, _fine_batch,
                                 _near_outline, augment, choose_steps, train_coarse, train_fine)

RULE_AT, RULE_PX = 10, 3


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    """Toy-sized nets: the real sizes would take minutes on cpu."""
    monkeypatch.setattr(predict, "TILE", 32)
    monkeypatch.setattr(predict, "COARSE_SIDE", 64)
    monkeypatch.setattr(predict, "COARSE_LEVELS", 3)
    monkeypatch.setattr(predict, "COARSE_WIDTH", 4)
    monkeypatch.setattr(predict, "FINE_LEVELS", 3)
    monkeypatch.setattr(predict, "FINE_WIDTH", 8)


def toy_page(seed, h=96, w=128):
    """Paper with a 3-px frame rule 10 px in from each edge, plus random 'streets' inside; the mask is the inside of
    the rule.  The prior is the mask blurred, a stand-in for the page pass's answer."""
    rng = np.random.default_rng(seed)
    lum = np.ones((h, w), np.float32)
    a, b = RULE_AT, RULE_AT + RULE_PX
    lum[a:h - a, a:w - a] = 0.0
    lum[b:h - b, b:w - b] = 1.0
    mask = np.zeros((h, w), np.uint8)
    mask[b:h - b, b:w - b] = 1
    lo_r, hi_r, lo_c, hi_c = b + 3, h - b - 3, b + 3, w - b - 3
    for _ in range(4):
        r = int(rng.integers(lo_r, hi_r - 2))
        lum[r:r + 2, lo_c:hi_c] = 0.0
        c = int(rng.integers(lo_c, hi_c - 2))
        lum[lo_r:hi_r, c:c + 2] = 0.0
    prior = ndimage.gaussian_filter(mask.astype(np.float32), 4.0)
    return Page(id=("toy", str(seed)), lum=lum, mask=mask, prior=prior)


def _rule_px(pg):
    m = np.zeros(pg.lum.shape, bool)
    h, w = m.shape
    a, b = RULE_AT, RULE_AT + RULE_PX
    m[a:h - a, a:w - a] = True
    m[b:h - b, b:w - b] = False
    return m


def _inside_ink_px(pg):
    return pg.mask.astype(bool) & (pg.lum < 0.5)


def _transforms():
    """The eight quarter-turn / mirror-flip maps of a 2-D array."""
    return [lambda x, k=k, f=f: np.rot90(np.flip(x, 1) if f else x, k) for k, f in itertools.product(range(4), (0, 1))]


def test_augment_keeps_arrays_aligned():
    lum = np.random.default_rng(1).random((6, 8)).astype(np.float32)
    mask = (lum > 0.5).astype(np.float32)
    seen = set()
    for seed in range(40):
        l2, m2, plain = augment(np.random.default_rng(seed), lum, mask, lum.copy())
        assert l2.shape == m2.shape == plain.shape
        # only the first array gets the brightness / contrast / gamma / blur changes; the others are moved only
        hits = [i for i, t in enumerate(_transforms()) if np.array_equal(t(lum), plain)]
        assert len(hits) == 1
        assert np.array_equal(_transforms()[hits[0]](mask), m2)
        seen.add(hits[0])
    assert len(seen) == 8


def test_augment_changes_only_the_first_arrays_levels():
    lum = np.random.default_rng(2).random((16, 16)).astype(np.float32)
    changed = 0
    for seed in range(10):
        l2, plain = augment(np.random.default_rng(seed), lum, lum.copy())
        assert l2.dtype == np.float32 and l2.min() >= 0.0 and l2.max() <= 1.0
        changed += not np.allclose(l2, plain)
    assert changed == 10


def test_fine_pass_learns_the_toy_frame():
    pages = [toy_page(s) for s in range(4)]
    net, _ = train_fine(pages, Recipe(steps=300, batch=2, check_every=100), "cpu")
    pg = toy_page(9)
    p = predict_tiles(net, pg.lum, pg.prior)
    assert p[_rule_px(pg)].mean() < 0.2 and p[_inside_ink_px(pg)].mean() > 0.8


def test_coarse_pass_learns_the_toy_frame():
    pages = [toy_page(s, h=128, w=96) if s % 2 else toy_page(s) for s in range(4)]   # mixed shapes in a batch
    net, _ = train_coarse(pages, Recipe(steps=600, batch=2, check_every=200), "cpu")
    pg = toy_page(9)
    p = predict_coarse(net, pg.lum)
    h, w = p.shape
    inside = np.zeros(p.shape, bool)
    inside[30:h - 30, 30:w - 30] = True
    outside = np.zeros(p.shape, bool)
    outside[:, :4] = outside[:, -4:] = outside[:4] = outside[-4:] = True
    assert p[inside].mean() > 0.8 and p[outside].mean() < 0.2


def test_training_is_deterministic():
    a, _ = train_coarse([toy_page(0)], Recipe(steps=5, batch=1), "cpu")
    b, _ = train_coarse([toy_page(0)], Recipe(steps=5, batch=1), "cpu")
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    c, _ = train_fine([toy_page(0)], Recipe(steps=5, batch=2), "cpu")
    d, _ = train_fine([toy_page(0)], Recipe(steps=5, batch=2), "cpu")
    assert all(torch.equal(x, y) for x, y in zip(c.state_dict().values(), d.state_dict().values()))


def test_snapshot_at_a_step_is_a_run_of_that_many_steps():
    """A snapshot taken at step s while training (and validating) for longer is, bit for bit, the net a run of s steps
    returns: the chosen step's net costs no extra training."""
    val = [toy_page(7)]
    for trainer, batch in ((train_coarse, 1), (train_fine, 2)):
        long, info = trainer([toy_page(0)], Recipe(steps=4, batch=batch, check_every=2), "cpu", val=val,
                             snapshots=True)
        assert sorted(info["snapshots"]) == [2, 4]
        short, _ = trainer([toy_page(0)], Recipe(steps=2, batch=batch, check_every=2), "cpu")
        for want, net_step in ((short.state_dict(), 2), (long.state_dict(), 4)):
            snap = info["snapshots"][net_step]
            assert list(snap) == list(want)
            assert all(torch.equal(snap[k], want[k]) for k in want)
        assert any(not torch.equal(info["snapshots"][2][k], info["snapshots"][4][k]) for k in want)
    _, info = train_coarse([toy_page(0)], Recipe(steps=2, batch=1, check_every=1), "cpu")
    assert "snapshots" not in info


def test_validation_curve_and_returned_net():
    val = [toy_page(7)]
    net, info = train_coarse([toy_page(0)], Recipe(steps=4, batch=1, check_every=2), "cpu", val=val)
    assert [s for s, _ in info["val_curve"]] == [2, 4]
    assert all(np.isfinite(v) and v > 0 for _, v in info["val_curve"])
    assert not net.training and next(net.parameters()).device.type == "cpu"
    net, info = train_fine([toy_page(0)], Recipe(steps=4, batch=2, check_every=2), "cpu", val=val)
    assert [s for s, _ in info["val_curve"]] == [2, 4]
    assert not net.training and next(net.parameters()).device.type == "cpu"
    _, info = train_fine([toy_page(0)], Recipe(steps=2, batch=1), "cpu")
    assert info == {"val_curve": []}


def test_outline_band():
    pg = toy_page(0)
    assert _near_outline(np.ones_like(pg.mask)).size == 0 and _near_outline(np.zeros_like(pg.mask)).size == 0
    big = np.zeros((700, 700), np.uint8)
    big[300:400, 300:400] = 1
    near = np.zeros(big.size, bool)
    near[_near_outline(big)] = True
    near = near.reshape(big.shape)
    edge = big.astype(bool) & ~ndimage.binary_erosion(big.astype(bool), border_value=1)
    d = ndimage.distance_transform_edt(~edge)
    assert np.array_equal(near, d <= OUTLINE_REACH) and not near.all()


def test_fine_pass_takes_pages_with_no_outline():
    pg = toy_page(0)
    full = Page(id=("toy", "full"), lum=pg.lum, mask=np.ones_like(pg.mask), prior=np.ones_like(pg.prior))
    small = Page(id=("toy", "small"), lum=pg.lum[:20, :24], mask=pg.mask[:20, :24], prior=pg.prior[:20, :24])
    net, _ = train_fine([full, small], Recipe(steps=3, batch=2), "cpu")
    assert all(torch.isfinite(v).all() for v in net.state_dict().values() if v.is_floating_point())
    # an outline crop of a page with no outline falls back to a uniform one: a whole-map crop, learned as map
    _, t, w = _fine_batch(np.random.default_rng(0), [full], [_near_outline(full.mask)], 4, 0.1)
    assert w.sum() > 0 and (t * w).sum() / w.sum() > 0.95


def test_fine_crops_have_an_unweighted_margin():
    tile, margin = predict.TILE, predict._margin()
    pages = [toy_page(0), toy_page(1, h=60, w=50)]
    x, t, w = _fine_batch(np.random.default_rng(3), pages, [_near_outline(p.mask) for p in pages], 6, 0.1)
    full = tile + 2 * margin
    assert x.shape == (6, 2, full, full) and t.shape == w.shape == (6, 1, full, full)
    core = np.zeros((full, full), bool)
    core[margin:margin + tile, margin:margin + tile] = True
    assert (w[:, :, ~core] == 0).all()
    assert (w[:, :, core].reshape(6, -1).sum(1) > 0).all()


def test_coarse_batches_are_the_prediction_square():
    pages = [toy_page(0), toy_page(1, h=128, w=60)]
    coarse = [_coarse_arrays(pg, 0.1) for pg in pages]
    x, t, w = _coarse_batch(np.random.default_rng(0), coarse, 4)
    side = predict.COARSE_SIDE
    assert x.shape == t.shape == w.shape == (4, 1, side, side)
    assert (x[w == 0] == 1.0).all() and (t[w == 0] == 0).all()
    arrays, scale = coarse[0]
    assert arrays[0].shape == (48, 64) and scale == pytest.approx(0.5)


def test_blur_is_measured_in_page_px():
    two_levels = np.zeros((32, 32), np.float32)
    two_levels[:, 16:] = 1.0
    blurred_any = False
    for seed in range(20):
        fine = augment(np.random.default_rng(seed), two_levels, scale=1.0)[0]
        coarse = augment(np.random.default_rng(seed), two_levels, scale=1e-3)[0]
        assert len(np.unique(coarse)) <= 2           # a blur of under 0.002 px leaves two levels
        blurred_any |= len(np.unique(fine)) > 2
    assert blurred_any


def test_fine_pass_needs_a_prior():
    pg = toy_page(0)
    with pytest.raises(ValueError):
        train_fine([Page(id=pg.id, lum=pg.lum, mask=pg.mask, prior=None)], Recipe(steps=1, batch=1), "cpu")


def test_choose_steps():
    assert choose_steps([[(250, 0.5), (500, 0.3), (750, 0.4)], [(250, 0.6), (500, 0.2), (750, 0.2)]]) == 500
    assert choose_steps([[(250, 0.3), (500, 0.3)]]) == 250
    with pytest.raises(ValueError):
        choose_steps([[(250, 0.3), (500, float("nan"))], [(250, 0.4), (500, 0.1)]])
