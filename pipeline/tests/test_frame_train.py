import dataclasses
import itertools
import logging
import threading

import numpy as np
import pytest
import torch
from scipy import ndimage

from sfzmaps.frame import predict, scores, train
from sfzmaps.frame.predict import predict_coarse, predict_tiles
from sfzmaps.frame.scores import UNSURE_PX
from sfzmaps.frame.train import (BLUR_SHARE, BLUR_SIGMA, BRIGHTNESS, CONTRAST, GAMMA, OUTLINE_REACH, SCALE_JITTER,
                                 Page, Recipe, _batches, _coarse_arrays, _coarse_draw, _coarse_make, _fine_aux,
                                 _fine_draw, _fine_make, _near_outline, _val_loss, apply_aug, choose_steps, draw_aug,
                                 train_coarse, train_fine)
from sfzmaps.nets.unet import Folded

RULE_AT, RULE_PX = 10, 3


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    """Toy-sized folded nets: the real sizes would take minutes on cpu."""
    monkeypatch.setattr(predict, "TILE", 32)
    monkeypatch.setattr(predict, "COARSE_SIDE", 64)
    monkeypatch.setattr(predict, "COARSE_FOLD", 4)
    monkeypatch.setattr(predict, "COARSE_LEVELS", 3)
    monkeypatch.setattr(predict, "COARSE_WIDTH", 8)
    monkeypatch.setattr(predict, "FINE_FOLD", 4)
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


def _numpy_augment(rng, *arrays, scale=1.0):
    """The numpy augmentation the device one replaced (Task 6), kept here as the reference."""
    turns, flip = int(rng.integers(4)), bool(rng.integers(2))
    gamma = float(np.exp(rng.uniform(-np.log(GAMMA), np.log(GAMMA))))
    contrast = 1.0 + float(rng.uniform(-CONTRAST, CONTRAST))
    brightness = float(rng.uniform(-BRIGHTNESS, BRIGHTNESS))
    blurred, sigma = rng.random() < BLUR_SHARE, float(rng.uniform(0.0, BLUR_SIGMA)) * scale
    out = []
    for i, a in enumerate(arrays):
        a = np.rot90(np.flip(a, 1) if flip else a, turns)
        if i == 0:
            a = np.clip(a.astype(np.float32), 0.0, 1.0) ** gamma
            a = np.clip((a - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
            if blurred and sigma > 0:
                a = ndimage.gaussian_filter(a, sigma, mode="nearest")
            a = a.astype(np.float32)
        out.append(np.ascontiguousarray(a))
    return tuple(out)


def _apply(aug, *arrays):
    return tuple(t.numpy() for t in apply_aug(aug, *(torch.from_numpy(a) for a in arrays)))


def test_device_augment_matches_geometry():
    """The same draws, applied in torch (here on cpu), turn and flip every array as the numpy version did, and change
    the luminance's levels as it did; the generator ends in the same state."""
    rng0 = np.random.default_rng(1)
    lum = rng0.random((12, 20)).astype(np.float32)
    prior, target, weight = (rng0.random((12, 20)).astype(np.float32) for _ in range(3))
    seen, blurred = set(), 0
    for seed in range(40):
        old_rng, new_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        want = _numpy_augment(old_rng, lum, prior, target, weight)
        aug = draw_aug(new_rng)
        got = _apply(aug, lum, prior, target, weight)
        for g, w in zip(got[1:], want[1:]):
            assert np.array_equal(g, w)
        np.testing.assert_allclose(got[0], want[0], atol=1e-5)
        assert old_rng.random() == new_rng.random()
        hits = [i for i, t in enumerate(_transforms()) if np.array_equal(t(prior), got[1])]
        assert len(hits) == 1
        seen.add(hits[0])
        blurred += aug.sigma > 0
    assert len(seen) == 8 and 0 < blurred < 40


def test_augment_changes_only_the_first_arrays_levels():
    lum = np.random.default_rng(2).random((16, 16)).astype(np.float32)
    changed = 0
    for seed in range(10):
        l2, plain = _apply(draw_aug(np.random.default_rng(seed)), lum, lum.copy())
        assert l2.dtype == np.float32 and l2.min() >= 0.0 and l2.max() <= 1.0
        changed += not np.allclose(l2, plain)
    assert changed == 10


def test_blur_is_measured_in_page_px():
    two_levels = np.zeros((32, 32), np.float32)
    two_levels[:, 16:] = 1.0
    blurred_any = False
    for seed in range(20):
        fine = _apply(draw_aug(np.random.default_rng(seed), scale=1.0), two_levels)[0]
        coarse = _apply(draw_aug(np.random.default_rng(seed), scale=1e-3), two_levels)[0]
        assert len(np.unique(coarse)) <= 2           # a blur of under 0.002 px leaves two levels
        blurred_any |= len(np.unique(fine)) > 2
    assert blurred_any


def test_fine_pass_learns_the_toy_frame():
    """With no help from its prior (1/2 all over the page), the pixel pass's correction learns the toy frame beyond
    the label's uncertain strip.  The toy's rule, 1-3 px outside the outline, lies within the strip and weighs 0, so
    each page gets an outer frame bar too (its 4 outermost px, 10-13 px outside the outline)."""
    def flat(pg):
        lum = pg.lum.copy()
        bar = np.ones(lum.shape, bool)
        bar[4:-4, 4:-4] = False
        lum[bar] = 0.0
        return dataclasses.replace(pg, lum=lum, prior=np.full(lum.shape, 0.5, np.float32))
    pages = [flat(toy_page(s)) for s in range(4)]
    # 600 steps: the folded net's 32-px core is 1.6% of its 256-px crop (the margin is the folded reach)
    net, _ = train_fine(pages, Recipe(steps=600, batch=2, check_every=200), "cpu")
    assert isinstance(net, Folded)
    pg = flat(toy_page(9))
    p = predict_tiles(net, pg.lum, pg.prior)
    sure = ~scores.unsure(pg.mask)
    assert not (_rule_px(pg) & sure).any()
    frame_ink = sure & ~pg.mask.astype(bool) & (pg.lum < 0.5)
    assert frame_ink.sum() > 0
    assert p[frame_ink].mean() < 0.2 and p[_inside_ink_px(pg) & sure].mean() > 0.8


def test_coarse_pass_learns_the_toy_frame():
    pages = [toy_page(s, h=128, w=96) if s % 2 else toy_page(s) for s in range(4)]   # mixed shapes in a batch
    net, _ = train_coarse(pages, Recipe(steps=600, batch=2, check_every=200), "cpu")
    assert isinstance(net, Folded)
    pg = toy_page(9)
    p = predict_coarse(net, pg.lum)
    h, w = p.shape
    inside = np.zeros(p.shape, bool)
    inside[30:h - 30, 30:w - 30] = True
    outside = np.zeros(p.shape, bool)
    outside[:, :4] = outside[:, -4:] = outside[:4] = outside[-4:] = True
    assert p[inside].mean() > 0.8 and p[outside].mean() < 0.2


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_training_is_deterministic():
    a, _ = train_coarse([toy_page(0)], Recipe(steps=5, batch=1), "cpu")
    b, _ = train_coarse([toy_page(0)], Recipe(steps=5, batch=1), "cpu")
    assert _same(a, b)
    c, _ = train_fine([toy_page(0)], Recipe(steps=5, batch=2), "cpu")
    d, _ = train_fine([toy_page(0)], Recipe(steps=5, batch=2), "cpu")
    assert _same(c, d)


def test_prefetch_same_weights():
    pages = [toy_page(0), toy_page(1, h=60, w=50)]
    for trainer in (train_coarse, train_fine):
        a, _ = trainer(pages, Recipe(steps=6, batch=2, prefetch=True), "cpu")
        b, _ = trainer(pages, Recipe(steps=6, batch=2, prefetch=False), "cpu")
        assert _same(a, b)
        c, _ = trainer(pages, Recipe(steps=6, batch=2, prefetch=True, seed=1), "cpu")
        assert not _same(a, c)


def _prefetchers():
    return [t for t in threading.enumerate() if t.name == "frame-prefetch"]


def test_prefetch_stops_cleanly():
    """The draws come in order; an error in the drawing thread is raised in the training's; an error in the
    training stops the drawing thread."""
    counter = iter(range(100))
    with _batches(lambda: next(counter), 10, True) as batches:
        assert list(batches) == list(range(10))
    assert not _prefetchers()

    def bad():
        raise ValueError("bad page")
    with pytest.raises(ValueError, match="bad page"):
        with _batches(bad, 10, True) as batches:
            list(batches)
    assert not _prefetchers()

    with pytest.raises(RuntimeError):
        with _batches(lambda: np.zeros(3), 1000, True) as batches:
            next(batches)
            raise RuntimeError("the step failed")
    assert not _prefetchers()


def test_validation_curve_and_returned_net(caplog):
    val = [toy_page(7)]
    with caplog.at_level(logging.INFO, logger="sfzmaps.frame.train"):
        net, info = train_coarse([toy_page(0)], Recipe(steps=4, batch=1, check_every=2), "cpu", val=val,
                                 label="fold 0 page inner 1")
    assert [s for s, _ in info["val_curve"]] == [2, 4]
    assert all(np.isfinite(v) and v > 0 for _, v in info["val_curve"])
    lines = [r.getMessage() for r in caplog.records if r.name == "sfzmaps.frame.train"]
    assert [m.split(": val ")[0] for m in lines] == ["fold 0 page inner 1 step 2/4", "fold 0 page inner 1 step 4/4"]
    assert [float(m.split(": val ")[1]) for m in lines] == pytest.approx([v for _, v in info["val_curve"]], abs=1e-4)
    assert not net.training and next(net.parameters()).device.type == "cpu"
    net, info = train_fine([toy_page(0)], Recipe(steps=4, batch=2, check_every=2), "cpu", val=val)
    assert [s for s, _ in info["val_curve"]] == [2, 4]
    assert not net.training and next(net.parameters()).device.type == "cpu"
    _, info = train_fine([toy_page(0)], Recipe(steps=2, batch=1), "cpu")
    assert info == {"val_curve": []}


def test_recipe_caps_at_4000_steps():
    assert Recipe().steps == 4000


def test_fine_loss_is_on_the_prior_logit_plus_the_correction(monkeypatch):
    """The pixel pass's training loss reads the summed logit: at the first step (head still zero) it is the prior's
    logit, clipped, exactly as prediction reads it."""
    seen = {}
    real_make, real_loss = train._fine_make, train._loss

    def make(*args):
        out = real_make(*args)
        seen.setdefault("x", out[0].clone())
        return out

    def loss(logits, target, weight):
        seen.setdefault("logits", logits.detach().clone())
        return real_loss(logits, target, weight)

    monkeypatch.setattr(train, "_fine_make", make)
    monkeypatch.setattr(train, "_loss", loss)
    train_fine([toy_page(0)], Recipe(steps=1, batch=2, prefetch=False), "cpu")
    p = seen["x"][:, 1:2].clamp(1e-4, 1 - 1e-4)
    assert (seen["x"][:, 1:2] == 0).any()                  # off the page the prior is 0, and its logit finite
    assert torch.isfinite(seen["logits"]).all()
    torch.testing.assert_close(seen["logits"], torch.log(p) - torch.log1p(-p))


def test_fine_weight_is_zero_in_the_uncertain_strip():
    """No px within UNSURE_PX of the label's outline (as scaled with the crop) carries weight; px beyond it do."""
    pages = [toy_page(0), toy_page(1)]
    _, t, w = _fine_batch(np.random.default_rng(7), pages, 16)
    near = UNSURE_PX * (1 - SCALE_JITTER) - 1.5          # the strip once a crop is scaled, less resampling
    far = UNSURE_PX * (1 + SCALE_JITTER) + 1.5
    weighed_far = 0
    for ti, wi in zip(t[:, 0], w[:, 0]):
        m = ti >= 0.5
        assert (wi[scores.unsure(m, near)] == 0).all()
        weighed_far += int(((wi > 0) & ~scores.unsure(m, far)).sum())
    assert weighed_far > 0


def test_coarse_weight_is_zero_in_the_strip_of_at_least_one_coarse_px():
    pg = toy_page(0)                                       # 96 x 128 on a 64 grid: 1/2, so the strip is 4 coarse px
    (_, target, weight), scale = _coarse_arrays(pg, 0.1)
    strip = scores.unsure(target >= 0.5, UNSURE_PX * scale)
    assert scale == pytest.approx(0.5) and strip.any()
    assert (weight[strip] == 0).all() and (weight[~strip] > 0).all()
    mask = np.zeros((96, 1024), np.uint8)                 # 1/16: 8 px would be half a coarse px; it is 1
    mask[20:76, 100:900] = 1
    wide = Page(id=("toy", "wide"), lum=np.ones(mask.shape, np.float32), mask=mask, prior=None)
    (_, target, weight), scale = _coarse_arrays(wide, 0.1)
    strip = scores.unsure(target >= 0.5, 1.0)
    assert scale == pytest.approx(1 / 16) and strip.any()
    assert (weight[strip] == 0).all() and (weight[~strip] > 0).all()


def test_validation_loss_ignores_the_strip():
    pg = toy_page(0)
    strip = scores.unsure(pg.mask)
    good = np.where(pg.mask == 1, 0.99, 0.01).astype(np.float32)
    bad = good.copy()
    bad[strip] = 1.0 - bad[strip]
    loss = _val_loss([pg])
    assert loss(lambda p: bad) == pytest.approx(loss(lambda p: good), abs=1e-12)
    bad[48, 64] = 0.01
    assert loss(lambda p: bad) > loss(lambda p: good)


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


def _fine_batch(rng, pages, batch, paper_weight=0.1):
    spec = _fine_draw(rng, pages, [_fine_aux(p) for p in pages], batch)
    return tuple(a.numpy() for a in _fine_make(spec, paper_weight, "cpu"))


def test_fine_pass_takes_pages_with_no_outline():
    pg = toy_page(0)
    full = Page(id=("toy", "full"), lum=pg.lum, mask=np.ones_like(pg.mask), prior=np.ones_like(pg.prior))
    small = Page(id=("toy", "small"), lum=pg.lum[:20, :24], mask=pg.mask[:20, :24], prior=pg.prior[:20, :24])
    net, _ = train_fine([full, small], Recipe(steps=3, batch=2), "cpu")
    assert all(torch.isfinite(v).all() for v in net.state_dict().values() if v.is_floating_point())
    # an outline crop of a page with no outline falls back to a uniform one: a whole-map crop, learned as map
    _, t, w = _fine_batch(np.random.default_rng(0), [full], 4)
    assert w.sum() > 0 and (t * w).sum() / w.sum() > 0.95


def test_fine_crops_have_an_unweighted_margin():
    tile, margin = predict.TILE, predict._margin()
    pages = [toy_page(0), toy_page(1, h=60, w=50)]
    x, t, w = _fine_batch(np.random.default_rng(3), pages, 6)
    full = tile + 2 * margin
    assert x.shape == (6, 2, full, full) and t.shape == w.shape == (6, 1, full, full)
    assert x.dtype == t.dtype == w.dtype == np.float32
    core = np.zeros((full, full), bool)
    core[margin:margin + tile, margin:margin + tile] = True
    assert (w[:, :, ~core] == 0).all()
    assert (w[:, :, core].reshape(6, -1).sum(1) > 0).all()


def test_fine_weight_is_exactly_zero_off_the_page():
    """Scaling a crop must not smear the page's weight onto the paper around it: on an all-ink page every px weighs
    paper_weight + 1 or more, so no weight lies strictly between 0 and paper_weight."""
    ink = Page(id=("toy", "ink"), lum=np.zeros((40, 30), np.float32), mask=np.ones((40, 30), np.uint8),
               prior=np.ones((40, 30), np.float32))
    rng = np.random.default_rng(5)
    for _ in range(5):
        _, _, w = _fine_batch(rng, [ink], 4, paper_weight=0.1)
        assert not ((w > 0) & (w < 0.1)).any()
        assert (w == 0).any() and (w > 0).any()


def test_coarse_batches_are_the_prediction_square():
    pages = [toy_page(0), toy_page(1, h=128, w=60)]
    coarse = [_coarse_arrays(pg, 0.1) for pg in pages]
    on_device = [tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in arrays) for arrays, _ in coarse]
    spec = _coarse_draw(np.random.default_rng(0), [s for _, s in coarse], 4)
    x, t, w = (a.numpy() for a in _coarse_make(spec, on_device, "cpu"))
    side = predict.COARSE_SIDE
    assert x.shape == t.shape == w.shape == (4, 1, side, side)
    for i, (k, aug) in enumerate(spec):
        h, wd = coarse[k][0][0].shape
        if aug.turns % 2:
            h, wd = wd, h
        pad = np.ones((side, side), bool)
        pad[:h, :wd] = False
        assert (x[i, 0][pad] == 1.0).all() and (t[i, 0][pad] == 0).all() and (w[i, 0][pad] == 0).all()
        assert (w[i, 0][~pad] > 0).any()
    arrays, scale = coarse[0]
    assert arrays[0].shape == (48, 64) and scale == pytest.approx(0.5)


def test_fine_pass_needs_a_prior():
    pg = toy_page(0)
    with pytest.raises(ValueError):
        train_fine([Page(id=pg.id, lum=pg.lum, mask=pg.mask, prior=None)], Recipe(steps=1, batch=1), "cpu")


def test_choose_steps():
    assert choose_steps([[(250, 0.5), (500, 0.3), (750, 0.4)], [(250, 0.6), (500, 0.2), (750, 0.2)]]) == 500
    assert choose_steps([[(250, 0.3), (500, 0.3)]]) == 250
    with pytest.raises(ValueError):
        choose_steps([[(250, 0.3), (500, float("nan"))], [(250, 0.4), (500, 0.1)]])
