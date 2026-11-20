import concurrent.futures
import itertools
import logging
import math
import threading

import numpy as np
import pytest
import torch
from scipy import ndimage

from sfzmaps.frame import predict, scores, train
from sfzmaps.frame.data import position_planes
from sfzmaps.frame.predict import Corrected, page_prior, predict_coarse, predict_tiles
from sfzmaps.frame.scores import UNSURE_PX
from sfzmaps.frame.train import (BLUR_SHARE, BLUR_SIGMA, BRIGHTNESS, CONTRAST, EDGE_CROP, GAMMA, OUTLINE_REACH,
                                 SCALE_JITTER, Page, Recipe, _batches, _coarse_arrays, _coarse_draw, _coarse_make,
                                 _draw_cut, _fine_draw, _fine_make, _near_mask, _near_outline, _page_priors,
                                 _prior_window, _Source, _val_loss, _window, _window_sure, apply_aug, draw_aug,
                                 train_joint)
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
    the rule."""
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
    return Page(id=("toy", str(seed)), lum=lum, mask=mask)


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


def test_joint_training_learns_the_toy_frame():
    """The realistic case: the toy's rule lies 1-3 px outside the label's outline, on the certain (frame) side.  Both
    passes trained together, read as the run reads them (the page pass's float16 prior, then the pixel pass): the rule
    is frame, the ink inside is map."""
    pages = [toy_page(s, h=128, w=96) if s % 2 else toy_page(s) for s in range(4)]   # mixed shapes in a batch
    (coarse, fine), info = train_joint(pages, Recipe(steps=600, batch=2, check_every=200), "cpu")
    assert isinstance(coarse, Folded) and isinstance(fine, Corrected)
    assert not coarse.training and not fine.training
    assert next(coarse.parameters()).device.type == next(fine.parameters()).device.type == "cpu"
    assert [s for s, _, _ in info["train_curve"]] == [200, 400, 600] and info["val_curve"] == []
    pg = toy_page(9)
    rule = _rule_px(pg)
    assert not (rule & scores.unsure(pg.mask)).any()
    p = predict_tiles(fine, pg.lum, page_prior(coarse, pg.lum))
    assert p[rule].mean() < 0.2 and p[_inside_ink_px(pg)].mean() > 0.8


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def _same_nets(a, b):
    return _same(a[0], b[0]) and _same(a[1], b[1])


def test_joint_training_is_deterministic_with_or_without_prefetch():
    pages = [toy_page(0), toy_page(1, h=60, w=50)]
    a, _ = train_joint(pages, Recipe(steps=6, batch=2, prefetch=True), "cpu")
    b, _ = train_joint(pages, Recipe(steps=6, batch=2, prefetch=True), "cpu")
    c, _ = train_joint(pages, Recipe(steps=6, batch=2, prefetch=False), "cpu")
    assert _same_nets(a, b) and _same_nets(a, c)
    d, _ = train_joint(pages, Recipe(steps=6, batch=2, seed=1), "cpu")
    assert not _same(a[0], d[0]) and not _same(a[1], d[1])


def test_cutting_on_threads_draws_the_same():
    """The cut pages and crops made on a thread pool are those made in turn, and the generator ends alike."""
    sources = [_Source(toy_page(0)), _Source(toy_page(1, h=60, w=50))]
    runs = []
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        for build in (map, pool.map):
            rng = np.random.default_rng(11)
            runs.append((_coarse_draw(rng, sources, 5, 0.1, build), _fine_draw(rng, sources, 5, build), rng.random()))
    (ca, fa, ra), (cb, fb, rb) = runs
    assert ra == rb
    for (xa, aug_a), (xb, aug_b) in zip(ca, cb, strict=True):
        assert aug_a == aug_b and all(np.array_equal(a, b) for a, b in zip(xa, xb, strict=True))
    for a, b in zip(fa, fb, strict=True):
        assert (a.key, a.top, a.left, a.aug) == (b.key, b.top, b.left, b.aug)
        assert np.array_equal(a.arrays, b.arrays) and np.array_equal(a.small, b.small)


def test_validation_is_logged_and_changes_nothing(caplog):
    """The validation curve is a log: the same weights with and without it."""
    pages, val = [toy_page(0)], [toy_page(7)]
    recipe = Recipe(steps=4, batch=2, check_every=2)
    plain, info = train_joint(pages, recipe, "cpu")
    assert info["val_curve"] == []
    with caplog.at_level(logging.INFO, logger="sfzmaps.frame.train"):
        nets, info = train_joint(pages, recipe, "cpu", val=val, label="fold 0")
    assert _same_nets(plain, nets)
    assert [s for s, _ in info["val_curve"]] == [2, 4]
    assert all(np.isfinite(v) and v > 0 for _, v in info["val_curve"])
    assert [s for s, _, _ in info["train_curve"]] == [2, 4]
    assert all(np.isfinite(v) and v > 0 for _, a, b in info["train_curve"] for v in (a, b))
    lines = [r.getMessage() for r in caplog.records if r.name == "sfzmaps.frame.train"]
    assert [m.split(":")[0] for m in lines] == ["fold 0 step 2/4", "fold 0 step 4/4"]
    assert [float(m.split(" val ")[1]) for m in lines] == pytest.approx([v for _, v in info["val_curve"]], abs=1e-4)


def test_recipe():
    assert (Recipe().steps, Recipe().check_every) == (2000, 250)


def test_cosine_schedule(monkeypatch):
    """The learning rate is `recipe.lr` at step 1, falls along a half cosine, and is exactly 0 after the last step
    (the last step itself still moves the weights)."""
    used, opts = [], []

    class Spy(torch.optim.Adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opts.append(self)

        def step(self, *args, **kwargs):
            used.append(self.param_groups[0]["lr"])
            return super().step(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", Spy)
    recipe = Recipe(steps=8, batch=1, check_every=4, lr=1e-3)
    train_joint([toy_page(0)], recipe, "cpu")
    assert len(opts) == 1 and len(used) == 8
    assert used[0] == recipe.lr
    assert used == pytest.approx([recipe.lr * 0.5 * (1 + math.cos(math.pi * s / 8)) for s in range(8)], abs=1e-12)
    assert used[-1] > 0 and opts[0].param_groups[0]["lr"] == 0.0


def test_pixel_loss_is_on_the_prior_logit_plus_the_correction(monkeypatch):
    """The pixel pass's training loss reads the summed logit: at the first step (head still zero) it is the prior's
    logit, clipped, exactly as prediction reads it."""
    seen, logits = {}, []
    real_make, real_loss = train._fine_make, train._loss

    def make(*args):
        out = real_make(*args)
        seen.setdefault("x", out[0].clone())
        return out

    def loss(lg, target, weight):
        logits.append(lg.detach().clone())
        return real_loss(lg, target, weight)

    monkeypatch.setattr(train, "_fine_make", make)
    monkeypatch.setattr(train, "_loss", loss)
    train_joint([toy_page(0)], Recipe(steps=1, batch=2, prefetch=False), "cpu")
    assert len(logits) == 2                                 # the page pass's, then the pixel pass's
    p = seen["x"][:, 1:2].clamp(1e-4, 1 - 1e-4)
    assert (seen["x"][:, 1:2] == 0).any()                  # off the page the prior is 0, and its logit finite
    assert torch.isfinite(logits[1]).all()
    torch.testing.assert_close(logits[1], torch.log(p) - torch.log1p(-p))


def test_each_pass_is_trained_by_its_own_loss(monkeypatch):
    """The prior enters the pixel pass with no gradient: the pixel loss reaches none of the page pass's weights."""
    nets, losses = [], []
    real_new, real_loss = predict.new_coarse_net, train._loss

    def new():
        nets.append(real_new())
        return nets[-1]

    def loss(lg, target, weight):
        out = real_loss(lg, target, weight)
        losses.append(out)
        if len(losses) == 2:
            grads = torch.autograd.grad(out, list(nets[0].parameters()), allow_unused=True, retain_graph=True)
            assert out.requires_grad and all(g is None for g in grads)
        return out

    monkeypatch.setattr(predict, "new_coarse_net", new)
    monkeypatch.setattr(train, "_loss", loss)
    train_joint([toy_page(0)], Recipe(steps=1, batch=2, prefetch=False), "cpu")
    assert len(losses) == 2 and len(nets) == 1


def _wake(net):
    """Random running statistics, so the eval-mode net differs from the train-mode one."""
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)


def test_the_prior_is_the_page_pass_now_on_the_plain_page():
    """A crop's prior is the page pass's eval-mode probability for the unaugmented cut page, as `predict_coarse`
    gives it for that page (to float rounding; then float16, as the run saves and reads it), cut to the crop's window,
    0 off the page; one per cut page (keyed by page and cut), so two cuts of one page get their own; the page pass is
    left in training mode."""
    torch.manual_seed(0)
    net = predict.new_coarse_net()
    _wake(net)
    pages = [toy_page(0), toy_page(1, h=60, w=50)]          # the page grid smaller, then larger, than the page
    sources = [_Source(pg) for pg in pages]
    cuts = {(0, (0, 0, 128, 96)): pages[0].lum, (0, (9, 4, 120, 90)): pages[0].lum[4:90, 9:120],
            (1, (0, 0, 50, 60)): pages[1].lum, (1, (2, 5, 47, 56)): pages[1].lum[5:56, 2:47]}
    smalls = {key: predict.coarse_resize(sources[key[0]].lum_img, key[1]) for key in cuts}
    net.train()
    priors = _page_priors(net, smalls, "cpu")
    assert net.training and set(priors) == set(cuts)
    assert not torch.equal(priors[(0, (0, 0, 128, 96))][:10, :10], priors[(0, (9, 4, 120, 90))][:10, :10])
    for key, lum in cuts.items():
        full = predict_coarse(net, lum)
        h, w = lum.shape
        assert tuple(priors[key].shape) == smalls[key].shape
        for top, left, side in ((0, 0, h), (-7, 5, 40), (h - 20, w - 30, 45), (3, -9, 70)):
            got = _prior_window(priors[key], (h, w), top, left, side).numpy()
            assert got.dtype == np.float32 and got.shape == (side, side)
            want = np.zeros((side, side), np.float32)
            r0, c0 = max(top, 0), max(left, 0)
            r1, c1 = min(top + side, h), min(left + side, w)
            want[r0 - top:r1 - top, c0 - left:c1 - left] = full[r0:r1, c0:c1]
            np.testing.assert_allclose(got, want, rtol=2e-3, atol=1e-6)
            assert np.array_equal(got, got.astype(np.float16).astype(np.float32))
            assert (got[want == 0] == 0).all() and (got[want > 0] > 0).all()


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
    (_, _, target, weight), scale = _coarse_arrays(_Source(pg), 0.1)
    strip = scores.unsure(target >= 0.5, UNSURE_PX * scale)
    assert scale == pytest.approx(0.5) and strip.any()
    assert (weight[strip] == 0).all() and (weight[~strip] > 0).all()
    mask = np.zeros((96, 1024), np.uint8)                 # 1/16: 8 px would be half a coarse px; it is 1
    mask[20:76, 100:900] = 1
    wide = Page(id=("toy", "wide"), lum=np.ones(mask.shape, np.float32), mask=mask)
    (_, _, target, weight), scale = _coarse_arrays(_Source(wide), 0.1)
    strip = scores.unsure(target >= 0.5, 1.0)
    assert scale == pytest.approx(1 / 16) and strip.any()
    assert (weight[strip] == 0).all() and (weight[~strip] > 0).all()


def _holed_page():
    """A 64 x 64 page (1 coarse px per page px on the test grid), all ink, map rows / columns 10..53 with a hole (a
    floating block) at 25..38."""
    mask = np.zeros((64, 64), np.uint8)
    mask[10:54, 10:54] = 1
    mask[25:39, 25:39] = 0
    return Page(id=("toy", "holed"), lum=np.zeros(mask.shape, np.float32), mask=mask)


# (row, col) of: 3 px outside the outline (a rule's place), 3 px inside it, the block side of the hole's outline and
# 3 px from it, the map side of the hole's outline (1 px and 3 px)
RULE_SIDE, MAP_SIDE = (30, 7), (30, 12)
BLOCK_SIDE, MAP_BY_HOLE = [(30, 25), (30, 27)], [(30, 24), (30, 22)]


def test_weights_follow_the_one_sided_strip():
    """Ruling R8: the frame side of every outline is certain (the labels are drawn inside the rule, holes on paper
    outside each block); only the map side is uncertain.  Both passes."""
    pg = _holed_page()
    (_, _, _, weight), scale = _coarse_arrays(_Source(pg), 0.1)
    assert scale == pytest.approx(1.0)
    sure = _window_sure(pg.mask, 0, 0, 64) > 0
    for r, c in [RULE_SIDE] + BLOCK_SIDE:
        assert weight[r, c] > 0 and sure[r, c], (r, c)
    for r, c in [MAP_SIDE] + MAP_BY_HOLE:
        assert weight[r, c] == 0 and not sure[r, c], (r, c)


def test_validation_loss_ignores_the_strip():
    pg = toy_page(0)
    strip = scores.unsure(pg.mask)
    good = np.where(pg.mask == 1, 0.99, 0.01).astype(np.float32)
    bad = good.copy()
    bad[strip] = 1.0 - bad[strip]
    loss = _val_loss([pg])
    assert loss(lambda i: bad) == pytest.approx(loss(lambda i: good), abs=1e-12)
    bad[48, 64] = 0.01
    assert loss(lambda i: bad) > loss(lambda i: good)


def test_outline_band():
    pg = toy_page(0)
    assert _near_outline(np.ones_like(pg.mask)).size == 0 and _near_outline(np.zeros_like(pg.mask)).size == 0
    assert not _near_mask(np.ones_like(pg.mask)).any()
    big = np.zeros((700, 700), np.uint8)
    big[300:400, 300:400] = 1
    near = np.zeros(big.size, bool)
    near[_near_outline(big)] = True
    near = near.reshape(big.shape)
    edge = big.astype(bool) & ~ndimage.binary_erosion(big.astype(bool), border_value=1)
    d = ndimage.distance_transform_edt(~edge)
    assert np.array_equal(near, d <= OUTLINE_REACH) and not near.all()
    assert np.array_equal(_near_mask(big), near)


def _fine_batch(rng, pages, batch, paper_weight=0.1):
    """A pixel-pass batch whose priors are 1/2 on every page's coarse grid."""
    spec = _fine_draw(rng, [_Source(p) for p in pages], batch)
    priors = {c.key: torch.full(c.small.shape, 0.5) for c in spec}
    return tuple(a.numpy() for a in _fine_make(spec, priors, paper_weight, "cpu"))


def test_fine_pass_takes_pages_with_no_outline():
    pg = toy_page(0)
    full = Page(id=("toy", "full"), lum=pg.lum, mask=np.ones_like(pg.mask))
    small = Page(id=("toy", "small"), lum=pg.lum[:20, :24], mask=pg.mask[:20, :24])
    nets, _ = train_joint([full, small], Recipe(steps=3, batch=2), "cpu")
    for net in nets:
        assert all(torch.isfinite(v).all() for v in net.state_dict().values() if v.is_floating_point())
    # an outline crop of a page with no outline falls back to a uniform one: a whole-map crop, learned as map
    _, t, w = _fine_batch(np.random.default_rng(0), [full], 4)
    assert w.sum() > 0 and (t * w).sum() / w.sum() > 0.95


def test_fine_crops_have_an_unweighted_margin():
    tile, margin = predict.TILE, predict._margin()
    pages = [toy_page(0), toy_page(1, h=60, w=50)]
    x, t, w = _fine_batch(np.random.default_rng(3), pages, 6)
    full = tile + 2 * margin
    assert x.shape == (6, 4, full, full) and t.shape == w.shape == (6, 1, full, full)
    assert x.dtype == t.dtype == w.dtype == np.float32
    core = np.zeros((full, full), bool)
    core[margin:margin + tile, margin:margin + tile] = True
    assert (w[:, :, ~core] == 0).all()
    assert (w[:, :, core].reshape(6, -1).sum(1) > 0).all()


def test_fine_weight_is_exactly_zero_off_the_page():
    """Scaling a crop must not smear the page's weight onto the paper around it: on an all-ink page every px weighs
    paper_weight + 1 or more, so no weight lies strictly between 0 and paper_weight."""
    ink = Page(id=("toy", "ink"), lum=np.zeros((40, 30), np.float32), mask=np.ones((40, 30), np.uint8))
    rng = np.random.default_rng(5)
    for _ in range(5):
        _, _, w = _fine_batch(rng, [ink], 4, paper_weight=0.1)
        assert not ((w > 0) & (w < 0.1)).any()
        assert (w == 0).any() and (w > 0).any()


def test_coarse_batches_are_the_prediction_square():
    """[lum, valid, edge] on the COARSE_SIDE square as `coarse_input` pads it: paper in lum, 0 in the planes, target
    and weight around the page; on the page the planes are the moved page's own (they turn and flip with it)."""
    pages = [toy_page(0), toy_page(1, h=128, w=60)]
    spec = _coarse_draw(np.random.default_rng(0), [_Source(pg) for pg in pages], 6, 0.1)
    x, t, w = (a.numpy() for a in _coarse_make(spec, "cpu"))
    side = predict.COARSE_SIDE
    assert x.shape == (6, 3, side, side) and t.shape == w.shape == (6, 1, side, side)
    turned = 0
    for i, (arrays, aug) in enumerate(spec):
        h, wd = arrays[0].shape
        if aug.turns % 2:
            h, wd = wd, h
            turned += h != wd
        pad = np.ones((side, side), bool)
        pad[:h, :wd] = False
        assert (x[i, 0][pad] == 1.0).all() and (x[i, 1:][:, pad] == 0).all()
        assert (t[i, 0][pad] == 0).all() and (w[i, 0][pad] == 0).all()
        assert np.array_equal(x[i, 1:, :h, :wd], np.stack(position_planes((h, wd))))
        assert (w[i, 0][~pad] > 0).any()
    assert turned
    arrays, scale = _coarse_arrays(_Source(pages[0]), 0.1)
    assert arrays[0].shape == (48, 64) and scale == pytest.approx(0.5)


def test_edge_cut_is_up_to_a_tenth_per_side():
    """Each side is cut by its own random 0 .. EDGE_CROP of the page's size that way; a box (left, top, right,
    bottom)."""
    assert EDGE_CROP == 0.1
    rng = np.random.default_rng(0)
    h, w = 200, 300
    cuts = np.array([_draw_cut(rng, (h, w)) for _ in range(400)])
    margins = np.stack([cuts[:, 0], cuts[:, 1], w - cuts[:, 2], h - cuts[:, 3]], 1)     # left, top, right, bottom
    limits = np.array([w, h, w, h]) * EDGE_CROP
    assert (margins >= 0).all() and (margins <= limits).all()
    assert (margins.max(0) > 0.9 * limits).all() and (margins.min(0) < 0.1 * limits).all()
    assert abs(np.corrcoef(margins[:, 0], margins[:, 2])[0, 1]) < 0.2         # independent per side
    assert _draw_cut(np.random.default_rng(1), (h, w)) == _draw_cut(np.random.default_rng(1), (h, w))


def test_page_pass_reads_the_cut_page():
    """A cut page's page-pass arrays are prediction's reading of that page alone (`coarse_input`): its luminance,
    its own position planes, its mask resized alike as the target, weight 0 on the target's strip."""
    pg = toy_page(0)
    box = (9, 4, 120, 90)
    lum, mask = pg.lum[4:90, 9:120], pg.mask[4:90, 9:120]
    (got_lum, planes, target, weight), scale = _coarse_arrays(_Source(pg), 0.1, box)
    arr, want_scale = predict.coarse_input(np.ascontiguousarray(lum))
    nh, nw = got_lum.shape
    assert scale == want_scale and (nh, nw) == (round(86 * scale), round(111 * scale)) and nw == 64
    assert np.array_equal(got_lum, arr[0, :nh, :nw]) and np.array_equal(planes, arr[1:, :nh, :nw])
    assert np.array_equal(target, predict.coarse_input(mask.astype(np.float32))[0][0, :nh, :nw])
    strip = scores.unsure(target >= 0.5, max(1.0, UNSURE_PX * scale))
    assert (weight[strip] == 0).all() and (weight[~strip] > 0).all()


def test_window_sure_is_the_cut_pages_strip():
    """The sure px of a window: `~scores.unsure` of the whole (cut) mask, cut to the window, 0 off the page; found
    from the window and a few px around it."""
    rng = np.random.default_rng(2)
    for trial in range(6):
        mask = np.zeros((80, 70), np.uint8)
        for _ in range(3):
            r, c = rng.integers(0, 60, 2)
            mask[r:r + int(rng.integers(5, 40)), c:c + int(rng.integers(5, 40))] = 1
        if trial == 5:
            mask[:] = 1                                      # no outline: no strip
        whole = (~scores.unsure(mask)).astype(np.float32)
        for top, left, side in ((0, 0, 80), (-10, 20, 40), (50, 50, 45), (3, -12, 30), (30, 30, 9)):
            got = _window_sure(mask, top, left, side)
            assert got.dtype == np.float32 and np.array_equal(got, _window(whole, top, left, side, 0.0))


def _cut(pg, box):
    left, top, right, bottom = box
    return pg.lum[top:bottom, left:right], pg.mask[top:bottom, left:right]


def test_fine_crops_read_the_cut_page(monkeypatch):
    """Every crop is cut from its own randomly cut page: its windows (luminance, mask, sure px, distance to the cut
    page's edge) and its prior's page are of that page; the batch's input is [lum, prior, valid, edge], the planes
    moved with the crop.  (Unscaled crops, so the windows can be compared exactly.)"""
    monkeypatch.setattr(train, "SCALE_JITTER", 0.0)
    pages = [toy_page(0), toy_page(1, h=60, w=50)]
    sources = [_Source(pg) for pg in pages]
    spec = _fine_draw(np.random.default_rng(4), sources, 8)
    full = predict.TILE + 2 * predict._margin()
    boxes = {c.box for c in spec}
    assert len(boxes) == 8 and all(c.key == (c.page, c.box) for c in spec)
    for c in spec:
        lum, mask = _cut(pages[c.page], c.box)
        assert c.shape == lum.shape and c.arrays.shape == (4, full, full)
        assert np.array_equal(c.arrays[0], _window(lum, c.top, c.left, full, 1.0))
        assert np.array_equal(c.arrays[1], _window(mask, c.top, c.left, full, 0.0))
        sure = _window((~scores.unsure(mask)).astype(np.float32), c.top, c.left, full, 0.0)
        assert np.array_equal(c.arrays[2], sure)
        valid, edge = position_planes(lum.shape, (c.top, c.left, full, full))
        assert np.array_equal(c.arrays[3], edge)
        assert np.array_equal(c.small, predict.coarse_input(np.ascontiguousarray(lum))[0][0][:c.small.shape[0],
                                                                                            :c.small.shape[1]])
    priors = {c.key: torch.full(c.small.shape, 0.5) for c in spec}
    x = _fine_make(spec, priors, 0.1, "cpu")[0].numpy()
    assert x.shape == (8, 4, full, full)
    for i, c in enumerate(spec):
        lum, _ = _cut(pages[c.page], c.box)
        planes = np.stack(position_planes(lum.shape, (c.top, c.left, full, full)))
        moved = np.rot90(np.flip(planes, 2) if c.aug.flip else planes, c.aug.turns, (1, 2))
        assert np.array_equal(x[i, 2:], moved)
        assert (x[i, 1][moved[0] == 0] == 0).all()                            # the prior: 0 off the page


def test_outline_crops_centre_near_the_outline_on_the_cut_page():
    pg = toy_page(0)
    near = _near_mask(pg.mask)
    spec = _fine_draw(np.random.default_rng(6), [_Source(pg)], 40)
    for i, c in enumerate(spec):
        side = c.arrays.shape[-1]
        cy, cx = c.top + side // 2, c.left + side // 2
        assert 0 <= cy < c.shape[0] and 0 <= cx < c.shape[1]
        if i % 2 == 0:
            assert near[cy + c.box[1], cx + c.box[0]]


def test_priors_are_the_page_pass_on_each_crops_cut_page(monkeypatch):
    """In training, each crop's prior comes from its own cut page (keyed by page and cut, not by sheet)."""
    seen = []
    real = train._fine_make

    def make(spec, priors, *args):
        seen.append((spec, priors))
        return real(spec, priors, *args)

    monkeypatch.setattr(train, "_fine_make", make)
    train_joint([toy_page(0)], Recipe(steps=2, batch=4, prefetch=False), "cpu")
    assert len(seen) == 2
    for spec, priors in seen:
        assert set(priors) == {c.key for c in spec} and len(priors) == 4       # one sheet, four cut pages
        for c in spec:
            assert tuple(priors[c.key].shape) == c.small.shape


def test_training_needs_labels():
    pg = toy_page(0)
    with pytest.raises(ValueError):
        train_joint([Page(id=pg.id, lum=pg.lum, mask=None)], Recipe(steps=1, batch=1), "cpu")
    with pytest.raises(ValueError):
        train_joint([pg], Recipe(steps=1, batch=1), "cpu", val=[Page(id=("toy", "v"), lum=pg.lum, mask=None)])
