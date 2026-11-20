import numpy as np
import pytest
import torch

from sfzmaps.frame import predict
from sfzmaps.frame.data import position_planes
from sfzmaps.nets.unet import Folded, folded_multiple, folded_reach


def _tiny_nets(monkeypatch, tile=64, side=96):
    """Folded nets at the real folds, with fewer levels and channels, so they run in seconds on cpu."""
    monkeypatch.setattr(predict, "TILE", tile)
    monkeypatch.setattr(predict, "COARSE_SIDE", side)
    monkeypatch.setattr(predict, "COARSE_LEVELS", 2)
    monkeypatch.setattr(predict, "COARSE_WIDTH", 4)
    monkeypatch.setattr(predict, "FINE_LEVELS", 3)
    monkeypatch.setattr(predict, "FINE_WIDTH", 4)
    torch.manual_seed(0)
    return predict.new_coarse_net(), predict.new_fine_net()


def test_real_sizes():
    """The real constants: folds of 4, the page pass sees the whole 512 px square (its reach is 1000), the tile margin
    is 256 (232 rounded up to the pooling grid of 32) and every size sits on its net's grid."""
    p = predict
    assert (p.COARSE_FOLD, p.COARSE_LEVELS, p.COARSE_WIDTH, p.COARSE_SIDE) == (4, 6, 16, 512)
    assert (p.FINE_FOLD, p.FINE_LEVELS, p.FINE_WIDTH, p.TILE) == (4, 4, 32, 256)
    assert folded_reach(p.COARSE_FOLD, p.COARSE_LEVELS) == 1000 >= p.COARSE_SIDE
    assert p.COARSE_SIDE % folded_multiple(p.COARSE_FOLD, p.COARSE_LEVELS) == 0
    assert folded_reach(p.FINE_FOLD, p.FINE_LEVELS) == 232 and p._margin() == 256
    assert p.TILE % folded_multiple(p.FINE_FOLD, p.FINE_LEVELS) == 0
    assert isinstance(p.new_coarse_net(), Folded) and isinstance(p.new_fine_net(), Folded)


def test_nets_read_the_position_planes():
    """The page pass reads [lum, valid, edge], the pixel pass [lum, prior, valid, edge] (Task 14), one output each."""
    for net, ch in ((predict.new_coarse_net(), 3), (predict.new_fine_net(), 4)):
        first, head = net.net.down[0][0], net.net.head
        assert first.in_channels == ch * net.fold ** 2 and head.out_channels == net.fold ** 2


def test_predict_shape_matches_page(monkeypatch):
    coarse, fine = _tiny_nets(monkeypatch, tile=128)
    lum = np.random.default_rng(0).random((517, 1283)).astype(np.float32)
    c, m = predict.predict_page(coarse, fine, lum, "cpu")
    for out in (c, m):
        assert out.shape == (517, 1283)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0


def _wake_head(net):
    """A random head, so the pixel pass's correction is not zero and the tiling is really tested."""
    torch.manual_seed(3)
    torch.nn.init.normal_(net.net.head.weight, std=0.5)
    torch.nn.init.normal_(net.net.head.bias, std=0.5)


def test_fine_net_starts_as_the_page_pass():
    """The pixel pass's head convolution starts at zero weights and bias (at the real sizes): an untrained net adds
    nothing to the prior's logit."""
    fine = predict.new_fine_net()
    assert isinstance(fine, Folded)
    head = fine.net.head
    assert torch.count_nonzero(head.weight) == 0 and torch.count_nonzero(head.bias) == 0
    assert any(torch.count_nonzero(p) for n, p in fine.named_parameters() if not n.startswith("net.head"))


def test_zero_head_returns_the_prior(monkeypatch):
    """Untrained, the pixel pass returns the page pass's answer: sigmoid(logit(clip(prior, 1e-4, 1 - 1e-4)))."""
    _, fine = _tiny_nets(monkeypatch)
    rng = np.random.default_rng(4)
    lum = rng.random((150, 230)).astype(np.float32)
    prior = rng.random((150, 230)).astype(np.float32)
    prior[:10] = 0.0                                   # clipped: 0 and 1 stay finite
    prior[-10:] = 1.0
    got = predict.predict_tiles(fine, lum, prior, "cpu")
    want = np.clip(prior, predict.PRIOR_CLIP, 1 - predict.PRIOR_CLIP)
    assert predict.PRIOR_CLIP == 1e-4
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_output_is_prior_logit_plus_correction(monkeypatch):
    _, fine = _tiny_nets(monkeypatch)
    _wake_head(fine)
    fine.eval()
    m = folded_multiple(predict.FINE_FOLD, predict.FINE_LEVELS)
    x = torch.rand((1, 4, 2 * m, 2 * m))
    x[0, 1, :3] = 0.0
    with torch.no_grad():
        inner = torch.nn.functional.pixel_shuffle(
            fine.net(torch.nn.functional.pixel_unshuffle(x, fine.fold)), fine.fold)
        p = x[:, 1:2].clamp(1e-4, 1 - 1e-4)
        torch.testing.assert_close(fine(x), torch.log(p) - torch.log1p(-p) + inner)


def _reference(net, lum, prior):
    """The run the tiling must equal (rulings R1/R5), built here from the definition: one run of the net on
    [lum, prior, valid, edge] of the whole page, padded by the tile margin on every side and then at right and bottom
    to the pixel pass's grid, with paper (1.0) in lum and 0 in the prior and both position planes; cropped back."""
    h, w = lum.shape
    m, mg = folded_multiple(predict.FINE_FOLD, predict.FINE_LEVELS), predict._margin()
    ph, pw = -(-(h + 2 * mg) // m) * m, -(-(w + 2 * mg) // m) * m
    x = np.zeros((4, ph, pw), np.float32)
    x[0] = 1.0
    x[0, mg:mg + h, mg:mg + w] = lum
    x[1, mg:mg + h, mg:mg + w] = prior
    x[2:, mg:mg + h, mg:mg + w] = np.stack(position_planes((h, w)))
    net.eval()
    with torch.no_grad():
        out = torch.sigmoid(net(torch.from_numpy(x)[None]))[0, 0].numpy()
    return out[mg:mg + h, mg:mg + w]


def test_tiling_has_no_seams(monkeypatch):
    """With four input channels: the tiles (training's size and prediction's) equal one run on the whole page."""
    _, fine = _tiny_nets(monkeypatch)
    assert isinstance(fine, Folded)
    _wake_head(fine)
    rng = np.random.default_rng(1)
    lum = rng.random((150, 230)).astype(np.float32)
    prior = rng.random((150, 230)).astype(np.float32)
    want = _reference(fine, lum, prior)
    for tile in (predict.TILE, 2 * predict.TILE):
        got = predict.predict_tiles(fine, lum, prior, "cpu", tile=tile)
        assert got.shape == lum.shape
        np.testing.assert_allclose(got, want, atol=1e-5)
    np.testing.assert_allclose(predict._predict_padded_whole(fine, lum, prior, "cpu"), want, atol=1e-5)


def test_near_computes_only_the_tiles_it_touches(monkeypatch):
    """With `near`, a tile whose core holds a near px is the full prediction's, bit for bit; every other px is the
    prior's (as float32); no near px at all is the prior throughout."""
    _, fine = _tiny_nets(monkeypatch)
    _wake_head(fine)
    rng = np.random.default_rng(5)
    lum = rng.random((150, 230)).astype(np.float32)
    prior = rng.random((150, 230)).astype(np.float16)
    near = np.zeros(lum.shape, bool)
    near[70, 140] = True                                   # in the tile at rows 64..127, columns 128..191
    full = predict.predict_tiles(fine, lum, prior, "cpu")
    got = predict.predict_tiles(fine, lum, prior, "cpu", near=near)
    assert got.dtype == np.float32 and got.shape == lum.shape
    done = np.zeros(lum.shape, bool)
    done[64:128, 128:192] = True
    np.testing.assert_array_equal(got[done], full[done])
    np.testing.assert_array_equal(got[~done], prior[~done].astype(np.float32))
    assert not np.array_equal(got, full)
    none = predict.predict_tiles(fine, lum, prior, "cpu", near=np.zeros(lum.shape, bool))
    np.testing.assert_array_equal(none, prior.astype(np.float32))


def test_coarse_input_pads_to_a_square():
    """[lum, valid, edge] on the COARSE_SIDE square: the resized page at top left, paper (1.0) in lum and 0 in the
    position planes around it; the planes are the resized page's own (`position_planes` of its shape)."""
    lum = np.zeros((3300, 5100), np.float32)
    for page in (lum, lum.T):
        arr, scale = predict.coarse_input(page)
        assert arr.shape == (3, predict.COARSE_SIDE, predict.COARSE_SIDE) and arr.dtype == np.float32
        assert scale == pytest.approx(predict.COARSE_SIDE / 5100, rel=1e-3)
        nh, nw = round(page.shape[0] * scale), round(page.shape[1] * scale)
        assert (arr[0, :nh, :nw] == 0).all() and (arr[0, nh:] == 1).all() and (arr[0, :, nw:] == 1).all()
        assert np.array_equal(arr[1:, :nh, :nw], np.stack(position_planes((nh, nw))))
        assert (arr[1:, nh:] == 0).all() and (arr[1:, :, nw:] == 0).all()


def test_predict_restores_training_mode(monkeypatch):
    coarse, fine = _tiny_nets(monkeypatch)
    lum = np.random.default_rng(2).random((100, 130)).astype(np.float32)
    coarse.train()
    fine.train()
    predict.predict_coarse(coarse, lum, "cpu")
    predict.predict_tiles(fine, lum, lum, "cpu")
    assert coarse.training and fine.training


def test_predict_keeps_pixels_aligned(monkeypatch):
    coarse_net, fine_net = _tiny_nets(monkeypatch)
    monkeypatch.setattr(predict, "_run", lambda net, x, device: x[0].copy())
    lum = np.ones((517, 1283), np.float32)
    lum[100:200, 900:1100] = 0.0
    tiled = predict.predict_tiles(fine_net, lum, lum, "cpu")
    np.testing.assert_array_equal(tiled, lum)
    coarse = predict.predict_coarse(coarse_net, lum, "cpu")
    assert coarse.shape == lum.shape
    dark = coarse < 0.5
    rows, cols = np.where(dark)
    assert abs(rows.min() - 100) <= 2 and abs(rows.max() - 199) <= 2
    assert abs(cols.min() - 900) <= 3 and abs(cols.max() - 1099) <= 3
