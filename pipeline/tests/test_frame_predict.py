import numpy as np
import pytest
import torch

from sfzmaps.frame import predict
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
    """The real constants: folds of 4, the page pass sees the whole page, the tile margin is 256 (232 rounded up to
    the pooling grid of 32) and every size sits on its net's grid."""
    p = predict
    assert (p.COARSE_FOLD, p.COARSE_LEVELS, p.COARSE_WIDTH, p.COARSE_SIDE) == (4, 7, 16, 1024)
    assert (p.FINE_FOLD, p.FINE_LEVELS, p.FINE_WIDTH, p.TILE) == (4, 4, 32, 256)
    assert folded_reach(p.COARSE_FOLD, p.COARSE_LEVELS) >= p.COARSE_SIDE
    assert p.COARSE_SIDE % folded_multiple(p.COARSE_FOLD, p.COARSE_LEVELS) == 0
    assert folded_reach(p.FINE_FOLD, p.FINE_LEVELS) == 232 and p._margin() == 256
    assert p.TILE % folded_multiple(p.FINE_FOLD, p.FINE_LEVELS) == 0
    assert isinstance(p.new_coarse_net(), Folded) and isinstance(p.new_fine_net(), Folded)


def test_predict_shape_matches_page(monkeypatch):
    coarse, fine = _tiny_nets(monkeypatch, tile=128)
    lum = np.random.default_rng(0).random((517, 1283)).astype(np.float32)
    c, m = predict.predict_page(coarse, fine, lum, "cpu")
    for out in (c, m):
        assert out.shape == (517, 1283)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_tiling_has_no_seams(monkeypatch):
    _, fine = _tiny_nets(monkeypatch)
    assert isinstance(fine, Folded)
    rng = np.random.default_rng(1)
    lum = rng.random((150, 230)).astype(np.float32)
    prior = rng.random((150, 230)).astype(np.float32)
    got = predict.predict_tiles(fine, lum, prior, "cpu")
    want = predict._predict_padded_whole(fine, lum, prior, "cpu")
    assert got.shape == lum.shape
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_coarse_input_pads_to_a_square():
    lum = np.zeros((3300, 5100), np.float32)
    for page in (lum, lum.T):
        arr, scale = predict.coarse_input(page)
        assert arr.shape == (predict.COARSE_SIDE, predict.COARSE_SIDE)
        assert scale == pytest.approx(1024 / 5100, rel=1e-3)
        nh, nw = round(page.shape[0] * scale), round(page.shape[1] * scale)
        assert (arr[:nh, :nw] == 0).all() and (arr[nh:] == 1).all() and (arr[:, nw:] == 1).all()


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
