import numpy as np
import pytest
import torch

from sfzmaps.frame import predict
from sfzmaps.nets.unet import UNet, multiple, reach


def _tiny_nets(monkeypatch, tile=64, side=96):
    monkeypatch.setattr(predict, "TILE", tile)
    monkeypatch.setattr(predict, "COARSE_SIDE", side)
    monkeypatch.setattr(predict, "COARSE_LEVELS", 3)
    torch.manual_seed(0)
    coarse = UNet(1, 1, 4, 3)
    fine = UNet(2, 1, 4, predict.FINE_LEVELS)
    return coarse, fine


def test_predict_shape_matches_page(monkeypatch):
    coarse, fine = _tiny_nets(monkeypatch)
    lum = np.random.default_rng(0).random((517, 1283)).astype(np.float32)
    c, m = predict.predict_page(coarse, fine, lum, "cpu")
    for out in (c, m):
        assert out.shape == (517, 1283)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_tiling_has_no_seams(monkeypatch):
    _, fine = _tiny_nets(monkeypatch)
    rng = np.random.default_rng(1)
    lum = rng.random((150, 230)).astype(np.float32)
    prior = rng.random((150, 230)).astype(np.float32)
    got = predict.predict_tiles(fine, lum, prior, "cpu")
    want = predict._predict_padded_whole(fine, lum, prior, "cpu")
    assert got.shape == lum.shape
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_coarse_reach_covers_page():
    assert reach(predict.COARSE_LEVELS) >= predict.COARSE_SIDE


def test_coarse_input_pads_to_a_square():
    lum = np.zeros((3300, 5100), np.float32)
    for page in (lum, lum.T):
        arr, scale = predict.coarse_input(page)
        assert arr.shape == (predict.COARSE_SIDE, predict.COARSE_SIDE)
        assert predict.COARSE_SIDE % multiple(predict.COARSE_LEVELS) == 0
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
    _tiny_nets(monkeypatch)
    monkeypatch.setattr(predict, "_run", lambda net, x, device: x[0].copy())
    lum = np.ones((517, 1283), np.float32)
    lum[100:200, 900:1100] = 0.0
    net = UNet(2, 1, 4, predict.FINE_LEVELS)
    tiled = predict.predict_tiles(net, lum, lum, "cpu")
    np.testing.assert_array_equal(tiled, lum)
    coarse = predict.predict_coarse(UNet(1, 1, 4, 3), lum, "cpu")
    assert coarse.shape == lum.shape
    dark = coarse < 0.5
    rows, cols = np.where(dark)
    assert abs(rows.min() - 100) <= 2 and abs(rows.max() - 199) <= 2
    assert abs(cols.min() - 900) <= 3 and abs(cols.max() - 1099) <= 3
