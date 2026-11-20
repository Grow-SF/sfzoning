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


def test_coarse_input_pads_to_multiple():
    lum = np.ones((3300, 5100), np.float32)
    arr, scale = predict.coarse_input(lum)
    m = multiple(predict.COARSE_LEVELS)
    assert max(arr.shape) % m == 0 and min(arr.shape) % m == 0
    assert min(arr.shape) >= 1024 or arr.shape[0] >= 1024 or arr.shape[1] >= 1024
    assert arr.shape[1] >= 1024
    assert scale == pytest.approx(1024 / 5100, rel=1e-3)
