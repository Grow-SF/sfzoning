import numpy as np
import torch

from sfzmaps.frame import model


def test_logits_have_the_page_shape(small_model):          # 300 x 457: not a multiple
    lum = np.random.default_rng(0).random((300, 457), dtype=np.float32)
    assert model.logits(model.new_net(), lum, "cpu").shape == (300, 457)


def test_pad_is_paper_and_a_multiple(small_model):
    p = model.pad(np.zeros((5, 9), np.float32))
    assert p.shape[0] % model.multiple() == 0 and p.shape[1] % model.multiple() == 0
    assert (p[:5, :9] == 0).all() and (p[5:, :] == model.PAPER).all() and (p[:, 9:] == model.PAPER).all()


def test_prob_is_the_mean_of_tempered_sigmoids(small_model):
    torch.manual_seed(0)
    a, b = model.new_net(), model.new_net()
    lum = np.full((40, 40), 0.7, np.float32)
    za, zb = model.logits(a, lum, "cpu"), model.logits(b, lum, "cpu")
    expect = (1 / (1 + np.exp(-za / 2)) + 1 / (1 + np.exp(-zb / 2))) / 2
    np.testing.assert_allclose(model.prob([a, b], lum, "cpu", temperature=2.0), expect, rtol=1e-5)


def test_logits_restore_train_mode(small_model):
    net = model.new_net().train()
    model.logits(net, np.ones((8, 8), np.float32), "cpu")
    assert net.training


def test_full_size_constants():
    assert model.multiple() == 256
