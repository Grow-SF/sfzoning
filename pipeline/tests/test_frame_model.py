import numpy as np
import torch

from sfzmaps.frame import model


def test_logits_have_a_plane_per_class_and_the_page_shape(small_model):      # 300 x 457: not a multiple
    lum = np.random.default_rng(0).random((300, 457), dtype=np.float32)
    z = model.logits(model.new_net(), lum, "cpu")
    assert z.shape == (model.CLASSES, 300, 457) == (3, 300, 457) and z.dtype == np.float32


def test_pad_is_paper_up_to_the_padded_shape(small_model):
    p = model.pad(np.zeros((5, 9), np.float32))
    assert p.shape == model.padded_shape(5, 9)
    assert (p[:5, :9] == 0).all() and (p[5:, :] == model.PAPER).all() and (p[:, 9:] == model.PAPER).all()


def test_padded_shape_is_a_multiple_of_the_bucket_and_the_net(small_model, monkeypatch):
    """Sides go up to multiples of both the shape bucket and the net's multiple (here 6 and 4: 12)."""
    monkeypatch.setattr(model, "BUCKET", 6)
    assert model.multiple() == 4
    assert model.padded_shape(5, 13) == (12, 24)
    assert model.padded_shape(12, 24) == (12, 24)
    assert model.pad(np.zeros((25, 1), np.float32)).shape == (36, 12)


def test_combine_is_a_tempered_softmax_mean():
    z = np.random.default_rng(0).normal(size=(3, 4, 5)).astype(np.float32)
    p = model.combine([z, 2 * z], temperature=2.0)
    np.testing.assert_allclose(p.sum(0), 1.0, rtol=1e-6)
    e1, e2 = np.exp(z / 2), np.exp(z)
    np.testing.assert_allclose(p, (e1 / e1.sum(0) + e2 / e2.sum(0)) / 2, rtol=1e-5)


def test_combine_of_large_logits_is_finite():
    z = np.array([1000.0, 0.0, -1000.0], np.float32)[:, None, None]
    np.testing.assert_array_equal(model.combine([z]), np.array([1.0, 0.0, 0.0], np.float32)[:, None, None])


def test_prob_combines_each_nets_logits(small_model):
    torch.manual_seed(0)
    a, b = model.new_net(), model.new_net()
    lum = np.full((40, 40), 0.7, np.float32)
    expect = model.combine([model.logits(a, lum, "cpu"), model.logits(b, lum, "cpu")], temperature=2.0)
    p = model.prob([a, b], lum, "cpu", temperature=2.0)
    assert p.shape == (3, 40, 40) and p.dtype == np.float32
    np.testing.assert_allclose(p, expect, rtol=1e-6)


def test_logits_restore_train_mode(small_model):
    net = model.new_net().train()
    model.logits(net, np.ones((8, 8), np.float32), "cpu")
    assert net.training


def test_full_size_constants():
    """The real net's sides are multiples of 256, which divides the bucket: pages are padded to multiples of 1,024."""
    assert model.multiple() == 256 and model.BUCKET % model.multiple() == 0
    assert model.padded_shape(1, 1025) == (1024, 2048)
