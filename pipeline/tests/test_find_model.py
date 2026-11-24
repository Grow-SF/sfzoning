"""The finder's network (`sfzmaps.find.model`): one luminance channel in, the three answers' logits out, at every px."""
import torch

from sfzmaps.find import model


def test_shapes():
    net = model.FindNet(8)
    out = net(torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 3, 64, 64)
