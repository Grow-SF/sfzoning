"""The finder's network (`sfzmaps.find.model`): one luminance channel in, the two answers' logits (street, block edge)
out, at every px; a model saved with the three answers of before is refused in words."""
import pytest
import torch

from sfzmaps.find import counts, model


def test_shapes():
    net = model.FindNet(8)
    out = net(torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 2, 64, 64)


def test_a_three_answer_model_is_refused_in_words():
    old = model.FindNet(4)
    old.head = torch.nn.Conv2d(4, 3, 1)                     # street, block edge and zoning border, as saved before
    with pytest.raises(ValueError, match="predates the two-answer finder"):
        counts.network(old.state_dict())
    assert counts.network(model.FindNet(4).state_dict()).head.out_channels == 2
