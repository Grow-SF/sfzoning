import pytest
import torch
import torch.nn as nn

from sfzmaps.nets.unet import Folded, UNet, folded_multiple, folded_reach, multiple, reach


def test_shapes():
    net = UNet(in_ch=2, out_ch=1, width=4, levels=3)
    y = net(torch.zeros(2, 2, 16, 24))
    assert y.shape == (2, 1, 16, 24)


def test_reach_matches_hatchnet():
    assert reach(5) == 122          # PR #70's REACH at LEVELS = 5 (its docstring says "about 120")
    assert multiple(9) == 256


def test_same_input_same_output():
    torch.manual_seed(0)
    a = UNet(1, 1, width=4, levels=3).eval()
    torch.manual_seed(0)
    b = UNet(1, 1, width=4, levels=3).eval()
    x = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a(x), b(x))


def test_folded_shapes_and_reach():
    net = Folded(2, 1, 4, 8, 3)
    assert net(torch.zeros(2, 2, 64, 96)).shape == (2, 1, 64, 96)
    assert folded_reach(4, 4) == 4 * reach(4)
    assert folded_multiple(4, 3) == 4 * multiple(3)


def test_folded_is_one_output_per_pixel():
    """With the inner U-Net the identity on channels, unshuffle then shuffle must give each px back where it was."""
    net = Folded(2, 2, 4, 4, 2)
    net.net = nn.Identity()
    x = torch.rand(3, 2, 16, 24, generator=torch.Generator().manual_seed(0))
    assert torch.equal(net(x), x)


def test_group_norm_is_batch_independent():
    torch.manual_seed(0)
    net = Folded(1, 1, fold=2, width=4, levels=2, norm="group").train()
    a, b = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
    alone = net(a)
    together = net(torch.cat([a, b]))[:1]
    assert torch.allclose(alone, together, atol=1e-6)


def test_batch_norm_default_keeps_its_parameters():
    keys = set(UNet(1, 1, width=4, levels=2).state_dict())
    assert any("running_mean" in k for k in keys)   # PR #70's checkpoints still load


def test_unknown_norm_refused():
    with pytest.raises(ValueError):
        UNet(1, 1, width=4, levels=2, norm="layer")
