import torch

from sfzmaps.nets.unet import UNet, multiple, reach


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
