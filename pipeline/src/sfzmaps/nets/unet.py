"""A small U-Net that scores every px of an image at its own resolution.  It came from PR #70's street finder
(`HatchNet`), with that finder's hatching-specific constants (district names, scale inputs) left out so any learned
reader can use it.

The network (`UNet`): `levels` blocks down (channels `width` x 1, 2, 4, 8, 8, ..., max-pooled between), `levels - 1`
blocks up (nearest upsampling, each joined to the block down at its resolution), two 3 x 3 convolutions with batch norm
and ReLU per block, and a 1 x 1 head.  A px's outputs depend on the input within `reach(levels)` px of it each way.
The height and width must be multiples of `multiple(levels)`, so the halvings come out even.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def multiple(levels: int) -> int:
    """The side length must be a multiple of this for the levels to halve evenly."""
    return 2 ** (levels - 1)


def reach(levels: int) -> int:
    """How far (px, each way) a px's outputs can see: each 3 x 3 convolution at level l adds 2^l px, each pooling
    2^l px, and each nearest upsampling to level l up to 2^l px (a px reads the px above it, half a step aside).  An
    upper bound: two convolutions per block down, `levels - 1` poolings, two convolutions per block up and
    `levels - 1` upsamplings."""
    down = sum(2 * 2 ** l for l in range(levels))
    up = sum(2 * 2 ** l for l in range(levels - 1))
    shifts = 2 * sum(2 ** l for l in range(levels - 1))          # the poolings and the upsamplings
    return down + up + shifts


def _conv(i: int, o: int) -> list[nn.Module]:
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(*_conv(i, o), *_conv(o, o))


class UNet(nn.Module):
    """The U-Net (see the module).  Input (n, in_ch, h, w); output (n, out_ch, h, w) logits."""

    def __init__(self, in_ch: int, out_ch: int, width: int = 16, levels: int = 5):
        super().__init__()
        self.width = width
        self.levels = levels
        ch = [width * min(2 ** i, 8) for i in range(levels)]
        self.down = nn.ModuleList([_block(in_ch, ch[0])] + [_block(ch[i], ch[i + 1]) for i in range(levels - 1)])
        self.up = nn.ModuleList([_block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(levels - 1))])
        self.head = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for u in self.up:
            s = skips.pop()
            x = u(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="nearest"), s], 1))
        return self.head(x)
