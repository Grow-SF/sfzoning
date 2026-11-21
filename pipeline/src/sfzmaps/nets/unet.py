"""A small U-Net that scores every px of an image at its own resolution.  It came from PR #70's street finder
(`HatchNet`), with that finder's hatching-specific constants (district names, scale inputs) left out so any learned
reader can use it.

The network (`UNet`): `levels` blocks down (channels `width` x 1, 2, 4, 8, 8, ..., max-pooled between), `levels - 1`
blocks up (nearest upsampling, each joined to the block down at its resolution), two 3 x 3 convolutions with batch norm
and ReLU per block, and a 1 x 1 head.  `norm="group"` swaps batch norm for group norm, whose statistics come from
each image alone (so one image per step trains the same as a batch; issue #71); the default stays batch norm, so PR
#70's checkpoints still load.  A px's outputs depend on the input within `reach(levels)` px of it each way.
The height and width must be multiples of `multiple(levels)`, so the halvings come out even.

The folded network (`Folded`): the input's `fold` x `fold` px blocks folded into channels (`pixel_unshuffle`), a U-Net
at 1/fold the size each way, and its outputs unfolded back (`pixel_shuffle`): one output per input px still, nothing
discarded.  On the Mac's GPU a convolution's backward costs 12-35 times its forward on layers with few channels and
millions of px (a U-Net's full-resolution top level); folded, no layer is both, and a training step is about 8 times
faster (issue #71, `scripts/spikes/issue71_speed4.py`).  Its reach is `folded_reach` page px; its sides must be
multiples of `folded_multiple`.
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


GROUPS = 32          # Wu & He 2018's default; a layer with fewer channels uses one group per channel


def _conv(i: int, o: int, norm: str) -> list[nn.Module]:
    if norm == "batch":
        n = nn.BatchNorm2d(o)
    elif norm == "group":
        n = nn.GroupNorm(min(GROUPS, o), o)
    else:
        raise ValueError(f"norm must be 'batch' or 'group', not {norm!r}")
    return [nn.Conv2d(i, o, 3, padding=1), n, nn.ReLU(inplace=True)]


def _block(i: int, o: int, norm: str) -> nn.Sequential:
    return nn.Sequential(*_conv(i, o, norm), *_conv(o, o, norm))


class UNet(nn.Module):
    """The U-Net (see the module).  Input (n, in_ch, h, w); output (n, out_ch, h, w) logits."""

    def __init__(self, in_ch: int, out_ch: int, width: int = 16, levels: int = 5, norm: str = "batch"):
        super().__init__()
        self.width = width
        self.levels = levels
        ch = [width * min(2 ** i, 8) for i in range(levels)]
        self.down = nn.ModuleList([_block(in_ch, ch[0], norm)] + [_block(ch[i], ch[i + 1], norm) for i in range(levels - 1)])
        self.up = nn.ModuleList([_block(ch[i + 1] + ch[i], ch[i], norm) for i in reversed(range(levels - 1))])
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


def folded_multiple(fold: int, levels: int) -> int:
    """A `Folded` net's sides must be multiples of this: the fold, then the inner U-Net's halvings."""
    return fold * multiple(levels)


def folded_reach(fold: int, levels: int) -> int:
    """How far (input px, each way) a `Folded` net's outputs see: the inner U-Net's reach in folded px, each `fold`
    input px.  Measured from a px's fold block, so a px sees up to `fold - 1` px more on one side; a tile margin
    rounded up to `folded_multiple` covers that (the tiling test checks it)."""
    return fold * reach(levels)


class Folded(nn.Module):
    """`pixel_shuffle(UNet(in_ch * fold², out_ch * fold², width, levels)(pixel_unshuffle(x, fold)), fold)` (see the
    module).  Input (n, in_ch, h, w), h and w multiples of `folded_multiple(fold, levels)`; output (n, out_ch, h, w)
    logits."""

    def __init__(self, in_ch: int, out_ch: int, fold: int, width: int, levels: int, norm: str = "batch"):
        super().__init__()
        self.fold = fold
        self.net = UNet(in_ch * fold * fold, out_ch * fold * fold, width, levels, norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pixel_shuffle(self.net(F.pixel_unshuffle(x, self.fold)), self.fold)
