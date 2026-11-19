"""The learned reader's network: a small U-Net over a sheet's scan as printed, every px scored at the scan's own
resolution.

Input per px (INPUTS channels): 0 the scan's luminance (`data.SheetRasters.ink`, 0 = ink, 1 = paper), every px as
scanned; 1 the sheet's measured scale, ft/px over FT_SCALE, the same at every px of a sheet (`inputs`).  The scale goes
in so one network reads sheets scanned at 2.72 and at 4.6-5.4 ft/px without rescaling the scan.

Outputs per px (OUTPUTS channels):
  STREET     0     the logit of the px being a street (not in a printed block); trained on today's blocks placed on
                   the scan (`train.fit_streets`)
  DISTRICTS  1-6   the six districts' logits, in NAMES order (trained later, through pooled lot losses)
  WEIGHT     7     the logit of how much the px counts when its lot's district evidence is pooled (trained later)

The network (`HatchNet`): LEVELS blocks down (channels `width` x 1, 2, 4, 8, 8, max-pooled between), LEVELS - 1 blocks
up (nearest upsampling, each joined to the block down at its resolution), two 3 x 3 convolutions with batch norm and
ReLU per block, and a 1 x 1 head.  A px's outputs depend on the input within REACH px of it each way (`reach`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NAMES = sorted(["1R", "2R", "C", "LI", "HI", "U"])
STREET = 0
DISTRICTS = slice(1, 1 + len(NAMES))
WEIGHT = 1 + len(NAMES)
OUTPUTS = 2 + len(NAMES)
INPUTS = 2
# Channels at full resolution.  Small on purpose: about 1.2 M weights against the 1921 sheets' 72 M labelled px, so
# the network has room to learn how a street looks (a white channel between parallel block outlines, never crossed by
# a pattern) and the six patterns, and little room to memorise where today's labels disagree with the 1921 print.
WIDTH = 16
# Resolutions, each half the one before.  Five give a reach (`reach`) of about 120 px each way: across a 1921 block
# at 4.6-5.4 ft/px (55-80 px) and from the middle of a blank 2R block to its outlines at 2.72 ft/px.
LEVELS = 5
MULTIPLE = 2 ** (LEVELS - 1)       # a side the levels halve evenly is a multiple of this
FT_SCALE = 5.0                     # ft/px is fed over this (the 1921 sheets' usual scale), so the plane is about 1


def reach(levels: int = LEVELS) -> int:
    """How far (px, each way) a px's outputs can see: each 3 x 3 convolution at level l adds 2^l px, each pooling
    2^l px, and each nearest upsampling to level l up to 2^l px (a px reads the px above it, half a step aside).  An
    upper bound: two convolutions per block down, LEVELS - 1 poolings, two convolutions per block up and LEVELS - 1
    upsamplings."""
    down = sum(2 * 2 ** l for l in range(levels))
    up = sum(2 * 2 ** l for l in range(levels - 1))
    shifts = 2 * sum(2 ** l for l in range(levels - 1))          # the poolings and the upsamplings
    return down + up + shifts


REACH = reach()


def inputs(ink: torch.Tensor, ft_per_px: torch.Tensor) -> torch.Tensor:
    """The network's input (n, INPUTS, h, w) from the scans' luminance (n, h, w) and each scan's ft/px (n,)."""
    plane = (ft_per_px.to(ink.dtype) / FT_SCALE)[:, None, None].expand_as(ink)
    return torch.stack([ink, plane], 1)


def _conv(i: int, o: int) -> list[nn.Module]:
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(*_conv(i, o), *_conv(o, o))


class HatchNet(nn.Module):
    """The U-Net (see the module).  Input (n, INPUTS, h, w) (`inputs`); output (n, OUTPUTS, h, w)."""

    def __init__(self, width: int = WIDTH):
        super().__init__()
        self.width = width
        ch = [width * min(2 ** i, 8) for i in range(LEVELS)]
        self.down = nn.ModuleList([_block(INPUTS, ch[0])] + [_block(ch[i], ch[i + 1]) for i in range(LEVELS - 1)])
        self.up = nn.ModuleList([_block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(LEVELS - 1))])
        self.head = nn.Conv2d(ch[0], OUTPUTS, 1)

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
