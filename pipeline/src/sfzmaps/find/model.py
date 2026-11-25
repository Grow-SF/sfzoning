"""The finder's network: the 1921 hatching reader's U-Net (`solver.hatchnet.model`, branch `hatch-experiments-archive`;
its levels, reach and blocks moved here unchanged) over a scan's luminance, two independent answers per px.

Input (n, 1, h, w): the scan's luminance as stored (1 paper, 0 ink), nothing else.  Output (n, 2, h, w): the logit
of each answer at every px, in ANSWERS order:
  STREET  0  the px is street (not in a printed block)
  EDGE    1  a printed block edge runs here

The blocks are the hatching reader's (`_block`: two 3 x 3 convolutions with batch norm and ReLU), LEVELS of them
down, LEVELS - 1 up, a 1 x 1 head; a px's answers see the input within `REACH` px of it each way.  h and w must be
multiples of MULTIPLE.  Reads and writes nothing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STREET, EDGE = 0, 1
ANSWERS = ("street", "edge")
WIDTH = 16          # engineering setting: channels at full resolution (the hatching reader's)
# Resolutions, each half the one before.  Five give a reach (`reach`) of about 120 px each way: across a 1921 block
# at 4.6-5.4 ft/px (55-80 px) and from the middle of a blank 2R block to its outlines at 2.72 ft/px.
LEVELS = 5
MULTIPLE = 2 ** (LEVELS - 1)       # a side the levels halve evenly is a multiple of this


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


def _conv(i: int, o: int) -> list[nn.Module]:
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(*_conv(i, o), *_conv(o, o))


__all__ = ["ANSWERS", "EDGE", "FindNet", "LEVELS", "MULTIPLE", "REACH", "STREET", "WIDTH"]


class FindNet(nn.Module):
    """The U-Net (see the module): channels `width` x 1, 2, 4, 8, 8 down the levels."""

    def __init__(self, width: int = WIDTH):
        super().__init__()
        self.width = width
        ch = [width * min(2 ** i, 8) for i in range(LEVELS)]
        self.down = nn.ModuleList([_block(1, ch[0])] + [_block(ch[i], ch[i + 1]) for i in range(LEVELS - 1)])
        self.up = nn.ModuleList([_block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(LEVELS - 1))])
        self.head = nn.Conv2d(ch[0], len(ANSWERS), 1)

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
