"""The finder's network: the branch's U-Net (`solver.hatchnet.model`) over a scan's luminance, three independent
answers per px.

Input (n, 1, h, w): the scan's luminance as stored (1 paper, 0 ink), nothing else.  Output (n, 3, h, w): the logit
of each answer at every px, in ANSWERS order:
  STREET  0  the px is street (not in a printed block)
  EDGE    1  a printed block edge runs here
  BORDER  2  a printed zoning border runs here

The blocks are hatchnet's (`_block`: two 3 x 3 convolutions with batch norm and ReLU), LEVELS of them down, LEVELS - 1
up, a 1 x 1 head; a px's answers see the input within `REACH` px of it each way.  h and w must be multiples of
MULTIPLE.  Reads and writes nothing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfzmaps.solver.hatchnet.model import LEVELS, MULTIPLE, REACH, _block

STREET, EDGE, BORDER = 0, 1, 2
ANSWERS = ("street", "edge", "border")
WIDTH = 16          # engineering setting: channels at full resolution (hatchnet's)

__all__ = ["ANSWERS", "BORDER", "EDGE", "FindNet", "LEVELS", "MULTIPLE", "REACH", "STREET", "WIDTH"]


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
