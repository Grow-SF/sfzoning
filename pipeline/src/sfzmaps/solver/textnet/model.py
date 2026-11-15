"""The text detector's network: a U-Net over the ink mask, every px scored at the page's own resolution.

Outputs per px (OUTPUTS channels): 0 the text logit; 1-4 the kind logits among text (data.KINDS: zoning code, block
number, street name, other); 5-6 the reading direction's cosine and sine (any length; only the direction counts).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 24        # channels at full resolution
LEVELS = 5        # resolutions, each half the one before
OUTPUTS = 7
MULTIPLE = 2 ** (LEVELS - 1)   # a page side the levels halve evenly is a multiple of this


def _block(i: int, o: int) -> nn.Sequential:
    """Two 3 x 3 convolutions, each with batch norm and ReLU."""
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class TextNet(nn.Module):
    """The U-Net: LEVELS blocks down (channels WIDTH, 2, 4, 8 and 8 x WIDTH, max-pooled between), LEVELS - 1 blocks up
    (nearest upsampling, each joined to the block down at its resolution), and a 1 x 1 head of OUTPUTS channels.
    Input: (n, 1, h, w) ink as 1, paper as 0."""

    def __init__(self, width: int = WIDTH):
        super().__init__()
        ch = [width * min(2 ** i, 8) for i in range(LEVELS)]
        self.down = nn.ModuleList([_block(1, ch[0])] + [_block(ch[i], ch[i + 1]) for i in range(LEVELS - 1)])
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
