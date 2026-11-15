"""The text detector's network: a U-Net over the ink mask, every px scored at the page's own resolution.

Outputs per px (OUTPUTS channels): 0 the text logit; 1-4 the kind logits among text (data.KINDS: zoning code, block
number, street name, other); 5-6 the reading direction's cosine and sine (any length; only the direction counts).  A
text-only network has channel 0 alone.

At half size the ink is folded first (`fold`: each 2 x 2 block of px into 4 channels at one px, every px kept), the
U-Net runs on that at half the px each way, and its head gives 4 x the outputs, unfolded (`unfold`) back to one set of
outputs per page px.  So a side must be even, and the multiple a side is padded to (`TextNet.multiple`) doubles.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

WIDTH = 24        # channels at full resolution
LEVELS = 5        # resolutions, each half the one before
OUTPUTS = 7
FOLD = 2          # px each way folded into channels at half size
MULTIPLE = 2 ** (LEVELS - 1)   # a page side the levels halve evenly is a multiple of this (at full size)


def fold(x: torch.Tensor) -> torch.Tensor:
    """(n, c, h, w) into (n, c * FOLD², h / FOLD, w / FOLD): each FOLD x FOLD block of px into channels, every px kept
    once.  h and w must be multiples of FOLD."""
    return F.pixel_unshuffle(x, FOLD)


def unfold(x: torch.Tensor) -> torch.Tensor:
    """`fold` undone: (n, c * FOLD², h, w) into (n, c, h * FOLD, w * FOLD)."""
    return F.pixel_shuffle(x, FOLD)


def _conv(i: int, o: int) -> list[nn.Module]:
    """One 3 x 3 convolution with batch norm and ReLU."""
    return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


def _block(i: int, o: int) -> nn.Sequential:
    """Two 3 x 3 convolutions, each with batch norm and ReLU."""
    return nn.Sequential(*_conv(i, o), *_conv(o, o))


def _light_block(i: int, o: int) -> nn.Sequential:
    """One 3 x 3 convolution with batch norm and ReLU."""
    return nn.Sequential(*_conv(i, o))


class TextNet(nn.Module):
    """The U-Net: LEVELS blocks down (channels `width`, 2, 4, 8 and 8 x `width`, max-pooled between), LEVELS - 1 blocks
    up (nearest upsampling, each joined to the block down at its resolution; two convolutions each, or one with
    `decoder` "light"), and a 1 x 1 head of `outputs` channels (OUTPUTS, or 1 for text alone).  With `half_size` the
    ink is folded before the blocks and the head's outputs unfolded after (see the module).
    Input: (n, 1, h, w) ink as 1, paper as 0; output (n, outputs, h, w)."""

    def __init__(self, width: int = WIDTH, outputs: int = OUTPUTS, half_size: bool = False, decoder: str = "full"):
        super().__init__()
        if decoder not in ("full", "light"):
            raise ValueError(f"decoder is full or light, not {decoder!r}")
        self.half_size = half_size
        self.multiple = MULTIPLE * (FOLD if half_size else 1)
        cells = FOLD * FOLD if half_size else 1
        ch = [width * min(2 ** i, 8) for i in range(LEVELS)]
        up = _block if decoder == "full" else _light_block
        self.down = nn.ModuleList([_block(cells, ch[0])] + [_block(ch[i], ch[i + 1]) for i in range(LEVELS - 1)])
        self.up = nn.ModuleList([up(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(LEVELS - 1))])
        self.head = nn.Conv2d(ch[0], outputs * cells, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.half_size:
            x = fold(x)
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for u in self.up:
            s = skips.pop()
            x = u(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="nearest"), s], 1))
        x = self.head(x)
        return unfold(x) if self.half_size else x
