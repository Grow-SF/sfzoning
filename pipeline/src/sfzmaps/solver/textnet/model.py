"""The text detector's network: a U-Net over the ink mask, every px scored at the page's own resolution.

Outputs per px (OUTPUTS channels): 0 the text logit; 1-4 the kind logits among text (data.KINDS: zoning code, block
number, street name, other); 5-6 the reading direction's cosine and sine (any length; only the direction counts).  A
text-only network has channel 0 alone.  A network with the word heads (`train.Recipe.heads="words"`, WORD_OUTPUTS
channels) has no sign and no side order in its geometry (a word's reading sign can't be told under the training's
quarter turns), so its channels are (`data.crop_targets(..., words=True)`):
  0          the text logit, 1-4 the kind logits (as above)
  AXIS 5-6   the reading AXIS: (cos 2t, sin 2t) of its angle t (any length; only the direction counts), t taken modulo
             180 degrees: a word read either way has one axis
  CORE 7     the logit of the px being in a word's core
  GEO  8-11  8, 9: dx, dy, the offset from the px to its word's centre in image px (x right, y down) over the word's
             height; 10, 11: the log of the word's length and of its height in px (along its axis and across it)
  TAG  12-15 the px's word tag, a vector classified to its own word's mean tag among its crop's words' (`train`)
A network with the word heads and the same-word affinity (`train.Recipe.heads="words_aff"`, WORD_AFF_OUTPUTS channels;
CRAFT's and instance segmentation's affinity) has those 16, the same layout, and
  AFF  16-17 16: the logit of the px and the px to its RIGHT being in one word; 17: of the px and the px BELOW it
             (`data.affinity_targets`)
`word_geometry` decodes AXIS and GEO into px and degrees.  The number of channels of each recipe's heads:
HEAD_OUTPUTS.

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
AXIS = slice(5, 7)             # the word heads' channels (see the module)
CORE = 7
GEO = slice(8, 12)
TAG = slice(12, 16)
WORD_OUTPUTS = 16
AFF = slice(16, 18)            # the same-word affinity to the right and below (heads "words_aff"; see the module)
WORD_AFF_OUTPUTS = 18
HEAD_OUTPUTS = {"all": OUTPUTS, "text": 1, "words": WORD_OUTPUTS,
                "words_aff": WORD_AFF_OUTPUTS}                       # channels, by `train.Recipe.heads`
GEO_LIMIT = 20.0               # a numerical guard, not a decision: the log sizes are read within +-this when decoded,
                               # so a size stays finite in float32 (e^20 px is far beyond any page)
FOLD = 2          # px each way folded into channels at half size
MULTIPLE = 2 ** (LEVELS - 1)   # a page side the levels halve evenly is a multiple of this (at full size)


def word_geometry(out: torch.Tensor) -> torch.Tensor:
    """The word heads' geometry decoded from a word network's outputs (`out` (..., WORD_OUTPUTS or WORD_AFF_OUTPUTS, h,
    w), the channels third from last): (..., 5, h, w) of each px's word centre offset from the px in px (x right, y down: dx and dy
    times the height), the word's length and height in px, and its axis in degrees in [0, 180) (half the angle of
    AXIS's (cos 2t, sin 2t))."""
    g = out[..., GEO, :, :]
    length, height = torch.exp(g[..., 2:4, :, :].clamp(-GEO_LIMIT, GEO_LIMIT)).unbind(-3)
    axis = torch.rad2deg(torch.atan2(out[..., AXIS.start + 1, :, :], out[..., AXIS.start, :, :])) / 2 % 180
    return torch.stack([g[..., 0, :, :] * height, g[..., 1, :, :] * height, length, height, axis], -3)


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
    `decoder` "light"), and a 1 x 1 head of `outputs` channels (OUTPUTS, 1 for text alone, WORD_OUTPUTS with the word
    heads).  With `half_size` the ink is folded before the blocks and the head's outputs unfolded after (see the
    module).
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
