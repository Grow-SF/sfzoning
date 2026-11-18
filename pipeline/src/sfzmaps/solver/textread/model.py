"""The text reader's networks in plain torch: the recogniser (`Reader`) and the 0/180 classifier (`Direction`).

The recogniser is docTR's `crnn_mobilenet_v3_small` re-expressed: torchvision's MobileNetV3-small feature layers
(`feat_extractor`, the same layer order and parameter names, so docTR's state dicts load) with the depthwise
convolutions of layers 2, 4 and 9 at stride (2, 1) (docTR's `mobilenet_v3_small_r`: a 32 px tall crop comes out one
row tall, its width divided by STRIDE), then a two-layer bidirectional LSTM of 128 units over the columns
(`decoder`) and a linear layer to the classes (`linear`): VOCAB and the CTC blank, the last class.

The input (`to_input`): a crop's grey px (uint8, paper 255) padded with paper on the right to whole CTC frames
(`padded`), divided by 255 and normalised by docTR's mean and sd, repeated over three channels.

The 0/180 classifier (`Direction`, spike S2's): four 3x3 conv blocks and a linear layer on their global mean, one
logit: the crop is upside down.
"""
import math

import numpy as np
import torch
from torch import nn

VOCAB = " &'(),-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = len(VOCAB)               # the CTC blank: the last class
CLASSES = len(VOCAB) + 1
HEIGHT = 32                      # px, every crop's height
STRIDE = 4                       # input px per CTC frame
MEAN, SD = 0.694, 0.299          # docTR's normalisation of the grey value
WIDTHS = (64, 96, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280)   # training batch widths
RNN_UNITS = 128
BN_EPS, BN_MOMENTUM = 0.001, 0.01
# torchvision's MobileNetV3-small inverted residual blocks: input channels, kernel, expanded channels, output
# channels, squeeze-and-excite, hard swish (else ReLU), stride
SMALL = ((16, 3, 16, 16, True, False, 2), (16, 3, 72, 24, False, False, 2), (24, 3, 88, 24, False, False, 1),
         (24, 5, 96, 40, True, True, 2), (40, 5, 240, 40, True, True, 1), (40, 5, 240, 40, True, True, 1),
         (40, 5, 120, 48, True, True, 1), (48, 5, 144, 48, True, True, 1), (48, 5, 288, 96, True, True, 2),
         (96, 5, 576, 96, True, True, 1), (96, 5, 576, 96, True, True, 1))
RECT = (2, 4, 9)                 # feature layers whose depthwise convolution strides (2, 1): height only
LAST = 576                       # the last feature layer's channels (6 x 96)


def _divisible(v: float, divisor: int = 8) -> int:
    """torchvision's `_make_divisible`: v rounded to a multiple of `divisor`, never more than 10% below v."""
    new = max(divisor, int(v + divisor / 2) // divisor * divisor)
    return new + divisor if new < 0.9 * v else new


class ConvNormAct(nn.Sequential):
    """torchvision's Conv2dNormActivation: a bias-free convolution (`0`), batch norm (`1`), an activation (`2`)."""

    def __init__(self, cin: int, cout: int, kernel: int, stride=1, groups: int = 1, act=nn.Hardswish):
        layers = [nn.Conv2d(cin, cout, kernel, stride, (kernel - 1) // 2, groups=groups, bias=False),
                  nn.BatchNorm2d(cout, eps=BN_EPS, momentum=BN_MOMENTUM)]
        if act is not None:
            layers.append(act(inplace=True))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    """torchvision's SqueezeExcitation with a hard sigmoid scale."""

    def __init__(self, channels: int, squeeze: int):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeeze, 1)
        self.fc2 = nn.Conv2d(squeeze, channels, 1)
        self.activation = nn.ReLU()
        self.scale_activation = nn.Hardsigmoid()

    def forward(self, x):
        s = self.scale_activation(self.fc2(self.activation(self.fc1(self.avgpool(x)))))
        return s * x


class InvertedResidual(nn.Module):
    """torchvision's MobileNetV3 block: expand (when the channels grow), depthwise, squeeze-excite, project; the
    input added back when the shape is kept."""

    def __init__(self, cin: int, kernel: int, expanded: int, cout: int, se: bool, hard: bool, stride):
        super().__init__()
        act = nn.Hardswish if hard else nn.ReLU
        layers = []
        if expanded != cin:
            layers.append(ConvNormAct(cin, expanded, 1, act=act))
        layers.append(ConvNormAct(expanded, expanded, kernel, stride, groups=expanded, act=act))
        if se:
            layers.append(SqueezeExcite(expanded, _divisible(expanded // 4)))
        layers.append(ConvNormAct(expanded, cout, 1, act=None))
        self.block = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.block(x)
        return y + x if self.residual else y


def features() -> nn.Sequential:
    """MobileNetV3-small's feature layers with RECT's depthwise strides (2, 1), initialised as torchvision does."""
    layers = [ConvNormAct(3, SMALL[0][0], 3, 2)]
    for i, (cin, k, exp, cout, se, hard, stride) in enumerate(SMALL, start=1):
        layers.append(InvertedResidual(cin, k, exp, cout, se, hard, (2, 1) if i in RECT else stride))
    layers.append(ConvNormAct(SMALL[-1][3], LAST, 1))
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class Reader(nn.Module):
    """The recogniser (see the module): `forward` gives the logits (n, frames, CLASSES)."""

    def __init__(self):
        super().__init__()
        self.feat_extractor = features()
        self.decoder = nn.LSTM(input_size=LAST, hidden_size=RNN_UNITS, num_layers=2, bidirectional=True,
                               batch_first=True)
        self.linear = nn.Linear(2 * RNN_UNITS, CLASSES)

    def forward(self, x):
        f = self.feat_extractor(x)
        c, h, w = f.shape[1:]
        f = f.reshape(-1, h * c, w).transpose(1, 2)
        return self.linear(self.decoder(f)[0])


class Direction(nn.Module):
    """p(upside down) of a crop, as a logit: four conv blocks, global average pool, one linear unit."""

    def __init__(self, width: int = 32):
        super().__init__()
        chans = [1, width, width * 2, width * 2, width * 4]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(chans[-1], 1)

    def forward(self, x):
        f = self.body(x[:, :1])
        return self.head(f.mean(dim=(2, 3)))[:, 0]


# --- the input --------------------------------------------------------------------------------------------------------

def encode(text: str) -> list[int]:
    return [VOCAB.index(c) for c in text]


def best_path(log_probs: torch.Tensor) -> str:
    """The best path's string of one crop's frames (frames x classes): each frame's likeliest class, repeats merged,
    blanks dropped."""
    out, last = [], None
    for k in log_probs.argmax(-1).tolist():
        if k != last and k != BLANK:
            out.append(VOCAB[k])
        last = k
    return "".join(out)


def padded(crop: np.ndarray) -> np.ndarray:
    """A crop padded with paper on the right to a whole number of CTC frames, at least two."""
    w = crop.shape[1]
    want = max(STRIDE * 2, math.ceil(w / STRIDE) * STRIDE)
    if want == w:
        return crop
    return np.pad(crop, ((0, 0), (0, want - w)), constant_values=255)


def to_input(batch: list[np.ndarray], quantize: bool = False) -> tuple[torch.Tensor, list[int]]:
    """A batch of crops (`padded`, then right-padded with paper to the longest, or with `quantize` to the next of
    WIDTHS so the GPU sees few shapes) as the networks' input, float32 (n, 3, HEIGHT, W); and each crop's frames."""
    crops = [padded(c) for c in batch]
    w = max(c.shape[1] for c in crops)
    if quantize:
        w = next((q for q in WIDTHS if q >= w), w)
    x = np.full((len(crops), HEIGHT, w), 255, np.uint8)
    for i, c in enumerate(crops):
        x[i, :, :c.shape[1]] = c
    t = (torch.from_numpy(x).float().div_(255.0) - MEAN) / SD
    return t[:, None].expand(-1, 3, -1, -1).contiguous(), [c.shape[1] // STRIDE for c in crops]


# --- weights --------------------------------------------------------------------------------------------------------

def save(path, net: nn.Module, **about) -> None:
    """The weights (on the CPU) and what made them (`about`: JSON-able) in one torch file."""
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "about": about}, path)


def load(path, kind: str = "reader") -> nn.Module:
    """A network written by `save` (or a bare state dict, as spike S2 wrote them), in eval mode on the CPU: "reader"
    or "direction"."""
    got = torch.load(path, map_location="cpu", weights_only=True)
    state = got["state_dict"] if "state_dict" in got else got
    net = Reader() if kind == "reader" else Direction()
    net.load_state_dict(state, strict=True)
    return net.eval()
