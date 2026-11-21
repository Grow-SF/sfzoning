"""The frame finder's one network (issue #71, spec part 3 and the amendment "removed blocks"): a folded U-Net reading
the whole page at full resolution and giving, per pixel, a logit for each of `CLASSES` answers: 0, map; 1, outside the
frame; 2, a removed block (a legend or title block, a compass rose or a "SECTION N" badge printed inside the frame,
deleted wherever it sits).  `combine` turns logits into the three chances, a softmax over the classes.

One network, one pass, no coarse page pass and no tiles' seams: its reach (about 2,000 px each way) is the context
a pixel needs.  GroupNorm, not batch norm, so the output does not depend on what else is in the batch (one page per
step).  FOLD, WIDTH, LEVELS, CLASSES and BUCKET are read when called, never bound at import, so tests can shrink the
net.

A page is read padded with paper at the bottom and right up to `padded_shape`: sides that are multiples of `BUCKET` px
(and of `multiple()`, the net's own), the same in training and prediction.  The bucket is computational: the Mac's
GPU builds its kernels for each new input shape, and every training sample had a new one (a random cut and scale).
Measured on the 21 labelled pages, "fold 16" net (issue #71, `scripts/spikes/issue71_shape_cost.py`): a step was
132 ms on 40 new shapes and 89 ms when the same shapes came again; padded up to multiples of 1,024 px the 40 samples
had 14 shapes and 15.6% more px, and a step was 96 ms once the shapes had been seen.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from sfzmaps.nets.unet import Folded, folded_multiple

# "fold 16", chosen at the first look (issue #71 ledger R11: held-out loss tied the plan's net, 1.7x faster):
FOLD = 16    # pixels folded into channels
WIDTH = 32   # channels of the first level
LEVELS = 5   # U-Net levels: with FOLD 16 a reach of 1,952 px (spec part 3: about 2,000)
PAPER = 1.0  # luminance of the padding: plain paper
CLASSES = 3  # 0 map, 1 outside the frame, 2 removed block (spec amendment "removed blocks")
BUCKET = 1024  # px: padded sides are multiples of this, so few shapes reach the GPU (computational; see the module)


def new_net() -> Folded:
    return Folded(1, CLASSES, FOLD, WIDTH, LEVELS, norm="group")


def multiple() -> int:
    """Page sides the net accepts: 256 at full size."""
    return folded_multiple(FOLD, LEVELS)


def padded_shape(h: int, w: int) -> tuple[int, int]:
    """(H, W) a page of h x w px is read at: each side up to a multiple of both `BUCKET` and `multiple()`."""
    m = math.lcm(multiple(), BUCKET)
    return -(-h // m) * m, -(-w // m) * m


def pad(lum: np.ndarray) -> np.ndarray:
    """`lum` with paper added at the bottom and right up to `padded_shape`."""
    h, w = lum.shape
    out = np.full(padded_shape(h, w), PAPER, np.float32)
    out[:h, :w] = lum
    return out


@torch.no_grad()
def logits(net: nn.Module, lum: np.ndarray, device: str) -> np.ndarray:
    """Float32 logits, shape `(CLASSES, *lum.shape)`.  Eval mode; the net's mode and device are restored."""
    was_training = net.training
    old = next(net.parameters()).device
    net.eval().to(device)
    try:
        x = torch.from_numpy(pad(lum))[None, None].to(device)
        z = net(x)[0, :, :lum.shape[0], :lum.shape[1]]
        return z.float().cpu().numpy().astype(np.float32)
    finally:
        net.to(old).train(was_training)


def combine(zs: Sequence[np.ndarray], temperature: float = 1.0) -> np.ndarray:
    """Float32 `(CLASSES, h, w)`: the mean over `zs` (each `(CLASSES, h, w)` logits) of softmax(z / temperature) over
    the classes (axis 0).  One temperature divides every class's logits (Guo et al. 2017's temperature scaling is
    multiclass).  Each softmax subtracts its largest logit first, so no logit overflows; float32 and in place, so a
    whole page needs two arrays of its three planes, whatever the number of nets."""
    total = None
    for z in zs:
        e = np.asarray(z, np.float32) / np.float32(temperature)
        e -= e.max(axis=0, keepdims=True)
        np.exp(e, out=e)
        e /= e.sum(axis=0, keepdims=True)
        if total is None:
            total = e
        else:
            total += e
    total /= np.float32(len(zs))
    return total


def prob(nets: Sequence[nn.Module], lum: np.ndarray, device: str, temperature: float = 1.0) -> np.ndarray:
    """The three chances `(CLASSES, *lum.shape)` of `nets` on `lum`: `combine` of each net's `logits`."""
    return combine([logits(n, lum, device) for n in nets], temperature)
