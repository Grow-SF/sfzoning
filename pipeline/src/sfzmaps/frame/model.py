"""The frame finder's one network (issue #71, spec part 3): a folded U-Net reading the whole page at full
resolution and giving, per pixel, the logit of "outside the map area".

One network, one pass, no coarse page pass and no tiles' seams: its reach (about 2,000 px each way) is the context
a pixel needs.  GroupNorm, not batch norm, so the output does not depend on what else is in the batch (one page per
step).  FOLD, WIDTH and LEVELS are read when called, never bound at import, so tests can shrink the net.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from sfzmaps.nets.unet import Folded, folded_multiple

FOLD = 8     # pixels folded into channels (the spike's network)
WIDTH = 32   # channels of the first level (the spike's network)
LEVELS = 6   # U-Net levels: with FOLD 8 a reach of 2,000 px (spec part 3)
PAPER = 1.0  # luminance of the padding: plain paper


def new_net() -> Folded:
    return Folded(1, 1, FOLD, WIDTH, LEVELS, norm="group")


def multiple() -> int:
    """Page sides the net accepts: 256 at full size."""
    return folded_multiple(FOLD, LEVELS)


def pad(lum: np.ndarray) -> np.ndarray:
    """`lum` with paper added at the bottom and right up to multiples of `multiple()`."""
    m = multiple()
    h, w = lum.shape
    out = np.full((-(-h // m) * m, -(-w // m) * m), PAPER, np.float32)
    out[:h, :w] = lum
    return out


@torch.no_grad()
def logits(net: nn.Module, lum: np.ndarray, device: str) -> np.ndarray:
    """Float32 logits of "outside", shape `lum.shape`.  Eval mode; the net's mode and device are restored."""
    was_training = net.training
    old = next(net.parameters()).device
    net.eval().to(device)
    try:
        x = torch.from_numpy(pad(lum))[None, None].to(device)
        z = net(x)[0, 0, :lum.shape[0], :lum.shape[1]]
        return z.float().cpu().numpy().astype(np.float32)
    finally:
        net.to(old).train(was_training)


def combine(zs: Sequence[np.ndarray], temperature: float = 1.0) -> np.ndarray:
    """Float32 mean over `zs` of sigmoid(z / temperature): P(outside)."""
    ps = [1.0 / (1.0 + np.exp(-np.asarray(z, np.float64) / temperature)) for z in zs]
    return np.mean(ps, axis=0).astype(np.float32)


def prob(nets: Sequence[nn.Module], lum: np.ndarray, device: str, temperature: float = 1.0) -> np.ndarray:
    return combine([logits(n, lum, device) for n in nets], temperature)
