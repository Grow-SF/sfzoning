"""Whole-page prediction for the learned frame finder: a page pass on the whole page at low resolution (every output
px sees the whole page), fed `[lum, *planes]`, then a pixel pass at full resolution over tiles, fed `[lum, prior,
*planes]` where `prior` is the page pass's probability that a px is map.

The position planes (`data.position_planes`, named in `data.POSITION_CHANNELS`; issue #71 Tasks 14-15) say where the
scan is: `valid` is 1 on the page (the default reads it alone), `edge` each px's distance to the page's nearest edge
as a fraction of its shorter side; both 0 off the page.  Each pass computes them from the page it reads (the page pass
from the resized page), and pads them with 0 where it pads the luminance with paper (1.0) and the prior with 0.0.

Both passes are folded U-Nets (`nets.unet.Folded`): 4 x 4 px folded into channels, one output per px still (issue
#71's speed amendment).  The page pass sees the whole COARSE_SIDE square (`folded_reach` = 1000 >= COARSE_SIDE =
512); the pixel pass sees `folded_reach(FINE_FOLD, FINE_LEVELS)` = 232 px each way, so each tile is read with a margin
of that, rounded up to the folded pooling grid (256).

The page pass's prior is float16 wherever it is kept or read (`page_prior`): saved, read by the pixel pass in
prediction, and in training (`train._prior_window`), so the pixel pass always reads the same precision.

The pixel pass is a correction of the page pass (`Corrected`): its logit is the prior's logit,
`logit(clip(prior, PRIOR_CLIP, 1 - PRIOR_CLIP))`, plus the net's output, and its head convolution starts at zero, so
an untrained pixel pass returns the page pass's answer.  The sum is inside the module, so training (whose loss reads
the summed logit), validation and prediction all use the one formula.  Off the page the prior is 0.0 (not map); the
clip keeps its logit finite."""
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from sfzmaps.frame import data
from sfzmaps.frame.data import position_planes, stacked_planes
from sfzmaps.nets.unet import Folded, folded_multiple, folded_reach

COARSE_SIDE = 512
COARSE_FOLD = 4
COARSE_LEVELS = 6
COARSE_WIDTH = 16
FINE_FOLD = 4
FINE_LEVELS = 4
FINE_WIDTH = 32
TILE = 256
# Whole-page prediction reads bigger cores than training's crops: the margin is the same, so the result is the same
# (seamless, R1/R5) but far fewer px are read twice.  A multiple of the fine net's stride.
PREDICT_TILE = 1536
PRIOR_CLIP = 1e-4       # the prior is clipped to [this, 1 - this] before its logit (finite: about +-9.2)

assert folded_reach(COARSE_FOLD, COARSE_LEVELS) >= COARSE_SIDE, "every page-pass output must see the whole page"
assert COARSE_SIDE % folded_multiple(COARSE_FOLD, COARSE_LEVELS) == 0, "COARSE_SIDE must sit on the page pass's grid"


def _planes() -> int:
    """How many position planes the nets read (`data.POSITION_CHANNELS`, read when a net is made)."""
    return len(data.POSITION_CHANNELS)


def new_coarse_net() -> Folded:
    """The page pass, reading `[lum, *planes]`."""
    return Folded(1 + _planes(), 1, COARSE_FOLD, COARSE_WIDTH, COARSE_LEVELS)


class Corrected(Folded):
    """The pixel pass: a `Folded` net reading `[lum, prior, *planes]` whose output is added to the prior's logit
    (see the module).  Its head convolution (the inner U-Net's last) starts at zero weights and bias."""

    def __init__(self, fold: int, width: int, levels: int):
        super().__init__(2 + _planes(), 1, fold, width, levels)
        nn.init.zeros_(self.net.head.weight)
        nn.init.zeros_(self.net.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x[:, 1:2].clamp(PRIOR_CLIP, 1.0 - PRIOR_CLIP)
        return torch.log(p) - torch.log1p(-p) + super().forward(x)


def new_fine_net() -> Corrected:
    return Corrected(FINE_FOLD, FINE_WIDTH, FINE_LEVELS)


def _fine_multiple() -> int:
    return folded_multiple(FINE_FOLD, FINE_LEVELS)


def coarse_size(h: int, w: int) -> tuple[int, int, float]:
    """(rows, columns, scale) of an h x w page on the page pass's grid: its long side COARSE_SIDE."""
    scale = COARSE_SIDE / max(h, w)
    return max(1, round(h * scale)), max(1, round(w * scale)), scale


def coarse_resize(img: Image.Image, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """A mode "F" image, or its `box` (left, top, right, bottom: training's cut page, read without copying the
    page), resized (Pillow BOX) so its long side is COARSE_SIDE; float32, unpadded."""
    left, top, right, bottom = box if box is not None else (0, 0, *img.size)
    nh, nw, _ = coarse_size(bottom - top, right - left)
    return np.array(img.resize((nw, nh), Image.BOX, box=(left, top, right, bottom)), np.float32)   # writable


def coarse_square(small: np.ndarray) -> np.ndarray:
    """(1 + planes, COARSE_SIDE, COARSE_SIDE) float32: a page on the page pass's grid (`coarse_resize`) and its
    position planes (`data.position_planes` of its shape) at top left, padded at the right and bottom with paper
    (1.0) in the luminance and 0 in the planes: every page, either way up, is read with the same frame of padding, in
    training and in prediction alike."""
    m = folded_multiple(COARSE_FOLD, COARSE_LEVELS)
    if COARSE_SIDE % m:
        raise ValueError(f"COARSE_SIDE {COARSE_SIDE} is not a multiple of {m}")
    nh, nw = small.shape
    out = np.zeros((1 + _planes(), COARSE_SIDE, COARSE_SIDE), np.float32)
    out[0] = 1.0
    out[0, :nh, :nw] = small
    out[1:, :nh, :nw] = stacked_planes((nh, nw))
    return out


def coarse_input(lum: np.ndarray) -> tuple[np.ndarray, float]:
    """The page pass's input for a page (`coarse_square` of the resized page) and the scale factor."""
    img = Image.fromarray(np.ascontiguousarray(lum, np.float32), mode="F")
    return coarse_square(coarse_resize(img)), coarse_size(*lum.shape)[2]


@contextmanager
def _eval_on(net: nn.Module, device: str):
    """The net in eval mode on `device`, put back as the caller had it (train/eval mode and device) afterwards."""
    was = net.training
    home = next(net.parameters()).device
    net.to(device).eval()
    try:
        yield net
    finally:
        net.train(was)
        net.to(home)


def _run(net: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    """Sigmoid probability of a (c, h, w) array, float32.  The net must be in eval mode (see `_eval_on`)."""
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(device)
        return torch.sigmoid(net(t))[0, 0].float().cpu().numpy()


def predict_coarse(net: nn.Module, lum: np.ndarray, device: str = "cpu") -> np.ndarray:
    """The page pass's probability, cropped of padding and resized (bilinear) to the page's shape."""
    h, w = lum.shape
    arr, _ = coarse_input(lum)
    with _eval_on(net, device):
        prob = _run(net, arr, device)
    nh, nw, _ = coarse_size(h, w)
    prob = prob[:nh, :nw]
    t = torch.from_numpy(np.ascontiguousarray(prob))[None, None]
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy().astype(np.float32)


def page_prior(net: nn.Module, lum: np.ndarray, device: str = "cpu") -> np.ndarray:
    """The page pass's map probability as it is saved and as the pixel pass reads it: `predict_coarse`, float16."""
    return predict_coarse(net, lum, device).astype(np.float16)


def _margin() -> int:
    """The pixel pass's reach (`folded_reach`) rounded up to its grid (`folded_multiple`), so every tile and the
    whole-page run fold and pool on the same grid."""
    m = _fine_multiple()
    return -(-folded_reach(FINE_FOLD, FINE_LEVELS) // m) * m


def _padded(lum: np.ndarray, prior: np.ndarray, margin: int, ph: int, pw: int) -> np.ndarray:
    """(2 + planes, ph, pw): [lum, prior, *planes] of the page at (margin, margin), paper (1.0) in lum and 0 in the
    prior (not map) and the position planes (off the page) around it."""
    h, w = lum.shape
    x = np.zeros((2 + _planes(), ph, pw), np.float32)
    x[0] = 1.0
    x[0, margin:margin + h, margin:margin + w] = lum
    x[1, margin:margin + h, margin:margin + w] = prior
    x[2:] = stacked_planes((h, w), (-margin, -margin, ph, pw))
    return x


def _predict_padded_whole(net: nn.Module, lum: np.ndarray, prior: np.ndarray, device: str = "cpu") -> np.ndarray:
    """The reference tiling must equal: one run on the page paper-padded by the margin on every side, then at right and
    bottom to a multiple of the pixel pass's grid (`folded_multiple`), cropped back to the page."""
    h, w = lum.shape
    m, mg = _fine_multiple(), _margin()
    ph, pw = -(-(h + 2 * mg) // m) * m, -(-(w + 2 * mg) // m) * m
    with _eval_on(net, device):
        out = _run(net, _padded(lum, prior, mg, ph, pw), device)
    return out[mg:mg + h, mg:mg + w]


def predict_tiles(net: nn.Module, lum: np.ndarray, prior: np.ndarray, device: str = "cpu",
                  near: np.ndarray | None = None, tile: int | None = None) -> np.ndarray:
    """The pixel pass's probability at full resolution.  TILE x TILE cores, each read with a margin of reach px of
    page or paper around it; only the core is kept.  With `near` (bool, the page's shape), only the tiles whose core
    holds a True px are read; every other px is the prior's (the validation curve's shortcut: the border is where
    the pixel pass works)."""
    TILE_ = tile or (TILE if near is not None else PREDICT_TILE)
    if TILE_ % _fine_multiple():
        raise ValueError(f"tile {TILE_} is not a multiple of {_fine_multiple()}")
    h, w = lum.shape
    mg = _margin()
    th, tw = -(-h // TILE_) * TILE_, -(-w // TILE_) * TILE_
    x = _padded(lum, prior, mg, th + 2 * mg, tw + 2 * mg)
    out = np.zeros((th, tw), np.float32)
    if near is not None:
        out[:h, :w] = prior
    with _eval_on(net, device):
        for y in range(0, th, TILE_):
            for xx in range(0, tw, TILE_):
                if near is not None and not near[y:y + TILE_, xx:xx + TILE_].any():
                    continue
                p = _run(net, x[:, y:y + TILE_ + 2 * mg, xx:xx + TILE_ + 2 * mg], device)
                out[y:y + TILE_, xx:xx + TILE_] = p[mg:mg + TILE_, mg:mg + TILE_]
    return out[:h, :w]


def predict_page(coarse_net: nn.Module, fine_net: nn.Module, lum: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """(coarse_prob, map_prob), both float32 of the page's shape."""
    coarse = predict_coarse(coarse_net, lum, device)
    return coarse, predict_tiles(fine_net, lum, coarse, device)
