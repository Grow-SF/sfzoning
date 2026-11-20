"""Whole-page prediction for the learned frame finder: a page pass on the whole page at low resolution (every output
px sees the whole page), then a pixel pass at full resolution over tiles, fed `[lum, prior]` where `prior` is the page
pass's probability that a px is map."""
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sfzmaps.nets.unet import UNet, multiple, reach

COARSE_SIDE = 1024
COARSE_LEVELS = 9
COARSE_WIDTH = 8
FINE_LEVELS = 5
FINE_WIDTH = 16
TILE = 512


def new_coarse_net() -> UNet:
    return UNet(1, 1, COARSE_WIDTH, COARSE_LEVELS)


def new_fine_net() -> UNet:
    return UNet(2, 1, FINE_WIDTH, FINE_LEVELS)


def coarse_input(lum: np.ndarray) -> tuple[np.ndarray, float]:
    """The page resized (Pillow BOX) so its long side is COARSE_SIDE, padded with paper (1.0) at the right and bottom
    to a COARSE_SIDE x COARSE_SIDE square: every page, either way up, is read with the same frame of padding, in
    training and in prediction alike.  Returns the array and the scale factor."""
    if COARSE_SIDE % multiple(COARSE_LEVELS):
        raise ValueError(f"COARSE_SIDE {COARSE_SIDE} is not a multiple of {multiple(COARSE_LEVELS)}")
    h, w = lum.shape
    scale = COARSE_SIDE / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(Image.fromarray(lum.astype(np.float32), mode="F").resize((nw, nh), Image.BOX), np.float32)
    out = np.ones((COARSE_SIDE, COARSE_SIDE), np.float32)
    out[:nh, :nw] = small
    return out, scale


@contextmanager
def _eval_on(net: UNet, device: str):
    """The net in eval mode on `device`, put back as the caller had it (train/eval mode and device) afterwards."""
    was = net.training
    home = next(net.parameters()).device
    net.to(device).eval()
    try:
        yield net
    finally:
        net.train(was)
        net.to(home)


def _run(net: UNet, x: np.ndarray, device: str) -> np.ndarray:
    """Sigmoid probability of a (c, h, w) array, float32.  The net must be in eval mode (see `_eval_on`)."""
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(device)
        return torch.sigmoid(net(t))[0, 0].float().cpu().numpy()


def predict_coarse(net: UNet, lum: np.ndarray, device: str = "cpu") -> np.ndarray:
    """The page pass's probability, cropped of padding and resized (bilinear) to the page's shape."""
    h, w = lum.shape
    arr, scale = coarse_input(lum)
    with _eval_on(net, device):
        prob = _run(net, arr[None], device)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    prob = prob[:nh, :nw]
    t = torch.from_numpy(np.ascontiguousarray(prob))[None, None]
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)[0, 0].numpy().astype(np.float32)


def _margin() -> int:
    """reach(FINE_LEVELS) rounded up to the pooling grid, so every tile and the whole-page run pool on the same grid."""
    m = multiple(FINE_LEVELS)
    return -(-reach(FINE_LEVELS) // m) * m


def _padded(lum: np.ndarray, prior: np.ndarray, margin: int, ph: int, pw: int) -> np.ndarray:
    """(2, ph, pw): the page at (margin, margin), paper (1.0) in lum and not-map (0.0) in prior around it."""
    h, w = lum.shape
    x = np.zeros((2, ph, pw), np.float32)
    x[0] = 1.0
    x[0, margin:margin + h, margin:margin + w] = lum
    x[1, margin:margin + h, margin:margin + w] = prior
    return x


def _predict_padded_whole(net: UNet, lum: np.ndarray, prior: np.ndarray, device: str = "cpu") -> np.ndarray:
    """The reference tiling must equal: one run on the page paper-padded by the margin on every side, then at right and
    bottom to a multiple of multiple(FINE_LEVELS), cropped back to the page."""
    h, w = lum.shape
    m, mg = multiple(FINE_LEVELS), _margin()
    ph, pw = -(-(h + 2 * mg) // m) * m, -(-(w + 2 * mg) // m) * m
    with _eval_on(net, device):
        out = _run(net, _padded(lum, prior, mg, ph, pw), device)
    return out[mg:mg + h, mg:mg + w]


def predict_tiles(net: UNet, lum: np.ndarray, prior: np.ndarray, device: str = "cpu") -> np.ndarray:
    """The pixel pass's probability at full resolution.  TILE x TILE cores, each read with a margin of reach px of
    page or paper around it; only the core is kept."""
    h, w = lum.shape
    mg = _margin()
    th, tw = -(-h // TILE) * TILE, -(-w // TILE) * TILE
    x = _padded(lum, prior, mg, th + 2 * mg, tw + 2 * mg)
    out = np.zeros((th, tw), np.float32)
    with _eval_on(net, device):
        for y in range(0, th, TILE):
            for xx in range(0, tw, TILE):
                p = _run(net, x[:, y:y + TILE + 2 * mg, xx:xx + TILE + 2 * mg], device)
                out[y:y + TILE, xx:xx + TILE] = p[mg:mg + TILE, mg:mg + TILE]
    return out[:h, :w]


def predict_page(coarse_net: UNet, fine_net: UNet, lum: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """(coarse_prob, map_prob), both float32 of the page's shape."""
    coarse = predict_coarse(coarse_net, lum, device)
    return coarse, predict_tiles(fine_net, lum, coarse, device)
