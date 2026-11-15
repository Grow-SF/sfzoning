"""The text detector's chance of text at every px of every sheet's page, from the model that held out one quarter.

The page's ink is scored TILE x TILE px at a time, each tile seen with PAD px of page around it (so its edges are
scored with their context) and padded with paper to a multiple of `model.MULTIPLE`; tiles with no ink are left at 0.
Writes data/solver/text-<q>/chance/<sheet>.png: uint8, the chance of text x 255.
"""
import logging
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from sfzmaps.solver import page, paths
from sfzmaps.solver.textnet import data, model, train

log = logging.getLogger(__name__)

TILE = 1024     # px, the side of the square scored at once
PAD = 64        # px of page around each tile that the model sees too


def load_model(q: str, device: str) -> model.TextNet:
    """The model trained with quarter `q` held out (`text_dir(q)/model.pt`), on `device`, for scoring."""
    f = paths.text_dir(q) / "model.pt"
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: train it first (`sfzmaps solver text {q}`)")
    net = model.TextNet()
    net.load_state_dict(torch.load(f, map_location="cpu")["state_dict"])
    return net.to(device).eval()


@torch.no_grad()
def chance(net: model.TextNet, ink: np.ndarray, device: str) -> np.ndarray:
    """The chance of text at every px of a page's ink (bool), as uint8 0..255."""
    h, w = ink.shape
    out = np.zeros((h, w), np.uint8)
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            y0, x0 = max(y - PAD, 0), max(x - PAD, 0)
            y1, x1 = min(y + TILE + PAD, h), min(x + TILE + PAD, w)
            win = ink[y0:y1, x0:x1]
            if not win.any():
                continue
            ph, pw = (-win.shape[0]) % model.MULTIPLE, (-win.shape[1]) % model.MULTIPLE
            xin = torch.from_numpy(np.pad(win, ((0, ph), (0, pw))).astype(np.float32)).to(device)[None, None]
            p = torch.sigmoid(net(xin)[0, 0, :win.shape[0], :win.shape[1]]).cpu().numpy()
            cy, cx = y - y0, x - x0
            th, tw = min(TILE, h - y), min(TILE, w - x)
            out[y:y + th, x:x + tw] = np.round(255 * p[cy:cy + th, cx:cx + tw]).astype(np.uint8)
    return out


def chance_dir(q: str) -> Path:
    """Where the chance of text of each sheet's page goes, for the model that held out `q`."""
    return paths.text_dir(q) / "chance"


def predict_sheets(q: str, device: str | None = None) -> None:
    """Score every warped sheet's page with the model that held out `q` and write `chance/<sheet>.png` (the folder
    is made afresh), in the device's `train.gpu_turn`."""
    device = device or train.default_device()
    train.settle_torch()
    out = chance_dir(q)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    with train.gpu_turn(device):
        net = load_model(q, device)
        for name in paths.sheet_names():
            t0 = time.perf_counter()
            ink = data.load_page(name) < page.INK_BELOW
            if not cv2.imwrite(str(out / f"{name}.png"), chance(net, ink, device)):
                raise OSError(f"can't write {out / f'{name}.png'}")
            log.info("%s scored (%.0f s)", name, time.perf_counter() - t0)
