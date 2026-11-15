"""The text detector's chance of text at every px of every sheet's page, from the model that held out one quarter.

The page's ink is scored TILE x TILE px at a time, each tile seen with PAD px of page around it (so its edges are
scored with their context) and padded with paper to the network's multiple (`model.TextNet.multiple`: even, so a
half-size network can fold it, the page's odd edge too); tiles with no ink are left at 0.  The network is the one its
saved recipe names (`load_model`), whatever the recipe.
On the Mac's GPU the scoring runs in the GPU's exclusive turn (`sfzmaps.gpu`), which the caller takes
(`stage.build_text`).
Writes data/solver/text-<q>/chance/<sheet>.png: uint8, the chance of text x 255.
"""
import logging
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from sfzmaps.solver import grid, paths
from sfzmaps.solver.textnet import data, model, train

log = logging.getLogger(__name__)

TILE = 1024     # px, the side of the square scored at once (even)
PAD = 64        # px of page around each tile that the model sees too (even)


def load_model(f: Path, device: str) -> model.TextNet:
    """A trained detector (the file `train.save_model` wrote) on `device`, for scoring: the network its recipe names,
    with its weights.  A file without a recipe is refused (ValueError), not guessed."""
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: train the detector first")
    saved = torch.load(f, map_location="cpu")
    if "recipe" not in saved:
        raise ValueError(f"{f} holds no recipe (it was saved before recipes): train the detector again")
    net = train.build_net(train.Recipe(**saved["recipe"]))
    net.load_state_dict(saved["state_dict"])
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
            ph, pw = (-win.shape[0]) % net.multiple, (-win.shape[1]) % net.multiple
            xin = torch.from_numpy(np.pad(win, ((0, ph), (0, pw))).astype(np.float32)).to(device)[None, None]
            p = torch.sigmoid(net(xin)[0, 0, :win.shape[0], :win.shape[1]]).cpu().numpy()
            cy, cx = y - y0, x - x0
            th, tw = min(TILE, h - y), min(TILE, w - x)
            out[y:y + th, x:x + tw] = np.round(255 * p[cy:cy + th, cx:cx + tw]).astype(np.uint8)
    return out


def predict_sheets(q: str, device: str | None = None) -> int:
    """Score every sheet's page (`paths.SHEETS`, once they are the sheets the warp placed: `paths.check_sheets`)
    with the model that held out `q` and write `paths.chance_dir(q)/<sheet>.png` (the folder is made afresh once the
    model has loaded).  On `mps` the caller holds the GPU's exclusive turn (`gpu.gpu_turn`).  Returns the number of
    sheets scored."""
    names = paths.check_sheets()
    device = device or train.default_device()
    train.settle_torch()
    net = load_model(paths.text_dir(q) / "model.pt", device)
    out = paths.chance_dir(q)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for name in names:
        t0 = time.perf_counter()
        ink = data.load_page(name) < grid.INK_BELOW
        if not cv2.imwrite(str(out / f"{name}.png"), chance(net, ink, device)):
            raise OSError(f"can't write {out / f'{name}.png'}")
        log.info("%s scored (%.0f s)", name, time.perf_counter() - t0)
    return len(names)
