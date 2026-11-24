"""Reading a page with the finder: the network's two chances (street, block edge) at every px.

`read_page` reads a page TILE px square at a time, each square seen with CONTEXT px of page round it (the network's
whole reach, model.REACH, rounded up to model.MULTIPLE) and the page's end padded with paper to model.MULTIPLE, as one
pass over the whole page pads it: the squares' corners and the context are multiples of model.MULTIPLE, so the
poolings fall on the whole page's grid and every px gets the numbers one pass over the whole page gives it (as
`hatchnet.train` reads a sheet).  A hand crop is read as `read_page` of the crop grown as the trainer shows it
(`data.grown`; `find.counts`).

A read is stored as uint8 levels round(255 p) (`save`, `load`; `stored` gives the same levels in memory), so every
reading downstream (`find.areas`, `find.counts`) sees only the 256 levels a stored read can tell apart.  A read stored
before the finder dropped its zoning-border answer (2026-11-23) holds a third key, which `load` does not read.

Reads nothing itself; `save` writes the path its caller gives (data/find/runs/<run>/reads/...)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from sfzmaps.find import model
from sfzmaps.solver.textread.store import save_npz

TILE = 1024                 # engineering setting: the side of the square read at once, px (memory only)
CONTEXT = math.ceil(model.REACH / model.MULTIPLE) * model.MULTIPLE     # px of page seen round each square (128)
LEVELS = 255                # a stored chance is round(LEVELS p) / LEVELS


@torch.no_grad()
def read_page(net: model.FindNet, lum: np.ndarray, keep: np.ndarray, device, tile: int = TILE) -> np.ndarray:
    """Float32 (2, h, w): the chance of each answer (model.ANSWERS order) at every px of the page `lum` (1 paper, 0 ink,
    as stored), 0 off `keep` (the frame cut's map area; the scan is shown there all the same).  The network is moved
    to `device` and put in eval mode.  ValueError for a `tile` that is not a multiple of model.MULTIPLE."""
    if tile % model.MULTIPLE or tile <= 0:
        raise ValueError(f"the tile must be a positive multiple of {model.MULTIPLE}, not {tile}")
    lum = np.asarray(lum, np.float32)
    keep = np.asarray(keep, bool)
    if keep.shape != lum.shape:
        raise ValueError(f"keep is {keep.shape}, the page {lum.shape}")
    net = net.to(device).eval()
    h, w = lum.shape
    out = np.zeros((len(model.ANSWERS), h, w), np.float32)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, x0 = max(y - CONTEXT, 0), max(x - CONTEXT, 0)
            y1, x1 = min(y + tile + CONTEXT, h), min(x + tile + CONTEXT, w)
            win = lum[y0:y1, x0:x1]
            win = np.pad(win, ((0, -win.shape[0] % model.MULTIPLE), (0, -win.shape[1] % model.MULTIPLE)),
                         constant_values=1.0)                          # paper
            p = torch.sigmoid(net(torch.from_numpy(np.ascontiguousarray(win))[None, None].to(device)))[0]
            p = p.cpu().numpy()
            th, tw = min(tile, h - y), min(tile, w - x)
            out[:, y:y + th, x:x + tw] = p[:, y - y0:y - y0 + th, x - x0:x - x0 + tw]
    out[:, ~keep] = 0
    return out


def levels(chances: np.ndarray) -> np.ndarray:
    """Uint8: round(255 p) of each chance."""
    return np.rint(np.clip(np.asarray(chances, np.float32), 0, 1) * LEVELS).astype(np.uint8)


def stored(chances: np.ndarray) -> np.ndarray:
    """Float32: the chances as a stored read gives them back (`levels` / 255)."""
    return levels(chances).astype(np.float32) / LEVELS


def save(path: Path, chances: np.ndarray) -> Path:
    """The read (2, h, w) at `path` (a compressed .npz, its folders made), as `levels` under the keys street and
    edge."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lv = levels(chances)
    save_npz(path, **{name: lv[i] for i, name in enumerate(model.ANSWERS)})
    return path


def load(path: Path) -> np.ndarray:
    """Float32 (2, h, w): a read `save` stored, each level / 255 (the keys street and edge; any other key is not
    read)."""
    with np.load(path) as z:
        return np.stack([z[name] for name in model.ANSWERS]).astype(np.float32) / LEVELS
