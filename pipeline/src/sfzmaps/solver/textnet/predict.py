"""The text detector's chance of text at every px of a sheet's page (`chance`), and with it the kind of text and its
reading direction (`outputs`), from a trained model (`load_model`).

The page's ink is scored TILE x TILE px at a time, each tile seen with PAD px of page around it (so its edges are
scored with their context) and padded with paper to the network's multiple (`model.TextNet.multiple`: even, so a
half-size network can fold it, the page's odd edge too); tiles with no ink are left at 0, and so are tiles with none of
the px asked for, when only some are.  The network is the one its saved recipe names (`load_model`), whatever the
recipe.  On the Mac's GPU the scoring runs in the GPU's exclusive turn (`sfzmaps.gpu`), which the caller takes
(`stage.build_text`, `score_stage.build_scores`, `trial.run_trials`, `reader_stage.build_reader_text`).  The chance
is uint8, the chance of text x 255.

`outputs` scores the tiles as `chance` does and keeps all three of the network's outputs (`Outputs`): the chance (the
same numbers as `chance`), the kind (`data.KINDS`: the softmax of the four kind logits, float16, and the most likely,
from the logits in float32) and the reading direction (the direction output scaled to unit length, float16 cosine and
sine).  The px of a tile not scored get chance 0, kind NO_KIND (255), probabilities 0 and direction (0, 0).
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from sfzmaps.solver.textnet import model, train

TILE = 1024     # px, the side of the square scored at once (even)
PAD = 64        # px of page around each tile that the model sees too (even)
NO_KIND = 255   # the kind of a px not scored (`data.NO_KIND`, the kind outside text in training)
TINY = 1e-12    # the least length a direction output is divided by (its unit vector), so none is divided by 0


def load_model(f: Path, device: str) -> model.TextNet:
    """A trained detector (the file `train.save_model` wrote) on `device`, for scoring: the network its recipe names,
    with its weights.  A file without a recipe is refused (ValueError), not guessed."""
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: train the detector first")
    saved = torch.load(f, map_location="cpu")
    if "recipe" not in saved:
        raise ValueError(f"{f} holds no recipe (it was saved before recipes): train the detector again")
    # Float32 whatever the recipe's precision: training in half precision keeps the weights in float32.
    net = train.build_net(train.Recipe(**saved["recipe"]))
    net.load_state_dict(saved["state_dict"])
    return net.to(device).eval()


@torch.no_grad()
def chance(net: model.TextNet, ink: np.ndarray, device: str, only: np.ndarray | None = None) -> np.ndarray:
    """The chance of text at every px of a page's ink (bool), as uint8 0..255.  `only` (bool, the page's shape): the
    px wanted; a tile with none of them is not scored (left 0), and the px wanted get the chance a whole page's
    scoring gives them."""
    h, w = ink.shape
    if only is not None and only.shape != ink.shape:
        raise ValueError(f"the px wanted are {only.shape}, the page {ink.shape}")
    out = np.zeros((h, w), np.uint8)
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            if only is not None and not only[y:y + TILE, x:x + TILE].any():
                continue
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


class Outputs(NamedTuple):
    """The detector's outputs at every px of a page (see the module)."""
    chance: np.ndarray       # uint8 (h, w): the chance of text x 255, as `chance` gives it
    kind: np.ndarray         # uint8 (h, w): the most likely kind (data.KINDS), NO_KIND where not scored
    kind_p: np.ndarray       # float16 (4, h, w): each kind's probability (the softmax of the kind logits)
    direction: np.ndarray    # float16 (2, h, w): the reading direction's cosine and sine, unit length


@torch.no_grad()
def outputs(net: model.TextNet, ink: np.ndarray, device: str, only: np.ndarray | None = None) -> Outputs:
    """The detector's three outputs (`Outputs`) at every px of a page's ink (bool), tile by tile as `chance` scores
    them (its chance the same numbers).  `only` (bool, the page's shape): the px wanted, as for `chance`.  A network
    trained for text alone has no kind or direction (ValueError)."""
    h, w = ink.shape
    if only is not None and only.shape != ink.shape:
        raise ValueError(f"the px wanted are {only.shape}, the page {ink.shape}")
    out = Outputs(np.zeros((h, w), np.uint8), np.full((h, w), NO_KIND, np.uint8), np.zeros((4, h, w), np.float16),
                  np.zeros((2, h, w), np.float16))
    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            if only is not None and not only[y:y + TILE, x:x + TILE].any():
                continue
            y0, x0 = max(y - PAD, 0), max(x - PAD, 0)
            y1, x1 = min(y + TILE + PAD, h), min(x + TILE + PAD, w)
            win = ink[y0:y1, x0:x1]
            if not win.any():
                continue
            ph, pw = (-win.shape[0]) % net.multiple, (-win.shape[1]) % net.multiple
            xin = torch.from_numpy(np.pad(win, ((0, ph), (0, pw))).astype(np.float32)).to(device)[None, None]
            o = net(xin)[0]
            if o.shape[0] != model.OUTPUTS:
                raise ValueError(f"the network has {o.shape[0]} output, not {model.OUTPUTS}: it was trained for text "
                                 "alone")
            cy, cx = y - y0, x - x0
            th, tw = min(TILE, h - y), min(TILE, w - x)
            o = o[:, cy:cy + th, cx:cx + tw]
            p = torch.sigmoid(o[0]).cpu().numpy()
            out.chance[y:y + th, x:x + tw] = np.round(255 * p).astype(np.uint8)
            logits = o[1:5]
            out.kind[y:y + th, x:x + tw] = torch.argmax(logits, 0).cpu().numpy().astype(np.uint8)
            out.kind_p[:, y:y + th, x:x + tw] = torch.softmax(logits, 0).cpu().numpy().astype(np.float16)
            cs = o[5:7]
            out.direction[:, y:y + th, x:x + tw] = (cs / cs.norm(dim=0, keepdim=True).clamp(min=TINY)).cpu().numpy() \
                .astype(np.float16)
    return out
