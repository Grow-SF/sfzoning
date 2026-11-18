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

A network whose recipe has `input="grey"` (`train.build_net` sets `net.input`) is given the page's grey levels (uint8)
in the `ink` argument of `chance`, `outputs` and `word_outputs`: over 255 as training gives them, a tile's padding white
paper, and no tile skipped for having no ink (a grey page has none to look for).

`word_outputs` is for a network with the word heads (`train.Recipe(heads="words")`, `model.WORD_OUTPUTS` channels, or
"words_aff", `model.WORD_AFF_OUTPUTS`, whose same-word affinity is `WordOutputs.aff`, or "words_code",
`model.WORD_CODE_OUTPUTS`, whose px's likeliest code class is `WordOutputs.code` and its chance `WordOutputs.code_p`):
the arrays `wordread.words.core_words` takes (`WordOutputs`), the page's tiles stitched into one page.  Each tile is
seen with `context` px of page around it, the network's whole reach (its receptive field), not PAD: so every px kept
gets the numbers one pass over the whole page would give it (the tile's own edges and the paper it is padded with
are beyond its reach), and a word on a tile's edge is one word's votes on both sides.  The tile's corner and the
context are multiples of the network's `multiple`, so its pooling falls on the whole page's grid, and its end is
padded as the whole page's is.  A px of a tile not scored is not a word's: its core logit is NOT_SCORED (a chance of
exactly 0 in double precision), its other outputs 0.  `chance` and `outputs` are unchanged.
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


def _grey(net: model.TextNet) -> bool:
    """Whether the network reads the page's grey levels (its recipe's `input`), not the 1-bit ink."""
    return getattr(net, "input", "ink") == "grey"


def _tile_input(net: model.TextNet, win: np.ndarray, device: str) -> torch.Tensor:
    """A tile's input (1, 1, h, w) on `device`, padded to the network's `multiple`.  An ink network's is the ink
    (bool) as 1 and 0, padded with paper (0), as ever.  A grey network's is the page's grey levels (uint8: ValueError
    otherwise) over 255, as training gives them (`data.crop_targets(..., grey=True)`), padded with white paper (1)."""
    ph, pw = (-win.shape[0]) % net.multiple, (-win.shape[1]) % net.multiple
    if _grey(net):
        if win.dtype != np.uint8:
            raise ValueError(f"a network trained on grey levels reads the page's uint8 grey levels, not {win.dtype}")
        arr = np.pad(win.astype(np.float32) / 255, ((0, ph), (0, pw)), constant_values=1.0)
    else:
        arr = np.pad(win, ((0, ph), (0, pw))).astype(np.float32)
    return torch.from_numpy(arr).to(device)[None, None]


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
            if not _grey(net) and not win.any():
                continue
            xin = _tile_input(net, win, device)
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
            if not _grey(net) and not win.any():
                continue
            xin = _tile_input(net, win, device)
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


# --- the word heads ---------------------------------------------------------------------------------------------------

NOT_SCORED = -1e4    # the core logit of a px not scored: sigmoid(-1e4) is 0 in double precision (no word's), finite
#                      so a log of it stays a number; float16 holds it exactly
NO_CODE = 255        # the code class of a px not scored (no class: the classes are 0 .. model.CODE_CLASSES - 1)


class WordOutputs(NamedTuple):
    """A word-heads network's outputs at every px of a page (see the module), as `wordread.words.core_words` takes
    them."""
    chance: np.ndarray       # uint8 (h, w): the chance of text x 255 (channel 0), as `chance` gives it
    kind_p: np.ndarray       # float16 (4, h, w): each kind's probability (the softmax of the kind logits)
    core: np.ndarray         # float32 (h, w): the word-core logit (`model.CORE`); NOT_SCORED where not scored
    geo: np.ndarray          # float32 (5, h, w): `model.word_geometry`: the offset in px from the px to its word's
    #                          centre (x right, y down), the word's length and height in px, and its axis in degrees in
    #                          [0, 180) (no reading sign: 0/180 is the reader's)
    tag: np.ndarray          # float32 (k, h, w): the word tag (`model.TAG`)
    aff: np.ndarray | None = None
    #                          float32 (2, h, w), a "words_aff" network's alone (None for "words"): the same-word
    #                          affinity logits (`model.AFF`), [0] of the px and the px to its right, [1] of the px and
    #                          the px below; NOT_SCORED where the px is not scored, and on the page's last column ([0])
    #                          and last row ([1]), whose pair has no second px
    code: np.ndarray | None = None
    #                          uint8 (h, w), a "words_code" network's alone (None otherwise): the px's likeliest class
    #                          of the code head (`model.CODE`, the argmax of its logits: 0 the background, 1 another
    #                          text, 2 on a code, `wordread.codes.classes`); NO_CODE where the px is not scored
    code_p: np.ndarray | None = None
    #                          float16 (h, w), with `code`: that class's chance (the softmax over the classes); 0 where
    #                          not scored


def context(net: model.TextNet) -> int:
    """The px of page each side of a px that can move its outputs (the network's receptive field's reach), rounded
    up to the network's `multiple`: per level, its convolutions (one px each at that level's px, two a block down, two
    or one a block up), its 2 x 2 pooling (one px of the level before) and its nearest upsampling (one px of the level
    below); at half size all of it at twice the px, and the fold's one px more."""
    convs_up = 2 if isinstance(net.up[0], torch.nn.Sequential) and len(net.up[0]) == 6 else 1
    levels = len(net.down)
    reach = sum(2 * 2 ** i for i in range(levels))                    # the blocks down
    reach += sum(2 ** (i - 1) for i in range(1, levels))              # the poolings before them
    reach += sum(convs_up * 2 ** i for i in range(levels - 1))        # the blocks up
    reach += sum(2 ** i for i in range(levels - 1))                   # the upsamplings before them
    if net.half_size:
        reach = model.FOLD * reach + 1
    return -(-reach // net.multiple) * net.multiple


@torch.no_grad()
def word_outputs(net: model.TextNet, ink: np.ndarray, device: str, only: np.ndarray | None = None,
                 tile: int = TILE) -> WordOutputs:
    """A word-heads network's outputs (`WordOutputs`) at every px of a page's ink (bool), `tile` x `tile` px at a time
    (a multiple of the network's `multiple`), each tile seen with `context(net)` px of page around it and the tiles
    stitched (see the module).  `only` (bool, the page's shape): the px wanted; a tile with none of them is not scored.
    A network without the word heads is refused (ValueError).  A "words_aff" network's affinity is stitched as the
    rest (`WordOutputs.aff`; None for a "words" network), and a "words_code" network's likeliest code class and its
    chance (`WordOutputs.code`, `code_p`; None for the others: the class's logits are not kept, a page's would be
    CODE_CLASSES floats a px)."""
    h, w = ink.shape
    if only is not None and only.shape != ink.shape:
        raise ValueError(f"the px wanted are {only.shape}, the page {ink.shape}")
    if tile % net.multiple:
        raise ValueError(f"a tile is a multiple of the network's {net.multiple} px, not {tile}")
    pad = context(net)
    k = model.TAG.stop - model.TAG.start
    channels = net.head.out_channels // (model.FOLD ** 2 if net.half_size else 1)
    coded = channels == model.WORD_CODE_OUTPUTS
    out = WordOutputs(np.zeros((h, w), np.uint8), np.zeros((4, h, w), np.float16),
                      np.full((h, w), NOT_SCORED, np.float32), np.zeros((5, h, w), np.float32),
                      np.zeros((k, h, w), np.float32),
                      np.full((2, h, w), NOT_SCORED, np.float32) if channels == model.WORD_AFF_OUTPUTS else None,
                      np.full((h, w), NO_CODE, np.uint8) if coded else None,
                      np.zeros((h, w), np.float16) if coded else None)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if only is not None and not only[y:y + tile, x:x + tile].any():
                continue
            y0, x0 = max(y - pad, 0), max(x - pad, 0)
            y1, x1 = min(y + tile + pad, h), min(x + tile + pad, w)
            win = ink[y0:y1, x0:x1]
            xin = _tile_input(net, win, device)
            o = net(xin)[0]
            if o.shape[0] not in (model.WORD_OUTPUTS, model.WORD_AFF_OUTPUTS, model.WORD_CODE_OUTPUTS):
                raise ValueError(f"the network has {o.shape[0]} outputs, not {model.WORD_OUTPUTS}, "
                                 f"{model.WORD_AFF_OUTPUTS} or {model.WORD_CODE_OUTPUTS}: it has no word heads")
            cy, cx = y - y0, x - x0
            th, tw = min(tile, h - y), min(tile, w - x)
            o = o[:, cy:cy + th, cx:cx + tw].float()
            win_out = (slice(y, y + th), slice(x, x + tw))
            out.chance[win_out] = np.round(255 * torch.sigmoid(o[0]).cpu().numpy()).astype(np.uint8)
            out.kind_p[(slice(None), *win_out)] = torch.softmax(o[1:5], 0).cpu().numpy().astype(np.float16)
            out.core[win_out] = o[model.CORE].cpu().numpy()
            out.geo[(slice(None), *win_out)] = model.word_geometry(o).cpu().numpy()
            out.tag[(slice(None), *win_out)] = o[model.TAG].cpu().numpy()
            if out.aff is not None:
                out.aff[(slice(None), *win_out)] = o[model.AFF].cpu().numpy()
            if coded:
                p, c = torch.softmax(o[model.CODE], 0).max(0)
                out.code[win_out] = c.cpu().numpy().astype(np.uint8)
                out.code_p[win_out] = p.cpu().numpy().astype(np.float16)
    if out.aff is not None:
        out.aff[0, :, -1] = NOT_SCORED         # no px to the right of the last column, none below the last row
        out.aff[1, -1, :] = NOT_SCORED
    return out
