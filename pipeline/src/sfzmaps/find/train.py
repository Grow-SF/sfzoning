"""The finder's trainer: tiles drawn fairly across the map eras from the lot-labelled sheets and the training crops
(`find.data`), and the network (`model.FindNet`) fitted to every label they carry as one likelihood.

Each step's batch takes equal numbers of tiles from every era present (the remainder to the eras first in sorted
order), so an era with few labelled px weighs as much as one with many.  A tile's centre is drawn uniformly from its
era's labelled px (`data.labelled`, of the answers trained), the tile shifted to lie inside its item, turned a random
number of quarter turns (the finder is not told which way is up), given the branch's intensity jitter
(`hatchnet.train`: ink and paper levels each moved) and then print-and-scan noise (`noised`, the owner's ruling R29:
these are old, poorly printed and poorly scanned maps).  The noise moves only the luminance, never a label.

The loss (`batch_loss`): every label of the answers trained, summed as one likelihood (`find.losses`: the dense labels
by `dense_nll`, the cut statements by `cut_nll`), divided by the batch's statements (`losses.statements`); a batch with
no statement is skipped and counted.  An answer left out of `Recipe.answers` gets no loss, no gradient and no weight
decay: its row of the network's head stays as it was made.  With `Recipe.unbroken`, the hand-crop tiles add the
branch's unbroken-line terms (`hatchnet.topo`, clDice: 1 - recall and 1 - precision) between the block-edge chance
and the hand block-edge label.  The loss is taken on the CPU, where its sums (`index_add`) are deterministic (as the
branch does).

Every `Recipe.every` steps (and at the last) the weights are a checkpoint: `on_checkpoint(step, state_dict)` scores it
and the highest is kept, the earliest of a tie; with no callback, the last.

Deterministic: torch's generator and two numpy generators (the tiles; the noise, apart so that a run with noise and
one without draw the same tiles) are seeded from `Recipe.seed`, and torch runs deterministic algorithms
(`hatchnet.train.settle_torch`).  On the Mac's GPU the caller holds the GPU's turn (`sfzmaps.gpu`).

Reads nothing itself; `save_run` writes data/find/runs/<run>/model.pt and log.json, `write_noise_examples` a PNG."""
from __future__ import annotations

import dataclasses
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

from sfzmaps.find import cuts as cutlib, data, losses, model
from sfzmaps.find.data import NONE, Labelled
from sfzmaps.solver.hatchnet import topo
from sfzmaps.solver.hatchnet import train as branch

log = logging.getLogger(__name__)

ANSWERS = model.ANSWERS
CHANNEL = {"street": model.STREET, "edge": model.EDGE, "border": model.BORDER}
CUT_ANSWER = {"edge": cutlib.EDGE, "border": cutlib.BORDER}
SPECK_AREA = 2.5            # noise setting: a speck's mean px, half of them 1 px and half 2 x 2


@dataclass(frozen=True)
class Recipe:
    """How the finder is trained.  Engineering settings, the branch's (`hatchnet.train`): the steps, tiles per step,
    the tile's side (px, a multiple of model.MULTIPLE), the network's width, the one-cycle schedule (`max_lr`,
    `pct_start`), AdamW's weight decay, the intensity jitter, and the steps between checkpoints (`every`).  `answers`:
    the answers trained (one left out gets no loss); `unbroken`: add the unbroken-line terms (see the module); `seed`.

    The noise (`noised`): engineering settings too, augmentation strengths, not anything a reading uses.  `noise`
    scales every part (0: none at all); each part's maximum: `noise_blur` the blur's sigma (px), `noise_shift` the
    paper/ink level's move, `noise_gain` the contrast's largest gain, `noise_specks` the share of px that are
    specks, `noise_grain` the grain's standard deviation."""
    steps: int = 3000
    batch: int = 16
    tile: int = 256
    width: int = 16
    max_lr: float = 2e-3
    pct_start: float = 0.3
    weight_decay: float = 1e-4
    jitter: float = 0.25
    every: int = 250
    answers: tuple[str, ...] = ANSWERS
    unbroken: bool = False
    seed: int = 0
    noise: float = 1.0
    noise_blur: float = 1.0
    noise_shift: float = 0.15
    noise_gain: float = 4.0
    noise_specks: float = 0.01
    noise_grain: float = 0.08


@dataclass
class Fitted:
    """The weights kept (on the CPU), the step they are from, the curve (a row at the first step, every `every` steps
    and the last), the device and the count of batches skipped for holding no statement."""
    state_dict: dict
    step: int
    curve: list[dict]
    device: str
    skipped: int = 0


# --- the noise --------------------------------------------------------------------------------------------------------

def noised(lum: np.ndarray, rng: np.random.Generator, recipe: Recipe) -> np.ndarray:
    """`lum` (float32, 1 paper 0 ink) with print-and-scan noise, kept in [0, 1]; with `recipe.noise` 0, `lum` itself.
    Each part's amount is drawn uniformly between 0 and its maximum times `recipe.noise`, in this order: a soft scan
    (Gaussian blur); light or heavy printing (clip(0.5 + (lum - 0.5 - shift) gain): shift either way, gain from 1 up);
    specks of dust and broken ink (blobs of 1 or 2 x 2 px turned to the other end, 1 - lum); and grain (Gaussian
    grey noise)."""
    if recipe.noise == 0:
        return lum
    a = recipe.noise
    blur = rng.uniform(0, a * recipe.noise_blur)
    shift = rng.uniform(-a * recipe.noise_shift, a * recipe.noise_shift)
    gain = 1 + rng.uniform(0, a * (recipe.noise_gain - 1))
    specks = rng.uniform(0, a * recipe.noise_specks)
    grain = rng.uniform(0, a * recipe.noise_grain)
    x = ndimage.gaussian_filter(np.asarray(lum, np.float32), blur) if blur > 0 else np.asarray(lum, np.float32)
    x = np.clip(0.5 + (x - 0.5 - shift) * gain, 0, 1)
    seed = rng.random(x.shape) < specks / SPECK_AREA
    big = seed & (rng.random(x.shape) < 0.5)
    speck = seed.copy()
    speck[1:] |= big[:-1]
    speck[:, 1:] |= big[:, :-1]
    speck[1:, 1:] |= big[:-1, :-1]
    x = np.where(speck, 1 - x, x)
    x = x + rng.normal(0, 1, x.shape) * grain
    return np.clip(x, 0, 1).astype(np.float32)


# --- tiles ------------------------------------------------------------------------------------------------------------

class _Era:
    """An era's labelled px: tile centres drawn uniformly from them, without listing them (a sheet has millions)."""

    def __init__(self, items: list[int], masks: list[np.ndarray]):
        self.items = items
        self.masks = masks
        self.rows = [np.cumsum(m.sum(axis=1), dtype=np.int64) for m in masks]       # labelled px up to each row
        self.ends = np.cumsum([int(r[-1]) for r in self.rows], dtype=np.int64)

    @property
    def total(self) -> int:
        return int(self.ends[-1]) if len(self.ends) else 0

    def centre(self, rng: np.random.Generator) -> tuple[int, int, int]:
        """(item, row, col) of a labelled px drawn uniformly."""
        k = int(rng.integers(self.total))
        i = int(np.searchsorted(self.ends, k, side="right"))
        k -= int(self.ends[i - 1]) if i else 0
        rows = self.rows[i]
        r = int(np.searchsorted(rows, k, side="right"))
        k -= int(rows[r - 1]) if r else 0
        return self.items[i], r, int(np.flatnonzero(self.masks[i][r])[k])


class Tiles:
    """Training tiles drawn from `items` (see the module).  ValueError for an item smaller than the tile (naming it),
    a tile side that is not a multiple of model.MULTIPLE, and an era with no labelled px."""

    def __init__(self, items: list[Labelled], recipe: Recipe):
        side = recipe.tile
        if side % model.MULTIPLE:
            raise ValueError(f"the tile's side {side} is not a multiple of {model.MULTIPLE}")
        if not items:
            raise ValueError("nothing to train on")
        for it in items:
            h, w = it.lum.shape
            if h < side or w < side:
                raise ValueError(f"{it.name}: {h} x {w} px is smaller than a {side} px tile")
        self.items, self.recipe = items, recipe
        self.eras = sorted({it.era for it in items})
        self.pools = {}
        for era in self.eras:
            mine = [i for i, it in enumerate(items) if it.era == era]
            pool = _Era(mine, [data.labelled(items[i], recipe.answers) for i in mine])
            if not pool.total:
                raise ValueError(f"era {era}: no labelled px")
            self.pools[era] = pool
        self.boxes = [None if it.cuts is None else data.cut_boxes(it.cuts, it.lum.shape[1]) for it in items]

    def shares(self) -> list[int]:
        """Tiles per era (in `eras` order): equal, the remainder to the first."""
        n, extra = divmod(self.recipe.batch, len(self.eras))
        return [n + (i < extra) for i in range(len(self.eras))]

    def tile(self, era: str, rng: np.random.Generator) -> Labelled:
        """A tile of `era` turned a random number of quarter turns, before jitter and noise."""
        side = self.recipe.tile
        i, r, c = self.pools[era].centre(rng)
        it = self.items[i]
        h, w = it.lum.shape
        y0, x0 = min(max(r - side // 2, 0), h - side), min(max(c - side // 2, 0), w - side)
        return data.turned(data.window(it, y0, x0, side, self.boxes[i]), int(rng.integers(4)))

    def draw(self, rng: np.random.Generator, noise_rng: np.random.Generator) -> list[Labelled]:
        """One batch: each era's share of tiles (`shares`), era by era, each jittered (from `rng`) then noised (from
        `noise_rng`)."""
        out = []
        j = self.recipe.jitter
        for era, n in zip(self.eras, self.shares()):
            for _ in range(n):
                t = self.tile(era, rng)
                lo, hi = rng.uniform(0.0, j), rng.uniform(1.0 - j, 1.0)
                lum = (lo + (hi - lo) * t.lum).astype(np.float32)
                out.append(dataclasses.replace(t, lum=noised(lum, noise_rng, self.recipe)))
        return out


# --- the loss ---------------------------------------------------------------------------------------------------------

def _dense(tiles: list[Labelled], name: str) -> tuple[np.ndarray, np.ndarray]:
    """(n, h, w) targets (1 yes) and weights (0 no label) of answer `name`."""
    lab = np.stack([getattr(t, name) for t in tiles])
    if name == "street":
        weight = np.where(lab != NONE, np.stack([t.street_w for t in tiles]), 0).astype(np.float32)
    else:
        weight = (lab != NONE).astype(np.float32)
    return (lab == 1).astype(np.float32), weight


def _cuts(tiles: list[Labelled], answer: int):
    """The tiles' cuts of `answer`, one after the other: px into the batch's flat (n, h, w), cut, prior, says,
    weight."""
    px, cut, prior, says, weight = [], [], [], [], []
    n = 0
    for b, t in enumerate(tiles):
        c = t.cuts
        if c is None:
            continue
        sel = c.answer == answer
        on = sel[c.cut]
        renumber = np.cumsum(sel) - 1 + n
        px.append(c.px[on] + b * t.lum.size)
        cut.append(renumber[c.cut[on]])
        prior.append(c.prior[on])
        says.append(c.says[sel])
        weight.append(c.weight[sel])
        n += int(sel.sum())
    if not n:
        return None
    return (np.concatenate(px).astype(np.int64), np.concatenate(cut).astype(np.int64),
            np.concatenate(prior).astype(np.float32), np.concatenate(says), np.concatenate(weight).astype(np.float32))


def batch_loss(net_out: torch.Tensor, tiles: list[Labelled], recipe: Recipe) -> torch.Tensor:
    """Minus the log likelihood of every label of the answers in `recipe.answers` on `tiles`, summed (the dense labels
    by `losses.dense_nll`, each cut by `losses.cut_nll` with its weight), under the logits `net_out` (n, 3, h, w);
    divided by nothing (`statements` counts them)."""
    total = net_out.new_zeros(())
    for name in recipe.answers:
        z = net_out[:, CHANNEL[name]]
        target, weight = _dense(tiles, name)
        total = total + losses.dense_nll(z, torch.from_numpy(target).to(z.device),
                                         torch.from_numpy(weight).to(z.device))
        found = _cuts(tiles, CUT_ANSWER[name]) if name in CUT_ANSWER else None
        if found is not None:
            px, cut, prior, says, weight = (torch.from_numpy(a).to(z.device) for a in found)
            total = total + losses.cut_nll(z.reshape(-1), px, cut, prior, says, weight)
    return total


def statements(tiles: list[Labelled], recipe: Recipe) -> float:
    """The statements `batch_loss` sums for `tiles` (`losses.statements` of every weight it uses)."""
    weights = []
    for name in recipe.answers:
        weights.append(_dense(tiles, name)[1])
        found = _cuts(tiles, CUT_ANSWER[name]) if name in CUT_ANSWER else None
        if found is not None:
            weights.append(found[4])
    return losses.statements(*weights)


def unbroken_loss(net_out: torch.Tensor, tiles: list[Labelled], recipe: Recipe) -> torch.Tensor:
    """The branch's unbroken-line terms (`hatchnet.topo.cldice_sums`) on the hand-crop tiles, pooled: 1 - recall
    (how much of the hand block-edge label the block-edge chance covers) + 1 - precision (how much of the chance's
    soft skeleton, over the px with a block-edge label, lies on the hand label); the skeleton half a tile deep (the
    branch's).  0 with no hand tile or with the block-edge answer left out."""
    hand = [i for i, t in enumerate(tiles) if t.hand]
    if not hand or "edge" not in recipe.answers:
        return net_out.new_zeros(())
    p = torch.sigmoid(net_out[torch.tensor(hand), model.EDGE])
    edge = torch.from_numpy(np.stack([tiles[i].edge for i in hand])).to(p.device)
    on, region = (edge == 1).to(p.dtype), (edge != NONE).to(p.dtype)
    s = topo.cldice_sums(p, on, on, recipe.tile // 2, region)
    return (1 - topo.ratio(s[0], s[1])) + (1 - topo.ratio(s[2], s[3]))


# --- the fit ----------------------------------------------------------------------------------------------------------

def _decay_keep(net: model.FindNet, answers) -> list[tuple[torch.nn.Parameter, torch.Tensor | float]]:
    """Each weight and how much of AdamW's decay it takes: 1, or for the head 0 on the rows of answers left out."""
    left = [CHANNEL[a] for a in ANSWERS if a not in answers]
    out = []
    for name, p in net.named_parameters():
        if name.startswith("head.") and left:
            keep = torch.ones_like(p)
            keep[left] = 0
            out.append((p, keep))
        else:
            out.append((p, 1.0))
    return out


def _state(net: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _zero() -> dict:
    """The sums a curve row is made of, since the last row."""
    return {"loss": 0.0, "statements": 0.0, "unbroken": 0.0, "steps": 0, "skipped": 0}


def fit(items: list[Labelled], recipe: Recipe, device, on_checkpoint=None) -> Fitted:
    """The network trained on `items` (see the module).  The network is made right after torch's generator is
    seeded.  ValueError for an answer not in model.ANSWERS; FloatingPointError for a loss that is not finite."""
    bad = sorted(set(recipe.answers) - set(ANSWERS))
    if bad or not recipe.answers:
        raise ValueError(f"answers must be some of {', '.join(ANSWERS)}; got {', '.join(recipe.answers) or 'none'}")
    branch.settle_torch()
    torch.manual_seed(recipe.seed)
    net = model.FindNet(recipe.width)
    tiles = Tiles(items, recipe)
    rng, noise_rng = np.random.default_rng(recipe.seed), np.random.default_rng([recipe.seed, 1])
    net.to(device).train()
    # AdamW's decoupled weight decay (p <- p (1 - lr decay) before the Adam step, as torch's AdamW does), applied here
    # so that an answer left out keeps its head row exactly as made
    opt = torch.optim.AdamW(net.parameters(), lr=recipe.max_lr, weight_decay=0.0)
    keep = _decay_keep(net, recipe.answers)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=recipe.max_lr, total_steps=recipe.steps,
                                                pct_start=recipe.pct_start)
    curve, skipped, best, best_value, last_row = [], 0, None, None, 0
    sums = _zero()
    for step in range(1, recipe.steps + 1):
        batch = tiles.draw(rng, noise_rng)
        n = statements(batch, recipe)
        row_step = None
        if n > 0:
            x = torch.from_numpy(np.stack([t.lum for t in batch]))[:, None].to(device)
            out = net(x).cpu()
            nll = batch_loss(out, batch, recipe)
            loss = nll / n
            extra = unbroken_loss(out, batch, recipe) if recipe.unbroken else None
            if extra is not None:
                loss = loss + extra
            if not torch.isfinite(loss):
                raise FloatingPointError(f"step {step}: the loss is not finite ({float(loss.detach())})")
            opt.zero_grad()
            loss.backward()
            with torch.no_grad():
                lr = opt.param_groups[0]["lr"]
                for p, k in keep:
                    p.mul_(1 - lr * recipe.weight_decay * k)
            opt.step()
            sched.step()
            row_step = float(nll.detach()) / n
            sums["loss"] += float(nll.detach())
            sums["statements"] += n
            sums["unbroken"] += float(extra.detach()) if extra is not None else 0.0
            sums["steps"] += 1
        else:
            skipped += 1
            sums["skipped"] += 1
        checkpoint = step % recipe.every == 0 or step == recipe.steps
        if step == 1 or checkpoint:
            row = {"step": step, "lr": float(opt.param_groups[0]["lr"]), "first_step": last_row + 1,
                   "loss_per_statement": sums["loss"] / sums["statements"] if sums["statements"] else math.nan,
                   "statements": sums["statements"], "step_loss_per_statement": row_step,
                   "skipped": sums["skipped"]}
            if recipe.unbroken:
                row["unbroken"] = sums["unbroken"] / sums["steps"] if sums["steps"] else math.nan
            sums = _zero()
            last_row = step
            if checkpoint:
                state = _state(net)
                value = on_checkpoint(step, state) if on_checkpoint is not None else None
                if value is not None:
                    row["chosen_by"] = float(value)
                if on_checkpoint is None or (value is not None and math.isfinite(value)
                                             and (best_value is None or value > best_value)):
                    best, best_value = (step, state), value
            curve.append(row)
            log.info("steps %d-%d: loss per statement %.4f (%.0f statements), %d skipped", row["first_step"], step,
                     row["loss_per_statement"], row["statements"], row["skipped"])
    if best is None:                                 # no checkpoint scored a finite value: the last
        best = (recipe.steps, _state(net))
    return Fitted(best[1], best[0], curve, str(device), skipped)


# --- the run's files --------------------------------------------------------------------------------------------------

def save_run(folder: Path, fitted: Fitted, recipe: Recipe, log_fields: dict) -> tuple[Path, Path]:
    """model.pt (the weights kept, their step, the recipe, the device: no clock time) and log.json (the recipe, the
    device, the curve, the skipped batches and `log_fields`) in `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    recipe_d = dataclasses.asdict(recipe) | {"answers": list(recipe.answers)}
    pt, js = folder / "model.pt", folder / "log.json"
    part = folder / ".model.pt.part"
    torch.save({"state_dict": fitted.state_dict, "step": fitted.step, "recipe": recipe_d, "device": fitted.device},
               part)
    part.replace(pt)
    out = {"recipe": recipe_d, "device": fitted.device, "step_kept": fitted.step, **log_fields,
           "curve": fitted.curve, "skipped_batches": fitted.skipped}
    js.write_text(json.dumps(out, indent=1) + "\n")
    return pt, js


ZOOM = 2                    # picture setting: the noise examples' enlargement (nearest neighbour)


def write_noise_examples(items: list[Labelled], recipe: Recipe, path: Path, per_era: int = 2) -> Path:
    """`per_era` training tiles of every era, each as trained without noise (turned and jittered) beside the same with
    noise, in rows, to judge by eye whether the noise looks like these maps."""
    from sfzmaps.find import look as pictures
    tiles = Tiles(items, recipe)
    rng, noise_rng = np.random.default_rng([recipe.seed, 2]), np.random.default_rng([recipe.seed, 3])
    rows, j = [], recipe.jitter
    for era in tiles.eras:
        for _ in range(per_era):
            t = tiles.tile(era, rng)
            lo, hi = rng.uniform(0.0, j), rng.uniform(1.0 - j, 1.0)
            clean = (lo + (hi - lo) * t.lum).astype(np.float32)
            noisy = noised(clean, noise_rng, recipe)
            pair = [(np.clip(a, 0, 1) * 255).astype(np.uint8) for a in (clean, noisy)]
            img = np.hstack([pair[0], np.full((clean.shape[0], 8), 255, np.uint8), pair[1]])
            img = cv2.resize(img, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
            band = np.full((28, img.shape[1]), 255, np.uint8)
            cv2.putText(band, f"{era}: {t.name}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
            rows.append(np.vstack([band, img, np.full((10, img.shape[1]), 255, np.uint8)]))
    img = cv2.cvtColor(np.vstack(rows), cv2.COLOR_GRAY2BGR)
    img = pictures._keyed(img, "training tiles: left as trained without noise (turned, jittered), right with noise",
                          ())
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path
