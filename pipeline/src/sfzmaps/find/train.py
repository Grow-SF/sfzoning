"""The finder's trainer: tiles drawn fairly across the map eras from the lot-labelled sheets and the training crops
(`find.data`), and the network (`model.FindNet`) fitted to every label they carry as one likelihood.

Each step's batch takes equal numbers of tiles from every era present (the remainder to the eras first in sorted
order), so an era with few labelled px weighs as much as one with many.  A tile's centre is drawn uniformly from its
era's labelled px (`data.labelled`).  With `Recipe.resize`, the tile is then shown at a print size drawn from the
sizes measured on the sheets the finder is used on (`data.print_sizes`): one series, each equally likely, then one of
its sheets, each equally likely; the factor is the item's ft per px over that sheet's, and the square around the
centre that becomes one tile at that factor is resized to the tile (`data.resized`: labels and cuts with it).  When the
square is bigger than the item, the tile is taken at the nearest factor whose square fits (counted).  The finder is
never told a scale: it learns every size.  The square (the tile) is shifted to lie inside its item, turned a random
number of quarter turns (the finder is not told which way is up), given the branch's intensity jitter
(`hatchnet.train`: ink and paper levels each moved) and then print-and-scan noise (`noised`, the owner's ruling R29:
these are old, poorly printed and poorly scanned maps).  The noise moves only the luminance, never a label.

The loss (`batch_loss`): every label of both answers, summed as one likelihood (`find.losses`: the dense street and
block-edge labels by `dense_nll`, the block-edge cut statements by `cut_nll`), divided by the batch's statements
(`losses.statements`); a batch with no statement is skipped and counted.  With `Recipe.unbroken`, the hand-crop tiles
add the branch's unbroken-line terms (`hatchnet.topo`, clDice: 1 - recall and 1 - precision) between the block-edge
chance and the hand block-edge label.  The loss is taken on the CPU, where its sums (`index_add`) are deterministic
(as the branch does).

The weights kept are the last step's.  Every `Recipe.every` steps (and at the last) `on_checkpoint(step, state_dict)`
scores the weights on the choosing crops (`counts.trace_scorer`: a dict such as `choose_found_share`,
`choose_panoptic_quality`, `choose_depth`); it goes into the curve row as the run's held-out trace and chooses
nothing.

Deterministic: torch's generator and three numpy generators (the tiles; the noise, apart so that a run with noise and
one without draw the same tiles; the print sizes, apart so that a run without resizing draws the tiles of before) are
seeded from `Recipe.seed`, and torch runs deterministic algorithms
(`hatchnet.train.settle_torch`).  On the Mac's GPU the caller holds the GPU's turn (`sfzmaps.gpu`).

Reads nothing itself; `save_run` writes data/find/runs/<run>/model.pt and log.json, `write_noise_examples`
data/find/runs/<run>/noise-examples.png (the path its caller gives)."""
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

from sfzmaps.find import data, handlabels, losses, model
from sfzmaps.find.data import NONE, Labelled
from sfzmaps.solver.hatchnet import topo
from sfzmaps.solver.hatchnet import train as branch

log = logging.getLogger(__name__)

SPECK_AREA = 2.5            # noise setting: a speck's mean px, half of them 1 px and half 2 x 2


@dataclass(frozen=True)
class Recipe:
    """How the finder is trained.  Engineering settings, the branch's (`hatchnet.train`): the steps, tiles per step,
    the tile's side (px, a multiple of model.MULTIPLE), the network's width, the one-cycle schedule (`max_lr`,
    `pct_start`), AdamW's weight decay, the intensity jitter, and the steps between checkpoints (`every`).  `unbroken`:
    add the unbroken-line terms (see the module); `resize`: show each tile at a print size drawn from the measured
    sizes (see the module; off: every tile at its item's own size, as before); `seed`.

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
    unbroken: bool = False
    resize: bool = True
    seed: int = 0
    noise: float = 1.0
    noise_blur: float = 1.0
    noise_shift: float = 0.15
    noise_gain: float = 4.0
    noise_specks: float = 0.01
    noise_grain: float = 0.08


@dataclass
class Fitted:
    """The weights kept (on the CPU: the last step's), their step, the curve (a row at the first step, every `every`
    steps and the last), the device, the count of batches skipped for holding no statement and the factors the tiles
    were shown at (`Tiles.factor_summary`; None without resizing)."""
    state_dict: dict
    step: int
    curve: list[dict]
    device: str
    skipped: int = 0
    factors: dict | None = None


# --- the noise --------------------------------------------------------------------------------------------------------

def noise_amounts(rng: np.random.Generator, recipe: Recipe) -> tuple[float, float, float, float, float]:
    """One tile's noise (`noised`): the blur's sigma, the level's shift, the gain, the share of specks and the
    grain's standard deviation, each drawn uniformly between 0 (gain 1; shift its negative maximum) and its maximum
    times `recipe.noise`."""
    a = recipe.noise
    return (rng.uniform(0, a * recipe.noise_blur), rng.uniform(-a * recipe.noise_shift, a * recipe.noise_shift),
            1 + rng.uniform(0, a * (recipe.noise_gain - 1)), rng.uniform(0, a * recipe.noise_specks),
            rng.uniform(0, a * recipe.noise_grain))


def noised(lum: np.ndarray, rng: np.random.Generator, recipe: Recipe) -> np.ndarray:
    """`lum` (float32, 1 paper 0 ink) with print-and-scan noise, kept in [0, 1]; with `recipe.noise` 0, `lum` itself.
    Each part's amount is drawn per call, so per tile (`noise_amounts`), in this order: a soft scan
    (Gaussian blur); light or heavy printing (clip(0.5 + (lum - 0.5 - shift) gain): shift either way, gain from 1 up);
    specks of dust and broken ink (blobs of 1 or 2 x 2 px turned to the other end, 1 - lum); and grain (Gaussian
    grey noise)."""
    if recipe.noise == 0:
        return lum
    blur, shift, gain, specks, grain = noise_amounts(rng, recipe)
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
    """Training tiles drawn from `items` (see the module), at the print sizes `sizes` (series -> each sheet's ft per
    px, `data.print_sizes`) with `recipe.resize`.  `factors` lists the factor of every tile drawn and `moved` counts
    those moved to fit their item.  ValueError for an item smaller than the tile or one of a held-out set
    (`handlabels.HELD_OUT`: a choosing or test crop), naming it; a tile side that is not a multiple of model.MULTIPLE;
    an era with no labelled px; with `recipe.resize`, no sizes, or an item with no ft per px or with cuts and no
    sigma."""

    def __init__(self, items: list[Labelled], recipe: Recipe, sizes: dict[str, list[float]] | None = None):
        side = recipe.tile
        if side % model.MULTIPLE:
            raise ValueError(f"the tile's side {side} is not a multiple of {model.MULTIPLE}")
        if not items:
            raise ValueError("nothing to train on")
        for it in items:
            h, w = it.lum.shape
            if h < side or w < side:
                raise ValueError(f"{it.name}: {h} x {w} px is smaller than a {side} px tile")
            if it.set in handlabels.HELD_OUT:
                raise ValueError(f"{it.name}: a {it.set} crop is never trained on")
        self.items, self.recipe = items, recipe
        self.eras = sorted({it.era for it in items})
        self.pools = {}
        for era in self.eras:
            mine = [i for i, it in enumerate(items) if it.era == era]
            pool = _Era(mine, [data.labelled(items[i]) for i in mine])
            if not pool.total:
                raise ValueError(f"era {era}: no labelled px")
            self.pools[era] = pool
        self.boxes = [None if it.cuts is None else data.cut_boxes(it.cuts, it.lum.shape[1]) for it in items]
        self.sizes, self.size_rng = None, None
        self.factors: list[float] = []
        self.moved = 0
        if recipe.resize:
            self.sizes = {s: [float(v) for v in vs] for s, vs in sorted((sizes or {}).items()) if len(vs)}
            if not self.sizes:
                raise ValueError("resizing the tiles needs the measured print sizes (`data.print_sizes`); none given")
            for it in items:
                if not (math.isfinite(it.ft_per_px) and it.ft_per_px > 0):
                    raise ValueError(f"{it.name}: no ft per px; its tiles cannot be shown at another print size")
                if it.cuts is not None and not (math.isfinite(it.sigma) and it.sigma > 0):
                    raise ValueError(f"{it.name}: its cuts have no sigma; they cannot be laid again at another size")
            self.size_rng = np.random.default_rng([recipe.seed, 2])

    def shares(self) -> list[int]:
        """Tiles per era (in `eras` order): equal, the remainder to the first."""
        n, extra = divmod(self.recipe.batch, len(self.eras))
        return [n + (i < extra) for i in range(len(self.eras))]

    def target(self) -> tuple[str, float]:
        """A print size to show a tile at, from the size generator: a series, each equally likely, then one of its
        sheets, each equally likely; (series, ft per px)."""
        names = list(self.sizes)
        series = names[int(self.size_rng.integers(len(names)))]
        found = self.sizes[series]
        return series, found[int(self.size_rng.integers(len(found)))]

    def tile(self, era: str, rng: np.random.Generator) -> Labelled:
        """A tile of `era` (with `recipe.resize`, at a drawn print size) turned a random number of quarter turns,
        before jitter and noise."""
        side = self.recipe.tile
        i, r, c = self.pools[era].centre(rng)
        it = self.items[i]
        h, w = it.lum.shape
        src = side
        if self.recipe.resize:
            # the square that becomes one tile at the factor f (the item's ft per px over the target's) is side / f px,
            # a whole number of px (at least one); when it is bigger than the item, the item's shorter side
            src = max(1, round(side * self.target()[1] / it.ft_per_px))
            if src > min(h, w):
                src = min(h, w)
                self.moved += 1
            self.factors.append(side / src)
        y0, x0 = min(max(r - src // 2, 0), h - src), min(max(c - src // 2, 0), w - src)
        return data.turned(data.resized(it, y0, x0, src, side, self.boxes[i]), int(rng.integers(4)))

    def factor_summary(self) -> dict | None:
        """The factors drawn so far: count, least, median, most, and how many were moved to fit their item; None
        without resizing."""
        if not self.recipe.resize:
            return None
        f = np.asarray(self.factors, float)
        if not len(f):
            return {"count": 0, "least": None, "median": None, "most": None, "moved_to_fit": self.moved}
        return {"count": len(f), "least": float(f.min()), "median": float(np.median(f)), "most": float(f.max()),
                "moved_to_fit": self.moved}

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


def _cuts(tiles: list[Labelled]):
    """The tiles' block-edge cuts, one after the other: px into the batch's flat (n, h, w), cut, prior, weight; None
    when there is none."""
    px, cut, prior, weight = [], [], [], []
    n = 0
    for b, t in enumerate(tiles):
        c = t.cuts
        if c is None:
            continue
        px.append(c.px + b * t.lum.size)
        cut.append(c.cut.astype(np.int64) + n)
        prior.append(c.prior)
        weight.append(c.weight)
        n += len(c.weight)
    if not n:
        return None
    return (np.concatenate(px).astype(np.int64), np.concatenate(cut), np.concatenate(prior).astype(np.float32),
            np.concatenate(weight).astype(np.float32))


def batch_loss(net_out: torch.Tensor, tiles: list[Labelled]) -> torch.Tensor:
    """Minus the log likelihood of every label on `tiles`, summed (the dense street and block-edge labels by
    `losses.dense_nll`, each block-edge cut by `losses.cut_nll` with its weight, saying an edge crosses), under the
    logits `net_out` (n, 2, h, w); divided by nothing (`statements` counts them)."""
    total = net_out.new_zeros(())
    for name, channel in (("street", model.STREET), ("edge", model.EDGE)):
        z = net_out[:, channel]
        target, weight = _dense(tiles, name)
        total = total + losses.dense_nll(z, torch.from_numpy(target).to(z.device),
                                         torch.from_numpy(weight).to(z.device))
    found = _cuts(tiles)
    if found is not None:
        z = net_out[:, model.EDGE]
        px, cut, prior, weight = (torch.from_numpy(a).to(z.device) for a in found)
        says = torch.ones(len(weight), dtype=torch.bool, device=z.device)
        total = total + losses.cut_nll(z.reshape(-1), px, cut, prior, says, weight)
    return total


def statements(tiles: list[Labelled]) -> float:
    """The statements `batch_loss` sums for `tiles` (`losses.statements` of every weight it uses)."""
    weights = [_dense(tiles, name)[1] for name in ("street", "edge")]
    found = _cuts(tiles)
    if found is not None:
        weights.append(found[3])
    return losses.statements(*weights)


def corridor(edge: np.ndarray) -> np.ndarray:
    """float32: 1 on the band a hand block edge's printed line can fill, the edge label (1) drawn as wide as the hand
    tool draws a line (`handlabels.BARRIER`, a square), 0 elsewhere.  The edge label is the outermost px of a block (or
    a stroke's px), so the middle of the printed line, where a predicted line's skeleton runs, lies a px or so in."""
    on = edge == 1
    if not on.any():
        return on.astype(np.float32)
    return ndimage.binary_dilation(on, np.ones((handlabels.BARRIER, handlabels.BARRIER), bool)).astype(np.float32)


def unbroken_loss(net_out: torch.Tensor, tiles: list[Labelled], recipe: Recipe) -> torch.Tensor:
    """The branch's unbroken-line terms (`hatchnet.topo.cldice_sums`) on the hand-crop tiles, pooled, as the branch
    uses them (a thin line for the recall, a wider corridor for the precision): 1 - recall (how much of the hand
    block-edge label, edge 1, the block-edge chance covers) + 1 - precision (how much of the chance's soft skeleton,
    over the px with a block-edge label, lies in the edge's `corridor`); the skeleton half a tile deep (the branch's).
    0 with no hand tile."""
    hand = [i for i, t in enumerate(tiles) if t.hand]
    if not hand:
        return net_out.new_zeros(())
    p = torch.sigmoid(net_out[torch.tensor(hand), model.EDGE])
    labels = np.stack([tiles[i].edge for i in hand])
    edge = torch.from_numpy(labels).to(p.device)
    on, region = (edge == 1).to(p.dtype), (edge != NONE).to(p.dtype)
    band = torch.from_numpy(np.stack([corridor(e) for e in labels])).to(p.device, p.dtype)
    s = topo.cldice_sums(p, on, band, recipe.tile // 2, region)
    return (1 - topo.ratio(s[0], s[1])) + (1 - topo.ratio(s[2], s[3]))


# --- the fit ----------------------------------------------------------------------------------------------------------

def _state(net: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _zero() -> dict:
    """The sums a curve row is made of, since the last row."""
    return {"loss": 0.0, "statements": 0.0, "unbroken": 0.0, "steps": 0, "skipped": 0}


def generators(recipe: Recipe) -> tuple[np.random.Generator, np.random.Generator]:
    """The tiles' and the noise's generators, from `recipe.seed` (the sizes' is `Tiles`' own, [seed, 2])."""
    return np.random.default_rng(recipe.seed), np.random.default_rng([recipe.seed, 1])


def fit(items: list[Labelled], recipe: Recipe, device, on_checkpoint=None,
        sizes: dict[str, list[float]] | None = None) -> Fitted:
    """The network trained on `items` (see the module), its tiles at the print sizes `sizes` with `recipe.resize`.
    The network is made right after torch's generator is seeded.  FloatingPointError for a loss that is not finite."""
    branch.settle_torch()
    torch.manual_seed(recipe.seed)
    net = model.FindNet(recipe.width)
    tiles = Tiles(items, recipe, sizes)
    rng, noise_rng = generators(recipe)
    net.to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=recipe.max_lr, weight_decay=recipe.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=recipe.max_lr, total_steps=recipe.steps,
                                                pct_start=recipe.pct_start)
    curve, skipped, last_row = [], 0, 0
    sums = _zero()
    for step in range(1, recipe.steps + 1):
        batch = tiles.draw(rng, noise_rng)
        n = statements(batch)
        row_step = None
        if n > 0:
            x = torch.from_numpy(np.stack([t.lum for t in batch]))[:, None].to(device)
            out = net(x).cpu()
            nll = batch_loss(out, batch)
            loss = nll / n
            extra = unbroken_loss(out, batch, recipe) if recipe.unbroken else None
            if extra is not None:
                loss = loss + extra
            if not torch.isfinite(loss):
                raise FloatingPointError(f"step {step}: the loss is not finite ({float(loss.detach())})")
            opt.zero_grad()
            loss.backward()
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
            if checkpoint and on_checkpoint is not None:         # the held-out trace: it chooses nothing
                row.update(on_checkpoint(step, _state(net)) or {})
            curve.append(row)
            log.info("steps %d-%d: loss per statement %.4f (%.0f statements), %d skipped", row["first_step"], step,
                     row["loss_per_statement"], row["statements"], row["skipped"])
    return Fitted(_state(net), recipe.steps, curve, str(device), skipped, tiles.factor_summary())


# --- the run's files --------------------------------------------------------------------------------------------------

def save_run(folder: Path, fitted: Fitted, recipe: Recipe, log_fields: dict) -> tuple[Path, Path]:
    """model.pt (the weights kept, their step, the recipe, the device: no clock time) and log.json (the recipe, the
    device, the curve, the skipped batches, the factors drawn when resizing and `log_fields`; a number that is not
    finite as null) in `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    recipe_d = dataclasses.asdict(recipe)
    pt, js = folder / "model.pt", folder / "log.json"
    part = folder / ".model.pt.part"
    torch.save({"state_dict": fitted.state_dict, "step": fitted.step, "recipe": recipe_d, "device": fitted.device},
               part)
    part.replace(pt)
    out = {"recipe": recipe_d, "device": fitted.device, "step_kept": fitted.step, **log_fields,
           "curve": fitted.curve, "skipped_batches": fitted.skipped}
    if fitted.factors is not None:
        out["factors"] = fitted.factors
    js.write_text(json.dumps(_finite(out), indent=1, allow_nan=False) + "\n")
    return pt, js


def _finite(v):
    """`v` with every float that is not finite (NaN, infinity) as None: JSON has no such numbers."""
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


ZOOM = 2                    # picture setting: the noise examples' enlargement (nearest neighbour)


def write_noise_examples(items: list[Labelled], recipe: Recipe, path: Path, per_era: int = 2,
                         sizes: dict[str, list[float]] | None = None) -> Path:
    """`per_era` training tiles of every era (at the print sizes `sizes` with `recipe.resize`), each as trained
    without noise (turned and jittered) beside the same with noise, in rows, to judge by eye whether the noise looks
    like these maps."""
    tiles = Tiles(items, recipe, sizes)
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
    title = np.full((36, rows[0].shape[1]), 255, np.uint8)
    cv2.putText(title, "training tiles: left as trained without noise (turned, jittered), right with noise", (6, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    img = np.vstack([title, *rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path
