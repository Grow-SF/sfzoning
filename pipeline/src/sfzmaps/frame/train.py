"""Training the learned frame finder's two passes (see `predict` for how they read a page).

Training reads a page exactly as prediction does, so batch norm and the borders see the same thing in both:

- The page pass (`train_coarse`) sees whole pages on the coarse grid, padded with paper to the COARSE_SIDE square
  (`predict.coarse_input`), and learns the label's mask resized to that grid with Pillow BOX: a soft target, each
  coarse px the fraction of its page px inside the map.
- The pixel pass (`train_fine`) sees crops at full resolution, fed `[lum, prior]`, each a `predict.TILE` x
  `predict.TILE` core with `predict._margin()` px of context around it, as `predict.predict_tiles` reads a tile; it
  learns the mask on the core only.  Half the cores centre on a px within `scores.BAND_PX * 4` of the label's
  outline, where the hard part is; the other half anywhere on the page.

Both losses are binary cross-entropy on logits, each px weighted by `paper_weight + ink_weight(lum)` (the ink counts,
as in the scores, and paper a little so the net still says something there).  Px outside the page (padding) and the
pixel pass's margins weigh 0.  Every sample is augmented (`augment`): quarter turns and mirror flips of every array
alike, and brightness, contrast, gamma and blur changes of the luminance alone, the blur measured in page px on either
grid; the pixel pass's crops are also scaled by up to `SCALE_JITTER` either way.  The page pass needs no scale change:
every page is resized to `predict.COARSE_SIDE` before it is read.

The learning rate is constant (Adam), so a net trained for n steps is the net a longer run had at step n: the step
count chosen from the validation curves (`choose_steps`) is the one the final model trains for.

Deterministic: each trainer seeds torch and one numpy generator with `recipe.seed`, makes torch use deterministic
algorithms and pins its cpu threads to `THREADS`; every random draw comes from that generator.  On mps the caller
holds the GPU's exclusive turn (`sfzmaps.gpu.gpu_turn`) for the whole run.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from sfzmaps.frame import predict, scores
from sfzmaps.frame.data import SheetId, ink_weight
from sfzmaps.nets.unet import UNet

THREADS = 4
SCALE_JITTER = 0.1      # the pixel pass's crops are scaled by a factor in 1 +- this
BLUR_SIGMA = 1.5        # px, the most a luminance blur spreads
BLUR_SHARE = 0.5        # the share of samples blurred
BRIGHTNESS = 0.1        # the luminance is shifted by up to this either way
CONTRAST = 0.2          # ... stretched about 0.5 by a factor in 1 +- this
GAMMA = 1.5             # ... and raised to a power between 1 / this and this
OUTLINE_REACH = scores.BAND_PX * 4   # px: the outline crops centre within this of the label's outline
WEIGHT_EPS = 1e-12      # the least the loss divides by, so a batch of no weight gives 0 (numerical; decides nothing)


@dataclasses.dataclass(frozen=True)
class Recipe:
    steps: int = 4000
    check_every: int = 250
    lr: float = 1e-3
    batch: int = 4
    paper_weight: float = 0.1
    seed: int = 0


@dataclasses.dataclass
class Page:
    """A page to train or validate on.  `lum` is the luminance (1 = paper), `mask` the label's map area (1 inside),
    `prior` the page pass's map probability at full resolution: needed by the pixel pass, None for the page pass."""
    id: SheetId
    lum: np.ndarray
    mask: np.ndarray | None
    prior: np.ndarray | None


def augment(rng: np.random.Generator, *arrays, scale: float = 1.0) -> tuple:
    """The same random quarter turn and mirror flip of every array, then brightness, contrast and gamma changes and,
    for `BLUR_SHARE` of the draws, a Gaussian blur (sigma up to `BLUR_SIGMA` page px) of the first array (the
    luminance) alone, kept within [0, 1].  `scale` is the arrays' px per page px (the page pass's coarse grid is
    smaller than the page), so the blur is the same on the page whatever grid it is applied on.  The first comes
    back float32; the others keep their dtype."""
    turns, flip = int(rng.integers(4)), bool(rng.integers(2))
    gamma = float(np.exp(rng.uniform(-np.log(GAMMA), np.log(GAMMA))))
    contrast = 1.0 + float(rng.uniform(-CONTRAST, CONTRAST))
    brightness = float(rng.uniform(-BRIGHTNESS, BRIGHTNESS))
    blurred, sigma = rng.random() < BLUR_SHARE, float(rng.uniform(0.0, BLUR_SIGMA)) * scale
    out = []
    for i, a in enumerate(arrays):
        a = np.rot90(np.flip(a, 1) if flip else a, turns)
        if i == 0:
            a = np.clip(a.astype(np.float32), 0.0, 1.0) ** gamma
            a = np.clip((a - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
            if blurred and sigma > 0:
                a = ndimage.gaussian_filter(a, sigma, mode="nearest")
            a = a.astype(np.float32)
        out.append(np.ascontiguousarray(a))
    return tuple(out)


def _settle(seed: int) -> np.random.Generator:
    """Seed torch, make it deterministic and pin its threads; the one numpy generator every draw comes from."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)
    return np.random.default_rng(seed)


def _loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean BCE.  A masked (weighted) sum, never boolean indexing: the backward of indexing has no
    deterministic kernel on mps."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * weight).sum() / weight.sum().clamp(min=WEIGHT_EPS)


def _val_loss(val: list[Page], prob: Callable[[Page], np.ndarray]) -> float:
    """The ink-weighted log-loss over all ink of every validation page, pooled (each page counts by its ink)."""
    num = den = 0.0
    for pg in val:
        w = ink_weight(pg.lum)
        total = float(w.sum(dtype=np.float64))
        if total <= 0:
            continue
        num += scores.log_loss(prob(pg), pg.mask, w, np.ones(pg.lum.shape, bool)) * total
        den += total
    return num / den if den else float("nan")


def _fit(net: UNet, draw: Callable[[], tuple[np.ndarray, ...]], recipe: Recipe, device: str,
         val_loss: Callable[[UNet], float] | None) -> tuple[UNet, dict]:
    """`recipe.steps` Adam steps on batches from `draw` ((x, target, weight) float32 arrays), the validation loss
    every `check_every` steps; the net back in eval mode on cpu.  FloatingPointError on a non-finite loss (read only
    at the checks, so the device is not waited on every step)."""
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=recipe.lr)
    curve = []
    for step in range(1, recipe.steps + 1):
        x, t, w = (torch.from_numpy(a).to(device) for a in draw())
        loss = _loss(net(x), t, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % recipe.check_every == 0:
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"training loss is {loss.item()} at step {step}")
            if val_loss is not None:
                curve.append((step, float(val_loss(net))))
    return net.eval().cpu(), {"val_curve": curve}


def _need(pages: list[Page], what: str, attrs: tuple[str, ...]) -> None:
    for pg in pages:
        for a in attrs:
            if getattr(pg, a) is None:
                raise ValueError(f"{what} page {pg.id} has no {a}")


# --- the page pass ---------------------------------------------------------------------------------------------------

def _coarse_arrays(pg: Page, paper_weight: float) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """((lum, target, weight), scale): the page on the coarse grid `predict.coarse_input` reads it on, without its
    padding, and that grid's px per page px."""
    h, w = pg.lum.shape
    arr, scale = predict.coarse_input(pg.lum)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    lum = arr[:nh, :nw]
    target = predict.coarse_input(pg.mask.astype(np.float32))[0][:nh, :nw]
    return (lum, target, (paper_weight + ink_weight(lum)).astype(np.float32)), scale


def _coarse_batch(rng: np.random.Generator, coarse: list[tuple[tuple[np.ndarray, ...], float]], batch: int):
    """`batch` whole pages, drawn with replacement and augmented, each padded at right and bottom to the
    COARSE_SIDE x COARSE_SIDE square `predict.coarse_input` reads on: paper in the input, weight 0."""
    samples = []
    for _ in range(batch):
        arrays, scale = coarse[int(rng.integers(len(coarse)))]
        samples.append(augment(rng, *arrays, scale=scale))
    side = predict.COARSE_SIDE
    x = np.ones((batch, 1, side, side), np.float32)
    t = np.zeros((batch, 1, side, side), np.float32)
    w = np.zeros((batch, 1, side, side), np.float32)
    for i, (lum, target, weight) in enumerate(samples):
        h, wd = lum.shape
        x[i, 0, :h, :wd], t[i, 0, :h, :wd], w[i, 0, :h, :wd] = lum, target, weight
    return x, t, w


def train_coarse(pages: list[Page], recipe: Recipe, device: str,
                 val: list[Page] | None = None) -> tuple[UNet, dict]:
    """The page pass trained on `pages` (see the module); returns (net in eval mode on cpu, {"val_curve": [(step,
    loss), ...]}), the curve empty without `val`."""
    _need(pages + (val or []), "page-pass", ("mask",))
    rng = _settle(recipe.seed)
    net = predict.new_coarse_net()
    coarse = [_coarse_arrays(pg, recipe.paper_weight) for pg in pages]
    val_loss = None
    if val:
        val_loss = lambda n: _val_loss(val, lambda pg: predict.predict_coarse(n, pg.lum, device))  # noqa: E731
    return _fit(net, lambda: _coarse_batch(rng, coarse, recipe.batch), recipe, device, val_loss)


# --- the pixel pass --------------------------------------------------------------------------------------------------

def _near_outline(mask: np.ndarray) -> np.ndarray:
    """Flat indices of the px within `OUTLINE_REACH` of the mask's outline; empty for a mask with no outline (all
    map or none)."""
    m = mask.astype(bool)
    edge = m & ~ndimage.binary_erosion(m, border_value=1)
    if not edge.any():
        return np.empty(0, np.int64)
    return np.flatnonzero(ndimage.distance_transform_edt(~edge) <= OUTLINE_REACH)


def _window(a: np.ndarray, top: int, left: int, side: int, fill: float) -> np.ndarray:
    """The side x side window of `a` at (top, left), float32, `fill` where it leaves the page."""
    out = np.full((side, side), fill, np.float32)
    h, w = a.shape
    r0, c0, r1, c1 = max(top, 0), max(left, 0), min(top + side, h), min(left + side, w)
    if r1 > r0 and c1 > c0:
        out[r0 - top:r1 - top, c0 - left:c1 - left] = a[r0:r1, c0:c1]
    return out


def _resized(a: np.ndarray, side: int) -> np.ndarray:
    if a.shape == (side, side):
        return a
    return np.asarray(Image.fromarray(a).resize((side, side), Image.BILINEAR), np.float32)


def _fine_crop(rng: np.random.Generator, pg: Page, near: np.ndarray, outline: bool, paper_weight: float):
    """One augmented crop (lum, prior, mask, weight) read as `predict.predict_tiles` reads a tile: a TILE x TILE core
    with `predict._margin()` px of context around it, the margin weighing 0 so only the core is learned.  The window
    is (TILE + 2 margin) / s px (s within 1 +- SCALE_JITTER), centred on a px near the outline (`outline` and the page
    has one) or anywhere on the page, resized to TILE + 2 margin.  Outside the page: paper, prior 0, mask 0, weight
    0."""
    tile, margin = predict.TILE, predict._margin()
    full = tile + 2 * margin
    side = max(1, round(full / (1.0 + float(rng.uniform(-SCALE_JITTER, SCALE_JITTER)))))
    h, w = pg.lum.shape
    if outline and near.size:
        cy, cx = divmod(int(near[int(rng.integers(near.size))]), w)
    else:
        cy, cx = int(rng.integers(h)), int(rng.integers(w))
    top, left = cy - side // 2, cx - side // 2
    lum = _window(pg.lum, top, left, side, 1.0)
    inside = np.zeros((side, side), np.float32)
    inside[max(-top, 0):min(h - top, side), max(-left, 0):min(w - left, side)] = 1.0
    arrays = [_resized(a, full) for a in (lum, _window(pg.prior, top, left, side, 0.0),
                                          _window(pg.mask.astype(np.float32), top, left, side, 0.0),
                                          inside * (paper_weight + ink_weight(lum)))]
    core = np.zeros((full, full), np.float32)
    core[margin:margin + tile, margin:margin + tile] = 1.0
    arrays[3] = arrays[3] * core
    return augment(rng, *arrays)


def _fine_batch(rng: np.random.Generator, pages: list[Page], near: list[np.ndarray], batch: int,
                paper_weight: float):
    """`batch` crops, every other one (from the first) an outline crop; pages drawn with replacement."""
    crops = []
    for i in range(batch):
        k = int(rng.integers(len(pages)))
        crops.append(_fine_crop(rng, pages[k], near[k], i % 2 == 0, paper_weight))
    x = np.stack([np.stack([c[0], c[1]]) for c in crops])
    t = np.stack([c[2] for c in crops])[:, None]
    w = np.stack([c[3] for c in crops])[:, None]
    return x, t, w


def train_fine(pages: list[Page], recipe: Recipe, device: str,
               val: list[Page] | None = None) -> tuple[UNet, dict]:
    """The pixel pass trained on `pages` (each with its page-pass `prior`; see the module); returns (net in eval mode
    on cpu, {"val_curve": [(step, loss), ...]}), the curve empty without `val`."""
    _need(pages + (val or []), "pixel-pass", ("mask", "prior"))
    rng = _settle(recipe.seed)
    net = predict.new_fine_net()
    near = [_near_outline(pg.mask) for pg in pages]
    val_loss = None
    if val:
        val_loss = lambda n: _val_loss(val, lambda pg: predict.predict_tiles(n, pg.lum, pg.prior, device))  # noqa: E731
    return _fit(net, lambda: _fine_batch(rng, pages, near, recipe.batch, recipe.paper_weight), recipe, device,
                val_loss)


# --- step choice -----------------------------------------------------------------------------------------------------

def choose_steps(curves: list[list[tuple[int, float]]]) -> int:
    """The step with the lowest mean validation loss across the curves (the inner folds'), over the steps every
    curve has; ties go to the earlier step.  ValueError if any loss is NaN (a validation set with no ink scores
    NaN, and NaN never compares) or no step is on every curve."""
    if any(np.isnan(v) for c in curves for _, v in c):
        raise ValueError("a validation curve has a NaN loss (no ink in its validation pages?)")
    tables = [dict(c) for c in curves]
    common = sorted(set.intersection(*(set(t) for t in tables))) if tables else []
    if not common:
        raise ValueError("no step is on every validation curve")
    means = [(float(np.mean([t[s] for t in tables])), s) for s in common]
    return min(means)[1]
