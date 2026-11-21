"""Training the frame finder's one network on whole pages (issue #71, spec parts 3-6).

Each step is one whole page at full resolution, read exactly as prediction reads a page (`model.pad`: paper at the
bottom and right up to multiples of `model.multiple()`), so there is no window edge that training sees and
prediction does not.  The step:

1. draws a labelled page uniformly, then its cut, its scale and its augmentation, in that order, from one generator;
2. cuts the page: each side by a random 0 .. that side's labelled frame depth (`side_depths`, `draw_cut`), so at the
   extreme the frame is gone and map ink runs to the page edge (a frameless side), the label cut alike;
3. scales the cut page by a factor in 1 +- `SCALE_JITTER` (bilinear for the luminance, nearest for the label);
4. turns and flips it, changes its brightness, contrast and gamma, and blurs it now and then (`apply_aug`);
5. takes one Adam step on the weighted log-loss (`_loss`) of "outside the map" (the target is 1 - the label's inside).

The weight of a px is `paper_weight + (1 - lum)`, from the luminance after the scale and before the photometric
changes, and 0 on the padding.  It never depends on the label: a weight that does moves the loss's minimiser away
from the true chance (spec part 4), and the decision downstream needs calibrated chances.  So there is no uncertain
strip along the outline either.

The learning rate is constant, with no schedule: the net after s steps of a longer run is then exactly the net of a
run of s steps, so the held-out loss at a check is the score of "train s steps", which the final model repeats
(spec part 6).  At every multiple of `check_every` the run logs the training loss averaged since the last check, the
held-out loss (`held_loss`) when held pages are given, and saves a snapshot when asked.  The held-out loss is a log
and a curve only: it changes no weight.

Where the work runs (issue #71's speed measurement: a device-side bilinear rescale of a whole page is 63 ms of a
213 ms step): the cpu draws every random value, cuts the page and scales it, from the page's luminance and label kept
as 8-bit Pillow images (`_Source`; the pages are 8-bit scans, so the luminance loses nothing) and handed over as
uint8 arrays (`_cut_and_scale`); the device does the turns and flips (of those bytes, a sixth of the time of
moving float32 arrays), the photometric changes, the blur and the padding (`_on_device`).  With `Recipe.prefetch` a
background thread (`_batches`) does the cpu side up to `PREFETCH` steps ahead while the device steps.

Deterministic: the trainer seeds torch and one numpy generator with `recipe.seed`, makes torch use deterministic
algorithms and pins its cpu threads to `THREADS` (`_settle`); every random value comes from that generator, drawn in
a fixed order by one thread (the prefetching one, or the training's own), so the weights are the same with or
without prefetch, and with or without held pages.  On mps the caller holds the GPU's exclusive turn
(`sfzmaps.gpu.gpu_turn`) for the whole run.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from sfzmaps.frame import model
from sfzmaps.frame.data import SheetId

log = logging.getLogger(__name__)

# From the two-pass trainer this replaces (`frame/train.py` at e580a00), unchanged:
THREADS = 4
SCALE_JITTER = 0.1      # each drawn page is scaled by a factor in 1 +- this (inherited, not measured: spec part 5)
BLUR_SIGMA = 1.5        # px, the most a luminance blur spreads
BLUR_SHARE = 0.5        # the share of samples blurred
BRIGHTNESS = 0.1        # the luminance is shifted by up to this either way
CONTRAST = 0.2          # ... stretched about 0.5 by a factor in 1 +- this
GAMMA = 1.5             # ... and raised to a power between 1 / this and this
BLUR_TRUNCATE = 4.0     # the blur's kernel reaches this many sigmas each way (scipy's gaussian_filter default)
WEIGHT_EPS = 1e-12      # the least the loss divides by, so a batch of no weight gives 0 (numerical; decides nothing)
PREFETCH = 2            # steps drawn ahead with `Recipe.prefetch`

HELD_EPS = 1e-6         # the held-out chances are clipped to [this, 1 - this], as `scores.log_loss` clips (numerical)


@dataclasses.dataclass(frozen=True)
class Recipe:
    """`steps` Adam steps at the constant rate `lr`, checked every `check_every` (1,000 gives the spec's four
    candidate step counts, 1,000 to 4,000); `paper_weight` the weight of a px of plain paper (ink adds 1 - lum);
    `prefetch` does the next steps' cpu side in a background thread: the same draws, so it never changes the
    weights."""
    steps: int = 4000
    check_every: int = 1000
    lr: float = 1e-3
    paper_weight: float = 0.1
    seed: int = 0
    prefetch: bool = True


@dataclasses.dataclass
class Page:
    """A page to train on or hold out.  `lum` is the float32 luminance (1 = paper), `inside` the label's map area
    (bool, True inside)."""
    id: SheetId
    lum: np.ndarray
    inside: np.ndarray | None


# --- augmentation: drawn on the cpu, applied on the device (from `frame/train.py` at e580a00, unchanged) -------------

@dataclasses.dataclass(frozen=True)
class Aug:
    """One sample's augmentation (`draw_aug`): `turns` quarter turns after a mirror `flip`, the luminance raised to
    `gamma`, stretched by `contrast` about 0.5, shifted by `brightness`, and blurred by `sigma` px of the arrays it
    is applied to (0: not blurred)."""
    turns: int
    flip: bool
    gamma: float
    contrast: float
    brightness: float
    sigma: float


def draw_aug(rng: np.random.Generator, scale: float = 1.0) -> Aug:
    """One sample's augmentation drawn from `rng`, always the same seven draws in the same order.  `scale` is the
    arrays' px per page px (the page pass's coarse grid is smaller than the page), so the blur is the same on the page
    whatever grid it is applied on; `BLUR_SHARE` of the draws are blurred."""
    turns, flip = int(rng.integers(4)), bool(rng.integers(2))
    gamma = float(np.exp(rng.uniform(-np.log(GAMMA), np.log(GAMMA))))
    contrast = 1.0 + float(rng.uniform(-CONTRAST, CONTRAST))
    brightness = float(rng.uniform(-BRIGHTNESS, BRIGHTNESS))
    blurred, sigma = rng.random() < BLUR_SHARE, float(rng.uniform(0.0, BLUR_SIGMA)) * scale
    return Aug(turns, flip, gamma, contrast, brightness, sigma if blurred else 0.0)


def _blur(a: torch.Tensor, sigma: float) -> torch.Tensor:
    """A Gaussian blur of a (h, w) tensor, as `scipy.ndimage.gaussian_filter(a, sigma, mode="nearest")`: a separable
    kernel `BLUR_TRUNCATE` sigmas each way, the edge px repeated beyond the edge."""
    radius = int(BLUR_TRUNCATE * sigma + 0.5)
    if radius == 0:
        return a
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).to(device=a.device, dtype=a.dtype)
    t = a[None, None]
    t = F.conv2d(F.pad(t, (radius, radius, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, radius, radius), mode="replicate"), k.view(1, 1, -1, 1))
    return t[0, 0]


def apply_aug(aug: Aug, lum: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """`aug` applied to tensors on any device, the luminance (h, w) and the others (..., h, w) (stacked planes move
    together): the same flip and turns of every plane (exact: a permutation of the px, so a weight of 0 stays 0),
    then the gamma, contrast and brightness changes and the blur of the luminance alone, kept within [0, 1].  The
    luminance comes back float32; the others keep their dtype."""
    return (_photometric(aug, _move(aug, lum)).contiguous(), *(_move(aug, a).contiguous() for a in others))


def _move(aug: Aug, a: torch.Tensor) -> torch.Tensor:
    """`aug`'s flip, then its quarter turns, of a (..., h, w) tensor (a view where torch can)."""
    return torch.rot90(torch.flip(a, (-1,)) if aug.flip else a, aug.turns, (-2, -1))


def _photometric(aug: Aug, lum: torch.Tensor) -> torch.Tensor:
    """`aug`'s gamma, contrast and brightness changes and its blur, of a (h, w) luminance, float32 within [0, 1]."""
    lum = lum.float().clamp(0.0, 1.0) ** aug.gamma
    lum = ((lum - 0.5) * aug.contrast + 0.5 + aug.brightness).clamp(0.0, 1.0)
    if aug.sigma > 0:
        lum = _blur(lum, aug.sigma)
    return lum


# --- shared pieces (from `frame/train.py` at e580a00, unchanged) -----------------------------------------------------

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


@contextlib.contextmanager
def _batches(draw: Callable[[], object], n: int, prefetch: bool) -> Iterator[Iterator[object]]:
    """An iterator over `n` results of `draw()`, in order.  With `prefetch`, a background thread calls `draw` up to
    `PREFETCH` ahead (and alone calls it meanwhile, so the same draws come in the same order).  An error in that
    thread is raised where the draw is asked for; leaving the block, by an error or not, stops and joins it."""
    if not prefetch:
        yield (draw() for _ in range(n))
        return
    ready, stop = queue.Queue(PREFETCH), threading.Event()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                ready.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def produce():
        try:
            for _ in range(n):
                if not put(draw()):
                    return
        except BaseException as e:      # noqa: BLE001 - handed to the training's thread, raised there
            put(e)

    def consume():
        for _ in range(n):
            item = ready.get()
            if isinstance(item, BaseException):
                raise item
            yield item

    worker = threading.Thread(target=produce, name="frame-prefetch", daemon=True)
    worker.start()
    try:
        yield consume()
    finally:
        stop.set()
        worker.join()


# --- the cut: how deep each side's frame is --------------------------------------------------------------------------

Cut = tuple[int, int, int, int]     # px taken off the (top, bottom, left, right) of a page


def side_depths(inside: np.ndarray) -> Cut:
    """(top, bottom, left, right): for each side, the median over that side's lines (the columns for top and bottom,
    the rows for left and right) of the distance in px from the page's edge to the line's first inside px.  Lines
    with no inside px are left out; a side with none gives 0.  The lower median (one line's own depth), so an even
    count of lines never gives a half px."""
    m = np.asarray(inside, bool)

    def depth(a: np.ndarray, axis: int) -> int:
        hit = a.any(axis=axis)
        if not hit.any():
            return 0
        return int(np.percentile(a.argmax(axis=axis)[hit], 50, method="lower"))

    return depth(m, 0), depth(m[::-1], 0), depth(m, 1), depth(m[:, ::-1], 1)


def draw_cut(rng: np.random.Generator, depths: Cut) -> Cut:
    """Px to cut off each side: 0 .. that side's depth (`side_depths`), each drawn on its own, in the order top,
    bottom, left, right."""
    return tuple(int(rng.integers(0, d + 1)) for d in depths)


# --- one sample: the cpu side, then the device side ------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Source:
    """A page as the cpu side reads it: its luminance as an 8-bit Pillow image (`lum`, mode L, 0 .. 255) and its
    label's inside as one (`inside`, mode L, 0 or 1), made once per training."""
    lum: Image.Image
    inside: Image.Image

    @classmethod
    def of(cls, pg: Page) -> _Source:
        lum = np.round(np.clip(pg.lum, 0.0, 1.0) * 255.0).astype(np.uint8)
        return cls(Image.fromarray(lum), Image.fromarray(pg.inside.astype(np.uint8)))   # 2-D uint8: mode L


def _cut_and_scale(src: _Source, cut: Cut, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """(lum, inside), uint8 arrays: the page cut by `cut`, then scaled by `scale` (Pillow BILINEAR for the luminance,
    NEAREST for the inside: px centres, as torch's nearest-exact).  Cut first and then scaled, so no px beyond the cut
    reaches the sample through the resampling filter; a scale of 1 resamples nothing."""
    top, bottom, left, right = cut
    w, h = src.lum.size
    box = (left, top, w - right, h - bottom)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"cut {cut} leaves nothing of a {h} x {w} page")
    lum, inside = src.lum.crop(box), src.inside.crop(box)
    if scale != 1.0:
        size = (max(1, round(lum.width * scale)), max(1, round(lum.height * scale)))
        lum = lum.resize(size, Image.Resampling.BILINEAR)
        inside = inside.resize(size, Image.Resampling.NEAREST)
    return np.array(lum), np.array(inside)


@torch.no_grad()
def _on_device(lum: np.ndarray, inside: np.ndarray, aug: Aug, paper_weight: float,
               device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x, target, weight), each (1, 1, H, W) on `device`, from the cpu side's uint8 arrays: `apply_aug(aug, lum,
    target, weight)` of the luminance, the target 1 - inside and the weight `paper_weight + (1 - lum)` (before the
    photometric changes), then padded at the bottom and right to multiples of `model.multiple()`: paper in x, 1
    (not map) in the target, 0 in the weight.

    Computed in a cheaper order with the same values: the flip and turns move the two uint8 arrays (6 ms on the
    largest page, mps), and the target, the weight and the float luminance are made from the moved bytes, instead of
    moving three float32 arrays (38 ms).  The target and the weight are functions of each px's own values and the
    move only permutes px, so either order gives the same numbers."""
    lum8, inside8 = (_move(aug, torch.from_numpy(a).to(device)).contiguous() for a in (lum, inside))
    lum_t = lum8.float() / 255.0
    target = 1.0 - inside8.float()
    weight = paper_weight + (1.0 - lum_t)
    lum_t = _photometric(aug, lum_t)
    h, w = lum_t.shape
    m = model.multiple()
    size = (1, 1, -(-h // m) * m, -(-w // m) * m)
    x = torch.full(size, model.PAPER, dtype=torch.float32, device=device)
    t = torch.ones(size, dtype=torch.float32, device=device)
    wt = torch.zeros(size, dtype=torch.float32, device=device)
    x[0, 0, :h, :w], t[0, 0, :h, :w], wt[0, 0, :h, :w] = lum_t, target, weight
    return x, t, wt


def make_sample(page: Page, cut: Cut, scale: float, aug: Aug, paper_weight: float,
                device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x, target, weight), each (1, 1, H, W): `page` cut by `cut`, scaled by `scale`, augmented by `aug` and padded,
    as a training step reads it (see the module)."""
    return _on_device(*_cut_and_scale(_Source.of(page), cut, scale), aug, paper_weight, device)


# --- held-out loss ---------------------------------------------------------------------------------------------------

def held_loss(net: nn.Module, held: Sequence[Page], device: str) -> tuple[float, float]:
    """(loss, ink): the ink-weighted log-loss of `model.prob([net], ...)` (P(outside), clipped to `HELD_EPS`) against
    the labels' outside, over every px of every held page, each px weighted by 1 - lum and the pages pooled; and the
    total ink (the sum of the weights), so curves can be pooled across runs.  NaN loss with no ink."""
    num = den = 0.0
    for pg in held:
        p = np.clip(model.prob([net], pg.lum, device), HELD_EPS, 1.0 - HELD_EPS)
        w = 1.0 - pg.lum
        loss = np.where(pg.inside, -np.log1p(-p), -np.log(p))
        num += float((loss * w).sum(dtype=np.float64))
        den += float(w.sum(dtype=np.float64))
    return (num / den if den > 0 else float("nan")), den


# --- the training ----------------------------------------------------------------------------------------------------

def _state(net: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}


def train(pages: Sequence[Page], recipe: Recipe, device: str, held: Sequence[Page] = (),
          snapshots: Path | None = None, label: str = "") -> tuple[nn.Module, dict]:
    """The frame finder's net trained on `pages` (see the module).  Returns (the net, in eval mode on cpu,
    {"train_curve": [[step, loss], ...], "held_curve": [[step, loss, ink], ...]}): at every multiple of
    `recipe.check_every`, the training loss averaged since the last check and, with `held`, `held_loss` (empty
    without).  Each check logs "<label> step <s>/<n>: train <loss>[, held <loss>] (<ms> ms/step)", the step time
    since the last check (the checks' own work left out), and saves the net's state dict to
    `snapshots / "step-<s:05d>.pt"` when `snapshots` is set.  FloatingPointError on a non-finite training loss (read
    at the checks only, so the device is not waited on every step)."""
    for pg in (*pages, *held):
        if pg.inside is None:
            raise ValueError(f"page {pg.id} has no label")
    rng = _settle(recipe.seed)
    net = model.new_net().to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=recipe.lr)
    sources = [_Source.of(pg) for pg in pages]
    depths = [side_depths(pg.inside) for pg in pages]
    if snapshots is not None:
        snapshots.mkdir(parents=True, exist_ok=True)

    def draw() -> tuple[np.ndarray, np.ndarray, Aug]:
        k = int(rng.integers(len(sources)))
        cut = draw_cut(rng, depths[k])
        scale = 1.0 + float(rng.uniform(-SCALE_JITTER, SCALE_JITTER))
        aug = draw_aug(rng)
        return (*_cut_and_scale(sources[k], cut, scale), aug)

    prefix = f"{label} " if label else ""
    train_curve, held_curve = [], []
    total, since = torch.zeros((), device=device), 0
    t0 = time.perf_counter()
    with _batches(draw, recipe.steps, recipe.prefetch) as batches:
        for step, (lum, inside, aug) in enumerate(batches, 1):
            x, t, w = _on_device(lum, inside, aug, recipe.paper_weight, device)
            loss = _loss(net(x), t, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach()
            since += 1
            if step % recipe.check_every:
                continue
            mean = float(total) / since
            ms = (time.perf_counter() - t0) / since * 1000.0
            total.zero_()
            since = 0
            if not math.isfinite(mean):
                raise FloatingPointError(f"training loss is {mean} by step {step}")
            train_curve.append([step, mean])
            line = f"{prefix}step {step}/{recipe.steps}: train {mean:.4f}"
            if held:
                held_curve.append([step, *held_loss(net, held, device)])
                line += f", held {held_curve[-1][1]:.4f}"
            log.info("%s (%.0f ms/step)", line, ms)
            if snapshots is not None:
                torch.save(_state(net), snapshots / f"step-{step:05d}.pt")
            t0 = time.perf_counter()
    return net.eval().cpu(), {"train_curve": train_curve, "held_curve": held_curve}
