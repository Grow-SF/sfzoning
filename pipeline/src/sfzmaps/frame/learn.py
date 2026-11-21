"""Training the frame finder's one network on whole pages (issue #71, spec parts 3-6).

Each step is one whole page at full resolution, read exactly as prediction reads a page (`model.pad`: paper at the
bottom and right up to `model.padded_shape`, multiples of `model.BUCKET` px), so there is no window edge that training
sees and prediction does not.  The step:

1. draws a labelled page uniformly, then its cut, its scale and its augmentation, in that order, from one generator;
2. cuts the page: each side by a random 0 .. that side's labelled frame depth (`side_depths`, `draw_cut`), so at the
   extreme the frame is gone and map ink runs to the page edge (a frameless side), the label cut alike;
3. scales the cut page by a factor in 1 +- `SCALE_JITTER` (bilinear for the luminance, nearest for the label);
4. changes its brightness, contrast and gamma and blurs it now and then (`_photometric`), then turns and flips it,
   the label alike (`_move`; the blur is the same each way, so it commutes with the turns and flips);
5. takes one Adam step on the weighted multiclass log-loss (`_loss`, cross-entropy) of the label's class per px: 0
   map, 1 outside the frame, 2 removed block (`data.class_mask`, `model.CLASSES`).  Multiclass log-loss is a strictly
   proper scoring rule, as the two-class one was, so the chances stay calibrated (spec amendment "removed blocks").

The weight of a px is `paper_weight + (1 - lum)`, from the luminance after the scale and before the photometric
changes, and 0 on the padding.  It never depends on the label: a weight that does moves the loss's minimiser away
from the true chance (spec part 4), and the decision downstream needs calibrated chances.  So there is no uncertain
strip along the outline either.

The learning rate is constant, with no schedule: the net after s steps of a longer run is then exactly the net of a
run of s steps, so the held-out loss at a check is the score of "train s steps", which the final model repeats
(spec part 6).  At every multiple of `check_every` the run logs the training loss averaged since the last check, the
held-out loss (`held_loss`) when held pages are given, and saves a snapshot when asked.  The held-out loss is a log
and a curve only: it changes no weight.

Where the work runs (issue #71's speed measurements, "fold 16" net on mps: a device-side bilinear rescale of a whole
page was 63 ms of a 213 ms step; later the device side was 49 ms of every step, in series with the net, and the cpu
side 75 ms on one thread): the cpu does everything but the net's own step, from the page's luminance and label kept
as 8-bit Pillow images (`_Source`; the pages are 8-bit scans, so the luminance loses nothing).  It cuts the page and
scales it (`_cut_and_scale`), then changes the 8-bit luminance's brightness, contrast and gamma by a 256-entry lookup
table and blurs it (`_photometric`, `_cpu_side`), each result rounded to the nearest of the 256 levels.  It hands the
device three uint8 planes: the changed luminance, the luminance before the changes (for the weight) and the classes.
The device turns and flips those bytes, makes the floats, the target and the weight and pads them (`_on_device`).
With `Recipe.prefetch` a background thread (`_batches`) draws up to `PREFETCH` steps ahead and hands each step's cpu
work to `DRAW_THREADS` threads while the device steps.

Deterministic: the trainer seeds torch and one numpy generator with `recipe.seed`, makes torch use deterministic
algorithms and pins its cpu threads to `THREADS` and OpenCV's to one per drawing thread (`_settle`); every random
value comes from that generator, drawn in a fixed order by one thread (the prefetching one, or the training's own)
before the work, which draws nothing, is handed out, and the results come back in the order drawn.  So the weights
are the same with or without prefetch, and with or without held pages.  On mps the caller holds the GPU's exclusive
turn (`sfzmaps.gpu.gpu_turn`) for the whole run.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import logging
import math
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import cv2
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
PREFETCH = 4            # steps drawn ahead with `Recipe.prefetch` (computational)
DRAW_THREADS = 3        # threads that do the drawn steps' cpu work (computational: Pillow and OpenCV let go of the GIL)

HELD_EPS = 1e-6         # the held-out chance of the labelled class is clipped to at least this (numerical)


@dataclasses.dataclass(frozen=True)
class Recipe:
    """`steps` Adam steps at the constant rate `lr`, checked every `check_every` (1,000 gives the spec's four
    candidate step counts, 1,000 to 4,000); `paper_weight` the weight of a px of plain paper (ink adds 1 - lum);
    `prefetch` draws the next steps in a background thread and does their cpu side on `DRAW_THREADS` threads: the
    same draws, so it never changes the weights."""
    steps: int = 4000
    check_every: int = 1000
    lr: float = 1e-3
    paper_weight: float = 0.1
    seed: int = 0
    prefetch: bool = True


@dataclasses.dataclass
class Page:
    """A page to train on or hold out.  `lum` is the float32 luminance (1 = paper), `cls` the label's class per px
    (uint8: 0 map, 1 outside the frame, 2 removed block; `data.class_mask`)."""
    id: SheetId
    lum: np.ndarray
    cls: np.ndarray | None


# --- augmentation: drawn and mostly applied on the cpu, the turns and flips on the device ---------------------------

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


def _move(aug: Aug, a: torch.Tensor) -> torch.Tensor:
    """`aug`'s flip, then its quarter turns, of a (..., h, w) tensor (a view where torch can)."""
    return torch.rot90(torch.flip(a, (-1,)) if aug.flip else a, aug.turns, (-2, -1))


def _table(aug: Aug) -> np.ndarray:
    """`aug`'s gamma, contrast and brightness changes as a 256-entry uint8 lookup table: level k is k / 255 raised to
    the gamma, stretched by the contrast about 0.5, shifted by the brightness, kept within [0, 1] and rounded to the
    nearest level."""
    v = (np.arange(256, dtype=np.float64) / 255.0) ** aug.gamma
    v = np.clip((v - 0.5) * aug.contrast + 0.5 + aug.brightness, 0.0, 1.0)
    return np.round(v * 255.0).astype(np.uint8)


def _blur(lum: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur of a (h, w) uint8 array, as `scipy.ndimage.gaussian_filter(lum, sigma, mode="nearest")` rounded
    to the nearest level: a separable kernel `BLUR_TRUNCATE` sigmas each way, the edge px repeated beyond the edge.
    OpenCV's `sepFilter2D` with scipy's kernel, not its `GaussianBlur`, whose 8-bit path rounds the kernel to fixed
    point (measured up to 2.9 levels off scipy on 1-bit noise; this is within half a level, at the same speed)."""
    radius = int(BLUR_TRUNCATE * sigma + 0.5)
    if radius == 0:
        return lum
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return cv2.sepFilter2D(lum, -1, k, k, borderType=cv2.BORDER_REPLICATE)


def _photometric(aug: Aug, lum: np.ndarray) -> np.ndarray:
    """`aug`'s gamma, contrast and brightness changes (`_table`), then its blur (`_blur`; sigma 0: none), of a (h, w)
    uint8 luminance; uint8."""
    out = cv2.LUT(lum, _table(aug))
    return _blur(out, aug.sigma) if aug.sigma > 0 else out


# --- shared pieces (from `frame/train.py` at e580a00; `_batches` now hands its jobs to `DRAW_THREADS` threads) -------

def _settle(seed: int) -> np.random.Generator:
    """Seed torch, make it deterministic and pin its threads, and OpenCV's to one (each drawing thread is one); the one
    numpy generator every draw comes from."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)
    cv2.setNumThreads(1)
    return np.random.default_rng(seed)


def _loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted mean cross-entropy of `logits` (n, CLASSES, H, W) against the class indices `target` (n, H, W), each
    px weighted by `weight` (n, H, W).  A masked (weighted) sum, never boolean indexing: the backward of indexing has
    no deterministic kernel on mps."""
    ce = F.cross_entropy(logits, target, reduction="none")
    return (ce * weight).sum() / weight.sum().clamp(min=WEIGHT_EPS)


Job = tuple[Callable[..., object], tuple]     # (work, args): work that draws nothing, and what it is called with


@contextlib.contextmanager
def _batches(draw: Callable[[], Job], n: int, prefetch: bool) -> Iterator[Iterator[object]]:
    """An iterator over `n` results, in order: each `draw()` gives a job `(work, args)`, its result `work(*args)`.
    One thread calls `draw`, in order: the training's own without `prefetch`, which then does each job as it asks for
    it; with `prefetch` a background thread, up to `PREFETCH` ahead, which hands the jobs to `DRAW_THREADS` threads
    whose results come back in the order drawn.  So the same draws come in the same order either way.  An error in
    `draw` or a job is raised where its result is asked for; leaving the block, by an error or not, stops and joins the
    threads (jobs not yet started are dropped)."""
    if not prefetch:
        yield (work(*args) for work, args in (draw() for _ in range(n)))
        return
    ready, stop = queue.Queue(PREFETCH), threading.Event()
    pool = concurrent.futures.ThreadPoolExecutor(DRAW_THREADS, thread_name_prefix="frame-cut")

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
                work, args = draw()
                if not put(pool.submit(work, *args)):
                    return
        except BaseException as e:      # noqa: BLE001 - handed to the training's thread, raised there
            put(e)

    def consume():
        for _ in range(n):
            item = ready.get()
            if isinstance(item, BaseException):
                raise item
            yield item.result()

    worker = threading.Thread(target=produce, name="frame-prefetch", daemon=True)
    worker.start()
    try:
        yield consume()
    finally:
        stop.set()
        worker.join()
        pool.shutdown(wait=True, cancel_futures=True)


# --- the cut: how deep each side's frame is --------------------------------------------------------------------------

Cut = tuple[int, int, int, int]     # px taken off the (top, bottom, left, right) of a page


def side_depths(cls: np.ndarray) -> Cut:
    """(top, bottom, left, right): for each side, the median over that side's lines (the columns for top and bottom,
    the rows for left and right) of the distance in px from the page's edge to the line's first px not outside the
    frame (`cls != 1`: map or a removed block).  Lines with no such px are left out; a side with none gives 0.  The
    lower median (one line's own depth), so an even count of lines never gives a half px."""
    m = np.asarray(cls) != 1

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
    label's classes as one (`cls`, mode L, 0 .. 2), made once per training."""
    lum: Image.Image
    cls: Image.Image

    @classmethod
    def of(cls, pg: Page) -> _Source:
        lum = np.round(np.clip(pg.lum, 0.0, 1.0) * 255.0).astype(np.uint8)
        return cls(Image.fromarray(lum), Image.fromarray(np.asarray(pg.cls, np.uint8)))   # 2-D uint8: mode L


def _cut_and_scale(src: _Source, cut: Cut, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """(lum, cls), uint8 arrays: the page cut by `cut`, then scaled by `scale` (Pillow BILINEAR for the luminance,
    NEAREST for the classes: px centres, as torch's nearest-exact).  Cut first and then scaled, so no px beyond the
    cut reaches the sample through the resampling filter; a scale of 1 resamples nothing."""
    top, bottom, left, right = cut
    w, h = src.lum.size
    box = (left, top, w - right, h - bottom)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"cut {cut} leaves nothing of a {h} x {w} page")
    lum, cls = src.lum.crop(box), src.cls.crop(box)
    if scale != 1.0:
        size = (max(1, round(lum.width * scale)), max(1, round(lum.height * scale)))
        lum = lum.resize(size, Image.Resampling.BILINEAR)
        cls = cls.resize(size, Image.Resampling.NEAREST)
    return np.array(lum), np.array(cls)


def _cpu_side(src: _Source, cut: Cut, scale: float, aug: Aug) -> tuple[np.ndarray, Aug]:
    """(planes, aug): `src` cut by `cut` and scaled by `scale` (`_cut_and_scale`), as (3, h, w) uint8 planes: the
    luminance after `aug`'s photometric changes and blur (`_photometric`), the luminance before them (the weight's)
    and the classes.  Draws nothing; `aug` comes back with the planes for the device's turns and flips."""
    lum, cls = _cut_and_scale(src, cut, scale)
    return np.stack((_photometric(aug, lum), lum, cls)), aug


@torch.no_grad()
def _on_device(planes: np.ndarray, aug: Aug, paper_weight: float,
               device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x, target, weight) on `device`, x (1, 1, H, W) and the target and the weight (1, H, W), from the cpu side's
    (3, h, w) uint8 planes (`_cpu_side`): moved together by `aug`'s flip and turns (`_move`, of the bytes: a sixth of
    the time of moving float32 arrays), then x the changed luminance / 255, the target the classes as int64 indices
    and the weight `paper_weight + (1 - lum)` of the luminance before the changes, padded at the bottom and right to
    `model.padded_shape`: paper in x, 1 (outside the frame) in the target, 0 in the weight."""
    moved = _move(aug, torch.from_numpy(planes).to(device)).contiguous()
    _, h, w = moved.shape
    size = (1, *model.padded_shape(h, w))
    x = torch.full((1, *size), model.PAPER, dtype=torch.float32, device=device)
    t = torch.ones(size, dtype=torch.int64, device=device)
    wt = torch.zeros(size, dtype=torch.float32, device=device)
    x[0, 0, :h, :w] = moved[0].float() / 255.0
    wt[0, :h, :w] = paper_weight + (1.0 - moved[1].float() / 255.0)
    t[0, :h, :w] = moved[2]
    return x, t, wt


def make_sample(page: Page, cut: Cut, scale: float, aug: Aug, paper_weight: float,
                device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(x, target, weight), x (1, 1, H, W), the target (int64 class indices) and the weight (1, H, W): `page` cut by
    `cut`, scaled by `scale`, augmented by `aug` and padded, as a training step reads it (see the module)."""
    return _on_device(*_cpu_side(_Source.of(page), cut, scale, aug), paper_weight, device)


# --- held-out loss ---------------------------------------------------------------------------------------------------

def held_loss(net: nn.Module, held: Sequence[Page], device: str) -> tuple[float, float]:
    """(loss, ink): the ink-weighted multiclass log-loss of `model.prob([net], ...)` against the labels' classes (-log
    of the chance given the labelled class, clipped to at least `HELD_EPS`), over every px of every held page, each px
    weighted by 1 - lum and the pages pooled; and the total ink (the sum of the weights), so curves can be pooled
    across runs.  NaN loss with no ink."""
    num = den = 0.0
    for pg in held:
        p = model.prob([net], pg.lum, device)
        own = np.take_along_axis(p, np.asarray(pg.cls, np.intp)[None], axis=0)[0]
        w = 1.0 - pg.lum
        loss = -np.log(np.maximum(own, HELD_EPS))
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
        if pg.cls is None:
            raise ValueError(f"page {pg.id} has no label")
    rng = _settle(recipe.seed)
    net = model.new_net().to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=recipe.lr)
    sources = [_Source.of(pg) for pg in pages]
    depths = [side_depths(pg.cls) for pg in pages]
    if snapshots is not None:
        snapshots.mkdir(parents=True, exist_ok=True)

    def draw() -> Job:
        k = int(rng.integers(len(sources)))
        cut = draw_cut(rng, depths[k])
        scale = 1.0 + float(rng.uniform(-SCALE_JITTER, SCALE_JITTER))
        aug = draw_aug(rng)
        return _cpu_side, (sources[k], cut, scale, aug)

    prefix = f"{label} " if label else ""
    train_curve, held_curve = [], []
    total, since = torch.zeros((), device=device), 0
    t0 = time.perf_counter()
    with _batches(draw, recipe.steps, recipe.prefetch) as batches:
        for step, (planes, aug) in enumerate(batches, 1):
            x, t, w = _on_device(planes, aug, recipe.paper_weight, device)
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
