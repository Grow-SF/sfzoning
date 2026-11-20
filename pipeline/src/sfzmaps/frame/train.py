"""Training the learned frame finder's two passes together (see `predict` for how they read a page).

`train_joint` trains the page pass and the pixel pass in one loop, from one generator and one seed (issue #71, Task
13).  Each step:

1. the page-pass loss on `batch` augmented whole pages;
2. the pixel-pass loss on `batch` crops, each crop's prior the page pass's own current probability for its page;
3. one Adam step on both nets' parameters, on the sum of the two losses.

Training reads a page exactly as prediction does, so batch norm and the borders see the same thing in both:

- The page pass sees whole pages on the coarse grid, padded with paper to the COARSE_SIDE square
  (`predict.coarse_input`), and learns the label's mask resized to that grid with Pillow BOX: a soft target, each
  coarse px the fraction of its page px inside the map.
- The pixel pass sees crops at full resolution, fed `[lum, prior]`, each a `predict.TILE` x `predict.TILE` core with
  `predict._margin()` px of context around it, as `predict.predict_tiles` reads a tile; it learns the mask on the core
  only.  Half the cores centre on a px within `scores.BAND_PX * 4` of the label's outline, where the hard part is; the
  other half anywhere on the page.  A crop's prior is what prediction would feed it now: the page pass, in eval mode
  and with no gradient, run once per step on each page the crops were cut from (unaugmented, as `coarse_input` reads
  it), upsampled bilinearly as `predict.predict_coarse` does but over the crop's window only, and rounded to float16
  as `predict.page_prior` saves it (`_page_priors`, `_prior_window`); then scaled and augmented with the crop.  The
  prior carries no gradient, so each pass is trained by its own loss alone.

Both losses are binary cross-entropy on logits, each px weighted by `paper_weight + ink_weight(lum)` (the ink counts,
as in the scores, and paper a little so the net still says something there).  Px outside the page (padding), the
pixel pass's margins and the label's uncertain strip (`scores.unsure`: map px within `scores.UNSURE_PX` of an
outline, holes' too, the frame side being certain; on the coarse grid `UNSURE_PX` times its scale, at least 1 coarse
px) weigh exactly 0.  The pixel pass's logit is the prior's plus the net's correction (`predict.Corrected`), so its
loss is on that sum, as prediction reads it.

Every sample is augmented: quarter turns and mirror flips of every array alike, and brightness, contrast, gamma and
blur changes of the luminance alone, the blur measured in page px on either grid; the pixel pass's crops are also
scaled by up to `SCALE_JITTER` either way.  The page pass needs no scale change: every page is resized to
`predict.COARSE_SIDE` before it is read.

Where the work runs (issue #71's speed amendment): the cpu draws every random value and cuts the crops
(`_coarse_draw`, `_fine_draw`); the device builds the batches from them (`_coarse_make`, `_fine_make`): the priors,
the crops' scaling, the turns and flips, the photometric changes and the blur, in torch (`apply_aug`).  With
`Recipe.prefetch` a background thread draws the next steps' pages and crops while the device steps (`_batches`).

The learning rate follows a half cosine from `recipe.lr` at step 1 to exactly 0 after the last step; no checkpoint is
chosen, the last step's nets are the model.  Every `check_every` steps one line is logged, "<label> step <s>/<n>: page
<loss>, pixel <loss>[, val <loss>]": the two training losses averaged since the last check and, with `val`, the
ink-weighted log-loss on the validation pages (`predict_tiles` only near the label's outline, the prior elsewhere).
The validation is a log only: it changes no weight.

Deterministic: the trainer seeds torch and one numpy generator with `recipe.seed`, makes torch use deterministic
algorithms and pins its cpu threads to `THREADS`; every random value comes from that generator, drawn in a fixed order
by one thread (the prefetching one, or the training's own), so the weights are the same with or without prefetch, and
with or without validation.  On mps the caller holds the GPU's exclusive turn (`sfzmaps.gpu.gpu_turn`) for the whole
run.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import math
import queue
import threading
from collections.abc import Callable, Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from sfzmaps.frame import predict, scores
from sfzmaps.frame.data import SheetId, ink_weight

log = logging.getLogger(__name__)

THREADS = 4
SCALE_JITTER = 0.1      # the pixel pass's crops are scaled by a factor in 1 +- this
BLUR_SIGMA = 1.5        # px, the most a luminance blur spreads
BLUR_SHARE = 0.5        # the share of samples blurred
BRIGHTNESS = 0.1        # the luminance is shifted by up to this either way
CONTRAST = 0.2          # ... stretched about 0.5 by a factor in 1 +- this
GAMMA = 1.5             # ... and raised to a power between 1 / this and this
BLUR_TRUNCATE = 4.0     # the blur's kernel reaches this many sigmas each way (scipy's gaussian_filter default)
OUTLINE_REACH = scores.BAND_PX * 4   # px: the outline crops centre within this of the label's outline
WEIGHT_EPS = 1e-12      # the least the loss divides by, so a batch of no weight gives 0 (numerical; decides nothing)
PREFETCH = 2            # steps drawn ahead with `Recipe.prefetch`


@dataclasses.dataclass(frozen=True)
class Recipe:
    """`prefetch` draws the next steps in a background thread: the same draws, so it never changes the weights."""
    steps: int = 2000
    check_every: int = 250
    lr: float = 1e-3
    batch: int = 4
    paper_weight: float = 0.1
    seed: int = 0
    prefetch: bool = True


@dataclasses.dataclass
class Page:
    """A page to train or validate on.  `lum` is the luminance (1 = paper), `mask` the label's map area (1 inside)."""
    id: SheetId
    lum: np.ndarray
    mask: np.ndarray | None


# --- augmentation: drawn on the cpu, applied on the device -----------------------------------------------------------

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
    def move(a: torch.Tensor) -> torch.Tensor:
        return torch.rot90(torch.flip(a, (-1,)) if aug.flip else a, aug.turns, (-2, -1))

    lum = move(lum).float().clamp(0.0, 1.0) ** aug.gamma
    lum = ((lum - 0.5) * aug.contrast + 0.5 + aug.brightness).clamp(0.0, 1.0)
    if aug.sigma > 0:
        lum = _blur(lum, aug.sigma)
    return (lum.contiguous(), *(move(a).contiguous() for a in others))


def _resize(a: torch.Tensor, side: int, mode: str) -> torch.Tensor:
    """A (..., h, w) tensor resized to (..., side, side), every (h, w) plane alike, in one call: "bilinear"
    (half-px centres) or "nearest-exact" (the same centres)."""
    if tuple(a.shape[-2:]) == (side, side):
        return a
    extra = {"align_corners": False} if mode == "bilinear" else {}
    out = F.interpolate(a.reshape(-1, 1, *a.shape[-2:]), size=(side, side), mode=mode, **extra)
    return out.reshape(*a.shape[:-2], side, side)


# --- shared pieces ---------------------------------------------------------------------------------------------------

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


def _val_loss(val: list[Page]) -> Callable[[Callable[[int], np.ndarray]], float]:
    """A function of `prob` (validation page i's map probability): the ink-weighted log-loss over all ink of every
    validation page beyond the label's uncertain strip (`scores.unsure`, found once here, not at every check), pooled
    (each page counts by that ink)."""
    sure = [~scores.unsure(pg.mask) for pg in val]

    def loss(prob: Callable[[int], np.ndarray]) -> float:
        num = den = 0.0
        for i, (pg, where) in enumerate(zip(val, sure)):
            w = ink_weight(pg.lum)
            total = float(w[where].sum(dtype=np.float64))
            if total <= 0:
                continue
            num += scores.log_loss(prob(i), pg.mask, w, where) * total
            den += total
        return num / den if den else float("nan")
    return loss


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


# --- the page pass ---------------------------------------------------------------------------------------------------

def _coarse_arrays(pg: Page, paper_weight: float) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    """((lum, target, weight), scale): the page on the coarse grid `predict.coarse_input` reads it on, without its
    padding, and that grid's px per page px.  The weight is 0 on the label's uncertain strip as that grid sees it:
    `scores.unsure` of the target's map side (>= 1/2) at `UNSURE_PX * scale` coarse px, at least 1."""
    h, w = pg.lum.shape
    arr, scale = predict.coarse_input(pg.lum)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    lum = arr[:nh, :nw]
    target = predict.coarse_input(pg.mask.astype(np.float32))[0][:nh, :nw]
    strip = scores.unsure(target >= 0.5, max(1.0, scores.UNSURE_PX * scale))
    weight = np.where(strip, 0.0, paper_weight + ink_weight(lum)).astype(np.float32)
    return (lum, target, weight), scale


def _coarse_draw(rng: np.random.Generator, scales: list[float], batch: int) -> list[tuple[int, Aug]]:
    """`batch` pages (indices, drawn with replacement), each with its augmentation (the blur in page px: `scales`
    are the pages' coarse px per page px)."""
    out = []
    for _ in range(batch):
        k = int(rng.integers(len(scales)))
        out.append((k, draw_aug(rng, scales[k])))
    return out


def _coarse_make(spec: list[tuple[int, Aug]], arrays: list[tuple[torch.Tensor, ...]], device: str):
    """The page-pass batch on the device: each drawn page's (lum, target, weight) (`arrays`, on the device already),
    augmented, and padded at right and bottom to the COARSE_SIDE x COARSE_SIDE square `predict.coarse_input` reads on:
    paper in the input, target and weight 0."""
    side, n = predict.COARSE_SIDE, len(spec)
    x = torch.ones((n, 1, side, side), dtype=torch.float32, device=device)
    t = torch.zeros((n, 1, side, side), dtype=torch.float32, device=device)
    w = torch.zeros((n, 1, side, side), dtype=torch.float32, device=device)
    for i, (k, aug) in enumerate(spec):
        lum, target, weight = apply_aug(aug, *arrays[k])
        h, wd = lum.shape
        x[i, 0, :h, :wd], t[i, 0, :h, :wd], w[i, 0, :h, :wd] = lum, target, weight
    return x, t, w


# --- the pixel pass's prior: the page pass as it is now --------------------------------------------------------------

def _page_priors(net: nn.Module, arrays: list[tuple[torch.Tensor, ...]], pages: Iterable[int],
                 device: str) -> dict[int, torch.Tensor]:
    """{page index: the page pass's map probability on the page's coarse grid, without the padding}, for each of
    `pages` once: the unaugmented pages (`arrays`' luminance, padded with paper to the COARSE_SIDE square as
    `predict.coarse_input` pads it) in one batch, the net in eval mode with no gradient, left as it was."""
    ks = sorted(set(pages))
    side = predict.COARSE_SIDE
    x = torch.ones((len(ks), 1, side, side), dtype=torch.float32, device=device)
    for i, k in enumerate(ks):
        h, w = arrays[k][0].shape
        x[i, 0, :h, :w] = arrays[k][0]
    with predict._eval_on(net, device), torch.no_grad():
        prob = torch.sigmoid(net(x))
    return {k: prob[i, 0, :arrays[k][0].shape[0], :arrays[k][0].shape[1]] for i, k in enumerate(ks)}


def _source(n_page: int, n_small: int, start: int, side: int):
    """Along one axis, for page px start .. start + side - 1: the coarse px each reads and their weights, as
    `F.interpolate(mode="bilinear", align_corners=False)` from n_small to n_page px finds them (the source position
    `(i + 0.5) * n_small / n_page - 0.5`, at least 0; its floor and the next px, at most the last), and whether the px
    is on the page."""
    i = np.arange(start, start + side)
    on = (i >= 0) & (i < n_page)
    src = np.maximum((np.clip(i, 0, n_page - 1) + 0.5) * (n_small / n_page) - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), n_small - 1)
    i1 = np.minimum(i0 + 1, n_small - 1)
    return i0, i1, (src - i0).astype(np.float32), on


def _prior_window(small: torch.Tensor, shape: tuple[int, int], top: int, left: int, side: int) -> torch.Tensor:
    """The side x side window at (top, left) of `small` (a page's coarse-grid probability, `_page_priors`) upsampled
    to the page's `shape` as `predict.predict_coarse` upsamples it, computed over the window alone; 0.0 (not map) off
    the page, as `predict_tiles` pads the prior; rounded to float16 as `predict.page_prior` saves it."""
    (h, w), dev = shape, small.device
    r0, r1, fr, on_r = _source(h, small.shape[0], top, side)
    c0, c1, fc, on_c = _source(w, small.shape[1], left, side)

    def t(a):
        return torch.from_numpy(a).to(dev)
    fr, fc = t(fr)[:, None], t(fc)[None]
    rows = small[t(r0)] * (1 - fr) + small[t(r1)] * fr
    out = rows[:, t(c0)] * (1 - fc) + rows[:, t(c1)] * fc
    out = out * t((on_r[:, None] & on_c[None]).astype(np.float32))
    return out.half().float()


# --- the pixel pass --------------------------------------------------------------------------------------------------

def _near_mask(mask: np.ndarray) -> np.ndarray:
    """Bool, True on the px within `OUTLINE_REACH` of the mask's outline; all False for a mask with no outline (all
    map or none)."""
    m = mask.astype(bool)
    edge = m & ~ndimage.binary_erosion(m, border_value=1)
    if not edge.any():
        return np.zeros(m.shape, bool)
    return ndimage.distance_transform_edt(~edge) <= OUTLINE_REACH


def _near_outline(mask: np.ndarray) -> np.ndarray:
    """Flat indices of the px within `OUTLINE_REACH` of the mask's outline (`_near_mask`)."""
    return np.flatnonzero(_near_mask(mask))


def _window(a: np.ndarray, top: int, left: int, side: int, fill: float) -> np.ndarray:
    """The side x side window of `a` at (top, left), float32, `fill` where it leaves the page."""
    out = np.full((side, side), fill, np.float32)
    h, w = a.shape
    r0, c0, r1, c1 = max(top, 0), max(left, 0), min(top + side, h), min(left + side, w)
    if r1 > r0 and c1 > c0:
        out[r0 - top:r1 - top, c0 - left:c1 - left] = a[r0:r1, c0:c1]
    return out


def _fine_aux(pg: Page) -> tuple[np.ndarray, np.ndarray]:
    """What the pixel pass's crops need of a page beside its arrays, found once per training: (the flat indices of
    the px near the label's outline (`_near_outline`), the uncertain strip (`scores.unsure`))."""
    return _near_outline(pg.mask), scores.unsure(pg.mask)


@dataclasses.dataclass(frozen=True)
class _Crop:
    """A pixel-pass crop as the cpu cuts it: `arrays`, (3, side, side) float32 windows of the luminance (paper off
    the page), the mask and the sure px (1 beyond the label's uncertain strip) (0 off the page), stacked so the device
    moves them in one go; the page (`page`, its index, and its `shape`) and the window's place on it (`top`, `left`);
    the page's rows [r0, r1) and columns [c0, c1) within the window (`inside`); and the augmentation.  The prior is
    added on the device (`_fine_make`): it is the page pass's as it is at the step."""
    arrays: np.ndarray
    page: int
    shape: tuple[int, int]
    top: int
    left: int
    inside: tuple[int, int, int, int]
    aug: Aug


def _fine_crop(rng: np.random.Generator, k: int, pg: Page, aux: tuple[np.ndarray, np.ndarray],
               outline: bool) -> _Crop:
    """One crop of page `k`, to be read as `predict.predict_tiles` reads a tile once `_fine_make` scales it: a window
    of (TILE + 2 margin) / s px (s within 1 +- SCALE_JITTER), centred on a px near the outline (`outline` and the page
    has one) or anywhere on the page.  `aux` is the page's `_fine_aux`."""
    near, strip = aux
    full = predict.TILE + 2 * predict._margin()
    side = max(1, round(full / (1.0 + float(rng.uniform(-SCALE_JITTER, SCALE_JITTER)))))
    h, w = pg.lum.shape
    if outline and near.size:
        cy, cx = divmod(int(near[int(rng.integers(near.size))]), w)
    else:
        cy, cx = int(rng.integers(h)), int(rng.integers(w))
    top, left = cy - side // 2, cx - side // 2
    inside = (max(-top, 0), min(h - top, side), max(-left, 0), min(w - left, side))
    arrays = np.stack([_window(pg.lum, top, left, side, 1.0), _window(pg.mask, top, left, side, 0.0),
                       _window(~strip, top, left, side, 0.0)])
    return _Crop(arrays=arrays, page=k, shape=(h, w), top=top, left=left, inside=inside, aug=draw_aug(rng))


def _fine_draw(rng: np.random.Generator, pages: list[Page], aux: list[tuple[np.ndarray, np.ndarray]],
               batch: int) -> list[_Crop]:
    """`batch` crops, every other one (from the first) an outline crop; pages drawn with replacement; `aux` holds
    each page's `_fine_aux`."""
    crops = []
    for i in range(batch):
        k = int(rng.integers(len(pages)))
        crops.append(_fine_crop(rng, k, pages[k], aux[k], i % 2 == 0))
    return crops


def _fine_make(spec: list[_Crop], priors: dict[int, torch.Tensor], paper_weight: float, device: str):
    """The pixel-pass batch on the device: each crop's prior window (`_prior_window` of its page's `priors` entry),
    then the crop scaled to TILE + 2 margin (luminance, prior and mask bilinear), weighted `paper_weight +
    ink_weight(lum)` on the core, on the page and beyond the label's uncertain strip only (the page's extent and the
    sure px scaled nearest, so the weight off the page, on the margins and on the strip is exactly 0), then
    augmented."""
    tile, margin = predict.TILE, predict._margin()
    full = tile + 2 * margin
    core = torch.zeros((full, full), dtype=torch.float32, device=device)
    core[margin:margin + tile, margin:margin + tile] = 1.0
    xs, ts, ws = [], [], []
    for c in spec:
        side = c.arrays.shape[-1]
        planes = torch.from_numpy(c.arrays).to(device)
        prior = _prior_window(priors[c.page], c.shape, c.top, c.left, side)
        lum, prior, mask = _resize(torch.stack([planes[0], prior, planes[1]]), full, "bilinear")
        r0, r1, c0, c1 = c.inside
        inside = torch.zeros((side, side), dtype=torch.float32, device=device)
        inside[r0:r1, c0:c1] = 1.0
        inside, sure = _resize(torch.stack([inside, planes[2]]), full, "nearest-exact")
        weight = inside * sure * core * (paper_weight + ink_weight(lum))
        lum, (prior, mask, weight) = apply_aug(c.aug, lum, torch.stack([prior, mask, weight]))
        xs.append(torch.stack([lum, prior]))
        ts.append(mask)
        ws.append(weight)
    return torch.stack(xs), torch.stack(ts)[:, None], torch.stack(ws)[:, None]


# --- the joint training ----------------------------------------------------------------------------------------------

def _validation(val: list[Page], device: str) -> Callable[[nn.Module, nn.Module], float]:
    """The validation loss of (page pass, pixel pass) as the run reads them (`predict.page_prior`, then
    `predict_tiles`), the pixel pass read only on the tiles near each label's outline (`_near_mask`, found once
    here), the prior elsewhere (`_val_loss`)."""
    loss = _val_loss(val)
    near = [_near_mask(pg.mask) for pg in val]

    def check(coarse: nn.Module, fine: nn.Module) -> float:
        def prob(i: int) -> np.ndarray:
            prior = predict.page_prior(coarse, val[i].lum, device)
            return predict.predict_tiles(fine, val[i].lum, prior, device, near=near[i])
        return loss(prob)
    return check


def train_joint(pages: list[Page], recipe: Recipe, device: str, val: list[Page] | None = None,
                label: str = "training") -> tuple[tuple[predict.Folded, predict.Corrected], dict]:
    """The page pass and the pixel pass trained together on `pages` (see the module).  Returns ((page pass, pixel
    pass), both in eval mode on cpu, {"train_curve": [(step, page loss, pixel loss), ...], "val_curve": [(step, val
    loss), ...]}) at every check, the training losses averaged since the last check, the validation curve empty
    without `val`.  `label` names the training in the lines logged.  FloatingPointError on a non-finite training
    loss (read at the checks only, so the device is not waited on every step)."""
    for pg in pages + (val or []):
        if pg.mask is None:
            raise ValueError(f"page {pg.id} has no label mask")
    rng = _settle(recipe.seed)
    coarse = predict.new_coarse_net().to(device).train()
    fine = predict.new_fine_net().to(device).train()
    found = [_coarse_arrays(pg, recipe.paper_weight) for pg in pages]
    scales = [scale for _, scale in found]
    arrays = [tuple(torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device) for a in arr)
              for arr, _ in found]
    aux = [_fine_aux(pg) for pg in pages]
    check = _validation(val, device) if val else None
    opt = torch.optim.Adam([*coarse.parameters(), *fine.parameters()], lr=recipe.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1.0 + math.cos(math.pi * s / recipe.steps)))

    def draw():
        return _coarse_draw(rng, scales, recipe.batch), _fine_draw(rng, pages, aux, recipe.batch)

    train_curve, val_curve = [], []
    sums, since = torch.zeros(2, device=device), 0
    with _batches(draw, recipe.steps, recipe.prefetch) as batches:
        for step, (coarse_spec, fine_spec) in enumerate(batches, 1):
            with torch.no_grad():
                priors = _page_priors(coarse, arrays, (c.page for c in fine_spec), device)
                xc, tc, wc = _coarse_make(coarse_spec, arrays, device)
                xf, tf, wf = _fine_make(fine_spec, priors, recipe.paper_weight, device)
            losses = torch.stack([_loss(coarse(xc), tc, wc), _loss(fine(xf), tf, wf)])
            opt.zero_grad(set_to_none=True)
            losses.sum().backward()
            opt.step()
            sched.step()
            sums += losses.detach()
            since += 1
            if step % recipe.check_every == 0:
                page_loss, pixel_loss = (sums / since).tolist()
                sums.zero_()
                since = 0
                if not (math.isfinite(page_loss) and math.isfinite(pixel_loss)):
                    raise FloatingPointError(f"training loss is page {page_loss}, pixel {pixel_loss} by step {step}")
                train_curve.append((step, page_loss, pixel_loss))
                line = f"{label} step {step}/{recipe.steps}: page {page_loss:.4f}, pixel {pixel_loss:.4f}"
                if check is not None:
                    val_curve.append((step, float(check(coarse, fine))))
                    line += f", val {val_curve[-1][1]:.4f}"
                log.info("%s", line)
    return (coarse.eval().cpu(), fine.eval().cpu()), {"train_curve": train_curve, "val_curve": val_curve}
