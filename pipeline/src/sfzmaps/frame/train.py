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
as in the scores, and paper a little so the net still says something there).  Px outside the page (padding), the
pixel pass's margins and the label's uncertain strip (`scores.unsure`: map px within `scores.UNSURE_PX` of an
outline, holes' too, the frame side being certain; on the coarse grid `UNSURE_PX` times its scale, at least 1 coarse
px) weigh exactly 0.  The pixel pass's logit
is the prior's plus the net's correction (`predict.Corrected`), so its loss is on that sum, as prediction reads it.
The validation loss, like the scores, leaves the strip out.

Every sample is augmented: quarter turns and mirror flips of every array alike, and brightness, contrast, gamma and
blur changes of the luminance alone, the blur measured in page px on either grid; the pixel pass's crops are also
scaled by up to `SCALE_JITTER` either way.  The page pass needs no scale change:
every page is resized to `predict.COARSE_SIDE` before it is read.

Where the work runs (issue #71's speed amendment): the cpu draws every random value and cuts the crops (`_coarse_draw`,
`_fine_draw`); the device builds the batch from them (`_coarse_make`, `_fine_make`): the crops' scaling, the turns
and flips, the photometric changes and the blur, in torch (`apply_aug`).  With `Recipe.prefetch` a background thread
draws the next batches while the device steps (`_batches`).

The learning rate is constant (Adam), so a net trained for n steps is the net a longer run had at step n: the step
count chosen from the validation curves (`choose_steps`) is the one the final model trains for.  Every validation
check logs one line, "<label> step <s>/<n>: val <loss>".

Deterministic: each trainer seeds torch and one numpy generator with `recipe.seed`, makes torch use deterministic
algorithms and pins its cpu threads to `THREADS`; every random value comes from that generator, drawn in a fixed order
by one thread (the prefetching one, or the training's own), so the weights are the same with or without prefetch.  On
mps the caller holds the GPU's exclusive turn (`sfzmaps.gpu.gpu_turn`) for the whole run.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import queue
import threading
from collections.abc import Callable, Iterator

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
PREFETCH = 2            # batches drawn ahead with `Recipe.prefetch`


@dataclasses.dataclass(frozen=True)
class Recipe:
    """`prefetch` draws the next batches in a background thread: the same batches, so it never changes the
    weights."""
    steps: int = 4000
    check_every: int = 250
    lr: float = 1e-3
    batch: int = 4
    paper_weight: float = 0.1
    seed: int = 0
    prefetch: bool = True


@dataclasses.dataclass
class Page:
    """A page to train or validate on.  `lum` is the luminance (1 = paper), `mask` the label's map area (1 inside),
    `prior` the page pass's map probability at full resolution: needed by the pixel pass, None for the page pass."""
    id: SheetId
    lum: np.ndarray
    mask: np.ndarray | None
    prior: np.ndarray | None


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


# --- training loop ---------------------------------------------------------------------------------------------------

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


def _val_loss(val: list[Page]) -> Callable[[Callable[[Page], np.ndarray]], float]:
    """A function of `prob` (a page's map probability): the ink-weighted log-loss over all ink of every validation
    page beyond the label's uncertain strip (`scores.unsure`, found once here, not at every check), pooled (each page
    counts by that ink)."""
    sure = [~scores.unsure(pg.mask) for pg in val]

    def loss(prob: Callable[[Page], np.ndarray]) -> float:
        num = den = 0.0
        for pg, where in zip(val, sure):
            w = ink_weight(pg.lum)
            total = float(w[where].sum(dtype=np.float64))
            if total <= 0:
                continue
            num += scores.log_loss(prob(pg), pg.mask, w, where) * total
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


def _fit(net: nn.Module, draw: Callable[[], object], make: Callable[[object], tuple[torch.Tensor, ...]],
         recipe: Recipe, device: str, val_loss: Callable[[nn.Module], float] | None, snapshots: bool = False,
         label: str = "training") -> tuple[nn.Module, dict]:
    """`recipe.steps` Adam steps, each on the batch `make` builds on the device ((x, target, weight) float32 tensors)
    from what `draw` drew on the cpu; the validation loss every `check_every` steps, logged as "<label> step <s>/<n>:
    val <loss>"; the net back in eval mode on cpu.  With `snapshots`, the info also holds {"snapshots": {step:
    state_dict on cpu}} at every check: the draws do not depend on `recipe.steps` and the validation changes nothing,
    so the snapshot at step s is the net a run of s steps returns.  FloatingPointError on a non-finite loss (read only
    at the checks, so the device is not waited on every step)."""
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=recipe.lr)
    curve, snaps = [], {}
    with _batches(draw, recipe.steps, recipe.prefetch) as batches:
        for step, spec in enumerate(batches, 1):
            with torch.no_grad():
                x, t, w = make(spec)
            loss = _loss(net(x), t, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % recipe.check_every == 0:
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"training loss is {loss.item()} at step {step}")
                if val_loss is not None:
                    checked = float(val_loss(net))
                    curve.append((step, checked))
                    log.info("%s step %d/%d: val %.4f", label, step, recipe.steps, checked)
                if snapshots:
                    snaps[step] = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
    info = {"val_curve": curve}
    if snapshots:
        info["snapshots"] = snaps
    return net.eval().cpu(), info


def _need(pages: list[Page], what: str, attrs: tuple[str, ...]) -> None:
    for pg in pages:
        for a in attrs:
            if getattr(pg, a) is None:
                raise ValueError(f"{what} page {pg.id} has no {a}")


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


def train_coarse(pages: list[Page], recipe: Recipe, device: str, val: list[Page] | None = None,
                 snapshots: bool = False, label: str = "page pass") -> tuple[nn.Module, dict]:
    """The page pass trained on `pages` (see the module); returns (net in eval mode on cpu, {"val_curve": [(step,
    loss), ...]}), the curve empty without `val`; with `snapshots`, also the net's state at every check (`_fit`).
    `label` names the training in the validation lines logged."""
    _need(pages + (val or []), "page-pass", ("mask",))
    rng = _settle(recipe.seed)
    net = predict.new_coarse_net()
    coarse = [_coarse_arrays(pg, recipe.paper_weight) for pg in pages]
    scales = [scale for _, scale in coarse]
    on_device = [tuple(torch.from_numpy(np.ascontiguousarray(a, np.float32)).to(device) for a in arrays)
                 for arrays, _ in coarse]
    val_loss = None
    if val:
        loss = _val_loss(val)
        val_loss = lambda n: loss(lambda pg: predict.predict_coarse(n, pg.lum, device))  # noqa: E731
    return _fit(net, lambda: _coarse_draw(rng, scales, recipe.batch),
                lambda spec: _coarse_make(spec, on_device, device), recipe, device, val_loss, snapshots, label)


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


def _fine_aux(pg: Page) -> tuple[np.ndarray, np.ndarray]:
    """What the pixel pass's crops need of a page beside its arrays, found once per training: (the flat indices of
    the px near the label's outline (`_near_outline`), the uncertain strip (`scores.unsure`))."""
    return _near_outline(pg.mask), scores.unsure(pg.mask)


@dataclasses.dataclass(frozen=True)
class _Crop:
    """A pixel-pass crop as the cpu cuts it: `arrays`, (4, side, side) float32 windows of the luminance (paper off
    the page), the prior, the mask and the sure px (1 beyond the label's uncertain strip) (0 off the page), stacked
    so the device moves them in one go; the page's rows [r0, r1) and columns [c0, c1) within the window; and the
    augmentation."""
    arrays: np.ndarray
    inside: tuple[int, int, int, int]
    aug: Aug


def _fine_crop(rng: np.random.Generator, pg: Page, aux: tuple[np.ndarray, np.ndarray], outline: bool) -> _Crop:
    """One crop, to be read as `predict.predict_tiles` reads a tile once `_fine_make` scales it: a window of
    (TILE + 2 margin) / s px (s within 1 +- SCALE_JITTER), centred on a px near the outline (`outline` and the page
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
    arrays = np.stack([_window(pg.lum, top, left, side, 1.0), _window(pg.prior, top, left, side, 0.0),
                       _window(pg.mask, top, left, side, 0.0), _window(~strip, top, left, side, 0.0)])
    return _Crop(arrays=arrays, inside=inside, aug=draw_aug(rng))


def _fine_draw(rng: np.random.Generator, pages: list[Page], aux: list[tuple[np.ndarray, np.ndarray]],
               batch: int) -> list[_Crop]:
    """`batch` crops, every other one (from the first) an outline crop; pages drawn with replacement; `aux` holds
    each page's `_fine_aux`."""
    crops = []
    for i in range(batch):
        k = int(rng.integers(len(pages)))
        crops.append(_fine_crop(rng, pages[k], aux[k], i % 2 == 0))
    return crops


def _fine_make(spec: list[_Crop], paper_weight: float, device: str):
    """The pixel-pass batch on the device: each crop scaled to TILE + 2 margin (luminance, prior and mask bilinear),
    weighted `paper_weight + ink_weight(lum)` on the core, on the page and beyond the label's uncertain strip only
    (the page's extent and the sure px scaled nearest, so the weight off the page, on the margins and on the strip
    is exactly 0), then augmented."""
    tile, margin = predict.TILE, predict._margin()
    full = tile + 2 * margin
    core = torch.zeros((full, full), dtype=torch.float32, device=device)
    core[margin:margin + tile, margin:margin + tile] = 1.0
    xs, ts, ws = [], [], []
    for c in spec:
        side = c.arrays.shape[-1]
        planes = torch.from_numpy(c.arrays).to(device)
        lum, prior, mask = _resize(planes[:3], full, "bilinear")
        r0, r1, c0, c1 = c.inside
        inside = torch.zeros((side, side), dtype=torch.float32, device=device)
        inside[r0:r1, c0:c1] = 1.0
        inside, sure = _resize(torch.stack([inside, planes[3]]), full, "nearest-exact")
        weight = inside * sure * core * (paper_weight + ink_weight(lum))
        lum, (prior, mask, weight) = apply_aug(c.aug, lum, torch.stack([prior, mask, weight]))
        xs.append(torch.stack([lum, prior]))
        ts.append(mask)
        ws.append(weight)
    return torch.stack(xs), torch.stack(ts)[:, None], torch.stack(ws)[:, None]


def train_fine(pages: list[Page], recipe: Recipe, device: str, val: list[Page] | None = None,
               snapshots: bool = False, label: str = "pixel pass") -> tuple[nn.Module, dict]:
    """The pixel pass trained on `pages` (each with its page-pass `prior`; see the module); returns (net in eval mode
    on cpu, {"val_curve": [(step, loss), ...]}), the curve empty without `val`; with `snapshots`, also the net's
    state at every check (`_fit`).  `label` names the training in the validation lines logged."""
    _need(pages + (val or []), "pixel-pass", ("mask", "prior"))
    rng = _settle(recipe.seed)
    net = predict.new_fine_net()
    aux = [_fine_aux(pg) for pg in pages]
    val_loss = None
    if val:
        loss = _val_loss(val)
        val_loss = lambda n: loss(lambda pg: predict.predict_tiles(n, pg.lum, pg.prior, device))  # noqa: E731
    return _fit(net, lambda: _fine_draw(rng, pages, aux, recipe.batch),
                lambda spec: _fine_make(spec, recipe.paper_weight, device), recipe, device, val_loss, snapshots,
                label)


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
