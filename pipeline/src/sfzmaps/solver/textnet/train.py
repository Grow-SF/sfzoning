"""Fitting the text detector, by a recipe (`Recipe`; its defaults are the old recipe, the one the 2000 detector of
each quarter of the city trained with before there was one detector).

Each step takes `batch` crops of `crop` x `crop` px from the sheets' pages (`data.SheetData`), centred on a px where a
crop may centre (`data.centres`, taken every CENTRE_STRIDE px each way): TEXT_SHARE of them on text, the rest on any
ink.  Each crop is scaled by a factor drawn log-uniformly from SCALE (so small street lettering also teaches big code
lettering) and turned a random number of quarter turns (the reading angle turns with it); its targets are drawn for it
alone (`data.crop_targets`), by the recipe's label rule (`labels`; "lines-unknown" needs the sheets' line px, which
`zones.prepare` fills before the training).  With `composites` p > 0, each crop is a composite with chance p, made by
the bank handed to `fit` (`composite.Bank`, built before the training; any object with `.crop(rng, sd, y, x, size,
scale, turns)`) from the same crop's centre, scale and turns and the same generator; with p = 0 that chance is never
drawn, so the batches are those of a recipe without composites, byte for byte.

The loss is binary cross-entropy on text over the px it reads, plus KIND_WEIGHT x the cross-entropy of the kind and
ANGLE_WEIGHT x (1 - cosine) of the reading direction, both on the text px it reads; a text-only network (`heads`
"text") has the text's alone.  It reads the known px (`ink_only` "off"), only the known ink ("ink": the chance of text
is only ever read at ink), or the known ink and the px next to it, the 8 around each ink px ("ink+1px").  AdamW
(WEIGHT_DECAY) with a one-cycle schedule rising to `max_lr` over the first `pct_start` of the steps.  With
`weight_average` the model kept is the moving average of the weights over the steps (`WeightAverage`).

With the word heads (`heads` "words": `model.WORD_OUTPUTS` channels; the sheets' word cores, each word's own labelled
rectangle, filled by `wordread.shrink.prepare` before the training), each crop also has its word targets
(`data.crop_targets(..., words=True)`), and the loss adds, to the per-ink head's (channels 0-6), the word terms
WORD_PARTS.  The geometry has no sign and no side order (`model`'s word layout): a word's reading sign can't be told
under the quarter turns, and a signed target gave the axis no gradient.  So with the word heads channels 5-6 are the
reading AXIS, (cos 2t, sin 2t), in the per-ink terms too: the per-ink head's terms are those of "all" (`loss_fn`,
unchanged) given twice the reading angle, so its 1 - cosine is 1 - cos 2(t' - t), the axis's.  Each word term is a
weighted mean over the px it reads, a px weighing `word_tensors`' known: 0 where the px is not known, else 1, or, with
word weights (`data.SheetData.weights`, Terra's doubt from `wordread.doubt.prepare`; none by default), its word's
weight at a core px; a word of weight 0 adds nothing.
  core       binary cross-entropy of the core over the known px, each class weighed by the inverse of its weight in
             the batch (the batch's own counts, no measuring pass and no mining): the mean over the core px plus the
             mean over the other px, divided by the number of the two the batch holds
  dice       1 - Dice of the core's chance over the known px
  centre     the Gaussian regression of the offset to the word's centre over its height (GEO 8-9 against geo's dx,
             dy): the mean over the coordinates and the px of (prediction - target)^2 / 2, so that with its learned
             weight below, exp(-s) L + s / 2 is the mean Gaussian negative log-likelihood of a coordinate (up to its
             constant) with the learned variance e^s; over the known core px whose geometry is known (`geo_known`:
             not a word reaching into a held-out area)
  size       the same of the log length and log height (GEO 10-11), its own learned variance
  axis       1 - cos 2(t' - t) of the axis (channels 5-6, read as (cos 2t, sin 2t)) against its word's, over the known
             core px (a word reaching into a held-out area keeps its axis)
  tag        each known core px's tag (channels 12-15) classified among its crop's classes: each of the crop's words,
             by its mean tag over its core px, and the background, by the mean tag over the crop's known px off the
             cores.  A softmax of -|tag - mean|^2 / T over the classes the crop has, T a learned temperature
             (exp of `WordLoss.log_temp`), and the cross-entropy to the px's own word: no margins
The terms are added by learned weights (homoscedastic uncertainty: Kendall, Gal & Cipolla, CVPR 2018), one learned
log-variance s_i a term (`WordLoss.log_var`):
    L_words = sum_i ( exp(-s_i) L_i + s_i / 2 )
their eq. 7 with s = log sigma^2: 1/sigma^2 L + log sigma, a classification term's form; a regression term's 1/2 in
front of L is a shift of s by log 2, so the one form serves every term.  The log-variances and the log-temperature
start at 0 and are trained with the network (AdamW, no weight decay on them, the one-cycle schedule); `fit` keeps
their last values on the network it returns (`learned_word_loss`), and `save_model` writes them beside the weights.
Every term is dense (masked sums, products and one cross-entropy, no indexing in the backward), so it runs on the
Mac's GPU in deterministic mode.  DICE_EPS and MASS_EPS are numerical guards (a sum divided by nothing), not
decisions.

The parts of the loss stay on the device, summed; every LOG_EVERY steps (and at the last) their mean over those steps
is read, logged and checked, so the device is waited on only then.  A mean that is not finite stops the training.
Each batch goes to the device without waiting for it either (`Copies`): a waiting copy to the Mac's GPU first waits
for all the work queued before it, so every step would wait for the last.

Every draw comes from one numpy generator and the weights from torch's, both seeded; torch runs deterministic
algorithms on THREADS threads.  With `prefetch` a background thread draws the batches ahead, from that one generator
and in the same order, so the weights are the same bit for bit.  On the Mac's GPU the training runs in the GPU's
exclusive turn (`sfzmaps.gpu`), which the caller takes (`stage.build_text`, `trial.run_trials`).

With `precision="float16"` the forward runs under autocast in half precision (22% faster per step on one fixed batch with nothing else
running, but 0-8% in the training loop, within the noise), with the loss in float32 and a gradient scaler; the weights stay float32, and prediction stays float32.

`save_model` writes the weights with the recipe they came from; `predict.load_model` builds the network that recipe
names.
"""
import collections
import contextlib
import dataclasses
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sfzmaps.solver.textnet import data, model

log = logging.getLogger(__name__)

ITERATIONS = 8000
CROP = 320              # px, the side of a training crop
BATCH = 12              # crops per step
LEARNING_RATE = 2e-3    # the one-cycle schedule's peak
PCT_START = 0.3         # the share of the steps the learning rate rises over
TEXT_SHARE = 0.5        # the share of crops centred on text (the others on any ink)
SCALE = (0.6, 1.6)      # the range of the factor each crop is scaled by
CENTRE_STRIDE = 4       # crops centre on every this-many-th px each way
WEIGHT_DECAY = 1e-4
KIND_WEIGHT = 0.5
ANGLE_WEIGHT = 0.5
THREADS = 4
LOG_EVERY = 250         # iterations between progress lines (and reads of the loss from the device)
PREFETCH = 2            # batches drawn ahead with `prefetch`
HELD = 2                # steps a batch is held after its copy to the GPU was queued (`Copies`)
DICE_EPS = 1e-6        # added to Dice's denominator: no px gives 0 / it (numerical; decides nothing)
MASS_EPS = 1e-12       # the least a weighted sum divides by: no weight gives 0 / it = 0 (numerical; decides nothing)
PARTS = ("text", "kind", "angle")
WORD_PARTS = ("core", "dice", "centre", "size", "axis", "tag")
CHOICES = {"ink_only": ("off", "ink", "ink+1px"), "heads": ("all", "text", "words"), "decoder": ("full", "light"),
           "precision": ("float32", "float16"), "labels": data.LABEL_RULES, "input": ("ink", "grey")}
NOT_IN_NAME = ("prefetch",)    # recipe settings that never change the weights (the same batches, drawn sooner)


@dataclasses.dataclass(frozen=True)
class Recipe:
    """How the detector is trained.  The defaults are the old recipe, the quarter models' (see the module; the build's is `stage.DEFAULT_RECIPE`).

      iterations      steps
      max_lr          the one-cycle schedule's peak learning rate
      pct_start       the share of the steps the learning rate rises over
      batch, crop     crops per step, and their side in px
      width           the network's channels at full resolution (`model.TextNet`)
      ink_only        the px the loss reads: "off" every known px, "ink" the known ink, "ink+1px" the known ink and
                      the px next to it
      half_size       the network runs at half the px each way, on the ink folded 2 x 2 into channels (`model.fold`);
                      crops must be even
      heads           "all" outputs (text, kind, reading direction), "text" alone, or "words": all with the word heads
                      (axis, core, centre and size, tag; see the module), their sheets' cores filled by
                      `wordread.shrink.prepare`
                      and the word terms added by learned weights (`WordLoss`)
      weight_average  the decay of the moving average of the weights kept as the model (`WeightAverage`); 0 is off
      prefetch        the next batches drawn in a background thread (the same batches)
      decoder         "full" two convolutions per block up, or "light" one
      precision       "float32", or "float16": the forward under `torch.autocast` in half precision, the output cast to
                      float32 before the loss, and the loss scaled by a `torch.amp.GradScaler` against underflow of
                      the half-precision gradients (torch 2.15 supports it on cpu and mps).  The weights, the
                      optimiser and the saved model stay float32.  The scaler's step reads its overflow check
                      back from the device each step (`.item()`), which waits for the GPU.
      labels          what a word box teaches (`data.crop_targets`): "boxes" every px of it as text, the lines through
                      it too; "lines-unknown" its line px unknown (the sheets' line px filled by `zones.prepare`)
      composites      the share of crops that are composites (`composite.Bank.crop`: real clean words onto real line
                      work, real line pieces across clean words, exact masks); 0 none, and no draw for it
      input           what the network reads: "ink" the 1-bit ink (1 on ink, 0 on paper), or "grey" the page's grey
                      levels over 255 (0 black, 1 white paper; `data.crop_targets(..., grey=True)`, bilinear when a
                      crop is scaled; the sheets must hold them, `load_sheets(grey=True)`).  Targets and the loss's
                      px rules are those of "ink" (the loss's ink is still the 1-bit ink).  A network at full size
                      (`half_size` off, the default) reads every px of the page; the full-size option needs no
                      other setting
    """
    iterations: int = ITERATIONS
    max_lr: float = LEARNING_RATE
    pct_start: float = PCT_START
    batch: int = BATCH
    crop: int = CROP
    width: int = model.WIDTH
    ink_only: str = "off"
    half_size: bool = False
    heads: str = "all"
    weight_average: float = 0.0
    prefetch: bool = False
    decoder: str = "full"
    precision: str = "float32"
    labels: str = "boxes"
    composites: float = 0.0
    input: str = "ink"

    def __post_init__(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.type is bool and not isinstance(v, bool):
                raise ValueError(f"recipe: {f.name} is True or False, not {v!r}")
            if f.type is int and (isinstance(v, bool) or not isinstance(v, (int, np.integer))):
                raise ValueError(f"recipe: {f.name} is a whole number, not {v!r}")
            if f.type is float and (isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating))):
                raise ValueError(f"recipe: {f.name} is a number, not {v!r}")
            if f.type in (int, float):
                object.__setattr__(self, f.name, f.type(v))     # one type per setting: one name, one saved form
            if f.name in CHOICES and v not in CHOICES[f.name]:
                raise ValueError(f"recipe: {f.name} is one of {', '.join(CHOICES[f.name])}, not {v!r}")
        for name, ok, rule in (("iterations", self.iterations >= 1, "at least 1"),
                               ("batch", self.batch >= 1, "at least 1"), ("crop", self.crop >= 1, "at least 1"),
                               ("width", self.width >= 1, "at least 1"), ("max_lr", self.max_lr > 0, "above 0"),
                               ("pct_start", 0 < self.pct_start < 1, "between 0 and 1"),
                               ("weight_average", 0 <= self.weight_average < 1, "at least 0 and below 1"),
                               ("composites", 0 <= self.composites <= 1, "between 0 and 1")):
            if not ok:
                raise ValueError(f"recipe: {name} must be {rule}, not {getattr(self, name)}")
        if self.half_size and self.crop % model.FOLD:
            raise ValueError(f"recipe: half_size folds each crop {model.FOLD} x {model.FOLD}, so the crop must be "
                             f"even, not {self.crop}")
        if self.input == "grey" and self.composites > 0:
            raise ValueError(f"recipe: composites are drawn as ink, with no grey levels: composites must be 0 with "
                             f"input grey, not {self.composites}")
        if self.heads == "words" and self.composites > 0:
            raise ValueError(f"recipe: the word heads learn from data.crop_targets' word targets, which composites "
                             f"don't make: composites must be 0 with heads words, not {self.composites}")

    def name(self) -> str:
        """A short id of the weights the recipe trains, the same every run (for folder names): each setting that
        differs from the old recipe's (the defaults), sorted by name, as "name=value" (a switch turned on as its name
        alone), joined by "-"; "default" when none differs.  The settings of NOT_IN_NAME, which never change the weights,
        are left out."""
        today = Recipe()
        changed = sorted((f.name, getattr(self, f.name)) for f in dataclasses.fields(self)
                         if f.name not in NOT_IN_NAME and getattr(self, f.name) != getattr(today, f.name))
        return "-".join(k if v is True else f"{k}={v}" for k, v in changed) or "default"


def build_net(recipe: Recipe) -> model.TextNet:
    """The (untrained) network a recipe names: its width, size, outputs (`model.HEAD_OUTPUTS`) and decoder."""
    return model.TextNet(width=recipe.width, outputs=model.HEAD_OUTPUTS[recipe.heads],
                         half_size=recipe.half_size, decoder=recipe.decoder)


def default_device() -> str:
    """The device the detector trains on unless told: "mps" where available, else "cpu"."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def settle_torch() -> None:
    """Pin torch's threads and make it use deterministic algorithms only (it raises on an op that has none)."""
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


class Pool:
    """Crop centres on many sheets, kept sheet by sheet (no one array of them all): `pool[j]` is the j-th centre,
    counting through the sheets in order and each sheet's centres in order, as (sheet index, row, col)."""

    def __init__(self, parts: list[np.ndarray]):
        self.parts = parts                                    # each sheet's (n, 2) (row, col)
        self.ends = np.cumsum([len(p) for p in parts], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, j) -> tuple[int, int, int]:
        i = int(np.searchsorted(self.ends, j, side="right"))
        r, c = self.parts[i][j - (int(self.ends[i - 1]) if i else 0)]
        return i, int(r), int(c)


def centre_pools(sheets: list[data.SheetData]) -> tuple[Pool, Pool]:
    """Where crops may centre on the sheets, on text and on ink (`data.centres` every CENTRE_STRIDE-th px),
    data.LOAD_THREADS sheets at a time.  Stops (ValueError) when either pool is empty."""
    with ThreadPoolExecutor(data.LOAD_THREADS) as ex:
        found = list(ex.map(lambda sd: data.centres(sd, CENTRE_STRIDE), sheets))
    pools = Pool([f[0] for f in found]), Pool([f[1] for f in found])
    if not len(pools[0]) or not len(pools[1]):
        raise ValueError(f"nothing to train on: {len(pools[0])} text and {len(pools[1])} ink px where a crop may "
                         "centre")
    return pools


def draw(sheets: list[data.SheetData], pools: tuple[Pool, Pool], recipe: Recipe, rng: np.random.Generator,
         bank=None) -> list[dict[str, np.ndarray]]:
    """One batch of `recipe.batch` crops: for each, text or ink, a centre from that pool, a scale and a number of
    turns; its targets by the recipe's label rule (and, with the word heads, its word targets), or, with chance
    `recipe.composites` (drawn only when it is above 0), a composite of `bank` from the same centre, scale and turns
    (ValueError without a bank)."""
    if recipe.composites > 0 and bank is None:
        raise ValueError(f"a recipe with composites ({recipe.composites}) draws them from a bank: none was given")
    extra = {"words": True} if recipe.heads == "words" else {}
    if recipe.input == "grey":
        extra["grey"] = True
    out = []
    for _ in range(recipe.batch):
        pool = pools[0] if rng.random() < TEXT_SHARE else pools[1]
        i, y, x = pool[rng.integers(len(pool))]
        scale = float(np.exp(rng.uniform(np.log(SCALE[0]), np.log(SCALE[1]))))
        turns = int(rng.integers(4))
        if recipe.composites > 0 and rng.random() < recipe.composites:
            out.append(bank.crop(rng, sheets[i], y, x, recipe.crop, scale, turns))
        else:
            out.append(data.crop_targets(sheets[i], y, x, recipe.crop, scale, turns, labels=recipe.labels, **extra))
    return out


def _next_to(ink: np.ndarray) -> np.ndarray:
    """Each px of (n, h, w) bool crops that is ink or one of the 8 px around an ink px of its crop."""
    h, w = ink.shape[1:]
    p = np.pad(ink, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(ink)
    for dy in range(3):
        for dx in range(3):
            out |= p[:, dy:dy + h, dx:dx + w]
    return out


def loss_weight(crops: list[dict[str, np.ndarray]], ink_only: str = "off") -> np.ndarray:
    """Which px of a batch the loss reads (n, h, w) bool: the known px, only their ink, or their ink and the px next
    to it (`ink_only` "off", "ink" or "ink+1px")."""
    known = np.stack([c["known"] for c in crops])
    if ink_only == "off":
        return known
    ink = np.stack([c["ink"] for c in crops])
    return known & (_next_to(ink) if ink_only == "ink+1px" else ink)


def batch_tensors(crops: list[dict[str, np.ndarray]], ink_only: str = "off",
                  input: str = "ink") -> tuple[torch.Tensor, ...]:
    """A batch's input and targets on the CPU: the input (n, 1, h, w) float (the ink, or with `input` "grey" the
    crops' grey levels over 255; see `Recipe.input`), text float, kind int64, angle (degrees) float and the loss weight
    (`loss_weight`) float."""
    def stack(key, dtype):
        return torch.from_numpy(np.stack([c[key] for c in crops]).astype(dtype))
    return (stack("grey" if input == "grey" else "ink", np.float32)[:, None], stack("text", np.float32),
            stack("kind", np.int64), stack("angle", np.float32),
            torch.from_numpy(loss_weight(crops, ink_only).astype(np.float32)))


def to_tensors(crops: list[dict[str, np.ndarray]], device, ink_only: str = "off",
               input: str = "ink") -> tuple[torch.Tensor, ...]:
    """`batch_tensors` on `device`."""
    return tuple(v.to(device) for v in batch_tensors(crops, ink_only, input))


def loss_fn(out: torch.Tensor, t: torch.Tensor, k: torch.Tensor, a: torch.Tensor,
            w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The loss of a batch (see the module) and its three parts (text, kind, angle) as one tensor on the batch's
    device (reading it waits for the device).  Px of weight `w` 0 add nothing to it.  A text-only network's (one
    output) is its text part; the other two are 0."""
    lt = (F.binary_cross_entropy_with_logits(out[:, 0], t, reduction="none") * w).sum() / w.sum().clamp(min=1)
    if out.shape[1] == 1:
        return lt, torch.stack([lt, torch.zeros_like(lt), torch.zeros_like(lt)])
    # Masked sums, not boolean indexing: the backward of indexing has no deterministic kernel on mps.
    m = ((t > 0) & (w > 0)).float()
    n = m.sum().clamp(min=1)
    lk = (F.cross_entropy(out[:, 1:5], torch.where(m > 0, k, torch.zeros_like(k)), reduction="none") * m).sum() / n
    cs = out[:, 5:7]
    rad = torch.deg2rad(a)
    cosine = (cs[:, 0] * torch.cos(rad) + cs[:, 1] * torch.sin(rad)) / cs.norm(dim=1).clamp(min=1e-6)
    la = ((1 - cosine) * m).sum() / n
    return lt + KIND_WEIGHT * lk + ANGLE_WEIGHT * la, torch.stack([lt, lk, la])


# --- the word heads --------------------------------------------------------------------------------------------------

def word_tensors(crops: list[dict[str, np.ndarray]]) -> tuple[torch.Tensor, ...]:
    """A batch's word targets on the CPU (`data.crop_targets(..., words=True)`): core (n, h, w) float, geo (n, 5, h,
    w) float, tag (n, h, w) int64, known (n, h, w) float, geo_known (n, h, w) float, and groups.  known is each px's
    weight in the word terms: 0 where the px is not known, else 1, or, for a crop with word weights (`weight`, from
    `SheetData.weights`), its word's weight at a core px.  A crop's tags are its core px of weight above 0, their words
    numbered 1, 2, ... in the order of their ids (0 elsewhere); groups is 1 .. the most words of a crop (int64), so the
    loss knows their number without reading the device."""
    def stack(key, dtype):
        return torch.from_numpy(np.stack([c[key] for c in crops]).astype(dtype))
    tags, weights = [], []
    for c in crops:
        w = c["known"].astype(np.float32)
        if "weight" in c:
            w = w * c["weight"]
        ids = np.where(w > 0, c["tag_id"], 0)
        words = np.unique(ids[ids > 0])
        tags.append(np.where(ids > 0, np.searchsorted(words, ids) + 1, 0))
        weights.append(w)
    most = max((int(t.max()) for t in tags), default=0)
    return (stack("core", np.float32), stack("geo", np.float32), torch.from_numpy(np.stack(tags).astype(np.int64)),
            torch.from_numpy(np.stack(weights)), stack("geo_known", np.float32),
            torch.arange(1, most + 1, dtype=torch.int64))


def _balanced_bce(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy weighted by the inverse of each class's weight in the batch (`weight` each px's): the
    weighted mean over the core px plus the weighted mean over the other px, divided by the number of the two
    classes the batch holds (0 with none)."""
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pos, neg = target * weight, (1 - target) * weight
    mass_pos, mass_neg = pos.sum(), neg.sum()
    classes = (mass_pos > 0).to(bce.dtype) + (mass_neg > 0).to(bce.dtype)
    means = (bce * pos).sum() / mass_pos.clamp(min=MASS_EPS) + (bce * neg).sum() / mass_neg.clamp(min=MASS_EPS)
    return means / classes.clamp(min=1)


def _tag_ce(tag_out: torch.Tensor, tag: torch.Tensor, groups: torch.Tensor, weight: torch.Tensor,
            core: torch.Tensor, log_temp: torch.Tensor) -> torch.Tensor:
    """The tag term (see the module): the weighted mean over the tagged core px of the cross-entropy of classifying
    each px's tag (`tag_out` (n, c, h, w)) among its crop's classes, class 0 the background (the crop's known px off
    the cores, `weight` above 0 and `core` 0) and class k its k-th word (`tag`, `groups`: `word_tensors`), by the
    softmax of -|tag - class mean|^2 / exp(`log_temp`) over the classes the crop holds.  Each class's px are found by
    a one-hot (matrix products, no indexing); a class a crop doesn't hold takes no part."""
    n, c, h, w = tag_out.shape
    if groups.numel() == 0:
        return tag_out.sum() * 0
    size = h * w
    words = (tag[:, None] == groups[None, :, None, None]).to(tag_out.dtype).reshape(n, len(groups), size)
    back = ((1 - core) * (weight > 0).to(core.dtype)).to(tag_out.dtype).reshape(n, 1, size)
    hot = torch.cat([back, words], 1)                                       # (n, 1 + K, P)
    count = hot.sum(-1)                                                     # (n, 1 + K) px of each class
    vec = tag_out.reshape(n, c, size).transpose(1, 2)                       # (n, P, c)
    mean = torch.bmm(hot, vec) / count.clamp(min=1)[..., None]              # (n, 1 + K, c); 0 for a class of no px
    far = ((vec ** 2).sum(-1)[..., None] - 2 * torch.bmm(vec, mean.transpose(1, 2))
           + (mean ** 2).sum(-1)[:, None, :]).clamp(min=0)                  # (n, P, 1 + K): |tag - mean|^2
    logit = torch.where((count > 0)[:, None, :], -far * torch.exp(-log_temp), torch.full_like(far, -torch.inf))
    on = (tag > 0).reshape(n, size)
    logit = torch.where(on[..., None], logit, torch.zeros_like(logit))      # px off the cores: read by nothing
    ce = F.cross_entropy(logit.transpose(1, 2), tag.reshape(n, size), reduction="none")
    m = weight.reshape(n, size) * on.to(weight.dtype)
    return (ce * m).sum() / m.sum().clamp(min=MASS_EPS)


def word_loss(out: torch.Tensor, core: torch.Tensor, geo: torch.Tensor, tag: torch.Tensor, known: torch.Tensor,
              geo_known: torch.Tensor, groups: torch.Tensor, log_var: torch.Tensor | None = None,
              log_temp: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The word heads' loss of a batch (see the module): sum_i exp(-s_i) L_i + s_i / 2 over the WORD_PARTS L_i, by
    the log-variances `log_var` (s, one a part) and the tag's log-temperature `log_temp` (both 0, their start, when
    not given: `WordLoss` holds the learned ones), and the parts themselves (unweighted) as one tensor on the batch's
    device; `out` the network's WORD_OUTPUTS channels, the rest `word_tensors`.  Each px counts by its weight
    (`known`).  The centre and size are read only at core px whose geometry is known (`geo_known`), the axis at every
    known core px."""
    if log_var is None:
        log_var = out.new_zeros(len(WORD_PARTS))
    if log_temp is None:
        log_temp = out.new_zeros(())
    logit = out[:, model.CORE]
    lc = _balanced_bce(logit, core, known)
    p = torch.sigmoid(logit)
    ld = 1 - 2 * (p * core * known).sum() / ((p * known).sum() + (core * known).sum() + DICE_EPS)
    on = core * known
    boxed = on * geo_known
    mass = boxed.sum().clamp(min=MASS_EPS)
    half_sq = (out[:, model.GEO] - geo[:, :4]) ** 2 / 2                     # (n, 4, h, w)
    lcentre = (half_sq[:, 0:2].mean(1) * boxed).sum() / mass
    lsize = (half_sq[:, 2:4].mean(1) * boxed).sum() / mass
    cs = out[:, model.AXIS]
    rad = torch.deg2rad(2 * geo[:, 4])
    cosine = (cs[:, 0] * torch.cos(rad) + cs[:, 1] * torch.sin(rad)) / cs.norm(dim=1).clamp(min=1e-6)
    la = ((1 - cosine) * on).sum() / on.sum().clamp(min=MASS_EPS)
    lt = _tag_ce(out[:, model.TAG], tag, groups, known, core, log_temp)
    parts = torch.stack([lc, ld, lcentre, lsize, la, lt])
    return (torch.exp(-log_var) * parts + log_var / 2).sum(), parts


class WordLoss(torch.nn.Module):
    """The word terms' learned settings (see the module), trained with the network: `log_var` (one log-variance s_i
    a part of WORD_PARTS, its weight exp(-s_i)) and `log_temp` (the tag's log-temperature), all starting at 0.
    Calling it gives `word_loss` by them."""

    def __init__(self):
        super().__init__()
        self.log_var = torch.nn.Parameter(torch.zeros(len(WORD_PARTS)))
        self.log_temp = torch.nn.Parameter(torch.zeros(()))

    def forward(self, out: torch.Tensor, *targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return word_loss(out, *targets, log_var=self.log_var, log_temp=self.log_temp)

    @torch.no_grad()
    def learned(self) -> dict:
        """The settings as plain numbers (reading them waits for the device): each part's log-variance and weight
        exp(-s), and the tag's temperature."""
        s = self.log_var.detach().cpu().double().tolist()
        return {"log_var": dict(zip(WORD_PARTS, s)), "weight": {k: float(np.exp(-v)) for k, v in zip(WORD_PARTS, s)},
                "temperature": float(np.exp(float(self.log_temp.detach().cpu())))}


class WeightAverage:
    """The exponential moving average of a network's state, taken after every step (`Recipe.weight_average`, the
    decay d): every floating-point entry, the weights and the batch norms' running statistics.  After t steps it is
    sum over steps s of (1 - d) d^(t - s) w_s, divided by 1 - d^t so the steps' shares sum to 1 (the untrained state
    has none).  Whole-number entries (the batch norms' step counts) stay the network's own."""

    def __init__(self, net: model.TextNet, decay: float):
        self.decay, self.steps = decay, 0
        self.sums = {k: torch.zeros_like(v) for k, v in net.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, net: model.TextNet) -> None:
        """Add the network's state after a step."""
        state = net.state_dict()
        for k, s in self.sums.items():
            s.mul_(self.decay).add_(state[k], alpha=1 - self.decay)
        self.steps += 1

    @torch.no_grad()
    def put_into(self, net: model.TextNet) -> None:
        """Set the network's floating-point state to the average."""
        state = net.state_dict()
        share = 1 - self.decay ** self.steps
        for k, s in self.sums.items():
            state[k].copy_(s / share)


class Copies:
    """Batches copied to a device without waiting (`non_blocking`), one call per step.  On the Mac's GPU such a copy is
    only queued: it reads the batch's CPU memory when the GPU gets to it.  So each batch is held, with an event
    recorded after its copy (`event`, by default `torch.mps.Event`), until HELD steps later, when that event is waited
    for (the GPU is then usually past it, so the wait costs nothing) and the batch let go; `close` waits for and lets
    go of the rest.  On the CPU the batch is its own copy and nothing is held."""

    def __init__(self, device, event=None):
        self.device = torch.device(device)
        self.event = event if event is not None else (torch.mps.Event if self.device.type == "mps" else None)
        self.held = collections.deque()

    def __call__(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        out = tuple(v.to(self.device, non_blocking=True) for v in batch)
        if self.event is not None:
            done = self.event()
            done.record()
            self.held.append((batch, done))
            while len(self.held) > HELD:
                self._let_go()
        return out

    def _let_go(self) -> None:
        batch, done = self.held.popleft()
        done.synchronize()          # the copy has read the batch: only now may its memory go
        del batch

    def close(self) -> None:
        while self.held:
            self._let_go()


@contextlib.contextmanager
def _batches(sheets: list[data.SheetData], pools: tuple[Pool, Pool], recipe: Recipe, rng: np.random.Generator,
             bank=None):
    """The training's batches (`batch_tensors` of `draw`, composites from `bank`; with the word heads followed by
    `word_tensors`), one per step, from `rng` in order: drawn when asked for or, with `recipe.prefetch`, by a
    background thread up to PREFETCH ahead (which alone uses `rng` and the bank meanwhile: the same batches, in the
    same order).  An error in that thread is raised where the batch is asked for."""
    def make():
        crops = draw(sheets, pools, recipe, rng, bank)
        grey = {"input": "grey"} if recipe.input == "grey" else {}     # the ink recipe's call is today's
        if recipe.heads == "words":
            return batch_tensors(crops, recipe.ink_only, **grey) + word_tensors(crops)
        return batch_tensors(crops, recipe.ink_only, **grey)

    if not recipe.prefetch:
        yield (make() for _ in range(recipe.iterations))
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
            for _ in range(recipe.iterations):
                if not put(make()):
                    return
        except BaseException as e:      # noqa: BLE001 - handed to the training's thread, raised there
            put(e)

    def consume():
        for _ in range(recipe.iterations):
            item = ready.get()
            if isinstance(item, BaseException):
                raise item
            yield item

    worker = threading.Thread(target=produce, name="textnet-prefetch", daemon=True)
    worker.start()
    try:
        yield consume()
    finally:
        stop.set()
        worker.join()


def fit(sheets: list[data.SheetData], recipe: Recipe, device: str | None = None, seed: int = 0,
        bank=None) -> model.TextNet:
    """A TextNet fitted on the sheets by `recipe` on `device` (default `default_device()`), its composites (if the
    recipe has any) from `bank`; see the module.  On `mps` the caller holds the GPU's exclusive turn (`gpu.gpu_turn`).
    ValueError, before any training, for a recipe with composites and no bank.  Raises FloatingPointError at the first
    log whose mean loss over the steps since the last is not finite."""
    if recipe.composites > 0 and bank is None:
        raise ValueError(f"a recipe with composites ({recipe.composites}) draws them from a bank: none was given")
    device = device or default_device()
    settle_torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pools = centre_pools(sheets)
    net = build_net(recipe).to(device)
    crit = WordLoss().to(device) if recipe.heads == "words" else None
    trained = net.parameters() if crit is None else [{"params": net.parameters()},
                                                      {"params": crit.parameters(), "weight_decay": 0.0}]
    opt = torch.optim.AdamW(trained, lr=recipe.max_lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=recipe.max_lr, total_steps=recipe.iterations,
                                                pct_start=recipe.pct_start)
    average = WeightAverage(net, recipe.weight_average) if recipe.weight_average else None
    names = PARTS + (WORD_PARTS if recipe.heads == "words" else ())
    sums, first, t0 = torch.zeros(len(names), device=device), 1, time.perf_counter()
    copies = Copies(device)
    half = recipe.precision == "float16"
    scaler = torch.amp.GradScaler(torch.device(device).type, enabled=half)
    with _batches(sheets, pools, recipe, rng, bank) as batches, contextlib.closing(copies):
        for it, batch in enumerate(batches, 1):
            x, t, k, a, w, *words = copies(batch)
            with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=half):
                out = net(x)
            out = out.float()                                   # the loss in float32, outside the autocast
            # with the word heads channels 5-6 are the axis: the per-ink terms are given twice the angle (the module)
            loss, parts = loss_fn(out, t, k, 2 * a if words else a, w)
            if words:
                more, word_parts = crit(out, *words)
                loss, parts = loss + more, torch.cat([parts, word_parts])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if average is not None:
                average.update(net)
            sums += parts.detach()
            if it % LOG_EVERY == 0 or it == recipe.iterations:
                mean = (sums / (it - first + 1)).cpu().numpy()
                if not np.isfinite(mean).all():
                    raise FloatingPointError(f"iterations {first}-{it}: the loss is not finite (mean "
                                             + ", ".join(f"{n} {v}" for n, v in zip(names, mean)) + ")")
                log.info("iterations %d-%d: " + ", ".join(f"{n} %.4f" for n in names) + " (%.1f s)", first, it,
                         *mean, time.perf_counter() - t0)
                if crit is not None:
                    got = crit.learned()
                    log.info("word terms' learned weights %s, tag temperature %.4g",
                             ", ".join(f"{k} {v:.4g}" for k, v in got["weight"].items()), got["temperature"])
                sums.zero_()
                first = it + 1
    if average is not None:
        average.put_into(net)
    if crit is not None:
        net.learned_word_loss = crit.learned()      # a plain attribute: not in the state, kept by `save_model`
    return net


def save_model(out: Path, net: model.TextNet, recipe: Recipe, device: str, seed: int = 0) -> Path:
    """Write a trained detector to `out` (its folder made): the weights (`state_dict`, on the CPU) and the recipe they
    came from (as a dict), with its iterations, the device and the seed; with the word heads, also the word terms'
    learned weights and temperature (`word_loss`, from `fit`'s `learned_word_loss`).  Returns `out`."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved = {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "recipe": dataclasses.asdict(recipe),
             "iterations": recipe.iterations, "device": device, "seed": seed}
    if getattr(net, "learned_word_loss", None) is not None:
        saved["word_loss"] = net.learned_word_loss
    torch.save(saved, out)
    return out
