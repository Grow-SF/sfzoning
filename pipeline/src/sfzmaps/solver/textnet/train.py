"""Fitting the text detector, by a recipe (`Recipe`; its defaults are today's recipe, the one the 2000 detector of
each quarter of the city trained with before there was one detector).

Each step takes `batch` crops of `crop` x `crop` px from the sheets' pages (`data.SheetData`), centred on a px where a
crop may centre (`data.centres`, taken every CENTRE_STRIDE px each way): TEXT_SHARE of them on text, the rest on any
ink.  Each crop is scaled by a factor drawn log-uniformly from SCALE (so small street lettering also teaches big code
lettering) and turned a random number of quarter turns (the reading angle turns with it); its targets are drawn for it
alone (`data.crop_targets`).

The loss is binary cross-entropy on text over the px it reads, plus KIND_WEIGHT x the cross-entropy of the kind and
ANGLE_WEIGHT x (1 - cosine) of the reading direction, both on the text px it reads; a text-only network (`heads`
"text") has the text's alone.  It reads the known px (`ink_only` "off"), only the known ink ("ink": the chance of text
is only ever read at ink), or the known ink and the px next to it, the 8 around each ink px ("ink+1px").  AdamW
(WEIGHT_DECAY) with a one-cycle schedule rising to `max_lr` over the first `pct_start` of the steps.  With
`weight_average` the model kept is the moving average of the weights over the steps (`WeightAverage`).

The parts of the loss stay on the device, summed; every LOG_EVERY steps (and at the last) their mean over those steps
is read, logged and checked, so the device is waited on only then.  A mean that is not finite stops the training.
Each batch goes to the device without waiting for it either (`Copies`): a waiting copy to the Mac's GPU first waits
for all the work queued before it, so every step would wait for the last.

Every draw comes from one numpy generator and the weights from torch's, both seeded; torch runs deterministic
algorithms on THREADS threads.  With `prefetch` a background thread draws the batches ahead, from that one generator
and in the same order, so the weights are the same bit for bit.  On the Mac's GPU the training runs in the GPU's
exclusive turn (`sfzmaps.gpu`), which the caller takes (`stage.build_text`, `trial.run_trials`).

With `precision="float16"` the forward runs under autocast in half precision (measured 22% faster per step on the M1
Max), with the loss in float32 and a gradient scaler; the weights stay float32, and prediction stays float32.

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
CHOICES = {"ink_only": ("off", "ink", "ink+1px"), "heads": ("all", "text"), "decoder": ("full", "light"),
           "precision": ("float32", "float16")}
NOT_IN_NAME = ("prefetch",)    # recipe settings that never change the weights (the same batches, drawn sooner)


@dataclasses.dataclass(frozen=True)
class Recipe:
    """How the detector is trained.  The defaults are today's recipe (see the module).

      iterations      steps
      max_lr          the one-cycle schedule's peak learning rate
      pct_start       the share of the steps the learning rate rises over
      batch, crop     crops per step, and their side in px
      width           the network's channels at full resolution (`model.TextNet`)
      ink_only        the px the loss reads: "off" every known px, "ink" the known ink, "ink+1px" the known ink and
                      the px next to it
      half_size       the network runs at half the px each way, on the ink folded 2 x 2 into channels (`model.fold`);
                      crops must be even
      heads           "all" outputs (text, kind, reading direction) or "text" alone
      weight_average  the decay of the moving average of the weights kept as the model (`WeightAverage`); 0 is off
      prefetch        the next batches drawn in a background thread (the same batches)
      decoder         "full" two convolutions per block up, or "light" one
      precision       "float32", or "float16": the forward under `torch.autocast` in half precision, the output cast to
                      float32 before the loss, and the loss scaled by a `torch.amp.GradScaler` against underflow of
                      the half-precision gradients (torch 2.15 supports it on cpu and mps).  The weights, the
                      optimiser and the saved model stay float32.  The scaler's step reads its overflow check
                      back from the device each step (`.item()`), which waits for the GPU.
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
                               ("weight_average", 0 <= self.weight_average < 1, "at least 0 and below 1")):
            if not ok:
                raise ValueError(f"recipe: {name} must be {rule}, not {getattr(self, name)}")
        if self.half_size and self.crop % model.FOLD:
            raise ValueError(f"recipe: half_size folds each crop {model.FOLD} x {model.FOLD}, so the crop must be "
                             f"even, not {self.crop}")

    def name(self) -> str:
        """A short id of the weights the recipe trains, the same every run (for folder names): each setting that
        differs from today's (the defaults), sorted by name, as "name=value" (a switch turned on as its name alone),
        joined by "-"; "default" when none differs.  The settings of NOT_IN_NAME, which never change the weights,
        are left out."""
        today = Recipe()
        changed = sorted((f.name, getattr(self, f.name)) for f in dataclasses.fields(self)
                         if f.name not in NOT_IN_NAME and getattr(self, f.name) != getattr(today, f.name))
        return "-".join(k if v is True else f"{k}={v}" for k, v in changed) or "default"


def build_net(recipe: Recipe) -> model.TextNet:
    """The (untrained) network a recipe names: its width, size, outputs and decoder."""
    return model.TextNet(width=recipe.width, outputs=model.OUTPUTS if recipe.heads == "all" else 1,
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


def draw(sheets: list[data.SheetData], pools: tuple[Pool, Pool], recipe: Recipe,
         rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """One batch of `recipe.batch` crops: for each, text or ink, a centre from that pool, a scale and a number of
    turns."""
    out = []
    for _ in range(recipe.batch):
        pool = pools[0] if rng.random() < TEXT_SHARE else pools[1]
        i, y, x = pool[rng.integers(len(pool))]
        scale = float(np.exp(rng.uniform(np.log(SCALE[0]), np.log(SCALE[1]))))
        out.append(data.crop_targets(sheets[i], y, x, recipe.crop, scale, int(rng.integers(4))))
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


def batch_tensors(crops: list[dict[str, np.ndarray]], ink_only: str = "off") -> tuple[torch.Tensor, ...]:
    """A batch's input and targets on the CPU: ink (n, 1, h, w) float, text float, kind int64, angle (degrees) float
    and the loss weight (`loss_weight`) float."""
    def stack(key, dtype):
        return torch.from_numpy(np.stack([c[key] for c in crops]).astype(dtype))
    return (stack("ink", np.float32)[:, None], stack("text", np.float32), stack("kind", np.int64),
            stack("angle", np.float32), torch.from_numpy(loss_weight(crops, ink_only).astype(np.float32)))


def to_tensors(crops: list[dict[str, np.ndarray]], device, ink_only: str = "off") -> tuple[torch.Tensor, ...]:
    """`batch_tensors` on `device`."""
    return tuple(v.to(device) for v in batch_tensors(crops, ink_only))


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
def _batches(sheets: list[data.SheetData], pools: tuple[Pool, Pool], recipe: Recipe, rng: np.random.Generator):
    """The training's batches (`batch_tensors` of `draw`), one per step, from `rng` in order: drawn when asked for or,
    with `recipe.prefetch`, by a background thread up to PREFETCH ahead (which alone uses `rng` meanwhile: the same
    batches, in the same order).  An error in that thread is raised where the batch is asked for."""
    def make():
        return batch_tensors(draw(sheets, pools, recipe, rng), recipe.ink_only)

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


def fit(sheets: list[data.SheetData], recipe: Recipe, device: str | None = None, seed: int = 0) -> model.TextNet:
    """A TextNet fitted on the sheets by `recipe` on `device` (default `default_device()`); see the module.  On `mps`
    the caller holds the GPU's exclusive turn (`gpu.gpu_turn`).  Raises FloatingPointError at the first log whose
    mean loss over the steps since the last is not finite."""
    device = device or default_device()
    settle_torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pools = centre_pools(sheets)
    net = build_net(recipe).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=recipe.max_lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=recipe.max_lr, total_steps=recipe.iterations,
                                                pct_start=recipe.pct_start)
    average = WeightAverage(net, recipe.weight_average) if recipe.weight_average else None
    sums, first, t0 = torch.zeros(3, device=device), 1, time.perf_counter()
    copies = Copies(device)
    half = recipe.precision == "float16"
    scaler = torch.amp.GradScaler(torch.device(device).type, enabled=half)
    with _batches(sheets, pools, recipe, rng) as batches, contextlib.closing(copies):
        for it, batch in enumerate(batches, 1):
            x, t, k, a, w = copies(batch)
            with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=half):
                out = net(x)
            loss, parts = loss_fn(out.float(), t, k, a, w)      # the loss in float32, outside the autocast
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
                    raise FloatingPointError(f"iterations {first}-{it}: the loss is not finite (mean text {mean[0]}, "
                                             f"kind {mean[1]}, angle {mean[2]})")
                log.info("iterations %d-%d: text %.4f, kind %.4f, angle %.4f (%.1f s)", first, it, *mean,
                         time.perf_counter() - t0)
                sums.zero_()
                first = it + 1
    if average is not None:
        average.put_into(net)
    return net


def save_model(out: Path, net: model.TextNet, recipe: Recipe, device: str, seed: int = 0) -> Path:
    """Write a trained detector to `out` (its folder made): the weights (`state_dict`, on the CPU) and the recipe they
    came from (as a dict), with its iterations, the device and the seed.  Returns `out`."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "recipe": dataclasses.asdict(recipe),
                "iterations": recipe.iterations, "device": device, "seed": seed}, out)
    return out
