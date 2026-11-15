"""Fitting the text detector with one quarter of the city held out.

Each step takes BATCH crops of CROP x CROP px from the sheets' pages, centred on a px that is not held out (`centre_ok`,
taken every CENTRE_STRIDE px each way): TEXT_SHARE of them on text, the rest on any ink.  Each crop is scaled by a
factor drawn log-uniformly from SCALE (so small street lettering also teaches big code lettering) and turned a random
number of quarter turns (the reading angle turns with it).  The loss is binary cross-entropy on text over the known
px, plus KIND_WEIGHT x the cross-entropy of the kind and ANGLE_WEIGHT x (1 - cosine) of the reading direction, both on
known text px.  AdamW at LEARNING_RATE with a one-cycle schedule.  Every draw comes from one numpy generator and the
weights from torch's, both seeded; torch runs deterministic algorithms on THREADS threads.  On the Mac's GPU the
training waits for its turn (`gpu_turn`): two processes using `mps` at once get wrong numbers back.  A loss that is
not finite stops the training.

Writes data/solver/text-<q>/model.pt: the weights (`state_dict`) with the iterations, device and seed they came from.
"""
import contextlib
import fcntl
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sfzmaps.solver import paths
from sfzmaps.solver.textnet import data, model

log = logging.getLogger(__name__)

ITERATIONS = 8000
CROP = 320              # px, the side of a training crop
BATCH = 12              # crops per step
TEXT_SHARE = 0.5        # the share of crops centred on text (the others on any ink)
SCALE = (0.6, 1.6)      # the range of the factor each crop is scaled by
CENTRE_STRIDE = 4       # crops centre on every this-many-th px each way
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
KIND_WEIGHT = 0.5
ANGLE_WEIGHT = 0.5
THREADS = 4
LOG_EVERY = 250         # iterations between progress lines
MPS_LOCK = Path("/tmp/sfzmaps-mps.lock")   # held by the one process whose turn it is on the Mac's GPU


def default_device() -> str:
    """The device a quarter trains on unless told: "mps" where available, else "cpu"."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


@contextlib.contextmanager
def gpu_turn(device: str):
    """Have the Mac's GPU to this process alone while the body runs, if `device` is "mps": wait until no other
    process holds MPS_LOCK (one file for the whole machine, as the GPU is), then hold it.  Two processes running the
    detector on `mps` at once get wrong numbers back: the same step on the same input gives a different output each
    time (a logit off by up to 2.5), and now and then a gradient that is all NaN; one process alone gets the same bits
    every time.  The CPU takes no turn."""
    if torch.device(device).type != "mps":
        yield
        return
    with open(MPS_LOCK, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("waiting for the GPU: another process holds %s", MPS_LOCK)
            t0 = time.perf_counter()
            fcntl.flock(f, fcntl.LOCK_EX)
            log.info("the GPU is ours after %.0f s", time.perf_counter() - t0)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def settle_torch() -> None:
    """Pin torch's threads and make it use deterministic algorithms only (it raises on an op that has none)."""
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


def centre_pools(sheets: list[data.SheetTargets]) -> tuple[np.ndarray, np.ndarray]:
    """Where crops may centre, as (sheet, row, col) rows (int64): every CENTRE_STRIDE-th px each way that is
    `centre_ok`, on text and on ink."""
    on_text, on_ink = [], []
    k = CENTRE_STRIDE
    for i, t in enumerate(sheets):
        ok = t.centre_ok[::k, ::k]
        for pool, mask in ((on_text, ok & (t.text[::k, ::k] > 0)), (on_ink, ok & t.ink[::k, ::k])):
            r, c = np.nonzero(mask)
            pool.append(np.c_[np.full(len(r), i), r * k, c * k].astype(np.int64))
    pools = np.concatenate(on_text), np.concatenate(on_ink)
    if not len(pools[0]) or not len(pools[1]):
        raise ValueError(f"nothing to train on: {len(pools[0])} text and {len(pools[1])} ink px outside the held-out "
                         "quarter")
    return pools


def crop(t: data.SheetTargets, y: int, x: int, scale: float, turns: int) -> dict[str, np.ndarray]:
    """The training crop centred on page px (row y, col x), moved inside the page: CROP / `scale` px of page resized
    to CROP x CROP (nearest px) and turned `turns` quarter turns counter-clockwise, the reading angle with it (in
    [-180, 180) degrees on text, 0 elsewhere).  Keys: ink, text, kind, angle, known."""
    h, w = t.ink.shape
    src = int(round(CROP / scale))
    y0 = int(np.clip(y - src // 2, 0, h - src))
    x0 = int(np.clip(x - src // 2, 0, w - src))
    sl = (slice(y0, y0 + src), slice(x0, x0 + src))
    angle = np.where(t.text[sl] > 0, (t.angle[sl].astype(np.int32) + 90 * turns + 180) % 360 - 180, 0)
    parts = {"ink": t.ink[sl], "text": t.text[sl], "kind": t.kind[sl], "angle": angle.astype(np.int16),
             "known": t.known[sl]}
    if src != CROP:
        parts = {k: cv2.resize(v.astype(np.uint8) if v.dtype == bool else v, (CROP, CROP),
                               interpolation=cv2.INTER_NEAREST).astype(v.dtype) for k, v in parts.items()}
    return {k: np.rot90(v, turns) for k, v in parts.items()}


def draw(sheets: list[data.SheetTargets], pools: tuple[np.ndarray, np.ndarray],
         rng: np.random.Generator) -> list[dict[str, np.ndarray]]:
    """One batch of BATCH crops: for each, text or ink, a centre from that pool, a scale and a number of turns."""
    out = []
    for _ in range(BATCH):
        pool = pools[0] if rng.random() < TEXT_SHARE else pools[1]
        i, y, x = pool[rng.integers(len(pool))]
        scale = float(np.exp(rng.uniform(np.log(SCALE[0]), np.log(SCALE[1]))))
        out.append(crop(sheets[i], int(y), int(x), scale, int(rng.integers(4))))
    return out


def to_tensors(crops: list[dict[str, np.ndarray]], device) -> tuple[torch.Tensor, ...]:
    """A batch's input and targets on `device`: ink (n, 1, h, w) float, text float, kind int64, angle (degrees) float
    and the loss weight (known) float."""
    def stack(key, dtype):
        return torch.from_numpy(np.stack([c[key] for c in crops]).astype(dtype)).to(device)
    return (stack("ink", np.float32)[:, None], stack("text", np.float32), stack("kind", np.int64),
            stack("angle", np.float32), stack("known", np.float32))


def loss_fn(out: torch.Tensor, t: torch.Tensor, k: torch.Tensor, a: torch.Tensor,
            w: torch.Tensor) -> tuple[torch.Tensor, tuple[float, float, float]]:
    """The loss of a batch (see the module) and its three parts (text, kind, angle).  Px with weight `w` 0 add
    nothing to it."""
    lt = (F.binary_cross_entropy_with_logits(out[:, 0], t, reduction="none") * w).sum() / w.sum().clamp(min=1)
    # Masked sums, not boolean indexing: the backward of indexing has no deterministic kernel on mps.
    m = ((t > 0) & (w > 0)).float()
    n = m.sum().clamp(min=1)
    lk = (F.cross_entropy(out[:, 1:5], torch.where(m > 0, k, torch.zeros_like(k)), reduction="none") * m).sum() / n
    cs = out[:, 5:7]
    rad = torch.deg2rad(a)
    cosine = (cs[:, 0] * torch.cos(rad) + cs[:, 1] * torch.sin(rad)) / cs.norm(dim=1).clamp(min=1e-6)
    la = ((1 - cosine) * m).sum() / n
    return lt + KIND_WEIGHT * lk + ANGLE_WEIGHT * la, (lt.item(), lk.item(), la.item())


def fit(sheets: list[data.SheetTargets], iterations: int = ITERATIONS, device: str | None = None,
        seed: int = 0) -> model.TextNet:
    """A TextNet fitted on the sheets' known px for `iterations` steps on `device` (default `default_device()`), in
    the device's `gpu_turn`.  Raises FloatingPointError at the first step whose loss is not finite."""
    device = device or default_device()
    settle_torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pools = centre_pools(sheets)
    with gpu_turn(device):
        net = model.TextNet().to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LEARNING_RATE, total_steps=iterations)
        t0, running = time.perf_counter(), np.zeros(3)
        for it in range(iterations):
            x, t, k, a, w = to_tensors(draw(sheets, pools, rng), device)
            loss, parts = loss_fn(net(x), t, k, a, w)
            if not np.isfinite(parts).all():
                raise FloatingPointError(f"iteration {it + 1}: the loss is not finite (text {parts[0]}, kind "
                                         f"{parts[1]}, angle {parts[2]})")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running = 0.98 * running + 0.02 * np.array(parts)
            if (it + 1) % LOG_EVERY == 0 or it + 1 == iterations:
                log.info("iteration %d: text %.4f, kind %.4f, angle %.4f (%.0f s)", it + 1, *running,
                         time.perf_counter() - t0)
    return net


def train_quarter(q: str, iterations: int = ITERATIONS, device: str | None = None, seed: int = 0) -> Path:
    """Fit the detector on every warped sheet with quarter `q` held out and write `text_dir(q)/model.pt`."""
    data.quarter_index(q)
    device = device or default_device()
    names = paths.sheet_names()
    if not names:
        raise FileNotFoundError(f"no warped {paths.SERIES} sheets: run the warp-{paths.SERIES} stage first")
    t0 = time.perf_counter()
    sheets = [data.sheet_targets(name, q) for name in names]
    log.info("%d sheets' targets read (%.0f s); training on %s", len(sheets), time.perf_counter() - t0, device)
    net = fit(sheets, iterations, device, seed)
    out = paths.text_dir(q) / "model.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "iterations": iterations,
                "device": device, "seed": seed}, out)
    return out
