"""The learned streets: the network's street output trained on today's blocks placed on the scans, and the printed
blocks found as the connected areas it calls "not street".

The target (`data.SheetRasters.block`): 1 where a px lies in today's blocks as placed on the scan, 0 where it doesn't,
255 unknown (outside the neatline, or another sheet's area).  A px is a street when its block is 0.  These labels are
noisy: today's alleys and lot gaps cross some printed 1921 blocks, today's blocks spill over some printed streets, and
the placement is off by a few ft.  The owner's rule the network has to find in the print: the hatching always falls
inside blocks and never crosses streets.  So it is trained to learn how a street LOOKS, not where today's streets are:

  - a small network (`model.WIDTH`: about 1.2 M weights against 72 M labelled px) that sees only the scan as printed
    and its scale, never a px's position; the noise isn't in the print, so it can only be fitted by memorising;
  - held-out areas (`held_out`): one VAL_CELL square in HELD_OUT on every sheet is never trained on.  Its labels are
    the held-out loss, measured every VAL_EVERY steps on fixed held-out tiles; the weights kept are those of the step
    with the least held-out loss (early stopping: past the point where fitting the training labels stops carrying
    over to areas it never saw, the later weights are not kept);
  - no flips or rotations: 1R's hatch has one fixed map angle (a flip would teach the other diagonal as 1R).  The only
    augmentation is intensity jitter (JITTER): each tile's ink and paper levels move, so the network keys on the
    drawing, not on exact black and white values (the 1948 scan is full-tone, grey ink on cream paper).

Each step takes BATCH tiles of TILE x TILE px, each centred on a training px drawn uniformly from every sheet's
labelled, not held-out px (every CENTRE_STRIDE-th px each way), and placed inside its page.  AdamW (WEIGHT_DECAY) with a
one-cycle schedule rising to MAX_LR over the first PCT_START of the steps.  Two street losses (`Recipe.street_loss`):

  v1  the binary cross-entropy of the street logit against `block` (street where block is 0) over the tile's labelled,
      not held-out px; the weights kept are those of the least held-out loss, measured every VAL_EVERY steps
  v2  (streets v2) three terms, summed (`street_terms`, `street_total`):
        - the street labels' negative log-likelihood under a label-noise model (Mnih & Hinton, ICML 2012), each px
          weighted by `street_w`, over the labelled px (`street` 0 or 1; 255 ignored): the map's label m is the print's
          truth seen through two rates, theta = P(m street | print block) (today's street the 1921 print doesn't show)
          and phi = P(m block | print street) (a 1921 street under today's block), so with p the network's street
          probability P(m = 1) = p (1 - phi) + (1 - p) theta and P(m = 0) = p phi + (1 - p)(1 - theta)
          (`noisy_nll`).  The two rates are fitted jointly with the network, by the same likelihood and the same
          optimiser and schedule, as two sigmoid-parameterised scalars (`NoiseRates`) that start at RATE_INIT;
        - 1 - the clDice recall (`topo`): every centre line covered by the predicted street, each centre px weighted by
          the posterior that it is truly a street, p (1 - phi) / P(m = 1), with p and the rates held fixed
          (`street_posterior`): a centre line across print the network reads as block is put down to theta and not
          forced;
        - 1 - the clDice precision: the predicted street's soft skeleton (`Recipe.iters` rounds) inside the corridors
          (street 1, weighted by street_w), over every px of the sheet's own area (labelled or ignored), so predicted
          street away from every corridor (open water, a blank tract) costs;
      a checkpoint every VAL_EVERY steps, and the one kept is that with the least held-out value of the same sum
      (`select_checkpoint`), its terms pooled over the fixed held-out tiles.

Blocks (`blocks`): the connected areas (4-connected: px that share a side) where the street probability is at most 1/2
(the two-class argmax: a block px is one the network calls more likely block than street), inside the neatline.  No
size, shape or other rule: every such area is a block.

Deterministic: one numpy generator and torch's, both seeded; torch runs deterministic algorithms on THREADS threads.
On the Mac's GPU the caller holds the GPU's exclusive turn (`sfzmaps.gpu`); `run` does.

$0: no API calls.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from sfzmaps import gpu
from sfzmaps.solver.hatchnet import data, model, topo
from sfzmaps.solver.textread.store import save_npz

log = logging.getLogger(__name__)

# The training tile's side.  256 px holds 8 of the widest 1R spacings (about 30 px, at 2.72 ft/px) and a whole text
# knockout (a block number is 40-80 px, the longest street name about 150 px at 2.72 ft/px); it is a little over the
# network's whole field (2 x model.REACH + 1 = 245 px), so a tile's middle px are trained with all the context the
# network can use; and at 4.6-5.4 ft/px it spans 3-4 blocks, at 2.72 ft/px about 2, so every tile has streets and
# blocks.
TILE = 256
BATCH = 16             # tiles per step
STEPS = 3000           # steps: 48 k tiles (the held-out loss decides which step's weights are kept)
MAX_LR = 2e-3          # the one-cycle schedule's peak
PCT_START = 0.3        # the share of the steps the learning rate rises over
WEIGHT_DECAY = 1e-4
# Intensity jitter: each tile's ink -> lo + (hi - lo) ink, lo ~ U(0, JITTER), hi ~ U(1 - JITTER, 1).  A quarter of the
# range each way keeps ink darker than paper always (lo < hi) while no exact level can be keyed on.
JITTER = 0.25
# Held-out areas: page squares of VAL_CELL px, cell (i, j) (row, column) held out when (i + 2 j) % HELD_OUT is
# HELD_OUT - 1: one in five, on every sheet, in every row and column of cells.  1024 px is 2.8-5.5 k ft: most of a
# held-out cell is far from any trained px.
VAL_CELL = 1024
HELD_OUT = 5
VAL_EVERY = 250        # steps between held-out measurements
VAL_TILES = 128        # fixed held-out tiles the held-out loss is measured on (8 M px)
CENTRE_STRIDE = 8      # training tiles centre on every this-many-th labelled px each way (memory only)
THREADS = 4
PRED_TILE = 1024       # px, the side of the square predicted at once (a multiple of model.MULTIPLE; memory only)
CONTEXT = math.ceil(model.REACH / model.MULTIPLE) * model.MULTIPLE    # px of page seen around each predicted square
FOUR = ndimage.generate_binary_structure(2, 1)                        # 4-connected: px that share a side
LOOK_SHEETS = ("20180102094149543", "20180102100525362")             # a dense sheet and the southern sheet
WINDOWS = {"20180102094149543": (2015, 650),       # the zoom's top-left (y, x), upright scan px: the target pictures'
           "20180102100525362": (1495, 2015)}
STREET_LOSSES = ("v1", "v2")
# The noise rates' start (theta = phi = RATE_INIT).  The likelihood alone can't tell the rates from the network: on
# print the network reads as block, P(m = 1) = p (1 - phi) + (1 - p) theta is the same for every (p, theta) on a line,
# so a network that says "street" with p = theta on solid hatching fits as well as one that says p = 0 with the rate
# theta.  The solution where the network reads the print (p near 0 on solid hatching, near 1 on a clear street) and
# the rates hold the labels' noise is reached by starting the rates ABOVE the noise and letting them come down, at
# the network's own learning rate, as the network grows sure (started below, the network takes the noise in instead).
# 1/4 is the middle of the rates a minority of wrong labels can have (0 to 1/2): above any such noise, while the labels
# keep 1 - theta - phi = 1/2 of their pull at the start (at theta + phi = 1 a label says nothing).  This identifies the
# rates only partly (synthetic towns, test_hatchnet_noise*): today's streets across hatched blocks give theta to within
# 0.005, but with labels flipped px by px (theta 0.12, phi 0.03) the network still takes in 19-31% of theta (0.083-0.097
# over three seeds), and phi ends at 0.057-0.072: under the one-cycle schedule a rate's logit moves at most about the
# sum of the learning rates (about 3), so a rate far below 1/4 is not reached.
RATE_INIT = 0.25
# The clDice skeleton depth (Recipe.iters): the widest street's half-width in px.  The widest learned half-width
# (data.summary.json street_labels.half_widths: 1,117 ft) is 207-411 px on the 1921 sheets (5.4-2.72 ft/px), more
# than a tile holds; TILE / 2 is the widest half-width a tile can hold, so every predicted street, or blob, in a tile
# is reduced to its skeleton and none escapes the precision by being wide.  (More rounds than a shape needs add
# nothing to its skeleton; they cost time: about 0.34 s a step on the M1 Max at 128 rounds.)


@dataclasses.dataclass(frozen=True)
class Recipe:
    """How the street output is trained (the defaults are the settings above).  `street_loss` "v1" or "v2" (see the
    module); `skeleton_iters` None means TILE / 2 (`iters`)."""
    steps: int = STEPS
    batch: int = BATCH
    tile: int = TILE
    width: int = model.WIDTH
    max_lr: float = MAX_LR
    pct_start: float = PCT_START
    weight_decay: float = WEIGHT_DECAY
    jitter: float = JITTER
    val_cell: int = VAL_CELL
    val_every: int = VAL_EVERY
    val_tiles: int = VAL_TILES
    street_loss: str = "v2"
    rate_init: float = RATE_INIT
    skeleton_iters: int | None = None

    @property
    def iters(self) -> int:
        """The clDice skeleton depth: `skeleton_iters`, else half the tile (see the note above Recipe)."""
        return self.skeleton_iters if self.skeleton_iters is not None else self.tile // 2


@dataclasses.dataclass
class Fitted:
    """A trained street network: the weights kept (on the CPU), the step they are from, the curve (one row per
    held-out measurement: step, learning rate, training loss since the last row, held-out loss and its terms), the
    device, the seconds taken, and the last step's weights (to show what the selection changed).  With the v2 loss
    also the noise rates kept and last ({"theta", "phi"}) and every checkpoint (step, state_dict, rates, val_loss)."""
    state_dict: dict
    best_step: int
    curve: list[dict]
    device: str
    seconds: float
    last_state_dict: dict | None = None
    rates: dict | None = None
    last_rates: dict | None = None
    checkpoints: list[dict] = dataclasses.field(default_factory=list)


def default_device() -> str:
    """"mps" where available, else "cpu"."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def settle_torch() -> None:
    """Pin torch's threads and make it use deterministic algorithms only (it raises on an op that has none)."""
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


def held_out(rows: np.ndarray, cols: np.ndarray, cell: int = VAL_CELL) -> np.ndarray:
    """Whether each px of the grid `rows` x `cols` (page px) is in a held-out cell (see VAL_CELL)."""
    return ((rows[:, None] // cell + 2 * (cols[None, :] // cell)) % HELD_OUT) == HELD_OUT - 1


def street_target(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(street, known): 1 where the target says street (block 0), and where the target is known (not 255)."""
    return (block == 0), (block != data.UNKNOWN)


def _centres(r: data.SheetRasters, cell: int, held: bool) -> np.ndarray:
    """(n, 2) int32 (row, col) of the sheet's labelled px every CENTRE_STRIDE-th px each way, held out or not."""
    h, w = r.block.shape
    rows, cols = np.arange(0, h, CENTRE_STRIDE), np.arange(0, w, CENTRE_STRIDE)
    sub = (r.block[::CENTRE_STRIDE, ::CENTRE_STRIDE] != data.UNKNOWN) & (held_out(rows, cols, cell) == held)
    i, j = np.nonzero(sub)
    return np.stack([rows[i], cols[j]], 1).astype(np.int32)


def _tile(r: data.SheetRasters, centre, side: int) -> tuple[int, int]:
    h, w = r.ink.shape
    if h < side or w < side:
        raise ValueError(f"{r.sheet}: a page of {r.ink.shape} px is smaller than a {side} px tile")
    return (int(np.clip(centre[0] - side // 2, 0, h - side)), int(np.clip(centre[1] - side // 2, 0, w - side)))


def _cut(r: data.SheetRasters, y0: int, x0: int, side: int, cell: int, held: bool):
    """A tile's ink, street target and the px its loss reads (labelled, and held out or not as `held`)."""
    ys, xs = np.arange(y0, y0 + side), np.arange(x0, x0 + side)
    street, known = street_target(r.block[y0:y0 + side, x0:x0 + side])
    reads = known & (held_out(ys, xs, cell) == held)
    return r.ink[y0:y0 + side, x0:x0 + side], street, reads


class _Pool:
    """Tile centres on many sheets: `pool[j]` is (sheet index, (row, col))."""

    def __init__(self, parts: list[np.ndarray]):
        self.parts = parts
        self.ends = np.cumsum([len(p) for p in parts], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, j: int):
        i = int(np.searchsorted(self.ends, j, side="right"))
        return i, self.parts[i][j - (int(self.ends[i - 1]) if i else 0)]


def _batch(rasters, pool: _Pool, rng: np.random.Generator, recipe: Recipe):
    """One step's tiles, drawn from `rng`: ink (jittered), street target, px read, and ft/px."""
    b, s = recipe.batch, recipe.tile
    x, t, k = np.empty((b, s, s), np.float32), np.empty((b, s, s), np.float32), np.empty((b, s, s), np.float32)
    ft = np.empty(b, np.float32)
    for n in range(b):
        i, c = pool[int(rng.integers(len(pool)))]
        r = rasters[i]
        y0, x0 = _tile(r, c, s)
        ink, street, reads = _cut(r, y0, x0, s, recipe.val_cell, held=False)
        lo, hi = rng.uniform(0.0, recipe.jitter), rng.uniform(1.0 - recipe.jitter, 1.0)
        x[n] = lo + (hi - lo) * ink
        t[n], k[n], ft[n] = street, reads, r.ft_per_px
    return x, t, k, ft


def _held_tiles(rasters, recipe: Recipe, rng: np.random.Generator):
    """The fixed held-out tiles (no jitter): ink, street target, held-out px read, ft/px; None when no sheet has a
    labelled held-out px."""
    pool = _Pool([_centres(r, recipe.val_cell, held=True) for r in rasters])
    if not len(pool):
        return None
    n, s = recipe.val_tiles, recipe.tile
    x, t, k = np.empty((n, s, s), np.float32), np.empty((n, s, s), np.float32), np.empty((n, s, s), np.float32)
    ft = np.empty(n, np.float32)
    for m, j in enumerate(rng.choice(len(pool), size=n, replace=len(pool) < n)):
        i, c = pool[int(j)]
        r = rasters[i]
        y0, x0 = _tile(r, c, s)
        x[m], t[m], k[m] = _cut(r, y0, x0, s, recipe.val_cell, held=True)
        ft[m] = r.ft_per_px
    return x, t, k, ft


def _loss(out: torch.Tensor, t: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The street logit's binary cross-entropy summed over the px read, and how many px were read."""
    bce = F.binary_cross_entropy_with_logits(out[:, model.STREET], t, reduction="none")
    return (bce * k).sum(), k.sum()


@torch.no_grad()
def _measure(net: model.HatchNet, held, device: str, batch: int) -> tuple[float, float]:
    """The held-out loss (mean binary cross-entropy) and agreement (the share of held-out px where the argmax is the
    target) over the fixed held-out tiles, in eval mode."""
    x, t, k, ft = held
    net.eval()
    loss = agree = n = 0.0
    for a in range(0, len(x), batch):
        xs = torch.from_numpy(x[a:a + batch]).to(device)
        ts, ks = torch.from_numpy(t[a:a + batch]).to(device), torch.from_numpy(k[a:a + batch]).to(device)
        out = net(model.inputs(xs, torch.from_numpy(ft[a:a + batch]).to(device)))
        s, c = _loss(out, ts, ks)
        hit = ((out[:, model.STREET] > 0).float() == ts).float()
        loss, agree, n = loss + float(s), agree + float((hit * ks).sum()), n + float(c)
    net.train()
    return (loss / n, agree / n) if n else (float("nan"), float("nan"))


# --- the v2 street loss ---------------------------------------------------------------------------------------------

class NoiseRates(nn.Module):
    """The label-noise model's two rates (see the module), theta = sigmoid(logits[0]) and phi = sigmoid(logits[1]),
    both starting at `start`.  Calling it gives (theta, phi)."""

    def __init__(self, start: float = RATE_INIT):
        super().__init__()
        self.logits = nn.Parameter(torch.full((2,), math.log(start / (1.0 - start))))

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        r = torch.sigmoid(self.logits)
        return r[0], r[1]

    @torch.no_grad()
    def values(self) -> dict:
        theta, phi = self()
        return {"theta": float(theta), "phi": float(phi)}


def _log_likelihoods(z: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor):
    """log P(m = 1) and log P(m = 0) under the noise model, from the street logit `z`, in log space."""
    lp, lq = F.logsigmoid(z), F.logsigmoid(-z)
    l1 = torch.logaddexp(lp + torch.log1p(-phi), lq + torch.log(theta))
    l0 = torch.logaddexp(lp + torch.log(phi), lq + torch.log1p(-theta))
    return lp, l1, l0


def noisy_nll(z: torch.Tensor, m: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Each px's negative log-likelihood of its label `m` (1 street, 0 not) given the street logit `z` and the rates:
    -log(p (1 - phi) + (1 - p) theta) where m is 1, -log(p phi + (1 - p)(1 - theta)) where it is 0.  With theta = phi
    = 0 it is the plain binary cross-entropy."""
    _, l1, l0 = _log_likelihoods(z, theta, phi)
    return -(m * l1 + (1 - m) * l0)


def street_posterior(z: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """The posterior that a px labelled street is truly a printed street, p (1 - phi) / (p (1 - phi) + (1 - p)
    theta), held fixed (no gradient): 1 with theta = 0."""
    with torch.no_grad():
        lp, l1, _ = _log_likelihoods(z, theta, phi)
        return torch.exp(lp + torch.log1p(-phi) - l1)


TERMS = ("nll", "weight", "recall_num", "recall_den", "precision_num", "precision_den")


def street_terms(z: torch.Tensor, m: torch.Tensor, w: torch.Tensor, c: torch.Tensor, v: torch.Tensor,
                 g: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor, iters: int) -> torch.Tensor:
    """The v2 street loss's sums (TERMS order) over the street logits `z` (n, h, w), pooled over the tiles:
    sum w nll, sum w; the recall's sums with the centre px `c` weighted by the posterior; the precision's sums over the
    region `g` against the corridor weights `v`.  m: the labels (1 street, 0 not); w: each px's label weight (0 where
    not read: ignored, or not this pass's part); c: 1 on the centre px read; v: street_w on street px, else 0; g: 1 on
    the px the precision reads (the sheet's own area, this pass's part)."""
    p = torch.sigmoid(z)
    bce = torch.stack([(w * noisy_nll(z, m, theta, phi)).sum(), w.sum()])
    return torch.cat([bce, topo.cldice_sums(p, c * street_posterior(z, theta, phi), v, iters, g)])


def bce_of(t: torch.Tensor) -> torch.Tensor:
    """The weighted mean noisy-label loss of `street_terms`' sums (0 where nothing is labelled)."""
    return torch.where(t[1] > 0, t[0] / t[1].clamp_min(torch.finfo(t.dtype).tiny), torch.zeros_like(t[1]))


def recall_of(t: torch.Tensor) -> torch.Tensor:
    return topo.ratio(t[2], t[3])


def precision_of(t: torch.Tensor) -> torch.Tensor:
    return topo.ratio(t[4], t[5])


def street_total(t: torch.Tensor) -> torch.Tensor:
    """The v2 street loss from `street_terms`' sums: the noisy-label loss + (1 - recall) + (1 - precision)."""
    return bce_of(t) + (1 - recall_of(t)) + (1 - precision_of(t))


def select_checkpoint(vals: list[float]) -> int:
    """The index of the checkpoint kept: the least held-out value, the first of a tie; no finite value: the last."""
    best = None
    for i, v in enumerate(vals):
        if math.isfinite(v) and (best is None or v < vals[best]):
            best = i
    return best if best is not None else len(vals) - 1


def _cut_v2(r: data.SheetRasters, y0: int, x0: int, side: int, cell: int, held: bool):
    """A tile's ink and its v2 label planes (`street_terms`' m, w, c, v, g), for the px held out or not as `held`."""
    ys, xs = np.arange(y0, y0 + side), np.arange(x0, x0 + side)
    cut = np.s_[y0:y0 + side, x0:x0 + side]
    part = held_out(ys, xs, cell) == held
    st, sw = r.street[cut], r.street_w[cut].astype(np.float32)
    street = st == 1
    m = street.astype(np.float32)
    w = np.where((st != data.UNKNOWN) & part, sw, 0.0).astype(np.float32)
    c = ((r.centre[cut] == 1) & street & part).astype(np.float32)
    v = np.where(street, sw, 0.0).astype(np.float32)
    g = ((r.block[cut] != data.UNKNOWN) & part).astype(np.float32)
    return r.ink[cut], m, w, c, v, g


def _batch_v2(rasters, pool: _Pool, rng: np.random.Generator, recipe: Recipe):
    """One step's tiles, drawn from `rng` exactly as `_batch` draws them: ink (jittered), m, w, c, v, g, ft/px."""
    b, s = recipe.batch, recipe.tile
    planes = [np.empty((b, s, s), np.float32) for _ in range(6)]
    ft = np.empty(b, np.float32)
    for n in range(b):
        i, c = pool[int(rng.integers(len(pool)))]
        r = rasters[i]
        y0, x0 = _tile(r, c, s)
        ink, *labels = _cut_v2(r, y0, x0, s, recipe.val_cell, held=False)
        lo, hi = rng.uniform(0.0, recipe.jitter), rng.uniform(1.0 - recipe.jitter, 1.0)
        planes[0][n] = lo + (hi - lo) * ink
        for a, lab in zip(planes[1:], labels):
            a[n] = lab
        ft[n] = r.ft_per_px
    return (*planes, ft)


def _held_tiles_v2(rasters, recipe: Recipe, rng: np.random.Generator):
    """The fixed held-out tiles (no jitter; the same tiles `_held_tiles` draws): ink, m, w, c, v, g, ft/px; None when
    no sheet has a labelled held-out px."""
    pool = _Pool([_centres(r, recipe.val_cell, held=True) for r in rasters])
    if not len(pool):
        return None
    n, s = recipe.val_tiles, recipe.tile
    planes = [np.empty((n, s, s), np.float32) for _ in range(6)]
    ft = np.empty(n, np.float32)
    for k, j in enumerate(rng.choice(len(pool), size=n, replace=len(pool) < n)):
        i, c = pool[int(j)]
        r = rasters[i]
        y0, x0 = _tile(r, c, s)
        for a, plane in zip(planes, _cut_v2(r, y0, x0, s, recipe.val_cell, held=True)):
            a[k] = plane
        ft[k] = r.ft_per_px
    return (*planes, ft)


@torch.no_grad()
def _measure_v2(net: model.HatchNet, rates: NoiseRates, held, device: str, recipe: Recipe) -> dict:
    """The held-out value of the v2 street loss, its terms pooled over the fixed held-out tiles, in eval mode: loss
    (`street_total`), bce, recall, precision; and the agreement (the label-weighted share of labelled held-out px where
    the argmax is the label)."""
    *planes, ft = held
    net.eval()
    theta, phi = rates()
    sums = torch.zeros(len(TERMS), device=device)
    agree = torch.zeros(2, device=device)
    for a in range(0, len(ft), recipe.batch):
        x, m, w, c, v, g = (torch.from_numpy(p[a:a + recipe.batch]).to(device) for p in planes)
        z = net(model.inputs(x, torch.from_numpy(ft[a:a + recipe.batch]).to(device)))[:, model.STREET]
        sums += street_terms(z, m, w, c, v, g, theta, phi, recipe.iters)
        agree += torch.stack([(((z > 0).float() == m).float() * w).sum(), w.sum()])
    net.train()
    t = sums.cpu().double()
    a, n = agree.cpu().double().tolist()
    return {"loss": float(street_total(t)), "bce": float(bce_of(t)), "recall": float(recall_of(t)),
            "precision": float(precision_of(t)), "agree": a / n if n else math.nan}


def _fit_v2(rasters: list[data.SheetRasters], seed: int, device: str, recipe: Recipe) -> Fitted:
    """The v2 street loss's training (see the module and `fit_streets`)."""
    bare = [r.sheet for r in rasters if r.street is None or r.street_w is None or r.centre is None]
    if bare:
        raise ValueError(f"no street labels (street, street_w, centre) on {', '.join(bare)}: run `hatchnet data` "
                         "again, or train with street_loss v1")
    settle_torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    held = _held_tiles_v2(rasters, recipe, np.random.default_rng([seed, 1]))
    pool = _Pool([_centres(r, recipe.val_cell, held=False) for r in rasters])
    if not len(pool):
        raise ValueError("nothing to train on: no labelled px outside the held-out areas")
    net = model.HatchNet(recipe.width).to(device)
    net.train()
    rates = NoiseRates(recipe.rate_init).to(device)
    opt = torch.optim.AdamW([{"params": list(net.parameters())},
                             {"params": list(rates.parameters()), "weight_decay": 0.0}],     # no prior on the rates
                            lr=recipe.max_lr, weight_decay=recipe.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=recipe.max_lr, total_steps=recipe.steps,
                                                pct_start=recipe.pct_start)
    curve, checkpoints = [], []
    sums, first, t0 = torch.zeros(len(TERMS), device=device), 1, time.perf_counter()
    for step in range(1, recipe.steps + 1):
        x, m, w, c, v, g, ft = (torch.from_numpy(a).to(device) for a in _batch_v2(rasters, pool, rng, recipe))
        z = net(model.inputs(x, ft))[:, model.STREET]
        theta, phi = rates()
        terms = street_terms(z, m, w, c, v, g, theta, phi, recipe.iters)
        loss = street_total(terms)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        sums += terms.detach()
        if step % recipe.val_every == 0 or step == recipe.steps:
            t = sums.cpu().double()
            train_loss = float(street_total(t))
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"steps {first}-{step}: the training loss is not finite ({train_loss})")
            val = _measure_v2(net, rates, held, device, recipe) if held else \
                {k: math.nan for k in ("loss", "bce", "recall", "precision", "agree")}
            r = rates.values()
            row = {"step": step, "lr": float(sched.get_last_lr()[0]), "train_loss": train_loss,
                   "train_bce": float(bce_of(t)), "train_recall": float(recall_of(t)),
                   "train_precision": float(precision_of(t)), **{f"val_{k}": v for k, v in val.items()},
                   **r, "seconds": time.perf_counter() - t0}
            curve.append(row)
            checkpoints.append({"step": step, "rates": r, "val_loss": val["loss"],
                                "state_dict": {n: a.detach().cpu().clone() for n, a in net.state_dict().items()}})
            log.info("steps %d-%d: train loss %.4f; held-out loss %.4f (noisy-label %.4f, recall %.4f, precision "
                     "%.4f), agreement %.4f; theta %.4f, phi %.4f (%.0f s)", first, step, train_loss, val["loss"],
                     val["bce"], val["recall"], val["precision"], val["agree"], r["theta"], r["phi"], row["seconds"])
            sums.zero_()
            first = step + 1
    k = select_checkpoint([c["val_loss"] for c in checkpoints])
    kept, last = checkpoints[k], checkpoints[-1]
    return Fitted(kept["state_dict"], kept["step"], curve, str(device), time.perf_counter() - t0,
                  last["state_dict"], kept["rates"], last["rates"], checkpoints)


def fit_streets(rasters: list[data.SheetRasters], seed: int = 0, device: str | None = None,
                recipe: Recipe = Recipe()) -> Fitted:
    """The street network trained on `rasters` with `recipe.street_loss` (see the module).  On `mps` the caller holds
    the GPU's exclusive turn.  Raises FloatingPointError when a held-out measurement's training loss is not finite."""
    device = device or default_device()
    if recipe.street_loss == "v2":
        return _fit_v2(rasters, seed, device, recipe)
    if recipe.street_loss != "v1":
        raise ValueError(f"street_loss is one of {STREET_LOSSES}, not {recipe.street_loss!r}")
    return _fit_v1(rasters, seed, device, recipe)


def _fit_v1(rasters: list[data.SheetRasters], seed: int, device: str, recipe: Recipe) -> Fitted:
    """The v1 street loss's training (Task 2's, unchanged)."""
    settle_torch()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    held = _held_tiles(rasters, recipe, np.random.default_rng([seed, 1]))
    pool = _Pool([_centres(r, recipe.val_cell, held=False) for r in rasters])
    if not len(pool):
        raise ValueError("nothing to train on: no labelled px outside the held-out areas")
    net = model.HatchNet(recipe.width).to(device)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=recipe.max_lr, weight_decay=recipe.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=recipe.max_lr, total_steps=recipe.steps,
                                                pct_start=recipe.pct_start)
    best, best_loss, best_step, curve = None, math.inf, 0, []
    sums, first, t0 = torch.zeros(2, device=device), 1, time.perf_counter()
    for step in range(1, recipe.steps + 1):
        x, t, k, ft = (torch.from_numpy(a).to(device) for a in _batch(rasters, pool, rng, recipe))
        out = net(model.inputs(x, ft))
        s, c = _loss(out, t, k)
        loss = s / c.clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        sums += torch.stack([s.detach(), c.detach()])
        if step % recipe.val_every == 0 or step == recipe.steps:
            s_sum, c_sum = sums.cpu().numpy().tolist()
            train_loss = s_sum / c_sum if c_sum else float("nan")
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"steps {first}-{step}: the training loss is not finite ({train_loss})")
            val_loss, val_agree = _measure(net, held, device, recipe.batch) if held else (math.nan, math.nan)
            row = {"step": step, "lr": float(sched.get_last_lr()[0]), "train_loss": train_loss,
                   "val_loss": val_loss, "val_agree": val_agree, "seconds": time.perf_counter() - t0}
            curve.append(row)
            log.info("steps %d-%d: train loss %.4f, held-out loss %.4f, agreement %.4f (%.0f s)", first, step,
                     train_loss, val_loss, val_agree, row["seconds"])
            if val_loss < best_loss or (best is None and not math.isfinite(val_loss) and step == recipe.steps):
                best_loss, best_step = val_loss, step
                best = {n: v.detach().cpu().clone() for n, v in net.state_dict().items()}
            sums.zero_()
            first = step + 1
    last = {n: v.detach().cpu().clone() for n, v in net.state_dict().items()}
    return Fitted(best, best_step, curve, str(device), time.perf_counter() - t0, last)


def train_streets(rasters: list[data.SheetRasters], seed: int = 0, device: str | None = None,
                  recipe: Recipe = Recipe()) -> dict:
    """The street network's weights (`fit_streets`, the step kept), on the CPU."""
    return fit_streets(rasters, seed, device, recipe).state_dict


@torch.no_grad()
def predict_streets(net: model.HatchNet, sheet: data.SheetRasters, tile: int = PRED_TILE) -> np.ndarray:
    """The street probability (float32) at every px of a sheet's page, on the network's device, in eval mode as given.

    The page is predicted `tile` px square at a time, each seen with CONTEXT px of page around it (the network's whole
    reach, model.REACH, rounded up to model.MULTIPLE) and its end padded with paper to model.MULTIPLE, as one pass
    over the whole page pads it: the square's corner and the context are multiples of model.MULTIPLE, so the pooling
    falls on the whole page's grid and every px gets the numbers one pass over the whole page would give it."""
    if tile % model.MULTIPLE:
        raise ValueError(f"the tile must be a multiple of {model.MULTIPLE}, not {tile}")
    dev = next(net.parameters()).device
    h, w = sheet.ink.shape
    ft = torch.tensor([sheet.ft_per_px], dtype=torch.float32, device=dev)
    out = np.empty((h, w), np.float32)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y0, x0 = max(y - CONTEXT, 0), max(x - CONTEXT, 0)
            y1, x1 = min(y + tile + CONTEXT, h), min(x + tile + CONTEXT, w)
            win = sheet.ink[y0:y1, x0:x1]
            win = np.pad(win, ((0, (-win.shape[0]) % model.MULTIPLE), (0, (-win.shape[1]) % model.MULTIPLE)),
                         constant_values=1.0)                  # paper
            xin = model.inputs(torch.from_numpy(np.ascontiguousarray(win, np.float32))[None].to(dev), ft)
            p = torch.sigmoid(net(xin)[0, model.STREET]).cpu().numpy()
            th, tw = min(tile, h - y), min(tile, w - x)
            out[y:y + th, x:x + tw] = p[y - y0:y - y0 + th, x - x0:x - x0 + tw]
    return out


def blocks(street_prob: np.ndarray, crop: np.ndarray) -> np.ndarray:
    """The printed blocks (int32: 0 none, 1.. each block): the 4-connected areas where `street_prob` is at most 1/2,
    inside the neatline `crop`."""
    lab, _ = ndimage.label((street_prob <= 0.5) & np.asarray(crop, bool), structure=FOUR)
    return lab.astype(np.int32)


# --- the saved model ------------------------------------------------------------------------------------------------

def save_model(f: Path, fitted: Fitted, recipe: Recipe, seed: int, last: bool = False,
               checkpoint: dict | None = None) -> Path:
    """Write the street network: the weights kept (or with `last`, the last step's; or a `checkpoint` of
    `fitted.checkpoints`), the step they are from, their noise rates (v2), the recipe, the seed, the device it trained
    on, the step kept and the curve: tensors and plain values only, so torch.load's default (weights only) reads it."""
    f = Path(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        state, step, rates = checkpoint["state_dict"], checkpoint["step"], checkpoint["rates"]
    elif last:
        state, step, rates = fitted.last_state_dict, recipe.steps, fitted.last_rates
    else:
        state, step, rates = fitted.state_dict, fitted.best_step, fitted.rates
    torch.save({"state_dict": state, "step": step, "rates": rates, "street_loss": recipe.street_loss,
                "recipe": dataclasses.asdict(recipe), "skeleton_iters": recipe.iters,
                "seed": seed, "device": fitted.device, "best_step": fitted.best_step, "curve": fitted.curve,
                "torch": str(torch.__version__)}, f)
    return f


def save_checkpoints(d: Path, fitted: Fitted, recipe: Recipe, seed: int) -> list[Path]:
    """Every checkpoint of a v2 fit, as `save_model` writes one: d/streets_<step>.pt."""
    return [save_model(Path(d) / f"streets_{c['step']:05d}.pt", fitted, recipe, seed, checkpoint=c)
            for c in fitted.checkpoints]


def load_model(f: Path, device: str) -> tuple[model.HatchNet, dict]:
    """A saved street network on `device` in eval mode, and what was saved with it."""
    saved = torch.load(f, map_location="cpu")
    net = model.HatchNet(saved["recipe"]["width"])
    net.load_state_dict(saved["state_dict"])
    return net.to(device).eval(), saved


def state_digest(state: dict) -> str:
    """A sha256 of a state dict's tensors' bytes, in name order (the same weights, the same digest)."""
    h = hashlib.sha256()
    for n in sorted(state):
        h.update(n.encode())
        h.update(state[n].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --- the run --------------------------------------------------------------------------------------------------------

def sheet_stats(r: data.SheetRasters, p: np.ndarray, b: np.ndarray, cell: int = VAL_CELL) -> dict:
    """How the learned streets agree with the target on the sheet's held-out and trained px, and its blocks."""
    street, known = street_target(r.block)
    h, w = r.block.shape
    held = held_out(np.arange(h), np.arange(w), cell)
    said = p > 0.5
    pc = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    out = {"ft_per_px": r.ft_per_px, "shape": list(r.ink.shape)}
    hit, read = street == said, known & held
    out["held_out_cells"] = [[i // cell, j // cell, int(m.sum()), int(hit[i:i + cell, j:j + cell][m].sum())]
                             for i in range(0, h, cell) for j in range(0, w, cell)
                             for m in [read[i:i + cell, j:j + cell]] if m.any()]  # (row, col, px, agreeing px)
    for name, m in (("held_out", known & held), ("trained", known & ~held)):
        n = int(m.sum())
        st, sd = street[m], said[m]
        bce = float(-(np.log(pc[m]) * st + np.log(1 - pc[m]) * ~st).mean()) if n else None
        share = float(st.mean()) if n else None
        out[name] = {"px": n, "agree": float((st == sd).mean()) if n else None, "bce": bce,
                     "target_street_share": share,
                     "street_precision": float(st[sd].mean()) if sd.any() else None,
                     "street_recall": float(sd[st].mean()) if st.any() else None,
                     "agree_n": int((st == sd).sum())}
    sizes = np.bincount(b.ravel())[1:]
    tgt, nt = ndimage.label(r.block == 1, structure=FOUR)
    out["blocks"] = {"count": int(len(sizes)),
                     "size_px_quantiles": {q: float(np.quantile(sizes, q / 100)) for q in (10, 50, 90)}
                     if len(sizes) else {},
                     "largest_px": int(sizes.max()) if len(sizes) else 0,
                     "target_blocks": int(nt),
                     "px_in_blocks": int(sizes.sum()), "crop_px": int(np.count_nonzero(r.crop))}
    if r.street is not None and r.street_w is not None and r.centre is not None:
        out["labels"] = _label_stats(r, said, held)
    out["street_prob_sha256"] = hashlib.sha256(p.tobytes()).hexdigest()
    return out


def _label_stats(r: data.SheetRasters, said: np.ndarray, held: np.ndarray) -> dict:
    """How the learned streets agree with the v2 street labels on the held-out and trained px: the labelled px, their
    weight, the weighted agreement, the street labels called street (weighted), and the centre px called street."""
    lab, street, cen = r.street != data.UNKNOWN, r.street == 1, r.centre == 1
    w = r.street_w.astype(np.float64)
    out = {}
    for name, part in (("held_out", held), ("trained", ~held)):
        m = lab & part
        wm, ws, c = w[m].sum(), w[m & street].sum(), cen & part
        out[name] = {"px": int(m.sum()), "weight": float(wm),
                     "agree": float((w[m] * (said[m] == street[m])).sum() / wm) if wm else None,
                     "street_called_street": float(w[m & street & said].sum() / ws) if ws else None,
                     "centre_px": int(c.sum()), "centre_called_street": float(said[c].mean()) if c.any() else None}
    return out


def _pooled(stats: dict, part: str) -> dict:
    n = sum(s[part]["px"] for s in stats.values())
    agree = sum(s[part]["agree_n"] for s in stats.values())
    bce = sum(s[part]["bce"] * s[part]["px"] for s in stats.values() if s[part]["px"])
    share = sum(s[part]["target_street_share"] * s[part]["px"] for s in stats.values() if s[part]["px"])
    share = share / n if n else None
    prior = (-(share * math.log(share) + (1 - share) * math.log(1 - share))) if share else None
    return {"px": n, "agree": agree / n if n else None, "bce": bce / n if n else None, "target_street_share": share,
            "constant_prior_bce": prior, "majority_agree": max(share, 1 - share) if share is not None else None}


def default_root(street_loss: str) -> Path:
    """Where a run writes by default: the experiment's folder for v1 (Task 2's run), its streets-v2/ for v2."""
    return data.EXPERIMENT if street_loss == "v1" else data.EXPERIMENT / "streets-v2"


def _pooled_labels(stats: dict, part: str) -> dict | None:
    """`_label_stats` pooled over the sheets that have it (weighted means)."""
    rows = [s["labels"][part] for s in stats.values() if "labels" in s]
    if not rows:
        return None
    w, c = sum(r["weight"] for r in rows), sum(r["centre_px"] for r in rows)
    return {"px": sum(r["px"] for r in rows), "weight": w,
            "agree": sum(r["agree"] * r["weight"] for r in rows if r["weight"]) / w if w else None,
            "centre_px": c, "centre_called_street":
                sum(r["centre_called_street"] * r["centre_px"] for r in rows if r["centre_px"]) / c if c else None}


def run(series: str, root: Path | None = None, device: str | None = None, seed: int = 0,
        recipe: Recipe = Recipe(), look_sheets=LOOK_SHEETS) -> dict:
    """Train the street network on every sheet of `series` (data.data_dir), predict every sheet, find its blocks, and
    write under `root` (default: `default_root`): models/streets.pt (the weights kept), models/streets_last.pt, with v2
    models/checkpoints/streets_<step>.pt, streets/<sheet>.npz (street_prob float32, blocks int32),
    streets/summary.json and the pictures of `look_sheets` in look/."""
    from sfzmaps.solver.hatchnet import look
    root = Path(root) if root is not None else default_root(recipe.street_loss)
    files = sorted(data.data_dir(series).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no rasters under {data.data_dir(series)}: run `hatchnet data --series {series}`")
    rasters = [data.read_rasters(f) for f in files]
    device = device or default_device()
    t0 = time.perf_counter()
    stats, last_stats, pictures = {}, {}, []
    with gpu.gpu_turn(True, device):
        fitted = fit_streets(rasters, seed, device, recipe)
        save_model(root / "models" / "streets.pt", fitted, recipe, seed)
        save_model(root / "models" / "streets_last.pt", fitted, recipe, seed, last=True)
        save_checkpoints(root / "models" / "checkpoints", fitted, recipe, seed)
        net, net_last = model.HatchNet(recipe.width), model.HatchNet(recipe.width)
        net.load_state_dict(fitted.state_dict)
        net_last.load_state_dict(fitted.last_state_dict)
        net, net_last = net.to(device).eval(), net_last.to(device).eval()
        t1 = time.perf_counter()
        for r in rasters:
            p = predict_streets(net, r)
            b = blocks(p, r.crop)
            f = root / "streets" / f"{r.sheet}.npz"
            f.parent.mkdir(parents=True, exist_ok=True)
            save_npz(f, street_prob=p, blocks=b, sheet=np.array(r.sheet), series=np.array(series))
            stats[r.sheet] = sheet_stats(r, p, b, recipe.val_cell)
            if r.sheet in look_sheets:
                pictures += look.all_pictures(r, p, b, root / "look", WINDOWS.get(r.sheet), recipe.val_cell)
            pictures.append(look.overview(r, p, b, root / "look", _note(stats[r.sheet])))
            log.info("%s: %d blocks, held-out agreement %s", r.sheet, stats[r.sheet]["blocks"]["count"],
                     stats[r.sheet]["held_out"]["agree"])
            # the last step's weights, measured the same way (not written: they show what early stopping changed)
            p = predict_streets(net_last, r)
            last_stats[r.sheet] = sheet_stats(r, p, blocks(p, r.crop), recipe.val_cell)
        t2 = time.perf_counter()
    summary = {"series": series, "seed": seed, "device": fitted.device, "torch": torch.__version__,
               "recipe": dataclasses.asdict(recipe), "reach_px": model.REACH, "context_px": CONTEXT,
               "weights": int(sum(v.numel() for v in net.parameters())),
               "street_loss": recipe.street_loss, "skeleton_iters": recipe.iters, "rates": fitted.rates,
               "last_rates": fitted.last_rates,
               "best_step": fitted.best_step, "curve": fitted.curve, "state_sha256": state_digest(fitted.state_dict),
               "seconds": {"train": fitted.seconds, "predict": t2 - t1, "total": t2 - t0},
               "held_out": _pooled(stats, "held_out"), "trained": _pooled(stats, "trained"),
               "labels": {k: _pooled_labels(stats, k) for k in ("held_out", "trained")},
               "sheets": stats, "pictures": [str(p) for p in pictures],
               "last_step": {"step": recipe.steps, "held_out": _pooled(last_stats, "held_out"),
                             "trained": _pooled(last_stats, "trained"),
                             "labels": {k: _pooled_labels(last_stats, k) for k in ("held_out", "trained")},
                             "sheets": last_stats}}
    (root / "streets" / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def _note(st: dict) -> str:
    a = st["held_out"]["agree"]
    return f"held-out agreement {a:.3f}" if a is not None else "no labelled held-out px"


def redraw(series: str, root: Path = data.EXPERIMENT, look_sheets=LOOK_SHEETS, cell: int = VAL_CELL) -> list[Path]:
    """The pictures again, from the written street maps: those of `look_sheets`, and every sheet's overview."""
    from sfzmaps.solver.hatchnet import look
    root = Path(root)
    f = root / "streets" / "summary.json"
    sheets = json.loads(f.read_text())["sheets"] if f.is_file() else {}
    out = []
    for g in sorted(data.data_dir(series).glob("*.npz")):
        r = data.read_rasters(g)
        with np.load(root / "streets" / g.name) as z:
            p, b = z["street_prob"], z["blocks"]
        if r.sheet in look_sheets:
            out += look.all_pictures(r, p, b, root / "look", WINDOWS.get(r.sheet), cell)
        out.append(look.overview(r, p, b, root / "look", _note(sheets[r.sheet]) if r.sheet in sheets else ""))
    return out
