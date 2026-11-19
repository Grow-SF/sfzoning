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
labelled, not held-out px (every CENTRE_STRIDE-th px each way), and placed inside its page.  The loss is the binary
cross-entropy of the street logit over the tile's labelled, not held-out px.  AdamW (WEIGHT_DECAY) with a one-cycle
schedule rising to MAX_LR over the first PCT_START of the steps.

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
import torch.nn.functional as F
from scipy import ndimage

from sfzmaps import gpu
from sfzmaps.solver.hatchnet import data, model
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


@dataclasses.dataclass(frozen=True)
class Recipe:
    """How the street output is trained (the defaults are the settings above)."""
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


@dataclasses.dataclass
class Fitted:
    """A trained street network: the weights kept (on the CPU), the step they are from, the curve (one row per
    held-out measurement: step, learning rate, mean training loss since the last row, held-out loss and agreement),
    the device, the seconds taken, and the last step's weights (to show what early stopping changed)."""
    state_dict: dict
    best_step: int
    curve: list[dict]
    device: str
    seconds: float
    last_state_dict: dict | None = None


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


def fit_streets(rasters: list[data.SheetRasters], seed: int = 0, device: str | None = None,
                recipe: Recipe = Recipe()) -> Fitted:
    """The street network trained on `rasters` (see the module).  On `mps` the caller holds the GPU's exclusive turn.
    Raises FloatingPointError when a held-out measurement's training loss is not finite."""
    device = device or default_device()
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

def save_model(f: Path, fitted: Fitted, recipe: Recipe, seed: int, last: bool = False) -> Path:
    """Write the street network: the weights kept (or with `last`, the last step's), the step they are from, the
    recipe, the seed, the device it trained on, the step kept and the curve: tensors and plain values only, so
    torch.load's default (weights only) reads it."""
    f = Path(f)
    f.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": fitted.last_state_dict if last else fitted.state_dict,
                "step": recipe.steps if last else fitted.best_step, "recipe": dataclasses.asdict(recipe),
                "seed": seed, "device": fitted.device, "best_step": fitted.best_step, "curve": fitted.curve,
                "torch": str(torch.__version__)}, f)
    return f


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
    out["street_prob_sha256"] = hashlib.sha256(p.tobytes()).hexdigest()
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


def run(series: str, root: Path = data.EXPERIMENT, device: str | None = None, seed: int = 0,
        recipe: Recipe = Recipe(), look_sheets=LOOK_SHEETS) -> dict:
    """Train the street network on every sheet of `series` (data.data_dir), predict every sheet, find its blocks, and
    write under `root`: models/streets.pt, streets/<sheet>.npz (street_prob float32, blocks int32), streets/summary.json
    and the pictures of `look_sheets` in look/."""
    from sfzmaps.solver.hatchnet import look
    root = Path(root)
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
               "best_step": fitted.best_step, "curve": fitted.curve, "state_sha256": state_digest(fitted.state_dict),
               "seconds": {"train": fitted.seconds, "predict": t2 - t1, "total": t2 - t0},
               "held_out": _pooled(stats, "held_out"), "trained": _pooled(stats, "trained"),
               "sheets": stats, "pictures": [str(p) for p in pictures],
               "last_step": {"step": recipe.steps, "held_out": _pooled(last_stats, "held_out"),
                             "trained": _pooled(last_stats, "trained"), "sheets": last_stats}}
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
