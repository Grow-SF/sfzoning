"""The text reader's training (stage `solver-text-train`, `sfzmaps solver text-train`): for each quarter of the city
(`paths.QUARTERS`), a reader and a 0/180 classifier that never saw that quarter or any answered block, by spike S2's
recipes (reader C and classifier D), from scratch.

Quarter q's data (`solver-text-crops`): the real crops whose footprint touches neither q nor the answered blocks
(`store.training_rows`: every kind, the non-text windows with target "" weighted 1 - Terra's miss rate), and for the
reader its own synthetic code crops (`synth_<q>`), SYNTH_SHARE of each draw.

A step draws BUCKET x BATCH crops from one seeded numpy generator, sorts them by width and cuts them into BUCKET batches
(less padding), used in a shuffled order.  Each crop is lightly augmented from the same generator: its width scaled
by up to WIDTH_JITTER either way, a few px of paper added at either end, a shift of up to SHIFT px up or down.  The
reader's loss: CTC's negative log likelihood of each crop's string over its length (1 for ""), weighted, the
weighted mean over the batch.  The classifier's: each crop turned a half turn or not (a fair coin from the generator),
binary cross-entropy on the turn.  AdamW (weight decay 1e-4) on a one-cycle schedule, gradients clipped at norm 5, a
fixed number of steps (no early stopping, nothing chosen on held-out crops).  Every LOG_EVERY steps the mean loss is
logged, and for a reader the best-path character error on up to MONITOR words of the held-out quarter on the 2000 scan
(logged only: nothing is chosen by it).

For trials, `fit` also takes crops of our own word boxes and sloppy crops of Terra's words (`extra`, made by
`wordread.readercrops`; never imported here), drawn at the recipe's shares and masked outside the box's own px when it
asks (READER_OPTIONS, all off: the stage's training is as before, draw for draw).

Each training runs in its own exclusive turn on the Mac's GPU (`sfzmaps.gpu`), one after another, so a waiting process
can take the GPU between them.  Seeded (torch and numpy, SEED); spike S2 found two runs of its loop on the GPU
identical step for step.

Writes data/solver/text-train/ (`store.TRAIN_DIR`): <q>.pt and <q>_direction.pt (`model.save`, with the recipe, the
seed and the counts), held.json (no times in it; per quarter the counts, among them the classifier's words whose
Terra angle reads at 180 degrees or more from their axis) and log.json (the steps' losses and times).  The reading
maths (`ctc`) is not imported here: a change to how the readers are read never retrains them.
"""
import json
import logging
import math
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sfzmaps import gpu
from sfzmaps.solver import paths, runs
from sfzmaps.solver.textread import model, store

log = logging.getLogger(__name__)

BATCH = 64
BUCKET = 8
WIDTH_JITTER = 0.12
SHIFT = 2
END_PAD = 6
SYNTH_SHARE = 0.4
LOG_EVERY = 500
LOSS_EVERY = 50          # the loss is read from the device every this many steps (and at the last)
MONITOR = 600
MONITOR_SEED = 7
SEED = 0
THREADS = 4              # torch's CPU threads while training (spike S2's)
READER = {"steps": 12000, "max_lr": 1.5e-3, "pct_start": 0.15, "synth": True}       # reader C
DIRECTION = {"steps": 3000, "max_lr": 2e-3, "pct_start": 0.15, "synth": False}      # classifier D
# The reader's training options, each off (a recipe's own keys override them; READER leaves them out, so what the stage
# writes is as before): `own` and `sloppy` the shares of each draw taken from the crops of our own boxes and the sloppy
# crops of Terra's words passed to `fit` (`extra`; `wordread.readercrops` makes them), `mask` whether every crop
# `extra` gives a mask is blanked outside it (`crops.mask_outside`'s rule; `Pool.joined`): as our boxes are read with
# `read.build_read(mask=True)`.
READER_OPTIONS = {"own": 0.0, "sloppy": 0.0, "mask": False}
WEIGHT_DECAY = 1e-4
CLIP = 5.0


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def augment(crop: np.ndarray, rng) -> np.ndarray:
    """A crop's width scaled by up to WIDTH_JITTER either way, 0-END_PAD px of paper added at each end, shifted by up
    to SHIFT px up or down (paper filling)."""
    h, w = crop.shape
    nw = max(8, int(round(w * math.exp(rng.uniform(-WIDTH_JITTER, WIDTH_JITTER)))))
    out = cv2.resize(crop, (nw, h), interpolation=cv2.INTER_AREA if nw < w else cv2.INTER_LINEAR)
    left, right = int(rng.integers(0, END_PAD + 1)), int(rng.integers(0, END_PAD + 1))
    out = np.pad(out, ((0, 0), (left, right)), constant_values=255)
    s = int(rng.integers(-SHIFT, SHIFT + 1))
    if s:
        out = np.roll(out, s, axis=0)
        if s > 0:
            out[:s] = 255
        else:
            out[s:] = 255
    return out


class Pool:
    """One training's crops (real, then synthetic), their targets and weights."""

    def __init__(self, real: store.Crops, rows: pd.DataFrame, idx: np.ndarray, synth: store.Crops | None = None,
                 synth_texts: list[str] | None = None, monitor: list[tuple[np.ndarray, str]] | None = None):
        self.items = [real[int(i)] for i in idx]
        self.texts = [str(t) for t in rows["text"].to_numpy()[idx]]
        self.weights = [float(w) for w in rows["weight"].to_numpy()[idx]]
        self.n_real = len(self.items)
        self.n_synth = 0
        if synth is not None:
            self.items += [synth[i] for i in range(len(synth))]
            self.texts += list(synth_texts)
            self.weights += [1.0] * len(synth)
            self.n_synth = len(synth)
        self.widths = np.array([c.shape[1] for c in self.items])
        self.monitor = monitor or []
        self.n_own = self.n_sloppy = 0
        self.row_of = np.asarray(idx, np.int64)      # each real crop's row of `rows`
        self.own_p = None

    def joined(self, extra: dict, mask: bool = False) -> "Pool":
        """A new pool: this one's crops, then `extra["own"]` (crops of our own boxes), then `extra["sloppy"]` (sloppy
        crops of Terra's words), each a list of (crop, target, meta: a dict or None), weight 1.  `extra["own_p"]`
        (optional): the chance each own-box crop is drawn by (uniform without).  With `mask`, every crop given a mask
        (bool, the crop's shape: the ink touching its rectangle, `read.kept_ink`) is paper outside it:
        an own-box or sloppy crop by its meta's "mask", a real crop by `extra["real_masks"]` ({row of `rows`:
        mask})."""
        out = Pool.__new__(Pool)
        out.__dict__.update(self.__dict__)
        own, sloppy = list(extra.get("own") or ()), list(extra.get("sloppy") or ())

        def crop(c, m):
            return c if m is None or not mask else np.where(m, c, 255).astype(np.uint8)

        real_masks = extra.get("real_masks") or {}
        base = list(self.items)
        if mask and real_masks:
            for k, row in enumerate(self.row_of):
                base[k] = crop(base[k], real_masks.get(int(row)))
        out.items = base + [crop(c, (m or {}).get("mask")) for c, _, m in own + sloppy]
        out.texts = self.texts + [str(t) for _, t, _ in own] + [str(t) for _, t, _ in sloppy]
        out.weights = self.weights + [1.0] * (len(own) + len(sloppy))
        out.n_own, out.n_sloppy = len(own), len(sloppy)
        out.widths = np.array([c.shape[1] for c in out.items])
        p = extra.get("own_p")
        if p is not None:
            p = np.asarray(p, float)
            if p.shape != (len(own),) or (p < 0).any() or p.sum() <= 0:
                raise ValueError(f"own_p: {p.shape} chances for {len(own)} own-box crops")
            out.own_p = p / p.sum()
        return out

    def batches(self, rng, synth_share: float, own: float = 0.0, sloppy: float = 0.0) -> list[np.ndarray]:
        """BUCKET batches of indices, sorted by width within the draw, in a shuffled order.  With `own` or `sloppy`,
        each draw's crops are split among the real, synthetic, own-box and sloppy crops by one multinomial draw at
        the shares (the real crops the rest), each part drawn alike (the own-box crops by `own_p` when set); without,
        as before."""
        n = BATCH * BUCKET
        if own or sloppy:
            synth = synth_share if self.n_synth else 0.0
            shares = [1.0 - synth - own - sloppy, synth, own, sloppy]
            counts = rng.multinomial(n, shares)
            sizes = [self.n_real, self.n_synth, self.n_own, self.n_sloppy]
            starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
            parts = []
            for g, (s, size, k) in enumerate(zip(starts, sizes, counts)):
                if k:
                    weighted = g == 2 and self.own_p is not None
                    parts.append(s + (rng.choice(size, k, p=self.own_p) if weighted else rng.integers(0, size, k)))
            idx = np.concatenate(parts)
        elif self.n_synth:
            k = rng.binomial(n, synth_share)
            idx = np.concatenate([rng.integers(0, self.n_real, n - k), self.n_real + rng.integers(0, self.n_synth, k)])
        else:
            idx = rng.integers(0, self.n_real, n)
        idx = idx[np.argsort(self.widths[idx], kind="stable")]
        chunks = [idx[i * BATCH:(i + 1) * BATCH] for i in range(BUCKET)]
        return [chunks[j] for j in rng.permutation(BUCKET)]


def targets(texts: list[str]) -> torch.Tensor:
    """The strings as CTC's padded targets (n, longest or 1)."""
    out = torch.zeros((len(texts), max(1, max(len(t) for t in texts))), dtype=torch.long)
    for i, t in enumerate(texts):
        if t:
            out[i, :len(t)] = torch.tensor(model.encode(t))
    return out


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def monitor_cer(net, pool: Pool, device: str) -> float | None:
    """The best-path character error on the pool's monitor words."""
    if not pool.monitor:
        return None
    net.eval()
    errs = chars = 0
    with torch.no_grad():
        order = sorted(range(len(pool.monitor)), key=lambda i: pool.monitor[i][0].shape[1])
        for s in range(0, len(order), 64):
            part = [pool.monitor[i] for i in order[s:s + 64]]
            x, frames = model.to_input([c for c, _ in part], quantize=True)
            lp = F.log_softmax(net(x.to(device)).float(), -1).cpu()
            for k, (_, text) in enumerate(part):
                errs += edit_distance(model.best_path(lp[k, :frames[k]]), text)
                chars += max(1, len(text))
    net.train()
    return errs / chars


def fit(kind: str, pool: Pool, recipe: dict, device: str, seed: int = SEED, tag: str = "",
        extra: dict | None = None) -> tuple:
    """One network ("reader" or "direction") trained on `pool` by `recipe` (see the module), in its own exclusive GPU
    turn on mps.  `extra`: crops of our own boxes and sloppy crops (`Pool.joined`), drawn at the recipe's `own` and
    `sloppy` shares and masked by its `mask` (READER_OPTIONS; at their defaults the draws are as without them).
    Returns the network (on the CPU, in eval mode) and its log."""
    opts = {k: recipe.get(k, v) for k, v in READER_OPTIONS.items()}
    if extra is not None:
        pool = pool.joined(extra, mask=bool(opts["mask"]))
    for name, have in (("own", pool.n_own), ("sloppy", pool.n_sloppy)):
        if opts[name] < 0 or (opts[name] > 0 and not have):
            raise ValueError(f"{tag}: a share {opts[name]} of {name} crops, and {have} of them passed in (extra)")
    if opts["own"] + opts["sloppy"] + (SYNTH_SHARE if pool.n_synth else 0.0) > 1.0:
        raise ValueError(f"{tag}: the shares of own-box, sloppy and synthetic crops sum past 1")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = model.Reader() if kind == "reader" else model.Direction()
    steps = recipe["steps"]
    out = {"kind": kind, "recipe": recipe, "seed": seed, "device": device, "batch": BATCH, "real": pool.n_real,
           "synth": pool.n_synth, "monitor": len(pool.monitor), "torch": torch.__version__, "steps": []}
    if extra is not None:
        out.update(own=pool.n_own, sloppy=pool.n_sloppy)
    with gpu.gpu_turn(True, device):
        t0 = time.perf_counter()
        net = net.to(device)
        net.train()
        opt = torch.optim.AdamW(net.parameters(), lr=recipe["max_lr"], weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=recipe["max_lr"], total_steps=steps,
                                                    pct_start=recipe["pct_start"])
        queue: list = []
        run_loss, run_n = 0.0, 0
        for step in range(1, steps + 1):
            if not queue:
                queue = pool.batches(rng, SYNTH_SHARE, opts["own"], opts["sloppy"])
            idx = queue.pop()
            crops = [augment(pool.items[i], rng) for i in idx]
            if kind == "direction":
                flip = rng.integers(0, 2, len(crops))
                crops = [np.rot90(c, 2) if f else c for c, f in zip(crops, flip)]
                x, _ = model.to_input(crops, quantize=True)
                logit = net(x.to(device))
                loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(flip.astype(np.float32)).to(device))
            else:
                x, frames = model.to_input(crops, quantize=True)
                texts = [pool.texts[i] for i in idx]
                w = torch.tensor([pool.weights[i] for i in idx], dtype=torch.float32)
                lens = torch.tensor([len(t) for t in texts], dtype=torch.long)
                lp = F.log_softmax(net(x.to(device)), -1).permute(1, 0, 2)
                nll = F.ctc_loss(lp, targets(texts).to(device), torch.tensor(frames, dtype=torch.long).to(device),
                                 lens.to(device), blank=model.BLANK, reduction="none", zero_infinity=True)
                loss = (nll / lens.clamp(min=1).to(device) * w.to(device)).sum() / w.sum().to(device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
            sched.step()
            if step % LOSS_EVERY == 0 or step == steps:
                run_loss += float(loss.detach().cpu())
                run_n += 1
            if step % LOG_EVERY == 0 or step == steps:
                mean = run_loss / max(run_n, 1)
                if not math.isfinite(mean):
                    raise FloatingPointError(f"{tag}: loss {mean} at step {step}")
                entry = {"step": step, "loss": round(mean, 5), "lr": sched.get_last_lr()[0],
                         "seconds": round(time.perf_counter() - t0, 1)}
                if kind == "reader":
                    cer = monitor_cer(net, pool, device)
                    entry["monitor_cer"] = None if cer is None else round(cer, 4)
                out["steps"].append(entry)
                log.info("%s %s", tag, entry)
                run_loss, run_n = 0.0, 0
        if gpu.on_gpu(device):
            torch.mps.synchronize()
        out["train_seconds"] = round(time.perf_counter() - t0, 1)
        net = net.cpu()
        if gpu.on_gpu(device):
            torch.mps.empty_cache()
    return net.eval(), out


def monitor_words(real: store.Crops, rows: pd.DataFrame, q: int) -> list[tuple[np.ndarray, str]]:
    """Up to MONITOR words of quarter q (their centre) on the 2000 scan, touching no answered block (seeded)."""
    ok = np.flatnonzero(((rows["series"] == paths.SERIES) & (rows["kind"] != "none") & rows["text"].notna()
                         & (rows["quarter"] == q) & ~rows["touch_blocks"].astype(bool)).to_numpy())
    ok = ok[np.random.default_rng(MONITOR_SEED).permutation(len(ok))[:MONITOR]]
    return [(real[int(i)], str(rows["text"].iat[int(i)])) for i in ok]


def load_inputs() -> tuple[store.Crops, pd.DataFrame]:
    f = store.CROPS_DIR / "crops.parquet"
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver text-crops` (stage solver-text-crops)")
    return store.Crops.load(store.CROPS_DIR / "crops.npz"), pd.read_parquet(f)


def build_train(device: str | None = None, quarters=None) -> dict:
    """Train every quarter's reader and classifier and write TRAIN_DIR (see the module).  `quarters`: a subset of
    quarter names (all when None; for trials only: the stage trains all four).  Returns the logs and seconds."""
    device = device or default_device()
    torch.set_num_threads(THREADS)
    cv2.setNumThreads(1)
    t0 = time.perf_counter()
    real, rows = load_inputs()
    names = list(paths.QUARTERS) if quarters is None else list(quarters)
    part = runs.fresh_folder(store.TRAIN_DIR)
    held, logs = {}, {}
    for name in names:
        q = paths.QUARTERS.index(name)
        idx = store.training_rows(rows, q)
        text_idx = store.training_rows(rows, q, text_only=True)
        syn = store.Crops.load(store.CROPS_DIR / f"synth_{name}.npz")
        syn_rows = pd.read_parquet(store.CROPS_DIR / f"synth_{name}.parquet")
        pool = Pool(real, rows, idx, syn, list(syn_rows["text"]), monitor_words(real, rows, q))
        log.info("%s held out: reader on %d real + %d synthetic crops (%s)", name, pool.n_real, pool.n_synth, device)
        net, lg = fit("reader", pool, READER, device, SEED, tag=f"{name} reader")
        model.save(part / f"{name}.pt", net, quarter=name, recipe=READER, seed=SEED, real=pool.n_real,
                   synth=pool.n_synth)
        del net, pool
        dpool = Pool(real, rows, text_idx)
        log.info("%s held out: 0/180 classifier on %d real crops", name, dpool.n_real)
        dnet, dlg = fit("direction", dpool, DIRECTION, device, SEED, tag=f"{name} direction")
        model.save(part / f"{name}_direction.pt", dnet, quarter=name, recipe=DIRECTION, seed=SEED, real=dpool.n_real)
        del dnet, dpool
        turned = np.mod(rows["angle"].to_numpy(float)[text_idx], 360.0) >= 180.0
        kinds = rows["kind"].to_numpy()[idx]
        held[name] = {"quarter": q, "seed": SEED, "reader": READER, "direction": DIRECTION,
                      "real_crops": int(len(idx)),
                      "real_by_kind": {k: int((kinds == k).sum()) for k in sorted(set(kinds))},
                      "synthetic_crops": int(len(syn)), "direction_crops": int(len(text_idx)),
                      "direction_words_turned": int(turned.sum()),
                      "monitor_words": lg["monitor"], "monitor_cer_last": lg["steps"][-1].get("monitor_cer")}
        logs[name] = {"reader": lg, "direction": dlg}
    (part / "held.json").write_text(json.dumps({"quarters": names, "device": device, "train": held,
                                                "synth_share": SYNTH_SHARE, "batch": BATCH, "bucket": BUCKET},
                                               indent=1) + "\n")
    (part / "log.json").write_text(json.dumps(logs, indent=1) + "\n")
    runs.replace_folder(part, store.TRAIN_DIR)
    return {"held": held, "device": device, "seconds": round(time.perf_counter() - t0, 1)}
