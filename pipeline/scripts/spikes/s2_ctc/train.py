"""Spike S2, step 2: train one reader on the Mac's GPU, in one exclusive turn (`sfzmaps.gpu.gpu_turn`).

Runs in the docTR overlay (docTR's model classes; the project's torch version pinned):

    uv run --with python-doctr --with torch==2.15.0 python scripts/spikes/s2_ctc/train.py B [--steps N] [--seed S]

Readers (survey section 4.6, step 2; all docTR's `crnn_mobilenet_v3_small`, our VOCAB, CTC with docTR's last-class
blank):
  A  fine-tuned from docTR's published weights (a baseline only: their training data is undisclosed).  The output layer
     starts from the pretrained rows of the characters both vocabularies share (and the blank).
  B  the same architecture from scratch (no pretrained backbone either).
  C  B plus the synthetic crops of synth.py (SYNTH_SHARE of each batch).
  D  the 0/180-degree crop classifier: a tiny CNN, p(the crop is upside down), on the training text crops as cut
     (Terra's direction, label 0) and turned a half turn (label 1).

Data: crops.py's training split (every kind; non-text crops with target "" weighted by 1 - Terra's miss rate).  Each
step draws BUCKET x BATCH crops from one seeded numpy generator, sorts them by width and cuts them into BUCKET batches
(less padding), in a shuffled order.  Light augmentation from the same generator: the width scaled by up to WIDTH_JITTER
either way, a few px of paper added at either end, a shift of up to SHIFT px up or down.  Loss: CTC's negative log
likelihood of each crop's string over its length (1 for ""), weighted, mean over the batch.  AdamW, one-cycle.  Every
LOG_EVERY steps the mean loss is logged, and the best-path character error on the monitor set (the validation area's
crops, which nothing is trained or scored on) is measured.  A fixed number of steps (no early stopping).

Writes data/solver/experiments/spike-s2/models/<reader>.pt (state dict) and <reader>.json (recipe, log, seconds).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

BATCH = 64
BUCKET = 8
WIDTH_JITTER = 0.12
SHIFT = 2
END_PAD = 6
SYNTH_SHARE = 0.4
LOG_EVERY = 500
MONITOR = 600
RECIPES = {
    "A": {"steps": 8000, "max_lr": 5e-4, "pct_start": 0.1, "pretrained": True, "synth": False},
    "B": {"steps": 12000, "max_lr": 1.5e-3, "pct_start": 0.15, "pretrained": False, "synth": False},
    "C": {"steps": 12000, "max_lr": 1.5e-3, "pct_start": 0.15, "pretrained": False, "synth": True},
    "D": {"steps": 3000, "max_lr": 2e-3, "pct_start": 0.15, "pretrained": False, "synth": False},
}


class Direction(nn.Module):
    """p(upside down) of a crop: four conv blocks, global average pool, one logit."""

    def __init__(self, width: int = 32):
        super().__init__()
        chans = [1, width, width * 2, width * 2, width * 4]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(chans[-1], 1)

    def forward(self, x):
        f = self.body(x[:, :1])
        return self.head(f.mean(dim=(2, 3)))[:, 0]


def build(reader: str):
    if reader == "D":
        return Direction()
    from doctr.models import crnn_mobilenet_v3_small
    if not RECIPES[reader]["pretrained"]:
        return crnn_mobilenet_v3_small(pretrained=False, pretrained_backbone=False, vocab=C.VOCAB)
    model = crnn_mobilenet_v3_small(pretrained=True, vocab=C.VOCAB)
    ref = crnn_mobilenet_v3_small(pretrained=True)            # docTR's own vocabulary and output layer
    rows = {c: i for i, c in enumerate(ref.vocab)}
    shared = [c for c in C.VOCAB if c in rows]
    with torch.no_grad():
        for i, c in enumerate(C.VOCAB):
            if c in rows:
                model.linear.weight[i] = ref.linear.weight[rows[c]]
                model.linear.bias[i] = ref.linear.bias[rows[c]]
        model.linear.weight[C.BLANK] = ref.linear.weight[len(ref.vocab)]
        model.linear.bias[C.BLANK] = ref.linear.bias[len(ref.vocab)]
    print(f"A: output rows from the pretrained layer for {len(shared)}/{len(C.VOCAB)} characters "
          f"(missing: {''.join(c for c in C.VOCAB if c not in rows)!r}) and the blank", flush=True)
    return model


def augment(crop: np.ndarray, rng) -> np.ndarray:
    import cv2
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
    """The training crops (and the synthetic ones for C) with their targets and weights."""

    def __init__(self, reader: str):
        crops = C.load_crops()
        m = crops.meta
        train = np.flatnonzero((m["split"] == "train").to_numpy())
        if reader == "D":
            train = train[(m["kind"].to_numpy()[train] != "none")]
        self.items = [crops[i] for i in train]
        self.texts = [m["text"].iat[i] for i in train]
        self.weights = [float(m["weight"].iat[i]) for i in train]
        self.n_real = len(self.items)
        self.n_synth = 0
        if RECIPES[reader]["synth"]:
            syn = C.Crops(C.OUT / "synth.npz", C.OUT / "synth_meta.pkl")
            self.items += [syn[i] for i in range(len(syn))]
            self.texts += list(syn.meta["text"])
            self.weights += [1.0] * len(syn)
            self.n_synth = len(syn)
        self.widths = np.array([c.shape[1] for c in self.items])
        mon = np.flatnonzero(((m["split"] == "") & m["touch_validation"] & ~m["touch_quarter"] & ~m["touch_blocks"]
                              & m["text"].notna() & (m["kind"] != "none")).to_numpy())
        mon = mon[np.random.default_rng(7).permutation(len(mon))[:MONITOR]]
        self.monitor = [(crops[i], m["text"].iat[i]) for i in mon]

    def batches(self, rng, synth_share: float):
        """BUCKET batches of indices, sorted by width within the draw, in a shuffled order."""
        n = BATCH * BUCKET
        if self.n_synth:
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
            out[i, :len(t)] = torch.tensor(C.encode(t))
    return out


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def monitor_cer(model, pool: Pool, device: str) -> float:
    model.eval()
    errs = chars = 0
    with torch.no_grad():
        order = sorted(range(len(pool.monitor)), key=lambda i: pool.monitor[i][0].shape[1])
        for s in range(0, len(order), 64):
            part = [pool.monitor[i] for i in order[s:s + 64]]
            x, frames = C.to_input([c for c, _ in part], quantize=True)
            lp = F.log_softmax(C.forward(model, x.to(device)).float(), -1).cpu()
            for k, (_, text) in enumerate(part):
                got = C.best_path(lp[k, :frames[k]])
                errs += edit_distance(got, text)
                chars += max(1, len(text))
    model.train()
    return errs / chars


def train(reader: str, steps: int | None, seed: int, device: str, tag: str | None = None) -> dict:
    from sfzmaps import gpu
    recipe = dict(RECIPES[reader])
    if steps:
        recipe["steps"] = steps
    tag = tag or reader
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t_load = time.perf_counter()
    pool = Pool(reader)
    log = {"reader": reader, "recipe": recipe, "seed": seed, "device": device, "batch": BATCH,
           "real": pool.n_real, "synth": pool.n_synth, "steps": [], "torch": torch.__version__}
    print(f"{tag}: {pool.n_real} real crops, {pool.n_synth} synthetic, monitor {len(pool.monitor)} "
          f"({time.perf_counter() - t_load:.0f} s)", flush=True)
    model = build(reader)
    with gpu.gpu_turn(True, device):
        t0 = time.perf_counter()
        model = model.to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=recipe["max_lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=recipe["max_lr"], total_steps=recipe["steps"],
                                                    pct_start=recipe["pct_start"])
        queue: list = []
        run_loss, run_n = 0.0, 0
        for step in range(1, recipe["steps"] + 1):
            if not queue:
                queue = pool.batches(rng, SYNTH_SHARE)
            idx = queue.pop()
            crops = [augment(pool.items[i], rng) for i in idx]
            if reader == "D":
                flip = rng.integers(0, 2, len(crops))
                crops = [np.rot90(c, 2) if f else c for c, f in zip(crops, flip)]
                x, _ = C.to_input(crops, quantize=True)
                logit = model(x.to(device))
                loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(flip.astype(np.float32)).to(device))
            else:
                x, frames = C.to_input(crops, quantize=True)
                texts = [pool.texts[i] for i in idx]
                w = torch.tensor([pool.weights[i] for i in idx], dtype=torch.float32)
                lens = torch.tensor([len(t) for t in texts], dtype=torch.long)
                tgt = targets(texts)
                logits = C.forward(model, x.to(device))
                lp = F.log_softmax(logits, -1).permute(1, 0, 2)
                nll = F.ctc_loss(lp, tgt.to(device), torch.tensor(frames, dtype=torch.long).to(device),
                                 lens.to(device), blank=C.BLANK, reduction="none", zero_infinity=True)
                loss = (nll / lens.clamp(min=1).to(device) * w.to(device)).sum() / w.sum().to(device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run_loss += float(loss.detach().cpu()) if step % 50 == 0 or step == recipe["steps"] else 0.0
            run_n += 1 if step % 50 == 0 or step == recipe["steps"] else 0
            if step % LOG_EVERY == 0 or step == recipe["steps"]:
                mean = run_loss / max(run_n, 1)
                if not math.isfinite(mean):
                    raise FloatingPointError(f"{tag}: loss {mean} at step {step}")
                entry = {"step": step, "loss": round(mean, 5), "lr": sched.get_last_lr()[0],
                         "seconds": round(time.perf_counter() - t0, 1)}
                if reader != "D":
                    entry["monitor_cer"] = round(monitor_cer(model, pool, device), 4)
                log["steps"].append(entry)
                print(tag, entry, flush=True)
                run_loss, run_n = 0.0, 0
        if device == "mps":
            torch.mps.synchronize()
        log["train_seconds"] = round(time.perf_counter() - t0, 1)
        model = model.cpu()
        if device == "mps":
            torch.mps.empty_cache()
    (C.OUT / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), C.OUT / "models" / f"{tag}.pt")
    (C.OUT / "models" / f"{tag}.json").write_text(json.dumps(log, indent=1) + "\n")
    print(f"{tag}: trained {recipe['steps']} steps in {log['train_seconds']:.0f} s", flush=True)
    return log


def first_losses(reader: str, steps: int, seed: int, device: str) -> list[float]:
    """The loss of each of the first `steps` steps, read every step (for the reproducibility check).  A simplified
    loop (lr / 10, no schedule, no clipping, no weights): it checks that the GPU gives the same numbers twice, not
    that `train`'s own loop does (review M5)."""
    from sfzmaps import gpu
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pool = Pool(reader)
    model = build(reader)
    out = []
    with gpu.gpu_turn(True, device):
        model = model.to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=RECIPES[reader]["max_lr"] / 10, weight_decay=1e-4)
        queue: list = []
        for _ in range(steps):
            if not queue:
                queue = pool.batches(rng, SYNTH_SHARE)
            idx = queue.pop()
            x, frames = C.to_input([augment(pool.items[i], rng) for i in idx], quantize=True)
            texts = [pool.texts[i] for i in idx]
            lens = torch.tensor([len(t) for t in texts], dtype=torch.long)
            tgt = targets(texts)
            lp = F.log_softmax(C.forward(model, x.to(device)), -1).permute(1, 0, 2)
            nll = F.ctc_loss(lp, tgt.to(device), torch.tensor(frames).to(device), lens.to(device), blank=C.BLANK,
                             reduction="none", zero_infinity=True)
            loss = (nll / lens.clamp(min=1).to(device)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            out.append(float(loss.detach().cpu()))
        if device == "mps":
            torch.mps.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reader", choices=sorted(RECIPES) + ["repro"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    torch.set_num_threads(4)
    if a.reader == "repro":           # GPU training reproducible? B's first steps twice, each in its own turn
        runs = [first_losses("B", a.steps or 300, a.seed, a.device) for _ in range(2)]
        same = [x == y for x, y in zip(*runs)]
        first_diff = next((i for i, s in enumerate(same) if not s), None)
        res = {"steps": len(same), "identical_steps": int(sum(same)), "first_difference": first_diff,
               "max_abs_diff": max(abs(x - y) for x, y in zip(*runs)), "run1_last": runs[0][-1],
               "run2_last": runs[1][-1]}
        (C.OUT / "models").mkdir(exist_ok=True)
        (C.OUT / "models" / "repro.json").write_text(json.dumps(res, indent=1) + "\n")
        print(res, flush=True)
        return
    # seed 0 keeps the reader's own name (A.pt ...); another seed is a training-variance replicate, "<reader>s<seed>"
    train(a.reader, a.steps, a.seed, a.device, tag=a.reader if a.seed == 0 else f"{a.reader}s{a.seed}")


if __name__ == "__main__":
    main()
