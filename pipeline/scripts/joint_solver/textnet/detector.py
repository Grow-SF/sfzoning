"""Our own text detector: a small U-Net over the merged page, every pixel scored at the scan's own resolution.
Outputs per pixel: text (logit), kind among text (4 logits: zoning code, block number, street name, other), reading
direction (cos, sin of the angle).  Fitted with one quarter held out; trained only on terra's saved boxes.

  train(q)    fit without quarter q (pixels of q carry no loss), save textnet/detector_{q}.pt
  predict(q)  run the model fitted without q over the whole page, save textnet/text_prob_{q}.npy (uint8, 0..255)
usage: detector.py train|predict QUARTER [ITERATIONS]"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

S = Path("/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad")
OUT = S / "textnet"
QUARTERS = ("nw", "ne", "sw", "se")
CROP, BATCH = 320, 12
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, w=24):
        super().__init__()
        ch = [w, 2 * w, 4 * w, 8 * w, 8 * w]
        self.down = nn.ModuleList([block(1, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(ch[0], 7, 1)

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for u in self.up:
            s = skips.pop()
            x = u(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="nearest"), s], 1))
        return self.head(x)


def load():
    D = np.load(OUT / "data.npz")
    return {k: D[k] for k in D.files}


def batch(D, q, rng, augment=True):
    """CROP x CROP windows centred on a training-quarter ink pixel; half of them on a text pixel.  Augmented: scaled by
    0.6-1.6 (so small street lettering also teaches big code lettering) and turned by a multiple of 90 degrees (the
    reading angle turns with it)."""
    import cv2
    ink, text, quarter = D["ink"], D["text"], D["quarter"]
    H, W = ink.shape
    if "_ink_pts" not in D:
        ys, xs = np.nonzero(ink[::4, ::4])
        D["_ink_pts"] = np.c_[ys * 4, xs * 4]
        ys, xs = np.nonzero(text[::4, ::4])
        D["_text_pts"] = np.c_[ys * 4, xs * 4]
    xb, tb, kb, ab, wb = [], [], [], [], []
    while len(xb) < BATCH:
        pts = D["_text_pts"] if rng.random() < 0.5 else D["_ink_pts"]
        y, x = pts[rng.integers(len(pts))]
        if quarter[y, x] == q:
            continue
        s = float(np.exp(rng.uniform(np.log(0.6), np.log(1.6)))) if augment else 1.0
        src = int(round(CROP / s))
        y0 = int(np.clip(y - src // 2, 0, H - src))
        x0 = int(np.clip(x - src // 2, 0, W - src))
        sl = (slice(y0, y0 + src), slice(x0, x0 + src))
        rot = int(rng.integers(4)) if augment else 0
        ok = ~D["unknown"][sl] & (quarter[sl] != q)
        ang = D["angle"][sl].astype(np.int32)
        ang = np.where(text[sl] > 0, (ang + 90 * rot + 180) % 360 - 180, -999)
        parts = [ink[sl], text[sl], D["kind"][sl], ang, ok]
        if src != CROP:
            parts = [cv2.resize(p.astype(np.int16 if p.dtype != np.uint8 else np.uint8), (CROP, CROP),
                                interpolation=cv2.INTER_NEAREST) for p in parts]
        parts = [np.rot90(p, rot) for p in parts]
        xb.append(parts[0])
        tb.append(parts[1])
        kb.append(parts[2])
        ab.append(parts[3])
        wb.append(parts[4])
    f = lambda a, dt: torch.from_numpy(np.stack(a).astype(dt)).to(DEV)
    return f(xb, np.float32)[:, None], f(tb, np.float32), f(kb, np.int64), f(ab, np.float32), f(wb, np.float32)


def loss_fn(out, t, k, a, w):
    lt = (F.binary_cross_entropy_with_logits(out[:, 0], t, reduction="none") * w).sum() / w.sum().clamp(min=1)
    m = (t > 0) & (w > 0)
    if m.any():
        lk = F.cross_entropy(out[:, 1:5].permute(0, 2, 3, 1)[m], (k[m] - 1).clamp(0, 3))
        cs = out[:, 5:7].permute(0, 2, 3, 1)[m]
        rad = torch.deg2rad(a[m])
        la = (1 - (cs[:, 0] * torch.cos(rad) + cs[:, 1] * torch.sin(rad)) / cs.norm(dim=1).clamp(min=1e-6)).mean()
    else:
        lk = la = out.sum() * 0
    return lt + 0.5 * lk + 0.5 * la, (lt.item(), lk.item(), la.item())


def train(qname, iters=6000):
    q = QUARTERS.index(qname)
    D = load()
    rng = np.random.default_rng(q)
    torch.manual_seed(q)
    model = UNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=iters)
    t0, run = time.time(), np.zeros(3)
    for it in range(iters):
        x, t, k, a, w = batch(D, q, rng)
        out = model(x)
        loss, parts = loss_fn(out, t, k, a, w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * np.array(parts)
        if (it + 1) % 250 == 0:
            print(f"{qname} it {it + 1}: text {run[0]:.4f} kind {run[1]:.4f} angle {run[2]:.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    torch.save(model.state_dict(), OUT / f"detector_{qname}.pt")


@torch.no_grad()
def predict(qname, tile=1024, pad=64):
    """the model fitted without q over the whole page: text chance (uint8 0..255), kind (argmax 1..4), angle (deg)"""
    D = load()
    model = UNet().to(DEV)
    model.load_state_dict(torch.load(OUT / f"detector_{qname}.pt", map_location=DEV))
    model.eval()
    ink = D["ink"]
    H, W = ink.shape
    prob = np.zeros((H, W), np.uint8)
    kind = np.zeros((H, W), np.uint8)
    ang = np.zeros((H, W), np.int16)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y0, x0 = max(y - pad, 0), max(x - pad, 0)
            y1, x1 = min(y + tile + pad, H), min(x + tile + pad, W)
            win = ink[y0:y1, x0:x1]
            if not win.any():
                continue
            ph, pw = (-win.shape[0]) % 16, (-win.shape[1]) % 16
            xin = torch.from_numpy(np.pad(win, ((0, ph), (0, pw))).astype(np.float32)).to(DEV)[None, None]
            o = model(xin)[0, :, :win.shape[0], :win.shape[1]].float().cpu().numpy()
            cy, cx = y - y0, x - x0
            h, w = min(tile, H - y), min(tile, W - x)
            sl = (slice(cy, cy + h), slice(cx, cx + w))
            prob[y:y + h, x:x + w] = np.round(255 / (1 + np.exp(-o[0][sl]))).astype(np.uint8)
            kind[y:y + h, x:x + w] = 1 + o[1:5][(slice(None),) + sl].argmax(0)
            ang[y:y + h, x:x + w] = np.round(np.rad2deg(np.arctan2(o[6][sl], o[5][sl]))).astype(np.int16)
    np.savez_compressed(OUT / f"text_pred_{qname}.npz", prob=prob, kind=kind, angle=ang)
    print(f"saved text_pred_{qname}.npz")


if __name__ == "__main__":
    cmd, qname = sys.argv[1], sys.argv[2]
    if cmd == "train":
        train(qname, int(sys.argv[3]) if len(sys.argv) > 3 else 6000)
    else:
        predict(qname)
