"""Constrained correction of today's fit (CPU).  A smooth displacement field u (grid every 50 px over the crop,
bilinear) added to today's placement.  Score:
  free lines   each block-line point not found today: distance to the nearest printed block-line centre running
               its way (reg2 maps), capped (c^2 d^2/(d^2+c^2))
  free corners each thrown-out corner: distance from the moved record corner to its printed corner, capped the same
  held sides   HOLD x mean (u . n)^2 at every held block-line point (n = its line's normal): held across the line,
               free to slide along it (a held side gets shorter or longer where a neighbouring side moves)
  held corners HOLD_CP x mean |u|^2 at today's held control points: held in both directions
  smooth       LAMBDA x sum over links of squared difference between neighbouring nodes / number of free points
Annealed on the target: blur 8 -> 4 -> 2 -> 1 px, cap 16 -> 8 -> 4 -> 3 px (16 covers the largest measured offset).
-> confit_NN_<tag>.npz (field on a page-px grid).  usage: confit5.py NN LAMBDA HOLD HOLD_CP TAG RATIO RIP"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

LAM, HOLD, HOLD_CP, TAG = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
RATIO, RIP = float(sys.argv[5]), float(sys.argv[6])
sys.path.insert(0, '/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/allyears')
from ay import *  # noqa: F403,E402
import json  # noqa: E402
FTPX = settings()['ftpx']
import os
Z2 = np.load(S + os.environ.get("REG", f"reg2_{NN}.npz"))
T8 = Z2["targets"]
C = np.load(S + f"con_{NN}.npz")
Hp, HN, HC = C["held_line"], C["held_n"], C["held_cp"]
PR = np.load(S + f"prop_{NN}.npz")
Fp, FN, FO = PR["free"], PR["free_n"], PR["free_off"]
FB = np.zeros(len(Fp), int)
CR, CS = C["corner_rec"], C["corner_scan"]
x0, y0 = (int(v) for v in C["origin"])
B, h, w = T8.shape
torch.set_num_threads(int(os.environ.get("CONFIT_THREADS", "8")))   # speed: 1 when many corrections run at once
SP = 50
gh, gw = int(np.ceil((h - 1) / SP)) + 1, int(np.ceil((w - 1) / SP)) + 1
ext = torch.tensor([(gw - 1) * SP, (gh - 1) * SP], dtype=torch.float32)
groups = [np.flatnonzero(FB == b) for b in range(B)]


def gp(xy):
    xy = torch.as_tensor(xy)
    return torch.stack([xy[:, 0] / ext[0] * 2 - 1, xy[:, 1] / ext[1] * 2 - 1], -1)[None, None]


def field(G, g):
    return F.grid_sample(G, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, :].T


class Sample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, maps):
        out = torch.zeros(3, xy.shape[0])
        g = torch.stack([xy[:, 0] / (w - 1) * 2 - 1, xy[:, 1] / (h - 1) * 2 - 1], -1)
        for b, idx in enumerate(groups):
            if len(idx):
                ii = torch.as_tensor(idx)
                out[:, ii] = F.grid_sample(maps[b:b + 1], g[ii][None, None], mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, :]
        ctx.save_for_backward(out[1:])
        return out[0]

    @staticmethod
    def backward(ctx, gr):
        (grad,) = ctx.saved_tensors
        return (gr[None] * grad).T, None


def own(xy):
    return T8[FB, np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1), np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)]


Ft, Ht, CRt, CSt, HNt, FNt, FOt = (torch.tensor(a) for a in (Fp, Hp, CR, CS, HN, FN, FO))
gF, gH, gC, gK = gp(Fp), gp(Hp), gp(CR), gp(HC)
print(f"p{NN}: free line points {len(Fp)}, free corners {len(CR)}, held side points {len(Hp)}, held control points {len(HC)}; grid {SP} px; lambda {LAM}, hold {HOLD}, hold_cp {HOLD_CP}", flush=True)
print(f"  start: free points within 1 px of a same-direction line {np.mean(own(Fp) <= 1):.1%}, within 2 px {np.mean(own(Fp) <= 2):.1%}; "
      f"corners off by px p50 {np.median(np.linalg.norm(CR - CS, axis=1)):.1f}", flush=True)
G = torch.zeros(1, 2, gh, gw, requires_grad=True)
t0 = time.time()
for blur, c in ((1.0, 3.0),):
    maps = None
    opt = torch.optim.Adam([G], lr=0.25)
    for it in range(1500):
        opt.zero_grad()
        data = torch.mean((torch.sum(field(G, gF) * FNt, dim=1) - FOt) ** 2)
        dc = torch.linalg.norm(CRt + field(G, gC) - CSt, dim=1) if len(CR) else torch.zeros(1)
        cor = torch.mean(c * c * dc * dc / (dc * dc + c * c)) / (c * c)
        held = torch.mean(torch.sum(field(G, gH) * HNt, dim=1) ** 2)
        heldcp = torch.mean(torch.sum(field(G, gK) ** 2, dim=1))
        # per link, on the scale of one free block-line point (data is a mean over the free points)
        smooth = (torch.sum((G[:, :, 1:, :] - G[:, :, :-1, :]) ** 2) + torch.sum((G[:, :, :, 1:] - G[:, :, :, :-1]) ** 2)) / len(Fp)
        mag = torch.sqrt(torch.sum(G[0] ** 2, dim=0) + 1e-6)
        nb = 0.25 * (mag[:-2, 1:-1] + mag[2:, 1:-1] + mag[1:-1, :-2] + mag[1:-1, 2:])
        excess = torch.relu(mag[1:-1, 1:-1] - RATIO * nb)
        rip = torch.sum(excess ** 2) / len(Fp)
        (data + cor + HOLD * held + HOLD_CP * heldcp + LAM * smooth + RIP * rip).backward()
        opt.step()
    del maps
    with torch.no_grad():
        uf, uh, uc, uk = field(G, gF).numpy(), field(G, gH).numpy(), field(G, gC).numpy(), field(G, gK).numpy()
    hm = np.abs(np.sum(uh * HN, axis=1))
    km = np.linalg.norm(uk, axis=1) * FTPX
    with torch.no_grad():
        jump = max(float(torch.linalg.norm(G[0, :, 1:, :] - G[0, :, :-1, :], dim=0).max()), float(torch.linalg.norm(G[0, :, :, 1:] - G[0, :, :, :-1], dim=0).max()))
        umax = float(torch.linalg.norm(G[0], dim=0).max())
    with torch.no_grad():
        mag = torch.linalg.norm(G[0], dim=0)
        nb = 0.25 * (mag[:-2, 1:-1] + mag[2:, 1:-1] + mag[1:-1, :-2] + mag[1:-1, 2:])
        big = mag[1:-1, 1:-1] > 1.0
        ratio = float((mag[1:-1, 1:-1][big] / (nb[big] + 1e-6)).max()) if big.any() else 0.0
    print(f"  biggest jump between neighbouring grid points {jump:.1f} px, biggest correction {umax:.1f} px; largest (correction / neighbours' average) where the correction is over 1 px: {ratio:.2f}")
    print(f"  blur {blur:3.0f}, cap {c:4.0f}: free within 1 px {np.mean(own(Fp + uf) <= 1):.1%}, within 2 px {np.mean(own(Fp + uf) <= 2):.1%}; "
          f"corners off px p50 {np.median(np.linalg.norm(CR + uc - CS, axis=1)):.1f}; held sides moved ACROSS px p50 {np.median(hm):.2f} p99 {np.percentile(hm, 99):.2f} max {hm.max():.2f}; held control points moved ft max {km.max():.2f}; "
          f"free moved px p90 {np.percentile(np.linalg.norm(uf, axis=1), 90):.1f}; {time.time() - t0:.0f}s", flush=True)
Hp_, Wp_ = (int(v) for v in C["shape"])
yy, xx = np.meshgrid(np.arange(0, Hp_ + SP, SP, dtype=np.float32), np.arange(0, Wp_ + SP, SP, dtype=np.float32), indexing="ij")
cx, cy = (xx - x0) / ext[0].item() * 2 - 1, (yy - y0) / ext[1].item() * 2 - 1
inside = (cx >= -1) & (cx <= 1) & (cy >= -1) & (cy <= 1)
gpage = F.grid_sample(G.detach(), torch.from_numpy(np.stack([cx, cy], -1))[None], mode="bilinear", padding_mode="border", align_corners=True)[0].numpy()
gpage[:, ~inside] = 0.0
np.savez(S + f"confit_{NN}_{TAG}.npz", G=gpage, spacing=SP, lam=LAM, hold=HOLD, hold_cp=HOLD_CP, Gc=G.detach().numpy()[0], origin=np.array([x0, y0]))
print("saved", S + f"confit_{NN}_{TAG}.npz")
