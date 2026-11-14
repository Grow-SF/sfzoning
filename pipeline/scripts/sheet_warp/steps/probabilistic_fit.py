"""The probabilistic correction of a sheet's placement (runs in the PyTorch environment).

Every block-side point (probabilistic_inputs.npz, placed by today's placement) is explained by the print with a
probability:
  printed      it lies on a printed block line running its way: the density across the line, N(d; 0, s), summed over
               every such line's centre pixels (block_line_weights.npz: each weighted by how likely it is a block line),
               weight 1 - eps
  not printed  missing from the print or hidden under heavy ink: uniform over every distance a point can have, weight eps
The placement's control points are evidence too: a 2-D Gaussian around each one's matched page pixel.  The correction is
a smooth displacement field (a grid every 50 px, bilinear) that makes all of it together most probable, with a prior
keeping each grid point's correction near zero where no printed line is near, a prior on how much the field may
stretch and shear (the spread of today's own placement), and a steep penalty where it would fold the scan over.  The
spread s starts at half a street width and halves each stage; after each stage eps and sigma (the printed lines' own
spread) are re-estimated from the points; the last stage is at s = sigma.
With "extra_evidence" it fits the extra evidence points (extra_evidence_inputs.npz), each counted by its weight.
-> probabilistic_correction.npz or extra_evidence_correction.npz (Gc: the field on the grid, spacing, origin: its
top-left page pixel).  usage: probabilistic_fit.py [extra_evidence]"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from warp_common import S, SHEET

NAME = "extra_evidence" if sys.argv[1:] == ["extra_evidence"] else "probabilistic"
ITERS = 800                        # optimiser steps per stage
torch.set_num_threads(3)           # fixed: the same threads every run
T8 = np.load(S + "fit_targets.npz")["targets"]
B, h, w = T8.shape
PI = np.load(S + f"{NAME}_inputs.npz")
Pp, PB, SW = PI["P"].astype(np.float32), PI["bin"], float(PI["street_px"])
N = len(Pp)
GB_, GX_ = PI["gbase"].astype(np.float32), PI["gpx"].astype(np.float32)
NG = len(GB_)
x0, y0 = (int(v) for v in PI["origin"])
# every block-side point is one observation; an extra evidence point counts by its weight, and only the block sides
# (kind 0) measure how well the print is fitted
WPT = torch.from_numpy(PI["wt"].astype(np.float32)) if "wt" in PI.files else torch.ones(N)
NOBS = float(WPT.sum()) + NG
BASE = (PI["kind"] == 0) if "kind" in PI.files else np.ones(N, bool)
DMAX = float(T8.max())
SP = 50
gh, gw = int(np.ceil((h - 1) / SP)) + 1, int(np.ceil((w - 1) / SP)) + 1
ext = torch.tensor([(gw - 1) * SP, (gh - 1) * SP], dtype=torch.float32)
groups = [torch.as_tensor(np.flatnonzero(PB == b)) for b in range(B)]


def gp(xy):
    xy = torch.as_tensor(xy)
    return torch.stack([xy[:, 0] / ext[0] * 2 - 1, xy[:, 1] / ext[1] * 2 - 1], -1)[None, None]


def field(G, g):
    return F.grid_sample(G, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, :].T


def sample(xy, maps1):
    """each point's value on the density map of its direction (B, 1, h, w); differentiable in xy"""
    g = torch.stack([xy[:, 0] / (w - 1) * 2 - 1, xy[:, 1] / (h - 1) * 2 - 1], -1)
    vals, idx = [], []
    for b, ii in enumerate(groups):
        if len(ii):
            vals.append(F.grid_sample(maps1[b:b + 1], g[ii][None, None], mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0, 0, :])
            idx.append(ii)
    return torch.zeros(xy.shape[0]).index_copy(0, torch.cat(idx), torch.cat(vals))


def jdet(G):
    """determinant of the map p -> p + u(p) on each grid cell (forward differences averaged over the cell's sides):
    below 0 the scan folds over itself"""
    ux_x = (G[0, 0, :, 1:] - G[0, 0, :, :-1]) / SP
    uy_x = (G[0, 1, :, 1:] - G[0, 1, :, :-1]) / SP
    ux_y = (G[0, 0, 1:, :] - G[0, 0, :-1, :]) / SP
    uy_y = (G[0, 1, 1:, :] - G[0, 1, :-1, :]) / SP
    ax, ay = 0.5 * (ux_x[1:, :] + ux_x[:-1, :]), 0.5 * (uy_x[1:, :] + uy_x[:-1, :])
    bx, by = 0.5 * (ux_y[:, 1:] + ux_y[:, :-1]), 0.5 * (uy_y[:, 1:] + uy_y[:, :-1])
    return (1 + ax) * (1 + by) - bx * ay


def nearest_d(xy, m=None):
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return T8[PB if m is None else PB[m], yi, xi].astype(np.float64)


def em(d, eps, sig, rounds=50):
    """eps, sigma of the two-part model on the distances d to the nearest same-way centre (half-normal + uniform
    over 0..DMAX)"""
    for _ in range(rounds):
        pl = (1 - eps) * 2 * np.exp(-d * d / (2 * sig * sig)) / (np.sqrt(2 * np.pi) * sig)
        pu = eps / DMAX
        r = pl / np.maximum(pl + pu, 1e-300)
        eps = float(1 - r.mean())
        sig = float(np.sqrt((r * d * d).sum() / max(r.sum(), 1e-9)))
    return eps, max(sig, 0.5)


def share(xy):
    d = nearest_d(xy[BASE], BASE)
    return np.mean(d <= 1), np.mean(d <= 2), np.mean(d <= 4)


eps, sig = em(nearest_d(Pp[BASE], BASE), 0.5, SW / 4)
TAU = sig          # prior on the correction: today's fit is believed close; spread = today's measured misfit
AREA = float(h * w)


def em2(r, eps, sig, rounds=50):
    """control points: 2-D Gaussian of spread sig around their matched px, or anywhere in the crop (eps)"""
    for _ in range(rounds):
        pl = (1 - eps) * np.exp(-r * r / (2 * sig * sig)) / (2 * np.pi * sig * sig)
        pu = eps / AREA
        q = pl / np.maximum(pl + pu, 1e-300)
        eps = float(np.clip(1 - q.mean(), 1e-4, 0.5))
        sig = float(np.sqrt((q * r * r).sum() / max(2 * q.sum(), 1e-9)))
    return eps, max(sig, 0.25)


geps, gsig = em2(np.linalg.norm(GB_ - GX_, axis=1), 0.05, 2.0)
print(f"control points {NG}: today eps {geps:.3f}, sigma {gsig:.2f} px", flush=True)
print(f"{SHEET}: points {N}; street width {SW:.1f} px; grid {SP} px; start (today): eps {eps:.2f}, sigma {sig:.2f} px; "
      f"within 1/2/4 px {share(Pp)[0]:.1%} {share(Pp)[1]:.1%} {share(Pp)[2]:.1%}", flush=True)
cent = np.load(S + "block_line_weights.npz")["weights"]
TAU_J = float(PI["tau_j"])         # how much today's placement stretches and shears across the sheet
G = torch.zeros(1, 2, gh, gw, requires_grad=True)
gP = gp(Pp)
gG = gp(GB_)
GBt, GXt = torch.from_numpy(GB_), torch.from_numpy(GX_)
Pt = torch.from_numpy(Pp)
t0 = time.time()
s = SW / 2          # the two printed edges of a street (one street width apart) stay two ridges, not one
stage = 0
while True:
    last = s <= sig
    if last:
        s = sig
    maps = np.zeros((B, 1, h, w), np.float32)
    for b in range(B):
        maps[b, 0] = gaussian_filter(np.asarray(cent[b], np.float32), s)   # summed 2-D Gaussians: along a line, N(d; 0, s)
    maps = torch.from_numpy(maps)
    opt = torch.optim.Adam([G], lr=0.25)
    for it in range(ITERS):
        opt.zero_grad()
        v = sample(Pt + field(G, gP), maps)
        nll = -torch.log((1 - eps) * torch.clamp(v, min=0) + eps / DMAX)
        rg = torch.sum((GBt + field(G, gG) - GXt) ** 2, dim=1)
        nllg = -torch.log((1 - geps) * torch.exp(-rg / (2 * gsig * gsig)) / (2 * np.pi * gsig * gsig) + geps / AREA)
        data = (torch.sum(nll * WPT) + torch.sum(nllg)) / NOBS * 2 * s * s
        prior = torch.sum(G ** 2) / (2 * TAU * TAU) / NOBS * 2 * s * s
        fold = torch.sum(torch.relu(-jdet(G)) ** 2) * 1e6
        ux_x = (G[0, 0, :, 1:] - G[0, 0, :, :-1]) / SP
        uy_x = (G[0, 1, :, 1:] - G[0, 1, :, :-1]) / SP
        ux_y = (G[0, 0, 1:, :] - G[0, 0, :-1, :]) / SP
        uy_y = (G[0, 1, 1:, :] - G[0, 1, :-1, :]) / SP
        shape = (torch.sum(ux_x ** 2) + torch.sum(uy_x ** 2) + torch.sum(ux_y ** 2) + torch.sum(uy_y ** 2)) / (2 * TAU_J ** 2) / NOBS * 2 * s * s
        (data + prior + shape + fold).backward()
        opt.step()
    del maps
    with torch.no_grad():
        u = field(G, gP).numpy()
        umax = float(torch.linalg.norm(G[0], dim=0).max())
        dmin = float(jdet(G).min())
        ug = field(G, gG).numpy()
    eps, sig = em(nearest_d((Pp + u)[BASE], BASE), eps, sig)
    a1, a2, a4 = share(Pp + u)
    print(f"  stage {stage}: s {s:.2f} px; within 1/2/4 px {a1:.1%} {a2:.1%} {a4:.1%}; eps {eps:.2f}, sigma {sig:.2f} px; control points off p99 {np.percentile(np.linalg.norm(GB_ + ug - GX_, axis=1), 99):.1f} px; "
          f"moved p50 {np.median(np.linalg.norm(u, axis=1)):.1f} p90 {np.percentile(np.linalg.norm(u, axis=1), 90):.1f} px, biggest correction {umax:.1f} px; smallest cell determinant {dmin:.2f}; {time.time() - t0:.0f}s", flush=True)
    stage += 1
    if last:
        break
    s = s / 2
np.savez(S + f"{NAME}_correction.npz", Gc=G.detach().numpy()[0], spacing=SP, origin=np.array([x0, y0]))
print("saved", S + f"{NAME}_correction.npz")
