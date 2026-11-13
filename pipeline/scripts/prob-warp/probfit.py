"""Probabilistic correction of today's fit, sheet NN.  No search windows, hit tests, held/free split or caps.
Every 2000 block-line point (probin_NN.npz, placed by today's fit) is explained by the print with a probability:
  printed      it lies on a printed block line running its way (reg2_NN.npz centres within 20 deg of its direction):
               density along the line's normal N(d; 0, s), summed over every such centre (the centres blurred by a
               Gaussian of width s), weight 1 - eps
  not printed  not printed here or hidden under heavy ink: uniform over every distance a point can have (0 to the
               distance maps' largest value, DMAX), weight eps
A smooth displacement field u (grid every 50 px, bilinear, as confit6.py) is fitted to make all points together as
probable as possible: minimise mean(-log p) * 2 s^2 (so a point near its line costs about d^2 at any s, the scale the
existing smoothness was set on) + LAMBDA * smooth + RIP * rip, with the existing settings (LAMBDA 1, RATIO 2,
RIP 1000, as correction p1).  s starts at half a street width (wider, a street's two printed edges blur into one
ridge down its middle and pull block lines into it) and halves each stage; after each stage eps and sigma
(the printed lines' own spread) are re-estimated from the points (EM on their distance to the nearest same-way
centre); the last stage is at s = sigma.  The warp's control points (matched corners) are evidence too: a 2-D
Gaussian around the matched page px with today's spread and outlier share (EM on today's residuals, kept fixed:
re-estimating after the fit only shrinks it).  Each block side counts once (its points share one weight), or with PERPOINT=1 every point counts once.
Prior on the correction: each grid point's correction ~ N(0, TAU), TAU = today's measured misfit (sigma at the start),
so with no printed line nearby it stays near zero.  No fold: a steep penalty where a grid cell's determinant of
p -> p + u(p) goes below 0 (the scan turned over); the smallest determinant is reported.
-> confit_NN_<TAG>.npz (same format as confit6.py: Gc on the crop grid, G on the page grid).
usage: probfit.py NN TAG [ITERS]"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

NN, TAG = sys.argv[1], sys.argv[2]
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 800
LAM, RATIO, RIP = 1.0, 2.0, 1000.0
S = os.environ.get("PROBDIR", "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/")
LOWMEM = bool(os.environ.get("LOWMEM"))
torch.set_num_threads(int(__import__("os").environ.get("THREADS", "5")))
R2 = np.load(S + f"reg2_{NN}.npz")
T8 = R2["targets"]
B, h, w = T8.shape
PI = np.load(S + f"probin_{NN}.npz")
Pp, PB, SW = PI["P"].astype(np.float32), PI["bin"], float(PI["street_px"])
SID = PI["sid"]
_u, _inv, _cnt = np.unique(SID, return_inverse=True, return_counts=True)
WPT = torch.from_numpy((1.0 / _cnt[_inv]).astype(np.float32))      # each block side counts once: its points share 1
NSIDE = len(_u)
if __import__("os").environ.get("PERPOINT"):      # every block-line point is its own observation
    WPT = torch.ones(len(SID))
NOBS = float(WPT.sum()) + 0.0
GB_, GX_ = PI["gbase"].astype(np.float32), PI["gpx"].astype(np.float32)
NG = len(GB_)
NOBS += NG
x0, y0 = (int(v) for v in PI["origin"])
Hp_, Wp_ = (int(v) for v in PI["shape"])
N = len(Pp)
# added evidence (sides_plus.py): each point's weight, and which points are block sides (kind 0)
if "wt" in PI.files:
    WPT = WPT * torch.from_numpy(PI["wt"].astype(np.float32))
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


class Sample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, maps):
        out = torch.zeros(3, xy.shape[0])
        g = torch.stack([xy[:, 0] / (w - 1) * 2 - 1, xy[:, 1] / (h - 1) * 2 - 1], -1)
        for b, ii in enumerate(groups):
            if len(ii):
                out[:, ii] = F.grid_sample(maps[b:b + 1], g[ii][None, None], mode="bilinear", padding_mode="zeros", align_corners=True)[0, :, 0, :]
        ctx.save_for_backward(out[1:])
        return out[0]

    @staticmethod
    def backward(ctx, gr):
        (grad,) = ctx.saved_tensors
        return (gr[None] * grad).T, None


def sample_ad(xy, maps1):
    """value maps only (B, 1, h, w); gradient w.r.t. xy through grid_sample"""
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
print(f"p{NN}: points {N}; street width {SW:.1f} px; grid {SP} px; start (today): eps {eps:.2f}, sigma {sig:.2f} px; "
      f"within 1/2/4 px {share(Pp)[0]:.1%} {share(Pp)[1]:.1%} {share(Pp)[2]:.1%}", flush=True)
cent = (T8 == 0)
if os.environ.get("WMAP"):        # block-line centres weighted by their probability of being a block line (regprep3)
    cent = np.load(S + f"regw_{NN}.npz")["weights"]
TAU_J = float(PI["tau_j"]) if "tau_j" in PI.files else None
SHAPE = bool(os.environ.get("SHAPE"))
print(f"centre weights: {'regw (width probability, text erased)' if os.environ.get('WMAP') else 'reg2 (in or out)'}; "
      f"shape prior: {'on, stretch spread %.4f' % TAU_J if SHAPE else 'off (smoothness LAMBDA %.1f, rip)' % LAM}", flush=True)
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
    maps = np.zeros((B, 1 if LOWMEM else 3, h, w), np.float32)
    for b in range(B):
        lb = gaussian_filter(np.asarray(cent[b], np.float32), s)       # summed 2-D Gaussians: along a line, N(d; 0, s)
        maps[b, 0] = lb
        if not LOWMEM:
            gy_, gx_ = np.gradient(lb)
            maps[b, 1], maps[b, 2] = gx_, gy_
    maps = torch.from_numpy(maps)
    opt = torch.optim.Adam([G], lr=0.25)
    for it in range(ITERS):
        opt.zero_grad()
        v = sample_ad(Pt + field(G, gP), maps) if LOWMEM else Sample.apply(Pt + field(G, gP), maps)
        nll = -torch.log((1 - eps) * torch.clamp(v, min=0) + eps / DMAX)
        rg = torch.sum((GBt + field(G, gG) - GXt) ** 2, dim=1)
        nllg = -torch.log((1 - geps) * torch.exp(-rg / (2 * gsig * gsig)) / (2 * np.pi * gsig * gsig) + geps / AREA)
        data = (torch.sum(nll * WPT) + torch.sum(nllg)) / NOBS * 2 * s * s
        smooth = (torch.sum((G[:, :, 1:, :] - G[:, :, :-1, :]) ** 2) + torch.sum((G[:, :, :, 1:] - G[:, :, :, :-1]) ** 2)) / N
        mag = torch.sqrt(torch.sum(G[0] ** 2, dim=0) + 1e-6)
        nb = 0.25 * (mag[:-2, 1:-1] + mag[2:, 1:-1] + mag[1:-1, :-2] + mag[1:-1, 2:])
        rip = torch.sum(torch.relu(mag[1:-1, 1:-1] - RATIO * nb) ** 2) / N
        prior = torch.sum(G ** 2) / (2 * TAU * TAU) / NOBS * 2 * s * s
        fold = torch.sum(torch.relu(-jdet(G)) ** 2) * 1e6
        if SHAPE:
            ux_x = (G[0, 0, :, 1:] - G[0, 0, :, :-1]) / SP
            uy_x = (G[0, 1, :, 1:] - G[0, 1, :, :-1]) / SP
            ux_y = (G[0, 0, 1:, :] - G[0, 0, :-1, :]) / SP
            uy_y = (G[0, 1, 1:, :] - G[0, 1, :-1, :]) / SP
            shape = (torch.sum(ux_x ** 2) + torch.sum(uy_x ** 2) + torch.sum(ux_y ** 2) + torch.sum(uy_y ** 2)) / (2 * TAU_J ** 2) / NOBS * 2 * s * s
            (data + prior + shape + fold).backward()
        else:
            (data + prior + LAM * smooth + RIP * rip + fold).backward()
        opt.step()
    del maps
    with torch.no_grad():
        u = field(G, gP).numpy()
        umax = float(torch.linalg.norm(G[0], dim=0).max())
        dmin = float(jdet(G).min())
        ug = field(G, gG).numpy()
    eps, sig = em(nearest_d((Pp + u)[BASE], BASE), eps, sig)
    a1, a2, a4 = share(Pp + u)
    print(f"  stage {stage}: s {s:.2f} px; within 1/2/4 px {a1:.1%} {a2:.1%} {a4:.1%}; eps {eps:.2f}, sigma {sig:.2f} px; control points eps {geps:.3f} sigma {gsig:.2f} px, off p99 {np.percentile(np.linalg.norm(GB_ + ug - GX_, axis=1), 99):.1f} px; "
          f"moved p50 {np.median(np.linalg.norm(u, axis=1)):.1f} p90 {np.percentile(np.linalg.norm(u, axis=1), 90):.1f} px, biggest correction {umax:.1f} px; smallest cell determinant {dmin:.2f}; {time.time() - t0:.0f}s", flush=True)
    stage += 1
    if last:
        break
    s = s / 2
yy, xx = np.meshgrid(np.arange(0, Hp_ + SP, SP, dtype=np.float32), np.arange(0, Wp_ + SP, SP, dtype=np.float32), indexing="ij")
cx, cy = (xx - x0) / ext[0].item() * 2 - 1, (yy - y0) / ext[1].item() * 2 - 1
inside = (cx >= -1) & (cx <= 1) & (cy >= -1) & (cy <= 1)
gpage = F.grid_sample(G.detach(), torch.from_numpy(np.stack([cx, cy], -1))[None], mode="bilinear", padding_mode="border", align_corners=True)[0].numpy()
gpage[:, ~inside] = 0.0
np.savez(S + f"confit_{NN}_{TAG}.npz", G=gpage, spacing=SP, lam=LAM, Gc=G.detach().numpy()[0], origin=np.array([x0, y0]),
         eps=eps, sigma=sig)
print("saved", S + f"confit_{NN}_{TAG}.npz")
