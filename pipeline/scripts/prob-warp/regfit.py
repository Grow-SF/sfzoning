"""Direct fit by annealed gradient descent (PyTorch, Apple GPU when available).
Moves the 2000 block-line points (reg_NN.npz, drawn onto the page by today's fit) by a smooth displacement
field so they land on printed block-line centres.  The field is a grid of 2-D displacements (page px),
bilinear between nodes.  Score per point: c^2 d^2 / (d^2 + c^2) (d = distance to the nearest block-line
centre on the blurred target): it stops growing past c, so a line with nothing printed near it can't pull
hard.  Plus LAMBDA x mean squared difference between neighbouring nodes (a uniform shift costs nothing).
Coarse to fine:  grid 800 -> 400 -> 200 -> 100 px, target blur 8 -> 4 -> 2 -> 1 px, cap c 16 -> 8 -> 4 -> 3 px
(16 covers the largest print-to-record offset measured, 15 px; 3 is just past the 2 px alignment).
-> regfit_NN_<tag>.npz: final grid, spacing.  usage: regfit.py NN LAMBDA TAG"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

NN, LAM, TAG = sys.argv[1], float(sys.argv[2]), sys.argv[3]
S = "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/"
Z = np.load(S + f"reg_{NN}.npz")
target, P = Z["target"], Z["points"]
H, W = (int(v) for v in Z["shape"])
import os
dev = torch.device(os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"p{NN}: device {dev}; points {len(P)}; lambda {LAM}", flush=True)
Pt = torch.tensor(P, device=dev)


def norm(xy):
    """page px -> grid_sample's [-1, 1] (align_corners=True over the page)."""
    return torch.stack([xy[:, 0] / (W - 1) * 2 - 1, xy[:, 1] / (H - 1) * 2 - 1], -1)


def at(img, xy):
    return F.grid_sample(img, norm(xy)[None, None], mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, :]


class Sample(torch.autograd.Function):
    """Distance at points, with its slope taken from precomputed slope images (no gradient through the
    image itself: that made one step too long for the GPU)."""
    @staticmethod
    def forward(ctx, xy, img3):
        v = at(img3, xy)                       # 3 x N: distance, d/dx, d/dy
        ctx.save_for_backward(v[1:])
        return v[0]

    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        return (g[None] * grad).T, None


def within(xy, px):
    d = target[np.clip(np.round(xy[:, 1]).astype(int), 0, H - 1), np.clip(np.round(xy[:, 0]).astype(int), 0, W - 1)]
    return float(np.mean(d <= px))


print(f"  start: within 1 px {within(P, 1):.1%}, within 2 px {within(P, 2):.1%}", flush=True)
stages = [(800, 8.0, 16.0), (400, 4.0, 8.0), (200, 2.0, 4.0), (100, 1.0, 3.0)]
stages = [s for s in stages if s[0] >= int(os.environ.get("FINEST", "100"))]
G = None
t0 = time.time()
for sp, blur, c in stages:
    gh, gw = int(np.ceil((H - 1) / sp)) + 1, int(np.ceil((W - 1) / sp)) + 1
    tbn = gaussian_filter(target, blur).astype(np.float32)
    gy_, gx_ = np.gradient(tbn)
    tb = torch.tensor(np.stack([tbn, gx_, gy_]).astype(np.float32), device=dev)[None]
    if G is None:
        G0 = torch.zeros(1, 2, gh, gw, device=dev)
    else:
        G0 = F.interpolate(G.detach(), size=(gh, gw), mode="bilinear", align_corners=True)
    # the grid covers [0, (gw-1)*sp] x [0, (gh-1)*sp]; points map to it in px / that extent
    ext = torch.tensor([(gw - 1) * sp, (gh - 1) * sp], device=dev, dtype=torch.float32)
    G = G0.clone().requires_grad_(True)
    opt = torch.optim.Adam([G], lr=0.25)
    gpos = torch.stack([Pt[:, 0] / ext[0] * 2 - 1, Pt[:, 1] / ext[1] * 2 - 1], -1)[None, None]
    for it in range(400):
        opt.zero_grad()
        u = F.grid_sample(G, gpos, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, :].T
        d = Sample.apply(Pt + u, tb)
        data = torch.mean(c * c * d * d / (d * d + c * c)) / (c * c)
        smooth = (torch.mean((G[:, :, 1:, :] - G[:, :, :-1, :]) ** 2) + torch.mean((G[:, :, :, 1:] - G[:, :, :, :-1]) ** 2))
        loss = data + LAM * smooth
        loss.backward()
        opt.step()
        if dev.type == "mps":
            torch.mps.synchronize()          # keep each GPU command short (macOS stops long ones)
    with torch.no_grad():
        u = F.grid_sample(G, gpos, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, :].T
        q = (Pt + u).cpu().numpy()
        um = torch.linalg.norm(u, dim=1).cpu().numpy()
    print(f"  grid {sp:4d} px, blur {blur:3.0f}, cap {c:4.0f}: within 1 px {within(q, 1):.1%}, within 2 px {within(q, 2):.1%}; "
          f"point moves px p50 {np.median(um):.2f} p90 {np.percentile(um, 90):.2f} max {um.max():.1f}; {time.time() - t0:.0f}s", flush=True)
np.savez(S + f"regfit_{NN}_{TAG}.npz", G=G.detach().cpu().numpy()[0], spacing=stages[-1][0], lam=LAM)
print("saved", S + f"regfit_{NN}_{TAG}.npz")
