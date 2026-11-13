"""Direction-aware direct fit (CPU).  As regfit.py, but each record point is scored on the distance map of
its own direction bin (reg2_NN.npz): it is drawn only to printed block-line centres running within ~20 deg of
its own block line.  The field lives in crop px (origin saved) and is written in page px for regapply.
-> regfit_NN_<tag>.npz.  usage: regfit2.py NN LAMBDA TAG"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

NN, LAM, TAG = sys.argv[1], float(sys.argv[2]), sys.argv[3]
S = "/private/tmp/claude-501/-Users-sbuss-workspace-sfzoning/eb18d2fc-e22a-48a6-afbc-57c31cd3e134/scratchpad/"
Z = np.load(S + f"reg2_{NN}.npz")
T8, P, PB = Z["targets"], Z["points"], Z["pbin"]
x0, y0 = (int(v) for v in Z["origin"])
B, h, w = T8.shape
torch.set_num_threads(8)
Pt = torch.tensor(P)
groups = [np.flatnonzero(PB == b) for b in range(B)]
print(f"p{NN}: CPU; points {len(P)}; bins {B}; crop {w}x{h}; lambda {LAM}", flush=True)


def within(xy, px):
    d = T8[PB, np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1), np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)]
    return float(np.mean(d <= px))


print(f"  start: within 1 px {within(P, 1):.1%}, within 2 px {within(P, 2):.1%}", flush=True)


class Sample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, maps):          # maps: B x 3 x h x w (distance, d/dx, d/dy)
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


stages = [(800, 8.0, 16.0), (400, 4.0, 8.0), (200, 2.0, 4.0), (100, 1.0, 3.0)]
G = None
t0 = time.time()
for sp, blur, c in stages:
    gh, gw = int(np.ceil((h - 1) / sp)) + 1, int(np.ceil((w - 1) / sp)) + 1
    maps = np.zeros((B, 3, h, w), np.float32)
    for b in range(B):
        tb = gaussian_filter(T8[b].astype(np.float32), blur)
        gy_, gx_ = np.gradient(tb)
        maps[b, 0], maps[b, 1], maps[b, 2] = tb, gx_, gy_
    maps = torch.from_numpy(maps)
    G0 = torch.zeros(1, 2, gh, gw) if G is None else F.interpolate(G.detach(), size=(gh, gw), mode="bilinear", align_corners=True)
    ext = torch.tensor([(gw - 1) * sp, (gh - 1) * sp], dtype=torch.float32)
    G = G0.clone().requires_grad_(True)
    opt = torch.optim.Adam([G], lr=0.25)
    gpos = torch.stack([Pt[:, 0] / ext[0] * 2 - 1, Pt[:, 1] / ext[1] * 2 - 1], -1)[None, None]
    for it in range(400):
        opt.zero_grad()
        u = F.grid_sample(G, gpos, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, :].T
        d = Sample.apply(Pt + u, maps)
        data = torch.mean(c * c * d * d / (d * d + c * c)) / (c * c)
        smooth = torch.mean((G[:, :, 1:, :] - G[:, :, :-1, :]) ** 2) + torch.mean((G[:, :, :, 1:] - G[:, :, :, :-1]) ** 2)
        (data + LAM * smooth).backward()
        opt.step()
    with torch.no_grad():
        u = F.grid_sample(G, gpos, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, :].T
        q = (Pt + u).numpy()
        um = torch.linalg.norm(u, dim=1).numpy()
    del maps
    print(f"  grid {sp:4d} px, blur {blur:3.0f}, cap {c:4.0f}: within 1 px {within(q, 1):.1%}, within 2 px {within(q, 2):.1%}; "
          f"point moves px p50 {np.median(um):.2f} p90 {np.percentile(um, 90):.2f} max {um.max():.1f}; {time.time() - t0:.0f}s", flush=True)
# write the field on a page-px grid (spacing sp, origin 0,0) for regapply: sample the crop field there
Gc = G.detach()
Hp, Wp = (int(v) for v in Z["shape"])
sp = stages[-1][0]
yy, xx = np.meshgrid(np.arange(0, Hp + sp, sp, dtype=np.float32), np.arange(0, Wp + sp, sp, dtype=np.float32), indexing="ij")
cx, cy = (xx - x0) / ((Gc.shape[3] - 1) * sp) * 2 - 1, (yy - y0) / ((Gc.shape[2] - 1) * sp) * 2 - 1
inside = (cx >= -1) & (cx <= 1) & (cy >= -1) & (cy <= 1)
gpage = F.grid_sample(Gc, torch.from_numpy(np.stack([cx, cy], -1))[None], mode="bilinear", padding_mode="border", align_corners=True)[0].numpy()
gpage[:, ~inside] = 0.0
np.savez(S + f"regfit_{NN}_{TAG}.npz", G=gpage, spacing=sp, lam=LAM)
print("saved", S + f"regfit_{NN}_{TAG}.npz")
