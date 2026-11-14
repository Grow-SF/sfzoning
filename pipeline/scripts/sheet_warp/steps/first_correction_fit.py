"""The first correction of a sheet's placement (runs in the PyTorch environment).  A smooth displacement field (a grid
every 50 px over the crop, bilinear between grid points) added to today's placement, the one that best does all of:
  free sides     move each free side point along its normal by its propagated shift (propagated_shifts.npz)
  free corners   bring each thrown-out corner to where the scan prints it, the miss capped softly at 3 px
  held sides     don't move held side points across their line (weight 100; they may slide along it)
  held corners   don't move the held control points at all (weight 1000)
  smooth         keep neighbouring grid points' corrections close
  no spikes      no grid point's correction more than twice its neighbours' average (weight 1000)
Every term is a mean over its points (smooth and no spikes: per free point), so each weighs the same on every sheet.
-> first_correction.npz (Gc: the field on the grid, spacing, origin: its top-left page pixel)."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from warp_common import S, SHEET

HOLD, HOLD_CP, RATIO, RIP, CAP = 100.0, 1000.0, 2.0, 1000.0, 3.0
torch.set_num_threads(8)
T8 = np.load(S + "fit_targets.npz")["targets"]
C = np.load(S + "first_correction_inputs.npz")
Hp, HN, HC = C["held_line"], C["held_n"], C["held_cp"]
PR = np.load(S + "propagated_shifts.npz")
Fp, FN, FO = PR["free"], PR["free_n"], PR["free_off"]
CR, CS = C["corner_rec"], C["corner_scan"]
x0, y0 = (int(v) for v in C["origin"])
B, h, w = T8.shape
SP = 50
gh, gw = int(np.ceil((h - 1) / SP)) + 1, int(np.ceil((w - 1) / SP)) + 1
ext = torch.tensor([(gw - 1) * SP, (gh - 1) * SP], dtype=torch.float32)


def gp(xy):
    xy = torch.as_tensor(xy)
    return torch.stack([xy[:, 0] / ext[0] * 2 - 1, xy[:, 1] / ext[1] * 2 - 1], -1)[None, None]


def field(G, g):
    return F.grid_sample(G, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0, :].T


CRt, CSt, HNt, FNt, FOt = (torch.tensor(a) for a in (CR, CS, HN, FN, FO))
gF, gH, gC, gK = gp(Fp), gp(Hp), gp(CR), gp(HC)
print(f"{SHEET}: free side points {len(Fp)}, thrown-out corners {len(CR)}, held side points {len(Hp)}, "
      f"held control points {len(HC)}", flush=True)
G = torch.zeros(1, 2, gh, gw, requires_grad=True)
t0 = time.time()
opt = torch.optim.Adam([G], lr=0.25)
for it in range(1500):
    opt.zero_grad()
    data = torch.mean((torch.sum(field(G, gF) * FNt, dim=1) - FOt) ** 2)
    dc = torch.linalg.norm(CRt + field(G, gC) - CSt, dim=1) if len(CR) else torch.zeros(1)
    cor = torch.mean(CAP * CAP * dc * dc / (dc * dc + CAP * CAP)) / (CAP * CAP)
    held = torch.mean(torch.sum(field(G, gH) * HNt, dim=1) ** 2)
    heldcp = torch.mean(torch.sum(field(G, gK) ** 2, dim=1))
    smooth = (torch.sum((G[:, :, 1:, :] - G[:, :, :-1, :]) ** 2) + torch.sum((G[:, :, :, 1:] - G[:, :, :, :-1]) ** 2)) / len(Fp)
    mag = torch.sqrt(torch.sum(G[0] ** 2, dim=0) + 1e-6)
    nb = 0.25 * (mag[:-2, 1:-1] + mag[2:, 1:-1] + mag[1:-1, :-2] + mag[1:-1, 2:])
    rip = torch.sum(torch.relu(mag[1:-1, 1:-1] - RATIO * nb) ** 2) / len(Fp)
    (data + cor + HOLD * held + HOLD_CP * heldcp + smooth + RIP * rip).backward()
    opt.step()
with torch.no_grad():
    uh = field(G, gH).numpy()
    umax = float(torch.linalg.norm(G[0], dim=0).max())
hm = np.abs(np.sum(uh * HN, axis=1))
print(f"  biggest correction {umax:.1f} px; held sides moved across their line px p99 {np.percentile(hm, 99):.2f}; "
      f"{time.time() - t0:.0f}s", flush=True)
np.savez(S + "first_correction.npz", Gc=G.detach().numpy()[0], spacing=SP, origin=np.array([x0, y0]))
print("saved", S + "first_correction.npz")
