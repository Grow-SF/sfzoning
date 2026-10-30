"""Residual diagnosis for one sheet: python resid.py SERIES SHEET [OUTDIR]"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import blocks_corners, out_dir
from sfzmaps import config
from sfzmaps.sources import datasf
from sfzmaps.transform import SheetTransform

series, sheet = sys.argv[1], sys.argv[2]
out = out_dir()
d = config.WORK_DIR / series / sheet
rf = json.loads((d / "refine.json").read_text())
t = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
ms = rf["matches"]
blocks, corners = blocks_corners()
cent = {b: g.centroid for b, g in zip(blocks.block_num, blocks.geometry)}
cblk = np.asarray(corners.block_num).astype(str)
lab = np.array([m["label"] for m in ms])
P = np.array([[m["px"] if m["px"] is not None else np.nan, m["py"] if m["py"] is not None else np.nan] for m in ms])
Wm = np.array([[m["wx"] if m["wx"] is not None else np.nan, m["wy"] if m["wy"] is not None else np.nan] for m in ms])
k = np.array([m["ref_idx"] for m in ms])
sel = np.isin(lab, ["inlier", "outlier"])
Q = t.to_world(P[sel]); R = Q - Wm[sel]; L = lab[sel]; K = k[sel]
dist = np.hypot(*R.T)
print(f"{series}/{sheet}: scored {sel.sum()}  inlier {np.sum(L=='inlier')} outlier {np.sum(L=='outlier')}")
print(f" frac<=6 {np.mean(dist<=6):.3f}  <=8 {np.mean(dist<=8):.3f}  <=10 {np.mean(dist<=10):.3f}  p50 {np.median(dist):.1f} p95 {np.quantile(dist,.95):.1f}")
for nm in ("inlier", "outlier"):
    dd = dist[L == nm]
    if len(dd): print(f" {nm}: n {len(dd)} p50 {np.median(dd):.1f} p95 {np.quantile(dd,.95):.1f}")
print(f" mean residual vector {R.mean(0).round(2)}  (global offset)")
# radial wrt modern block centroid: + = scan corner is outside the modern block
C = np.array([[cent[cblk[i]].x, cent[cblk[i]].y] for i in K])
u = Wm[sel] - C; u /= np.linalg.norm(u, axis=1)[:, None]
rad = (R * u).sum(1)
print(f" radial component (+ outward): median {np.median(rad):.2f}  mean {rad.mean():.2f}  inliers median {np.median(rad[L=='inlier']):.2f}")
# corner-bisector component: corners of a rectangle; inset shows up along bisector
# spatial coherence: neighbour-mean residual within radius
Wq = Wm[sel]
tree = cKDTree(Wq)
for rr in (150, 300, 600, 1200):
    nb = tree.query_ball_point(Wq, rr)
    sm = np.array([R[[j for j in n if j != i]].mean(0) if len(n) > 3 else [np.nan, np.nan] for i, n in enumerate(nb)])
    ok = np.isfinite(sm).all(1)
    var_tot = (R[ok] ** 2).sum(1).mean(); var_res = ((R[ok] - sm[ok]) ** 2).sum(1).mean()
    print(f"  neighbour mean within {rr} ft explains {1 - var_res/var_tot:.0%} of residual power (n={ok.sum()}, mean |nbr| {np.hypot(*sm[ok].T).mean():.1f})")
# residual vs block-centroid radial after removing smooth field
fig, ax = plt.subplots(1, 2, figsize=(20, 11))
q = ax[0].quiver(Wq[:, 0], Wq[:, 1], R[:, 0], R[:, 1], dist, angles="xy", scale_units="xy", scale=1/25, cmap="viridis", clim=(0, 15), width=0.0015)
fig.colorbar(q, ax=ax[0], label="|resid| ft")
ax[0].set_title(f"{series}/{sheet} residuals (x25)"); ax[0].set_aspect("equal")
for nm, c in (("real_change", "red"), ("no_equivalent", "lightgray")):
    s = lab == nm
    if s.any() and np.isfinite(P[s]).all(1).any():
        pp = P[s][np.isfinite(P[s]).all(1)]
        ww = t.to_world(pp)
        ax[1].scatter(ww[:, 0], ww[:, 1], s=4, c=c, label=nm)
sc = ax[1].scatter(Wq[:, 0], Wq[:, 1], c=dist, s=6, cmap="RdYlGn_r", vmin=0, vmax=12)
fig.colorbar(sc, ax=ax[1]); ax[1].legend(); ax[1].set_aspect("equal"); ax[1].set_title("|resid| (scored), real_change red, no_eq gray")
fig.tight_layout(); fn = out / f"resid_{series}_{sheet}.png"; fig.savefig(fn, dpi=70); print(" wrote", fn)
np.savez(out / f"resid_{series}_{sheet}.npz", P=P[sel], W=Wq, R=R, L=L, K=K, rad=rad)
