"""Per-block decomposition of residuals (needs resid.py output): uv run python scripts/tuning/block.py 1948 1"""
import sys, json, pickle
import numpy as np
from pathlib import Path
from _common import blocks_corners, out_dir
from sfzmaps import config
from sfzmaps.transform import SheetTransform
series, sheet = sys.argv[1], sys.argv[2]
D = out_dir()
z = np.load(D / f"resid_{series}_{sheet}.npz")
blocks, corners = blocks_corners()
cblk = np.asarray(corners.block_num).astype(str)
t = SheetTransform.from_json(json.loads((config.WORK_DIR/series/sheet/"transform.json").read_text()))
p0 = np.array([[3000., 3000.]]); 
J = (t.to_world(p0 + [[10, 0]]) - t.to_world(p0)) / 10
print("ft per px ~", np.hypot(*J[0]).round(3))
R, W, K = z["R"], z["W"], z["K"]
b = cblk[K]
tot = (R**2).sum(1).mean()
# per-block translation
Rt = R.copy(); Rs = R.copy()
from collections import defaultdict
idx = defaultdict(list)
for i, bb in enumerate(b): idx[bb].append(i)
nb = 0
for bb, ii in idx.items():
    if len(ii) < 3: Rt[ii] = np.nan; Rs[ii] = np.nan; continue
    nb += 1
    ii = np.array(ii)
    Rt[ii] = R[ii] - R[ii].mean(0)
    # per-block affine (translation + 2x2 linear) of residual vs local position
    X = np.c_[W[ii] - W[ii].mean(0), np.ones(len(ii))]
    if len(ii) >= 4:
        coef, *_ = np.linalg.lstsq(X, R[ii], rcond=None)
        Rs[ii] = R[ii] - X @ coef
    else:
        Rs[ii] = np.nan
ok = np.isfinite(Rt).all(1); ok2 = np.isfinite(Rs).all(1)
print(f"blocks with >=3 matched corners: {nb}; corners {ok.sum()}")
print(f" residual power {tot:.1f} ft^2 (rms {np.sqrt(tot):.1f})")
print(f" after removing per-block translation: rms {np.sqrt((Rt[ok]**2).sum(1).mean()):.1f} (vs {np.sqrt((R[ok]**2).sum(1).mean()):.1f} on same corners), dof-corrected")
print(f" after per-block affine (>=4 corners): rms {np.sqrt((Rs[ok2]**2).sum(1).mean()):.1f} vs {np.sqrt((R[ok2]**2).sum(1).mean()):.1f}")
# expected reduction with pure noise: translation removes 1/n of power
ns = np.array([len(idx[bb]) for bb in b])
print(" mean corners/block", ns[ok].mean().round(1), " pure-noise expectation after translation: rms", np.sqrt((R[ok]**2).sum(1).mean()*np.mean(1-1/ns[ok])).round(1))
# per-block size error: projection of residual onto (corner - block centre) along x and y
sx = []; sy = []
for bb, ii in idx.items():
    if len(ii) < 4: continue
    ii = np.array(ii); dx = W[ii] - W[ii].mean(0)
    # least squares stretch: R_x = ax*dx, R_y = ay*dy
    ax = (R[ii,0]*dx[:,0]).sum()/(dx[:,0]**2).sum(); ay = (R[ii,1]*dx[:,1]).sum()/(dx[:,1]**2).sum()
    sx.append(ax); sy.append(ay)
sx, sy = np.array(sx), np.array(sy)
print(f" per-block stretch x: median {np.median(sx)*100:.2f}% IQR {np.quantile(sx,.25)*100:.2f}..{np.quantile(sx,.75)*100:.2f}%; y: median {np.median(sy)*100:.2f}% IQR {np.quantile(sy,.25)*100:.2f}..{np.quantile(sy,.75)*100:.2f}%")
# anisotropy: residual covariance
print(" residual cov (ft^2):", np.cov(R.T).round(1).tolist())
