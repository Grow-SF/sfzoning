"""Solve block-side shifts by carrying them outward from what is right (sheet NN, today's fit, page px).
  start    held sides (dm_held: on the print today) are solved with shift (0, 0)
  order    repeatedly the free side nearest the solved ones (by its closest point)
  prior    the 2-D shift of its 30 nearest solved points, weighted 1 / distance^2
  search   each of its points looks across its own line, from (point + prior), for the nearest printed block-weight
           centre running its way (reg2 maps), up to just under half a street width each way (the line across the
           street is a full street width away)
  solved   prior + (median of the found offsets) x the side's normal; if none of its points find a line: the prior
-> prop_NN.npz: per free side point (crop px) its target offset along its normal; per side the solved shift.
usage: propagate.py NN"""
import heapq
import json
import sys

import numpy as np
import shapely
from scipy.spatial import cKDTree

from sfzmaps import config, linework
from sfzmaps.transform import SheetTransform

from ay import *  # noqa: F403
T0 = SheetTransform.from_json(json.loads((D / "transform.json").read_text()))
R2 = np.load(S + f"reg2_{NN}.npz")
TG = R2["targets"]
x0, y0 = (int(v) for v in R2["origin"])
B, h, w = TG.shape
B_ = json.loads((D / "blocks_px.geojson").read_text())
SW, _ = linework.estimate_street_px([shapely.geometry.shape(f["geometry"]) for f in B_["features"]])
SD = np.load(S + f"dm_sides_{NN}.npz")
held_side = np.load(S + f"dm_held_{NN}.npz")["held"]
sid = SD["sid"]
_pp, _pa, _pb = side_pixels(T0, SD)          # E4: computed once per sheet (bit-identical)
P = _pp - [x0, y0]
dv = _pb - _pa
L = np.linalg.norm(dv, axis=1)
tng = dv / np.maximum(L, 1e-9)[:, None]
nrm = np.column_stack([-tng[:, 1], tng[:, 0]])
pbin = np.round((np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180) / 10.0).astype(int) % B
ok = np.isfinite(P).all(1) & (P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)
ns = sid.max() + 1
members = [np.flatnonzero((sid == s) & ok) for s in range(ns)]
shift = np.full((len(P), 2), np.nan)
solved_side = np.zeros(ns, bool)
for s in range(ns):
    if held_side[s] and len(members[s]):
        shift[members[s]] = 0.0
        solved_side[s] = True
reach = np.arange(-(SW / 2 - 0.5), SW / 2 - 0.5 + 0.01, 0.5)
print(f"p{NN}: street width {SW:.1f} px, search +-{reach.max():.1f} px around the prior; sides {ns}, held {solved_side.sum()}", flush=True)


def solved_tree():
    idx = np.flatnonzero(~np.isnan(shift[:, 0]))
    return cKDTree(P[idx]), idx


tree, sidx = solved_tree()
pending = [s for s in range(ns) if not solved_side[s] and len(members[s])]
dist0 = {s: float(tree.query(P[members[s]])[0].min()) for s in pending}
heap = [(dist0[s], s) for s in pending]
heapq.heapify(heap)
found_share, moved = [], {}
k = 0
while heap:
    _, s = heapq.heappop(heap)
    if solved_side[s]:
        continue
    m = members[s]
    dd, ii = tree.query(P[m].mean(0)[None], k=min(30, len(sidx)))
    dd, ii = np.atleast_1d(dd[0]), np.atleast_1d(ii[0])
    wts = 1.0 / np.maximum(dd, 1.0) ** 2
    prior = (shift[sidx[ii]] * wts[:, None]).sum(0) / wts.sum()
    q = P[m][:, None, :] + prior[None, None, :] + nrm[m][:, None, :] * reach[None, :, None]
    xi = np.clip(np.round(q[..., 0]).astype(int), 0, w - 1); yi = np.clip(np.round(q[..., 1]).astype(int), 0, h - 1)
    dmap = TG[pbin[m][:, None], yi, xi].astype(float)
    best = np.argmin(dmap + 1e-3 * np.abs(reach)[None, :], axis=1)
    hit = dmap[np.arange(len(m)), best] <= 1.0
    found_share.append(hit.mean())
    if hit.any():
        off = float(np.median(reach[best[hit]]))
        nn_ = np.median(nrm[m], axis=0)
        sv = prior + off * nn_ / max(np.linalg.norm(nn_), 1e-9)
    else:
        sv = prior
    shift[m] = sv
    moved[s] = sv
    solved_side[s] = True
    k += 1
    if k % 50 == 0:
        tree, sidx = solved_tree()
        rest = [(float(tree.query(P[members[t]])[0].min()), t) for t in range(ns) if not solved_side[t] and len(members[t])]
        heap = rest
        heapq.heapify(heap)
fs = np.array(found_share)
mv = np.array([np.linalg.norm(v) for v in moved.values()])
print(f"free sides solved {len(moved)}: share of their points finding a line near the prior p50 {np.median(fs):.0%}; "
      f"sides finding none {np.mean(fs == 0):.0%}; solved shift px p50 {np.median(mv):.1f} p90 {np.percentile(mv, 90):.1f} max {mv.max():.1f}")
free = ok & ~held_side[sid]
tgt = np.sum(shift[free] * nrm[free], axis=1)          # target offset along each point's normal
np.savez(S + f"prop_{NN}.npz", free=P[free].astype(np.float32), free_n=nrm[free].astype(np.float32), free_off=tgt.astype(np.float32),
         side_shift=np.array([moved.get(s, (0.0, 0.0)) for s in range(ns)]), side_free=~held_side,
         side_mid=np.array([P[members[s]].mean(0) if len(members[s]) else (np.nan, np.nan) for s in range(ns)]) + [x0, y0])
