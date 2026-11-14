"""Trial: a zoning code for every lot, no closed areas (no chain change).
1. the unit is the lot (the April 2000 table's lot shapes; its codes are used only to fit the rates and to score).
2. a printed code is evidence for the lot it sits on (or the nearest lot, when printed in a street).
3. for each pair of neighbouring lots, the share f of their shared border that has a wall line along it: a wall within the
   street between them, or on their lot line, read with the measured spread of walls about lot lines (7.3 ft).
   P(the two differ | f, touching or across a street) is fitted on one half of the city.
4. all lots are solved together: each shared border is a conductor of strength length x P(same)/P(differ), each printed code
   feeds its lot; a lot's probability of a code is the share of current reaching it from that code's labels.
5. score: share of lots, and of lot area, given the table's code.  Fitted on the west half, reported on the east half too.
usage: lotcode.py OUT_CITY.png OUT_BERNAL.png"""
import sys

import cv2
import numpy as np
import shapely
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

import lotwalls
import truth

RES = lotwalls.RES
K = truth.K
SG = float(np.load(K / "coarsen_meas.npz")["sigma"])
ALIAS = {"NO BEACH": "NORTH BEACH", "FILLMORE": "UPPER FILLMORE", "WEST PORTAL": "WEST PORTAL AVENUE", "CR-NC": "CRNC",
         "UPR MARKET": "UPPER MARKET", "24TH-NOE": "24TH-NOE VALLEY", "24TH-MISSION": "24TH MISSION", "HAYES": "HAYES-GOUGH"}


def canon(c):
    c = truth.norm(c).removesuffix(" NCD")
    return ALIAS.get(c, c)


B = lotwalls.build("city5")
T = truth.load(B)
g, lot, grown, pairs, ins = T
N = len(g)
x0, y0 = B["x0"], B["y0"]
s0 = truth.score(T, B["comp"])
scored, main = s0["lots_in"], s0["main"]
code = np.array([canon(c) for c in g["code"].values])
area = g["area"].values
cen = shapely.get_coordinates(shapely.centroid(g.geometry.values))
west = cen[:, 0] < np.median(cen[scored, 0])

# 3. wall along each shared border
dw = cv2.distanceTransform((B["wall"] == 0).astype(np.uint8), cv2.DIST_L2, 5)
hh = cv2.distanceTransform((lot < 0).astype(np.uint8), cv2.DIST_L2, 5)
keys = pairs[:, 0] * N + pairs[:, 1]
cov, cnt = np.zeros(len(pairs)), np.zeros(len(pairs))
for A, Bq in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))), ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
    ga, gb = grown[A], grown[Bq]
    m = ga != gb
    k = np.minimum(ga[m], gb[m]).astype(np.int64) * N + np.maximum(ga[m], gb[m])
    i = np.searchsorted(keys, k)
    d = np.maximum(np.minimum(dw[A][m], dw[Bq][m]) - np.maximum(hh[A][m], hh[Bq][m]), 0) * RES
    cov += np.bincount(i, np.exp(-.5 * (d / SG) ** 2), minlength=len(pairs))
    cnt += np.bincount(i, minlength=len(pairs))
del dw, hh
a, b, tw, sw = pairs.T
f = cov / np.maximum(cnt, 1)
L = (tw + sw) * RES
touch = tw >= sw
known = g["known"].values
fit_ok = known[a] & known[b]
diff = code[a] != code[b]


def logit_fit(x, y, w):
    X = np.c_[np.ones(len(x)), x]
    beta = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ beta))
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H + 1e-9 * np.eye(2), X.T @ (w * (y - p)))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


pd_ = np.zeros(len(pairs))
print("3. P(the two lots differ) by the share of their border with a wall along it; fitted on the west half:")
for name, tm in (("touching", touch), ("across a street", ~touch)):
    tr = tm & fit_ok & west[a] & west[b]
    beta = logit_fit(f[tr], diff[tr].astype(float), L[tr])
    pd_[tm] = 1 / (1 + np.exp(-(beta[0] + beta[1] * f[tm])))
    q = 1 / (1 + np.exp(-(beta[0] + beta[1] * np.array([0, .25, .5, .75, 1]))))
    print(f"   {name}: wall along 0 / 25 / 50 / 75 / 100% of the border -> " + " / ".join(f"{v:.1%}" for v in q))
    for lo, hi in ((0, .1), (.1, .5), (.5, .9), (.9, 1.01)):
        mm = tm & fit_ok & (f >= lo) & (f < hi)
        print(f"      measured, share {lo:.1f}-{min(hi, 1):.1f}: {L[mm].sum():>12,.0f} ft, a boundary {L[mm & diff].sum() / max(L[mm].sum(), 1):.1%}")
pd_ = np.clip(pd_, 1e-4, 1 - 1e-4)
cond = L * (1 - pd_) / pd_

# 2. printed codes
names = sorted({canon(c) for c in B["codes"]})
sc = np.round((B["ctrs"][:, 0] - x0) / RES).astype(int)
sr = np.round((B["ctrs"][:, 1] - y0) / RES).astype(int)
okl = ins[sr, sc]
lab_lot = grown[sr, sc][okl]
lab_code = np.array([names.index(canon(c)) for c in B["codes"]])[okl]
lab_xy = B["ctrs"][okl]
lab_area = B["lab_comp"][okl]
kl = known[lab_lot]
print(f"2. printed codes {len(lab_lot):,}; the lot under a printed code has that code in the table: {(np.array(names)[lab_code[kl]] == code[lab_lot[kl]]).mean():.1%}")
print(f"   lots whose table code is printed nowhere on the map: {int((scored & ~np.isin(code, names)).sum()):,}")

# 4. solve
W = coo_matrix((np.r_[cond, cond], (np.r_[a, b], np.r_[b, a])), shape=(N, N)).tocsr()
deg = np.asarray(W.sum(1)).ravel()


def solve(s):
    S = np.zeros((N, len(names)))
    np.add.at(S, (lab_lot, lab_code), s)
    M = (diags(deg + S.sum(1) + 1e-9) - W).tocsc()
    X = splu(M).solve(S)
    return X


def report(name, pred, m):
    okk = pred[m] == code[m]
    return f"{name}: lots right {okk.mean():.1%}, lot area right {area[m][okk].sum() / area[m].sum():.1%}"


code_i = np.array([names.index(c) if c in names else -1 for c in code])
best = None
print("4. strength of a printed code (ft of clear border it is worth), judged on the west half:")
for s in (3000, 10000, 30000, 100000, 1000000, 10000000):
    X = solve(s)
    pred = X.argmax(1)
    r = (pred[scored & west] == code_i[scored & west]).mean()
    print(f"   {s:>6}: west {r:.2%}, east {(pred[scored & ~west] == code_i[scored & ~west]).mean():.2%}")
    if best is None or r > best[0]:
        best = (r, s, pred, X)
_, s, pred, X = best
predc = np.array(names)[pred]
conf = X.max(1) / np.maximum(X.sum(1), 1e-30)

# the chain's closed areas on the same lots: a lot takes the nearest printed code inside its own area
base = np.full(N, "", object)
strict = np.full(N, "", object)
by = {}
for j, A in enumerate(lab_area):
    by.setdefault(int(A), []).append(j)
idx = np.flatnonzero(scored)
o = idx[np.argsort(main[idx])]
st = np.r_[0, np.flatnonzero(np.diff(main[o])) + 1, len(o)]
for u, v in zip(st[:-1], st[1:]):
    js = by.get(int(main[o[u]]))
    if not js:
        continue
    js = np.array(js)
    nn = cKDTree(lab_xy[js]).query(cen[o[u:v]])[1]
    base[o[u:v]] = np.array(names)[lab_code[js[nn]]]
    if len(set(lab_code[js])) == 1:
        strict[o[u:v]] = names[lab_code[js[0]]]
print(f"5. score, {int(scored.sum()):,} lots (west half fitted, east half not):")
for nm, pr in (("chain areas, only areas holding one code", strict), ("chain areas, nearest printed code in the lot's area", base), ("lot trial", predc)):
    print("   " + report(nm, pr, scored) + " | east half " + report("", pr, scored & ~west)[2:])
right = predc == code
for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, 1.01)):
    mm = scored & (conf >= lo) & (conf < hi)
    print(f"   lot trial, its own probability {lo:.2f}-{min(hi, 1):.2f}: {int(mm.sum()):>7,} lots, right {right[mm].mean():.1%}")
from collections import Counter
bad = Counter(zip(predc[scored & ~right], code[scored & ~right]))
print("   most common wrong (given | table | lots): " + "; ".join(f"{p} | {c} | {v}" for (p, c), v in bad.most_common(12)))
np.savez_compressed(truth.D / "lotcode_run.npz", pred=predc.astype(str), conf=conf, scored=scored, f=f, pd=pd_, base=base.astype(str))

# pictures: right pale green, wrong red, chain-right-but-trial-wrong dark red, not scored grey
val = np.zeros(N + 1, np.uint8)                       # 0 none, 1 not scored, 2 right, 3 wrong
val[:N] = np.where(~scored, 1, np.where(right, 2, 3))
COL = np.array([(255, 255, 255), (225, 225, 225), (170, 215, 170), (40, 40, 225)], np.uint8)


def picture(r0, r1, c0, c1, f_, zoom):
    v = val[lot[r0:r1:f_, c0:c1:f_]]
    img = COL[v]
    if zoom > 1:
        img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        wl = cv2.resize(B["wall"][r0:r1, c0:c1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        img[wl > 0] = (90, 90, 90)
    return img[::-1]


H, Wd = lot.shape
cv2.imwrite(sys.argv[1], picture(0, H, 0, Wd, 5, 1))
X_, Y_, half = 6009700, 2098720, 600
kc, kr, hp = int(round((X_ - x0) / RES)), int(round((Y_ - y0) / RES)), int(round(half / RES))
cv2.imwrite(sys.argv[2], picture(kr - hp, kr + hp, kc - hp, kc + hp, 1, 3))
