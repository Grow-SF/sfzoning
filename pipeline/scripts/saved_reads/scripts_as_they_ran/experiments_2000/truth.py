"""The answer set and the score.  Answer: DataSF 'Historical Zoning Districts - April 2000', a zoning code on every lot.
On the wall grid (4 ft): every lot is painted, then grown to its nearest-lot region so lots facing across a street become
neighbours.  Each pair of neighbouring lots has a shared border length; the answer says whether their codes differ.
score(area): area = labels of closed areas (0 = wall).  A lot belongs to the area holding most of its own pixels; a pair is
'separated' when the two lots belong to different areas.  Reported, all in feet of shared border:
  found  = true boundary that is separated / all true boundary        (recall)
  right  = separated border that is true boundary / all separated     (precision)
  score  = 2 found right / (found + right)                             (the one number)
Pairs with a lot whose code is blank or split ('RH-1/RH-2') are left out."""
import re
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import shapely
from scipy import ndimage

import lotwalls

K = Path(__file__).parent
D = K.parent / "zon2000"
RES = lotwalls.RES


def norm(z):
    z = re.sub(r"\s+", " ", str(z or "").strip().upper())
    z = re.sub(r"^(RH|RM|RC|M|C)(\d)", r"\1-\2", z)
    return z


def load(B):
    """lots (GeoDataFrame, code, known), lot raster, grown raster, pairs (a, b, touch px, street px), mapped mask"""
    f = D / "truth_grid.npz"
    g = gpd.read_parquet(D / "april2000.parquet")
    g["code"] = g["zoning"].map(norm)
    g["known"] = (g["code"] != "") & ~g["code"].str.contains("/")
    H, W = B["H"], B["W"]
    if f.exists():
        z = np.load(f)
        return g, z["lot"], z["grown"], z["pairs"], z["inside"]
    lot = np.full((H, W), -1, np.int32)
    for i in np.argsort(-g.geometry.area.values):                  # big lots first, so a lot inside another stays visible
        for p in lotwalls.polys(g.geometry.values[i]):
            cv2.fillPoly(lot, [B["px"](shapely.get_coordinates(p.exterior))], int(i))
    inside = np.zeros((H, W), np.uint8)
    for p in lotwalls.polys(B["fp"]):
        cv2.fillPoly(inside, [B["px"](shapely.get_coordinates(p.exterior))], 1)
    idx = ndimage.distance_transform_edt(lot < 0, return_distances=False, return_indices=True)
    grown = lot[idx[0], idx[1]]
    del idx
    N = len(g)
    keys, tw, sw_ = [], [], []
    for a, b, ta, tb in ((grown[:, :-1], grown[:, 1:], lot[:, :-1], lot[:, 1:]), (grown[:-1], grown[1:], lot[:-1], lot[1:])):
        m = a != b
        lo, hi = np.minimum(a[m], b[m]).astype(np.int64), np.maximum(a[m], b[m]).astype(np.int64)
        t = (ta[m] >= 0) & (tb[m] >= 0)
        keys.append(lo * N + hi), tw.append(t), sw_.append(~t)
    keys, tw, sw_ = np.concatenate(keys), np.concatenate(tw), np.concatenate(sw_)
    u, inv = np.unique(keys, return_inverse=True)
    pairs = np.c_[u // N, u % N, np.bincount(inv, tw), np.bincount(inv, sw_)].astype(np.int64)
    np.savez_compressed(f, lot=lot, grown=grown, pairs=pairs, inside=inside.astype(bool))
    return g, lot, grown, pairs, inside.astype(bool)


def lot_area(lot, area):
    """for each lot seen in the window: its main area, its pixels there, its pixels in any area"""
    m = (lot >= 0) & (area > 0)
    li, ai = lot[m].astype(np.int64), area[m].astype(np.int64)
    M = int(ai.max()) + 1 if len(ai) else 1
    u, c = np.unique(li * M + ai, return_counts=True)
    ul, ua = u // M, u % M
    o = np.lexsort((-c, ul))
    ul, ua, c = ul[o], ua[o], c[o]
    first = np.r_[True, ul[1:] != ul[:-1]]
    tot = np.add.reduceat(c, np.flatnonzero(first)) if len(c) else c
    return ul[first], ua[first], c[first], tot


def score(T, area, win=None, inside=None):
    """T = load(B); area labels on the full grid, or on the window win = (r0, r1, c0, c1).  Lots must lie wholly in the window
    (and in the mapped area)."""
    g, lot, grown, pairs, ins = T
    r0, r1, c0, c1 = win or (0, lot.shape[0], 0, lot.shape[1])
    L = lot[r0:r1, c0:c1]
    N = len(g)
    full = np.bincount(lot[lot >= 0], minlength=N) if "full" not in CACHE else CACHE["full"]
    CACHE["full"] = full
    here = np.bincount(L[L >= 0], minlength=N)
    good = np.bincount(L[(L >= 0) & ins[r0:r1, c0:c1]], minlength=N)
    whole = (here == full) & (good == full) & (full > 0)
    ul, ua, cm, tot = lot_area(L, area[r0:r1, c0:c1] if area.shape == lot.shape else area)
    main = np.full(N, -1, np.int64)
    main[ul] = ua
    cut = np.zeros(N)
    cut[ul] = 1 - cm / np.maximum(tot, 1)
    a, b, tw, sw_ = pairs.T
    known = g["known"].values
    code = g["code"].values
    ok = whole[a] & whole[b] & known[a] & known[b] & (main[a] >= 0) & (main[b] >= 0)
    a, b, tw, sw_ = a[ok], b[ok], tw[ok], sw_[ok]
    diff = code[a] != code[b]
    sep = main[a] != main[b]
    out = {}
    for name, w in (("all", tw + sw_), ("touching", tw), ("street", sw_)):
        w = w * RES
        true, septot, hit = w[diff].sum(), w[sep].sum(), w[diff & sep].sum()
        found, right = hit / max(true, 1e-9), hit / max(septot, 1e-9)
        out[name] = dict(true_ft=true, found_ft=hit, missed_ft=true - hit, false_ft=septot - hit, found=found, right=right,
                         score=2 * found * right / max(found + right, 1e-9))
    lots_in = whole & known & (main >= 0)
    out["lots"] = int(lots_in.sum())
    out["cut"] = float((cut[lots_in] * full[lots_in]).sum() / max(full[lots_in].sum(), 1))
    out["pairs"] = (a, b, diff, sep)
    out["main"], out["lots_in"] = main, lots_in
    return out


CACHE = {}


def classmap(T, s):
    """on the grid: 1 true boundary found, 2 true boundary missed, 3 false wall (a border the walls separate, same code)"""
    g, lot, grown, pairs, ins = T
    N = len(g)
    a, b, diff, sep = s["pairs"]
    cls = np.where(diff & sep, 1, np.where(diff, 2, np.where(sep, 3, 0))).astype(np.uint8)
    keys = a * N + b
    o = np.argsort(keys)
    keys, cls = keys[o], cls[o]
    out = np.zeros(grown.shape, np.uint8)
    for sl_a, sl_b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))), ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        ga, gb = grown[sl_a], grown[sl_b]
        m = ga != gb
        k = np.minimum(ga[m], gb[m]).astype(np.int64) * N + np.maximum(ga[m], gb[m])
        i = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
        v = np.where(keys[i] == k, cls[i], 0).astype(np.uint8)
        t = out[sl_a]
        t[m] = np.maximum(t[m], v)
    return out


def line(name, s):
    t = s["all"]
    return (f"{name}: score {t['score']:.3f} (found {t['found']:.1%} of {t['true_ft']:,.0f} ft true boundary; right {t['right']:.1%}; "
            f"missed {t['missed_ft']:,.0f} ft, false {t['false_ft']:,.0f} ft; lots {s['lots']:,}; lot area cut off by walls {s['cut']:.1%})")


if __name__ == "__main__":
    B = lotwalls.build("city5")
    print("B keys", sorted(B.keys()), "grid", B["H"], B["W"], flush=True)
    T = load(B)
    g, lot, grown, pairs, ins = T
    print("lots painted", len(np.unique(lot[lot >= 0])), "of", len(g), "; known code", int(g['known'].sum()), "; pairs", len(pairs))
    s = score(T, B["comp"])
    print(line("chain walls, whole city", s))
    for k in ("touching", "street"):
        t = s[k]
        print(f"   {k}: score {t['score']:.3f}, found {t['found']:.1%} of {t['true_ft']:,.0f} ft, right {t['right']:.1%}, false {t['false_ft']:,.0f} ft")
