"""Where each printed label belongs, learned once for every year.

A label either STAYS (it names the district its text sits in) or points into a nearby walled area, for example a box
too small for its text, reached by a leader line.  Per label the options are: stay (the lot under its text), or each
walled area within REACH ft (that area's lot nearest the text).  Areas are lots joined by borders the held-out wall
odds call open.

Features per option: stay; log distance from the text; log area size; a pen stroke running from the text into the
area and its width; labels already in the area (same code / another code / none); how well the area's lot sizes fit
the code; for stay: another code's label in the label's own area, the label printed in a street, its trust.

Answers: the hand-checked leader moves in label_fixes.csv (rows with a fix point) where they match; otherwise the
table: stay if the lot under the text has the label's code, else the nearest option whose lots are mostly that code.
A conditional logit with a small ridge, fitted four times, each time without one quarter.  Every label gets each
model's chances, so a holdout's solve uses only the model that did not see it."""
import csv
import json
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .codes import canon
from .data import FT_PER_PX, INPUTS, LABEL_FIXES, OUT, QUARTERS, city, quarter_of
from .labels import print_set

REACH = 500.0
SERIES = "2000-digital-zoning"
FEATURES = ["stay", "log distance", "log area lots", "stroke link", "link width", "area has same code",
            "another code's label (own area / option)", "printed in a street", "trust", "lot fit", "area empty",
            "link x distance"]


def _leader_moves(label_fixes):
    """the hand-checked leader moves: per sheet, (label position, fix position, match radius, code) in map feet"""
    import os
    os.environ.setdefault("SFZMAPS_DATA_DIR", str(OUT.parent / "data_ay2"))
    from sfzmaps import config
    from sfzmaps.transform import SheetTransform
    moves = {}
    for fx in csv.DictReader(open(label_fixes)):
        if not (fx.get("fix_x") or "").strip():
            continue
        sheet = fx["sheet"].rsplit("-p", 1)[1]
        f = config.WORK_DIR / SERIES / fx["sheet"] / "transform.json"
        t = SheetTransform.from_json(json.loads(f.read_text()))
        p0, p1, pd = t.to_world(np.array([[float(fx["x"]), float(fx["y"])], [float(fx["fix_x"]), float(fx["fix_y"])],
                                          [float(fx["x"]) + 40, float(fx["y"])]]))
        moves.setdefault(sheet, []).append((p0, p1, float(np.hypot(*(pd - p0))), canon(fx["code"])))
    return moves


def _options(label_fixes):
    """per label with a reading: its quarter, whether a hand move answers it, and its options
    [(features, is the answer, lot)]"""
    import shapely

    from .data import wall_odds
    C = city()
    grown = C.grown_raster
    H, W = grown.shape
    N, a, b, cen = C.n, C.a, C.b, C.centroid
    lot_tree = cKDTree(cen)
    L = np.load(INPUTS / "labread2_cwfwxnw.npz", allow_pickle=True)
    lx, ly, sheet, under, street = L["x"], L["y"], np.asarray(L["sheet"]).astype(str), L["lab_lot"], L["street"]
    n_labels = len(under)
    lq = quarter_of(np.c_[lx, ly], cen[C.scored])
    S = np.load(INPUTS / "zstrokes_old.npz")
    xy, off, width = S["xy"], S["off"], S["wmed"]
    end1, end2 = xy[off[:-1]], xy[off[1:] - 1]
    end_tree = cKDTree(np.r_[end1, end2])
    n_strokes = len(width)
    LZ = np.load(INPUTS / "zlotchar2.npz", allow_pickle=True)
    lz_code = {c: i for i, c in enumerate(LZ["codes"].astype(str))}
    size_bin = np.clip(np.digitize(np.log(np.maximum(shapely.area(C.lots.geometry.values), 1.0)), LZ["edges"]) - 1, 0,
                       LZ["PMI"].shape[2] - 1)
    moves = _leader_moves(label_fixes)
    rows = []
    for qi, q in enumerate(QUARTERS):
        R = np.load(INPUTS / f"lotclf6_{print_set(q)}.npz", allow_pickle=True)
        names = [canon(n) for n in R["names"].astype(str)]
        top, top_trust = {}, {}
        for j, k, t in zip(R["cj"], R["ck"], R["cr"]):
            if j < n_labels and t > top_trust.get(j, -1):
                top[j], top_trust[j] = names[k], float(t)
        open_ = wall_odds(q) <= 0
        _, area = connected_components(coo_matrix((np.ones(int(open_.sum())), (a[open_], b[open_])), shape=(N, N)),
                                       directed=False)
        area_size = np.bincount(area)
        labels_in = {}
        for j in range(n_labels):
            if under[j] >= 0 and j in top:
                labels_in.setdefault(area[under[j]], []).append(top[j])
        majority = {}
        for j in np.flatnonzero(lq == qi):
            if under[j] < 0 or j not in top:
                continue
            c = top[j]
            own = area[under[j]]
            pmi = LZ["PMI"][qi][lz_code[c]] if c in lz_code else np.zeros(LZ["PMI"].shape[2])
            near = {}
            for i in lot_tree.query_ball_point([lx[j], ly[j]], REACH):
                if area[i] == own:
                    continue
                d = float(np.hypot(cen[i, 0] - lx[j], cen[i, 1] - ly[j]))
                if area[i] not in near or d < near[area[i]][0]:
                    near[area[i]] = (d, i)
            link = {}                                    # strokes from the text into an area: the far end lies on
            for s_ in end_tree.query_ball_point([lx[j], ly[j]], 40.0):    # one of the area's lots (within 12 ft)
                s, end = s_ % n_strokes, s_ // n_strokes
                far = end2[s] if end == 0 else end1[s]
                for i in lot_tree.query_ball_point(far, 60.0):
                    c_, r_ = C.to_grid(far)
                    win = grown[max(r_ - 3, 0):r_ + 4, max(c_ - 3, 0):c_ + 4]
                    if (win == i).any():
                        link[area[i]] = min(link.get(area[i], 99.0), float(width[s]))
            others = [x for x in labels_in.get(own, []) if x != c]

            def size_fit(lots, pmi=pmi):
                return float(np.mean(pmi[size_bin[lots]])) if len(lots) else 0.0

            answer = None
            for p0, p1, rad, fc in moves.get(sheet[j], []):
                if fc == c and np.hypot(lx[j] - p0[0], ly[j] - p0[1]) <= rad:
                    col, row = (int(round((p1[k] - C.origin[k]) / FT_PER_PX)) for k in (0, 1))
                    l2 = grown[row, col] if 0 <= row < H and 0 <= col < W else -1
                    if l2 >= 0:
                        answer = "stay" if area[l2] == own else area[l2]
                    break
            by_hand = answer is not None
            if answer is None and C.known[under[j]]:
                if C.table[under[j]] == c:
                    answer = "stay"
                else:
                    best = None
                    for ar, (d, i) in near.items():
                        if ar not in majority:
                            lots = np.flatnonzero((area == ar) & C.known)[:400]
                            majority[ar] = Counter(C.table[lots].tolist()).most_common(1)[0][0] if len(lots) else ""
                        if majority[ar] == c and (best is None or d < best[0]):
                            best = (d, ar)
                    answer = best[1] if best else None
            feats = [([1.0, 0.0, np.log1p(area_size[own]), 0.0, 0.0, 0.0, float(len(others) > 0), float(street[j]),
                       top_trust[j], size_fit(np.flatnonzero(area == own)[:400]), 0.0, 0.0],
                      answer == "stay", under[j])]
            for ar, (d, i) in near.items():
                labs = labels_in.get(ar, [])
                feats.append(([0.0, np.log1p(d), np.log1p(area_size[ar]), float(ar in link),
                               np.log1p(link.get(ar, 0.0)), float(c in labs), float(any(x != c for x in labs)),
                               0.0, 0.0, size_fit(np.flatnonzero(area == ar)[:400]), float(len(labs) == 0),
                               float(ar in link) * np.log1p(d)], answer == ar, i))
            rows.append((j, qi, by_hand, feats, answer is not None and any(f[1] for f in feats)))
    return rows


def _neg_log_lik(w, X, y, ridge):
    total, grad = 0.0, np.zeros_like(w)
    for Xi, yi in zip(X, y):
        s = Xi @ w
        s -= s.max()
        p = np.exp(s) / np.exp(s).sum()
        total -= np.log(max(p[yi], 1e-12))
        grad -= Xi[yi] - p @ Xi
    return total + ridge * w @ w, grad + 2 * ridge * w


def fit(label_fixes=LABEL_FIXES):
    """fit the four held-out models, report how often each picks the answer, write placement.npz"""
    rows = _options(label_fixes)
    print(f"labels {len(rows):,}; with an answer {sum(r[4] for r in rows):,} (hand moves "
          f"{sum(r[2] for r in rows if r[4])}); options per label median {int(np.median([len(r[3]) for r in rows]))}")
    out, scored = [], []
    for qi in range(4):
        train = [r for r in rows if r[1] != qi and r[4]]
        X = [np.array([f[0] for f in r[3]], float) for r in train]
        y = [int(np.flatnonzero([f[1] for f in r[3]])[0]) for r in train]
        mu = np.mean(np.vstack(X), 0)
        sd = np.std(np.vstack(X), 0) + 1e-9
        w = minimize(_neg_log_lik, np.zeros(len(FEATURES)), args=([(x - mu) / sd for x in X], y, 1.0), jac=True,
                     method="L-BFGS-B").x
        for r in rows:                                   # every label, by the model fitted without quarter qi
            s = ((np.array([f[0] for f in r[3]], float) - mu) / sd) @ w
            p = np.exp(s - s.max())
            p /= p.sum()
            for oi, f in enumerate(r[3]):
                out.append((r[0], f[2], p[oi], oi == 0, qi))
            if r[1] == qi and r[4]:
                scored.append((r[1], r[2], int(np.flatnonzero([f[1] for f in r[3]])[0]), int(np.argmax(p))))
        if qi == 0:
            print("weights without the north-west (standardised): "
                  + ", ".join(f"{n} {v:+.2f}" for n, v in zip(FEATURES, w)))
    R = np.array(scored)
    for name, m in (("labels with an answer", np.ones(len(R), bool)), ("answer = move", R[:, 2] != 0),
                    ("answer = stay", R[:, 2] == 0), ("hand moves", R[:, 1] == 1)):
        if m.any():
            print(f"{name:24s} {int(m.sum()):5,}: right {np.mean(R[m, 3] == R[m, 2]):6.1%} "
                  f"(always stay {np.mean(R[m, 2] == 0):6.1%})")
    opts = np.array(out, dtype=object)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "placement.npz", label=opts[:, 0].astype(int), lot=opts[:, 1].astype(int),
                        chance=opts[:, 2].astype(float), stay=opts[:, 3].astype(bool), holdout=opts[:, 4].astype(int))
    print(f"saved {OUT / 'placement.npz'}: {len(opts):,} options for {len(set(opts[:, 0].tolist())):,} labels")


def load_options(q):
    """label -> [(lot, chance, is_stay)] for holdout q, stay first"""
    P = np.load(OUT / "placement.npz")
    m = P["holdout"] == QUARTERS.index(q)
    options = {}
    for label, lot, chance, stay in zip(P["label"][m], P["lot"][m], P["chance"][m], P["stay"][m]):
        options.setdefault(int(label), []).append((int(lot), float(chance), bool(stay)))
    return options
