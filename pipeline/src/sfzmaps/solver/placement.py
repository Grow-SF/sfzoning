"""Where each printed label belongs, learned once for every year.

A label either STAYS (it names the district its text sits in) or points into a nearby walled area, for example a box
too small for its text, reached by a leader line.  Per label the options are: stay (the lot under its text), or each
walled area within REACH ft (that area's lot nearest the text).  Areas are lots joined by borders the held-out wall
odds call open (odds <= 0).

Features per option: stay; log distance from the text; log area size; a pen stroke (strokes.npz) running from the
text into the area and its width; labels already in the area (same code / another code / none); how well the area's
lot sizes fit the code (lotsize.npz); for stay: another code's label in the label's own area, the label printed in a
street, its trust.

Answers: the hand-checked leader moves (the rows of label_fixes.csv with a fix point, placed on the map through each
sheet's corrected warp) where they match; otherwise the table as downloaded: stay if the lot under the text has the
label's code, else the nearest option whose lots are mostly that code.  A conditional logit with a small ridge,
fitted four times, each time without the labels whose lot (under the text) is in one quarter.  Every label gets each
model's chances, so a holdout's solve uses only the model that did not see it.

Writes placement.npz into the run's folder (`runs.solve_dir`): per option, `label` (a prints.parquet row), `lot`,
`chance`, `stay` and `holdout` (the quarter the model held out: nw 0, ne 1, sw 2, se 3); and `walls`, the digest of
the walls it was fitted with (`city.walls_digest`): a solve with other walls refuses it."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from sfzmaps import config
from sfzmaps.solver import paths, runs
from sfzmaps.solver.city import city, wall_odds, walls_digest
from sfzmaps.solver.codes import canon
from sfzmaps.solver.trust import load_prints

REACH = 500.0               # ft from the text: the walled areas a label may point into
LINK_REACH = 40.0           # ft: a stroke end this near the text starts a link
LINK_LOT_REACH = 60.0       # ft: lots whose centroid is this near the stroke's far end are looked at
LINK_WINDOW = 3             # px: the far end lies on a lot when the lot's grown px are within this of it (12 ft)
MATCH_PX = 40.0             # page px: a hand move matches a label within this of the move's label position
MAX_AREA_LOTS = 400         # an area's lot sizes and majority code are read from its first this many lots
RIDGE = 1.0
FEATURES = ["stay", "log distance", "log area lots", "stroke link", "link width", "area has same code",
            "another code's label (own area / option)", "printed in a street", "trust", "lot fit", "area empty",
            "link x distance"]


def placement_file(folder) -> Path:
    """the placement of the run whose folder is `folder`"""
    return Path(folder) / "placement.npz"


def leader_moves(label_fixes: Path | None = None) -> dict:
    """The hand-checked leader moves: per sheet, [(label position, fix position, match radius, code)] in map feet,
    placed by each sheet's corrected warp (data/warp/<series>/<sheet>/transform.json).  The radius is MATCH_PX page
    px at the label, in feet."""
    from sfzmaps.transform import SheetTransform
    moves, transforms = {}, {}
    with open(label_fixes or runs.LABEL_FIXES, newline="") as f:
        for fx in csv.DictReader(f):
            if not (fx.get("fix_x") or "").strip():
                continue
            sheet = fx["sheet"]
            if sheet not in transforms:
                folder = config.DATA_DIR / "warp" / paths.SERIES / sheet
                transforms[sheet] = SheetTransform.from_json(json.loads((folder / "transform.json").read_text()),
                                                             folder=folder)
            x, y = float(fx["x"]), float(fx["y"])
            p0, p1, pd = transforms[sheet].to_world(np.array([[x, y], [float(fx["fix_x"]), float(fx["fix_y"])],
                                                              [x + MATCH_PX, y]]))
            moves.setdefault(sheet, []).append((p0, p1, float(np.hypot(*(pd - p0))), canon(fx["code"])))
    return moves


def _options(walls_dir=None):
    """per label with a reading: its quarter, whether a hand move answers it, and its options
    [(features, is the answer, lot)]"""
    import shapely
    C = city()
    grown = C.grown_raster
    H, W = grown.shape
    N, a, b, cen = C.n, C.a, C.b, C.centroid
    lot_tree = cKDTree(cen)
    found = load_prints()
    if not len(found):
        return []
    lx, ly = found["x"].to_numpy(float), found["y"].to_numpy(float)
    sheet = found["sheet"].astype(str).to_numpy()
    under = found["lab_lot"].to_numpy(np.int64)
    street = found["street"].to_numpy(bool)
    n_labels = len(under)
    lq = np.where(under >= 0, C.quarter[np.maximum(under, 0)], -1)
    S = np.load(paths.LABELS_DIR / "strokes.npz")
    xy, off, width = S["xy"].astype(float), S["off"], S["wmed"]
    end1, end2 = xy[off[:-1]], xy[off[1:] - 1]
    end_tree = cKDTree(np.r_[end1, end2])
    n_strokes = len(width)
    LZ = np.load(paths.LABELS_DIR / "lotsize.npz")
    lz_code = {c: i for i, c in enumerate(LZ["codes"].astype(str))}
    size_bin = np.clip(np.digitize(np.log(np.maximum(shapely.area(C.lots.geometry.values), 1.0)), LZ["edges"]) - 1, 0,
                       LZ["PMI"].shape[2] - 1)
    moves = leader_moves() if paths.is_2000() else {}    # the hand leader moves are the 2000 sheets'
    rows = []
    for qi, q in enumerate(paths.QUARTERS):
        R = np.load(runs.SOLVE_LABELS_DIR / f"readings_{q}.npz")
        names = [canon(n) for n in R["names"].astype(str)]
        top, top_trust = {}, {}
        for j, k, t in zip(R["print_index"], R["code_index"], R["trust"]):
            if j < n_labels and t > top_trust.get(j, -1):
                top[j], top_trust[j] = names[k], float(t)
        open_ = wall_odds(q, walls_dir) <= 0
        _, area = connected_components(coo_matrix((np.ones(int(open_.sum())), (a[open_], b[open_])), shape=(N, N)),
                                       directed=False)
        area_size = np.bincount(area)
        labels_in = {}
        for j in range(n_labels):
            if under[j] >= 0 and j in top:
                labels_in.setdefault(area[under[j]], []).append(top[j])
        majority = {}
        for j in np.flatnonzero(lq == qi):
            if j not in top:
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
            for s_ in end_tree.query_ball_point([lx[j], ly[j]], LINK_REACH):     # one of the area's lots
                s, end = s_ % n_strokes, s_ // n_strokes
                far = end2[s] if end == 0 else end1[s]
                for i in lot_tree.query_ball_point(far, LINK_LOT_REACH):
                    c_, r_ = C.px(far)
                    win = grown[max(r_ - LINK_WINDOW, 0):r_ + LINK_WINDOW + 1,
                                max(c_ - LINK_WINDOW, 0):c_ + LINK_WINDOW + 1]
                    if (win == i).any():
                        link[area[i]] = min(link.get(area[i], 99.0), float(width[s]))
            others = [x for x in labels_in.get(own, []) if x != c]

            def size_fit(lots, pmi=pmi):
                return float(np.mean(pmi[size_bin[lots]])) if len(lots) else 0.0

            answer = None
            for p0, p1, rad, fc in moves.get(sheet[j], []):
                if fc == c and np.hypot(lx[j] - p0[0], ly[j] - p0[1]) <= rad:
                    col, row = C.px(p1)
                    l2 = grown[row, col] if 0 <= row < H and 0 <= col < W else -1
                    if l2 >= 0:
                        answer = "stay" if area[l2] == own else area[l2]
                    break
            by_hand = answer is not None
            if answer is None and C.known[under[j]]:
                if C.table_raw[under[j]] == c:
                    answer = "stay"
                else:
                    best = None
                    for ar, (d, i) in near.items():
                        if ar not in majority:
                            lots = np.flatnonzero((area == ar) & C.known)[:MAX_AREA_LOTS]
                            majority[ar] = (Counter(C.table_raw[lots].tolist()).most_common(1)[0][0] if len(lots)
                                            else "")
                        if majority[ar] == c and (best is None or d < best[0]):
                            best = (d, ar)
                    answer = best[1] if best else None
            feats = [([1.0, 0.0, np.log1p(area_size[own]), 0.0, 0.0, 0.0, float(len(others) > 0), float(street[j]),
                       top_trust[j], size_fit(np.flatnonzero(area == own)[:MAX_AREA_LOTS]), 0.0, 0.0],
                      answer == "stay", under[j])]
            for ar, (d, i) in near.items():
                labs = labels_in.get(ar, [])
                feats.append(([0.0, np.log1p(d), np.log1p(area_size[ar]), float(ar in link),
                               np.log1p(link.get(ar, 0.0)), float(c in labs), float(any(x != c for x in labs)),
                               0.0, 0.0, size_fit(np.flatnonzero(area == ar)[:MAX_AREA_LOTS]), float(len(labs) == 0),
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


def fit_model(rows, qi: int):
    """The model fitted without quarter qi: (mu, sd, w), the features' means and spreads over its training options
    and the standardised weights."""
    train = [r for r in rows if r[1] != qi and r[4]]
    X = [np.array([f[0] for f in r[3]], float) for r in train]
    y = [int(np.flatnonzero([f[1] for f in r[3]])[0]) for r in train]
    mu = np.mean(np.vstack(X), 0)
    sd = np.std(np.vstack(X), 0) + 1e-9
    w = minimize(_neg_log_lik, np.zeros(len(FEATURES)), args=([(x - mu) / sd for x in X], y, RIDGE), jac=True,
                 method="L-BFGS-B").x
    return mu, sd, w


def fit(folder, walls_dir=None, log=print, models=None) -> dict:
    """Fit the four held-out models with the walls of `walls_dir` (default WALLS_DIR), report how often each picks
    the answer on its held-out labels, write placement.npz into the run's `folder`.  Returns the counts and hit
    rates.  `models` (four (mu, sd, w), nw ne sw se: the 2000 map's, `transfer`): apply them instead of fitting,
    for a series with no answers of its own."""
    rows = _options(walls_dir)
    summary = {"labels": len(rows), "with_answer": sum(r[4] for r in rows),
               "hand_moves": sum(r[2] for r in rows if r[4]),
               "median_options": int(np.median([len(r[3]) for r in rows])) if rows else 0}
    log(f"labels {summary['labels']:,}; with an answer {summary['with_answer']:,} (hand moves "
        f"{summary['hand_moves']}); options per label median {summary['median_options']}")
    out, scored = [], []
    for qi in range(4):
        mu, sd, w = fit_model(rows, qi) if models is None else models[qi]
        for r in rows:                                   # every label, by the model fitted without quarter qi
            s = ((np.array([f[0] for f in r[3]], float) - mu) / sd) @ w
            p = np.exp(s - s.max())
            p /= p.sum()
            for oi, f in enumerate(r[3]):
                out.append((r[0], f[2], p[oi], oi == 0, qi))
            if r[1] == qi and r[4] and models is None:
                scored.append((r[1], r[2], int(np.flatnonzero([f[1] for f in r[3]])[0]), int(np.argmax(p))))
        if qi == 0:
            log("weights without the north-west (standardised): "
                + ", ".join(f"{n} {v:+.2f}" for n, v in zip(FEATURES, w)))
    R = np.array(scored).reshape(-1, 4)
    summary["held_out"] = {}
    for name, m in (("labels with an answer", np.ones(len(R), bool)), ("answer = move", R[:, 2] != 0),
                    ("answer = stay", R[:, 2] == 0), ("hand moves", R[:, 1] == 1)):
        if m.any():
            right, stay = float(np.mean(R[m, 3] == R[m, 2])), float(np.mean(R[m, 2] == 0))
            summary["held_out"][name] = {"n": int(m.sum()), "right": right, "always_stay": stay}
            log(f"{name:24s} {int(m.sum()):5,}: right {right:6.1%} (always stay {stay:6.1%})")
    opts = np.array(out, dtype=object)
    Path(folder).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(placement_file(folder), label=opts[:, 0].astype(int), lot=opts[:, 1].astype(int),
                        chance=opts[:, 2].astype(float), stay=opts[:, 3].astype(bool), holdout=opts[:, 4].astype(int),
                        walls=walls_digest(walls_dir))
    summary["options"] = len(opts)
    log(f"saved {placement_file(folder)}: {len(opts):,} options for {len(set(opts[:, 0].tolist())):,} labels")
    return summary


def load_options(q: str, folder, walls_dir: Path | None = None) -> dict:
    """label -> [(lot, chance, is_stay)] for holdout q, stay first, from the placement in the run's `folder`.  It
    must have been fitted with the walls of `walls_dir` (default WALLS_DIR): its walled areas come from them."""
    f = placement_file(folder)
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver placement`")
    P = np.load(f)
    if "walls" not in P.files or str(P["walls"]) != walls_digest(walls_dir):
        raise ValueError(f"{f} was fitted with other walls than {walls_dir or runs.WALLS_DIR}: rerun "
                         "`sfzmaps solver placement` with the same --walls")
    m = P["holdout"] == paths.QUARTERS.index(q)
    options = {}
    for label, lot, chance, stay in zip(P["label"][m], P["lot"][m], P["chance"][m], P["stay"][m]):
        options.setdefault(int(label), []).append((int(lot), float(chance), bool(stay)))
    return options
