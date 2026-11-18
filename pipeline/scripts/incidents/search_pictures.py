"""Pictures of a search incident at 2x the grid (each grid px = 2 x 2 image px), in ONE quarter's solve (not the
join, whose quarter split would draw a line of its own): the page's ink, the table's district borders, the quarter's
solved answer's and the lower-energy answer's (the census's cheapest answer right on the incident), the labels.

    uv run python scripts/incidents/search_pictures.py ID [--window C0 R0 C1 R1]

  search-<ID>-<q>_answers.png  ink grey; table borders green (widest), the solve's blue (middle), the lower answer's
                               red (thin, on top); the lots the lower answer changes tinted by their change (one tint
                               per from -> to); labels as black dots with their top reading and trust
  search-<ID>-<q>_walls.png    every pair's wall probability under the quarter's walls (white 0 .. red 1), the pairs
                               the two answers draw differently wide
Window: grid px (column, row) of the crop; default the changed lots plus a margin.  Written into
data/solver/experiments/incidents-search-diagnosis/pictures/.  Reads only."""
import argparse
import json
from collections import Counter

import cv2
import numpy as np
from scipy.special import expit

from common import CENSUS_DIR
from sfzmaps.solver import grid, runs
from sfzmaps.solver.city import city, run_file, wall_odds
from sfzmaps.solver.codes import canon, table_vocabulary
from sfzmaps.solver.labels import readings
from sfzmaps.solver.trust import load_prints

S, MARGIN = 2, 60
OUT = runs.EXPERIMENTS_DIR / "incidents-search-diagnosis" / "pictures"
TINTS = [(150, 235, 255), (150, 200, 255), (200, 230, 170), (230, 190, 230), (200, 200, 140), (170, 220, 220)]


def up(m):
    return np.kron(m.astype(np.uint8), np.ones((S, S), np.uint8))


def wide(m, w):
    u = up(m)
    return cv2.dilate(u, np.ones((w, w), np.uint8)) > 0 if w > 1 else u > 0


def legend(img, lines):
    strip = np.full((22 * len(lines) + 8, img.shape[1], 3), 255, np.uint8)
    for i, (text, col) in enumerate(lines):
        y, x = 20 + 22 * i, 6
        if col is not None:
            cv2.rectangle(strip, (x, y - 12), (x + 22, y + 2), col, -1)
            x += 30
        cv2.putText(strip, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def main(incident: int, window=None) -> None:
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    rec = next(t for t in doc["list"] if t["id"] == incident)
    q = rec["quarter"]
    C = city()
    a, b = C.a, C.b
    R = np.load(run_file("final", f"-{q}", runs.SOLVE_DIR), allow_pickle=True)
    x = np.array([canon(n) for n in np.asarray(R["names"]).astype(str)])[R["pred"]]
    z = np.load(CENSUS_DIR / "answers" / f"cheapest-right-{incident}.npz")
    y = x.copy()
    y[z["lots"]] = [canon(c) for c in z["codes"].astype(str)]
    changed = np.flatnonzero(x != y)
    both = C.known[a] & C.known[b]
    tline = both & (C.table[a] != C.table[b])
    dx, dy = x[a] != x[b], y[a] != y[b]
    lot, grown = C.lot_raster, C.grown_raster
    if window:
        c0, r0, c1, r1 = window
    else:
        m = np.zeros(C.n + 1, bool)
        m[changed] = True
        rr, cc = np.nonzero(m[np.where(lot >= 0, lot, C.n)])
        r0, r1 = max(rr.min() - MARGIN, 0), min(rr.max() + MARGIN, grid.HEIGHT)
        c0, c1 = max(cc.min() - MARGIN, 0), min(cc.max() + MARGIN, grid.WIDTH)
    ink = C.scan()[r0:r1, c0:c1]
    L, G = lot[r0:r1, c0:c1], grown[r0:r1, c0:c1]
    H, W = L.shape
    keys = a.astype(np.int64) * C.n + b
    pair_px = np.full((H, W), -1, np.int64)
    for gA, gB, s1, s2 in ((G[:, :-1], G[:, 1:], (slice(None), slice(0, W - 1)), (slice(None), slice(1, W))),
                           (G[:-1], G[1:], (slice(0, H - 1), slice(None)), (slice(1, H), slice(None)))):
        lo_, hi_ = np.minimum(gA, gB).astype(np.int64), np.maximum(gA, gB).astype(np.int64)
        k = np.minimum(np.searchsorted(keys, lo_ * C.n + hi_), len(keys) - 1)
        ok = (gA != gB) & (keys[k] == lo_ * C.n + hi_)
        for sl in (s1, s2):
            sub = pair_px[sl]
            sub[ok] = k[ok]
            pair_px[sl] = sub
    has, pk = pair_px >= 0, np.maximum(pair_px, 0)
    P = load_prints()
    rd = readings(q, table_vocabulary(C.lots))
    top = {}
    for lab, code, t, far in zip(rd.label, rd.code, rd.trust, rd.far):
        if not far and t > top.get(int(lab), ("", -1))[1]:
            top[int(lab)] = (rd.names[code], t)
    cr = grid.to_grid(P[["x", "y"]].to_numpy())

    def labels_on(img):
        for j, (c, r) in enumerate(cr):
            if c0 <= c < c1 and r0 <= r < r1 and j in top:
                px, py = int((c - c0) * S), int((r - r0) * S)
                cv2.circle(img, (px, py), 4, (0, 0, 0), -1)
                txt = f"{top[j][0]} {top[j][1]:.2f}"
                for col, th in (((255, 255, 255), 3), ((0, 0, 0), 1)):
                    cv2.putText(img, txt, (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"search-{incident}-{q}" + ("_window" if window else "")
    Ls = np.where(L >= 0, L, C.n)
    img = np.full((H * S, W * S, 3), 255, np.uint8)
    trans = Counter(zip(x[changed].tolist(), y[changed].tolist())).most_common(len(TINTS))
    tint_lines = []
    for ((f, t), _), tint in zip(trans, TINTS):
        m = np.zeros(C.n + 1, bool)
        m[changed[(x[changed] == f) & (y[changed] == t)]] = True
        img[up(m[Ls]) > 0] = tint
        tint_lines.append((f"lots the lower answer changes {f} -> {t}: {int(m.sum())}", tint))
    g = up(ink) > 0
    img[g] = (img[g] * 0.35 + np.array([110, 110, 110]) * 0.65).astype(np.uint8)
    TABLE, RUN, LOWER = (60, 180, 60), (220, 120, 20), (40, 30, 220)
    img[wide(has & tline[pk], 9)] = TABLE
    img[wide(has & dx[pk], 5)] = RUN
    img[wide(has & dy[pk], 2)] = LOWER
    labels_on(img)
    img = legend(img, [
        (f"#{incident} ({rec['lots_count']} lots, table {rec['table']}, solved {rec['solved']}): quarter {q}'s solve "
         f"alone (not the join); lower answer dE {sum(rec['groups']['cheapest_right_on_incident'].values()):+.1f}",
         None),
        (f"2x the grid (1 image px = 2 ft); crop {W * 4} x {H * 4} ft; grey: page ink", None),
        *tint_lines,
        ("table district border (widest)", TABLE), (f"final-{q}'s border (middle)", RUN),
        ("the lower-energy answer's border (thin, on top)", LOWER),
        ("black dot: a printed label, with its top reading and trust", None)])
    cv2.imwrite(str(OUT / f"{stem}_answers.png"), img)
    # walls
    t = expit(wall_odds(q))
    img = np.full((H * S, W * S, 3), 255, np.uint8)
    img[up(ink) > 0] = (205, 205, 205)
    flip = dx != dy
    vals = np.round(t * 20) / 20
    for w_, sel in ((3, has & ~flip[pk]), (9, has & flip[pk])):
        for v_ in np.unique(vals[pk][sel]) if sel.any() else []:
            m = sel & (vals[pk] == v_)
            col = (1 - v_) * np.array([235, 235, 235]) + v_ * np.array([20, 20, 200])
            img[wide(m, w_)] = col.astype(np.uint8)
    labels_on(img)
    img = legend(img, [(f"#{incident}: wall probability, the stage's walls ({q}): white 0 .. red 1", None),
                       ("thin: every neighbour pair; wide: pairs drawn differently by the solve and the lower answer",
                        None), ("grey: page ink; black dot: a printed label", None)])
    cv2.imwrite(str(OUT / f"{stem}_walls.png"), img)
    print("wrote", OUT / f"{stem}_answers.png", OUT / f"{stem}_walls.png", f"crop {c0} {r0} {c1} {r1}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("id", type=int)
    ap.add_argument("--window", type=int, nargs=4, metavar=("C0", "R0", "C1", "R1"))
    args = ap.parse_args()
    main(args.id, args.window)
