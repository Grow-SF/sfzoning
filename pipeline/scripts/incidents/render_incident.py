"""Pictures of a census incident, at 2x the grid (each grid px = 2 x 2 image px), cropped to its changed set D plus a
margin, or to a map-ft window.

    uv run python scripts/incidents/render_incident.py ID [--other RUN] [--window X0 Y0 X1 Y1] [--walls-b DIR]

  <ID>_lines.png   page ink grey; the table's district borders (green, widest), the run's (blue, middle) and the
                   other run's (red, thin, on top; default J7, `import_j7.py`); the incident's lots tinted yellow, the
                   rest of D orange; each label in the crop as a black dot with its top reading and trust
  <ID>_walls.png   every neighbour pair's border coloured by its wall probability under the run's walls for the
                   incident's quarter (white 0 .. red 1); with --walls-b, by the change from those walls to DIR's
                   (blue lower, red higher, +-0.5); the pairs whose drawn state differs between the run and the other
                   run drawn wide
Written into CENSUS_DIR/pictures/ (suffix _window with --window).  Reads only."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.special import expit

from common import CENSUS_DIR, changed_set, load_costs, run_codes
from sfzmaps.solver import grid, runs
from sfzmaps.solver.city import city
from sfzmaps.solver.codes import table_vocabulary
from sfzmaps.solver.labels import readings
from sfzmaps.solver.trust import load_prints

S = 2               # image px per grid px
MARGIN = 60         # grid px


def up(m):
    return np.kron(m.astype(np.uint8), np.ones((S, S), np.uint8))


def wide(m, w):
    u = up(m)
    return cv2.dilate(u, np.ones((w, w), np.uint8)) > 0 if w > 1 else u > 0


def legend(img, lines):
    """a white strip above the picture with one line of text per entry: (text, colour swatch or None)"""
    strip = np.full((22 * len(lines) + 8, img.shape[1], 3), 255, np.uint8)
    for i, (text, col) in enumerate(lines):
        y, x = 20 + 22 * i, 6
        if col is not None:
            cv2.rectangle(strip, (x, y - 12), (x + 22, y + 2), col, -1)
            x += 30
        cv2.putText(strip, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([strip, img])


def main(incident: int, other: str, window, walls_b) -> None:
    doc = json.loads((CENSUS_DIR / "incidents.json").read_text())
    rec = next(t for t in doc["list"] if t["id"] == incident)
    run, q = doc["run"], rec["quarter"]
    C = city()
    a, b = C.a, C.b
    codes, codes_o = run_codes(run), run_codes(other)
    both = C.known[a] & C.known[b]
    tline = both & (C.table[a] != C.table[b])
    d_run, d_other = codes[a] != codes[b], codes_o[a] != codes_o[b]
    I = np.array(rec["lots"])
    D = changed_set(load_costs(run, q), I)
    lot, grown = C.lot_raster, C.grown_raster
    if window:
        x0, y0, x1, y1 = window
        c0, r0 = np.floor(grid.to_grid([x0, y1])).astype(int)
        c1, r1 = np.ceil(grid.to_grid([x1, y0])).astype(int)
        suffix = "_window"
    else:
        inD = np.zeros(C.n + 1, bool)
        inD[D] = True
        rr, cc = np.nonzero(inD[np.where(lot >= 0, lot, C.n)])
        r0, r1 = max(rr.min() - MARGIN, 0), min(rr.max() + MARGIN, grid.HEIGHT)
        c0, c1 = max(cc.min() - MARGIN, 0), min(cc.max() + MARGIN, grid.WIDTH)
        suffix = ""
    ink = C.scan()[r0:r1, c0:c1]
    L, G = lot[r0:r1, c0:c1], grown[r0:r1, c0:c1]
    H, W = L.shape
    keys = a.astype(np.int64) * C.n + b
    pair_px = np.full((H, W), -1, np.int64)       # one pair per px (the last written): enough to colour a line
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
                x, y = int((c - c0) * S), int((r - r0) * S)
                cv2.circle(img, (x, y), 4, (0, 0, 0), -1)
                txt = f"{top[j][0]} {top[j][1]:.2f}"
                for col, th in (((255, 255, 255), 3), ((0, 0, 0), 1)):
                    cv2.putText(img, txt, (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)

    out = CENSUS_DIR / "pictures"
    out.mkdir(parents=True, exist_ok=True)
    # picture 1: the lines
    img = np.full((H * S, W * S, 3), 255, np.uint8)
    inI, inD = np.zeros(C.n + 1, bool), np.zeros(C.n + 1, bool)
    inI[I], inD[D] = True, True
    Ls = np.where(L >= 0, L, C.n)
    img[up(inI[Ls]) > 0] = (150, 235, 255)             # BGR light yellow
    img[up(inD[Ls] & ~inI[Ls]) > 0] = (150, 200, 255)  # light orange
    g = up(ink) > 0
    img[g] = (img[g] * 0.35 + np.array([110, 110, 110]) * 0.65).astype(np.uint8)
    TABLE, RUN, OTHER = (60, 180, 60), (220, 120, 20), (40, 30, 220)
    img[wide(has & tline[pk], 9)] = TABLE
    img[wide(has & d_run[pk], 5)] = RUN
    img[wide(has & d_other[pk], 2)] = OTHER
    labels_on(img)
    img = legend(img, [
        (f"#{incident}: {rec['lots_count']} lots, table {rec['table']}, solved {rec['solved']}; quarter {q}", None),
        (f"2x the grid (1 image px = 2 ft); crop {W * 4} x {H * 4} ft; grey: page ink", None),
        ("tint: yellow the incident's lots, orange the rest of its changed set D", None),
        ("table district border (widest)", TABLE), (f"{run}'s border (middle)", RUN),
        (f"{other}'s border (thin, on top)", OTHER),
        ("black dot: a printed label, with its top reading and trust", None)])
    cv2.imwrite(str(out / f"{incident}_lines{suffix}.png"), img)
    # picture 2: the walls
    lo_a = np.load(runs.WALLS_DIR / f"walls_{q}.npz")["lo_ab"].astype(float)
    if walls_b:
        v = expit(np.load(Path(walls_b) / f"walls_{q}.npz")["lo_ab"].astype(float)) - expit(lo_a)
        t = np.clip(v / 0.5, -1, 1)
        title = f"change in wall probability: {walls_b} minus the stage's walls ({q}); blue lower, red higher"
    else:
        t = expit(lo_a)
        title = f"wall probability, the stage's walls ({q}): white 0 .. red 1"
    img = np.full((H * S, W * S, 3), 255, np.uint8)
    img[up(ink) > 0] = (205, 205, 205)
    flip = d_run != d_other
    vals = np.round(t * 20) / 20
    for w_, sel in ((3, has & ~flip[pk]), (9, has & flip[pk])):
        for v_ in np.unique(vals[pk][sel]) if sel.any() else []:
            m = sel & (vals[pk] == v_)
            if walls_b:
                col = ((1 - v_) * np.array([235, 235, 235]) + v_ * np.array([20, 20, 200]) if v_ >= 0 else
                       (1 + v_) * np.array([235, 235, 235]) - v_ * np.array([200, 60, 20]))
            else:
                col = (1 - v_) * np.array([235, 235, 235]) + v_ * np.array([20, 20, 200])
            img[wide(m, w_)] = col.astype(np.uint8)
    labels_on(img)
    img = legend(img, [(f"#{incident}: {title}", None),
                       (f"thin: every neighbour pair; wide: pairs drawn differently by {run} and {other}", None),
                       ("grey: page ink; black dot: a printed label", None)])
    cv2.imwrite(str(out / f"{incident}_walls{suffix}.png"), img)
    print("wrote", out / f"{incident}_lines{suffix}.png", out / f"{incident}_walls{suffix}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("id", type=int)
    ap.add_argument("--other", default="j7", help="the run drawn thin on top (default J7)")
    ap.add_argument("--window", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    ap.add_argument("--walls-b", type=Path, help="colour the walls picture by the change to this walls folder")
    args = ap.parse_args()
    main(args.id, args.other, args.window, args.walls_b)
