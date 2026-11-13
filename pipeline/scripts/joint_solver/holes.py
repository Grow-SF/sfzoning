"""Why each hole in a run's district lines is there (measures.py defines a hole), and the data errors among them.

Per hole: the solve's wall chance there; the raw scan's ink on the table's line, beside it, and as a heavy band; the
wrongly coded lots touching it (how many, and the size of the wrong area they belong to), the lot's width and how much
of it is inked; the labels on those lots; and how the energy would change if they took their table codes.
Causes, in this order:
  BIG AREA                      the hole is the edge of a wrong area of 20+ lots
  NOT PRINTED (offset)          little ink on the table's line (offset: the ink sits beside it)
  SLIVER UNDER THE BAND         read as no wall, and the wrong lot is at most 16 ft wide: it lies under the line
  PRINTED ONE LOT OVER          read as no wall, the heavy band sits 0.4+ of a lot width into the wrong lot
  ON THE LINE, READ AS NO WALL  read as no wall, ink on the line
  LABEL PULL                    read as a wall, a label on the patch reads the run's code
  CHEAPER STEP                  read as a wall, no label pull: the energy prefers the step
Data errors: a sliver, one-lot-over or on-the-line hole whose wrong lot is at least half inked.  The printed line is
drawn over the parcel and the table puts the parcel on the other side; nothing a reader can do about it.  They are
listed for publishing and left out of the scores."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .borders import border_paths
from .data import FT_PER_PX, OUT, QUARTERS, RUNS, city, load_run
from .measures import DATA_ERRORS, Table, holes

DATA_ERRORS_CSV = OUT / "data-errors-2000-digital.csv"
DATA_ERROR_CAUSES = {"SLIVER UNDER THE BAND", "PRINTED ONE LOT OVER", "ON THE LINE, READ AS NO WALL"}


def _ink_per_border(M):
    """per border: share of its points with ink within 8 ft of the expected line, with a band beside it (12-60 ft
    away) while the line itself is blank, and with a heavy band (16 of 20 ft inked) within 8 ft"""
    Z = border_paths()
    k, prof, J = Z["k"].astype(np.int64), Z["prof"], int(Z["J"])
    D = 2 * J + 1
    cs = np.c_[np.zeros(len(prof)), np.cumsum(prof, 1)]
    on = (cs[:, J + 3] - cs[:, J - 2]) > 0
    win = cs[:, np.arange(5, D + 1)] - cs[:, np.arange(0, D - 4)]          # 5-px windows
    centre = np.arange(0, D - 4) + 2 - J
    beside = ~on & (win[:, (np.abs(centre) >= 3) & (np.abs(centre) <= 15)] >= 4).any(1)
    heavy = (win[:, np.abs(centre) <= 2] >= 4).any(1)
    n = np.maximum(np.bincount(k, minlength=M), 1)
    return (np.bincount(k, on, minlength=M) / n, np.bincount(k, beside, minlength=M) / n,
            np.bincount(k, heavy, minlength=M) / n, k, win, centre)


def causes(run):
    """one row per hole of the run (data errors not excluded); needs the run's costs (solve --keep-costs)"""
    import shapely
    C = city()
    a, b, L, M = C.a, C.b, C.border_ft, len(C.a)
    T = Table(np.zeros(C.n, bool))
    codes = load_run(run)
    drawn = codes[a] != codes[b]
    hole_list, _, _ = holes(T, drawn)
    ink_on, ink_off, heavy_on, k, win, centre = _ink_per_border(M)
    wrong = C.scored & C.known & (codes != C.table)
    key = np.char.add(np.char.add(C.table, "|"), codes)
    wm = wrong[a] & wrong[b] & (key[a] == key[b])
    _, wcomp = connected_components(coo_matrix((np.ones(int(wm.sum())), (a[wm], b[wm])), shape=(C.n, C.n)),
                                    directed=False)
    wsize = np.bincount(wcomp[wrong], minlength=C.n)
    costs = {q: np.load(RUNS / f"{run}-{q}-costs.npz", allow_pickle=True) for q in QUARTERS}
    lot_ink = C.lot_ink()
    rows = []
    for ks in hole_list:
        q = QUARTERS[np.bincount(C.quarter[a[ks]]).argmax()]
        D = costs[q]
        U, odds, p, tcodes = D["U"].astype(float), D["odds"].astype(float), D["codes"], D["table_codes"]
        names = D["names"].astype(str).tolist()
        wall = float(np.mean(1 / (1 + np.exp(-odds[ks]))))
        on = float(np.average(ink_on[ks], weights=L[ks]))
        off = float(np.average(ink_off[ks], weights=L[ks]))
        side = np.unique(np.r_[a[ks], b[ks]])
        wl = side[wrong[side]]
        area = int(wsize[wcomp[wl]].max()) if len(wl) else 0
        patch = np.flatnonzero(np.isin(wcomp, wcomp[wl]) & wrong) if len(wl) else np.zeros(0, int)
        patch = patch if area < 20 else wl
        labs = [(names[D["code"][r]], round(float(D["trust"][r]), 2)) for r in np.flatnonzero(np.isin(D["lot"], patch))]
        pull = any(c in set(codes[patch].tolist()) for c, _ in labs)
        qq = p.copy()
        fix = patch[tcodes[patch] >= 0]
        qq[fix] = tcodes[fix]
        dU = float((U[fix, qq[fix]] - U[fix, p[fix]]).sum())
        dD = float((np.maximum(-odds, 0) * ((qq[a] != qq[b]).astype(float) - (p[a] != p[b]))).sum())
        dS = float((np.maximum(odds, 0) * ((qq[a] == qq[b]).astype(float) - (p[a] == p[b]))).sum())
        pts = np.flatnonzero(np.isin(k, ks))             # the heavy band's centre, from the table's line toward the
        best = np.argmax(win - 1e-3 * np.abs(centre)[None, :], axis=1)[pts]     #   wrong lot (ft)
        off_ft = centre[best] * FT_PER_PX
        band_off = float(np.median(np.where(np.isin(b[k[pts]], wl), off_ft, -off_ft))) if len(pts) else float("nan")
        if len(wl):
            xy = shapely.get_coordinates(shapely.minimum_rotated_rectangle(C.lots.geometry.values[wl[0]]))[:4]
            lot_w = float(np.hypot(*(np.diff(np.r_[xy, xy[:1]], axis=0)).T).min())
        else:
            lot_w = float("nan")
        inked = float(lot_ink[wl].mean()) if len(wl) else float("nan")
        if area >= 20:
            cause = "BIG AREA"
        elif on < 0.3:
            cause = "NOT PRINTED (offset)" if off >= 0.5 else "NOT PRINTED"
        elif wall < 0.5 and lot_w <= 16:
            cause = "SLIVER UNDER THE BAND"
        elif wall < 0.5 and band_off >= 0.4 * lot_w:
            cause = "PRINTED ONE LOT OVER"
        elif wall < 0.5:
            cause = "ON THE LINE, READ AS NO WALL"
        elif pull:
            cause = "LABEL PULL"
        else:
            cause = "CHEAPER STEP"
        rows.append(dict(cause=cause, q=q, ft=float(L[ks].sum()), wall=wall, ink=on, beside=off,
                         heavy=float(np.average(heavy_on[ks], weights=L[ks])), band_off=band_off, lot_w=lot_w,
                         lot_ink=inked, area=area, labels=labs[:5], dU=dU, dD=dD, dS=dS,
                         table=f"{C.table[a[ks[0]]]}|{C.table[b[ks[0]]]}", got=f"{codes[a[ks[0]]]}|{codes[b[ks[0]]]}",
                         lot=int(wl[0]) if len(wl) else int(a[ks[0]]), wrong_lots=[int(x) for x in wl],
                         borders=[int(x) for x in ks]))
    return rows


def report_causes(run, picture=None, save_data_errors=False):
    rows = causes(run)
    print(f"holes: {len(rows)}")
    for c, n in Counter(r["cause"] for r in rows).most_common():
        print(f"  {c:28s} {n}")
    for r in sorted(rows, key=lambda r: (r["cause"], -r["ft"])):
        print(f"{r['cause'][:14]:14s} {r['q']} {r['table']:16s} {r['ft']:4.0f} ft | wall {r['wall']:.2f} heavy "
              f"{r['heavy']:.2f} ink {r['ink']:.2f} | band {r['band_off']:+.0f} ft into a {r['lot_w']:.0f}-ft lot, "
              f"{r['lot_ink']:.0%} inked | wrong lots {len(r['wrong_lots'])} (area {r['area']}); energy to fix: "
              f"labels {r['dU']:+.1f}, differ {r['dD']:+.1f}, same {r['dS']:+.1f}; labels {r['labels']}")
    errors = [r for r in rows if r["cause"] in DATA_ERROR_CAUSES and r["lot_ink"] >= 0.5]
    print(f"data errors: {len(errors)} holes, {len({lot for r in errors for lot in r['wrong_lots']})} parcels")
    if save_data_errors:
        _save_data_errors(errors, load_run(run))
    if picture:
        _picture(rows, load_run(run), picture)


def _save_data_errors(errors, codes):
    C = city()
    lonlat = C.lots.to_crs(4326).geometry.representative_point()
    with open(DATA_ERRORS_CSV, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["blklot", "table_code", "drawing_code", "lot_width_ft", "lot_inked_share", "lon", "lat", "note"])
        for r in errors:
            for lot in r["wrong_lots"]:
                w.writerow([C.lots["blklot"].values[lot], C.table[lot], codes[lot], round(r["lot_w"]),
                            round(r["lot_ink"], 2), round(lonlat.x.values[lot], 6), round(lonlat.y.values[lot], 6),
                            "the printed district line is drawn over this parcel; the table puts the parcel on the "
                            "other side"])
    lots = sorted({lot for r in errors for lot in r["wrong_lots"]})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_ERRORS, lots=np.array(lots, int))
    print(f"saved {DATA_ERRORS_CSV} and {DATA_ERRORS} ({len(lots)} parcels)")


def _picture(rows, codes, out, F=5):
    """per hole, 240 ft around the wrong lot: the raw scan, table lines black (the hole red), the run's borders blue,
    the wrong lot outlined magenta; grouped by cause"""
    import cv2
    C = city()
    a, b, N, M = C.a, C.b, C.n, len(C.a)
    T = Table(np.zeros(C.n, bool))
    drawn = codes[a] != codes[b]
    grown = C.grown_raster
    Hh, Ww = grown.shape
    ink, row0, col0 = C.scan()
    keys = a.astype(np.int64) * N + b
    panels = []
    for r in sorted(rows, key=lambda r: (r["cause"], -r["ft"])):
        c0, r0 = C.to_grid(C.centroid[r["lot"]] - 120)
        c0, r0 = max(int(c0), 0), max(int(r0), 0)
        c1, r1 = min(c0 + 60, Ww), min(r0 + 60, Hh)
        pr, pc = row0 - np.arange(r0, r1), np.arange(c0, c1) - col0
        page = np.full((r1 - r0, c1 - c0), 255, np.uint8)
        okr, okc = (pr >= 0) & (pr < ink.shape[0]), (pc >= 0) & (pc < ink.shape[1])
        page[np.ix_(np.flatnonzero(okr), np.flatnonzero(okc))] = np.where(ink[np.ix_(pr[okr], pc[okc])], 0, 255)
        faint = cv2.cvtColor((255 - (255 - page) * 0.4).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        big = cv2.resize(faint, None, fx=F, fy=F, interpolation=cv2.INTER_NEAREST)
        gw = grown[r0:r1, c0:c1]
        hole = set(r["borders"])
        for A_, B_ in ((gw[:, :-1], gw[:, 1:]), (gw[:-1, :], gw[1:, :])):
            m = (A_ != B_) & (A_ >= 0) & (B_ >= 0)
            lo, hi = np.minimum(A_[m], B_[m]).astype(np.int64), np.maximum(A_[m], B_[m]).astype(np.int64)
            kk = np.clip(np.searchsorted(keys, lo * N + hi), 0, M - 1)
            hit = keys[kk] == lo * N + hi
            ys, xs = np.nonzero(m)
            for y, x, k_ in zip(ys[hit], xs[hit], kk[hit]):
                if drawn[k_]:
                    cv2.rectangle(big, (x * F - 1, y * F - 1), (x * F + F, y * F + F), (235, 140, 0), 2)
                if T.line[k_]:
                    col = (30, 30, 230) if k_ in hole else (20, 20, 20)
                    cv2.rectangle(big, (x * F + 1, y * F + 1), (x * F + F - 2, y * F + F - 2), col, -1)
        outline = cv2.resize((gw == r["lot"]).astype(np.uint8), None, fx=F, fy=F, interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(big, cnts, -1, (200, 0, 200), 2)
        big = big[::-1].copy()
        cap = np.full((34, big.shape[1], 3), 255, np.uint8)
        cv2.putText(cap, r["cause"][:28], (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 150), 1)
        text = f"{r['table']} wall {r['wall']:.2f} lot {r['lot_w']:.0f}ft band {r['band_off']:+.0f}ft"
        cv2.putText(cap, text, (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        panels.append(np.vstack([cap, big]))
    if not panels:
        return
    while len(panels) % 4:
        panels.append(np.full_like(panels[0], 255))
    img = np.vstack([np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)])
    top = np.full((24, img.shape[1], 3), 255, np.uint8)
    cv2.putText(top, "Every hole, by cause: table lines black (the hole red), the run's borders blue, the wrongly "
                "coded lot outlined magenta; 240 ft", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack([top, img]))
    print(f"wrote {out}")
