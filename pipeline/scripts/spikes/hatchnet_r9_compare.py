"""Spike: ruling R9 (a place inside several sheets' neatlines belongs to the sheet that prints it) measured on the 1921
hatchnet data, before and after.  Read-only on the data: it reads data/solver/experiments/hatch-cnn/data/1921-before-r9
(a plain copy of the data made before the R9 rerun) and data/1921 (after), and writes only hatch-cnn/look/r9/
(compare.json and the corner pictures).  CPU, one maths thread per worker.  $0.

    uv run python scripts/spikes/hatchnet_r9_compare.py [--workers 6]

What it measures, before and after:
  - each lot's sheets: its owned px per sheet (the `lot` rasters); the lots whose majority sheet (the one holding most of
    its px) changed, per (before, after) pair; the lots whose centre_sheet changed, per pair;
  - the Task 4 reviewer's blank-pooled measure (task-4-review.md, Important 1), as its scratch scripts computed it: for
    each sheet a lot lies on, the share of its block's px (the lot table's `block`, the assessor block) within R_NEAR px
    of ink (scan < 0.5); a lot is blank when that share is 0 on every sheet it is pooled on;
  - the two random lots on the join corner (4097020, 4097049B);
  - for the read lots that were blank-pooled before, whether centre_sheet now agrees with read_sheet.
Pictures: the two join corners (094308402's "Join Sec. 9", 094254223's top left), before and after, each on the blank
sheet and on the sheet that prints it, at the scan's own px: the owned lots outlined (red: the corner's blank-pooled lots;
blue: the others), px the sheet does not own darkened.
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from sfzmaps.solver.hatchnet import data

AFTER = data.data_dir("1921")
BEFORE = AFTER.parent / "1921-before-r9"
OUT = data.LOOK / "r9"
R_NEAR = 8                      # the reviewer's: a block px is "near ink" when ink (scan < 0.5) lies within 8 px
CORNERS = {"20180102094308402": "20180102094317284",       # the blank sheet: the sheet that prints its corner
           "20180102094254223": "20180102094149543"}
RANDOM = ("4097020", "4097049B")
PAD = 40                        # px around a corner's lots in the pictures


def _init():
    cv2.setNumThreads(1)


def sheet_job(job) -> dict:
    """One sheet: each lot's owned px before and after, and per assessor block its px and px near ink."""
    sheet, bcode, nb, n = job
    rb, ra = data.read_rasters(BEFORE / f"{sheet}.npz"), data.read_rasters(AFTER / f"{sheet}.npz")
    if not np.array_equal(rb.ink, ra.ink):
        raise ValueError(f"{sheet}: the scan differs before and after")
    near = ndimage.distance_transform_edt(ra.ink >= 0.5) <= R_NEAR
    out = {"sheet": sheet}
    for tag, r in (("before", rb), ("after", ra)):
        on = r.lot >= 0
        lid = r.lot[on].astype(np.int64)
        b = bcode[lid]
        out[tag] = {"lot_px": np.bincount(lid, minlength=n).astype(np.int64),
                    "bpx": np.bincount(b, minlength=nb).astype(np.int64),
                    "bnear": np.bincount(b, weights=near[on], minlength=nb)}
    return out


def blank_lots(px: np.ndarray, bpx: np.ndarray, bnear: np.ndarray, bcode: np.ndarray) -> np.ndarray:
    """The reviewer's measure: lots with px whose block has no px near ink on every sheet they are pooled on.  px:
    (sheets, lots); bpx, bnear: (sheets, blocks)."""
    share = np.where(bpx > 0, bnear / np.maximum(bpx, 1), 0.0)[:, bcode]           # (sheets, lots)
    on = px > 0
    max_near = np.where(on, share, -1.0).max(axis=0)
    return on.any(axis=0) & (max_near == 0)


def pooled_on(px: np.ndarray, sheets: list[str], i: int) -> str:
    return "+".join(s[-9:] for s, v in zip(sheets, px[:, i]) if v > 0) or "-"


def counts(values) -> dict:
    s = pd.Series(list(values)).value_counts()
    return {str(k): int(v) for k, v in s.sort_values(ascending=False).items()}


def outlines(lot: np.ndarray) -> np.ndarray:
    """The px of each lot that touch another lot or no lot (4-neighbours)."""
    on = lot >= 0
    edge = np.zeros(lot.shape, bool)
    edge[:-1] |= lot[:-1] != lot[1:]
    edge[1:] |= lot[1:] != lot[:-1]
    edge[:, :-1] |= lot[:, :-1] != lot[:, 1:]
    edge[:, 1:] |= lot[:, 1:] != lot[:, :-1]
    return edge & on


def panel(r: data.SheetRasters, box, corner: np.ndarray, title: list[str]) -> np.ndarray:
    """The scan in `box` (y0, y1, x0, x1) at its own px: px the sheet does not own darkened, owned lots outlined."""
    y0, y1, x0, x1 = box
    ink = r.ink[y0:y1, x0:x1]
    img = cv2.cvtColor(np.round(ink * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
    crop = r.crop[y0:y1, x0:x1].astype(bool)
    other = (r.block[y0:y1, x0:x1] == data.UNKNOWN) & crop
    img[other] = img[other] * 0.45 + np.array([90, 90, 90], np.float32) * 0.55
    img[~crop] *= 0.35
    img = np.round(img).astype(np.uint8)
    lot = r.lot[y0:y1, x0:x1]
    edge = outlines(lot)
    mine = np.zeros(lot.shape, bool)
    mine[lot >= 0] = corner[lot[lot >= 0]]
    img[edge & ~mine] = (255, 120, 0)
    img[edge & mine] = (0, 0, 255)
    return data._banner(img, title, 0.9)


def side_by_side(rows: list[list[np.ndarray]]) -> np.ndarray:
    h = max(p.shape[0] for row in rows for p in row)
    w = [max(row[k].shape[1] for row in rows) for k in range(len(rows[0]))]
    gap = 12
    out = []
    for row in rows:
        cells = []
        for k, p in enumerate(row):
            c = np.full((h, w[k], 3), 255, np.uint8)
            c[:p.shape[0], :p.shape[1]] = p
            cells.append(c)
            cells.append(np.full((h, gap, 3), 255, np.uint8))
        out.append(np.hstack(cells[:-1]))
        out.append(np.full((gap, out[-1].shape[1], 3), 255, np.uint8))
    return np.vstack(out[:-1])


def box_of(lot: np.ndarray, want: np.ndarray, shape) -> tuple[int, int, int, int] | None:
    on = lot >= 0
    m = np.zeros(lot.shape, bool)
    m[on] = want[lot[on]]
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    h, w = shape
    return (max(0, ys.min() - PAD), min(h, ys.max() + 1 + PAD), max(0, xs.min() - PAD), min(w, xs.max() + 1 + PAD))


def pictures(corner_lots: dict[str, np.ndarray]) -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    legend = ["red: the corner's lots blank-pooled before R9   blue: other lots", "dark: px this sheet does not own"]
    for blank_sheet, printing in CORNERS.items():
        want = corner_lots[blank_sheet]
        r = {(t, s): data.read_rasters((BEFORE if t == "before" else AFTER) / f"{s}.npz")
             for t in ("before", "after") for s in (blank_sheet, printing)}
        b_blank = box_of(r["before", blank_sheet].lot, want, r["before", blank_sheet].ink.shape)
        b_print = box_of(r["after", printing].lot, want, r["after", printing].ink.shape)
        rows = []
        for t in ("before", "after"):
            row = []
            for s, box in ((blank_sheet, b_blank), (printing, b_print)):
                rr = r[t, s]
                y0, y1, x0, x1 = box
                lots = rr.lot[y0:y1, x0:x1]
                n_corner = len(np.unique(lots[(lots >= 0)][want[lots[lots >= 0]]]))
                role = "blank corner" if s == blank_sheet else "prints the place"
                row.append(panel(rr, box, want, [f"{t.upper()} R9: {s} ({role})",
                                                  f"scan px y {y0}-{y1}, x {x0}-{x1}; red lots owned here: {n_corner}",
                                                  *legend]))
            rows.append(row)
        f = OUT / f"corner_{blank_sheet}.png"
        cv2.imwrite(str(f), side_by_side(rows))
        written.append(str(f))
    return written


def moved_blocks(workers: int) -> dict:
    """The blocks R9 gives to a sheet other than the highest-drawn one with px of them, per (R9's sheet < the top
    sheet) pair: how many, their lots, each sheet's ft/px, and the quantiles of the top sheet's mean ink over the block
    as a share of R9's sheet's (near 0: the top sheet is blank paper there; near 1: both print it)."""
    src = data.sources("1921")
    names = data.sheet_names(src)
    order = data.draw_order(src, names)
    lot, blocks = data.solver_grid(src)
    bid, n = data.block_ids(blocks)
    ink = data.block_ink(src, order, bid, n, workers)
    has = ink.px[:, 1:] > 0
    top = np.where(has.any(axis=0), len(order) - 1 - np.argmax(has[::-1], axis=0), -1)
    win = ink.owners()[1:]
    moved = (has.sum(axis=0) > 1) & (win != top)
    mean = np.where(ink.px > 0, ink.ink / np.maximum(ink.px, 1), np.nan)[:, 1:]
    on = lot >= 0
    lot_block = np.zeros(int(lot.max()) + 1, np.int64)
    lot_block[lot[on]] = bid[on]                     # each lot's block (the lots are in the filled blocks)
    lots_of_block = np.bincount(lot_block, minlength=n + 1)[1:]
    scale = {s: json.loads((AFTER / "summary.json").read_text())["sheets"][s]["ft_per_px"] for s in order}
    out = {}
    for w, t in sorted(set(zip(win[moved], top[moved]))):
        m = moved & (win == w) & (top == t)
        ratio = mean[t][m] / mean[w][m]
        out[f"{order[w]} < {order[t]}"] = {
            "blocks": int(m.sum()), "lots": int(lots_of_block[m].sum()),
            "ft_per_px": [round(scale[order[w]], 2), round(scale[order[t]], 2)],
            "top_over_r9_ink_q10_q50_q90": [round(float(v), 3) for v in np.quantile(ratio, [0.1, 0.5, 0.9])],
            "top_ink_q50": round(float(np.median(mean[t][m])), 3), "r9_ink_q50": round(float(np.median(mean[w][m])), 3)}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["lots"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    tb = pd.read_parquet(BEFORE / "lots.parquet")
    ta = pd.read_parquet(AFTER / "lots.parquet")
    if not (tb["lot"].equals(ta["lot"]) and tb["blklot"].equals(ta["blklot"]) and tb["block"].equals(ta["block"])):
        raise ValueError("the lot tables' rows differ")
    n = len(ta)
    bcode, buniq = pd.factorize(ta["block"])
    bcode = bcode.astype(np.int64)
    nb = len(buniq)
    sheets = sorted(f.stem for f in AFTER.glob("*.npz"))
    if sheets != sorted(f.stem for f in BEFORE.glob("*.npz")):
        raise ValueError("the sheets differ")
    jobs = [(s, bcode, nb, n) for s in sheets]
    with get_context("spawn").Pool(min(a.workers, len(jobs)), initializer=_init) as pool:
        done = {d["sheet"]: d for d in pool.imap_unordered(sheet_job, jobs)}
    m = {t: {k: np.stack([done[s][t][k] for s in sheets]) for k in ("lot_px", "bpx", "bnear")}
         for t in ("before", "after")}
    px_b, px_a = m["before"]["lot_px"], m["after"]["lot_px"]
    has_b, has_a = px_b.sum(axis=0) > 0, px_a.sum(axis=0) > 0
    maj_b = np.where(has_b, np.argmax(px_b, axis=0), -1)
    maj_a = np.where(has_a, np.argmax(px_a, axis=0), -1)
    name = np.array([s[-9:] for s in sheets] + ["(none)"], dtype=object)
    moved = maj_b != maj_a
    res: dict = {"lots": n, "lots_with_px": {"before": int(has_b.sum()), "after": int(has_a.sum())},
                 "lots_any_px_changed": int((px_b != px_a).any(axis=0).sum()),
                 "lots_sheet_set_changed": int(((px_b > 0) != (px_a > 0)).any(axis=0).sum()),
                 "lots_majority_sheet_changed": int(moved.sum()),
                 "majority_sheet_changed_by_pair": counts(f"{name[i]} -> {name[j]}"
                                                          for i, j in zip(maj_b[moved], maj_a[moved])),
                 "majority_sheet_changed_by_quarter": counts(ta["quarter"].to_numpy()[moved])}
    cb, ca = tb["centre_sheet"].to_numpy(), ta["centre_sheet"].to_numpy()
    cm = cb != ca
    res["lots_centre_sheet_changed"] = int(cm.sum())
    res["centre_sheet_changed_by_pair"] = counts(f"{(x or '(none)')[-9:]} -> {(y or '(none)')[-9:]}"
                                                 for x, y in zip(cb[cm], ca[cm]))
    # the reviewer's blank-pooled measure
    blank = {t: blank_lots(m[t]["lot_px"], m[t]["bpx"], m[t]["bnear"], bcode) for t in ("before", "after")}
    res["blank_pooled"] = {}
    for t in ("before", "after"):
        px = m[t]["lot_px"]
        ids = np.flatnonzero(blank[t])
        res["blank_pooled"][t] = {"lots": int(len(ids)),
                                  "by_pooled_sheets": counts(pooled_on(px, sheets, i) for i in ids)}
    corner_lots, res["join_corners"] = {}, {}
    for s in CORNERS:
        k = sheets.index(s)
        c = blank["before"] & (px_b[k] > 0) & ((px_b > 0).sum(axis=0) == 1)       # blank-pooled on this sheet alone
        corner_lots[s] = c
        ids = np.flatnonzero(c)
        read = ids[ta["read_code"].to_numpy()[ids] != ""]
        res["join_corners"][s] = {
            "printing_sheet": CORNERS[s], "lots_blank_before": int(len(ids)),
            "still_blank_after": int(blank["after"][ids].sum()),
            "still_blank_after_blklots": [str(b) for b in ta["blklot"].to_numpy()[ids[blank["after"][ids]]]],
            "after_pooled_on": counts(pooled_on(px_a, sheets, i) for i in ids),
            "read_lots": int(len(read)),
            "read_sheet_before": counts(tb["read_sheet"].to_numpy()[read]),
            "read_sheet_after": counts(ta["read_sheet"].to_numpy()[read]),
            "centre_sheet_before": counts(cb[read]), "centre_sheet_after": counts(ca[read]),
            "centre_equals_read_before": int((cb[read] == tb["read_sheet"].to_numpy()[read]).sum()),
            "centre_equals_read_after": int((ca[read] == ta["read_sheet"].to_numpy()[read]).sum()),
            "read_pooled_on_read_sheet_after": int(sum(px_a[sheets.index(rs), i] > 0 for i, rs in
                                                       zip(read, ta["read_sheet"].to_numpy()[read]))),
            "answered": int((ta["answer"].to_numpy()[ids] != "").sum())}
    both = corner_lots[list(CORNERS)[0]] | corner_lots[list(CORNERS)[1]]
    other = blank["before"] & ~both
    res["blank_pooled"]["join_corners_before"] = int(both.sum())
    res["blank_pooled"]["join_corners_after"] = int((blank["after"] & both).sum())
    res["blank_pooled"]["other_before"] = int(other.sum())
    res["blank_pooled"]["other_still_blank_after"] = int((blank["after"] & other).sum())
    res["blank_pooled"]["new_after"] = int((blank["after"] & ~blank["before"]).sum())
    new = np.flatnonzero(blank["after"] & ~blank["before"])
    res["blank_pooled"]["new_after_by_pooled_sheets"] = counts(pooled_on(px_a, sheets, i) for i in new)
    res["blank_pooled"]["new_after_examples"] = [str(b) for b in ta["blklot"].to_numpy()[new[:20]]]
    # all read lots: centre_sheet against read_sheet
    rd = ta["read_code"].to_numpy() != ""
    res["read_lots_centre_equals_read"] = {"before": int((rd & (cb == tb["read_sheet"].to_numpy())).sum()),
                                           "after": int((rd & (ca == ta["read_sheet"].to_numpy())).sum()),
                                           "read_lots": int(rd.sum())}
    res["random_lots"] = {}
    for b in RANDOM:
        i = int(np.flatnonzero(ta["blklot"].to_numpy() == b)[0])
        res["random_lots"][b] = {
            "lot": i, "answer": str(ta.loc[i, "answer"]), "read": str(ta.loc[i, "read_code"]),
            "read_sheet": str(ta.loc[i, "read_sheet"]),
            "before": {"px": {sheets[k]: int(px_b[k, i]) for k in np.flatnonzero(px_b[:, i])},
                       "centre_sheet": str(cb[i]), "blank_pooled": bool(blank["before"][i])},
            "after": {"px": {sheets[k]: int(px_a[k, i]) for k in np.flatnonzero(px_a[:, i])},
                      "centre_sheet": str(ca[i]), "blank_pooled": bool(blank["after"][i])}}
    res["moved_blocks"] = moved_blocks(a.workers)
    res["pictures"] = pictures(corner_lots)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compare.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    main()
