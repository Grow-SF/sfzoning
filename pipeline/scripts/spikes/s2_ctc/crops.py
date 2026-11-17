"""Spike S2, step 1: the crops a CTC reader learns from and is scored on (survey section 4.6, step 1).

A measurement, not shipped code (brief: .superpowers/sdd/2026-11-16-search-move-and-reader-spikes/task-S2-brief.md).

    uv run python scripts/spikes/s2_ctc/crops.py          # writes data/solver/experiments/spike-s2/crops.npz + .pkl

Every crop is a word rectified along Terra's reading angle from the full-resolution page (`data/work/<series>/<sheet>/
page.png`), as the code reader straightens its labels (`codereader.labels.frame` / `sample`: an exact px copy for a
quarter turn, bilinear otherwise, white beyond the page): Terra's word rectangle (`labels.word_size`) with PAD_X word
heights either side along the line and PAD_Y above and below, then resized to HEIGHT px tall (aspect kept; INTER_AREA
when shrinking).  The crop reads left to right; its half turn (`np.rot90(crop, 2)`) is the 180-degree copy.

A slanted word's height (any angle but a quarter turn) is solved from Terra's axis box and angle (`word_rect`), and
that inversion is ill-conditioned: a loose box gives a height of 2-7 px (a quarter of the training street names under
8 px), and the crop is a sliver of one stroke.  So a slanted word's height is floored at its kind's FLOOR_QUANTILE
quantile of the training split's quarter-turn words (whose heights are their box's sides, exact), and its length then
follows from the box and that height as `word_rect` lays out a near-diagonal word (`word_geometry`).
CHOSEN AFTER SEEING RESULTS (review S2, I4): the floor was found by looking at reader C's misses on the owner's
answered labels in the first run (v1/, which cut slivers and was a tie with the generative reader on both measures),
and every reader was retrained on these crops.  v1's crops and numbers are kept in v1/.

The maps: the 2000 drawing's three scans (MAPS: the 2000 digital scan and its two earlier scans, one modern lettering),
every word Terra boxed (every kind).  Hold-outs exactly as R1's `SheetData.held` (`textnet.reader_stage`), on every map
at once through each px's map point:
  the quarter       `data.Quarter(HOLD_QUARTER)` (se: dense, the most words; the north-east's dense code lettering stays
                    in training, per "fit on dense, report held-out")
  the validation    `data.VALIDATION`
  the answered      every block the owner's answers reach (`reader_stage.held_blocks` -> `load_block_area`)
A crop is a training crop only when NO px of its page footprint is in any of the three (`data._held` over the
footprint's window).  Scoring sets:
  quarter    2000-digital-zoning words whose centre px is in the held quarter
  answered   the owner's answered labels with Terra's box (box-found), from the generative reader's own label list
             (labs.pkl, copied into inputs/), cropped the same way from their box and angle
  nontext    windows of word size on ink outside every Terra word and outside the unknown px (Sol's points within R1's
             measured radius, the other passes' boxes), at Terra's own angles: target "" (training ones weighted
             1 - Terra's measured miss rate, `miss_rate`)
Every crop records its centre's map point, quarter and the block of the lot under it (lots.npz `grown`), so the hold-
outs can be re-checked by a second route (`test_s2_ctc.py`).
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")

import dataclasses  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sfzmaps import config  # noqa: E402
from sfzmaps.solver import grid, paths, saved  # noqa: E402
from sfzmaps.solver.codereader import labels as L  # noqa: E402
from sfzmaps.solver.textnet import boxes, data, reader_stage  # noqa: E402

cv2.setNumThreads(1)

OUT = config.DATA_DIR / "solver" / "experiments" / "spike-s2"
INPUTS = OUT / "inputs"
MAPS = ("2000-digital-zoning", "1980-1990-zoning", "1990-1996-zoning")
HOLD_QUARTER = 3            # se (paths.QUARTERS)
HEIGHT = 32                 # px, every crop's height
PAD_X, PAD_Y = 0.4, 0.2     # word heights of page around Terra's word, along and across the line
MAX_WIDTH = 1024            # px; a longer crop is squeezed to it
NONTEXT_SHARE = 0.2         # non-text windows per word, per sheet
NONTEXT_TRIES = 60          # draws per wanted window
VOCAB = " &'(),-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WORKERS = 4
SEED = 20261116
FLOOR_QUANTILE = 0.1        # a slanted word's height floor: this quantile of its kind's quarter-turn training heights


def norm_text(text: str) -> str | None:
    """Terra's string as the reader learns it: upper case, zero-width spaces dropped, single spaces; None when a
    character is outside VOCAB."""
    t = " ".join(str(text).replace("​", "").upper().split())
    return t if t and all(c in VOCAB for c in t) else None


def word_geometry(box, angle: float, floor: float = 0.0) -> tuple[float, float]:
    """A word's length and height in page px: `labels.word_size`, but a slanted word's height at least `floor`, its
    length then from the box at that height (`textnet.data.word_rect`'s near-diagonal layout)."""
    length, height = L.word_size(box, angle)
    turns = angle / 90
    if turns != round(turns) and height < floor:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        a = np.deg2rad(angle)
        c, s = abs(np.cos(a)), abs(np.sin(a))
        height = float(floor)
        length = float((w - height * s) / c if c >= s else (h - height * c) / s)
        length = max(length, height)
    return float(length), float(height)


def rectify(page: np.ndarray, box, angle: float, pad_x: float = PAD_X, pad_y: float = PAD_Y, floor: float = 0.0):
    """A word's crop (uint8, HEIGHT px tall, reading left to right), its page frame (`labels.frame`), its page-px
    shape before resizing and the word's (length, height) in page px (`word_geometry` with `floor`)."""
    x0, y0, x1, y1 = box
    length, height = word_geometry(box, angle, floor)
    w = max(4, round(length + 2 * pad_x * height))
    h = max(4, round(height * (1 + 2 * pad_y)))
    a = L.frame(((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2), w, h, angle)
    raw = L.sample(page, a, (h, w), 255)
    out_w = min(MAX_WIDTH, max(4, round(w * HEIGHT / h)))
    crop = cv2.resize(raw, (out_w, HEIGHT), interpolation=cv2.INTER_AREA if h > HEIGHT else cv2.INTER_LINEAR)
    return crop, a, (h, w), (float(length), float(height))


def _footprint(a, shape, page_shape):
    return L._footprint(a, shape, 1, page_shape)


def _areas(sd, held, x0, y0, x1, y1):
    """Which held areas (in `held`'s order) the page window touches."""
    return [bool(data.in_area(sd, area, y0, x0, y1 - y0, x1 - x0).any()) for area in held]


def _centre_info(sd, cx: float, cy: float) -> dict:
    """The map point of a page point (its quarter and lot block are added in the main process, `add_blocks`)."""
    xy = data._world_at(sd.world, np.array([int(round(cy))]), np.array([int(round(cx))]))
    return {"mx": float(xy[0][0, 0]), "my": float(xy[1][0, 0])}


def add_blocks(df: pd.DataFrame) -> None:
    """Each crop's centre quarter (`grid.quarter_of`) and the block of the lot under it (lots.npz `grown`, "" off the
    lots), from its map point; crops without one (answered labels: their own page point) are found here too."""
    with np.load(paths.LOTS_DIR / "lots.npz") as z:
        grown = z["grown"]
    lot_block = pd.read_parquet(paths.LOTS_DIR / "table.parquet", columns=["block_num"])["block_num"].astype(
        str).to_numpy()
    xy = df[["mx", "my"]].to_numpy(float)
    df["quarter"] = grid.quarter_of(xy)
    cr = np.floor(grid.to_grid(xy)).astype(np.int64)
    on = (cr[:, 1] >= 0) & (cr[:, 1] < grown.shape[0]) & (cr[:, 0] >= 0) & (cr[:, 0] < grown.shape[1])
    lot = np.full(len(df), -1, np.int64)
    lot[on] = grown[cr[on, 1], cr[on, 0]]
    df["block"] = np.where(lot >= 0, lot_block[np.maximum(lot, 0)], "")


_W = {}


def _init(held, radius, miss_weight, floors):
    cv2.setNumThreads(1)
    _W.update(held=held, radius=radius, miss_weight=miss_weight, floors=floors)


def quarter_turn_heights(job) -> list[tuple[str, float]]:
    """(kind, height) of one sheet's quarter-turn words that are training words (no footprint px held out)."""
    series, sheet, terra, others = job
    held = _W["held"]
    sd = data.sheet_data(series, sheet, hold_out_validation=False, terra=terra, others=others, placement=True)
    out = []
    for b in boxes.terra_boxes(series, sheet, terra):
        angle = float(b.get("angle_deg") or 0.0)
        if angle / 90 != round(angle / 90) or norm_text(b["text"]) is None:
            continue
        length, height = L.word_size(b["box"], angle)
        w, h = max(4, round(length + 2 * PAD_X * height)), max(4, round(height * (1 + 2 * PAD_Y)))
        x0, y0, x1, y1 = b["box"]
        a = L.frame(((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2), w, h, angle)
        fx0, fy0, fx1, fy1 = _footprint(a, (h, w), (sd.height, sd.width))
        if not any(_areas(sd, held, fx0, fy0, fx1, fy1)):
            out.append((b["kind"], float(height)))
    return out


def sheet_crops(job):
    """Every crop of one sheet: its words (train / quarter / neither) and its non-text windows."""
    series, sheet, terra, others = job
    held, radius, miss_weight, floors = _W["held"], _W["radius"], _W["miss_weight"], _W["floors"]
    t0 = time.perf_counter()
    sd = data.sheet_data(series, sheet, hold_out_validation=False, terra=terra, others=others, placement=True,
                         point_radius=radius)
    sd = dataclasses.replace(sd, held=held)
    page = data.load_page(sheet, series)
    found = boxes.terra_boxes(series, sheet, terra)
    rng = np.random.default_rng([SEED, hash_name(sheet)])
    crops, meta = [], []
    for j, b in enumerate(found):
        text = norm_text(b["text"])
        angle = float(b.get("angle_deg") or 0.0)
        crop, a, shape, (length, height) = rectify(page, b["box"], angle, floor=floors.get(b["kind"], 0.0))
        x0, y0, x1, y1 = _footprint(a, shape, page.shape)
        touch = _areas(sd, held, x0, y0, x1, y1)
        cx, cy = (b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2
        info = _centre_info(sd, cx, cy)
        in_q = bool(data.in_area(sd, held[0], int(round(cy)), int(round(cx)), 1, 1)[0, 0])
        split = "train" if not any(touch) else ("quarter" if in_q and series == paths.SERIES else "")
        crops.append(crop)
        meta.append({"series": series, "sheet": sheet, "word": j, "kind": b["kind"], "terra": b["text"], "text": text,
                     "angle": angle, "direction": b["direction"], "box": list(map(float, b["box"])),
                     "length": length, "height": height, "split": split if text is not None else "",
                     "touch_quarter": touch[0], "touch_validation": touch[1], "touch_blocks": touch[2],
                     "weight": 1.0, **info})
    # non-text windows: the size and angle of one of this sheet's words, centred on ink, touching no word and no
    # unknown px; a training window touches no held area, a quarter window (2000 only) has its centre in the quarter
    want = round(NONTEXT_SHARE * len(found))
    ink_rows = np.flatnonzero(np.unpackbits(sd.ink_bits, axis=1, count=sd.width).any(axis=1))
    made = 0
    for _ in range(want * NONTEXT_TRIES):
        if made >= want or not len(found) or not len(ink_rows):
            break
        b = found[int(rng.integers(len(found)))]
        angle = float(b.get("angle_deg") or 0.0)
        bw, bh = b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]
        y = int(ink_rows[int(rng.integers(len(ink_rows)))])
        row = data._ink_window(sd, y, 0, 1, sd.width)[0]
        xs = np.flatnonzero(row)
        if not len(xs):
            continue
        x = int(xs[int(rng.integers(len(xs)))])
        box = (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2)
        crop, a, shape, (length, height) = rectify(page, box, angle, floor=floors.get(b["kind"], 0.0))
        x0, y0, x1, y1 = _footprint(a, shape, page.shape)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        word = data._words(sd, y0, x0, y1 - y0, x1 - x0)
        if (word > 0).any() or data._unknown(sd, word, y0, x0, y1 - y0, x1 - x0).any():
            continue
        touch = _areas(sd, held, x0, y0, x1, y1)
        in_q = bool(data.in_area(sd, held[0], y, x, 1, 1)[0, 0])
        split = "train" if not any(touch) else ("quarter" if in_q and series == paths.SERIES else "")
        if not split:
            continue
        info = _centre_info(sd, x, y)
        crops.append(crop)
        meta.append({"series": series, "sheet": sheet, "word": -1, "kind": "none", "terra": "", "text": "",
                     "angle": angle, "direction": "", "box": list(map(float, box)), "length": length,
                     "height": height, "split": split, "touch_quarter": touch[0], "touch_validation": touch[1],
                     "touch_blocks": touch[2], "weight": miss_weight if split == "train" else 1.0, **info})
        made += 1
    return series, sheet, crops, meta, made, want, time.perf_counter() - t0


def hash_name(name: str) -> int:
    """A stable number for a sheet's name (Python's hash is salted per process)."""
    return int.from_bytes(name.encode()[-8:].rjust(8, b"\0"), "little") % (2 ** 31)


def miss_rate(recs, others_by_sheet) -> dict:
    """Terra's miss rate on the 2000 sheets: the share of the other passes' boxes that no Terra word overlaps."""
    missed = total = 0
    for sheet in paths.SHEETS:
        sd = data.sheet_data(paths.SERIES, sheet, hold_out_validation=False,
                             terra=[r for r in recs if r["series"] == paths.SERIES and r["sheet"] == sheet],
                             others=others_by_sheet.get(sheet, []), placement=False)
        for rect in sd.other:
            x0, y0 = np.maximum(rect.min(axis=0), 0)
            x1, y1 = np.minimum(rect.max(axis=0) + 1, [sd.width, sd.height])
            if x1 <= x0 or y1 <= y0:
                continue
            word = data._words(sd, y0, x0, y1 - y0, x1 - x0)
            own = data._paint(rect[None], np.ones(1, np.int32), y0, x0, y1 - y0, x1 - x0)
            total += 1
            missed += int(not ((word > 0) & (own > 0)).any())
    return {"other_pass_boxes": total, "no_terra_word": missed, "rate": missed / max(total, 1)}


def answered_crops(labs, floor: float) -> tuple[list, list]:
    """The owner's answered labels with Terra's box (box-found), cropped like the words (a code's height floor)."""
    crops, meta = [], []
    pages, places = {}, {}
    for lab in labs:
        if not lab.held_out or lab.box is None:
            continue
        if lab.sheet not in pages:
            pages[lab.sheet] = data.load_page(lab.sheet)
            places[lab.sheet] = data.load_transform(lab.sheet)
        mx, my = places[lab.sheet].to_world(np.array([lab.point], float))[0]
        crop, a, shape, (length, height) = rectify(pages[lab.sheet], lab.box, lab.angle, floor=floor)
        crops.append(crop)
        meta.append({"series": paths.SERIES, "sheet": lab.sheet, "word": -1, "kind": "zoning code",
                     "terra": lab.terra_text, "text": norm_text(lab.terra_text) or "", "angle": float(lab.angle),
                     "direction": lab.direction, "box": list(map(float, lab.box)), "length": length,
                     "height": height, "split": "answered", "index": int(lab.index),
                     "point": [float(v) for v in lab.point], "answer": lab.answer, "flagged": bool(lab.flagged),
                     "fold": int(lab.fold), "lab_block": lab.block, "case_blocks": list(lab.case_blocks),
                     "weight": 1.0, "mx": float(mx), "my": float(my)})
    return crops, meta


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    blocks = reader_stage.held_blocks()
    area = reader_stage.load_block_area(blocks)
    r1 = json.loads((paths.READER_TEXT_DIR / reader_stage.HELD_FILE).read_text())
    if r1["blocks"] != blocks:
        raise ValueError("the answered blocks differ from R1's held.json")
    radius = int(r1["unknown_radius"]["px"])
    held = (data.Quarter(HOLD_QUARTER), data.VALIDATION, area)
    recs = boxes.terra_records()
    boxes.check_text_sheets(recs)
    others_by_sheet: dict[str, list[dict]] = {}
    for f in data._other_files():
        for b in saved.answers(f"{data.OTHER_PASSES}/{f.name.removesuffix('.jsonl.gz')}"):
            others_by_sheet.setdefault(b.get("sheet"), []).append(b)
    miss = miss_rate(recs, others_by_sheet)
    print(f"answered blocks {len(blocks)}, unknown radius {radius} px, Terra's miss rate {miss}", flush=True)
    by_sheet: dict[tuple[str, str], list[dict]] = {}
    for r in recs:
        by_sheet.setdefault((r["series"], r["sheet"]), []).append(r)
    jobs = [(s, name, by_sheet.get((s, name), []), others_by_sheet.get(name, []) if s == paths.SERIES else [])
            for s in MAPS for name in paths.TEXT_SHEETS[s]]
    crops, meta = [], []
    with Pool(WORKERS, initializer=_init, initargs=(held, radius, 1.0 - miss["rate"], {})) as pool:
        found = [kh for part in pool.imap(quarter_turn_heights, jobs) for kh in part]
    floors = {k: float(np.quantile([h for kk, h in found if kk == k], FLOOR_QUANTILE))
              for k in sorted({k for k, _ in found})}
    print(f"slanted words' height floors (the {FLOOR_QUANTILE} quantile of quarter-turn training words): {floors}",
          flush=True)
    with Pool(WORKERS, initializer=_init, initargs=(held, radius, 1.0 - miss["rate"], floors)) as pool:
        for series, sheet, c, m, made, want, sec in pool.imap(sheet_crops, jobs):
            crops += c
            meta += m
            print(f"{series} {sheet}: {len(c)} crops ({made}/{want} non-text) in {sec:.0f} s", flush=True)
    labs = pickle.load(open(INPUTS / "labs.pkl", "rb"))
    fresh = {(lab.index, lab.point): lab for lab in L.load_labels(crops=False)}
    for lab in labs:      # the generative reader's labels against today's: the same answers and boxes
        f = fresh.get((lab.index, lab.point))
        if f is None or f.answer != lab.answer or f.box != lab.box or f.held_out != lab.held_out:
            raise ValueError(f"labs.pkl differs from today's labels at {lab.index} {lab.point}")
    c, m = answered_crops(labs, floors["zoning code"])
    crops += c
    meta += m
    widths = np.array([cr.shape[1] for cr in crops], np.int32)
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
    flat = np.concatenate([cr for cr in crops], axis=1) if crops else np.zeros((HEIGHT, 0), np.uint8)
    np.savez(OUT / "crops.npz", flat=flat, offsets=offsets, widths=widths)
    df = pd.DataFrame(meta)
    add_blocks(df)
    df.to_pickle(OUT / "crops_meta.pkl")
    summary = {"maps": MAPS, "hold_quarter": paths.QUARTERS[HOLD_QUARTER], "answered_blocks": len(blocks),
               "slanted_height_floors": floors, "floor_quantile": FLOOR_QUANTILE,
               "unknown_radius": radius, "miss": miss, "height": HEIGHT, "pad": [PAD_X, PAD_Y],
               "counts": {f"{s}/{k}": int(n) for (s, k), n in df.groupby(["split", "kind"]).size().items()},
               "dropped_text": int(df["text"].isna().sum()),
               "seconds": round(time.time() - t0, 1)}
    (OUT / "crops_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
