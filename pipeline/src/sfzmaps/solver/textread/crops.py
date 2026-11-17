"""The real crops the text reader learns from and is scored on (spike S2's `crops.py`, held out by quarter), and the
crop of one of our word boxes.

Every crop is a word rectified along its reading angle from the full-resolution page (`data/work/<series>/<sheet>/
page.png`), as the code reader straightens its labels (`codereader.labels.frame` / `sample`: an exact px copy for a
quarter turn, bilinear otherwise, white beyond the page): the word's rectangle with PAD_X word heights either side
along the line and PAD_Y above and below, resized to HEIGHT px tall (aspect kept; INTER_AREA when shrinking), at most
MAX_WIDTH px wide.  Its half turn (`np.rot90(crop, 2)`) is the 180-degree copy.

Terra's words (`rectify`): the word rectangle of Terra's box at Terra's angle (`codereader.labels.word_size`).  A
slanted word's height (any angle but a quarter turn) is solved from the axis box and the angle, and that is
ill-conditioned (a loose box gives 2-7 px), so it is floored at its kind's FLOOR_QUANTILE quantile of the quarter-turn
words outside the answered blocks (exact: their box's sides), its length then following from the box (`word_geometry`).
Our boxes (`box_crop`): the ink-trimmed rectangle (`wordboxes`' t corners, its `t_length` and `t_height`) along the
box's axis: the glyphs fill the crop's height as Terra's tight boxes do.  A box's own px (a page mask) are cut the same
way (`box_mask`), and a crop can be blanked outside them (`mask_outside`; reading with `read.build_read(mask=True)`).

The words (`sheet_crops`): every word Terra boxed (every kind) on the 2000 drawing's three scans (MAPS), and windows
of non-text: a word's size and angle centred on ink outside every Terra word and every unknown px (Sol's points within
the code reader's detectors' radius, the other passes' boxes), with target "" and weight 1 - Terra's measured miss
rate (`miss_rate`).  Each crop records which held-out areas its page footprint touches, through each px's map point on
every scan at once: each quarter of the city (`touch_<q>`, `data.Quarter`) and the answered blocks (`touch_blocks`,
`textnet.reader_stage.load_block_area`).  Quarter q's reader trains only on crops touching neither q nor the answered
blocks; a non-text window touching the answered blocks is not kept.  Every crop records its centre's map point, its
quarter and the block of the lot under it, so the hold-outs can be checked by a second route.

The answered labels (`answered_crops`): the owner's code-check answers joined to the prints
(`codereader.labels.load_labels`: `held_out`), each with its point, answer and quarter (its point's map point), and
those with Terra's box cropped like the words (a code's height floor).  Scored only, never trained on.
"""
import dataclasses
import json
import time
from multiprocessing import get_context

import cv2
import numpy as np
import pandas as pd

from sfzmaps.solver import grid, paths, saved
from sfzmaps.solver.codereader import labels as L
from sfzmaps.solver.textnet import boxes, data, reader_stage
from sfzmaps.solver.textread import model, store

MAPS = ("2000-digital-zoning", "1980-1990-zoning", "1990-1996-zoning")
HEIGHT = model.HEIGHT
PAD_X, PAD_Y = 0.4, 0.2     # word heights of page around a word, along and across the line
MAX_WIDTH = 1024            # px; a longer crop is squeezed to it
NONTEXT_SHARE = 0.2         # non-text windows per word, per sheet
NONTEXT_TRIES = 60          # draws per wanted window
WORKERS = 4
SEED = 20261116
FLOOR_QUANTILE = 0.1        # a slanted word's height floor: this quantile of its kind's quarter-turn heights
TOUCH = store.TOUCH


def norm_text(text) -> str | None:
    """Terra's string as the reader learns it: upper case, zero-width spaces dropped, single spaces; None when a
    character is outside the reader's VOCAB."""
    t = " ".join(str(text).replace("​", "").upper().split())
    return t if t and all(c in model.VOCAB for c in t) else None


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


def frame_of(centre: tuple[float, float], length: float, height: float, angle: float):
    """The page frame (`labels.frame`) and the page-px shape (h, w) of the crop of a word of `length` x `height` page
    px centred on the page px index `centre`, reading at `angle` (see the module)."""
    w = max(4, round(length + 2 * PAD_X * height))
    h = max(4, round(height * (1 + 2 * PAD_Y)))
    return L.frame(centre, w, h, angle), (h, w)


def cut(page: np.ndarray, centre: tuple[float, float], length: float, height: float, angle: float, fill=255,
        origin=(0, 0)):
    """The crop of a word of `length` x `height` page px centred on the page px index `centre`, reading at `angle`
    (see the module); its page frame (`labels.frame`) and its page-px shape before resizing.  `page` may be a window
    of the page whose px (0, 0) is the page's px `origin` (holding every px the crop reads: the same crop), and a bool
    image (a mask: True where any px the crop's px reads is True; `fill` False)."""
    a, (h, w) = frame_of(centre, length, height, angle)
    raw = L.sample(page, a, (h, w), fill, origin)
    out_w = min(MAX_WIDTH, max(4, round(w * HEIGHT / h)))
    how = cv2.INTER_AREA if h > HEIGHT else cv2.INTER_LINEAR
    if raw.dtype == bool:
        crop = cv2.resize(raw.view(np.uint8) * np.uint8(255), (out_w, HEIGHT), interpolation=how) > 0
    else:
        crop = cv2.resize(raw, (out_w, HEIGHT), interpolation=how)
    return crop, a, (h, w)


def rectify(page: np.ndarray, box, angle: float, floor: float = 0.0):
    """A Terra word's crop (see the module), its page frame, its page-px shape and the word's (length, height)."""
    x0, y0, x1, y1 = box
    length, height = word_geometry(box, angle, floor)
    crop, a, shape = cut(page, ((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2), length, height, angle)
    return crop, a, shape, (length, height)


def box_cut(page: np.ndarray, corners: np.ndarray, length: float, height: float, axis: float, fill=255,
            origin=(0, 0)):
    """`box_crop`, with its page frame and page-px shape (`cut`'s three), from the page or a window of it (`cut`)."""
    cx, cy = np.asarray(corners, float).reshape(4, 2).mean(0) - 0.5
    return cut(page, (float(cx), float(cy)), max(float(length), 1.0), max(float(height), 1.0), float(axis), fill,
               origin)


def box_crop(page: np.ndarray, corners: np.ndarray, length: float, height: float, axis: float) -> np.ndarray:
    """One of our word boxes' crop, read at its axis (0 degrees): `corners` its trimmed rectangle (4 x 2, page px edge
    coordinates), `length` and `height` that rectangle's sides along and across the axis (degrees)."""
    return box_cut(page, corners, length, height, axis)[0]


def box_mask(mask: np.ndarray, corners: np.ndarray, length: float, height: float, axis: float,
             origin=(0, 0)) -> np.ndarray:
    """A page mask (bool; the whole page, or a window of it whose px (0, 0) is the page's px `origin`) cut as
    `box_crop` cuts the page: bool, the crop's shape, True where any px the crop's px reads is True (False beyond the
    mask)."""
    return box_cut(mask, corners, length, height, axis, False, origin)[0]


def mask_outside(crop: np.ndarray, word_mask: np.ndarray) -> np.ndarray:
    """The crop with every px outside `word_mask` (bool, the crop's shape) paper (255): the word alone, as Mask
    TextSpotter v3 reads it.  ValueError when the shapes differ."""
    if np.shape(word_mask) != np.shape(crop):
        raise ValueError(f"a mask of {np.shape(word_mask)} for a crop of {np.shape(crop)}")
    return np.where(word_mask, crop, 255).astype(np.uint8)


def _footprint(a, shape, page_shape):
    return L._footprint(a, shape, 1, page_shape)


def _areas(sd, held, x0, y0, x1, y1) -> list[bool]:
    """Which held areas (in `held`'s order) the page window touches."""
    return [bool(data.in_area(sd, area, y0, x0, y1 - y0, x1 - x0).any()) for area in held]


def _centre_point(sd, cx: float, cy: float) -> dict:
    xy = data._world_at(sd.world, np.array([int(round(cy))]), np.array([int(round(cx))]))
    return {"mx": float(xy[0][0, 0]), "my": float(xy[1][0, 0])}


def add_blocks(df: pd.DataFrame) -> None:
    """Each row's quarter (`grid.quarter_of` of its map point `mx`, `my`) and the block of the lot under it (lots.npz
    `grown`; "" off the lots)."""
    with np.load(paths.LOTS_DIR / "lots.npz") as z:
        grown = z["grown"]
    lot_block = pd.read_parquet(paths.LOTS_DIR / "table.parquet", columns=["block_num"])["block_num"].astype(
        str).to_numpy()
    xy = df[["mx", "my"]].to_numpy(float)
    df["quarter"] = grid.quarter_of(xy) if len(df) else np.zeros(0, np.int8)
    cr = np.floor(grid.to_grid(xy)).astype(np.int64).reshape(-1, 2)
    on = (cr[:, 1] >= 0) & (cr[:, 1] < grown.shape[0]) & (cr[:, 0] >= 0) & (cr[:, 0] < grown.shape[1])
    lot = np.full(len(df), -1, np.int64)
    lot[on] = grown[cr[on, 1], cr[on, 0]]
    df["block"] = np.where(lot >= 0, lot_block[np.maximum(lot, 0)], "")


def hash_name(name: str) -> int:
    """A stable number for a sheet's name (Python's hash is salted per process)."""
    return int.from_bytes(name.encode()[-8:].rjust(8, b"\0"), "little") % (2 ** 31)


# --- per sheet, in worker processes ---------------------------------------------------------------------------------

_W: dict = {}


def _init(held, radius, miss_weight, floors):
    cv2.setNumThreads(1)
    _W.update(held=held, radius=radius, miss_weight=miss_weight, floors=floors)


def quarter_turn_heights(job) -> list[tuple[str, float]]:
    """(kind, height) of one sheet's quarter-turn words whose footprint touches no answered block."""
    series, sheet, terra, others = job
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
        if not _areas(sd, (_W["held"][-1],), fx0, fy0, fx1, fy1)[0]:
            out.append((b["kind"], float(height)))
    return out


def sheet_crops(job):
    """Every crop of one sheet: its words and its non-text windows (see the module), with their rows."""
    series, sheet, terra, others = job
    held, radius, miss_weight, floors = _W["held"], _W["radius"], _W["miss_weight"], _W["floors"]
    t0 = time.perf_counter()
    sd = data.sheet_data(series, sheet, hold_out_validation=False, terra=terra, others=others, placement=True,
                         point_radius=radius)
    sd = dataclasses.replace(sd, held=held)
    page = data.load_page(sheet, series)
    found = boxes.terra_boxes(series, sheet, terra)
    rng = np.random.default_rng([SEED, hash_name(sheet)])
    crops, rows = [], []
    for j, b in enumerate(found):
        angle = float(b.get("angle_deg") or 0.0)
        crop, a, shape, (length, height) = rectify(page, b["box"], angle, floor=floors.get(b["kind"], 0.0))
        x0, y0, x1, y1 = _footprint(a, shape, page.shape)
        cx, cy = (b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2
        crops.append(crop)
        rows.append({"series": series, "sheet": sheet, "word": j, "kind": b["kind"], "terra": b["text"],
                     "text": norm_text(b["text"]), "angle": angle, "direction": b["direction"],
                     **dict(zip(("box_x0", "box_y0", "box_x1", "box_y1"), map(float, b["box"]))),
                     "length": length, "height": height, "weight": 1.0,
                     **dict(zip(TOUCH, _areas(sd, held, x0, y0, x1, y1))), **_centre_point(sd, cx, cy)})
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
        xs = np.flatnonzero(data._ink_window(sd, y, 0, 1, sd.width)[0])
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
        if touch[-1]:
            continue
        crops.append(crop)
        rows.append({"series": series, "sheet": sheet, "word": -1, "kind": "none", "terra": "", "text": "",
                     "angle": angle, "direction": "",
                     **dict(zip(("box_x0", "box_y0", "box_x1", "box_y1"), map(float, box))),
                     "length": length, "height": height, "weight": miss_weight, **dict(zip(TOUCH, touch)),
                     **_centre_point(sd, x, y)})
        made += 1
    return series, sheet, crops, rows, made, want, time.perf_counter() - t0


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


def answered_crops(labels: list, floor: float) -> tuple[list[np.ndarray], pd.DataFrame]:
    """The owner's answered labels (`held_out`): one row each, in the labels' order, and the crops of those with
    Terra's box (`crop` the row of its crop among them, -1 without a box)."""
    crops, rows = [], []
    pages, places = {}, {}
    for lab in labels:
        if not lab.held_out:
            continue
        if lab.sheet not in pages:
            pages[lab.sheet] = data.load_page(lab.sheet)
            places[lab.sheet] = data.load_transform(lab.sheet)
        mx, my = places[lab.sheet].to_world(np.array([lab.point], float))[0]
        row = {"index": int(lab.index), "sheet": lab.sheet, "kind": int(lab.kind), "point_x": float(lab.point[0]),
               "point_y": float(lab.point[1]), "answer": lab.answer, "flagged": bool(lab.flagged),
               "fold": int(lab.fold), "lab_block": lab.block, "terra": lab.terra_text, "angle": float(lab.angle),
               "direction": lab.direction, "mx": float(mx), "my": float(my), "crop": -1,
               **dict(zip(("box_x0", "box_y0", "box_x1", "box_y1"),
                          map(float, lab.box) if lab.box is not None else [np.nan] * 4))}
        if lab.box is not None:
            crop, _, _, (length, height) = rectify(pages[lab.sheet], lab.box, lab.angle, floor=floor)
            row["crop"] = len(crops)
            crops.append(crop)
        rows.append(row)
    return crops, pd.DataFrame(rows)


# --- every crop -----------------------------------------------------------------------------------------------------

def make_crops(labels: list, workers: int = WORKERS) -> dict:
    """The words' and non-text windows' crops of every sheet of MAPS, and the answered labels' (see the module).
    `labels`: `codereader.labels.load_labels(crops=False)`.  Returns {"crops": list, "rows": DataFrame, "answered":
    list, "answered_rows": DataFrame, "summary": dict}."""
    t0 = time.perf_counter()
    blocks = sorted(L.held_out_blocks(labels))
    area = reader_stage.load_block_area(blocks)
    r1 = json.loads((paths.READER_TEXT_DIR / reader_stage.HELD_FILE).read_text())
    if r1["blocks"] != blocks:
        raise ValueError("the answered blocks differ from the code reader's detectors' held.json: rebuild "
                         "solver-text-reader")
    radius = int(r1["unknown_radius"]["px"])
    held = (*(data.Quarter(q) for q in range(len(paths.QUARTERS))), area)
    recs = boxes.terra_records()
    boxes.check_text_sheets(recs)
    others_by_sheet: dict[str, list[dict]] = {}
    for f in data._other_files():
        for b in saved.answers(f"{data.OTHER_PASSES}/{f.name.removesuffix('.jsonl.gz')}"):
            others_by_sheet.setdefault(b.get("sheet"), []).append(b)
    miss = miss_rate(recs, others_by_sheet)
    by_sheet: dict[tuple[str, str], list[dict]] = {}
    for r in recs:
        by_sheet.setdefault((r["series"], r["sheet"]), []).append(r)
    jobs = [(s, name, by_sheet.get((s, name), []), others_by_sheet.get(name, []) if s == paths.SERIES else [])
            for s in MAPS for name in paths.TEXT_SHEETS[s]]
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(held, radius, 1.0 - miss["rate"], {})) as pool:
        found = [kh for part in pool.imap(quarter_turn_heights, jobs) for kh in part]
    floors = {k: float(np.quantile([h for kk, h in found if kk == k], FLOOR_QUANTILE))
              for k in sorted({k for k, _ in found})}
    crops, rows = [], []
    with ctx.Pool(workers, initializer=_init, initargs=(held, radius, 1.0 - miss["rate"], floors)) as pool:
        for series, sheet, c, r, made, want, sec in pool.imap(sheet_crops, jobs):
            crops += c
            rows += r
    df = pd.DataFrame(rows)
    add_blocks(df)
    ans_crops, ans = answered_crops(labels, floors["zoning code"])
    add_blocks(ans)
    summary = {"maps": list(MAPS), "answered_blocks": len(blocks), "unknown_radius": radius, "miss": miss,
               "slanted_height_floors": floors, "floor_quantile": FLOOR_QUANTILE, "height": HEIGHT,
               "pad": [PAD_X, PAD_Y], "crops": len(df),
               "by_kind": {k: int(n) for k, n in df.groupby("kind").size().items()},
               "untrainable_text": int(((df["kind"] != "none") & df["text"].isna()).sum()),
               "answered": int(len(ans)), "answered_with_terra_box": int((ans["crop"] >= 0).sum()),
               "seconds": round(time.perf_counter() - t0, 1)}
    return {"crops": crops, "rows": df, "answered": ans_crops, "answered_rows": ans, "summary": summary}
