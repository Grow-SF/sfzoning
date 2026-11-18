"""Word boxes on every 2000 sheet, grouped from our own text detector's chance (stage `solver-boxes`,
`sfzmaps solver boxes`).  No outside input: the boxes come from the held-out detectors' per-px chance
(`paths.READER_TEXT_DIR/<sheet>/{chance.png, kind.npz}`, each px from the detector that never saw its quarter of the
city or the answered blocks) and the page's ink.  Terra's saved boxes are read to CHOOSE and to SCORE, never as input to
the boxes (see "The level").

A box is one connected component (8-connectivity) of the px that are at or above the LEVEL in chance and within INK_R px
of ink (the detector reads ink only, so its chance on blank paper is no evidence), among components holding at least
MIN_PX ink px.  The component's px, weighted by chance, give its centre and second moments; the AXIS is the major
axis (degrees counter-clockwise from left-to-right, modulo 180, as Terra's reading angle); the grouped box is the
extent of the component's px along and across the axis.  The detector's chance spreads a few px past the ink, so each
box also records the TRIMMED box: the same axes, the extent of the component's ink px alone (the glyphs, which is
what the reader should crop).  The ports of spike S3's grouping (`scripts/spikes/s3_boxes.py`, level "I0.80").

The level, chosen per quarter of the city (`paths.QUARTERS`) from LEVELS: the level that finds most of Terra's
zoning-code words (centre in the quarter's complement, IoU >= IOU) on the OTHER three quarters; ties go to the higher
level.  The boxes whose centre's map point is in quarter q are then the boxes of q's level.  Terra's words are used
for this choice and for the scores only.

Each kept box is placed on the map as the prints are (`prints._place`: its centre must be inside the sheet's footprint
and on the mapped area; others are dropped and counted): `x`, `y`, `lab_lot` (the lot under it) and `street`.

Writes data/solver/boxes/:
  boxes.parquet  one row per box (columns in COLUMNS)
  levels.json    the chosen level per quarter and the table it was chosen from
  summary.json   per quarter: the zoning-code words found (IoU >= IOU), with the axis within AXIS_OK degrees, the
                 trimmed box's recall; the boxes kept and dropped off the map; how many of the owner's answered labels
                 get a box.  Nothing in it is a time.
"""
import json
import logging
import time
from multiprocessing import get_context

import cv2
import numpy as np
import pandas as pd
import shapely

from sfzmaps import config
from sfzmaps.solver import grid, lots, page, paths, prints, runs
from sfzmaps.solver.textnet import boxes as textboxes
from sfzmaps.solver.textnet import data
from sfzmaps.transform import SheetTransform

log = logging.getLogger(__name__)

BOXES_DIR = paths.SOLVER_DIR / "boxes"
LEVELS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0)   # the chance levels tried: the held-out detectors' chance
#   is near 1 on most ink, so the useful levels sit high (spike S3's 0.5 to 0.9 found 20 to 60% of the codes)
INK_R = 4          # px: the chance counts only within this of ink
MIN_PX = 20        # a component with fewer ink px is a speck
TOUCH_IOU = 0.1    # a word is "touched" by a box at this IoU (a looser find: what the misses overlap at all)
IOU = 0.3          # a word is found when a box meets it at this IoU
AXIS_OK = 10.0     # degrees: a found word's axis is usable within this
WORKERS = 4
KIND_NAMES = ("zoning", "block", "street", "other")        # data.KINDS' order
KIND = data.KINDS["zoning code"]

CORNER_COLS = [f"{k}{i}{a}" for k in "gt" for i in range(4) for a in "xy"]    # g: the grouped box, t: the trimmed
COLUMNS = (["sheet", "i", "level", "quarter", "page_x", "page_y", "axis", "length", "height", "t_length", "t_height",
            "mean_chance", "ink_px", *[f"p_{k}" for k in KIND_NAMES], "x", "y", "lab_lot", "street"] + CORNER_COLS)


def level_name(level: float) -> str:
    return f"{level:.3f}"


# --- grouping ---------------------------------------------------------------------------------------------------

def ink_gate(ink: np.ndarray, radius: int = INK_R) -> np.ndarray:
    """The px within `radius` px of ink (an ellipse of the diameter 2 radius + 1), bool."""
    k = 2 * radius + 1
    return cv2.dilate(ink.view(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).view(bool)


def group(chance: np.ndarray, ink: np.ndarray, gate: np.ndarray, level: float, kind_p: np.ndarray | None = None,
          min_px: int = MIN_PX) -> dict:
    """The boxes of one page at one level (see the module): `chance` uint8 (chance x 255), `ink` and `gate` bool,
    `kind_p` (4, h, w) the kinds' probabilities (None: not asked).  Arrays over the boxes: `g` and `t` the grouped and
    the trimmed corners (n, 4, 2) in page px (edge coordinates: a px is [i, i + 1)), `axis` degrees modulo 180, `length`
    along the axis and `height` across it (the grouped box), `t_length` and `t_height` (the trimmed), `mean`
    the mean chance and `kind` the mean kind probabilities over the box's ink px, `ink_px`."""
    thr = int(np.ceil(level * 255 - 1e-9))
    mask = (chance >= thr) & gate
    _, lab = cv2.connectedComponents(mask.view(np.uint8), connectivity=8)
    ys, xs = np.nonzero(mask)
    n0 = np.zeros(0)
    empty = {"g": np.zeros((0, 4, 2)), "t": np.zeros((0, 4, 2)), "axis": n0, "length": n0, "height": n0,
             "t_length": n0, "t_height": n0, "mean": n0, "kind": np.zeros((0, 4)), "ink_px": np.zeros(0, int)}
    if not len(ys):
        return empty
    lab_px = lab[ys, xs]
    order = np.argsort(lab_px, kind="stable")
    lab_px, xi, yi = lab_px[order], xs[order], ys[order]
    xs, ys = xi + 0.5, yi + 0.5
    w = chance[yi, xi] / 255.0
    on_ink = ink[yi, xi]
    start = np.flatnonzero(np.r_[True, lab_px[1:] != lab_px[:-1]])
    cnt = np.diff(np.r_[start, len(lab_px)])
    rep = lambda a: np.repeat(a, cnt)       # noqa: E731
    m = np.add.reduceat(w, start)
    cx, cy = np.add.reduceat(w * xs, start) / m, np.add.reduceat(w * ys, start) / m
    rx, ry = xs - rep(cx), ys - rep(cy)
    sxx, syy, sxy = (np.add.reduceat(w * a, start) / m for a in (rx * rx, ry * ry, rx * ry))
    th = 0.5 * np.arctan2(2 * sxy, sxx - syy)                    # the major axis, image coordinates (y down)
    vx, vy = np.cos(th), np.sin(th)
    p = rx * rep(vx) + ry * rep(vy)
    q = -rx * rep(vy) + ry * rep(vx)
    p0, p1 = np.minimum.reduceat(p, start) - 0.5, np.maximum.reduceat(p, start) + 0.5
    q0, q1 = np.minimum.reduceat(q, start) - 0.5, np.maximum.reduceat(q, start) + 0.5
    n_ink = np.add.reduceat(on_ink.astype(np.int64), start)
    keep = n_ink >= min_px
    # the ink alone (a kept component holds at least min_px >= 1 ink px)
    big = np.inf
    pi0 = np.minimum.reduceat(np.where(on_ink, p, big), start) - 0.5
    pi1 = np.maximum.reduceat(np.where(on_ink, p, -big), start) + 0.5
    qi0 = np.minimum.reduceat(np.where(on_ink, q, big), start) - 0.5
    qi1 = np.maximum.reduceat(np.where(on_ink, q, -big), start) + 0.5
    with np.errstate(invalid="ignore"):
        mean = np.add.reduceat(np.where(on_ink, w, 0.0), start) / np.maximum(n_ink, 1)
        kind = np.zeros((len(start), 4))
        if kind_p is not None:
            for k in range(4):
                kind[:, k] = np.add.reduceat(np.where(on_ink, kind_p[k][yi, xi].astype(np.float64), 0.0),
                                             start) / np.maximum(n_ink, 1)
    v, nrm = np.stack([vx, vy], 1), np.stack([-vy, vx], 1)
    c = np.stack([cx, cy], 1)

    def corners(a0, a1, b0, b1):
        return np.stack([c + v * a0[:, None] + nrm * b0[:, None], c + v * a1[:, None] + nrm * b0[:, None],
                         c + v * a1[:, None] + nrm * b1[:, None], c + v * a0[:, None] + nrm * b1[:, None]], 1)

    g = corners(p0, p1, q0, q1)
    t = corners(np.where(keep, pi0, 0), np.where(keep, pi1, 0), np.where(keep, qi0, 0), np.where(keep, qi1, 0))
    axis = np.rad2deg(np.arctan2(-vy, vx)) % 180.0               # counter-clockwise from left-to-right
    return {"g": g[keep], "t": t[keep], "axis": axis[keep], "length": (p1 - p0)[keep], "height": (q1 - q0)[keep],
            "t_length": (pi1 - pi0)[keep], "t_height": (qi1 - qi0)[keep], "mean": mean[keep], "kind": kind[keep],
            "ink_px": n_ink[keep]}


# --- matching Terra's words ---------------------------------------------------------------------------------------

def poly_area(c: np.ndarray) -> np.ndarray:
    """The areas of polygons (n, k, 2)."""
    if not len(c):
        return np.zeros(0)
    x, y = c[..., 0], c[..., 1]
    return 0.5 * np.abs((x * np.roll(y, -1, 1) - np.roll(x, -1, 1) * y).sum(1))


def pairs(a: np.ndarray, b: np.ndarray):
    """For every convex quadrilateral of `a` and of `b` (n, 4, 2) that overlap: their indices and the overlap's area."""
    if not len(a) or not len(b):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    al, ah, bl, bh = a.min(1), a.max(1), b.min(1), b.max(1)
    hit = ((al[:, None, 0] < bh[None, :, 0]) & (ah[:, None, 0] > bl[None, :, 0])
           & (al[:, None, 1] < bh[None, :, 1]) & (ah[:, None, 1] > bl[None, :, 1]))
    ia, ib = np.nonzero(hit)
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    inter = np.array([cv2.intersectConvexConvex(a32[i], b32[j])[0] for i, j in zip(ia, ib)])
    pos = inter > 0
    return ia[pos], ib[pos], inter[pos]


def axis_error(a, b):
    """The angle between two axes (degrees, modulo 180), in [0, 90]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 180.0
    return np.minimum(d, 180.0 - d)


def best_axis_match(words: np.ndarray, word_axes: np.ndarray, found: np.ndarray, axes: np.ndarray):
    """`best_match` with the axis error to the best polygon: (best IoU (n,), axis error (n,), NaN where none)."""
    n = len(words)
    best, err = np.zeros(n), np.full(n, np.nan)
    if not n or not len(found):
        return best, err
    wa, fa = poly_area(words), poly_area(found)
    i, j, inter = pairs(words, found)
    iou = inter / (wa[i] + fa[j] - inter)
    bj = np.full(n, -1)
    for k in np.argsort(iou):
        best[i[k]], bj[i[k]] = iou[k], j[k]
    has = bj >= 0
    err[has] = axis_error(word_axes[has], axes[bj[has]])
    return best, err


# --- a sheet ------------------------------------------------------------------------------------------------------

_SHARED: dict = {}      # the lots and the mapped area, read once before the workers fork


def _load_shared() -> None:
    _SHARED["lots"] = lots.load_lots()
    _SHARED["mapped"] = page.load_sheets() > 0
    _SHARED["terra"] = textboxes.terra_records()


def placement(sheet: str):
    """A 2000 sheet's corrected placement and footprint, as the warp wrote them (transform.json and
    warped.footprint.geojson): (SheetTransform, shapely footprint in map ft)."""
    warp = config.DATA_DIR / "warp" / paths.SERIES / sheet
    transform = SheetTransform.from_json(json.loads((warp / "transform.json").read_text()), folder=warp)
    return transform, shapely.from_geojson((warp / "warped.footprint.geojson").read_text())


def place(centres: np.ndarray, sheet: str, transform, footprint, lot_rasters, mapped) -> dict:
    """Boxes placed on the map as the prints are (`prints._place`): those whose centre (page px, (n, 2)) lands inside
    the sheet's footprint and on the mapped area, in order.  Each kept box's position in `centres` (`i`), its centre
    (`page_x`, `page_y`), its quarter (`grid.quarter_of` of its map point), its map point (`x`, `y`), the lot under
    it (`lab_lot`) and whether it is in a street; lists of the kept boxes (quarter an array).  The placement of every
    box of `sheet_boxes`, and of any other boxes on a 2000 sheet."""
    found = [{"page_x": float(c[0]), "page_y": float(c[1]), "i": k}
             for k, c in enumerate(np.asarray(centres, float).reshape(-1, 2))]
    placed = prints._place(found, sheet, transform, footprint, lot_rasters, mapped)
    k = np.array([p["i"] for p in placed], int)
    return {"i": k, "page_x": [p["page_x"] for p in placed], "page_y": [p["page_y"] for p in placed],
            "quarter": grid.quarter_of(np.array([[p["x"], p["y"]] for p in placed])) if len(k)
            else np.zeros(0, np.int8),
            "x": [p["x"] for p in placed], "y": [p["y"] for p in placed],
            "lab_lot": [p["lab_lot"] for p in placed], "street": [p["street"] for p in placed]}


def _read_chance(sheet: str, shape) -> np.ndarray:
    f = paths.READER_TEXT_DIR / sheet / "chance.png"
    c = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if c is None:
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver text-reader`")
    if c.shape != tuple(shape):
        raise ValueError(f"{f} is {c.shape[1]} x {c.shape[0]} px, its page {shape[1]} x {shape[0]}")
    return c


def sheet_boxes(sheet: str) -> dict:
    """One sheet's boxes at every level, placed on the map, and Terra's zoning-code words scored against them:
    {"levels": {level name: DataFrame (COLUMNS) of the placed boxes}, "off_map": {level name: boxes dropped},
    "words": {level name: DataFrame, one row per zoning-code word: quarter, best IoU of the grouped box, its axis error,
    the best IoU of the trimmed box}}."""
    ink = data.load_page(sheet) < grid.INK_BELOW
    gate = ink_gate(ink)
    chance = _read_chance(sheet, ink.shape)
    with np.load(paths.READER_TEXT_DIR / sheet / "kind.npz") as z:
        kind_p = z["p"]
    transform, footprint = placement(sheet)
    found_words = [b for b in textboxes.terra_boxes(paths.SERIES, sheet, _SHARED["terra"])
                   if data.KINDS.get(b["kind"], data.KINDS["other"]) == KIND]
    rects = [data.word_rect(b["box"], float(b.get("angle_deg") or 0.0)) for b in found_words]
    words = np.array(rects, float).reshape(-1, 4, 2)
    word_axis = np.array([data.reading_angle(r) for r in rects], float) % 180.0
    wq = (grid.quarter_of(transform.to_world(words.mean(1))) if len(words) else np.zeros(0, np.int8))
    out = {"levels": {}, "off_map": {}, "words": {}}
    for level in LEVELS:
        name = level_name(level)
        G = group(chance, ink, gate, level, kind_p)
        best, err = best_axis_match(words, word_axis, G["g"], G["axis"])
        tbest, _ = best_axis_match(words, word_axis, G["t"], G["axis"])
        out["words"][name] = pd.DataFrame({"quarter": wq, "best": best, "err": err, "tbest": tbest})
        placed = place(G["g"].mean(1), sheet, transform, footprint, _SHARED["lots"], _SHARED["mapped"])
        k = placed["i"]
        out["off_map"][name] = len(G["g"]) - len(k)
        df = pd.DataFrame({"sheet": sheet, "i": k, "level": level, "quarter": placed["quarter"],
                           "page_x": placed["page_x"], "page_y": placed["page_y"]})
        df["axis"], df["length"], df["height"] = G["axis"][k], G["length"][k], G["height"][k]
        df["t_length"], df["t_height"] = G["t_length"][k], G["t_height"][k]
        df["mean_chance"], df["ink_px"] = G["mean"][k], G["ink_px"][k]
        for c, nm in enumerate(KIND_NAMES):
            df[f"p_{nm}"] = G["kind"][k, c]
        df["x"], df["y"] = placed["x"], placed["y"]
        df["lab_lot"], df["street"] = placed["lab_lot"], placed["street"]
        for key, arr in (("g", G["g"]), ("t", G["t"])):
            flat = arr[k].reshape(len(k), 8)
            for c in range(8):
                df[f"{key}{c // 2}{'xy'[c % 2]}"] = flat[:, c]
        out["levels"][name] = df
    return out


def _job(sheet: str) -> dict:
    cv2.setNumThreads(1)
    return sheet_boxes(sheet)


# --- the level ----------------------------------------------------------------------------------------------------

def choose_levels(words: dict[str, pd.DataFrame]) -> dict:
    """The level of each quarter and the table it was chosen from.  `words`: level name -> every sheet's
    zoning-code words (quarter, best).  A quarter's level is the one with the highest recall (best >= IOU) on the words
    of the other quarters; ties to the higher level.  {"level": {quarter: level}, "recall_on_others": {quarter:
    {level name: recall}}}."""
    table = {}
    for q, qn in enumerate(paths.QUARTERS):
        table[qn] = {}
        for name, w in words.items():
            o = w[w["quarter"] != q]
            table[qn][name] = float((o["best"] >= IOU).mean()) if len(o) else 0.0
    chosen = {qn: max(LEVELS, key=lambda lv: (t[level_name(lv)], lv)) for qn, t in table.items()}
    return {"level": chosen, "recall_on_others": table}


def keep_chosen(boxes: dict[str, pd.DataFrame], levels: dict[str, float]) -> pd.DataFrame:
    """The boxes whose quarter's level is their own: `boxes` level name -> every sheet's placed boxes at that level,
    `levels` quarter name -> level."""
    parts = []
    for q, qn in enumerate(paths.QUARTERS):
        df = boxes[level_name(levels[qn])]
        parts.append(df[df["quarter"] == q])
    return pd.concat(parts, ignore_index=True)


def quarter_measures(words: dict[str, pd.DataFrame], levels: dict[str, float]) -> dict:
    """Per quarter, on its own zoning-code words at its level: the words, the share found (IoU >= IOU) by the
    grouped box and by the trimmed box, the share touched at TOUCH_IOU, and the share found with the axis within
    AXIS_OK degrees."""
    out = {}
    for q, qn in enumerate(paths.QUARTERS):
        w = words[level_name(levels[qn])]
        w = w[w["quarter"] == q]
        n = len(w)
        found = w["best"] >= IOU
        out[qn] = {"level": levels[qn], "words": int(n), "found": int(found.sum()),
                   "recall": float(found.mean()) if n else None,
                   "trimmed_recall": float((w["tbest"] >= IOU).mean()) if n else None,
                   "touched": float((w["best"] >= TOUCH_IOU).mean()) if n else None,
                   "axis_within_10": float((found & (w["err"] <= AXIS_OK)).mean()) if n else None}
    return out


def answered_with_box(df: pd.DataFrame) -> dict:
    """How many of the owner's answered labels (`codereader.labels`; used here only to count) get a box among the
    kept boxes `df`: of those with a Terra box, how many meet it at IOU; of all answered, how many have a box
    holding their point."""
    from sfzmaps.solver.codereader import labels
    held = [lab for lab in labels.load_labels(crops=False) if lab.held_out]
    with_terra = [lab for lab in held if lab.box is not None]
    hit_iou = hit_point = 0
    for lab in held:
        mine = df[df["sheet"] == lab.sheet]
        g = mine[[f"g{i}{a}" for i in range(4) for a in "xy"]].to_numpy(float).reshape(-1, 4, 2)
        px, py = lab.point
        if len(g) and any(cv2.pointPolygonTest(c.astype(np.float32).reshape(-1, 1, 2), (float(px), float(py)),
                                               False) >= 0 for c in g[(np.abs(g.mean(1) - [px, py]).max(1) < 400)]):
            hit_point += 1
        if lab.box is not None and len(g):
            rect = data.word_rect(lab.box, float(lab.angle)).astype(float).reshape(1, 4, 2)
            best, _ = best_axis_match(rect, np.zeros(1), g, np.zeros(len(g)))
            hit_iou += int(best[0] >= IOU)
    return {"answered": len(held), "with_terra_box": len(with_terra), "terra_box_met_at_iou": hit_iou,
            "point_inside_a_box": hit_point, "iou": IOU}


# --- the stage ----------------------------------------------------------------------------------------------------

def build_boxes(workers: int = WORKERS) -> dict:
    """Group every 2000 sheet's boxes, choose each quarter's level, and write BOXES_DIR (see the module).  Returns the
    summary and the seconds taken."""
    t0 = time.perf_counter()
    names = paths.check_sheets()
    _load_shared()
    ctx = get_context("fork")
    with ctx.Pool(min(workers, len(names))) as pool:
        results = pool.map(_job, names, chunksize=1)
    log.info("grouped %d sheets at %d levels in %.0f s", len(names), len(LEVELS), time.perf_counter() - t0)
    words = {level_name(lv): pd.concat([r["words"][level_name(lv)] for r in results], ignore_index=True)
             for lv in LEVELS}
    chosen = choose_levels(words)
    levels = chosen["level"]
    kept = keep_chosen({level_name(lv): pd.concat([r["levels"][level_name(lv)] for r in results], ignore_index=True)
                        for lv in LEVELS}, levels)
    kept = kept.sort_values(["sheet", "quarter", "i"], kind="stable").reset_index(drop=True)[COLUMNS]
    kept["quarter"] = kept["quarter"].astype("int8")
    kept["street"] = kept["street"].astype(bool)
    kept["lab_lot"] = kept["lab_lot"].astype("int32")
    kept["i"] = kept["i"].astype("int32")
    off_map = {level_name(lv): int(sum(r["off_map"][level_name(lv)] for r in results)) for lv in LEVELS}
    summary = {"quarters": quarter_measures(words, levels), "boxes": int(len(kept)),
               "boxes_by_quarter": {qn: int((kept["quarter"] == q).sum()) for q, qn in enumerate(paths.QUARTERS)},
               "off_map_by_level": off_map, "answered": answered_with_box(kept),
               "min_ink_px": MIN_PX, "ink_radius_px": INK_R}
    part = runs.fresh_folder(BOXES_DIR)
    kept.to_parquet(part / "boxes.parquet", index=False)
    (part / "levels.json").write_text(json.dumps(chosen, indent=1) + "\n")
    (part / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    runs.replace_folder(part, BOXES_DIR)
    return {**summary, "seconds": round(time.perf_counter() - t0, 1)}


def load_boxes() -> pd.DataFrame:
    """Read boxes.parquet."""
    f = BOXES_DIR / "boxes.parquet"
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver boxes` (stage solver-boxes)")
    return pd.read_parquet(f)
