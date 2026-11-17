"""The text reader's training crops beyond Terra's boxes: crops of our own word boxes, sloppy crops of Terra's words,
and masking outside the word.  The reader reads 96% of the owner's answered labels right in Terra's tight boxes and
41% in our own (`wordread.measures`): it has never practised on boxes like ours.  Nothing here changes what a stage
does: `textread.train.fit` takes these crops as `extra`, behind its off-by-default options (`READER_OPTIONS`).

  Terra's words   (`terra_frame`) the 2000 sheets' words from `solver-text-crops`' rows: Terra's string as the reader
                  learns it, its rectangle as `crops.rectify` cuts it (`cx`, `cy` its centre's page px index, `length`,
                  `height`, the reading `angle`), the held-out areas its crop touches (`crops.TOUCH`), and its
                  rectangle as Task 1 matches it (`c0x`..`c3y`, `textnet.boxes.word_rect`)
  matching        Task 1's (`measures`): each of our boxes (its grouped rectangle) goes to the Terra word it meets
                  best, kept at IoU >= IOU (`matches`)
  own-box crops   (`own_box_crops`) each matched box cut as the reader reads it (`crops.box_crop`), turned a half turn
                  when Terra's word reads the other way along its axis, with Terra's string as its target (ABCNet v2's
                  training on its own detections).  Kept only when neither the crop's footprint nor Terra's word
                  touches the answered blocks, or the reader's held-out quarter when one is given
  the offsets     (`offset_distribution`) how our matched boxes sit on Terra's words, in Terra's word heights H, in our
                  box's own frame (its reading direction the axis turned towards Terra's): our trimmed box's start and
                  end along it, each minus Terra's (`start`, `end`; Terra's word from -L/2 to L/2 about its centre), its
                  centre's offset down the letters (`cross`), its angle minus Terra's (`angle`, degrees in [-90, 90])
                  and its height over Terra's (`height_ratio`): cutting Terra's word with one pair's offsets cuts that
                  box's own crop.  Also the share of matched words holding line ink (`line_share`: at least
                  `composite.CROSS_PX` line px that are ink inside Terra's rectangle, column `line_ink`), and the
                  share of matched boxes holding part of another Terra word (`neighbour_share`), with the largest
                  share of such a word's rectangle inside the box, for each (`neighbour_part`)
  sloppy crops    (`sloppy_crop`) Terra's word cut with one matched pair's offsets (drawn together: their spread and
                  their ties are the measured ones), as Weinman's synthesizer makes its words: with probability
                  `line_share` a real line piece (a `composite.Bank`'s, `piece_patches`), turned a random number of
                  quarter turns, ORed through a letter px of the word; with probability `neighbour_share` the nearest
                  other word (`with_neighbours`) moved straight towards the word until a drawn `neighbour_part` of its
                  rectangle lies inside the crop's rectangle, and ORed in.  The page itself is never changed.
                  `sloppy_crops` makes them for a reader: only of words, and with neighbours, touching neither its
                  held-out quarter nor the answered blocks, and kept only when the crop's footprint touches neither
  masking         (`mask_outside`; `crops.box_mask`, `textread.read.word_masks`) px outside a box's own px (the
                  component it was grouped from, grown by `wordboxes.INK_R`) are paper (Mask TextSpotter v3)

`build_offsets` measures the offsets for each held-out quarter (the pairs whose box is outside it and whose word
touches neither it nor an answered block) from today's boxes and writes OFFSETS_FILE
(`python -m sfzmaps.solver.wordread.readercrops`).  Seeded throughout; no stage imports this module.
"""
import dataclasses
import json
import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import shapely

from sfzmaps.solver import grid, paths, runs, wordboxes
from sfzmaps.solver.codereader import labels as L
from sfzmaps.solver.textnet import boxes as textboxes
from sfzmaps.solver.textnet import composite, data, reader_stage, zones
from sfzmaps.solver.textread import crops as K
from sfzmaps.solver.textread import store
from sfzmaps.solver.wordread import measures

log = logging.getLogger(__name__)

IOU = 0.5
OFFSETS = ("start", "end", "cross", "angle", "height_ratio")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
BISECT = 40                       # halvings of the distance a neighbour is moved
OFFSETS_FILE = runs.EXPERIMENTS_DIR / "wordread" / "reader" / "offsets.json"
TOUCH = store.TOUCH
mask_outside = K.mask_outside


# --- Terra's words ----------------------------------------------------------------------------------------------------

def _rect(cx: float, cy: float, length: float, height: float, angle: float) -> np.ndarray:
    """A word rectangle's corners (4, 2), page px index coordinates: start-top, end-top, end-bottom, start-bottom."""
    (ux, uy), (vx, vy) = L._axes(angle)
    u, v = np.array([ux, uy]) * length / 2, np.array([vx, vy]) * height / 2
    c = np.array([cx, cy], float)
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


def terra_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Terra's words on the 2000 sheets with a target, from rows of `solver-text-crops`' crops.parquet (see the
    module), positions 0..n-1; `line_ink` too when the rows have it."""
    w = rows[(rows["series"] == paths.SERIES) & (rows["word"] >= 0) & rows["text"].notna()]
    w = w[w["text"].astype(str) != ""].reset_index(drop=True)
    box = w[["box_x0", "box_y0", "box_x1", "box_y1"]].to_numpy(float)
    angle = w["angle"].to_numpy(float)
    rects = np.array([textboxes.word_rect(b, a) for b, a in zip(box, angle)], float).reshape(-1, 4, 2)
    out = pd.DataFrame({"sheet": w["sheet"].astype(str).to_numpy(), "word": w["word"].astype(int).to_numpy(),
                        "text": w["text"].astype(str).to_numpy(), "kind": w["kind"].to_numpy(),
                        "quarter": w["quarter"].astype(int).to_numpy(), "angle": angle,
                        **{c: box[:, k] for k, c in enumerate(("box_x0", "box_y0", "box_x1", "box_y1"))},
                        "cx": (box[:, 0] + box[:, 2] - 1) / 2, "cy": (box[:, 1] + box[:, 3] - 1) / 2,
                        "length": w["length"].to_numpy(float), "height": w["height"].to_numpy(float),
                        **{t: w[t].astype(bool).to_numpy() for t in TOUCH},
                        **dict(zip(measures.C_COLS, rects.reshape(-1, 8).T))})
    if "line_ink" in w:
        out["line_ink"] = w["line_ink"].astype(int).to_numpy()
    return out


def with_neighbours(terra: pd.DataFrame) -> pd.DataFrame:
    """`terra` with each word's nearest other word of the frame on its sheet (by the gap between their rectangles,
    `rectify`'s): `n_word` (-1 when it has none) and its `n_cx`, `n_cy`, `n_length`, `n_height`, `n_angle`."""
    out = terra.reset_index(drop=True).copy()
    near = np.full(len(out), -1)
    sheets = out["sheet"].to_numpy()
    geo = out[["cx", "cy", "length", "height", "angle"]].to_numpy(float)
    for sheet in sorted(set(sheets)):
        idx = np.flatnonzero(sheets == sheet)
        if len(idx) < 2:
            continue
        polys = shapely.polygons([_rect(*g) for g in geo[idx]])
        src, dst = shapely.STRtree(polys).query_nearest(polys, exclusive=True, all_matches=False)
        near[idx[src]] = idx[dst]
    has = near >= 0
    out["n_word"] = np.where(has, out["word"].to_numpy()[np.maximum(near, 0)], -1)
    for k, c in enumerate(("n_cx", "n_cy", "n_length", "n_height", "n_angle")):
        out[c] = np.where(has, geo[np.maximum(near, 0), k], np.nan)
    return out


# --- our boxes against Terra's words ----------------------------------------------------------------------------------

def matches(boxes: pd.DataFrame, terra: pd.DataFrame, iou: float = IOU) -> pd.DataFrame:
    """Our boxes matched to a Terra word at `iou` or more (Task 1's rule: each box to the word it meets best): `box`
    and `word` their positions in the frames, `iou`."""
    ti, bj, v = measures._overlaps(boxes, terra)
    word, best = measures._best_word(ti, bj, v, len(boxes))
    pos = np.flatnonzero((best >= iou) & (word >= 0))
    return pd.DataFrame({"box": pos, "word": word[pos], "iou": best[pos]})


def _turned(terra_angle: float, axis: float) -> bool:
    """Whether Terra's word reads the other way along our box's axis (more than a quarter turn from it)."""
    return math.cos(math.radians(float(terra_angle) - float(axis))) < 0


def _corners(b) -> np.ndarray:
    return np.array([[b[f"t{i}x"], b[f"t{i}y"]] for i in range(4)], float)


def _pair_offsets(b, t) -> tuple[float, ...]:
    """One matched pair's offsets (see the module): our box `b` (a boxes.parquet row), Terra's word `t`."""
    axis = float(b["axis"]) + (180.0 if _turned(t["angle"], b["axis"]) else 0.0)
    (ux, uy), (vx, vy) = L._axes(axis)
    ox, oy = _corners(b).mean(0) - 0.5
    dx, dy = ox - t["cx"], oy - t["cy"]
    along, across = dx * ux + dy * uy, dx * vx + dy * vy
    length, height = float(t["length"]), float(t["height"])
    lo, ho = float(b["t_length"]), float(b["t_height"])
    return ((along - lo / 2 + length / 2) / height, (along + lo / 2 - length / 2) / height, across / height,
            (axis - float(t["angle"]) + 180.0) % 360.0 - 180.0, ho / height)


def _held(flags, quarter: int | None) -> bool:
    return bool(flags["touch_blocks"]) or (quarter is not None and bool(flags[f"touch_{paths.QUARTERS[quarter]}"]))


def offset_distribution(boxes: pd.DataFrame, terra: pd.DataFrame, iou: float = IOU,
                        quarter: int | None = None) -> dict:
    """The measured spread of our matched boxes against Terra's words (see the module): `n` pairs, `offsets` (each
    a list, pair by pair), their `mean`, `sd` and `quantiles`, `turned` (pairs read a half turn from the axis),
    `line_share` (None without `line_ink`), `neighbour_share` and `neighbour_part`.  Pairs whose word touches an
    answered block are left out, and with `quarter` those whose word touches it or whose box is in it."""
    m = matches(boxes, terra, iou)
    keep = [not _held(terra.iloc[w], quarter) and (quarter is None or int(boxes.iloc[p]["quarter"]) != quarter)
            for p, w in zip(m["box"], m["word"])]
    m = m[np.array(keep, bool)].reset_index(drop=True)
    rows = [_pair_offsets(boxes.iloc[p], terra.iloc[w]) for p, w in zip(m["box"], m["word"])]
    off = np.array(rows, float).reshape(-1, len(OFFSETS))
    out = {"iou": iou, "n": int(len(m)), "offsets": {k: off[:, j].tolist() for j, k in enumerate(OFFSETS)},
           "mean": {k: float(off[:, j].mean()) if len(m) else None for j, k in enumerate(OFFSETS)},
           "sd": {k: float(off[:, j].std()) if len(m) else None for j, k in enumerate(OFFSETS)},
           "quantiles": {k: dict(zip(map(str, QUANTILES), np.quantile(off[:, j], QUANTILES).tolist()))
                         if len(m) else None for j, k in enumerate(OFFSETS)},
           "turned": int(sum(_turned(terra.iloc[w]["angle"], boxes.iloc[p]["axis"])
                             for p, w in zip(m["box"], m["word"])))}
    if "line_ink" in terra:
        crossed = terra["line_ink"].to_numpy()[m["word"].to_numpy()] >= composite.CROSS_PX
        out["line_share"] = float(crossed.mean()) if len(m) else None
        out["line_words"] = int(crossed.sum())
    else:
        out["line_share"], out["line_words"] = None, None
    # each matched box's largest share of another Terra word's rectangle inside its trimmed box
    part = np.zeros(len(m))
    sheet_of = boxes["sheet"].astype(str).to_numpy()
    tsheet = terra["sheet"].astype(str).to_numpy()
    geo = terra[["cx", "cy", "length", "height", "angle"]].to_numpy(float)
    for sheet in sorted(set(sheet_of[m["box"]])):
        mine = np.flatnonzero(sheet_of[m["box"]] == sheet)
        words = np.flatnonzero(tsheet == sheet)
        ours = np.array([_corners(boxes.iloc[p]) - 0.5 for p in m["box"].to_numpy()[mine]], float).reshape(-1, 4, 2)
        theirs = np.array([_rect(*g) for g in geo[words]], float).reshape(-1, 4, 2)
        i, j, inter = wordboxes.pairs(ours, theirs)
        share = inter / np.maximum(wordboxes.poly_area(theirs)[j], 1e-9)
        other = words[j] != m["word"].to_numpy()[mine][i]
        np.maximum.at(part, mine[i[other]], share[other])
    holds = part > 0
    out["neighbour_share"] = float(holds.mean()) if len(m) else None
    out["neighbour_part"] = part[holds].tolist()
    return out


# --- which held-out areas a crop touches ------------------------------------------------------------------------------

class HeldAreas:
    """Which held-out areas a page window of a 2000 sheet touches, as `solver-text-crops` records its crops' (TOUCH:
    each quarter of the city, then the answered blocks; through every px's map point, the sheet's placement):
    `touch(sheet, (x0, y0, x1, y1), page shape) -> {name: bool}`."""

    def __init__(self, blocks=None):
        blocks = reader_stage.held_blocks() if blocks is None else blocks
        self.held = (*(data.Quarter(q) for q in range(len(paths.QUARTERS))), reader_stage.load_block_area(blocks))
        self._sheets: dict = {}

    def __call__(self, sheet: str, window, shape) -> dict[str, bool]:
        sd = self._sheets.get(sheet)
        if sd is None:
            world = data.placement_samples(data.load_transform(sheet), tuple(shape))
            sd = self._sheets[sheet] = SimpleNamespace(series=paths.SERIES, sheet=sheet, world=world)
        x0, y0, x1, y1 = (int(v) for v in window)
        if x1 <= x0 or y1 <= y0:
            return {k: False for k in TOUCH}
        return dict(zip(TOUCH, K._areas(sd, self.held, x0, y0, x1, y1)))


def _pages(pages):
    """A page getter: `pages[sheet]`, or each 2000 page read from disk (the last one kept)."""
    if pages is not None:
        return lambda sheet: pages[sheet]
    last: dict = {}

    def get(sheet):
        if sheet not in last:
            last.clear()
            last[sheet] = data.load_page(sheet)
        return last[sheet]
    return get


# --- crops of our own boxes -------------------------------------------------------------------------------------------

def own_box_crops(boxes: pd.DataFrame, terra: pd.DataFrame, iou: float = IOU, quarter: int | None = None,
                  pages=None, touch=None, word_masks=None) -> list[tuple[np.ndarray, str, dict]]:
    """Our boxes matched to a Terra word at `iou`, each cut as `crops.box_crop` cuts it (turned a half turn when
    Terra's word reads the other way), with Terra's string: (crop, string, meta), by sheet and box.  Left out: a box
    whose crop's footprint or whose word touches an answered block, or (with `quarter`) that quarter.  `pages`: sheet
    -> page (read from disk when None); `touch`: `HeldAreas` (made when None); `word_masks`: (sheet, rows) -> {row:
    (bool window, origin)} (`textread.read.word_masks`), when given each meta's `mask` is the box's own px cut as the
    crop (turned with it).  meta: `sheet`, `row` (the box's index label), `word` (Terra's word on the sheet), `iou`,
    `turned`, `kind`, `quarter` (the box's) and the TOUCH flags (crop or word)."""
    touch = HeldAreas() if touch is None else touch
    page_of = _pages(pages)
    m = matches(boxes, terra, iou)
    sheet_of = boxes["sheet"].astype(str).to_numpy()
    out = []
    for sheet in sorted(set(sheet_of[m["box"]])):
        mine = m[sheet_of[m["box"]] == sheet].sort_values("box")
        page = page_of(sheet)
        rows = [{**boxes.iloc[p].to_dict(), "row": boxes.index[p]} for p in mine["box"]]
        masks = word_masks(sheet, rows) if word_masks is not None else None
        for (_, w, v), r in zip(mine.itertuples(index=False), rows, strict=True):
            t = terra.iloc[w]
            corners = _corners(r)
            crop, a, shape = K.box_cut(page, corners, r["t_length"], r["t_height"], r["axis"])
            flags = touch(sheet, K._footprint(a, shape, page.shape), page.shape)
            flags = {k: bool(flags[k]) or bool(t[k]) for k in TOUCH}
            if _held(flags, quarter):
                continue
            turned = _turned(t["angle"], r["axis"])
            if turned:
                crop = np.ascontiguousarray(np.rot90(crop, 2))
            meta = {"sheet": sheet, "row": r["row"], "word": int(t["word"]), "iou": float(v), "turned": turned,
                    "kind": t["kind"], "quarter": int(r["quarter"]), **flags}
            if masks is not None:
                win, origin = masks[r["row"]]
                mk = K.box_mask(win, corners, r["t_length"], r["t_height"], r["axis"], origin)
                meta["mask"] = np.ascontiguousarray(np.rot90(mk, 2)) if turned else mk
            out.append((crop, str(t["text"]), meta))
    return out


# --- sloppy crops of Terra's words ------------------------------------------------------------------------------------

def _paste(win: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    """OR a patch's ink (its px darker than paper) into the window at (top, left), cut to the window."""
    h, w = win.shape
    py0, px0 = max(0, -top), max(0, -left)
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(h, top + patch.shape[0]), min(w, left + patch.shape[1])
    if y1 > y0 and x1 > x0:
        win[y0:y1, x0:x1] = np.minimum(win[y0:y1, x0:x1], patch[py0:py0 + y1 - y0, px0:px0 + x1 - x0])


def _inside(shape, origin, cx, cy, length, height, angle) -> np.ndarray:
    """Which px of a window (its px (0, 0) the page's px `origin`) have their centre in a word rectangle."""
    (ux, uy), (vx, vy) = L._axes(angle)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dx, dy = xx + origin[0] - cx, yy + origin[1] - cy
    return (np.abs(dx * ux + dy * uy) <= length / 2) & (np.abs(dx * vx + dy * vy) <= height / 2)


def _line_through(win, origin, t, pieces, rng) -> dict:
    """OR a line piece (drawn, turned a random number of quarter turns) through a letter px of Terra's word `t`: a
    px of the piece's ink on an ink px of the word inside the window (any px of its rectangle when it has none)."""
    inside = _inside(win.shape, origin, t["cx"], t["cy"], t["length"], t["height"], t["angle"])
    ys, xs = np.nonzero(inside & (win < grid.INK_BELOW))
    if not len(ys):
        ys, xs = np.nonzero(inside)
    if not len(ys):
        return {}
    k = int(rng.integers(len(ys)))
    j, turns = int(rng.integers(len(pieces))), int(rng.integers(4))
    patch = np.rot90(pieces[j], turns)
    py, px = np.nonzero(patch < 255)
    m = int(rng.integers(len(py)))
    _paste(win, patch, int(ys[k] - py[m]), int(xs[k] - px[m]))
    return {"line": True, "piece": j, "turns": turns, "line_at": (int(xs[k] + origin[0]), int(ys[k] + origin[1]))}


def _neighbour_into(win, origin, page, t, rect, parts, rng) -> dict:
    """OR the nearest other word (its rectangle's px, as printed) into the window, moved straight from its place
    towards the word until a drawn share of its rectangle (`parts`) lies inside the crop's rectangle `rect` (centre,
    length, height, angle), to the nearest px."""
    f = float(parts[int(rng.integers(len(parts)))])
    n = np.array([t["n_cx"], t["n_cy"]], float)
    nl, nh, na = float(t["n_length"]), float(t["n_height"]), float(t["n_angle"])
    n0 = _rect(n[0], n[1], nl, nh, na).astype(np.float32)
    area = max(nl * nh, 1e-9)
    (c, length, height, angle) = rect
    r = _rect(c[0], c[1], length, height, angle).astype(np.float32)
    d = n - [t["cx"], t["cy"]]
    d = d / np.hypot(*d) if np.hypot(*d) > 0 else np.array(L._axes(float(t["angle"]))[0], float)

    def share(shift) -> float:
        moved = n0 + np.asarray(shift, np.float32)
        return float(cv2.intersectConvexConvex(r, moved)[0]) / area

    def at(s):
        return np.asarray(c, float) + d * s - n

    if share(at(0.0)) > f:
        lo, hi = 0.0, (length + height + nl + nh) / 2 + 1.0
        for _ in range(BISECT):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if share(at(mid)) > f else (lo, mid)
        s = hi
    else:
        s = 0.0
    dx, dy = (int(v) for v in np.rint(at(s)))
    bx0, by0 = max(int(np.floor(n0[:, 0].min())), 0), max(int(np.floor(n0[:, 1].min())), 0)
    bx1 = min(int(np.ceil(n0[:, 0].max())) + 1, page.shape[1])
    by1 = min(int(np.ceil(n0[:, 1].max())) + 1, page.shape[0])
    if bx1 > bx0 and by1 > by0:
        sub = page[by0:by1, bx0:bx1]
        own = _inside(sub.shape, (bx0, by0), n[0], n[1], nl, nh, na)
        _paste(win, np.where(own, sub, 255).astype(np.uint8), by0 + dy - origin[1], bx0 + dx - origin[0])
    return {"neighbour": True, "part": f, "part_got": share((dx, dy)), "n_word": int(t["n_word"])}


def sloppy_crop(page: np.ndarray, terra_word, dist: dict, pieces: list[np.ndarray], rng,
                info: dict | None = None) -> np.ndarray:
    """A sloppy crop of Terra's word `terra_word` (a row of `terra_frame`, with `with_neighbours`' columns for a
    neighbour) on `page` (see the module): one pair's offsets of `dist` (`offset_distribution`), then a line piece of
    `pieces` (uint8 patches, paper 255 off the piece, `piece_patches`) with probability `dist["line_share"]`, then the
    nearest other word's fragment with probability `dist["neighbour_share"]`, each from `rng` alone.  `info` (a dict)
    gets what was drawn: the offsets, `draw`, `window` (the page px the crop reads), `line` (and `piece`, `turns`,
    `line_at`), `neighbour` (and `part`, `part_got`, `n_word`)."""
    t = terra_word
    cx, cy, length0, height0, angle0 = (float(t[k]) for k in ("cx", "cy", "length", "height", "angle"))
    off = dist["offsets"]
    k = int(rng.integers(len(off["start"])))
    s, e, x, da, r = (float(off[n][k]) for n in OFFSETS)
    angle = angle0 + da
    (ux, uy), (vx, vy) = L._axes(angle)
    length, height = max(length0 + (e - s) * height0, 1.0), max(r * height0, 1.0)
    along, across = (s + e) * height0 / 2, x * height0
    centre = (cx + ux * along + vx * across, cy + uy * along + vy * across)
    a, shape = K.frame_of(centre, length, height, angle)
    x0, y0, x1, y1 = K._footprint(a, shape, page.shape)
    got = {"draw": k, **dict(zip(OFFSETS, (s, e, x, da, r))), "window": (x0, y0, x1, y1), "line": False,
           "neighbour": False}
    u_line, u_near = rng.random(), rng.random()
    if x1 <= x0 or y1 <= y0:                     # off the page: paper
        crop = K.cut(page, centre, length, height, angle)[0]
    else:
        win = page[y0:y1, x0:x1].copy()
        if u_line < float(dist.get("line_share") or 0.0) and len(pieces):
            got.update(_line_through(win, (x0, y0), t, pieces, rng))
        if (u_near < float(dist.get("neighbour_share") or 0.0) and int(t.get("n_word", -1)) >= 0
                and dist.get("neighbour_part")):
            got.update(_neighbour_into(win, (x0, y0), page, t, (centre, length, height, angle),
                                       dist["neighbour_part"], rng))
        crop = K.cut(win, centre, length, height, angle, origin=(x0, y0))[0]
    if info is not None:
        info.update(got)
    return crop


def sloppy_crops(terra: pd.DataFrame, dist: dict, pieces: list[np.ndarray], seed: int, quarter: int | None = None,
                 copies: int = 1, pages=None, touch=None) -> list[tuple[np.ndarray, str, dict]]:
    """`copies` sloppy crops of every word of `terra` touching neither the answered blocks nor (with `quarter`) that
    quarter, its neighbour the nearest such word, each from its own generator (`default_rng([seed, sheet, word,
    copy])`); a crop whose footprint touches either is left out.  (crop, string, meta: `sheet`, `word`, `copy`,
    `n_word`, what was drawn and the TOUCH flags), by sheet and word.  `pieces` must hold no held-out ink (a bank
    built with the same areas held out)."""
    touch = HeldAreas() if touch is None else touch
    page_of = _pages(pages)
    keep = ~terra["touch_blocks"].to_numpy(bool)
    if quarter is not None:
        keep &= ~terra[f"touch_{paths.QUARTERS[quarter]}"].to_numpy(bool)
    t = with_neighbours(terra[keep])
    out = []
    for sheet in sorted(set(t["sheet"])):
        page = page_of(sheet)
        for _, row in t[t["sheet"] == sheet].iterrows():
            for c in range(copies):
                rng = np.random.default_rng([seed, K.hash_name(sheet), int(row["word"]), c])
                info: dict = {}
                crop = sloppy_crop(page, row, dist, pieces, rng, info)
                flags = touch(sheet, info["window"], page.shape)
                if _held(flags, quarter):
                    continue
                out.append((crop, str(row["text"]), {"sheet": sheet, "word": int(row["word"]), "copy": c,
                                                     "n_word": int(row["n_word"]), **info,
                                                     **{k: bool(v) for k, v in flags.items()}}))
    return out


def piece_patches(bank, pages=None, piece_mask=None) -> list[np.ndarray]:
    """The 2000 sheets' line pieces of a `composite.Bank` (references, none from its held-out areas) as patches for
    `sloppy_crop`: each piece's box of its page, the piece's px as printed, paper elsewhere.  `piece_mask`:
    `composite.piece_mask` (for tests)."""
    piece_mask = composite.piece_mask if piece_mask is None else piece_mask
    page_of = _pages(pages)
    p = bank.pieces[paths.SERIES]
    out = []
    for j in range(len(p)):
        sd = bank.sheets[int(p.sheet[j])]
        x0, y0, x1, y1 = (int(v) for v in p.box[j])
        mask = piece_mask(sd, p.box[j], p.seed[j])
        out.append(np.where(mask, page_of(sd.sheet)[y0:y1, x0:x1], 255).astype(np.uint8))
    return out


# --- the measured offsets ---------------------------------------------------------------------------------------------

def line_ink(terra: pd.DataFrame) -> np.ndarray:
    """Per word of `terra`: the line px that are ink inside Terra's rectangle (`composite.crossings`; the line px of
    `zones.sheet_lines`, kept in `zones.LINES_DIR`)."""
    out = np.zeros(len(terra), np.int64)
    recs = textboxes.terra_records()
    for sheet in sorted(set(terra["sheet"])):
        sd = data.sheet_data(paths.SERIES, sheet, hold_out_validation=False, terra=recs, placement=True)
        sd = dataclasses.replace(sd, line_bits=zones.sheet_lines(sd, zones.LINES_DIR))
        cross = composite.crossings(sd)
        mine = np.flatnonzero(terra["sheet"].to_numpy() == sheet)
        words = terra["word"].to_numpy()[mine]
        if words.max(initial=-1) >= len(cross):
            raise ValueError(f"{sheet}: Terra's words of crops.parquet are not the sheet's ({len(cross)} words)")
        out[mine] = cross[words, 1]
    return out


def _rounded(d: dict, places: int = 6) -> dict:
    out = dict(d)
    out["offsets"] = {k: [round(v, places) for v in vs] for k, vs in d["offsets"].items()}
    out["neighbour_part"] = [round(v, places) for v in d["neighbour_part"]]
    return out


def build_offsets(out: Path = OFFSETS_FILE) -> dict:
    """Measure the offsets of today's boxes (`solver-boxes`) against Terra's words (`solver-text-crops`) for each
    held-out quarter and write them to `out`: {"quarters": {name: offset_distribution}, ...}."""
    t0 = time.perf_counter()
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    terra = terra_frame(pd.read_parquet(store.CROPS_DIR / "crops.parquet"))
    terra["line_ink"] = line_ink(terra)
    got = {name: _rounded(offset_distribution(boxes, terra, quarter=q)) for q, name in enumerate(paths.QUARTERS)}
    result = {"what": "our boxes (solver-boxes) matched to Terra's 2000 words at IoU >= 0.5, per held-out quarter "
                      "(pairs whose box is outside it and whose word touches neither it nor an answered block); "
                      "offsets in Terra's word heights (see wordread.readercrops)",
              "iou": IOU, "boxes": int(len(boxes)), "terra_words": int(len(terra)),
              "cross_px": composite.CROSS_PX, "quarters": got}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    log.info("offsets of %s pairs written to %s in %.0f s", {q: d["n"] for q, d in got.items()}, out,
             time.perf_counter() - t0)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build_offsets()
