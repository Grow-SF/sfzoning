"""The text reader's training crops beyond Terra's boxes: crops of our own word boxes, sloppy crops of Terra's words,
and masking outside the word.  The reader reads 96% of the owner's answered labels right in Terra's tight boxes and
41% in our own (`wordread.measures`): it has never practised on boxes like ours.  Nothing here changes what a stage
does: `textread.train.fit` takes these crops as `extra` (`extra_pool`), behind its off-by-default options
(`READER_OPTIONS`).  Every function that makes or measures crops for a reader takes its held-out quarter, and stops
without one.

  Terra's words   (`terra_frame`) the 2000 sheets' words from `solver-text-crops`' rows: Terra's string as the reader
                  learns it, its rectangle as `crops.rectify` cuts it (`cx`, `cy` its centre's page px index, `length`,
                  `height`, the reading `angle`), the held-out areas its crop touches (`crops.TOUCH`), and its
                  rectangle as Task 1 matches it (`c0x`..`c3y`, `textnet.boxes.word_rect`)
  matching        Task 1's (`measures`): each of our boxes (its grouped rectangle) goes to the Terra word it meets
                  best, kept at IoU >= IOU (`matches`)
  own-box crops   (`own_box_crops`) each matched box cut as the reader reads it (`crops.box_crop`), turned a half turn
                  when Terra's word reads the other way along its axis, with Terra's string as its target (ABCNet v2's
                  training on its own detections).  Kept only when neither the crop's footprint nor Terra's word
                  touches the reader's held-out quarter or an answered block.  `extra_pool(kind_weights="terra")`
                  draws them in the kinds' shares of the training quarters' Terra words (`own_draw_weights`)
  the offsets     (`offset_distribution`) how our matched boxes sit on Terra's words, pair by pair, in Terra's word
                  heights H, in our box's own frame (its reading direction the axis turned towards Terra's): our
                  trimmed box's start and end along it, each minus Terra's (`start`, `end`; Terra's word from -L/2 to
                  L/2 about its centre), its centre's offset down the letters (`cross`), its angle minus Terra's
                  (`angle`, degrees in [-90, 90]), its height over Terra's (`height_ratio`), and the largest share of
                  another Terra word's rectangle inside the box (`neighbour`, 0 for none): cutting Terra's word with
                  one pair's offsets cuts that box's own crop.  Also: the share of matched words holding line ink
                  (`line_share`: at least `composite.CROSS_PX` line px that are ink inside Terra's rectangle), the
                  thickness of that line ink (`line_thickness`, `thickness` on the sheet's line ink) and its share of
                  the word's ink (`line_cover`); how much of a held neighbour lies on the word's own rectangle
                  (`neighbour_on_word`); and the kinds' shares of the training quarters' Terra words (`kind_shares`)
  sloppy crops    (`sloppy_crop`) Terra's word cut with one matched pair's offsets, its neighbour share with them
                  (drawn together: their spread and their ties are the measured ones), as Weinman's synthesizer makes
                  its words.  With probability `line_share` a real line piece (a `composite.Bank`'s, no thicker than
                  the measured crossings' 99th percentile, `piece_patches`), turned a random number of quarter turns,
                  is ORed through a letter px of the word; a draw whose piece covers more of the word's ink than the
                  crossings' 99th percentile (`line_cover`) is drawn again, so the line rate stays `line_share`.  When
                  the pair held a neighbour, the nearest other word (`with_neighbours`) is moved (its old place made
                  paper) along the word's reading axis or across it, on its own side first, until the pair's share of
                  it lies inside the crop's rectangle, never on the word's rectangle (ON_WORD; as near as it may come
                  when the crop holds no room beside the word).  The page itself is never changed.  `sloppy_crops`
                  makes them for a reader: only of words, and with neighbours, touching neither its held-out quarter
                  nor the answered blocks, and kept only when the crop's footprint touches neither
  masking         (`mask_outside`; `crops.box_mask`, `textread.read.word_masks`) px outside a box's own px (the
                  component it was grouped from, grown by `wordboxes.INK_R`) are paper (Mask TextSpotter v3).  For
                  training with the mask, Terra's crops (`real_masks`) and the sloppy crops (`terra_masks`) are masked
                  by the components meeting Terra's rectangle (`rect_masks`; the level of the word's quarter), so the
                  reader learns and reads the same kind of crop; a word no component meets is left as cut

`build_offsets` measures the offsets for each held-out quarter (the pairs whose box is outside it and whose word
touches neither it nor an answered block) from today's boxes and writes OFFSETS_FILE
(`python -m sfzmaps.solver.wordread.readercrops`).  Seeded throughout; no stage imports this module.
"""
import collections
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
from sfzmaps.solver.textread import read, store
from sfzmaps.solver.wordread import measures

log = logging.getLogger(__name__)

IOU = 0.5
OFFSETS = ("start", "end", "cross", "angle", "height_ratio")
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
CAP = 0.99                        # the quantile of the measured crossings a pasted line may reach
LINE_TRIES = 100                  # line draws for one crop before it is left without (a guard; counted)
BISECT = 40                       # halvings of a neighbour's move
STEP = 0.5                        # px between the places a neighbour is tried along its way in
ON_WORD = 0.0                     # the most of a moved neighbour's rectangle on the word's: never on the word.  (The
#   measured `neighbour_on_word` is no guide: its 95th percentile is 1, block numbers lying wholly inside a slanted
#   code's loose Terra rectangle, not on its letters.)
KIND_NAMES = tuple(data.KINDS)    # zoning code, block number, street name, other
OFFSETS_FILE = runs.EXPERIMENTS_DIR / "wordread" / "reader" / "offsets.json"
TOUCH = store.TOUCH
mask_outside = K.mask_outside


def _need_quarter(quarter, what: str) -> int:
    if quarter is None:
        raise ValueError(f"{what} needs the reader's held-out quarter: with none it would use that quarter's words")
    return int(quarter)


# --- Terra's words ----------------------------------------------------------------------------------------------------

def _rect(cx: float, cy: float, length: float, height: float, angle: float) -> np.ndarray:
    """A word rectangle's corners (4, 2), page px index coordinates: start-top, end-top, end-bottom, start-bottom."""
    (ux, uy), (vx, vy) = L._axes(angle)
    u, v = np.array([ux, uy]) * length / 2, np.array([vx, vy]) * height / 2
    c = np.array([cx, cy], float)
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


def terra_frame(rows: pd.DataFrame) -> pd.DataFrame:
    """Terra's words on the 2000 sheets with a target, from rows of `solver-text-crops`' crops.parquet (see the
    module), positions 0..n-1; `line_ink`, `word_ink` and `line_thick` too when the rows have them."""
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
    for c in ("line_ink", "word_ink", "line_thick"):
        if c in w:
            out[c] = w[c].to_numpy()
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


def _training_words(terra: pd.DataFrame, quarter: int) -> np.ndarray:
    """Which words of `terra` touch neither `quarter` nor an answered block, bool."""
    return ~terra["touch_blocks"].to_numpy(bool) & ~terra[f"touch_{paths.QUARTERS[quarter]}"].to_numpy(bool)


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


def _held(flags, quarter: int) -> bool:
    return bool(flags["touch_blocks"]) or bool(flags[f"touch_{paths.QUARTERS[quarter]}"])


def _summary(x) -> dict:
    """`n`, `quantiles` (QUANTILES) and `q99` (the CAP quantile) of measured values (None without any)."""
    x = np.asarray(x, float)
    if not len(x):
        return {"n": 0, "quantiles": None, "q99": None}
    return {"n": int(len(x)), "quantiles": dict(zip(map(str, QUANTILES), np.quantile(x, QUANTILES).tolist())),
            "q99": float(np.quantile(x, CAP))}


def _area(r: np.ndarray) -> float:
    return float(wordboxes.poly_area(np.asarray(r, float).reshape(1, 4, 2))[0])


def _meet(a: np.ndarray, b: np.ndarray) -> float:
    """The area two convex quadrilaterals (4, 2) share."""
    return float(cv2.intersectConvexConvex(np.asarray(a, np.float32), np.asarray(b, np.float32))[0])


def offset_distribution(boxes: pd.DataFrame, terra: pd.DataFrame, *, quarter: int, iou: float = IOU) -> dict:
    """The measured spread of our matched boxes against Terra's words for the reader holding `quarter` out (see the
    module): `n` pairs, `offsets` (each a list, pair by pair, `neighbour` too), the offsets' `mean`, `sd` and
    `quantiles`, `turned` (pairs read a half turn from the axis), `line_share` (None without `line_ink`),
    `line_thickness` and `line_cover` (`_summary`; with `word_ink` and `line_thick`), `neighbour_share`,
    `neighbour_part`, `neighbour_on_word` (`_summary`) and `kind_shares`.  Left out: pairs whose word touches an
    answered block or the quarter, or whose box is in it.  ValueError without a quarter."""
    quarter = _need_quarter(quarter, "offset_distribution")
    m = matches(boxes, terra, iou)
    keep = [not _held(terra.iloc[w], quarter) and int(boxes.iloc[p]["quarter"]) != quarter
            for p, w in zip(m["box"], m["word"])]
    m = m[np.array(keep, bool)].reset_index(drop=True)
    words = m["word"].to_numpy()
    rows = [_pair_offsets(boxes.iloc[p], terra.iloc[w]) for p, w in zip(m["box"], words)]
    off = np.array(rows, float).reshape(-1, len(OFFSETS))
    out = {"iou": iou, "quarter": paths.QUARTERS[quarter], "n": int(len(m)),
           "offsets": {k: off[:, j].tolist() for j, k in enumerate(OFFSETS)},
           "mean": {k: float(off[:, j].mean()) if len(m) else None for j, k in enumerate(OFFSETS)},
           "sd": {k: float(off[:, j].std()) if len(m) else None for j, k in enumerate(OFFSETS)},
           "quantiles": {k: _summary(off[:, j])["quantiles"] for j, k in enumerate(OFFSETS)},
           "turned": int(sum(_turned(terra.iloc[w]["angle"], boxes.iloc[p]["axis"])
                             for p, w in zip(m["box"], words)))}
    out["line_share"], out["line_words"] = None, None
    out["line_thickness"], out["line_cover"] = _summary([]), _summary([])
    if "line_ink" in terra:
        crossed = terra["line_ink"].to_numpy()[words] >= composite.CROSS_PX
        out["line_share"] = float(crossed.mean()) if len(m) else None
        out["line_words"] = int(crossed.sum())
        if "word_ink" in terra and "line_thick" in terra:
            mine = words[crossed]
            out["line_thickness"] = _summary(terra["line_thick"].to_numpy(float)[mine])
            ink = terra["word_ink"].to_numpy(float)[mine]
            out["line_cover"] = _summary(terra["line_ink"].to_numpy(float)[mine][ink > 0] / ink[ink > 0])
    # each matched box's largest share of another Terra word's rectangle inside its trimmed box, and how much of that
    # word lies on the matched word's own rectangle
    part, on_word = np.zeros(len(m)), np.zeros(len(m))
    sheet_of = boxes["sheet"].astype(str).to_numpy()
    tsheet = terra["sheet"].astype(str).to_numpy()
    geo = terra[["cx", "cy", "length", "height", "angle"]].to_numpy(float)
    for sheet in sorted(set(sheet_of[m["box"]])):
        mine = np.flatnonzero(sheet_of[m["box"]] == sheet)
        others = np.flatnonzero(tsheet == sheet)
        ours = np.array([_corners(boxes.iloc[p]) - 0.5 for p in m["box"].to_numpy()[mine]], float).reshape(-1, 4, 2)
        theirs = np.array([_rect(*g) for g in geo[others]], float).reshape(-1, 4, 2)
        i, j, inter = wordboxes.pairs(ours, theirs)
        share = inter / np.maximum(wordboxes.poly_area(theirs)[j], 1e-9)
        for a, b, s in zip(i, j, share):
            k = mine[a]
            if others[b] != words[k] and s > part[k]:
                part[k] = s
                on_word[k] = _meet(theirs[b], _rect(*geo[words[k]])) / max(_area(theirs[b]), 1e-9)
    holds = part > 0
    out["offsets"]["neighbour"] = part.tolist()
    out["neighbour_share"] = float(holds.mean()) if len(m) else None
    out["neighbour_part"] = part[holds].tolist()
    out["neighbour_on_word"] = _summary(on_word[holds])
    kinds = collections.Counter(terra.loc[_training_words(terra, quarter), "kind"])
    total = sum(kinds.values())
    out["kind_shares"] = {k: kinds.get(k, 0) / total if total else 0.0 for k in KIND_NAMES}
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

def own_box_crops(boxes: pd.DataFrame, terra: pd.DataFrame, *, quarter: int, iou: float = IOU, pages=None,
                  touch=None, word_masks=None) -> list[tuple[np.ndarray, str, dict]]:
    """Our boxes matched to a Terra word at `iou`, each cut as `crops.box_crop` cuts it (turned a half turn when
    Terra's word reads the other way), with Terra's string: (crop, string, meta), by sheet and box, for the reader
    holding `quarter` out.  Left out: a box whose crop's footprint or whose word touches an answered block or the
    quarter.  `pages`: sheet -> page (read from disk when None); `touch`: `HeldAreas` (made when None); `word_masks`:
    (sheet, rows) -> {row: (bool window, origin)} (`textread.read.word_masks`), when given each meta's `mask` is the
    box's own px cut as the crop (turned with it).  meta: `sheet`, `row` (the box's index label), `word` (Terra's word
    on the sheet), `iou`, `turned`, `kind`, `quarter` (the box's) and the TOUCH flags (crop or word).  ValueError
    without a quarter."""
    quarter = _need_quarter(quarter, "own_box_crops")
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


def own_draw_weights(own: list, shares: dict) -> np.ndarray:
    """The chance each own-box crop (`own_box_crops`' triples) is drawn by so that their kinds come in `shares` (kind
    -> share, `offset_distribution`'s `kind_shares`): each kind its share, split evenly among its crops; the kinds
    with no crop give theirs to the others in proportion."""
    kinds = [str((meta or {}).get("kind")) for _, _, meta in own]
    count = collections.Counter(kinds)
    total = sum(float(shares.get(k, 0.0)) for k in count)
    if not own or total <= 0:
        raise ValueError(f"no own-box crop of a kind with a share ({dict(count)}; {shares})")
    p = np.array([float(shares.get(k, 0.0)) / total / count[k] for k in kinds])
    return p / p.sum()


def extra_pool(own: list, sloppy: list = (), kind_weights: str | None = None, kind_shares: dict | None = None,
               real_masks: dict | None = None) -> dict:
    """`textread.train.fit`'s `extra`: the own-box and sloppy crops; with `kind_weights="terra"` the own-box crops
    drawn in `kind_shares` (`own_draw_weights`; off by default: drawn alike); `real_masks` (`real_masks`) for a
    training with the mask."""
    out = {"own": list(own), "sloppy": list(sloppy)}
    if kind_weights == "terra":
        if kind_shares is None:
            raise ValueError("kind_weights='terra' needs the measured kind_shares")
        out["own_p"] = own_draw_weights(out["own"], kind_shares)
    elif kind_weights is not None:
        raise ValueError(f"kind_weights: None or 'terra', not {kind_weights!r}")
    if real_masks is not None:
        out["real_masks"] = real_masks
    return out


# --- masks of Terra's words -------------------------------------------------------------------------------------------

def _levels() -> dict:
    """The boxes' level per quarter name (`solver-boxes`' levels.json)."""
    return json.loads((wordboxes.BOXES_DIR / "levels.json").read_text())["level"]


def rect_masks(sheet: str, rects: list[tuple], page: np.ndarray | None = None, chance: np.ndarray | None = None,
               levels: dict | None = None) -> dict:
    """For word rectangles (key, cx, cy, length, height, angle, quarter) of a 2000 sheet: the components our boxes
    are made of (`read.components`, at the level of the rectangle's quarter) that hold a px of the rectangle, grown by
    wordboxes.INK_R (`read.grown`): {key: (bool window, (x0, y0))}; a rectangle no component meets has none.  `page`,
    `chance` and `levels` read from disk when None."""
    page = data.load_page(sheet) if page is None else page
    ink = page < grid.INK_BELOW
    chance = wordboxes._read_chance(sheet, ink.shape) if chance is None else chance
    levels = _levels() if levels is None else levels
    gate = wordboxes.ink_gate(ink)
    by_level: dict = {}
    for r in rects:
        by_level.setdefault(float(levels[paths.QUARTERS[int(r[6])]]), []).append(r)
    out = {}
    h, w = ink.shape
    for level in sorted(by_level):
        c = read.components(ink, chance, level, gate)
        index = {int(label): i for i, label in enumerate(c["label"])}
        for key, cx, cy, length, height, angle, _ in by_level[level]:
            corners = _rect(cx, cy, length, height, angle)
            x0, y0 = max(int(np.floor(corners[:, 0].min())), 0), max(int(np.floor(corners[:, 1].min())), 0)
            x1 = min(int(np.ceil(corners[:, 0].max())) + 1, w)
            y1 = min(int(np.ceil(corners[:, 1].max())) + 1, h)
            if x1 <= x0 or y1 <= y0:
                continue
            inside = _inside((y1 - y0, x1 - x0), (x0, y0), cx, cy, length, height, angle)
            found = [index[int(v)] for v in np.unique(c["lab"][y0:y1, x0:x1][inside]) if int(v) in index]
            if found:
                out[key] = read.grown(c["lab"], c["label"][found], c["box"][found])
    return out


def terra_masks(sheet: str, words: pd.DataFrame, **kw) -> dict:
    """`rect_masks` of Terra's words (rows of `terra_frame`), keyed by their `word`: `sloppy_crops`' `masks`."""
    rects = [(int(t["word"]), t["cx"], t["cy"], t["length"], t["height"], t["angle"], int(t["quarter"]))
             for _, t in words.iterrows()]
    return rect_masks(sheet, rects, **kw)


def real_masks(rows: pd.DataFrame, pages=None, chances=None, levels=None) -> dict:
    """The masks of the 2000 sheets' crops of `solver-text-crops` (words and non-text windows; rows of crops.parquet):
    the components meeting each crop's rectangle (`rect_masks`), cut as the crop was (`crops.rectify`), keyed by the
    row's position; a crop no component meets has none, nor has another map's.  `pages`, `chances`: sheet -> array
    (read from disk when None); `levels`: quarter name -> level."""
    page_of = _pages(pages)
    pos = np.flatnonzero((rows["series"] == paths.SERIES).to_numpy())
    sheet_of = rows["sheet"].astype(str).to_numpy()
    box = rows[["box_x0", "box_y0", "box_x1", "box_y1"]].to_numpy(float)
    geo = {int(p): ((box[p, 0] + box[p, 2] - 1) / 2, (box[p, 1] + box[p, 3] - 1) / 2,
                    float(rows["length"].iat[p]), float(rows["height"].iat[p]), float(rows["angle"].iat[p]))
           for p in pos}
    out = {}
    for sheet in sorted(set(sheet_of[pos])):
        mine = pos[sheet_of[pos] == sheet]
        rects = [(int(p), *geo[int(p)], int(rows["quarter"].iat[p])) for p in mine]
        got = rect_masks(sheet, rects, page_of(sheet), None if chances is None else chances[sheet], levels)
        for p, (win, origin) in got.items():
            cx, cy, length, height, angle = geo[p]
            out[p] = K.cut(win, (cx, cy), length, height, angle, fill=False, origin=origin)[0]
    return out


# --- sloppy crops of Terra's words ------------------------------------------------------------------------------------

def _place(dst: np.ndarray, src: np.ndarray, top: int, left: int, op) -> None:
    """dst = op(dst, src) where `src` lies on `dst` at (top, left), cut to `dst`."""
    h, w = dst.shape
    py0, px0 = max(0, -top), max(0, -left)
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(h, top + src.shape[0]), min(w, left + src.shape[1])
    if y1 > y0 and x1 > x0:
        dst[y0:y1, x0:x1] = op(dst[y0:y1, x0:x1], src[py0:py0 + y1 - y0, px0:px0 + x1 - x0])


def _inside(shape, origin, cx, cy, length, height, angle) -> np.ndarray:
    """Which px of a window (its px (0, 0) the page's px `origin`) have their centre in a word rectangle."""
    (ux, uy), (vx, vy) = L._axes(angle)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    dx, dy = xx + origin[0] - cx, yy + origin[1] - cy
    return (np.abs(dx * ux + dy * uy) <= length / 2) & (np.abs(dx * vx + dy * vy) <= height / 2)


def thickness(mask: np.ndarray) -> float:
    """The thickness of ink (bool): twice the largest distance (L2) from one of its px to paper."""
    m = np.pad(np.asarray(mask, bool), 1).view(np.uint8)
    return 2.0 * float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max()) if m.any() else 0.0


def _line_through(win, origin, t, pieces, rng, cover_cap) -> dict:
    """OR a line piece (drawn, turned a random number of quarter turns) through a letter px of Terra's word `t`: a
    px of the piece's ink on an ink px of the word inside the window (any px of its rectangle when it has none).  A
    draw whose piece px are more than `cover_cap` of the ink inside the word's rectangle is drawn again (up to
    LINE_TRIES draws)."""
    inside = _inside(win.shape, origin, t["cx"], t["cy"], t["length"], t["height"], t["angle"])
    ys, xs = np.nonzero(inside & (win < grid.INK_BELOW))
    if not len(ys):
        ys, xs = np.nonzero(inside)
    if not len(ys):
        return {}
    rejected = 0
    for _ in range(LINE_TRIES):
        k = int(rng.integers(len(ys)))
        j, turns = int(rng.integers(len(pieces))), int(rng.integers(4))
        patch = np.rot90(pieces[j], turns)
        py, px = np.nonzero(patch < 255)
        m = int(rng.integers(len(py)))
        top, left = int(ys[k] - py[m]), int(xs[k] - px[m])
        trial = win.copy()
        _place(trial, patch, top, left, np.minimum)
        on = np.zeros(win.shape, bool)
        _place(on, patch < 255, top, left, np.logical_or)
        cover = float((on & inside).sum() / max(((trial < grid.INK_BELOW) & inside).sum(), 1))
        if cover_cap is not None and cover > cover_cap:
            rejected += 1
            continue
        win[:] = trial
        return {"line": True, "piece": j, "turns": turns, "line_at": (int(xs[k] + origin[0]), int(ys[k] + origin[1])),
                "line_cover": cover, "line_rejected": rejected}
    return {"line": False, "line_gave_up": True, "line_rejected": rejected}


def _neighbour_into(win, origin, page, t, rect, f, cap) -> dict:
    """Move the nearest other word (its rectangle's px, as printed; its old place in the window made paper, off the
    word's own rectangle) along the word's reading axis or across it, on the neighbour's own side first, to the
    farthest place where a share `f` of its rectangle lies inside the crop's rectangle `rect` (centre, length, height,
    angle) and at most `cap` of it on the word's rectangle; when no direction reaches `f`, to the place of the most
    share within the cap.  To the nearest px, never past the cap."""
    n = np.array([t["n_cx"], t["n_cy"]], float)
    nl, nh, na = float(t["n_length"]), float(t["n_height"]), float(t["n_angle"])
    n0 = _rect(n[0], n[1], nl, nh, na)
    area = max(nl * nh, 1e-9)
    wc = np.array([t["cx"], t["cy"]], float)
    word = _rect(wc[0], wc[1], t["length"], t["height"], t["angle"])
    centre, length, height, angle = rect
    crop = _rect(centre[0], centre[1], length, height, angle)
    (ux, uy), (vx, vy) = L._axes(float(t["angle"]))
    real = n - wc
    pu, pv = real @ [ux, uy], real @ [vx, vy]
    along = (np.array([ux, uy]) * (1 if pu >= 0 else -1), "along")
    across = (np.array([vx, vy]) * (1 if pv >= 0 else -1), "across")
    order = [along, across] if abs(pu) >= abs(pv) else [across, along]
    order += [(-d, side) for d, side in order]
    far = (float(np.hypot(*(np.asarray(centre) - wc))) + (length + height + nl + nh + t["length"] + t["height"]) / 2
           + 2.0)

    def shares(shift) -> tuple[float, float]:
        moved = n0 + shift
        return _meet(crop, moved) / area, _meet(word, moved) / area

    best = None
    for d, side in order:
        at = lambda s, d=d: wc + d * s - n                          # noqa: E731
        lo, hi = 0.0, far                                            # the nearest place within the cap
        if shares(at(0.0))[1] > cap:
            for _ in range(BISECT):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if shares(at(mid))[1] > cap else (lo, mid)
            lo = hi
        s_cap = lo
        grid_s = np.arange(far, s_cap - STEP, -STEP)
        grid_s = np.append(grid_s[grid_s > s_cap], s_cap)
        inside = np.array([shares(at(s))[0] for s in grid_s])
        reach = np.flatnonzero(inside >= f)
        if len(reach):                                               # the farthest place holding f inside
            k = int(reach[0])
            a, b = grid_s[k], (grid_s[k - 1] if k else grid_s[k])
            for _ in range(BISECT):
                mid = (a + b) / 2
                a, b = (mid, b) if shares(at(mid))[0] >= f else (a, mid)
            best = (d, side, a)
            break
        k = int(np.argmax(inside))
        if best is None or inside[k] > best[3]:
            best = (d, side, float(grid_s[k]), float(inside[k]))
    d, side, s = best[:3]
    shift = np.rint(wc + d * s - n)
    for _ in range(int(far / STEP) + 1):                           # to the nearest px, then out past the cap if need be
        if shares(shift)[1] <= cap + 1e-9:
            break
        s += STEP
        shift = np.rint(wc + d * s - n)
    dx, dy = int(shift[0]), int(shift[1])
    bx0, by0 = max(int(np.floor(n0[:, 0].min())), 0), max(int(np.floor(n0[:, 1].min())), 0)
    bx1 = min(int(np.ceil(n0[:, 0].max())) + 1, page.shape[1])
    by1 = min(int(np.ceil(n0[:, 1].max())) + 1, page.shape[0])
    if bx1 > bx0 and by1 > by0:
        sub = page[by0:by1, bx0:bx1]
        own = _inside(sub.shape, (bx0, by0), n[0], n[1], nl, nh, na)
        off_word = ~_inside(sub.shape, (bx0, by0), wc[0], wc[1], t["length"], t["height"], t["angle"])
        _place(win, np.where(own & off_word, 0, 255).astype(np.uint8), by0 - origin[1], bx0 - origin[0],
               lambda a, b: np.where(b == 0, np.uint8(255), a))           # its old place: paper
        _place(win, np.where(own, sub, 255).astype(np.uint8), by0 + dy - origin[1], bx0 + dx - origin[0], np.minimum)
    inside_got, on_word = shares(shift)
    return {"neighbour": True, "part": float(f), "part_got": inside_got, "on_word": on_word, "side": side,
            "shift": (dx, dy), "n_word": int(t["n_word"])}


def sloppy_crop(page: np.ndarray, terra_word, dist: dict, pieces: list[np.ndarray], rng,
                info: dict | None = None) -> np.ndarray:
    """A sloppy crop of Terra's word `terra_word` (a row of `terra_frame`, with `with_neighbours`' columns for a
    neighbour) on `page` (see the module): one pair of `dist` (`offset_distribution`): its offsets, and its neighbour
    share when the word has a neighbour; a line piece of `pieces` (uint8 patches, paper 255 off the piece,
    `piece_patches`) with probability `dist["line_share"]`, its cover capped at `dist["line_cover"]["q99"]`; the
    neighbour never on the word (ON_WORD); each from `rng` alone.
    `info` (a dict) gets what was drawn: the offsets, `draw`, `frame` (centre, length, height, angle: `crops.cut`'s),
    `window` (the page px the crop reads), `line` (and `piece`, `turns`, `line_at`, `line_cover`, `line_rejected`,
    `line_gave_up`), `neighbour` (and `part`, `part_got`, `on_word`, `side`, `shift`, `n_word`)."""
    t = terra_word
    cx, cy, length0, height0, angle0 = (float(t[k]) for k in ("cx", "cy", "length", "height", "angle"))
    off = dist["offsets"]
    k = int(rng.integers(len(off["start"])))
    s, e, x, da, r = (float(off[n][k]) for n in OFFSETS)
    part = float(off["neighbour"][k]) if "neighbour" in off else 0.0
    angle = angle0 + da
    (ux, uy), (vx, vy) = L._axes(angle)
    length, height = max(length0 + (e - s) * height0, 1.0), max(r * height0, 1.0)
    along, across = (s + e) * height0 / 2, x * height0
    centre = (cx + ux * along + vx * across, cy + uy * along + vy * across)
    a, shape = K.frame_of(centre, length, height, angle)
    x0, y0, x1, y1 = K._footprint(a, shape, page.shape)
    got = {"draw": k, **dict(zip(OFFSETS, (s, e, x, da, r))), "frame": (centre, length, height, angle),
           "window": (x0, y0, x1, y1), "line": False, "line_rejected": 0, "line_gave_up": False, "neighbour": False}
    u_line = rng.random()
    if x1 <= x0 or y1 <= y0:                     # off the page: paper
        crop = K.cut(page, centre, length, height, angle)[0]
    else:
        win = page[y0:y1, x0:x1].copy()
        if u_line < float(dist.get("line_share") or 0.0) and len(pieces):
            got.update(_line_through(win, (x0, y0), t, pieces, rng, (dist.get("line_cover") or {}).get("q99")))
        if part > 0 and int(t.get("n_word", -1)) >= 0:
            got.update(_neighbour_into(win, (x0, y0), page, t, (centre, length, height, angle), part, ON_WORD))
        crop = K.cut(win, centre, length, height, angle, origin=(x0, y0))[0]
    if info is not None:
        info.update(got)
    return crop


def sloppy_crops(terra: pd.DataFrame, dist: dict, pieces: list[np.ndarray], seed: int, *, quarter: int,
                 copies: int = 1, pages=None, touch=None, masks=None) -> list[tuple[np.ndarray, str, dict]]:
    """`copies` sloppy crops of every word of `terra` touching neither the answered blocks nor `quarter`, its
    neighbour the nearest such word, each from its own generator (`default_rng([seed, sheet, word, copy])`); a crop
    whose footprint touches either is left out.  (crop, string, meta: `sheet`, `word`, `copy`, `n_word`, what was
    drawn, the TOUCH flags, and with `masks` ((sheet, words) -> {word: (bool window, origin)}, `terra_masks`) the
    `mask` of its word's components cut as the crop), by sheet and word.  `pieces` must hold no held-out ink
    (`piece_patches` checks its bank).  ValueError without a quarter."""
    quarter = _need_quarter(quarter, "sloppy_crops")
    touch = HeldAreas() if touch is None else touch
    page_of = _pages(pages)
    t = with_neighbours(terra[_training_words(terra, quarter)])
    out = []
    for sheet in sorted(set(t["sheet"])):
        page = page_of(sheet)
        mine = t[t["sheet"] == sheet]
        found = masks(sheet, mine) if masks is not None else {}
        for _, row in mine.iterrows():
            for c in range(copies):
                rng = np.random.default_rng([seed, K.hash_name(sheet), int(row["word"]), c])
                info: dict = {}
                crop = sloppy_crop(page, row, dist, pieces, rng, info)
                flags = touch(sheet, info["window"], page.shape)
                if _held(flags, quarter):
                    continue
                meta = {"sheet": sheet, "word": int(row["word"]), "copy": c, "n_word": int(row["n_word"]),
                        "kind": row["kind"], **info, **{k: bool(v) for k, v in flags.items()}}
                if int(row["word"]) in found:
                    win, origin = found[int(row["word"])]
                    meta["mask"] = K.cut(win, *info["frame"], fill=False, origin=origin)[0]
                out.append((crop, str(row["text"]), meta))
    return out


def _same_area(a, b) -> bool:
    return (isinstance(a, data.GridArea) and isinstance(b, data.GridArea) and a.col0 == b.col0 and a.row0 == b.row0
            and np.shape(a.mask) == np.shape(b.mask) and np.array_equal(a.mask, b.mask))


def piece_patches(bank, *, quarter: int, max_thickness: float, pages=None, piece_mask=None, block_area=None,
                  info: dict | None = None) -> list[np.ndarray]:
    """The 2000 sheets' line pieces of a `composite.Bank` as patches for `sloppy_crop`: each piece's box of its page,
    the piece's px as printed, paper elsewhere; a piece thicker than `max_thickness` (`thickness`; the measured
    crossings' `line_thickness` q99) is dropped.  ValueError unless every sheet of the bank holds out `quarter` and the
    answered blocks (`block_area`, read when None).  `piece_mask`: `composite.piece_mask` (for tests); `info` gets
    the counts."""
    quarter = _need_quarter(quarter, "piece_patches")
    block_area = reader_stage.load_block_area(reader_stage.held_blocks()) if block_area is None else block_area
    for sd in bank.sheets:
        held = tuple(getattr(sd, "held", ()))
        if data.Quarter(quarter) not in held or not any(_same_area(a, block_area) for a in held):
            raise ValueError(f"the bank's sheet {sd.sheet} has not held out quarter {paths.QUARTERS[quarter]} and the "
                             "answered blocks: its line pieces could come from them")
    piece_mask = composite.piece_mask if piece_mask is None else piece_mask
    page_of = _pages(pages)
    p = bank.pieces[paths.SERIES]
    out, thick = [], 0
    for j in range(len(p)):
        sd = bank.sheets[int(p.sheet[j])]
        x0, y0, x1, y1 = (int(v) for v in p.box[j])
        mask = piece_mask(sd, p.box[j], p.seed[j])
        if thickness(mask) > max_thickness:
            thick += 1
            continue
        out.append(np.where(mask, page_of(sd.sheet)[y0:y1, x0:x1], 255).astype(np.uint8))
    if info is not None:
        info.update(pieces=len(p), kept=len(out), too_thick=thick)
    return out


# --- the measured offsets ---------------------------------------------------------------------------------------------

def line_measures(terra: pd.DataFrame) -> pd.DataFrame:
    """Per word of `terra` (its rectangle as `data.SheetData.words`): `line_ink` the line px that are ink inside it
    (`composite.crossings`' second count; the line px of `zones.sheet_lines`, kept in `zones.LINES_DIR`), `word_ink`
    the ink px inside it, and `line_thick` the thickness of that line ink (twice the largest distance to paper of
    one of its px, on the sheet's whole line ink; 0 without)."""
    n_line, n_ink, thick = np.zeros(len(terra), np.int64), np.zeros(len(terra), np.int64), np.zeros(len(terra))
    recs = textboxes.terra_records()
    for sheet in sorted(set(terra["sheet"])):
        sd = data.sheet_data(paths.SERIES, sheet, hold_out_validation=False, terra=recs, placement=True)
        sd = dataclasses.replace(sd, line_bits=zones.sheet_lines(sd, zones.LINES_DIR))
        line_ink = composite._line_page(sd) & np.unpackbits(sd.ink_bits, axis=1, count=sd.width).view(bool)
        dist = cv2.distanceTransform(np.pad(line_ink, 1).view(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
        mine = np.flatnonzero(terra["sheet"].to_numpy() == sheet)
        for k in mine:
            j = int(terra["word"].iat[k])
            if j >= len(sd.words):
                raise ValueError(f"{sheet}: Terra's words of crops.parquet are not the sheet's ({len(sd.words)})")
            box = composite._clipped_box(sd, sd.words[j])
            if box is None:
                continue
            y0, x0, h, w = box
            inside = composite._rect_in(sd.words[j], *box)
            ink = inside & data._ink_window(sd, *box)
            li = ink & line_ink[y0:y0 + h, x0:x0 + w]
            n_line[k], n_ink[k] = int(li.sum()), int(ink.sum())
            thick[k] = 2.0 * float(dist[y0:y0 + h, x0:x0 + w][li].max()) if li.any() else 0.0
    return pd.DataFrame({"line_ink": n_line, "word_ink": n_ink, "line_thick": thick}, index=terra.index)


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
    terra[["line_ink", "word_ink", "line_thick"]] = line_measures(terra)
    got = {name: _rounded(offset_distribution(boxes, terra, quarter=q)) for q, name in enumerate(paths.QUARTERS)}
    result = {"what": "our boxes (solver-boxes) matched to Terra's 2000 words at IoU >= 0.5, per held-out quarter "
                      "(pairs whose box is outside it and whose word touches neither it nor an answered block); "
                      "offsets in Terra's word heights (see wordread.readercrops)",
              "iou": IOU, "boxes": int(len(boxes)), "terra_words": int(len(terra)),
              "cross_px": composite.CROSS_PX, "cap_quantile": CAP, "quarters": got}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    log.info("offsets of %s pairs written to %s in %.0f s", {q: d["n"] for q, d in got.items()}, out,
             time.perf_counter() - t0)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    build_offsets()
