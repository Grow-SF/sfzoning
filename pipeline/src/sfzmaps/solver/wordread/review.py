"""The Mission Street review page: the text reader's output on one whole 2000 sheet, against Terra's words, as one
self-contained HTML page.  The owner judges every step of the plan 2026-11-17-map-text-fields-way by it.

The sheet (`mission_sheet`): the 2000 sheet whose footprint (the warp's `warped.footprint.geojson`, map ft) holds the
longest length of DataSF's MISSION ST centreline (`streetside.load_streets`, the solver's centreline loader).

What the page shows (`page_data`, then `render`), one job per layer, the calm view first:
  the image     the page at full resolution as palette PNG tiles: paper and the ink in one neutral grey (inverted
                with the theme).  Its own layer of tiles, off at first, draws red the ink the detector calls text
                (chance >= TEXT_CHANCE; each px from the fold detector that held out its quarter); nothing is drawn
                filled over the ink.
  our boxes     (SVG, thin strokes of a constant screen width) every box read as text, its outline in the colour of
                its kind (code, block number, street name, other; a colour-blind-safe set for both themes).  The
                boxes read as no text ("fragments") are a layer of their own, off at first.
  readings      (a screen layer drawn by the page's script) one small upright pill per box, at a constant screen size,
                just outside the box: green when our reading is Terra's string, purple "ours ≠ Terra's" when it
                isn't, neutral for text Terra lacks.  Drawn greedily (codes, then wrong readings, then the rest), a
                pill that would overlap one already drawn is left out; more appear as the owner zooms in.  On by
                itself from READINGS_FROM x, or when the owner turns it on.
  layers        chips turn on Terra's words, the words Terra saw that no box matches ("missed", `miss_reason`), the
                text ink, the fragments, and the details (kind, chance, angle); a filter keeps only the boxes whose
                reading differs from Terra's.  The chips' state is kept in the viewer's browser.
  versions      `render(..., sets=)`: several box sets over one image (e.g. crude-kind, crude-map, today) behind a
                segmented control that keeps the view.
  wrong labels  the owner's answered labels the version reads wrong (`answered_labels`), each with its cause when
                given (label_causes.py), stepped through with prev/next: each centres the map on its box.
  tapping       a box or a missed word opens everything known about it (a bottom sheet on a phone, a side panel on a
                wide screen)
  about         the top-line numbers of the boxes (`measures.box_numbers`) and of the readings, with the change since
                the run before, the reasons' table and what drew the page, folded away

The reading shown for a box: when the reader's likeliest hypothesis is a code, its likeliest code given a code; for
other text, the reader's open best path in the direction the classifier chose (`textread.model.best_path`, by the
fold reader that never saw the box's quarter); for "none", nothing.  It is compared with Terra's string after
`textread.crops.norm_text`, and a printed form of the code list counts as its code (`colour`).

Why a word was missed (`miss_reason`): from the detector's own chance on the word's ink, at the level its quarter's
boxes were grouped at (`wordboxes`: the connected px at or above the level within INK_R px of ink, the components).
The component holding most of its above-level ink is its main component.
  outside_map       its centre is off the sheet's footprint or the mapped area: a box there is dropped
  below_level       less than BELOW_SHARE of its ink reaches the level (or it has no ink)
  merged_with_word  the main component also holds another Terra word: MERGE_PART or more of that word's ink at the
                    level outside this word's rectangle, or most of that word's own above-level ink
  merged_into_line  the main component, holding no other word, is more than LINE_RATIO times the word's length
  split             its above-level ink falls in two or more components, each holding SPLIT_PART of it or more
  wrong_size        the rest: one component holds it, alone and not a line, but its box doesn't fit the word
in that order (the controller's ruling: a box never forming off the map or below the level comes first; a merge
matters only once the ink was called text).  The details give the highest and the mean chance on its
ink, the share of its ink at or above the level, the main component's length and ink px, the other words it holds,
and the reader's reading at Terra's box when it is given.

The green/purple comparison folds a trailing "." and runs of spaces (`colour`), so "ST" is Terra's "ST.".

For boxes that are not grouped from components of the chance (the word heads' boxes, Task 8), `page_data` takes
`comps`, the components the reasons read in place of the chance's at each level (e.g. the px of each box,
`box_components`), with the `reason_text` that says so; `visible`, the px drawn as they are (the rest of the page faded:
paper and ink in FADED's colours, under no box or missed word); and `notes`, text written on the page (e.g. over a
faded quarter).  Left out, the page is drawn as before, byte for byte.
"""
import base64
import html
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely
from PIL import Image

from sfzmaps import config
from sfzmaps.solver import grid, page, paths, streetside, wordboxes
from sfzmaps.solver.textnet import data
from sfzmaps.solver.textread import crops, store
from sfzmaps.solver.wordread import measures

MISSION = "MISSION ST"
TEXT_CHANCE = 0.5          # red: the ink the detector calls text
MATCH_IOU = 0.5            # the page's matching level (the panels give PANEL_IOU too)
PANEL_IOU = (0.5, 0.3)
BELOW_SHARE = 0.5          # a word with less of its ink at or above the level is below it
LINE_RATIO = 3.0           # a component this many times a word's length is a line
SPLIT_PART = measures.SPLIT_PART
MERGE_PART = 0.1           # a component holds another word when it holds this share of that word's ink
REASONS = ("outside_map", "below_level", "merged_with_word", "merged_into_line", "split", "wrong_size")
REASON_SHORT = {"merged_with_word": "merged with a word", "merged_into_line": "merged into a line",
                "outside_map": "off the map", "below_level": "below level", "split": "split",
                "wrong_size": "wrong size"}
REASON_TEXT = {
    "merged_with_word": "its ink joined a component that also holds another Terra word",
    "merged_into_line": f"its ink joined a component more than {LINE_RATIO:g}x the word's length, holding no other "
                        "word (a line read as text)",
    "outside_map": "its centre is off the sheet's footprint or the mapped area, where boxes are dropped",
    "below_level": "most of its ink is below the level its quarter's boxes are grouped at",
    "split": "its ink at the level fell into two or more components",
    "wrong_size": "one component holds it, alone and not a line, but its box doesn't fit the word",
}
KIND_SHORT = {"zoning": "code", "block": "block", "street": "street", "other": "other"}
KIND_CLASSES = ("code", "block", "street", "other")     # the outlines' colours (CSS classes k-<kind>)
SPLIT_KIND = {"zoning code": "code", "block number": "block", "street name": "street", "other": "other"}
CAUSE_NAMES = {"merged": "merged", "partial": "partial", "clean": "misread", "no_box": "no box", "other": "other"}
TILE = 2048                # px: the image is cut into tiles at most this big each way
READINGS_FROM = 2.0        # the readings come on by themselves from this zoom (screen px per page px) up
MAX_SCALE = 24.0
BAND = 400                 # px: the wrong labels are listed in bands this tall down the page, left to right in each

# the image's palette (RGB), by index: `page_data`'s pixels.  The page draws the paper's tints as paper and the text
# ink as the other ink in its base tiles, and the text ink red in a layer of its own (`render`)
COLOURS = {"paper": (255, 255, 255), "green_paper": (201, 234, 211), "yellow_paper": (250, 236, 166),
           "both_paper": (198, 220, 145), "other_ink": (140, 140, 140), "text_ink": (221, 34, 34)}
INDEX = {name: k for k, name in enumerate(COLOURS)}
FADED = {"faded_paper": (238, 238, 234), "faded_ink": (206, 206, 202)}     # `page_data(..., visible=)`: the rest
FADED_INDEX = {name: len(COLOURS) + k for k, name in enumerate(FADED)}
NOTE = "#8a8a84"           # the overlay's notes (`page_data(..., notes=)`)


# --- the sheet ------------------------------------------------------------------------------------------------------

def sheet_footprint(sheet: str):
    """A 2000 sheet's footprint (map ft), as the warp wrote it."""
    return shapely.from_geojson((data.warp_dir(sheet) / "warped.footprint.geojson").read_text())


def mission_lengths(streets=None, footprints: dict | None = None) -> dict[str, float]:
    """The length (ft) of the MISSION ST centreline inside each sheet's footprint.  `streets`: DataSF's centrelines
    (`streetside.load_streets` when None); `footprints`: sheet -> footprint (every placed 2000 sheet's when None)."""
    streets = streetside.load_streets() if streets is None else streets
    line = shapely.union_all(streets.geometry[(streets["streetname"] == MISSION).to_numpy()].to_numpy())
    if footprints is None:
        footprints = {s: sheet_footprint(s) for s in paths.sheet_names()}
    return {s: float(shapely.intersection(fp, line).length) for s, fp in footprints.items()}


def mission_sheet(streets=None, footprints: dict | None = None) -> str:
    """The sheet holding the longest length of MISSION ST (`mission_lengths`; ties: the first by name)."""
    got = mission_lengths(streets, footprints)
    return max(sorted(got), key=lambda s: got[s])


def on_the_map(sheet: str, xy: np.ndarray) -> np.ndarray:
    """Whether each page point (n, 2) is placed where a box is kept (`prints._place`): inside the sheet's footprint
    and on the mapped area."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if not len(xy):
        return np.zeros(0, bool)
    world = data.load_transform(sheet).to_world(xy)
    inside = shapely.contains(sheet_footprint(sheet), shapely.points(world))
    mapped = page.load_sheets() > 0
    cr = np.floor(grid.to_grid(world)).astype(np.int64)
    h, w = mapped.shape
    on = (cr[:, 1] >= 0) & (cr[:, 1] < h) & (cr[:, 0] >= 0) & (cr[:, 0] < w)
    on[on] = mapped[cr[on, 1], cr[on, 0]]
    return inside & on


# --- the components at a level, and why a word was missed --------------------------------------------------------

def level_threshold(level: float) -> int:
    """The uint8 chance (chance x 255) at or above which a px is at `level`, as `wordboxes.group` rounds it."""
    return int(np.ceil(level * 255 - 1e-9))


@dataclass
class Components:
    """A page's components at a level: `labels` int32 (h, w), 0 off every component; per label (index 0 unused):
    `length`, the extent of its px along its chance-weighted major axis (px, as `wordboxes.group` measures a box),
    `ink_px`, and `bbox` (x0, y0, x1, y1: its px's first and last column and row)."""
    labels: np.ndarray
    length: np.ndarray
    ink_px: np.ndarray
    bbox: np.ndarray

    @classmethod
    def of_labels(cls, labels: np.ndarray, chance: np.ndarray, ink: np.ndarray) -> "Components":
        labels = np.asarray(labels)
        n = int(labels.max()) + 1 if labels.size else 1
        length, ink_px, bbox = np.zeros(n), np.zeros(n, np.int64), np.zeros((n, 4), np.int64)
        ys, xs = np.nonzero(labels)
        if not len(ys):
            return cls(labels, length, ink_px, bbox)
        lab = labels[ys, xs].astype(np.int64)
        w = np.maximum(chance[ys, xs] / 255.0, 1e-3)
        x, y = xs + 0.5, ys + 0.5
        m = np.bincount(lab, w, n)
        m[m == 0] = 1.0
        cx, cy = np.bincount(lab, w * x, n) / m, np.bincount(lab, w * y, n) / m
        rx, ry = x - cx[lab], y - cy[lab]
        sxx, syy, sxy = (np.bincount(lab, w * a, n) / m for a in (rx * rx, ry * ry, rx * ry))
        th = 0.5 * np.arctan2(2 * sxy, sxx - syy)
        p = rx * np.cos(th)[lab] + ry * np.sin(th)[lab]
        order = np.argsort(lab, kind="stable")
        ls, ps = lab[order], p[order]
        start = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]])
        ids = ls[start]
        length[ids] = np.maximum.reduceat(ps, start) - np.minimum.reduceat(ps, start) + 1.0
        ink_px[:] = np.bincount(lab, ink[ys, xs].astype(np.int64), n)
        xo, yo = xs[order], ys[order]
        bbox[ids] = np.stack([np.minimum.reduceat(xo, start), np.minimum.reduceat(yo, start),
                              np.maximum.reduceat(xo, start), np.maximum.reduceat(yo, start)], 1)
        return cls(labels, length, ink_px, bbox)


def box_components(boxes: pd.DataFrame, chance: np.ndarray, ink: np.ndarray) -> Components:
    """The components of boxes that are not grouped from the chance (the word heads' boxes): each box's px (its
    grouped rectangle `g`, filled as `cv2.fillPoly` fills it), later boxes over earlier ones, numbered 1, 2, ... in
    the order of `boxes`."""
    labels = np.zeros(ink.shape, np.int32)
    g = boxes[measures.G_COLS].to_numpy(float).reshape(-1, 4, 2)
    for k, poly in enumerate(g, 1):
        cv2.fillPoly(labels, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], k)
    return Components.of_labels(labels, chance, ink)


def components(chance: np.ndarray, ink: np.ndarray, level: float, gate: np.ndarray | None = None) -> Components:
    """The components (8-connected) of the px at or above `level` in `chance` (uint8, chance x 255) and within
    `wordboxes.INK_R` px of `ink` (`gate`, made from the ink when None): the px a box is grouped from."""
    gate = wordboxes.ink_gate(ink) if gate is None else gate
    mask = np.ascontiguousarray((chance >= level_threshold(level)) & gate)
    _, labels = cv2.connectedComponents(mask.view(np.uint8), connectivity=8)
    return Components.of_labels(labels, chance, ink)


def _rect_mask(rect: np.ndarray, shape) -> tuple[np.ndarray, int, int]:
    """A quadrilateral's px (filled as `cv2.fillPoly` fills it, corners rounded) in the window of the page holding
    it: the mask, its top row and its left column."""
    pts = np.round(np.asarray(rect, float).reshape(-1, 2)).astype(np.int32)
    x0, y0 = np.maximum(pts.min(0), 0)
    x1, y1 = np.minimum(pts.max(0) + 1, [shape[1], shape[0]])
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), bool), 0, 0
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [pts - [x0, y0]], 1)
    return m.astype(bool), int(y0), int(x0)


def _poly_in(rect: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """A quadrilateral's px (as `_rect_mask` fills it) in the h x w window whose top-left px is (y0, x0)."""
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(rect, float).reshape(-1, 2)).astype(np.int32) - [x0, y0]], 1)
    return m.astype(bool)


def _held_words(rect, main: int, comps: "Components", chance, ink, thr: int, others, names) -> list:
    """The names of the `others` (rects (m, 4, 2)) whose ink at or above the level, outside `rect`, lies MERGE_PART or
    more in component `main`."""
    if not main or others is None or not len(others):
        return []
    o = np.asarray(others, float).reshape(-1, 4, 2)
    x0, y0, x1, y1 = comps.bbox[main]
    lo, hi = o.min(1), o.max(1)
    out = []
    for k in np.flatnonzero((lo[:, 0] <= x1 + 1) & (hi[:, 0] >= x0) & (lo[:, 1] <= y1 + 1) & (hi[:, 1] >= y0)):
        m, oy, ox = _rect_mask(o[k], ink.shape)
        if not m.size:
            continue
        win = (slice(oy, oy + m.shape[0]), slice(ox, ox + m.shape[1]))
        theirs = m & ink[win] & ~_poly_in(rect, oy, ox, *m.shape)
        n = int(theirs.sum())
        held = theirs & (chance[win] >= thr) & (comps.labels[win] == main)
        if n and held.sum() >= MERGE_PART * n:
            out.append(names[k] if names is not None else int(k))
    return out


def miss_reason(word_rect, chance: np.ndarray, ink: np.ndarray, components, level: float, *, on_map: bool = True,
                reading: dict | None = None, others=None, other_names=None, shares=()) -> tuple[str, dict]:
    """Why a Terra word (`word_rect`: its 4 corners in page px, the first edge along its reading direction) has no
    box: one of REASONS (see the module) and the details.  `chance` uint8 (chance x 255) and `ink` bool, the page's;
    `components` the page's `Components` at `level` (or their bare label image); `on_map` whether its centre is
    placed where boxes are kept; `reading` the reader's reading at Terra's box, copied into the details; `others`
    the page's other Terra words (rects (m, 4, 2)) and `other_names` their names, to find the words its main component
    holds (`merged_with`); `shares` the names of the other words whose own main component is this one's
    (`miss_reasons` finds them)."""
    comps = components if isinstance(components, Components) else Components.of_labels(components, chance, ink)
    rect = np.asarray(word_rect, float).reshape(4, 2)
    length = float(np.hypot(*(rect[1] - rect[0])))
    height = float(np.hypot(*(rect[3] - rect[0])))
    inside, y0, x0 = _rect_mask(rect, ink.shape)
    win = (slice(y0, y0 + inside.shape[0]), slice(x0, x0 + inside.shape[1]))
    word_ink = inside & ink[win]
    c = chance[win][word_ink]
    thr = level_threshold(level)
    above = c >= thr
    n_ink, n_above = int(word_ink.sum()), int(above.sum())
    labs = comps.labels[win][word_ink][above]
    ids, counts = np.unique(labs[labs > 0], return_counts=True)
    main = int(ids[np.argmax(counts)]) if len(ids) else 0
    details = {"max_chance": float(c.max()) / 255 if n_ink else 0.0,
               "mean_chance": float(c.mean()) / 255 if n_ink else 0.0,
               "share_above": n_above / n_ink if n_ink else 0.0, "ink_px": n_ink, "level": float(level),
               "word_length_px": length, "word_height_px": height, "components": len(ids), "component": main,
               "component_px": float(comps.length[main]) if main else 0.0,
               "component_ink_px": int(comps.ink_px[main]) if main else 0,
               "component_share": float(counts.max()) / n_above if len(ids) else 0.0}
    held = _held_words(rect, main, comps, chance, ink, thr, others, other_names)
    shares = list(shares) if main else []
    details["merged_with"] = (held + [s for s in shares if s not in held])[:5]
    details["shares_component_with"] = shares[:5]
    details["on_map"] = bool(on_map)
    if reading is not None:
        details["reading"] = reading
    if not on_map:
        return "outside_map", details
    if not n_ink or details["share_above"] < BELOW_SHARE:
        return "below_level", details
    if details["merged_with"]:
        return "merged_with_word", details
    if details["component_px"] > LINE_RATIO * length:
        return "merged_into_line", details
    if int((counts >= SPLIT_PART * n_above).sum()) >= 2:
        return "split", details
    return "wrong_size", details


def miss_reasons(terra: pd.DataFrame, chance: np.ndarray, ink: np.ndarray, levels: dict, on_map, words,
                 terra_reads: dict | None = None, comps: "Components | None" = None) -> dict[int, tuple[str, dict]]:
    """`miss_reason` of each of Terra's words at positions `words` (of `terra`; its quarter's level from `levels`,
    quarter name -> level), the page's components found once per level (`comps`, when given, at every level),
    every other word of `terra` the `others` and, of those, the words whose own main component is this word's its
    `shares`."""
    gate = wordboxes.ink_gate(ink)
    found: dict[float, Components] = {}
    rects = terra[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2)
    names = [str(s) for s in terra["text"]]
    level = [float(levels[paths.QUARTERS[int(q)]]) for q in terra["quarter"]]
    held_by: dict[tuple, list[int]] = {}
    main = []
    for t in range(len(terra)):
        if level[t] not in found:
            found[level[t]] = components(chance, ink, level[t], gate) if comps is None else comps
        main.append(miss_reason(rects[t], chance, ink, found[level[t]], level[t])[1]["component"])
        if main[t]:
            held_by.setdefault((level[t], main[t]), []).append(t)
    out = {}
    for t in sorted({int(t) for t in words}):
        shares = [names[o] for o in held_by.get((level[t], main[t]), []) if o != t] if main[t] else []
        keep = np.arange(len(terra)) != t
        out[t] = miss_reason(rects[t], chance, ink, found[level[t]], level[t], on_map=bool(on_map[t]),
                             reading=(terra_reads or {}).get(t), others=rects[keep],
                             other_names=[n for n, k in zip(names, keep) if k], shares=shares)
    return out


# --- the readings -----------------------------------------------------------------------------------------------------

def _norm(s) -> str:
    """A string as the reader learns it (`crops.norm_text`), or upper case and single spaces when it holds a character
    outside the reader's alphabet; "" for None."""
    if s is None:
        return ""
    t = crops.norm_text(s)
    return t if t is not None else " ".join(str(s).replace("\u200b", "").upper().split())


def _fold(s) -> str:
    """`_norm`, then a trailing "." dropped and runs of spaces made one (the green/purple comparison only)."""
    return " ".join(_norm(s).rstrip(".").split())


def colour(ours: str, terra: str | None, forms: dict | None = None) -> str:
    """"green" when our reading is Terra's string (both `_fold`ed; with `forms`, printed form -> code, a printed form
    counts as its code), "purple" when it isn't, "none" with no Terra word."""
    if terra is None:
        return "none"
    a, b = _fold(ours), _fold(terra)
    code = forms.get(_norm(terra), forms.get(b)) if forms else None
    if a == b or (code is not None and _fold(code) == a):
        return "green"
    return "purple"


def shown(read: dict | None) -> str:
    """The reading the page shows for a box (see the module)."""
    if not read or read.get("top", "none") == "none":
        return ""
    closed = read.get("closed") or {}
    if read["top"] in closed:
        return min(closed.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return str(read.get("text") or "")


def reads_for_page(folder: Path | None = None) -> dict:
    """`measures.reads_from_store` of a text-read folder (`store.READ_DIR` when None), each reading with what the page
    shows too: `text` (the open best path in the classifier's direction), `p_180`, `best_0`, `best_180`, and `open`
    (the open hypotheses' mass: each code, "other", "none")."""
    folder = folder or store.READ_DIR
    out = measures.reads_from_store(folder)
    r = pd.read_parquet(folder / "reads.parquet")
    with np.load(folder / "reads.npz") as z:
        post, form_code, codes = z["post"], [str(c) for c in z["form_code"]], [str(c) for c in z["codes"]]
    code_of = np.array([codes.index(c) for c in form_code], np.int64)
    by_code = np.zeros((len(post), len(codes)))
    for k in range(len(codes)):
        by_code[:, k] = post[:, :-2][:, code_of == k].sum(1)
    for k, row in enumerate(r.itertuples(index=False)):
        q = float(row.p_classifier)
        best_0, best_180 = getattr(row, "best_0", None), getattr(row, "best_180", None)
        out[int(row.row)].update({
            "p_180": q, "best_0": best_0, "best_180": best_180,
            "text": (best_180 if q > 0.5 else best_0) if best_0 is not None else None,
            "open": {**dict(zip(codes, by_code[k].tolist())), "other": float(post[k, -2]), "none": float(post[k, -1])}})
    return out


def terra_on_sheet(sheet: str) -> pd.DataFrame:
    """Terra's words on a 2000 sheet (`measures.terra_words`), with each word's axis box (box_x0..box_y1), Terra's
    reading angle (`angle`) and direction."""
    from sfzmaps.solver.textnet import boxes as textboxes
    t = measures.terra_words([sheet])
    found = textboxes.terra_boxes(paths.SERIES, sheet)
    if len(found) != len(t):
        raise ValueError(f"{sheet}: {len(found)} Terra boxes but {len(t)} words")
    for k, name in enumerate(("box_x0", "box_y0", "box_x1", "box_y1")):
        t[name] = [float(b["box"][k]) for b in found]
    t["angle"] = [float(b.get("angle_deg") or 0.0) for b in found]
    t["direction"] = [b["direction"] for b in found]
    return t


def terra_box_reads(sheet: str, terra: pd.DataFrame, words, readers: Path | None = None) -> dict[int, dict]:
    """The reading at Terra's box of each word at positions `words` (of `terra`, from `terra_on_sheet`): its crop
    (`crops.rectify` at Terra's angle, a slanted word's height floored as the reader's crops are) read as cut and a
    half turn by the fold reader of the word's quarter (the one that never saw it; `readers`, `store.TRAIN_DIR` when
    None), weighted by that quarter's 0/180 classifier, as `textread.read` reads the answered labels at Terra's box.
    On the CPU, one thread.  {position: {"top", "p_top", "code", "p_code", "text", "p_180"}}."""
    import torch

    from sfzmaps.solver.textread import model, read
    readers = readers or store.TRAIN_DIR
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    codes = read.CodeList.load()
    floors = json.loads((store.CROPS_DIR / "summary.json").read_text())["crops"]["slanted_height_floors"]
    pg = data.load_page(sheet)
    nets: dict[str, tuple] = {}
    out = {}
    for t in words:
        row = terra.iloc[int(t)]
        q = paths.QUARTERS[int(row["quarter"])]
        if q not in nets:
            nets[q] = (model.load(readers / f"{q}.pt"), model.load(readers / f"{q}_direction.pt", "direction"))
        box = [row["box_x0"], row["box_y0"], row["box_x1"], row["box_y1"]]
        crop = crops.rectify(pg, box, float(row["angle"]), floor=float(floors.get(row["kind"], 0.0)))[0]
        got = read.read_one(*nets[q], crop, codes.forms)
        w = read.weight(got)
        c = read.combine(got, w, codes)
        hyp = read.open_by_code(c["post"], codes)
        top = min(hyp.items(), key=lambda kv: (-kv[1], kv[0]))
        k = int(np.argmax(c["log_closed"]))
        out[int(t)] = {"top": top[0], "p_top": float(top[1]), "code": codes.codes[k],
                       "p_code": float(np.exp(c["log_closed"][k])), "text": got["best"][1 if w > 0.5 else 0],
                       "p_180": w}
    return out


def load_levels(path: Path | None = None) -> dict[str, float]:
    """Each quarter's grouping level (`wordboxes`' levels.json)."""
    return json.loads((path or wordboxes.BOXES_DIR / "levels.json").read_text())["level"]


def load_forms() -> dict[str, str]:
    """The code list's printed forms: form -> code (the reader's prior.json)."""
    return json.loads((store.CROPS_DIR / "prior.json").read_text())["forms"]


# --- what the page draws ------------------------------------------------------------------------------------------

def _fill(shape, polys) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    if len(polys):
        cv2.fillPoly(m, [np.round(np.asarray(p, float).reshape(-1, 2)).astype(np.int32) for p in polys], 1)
    return m.astype(bool)


def palette_image(ink: np.ndarray, text: np.ndarray, green, yellow) -> np.ndarray:
    """The image's palette indices (`COLOURS`' order): the paper tinted inside the `green` and the `yellow`
    polygons, then the ink over it (red where `text`), so no shading covers ink."""
    g, y = _fill(ink.shape, green), _fill(ink.shape, yellow)
    idx = np.full(ink.shape, INDEX["paper"], np.uint8)
    idx[g] = INDEX["green_paper"]
    idx[y] = INDEX["yellow_paper"]
    idx[g & y] = INDEX["both_paper"]
    idx[ink] = INDEX["other_ink"]
    idx[ink & text] = INDEX["text_ink"]
    return idx


def _r(v, k=3):
    return None if v is None or not np.isfinite(float(v)) else round(float(v), k)


def _reading_angle(rect: np.ndarray) -> float:
    dx, dy = rect[1] - rect[0]
    return float(np.rad2deg(np.arctan2(-dy, dx)) % 360.0)


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else dict(o)))


def text_ink_of(ink: np.ndarray, chance: np.ndarray) -> np.ndarray:
    """The ink the page draws red: ink whose chance (uint8, chance x 255) is TEXT_CHANCE or more."""
    return ink & (chance >= level_threshold(TEXT_CHANCE))


def matching(boxes: pd.DataFrame, terra: pd.DataFrame):
    """`measures`' matching of our boxes (positions) and Terra's words (positions): each box's best word (-1: none)
    and the IoU with it; each word's best box (-1: none) and the IoU with it."""
    ti, bj, v = measures._overlaps(boxes, terra)
    word, best = measures._best_word(ti, bj, v, len(boxes))
    t_best, t_box = np.zeros(len(terra)), np.full(len(terra), -1)
    for k in np.lexsort((bj, v)):               # ascending IoU: a word's highest is written last
        t_best[ti[k]], t_box[ti[k]] = v[k], bj[k]
    return word, best, t_box, t_best


def matched_at(word: np.ndarray, best: np.ndarray, n_words: int, iou: float):
    """At `iou`: whether each box is matched (to its best word), and whether each of the `n_words` words is."""
    won = (best >= iou) & (word >= 0)
    m = np.zeros(n_words, bool)
    m[word[won]] = True
    return won, m


def missed_words(boxes: pd.DataFrame, terra: pd.DataFrame, iou: float = MATCH_IOU) -> np.ndarray:
    """The positions of Terra's words that no box matches at `iou` (`measures`' rule)."""
    word, best, _, _ = matching(boxes, terra)
    return np.flatnonzero(~matched_at(word, best, len(terra), iou)[1])


def _cause_note(c: dict) -> str:
    """A wrong label's cause in a few words (label_causes.py's row): the detail, where the other word lies, and it."""
    bits = [str(c.get("detail") or "")]
    if c.get("placement") and isinstance(c["placement"], str):
        bits.append(c["placement"])
    if c.get("other_text") and isinstance(c["other_text"], str):
        bits.append(f'with “{c["other_text"]}”')
    return ", ".join(b for b in bits if b)


def answered_labels(boxes: pd.DataFrame, reads: dict, answers: pd.DataFrame, causes: pd.DataFrame | None = None,
                    band: int = BAND) -> list[dict]:
    """The owner's answered labels of one sheet (`answers`), each with the box its point goes to and whether our
    likeliest code there is the answer (`measures.label_right`, the headline's rule: no box is wrong), our code, and,
    when `causes` (label_causes.py's rows: point_x, point_y, truth, cause, detail, placement, other_text) holds the
    label's point and answer, its cause.  In bands `band` px tall down the page, left to right in each."""
    right = measures.label_right(boxes, reads, answers).set_index("label")
    named = {}
    if causes is not None and len(causes):
        for c in causes.to_dict("records"):
            named[(round(float(c["point_x"]), 1), round(float(c["point_y"]), 1), str(c["truth"]))] = c
    out = []
    for k, a in enumerate(answers.to_dict("records")):
        if k not in right.index:
            continue
        row = right.loc[k, "row"]
        has = not (isinstance(row, (int, np.integer)) and int(row) == -1)
        c = named.get((round(float(a["point_x"]), 1), round(float(a["point_y"]), 1), str(a["truth"])))
        out.append({"k": k, "x": _r(a["point_x"], 1), "y": _r(a["point_y"], 1), "truth": str(a["truth"]),
                    "terra": None if a.get("terra") is None else str(a["terra"]),
                    "box": f"b{row}" if has else None,
                    "read": measures._closed_best(reads.get(row)) if has else None,
                    "right": bool(right.loc[k, "right"]),
                    "cause": c["cause"] if c else None, "note": _cause_note(c) if c else ""})
    out.sort(key=lambda r: (int(r["y"] // band), r["x"], r["k"]))
    return out


def page_data(sheet: str, boxes: pd.DataFrame, reads: dict, terra: pd.DataFrame, chance: np.ndarray,
              before: dict | None = None, *, ink: np.ndarray | None = None, levels: dict | None = None,
              on_map: np.ndarray | None = None, terra_reads: dict | None = None, answers: pd.DataFrame | None = None,
              forms: dict | None = None, models=(), run: str = "", iou: float = MATCH_IOU,
              comps: Components | None = None, visible: np.ndarray | None = None, reason_text: dict | None = None,
              notes=(), causes: pd.DataFrame | None = None) -> dict:
    """Everything the page draws for `sheet`: `boxes` our boxes (wordboxes' columns; the index is the row `reads` is
    keyed by), `reads` their readings (`reads_for_page`), `terra` Terra's words (`measures.terra_words`' columns),
    `chance` the sheet's chance of text (uint8, chance x 255), `before` the numbers of the run before (None: none).
    Read from disk when None: `ink` (the page < grid.INK_BELOW), `levels` (quarter name -> grouping level), `on_map`
    (each Terra word's centre placed where boxes are kept), `forms` (printed form -> code).  `terra_reads`: Terra word
    position -> the reading at Terra's box; `answers`: the owner's answered labels on the sheet
    (`measures.answers_from_store`' columns); `models`: lines naming the models that drew it.  `comps`, `visible`,
    `reason_text` and `notes`: see the module (None, None, None and none: the page as before).  `causes`: the wrong
    labels' causes (`answered_labels`).

    Returns {"sheet", "run", "shape", "pixels" (`palette_image`), "text_ink_px", "boxes", "missed", "terra_words"
    (every Terra word's outline), "labels" (`answered_labels`; none without answers), "numbers", "delta"
    (`measures.compare` with `before`, or None), "models", "iou", "levels"}."""
    boxes = boxes[boxes["sheet"] == sheet] if "sheet" in boxes.columns else boxes
    terra = terra[terra["sheet"] == sheet].reset_index(drop=True)
    if ink is None:
        ink = data.load_page(sheet) < grid.INK_BELOW
    levels = load_levels() if levels is None else levels
    forms = load_forms() if forms is None else forms
    rects = terra[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2)
    if on_map is None:
        on_map = on_the_map(sheet, rects.mean(1))
    terra_reads = terra_reads or {}
    if answers is not None:
        answers = answers[answers["sheet"] == sheet].reset_index(drop=True)

    word, best, t_box, t_best = matching(boxes, terra)
    levels_used = sorted({iou, *PANEL_IOU})
    won, matched = matched_at(word, best, len(terra), iou)
    _, matched_hi = matched_at(word, best, len(terra), max(levels_used))
    reasons = miss_reasons(terra, chance, ink, levels, on_map, np.flatnonzero(~matched_hi), terra_reads, comps)

    # our boxes
    kinds = [k for k in KIND_SHORT if f"p_{k}" in boxes.columns]
    recs, green_polys = [], []
    reads_count: dict[str, dict] = {"all": {"matched": 0, "same": 0, "differs": 0}}
    g = boxes[measures.G_COLS].to_numpy(float).reshape(-1, 4, 2)
    for pos, (row, b) in enumerate(boxes.iterrows()):
        r = reads.get(row) or {}
        top = r.get("top", "none")
        text = shown(r)
        w = int(word[pos])
        t_text = terra["text"].iloc[w] if w >= 0 else None
        if won[pos]:
            status, col = "matched", colour(text, t_text, forms)
            kind = terra["kind"].iloc[w]
            for key in ("all", kind):
                c = reads_count.setdefault(key, {"matched": 0, "same": 0, "differs": 0})
                c["matched"] += 1
                c["same" if col == "green" else "differs"] += 1
        else:
            status, col = ("none" if top == "none" else "text"), "none"
            if status == "text":
                green_polys.append(g[pos])
        p_180 = float(r.get("p_180") or 0.0)
        axis = float(b.get("axis", 0.0))
        pk = {KIND_SHORT[k]: _r(b[f"p_{k}"]) for k in kinds}
        hyp = sorted((r.get("open") or {}).items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        recs.append({"id": f"b{row}", "row": int(row) if isinstance(row, (int, np.integer)) else str(row),
                     "quarter": paths.QUARTERS[int(b["quarter"])] if "quarter" in b else None,
                     "level": _r(b.get("level")), "poly": np.round(g[pos], 1).tolist(),
                     "angle": round((axis + (180.0 if p_180 > 0.5 else 0.0)) % 360.0, 1), "axis": round(axis, 1),
                     "length": _r(b.get("length"), 1), "height": _r(b.get("height"), 1),
                     "kind": max(pk, key=pk.get) if pk else None, "p_kind": pk, "chance": _r(b.get("mean_chance")),
                     "ink_px": int(b["ink_px"]) if "ink_px" in b else None, "top": top, "shown": text,
                     "top3": [[h, _r(p)] for h, p in hyp], "best_0": r.get("best_0"), "best_180": r.get("best_180"),
                     "p_180": _r(p_180), "status": status, "colour": col, "terra": t_text,
                     "terra_kind": terra["kind"].iloc[w] if w >= 0 else None, "iou": _r(best[pos])})

    # Terra's words we missed
    missed, yellow_polys = [], []
    for t in np.flatnonzero(~matched):
        reason, det = reasons[int(t)]
        det = {k: (_r(v_) if isinstance(v_, float) else v_) for k, v_ in det.items()}
        bb = int(t_box[t])
        missed.append({"id": f"t{t}", "pos": int(t), "text": str(terra["text"].iloc[t]), "kind": terra["kind"].iloc[t],
                       "quarter": paths.QUARTERS[int(terra["quarter"].iloc[t])], "poly": np.round(rects[t], 1).tolist(),
                       "angle": round(_reading_angle(rects[t]), 1), "reason": reason, "details": det,
                       "best_iou": _r(t_best[t]), "best_box": f"b{boxes.index[bb]}" if bb >= 0 else None})
        yellow_polys.append(rects[t])

    # the numbers
    text_ink = text_ink_of(ink, chance)
    numbers = {"boxes": measures.box_numbers(boxes, terra, reads, iou=tuple(sorted(levels_used, reverse=True)),
                                             reason_fn=lambda t, *a: reasons[t][0] if t in reasons else
                                             measures.geometric_reason(t, *a)),
               "reads": reads_count, "iou": iou}
    numbers["codes"] = measures.read_numbers(boxes, terra, reads, forms=forms, answers=answers, iou=iou)
    if answers is not None and len(answers):
        numbers["headline"] = measures.headline(boxes, reads, answers, paths.QUARTERS)
        numbers["terra_right"] = int(sum(forms.get(_norm(a)) == tr
                                         for a, tr in zip(answers["terra"], answers["truth"])))
    numbers = _jsonable(numbers)
    for lv in numbers["boxes"].values():
        lv.pop("by_quarter", None)
    labels = answered_labels(boxes, reads, answers, causes) if answers is not None and len(answers) else []
    at_box: dict[str, list] = {}
    for lab in labels:
        if lab["box"]:
            at_box.setdefault(lab["box"], []).append({"truth": lab["truth"], "right": lab["right"]})
    for rec in recs:
        if rec["id"] in at_box:
            rec["answers"] = at_box[rec["id"]]
    terra_words = [{"id": f"w{t}", "poly": np.round(rects[t], 1).tolist(), "text": str(terra["text"].iloc[t]),
                    "kind": terra["kind"].iloc[t]} for t in range(len(terra))]
    pixels = palette_image(ink, text_ink, green_polys, yellow_polys)
    extra = {}
    if visible is not None:
        pixels[~visible & ~ink] = FADED_INDEX["faded_paper"]
        pixels[~visible & ink] = FADED_INDEX["faded_ink"]
        extra["faded"] = True
    if reason_text is not None:
        extra["reason_text"] = {**REASON_TEXT, **reason_text}
    if notes:
        extra["notes"] = [[float(x), float(y), str(t)] for x, y, t in notes]
    return {**extra, "sheet": sheet, "run": run, "shape": list(ink.shape),
            "pixels": pixels, "text_ink_px": int(text_ink[visible].sum() if visible is not None else text_ink.sum()),
            "boxes": recs, "missed": missed, "terra_words": terra_words, "labels": labels, "numbers": numbers,
            "delta": measures.compare(numbers, before) if before else None, "models": list(models), "iou": iou,
            "levels": {k: float(v_) for k, v_ in levels.items()}}


def red_px_of(d: dict) -> int:
    """The px the page should show red: the ink the detector calls text."""
    return int(d["text_ink_px"])


# --- the page -----------------------------------------------------------------------------------------------------

def _tile_png(idx: np.ndarray, layer: str = "base", faded: bool = False) -> str | None:
    """One tile of `page_data`'s pixels as a base64 palette PNG.  "base": the paper and every ink in one grey (the
    paper's tints drawn as paper, the text ink as the other ink; the faded colours when `faded`).  "text": the text ink
    red on a transparent ground, None when the tile holds none."""
    buf = io.BytesIO()
    if layer == "text":
        on = idx == INDEX["text_ink"]
        if not on.any():
            return None
        im = Image.fromarray(on.astype(np.uint8), "P")
        im.putpalette([0, 0, 0, *COLOURS["text_ink"]])
        im.save(buf, "PNG", optimize=True, transparency=0)
    else:
        base = {**COLOURS, "green_paper": COLOURS["paper"], "yellow_paper": COLOURS["paper"],
                "both_paper": COLOURS["paper"], "text_ink": COLOURS["other_ink"]}
        im = Image.fromarray(idx, "P")
        im.putpalette([c for rgb in [*base.values(), *(FADED.values() if faded else ())] for c in rgb])
        im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _f(v, k=1) -> str:
    return f"{v:.{k}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def _path(polys) -> str:
    return "".join("M" + "L".join(f"{_f(x)} {_f(y)}" for x, y in np.asarray(p, float).reshape(-1, 2)) + "Z"
                   for p in polys)


def _kind_class(rec: dict) -> str:
    """A box's kind as the page colours it: one of KIND_CLASSES (a box of no kind is "other")."""
    k = rec.get("kind")
    k = KIND_SHORT.get(k, k)
    return k if k in KIND_CLASSES else "other"


def _svg(w: int, h: int, body: str, attrs: str = "") -> str:
    return (f'<svg class="ov"{attrs} xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}">{body}</svg>')


def _outlines(d: dict, key: str, hidden: bool = False) -> str:
    """One version's outlines (SVG in page px; every stroke a constant screen width, unfilled, and no text): the
    boxes read as text by kind, the same boxes again for the filter of the readings that differ from Terra's, the
    fragments (read as no text), and Terra's words no box matches."""
    h, w = d["shape"]
    kinds: dict = {k: [] for k in KIND_CLASSES}
    differ: dict = {k: [] for k in KIND_CLASSES}
    frags = []
    for b in d["boxes"]:
        if b["status"] == "none":
            frags.append(b["poly"])
            continue
        kinds[_kind_class(b)].append(b["poly"])
        if b["colour"] == "purple":
            differ[_kind_class(b)].append(b["poly"])

    def strokes(groups):
        return "".join(f'<path class="ln k-{k}" fill="none" d="{_path(p)}"/>' for k, p in groups.items() if p)
    miss = _path([m["poly"] for m in d["missed"]])
    body = (f'<g class="L-frags"><path class="ln k-frag" fill="none" d="{_path(frags)}"/></g>'
            f'<g class="L-boxes">{strokes(kinds)}</g><g class="L-diff">{strokes(differ)}</g>'
            f'<g class="L-missed"><path class="ln halo" fill="none" d="{miss}"/>'
            f'<path class="ln miss" fill="none" d="{miss}"/></g>')
    return f'<div class="setov" data-set="{_e(key)}"{" hidden" if hidden else ""}>{_svg(w, h, body)}</div>'


def _shared_overlays(d: dict) -> str:
    """The overlays every version shares: Terra's words, the notes (upright, in page px) and the tapped box's
    highlight."""
    h, w = d["shape"]
    terra = _path([t["poly"] for t in d.get("terra_words", [])])
    out = [_svg(w, h, f'<path class="ln terra" fill="none" d="{terra}"/>', ' id="terra-ov"')]
    notes = "".join(f'<text x="{_f(float(x))}" y="{_f(float(y))}" font-size="72" fill="{NOTE}" text-anchor="middle" '
                    f'font-family="-apple-system,Helvetica,Arial,sans-serif">{_e(t)}</text>'
                    for x, y, t in d.get("notes", ()))
    if notes:
        out.append(_svg(w, h, notes))
    out.append(_svg(w, h, '<path id="hl-halo" class="ln hl-halo" fill="none" d=""/>'
                          '<path id="hl-line" class="ln hl-line" fill="none" d=""/>', ' id="hl-ov"'))
    return "".join(out)


def _set_numbers(d: dict) -> dict:
    """The header's numbers for one version: the owner's labels read right on the sheet (of n), the Terra words
    matched, the Terra words missed because merged with another word, and the matched boxes read unlike Terra."""
    b = d["numbers"]["boxes"][f'{d["iou"]}']
    hl = d["numbers"].get("headline") or {}
    return {"right": hl.get("right"), "n": hl.get("n"), "matched": b["matched"],
            "merged": (b.get("missed_reasons") or {}).get("merged_with_word", 0),
            "differs": d["numbers"]["reads"]["all"]["differs"]}


def _n(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return f"{v:,}"


def _pct(a, b) -> str:
    return f"{100 * a / b:.0f}%" if b else "–"


def _signed(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:+.3f}".rstrip("0").rstrip(".") if v else "±0"
    return f"{v:+,}" if v else "±0"


def _get(dct, *keys):
    for k in keys:
        if not isinstance(dct, dict) or k not in dct:
            return None
        dct = dct[k]
    return dct


def _panels(d: dict) -> str:
    """The two top-line panels and the reasons (HTML, in the folded explanations)."""
    nb = d["numbers"]["boxes"]
    lv, other = f'{d["iou"]}', [k for k in nb if k != f'{d["iou"]}']
    b = nb[lv]
    delta = d["delta"]
    rd = d["numbers"]["reads"]["all"]
    alt = "".join(f' · at IoU ≥ {k}: matched <b>{_n(nb[k]["matched"])}</b>, missed {_n(nb[k]["missed"])}, '
                  f'ratio {_n(nb[k]["matched_to_missed"])} : 1' for k in other)
    rows = [(f'Terra saw <b>{_n(b["terra"])}</b> words · we found <b>{_n(b["ours"])}</b> boxes '
             f'({_signed(b["ours"] - b["terra"])} against Terra)'),
            (f'Matched to a Terra word (IoU ≥ {lv}) <b>{_n(b["matched"])}</b> ({_pct(b["matched"], b["terra"])}) · '
             f'missed <b>{_n(b["missed"])}</b> ({_pct(b["missed"], b["terra"])}, the missed words)'),
            (f'Ours, not in Terra: read as text <b>{_n(b["ours_unmatched_text"])}</b> · read as none '
             f'<b>{_n(b["ours_unmatched_none"])}</b> (the fragments)'),
            f'Matched : missed = <b>{_n(b["matched_to_missed"])} : 1</b>{alt}']
    if delta:
        db = _get(delta, "boxes", lv) or {}
        rows.append(f'vs last run: matched <b>{_signed(db.get("matched"))}</b> · missed '
                    f'{_signed(db.get("missed"))} · our boxes {_signed(db.get("ours"))} · green '
                    f'{_signed(db.get("ours_unmatched_text"))} · grey {_signed(db.get("ours_unmatched_none"))} · '
                    f'ratio {_signed(db.get("matched_to_missed"))}')
    else:
        rows.append("vs last run: none yet (this is the first page)")
    boxes_html = "<br>".join(rows)
    hl, tr = d["numbers"].get("headline"), d["numbers"].get("terra_right")
    shared = (f' (they meet {_n(b["matched"])} words: a word met by two boxes is read twice)'
              if rd["matched"] != b["matched"] else "")
    cf = d["numbers"].get("codes") or {}
    cva = cf.get("codes_right_vs_answers") or {}
    on_labels = (f' · on the owner\'s labels whose box is matched: ours {_n(cva["ours"])} / {_n(cva["n"])}, Terra\'s '
                 f'{_n(cva["terra"])} / {_n(cva["n"])}' if cva else "")
    trows = [(f'All matched boxes, any kind, string vs Terra\'s: <b>{_n(rd["matched"])}</b> boxes{shared} · same '
              f'<b>{_n(rd["same"])}</b> ({_pct(rd["same"], rd["matched"])}, green pills) · different '
              f'<b>{_n(rd["differs"])}</b> (purple pills)'),
             (f'Zoning-code forms only (Terra\'s string a printed form of the code list, our likeliest code): same '
              f'<b>{_n(cf.get("same_as_terra"))}</b> · different <b>{_n(cf.get("differs"))}</b> · '
              f'{_n(cf.get("not_a_form"))} other matched boxes{on_labels}')]
    if hl:
        ci = hl.get("ci") or [None, None]
        trows.append(f'The owner\'s answered labels on this sheet: <b>{_n(hl["right"])} / {_n(hl["n"])}</b> codes '
                     f'right at our boxes ({_pct(hl["right"], hl["n"])}, 95% {_n(ci[0])}–{_n(ci[1])}; a missed box '
                     f'counts wrong; {_n(hl["with_box"])} have a box) · Terra\'s strings right <b>{_n(tr)}</b> '
                     f'({_signed(hl["right"] - tr) if tr is not None else "–"} against Terra)')
    if delta:
        dr = _get(delta, "reads", "all") or {}
        dc = _get(delta, "codes") or {}
        trows.append(f'vs last run: any kind same <b>{_signed(dr.get("same"))}</b>, different '
                     f'{_signed(dr.get("differs"))} · code forms same <b>{_signed(dc.get("same_as_terra"))}</b>, '
                     f'different {_signed(dc.get("differs"))} · codes right '
                     f'{_signed(_get(delta, "headline", "right"))}')
    else:
        trows.append("vs last run: none yet")
    reasons = b.get("missed_reasons") or {}
    rrows = "".join(f'<tr><td><b>{_e(r.replace("_", " "))}</b></td><td class="num">{_n(reasons.get(r, 0))}</td>'
                    f'<td>{_e(d.get("reason_text", REASON_TEXT)[r])}</td></tr>' for r in REASONS)
    return (f'<div class="grid"><section class="box"><h2>Boxes (whole sheet)</h2>{boxes_html}</section>'
            f'<section class="box"><h2>Text read</h2>{"<br>".join(trows)}</section>'
            f'<section class="box"><h2>Why we missed Terra\'s words (IoU ≥ {lv})</h2>'
            f'<table class="rs">{rrows}</table></section></div>')


STYLE = """
:root {
  /* Layout: one screen.  A compact header and toolbar, the scan filling the rest; what a tap opens slides up over
     the map on a phone and sits in a side panel on a wide screen.  Box colours: Okabe-Ito, colour-blind safe. */
  --bg: #f2f3f0; --surface: #ffffff; --fg: #1b1f1c; --muted: #59625c; --rule: #d6dad3; --view: #e3e6e0;
  --map-paper: #ffffff; --map-filter: none;
  --k-code: #0072b2; --k-block: #c98200; --k-street: #009e73; --k-other: #303030; --k-frag: #8c8c8c;
  --terra: #3f3f3f; --miss: #c2338a; --halo: rgba(255, 255, 255, .85); --sel: #ffd500; --sel-line: #111111;
  --ok-bg: #d6efdc; --ok-fg: #0b4f23; --bad-bg: #6a3cbf; --bad-fg: #ffffff; --ti: #dd2222;
  --pill: rgba(255, 255, 255, .95); --pill-fg: #1b1f1c; --pill-edge: rgba(0, 0, 0, .24); --shadow: rgba(0, 0, 0, .16);
  --on: #1b1f1c; --on-fg: #ffffff;
  --font: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #101311; --surface: #1a1e1b; --fg: #e5eae6; --muted: #9aa49e; --rule: #2e3530; --view: #0b0d0c;
  --map-paper: #1a1a1a; --map-filter: invert(.9);
  --k-code: #56b4e9; --k-block: #f0b43c; --k-street: #3ccb9c; --k-other: #e8e8e8; --k-frag: #6e6e6e;
  --terra: #d2d2d2; --miss: #f47cc6; --halo: rgba(0, 0, 0, .8); --sel: #ffd84d; --sel-line: #ffffff;
  --ok-bg: #143a20; --ok-fg: #a6e9bb; --bad-bg: #8659e3; --bad-fg: #ffffff; --ti: #dd2222;
  --pill: rgba(24, 28, 25, .95); --pill-fg: #e5eae6; --pill-edge: rgba(255, 255, 255, .26); --shadow: rgba(0, 0, 0, .5);
  --on: #e5eae6; --on-fg: #101311; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #101311; --surface: #1a1e1b; --fg: #e5eae6; --muted: #9aa49e; --rule: #2e3530; --view: #0b0d0c;
  --map-paper: #1a1a1a; --map-filter: invert(.9);
  --k-code: #56b4e9; --k-block: #f0b43c; --k-street: #3ccb9c; --k-other: #e8e8e8; --k-frag: #6e6e6e;
  --terra: #d2d2d2; --miss: #f47cc6; --halo: rgba(0, 0, 0, .8); --sel: #ffd84d; --sel-line: #ffffff;
  --ok-bg: #143a20; --ok-fg: #a6e9bb; --bad-bg: #8659e3; --bad-fg: #ffffff; --ti: #dd2222;
  --pill: rgba(24, 28, 25, .95); --pill-fg: #e5eae6; --pill-edge: rgba(255, 255, 255, .26); --shadow: rgba(0, 0, 0, .5);
  --on: #e5eae6; --on-fg: #101311; color-scheme: dark }
html, body { height: 100%; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.45 var(--font); }
[hidden] { display: none !important; }
button { font: inherit; color: inherit; }
button:focus-visible, #vp:focus-visible { outline: 2px solid var(--k-code); outline-offset: 2px; }
.app { box-sizing: border-box; height: 100%; display: flex; flex-direction: column; gap: 8px; padding-block: 8px 10px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.top { display: flex; flex-wrap: wrap; align-items: center; gap: 2px 14px; }
.top h1 { order: 0; flex: 1 1 auto; min-width: 0; margin: 0; font-size: 17px; font-weight: 650; letter-spacing: -.01em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.top h1 .sid { color: var(--muted); font-weight: 500; }
#aboutb { order: 1; }
.nums { order: 2; flex: 1 1 100%; display: flex; flex-wrap: wrap; gap: 0 14px; color: var(--muted); font-size: 13px;
  font-variant-numeric: tabular-nums; }
.nums b { color: var(--fg); font-size: 15px; font-weight: 650; }
.legend { order: 3; flex: 1 1 100%; min-width: 0; display: flex; align-items: center; gap: 9px; overflow-x: auto;
  white-space: nowrap; font-size: 12px; color: var(--muted); scrollbar-width: none; padding: 3px 2px; }
.legend > span { display: inline-flex; align-items: center; gap: 4px; }
.legend .sw { width: 12px; }
.legend .lp { font-size: 11px; line-height: 14px; }
@media (max-width: 479px) { .wd { display: none; } }
.legend::-webkit-scrollbar, .chips::-webkit-scrollbar { display: none; }
.sw { display: inline-block; width: 16px; height: 0; border-top: 2.5px solid var(--c); flex: none; }
.s-code { --c: var(--k-code); } .s-block { --c: var(--k-block); } .s-street { --c: var(--k-street); }
.s-other { --c: var(--k-other); } .s-frag { --c: var(--k-frag); border-top-width: 1.5px; } .s-miss { --c: var(--miss); }
.s-terra { --c: var(--terra); border-top-style: dashed; border-top-width: 1.5px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; background: var(--ti); flex: none; }
.lp, .pl { font: 600 12px/16px var(--font); padding: 1px 5px; border-radius: 5px; white-space: nowrap;
  background: var(--pill); color: var(--pill-fg); box-shadow: 0 0 0 1px var(--pill-edge), 0 1px 3px var(--shadow); }
.lp.ok, .pl.ok { background: var(--ok-bg); color: var(--ok-fg); }
.lp.bad, .pl.bad { background: var(--bad-bg); color: var(--bad-fg); }
.pl .ne { opacity: .8; font-weight: 500; }
.pl { position: absolute; left: 0; top: 0; }
.pl.miss { color: var(--miss); box-shadow: 0 0 0 1.5px var(--miss), 0 1px 3px var(--shadow); }
.pl.tag { font: 500 11px/14px var(--mono); color: var(--muted); }
.pl.kc-code { --kc: var(--k-code); } .pl.kc-block { --kc: var(--k-block); } .pl.kc-street { --kc: var(--k-street); }
.pl.kc-other { --kc: var(--k-other); }
.pl[class*="kc-"] { border-left: 3px solid var(--kc); border-radius: 3px 5px 5px 3px; }
.pl.sel { box-shadow: 0 0 0 2.5px var(--sel), 0 1px 4px var(--shadow); }
.bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.seg { flex: 1 1 100%; display: flex; gap: 2px; padding: 2px; background: var(--view); border: 1px solid var(--rule);
  border-radius: 12px; }
.seg button { flex: 1 1 0; min-height: 40px; border: 0; border-radius: 10px; background: none; color: var(--muted);
  font-weight: 600; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--surface); color: var(--fg); box-shadow: 0 1px 3px var(--shadow); }
.chips { flex: 1 1 100%; min-width: 0; display: flex; gap: 6px; overflow-x: auto; scrollbar-width: none; }
.chip { flex: none; display: inline-flex; align-items: center; gap: 6px; min-height: 40px; padding: 0 13px;
  border-radius: 20px; border: 1px solid var(--rule); background: var(--surface); font-size: 13px; font-weight: 550;
  cursor: pointer; }
.chip[aria-pressed="true"] { background: var(--on); color: var(--on-fg); border-color: var(--on); }
.chip .auto, .chip .cnt { font-weight: 400; opacity: .7; font-variant-numeric: tabular-nums; }
.tb, .ib { min-width: 40px; min-height: 40px; border-radius: 10px; border: 1px solid var(--rule);
  background: var(--surface); cursor: pointer; }
.tb { padding: 0 12px; font-size: 13px; font-weight: 600; }
.ib { font-size: 20px; line-height: 1; padding: 0; }
.work { position: relative; flex: 1 1 auto; min-height: 260px; display: flex; gap: 12px; }
#vp { position: relative; flex: 1 1 auto; min-width: 0; overflow: hidden; touch-action: none; background: var(--view);
  border: 1px solid var(--rule); border-radius: 12px; cursor: grab; user-select: none; -webkit-user-select: none; }
#vp.drag { cursor: grabbing; }
#stage { position: absolute; left: 0; top: 0; transform-origin: 0 0; background: var(--map-paper); }
#stage img { position: absolute; image-rendering: pixelated; pointer-events: none; -webkit-user-drag: none;
  max-width: none; }
#stage img.base { filter: var(--map-filter); }
#stage img.ti { display: none; }
#vp.on-ti #stage img.ti { display: block; }
.setov { position: absolute; left: 0; top: 0; }
.ov { position: absolute; left: 0; top: 0; overflow: visible; pointer-events: none; }
.ln { vector-effect: non-scaling-stroke; fill: none; stroke-linejoin: round; stroke-width: 1.5px; }
.k-code { stroke: var(--k-code); } .k-block { stroke: var(--k-block); } .k-street { stroke: var(--k-street); }
.k-other { stroke: var(--k-other); } .k-frag { stroke: var(--k-frag); stroke-width: 1px; }
.terra { stroke: var(--terra); stroke-width: 1px; stroke-dasharray: 3 3; }
.halo { stroke: var(--halo); stroke-width: 5px; } .miss { stroke: var(--miss); stroke-width: 2.5px; }
.hl-halo { stroke: var(--sel); stroke-width: 8px; } .hl-line { stroke: var(--sel-line); stroke-width: 2px; }
.L-frags, .L-diff, .L-missed, #terra-ov { display: none; }
#vp.on-frags .L-frags, #vp.on-missed .L-missed, #vp.f-diff .L-diff { display: inline; }
#vp.on-terra #terra-ov { display: block; }
#vp.f-diff .L-boxes { display: none; }
#lab { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
.xm { position: absolute; left: 0; top: 0; width: 0; height: 0; }
.xm::before, .xm::after { content: ""; position: absolute; background: var(--sel-line);
  box-shadow: 0 0 0 1.5px var(--sel); }
.xm::before { left: -9px; top: -1px; width: 18px; height: 2px; }
.xm::after { left: -1px; top: -9px; width: 2px; height: 18px; }
.zl { position: absolute; left: 8px; top: 8px; font: 600 12px/1 var(--mono); padding: 6px 7px; border-radius: 8px;
  background: var(--pill); color: var(--muted); box-shadow: 0 0 0 1px var(--pill-edge); pointer-events: none; }
.ctl { position: absolute; right: 8px; top: 8px; display: flex; flex-direction: column; gap: 6px; }
.ctl button { width: 40px; height: 40px; border-radius: 10px; border: 1px solid var(--rule); background: var(--surface);
  font-size: 20px; line-height: 1; cursor: pointer; box-shadow: 0 1px 3px var(--shadow); padding: 0; }
.ctl #zfit { font-size: 12px; font-weight: 650; }
@media (pointer: coarse) { .ctl #zin, .ctl #zout { display: none; } }
.nav { position: absolute; left: 50%; bottom: 10px; transform: translateX(-50%); display: flex;
  background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; box-shadow: 0 2px 10px var(--shadow);
  overflow: hidden; max-width: calc(100% - 20px); }
.nav button { min-width: 46px; min-height: 42px; border: 0; background: none; font-size: 22px; line-height: 1;
  cursor: pointer; padding: 0; }
.nav .navc { font-size: 13px; padding: 0 12px; white-space: nowrap; border-left: 1px solid var(--rule);
  border-right: 1px solid var(--rule); }
#panel { position: absolute; left: 0; right: 0; bottom: 0; max-height: 52%; overflow: auto;
  overscroll-behavior: contain;
  background: var(--surface); border: 1px solid var(--rule); border-radius: 14px 14px 12px 12px;
  box-shadow: 0 -6px 24px var(--shadow); display: none; z-index: 5; }
#panel.open { display: block; }
.work.open .nav { display: none; }
.ph { position: sticky; top: 0; z-index: 1; display: flex; align-items: center; gap: 6px; padding: 8px 8px 8px 12px;
  background: var(--surface); border-bottom: 1px solid var(--rule); }
.ph h2 { flex: 1 1 auto; min-width: 0; margin: 0; font-size: 14px; font-weight: 650; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.ph .pos { font: 600 13px var(--mono); color: var(--muted); }
.muted, .sub { color: var(--muted); font-weight: 400; }
.pb { padding: 10px 12px 14px; }
.vs { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; margin: 0 0 10px; }
.vs .lp { font-size: 14px; line-height: 20px; padding: 2px 8px; }
.vs .k { color: var(--muted); font-size: 12px; }
.why { margin: 0 0 10px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 8px; }
.why .k { color: var(--muted); font-size: 12px; }
.kv { width: 100%; border-collapse: collapse; font-size: 13px; }
.kv th { text-align: left; font-weight: 500; color: var(--muted); padding: 6px 10px 6px 0; vertical-align: top;
  white-space: nowrap; border-top: 1px solid var(--rule); }
.kv td { padding: 6px 0; border-top: 1px solid var(--rule); overflow-wrap: anywhere; }
.kv .tb { margin-top: 4px; }
.cause { display: inline-block; flex: none; font: 600 11px/1 var(--font); padding: 4px 8px; border-radius: 999px;
  background: var(--view); border: 1px solid var(--rule); white-space: nowrap; }
.cf { display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 12px 6px; }
.cf button { min-height: 40px; padding: 0 12px; border-radius: 20px; border: 1px solid var(--rule);
  background: var(--surface); font-size: 13px; cursor: pointer; }
.cf button[aria-pressed="true"] { background: var(--on); color: var(--on-fg); border-color: var(--on); }
.ml { list-style: none; margin: 0; padding: 0 0 6px; }
.ml button { width: 100%; min-height: 44px; display: flex; align-items: center; gap: 10px; padding: 6px 12px;
  border: 0; border-top: 1px solid var(--rule); background: none; text-align: left; cursor: pointer; font-size: 14px; }
.ml button[aria-current="true"] { background: var(--view); }
.ml .mi { flex: 1 1 auto; min-width: 0; }
.ml .mi b { font-weight: 650; }
.empty { margin: 0; padding: 12px; color: var(--muted); }
#about { flex: none; max-height: 50%; overflow: auto; background: var(--surface); border: 1px solid var(--rule);
  border-radius: 12px; padding: 12px 14px; font-size: 13.5px; }
#about h2 { font-size: 15px; margin: 0 0 6px; }
#about h3 { font: 600 12px/1.2 var(--font); letter-spacing: .06em; text-transform: uppercase; color: var(--muted);
  margin: 14px 0 6px; }
#about p, #about li { max-width: 78ch; overflow-wrap: anywhere; }
#about ul { padding-left: 18px; margin: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 10px;
  margin-top: 12px; }
.box { border: 1px solid var(--rule); padding: 10px 12px; min-width: 0; overflow-wrap: anywhere; border-radius: 8px; }
.box h2 { font: 600 12px/1.2 var(--mono) !important; letter-spacing: .06em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 8px !important; }
.tblwrap { overflow-x: auto; }
table.rs, table.cmp { border-collapse: collapse; width: 100%; font-size: 13px; }
table.rs td, table.cmp td, table.cmp th { border-top: 1px solid var(--rule); padding: 4px 8px 4px 0;
  vertical-align: top;
  text-align: left; }
table.cmp th { font-weight: 600; white-space: nowrap; } table.cmp td { font-variant-numeric: tabular-nums; }
table.rs td:first-child, td.num { white-space: nowrap; }
td.num { text-align: right; font-variant-numeric: tabular-nums; padding-right: 10px; }
.models { color: var(--muted); font-size: 12.5px; margin-top: 12px; overflow-wrap: anywhere; }
@media (min-width: 960px) {
  .top { flex-wrap: nowrap; } .top h1 { flex: none; }
  .nums { order: 1; flex: 0 1 auto; } .legend { order: 2; flex: 0 1 auto; } #aboutb { order: 3; margin-left: auto; }
  .seg { flex: none; width: 340px; } .chips { flex: 1 1 0; }
  #panel { position: static; display: block; width: 360px; flex: none; max-height: none; box-shadow: none;
    border-radius: 12px; }
  .work.open .nav { display: flex; }
  .xp { display: none; }
}
"""

SCRIPT = r"""
(function () {
  'use strict';
  const D = JSON.parse(document.getElementById('data').textContent);
  const W = D.shape[1], H = D.shape[0], MAXS = D.max_scale, READ_FROM = D.readings_from;
  const $ = id => document.getElementById(id);
  const vp = $('vp'), st = $('stage'), lab = $('lab'), panel = $('panel'), work = $('work'), zl = $('zl');
  const hlHalo = $('hl-halo'), hlLine = $('hl-line'), nav = $('nav'), navc = $('navc'), xm = $('xm');
  const KIND = { zoning: 'code', code: 'code', block: 'block', street: 'street', other: 'other' };
  const KIND_NAME = { code: 'Zoning code', block: 'Block number', street: 'Street name', other: 'Other text' };
  const kindOf = r => KIND[r.kind] || 'other';
  const ENT = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
  const esc = t => String(t == null ? '' : t).replace(/[&<>"]/g, c => ENT[c]);
  const f2 = v => v == null ? '–' : (+v).toFixed(2);
  const pc = v => v == null ? '–' : Math.round(100 * v) + '%';
  const wide = () => window.matchMedia('(min-width: 960px)').matches;

  // each outline's bounding box, area and length, once
  function prep(r, t) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, a = 0;
    const p = r.poly;
    for (let i = 0; i < p.length; i++) {
      const x = p[i][0], y = p[i][1], q = p[(i + 1) % p.length];
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      a += x * q[1] - q[0] * y;
    }
    r._t = t; r._bb = [x0, y0, x1, y1]; r._area = Math.abs(a) / 2; r._len = Math.max(x1 - x0, y1 - y0);
  }
  const ids = {};
  for (const k of D.order) {
    const S = D.sets[k], m = ids[k] = new Map();
    S.boxes.forEach(b => { prep(b, 'box'); m.set(b.id, b); });
    S.missed.forEach(t => { prep(t, 'miss'); m.set(t.id, t); });
    S.missed.forEach(t => { const b = t.best_box && m.get(t.best_box); if (b) (b._missed = b._missed || []).push(t); });
    S.wrong = (S.labels || []).filter(l => !l.right);
  }
  D.terra.forEach(w => prep(w, 'terra'));

  // the layers, remembered in this browser only
  const KEY = 'sfzoning-wordread-review-layers';
  const L = { readings: null, missed: false, terra: false, textink: false, frags: false, details: false, diff: false };
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || 'null');
    if (v && typeof v === 'object') for (const k in L) if (k in v) L[k] = v[k];
  } catch (e) { /* no storage: the defaults */ }
  function save() { try { localStorage.setItem(KEY, JSON.stringify(L)); } catch (e) { /* no storage */ } }
  let key = D.order[0], s = 1, tx = 0, ty = 0, minS = 0.05;
  let sel = null, navI = -1, causeF = null, mode = 'list';
  const S = () => D.sets[key];
  const readingsOn = () => L.readings == null ? s >= READ_FROM : !!L.readings;
  const list = () => causeF ? S().wrong.filter(l => l.cause === causeF) : S().wrong;

  const chips = Array.from(document.querySelectorAll('.chip[data-layer]'));
  function syncChips() {
    chips.forEach(c => {
      const k = c.dataset.layer, on = k === 'readings' ? readingsOn() : !!L[k];
      c.setAttribute('aria-pressed', String(on));
      const a = c.querySelector('.auto');
      if (a) a.hidden = L.readings != null;
    });
    vp.classList.toggle('on-missed', !!L.missed); vp.classList.toggle('on-terra', !!L.terra);
    vp.classList.toggle('on-ti', !!L.textink); vp.classList.toggle('on-frags', !!L.frags);
    vp.classList.toggle('f-diff', !!L.diff);
  }
  chips.forEach(c => c.addEventListener('click', () => {
    const k = c.dataset.layer;
    if (k === 'readings') L.readings = !readingsOn(); else L[k] = !L[k];
    save(); syncChips(); schedule();
  }));

  function syncSet() {
    const n = S().nums;
    $('n-right').textContent = n.right == null ? '–' : n.right + '/' + n.n;
    $('n-match').textContent = n.matched.toLocaleString('en-US');
    $('n-merged').textContent = n.merged.toLocaleString('en-US');
    const nd = $('n-diff');
    if (nd) nd.textContent = n.differs.toLocaleString('en-US');
    document.querySelectorAll('.setov').forEach(e => { e.hidden = e.dataset.set !== key; });
    document.querySelectorAll('.seg button')
      .forEach(b => b.setAttribute('aria-pressed', String(b.dataset.set === key)));
  }
  document.querySelectorAll('.seg button').forEach(b => b.addEventListener('click', () => {
    if (b.dataset.set === key) return;
    const keep = sel && sel.label ? sel.label.k : null;
    key = b.dataset.set; syncSet();
    if (causeF && !list().length) causeF = null;
    sel = null; navI = -1; drawHL();
    const i = keep == null ? -1 : list().findIndex(l => l.k === keep);
    if (i >= 0) openMistake(i, false);
    else if (mode === 'detail') close();
    else { renderPanel(); renderNav(); schedule(); }
  }));

  // pan and zoom (the view stays when the version changes)
  function apply() {
    st.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + s + ')';
    zl.textContent = (s >= 1 ? s.toFixed(1) : s.toFixed(2)) + 'x';
    syncChips(); schedule();
  }
  function fit() {
    const r = vp.getBoundingClientRect();
    s = Math.min(r.width / W, r.height / H); minS = s * 0.5;
    tx = (r.width - W * s) / 2; ty = (r.height - H * s) / 2; apply();
  }
  function zoomAt(px, py, f) {
    const ns = Math.min(MAXS, Math.max(minS, s * f));
    tx = px - (px - tx) * ns / s; ty = py - (py - ty) * ns / s; s = ns; apply();
  }
  function centre(f) { const r = vp.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, f); }
  $('zin').onclick = () => centre(2);
  $('zout').onclick = () => centre(0.5);
  $('zfit').onclick = fit;
  vp.addEventListener('wheel', e => {
    e.preventDefault();
    const r = vp.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)));
  }, { passive: false });
  const pts = new Map(); let moved = 0, startT = 0, pinch = null;
  vp.addEventListener('pointerdown', e => {
    if (e.target.closest('button')) return;
    vp.setPointerCapture(e.pointerId);
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pts.size === 1) { moved = 0; startT = Date.now(); vp.classList.add('drag'); }
    if (pts.size === 2) pinch = null;
  });
  vp.addEventListener('pointermove', e => {
    if (!pts.has(e.pointerId)) return;
    const p = pts.get(e.pointerId), r = vp.getBoundingClientRect();
    if (pts.size === 1) {
      tx += e.clientX - p.x; ty += e.clientY - p.y; moved += Math.abs(e.clientX - p.x) + Math.abs(e.clientY - p.y);
      p.x = e.clientX; p.y = e.clientY; apply(); return;
    }
    p.x = e.clientX; p.y = e.clientY; moved += 10;
    const [a, b] = [...pts.values()];
    const d = Math.hypot(a.x - b.x, a.y - b.y), mx = (a.x + b.x) / 2 - r.left, my = (a.y + b.y) / 2 - r.top;
    if (pinch) { tx += mx - pinch.mx; ty += my - pinch.my; zoomAt(mx, my, d / pinch.d); }
    pinch = { d: d, mx: mx, my: my };
  });
  function up(e) {
    if (!pts.has(e.pointerId)) return;
    const single = pts.size === 1;
    pts.delete(e.pointerId); pinch = null;
    if (!pts.size) vp.classList.remove('drag');
    if (single && moved < 8 && Date.now() - startT < 600) {
      const r = vp.getBoundingClientRect();
      tap((e.clientX - r.left - tx) / s, (e.clientY - r.top - ty) / s);
    }
  }
  vp.addEventListener('pointerup', up); vp.addEventListener('pointercancel', up);
  vp.addEventListener('dblclick', e => {
    if (e.target.closest('button')) return;
    const r = vp.getBoundingClientRect(); zoomAt(e.clientX - r.left, e.clientY - r.top, 2);
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && sel) close(); });

  // a tap: the smallest outline under it, else the nearest within 12 screen px
  function inside(poly, x, y) {
    let c = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  }
  function shownBox(b) { return b.status === 'none' ? !!L.frags : !(L.diff && b.colour !== 'purple'); }
  function hit(x, y) {
    const tol = 12 / s, A = S(), cands = [];
    A.boxes.forEach(b => { if (shownBox(b)) cands.push([b, b.status === 'none' ? 1 : 0]); });
    if (L.missed) A.missed.forEach(m => cands.push([m, 0]));
    if (L.terra) D.terra.forEach(w => cands.push([w, 2]));
    let best = null, bk = null;
    for (const [r, rank] of cands) {
      const bb = r._bb;
      if (x < bb[0] - tol || x > bb[2] + tol || y < bb[1] - tol || y > bb[3] + tol) continue;
      const inn = inside(r.poly, x, y);
      const dist = inn ? 0 : Math.hypot(Math.max(bb[0] - x, 0, x - bb[2]), Math.max(bb[1] - y, 0, y - bb[3]));
      const k = inn ? [0, rank, r._area] : [1, dist, rank];
      if (!bk || k[0] < bk[0] || (k[0] === bk[0] && (k[1] < bk[1] || (k[1] === bk[1] && k[2] < bk[2])))) {
        best = r; bk = k;
      }
    }
    return best;
  }
  function tap(x, y) {
    const r = hit(x, y);
    if (!r) { if (sel || (!wide() && panel.classList.contains('open'))) close(); return; }
    const i = r._t === 'box' ? list().findIndex(l => l.box === r.id) : -1;
    if (i >= 0) openMistake(i, false); else { navI = -1; select(r, null); }
  }

  // the highlight, the panel and the wrong labels
  const pathOf = p => 'M' + p.map(q => q[0] + ' ' + q[1]).join('L') + 'Z';
  function drawHL() {
    const d = sel && sel.rec ? pathOf(sel.rec.poly) : '';
    hlHalo.setAttribute('d', d); hlLine.setAttribute('d', d);
  }
  function openPanel() { panel.classList.add('open'); work.classList.add('open'); panel.scrollTop = 0; }
  function select(rec, label) {
    sel = { rec: rec, label: label }; mode = 'detail';
    drawHL(); renderPanel(); openPanel(); renderNav(); schedule();
  }
  function close() {
    sel = null; mode = 'list'; drawHL();
    if (wide()) renderPanel(); else { panel.classList.remove('open'); work.classList.remove('open'); }
    renderNav(); schedule();
  }
  function showList() { sel = null; mode = 'list'; drawHL(); renderPanel(); openPanel(); renderNav(); schedule(); }
  function openMistake(i, move) {
    const items = list();
    if (!items.length) return;
    navI = ((i % items.length) + items.length) % items.length;
    const l = items[navI], b = l.box ? ids[key].get(l.box) : null;
    if (move !== false) focusOn(b ? b._bb : [l.x - 30, l.y - 15, l.x + 30, l.y + 15]);
    select(b, l);
  }
  function focusOn(bb) {
    const r = vp.getBoundingClientRect(), avail = wide() ? r.height : r.height * 0.46;
    const bw = Math.max(bb[2] - bb[0], 24), bh = Math.max(bb[3] - bb[1], 12);
    s = Math.min(MAXS, Math.max(READ_FROM, Math.min(4, 0.6 * r.width / bw, 0.6 * avail / bh)));
    tx = r.width / 2 - (bb[0] + bb[2]) / 2 * s; ty = avail / 2 - (bb[1] + bb[3]) / 2 * s; apply();
  }
  function renderNav() {
    const all = S().wrong, items = list();
    nav.hidden = !all.length;
    navc.innerHTML = navI >= 0 ? '<b>' + (navI + 1) + '</b> / ' + items.length + ' wrong'
      : '<b>' + items.length + '</b> wrong label' + (items.length === 1 ? '' : 's');
  }
  nav.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    const a = b.dataset.act;
    if (a === 'prev') openMistake(navI < 0 ? -1 : navI - 1);
    else if (a === 'next') openMistake(navI + 1);
    else showList();
  });
  panel.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    const act = b.dataset.act;
    if (act === 'close') close();
    else if (act === 'prev') openMistake(navI - 1);
    else if (act === 'next') openMistake(navI + 1);
    else if (act === 'list') showList();
    else if (act === 'item') openMistake(+b.dataset.i);
    else if (act === 'cause') { causeF = b.dataset.c || null; navI = -1; renderPanel(); renderNav(); }
    else if (act === 'open') {
      const r = ids[key].get(b.dataset.id);
      if (r) { navI = -1; focusOn(r._bb); select(r, null); }
    }
  });
  const causeChip = l => l.cause ? '<span class="cause">' + esc(D.causes[l.cause] || l.cause) + '</span>' : '';
  const row = (k, v) => '<tr><th>' + k + '</th><td>' + v + '</td></tr>';
  function renderPanel() { panel.innerHTML = mode === 'detail' && sel ? detailHtml() : listHtml(); }
  function listHtml() {
    const A = S(), all = A.wrong, items = list();
    let h = '<div class="ph"><h2>Labels read wrong <span class="muted">' + all.length +
      (A.nums.n != null ? ' of ' + A.nums.n : '') + '</span></h2>' +
      '<button class="ib xp" data-act="close" aria-label="Close">×</button></div>';
    if (!(A.labels || []).length) return h + '<p class="empty">No answered labels on this sheet.</p>';
    if (!all.length) return h + '<p class="empty">This version reads every answered label right.</p>';
    const counts = {};
    all.forEach(l => { if (l.cause) counts[l.cause] = (counts[l.cause] || 0) + 1; });
    const order = Object.keys(D.causes).filter(c => counts[c]);
    if (order.length) {
      h += '<div class="cf" role="group" aria-label="Cause"><button data-act="cause" data-c="" aria-pressed="' +
        String(!causeF) + '">All ' + all.length + '</button>' + order.map(c => '<button data-act="cause" data-c="' + c +
        '" aria-pressed="' + String(causeF === c) + '">' + esc(D.causes[c]) + ' ' + counts[c] + '</button>').join('') +
        '</div>';
    }
    return h + '<ol class="ml">' + items.map((l, i) => '<li><button data-act="item" data-i="' + i + '"' +
      (i === navI ? ' aria-current="true"' : '') + '>' + causeChip(l) + '<span class="mi"><b>' + esc(l.truth) +
      '</b> <span class="sub">read</span> ' +
      (l.read ? '<b>' + esc(l.read) + '</b>' : '<span class="sub">no box</span>') +
      '</span></button></li>').join('') + '</ol>';
  }
  function detailHtml() {
    const r = sel.rec, l = sel.label, items = list();
    const title = l ? 'Your label' : !r ? '' : r._t === 'box' ? KIND_NAME[kindOf(r)] + ' box' :
      r._t === 'miss' ? 'Missed word' : 'Terra’s word';
    let h = '<div class="ph">';
    if (l && navI >= 0) h += '<button class="ib" data-act="prev" aria-label="Previous wrong label">‹</button>' +
      '<span class="pos">' + (navI + 1) + '/' + items.length + '</span>' +
      '<button class="ib" data-act="next" aria-label="Next wrong label">›</button>';
    h += '<h2>' + esc(title) + '</h2>';
    if (l) h += '<button class="tb" data-act="list">List</button>';
    h += '<button class="ib" data-act="close" aria-label="Close">×</button></div><div class="pb">';
    if (l) h += labelHtml(l);
    if (r && r._t === 'box') h += boxHtml(r);
    else if (r && r._t === 'miss') h += missHtml(r);
    else if (r && r._t === 'terra') h += terraHtml(r);
    return h + '</div>';
  }
  function labelHtml(l) {
    return '<div class="vs"><span class="k">your answer</span><span class="lp">' + esc(l.truth) + '</span>' +
      '<span class="k">our code</span>' + (l.read ? '<span class="lp ' + (l.right ? 'ok' : 'bad') + '">' +
      esc(l.read) + '</span>' : '<span class="lp">no box</span>') +
      (l.terra != null ? '<span class="k">Terra</span><span class="lp">' + esc(l.terra) + '</span>' : '') + '</div>' +
      (l.cause ? '<p class="why"><span class="k">why</span> ' + causeChip(l) + (l.note ? ' <span class="sub">' +
        esc(l.note) + '</span>' : '') + '</p>' : '');
  }
  function readingPills(b) {
    const o = b.shown || '∅';
    const note = t => ' <span class="sub">' + t + '</span>';
    const pill = (cls, t) => '<span class="lp' + cls + '">' + esc(t) + '</span>';
    if (b.status === 'matched' && b.colour === 'green') return pill(' ok', o) + note('= Terra');
    if (b.status === 'matched') return pill(' bad', o + ' ≠ ' + b.terra) + note('ours ≠ Terra’s');
    if (b.status === 'text') return pill('', b.shown || '?') + note('Terra has no word here');
    return '<span class="sub">no text: a fragment</span>';
  }
  function boxHtml(b) {
    const kinds = Object.entries(b.p_kind || {}).sort((a, c) => c[1] - a[1])
      .map(([k, p]) => k + ' ' + f2(p)).join(' · ');
    const top3 = (b.top3 || []).map(([h, p]) => esc(h) + ' ' + f2(p)).join(' · ');
    const ans = (b.answers || [])
      .map(a => '<b>' + esc(a.truth) + '</b> ' + (a.right ? 'read right' : 'read wrong')).join('<br>');
    const missed = (b._missed || [])
      .map(m => '“' + esc(m.text) + '”: ' + esc(D.reason_short[m.reason] || m.reason)).join('<br>');
    return '<table class="kv">' +
      row('Reading', readingPills(b)) +
      (ans ? row('Your answer', ans) : '') +
      row('Kind', esc(KIND_NAME[kindOf(b)].toLowerCase()) +
          (b.split_kind ? ' <span class="sub">(the split’s)</span>' : '') +
          (kinds ? '<br><span class="sub">' + kinds + '</span>' : '')) +
      row('Chance', 'mean ' + f2(b.chance) + ' on ' + esc(b.ink_px) + ' ink px') +
      row('Angle', esc(b.angle) + '° reading direction <span class="sub">· axis ' + esc(b.axis) + '° · p(180°) ' +
          f2(b.p_180) + '</span>') +
      row('Reader', (top3 || '–') + '<br><span class="sub">0°: “' + esc(b.best_0) + '” · 180°: “' + esc(b.best_180) +
          '”</span>') +
      row('Terra', b.terra == null ? 'no Terra word overlaps' : '“' + esc(b.terra) + '” <span class="sub">' +
          esc(b.terra_kind) + ' · IoU ' + f2(b.iou) + (b.status === 'matched' ? '' : ', no match') + '</span>') +
      (missed ? row('Missed here', missed) : '') +
      row('Box', esc(b.id) + ' <span class="sub">· ' + esc(String(b.quarter || '').toUpperCase()) + ' · ' +
          esc(b.length) +
          ' × ' + esc(b.height) + ' px · grouped at ' + esc(b.level) + '</span>') + '</table>';
  }
  function missHtml(m) {
    const d = m.details || {}, r = d.reading;
    return '<div class="vs"><span class="lp" style="box-shadow:0 0 0 1.5px var(--miss)">' + esc(m.text) + '</span>' +
      '<span class="k">' + esc(m.kind) + ', no box matches it</span></div><table class="kv">' +
      row('Why', '<b>' + esc(D.reason_short[m.reason] || m.reason) + '</b>: ' + esc(D.reasons[m.reason])) +
      row('How close', 'highest chance on its ink ' + pc(d.max_chance) + ' · mean ' + pc(d.mean_chance) + ' · ' +
          pc(d.share_above) + ' of its ' + esc(d.ink_px) + ' ink px at or above ' + esc(d.level)) +
      row('Components', esc(d.components) + ' hold its ink' + (d.component ? '; the main one ' +
          Math.round(d.component_px) + ' px long (the word ' + Math.round(d.word_length_px) + ' × ' +
          Math.round(d.word_height_px) + ' px)' : '')) +
      row('Also holds', (d.merged_with || []).length ? d.merged_with.map(t => '“' + esc(t) + '”').join(', ') :
          'no other Terra word') +
      row('Our best box', m.best_box ? esc(m.best_box) + ' <span class="sub">· IoU ' + f2(m.best_iou) + '</span><br>' +
          '<button class="tb" data-act="open" data-id="' + esc(m.best_box) + '">Open that box</button>' :
          'none overlaps') +
      row('Reader at Terra’s box', r ? '“' + esc(r.text) + '” · ' + esc(r.top) + ' ' + f2(r.p_top) +
          (r.code ? ' · code ' + esc(r.code) + ' ' + f2(r.p_code) : '') : '–') +
      row('Quarter', esc(String(m.quarter || '').toUpperCase())) + '</table>';
  }
  function terraHtml(w) {
    const m = ids[key].get('t' + w.id.slice(1));
    return '<div class="vs"><span class="lp">' + esc(w.text) + '</span><span class="k">' + esc(w.kind) +
      '</span></div>' +
      '<table class="kv">' + row('This version', m ? 'misses it: ' + esc(D.reason_short[m.reason] || m.reason) +
      '<br><button class="tb" data-act="open" data-id="' + esc(m.id) + '">Why</button>' : 'a box matches it') +
      '</table>';
  }

  // the screen layer: upright pills of one size, greedily placed, none overlapping
  const cv = document.createElement('canvas').getContext('2d');
  const FAMILY = getComputedStyle(document.body).fontFamily;
  const MONO = getComputedStyle(document.documentElement).getPropertyValue('--mono').trim() || 'monospace';
  const FONT = '600 12px ' + FAMILY, TFONT = '500 11px ' + MONO;
  const widths = new Map();
  function textW(t, font) {
    const k = font + '|' + t;
    let w = widths.get(k);
    if (w == null) { cv.font = font; w = cv.measureText(t).width; widths.set(k, w); }
    return w;
  }
  const pool = [];
  let queued = false;
  function schedule() { if (!queued) { queued = true; requestAnimationFrame(layout); } }
  const CELL = 64, GAP = 3, MAXL = 400;
  function Grid() {
    const m = new Map(), key = (gx, gy) => (gy + 4) * 4096 + gx + 4;
    function each(x, y, w, h, f) {
      for (let gy = Math.floor(y / CELL); gy <= Math.floor((y + h) / CELL); gy++)
        for (let gx = Math.floor(x / CELL); gx <= Math.floor((x + w) / CELL); gx++) if (f(key(gx, gy))) return true;
      return false;
    }
    return {
      add(x, y, w, h, ref) {
        const q = [x, y, w, h, ref];
        each(x, y, w, h, k => { let c = m.get(k); if (!c) m.set(k, c = []); c.push(q); });
      },
      hits(x, y, w, h, pad, skip) {
        return each(x, y, w, h, k => (m.get(k) || []).some(q => q[4] !== skip && x < q[0] + q[2] + pad &&
          q[0] < x + w + pad && y < q[1] + q[3] + pad && q[1] < y + h + pad));
      }
    };
  }
  function layout() {
    queued = false;
    const vw = vp.clientWidth, vh = vp.clientHeight, A = S();
    const R = readingsOn() || !!L.diff, T = !!L.details, M = !!L.missed && R;
    const pills = Grid(), boxes = Grid(), vr = vp.getBoundingClientRect();
    const block = el => {
      const b = el.getBoundingClientRect();
      pills.add(b.left - vr.left, b.top - vr.top, b.width, b.height, null);
    };
    vp.querySelectorAll('.ctl, .nav, .zl').forEach(el => { if (el.offsetParent !== null) block(el); });
    if (!wide() && panel.classList.contains('open')) block(panel);
    const x0 = -tx / s, y0 = -ty / s, x1 = (vw - tx) / s, y1 = (vh - ty) / s;
    const seen = bb => bb[2] >= x0 && bb[0] <= x1 && bb[3] >= y0 && bb[1] <= y1;
    const scr = bb => [bb[0] * s + tx, bb[1] * s + ty, bb[2] * s + tx, bb[3] * s + ty];
    const me = sel && sel.rec;
    if (me && seen(me._bb)) { const q = scr(me._bb); pills.add(q[0], q[1], q[2] - q[0], q[3] - q[1], me); }
    const cands = [];
    const fence = r => {          // a box a pill had better not cover (not one filling a quarter of the view)
      const q = scr(r._bb);
      if ((q[2] - q[0]) * (q[3] - q[1]) < 0.25 * vw * vh) boxes.add(q[0], q[1], q[2] - q[0], q[3] - q[1], r);
    };
    for (const b of A.boxes) {
      if (!shownBox(b) || !seen(b._bb)) continue;
      if (b.status !== 'none') fence(b);
      if (!R && !T) continue;
      const mine = b === me;
      // a box Terra lacks with no reading has nothing to say: its outline alone shows it
      const says = b.status === 'matched' || !!b.shown || mine;
      if (R && b.status !== 'none' && says) cands.push({ r: b, t: 'read', p: mine ? -2 : kindOf(b) === 'code' ?
        (b.colour === 'purple' ? 0 : 0.5) : (b.colour === 'purple' ? 1 : 2) });
      if (T) cands.push({ r: b, t: 'tag', p: mine ? -1 : 3 });
    }
    if (L.missed) for (const m of A.missed) {
      if (!seen(m._bb)) continue;
      fence(m);
      if (M) cands.push({ r: m, t: 'miss', p: m === me ? -2 : 1.5 });
    }
    cands.sort((a, b) => a.p - b.p || b.r._len - a.r._len);
    let n = 0;
    for (const c of cands) {
      if (n >= MAXL) break;
      const r = c.r;
      let text, cls, font = FONT, h = 18, inner = null, stripe = 0;
      if (c.t === 'tag') {
        text = kindOf(r) + ' · ' + f2(r.chance) + ' · ' + (Math.round(r.angle) % 360) + '°';
        cls = 'pl tag'; font = TFONT; h = 16;
      } else if (c.t === 'miss') { text = r.text; cls = 'pl miss'; }
      else {
        stripe = 3;
        if (r.status === 'matched' && r.colour === 'purple') {
          const o = r.shown || '∅';
          text = o + ' ≠ ' + r.terra; inner = esc(o) + '<span class="ne"> ≠ </span>' + esc(r.terra); cls = 'pl bad';
        } else if (r.status === 'matched') { text = r.shown || '∅'; cls = 'pl ok'; }
        else { text = r.shown || '?'; cls = 'pl'; }
        cls += ' kc-' + kindOf(r);
      }
      if (r === me) cls += ' sel';
      const w = Math.ceil(textW(text, font)) + 12 + stripe;
      const q = scr(r._bb), cx = (q[0] + q[2]) / 2, cy = (q[1] + q[3]) / 2;
      const above = [cx - w / 2, q[1] - GAP - h], below = [cx - w / 2, q[3] + GAP];
      const right = [q[2] + GAP + 1, cy - h / 2], left = [q[0] - GAP - 1 - w, cy - h / 2];
      const spots = c.t === 'tag' ? [below, above, right, left] : [above, below, right, left];
      const fits = (x, y) => x >= 2 && y >= 2 && x + w <= vw - 2 && y + h <= vh - 2 &&
        !pills.hits(x, y, w, h, 2);
      // first a spot clear of every other box, else any spot clear of the pills already drawn
      const at = spots.find(([x, y]) => fits(x, y) && !boxes.hits(x, y, w, h, 0, r)) ||
        spots.find(([x, y]) => fits(x, y));
      if (!at) continue;
      pills.add(at[0], at[1], w, h, null);
      const e = pool[n] || (pool[n] = lab.appendChild(document.createElement('div')));
      e.className = cls; e.hidden = false;
      if (inner) e.innerHTML = inner; else e.textContent = text;
      e.style.transform = 'translate(' + Math.round(at[0]) + 'px,' + Math.round(at[1]) + 'px)';
      n++;
    }
    for (let i = n; i < pool.length; i++) pool[i].hidden = true;
    if (sel && sel.label) {
      xm.hidden = false;
      xm.style.transform = 'translate(' + Math.round(sel.label.x * s + tx) + 'px,' +
        Math.round(sel.label.y * s + ty) + 'px)';
    } else xm.hidden = true;
  }

  const ab = $('aboutb'), about = $('about');
  ab.addEventListener('click', () => {
    about.hidden = !about.hidden; ab.setAttribute('aria-expanded', String(!about.hidden)); schedule();
  });

  // start: the whole sheet in view (a link's #view=x,y,zoom, #set=, #layers=, #nav=, #tap=x,y and #theme= are
  // for checking the page by eye from a script)
  const hash = location.hash;
  const th = /theme=(light|dark)/.exec(hash);
  if (th) document.documentElement.setAttribute('data-theme', th[1]);
  const sk = /set=([\w-]+)/.exec(hash);
  if (sk && D.sets[sk[1]]) key = sk[1];
  const ly = /layers=([\w,]*)/.exec(hash);
  if (ly) {
    for (const k in L) L[k] = k === 'readings' ? null : false;
    ly[1].split(',').forEach(k => { if (k in L) L[k] = true; });
  }
  syncSet(); syncChips(); renderPanel(); renderNav();
  const v = /view=([\d.]+),([\d.]+),([\d.]+)/.exec(hash);
  if (v) {
    const r = vp.getBoundingClientRect();
    s = +v[3]; tx = r.width / 2 - +v[1] * s; ty = r.height / 2 - +v[2] * s; apply();
  } else fit();
  const nv = /nav=(\d+)/.exec(hash);
  if (nv) openMistake(+nv[1], !v);
  const tp = /tap=([\d.]+),([\d.]+)/.exec(hash);
  if (tp) tap(+tp[1], +tp[2]);
  let lastW = vp.clientWidth;
  window.addEventListener('resize', () => {
    if (s <= minS * 2.01 || Math.abs(vp.clientWidth - lastW) > 200) fit(); else schedule();
    lastW = vp.clientWidth;
    if (wide() && mode === 'list') renderPanel();
  });
})();
"""


def _header(d: dict, sets) -> str:
    """The compact header: the sheet, the selected version's three numbers, the legend in one line."""
    m = re.search(r"-p(\d+)$", str(d["sheet"]))
    short = f"p{m.group(1)}" if m else str(d["sheet"])
    n = _set_numbers(sets[0][2])
    right = f'{n["right"]}/{n["n"]}' if n["right"] is not None else "–"
    swatches = "".join(f'<span><i class="sw s-{k}"></i>{k}</span>' for k in KIND_CLASSES)
    return ('<header class="top">'
            f'<h1 title="{_e(d["sheet"])}">Mission St <span class="sid">· {_e(short)}</span></h1>'
            '<button class="tb" id="aboutb" aria-expanded="false" aria-controls="about">About</button>'
            f'<div class="nums"><span><b id="n-right">{right}</b> labels right</span>'
            f'<span><b id="n-match">{n["matched"]:,}</b> matched<span class="wd"> words</span></span>'
            f'<span><b id="n-merged">{n["merged"]:,}</b> merged<span class="wd"> words</span></span></div>'
            f'<div class="legend" aria-label="Legend">{swatches}<span class="lp ok">= Terra</span>'
            '<span class="lp bad" title="our reading differs from Terra\'s">≠ Terra</span></div></header>')


def _toolbar(sets) -> str:
    """The version control (when there are several) and the layers' chips."""
    seg = ""
    if len(sets) > 1:
        seg = ('<div class="seg" role="group" aria-label="Boxes from">'
               + "".join(f'<button data-set="{_e(k)}" aria-pressed="{str(i == 0).lower()}">{_e(name)}</button>'
                         for i, (k, name, _) in enumerate(sets)) + "</div>")
    n = _set_numbers(sets[0][2])
    chips = [("readings", f'Readings<span class="auto"> · from {READINGS_FROM:g}x</span>'),
             ("missed", '<i class="sw s-miss"></i>Missed words'),
             ("terra", '<i class="sw s-terra"></i>Terra\'s words'),
             ("textink", '<i class="dot"></i>Text ink'),
             ("frags", '<i class="sw s-frag"></i>Fragments'),
             ("details", "Details"),
             ("diff", f'≠ Terra only <span class="cnt" id="n-diff">{n["differs"]:,}</span>')]
    return (f'<div class="bar">{seg}<div class="chips" role="group" aria-label="Layers">'
            + "".join(f'<button class="chip" data-layer="{k}" aria-pressed="false">{t}</button>' for k, t in chips)
            + "</div></div>")


def _about(d: dict, about: str) -> str:
    """The folded explanations: what each layer draws, the build's own note, the panels and what drew the page."""
    h, w = d["shape"]
    run = f' · run <b>{_e(d["run"])}</b>' if d.get("run") else ""
    models = "".join(f"<div>{_e(m)}</div>" for m in d["models"])
    layers = (
        '<ul><li><b>The scan</b>: the page\'s ink binarised (px darker than '
        f'{grid.INK_BELOW}, the detector\'s own input), all in one grey; not the scan\'s grey levels.</li>'
        '<li><b>Box outlines</b>: our boxes read as text, coloured by kind (code, block number, street name, other): '
        'crude-kind\'s by the kind its split gave the word, the others\' by the box\'s likeliest kind.</li>'
        '<li><b>Readings</b>: one upright pill per box, beside it: green when our reading is Terra\'s string (a '
        'trailing "." and doubled spaces folded; a printed form counts as its code), purple "ours ≠ Terra\'s" when '
        f'it is not, white for text Terra has no word for.  On by themselves from {READINGS_FROM:g}x.  Codes are '
        'drawn first, then the readings that differ, then the rest; a pill that would overlap another waits until '
        'you zoom in.</li>'
        f'<li><b>Missed words</b>: Terra\'s words no box matches at IoU ≥ {d["iou"]}, thick pink outlines; tap one '
        'for why.</li><li><b>Terra\'s words</b>: every word Terra saw, thin dashed outlines.</li>'
        f'<li><b>Text ink</b>: red, the ink the detector calls text (chance ≥ {TEXT_CHANCE}).</li>'
        '<li><b>Fragments</b>: our boxes the reader calls no text, thin grey.</li>'
        '<li><b>Details</b>: each box\'s kind, mean chance and reading angle.</li>'
        '<li><b>≠ Terra only</b>: keeps the boxes whose reading differs from Terra\'s.</li>'
        '<li><b>Wrong labels</b> (‹ › at the foot of the map): your answered labels this version reads wrong at its '
        'own boxes (a label with no box counts wrong), with why when known: merged (the box holds another Terra '
        'word too), partial (it holds half or less of the code word), misread (a clean box read wrong), no box, '
        'other.</li></ul>')
    return (f'<section id="about" hidden aria-label="About this page"><h2>About this page</h2>'
            f'<p>Sheet <b>{_e(d["sheet"])}</b>{run} · {w:,} × {h:,} px at full resolution · boxes matched to Terra\'s '
            f'words at IoU ≥ {d["iou"]}.  Drag or pinch to move and zoom; tap a box for everything about it.</p>'
            f'<h3>What is drawn</h3>{layers}{about}{_panels(d)}'
            f'<div class="models"><b>Drawn by</b>{models}</div></section>')


def render(d: dict, out: Path, *, sets=None, title: str = "Mission Sheet Review", about: str = "") -> Path:
    """Write the page of `page_data`'s `d` to `out` (one self-contained HTML file, the content of a page without
    `<html>`, `<head>` or `<body>`: an Artifact's page).  `sets`: the versions the page flips between, (key, name,
    `page_data` of that version) in order, the first shown first (None: `d` alone); every version is drawn over
    `d`'s image.  `about`: HTML added to the folded explanations."""
    sets = list(sets) if sets else [("run", d.get("run") or "this run", d)]
    h, w = d["shape"]
    faded = bool(d.get("faded"))
    tiles = []
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            part = d["pixels"][y0:y0 + TILE, x0:x0 + TILE]
            for layer, cls in (("base", "base"), ("text", "ti")):
                png = _tile_png(part, layer, faded)
                if png is not None:
                    tiles.append(f'<img class="tile {cls}" alt="" data-x="{x0}" data-y="{y0}" '
                                 f'width="{part.shape[1]}" height="{part.shape[0]}" style="left:{x0}px;top:{y0}px" '
                                 f'src="data:image/png;base64,{png}">')
    js_data = {"shape": d["shape"], "order": [k for k, _, _ in sets], "readings_from": READINGS_FROM,
               "max_scale": MAX_SCALE, "reasons": d.get("reason_text", REASON_TEXT), "reason_short": REASON_SHORT,
               "causes": CAUSE_NAMES, "terra": d.get("terra_words", []),
               "sets": {k: {"name": name, "boxes": dk["boxes"], "missed": dk["missed"], "labels": dk.get("labels", []),
                            "nums": _set_numbers(dk)} for k, name, dk in sets}}
    payload = json.dumps(js_data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    overlays = "".join(_outlines(dk, k, i > 0) for i, (k, _, dk) in enumerate(sets))
    page_html = (
        f'<title>{_e(title)}</title>\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<meta name="claude:viewport-fit-sides" content="cover">\n'
        f"<style>{STYLE}</style>\n"
        f'<div class="app">{_header(d, sets)}{_about(d, about)}{_toolbar(sets)}'
        '<div class="work" id="work"><div id="vp" aria-label="The sheet: drag or pinch to move and zoom, tap a box">'
        f'<div id="stage" style="width:{w}px;height:{h}px">{"".join(tiles)}{overlays}{_shared_overlays(d)}</div>'
        '<div id="lab" aria-hidden="true"><div class="xm" id="xm" hidden></div></div>'
        '<div class="zl" id="zl"></div>'
        '<div class="ctl"><button id="zin" aria-label="Zoom in">+</button>'
        '<button id="zout" aria-label="Zoom out">−</button><button id="zfit" aria-label="Fit the sheet">Fit</button>'
        '</div>'
        '<div class="nav" id="nav"><button data-act="prev" aria-label="Previous wrong label">‹</button>'
        '<button data-act="list" class="navc" id="navc">wrong labels</button>'
        '<button data-act="next" aria-label="Next wrong label">›</button></div>'
        '</div><aside id="panel" aria-label="Details"></aside></div></div>\n'
        f'<script type="application/json" id="data">{payload}</script>\n'
        f"<script>{SCRIPT}</script>\n")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page_html, encoding="utf-8")
    return out


# --- reading the page back (the tests' check that no shading covers the red) -----------------------------------------

_TILE_RE = re.compile(r'<img class="tile (\w+)"[^>]*data-x="(\d+)" data-y="(\d+)"[^>]*'
                      r'src="data:image/png;base64,([^"]+)"')
_SHAPE_RE = re.compile(r"<(polygon|rect|path)\b([^>]*)>")
_ATTR_RE = re.compile(r'([\w:-]+)="([^"]*)"')


def _rgb(fill: str):
    """An SVG fill as (r, g, b, alpha), or None for none or transparent."""
    f = fill.strip().lower()
    if f in ("", "none", "transparent"):
        return None
    if f.startswith("#"):
        f = f[1:]
        if len(f) == 3:
            f = "".join(c * 2 for c in f)
        return int(f[0:2], 16), int(f[2:4], 16), int(f[4:6], 16), 1.0
    m = re.match(r"rgba?\(([^)]*)\)", f)
    if m:
        v = [float(x) for x in m.group(1).split(",")]
        return int(v[0]), int(v[1]), int(v[2]), v[3] if len(v) > 3 else 1.0
    return 0, 0, 0, 1.0          # a named colour: not red


def page_image(path: Path, layers=("base", "ti")) -> np.ndarray:
    """The page's image as drawn in the light theme (uint8 (h, w, 3) RGB): its tiles of `layers` in order ("base" the
    paper and grey ink, "ti" the text ink layer, transparent off the red), and over them every filled polygon, rect or
    path of the overlay (`fill` not none), at its fill and opacity."""
    text = Path(path).read_text(encoding="utf-8")
    tiles = [(layer, int(x), int(y), np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGBA")))
             for layer, x, y, b in _TILE_RE.findall(text)]
    h = max(y + t.shape[0] for _, _, y, t in tiles)
    w = max(x + t.shape[1] for _, x, _, t in tiles)
    img = np.zeros((h, w, 3), np.uint8)
    for want in layers:
        for layer, x, y, t in tiles:
            if layer == want:
                win = img[y:y + t.shape[0], x:x + t.shape[1]]
                a = t[..., 3:] / 255.0
                win[:] = np.round(win * (1 - a) + t[..., :3] * a).astype(np.uint8)
    for tag, attrs in _SHAPE_RE.findall(text):
        a = dict(_ATTR_RE.findall(attrs))
        fill = _rgb(a.get("fill", "#000"))          # SVG fills a shape black by default
        if fill is None:
            continue
        alpha = fill[3] * float(a.get("fill-opacity", 1)) * float(a.get("opacity", 1))
        mask = np.zeros((h, w), np.uint8)
        if tag == "rect":
            x0, y0 = float(a.get("x", 0)), float(a.get("y", 0))
            pts = [np.array([[x0, y0], [x0 + float(a["width"]), y0], [x0 + float(a["width"]), y0 + float(a["height"])],
                             [x0, y0 + float(a["height"])]])]
        elif tag == "polygon":
            pts = [np.array([float(v) for v in re.split(r"[\s,]+", a["points"].strip())]).reshape(-1, 2)]
        else:
            pts = [np.array([float(v) for v in re.findall(r"-?[\d.]+", part)]).reshape(-1, 2)
                   for part in re.split(r"[Mm]", a.get("d", "")) if part.strip()]
        cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in pts if len(p) >= 3], 1)
        on = mask.astype(bool)
        img[on] = np.round(img[on] * (1 - alpha) + np.array(fill[:3]) * alpha).astype(np.uint8)
    return img


def red_px_after_overlay(path: Path) -> int:
    """The px of the page as drawn (`page_image`) still in the text ink's red."""
    return int((page_image(path) == np.array(COLOURS["text_ink"], np.uint8)).all(-1).sum())


def experiments_dir() -> Path:
    """Where the review pages are written: data/solver/experiments/wordread/review."""
    return config.DATA_DIR / "solver" / "experiments" / "wordread" / "review"
