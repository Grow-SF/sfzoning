"""The Mission Street review page: the text reader's output on one whole 2000 sheet, against Terra's words, as one
self-contained HTML page.  The owner judges every step of the plan 2026-11-17-map-text-fields-way by it.

The sheet (`mission_sheet`): the 2000 sheet whose footprint (the warp's `warped.footprint.geojson`, map ft) holds the
longest length of DataSF's MISSION ST centreline (`streetside.load_streets`, the solver's centreline loader).

What the page shows (`page_data`, then `render`):
  the image     the page at full resolution as palette PNG tiles: white paper, grey ink, and red the ink the detector
                calls text (chance >= TEXT_CHANCE; each px from the fold detector that held out its quarter).  The paper
                inside a green or a yellow box is tinted, never the ink, so the shading sits under the red.
  our boxes     (SVG) every box: its grouped rectangle, an arrow along its reading direction (its axis, turned a half
                turn when the reader's 0/180 classifier says so) and a tag (kind, mean chance, reading angle).
                  matched  its best Terra word meets it at IoU >= the page's level (`measures`' rule): its reading in
                           green when it is Terra's, purple when it isn't, with Terra's string after it in red
                  text     it matches no word and is read as text: shaded green (text Terra missed)
                  none     it matches no word and is read as no text: a thin grey outline (junk)
  missed words  Terra's words that no box matches: shaded yellow, dashed, with `miss_reason`'s code and details
  panels        the top-line numbers of the boxes (`measures.box_numbers`) and of the readings, with the change
                since the run before
  tapping       a box or a missed word shows everything known about it

The reading shown for a box: when the reader's likeliest hypothesis is a code, its likeliest code given a code; for
other text, the reader's open best path in the direction the classifier chose (`textread.model.best_path`, by the
fold reader that never saw the box's quarter); for "none", nothing.  It is compared with Terra's string after
`textread.crops.norm_text`, and a printed form of the code list counts as its code (`colour`).

Why a word was missed (`miss_reason`): from the detector's own chance on the word's ink, at the level its quarter's
boxes were grouped at (`wordboxes`: the connected px at or above the level within INK_R px of ink, the components).
The component holding most of its above-level ink is its main component.
  merged_with_word  the main component also holds another Terra word: MERGE_PART or more of that word's ink at the
                    level outside this word's rectangle, or most of that word's own above-level ink
  merged_into_line  the main component, holding no other word, is more than LINE_RATIO times the word's length
  outside_map       its centre is off the sheet's footprint or the mapped area: a box there is dropped
  below_level       less than BELOW_SHARE of its ink reaches the level (or it has no ink)
  split             its above-level ink falls in two or more components, each holding SPLIT_PART of it or more
  wrong_size        the rest: one component holds it, alone and not a line, but its box doesn't fit the word
in that order (the controller's ruling, Task 2 fix round 1).  The details give the highest and the mean chance on its
ink, the share of its ink at or above the level, the main component's length and ink px, the other words it holds,
and the reader's reading at Terra's box when it is given.

The green/purple comparison folds a trailing "." and runs of spaces (`colour`), so "ST" is Terra's "ST.".
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
REASONS = ("merged_with_word", "merged_into_line", "outside_map", "below_level", "split", "wrong_size")
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
TILE = 2048                # px: the image is cut into tiles at most this big each way
LABELS_FROM = 0.8          # the overlay's readings and arrows are drawn from this zoom (screen px per page px) up,
TAGS_FROM = 2.0            # its tags and the missed words' details from this
MAX_SCALE = 24.0

# the image's palette (RGB), by index
COLOURS = {"paper": (255, 255, 255), "green_paper": (201, 234, 211), "yellow_paper": (250, 236, 166),
           "both_paper": (198, 220, 145), "other_ink": (154, 154, 154), "text_ink": (221, 34, 34)}
INDEX = {name: k for k, name in enumerate(COLOURS)}
# the overlay's colours (on the white map, in both themes)
MATCH, DIFF, TERRA_RED, BOX, FOUND, JUNK, MISSED, MISSED_TEXT = (
    "#1a8a3a", "#7b3fbf", "#d22222", "#333333", "#2a8a4a", "#9a9a9a", "#b08a00", "#5a4600")


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
    if details["merged_with"]:
        return "merged_with_word", details
    if details["component_px"] > LINE_RATIO * length:
        return "merged_into_line", details
    if not on_map:
        return "outside_map", details
    if not n_ink or details["share_above"] < BELOW_SHARE:
        return "below_level", details
    if int((counts >= SPLIT_PART * n_above).sum()) >= 2:
        return "split", details
    return "wrong_size", details


def miss_reasons(terra: pd.DataFrame, chance: np.ndarray, ink: np.ndarray, levels: dict, on_map, words,
                 terra_reads: dict | None = None) -> dict[int, tuple[str, dict]]:
    """`miss_reason` of each of Terra's words at positions `words` (of `terra`; its quarter's level from `levels`,
    quarter name -> level), the page's components found once per level, every other word of `terra` the `others`
    and, of those, the words whose own main component is this word's its `shares`."""
    gate = wordboxes.ink_gate(ink)
    found: dict[float, Components] = {}
    rects = terra[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2)
    names = [str(s) for s in terra["text"]]
    level = [float(levels[paths.QUARTERS[int(q)]]) for q in terra["quarter"]]
    held_by: dict[tuple, list[int]] = {}
    main = []
    for t in range(len(terra)):
        if level[t] not in found:
            found[level[t]] = components(chance, ink, level[t], gate)
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


def page_data(sheet: str, boxes: pd.DataFrame, reads: dict, terra: pd.DataFrame, chance: np.ndarray,
              before: dict | None = None, *, ink: np.ndarray | None = None, levels: dict | None = None,
              on_map: np.ndarray | None = None, terra_reads: dict | None = None, answers: pd.DataFrame | None = None,
              forms: dict | None = None, models=(), run: str = "", iou: float = MATCH_IOU) -> dict:
    """Everything the page draws for `sheet`: `boxes` our boxes (wordboxes' columns; the index is the row `reads` is
    keyed by), `reads` their readings (`reads_for_page`), `terra` Terra's words (`measures.terra_words`' columns),
    `chance` the sheet's chance of text (uint8, chance x 255), `before` the numbers of the run before (None: none).
    Read from disk when None: `ink` (the page < grid.INK_BELOW), `levels` (quarter name -> grouping level), `on_map`
    (each Terra word's centre placed where boxes are kept), `forms` (printed form -> code).  `terra_reads`: Terra word
    position -> the reading at Terra's box; `answers`: the owner's answered labels on the sheet
    (`measures.answers_from_store`' columns); `models`: lines naming the models that drew it.

    Returns {"sheet", "run", "shape", "pixels" (`palette_image`), "text_ink_px", "boxes", "missed", "numbers",
    "delta" (`measures.compare` with `before`, or None), "models", "iou", "levels"}."""
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
    reasons = miss_reasons(terra, chance, ink, levels, on_map, np.flatnonzero(~matched_hi), terra_reads)

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
    return {"sheet": sheet, "run": run, "shape": list(ink.shape),
            "pixels": palette_image(ink, text_ink, green_polys, yellow_polys), "text_ink_px": int(text_ink.sum()),
            "boxes": recs, "missed": missed, "numbers": numbers,
            "delta": measures.compare(numbers, before) if before else None, "models": list(models), "iou": iou,
            "levels": {k: float(v_) for k, v_ in levels.items()}}


def red_px_of(d: dict) -> int:
    """The px the page should show red: the ink the detector calls text."""
    return int(d["text_ink_px"])


# --- the page -----------------------------------------------------------------------------------------------------

def _png_b64(idx: np.ndarray) -> str:
    im = Image.fromarray(idx, "P")
    im.putpalette([c for rgb in COLOURS.values() for c in rgb])
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _f(v, k=1) -> str:
    return f"{v:.{k}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def _path(polys) -> str:
    return "".join("M" + "L".join(f"{_f(x)} {_f(y)}" for x, y in np.asarray(p, float).reshape(-1, 2)) + "Z"
                   for p in polys)


def _frame(rec: dict):
    """A box's or word's centre, reading direction v, its down-the-page normal n, length and height."""
    p = np.asarray(rec["poly"], float)
    a = np.deg2rad(rec["angle"])
    v, n = np.array([np.cos(a), -np.sin(a)]), np.array([np.sin(a), np.cos(a)])
    length = float(np.abs((p - p.mean(0)) @ v).max() * 2)
    height = float(np.abs((p - p.mean(0)) @ n).max() * 2)
    return p.mean(0), v, n, max(length, 4.0), max(height, 4.0)


def _text(xy, angle, size, fill, body, cls="") -> str:
    x, y = xy
    c = f' class="{cls}"' if cls else ""
    return (f'<text{c} transform="translate({_f(x)} {_f(y)}) rotate({_f(-angle)})" font-size="{_f(size)}" '
            f'fill="{fill}">{body}</text>')


def _overlay(d: dict) -> str:
    """The SVG overlay (page px): outlines and arrows as a few paths, the words as text."""
    h, w = d["shape"]
    out = [(f'<svg id="ov" xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            'font-family="ui-monospace,Menlo,Consolas,monospace">')]
    outlines = {"matched": [], "text": [], "none": []}
    arrows = {"matched": [], "text": [], "none": []}
    words = []
    for b in d["boxes"]:
        outlines[b["status"]].append(b["poly"])
        c, v, n, length, height = _frame(b)
        base = c + n * (height / 2 + 1.5)
        a0, a1 = base - v * length * 0.35, base + v * length * 0.35
        hd = min(4.0, 0.25 * length)
        h1, h2 = a1 - v * hd + n * hd * 0.6, a1 - v * hd - n * hd * 0.6
        arrows[b["status"]].append(f"M{_f(a0[0])} {_f(a0[1])}L{_f(a1[0])} {_f(a1[1])}M{_f(h1[0])} {_f(h1[1])}"
                                   f"L{_f(a1[0])} {_f(a1[1])}L{_f(h2[0])} {_f(h2[1])}")
        fs = float(np.clip(0.6 * height, 6.0, 12.0))
        ts = max(4.0, fs * 0.55)
        kind = KIND_SHORT.get(b["kind"], b["kind"] or "?")
        tag = f'{kind} · {b["chance"] or 0.0:.2f} · {round(b["angle"]) % 360}°'
        tag_at = c - v * length / 2 + n * (height / 2 + 3.0 + ts)
        tag_fill = {"matched": BOX, "text": FOUND, "none": JUNK}[b["status"]]
        words.append(_text(tag_at, b["angle"], ts, tag_fill, _e(tag), "tag"))
        if b["status"] == "none":
            continue
        top_at = c - v * length / 2 - n * (height / 2 + 2.0)
        if b["status"] == "matched":
            fill = MATCH if b["colour"] == "green" else DIFF
            body = _e(b["shown"] or "∅")
            if b["colour"] == "purple":
                body += f' <tspan fill="{TERRA_RED}">(Terra: {_e(b["terra"])})</tspan>'
        else:
            fill, body = BOX, _e(b["shown"] or "?")
        words.append(_text(top_at, b["angle"], fs, fill, body, "rd"))
    for m in d["missed"]:
        c, v, n, length, height = _frame(m)
        fs = float(np.clip(0.6 * height, 6.0, 12.0))
        ts = max(4.0, fs * 0.5)
        det = m["details"]
        words.append(_text(c - v * length / 2 - n * (height / 2 + 2.0), m["angle"], fs, TERRA_RED,
                           f'Terra: {_e(m["text"])}', "rd"))
        line1 = (f'{REASON_SHORT[m["reason"]]} · max {det["max_chance"] or 0.0:.2f} · '
                 f'{round(100 * (det["share_above"] or 0.0))}% ≥ {det["level"]:g}')
        bits = []
        if m["reason"] == "merged_with_word":
            bits.append(f'with {_e(", ".join(det.get("merged_with", [])[:2]))} ({round(det["component_px"])} px)')
        elif m["reason"] == "merged_into_line":
            bits.append(f'line {round(det["component_px"])} px')
        elif m["reason"] == "split":
            bits.append(f'{det["components"]} parts')
        rd = det.get("reading")
        if rd:
            said = rd.get("text") or "∅"
            bits.append(f"reader: {_e(said)} " + (f'{rd["p_top"]:.2f}' if rd["top"] == said
                                                  else f'({_e(rd["top"])} {rd["p_top"]:.2f})'))
        at = c - v * length / 2 + n * (height / 2 + 2.0 + ts)
        words.append(_text(at, m["angle"], ts, MISSED_TEXT, _e(line1), "tag"))
        if bits:
            words.append(_text(at + n * ts * 1.25, m["angle"], ts, MISSED_TEXT, " · ".join(bits), "tag"))
    out.append(f'<path class="ln" d="{_path([m["poly"] for m in d["missed"]])}" fill="none" stroke="{MISSED}" '
               'stroke-width="1.5" stroke-dasharray="5 3"/>')
    for status, stroke, width in (("none", JUNK, 0.8), ("text", FOUND, 1.4), ("matched", BOX, 1.4)):
        out.append(f'<path class="ln" d="{_path(outlines[status])}" fill="none" stroke="{stroke}" '
                   f'stroke-width="{width}"/>')
        out.append(f'<path class="ln lbl" d="{"".join(arrows[status])}" fill="none" stroke="{stroke}" '
                   f'stroke-width="{width}"/>')
    out.append('<g class="lbl">' + "".join(words) + "</g></svg>")
    return "".join(out)


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
    """The two top-line panels, the legend and the reasons (HTML)."""
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
             f'missed <b>{_n(b["missed"])}</b> ({_pct(b["missed"], b["terra"])}, yellow)'),
            (f'Ours, not in Terra: read as text <b>{_n(b["ours_unmatched_text"])}</b> (green) · read as none '
             f'<b>{_n(b["ours_unmatched_none"])}</b> (grey outlines)'),
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
              f'<b>{_n(rd["same"])}</b> ({_pct(rd["same"], rd["matched"])}, green) · different '
              f'<b>{_n(rd["differs"])}</b> (purple)'),
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
                    f'<td>{_e(REASON_TEXT[r])}</td></tr>' for r in REASONS)
    legend = (f'<span class="sw" style="background:rgb{COLOURS["text_ink"]}"></span> ink the detector calls text '
              f'(chance ≥ {TEXT_CHANCE})<br>'
              f'<span class="sw" style="background:rgb{COLOURS["other_ink"]}"></span> other ink<br>'
              '<span class="chip" style="border:2px solid var(--ink)">box</span> our box; arrow = reading direction; '
              'tag = kind · chance · angle<br>'
              f'<b style="color:var(--match)">RH-2</b> our reading = Terra\'s<br>'
              f'<b style="color:var(--diff)">NC-2</b> <span style="color:var(--text)">(Terra: NC-3)</span> our reading '
              'differs<br>'
              f'<span class="chip" style="background:rgb{COLOURS["green_paper"]};border:1.5px solid {FOUND};'
              'color:#1d2320">green shade</span> we found text Terra missed (its reading in black)<br>'
              f'<span class="chip" style="border:1px solid {JUNK};color:{JUNK}">grey</span> our box read as no text'
              '<br>'
              f'<span class="chip" style="background:rgb{COLOURS["yellow_paper"]};border:1.5px dashed {MISSED};'
              'color:#1d2320">yellow shade</span> Terra saw a word we missed, with why')
    return (f'<div class="grid"><section class="box"><h2>Boxes (whole sheet)</h2>{boxes_html}</section>'
            f'<section class="box"><h2>Text read</h2>{"<br>".join(trows)}</section>'
            f'<section class="box"><h2>Legend</h2>{legend}</section>'
            f'<section class="box"><h2>Why we missed Terra\'s words (IoU ≥ {lv})</h2>'
            f'<table class="rs">{rrows}</table></section></div>')


STYLE = """
:root { --bg: #f4f5f2; --paper: #ffffff; --ink: #1d2320; --muted: #5b655f; --rule: #d7dbd4; --text: #d22222;
  --match: #1a8a3a; --diff: #7b3fbf; --view: #e7e9e4;
  --font: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; --mono: ui-monospace, Menlo, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #141816; --paper: #1b201d;
  --ink: #e5eae6; --muted: #a0aaa4; --rule: #2e3531; --text: #ff6b6b; --match: #4cc274; --diff: #b48cf0;
  --view: #0d100f; color-scheme: dark } }
:root[data-theme="dark"] { --bg: #141816; --paper: #1b201d; --ink: #e5eae6; --muted: #a0aaa4; --rule: #2e3531;
  --text: #ff6b6b; --match: #4cc274; --diff: #b48cf0; --view: #0d100f; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--font); margin: 0; }
.wrap { max-width: 1280px; margin: 0 auto; padding-block: 20px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-size: 22px; margin: 0 0 4px; text-wrap: balance; }
.sub { color: var(--muted); margin: 0 0 14px; overflow-wrap: anywhere; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); gap: 12px; }
.box { background: var(--paper); border: 1px solid var(--rule); padding: 12px 14px; min-width: 0;
  overflow-wrap: anywhere; }
.box h2 { font: 600 12px/1.2 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted);
  margin: 0 0 8px; }
.chip { display: inline-block; padding: 0 6px; }
.sw { display: inline-block; width: 12px; height: 12px; vertical-align: -1px; }
table.rs { border-collapse: collapse; width: 100%; font-size: 14px; }
table.rs td { border-top: 1px solid var(--rule); padding: 4px 6px 4px 0; vertical-align: top;
  overflow-wrap: normal; }
table.rs td:first-child, td.num { white-space: nowrap; }
td.num { text-align: right; font-variant-numeric: tabular-nums; padding-right: 10px; }
.viewer { margin-top: 12px; border: 1px solid var(--rule); background: var(--paper); }
.bar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; padding: 8px;
  border-bottom: 1px solid var(--rule); }
.bar button { font: 600 14px var(--font); min-width: 40px; min-height: 36px; padding: 0 10px; border-radius: 6px;
  border: 1px solid var(--rule); background: var(--bg); color: var(--ink); cursor: pointer; }
.bar .hint { color: var(--muted); font-size: 13px; }
#vp { position: relative; height: min(78vh, 900px); overflow: hidden; touch-action: none; background: var(--view);
  cursor: grab; user-select: none; -webkit-user-select: none; }
#vp.drag { cursor: grabbing; }
#stage { position: absolute; left: 0; top: 0; transform-origin: 0 0; background: #fff; }
#stage img { position: absolute; image-rendering: pixelated; pointer-events: none; -webkit-user-drag: none; }
#ov { position: absolute; left: 0; top: 0; pointer-events: none; overflow: visible; }
#ov .ln { vector-effect: non-scaling-stroke; }
#stage.far .lbl, #stage.mid .tag { display: none; }
#info { position: absolute; left: 8px; right: 8px; bottom: 8px; max-height: 46%; overflow: auto; display: none;
  background: var(--paper); color: var(--ink); border: 1px solid var(--rule); border-radius: 8px; padding: 10px 12px;
  font-size: 13px; box-shadow: 0 4px 18px rgba(0,0,0,.25); cursor: auto; }
#info.on { display: block; }
#info h3 { font-size: 14px; margin: 6px 0 4px; }
#info table { border-collapse: collapse; width: 100%; }
#info td { padding: 2px 6px 2px 0; vertical-align: top; border-top: 1px solid var(--rule); }
#info td:first-child { color: var(--muted); white-space: nowrap; }
#info .x { float: right; border: 0; background: none; color: var(--ink); font-size: 20px; cursor: pointer;
  min-width: 36px; min-height: 32px; }
@media (min-width: 900px) { #info { left: auto; width: 380px; top: 8px; max-height: none; } }
.models { color: var(--muted); font-size: 13px; margin-top: 12px; overflow-wrap: anywhere; }
"""

SCRIPT = """
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const W = D.shape[1], H = D.shape[0], LBL = %(labels)s, TAGS = %(tags)s, MAXS = %(maxs)s;
  const vp = document.getElementById('vp'), st = document.getElementById('stage');
  const info = document.getElementById('info'), zl = document.getElementById('zl');
  let s = 1, tx = 0, ty = 0, minS = 0.05;
  function apply() {
    st.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + s + ')';
    st.classList.toggle('far', s < LBL); st.classList.toggle('mid', s < TAGS);
    zl.textContent = (s >= 1 ? s.toFixed(1) : s.toFixed(2)) + 'x';
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
  document.getElementById('zin').onclick = () => centre(2);
  document.getElementById('zout').onclick = () => centre(0.5);
  document.getElementById('zfit').onclick = fit;
  vp.addEventListener('wheel', e => {
    e.preventDefault();
    const r = vp.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)));
  }, { passive: false });
  const pts = new Map(); let moved = 0, startT = 0, pinch = null;
  vp.addEventListener('pointerdown', e => {
    if (info.contains(e.target)) return;
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
    const r = vp.getBoundingClientRect(); zoomAt(e.clientX - r.left, e.clientY - r.top, 2);
  });
  function inside(poly, x, y) {
    let c = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  }
  function near(poly, x, y, tol) {
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (const [px, py] of poly) {
      x0 = Math.min(x0, px); y0 = Math.min(y0, py); x1 = Math.max(x1, px); y1 = Math.max(y1, py);
    }
    if (x < x0 - tol || x > x1 + tol || y < y0 - tol || y > y1 + tol) return false;
    return inside(poly, x, y) || tol > 0;
  }
  const ENT = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
  const esc = t => String(t == null ? '' : t).replace(/[&<>"]/g, c => ENT[c]);
  const f2 = v => v == null ? '–' : (+v).toFixed(2);
  const pc = v => v == null ? '–' : Math.round(100 * v) + '%%';
  const row = (k, v) => '<tr><td>' + k + '</td><td>' + v + '</td></tr>';
  function boxHtml(b) {
    const kinds = Object.entries(b.p_kind || {}).sort((a, c) => c[1] - a[1])
      .map(([k, p]) => k + ' ' + f2(p)).join(' · ');
    const top3 = (b.top3 || []).map(([h, p]) => esc(h) + ' ' + f2(p)).join(' · ');
    const what = { matched: 'matched to a Terra word', text: 'not in Terra, read as text (green)',
                   none: 'not in Terra, read as no text (grey)' }[b.status];
    const col = b.colour === 'green' ? 'var(--match)' : b.colour === 'purple' ? 'var(--diff)' : 'var(--ink)';
    return '<h3>Our box ' + esc(b.id) + ': ' + what + '</h3><table>' +
      row('reading', '<b style="color:' + col + '">' + (esc(b.shown) || '∅ (no text)') + '</b>' +
          (b.colour === 'purple' ? ' <span style="color:var(--text)">(Terra: ' + esc(b.terra) + ')</span>' : '')) +
      row('top three', top3 || '–') +
      row('best paths', '0°: “' + esc(b.best_0) + '” · 180°: “' + esc(b.best_180) + '” · p(180°) ' +
          f2(b.p_180)) +
      row('reading direction', b.angle + '° (axis ' + b.axis + '°)') +
      row('kind', kinds || '–') +
      row('chance', 'mean ' + f2(b.chance) + ' on its ' + b.ink_px + ' ink px · grouped at ' + b.level) +
      row('size', b.length + ' × ' + b.height + ' px') +
      row('Terra', b.terra == null ? 'no word overlaps' :
          '“' + esc(b.terra) + '” (' + esc(b.terra_kind) + ') · IoU ' + f2(b.iou)) +
      row('quarter', b.quarter + ' (read by its fold reader)') + '</table>';
  }
  function missHtml(m) {
    const d = m.details || {}, r = d.reading;
    return '<h3>Terra word we missed: “' + esc(m.text) + '” (' + esc(m.kind) + ')</h3><table>' +
      row('why', '<b>' + esc(m.reason.replace(/_/g, ' ')) + '</b>: ' + esc(D.reasons[m.reason])) +
      row('how close', 'highest chance on its ink ' + pc(d.max_chance) + ' · mean ' + pc(d.mean_chance) + ' · ' +
          pc(d.share_above) + ' of its ' + d.ink_px + ' ink px at or above ' + d.level) +
      row('components', d.components + ' hold its ink at the level' + (d.component ? '; the main one ' +
          Math.round(d.component_px) + ' px long (the word ' + Math.round(d.word_length_px) + ' × ' +
          Math.round(d.word_height_px) + ' px), ' + d.component_ink_px + ' ink px, ' + pc(d.component_share) +
          ' of its ink' : '')) +
      row('that component also holds', (d.merged_with || []).length ?
          (d.merged_with || []).map(t => '“' + esc(t) + '”').join(', ') : 'no other Terra word') +
      row('our best box', m.best_box ? esc(m.best_box) + ' · IoU ' + f2(m.best_iou) : 'none overlaps') +
      row('reader at Terra\\'s box', r ? '“' + esc(r.text) + '” · ' + esc(r.top) + ' ' + f2(r.p_top) +
          ' · code ' + esc(r.code) + ' ' + f2(r.p_code) + ' · p(180°) ' + f2(r.p_180) : '–') +
      row('quarter', m.quarter) + '</table>';
  }
  function tap(x, y) {
    const tol = 4 / s;
    let hits = D.missed.filter(m => near(m.poly, x, y, 0)).map(missHtml)
      .concat(D.boxes.filter(b => near(b.poly, x, y, 0)).map(boxHtml));
    if (!hits.length) hits = D.missed.filter(m => near(m.poly, x, y, tol)).map(missHtml)
      .concat(D.boxes.filter(b => near(b.poly, x, y, tol)).map(boxHtml));
    if (!hits.length) { info.classList.remove('on'); return; }
    info.innerHTML = '<button class="x" aria-label="Close">×</button><div class="pt">page px ' + Math.round(x) + ', ' +
      Math.round(y) + '</div>' + hits.join('');
    info.querySelector('.x').onclick = () => info.classList.remove('on');
    info.classList.add('on');
  }
  function fromHash() {
    const m = /view=([\\d.]+),([\\d.]+),([\\d.]+)/.exec(location.hash);
    if (!m) return false;
    const r = vp.getBoundingClientRect();
    s = +m[3]; tx = r.width / 2 - +m[1] * s; ty = r.height / 2 - +m[2] * s; apply();
    const t = /tap=([\\d.]+),([\\d.]+)/.exec(location.hash);
    if (t) tap(+t[1], +t[2]);
    return true;
  }
  if (!fromHash()) fit();
  window.addEventListener('resize', () => { if (s <= minS * 2.01) fit(); });
})();
"""


def render(d: dict, out: Path) -> Path:
    """Write the page of `page_data`'s `d` to `out` (one self-contained HTML file, the content of a page without
    `<html>`, `<head>` or `<body>`: an Artifact's page)."""
    h, w = d["shape"]
    tiles = []
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            part = d["pixels"][y0:y0 + TILE, x0:x0 + TILE]
            tiles.append(f'<img class="tile" alt="" data-x="{x0}" data-y="{y0}" width="{part.shape[1]}" '
                         f'height="{part.shape[0]}" style="left:{x0}px;top:{y0}px" '
                         f'src="data:image/png;base64,{_png_b64(part)}">')
    js_data = {"shape": d["shape"], "boxes": d["boxes"], "missed": d["missed"], "reasons": REASON_TEXT}
    payload = json.dumps(js_data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    models = "".join(f"<div>{_e(m)}</div>" for m in d["models"])
    run = f' · run <b>{_e(d["run"])}</b>' if d.get("run") else ""
    page_html = (
        '<title>Mission Sheet Review</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
        f"<style>{STYLE}</style>\n"
        '<div class="wrap">'
        f'<h1>The Mission Street sheet: our text reader against Terra</h1>'
        f'<p class="sub">Sheet <b>{_e(d["sheet"])}</b>{run} · {w:,} × {h:,} px at full resolution · '
        f'boxes matched to Terra\'s words at IoU ≥ {d["iou"]} · {len(d["boxes"]):,} boxes, '
        f'{len(d["missed"]):,} missed words drawn. The image is the page\'s ink binarised (px darker than '
        f'{grid.INK_BELOW}, the detector\'s own input) and drawn grey or red, not the scan\'s grey levels.</p>'
        f"{_panels(d)}"
        '<div class="viewer"><div class="bar"><button id="zin" aria-label="Zoom in">+</button>'
        '<button id="zout" aria-label="Zoom out">−</button><button id="zfit">Fit</button><span id="zl" '
        'class="hint"></span><span class="hint">Drag or pinch to move and zoom; the readings appear from '
        f'{LABELS_FROM:g}x, the tags and the reasons from {TAGS_FROM:g}x. Tap a box for everything about it.</span>'
        '</div>'
        f'<div id="vp"><div id="stage" style="width:{w}px;height:{h}px">{"".join(tiles)}{_overlay(d)}</div>'
        '<div id="info"></div></div></div>'
        f'<div class="models"><b>Drawn by</b>{models}</div></div>\n'
        f'<script type="application/json" id="data">{payload}</script>\n'
        f"<script>{SCRIPT % {'labels': LABELS_FROM, 'tags': TAGS_FROM, 'maxs': MAX_SCALE}}</script>\n")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page_html, encoding="utf-8")
    return out


# --- reading the page back (the tests' check that no shading covers the red) -----------------------------------------

_TILE_RE = re.compile(r'<img class="tile"[^>]*data-x="(\d+)" data-y="(\d+)"[^>]*src="data:image/png;base64,([^"]+)"')
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


def page_image(path: Path) -> np.ndarray:
    """The page's image as drawn (uint8 (h, w, 3) RGB): its tiles, and over them every filled polygon, rect or path of
    the overlay (`fill` not none), at its fill and opacity."""
    text = Path(path).read_text(encoding="utf-8")
    tiles = [(int(x), int(y), np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")))
             for x, y, b in _TILE_RE.findall(text)]
    h = max(y + t.shape[0] for _, y, t in tiles)
    w = max(x + t.shape[1] for x, _, t in tiles)
    img = np.zeros((h, w, 3), np.uint8)
    for x, y, t in tiles:
        img[y:y + t.shape[0], x:x + t.shape[1]] = t
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
