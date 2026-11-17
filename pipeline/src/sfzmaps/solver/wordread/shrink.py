"""The word heads' cores (`prepare`): each Terra word's OWN labelled rectangle (`data.SheetData.words`), as
`data.crop_targets(..., words=True)` paints them for `train.Recipe.heads="words"`.  No shrink ratio (the owner,
2026-11-17: no picked core size): words that touch are told apart by the learned tags, the geometry votes (Task 6)
and the learned box model (Task 15), not by cores cut smaller by a chosen number.

Kept only as a record of the overlap measurement made before that ruling (commits 87f3d82, da3a149; its numbers in
SHRINK_FILE, experiments/wordread/detector/shrink.json): `core`, `unclip`, `cores_px`, `measure_r`, `choose`, `pool`
and `write_r`.  NOTHING USES r: no target, training or reading reads it.  What they computed:

A rectangle (4 corners (x, y), in order around it, the first edge along the reading direction, as `boxes.word_rect`
makes them) of length w along its first edge and height h is shrunk by D = A (1 - r^2) / L on every side, A = w h its
area and L = 2 (w + h) its perimeter (DBNet, Liao et al. 2020; the closed form of Vatti clipping for a rectangle): the
core is (w - 2D) x (h - 2D), with the same centre, the same axes and its corners in the same order, so its first edge
still runs along the reading direction.  As h - 2D = h (h + w r^2) / (w + h), a core never vanishes.  `unclip` grows
a core back by the D whose shrink gives it (the positive root of 4 (1 + r^2) D^2 + 2 r^2 (w' + h') D - (1 - r^2) w' h'
= 0, w' x h' the core): `unclip(core(rect, r), r)` is `rect`.

r (`measure_r`, `choose`; record only): the largest of R_GRID (0.3 to 0.7 by 0.05) at which at most MAX_SHARE (1%) of
the neighbouring pairs of words have cores that overlap NEWLY: at that r but not at the grid's first, 0.3.  The pairs
whose cores already overlap at 0.3 (the floor) overlap at every r: they are Terra's box errors (a word read twice
across a tile seam, a box grown over the next word), which no shrink separates and the doubt flags are for; the rule
is that shrinking must not merge words Terra keeps apart.  Two words of a sheet are neighbours when the gap between
their rectangles (0 when they touch or overlap) is at most one word height: the smaller of the two words' heights (the
second edge's length).  Two cores overlap when a px is filled by both, each painted as the detector learns it (its
corners rounded to px, filled as `cv2.fillPoly` fills them, `data._paint`); `touching` also counts a px of one next to
(the 8 around) a px of the other, the cores the connected px would join.  A core lies inside its word, so only words
whose rectangles overlap, or come within TOUCH_GAP px of each other, can have overlapping or touching cores.

`write_r` measures the curve on the training maps' sheets (`paths.TEXT_TRAIN_SERIES`) once per fold of the reader's
split (`paths.QUARTERS`), the words whose centre px is in the fold's quarter or an answered block
(`reader_stage.held_blocks`) left out, as `reader_stage.measure_radius` leaves them out; the folds' counts are pooled
(`pool`: their pairs and new overlaps summed) into one r; written to SHRINK_FILE (`python -m
sfzmaps.solver.wordread.shrink`).  Measured: r 0.40, the floor 12.6% of the pairs.

Neither `textnet.data` nor `textnet.train` imports this module (their stages' import closure is unchanged): the
cores reach the detector in `data.SheetData.cores`, filled by `prepare` before the training.
"""
import dataclasses
import json
import logging
from collections.abc import Mapping

import cv2
import numpy as np

from sfzmaps.solver import paths
from sfzmaps.solver.textnet import data

log = logging.getLogger(__name__)

R_GRID = tuple(round(0.3 + 0.05 * i, 2) for i in range(9))    # 0.3, 0.35, ..., 0.7
MAX_SHARE = 0.01           # of the neighbouring pairs, at most this share may have overlapping cores
TOUCH_GAP = 4.0            # px: words this far apart or more can't have cores sharing or touching a px
PAIR_CHUNK = 256           # words compared against all the others at a time when finding neighbours
SHRINK_FILE = paths.SOLVER_DIR / "experiments" / "wordread" / "detector" / "shrink.json"
_EIGHT = np.ones((3, 3), np.uint8)


# --- shrinking and growing -------------------------------------------------------------------------------------------

def _frame(rect) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A rectangle's (or many's, (..., 4, 2)) centre, axes and size: the unit vector along its first edge (u1), the
    one across it towards its last corner (u2), its extents along each (w, h), and its centre (the middle of those
    extents).  A first edge of no length runs left to right."""
    rect = np.asarray(rect, float)
    e1 = rect[..., 1, :] - rect[..., 0, :]
    n1 = np.linalg.norm(e1, axis=-1, keepdims=True)
    u1 = np.where(n1 > 0, e1 / np.where(n1 > 0, n1, 1), [1.0, 0.0])
    side = rect[..., 3, :] - rect[..., 0, :]
    across = side - (side * u1).sum(-1, keepdims=True) * u1
    n2 = np.linalg.norm(across, axis=-1, keepdims=True)
    perp = np.stack([-u1[..., 1], u1[..., 0]], -1)
    u2 = np.where(n2 > 1e-9, across / np.where(n2 > 1e-9, n2, 1), perp)
    a = (rect * u1[..., None, :]).sum(-1)               # each corner along u1, (..., 4)
    b = (rect * u2[..., None, :]).sum(-1)
    w, h = a.max(-1) - a.min(-1), b.max(-1) - b.min(-1)
    centre = (u1 * ((a.max(-1) + a.min(-1)) / 2)[..., None] + u2 * ((b.max(-1) + b.min(-1)) / 2)[..., None])
    return centre, u1, u2, w, h


_SIGNS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)    # each corner's side of the centre, along u1 and u2


def _rebuild(centre, u1, u2, w, h) -> np.ndarray:
    """The rectangle (..., 4, 2) of that centre, axes and size, its corners in `_frame`'s order."""
    half = np.stack([w / 2, h / 2], -1)[..., None, :] * _SIGNS          # (..., 4, 2): along u1, along u2
    return centre[..., None, :] + half[..., :1] * u1[..., None, :] + half[..., 1:] * u2[..., None, :]


def offset(w, h, r: float) -> np.ndarray:
    """DBNet's shrink offset D = A (1 - r^2) / L of a w x h rectangle (0 for one of no perimeter)."""
    w, h = np.asarray(w, float), np.asarray(h, float)
    perimeter = 2 * (w + h)
    return np.where(perimeter > 0, w * h * (1 - r * r) / np.where(perimeter > 0, perimeter, 1), 0.0)


def core(rect, r: float) -> np.ndarray:
    """The core of a word's rectangle (4 corners (x, y), or many, (..., 4, 2)): shrunk by `offset` on every side,
    float corners in the same order (see the module)."""
    centre, u1, u2, w, h = _frame(rect)
    d = offset(w, h, r)
    return _rebuild(centre, u1, u2, np.maximum(w - 2 * d, 0), np.maximum(h - 2 * d, 0))


def unclip(core_rect, r: float) -> np.ndarray:
    """The rectangle whose `core` at `r` is `core_rect` (4 corners, or (..., 4, 2)): grown by the D that shrink took
    (see the module), float corners in the same order."""
    centre, u1, u2, w, h = _frame(core_rect)
    r2 = r * r
    s, p = w + h, w * h
    d = (-2 * r2 * s + np.sqrt(4 * r2 * r2 * s * s + 16 * (1 + r2) * (1 - r2) * p)) / (8 * (1 + r2))
    return _rebuild(centre, u1, u2, w + 2 * d, h + 2 * d)


def cores_px(words: np.ndarray, r: float) -> np.ndarray:
    """The cores of words (int32 (n, 4, 2), `data.SheetData.words`) as the detector learns them: corners rounded to
    page px, int32 (n, 4, 2)."""
    words = np.asarray(words).reshape(-1, 4, 2)
    return np.round(core(words, r)).reshape(-1, 4, 2).astype(np.int32)


def prepare(sheets: list[data.SheetData]) -> list[data.SheetData]:
    """The sheets with their words' cores (`data.SheetData.cores`) for the word heads' targets
    (`data.crop_targets(..., words=True)`): each word's own labelled rectangle, `words`, unshrunk (see the module);
    everything else shared with `sheets`.  Run it before the training."""
    return [dataclasses.replace(sd, cores=sd.words) for sd in sheets]


# --- which r ---------------------------------------------------------------------------------------------------------

def heights(words: np.ndarray) -> np.ndarray:
    """Each word's height: its rectangle's second edge's length."""
    w = np.asarray(words, float).reshape(-1, 4, 2)
    return np.linalg.norm(w[:, 2] - w[:, 1], axis=-1)


def _point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The distance from points p to segments a-b, (..., 2) each."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def gaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The gap between convex quadrilaterals a[k] and b[k] ((m, 4, 2) each): 0 when they touch or overlap (no axis
    of their 8 edges separates them), else the shortest distance from a corner of one to an edge of the other."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    edges = np.concatenate([np.roll(a, -1, 1) - a, np.roll(b, -1, 1) - b], 1)        # (m, 8, 2)
    normals = np.stack([-edges[..., 1], edges[..., 0]], -1)
    pa = np.einsum("mkc,mjc->mkj", normals, a)                                        # (m, 8 axes, 4 corners)
    pb = np.einsum("mkc,mjc->mkj", normals, b)
    apart = (pa.max(-1) < pb.min(-1)) | (pb.max(-1) < pa.min(-1))
    meet = ~apart.any(-1)
    ends = [(x, np.roll(x, -1, 1)) for x in (a, b)]
    d = np.full(len(a), np.inf)
    for pts, (s0, s1) in ((a, ends[1]), (b, ends[0])):
        dist = _point_segment(pts[:, :, None, :], s0[:, None, :, :], s1[:, None, :, :])   # (m, 4, 4)
        d = np.minimum(d, dist.reshape(len(a), -1).min(-1))
    return np.where(meet, 0.0, d)


def neighbour_pairs(words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A sheet's neighbouring pairs of words (see the module): (i, j) with i < j, int (m, 2), and the gap between
    each pair's rectangles."""
    w = np.asarray(words, float).reshape(-1, 4, 2)
    n = len(w)
    hgt = heights(w)
    lo, hi = w.min(1), w.max(1)
    found, found_gaps = [], []
    for i0 in range(0, n, PAIR_CHUNK):
        i = np.arange(i0, min(i0 + PAIR_CHUNK, n))
        # the axis boxes' gap is at most the rectangles': a cheap first cut
        gx = np.maximum(np.maximum(lo[None, :, 0] - hi[i, None, 0], lo[i, None, 0] - hi[None, :, 0]), 0)
        gy = np.maximum(np.maximum(lo[None, :, 1] - hi[i, None, 1], lo[i, None, 1] - hi[None, :, 1]), 0)
        reach = np.minimum(hgt[i, None], hgt[None, :])
        ii, jj = np.nonzero((np.hypot(gx, gy) <= reach) & (np.arange(n)[None, :] > i[:, None]))
        ii = i[ii]
        if not len(ii):
            continue
        g = gaps(w[ii], w[jj])
        keep = g <= np.minimum(hgt[ii], hgt[jj])
        found.append(np.c_[ii[keep], jj[keep]])
        found_gaps.append(g[keep])
    if not found:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    return np.concatenate(found).astype(np.int64), np.concatenate(found_gaps)


def _meet(c1: np.ndarray, c2: np.ndarray) -> tuple[bool, bool]:
    """Whether two cores (int32 (4, 2) page px) share a px when painted, and whether a px of one is on or next to
    (the 8 around) a px of the other."""
    lo = np.minimum(c1.min(0), c2.min(0)) - 1
    hi = np.maximum(c1.max(0), c2.max(0)) + 2
    a = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
    b = np.zeros_like(a)
    cv2.fillPoly(a, [(c1 - lo).reshape(-1, 1, 2).astype(np.int32)], 1)
    cv2.fillPoly(b, [(c2 - lo).reshape(-1, 1, 2).astype(np.int32)], 1)
    return bool((a & b).any()), bool((cv2.dilate(a, _EIGHT) & b).any())


def measure_r(words_by_sheet) -> dict:
    """r (see the module) from the words of each sheet (`words_by_sheet`: {sheet: int32 (n, 4, 2) rectangles}, or
    a list of them): `r` the largest of R_GRID whose new share is at most MAX_SHARE (`choose`); `curve` per r of
    R_GRID: the pairs whose cores overlap (`overlapping`, `share`), those of them that do not at the grid's first r
    (`new`, `new_share`), and those whose cores touch or overlap (`touching`, `touching_share`); `floor` the pairs
    overlapping at the first r and `floor_share`; `pairs` the neighbouring pairs, `words` the words and `close_pairs`
    those within TOUCH_GAP px."""
    sheets = list(words_by_sheet.values()) if isinstance(words_by_sheet, Mapping) else list(words_by_sheet)
    total_words, close = 0, []          # (sheet's words, i, j) of the neighbours close enough to share a px
    pairs = 0
    for words in sheets:
        words = np.asarray(words).reshape(-1, 4, 2)
        total_words += len(words)
        ij, g = neighbour_pairs(words)
        pairs += len(ij)
        close += [(words, int(i), int(j)) for (i, j), gap in zip(ij, g) if gap < TOUCH_GAP]
    met = []                            # per r of R_GRID: each close pair's (overlap, touch)
    for r in R_GRID:
        cores = {}
        found = []
        for words, i, j in close:
            key = id(words)
            if key not in cores:
                cores[key] = cores_px(words, r)
            found.append(_meet(cores[key][i], cores[key][j]))
        met.append(np.array(found, bool).reshape(-1, 2))
    floor = met[0][:, 0]
    curve = [{"r": r, "overlapping": int(m[:, 0].sum()), "new": int((m[:, 0] & ~floor).sum()),
              "touching": int(m[:, 1].sum())} for r, m in zip(R_GRID, met)]
    return {**_shares({"pairs": pairs, "words": total_words, "close_pairs": len(close), "floor": int(floor.sum()),
                       "curve": curve}), "max_share": MAX_SHARE, "grid": list(R_GRID), "rule": RULE}


RULE = ("the largest r of the grid at which at most max_share of the neighbouring pairs (rectangles at most the "
        "smaller word height apart) have cores sharing a painted px that do not at the grid's first r (the floor: "
        "Terra's box errors, overlapping at every r); cores shrunk by D = A (1 - r^2) / L, corners rounded to px")


def _shares(m: dict) -> dict:
    """A measurement's counts with their shares of its pairs, and its r (`choose`)."""
    n = m["pairs"]

    def share(k):
        return k / n if n else 0.0
    curve = [{**c, "share": share(c["overlapping"]), "new_share": share(c["new"]),
              "touching_share": share(c["touching"])} for c in m["curve"]]
    return {**m, "r": choose(curve), "floor_share": share(m["floor"]), "curve": curve}


def choose(curve: list[dict]) -> float | None:
    """The largest r of a curve whose new share is at most MAX_SHARE (None when none is)."""
    ok = [c["r"] for c in curve if c["new_share"] <= MAX_SHARE]
    return max(ok) if ok else None


def pool(measured) -> dict:
    """Measurements (`measure_r`, e.g. one per fold) as one: their pairs, words, close pairs, floors and each r's
    counts summed, the shares and r (`choose`) taken from the sums."""
    measured = list(measured)
    grids = {tuple(m["grid"]) for m in measured}
    if len(grids) != 1:
        raise ValueError(f"the measurements were taken on {len(grids)} grids of r; pooling needs one")
    total = {k: sum(m[k] for m in measured) for k in ("pairs", "words", "close_pairs", "floor")}
    curve = [{"r": c["r"], **{k: sum(m["curve"][i][k] for m in measured) for k in ("overlapping", "new", "touching")}}
             for i, c in enumerate(measured[0]["curve"])]
    return {**_shares({**total, "curve": curve}), "max_share": MAX_SHARE, "grid": list(grids.pop()), "rule": RULE,
            "pooled": len(measured)}


# --- measured on the training maps -----------------------------------------------------------------------------------

def _kept_words(sd: data.SheetData, areas) -> np.ndarray:
    """A sheet's words whose centre px (the corners' mean, rounded, kept on the page) is in none of `areas`."""
    keep = []
    for j, rect in enumerate(sd.words.astype(float)):
        x, y = np.clip(np.rint(rect.mean(axis=0)).astype(int), 0, [sd.width - 1, sd.height - 1])
        if not any(data.in_area(sd, a, int(y), int(x), 1, 1)[0, 0] for a in areas):
            keep.append(j)
    return sd.words[np.array(keep, int)].reshape(-1, 4, 2)


def write_r(out=SHRINK_FILE) -> dict:
    """Measure the curve on the training maps per fold, pool the folds into r (see the module), and write both to
    `out` (its folder made)."""
    from sfzmaps.solver.textnet import reader_stage    # torch and the lots: only for this measurement

    blocks = reader_stage.held_blocks()
    area = reader_stage.load_block_area(blocks)
    sheets = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False)
    folds = {}
    for q, name in enumerate(paths.QUARTERS):
        areas = (data.Quarter(q), area)
        words = {f"{sd.series}/{sd.sheet}": _kept_words(sd, areas) for sd in sheets}
        folds[name] = measure_r(words)
        log.info("fold %s held out: r %s over %d pairs of %d words (floor %.4f)", name, folds[name]["r"],
                 folds[name]["pairs"], folds[name]["words"], folds[name]["floor_share"])
    pooled = pool(folds.values())
    rs = {name: f["r"] for name, f in folds.items()}
    got = {"r": pooled["r"], "pooled": pooled, "r_by_fold": rs, "same_every_fold": len(set(rs.values())) == 1,
           "series": list(paths.TEXT_TRAIN_SERIES), "sheets": len(sheets), "blocks_held": len(blocks),
           "left_out": "words whose centre px is in the fold's quarter or an answered block", "folds": folds}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(got, indent=1) + "\n")
    return got


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data.LOAD_THREADS = 1          # one page decoded at a time: the machine is shared
    result = write_r()
    print(json.dumps({k: v for k, v in result.items() if k not in ("folds", "pooled")}, indent=1))
    p = result["pooled"]
    print(f"pooled: {p['pairs']} pairs, floor {p['floor']} ({p['floor_share']:.4f}); new by r:",
          ", ".join(f"{c['r']}: {c['new']} ({c['new_share']:.4f})" for c in p["curve"]))
