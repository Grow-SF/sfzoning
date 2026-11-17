"""Word boxes from the detector's word heads (`textnet.train.Recipe.heads="words"`, `textnet.model` CORE, GEO, TAG):
a word is a component of the word-core chance, split where its px tags say two words, grown back to the word by
DBNet's unclip or rebuilt from its px's geometry votes (EAST's locality-aware NMS).  Not by connected ink: two words
whose ink touches keep separate cores, and a line touching a word is not in its core.

`core_words(core, geo, tag, ink, level, r, refine, ...)` works on one page's arrays (the net's outputs over the page
are `Task 8`'s: `textnet.predict` is not used here):
  core   float (h, w): the chance a px is in a word's core (the CORE logit's sigmoid), in [0, 1]
  geo    float (5, h, w) or (6, h, w): the distances in px from a px to its word's top, bottom, start and end edges
         (`textnet.model.distances` of GEO), then its reading direction: the angle in degrees counter-clockwise from
         left-to-right (as `data.crop_targets(..., words=True)`'s geo), or its cosine and sine (the net's channels 5-6,
         any length)
  tag    float (k, h, w): each px's word tag (TAG)
  ink    bool (h, w): the page's ink

1. The words: the 8-connected components of the px with core >= `level`.
2. The split (PAN's tags, as a likelihood): a word's px tags are Gaussian about the word's mean tag, the same spread
   `tag_sigma` on every channel.  A component is split when a model of two words (a two-part mixture of such
   Gaussians, fitted by EM: `two_word_llr`) is likelier than one word, the second word's extra parameters charged as
   `split_prior`, the log prior odds of two words against one: split when log L2 - log L1 + split_prior > 0.  Each
   px goes to the word it is likelier in; each word's px fall into 8-connected pieces, and every piece is a component
   tested again (a chain of three words splits twice).  Then the pieces are decided by the same model: two touching
   pieces are joined while one word is at least as likely as two (the two means against one, `split_prior` charged
   for the second), the likeliest join first.  So px the mixture put on the wrong side, a few at a time (each piece
   of them a would-be word), join the word around them, and two words stay apart only on a word's worth of evidence.
   `tag_sigma` and `split_prior` are measured on the training quarters' predictions (Task 8); nothing here fixes a
   distance between tags.
3. The box, by `refine`:
   - "unclip" (DBNet): the minimum-area rectangle of the word's core px, grown by the D whose shrink at `r` gives it
     (`shrink.unclip`).  Its first edge is the rectangle's side nearest the word's reading direction (the
     core-chance-weighted sum of its px's directions).
   - "votes" (EAST): every core px votes for its word's rectangle (its distances to the four edges along its own
     reading direction: `data._geometry` inverted).  The votes are laid with their first edge nearest the word's
     reading direction, then merged by score-weighted locality-aware NMS, the score each px's core chance: in px
     order (row by row) a vote merges into the running box when their IoU exceeds `merge_iou` (the corners averaged,
     weighted by score, the scores summed), else the running box is set aside and the vote starts the next; the boxes
     set aside are then merged, in falling score, into the first kept box they meet at above `merge_iou`.  The word's
     box is the one with the most score (a minority of px voting for something else, a core run on along a line, is
     outvoted, never averaged in).  The merged corners are squared up to a rectangle (its centre, the mean direction
     of its two long edges, their mean lengths).  `merge_iou` is measured in Task 8.
   Either box is clipped to the page: a box running off the page keeps its axes and becomes the extent along them of
   its part on the page (never wrapped, never dropped); a box with no area keeps its corners moved onto the page.

Coordinates.  The cores were painted from rectangles with corners at px numbers (`data._paint`, as `cv2.fillPoly`
fills them) and the distances measured from a px's numbers (col, row) (`data._geometry`), both in the frame of
Terra's rectangles; so here a core px stands at (col, row), and the boxes come back in that frame: page px edge
coordinates (a px is [i, i + 1)), as `wordboxes`' boxes and Terra's words are compared.  The ink px (edge coordinates,
its centre at (col + 0.5, row + 0.5), a px wide) give the trimmed box.

The table (`wordboxes.COLUMNS`, the same meaning, so the reader reads these boxes unchanged; then EXTRA_COLUMNS):
  g corners       the box (4 corners, the first edge along the reading direction, the second towards the letters'
                  feet, as `boxes.word_rect` lays a word)
  t corners       the trimmed box: the box's axes, the extent of the ink px inside the box (each a px wide), within
                  the box; the box itself when it holds no ink
  axis            the first edge's direction, degrees counter-clockwise from left-to-right, modulo 180
  length, height  the box's sides along and across it; t_length, t_height the trimmed box's
  page_x, page_y  the box's centre
  ink_px          the ink px inside the box; p_<kind> their mean kind probabilities (`kind_p`, else 0)
  mean_chance     the word's chance: the mean core chance over its core px (as p_word)
  sheet, i, level the sheet given, the box's number on the page (the words in the order of their first px, row by
                  row), the level
  quarter, x, y, lab_lot, street   not placed yet: -1, NaN, NaN, -1, False.  Place the boxes as
                  `wordboxes.sheet_boxes` does (`prints._place` of page_x, page_y; `grid.quarter_of`) before reading.
  form            "unclip" or "votes"; "+split" when the word was cut from a larger component by its tags
  core_px         the word's core px
  tag_spread      the spread of the word's own px tags about their mean (the root mean square over px and channels:
                  the one-word model's fitted sigma), what `tag_sigma` is measured from
  p_word          the word's chance (the mean core chance over its core px): later steps weigh a box by it rather
                  than drop it
  split_llr       of a word cut from a larger component, the log-likelihood ratio of it standing apart from its
                  likeliest touching word against the two joined (their two mean tags against one, under
                  `tag_sigma`); above -split_prior by the rule; NaN for a whole component

`choose_level(boxes, reads, terra, forms)`: the level of each quarter of the city, chosen on the other three quarters
by Task 1's headline (`measures.label_right`: read right at our boxes, the box holding the point, the reader's surest
of several; no box is wrong) over Terra's code words (Terra's string as the truth, its rectangle's centre as the
point), never the owner's answers.  Ties go to the higher level.  The levels are the caller's: none is fixed here.
"""
import math

import cv2
import numpy as np
import pandas as pd

from sfzmaps.solver import paths, wordboxes
from sfzmaps.solver.wordread import measures, shrink

REFINES = ("unclip", "votes")
EXTRA_COLUMNS = ("form", "core_px", "tag_spread", "p_word", "split_llr")
EM_STEPS = 100        # numerical only: the two-word fit stops after this many EM steps ...
EM_TOL = 1e-9         # ... or once its log likelihood rises by less than this per px


# --- the tag split -------------------------------------------------------------------------------------------------

def two_word_llr(tags: np.ndarray, sigma: float) -> tuple[float, np.ndarray]:
    """log L(two words) - log L(one word) of px tags (n, k), each word's tags Gaussian about its mean with sd `sigma`
    on every channel; the two-word model a two-part mixture (its weight and both means fitted by EM, started at the
    tag farthest from the mean and the tag farthest from that).  Also each px's chance of being in the first word.
    0 and ones for fewer than 2 px."""
    t = np.asarray(tags, np.float64).reshape(len(tags), -1)
    n = len(t)
    if n < 2:
        return 0.0, np.ones(n)
    inv = 1.0 / (2.0 * sigma * sigma)
    mean = t.mean(0)
    ss1 = float(((t - mean) ** 2).sum())
    m1 = t[int(np.argmax(((t - mean) ** 2).sum(1)))]
    m2 = t[int(np.argmax(((t - m1) ** 2).sum(1)))]
    w = 0.5
    prev = -math.inf
    for _ in range(EM_STEPS):
        la = (math.log(w) if w > 0 else -math.inf) - ((t - m1) ** 2).sum(1) * inv
        lb = (math.log(1 - w) if w < 1 else -math.inf) - ((t - m2) ** 2).sum(1) * inv
        ll = float(np.logaddexp(la, lb).sum())
        with np.errstate(invalid="ignore"):
            resp = 0.5 * (1.0 + np.tanh(np.nan_to_num(la - lb, nan=0.0) / 2))     # sigmoid(la - lb), stable
        if ll - prev <= EM_TOL * n:
            break
        prev = ll
        n1 = float(resp.sum())
        if n1 <= 0 or n1 >= n:
            break                            # one part took every px: one word
        w = n1 / n
        m1 = (resp[:, None] * t).sum(0) / n1
        m2 = ((1 - resp)[:, None] * t).sum(0) / (n - n1)
    return ll + ss1 * inv, resp


def _pieces(idx: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> list[np.ndarray]:
    """The 8-connected pieces of the px idx (ascending indices into ys, xs), each ascending, in the order of their
    first px."""
    y, x = ys[idx], xs[idx]
    y0, x0 = int(y.min()), int(x.min())
    canvas = np.zeros((int(y.max()) - y0 + 1, int(x.max()) - x0 + 1), np.uint8)
    canvas[y - y0, x - x0] = 1
    count, lab = cv2.connectedComponents(canvas, connectivity=8)
    if count == 2:
        return [idx]
    own = lab[y - y0, x - x0]
    order = np.argsort(own, kind="stable")
    parts = np.split(idx[order], np.flatnonzero(np.diff(own[order])) + 1)
    return sorted(parts, key=lambda p: int(p[0]))


_SHIFTS = ((0, 1), (1, -1), (1, 0), (1, 1))     # each 8-neighbour pair of px once


def _touching(pieces: list[np.ndarray], ys, xs) -> set[tuple[int, int]]:
    """The pairs (a < b) of pieces (disjoint px index arrays) with a px of one among the 8 around a px of the other."""
    every = np.concatenate(pieces)
    y0, x0 = int(ys[every].min()), int(xs[every].min())
    canvas = np.full((int(ys[every].max()) - y0 + 1, int(xs[every].max()) - x0 + 1), -1, np.int64)
    for k, p in enumerate(pieces):
        canvas[ys[p] - y0, xs[p] - x0] = k
    h, w = canvas.shape
    pairs = set()
    for dy, dx in _SHIFTS:
        a = canvas[0:h - dy, max(0, -dx):w - max(0, dx)]
        b = canvas[dy:h, max(0, dx):w - max(0, -dx)]
        hit = (a >= 0) & (b >= 0) & (a != b)
        pairs |= {(int(min(i, j)), int(max(i, j))) for i, j in zip(a[hit], b[hit])}
    return pairs


def _apart(n_a: int, sum_a: np.ndarray, n_b: int, sum_b: np.ndarray, inv: float) -> float:
    """log L(two words) - log L(one word) of two px sets' tags under the one spread (the two means against one: Ward's
    cost n_a n_b / (n_a + n_b) |mean_a - mean_b|^2 / (2 sigma^2))."""
    d = sum_a / n_a - sum_b / n_b
    return n_a * n_b / (n_a + n_b) * float(d @ d) * inv


def _split(idx: np.ndarray, ys, xs, tags: np.ndarray, sigma: float, prior: float) -> list[tuple]:
    """A component's words (see the module): (px indices, the log-likelihood ratio of the word standing apart from
    its likeliest touching word, NaN for a whole component).

    Proposed by splitting: wherever the two-word mixture is likelier (`two_word_llr` + prior > 0), each px goes to
    the part it is likelier in and each part's 8-connected pieces are tested again.  Then decided on the pieces: two
    touching pieces are joined while one word is at least as likely as two (`_apart` + prior <= 0), the likeliest
    join first.  So a few px the mixture put on the wrong side (they make pieces of their own) join the word around
    them unless they are a word's worth of evidence apart, and nothing but `sigma` and `prior` decides."""
    pieces, stack = [], [idx]
    while stack:
        cur = stack.pop()
        llr, first = two_word_llr(tags[cur], sigma)
        first = first > 0.5
        if llr + prior > 0 and first.any() and not first.all():
            for part in (cur[first], cur[~first]):
                stack.extend(reversed(_pieces(part, ys, xs)))
        else:
            pieces.append(cur)
    if len(pieces) == 1:
        return [(idx, math.nan)]
    inv = 1.0 / (2.0 * sigma * sigma)
    pieces.sort(key=lambda p: int(p[0]))
    members = {k: [p] for k, p in enumerate(pieces)}
    count = {k: len(p) for k, p in enumerate(pieces)}
    total = {k: tags[p].sum(0) for k, p in enumerate(pieces)}
    touch = _touching(pieces, ys, xs)
    while touch:
        cost, (a, b) = min((_apart(count[a], total[a], count[b], total[b], inv), (a, b)) for a, b in sorted(touch))
        if cost + prior > 0:
            break                               # every touching pair is likelier as two words
        members[a] += members.pop(b)
        count[a] += count.pop(b)
        total[a] = total[a] + total.pop(b)
        renamed = ((a if i == b else i, a if j == b else j) for i, j in touch)       # b is now part of a
        touch = {(min(i, j), max(i, j)) for i, j in renamed if i != j}
    if len(members) == 1:
        return [(idx, math.nan)]
    apart = {k: math.inf for k in members}      # each word against its likeliest touching word
    for i, j in touch:
        cost = _apart(count[i], total[i], count[j], total[j], inv)
        apart[i], apart[j] = min(apart[i], cost), min(apart[j], cost)
    return [(np.sort(np.concatenate(parts)), apart[k]) for k, parts in members.items()]


# --- rectangles ----------------------------------------------------------------------------------------------------

def _rect(c, v, n, a0, a1, b0, b1) -> np.ndarray:
    """The rectangle of centre c, axes v (along) and n (across, towards the feet), extents [a0, a1] x [b0, b1]: its
    corners in `boxes.word_rect`'s order."""
    return np.array([c + v * a0 + n * b0, c + v * a1 + n * b0, c + v * a1 + n * b1, c + v * a0 + n * b1])


def _frame(rect: np.ndarray):
    """A rectangle's centre, axes (its first edge's direction, and that turned towards the feet) and its corners'
    extents along them: c, v, n, (a0, a1, b0, b1)."""
    e = rect[1] - rect[0]
    norm = math.hypot(*e)
    v = e / norm if norm > 0 else np.array([1.0, 0.0])
    n = np.array([-v[1], v[0]])
    c = rect.mean(0)
    a, b = (rect - c) @ v, (rect - c) @ n
    return c, v, n, (a.min(), a.max(), b.min(), b.max())


def _oriented(c, sides: list[tuple[np.ndarray, float]], ref: np.ndarray) -> np.ndarray:
    """A rectangle of centre c given as its two axes with their lengths ((unit vector, length), twice), laid with its
    first edge along the axis direction nearest `ref` (with no `ref`, the longer side, left to right or upwards)."""
    cands = []
    for k, (u, length) in enumerate(sides):
        other = sides[1 - k][1]
        for sgn in (1.0, -1.0):
            cands.append((sgn * u, length, other))
    if ref @ ref > 0:
        v, along, across = max(cands, key=lambda cd: float(cd[0] @ ref))
    else:
        v, along, across = max(cands, key=lambda cd: (cd[1], float(cd[0][0]), -float(cd[0][1])))
    n = np.array([-v[1], v[0]])
    return _rect(c, v, n, -along / 2, along / 2, -across / 2, across / 2)


def _unclip_box(px: np.ndarray, ref: np.ndarray, r: float) -> np.ndarray:
    """DBNet's box of a word's core px (m, 2) (x, y): their minimum-area rectangle, laid by the reading direction,
    unclipped at r."""
    p0, p1, p2, _ = cv2.boxPoints(cv2.minAreaRect(px.astype(np.float32))).astype(np.float64)
    sides = []
    for e in (p1 - p0, p2 - p1):
        length = math.hypot(*e)
        sides.append((e / length if length > 0 else None, length))
    if sides[0][0] is None and sides[1][0] is None:
        sides = [(np.array([1.0, 0.0]), 0.0), (np.array([0.0, 1.0]), 0.0)]
    elif sides[0][0] is None:
        u = sides[1][0]
        sides[0] = (np.array([-u[1], u[0]]), 0.0)
    elif sides[1][0] is None:
        u = sides[0][0]
        sides[1] = (np.array([-u[1], u[0]]), 0.0)
    return shrink.unclip(_oriented((p0 + p2) / 2, sides, ref), r)


def _areas(q: np.ndarray) -> np.ndarray:
    x, y = q[..., 0], q[..., 1]
    return 0.5 * np.abs((x * np.roll(y, -1, -1) - np.roll(x, -1, -1) * y).sum(-1))


def _iou(a32: np.ndarray, b32: np.ndarray, area_a: float, area_b: float) -> float:
    inter = cv2.intersectConvexConvex(a32, b32)[0]
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _lanms(quads: np.ndarray, score: np.ndarray, merge_iou: float) -> np.ndarray:
    """The votes (m, 4, 2), in px order, merged by score-weighted locality-aware NMS (see the module): the merged box
    with the most score (its corners, not yet squared up)."""
    q32 = quads.astype(np.float32)
    area = _areas(quads)
    groups = []                                     # [score-weighted corner sum, score]
    total, weight = quads[0] * score[0], float(score[0])
    cur32, cur_area = q32[0], float(area[0])
    for k in range(1, len(quads)):
        if _iou(cur32, q32[k], cur_area, float(area[k])) > merge_iou:
            total = total + quads[k] * score[k]
            weight += float(score[k])
            cur = total / weight
            cur32, cur_area = cur.astype(np.float32), float(_areas(cur))
        else:
            groups.append([total, weight])
            total, weight = quads[k] * score[k], float(score[k])
            cur32, cur_area = q32[k], float(area[k])
    groups.append([total, weight])
    kept = []                                       # [corner sum, score, box, box32, area]
    for total, weight in sorted(groups, key=lambda gr: -gr[1]):
        box = total / weight
        box32, box_area = box.astype(np.float32), float(_areas(box))
        for kk in kept:
            if _iou(kk[3], box32, kk[4], box_area) > merge_iou:
                kk[0] = kk[0] + total
                kk[1] += weight
                kk[2] = kk[0] / kk[1]
                kk[3], kk[4] = kk[2].astype(np.float32), float(_areas(kk[2]))
                break
        else:
            kept.append([total, weight, box, box32, box_area])
    return max(kept, key=lambda kk: kk[1])[2]      # the first of equal scores


def _squared(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """A merged quadrilateral of votes as a rectangle: its corners' centre, the mean direction of its first and third
    edges, their mean lengths, and the mean lengths of the other two; read the way of `ref`."""
    e = (q[1] - q[0]) + (q[2] - q[3])
    norm = math.hypot(*e)
    v = e / norm if norm > 0 else np.array([1.0, 0.0])
    if ref @ v < 0:
        v = -v
    n = np.array([-v[1], v[0]])
    along = abs(float(((q[1] - q[0]) @ v + (q[2] - q[3]) @ v) / 2))
    across = abs(float(((q[3] - q[0]) @ n + (q[2] - q[1]) @ n) / 2))
    return _rect(q.mean(0), v, n, -along / 2, along / 2, -across / 2, across / 2)


def _votes(px: np.ndarray, along: np.ndarray, dist: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Each core px's vote (m, 4, 2): its word's rectangle from its distances (top, bottom, start, end) along its own
    reading direction (`data._geometry` inverted), laid with the first edge nearest `ref`."""
    up = np.stack([along[:, 1], -along[:, 0]], -1)              # (-sin, -cos): towards the letters' tops
    top, bottom, start, end = dist.T
    a = np.stack([-start, end, end, -start], -1)[..., None]
    b = np.stack([top, top, -bottom, -bottom], -1)[..., None]
    q = px[:, None, :] + a * along[:, None, :] + b * up[:, None, :]
    if ref @ ref > 0:
        e = np.roll(q, -1, 1) - q
        norm = np.linalg.norm(e, axis=-1)
        dots = (e @ ref) / np.where(norm > 0, norm, np.inf)
        k = np.argmax(dots, 1)
        q = np.take_along_axis(q, ((np.arange(4)[None, :] + k[:, None]) % 4)[..., None], 1)
    return q


def _on_page(rect: np.ndarray, width: int, height: int) -> np.ndarray:
    """The box clipped to the page [0, width] x [0, height] (see the module)."""
    if (rect >= 0).all() and (rect[:, 0] <= width).all() and (rect[:, 1] <= height).all():
        return rect
    page = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    area, poly = cv2.intersectConvexConvex(rect.astype(np.float32), page)
    if poly is None or area <= 0:
        return np.clip(rect, 0, [width, height])
    c, v, n, _ = _frame(rect)
    pts = poly.reshape(-1, 2).astype(np.float64)
    a, b = (pts - c) @ v, (pts - c) @ n
    return _rect(c, v, n, a.min(), a.max(), b.min(), b.max())


def _inked(rect: np.ndarray, ink: np.ndarray, kind_p: np.ndarray | None):
    """The trimmed box of a box (see the module), its ink px and their mean kind probabilities (4,)."""
    h, w = ink.shape
    x0, y0 = np.maximum(np.floor(rect.min(0)).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(rect.max(0)).astype(int), [w, h])
    kinds = np.zeros(4)
    if x0 >= x1 or y0 >= y1:
        return rect, 0, kinds
    yy, xx = np.nonzero(ink[y0:y1, x0:x1])
    c, v, n, (a0, a1, b0, b1) = _frame(rect)
    p = np.stack([xx + x0 + 0.5, yy + y0 + 0.5], -1)
    a, b = (p - c) @ v, (p - c) @ n
    inside = (a >= a0) & (a <= a1) & (b >= b0) & (b <= b1)
    count = int(inside.sum())
    if not count:
        return rect, 0, kinds
    a, b = a[inside], b[inside]
    if kind_p is not None:
        kinds = kind_p[:, yy[inside] + y0, xx[inside] + x0].astype(np.float64).mean(1)
    t = _rect(c, v, n, max(a0, a.min() - 0.5), min(a1, a.max() + 0.5), max(b0, b.min() - 0.5), min(b1, b.max() + 0.5))
    return t, count, kinds


# --- the words of a page -------------------------------------------------------------------------------------------

def _table(rows: list[dict]) -> pd.DataFrame:
    """The words' rows as the table (wordboxes.COLUMNS then EXTRA_COLUMNS), typed as `wordboxes.build_boxes` types
    them."""
    floats = ("level", "page_x", "page_y", "axis", "length", "height", "t_length", "t_height", "mean_chance", "x",
              "y", "tag_spread", "p_word", "split_llr", *wordboxes.CORNER_COLS,
              *[f"p_{k}" for k in wordboxes.KIND_NAMES])
    return pd.DataFrame(rows, columns=[*wordboxes.COLUMNS, *EXTRA_COLUMNS]).astype(
        {"i": "int32", "quarter": "int8", "lab_lot": "int32", "street": bool, "ink_px": "int64", "core_px": "int64",
         **{c: "float64" for c in floats}})


def core_words(core, geo, tag, ink, level: float, r: float, refine: str, *, tag_sigma: float, split_prior: float,
               merge_iou: float | None = None, kind_p: np.ndarray | None = None, sheet: str = "") -> pd.DataFrame:
    """One page's word boxes from the word heads (see the module): `core` (h, w), `geo` (5 or 6, h, w), `tag` (k, h,
    w), `ink` (h, w) bool; `level` the core chance a word's px reach; `r` the shrink ratio of the cores; `refine`
    "unclip" or "votes"; `tag_sigma` the within-word spread of the tags and `split_prior` the log prior odds of two
    words against one (both measured, Task 8); `merge_iou` the IoU above which two votes merge (measured, Task 8;
    needed by "votes"); `kind_p` (4, h, w) the kinds' probabilities (optional); `sheet` written to the table.
    A DataFrame of wordboxes.COLUMNS then EXTRA_COLUMNS, one row per word."""
    if refine not in REFINES:
        raise ValueError(f"refine is one of {', '.join(REFINES)}, not {refine!r}")
    if refine == "votes" and merge_iou is None:
        raise ValueError("refine='votes' needs merge_iou, the IoU above which two votes merge (measured, Task 8)")
    if not 0 < level <= 1:
        raise ValueError(f"level is a core chance in (0, 1], not {level}")
    if not tag_sigma > 0:
        raise ValueError(f"tag_sigma is a spread, above 0, not {tag_sigma}")
    core = np.asarray(core)
    height, width = core.shape
    geo = np.asarray(geo)
    if geo.shape[0] not in (5, 6) or geo.shape[1:] != core.shape:
        raise ValueError(f"geo is (5 or 6, {height}, {width}), not {geo.shape}")
    tag = np.asarray(tag)
    if tag.ndim != 3 or tag.shape[1:] != core.shape:
        raise ValueError(f"tag is (k, {height}, {width}), not {tag.shape}")
    ink = np.asarray(ink, bool)

    mask = np.ascontiguousarray(core >= level)
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return _table([])
    _, lab = cv2.connectedComponents(mask.view(np.uint8), connectivity=8)
    chance = core[ys, xs].astype(np.float64)
    dist = np.moveaxis(geo[:4, ys, xs].astype(np.float64), 0, -1)
    if geo.shape[0] == 5:
        theta = np.deg2rad(geo[4, ys, xs].astype(np.float64))
    else:
        theta = np.arctan2(geo[5, ys, xs].astype(np.float64), geo[4, ys, xs].astype(np.float64))
    along = np.stack([np.cos(theta), -np.sin(theta)], -1)          # the reading direction, page px (y down)
    tags = np.moveaxis(tag[:, ys, xs].astype(np.float64), 0, -1)
    px = np.stack([xs, ys], -1).astype(np.float64)

    own = lab[ys, xs]
    order = np.argsort(own, kind="stable")
    found = []
    for comp in np.split(order, np.flatnonzero(np.diff(own[order])) + 1):
        found += _split(np.sort(comp), ys, xs, tags, tag_sigma, split_prior)
    found.sort(key=lambda f: int(f[0][0]))

    rows = []
    for i, (idx, llr) in enumerate(found):
        s = chance[idx]
        ref = (s[:, None] * along[idx]).sum(0)
        if refine == "unclip":
            box = _unclip_box(px[idx], ref, r)
        else:
            box = _squared(_lanms(_votes(px[idx], along[idx], dist[idx], ref), s, merge_iou), ref)
        g = _on_page(box, width, height)
        t, ink_px, kinds = _inked(g, ink, kind_p)
        _, v, _, (a0, a1, b0, b1) = _frame(g)
        _, _, _, (ta0, ta1, tb0, tb1) = _frame(t)
        spread = float(np.sqrt(((tags[idx] - tags[idx].mean(0)) ** 2).mean()))
        p_word = float(s.mean())
        centre = g.mean(0)
        rows.append({"sheet": sheet, "i": i, "level": float(level), "quarter": -1,
                     "page_x": float(centre[0]), "page_y": float(centre[1]),
                     "axis": float(np.rad2deg(np.arctan2(-v[1], v[0])) % 180.0),
                     "length": float(a1 - a0), "height": float(b1 - b0),
                     "t_length": float(ta1 - ta0), "t_height": float(tb1 - tb0),
                     "mean_chance": p_word, "ink_px": ink_px,
                     **{f"p_{k}": float(kinds[c]) for c, k in enumerate(wordboxes.KIND_NAMES)},
                     "x": math.nan, "y": math.nan, "lab_lot": -1, "street": False,
                     **dict(zip(wordboxes.CORNER_COLS, [*g.reshape(8).tolist(), *t.reshape(8).tolist()])),
                     "form": refine if math.isnan(llr) else f"{refine}+split", "core_px": len(idx),
                     "tag_spread": spread, "p_word": p_word, "split_llr": float(llr)})
    return _table(rows)


# --- the level -----------------------------------------------------------------------------------------------------

def terra_labels(terra: pd.DataFrame, forms: dict | None = None) -> pd.DataFrame:
    """Terra's code words as labels for the headline (`measures.label_right`): sheet, point_x, point_y (the centre of
    its rectangle), truth (the code its string is a printed form of, `forms`; None: the string itself) and quarter;
    words whose string is no code are left out."""
    code = [measures._code_of(t, forms) for t in terra["text"]]
    keep = np.array([c is not None for c in code], bool)
    centre = terra[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2).mean(1)
    return pd.DataFrame({"sheet": terra["sheet"].to_numpy()[keep], "point_x": centre[keep, 0],
                         "point_y": centre[keep, 1], "truth": [c for c, k in zip(code, keep) if k],
                         "quarter": terra["quarter"].to_numpy()[keep].astype(int)})


def _right(boxes: pd.DataFrame, reads: dict, labels: pd.DataFrame) -> np.ndarray:
    """Each label read right (1) or not (0, a label with no box too), by `measures.label_right`, a sheet at a time."""
    out = np.zeros(len(labels))
    for sheet, lab in labels.groupby("sheet", sort=True):
        mine = boxes[boxes["sheet"] == sheet]
        d = measures.label_right(mine, reads, lab.reset_index(drop=True))
        out[lab.index.to_numpy()[d["label"].to_numpy(int)]] = d["right"].to_numpy(float)
    return out


def choose_level(boxes: dict, reads: dict, terra: pd.DataFrame, forms: dict | None = None,
                 quarters: tuple[str, ...] = paths.QUARTERS) -> dict:
    """The level of each quarter (see the module): `boxes` {level: the boxes at that level (sheet, g corners; their
    index the reads' keys)}, `reads` {level: {box index: reading}} (`measures` readings), `terra` Terra's words
    (`measures.terra_words`), `forms` printed form -> code.  {"level": {quarter: level}, "right_on_others": {quarter:
    {level name: share right}}, "words_on_others": {quarter: Terra code words of the other quarters}}."""
    labels = terra_labels(terra, forms)
    levels = sorted(boxes)
    right = {lv: _right(boxes[lv], reads[lv], labels) for lv in levels}
    q = labels["quarter"].to_numpy()
    table, counts = {}, {}
    for k, name in enumerate(quarters):
        others = q != k
        counts[name] = int(others.sum())
        table[name] = {wordboxes.level_name(lv): float(right[lv][others].mean()) if others.any() else 0.0
                       for lv in levels}
    chosen = {name: max(levels, key=lambda lv: (table[name][wordboxes.level_name(lv)], lv)) for name in quarters}
    return {"level": chosen, "right_on_others": table, "words_on_others": counts}
