"""Box choice by the reader: for each of our word boxes, several versions of it, and the version the reader likes best.

Our boxes often hold two words, or cut a word short.  The reader scores any string exactly (`ctc.string_log_probs`), so
instead of trusting the grouping we read each version of the box and keep the one with the highest log posterior (the
way Strabo picks a reading by its confidence).  The versions (`candidates`), each an array (k, 4, 2) of k word
rectangles' corners in page px (edge coordinates):

  as_found        the box as it is (always first: the tie-break)
  split_gap       two words, cut at the widest empty gap in the ink along the box's axis (it exists when the ink has
                  a gap at all; no minimum width: the gap is the choice model's feature)
  join_nearest    one word, the box and the nearest other box however far (the distance between the two rectangles
                  in the box's frame; the box itself and any box overlapping it are left out); the distance is the
                  choice model's feature, so the fit decides how it lowers the odds
  trim_start      the box without its first character's width, the median character width of its kind
  trim_end        the same at the other end (needs a box longer than one such width)

No fixed threshold decides anything here: the split's gap and the join's distance (in the kind's character widths) are
features of the fitted choice (`fit_choice`, `choose(params=...)`), the character widths are measured, and the only
constants are numerical (a tie tolerance, a tiny ridge).

`choose` reads each word of each version with `read_fn(corners, box)`, which returns the log posterior (over the ink
alone) of each hypothesis: the strings it can read, and "other" and "none".  A word's score is the log of its best
hypothesis' posterior, after `prior_fn(hypothesis, box)` (a log prior at the box's place, Task 10's context packs; None
is the ink alone, `read_fn`'s as returned) is added to every hypothesis, "none" and "other" included, and the
hypotheses are renormalised.  A
version's score is the SUM of its words' scores: a design call, so that versions with different numbers of words
compare on one scale (each word's score already counts its own "none" and "other" mass, as the reader returns it).  The
sum charges a second word its own uncertainty, so a split must read two words clearly better than one; ties keep the
earlier version (`as_found` first).

The median character width of a kind is measured from Terra's words (`measure_char_widths`): box length along the
reading over string length, the median per kind, outside the answered blocks; `held_out` quarters' variants leave a
quarter's words out (for scoring that quarter).  Saved in `CHAR_WIDTH`.  Nothing here asks a model.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sfzmaps.solver import paths

KIND_NAMES = ("zoning", "block", "street", "other")        # wordboxes.KIND_NAMES, copied: no import of wordboxes
TERRA_KINDS = {"zoning code": "zoning", "block number": "block", "street name": "street", "other": "other"}
CHAR_WIDTH = paths.SOLVER_DIR / "experiments" / "wordread" / "choose" / "char_width.json"
RIDGE = 1e-6               # numerical: keeps a separable fit finite and a version never seen at 0; decides nothing
EPS = 1e-9


# --- geometry, in the box's own frame ---------------------------------------------------------------------------------

def _frame(box: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The box's centre, and its unit vectors along and across its axis (image coordinates, y down)."""
    corners = np.array([[box[f"t{i}x"], box[f"t{i}y"]] for i in range(4)], float)
    a = math.radians(float(box["axis"]))
    return corners.mean(0), np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def _rect(c: np.ndarray, u: np.ndarray, v: np.ndarray, p0: float, p1: float, q0: float, q1: float) -> np.ndarray:
    """The rectangle [p0, p1] along u and [q0, q1] across it from the point c: corners (4, 2)."""
    return np.array([c + p * u + q * v for p, q in ((p0, q0), (p1, q0), (p1, q1), (p0, q1))])


def box_kind(box: dict) -> str | None:
    """The kind the box's detector says (the largest of its p_<kind>), None with every p_<kind> zero or missing."""
    p = [float(box.get(f"p_{k}", 0.0)) for k in KIND_NAMES]
    return KIND_NAMES[int(np.argmax(p))] if max(p) > 0 else None


class Cand(np.ndarray):
    """A candidate's corners (k, 4, 2) that also carries `gap`, the px of space that separates its words from the
    box's (the split's gap, the space to the joined neighbour; 0 for the rest)."""

    gap = 0.0

    def __array_finalize__(self, obj):
        self.gap = getattr(obj, "gap", 0.0)


def _cand(rects, gap: float = 0.0) -> Cand:
    out = np.asarray(rects, float).view(Cand)
    out.gap = float(gap)
    return out


def dims(corners: np.ndarray, axis: float) -> tuple[float, float]:
    """A candidate rectangle's (length along the axis, height across it)."""
    a = math.radians(axis)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    p, q = corners @ u, corners @ v
    return float(p.max() - p.min()), float(q.max() - q.min())


def _ink_along(box: dict, ink: np.ndarray, c, u, v, half_len: float, half_h: float) -> np.ndarray:
    """The along-axis positions p (relative to the centre) of the ink px inside the box."""
    reach = int(math.ceil(abs(half_len * u[0]) + abs(half_h * v[0]))), int(math.ceil(abs(half_len * u[1])
                                                                                      + abs(half_h * v[1])))
    x0, x1 = max(0, int(c[0]) - reach[0] - 1), min(ink.shape[1], int(c[0]) + reach[0] + 2)
    y0, y1 = max(0, int(c[1]) - reach[1] - 1), min(ink.shape[0], int(c[1]) + reach[1] + 2)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(0)
    ys, xs = np.nonzero(ink[y0:y1, x0:x1])
    d = np.stack([xs + x0 + 0.5, ys + y0 + 0.5], 1) - c
    p, q = d @ u, d @ v
    return p[(np.abs(q) <= half_h) & (p >= -half_len) & (p <= half_len)]


def _widest_gap(p: np.ndarray) -> tuple[float, float] | None:
    """The widest interior gap (lo, hi) between consecutive occupied 1 px bins of the ink's positions, None with no
    gap of at least one px; the leftmost wins a tie."""
    if not len(p):
        return None
    bins = np.unique(np.floor(p).astype(int))
    gaps = np.diff(bins) - 1
    if not len(gaps) or gaps.max() < 1:
        return None
    k = int(np.argmax(gaps))
    return float(bins[k] + 1), float(bins[k + 1])


# --- the versions of a box ---------------------------------------------------------------------------------------------

def load_char_widths(f: Path | None = None, held_out: str | None = None) -> dict[str, float]:
    """The median character width per kind from `f` (default CHAR_WIDTH): the pool of every quarter, or with `held_out`
    a quarter's name the fit that never saw that quarter (a kind it has no words for takes the pool's)."""
    saved = json.loads((CHAR_WIDTH if f is None else f).read_text())
    out = dict(saved["all"])
    if held_out is not None:
        out.update(saved["held_out"][held_out])
    return out


def candidates(box: dict, ink: np.ndarray, neighbours: list[dict], widths: dict[str, float]
               ) -> list[tuple[str, Cand]]:
    """The versions of `box` (a row of boxes.parquet: the trimmed corners `t<i>x`/`t<i>y`, `axis`, `t_length`,
    `t_height`, `p_<kind>`) as (name, corners (k, 4, 2)), in the order of the module: `ink` the page's ink (bool, page
    px), `neighbours` the other boxes on the page (rows alike; the box itself and any box overlapping it are not
    joined), `widths` the median character width per kind, REQUIRED (`load_char_widths(held_out=q)`; ValueError when
    None).  A version that does not exist for this box (no gap to split at, nothing to join, too short to trim, a kind
    unknown for the trims) is left out.  Each array carries its `gap` (see `Cand`)."""
    if widths is None:
        raise ValueError("candidates needs the character widths (load_char_widths(held_out=quarter))")
    c, u, v = _frame(box)
    length, height = float(box["t_length"]), float(box["t_height"])
    hl, hh = length / 2, height / 2
    out = [("as_found", _cand(_rect(c, u, v, -hl, hl, -hh, hh)[None]))]
    gap = _widest_gap(_ink_along(box, ink, c, u, v, hl, hh))
    if gap is not None:
        lo, hi = gap
        out.append(("split_gap", _cand(np.stack([_rect(c, u, v, -hl, lo, -hh, hh), _rect(c, u, v, hi, hl, -hh, hh)]),
                                       hi - lo)))
    best = None
    for n in neighbours:
        nc, _, _ = _frame(n)
        d = nc - c
        p, q = float(d @ u), float(d @ v)
        nl, nh = float(n["t_length"]), float(n["t_height"])
        along = abs(p) - nl / 2 - hl          # the space between the two rectangles along the axis and across it
        across = abs(q) - nh / 2 - hh         # (both < 0: they overlap)
        if along < 0 and across < 0:
            continue
        dist = float(np.hypot(max(along, 0.0), max(across, 0.0)))
        if best is None or dist < best[0]:
            best = (dist, p, q, nl, nh)
    if best is not None:
        dist, p, q, nl, nh = best
        out.append(("join_nearest", _cand(_rect(c, u, v, min(-hl, p - nl / 2), max(hl, p + nl / 2),
                                                min(-hh, q - nh / 2), max(hh, q + nh / 2))[None], dist)))
    w = widths.get(box_kind(box))
    if w is not None and length > w:
        out.append(("trim_start", _cand(_rect(c, u, v, -hl + w, hl, -hh, hh)[None])))
        out.append(("trim_end", _cand(_rect(c, u, v, -hl, hl - w, -hh, hh)[None])))
    return out


# --- the choice ---------------------------------------------------------------------------------------------------------

def _logsumexp(a: np.ndarray) -> float:
    m = float(a.max())
    return m + math.log(float(np.exp(a - m).sum()))


def word_score(log_post: dict[str, float], box: dict, prior_fn=None) -> tuple[str, float]:
    """A word's best hypothesis and the log of its posterior: `log_post` (from `read_fn`) plus `prior_fn(h, box)` on
    every hypothesis and renormalised over them all (no prior: `log_post` as given).  Ties go to the first name."""
    names = sorted(log_post)
    lp = np.array([log_post[h] for h in names])
    if prior_fn is None:
        post = lp                      # the reader's own posterior, as it returned it
    else:
        lp = lp + np.array([float(prior_fn(h, box)) for h in names])
        post = lp - _logsumexp(lp)
    k = int(np.argmax(post))
    return names[k], float(post[k])


def _scored(box: dict, cands, read_fn, prior_fn=None) -> tuple[dict, dict]:
    """Every version's score (the sum of its words' `word_score`) and its words' best hypotheses."""
    if not cands or cands[0][0] != "as_found":
        raise ValueError("the candidates must start with as_found")
    scores, words = {}, {}
    for name, rects in cands:
        got = [word_score(read_fn(r, box), box, prior_fn) for r in rects]
        scores[name] = float(sum(s for _, s in got))
        words[name] = [h for h, _ in got]
    return scores, words


def _gap_width(box: dict, widths: dict[str, float]) -> float:
    """The character width the gaps are divided by: the box's kind's, the median of the kinds' with an unknown kind."""
    w = widths.get(box_kind(box))
    return float(w) if w else float(np.median(list(widths.values())))


def choice_features(box: dict, cands, read_fn, widths: dict[str, float], prior_fn=None) -> pd.DataFrame:
    """One row per version for the choice model: `cand` (name), `score` (the sum of its words' log posteriors) and
    `gap` (the split's gap or the space to the joined neighbour, in the box's kind's character widths; 0 for as_found
    and the trims)."""
    scores, _ = _scored(box, cands, read_fn, prior_fn)
    w = _gap_width(box, widths)
    return pd.DataFrame({"cand": [n for n, _ in cands], "score": [scores[n] for n, _ in cands],
                         "gap": [float(getattr(r, "gap", 0.0)) / w for _, r in cands]})


def _bbox(rect: np.ndarray) -> tuple[float, float, float, float]:
    return float(rect[:, 0].min()), float(rect[:, 1].min()), float(rect[:, 0].max()), float(rect[:, 1].max())


def _iou(a, b) -> float:
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def label_of(cands, terra_words: list) -> str | None:
    """The version whose rectangles best match Terra's words (x0, y0, x1, y1, page px, each word used once): the
    highest mean over the version's rectangles of the IoU with their (greedily, best first) assigned word, so a
    split matches two Terra words and a split of one word matches worse than the whole.  None when no version
    overlaps a word at all (no minimum IoU); the earlier version wins a tie."""
    best, best_m = None, EPS
    for name, rects in cands:
        left, got = list(range(len(terra_words))), []
        for r in rects:
            b = _bbox(r)
            ious = [(_iou(b, terra_words[k]), k) for k in left]
            if ious:
                m, k = max(ious)
                got.append(m)
                left.remove(k)
            else:
                got.append(0.0)
        m = float(np.mean(got))
        if m > best_m + EPS:
            best, best_m = name, m
    return best


NON_BASE = ("split_gap", "join_nearest", "trim_start", "trim_end")      # the versions with an alpha (as_found: 0)
GAPPED = ("split_gap", "join_nearest")                                    # the versions with a gamma


def fit_choice(rows: pd.DataFrame, ridge: float = RIDGE) -> dict:
    """The conditional logit P(c) proportional to exp(beta score_c + alpha_c + gamma_c gap_c) over a box's versions,
    fitted by maximum likelihood (L-BFGS, with a ridge of `ridge` on alpha and gamma only, so a version never seen
    stays at 0 and separable data stay finite).  alpha and gamma of as_found are 0.  `rows`: one per version of each
    box, `choice_features`'s columns plus `box` (an id) and `label` (bool: the version `label_of` gave, one per box;
    a box without exactly one is not used).  Rows may come only from the training quarters' Terra-matched boxes
    outside the answered blocks: a true `held_out` or `answered` column is refused (ValueError).  Returns {"beta",
    "alpha": {version: value}, "gamma": {version: value}, "boxes", "nll"}."""
    for flag in ("held_out", "answered"):
        if flag in rows and rows[flag].astype(bool).any():
            raise ValueError(f"fit_choice: rows flagged {flag} may not be used to fit")
    ok = rows.groupby("box")["label"].transform(lambda x: x.astype(bool).sum()) == 1
    rows = rows[ok].reset_index(drop=True)
    if not len(rows):
        raise ValueError("fit_choice: no box with exactly one labelled version")
    ids, grp = np.unique(rows["box"].to_numpy(), return_inverse=True)
    score, gap = rows["score"].to_numpy(float), rows["gap"].to_numpy(float)
    names = rows["cand"].to_numpy()
    where = {n: np.flatnonzero(names == n) for n in NON_BASE}
    lab = rows["label"].to_numpy(bool)

    def unpack(th):
        return th[0], dict(zip(NON_BASE, th[1:1 + len(NON_BASE)])), dict(zip(GAPPED, th[1 + len(NON_BASE):]))

    def nll(th):
        beta, al, ga = unpack(th)
        u = beta * score
        for n in NON_BASE:
            u[where[n]] += al[n]
        for n in GAPPED:
            u[where[n]] += ga[n] * gap[where[n]]
        m = np.full(len(ids), -np.inf)
        np.maximum.at(m, grp, u)
        z = np.zeros(len(ids))
        np.add.at(z, grp, np.exp(u - m[grp]))
        pen = ridge * (sum(v * v for v in al.values()) + sum(v * v for v in ga.values()))
        return float(-(u[lab] - m[grp[lab]] - np.log(z[grp[lab]])).sum() + pen)

    res = minimize(nll, np.zeros(1 + len(NON_BASE) + len(GAPPED)), method="L-BFGS-B")
    beta, al, ga = unpack(res.x)
    return {"beta": float(beta), "alpha": {"as_found": 0.0, **{k: float(v) for k, v in al.items()}},
            "gamma": {"as_found": 0.0, **{k: float(v) for k, v in ga.items()}}, "boxes": int(len(ids)),
            "nll": float(res.fun)}


def choose(box: dict, cands: list[tuple[str, np.ndarray]], read_fn, prior_fn=None, params: dict | None = None,
           widths: dict[str, float] | None = None) -> tuple[str, dict]:
    """The version of `box` to keep, and a dict: `scores` (every version's), `words` (the chosen version's best
    hypothesis per word).  `read_fn(corners, box)` reads one word rectangle (4, 2) and returns {hypothesis: log
    posterior}, "other" and "none" among them.

    `params` None: the version with the highest score (the sum over its words of `word_score`), the first of equal
    scores winning, so `cands` must start with as_found.  `params` (`fit_choice`, with `widths` for the gaps): the
    argmax of the fitted logit over the versions, the same tie-break; the dict then also holds `probs` (every
    version's) and `prob` (the chosen one's)."""
    scores, words = _scored(box, cands, read_fn, prior_fn)
    best = cands[0][0]
    if params is None:
        for name, _ in cands[1:]:
            if scores[name] > scores[best] + EPS:
                best = name
        return best, {"scores": scores, "words": words[best]}
    if widths is None:
        raise ValueError("choose with params needs the character widths")
    gap_w = _gap_width(box, widths)
    u = np.array([params["beta"] * scores[n] + params["alpha"].get(n, 0.0)
                  + params["gamma"].get(n, 0.0) * float(getattr(r, "gap", 0.0)) / gap_w for n, r in cands])
    p = np.exp(u - _logsumexp(u))
    k = 0
    for i in range(1, len(cands)):
        if p[i] > p[k] + EPS:
            k = i
    best = cands[k][0]
    return best, {"scores": scores, "words": words[best], "probs": dict(zip((n for n, _ in cands), p.tolist())),
                  "prob": float(p[k])}


# --- measuring the character widths -------------------------------------------------------------------------------------

def _medians(rows) -> dict[str, float]:
    out = {}
    for terra, kind in TERRA_KINDS.items():
        sel = rows[rows["kind"] == terra]
        if len(sel):
            out[kind] = float(np.median(sel["length"].to_numpy(float) / sel["text"].str.len().to_numpy(float)))
    return out


def measure_char_widths(crops) -> dict:
    """The median character width (page px) per kind from Terra's words, `crops` a crops.parquet frame
    (`textread.store`): a word's `length` along its reading over the length of its `text`, the words with a text,
    outside the answered blocks (`touch_blocks`).  {"all": the pool, "held_out": {quarter name: the pool without that
    quarter's words}}; a kind with no word is left out."""
    ok = (crops["word"] >= 0) & crops["text"].notna() & (crops["kind"] != "none") & ~crops["touch_blocks"].astype(bool)
    rows = crops[ok]
    rows = rows[rows["text"].str.len() > 0]
    return {"all": _medians(rows),
            "held_out": {q: _medians(rows[rows["quarter"] != i]) for i, q in enumerate(paths.QUARTERS)}}


def write_char_widths(f: Path | None = None) -> dict:
    """Measure from data/solver/text-crops/crops.parquet and save as JSON (`CHAR_WIDTH`)."""
    import pandas as pd

    from sfzmaps.solver.textread import store

    widths = measure_char_widths(pd.read_parquet(store.CROPS_DIR / "crops.parquet"))
    f = CHAR_WIDTH if f is None else f
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(widths, indent=1, sort_keys=True))
    return widths
