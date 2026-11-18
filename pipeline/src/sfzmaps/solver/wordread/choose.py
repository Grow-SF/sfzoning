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
features of the fitted quality (`fit_quality`, `choose(params=...)`), the character widths are measured, and the only
constants are numerical (a tie tolerance, a tiny ridge, a floor under log(1 - quality)).

The fitted choice (`fit_quality`) models each version's OWN quality, the chance it holds exactly one whole word
(`label_weights`, by Terra's words' ink), from its words' read scores, its gap features, the box's kind, the version
as a fitted offset and, when given, Task 15's box model's log P(count = 1) of each word's crop; a split's quality is
the product of its halves'.  `choose` keeps the version of the highest quality.  The versions do not compete for
shares of one box's probability (a conditional logit did, and near-copies, as found and its two trims, split their
outcome's share three ways against a lone join: Task 11's NE run picked the join for 64% of boxes).

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sfzmaps.solver import paths
from sfzmaps.solver.wordread import boxmodel

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


@dataclass(frozen=True)
class Candidate:
    """A version of a box: its `name`, its word rectangles `rects` (k, 4, 2), and the two features the choice model
    takes (plain floats, so a candidate pickles and its features go to a table): `along`, the split's gap or the
    space along the axis to the joined neighbour, in the box's kind's character widths; `across`, the space across the
    axis to the joined neighbour in word heights (0 for the rest).  It iterates as (name, rects)."""

    name: str
    rects: np.ndarray
    along: float = 0.0
    across: float = 0.0

    def __iter__(self):
        return iter((self.name, self.rects))


def _gap_width(box: dict, widths: dict[str, float]) -> float:
    """The character width the gaps are divided by: the box's kind's, the median of the kinds' with an unknown kind."""
    w = widths.get(box_kind(box))
    return float(w) if w else float(np.median(list(widths.values())))


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
               ) -> list[Candidate]:
    """The versions of `box` (a row of boxes.parquet: the trimmed corners `t<i>x`/`t<i>y`, `axis`, `t_length`,
    `t_height`, `p_<kind>`) as `Candidate`s (name, corners (k, 4, 2)), in the order of the module: `ink` the page's ink (bool, page
    px), `neighbours` the other boxes on the page (rows alike; the box itself and any box overlapping it are not
    joined), `widths` the median character width per kind, REQUIRED (`load_char_widths(held_out=q)`; ValueError when
    None).  A version that does not exist for this box (no gap to split at, nothing to join, too short to trim, a kind
    unknown for the trims) is left out.  Each carries its features (see `Candidate`)."""
    if widths is None:
        raise ValueError("candidates needs the character widths (load_char_widths(held_out=quarter))")
    c, u, v = _frame(box)
    length, height = float(box["t_length"]), float(box["t_height"])
    hl, hh = length / 2, height / 2
    cw = _gap_width(box, widths)
    out = [Candidate("as_found", _rect(c, u, v, -hl, hl, -hh, hh)[None])]
    gap = _widest_gap(_ink_along(box, ink, c, u, v, hl, hh))
    if gap is not None:
        lo, hi = gap
        out.append(Candidate("split_gap", np.stack([_rect(c, u, v, -hl, lo, -hh, hh), _rect(c, u, v, hi, hl, -hh, hh)]),
                             (hi - lo) / cw))
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
            best = (dist, p, q, nl, nh, max(along, 0.0), max(across, 0.0))
    if best is not None:
        _, p, q, nl, nh, along, across = best
        out.append(Candidate("join_nearest", _rect(c, u, v, min(-hl, p - nl / 2), max(hl, p + nl / 2),
                                                    min(-hh, q - nh / 2), max(hh, q + nh / 2))[None],
                             along / cw, across / height))
    w = widths.get(box_kind(box))
    if w is not None and length > w:
        out.append(Candidate("trim_start", _rect(c, u, v, -hl + w, hl, -hh, hh)[None]))
        out.append(Candidate("trim_end", _rect(c, u, v, -hl, hl - w, -hh, hh)[None]))
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


def _scored(box: dict, cands, read_fn, prior_fn=None) -> tuple[dict, dict, dict]:
    """Every version's score (the sum of its words' `word_score`), its words' best hypotheses and its words' own
    scores."""
    if not cands or cands[0].name != "as_found":
        raise ValueError("the candidates must start with as_found")
    scores, words, each = {}, {}, {}
    for name, rects in cands:
        got = [word_score(read_fn(r, box), box, prior_fn) for r in rects]
        each[name] = [float(s) for _, s in got]
        scores[name] = float(sum(each[name]))
        words[name] = [h for h, _ in got]
    return scores, words, each


def _features(box: dict, cands, scores: dict, each: dict, count_fn=None) -> pd.DataFrame:
    f = pd.DataFrame({"cand": [c.name for c in cands], "score": [scores[c.name] for c in cands],
                      "word_scores": [list(each[c.name]) for c in cands], "gap": [c.along for c in cands],
                      "across": [c.across for c in cands]})
    f["kind"] = box_kind(box) or "unknown"
    if count_fn is not None:
        f["word_count1"] = [[float(count_fn(r, box)) for r in c.rects] for c in cands]
    return f


def choice_features(box: dict, cands, read_fn, prior_fn=None, count_fn=None) -> pd.DataFrame:
    """One row per version for the quality model: `cand` (name), `score` (the sum of its words' log posteriors, the
    plain choice's), `word_scores` (each word's own log posterior: one for most versions, two for a split), `gap` (the
    split's gap or the join's space along the axis, in the kind's character widths), `across` (the join's space across
    the axis, in word heights; both 0 for as_found and the trims) and `kind` (the box's, `box_kind`; "unknown" without
    one).  With `count_fn(corners, box)` (Task 15's box model: a word rectangle's log P(count = 1), e.g.
    `boxmodel.box_logp` of its `boxmodel.box_crop`) also `word_count1`, each word's."""
    scores, _, each = _scored(box, cands, read_fn, prior_fn)
    return _features(box, cands, scores, each, count_fn)


def _corners(word) -> np.ndarray:
    """A Terra word as corners (4, 2): (x0, y0, x1, y1) is an upright box, an array (4, 2) a slanted word's rotated
    rectangle."""
    a = np.asarray(word, float)
    if a.shape == (4,):
        x0, y0, x1, y1 = a
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return a.reshape(4, 2)


def label_weights(cands: list[Candidate], terra_words: list, ink: np.ndarray) -> dict[str, float]:
    """Each version's chance of holding exactly one Terra word, by ink and not by rectangles: for every rectangle of the
    version, each overlapping word's share of its own ink inside the rectangle (`boxmodel.ink_shares`) are independent
    chances, and the rectangle holds exactly one word with the Poisson-binomial P(count = 1) of them
    (`boxmodel.poisson_binomial`); a version of several rectangles (a split) holds one word in each with the product.
    `terra_words`: the training Terra words (page px; see `_corners`), `ink` the page's ink (bool).  A version holding
    no word, or two, weighs about 0; no IoU anywhere."""
    words = np.array([_corners(w) for w in terra_words], float).reshape(-1, 4, 2)
    rects = np.concatenate([np.asarray(c.rects, float) for c in cands])
    ri, _, share = boxmodel.ink_shares(rects, words, ink) if len(words) else (np.zeros(0, int), None, np.zeros(0))
    out, at = {}, 0
    for c in cands:
        w = 1.0
        for k in range(len(c.rects)):
            pb = boxmodel.poisson_binomial(share[ri == at + k])
            w *= float(pb[1]) if len(pb) > 1 else 0.0
        out[c.name] = w
        at += len(c.rects)
    return out


TYPES = ("as_found", "split_gap", "join_nearest", "trim_start", "trim_end")
NON_BASE = TYPES[1:]                                       # the versions with an offset alpha (as_found's: the intercept)
GAPPED = ("split_gap", "join_nearest")                     # the versions with a gamma (along)
ACROSS = ("join_nearest",)                                 # the versions with a gamma across
KIND_OFFSETS = ("block", "street", "other", "unknown")     # the box kinds with an offset (zoning's: the intercept)
NEED = ("box", "cand", "word_scores", "gap", "across", "kind", "weight")
TINY = 1e-300              # numerical: keeps log(1 - quality) finite at a quality of exactly 1; decides nothing


def _pieces(rows: pd.DataFrame, count: bool) -> tuple[np.ndarray, np.ndarray]:
    """The design matrix of every word of every version (one row per word: a split has two) and the version (position
    in `rows`) each word belongs to.  Columns: the intercept, alpha per NON_BASE, gamma (along) per GAPPED, gamma
    across per ACROSS, the offset per KIND_OFFSETS, the word's score (beta) and, with `count`, its log P(count = 1)."""
    ws = [np.asarray(w, float).ravel() for w in rows["word_scores"]]
    owner = np.repeat(np.arange(len(rows)), [len(w) for w in ws]).astype(int)
    names = rows["cand"].to_numpy(dtype=object)[owner]
    kinds = rows["kind"].to_numpy(dtype=object)[owner]
    along, across = rows["gap"].to_numpy(float)[owner], rows["across"].to_numpy(float)[owner]
    cols = [np.ones(len(owner))]
    cols += [(names == t).astype(float) for t in NON_BASE]
    cols += [np.where(names == t, along, 0.0) for t in GAPPED]
    cols += [np.where(names == t, across, 0.0) for t in ACROSS]
    cols += [(kinds == k).astype(float) for k in KIND_OFFSETS]
    cols.append(np.concatenate(ws) if ws else np.zeros(0))
    if count:
        c1 = [np.asarray(c, float).ravel() for c in rows["word_count1"]]
        if any(len(c) != len(w) for c, w in zip(c1, ws)):
            raise ValueError("word_count1 needs one value per word, as word_scores")
        cols.append(np.concatenate(c1) if c1 else np.zeros(0))
    return np.stack(cols, 1), owner


def _unpack(th: np.ndarray, count: bool) -> dict:
    k = 1
    al = dict(zip(NON_BASE, th[k:k + len(NON_BASE)]))
    k += len(NON_BASE)
    ga = dict(zip(GAPPED, th[k:k + len(GAPPED)]))
    k += len(GAPPED)
    gx = dict(zip(ACROSS, th[k:k + len(ACROSS)]))
    k += len(ACROSS)
    kd = dict(zip(KIND_OFFSETS, th[k:k + len(KIND_OFFSETS)]))
    k += len(KIND_OFFSETS)
    return {"intercept": float(th[0]), "beta": float(th[k]), "count1": float(th[k + 1]) if count else None,
            "alpha": {"as_found": 0.0, **{n: float(v) for n, v in al.items()}},
            "gamma": {"as_found": 0.0, **{n: float(v) for n, v in ga.items()}},
            "gamma_across": {"as_found": 0.0, **{n: float(v) for n, v in gx.items()}},
            "kind": {KIND_NAMES[0]: 0.0, **{n: float(v) for n, v in kd.items()}}}


def _theta(params: dict) -> np.ndarray:
    th = [params["intercept"], *(params["alpha"].get(n, 0.0) for n in NON_BASE),
          *(params["gamma"].get(n, 0.0) for n in GAPPED), *(params["gamma_across"].get(n, 0.0) for n in ACROSS),
          *(params["kind"].get(k, 0.0) for k in KIND_OFFSETS), params["beta"]]
    if params.get("count1") is not None:
        th.append(params["count1"])
    return np.array(th, float)


def _log_quality(X: np.ndarray, owner: np.ndarray, n: int, th: np.ndarray):
    """Each word's eta and each version's log quality (the sum over its words of log sigmoid(eta))."""
    eta = X @ th
    return eta, np.bincount(owner, -np.logaddexp(0.0, -eta), minlength=n)


def predict_quality(params: dict, rows: pd.DataFrame) -> np.ndarray:
    """Each version's (row's) quality under `params` (`fit_quality`): the product over its words of sigmoid(eta)."""
    count = params.get("count1") is not None
    if count and "word_count1" not in rows:
        raise ValueError("these params weigh the box model's log P(count = 1): the rows need `word_count1` "
                         "(choose: pass count_fn)")
    X, owner = _pieces(rows.reset_index(drop=True), count)
    return np.exp(_log_quality(X, owner, len(rows), _theta(params))[1])


def fit_quality(rows: pd.DataFrame, ridge: float = RIDGE) -> dict:
    """Each version's own quality: a fractional logistic regression of its label weight w (`label_weights`: the chance
    it holds exactly one whole Terra word, a SOFT target in [0, 1]) on its features, every version its own observation
    (versions do not compete for shares: near-copies such as as_found and its trims can't split a vote).  A word's
    eta = intercept + alpha_version + gamma_version along + gamma'_version across + kind_offset + beta score_word
    (+ count1 log P(count = 1)_word when the rows carry `word_count1`, Task 15's box model; fitted only then); a
    version's quality is the product over its words of sigmoid(eta) (a split's: its halves', as its weight is), fitted
    by maximum quasi-likelihood, sum of w log q + (1 - w) log(1 - q), by L-BFGS with its exact gradient and a tiny
    numerical ridge `ridge` on every weight (a version or kind never seen stays at 0, a separated fit stays finite).
    alpha, gamma and gamma' of as_found and the offset of a zoning box are 0.

    `rows`: one per version of each box, `choice_features`'s columns (`cand`, `word_scores`, `gap`, `across`, `kind`;
    `word_count1` optional) plus `box` (an id) and `weight`.  Rows may come only from the training quarters' boxes
    outside the answered blocks: the columns `held_out` and `answered` are REQUIRED (ValueError when missing) and a
    true one is refused.  Returns {"intercept", "beta", "count1" (None without the column), "alpha", "gamma",
    "gamma_across" ({version: value}), "kind" ({kind: value}), "boxes", "versions", "nll", "converged",
    "iterations"}."""
    for flag in ("held_out", "answered"):
        if flag not in rows:
            raise ValueError(f"fit_quality: rows need a `{flag}` column (False for a row that may be used)")
        if rows[flag].astype(bool).any():
            raise ValueError(f"fit_quality: rows flagged {flag} may not be used to fit")
    missing = [c for c in NEED if c not in rows]
    if missing or not len(rows):
        raise ValueError(f"fit_quality: needs rows with the columns {NEED} (missing {missing})")
    rows = rows.reset_index(drop=True)
    count = "word_count1" in rows
    X, owner = _pieces(rows, count)
    w = np.clip(rows["weight"].to_numpy(float), 0.0, 1.0)
    n = len(rows)

    def nll(th):
        eta, lq = _log_quality(X, owner, n, th)
        l1q = np.log(np.maximum(-np.expm1(lq), TINY))                       # log(1 - quality)
        f = float(-(w * lq + (1.0 - w) * l1q).sum() + ridge * th @ th)
        # d f / d eta_word = -(w - q) sigmoid(-eta_word) / (1 - q), the ratio <= 1 (it is 1 for a one-word version)
        ratio = np.exp(np.minimum(-np.logaddexp(0.0, eta) - l1q[owner], 0.0))
        return f, X.T @ (-(w - np.exp(lq))[owner] * ratio) + 2.0 * ridge * th

    res = minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 10_000})
    out = _unpack(res.x, count)
    out.update(boxes=int(rows["box"].nunique()), versions=int(n), nll=float(res.fun), converged=bool(res.success),
               iterations=int(res.nit))
    return out


def choose(box: dict, cands: list[Candidate], read_fn, prior_fn=None, params: dict | None = None, count_fn=None
           ) -> tuple[str, dict]:
    """The version of `box` to keep, and a dict: `scores` (every version's), `words` (the chosen version's best
    hypothesis per word).  `read_fn(corners, box)` reads one word rectangle (4, 2) and returns {hypothesis: log
    posterior}, "other" and "none" among them.

    `params` None: the version with the highest score (the sum over its words of `word_score`), the first of equal
    scores winning, so `cands` must start with as_found.  `params` (`fit_quality`; the gaps are in the candidates): the
    version of the highest predicted quality (each version's own, `predict_quality`: a split's the product of its
    halves'), the same tie-break; the dict then also holds `quality` (every version's) and `prob` (the chosen one's).
    Params fitted with the box model's column need `count_fn(corners, box)`, its log P(count = 1) of a word rectangle
    (ValueError without)."""
    scores, words, each = _scored(box, cands, read_fn, prior_fn)
    best = cands[0].name
    if params is None:
        for name in (c.name for c in cands[1:]):
            if scores[name] > scores[best] + EPS:
                best = name
        return best, {"scores": scores, "words": words[best]}
    count = params.get("count1") is not None
    if count and count_fn is None:
        raise ValueError("these params weigh the box model's log P(count = 1): pass count_fn")
    q = predict_quality(params, _features(box, cands, scores, each, count_fn if count else None))
    k = 0
    for i in range(1, len(cands)):
        if q[i] > q[k] + EPS:
            k = i
    best = cands[k].name
    return best, {"scores": scores, "words": words[best], "quality": dict(zip((c.name for c in cands), q.tolist())),
                  "prob": float(q[k])}


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
