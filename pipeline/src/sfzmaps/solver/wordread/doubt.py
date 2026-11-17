"""Doubt weights for Terra's words: each word's probability of being right, so a wrong label counts less in training.

Some of Terra's word boxes are wrong: a word cut at a tile's edge and read again whole from the next tile, a box grown
over the next word, a near-45-degree word whose height `boxes.word_rect` had to force.  No word is cut, flagged or
left out here: each gets two weights, fitted on the words Terra was shown twice.

Terra was shown each sheet in overlapping tiles (`boxes.terra_boxes`; 768 px tiles about 583 px apart), so a word in
the overlap of two tiles was shown twice, and each tile's answer read it on its own.  `terra_boxes` keeps one read
(the one farthest from its tile's edge) and drops a second whose box overlaps it by IoU > `boxes.SAME_TEXT_IOU`; the
second reads it drops are the ones used here, every tile's read (`tile_reads`) as Terra gave it.

The features of a Terra word (`doubt_features`, one row per word of a sheet, in the order of `SheetData.words`; the
sheet's line px, `SheetData.line_bits`, are needed: `zones.sheet_lines`):
  ink_outside    the share of the px of its letters' ink pieces (the 8-connected components of the page's ink less
                 its line px that meet its rectangle, `SheetData.words` as `data._paint` fills it) that lie outside
                 its rectangle; a rectangle holding no such ink has all of it outside: 1
  line_share     the share of its rectangle's px that are line ink (ink and line px)
  axis           |cos^2 a - sin^2 a| of its reading angle a (1 for a word along or across the page, 0 at 45 degrees:
                 how well its axis box gives its height, the continuous form of `boxes.word_rect`'s forced case); the
                 fit reads axis and axis^2 (the pairs' IoU is lowest between: a height solved from an ill-conditioned
                 box at 20-35 degrees disagrees more than the forced one at 45)
  edge           Terra's distance (px) of its box from its tile's edge; the fit reads log(1 + edge)
  kind           Terra's kind
  read_twice     another tile's window holds its whole box: Terra was shown it twice
  paired         such a tile read something whose rectangle overlaps the word's (IoU above 0)
  other_*        the other read: of the other tiles' reads (every one, before the drop), the one whose rectangle
                 overlaps the word's with the highest IoU.  ANY positive overlap pairs (argmax, no IoU cut), so the
                 other read may be a neighbouring word (the word's own read in the other tile missed, or cut): such
                 pairs enter the IoU fit as they are, as the low-IoU component of its mixture.  Its text, the pair's
                 IoU (`rect_iou` of the two reads' `word_rect`s, each at its own angle) and whether the two strings
                 agree once folded (upper case, zero-width spaces dropped, single spaces: `crops.norm_text`'s folding)
  touch_<name>   with `areas`, whether its rectangle holds a px of each area (the held-out quarters and the answered
                 blocks, as the reader's crops record them)
Reported only: forced_height (`word_rect`'s near-45-degree case), line_of_ink (the share of the ink inside its
rectangle that is line px), ink_px.  The forms were chosen by the held-out fit (Follow-up round 1): line ink as a share
of the rectangle over a share of its ink, axis and axis^2 over axis alone, the distance from the nearest quarter turn
or the forced case.

The fit (`fit_doubt`, on the training quarters' words shown twice outside the answered blocks; the rows say so in
`held_out` and `answered`, and a true one is refused), from x = (one intercept per kind, ink_outside, line_share, axis,
axis^2, log(1 + edge)), a zero-inflated model of the second read:
  paired  a logistic model of P(the other tile read something there | x): a miss is an outcome, not dropped
  agree   given a pair, a logistic model of P(the two strings agree | x)
  IoU     given a pair, P(IoU = 1 | x) (two reads of one rectangle) by a logistic (`one`), and otherwise a mixture
          of two beta regressions (Ospina and Ferrari's inflated beta, Ferrari and Cribari-Neto's beta regression):
          with P(same word | x) = logistic (`same`), the same word's IoU ~ Beta(mu phi, (1 - mu) phi), logit mu =
          x . `mean`, log phi = x . `precision`, and a neighbour's IoU ~ Beta with one mean and precision
          (`neighbour`).  The mixture fitted every held-out quarter better than one beta (report, Follow-up round 1).
Each by maximum likelihood (L-BFGS, analytic gradients), with RIDGE on every coefficient only to keep a separable fit
finite.

The weights (`weights_of`, `doubt_weights`), per word, with no cut and no flag:
  box     the expected IoU of its second read, a miss counting 0: P(paired) (P(IoU = 1) + (1 - P(IoU = 1))
          (P(same) mu + (1 - P(same)) mu_neighbour)): the word's weight in the detector's word terms
          (`data.SheetData.weights`, `prepare`)
  string  P(string right), taken as P(paired) P(agree): for the reader's targets
Both are predictions from the word's features alone (its own pair, when it has one, is one of the fit's rows).  A
word read once lies farther from its tile's edge (up to 369 px) than any word shown twice (up to 251): its edge term
is extrapolated, and `counts` reports the weights beyond the fit's edges.

`write_doubt` measures the training maps' words (`paths.TEXT_TRAIN_SERIES`), fits once per fold of the reader's split
(`fit_folds`: `paths.QUARTERS`, the words touching the fold's quarter or an answered block left out of its fit) and
writes the features (FEATURES_FILE) and, per fold, the fitted parameters, every word's weights with its sheet's
words' hash (`words_hash`, checked by `prepare`) and the counts (DOUBT_FILE).

Neither `textnet.data` nor `textnet.train` imports this module: the weights reach the detector in
`data.SheetData.weights`, filled by `prepare` before the training.  Nothing here asks a model.
"""
import dataclasses
import hashlib
import json
import logging
from collections.abc import Mapping

import cv2
import numpy as np
import pandas as pd
import shapely
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, log_expit, logsumexp

from sfzmaps.solver import paths
from sfzmaps.solver.textnet import boxes, data

log = logging.getLogger(__name__)

DOUBT_FILE = paths.SOLVER_DIR / "experiments" / "wordread" / "detector" / "doubt.json"
FEATURES_FILE = DOUBT_FILE.with_name("doubt_features.parquet")
KIND_NAMES = tuple(data.KINDS)                 # Terra's kinds, in the order of their KINDS index
FEATURES = (*(f"kind={k}" for k in KIND_NAMES), "ink_outside", "line_share", "axis", "axis2", "log_edge")
NEAR_DIAGONAL = 0.25    # `boxes.word_rect`'s: |cos^2 - sin^2| of the angle at or below it forces the height (its code,
                        # not a choice here; copied, as boxes.py is in the prints stage's closure; a test pins it;
                        # reported only)
RIDGE = 1e-6            # numerical: keeps a separable fit finite; decides nothing
QUANTILES = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)    # reported only


# --- Terra's reads ---------------------------------------------------------------------------------------------------

def _sheet_records(series: str, sheet: str, records: list[dict] | None) -> list[dict]:
    return [r for r in (boxes.terra_records() if records is None else records)
            if r.get("series") == series and r.get("sheet") == sheet]


def tile_reads(series: str, sheet: str, records: list[dict] | None = None) -> list[dict]:
    """Every piece of text Terra read on a sheet, every tile's, before `boxes.terra_boxes` drops repeats: Terra's
    fields, `box` (page px), `tile` (its tile's top-left corner) and `page_box` (its tile, page px), decoded as
    `terra_boxes` decodes them.  `records`: the lines of TERRA_READ (read from disk when None)."""
    out = []
    for r in _sheet_records(series, sheet, records):
        tx, ty = r["page_box"][:2]
        for it in boxes.TextItems.model_validate(r["answer"]).items:
            out.append({**it.model_dump(), "box": [it.x0 + tx, it.y0 + ty, it.x1 + tx, it.y1 + ty],
                        "tile": [tx, ty], "page_box": list(r["page_box"])})
    return out


def axis_of(angle_deg) -> np.ndarray:
    """|cos^2 a - sin^2 a| (= |cos 2a|) of reading angles `angle_deg` (degrees; any shape): 1 along or across the
    page, 0 at 45 degrees."""
    a = np.deg2rad(np.asarray(angle_deg, float))
    c, s = np.abs(np.cos(a)), np.abs(np.sin(a))
    return np.abs(c * c - s * s)


def forced_height(angle_deg) -> np.ndarray:
    """Whether `boxes.word_rect` forces a word's height at reading angle `angle_deg` (degrees; any shape): its
    near-45-degree case, where the box can't give the height.  Reported only (the fit reads `axis_of`)."""
    return ~(axis_of(angle_deg) > NEAR_DIAGONAL)


def _norm(text) -> str:
    """A string folded for comparison: upper case, zero-width spaces dropped, single spaces (`crops.norm_text`'s
    folding, without its vocabulary)."""
    return " ".join(str(text).replace("​", "").upper().split())


def rect_iou(a: np.ndarray, b: np.ndarray) -> float:
    """The IoU of two word rectangles (4 corners each): 1 exactly when they are the same rectangle."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.array_equal(a, b):
        return 1.0
    pa, pb = shapely.Polygon(a), shapely.Polygon(b)
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0


def _rect(item: dict) -> np.ndarray:
    return boxes.word_rect(item["box"], float(item.get("angle_deg") or 0.0))


def _other_reads(kept: list[dict], reads: list[dict]) -> list[dict]:
    """For each kept word: the number of other tiles whose window holds its whole box and, among those tiles' reads,
    the one whose rectangle overlaps its own with the highest IoU (None when none overlaps; any overlap pairs)."""
    tiles = {tuple(r["tile"]): r["page_box"] for r in reads}
    rects = [_rect(r) for r in reads]
    lo = np.array([r.min(0) for r in rects]).reshape(-1, 2)
    hi = np.array([r.max(0) for r in rects]).reshape(-1, 2)
    tile_of = [tuple(r["tile"]) for r in reads]
    out = []
    for k in kept:
        x0, y0, x1, y1 = k["box"]
        own = tuple(k["tile"])
        others = {t for t, (tx0, ty0, tx1, ty1) in tiles.items()
                  if t != own and tx0 <= x0 and ty0 <= y0 and x1 <= tx1 and y1 <= ty1}
        best, best_iou = None, 0.0
        if others:
            rect = _rect(k)
            klo, khi = rect.min(0), rect.max(0)
            near = np.flatnonzero((lo[:, 0] < khi[0]) & (hi[:, 0] > klo[0]) & (lo[:, 1] < khi[1]) & (hi[:, 1] > klo[1]))
            for j in near:
                if tile_of[j] not in others:
                    continue
                v = rect_iou(rect, rects[j])
                if v > best_iou:
                    best, best_iou = reads[j], v
        out.append({"other_tiles": len(others), "other": best, "iou": best_iou})
    return out


# --- the features ----------------------------------------------------------------------------------------------------

def words_hash(sd: data.SheetData) -> str:
    """A hash of a sheet's words' rectangles (`SheetData.words`): the weights of one sheet fit only its own words."""
    w = np.ascontiguousarray(sd.words, np.int32)
    return hashlib.sha1(str(w.shape).encode() + w.tobytes()).hexdigest()


def _window(rect: np.ndarray, sd: data.SheetData) -> tuple[int, int, int, int] | None:
    """The page window (x0, y0, w, h) holding a rectangle's px (its corners' bounding box), cut to the page."""
    x0, y0 = np.maximum(rect.min(0), 0)
    x1, y1 = np.minimum(rect.max(0) + 1, [sd.width, sd.height])
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def doubt_features(sd: data.SheetData, records: list[dict] | None = None, areas: Mapping | None = None
                   ) -> pd.DataFrame:
    """One row per Terra word of the sheet, in the order of `sd.words` (see the module): series, sheet, word (its
    index), text, kind, angle, edge, ink_outside, line_share, axis, line_of_ink, ink_px, forced_height, read_twice,
    other_tiles, paired, other_text, other_iou, agree (1.0 / 0.0; NaN unpaired), and a bool column per entry of
    `areas` ({column name: data.MapRect / Quarter / GridArea}; needs the sheet's placement).  `records`: the lines of
    TERRA_READ (read from disk when None).  ValueError when the sheet holds no line px (`zones.sheet_lines`) or its
    words are not Terra's words of the sheet (`boxes.terra_boxes`, as `data.sheet_data` reads them)."""
    if sd.line_bits.shape != sd.ink_bits.shape:
        raise ValueError(f"{sd.series} {sd.sheet}: no line px (SheetData.line_bits {sd.line_bits.shape}): the doubt "
                         "features need them (textnet.zones.sheet_lines)")
    recs = _sheet_records(sd.series, sd.sheet, records)
    kept = boxes.terra_boxes(sd.series, sd.sheet, recs)
    rects = np.round(np.array([_rect(k) for k in kept], float).reshape(-1, 4, 2)).astype(np.int32)
    if rects.shape != sd.words.shape or not np.array_equal(rects, sd.words):
        raise ValueError(f"{sd.series} {sd.sheet}: the sheet's {len(sd.words)} words are not Terra's words of the "
                         f"sheet ({len(kept)}, boxes.terra_boxes)")
    areas = dict(areas or {})
    ink = data.page_ink(sd)
    line = np.unpackbits(sd.line_bits, axis=1, count=sd.width).view(bool)
    letters = ink & ~line
    _, labels, stats, _ = cv2.connectedComponentsWithStats(letters.view(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    size = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    rows = []
    for j, (k, other) in enumerate(zip(kept, _other_reads(kept, tile_reads(sd.series, sd.sheet, recs)))):
        win = _window(sd.words[j], sd)
        inside = total = ink_in = line_in = area = 0
        touch = dict.fromkeys(areas, False)
        if win is not None:
            x0, y0, w, h = win
            mask = data._paint(sd.words[j:j + 1], np.ones(1, np.int32), y0, x0, h, w) > 0
            lab = labels[y0:y0 + h, x0:x0 + w][mask]
            ids, counts = np.unique(lab[lab > 0], return_counts=True)
            inside, total = int(counts.sum()), int(size[ids].sum())
            ink_w, line_w = ink[y0:y0 + h, x0:x0 + w][mask], line[y0:y0 + h, x0:x0 + w][mask]
            ink_in, line_in, area = int(ink_w.sum()), int((ink_w & line_w).sum()), int(mask.sum())
            touch = {name: bool(data.in_area(sd, area_, y0, x0, h, w)[mask].any()) for name, area_ in areas.items()}
        o = other["other"]
        angle = float(k.get("angle_deg") or 0.0)
        rows.append({"series": sd.series, "sheet": sd.sheet, "word": j, "text": k["text"], "kind": k["kind"],
                     "angle": angle, "edge": float(k["edge"]),
                     "ink_outside": 1.0 - inside / total if total else 1.0,
                     "line_share": line_in / area if area else 0.0, "axis": float(axis_of(angle)),
                     "line_of_ink": line_in / ink_in if ink_in else 0.0, "ink_px": ink_in,
                     "forced_height": bool(forced_height(angle)),
                     "read_twice": other["other_tiles"] > 0, "other_tiles": other["other_tiles"],
                     "paired": o is not None, "other_text": None if o is None else o["text"],
                     "other_iou": np.nan if o is None else other["iou"],
                     "agree": np.nan if o is None else float(_norm(o["text"]) == _norm(k["text"])), **touch})
    cols = ["series", "sheet", "word", "text", "kind", "angle", "edge", "ink_outside", "line_share", "axis",
            "line_of_ink", "ink_px", "forced_height", "read_twice", "other_tiles", "paired", "other_text", "other_iou",
            "agree", *areas]
    return pd.DataFrame(rows, columns=cols)


def design(features: pd.DataFrame) -> np.ndarray:
    """The fit's x per row (FEATURES): one intercept per Terra kind, ink_outside, line_share, axis, axis^2 and
    log(1 + edge).  ValueError for an unknown kind, a feature that is not a number or an edge below 0."""
    kind = features["kind"].to_numpy()
    unknown = sorted(set(kind) - set(KIND_NAMES))
    if unknown:
        raise ValueError(f"doubt: kinds {unknown} are not Terra's ({', '.join(KIND_NAMES)})")
    edge = features["edge"].to_numpy(float)
    if (edge < 0).any():
        raise ValueError("doubt: a tile edge distance below 0")
    axis = features["axis"].to_numpy(float)
    x = np.column_stack([*(kind == k for k in KIND_NAMES), features["ink_outside"].to_numpy(float),
                         features["line_share"].to_numpy(float), axis, axis * axis, np.log1p(edge)]).astype(float)
    if not np.isfinite(x).all():
        raise ValueError("doubt: every feature must be a number for every word")
    return x


# --- the fit ---------------------------------------------------------------------------------------------------------

def _logistic(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    """Maximum-likelihood logistic coefficients of y (0/1) on x, RIDGE on each, and the negative log-likelihood."""
    def nll(b):
        z = x @ b
        f = -(y * log_expit(z) + (1 - y) * log_expit(-z)).sum() + ridge * b @ b
        return f, x.T @ (expit(z) - y) + 2 * ridge * b

    res = minimize(nll, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    return res.x, float(res.fun)


def _beta_terms(x: np.ndarray, ly: np.ndarray, l1y: np.ndarray, g: np.ndarray, d: np.ndarray):
    """Each row's beta log-density (logit mu = x . g, log phi = x . d) and its gradients by g and d (rows x k each)."""
    mu = expit(x @ g)
    phi = np.exp(x @ d)
    a, b = mu * phi, (1 - mu) * phi
    ll = gammaln(phi) - gammaln(a) - gammaln(b) + (a - 1) * ly + (b - 1) * l1y
    da, db = digamma(a), digamma(b)
    dl_dmu = phi * (-da + db + ly - l1y)
    dl_dphi = digamma(phi) - mu * da - (1 - mu) * db + mu * ly + (1 - mu) * l1y
    return ll, x * (dl_dmu * mu * (1 - mu))[:, None], x * (dl_dphi * phi)[:, None]


def _beta(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Maximum-likelihood beta regression of y in (0, 1) on x, its mean and its precision each modelled (logit mu =
    x . g, log phi = x . d: the variable-dispersion beta regression), RIDGE on each: g, d and the negative
    log-likelihood."""
    ly, l1y = np.log(y), np.log1p(-y)
    k = x.shape[1]

    def nll(th):
        ll, gg, gd = _beta_terms(x, ly, l1y, th[:k], th[k:])
        return -ll.sum() + ridge * th @ th, -np.concatenate([gg.sum(0), gd.sum(0)]) + 2 * ridge * th

    res = minimize(nll, np.zeros(2 * k), jac=True, method="L-BFGS-B")
    return res.x[:k], res.x[k:], float(res.fun)


def _beta_mixture(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[dict, float]:
    """Maximum-likelihood mixture of two beta regressions of y in (0, 1): with P(same | x) = logistic(x . c), the
    same word's beta (logit mu = x . g, log phi = x . d) and a neighbour's (one mean and one precision, logit m and
    log p), RIDGE on each.  Started from the one-beta fit for the same word and a low mean for the neighbour (the
    components are told apart by where they start).  Returns ({"same", "mean", "precision", "neighbour"}, the
    negative log-likelihood)."""
    ly, l1y = np.log(y), np.log1p(-y)
    k = x.shape[1]
    ones = np.ones((len(y), 1))
    g0, d0, _ = _beta(x, y, ridge)
    q = np.quantile(y, 0.05)
    kinds = len(KIND_NAMES)                             # x's first columns: one intercept per kind
    start = np.concatenate([np.full(kinds, 3.0), np.zeros(k - kinds), g0, d0, [np.log(q / (1 - q)), 0.0]])

    def nll(th):
        c, g, d, nb = th[:k], th[k:2 * k], th[2 * k:3 * k], th[3 * k:]
        ls, gg, gd = _beta_terms(x, ly, l1y, g, d)
        ln, gn_m, gn_p = _beta_terms(ones, ly, l1y, nb[:1], nb[1:])
        z = x @ c
        both = np.stack([log_expit(z) + ls, log_expit(-z) + ln])
        total = logsumexp(both, axis=0)
        r = np.exp(both[0] - total)                     # each row's chance of being the same word's
        grad = np.concatenate([(x * (r - expit(z))[:, None]).sum(0), (gg * r[:, None]).sum(0),
                               (gd * r[:, None]).sum(0), (gn_m * (1 - r)[:, None]).sum(0),
                               (gn_p * (1 - r)[:, None]).sum(0)])
        return -total.sum() + ridge * th @ th, -grad + 2 * ridge * th

    res = minimize(nll, start, jac=True, method="L-BFGS-B")
    th = res.x
    return {"same": th[:k], "mean": th[k:2 * k], "precision": th[2 * k:3 * k],
            "neighbour": {"mean": float(th[3 * k]), "precision": float(th[3 * k + 1])}}, float(res.fun)


def fit_doubt(features: pd.DataFrame, ridge: float = RIDGE) -> dict:
    """The doubt model (see the module) fitted on the rows of `features` shown twice (`doubt_features`' columns kind,
    ink_outside, line_share, axis, edge, read_twice, paired, agree, other_iou): P(paired) on all of them, the rest on
    the paired ones.  Rows may come only from the training quarters' words outside the answered blocks: the columns
    `held_out` and `answered` are REQUIRED (ValueError when missing) and a true one is refused.  Returns {"features",
    "paired", "agree", "one", "same", "mean", "precision" (each {feature: coefficient}), "neighbour": {"mean",
    "precision"} (logit and log), "rows", "paired_rows", "paired_share", "agree_share", "one_share", "nll"}."""
    for flag in ("held_out", "answered"):
        if flag not in features:
            raise ValueError(f"fit_doubt: rows need a `{flag}` column (False for a row that may be used)")
        if features[flag].astype(bool).any():
            raise ValueError(f"fit_doubt: rows flagged {flag} may not be used to fit")
    rows = features[features["read_twice"].astype(bool)]
    if not len(rows):
        raise ValueError("fit_doubt: no word shown twice to fit on")
    x = design(rows)
    paired = rows["paired"].to_numpy(bool)
    if not paired.any():
        raise ValueError("fit_doubt: no word shown twice was read twice")
    agree = rows["agree"].to_numpy(float)[paired]
    iou = rows["other_iou"].to_numpy(float)[paired]
    if not ((agree == 0) | (agree == 1)).all() or not ((iou > 0) & (iou <= 1)).all():
        raise ValueError("fit_doubt: a paired row needs agree 0 or 1 and an IoU in (0, 1]")
    b_paired, nll_paired = _logistic(x, paired.astype(float), ridge)
    xp = x[paired]
    b_agree, nll_agree = _logistic(xp, agree, ridge)
    one = (iou == 1).astype(float)
    b_one, nll_one = _logistic(xp, one, ridge)
    mix, nll_iou = _beta_mixture(xp[iou < 1], iou[iou < 1], ridge)
    names = list(FEATURES)

    def named(v):
        return dict(zip(names, map(float, v)))

    return {"features": names, "paired": named(b_paired), "agree": named(b_agree), "one": named(b_one),
            "same": named(mix["same"]), "mean": named(mix["mean"]), "precision": named(mix["precision"]),
            "neighbour": mix["neighbour"], "rows": int(len(rows)), "paired_rows": int(paired.sum()),
            "paired_share": float(paired.mean()), "agree_share": float(agree.mean()), "one_share": float(one.mean()),
            "nll": {"paired": nll_paired, "agree": nll_agree, "one": nll_one, "iou": nll_iou}}


# --- the weights -----------------------------------------------------------------------------------------------------

def _coef(params: dict, part: str) -> np.ndarray:
    if list(params["features"]) != list(FEATURES):
        raise ValueError(f"doubt: the parameters are for {params['features']}, not {list(FEATURES)}")
    return np.array([params[part][k] for k in FEATURES], float)


def weights_of(features: pd.DataFrame, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """(box, string) weights of each row of `features` by the fitted `params` (see the module), float64."""
    x = design(features)
    p_paired = expit(x @ _coef(params, "paired"))
    p_one = expit(x @ _coef(params, "one"))
    p_same = expit(x @ _coef(params, "same"))
    interior = p_same * expit(x @ _coef(params, "mean")) + (1 - p_same) * expit(params["neighbour"]["mean"])
    box = p_paired * (p_one + (1 - p_one) * interior)
    return box, p_paired * expit(x @ _coef(params, "agree"))


def doubt_weights(sd: data.SheetData | None, params: dict, features: pd.DataFrame | None = None,
                  records: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each Terra word's (box_weight, string_weight) on a sheet (see the module): its expected box IoU and P(string
    right), from its features (`features`, or `doubt_features(sd, records)` when None).  No cut, no flag."""
    if features is None:
        features = doubt_features(sd, records)
    return weights_of(features, params)


def prepare(sheets: list[data.SheetData], weights: Mapping, which: str = "box") -> list[data.SheetData]:
    """The sheets with each word's weight (`which`: "box", the default, or "string") in `SheetData.weights`
    (float32, in the order of its words), everything else shared (`dataclasses.replace`).  `weights`:
    {"<series>/<sheet>": {"words": `words_hash` of the sheet the weights were made for, "box": [...], "string":
    [...]}} (`load_weights`).  ValueError for a sheet with no weights, weights made for other words, or another
    number of weights than words."""
    out = []
    for sd in sheets:
        key = f"{sd.series}/{sd.sheet}"
        if key not in weights:
            raise ValueError(f"doubt: no weights for {key}")
        entry = weights[key]
        if entry.get("words") != words_hash(sd):
            raise ValueError(f"doubt: the weights of {key} were made for other words (words hash "
                             f"{entry.get('words')}, the sheet's {words_hash(sd)})")
        w = np.asarray(entry[which], np.float32)
        if w.shape != (len(sd.words),):
            raise ValueError(f"doubt: {key} has {len(sd.words)} words but {w.size} weights")
        out.append(dataclasses.replace(sd, weights=w))
    return out


def load_weights(fold: str, path=DOUBT_FILE) -> dict[str, dict]:
    """The fold's weights of every training sheet from DOUBT_FILE: {"<series>/<sheet>": {"words", "box", "string"}}
    (for `prepare`).  ValueError for a fold that is not one of `paths.QUARTERS` or not in the file."""
    if fold not in paths.QUARTERS:
        raise ValueError(f"doubt: fold {fold!r} is not one of {', '.join(paths.QUARTERS)}")
    folds = json.loads(path.read_text())["folds"]
    if fold not in folds:
        raise ValueError(f"doubt: {path} holds no fold {fold!r} (it holds {', '.join(folds)})")
    return {k: {"words": v["words"], "box": np.asarray(v["box"], np.float32),
                "string": np.asarray(v["string"], np.float32)} for k, v in folds[fold]["weights"].items()}


# --- measured on the training maps -----------------------------------------------------------------------------------

def _quantiles(v: np.ndarray) -> dict:
    return {str(q): round(float(np.quantile(v, q)), 4) for q in QUANTILES} if len(v) else {}


def counts(features: pd.DataFrame) -> dict:
    """The double reads among `features`' words: words, shown twice, paired, shown twice but read once, the paired
    words' agreement and IoU (quantiles; the share at 1), the features' quantiles, and the same per series and kind."""
    def one(f):
        p = f[f["paired"]]
        return {"words": int(len(f)), "read_twice": int(f["read_twice"].sum()), "paired": int(len(p)),
                "read_once_of_twice": int((f["read_twice"] & ~f["paired"]).sum()),
                "agree_share": round(float(p["agree"].mean()), 4) if len(p) else None,
                "iou_one_share": round(float((p["other_iou"] == 1).mean()), 4) if len(p) else None,
                "iou": _quantiles(p["other_iou"].to_numpy(float)),
                "forced_height": int(f["forced_height"].sum()), "no_ink": int((f["ink_px"] == 0).sum()),
                **{c: _quantiles(f[c].to_numpy(float)) for c in ("ink_outside", "line_share", "line_of_ink", "axis",
                                                                 "edge")}}
    return {"all": one(features), "series": {s: one(g) for s, g in features.groupby("series", sort=False)},
            "kind": {k: one(g) for k, g in features.groupby("kind", sort=False)}}


def fit_folds(feats: pd.DataFrame, hashes: Mapping[tuple[str, str], str]) -> dict:
    """Per fold of `paths.QUARTERS`: the fit on the words touching neither the fold's quarter (`touch_<q>`) nor an
    answered block (`touch_blocks`), and every word's weights by it (with its sheet's `hashes[(series, sheet)]`), the
    counts of the words fitted, and the weights' quantiles per series and kind, in and beyond the fit's edges."""
    folds = {}
    for q in paths.QUARTERS:
        free = (~feats[f"touch_{q}"].astype(bool) & ~feats["touch_blocks"].astype(bool)).to_numpy()
        params = fit_doubt(feats[free].assign(held_out=False, answered=False))
        box, string = weights_of(feats, params)
        weights = {f"{s}/{sh}": {"words": hashes[(s, sh)], "box": [round(float(v), 6) for v in box[i]],
                                 "string": [round(float(v), 6) for v in string[i]]}
                   for (s, sh), i in feats.groupby(["series", "sheet"], sort=False).indices.items()}
        dist: dict = {}
        for (s, k), i in feats.groupby(["series", "kind"], sort=False).indices.items():
            dist.setdefault(s, {})[k] = {"box": _quantiles(box[i]), "string": _quantiles(string[i]),
                                         "words": int(len(i))}
        fitted = feats[free & feats["read_twice"].to_numpy(bool)]
        beyond = (feats["edge"] > fitted["edge"].max()).to_numpy()
        support = {"fit_edge_max": float(fitted["edge"].max()), "words_beyond": int(beyond.sum()),
                   "box_beyond": _quantiles(box[beyond]), "box_within": _quantiles(box[~beyond]),
                   "string_beyond": _quantiles(string[beyond]), "string_within": _quantiles(string[~beyond])}
        folds[q] = {"params": params, "fit_counts": counts(feats[free]), "weights_by_series_kind": dist,
                    "edge_support": support,
                    "left_out": "words whose rectangle touches the fold's quarter or an answered block",
                    "weights": weights}
        log.info("fold %s: %d rows shown twice, %d paired; paired %s; agree %s", q, params["rows"],
                 params["paired_rows"], params["paired"], params["agree"])
    return folds


def write_doubt(out=DOUBT_FILE, features_out=FEATURES_FILE) -> dict:
    """Measure the training maps' words (their line px from `zones.LINES_DIR`), fit each fold (`fit_folds`), and
    write the features to `features_out` and the folds' parameters, weights and counts to `out` (folders made; see
    the module)."""
    from sfzmaps.solver.textnet import reader_stage, zones    # torch and the lots: only for this measurement

    blocks = reader_stage.held_blocks()
    areas = {**{f"touch_{q}": data.Quarter(i) for i, q in enumerate(paths.QUARTERS)},
             "touch_blocks": reader_stage.load_block_area(blocks)}
    records = boxes.terra_records()
    frames, hashes = [], {}
    for s in paths.TEXT_TRAIN_SERIES:
        for sd in data.load_sheets([s], hold_out_validation=False):
            sd = dataclasses.replace(sd, line_bits=zones.sheet_lines(sd, zones.LINES_DIR))
            frames.append(doubt_features(sd, records, areas))
            hashes[(sd.series, sd.sheet)] = words_hash(sd)
            log.info("%s %s: %d words, %d shown twice", sd.series, sd.sheet, len(frames[-1]),
                     int(frames[-1]["read_twice"].sum()))
    feats = pd.concat(frames, ignore_index=True)
    features_out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(features_out)
    got = {"series": list(paths.TEXT_TRAIN_SERIES), "words": int(len(feats)), "blocks_held": len(blocks),
           "counts": counts(feats), "folds": fit_folds(feats, hashes)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(got) + "\n")
    return got


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cv2.setNumThreads(1)
    data.LOAD_THREADS = 1          # one page decoded at a time: the machine is shared
    result = write_doubt()
    print(json.dumps(result["counts"]["all"], indent=1))
    for q, f in result["folds"].items():
        print(q, json.dumps(f["params"]))
