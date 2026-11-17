"""Doubt weights for Terra's words: each word's probability of being right, so a wrong label counts less in training.

Some of Terra's word boxes are wrong: a word cut at a tile's edge and read again whole from the next tile, a box grown
over the next word, a near-45-degree word whose height `boxes.word_rect` had to force.  No word is cut or flagged
here: each gets two weights, fitted on the words Terra read twice.

Terra was shown each sheet in overlapping tiles (`boxes.terra_boxes`; 768 px tiles about 583 px apart), so a word in
the overlap of two tiles was shown twice, and each tile's answer read it on its own.  `terra_boxes` keeps one read
(the one farthest from its tile's edge) and drops a second whose box overlaps it by IoU > `boxes.SAME_TEXT_IOU`; the
second reads it drops are the ones used here, every tile's read (`tile_reads`) as Terra gave it.

The features of a Terra word (`doubt_features`, one row per word of a sheet, in the order of `SheetData.words`):
  ink_outside    the share of the px of its ink pieces (the 8-connected components of the page's ink that meet its
                 rectangle, `SheetData.words` as `data._paint` fills it) that lie outside its rectangle; a rectangle
                 holding no ink has all of its ink outside: 1
  forced_height  its height was forced by `boxes.word_rect`'s near-45-degree case (TYPICAL_HEIGHT)
  kind           Terra's kind
  read_twice     another tile's window holds its whole box: Terra was shown it twice
  other_*        for a word read twice, the other read: of the other tiles' reads (every one, before the drop), the
                 one whose rectangle overlaps the word's with the highest IoU (argmax; no IoU cut), its text, the
                 pair's IoU (`rect_iou` of the two reads' `word_rect`s, each at its own angle) and whether the two
                 strings agree once folded (upper case, zero-width spaces dropped, single spaces: `crops.norm_text`'s
                 folding); `paired` when such a read exists (a word shown twice may have been read once)
  touch_<name>   with `areas`, whether its rectangle holds a px of each area (the held-out quarters and the answered
                 blocks, as the reader's crops record them)

The fit (`fit_doubt`, on the training quarters' paired words outside the answered blocks; the rows say so in
`held_out` and `answered`, and a true one is refused), from x = (one intercept per kind, ink_outside, forced_height):
  agree   a logistic model of P(the two strings agree | x)
  IoU     a one-inflated beta regression of the pair's IoU (Ospina and Ferrari's inflated beta regression): P(IoU = 1 |
          x), two reads of one rectangle, by a logistic (`one`), and otherwise IoU ~ Beta(mu phi, (1 - mu) phi) with
          logit mu = x . `mean` and log phi = x . `precision` (the variable-dispersion beta regression).  The plain
          beta regression has no room for the point mass at 1 (two identical reads: 5% of the pairs); one precision
          for every word fitted the IoU of every held-out quarter worse (by 0.14-0.20 nats a pair) and put its mean
          0.03 below the observed.
Each by maximum likelihood (L-BFGS), with RIDGE on every coefficient only to keep a separable fit finite.

The weights (`doubt_weights`), per word, with no cut and no flag:
  box     the expected IoU of its box, P(IoU = 1 | x) + (1 - P(IoU = 1 | x)) mu(x): the word's weight in the
          detector's word terms (`data.SheetData.weights`, `prepare`)
  string  P(string right), taken as the fitted P(the two reads agree | x): for the reader's targets
Both are predictions from the word's features alone (its own pair, when it has one, is one of the fit's rows).

`write_doubt` measures the training maps' words (`paths.TEXT_TRAIN_SERIES`), fits once per fold of the reader's split
(`paths.QUARTERS`: the words touching the fold's quarter or an answered block left out of its fit) and writes the
features (FEATURES_FILE) and, per fold, the fitted parameters, every word's weights and the counts (DOUBT_FILE).

Neither `textnet.data` nor `textnet.train` imports this module: the weights reach the detector in
`data.SheetData.weights`, filled by `prepare` before the training.  Nothing here asks a model.
"""
import dataclasses
import json
import logging
from collections.abc import Mapping

import cv2
import numpy as np
import pandas as pd
import shapely
from scipy.optimize import minimize
from scipy.special import digamma, expit, gammaln, log_expit

from sfzmaps.solver import paths
from sfzmaps.solver.textnet import boxes, data

log = logging.getLogger(__name__)

DOUBT_FILE = paths.SOLVER_DIR / "experiments" / "wordread" / "detector" / "doubt.json"
FEATURES_FILE = DOUBT_FILE.with_name("doubt_features.parquet")
KIND_NAMES = tuple(data.KINDS)                 # Terra's kinds, in the order of their KINDS index
FEATURES = (*(f"kind={k}" for k in KIND_NAMES), "ink_outside", "forced_height")
NEAR_DIAGONAL = 0.25    # `boxes.word_rect`'s: |cos^2 - sin^2| of the angle at or below it forces the height (its code,
                        # not a choice here; copied, as boxes.py is in the prints stage's closure; a test pins it)
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


def forced_height(angle_deg) -> np.ndarray:
    """Whether `boxes.word_rect` forces a word's height at reading angle `angle_deg` (degrees; any shape): its
    near-45-degree case, where the box can't give the height."""
    a = np.deg2rad(np.asarray(angle_deg, float))
    c, s = np.abs(np.cos(a)), np.abs(np.sin(a))
    return ~(np.abs(c * c - s * s) > NEAR_DIAGONAL)


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
    the one whose rectangle overlaps its own with the highest IoU (None when none overlaps)."""
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
    index), text, kind, edge (Terra's distance from its tile's edge, reported only), ink_outside, ink_px (its ink px
    inside), forced_height, read_twice, other_tiles, paired, other_text, other_iou, agree (1.0 / 0.0; NaN unpaired),
    and a bool column per entry of `areas` ({column name: data.MapRect / Quarter / GridArea}; needs the sheet's
    placement).  `records`: the lines of TERRA_READ (read from disk when None).  ValueError when the sheet's words
    are not Terra's words of the sheet (`boxes.terra_boxes`, as `data.sheet_data` reads them)."""
    recs = _sheet_records(sd.series, sd.sheet, records)
    kept = boxes.terra_boxes(sd.series, sd.sheet, recs)
    rects = np.round(np.array([_rect(k) for k in kept], float).reshape(-1, 4, 2)).astype(np.int32)
    if rects.shape != sd.words.shape or not np.array_equal(rects, sd.words):
        raise ValueError(f"{sd.series} {sd.sheet}: the sheet's {len(sd.words)} words are not Terra's words of the "
                         f"sheet ({len(kept)}, boxes.terra_boxes)")
    areas = dict(areas or {})
    _, labels, stats, _ = cv2.connectedComponentsWithStats(data.page_ink(sd).view(np.uint8), connectivity=8,
                                                               ltype=cv2.CV_32S)
    size = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    rows = []
    for j, (k, other) in enumerate(zip(kept, _other_reads(kept, tile_reads(sd.series, sd.sheet, recs)))):
        win = _window(sd.words[j], sd)
        inside = total = 0
        touch = dict.fromkeys(areas, False)
        if win is not None:
            x0, y0, w, h = win
            mask = data._paint(sd.words[j:j + 1], np.ones(1, np.int32), y0, x0, h, w) > 0
            lab = labels[y0:y0 + h, x0:x0 + w][mask]
            ids, counts = np.unique(lab[lab > 0], return_counts=True)
            inside, total = int(counts.sum()), int(size[ids].sum())
            touch = {name: bool(data.in_area(sd, area, y0, x0, h, w)[mask].any()) for name, area in areas.items()}
        o = other["other"]
        rows.append({"series": sd.series, "sheet": sd.sheet, "word": j, "text": k["text"], "kind": k["kind"],
                     "edge": float(k["edge"]), "ink_outside": 1.0 - inside / total if total else 1.0,
                     "ink_px": inside, "forced_height": bool(forced_height(float(k.get("angle_deg") or 0.0))),
                     "read_twice": other["other_tiles"] > 0, "other_tiles": other["other_tiles"],
                     "paired": o is not None, "other_text": None if o is None else o["text"],
                     "other_iou": np.nan if o is None else other["iou"],
                     "agree": np.nan if o is None else float(_norm(o["text"]) == _norm(k["text"])), **touch})
    cols = ["series", "sheet", "word", "text", "kind", "edge", "ink_outside", "ink_px", "forced_height", "read_twice",
            "other_tiles", "paired", "other_text", "other_iou", "agree", *areas]
    return pd.DataFrame(rows, columns=cols)


def design(features: pd.DataFrame) -> np.ndarray:
    """The fit's x per row (FEATURES): one intercept per Terra kind, ink_outside and forced_height.  ValueError for an
    unknown kind or a feature that is not a number."""
    kind = features["kind"].to_numpy()
    unknown = sorted(set(kind) - set(KIND_NAMES))
    if unknown:
        raise ValueError(f"doubt: kinds {unknown} are not Terra's ({', '.join(KIND_NAMES)})")
    x = np.column_stack([*(kind == k for k in KIND_NAMES), features["ink_outside"].to_numpy(float),
                         features["forced_height"].to_numpy(bool)]).astype(float)
    if not np.isfinite(x).all():
        raise ValueError("doubt: ink_outside must be a number for every word")
    return x


# --- the fit ---------------------------------------------------------------------------------------------------------

def _logistic(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    """Maximum-likelihood logistic coefficients of y (0/1) on x, RIDGE on each."""
    def nll(b):
        z = x @ b
        p = expit(z)
        f = -(y * log_expit(z) + (1 - y) * log_expit(-z)).sum() + ridge * b @ b
        return f, x.T @ (p - y) + 2 * ridge * b

    res = minimize(nll, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    return res.x, float(res.fun)


def _beta(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Maximum-likelihood beta regression of y in (0, 1) on x, its mean and its precision each modelled (logit mu =
    x . g, log phi = x . d: the variable-dispersion beta regression), RIDGE on each: g, d and the negative
    log-likelihood."""
    ly, l1y = np.log(y), np.log1p(-y)
    k = x.shape[1]

    def nll(th):
        g, d = th[:k], th[k:]
        mu = expit(x @ g)
        phi = np.exp(x @ d)
        a, b = mu * phi, (1 - mu) * phi
        f = -(gammaln(phi) - gammaln(a) - gammaln(b) + (a - 1) * ly + (b - 1) * l1y).sum() + ridge * th @ th
        da, db = digamma(a), digamma(b)
        # the log-likelihood's derivatives by mu and phi, through a = mu phi and b = (1 - mu) phi
        dl_dmu = phi * (-da + db + ly - l1y)
        dl_dphi = digamma(phi) - mu * da - (1 - mu) * db + mu * ly + (1 - mu) * l1y
        grad = np.concatenate([-(x.T @ (dl_dmu * mu * (1 - mu))), -(x.T @ (dl_dphi * phi))])
        return f, grad + 2 * ridge * th

    res = minimize(nll, np.zeros(2 * k), jac=True, method="L-BFGS-B")
    return res.x[:k], res.x[k:], float(res.fun)


def fit_doubt(features: pd.DataFrame, ridge: float = RIDGE) -> dict:
    """The doubt model (see the module) fitted on the paired rows of `features` (`doubt_features`' columns kind,
    ink_outside, forced_height, paired, agree, other_iou).  Rows may come only from the training quarters' words
    outside the answered blocks: the columns `held_out` and `answered` are REQUIRED (ValueError when missing) and a
    true one is refused.  Returns {"features", "agree", "one", "mean", "precision" (each {feature: coefficient}),
    "rows", "agree_share", "one_share", "nll": {"agree", "one", "mean"}}."""
    for flag in ("held_out", "answered"):
        if flag not in features:
            raise ValueError(f"fit_doubt: rows need a `{flag}` column (False for a row that may be used)")
        if features[flag].astype(bool).any():
            raise ValueError(f"fit_doubt: rows flagged {flag} may not be used to fit")
    rows = features[features["paired"].astype(bool)]
    if not len(rows):
        raise ValueError("fit_doubt: no word read twice with a pair to fit on")
    x = design(rows)
    agree = rows["agree"].to_numpy(float)
    iou = rows["other_iou"].to_numpy(float)
    if not ((agree == 0) | (agree == 1)).all() or not ((iou > 0) & (iou <= 1)).all():
        raise ValueError("fit_doubt: a paired row needs agree 0 or 1 and an IoU in (0, 1]")
    b_agree, nll_agree = _logistic(x, agree, ridge)
    one = (iou == 1).astype(float)
    b_one, nll_one = _logistic(x, one, ridge)
    part = iou < 1
    g, d, nll_mean = _beta(x[part], iou[part], ridge)
    names = list(FEATURES)
    return {"features": names, "agree": dict(zip(names, map(float, b_agree))),
            "one": dict(zip(names, map(float, b_one))), "mean": dict(zip(names, map(float, g))),
            "precision": dict(zip(names, map(float, d))), "rows": int(len(rows)), "agree_share": float(agree.mean()),
            "one_share": float(one.mean()), "nll": {"agree": nll_agree, "one": nll_one, "mean": nll_mean}}


# --- the weights -----------------------------------------------------------------------------------------------------

def _coef(params: dict, part: str) -> np.ndarray:
    if list(params["features"]) != list(FEATURES):
        raise ValueError(f"doubt: the parameters are for {params['features']}, not {list(FEATURES)}")
    return np.array([params[part][k] for k in FEATURES], float)


def weights_of(features: pd.DataFrame, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """(box, string) weights of each row of `features` by the fitted `params` (see the module), float64."""
    x = design(features)
    p_one = expit(x @ _coef(params, "one"))
    box = p_one + (1 - p_one) * expit(x @ _coef(params, "mean"))
    return box, expit(x @ _coef(params, "agree"))


def doubt_weights(sd: data.SheetData | None, params: dict, features: pd.DataFrame | None = None,
                  records: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each Terra word's (box_weight, string_weight) on a sheet (see the module): its expected box IoU and P(string
    right), from its features (`features`, or `doubt_features(sd, records)` when None).  No cut, no flag."""
    if features is None:
        features = doubt_features(sd, records)
    return weights_of(features, params)


def prepare(sheets: list[data.SheetData], weights: Mapping) -> list[data.SheetData]:
    """The sheets with each word's box weight in `SheetData.weights` (float32, in the order of its words), everything
    else shared (`dataclasses.replace`).  `weights`: {"<series>/<sheet>": the sheet's box weights}.  ValueError for a
    sheet with no weights or with another number of them than words."""
    out = []
    for sd in sheets:
        key = f"{sd.series}/{sd.sheet}"
        if key not in weights:
            raise ValueError(f"doubt: no weights for {key}")
        w = np.asarray(weights[key], np.float32)
        if w.shape != (len(sd.words),):
            raise ValueError(f"doubt: {key} has {len(sd.words)} words but {w.size} weights")
        out.append(dataclasses.replace(sd, weights=w))
    return out


def load_weights(fold: str, path=DOUBT_FILE, which: str = "box") -> dict[str, np.ndarray]:
    """The fold's weights of every training sheet from DOUBT_FILE ({"<series>/<sheet>": float32}), the box weights
    (for `prepare`) or the string weights (`which` "string")."""
    got = json.loads(path.read_text())["folds"][fold]["weights"]
    return {k: np.asarray(v[which], np.float32) for k, v in got.items()}


# --- measured on the training maps -----------------------------------------------------------------------------------

def _quantiles(v: np.ndarray) -> dict:
    return {str(q): round(float(np.quantile(v, q)), 4) for q in QUANTILES} if len(v) else {}


def counts(features: pd.DataFrame) -> dict:
    """The double reads among `features`' words: words, read twice, paired, shown twice but read once, the paired
    words' agreement and IoU (quantiles; the share at 1), and the same per series and per kind."""
    def one(f):
        p = f[f["paired"]]
        return {"words": int(len(f)), "read_twice": int(f["read_twice"].sum()), "paired": int(len(p)),
                "read_once_of_twice": int((f["read_twice"] & ~f["paired"]).sum()),
                "agree_share": round(float(p["agree"].mean()), 4) if len(p) else None,
                "iou_one_share": round(float((p["other_iou"] == 1).mean()), 4) if len(p) else None,
                "iou": _quantiles(p["other_iou"].to_numpy(float)),
                "forced_height": int(f["forced_height"].sum()), "no_ink": int((f["ink_px"] == 0).sum()),
                "ink_outside": _quantiles(f["ink_outside"].to_numpy(float))}
    return {"all": one(features), "series": {s: one(g) for s, g in features.groupby("series", sort=False)},
            "kind": {k: one(g) for k, g in features.groupby("kind", sort=False)}}


def write_doubt(out=DOUBT_FILE, features_out=FEATURES_FILE) -> dict:
    """Measure the training maps' words, fit each fold, and write the features to `features_out` and the folds'
    parameters, weights and counts to `out` (folders made; see the module)."""
    from sfzmaps.solver.textnet import reader_stage    # torch and the lots: only for this measurement

    blocks = reader_stage.held_blocks()
    areas = {**{f"touch_{q}": data.Quarter(i) for i, q in enumerate(paths.QUARTERS)},
             "touch_blocks": reader_stage.load_block_area(blocks)}
    records = boxes.terra_records()
    frames = []
    for s in paths.TEXT_TRAIN_SERIES:
        for sd in data.load_sheets([s], hold_out_validation=False):
            frames.append(doubt_features(sd, records, areas))
            log.info("%s %s: %d words, %d read twice", sd.series, sd.sheet, len(frames[-1]),
                     int(frames[-1]["read_twice"].sum()))
    feats = pd.concat(frames, ignore_index=True)
    features_out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(features_out)
    folds = {}
    for q in paths.QUARTERS:
        free = ~feats[f"touch_{q}"] & ~feats["touch_blocks"]
        rows = feats[free].assign(held_out=False, answered=False)
        params = fit_doubt(rows)
        box, string = weights_of(feats, params)
        weights = {f"{s}/{sh}": {"box": [round(float(v), 6) for v in box[i]],
                                 "string": [round(float(v), 6) for v in string[i]]}
                   for (s, sh), i in feats.groupby(["series", "sheet"], sort=False).indices.items()}
        dist: dict = {}
        for (s, k), i in feats.groupby(["series", "kind"], sort=False).indices.items():
            dist.setdefault(s, {})[k] = {"box": _quantiles(box[i]), "string": _quantiles(string[i]),
                                         "words": int(len(i))}
        folds[q] = {"params": params, "fit_counts": counts(feats[free]), "weights_by_series_kind": dist,
                    "left_out": "words whose rectangle touches the fold's quarter or an answered block",
                    "weights": weights}
        log.info("fold %s: %d paired rows; agree %s; one %s; mean %s; precision %s", q, params["rows"],
                 params["agree"], params["one"], params["mean"], params["precision"])
    got = {"series": list(paths.TEXT_TRAIN_SERIES), "words": int(len(feats)), "blocks_held": len(blocks),
           "counts": counts(feats), "folds": folds}
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
        print(q, json.dumps({k: v for k, v in f["params"].items()}, indent=None))
