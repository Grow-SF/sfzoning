"""The wall odds (`sfzmaps solver walls`): per pair of neighbouring lots (the rows of lots.npz `pairs`), the log odds
that a printed district line runs between them, fitted held out by quarter.

The features (`design`): the border evidence of `wallevidence` (F), F again times "the pair faces across a street"
(no touching px), and the lots' character: hats of the difference of their log areas (AREA_DIFFERENCE_KNOTS), the
same times across a street, and hats of the smaller log area (SMALLER_AREA_KNOTS; areas in square feet).  The
letter share of each stroke comes from Terra's text boxes (`prints/terra_boxes.json`, every kind), each drawn as its
word's rectangle on the sheet's page, kept when its centre is inside the sheet's crop (`page/frames.json`) and in no
cut box, and put on the grid through the sheet's corrected placement.

One ridge logistic regression per quarter q (`fit_quarter`: Newton's method from zero, RIDGE on every weight, at
most MAX_STEPS steps, stopping when no weight moves TOLERANCE).  Its target is that the two lots' codes in the
table as downloaded differ, and it trains only on pairs with both lots known and both outside q.  Every pair gets
the odds of every fit; the solve reads quarter q's.  No whole-wall stage: how a line holds together along its length
is left to the solve's energy.

Writes, under data/solver/walls/:
  walls_{q}.npz     lo_ab: float32 per pair row, the log odds from the fit without quarter q
  evaluation.json   per quarter: the log loss on the held-out pairs (both lots known and in q), their number, and the
                    training pairs' number and log loss; the number of pairs and features; the seconds
"""
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from sfzmaps.solver import lots, page, paths, wallevidence
from sfzmaps.solver.wallevidence import hats

log = logging.getLogger(__name__)

AREA_DIFFERENCE_KNOTS = np.array([0.0, 0.3, 0.7, 1.2, 2.0, 3.0, 5.0])   # |log area a - log area b|
SMALLER_AREA_KNOTS = np.array([5.0, 7.0, 7.8, 8.5, 9.5, 11.0, 13.0])     # the smaller lot's log area (sq ft)
RIDGE = 0.1
MAX_STEPS = 60
TOLERANCE = 1e-7
PROBABILITY_FLOOR = 1e-6    # the log loss counts a probability below this as this


def design(F: np.ndarray, lots_: lots.Lots, log_area: np.ndarray) -> np.ndarray:
    """The fit's features per pair (float32): F, F times across a street, and the lots' character."""
    street = (lots_.touching == 0).astype(np.float32)[:, None]
    la, lb = log_area[lots_.a], log_area[lots_.b]
    difference = hats(np.abs(la - lb), AREA_DIFFERENCE_KNOTS)
    return np.concatenate([F, F * street, difference, difference * street,
                           hats(np.minimum(la, lb), SMALLER_AREA_KNOTS)], 1).astype(np.float32)


def table_lines(lots_: lots.Lots) -> np.ndarray:
    """1.0 where the two lots' codes in the table as downloaded differ, else 0.0 (meaningful where both are known)."""
    return (lots_.codes[lots_.a] != lots_.codes[lots_.b]).astype(float)


def training_pairs(lots_: lots.Lots, q: int) -> np.ndarray:
    """The pairs quarter q's fit learns from: both lots known and both outside q (by each lot's quarter)."""
    return (lots_.known[lots_.a] & lots_.known[lots_.b]
            & (lots_.quarter[lots_.a] != q) & (lots_.quarter[lots_.b] != q))


def held_out_pairs(lots_: lots.Lots, q: int) -> np.ndarray:
    """The pairs quarter q's fit is scored on: both lots known and both in q."""
    return (lots_.known[lots_.a] & lots_.known[lots_.b]
            & (lots_.quarter[lots_.a] == q) & (lots_.quarter[lots_.b] == q))


def fit_quarter(F: np.ndarray, y: np.ndarray, train_mask: np.ndarray, ridge: float = RIDGE) -> np.ndarray:
    """Ridge logistic regression of y on F's rows in `train_mask`: the weights (no separate intercept; the hats sum
    to one).  Newton's method from zero."""
    X = np.asarray(F[train_mask], float)
    t = np.asarray(y, float)[train_mask]
    w = np.zeros(X.shape[1])
    eye = ridge * np.eye(X.shape[1])
    for _ in range(MAX_STEPS):
        p = expit(X @ w)
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + eye, X.T @ (t - p) - ridge * w)
        w += step
        if np.abs(step).max() < TOLERANCE:
            break
    return w


def log_loss(lo: np.ndarray, y: np.ndarray) -> float:
    """The mean log loss of log odds `lo` against 0/1 `y`."""
    p = expit(np.asarray(lo, float))
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(np.clip(p, PROBABILITY_FLOOR, 1))
                          + (1 - y) * np.log(np.clip(1 - p, PROBABILITY_FLOOR, 1))))


def fit_walls(X: np.ndarray, lots_: lots.Lots) -> tuple[dict[str, np.ndarray], dict]:
    """One fit per quarter.  Returns each quarter's odds for every pair (float32) and the evaluation."""
    y = table_lines(lots_)
    odds, quarters = {}, {}
    for qi, q in enumerate(paths.QUARTERS):
        train, held = training_pairs(lots_, qi), held_out_pairs(lots_, qi)
        lo = X @ fit_quarter(X, y, train)
        odds[q] = lo.astype(np.float32)
        quarters[q] = {"held_out_log_loss": round(log_loss(lo[held], y[held]), 6), "held_out_pairs": int(held.sum()),
                       "training_log_loss": round(log_loss(lo[train], y[train]), 6),
                       "training_pairs": int(train.sum())}
        log.info("%s: held-out log loss %.4f on %d pairs", q, quarters[q]["held_out_log_loss"], held.sum())
    return odds, {"pairs": len(y), "features": int(X.shape[1]), "quarters": quarters}


def write_walls(folder: Path, odds: dict[str, np.ndarray], evaluation: dict) -> None:
    """Write walls_{q}.npz (lo_ab) for each quarter and evaluation.json into `folder`."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for q in paths.QUARTERS:
        np.savez_compressed(folder / f"walls_{q}.npz", lo_ab=odds[q].astype(np.float32))
    (folder / "evaluation.json").write_text(json.dumps(evaluation, indent=1) + "\n")


def terra_text(shape) -> np.ndarray:
    """Terra's text boxes on the grid (bool, `shape` from the grid's top-left corner): every sheet's boxes
    (`wallevidence.text_rings`) through its corrected placement."""
    from sfzmaps import config
    from sfzmaps.transform import SheetTransform
    f = paths.PRINTS_DIR / "terra_boxes.json"
    if not f.is_file():
        raise FileNotFoundError(f"can't read {f}: run `sfzmaps solver prints`")
    boxes = json.loads(f.read_text())
    frames = json.loads((paths.PAGE_DIR / "frames.json").read_text())
    cuts = page.load_cuts()
    rings = []
    for name in paths.sheet_names():
        if name not in frames:
            raise KeyError(f"{paths.PAGE_DIR / 'frames.json'} has no frame for {name}: run `sfzmaps solver page`")
        folder = config.DATA_DIR / "warp" / paths.SERIES / name
        tr = SheetTransform.from_json(json.loads((folder / "transform.json").read_text()), folder=folder)
        kept = wallevidence.text_rings(boxes.get(name, []), frames[name], cuts.get(name, []))
        rings += [tr.to_world(r) for r in kept]
    return wallevidence.text_raster(rings, shape)


def build_walls(workers: int | None = None) -> dict:
    """Fit the wall odds and write walls_{q}.npz and evaluation.json into WALLS_DIR.  `workers`: processes for the
    closing paths (default half the CPUs).  Returns the evaluation."""
    import geopandas as gpd
    import shapely
    t0 = time.perf_counter()
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    lots_ = lots.load_lots()
    mapped = page.load_sheets() > 0
    strokes_file = paths.LABELS_DIR / "strokes.npz"
    if not strokes_file.is_file():
        raise FileNotFoundError(f"can't read {strokes_file}: run `sfzmaps solver labels`")
    strokes = dict(np.load(strokes_file))
    text = terra_text(lots_.lot.shape)
    letter = wallevidence.letter_share(strokes, text)
    log.info("letter share: %d of %d strokes touch text (%.0f s)", (letter > 0).sum(), len(letter),
             time.perf_counter() - t0)
    del text
    F = wallevidence.border_features(lots_, mapped, strokes, letter, workers=workers)
    table = gpd.read_parquet(paths.LOTS_DIR / "table.parquet")
    log_area = np.log(np.maximum(shapely.area(table.geometry.values), 1.0))
    X = design(F, lots_, log_area)
    del F
    odds, evaluation = fit_walls(X, lots_)
    evaluation["strokes_touching_text"] = int((letter > 0).sum())
    evaluation["seconds"] = round(time.perf_counter() - t0, 1)
    write_walls(paths.WALLS_DIR, odds, evaluation)
    return evaluation

