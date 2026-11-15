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

Candidate features (CANDIDATES, in the order they are tried; none by default, which is the set above): the five of
`wallevidence.OPTIONS` (text_weighted, text_weighted_beside_letters, street_crossing, no_city_limit,
text_discounted_enclosure), read through the detector's out-of-fold text chance (`textgrid.text_chance`, all four
quarters) and the merged page's ink, and
  context           (`context`, after the lots' character) touching pairs: hats over KIND_KNOTS of how the line
                    between the lots' centroids runs along their long axes (the mean |cos| with each lot's minimum
                    rotated rectangle's long side: 0 side by side, 0.5 between, 1 back to back); pairs across a
                    street: hats over STREET_WIDTH_KNOTS of the street's width (the distance between the two lots)
`evaluate_candidates` scores several configurations of them from inputs read once, and writes nothing.

Writes, under data/solver/walls/:
  walls_{q}.npz     lo_ab: float32 per pair row, the log odds from the fit without quarter q
  evaluation.json   per quarter: the log loss on the held-out pairs (both lots known and in q), their number, and the
                    training pairs' number and log loss; the number of pairs and features; the candidate features
                    used; the strokes touching text (where Terra's letter share is read) (no times: the same inputs
                    give the same bytes)
  text_chance.png   the detector's text chance on the grid that the fit read, when a candidate reads it
                    (`textgrid.write_text_chance`; an output only, never read back)
"""
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit

from sfzmaps.solver import lots, page, paths, wallevidence
from sfzmaps.solver.wallevidence import hats

log = logging.getLogger(__name__)

AREA_DIFFERENCE_KNOTS = np.array([0.0, 0.3, 0.7, 1.2, 2.0, 3.0, 5.0])   # |log area a - log area b|
SMALLER_AREA_KNOTS = np.array([5.0, 7.0, 7.8, 8.5, 9.5, 11.0, 13.0])     # the smaller lot's log area (sq ft)
KIND_KNOTS = np.array([0.0, 0.5, 1.0])      # touching lots: side by side, between, back to back
# Pairs across a street: the street's width, ft.  Knots at the peaks of the 2000 table's widths (length-weighted:
# 25, 35-40, 50, 60, 68.75, 80-82.5, 100, 120-125 ft), then the wide open ground of parks and water.
STREET_WIDTH_KNOTS = np.array([0.0, 25.0, 40.0, 50.0, 60.0, 68.75, 82.5, 100.0, 125.0, 200.0, 500.0])
RIDGE = 0.1
MAX_STEPS = 60
TOLERANCE = 1e-7
PROBABILITY_FLOOR = 1e-6    # the log loss counts a probability below this as this

CONTEXT = "context"
CANDIDATES = (wallevidence.TEXT_WEIGHTED, wallevidence.TEXT_WEIGHTED_BESIDE_LETTERS, wallevidence.STREET_CROSSING,
              CONTEXT, wallevidence.NO_CITY_LIMIT, wallevidence.TEXT_DISCOUNTED_ENCLOSURE)


def check_candidates(names) -> tuple[str, ...]:
    """The candidate features named, in CANDIDATES order; an unknown name, or both text weightings, is an error."""
    if isinstance(names, str):
        raise TypeError(f"a configuration is a list of candidate names, not the string {names!r}")
    named = set(names)
    unknown = sorted(named - set(CANDIDATES))
    if unknown:
        raise ValueError(f"no candidate feature {unknown[0]!r}: one of {', '.join(CANDIDATES)}")
    wallevidence.check_options(named - {CONTEXT})
    return tuple(c for c in CANDIDATES if c in named)


def design(F: np.ndarray, lots_: lots.Lots, log_area: np.ndarray, context_: np.ndarray | None = None) -> np.ndarray:
    """The fit's features per pair (float32): F, F times across a street, and the lots' character; then the
    `context` columns when given."""
    street = (lots_.touching == 0).astype(np.float32)[:, None]
    la, lb = log_area[lots_.a], log_area[lots_.b]
    difference = hats(np.abs(la - lb), AREA_DIFFERENCE_KNOTS)
    parts = [F, F * street, difference, difference * street, hats(np.minimum(la, lb), SMALLER_AREA_KNOTS)]
    if context_ is not None:
        parts.append(context_)
    return np.concatenate(parts, 1).astype(np.float32)


def long_axes(geometries) -> np.ndarray:
    """Each lot's long axis (n, 2), a unit vector: the longer side of its minimum rotated rectangle (the line itself
    for a lot with no area; 0 for a point or an empty lot)."""
    import shapely
    with np.errstate(divide="ignore", invalid="ignore"):     # it divides by zero along sides parallel to an axis
        rect = shapely.minimum_rotated_rectangle(np.asarray(geometries, dtype=object))
    out = np.zeros((len(rect), 2))
    for i, r in enumerate(rect):
        xy = shapely.get_coordinates(r)
        if len(xy) < 2:
            continue
        e1 = xy[1] - xy[0]
        e2 = xy[2] - xy[1] if len(xy) > 3 else e1
        e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        n = np.linalg.norm(e)
        if n > 0:
            out[i] = e / n
    return out


def context(geometries, lots_: lots.Lots) -> np.ndarray:
    """The context of every pair (float32 (pairs, len(KIND_KNOTS) + len(STREET_WIDTH_KNOTS))): for touching pairs,
    hats of the touching kind (the mean over the two lots of |cos| between the line joining their centroids and the
    lot's long axis); for pairs across a street, hats of the distance between the two lots (ft).  `geometries`: the
    lots' outlines in lot order (EPSG:2227 ft)."""
    import shapely
    geometries = np.asarray(geometries, dtype=object)
    a, b = lots_.a, lots_.b
    centre = lots.lot_centroids(geometries)
    if not np.isfinite(centre[np.r_[a, b]]).all():
        raise ValueError("a lot of a neighbouring pair has no centroid")
    axis = long_axes(geometries)
    v = centre[b] - centre[a]
    v /= np.maximum(np.linalg.norm(v, axis=1), 1e-9)[:, None]
    kind = 0.5 * (np.abs((v * axis[a]).sum(1)) + np.abs((v * axis[b]).sum(1)))
    touching = lots_.touching > 0
    width = np.zeros(len(a))
    width[~touching] = shapely.distance(geometries[a[~touching]], geometries[b[~touching]])
    return np.concatenate([hats(kind, KIND_KNOTS) * touching[:, None],
                           hats(width, STREET_WIDTH_KNOTS) * ~touching[:, None]], 1).astype(np.float32)


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


@dataclass
class Inputs:
    """What the walls read: the lots, the mapped area, the strokes, each lot's log area and outline, and as the
    candidates need them each stroke's Terra letter share, the detector's text chance and the merged page's ink."""
    lots: lots.Lots
    mapped: np.ndarray
    strokes: dict
    log_area: np.ndarray
    geometries: np.ndarray | None
    letter: np.ndarray | None
    text: np.ndarray | None
    ink: np.ndarray | None


def load_inputs(configs) -> Inputs:
    """Read, once, what the walls need to fit every configuration of `configs` (each a list of candidate names; see
    `Inputs`).  The text chance is computed from all four quarters' detectors (`textgrid.text_chance`)."""
    import geopandas as gpd
    import shapely

    from sfzmaps.solver import textgrid
    t0 = time.perf_counter()
    configs = [check_candidates(c) for c in configs]
    used = {c for config in configs for c in config}
    lots_ = lots.load_lots()
    mapped = page.load_sheets() > 0
    strokes_file = paths.LABELS_DIR / "strokes.npz"
    if not strokes_file.is_file():
        raise FileNotFoundError(f"can't read {strokes_file}: run `sfzmaps solver labels`")
    strokes = dict(np.load(strokes_file))
    letter = None
    if any(wallevidence.TEXT_WEIGHTED not in config for config in configs):      # Terra's letter share is read
        boxes = terra_text(lots_.lot.shape)
        letter = wallevidence.letter_share(strokes, boxes)
        del boxes
        log.info("letter share: %d of %d strokes touch text (%.0f s)", (letter > 0).sum(), len(letter),
                 time.perf_counter() - t0)
    text = textgrid.text_chance() if used & set(wallevidence.READS_TEXT) else None
    ink = page.load_ink() if wallevidence.STREET_CROSSING in used else None
    geometries = gpd.read_parquet(paths.LOTS_DIR / "table.parquet").geometry.values
    log_area = np.log(np.maximum(shapely.area(geometries), 1.0))
    return Inputs(lots_, mapped, strokes, log_area, geometries if CONTEXT in used else None, letter, text, ink)


def candidate_design(inputs: Inputs, candidates=(), workers: int = 1) -> np.ndarray:
    """The fit's features (`design`) of every pair under `candidates` (none: the set the module describes)."""
    candidates = check_candidates(candidates)
    options = [c for c in candidates if c != CONTEXT]
    F = wallevidence.border_features(inputs.lots, inputs.mapped, inputs.strokes, inputs.letter, workers=workers,
                                     options=options, text=inputs.text, ink=inputs.ink)
    extra = context(inputs.geometries, inputs.lots) if CONTEXT in candidates else None
    return design(F, inputs.lots, inputs.log_area, extra)


def evaluate_candidates(configs, workers: int | None = None) -> list[dict]:
    """Fit the walls under each configuration of `configs` (each a list of candidate names; [] is the default set)
    per held-out quarter, as `build_walls` fits them (the table as downloaded), from inputs read once.  Returns one
    evaluation per configuration, in order (per quarter the held-out log loss and pairs, the training log loss and
    pairs; the pairs, features and candidates).  Writes nothing."""
    configs = [check_candidates(c) for c in configs]
    if not configs:
        raise ValueError("no configuration to evaluate")
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    inputs = load_inputs(configs)
    out = []
    for candidates in configs:
        _, evaluation = fit_walls(candidate_design(inputs, candidates, workers), inputs.lots)
        out.append({**evaluation, "candidates": list(candidates)})
    return out


def build_walls(workers: int | None = None, candidates=()) -> dict:
    """Fit the wall odds with the candidate features `candidates` (none by default) and write walls_{q}.npz and
    evaluation.json into WALLS_DIR, and text_chance.png (`textgrid.write_text_chance`) when a candidate reads it.
    They are written into a fresh folder that then takes WALLS_DIR's place (`paths.fresh_folder`,
    `paths.replace_folder`), so nothing of an earlier fit is left.  `workers`: processes for the closing paths
    (default half the CPUs).  Returns the evaluation and the seconds taken (not written)."""
    from sfzmaps.solver import textgrid
    t0 = time.perf_counter()
    candidates = check_candidates(candidates)
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    inputs = load_inputs([candidates])
    X = candidate_design(inputs, candidates, workers)
    odds, evaluation = fit_walls(X, inputs.lots)
    evaluation["candidates"] = list(candidates)
    if inputs.letter is not None:
        evaluation["strokes_touching_text"] = int((inputs.letter > 0).sum())
    folder = paths.fresh_folder(paths.WALLS_DIR)
    write_walls(folder, odds, evaluation)
    if inputs.text is not None:
        textgrid.write_text_chance(folder, inputs.text)
    paths.replace_folder(folder, paths.WALLS_DIR)
    return {**evaluation, "seconds": round(time.perf_counter() - t0, 1)}

