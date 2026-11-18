"""The code reader's text detectors, as their build stage makes them (`sfzmaps solver text-reader`, stage
`solver-text-reader`): four detectors, one per quarter of the city (`paths.QUARTERS`, by the solver's split,
`grid.quarter_of`), each trained by the detector stage's recipe (`stage.DEFAULT_RECIPE`, seed 0) on every sheet Terra
read of the training maps (`paths.TEXT_TRAIN_SERIES`) with its quarter held out, and the answered blocks held out of
all four; then every 2000 sheet's page (`paths.SHEETS`) scored, each px by the detector whose held-out quarter holds
it.  So the code reader reads each label with outputs of a detector that never saw its quarter, and no detector has
trained on an answered block.  The `solver-text` stage's detector, trained on every px, is not used by the reader.

Held out by map area, through each px's map point (`data.SheetData.held`, as a trial holds the validation area out),
on every training map at once (the 2000 drawing's three scans are one drawing, and so are 1964 and 1968):
  the quarter        `data.Quarter(q)`
  the answered blocks  every block the owner's answers reach (`codereader.labels.held_out_blocks` of every label: the
                     blocks of the answered labels' lots and of the lots under the answered cases' own points), as
                     the grid px whose nearest lot (lots.npz `grown`, as a label's lot is found) is in one of them
                     (`block_area`)

The unknown radius (`measure_radius`): around each of Sol's label points the whole word a label may print is taught
neither as text nor as not-text (`data.SheetData.point_radius`).  The point may sit anywhere on its word, and the word
may run in any direction, so the disc reaches the longest printed code word: the longest of the training maps' Terra
words of kind "zoning code" (each the word's rectangle as the detector learns it, `data.SheetData.words`), the words
whose centre is in an answered block left out; in page px, rounded up.  One radius for every map.

Writes data/solver/text-reader/ (`paths.READER_TEXT_DIR`):
  models/<q>.pt          each quarter's detector (`train.save_model`: the weights, the recipe, the device, the seed)
  <sheet>/chance.png     the chance of text x 255, uint8 (`predict.outputs`), each px from the detector of its quarter
  <sheet>/kind.npz       `kind` the most likely kind (uint8, `data.KINDS`; 255 on a tile with no ink, never scored)
                         and `p` the four kinds' probabilities (float16, (4, h, w))
  <sheet>/direction.npz  `direction`: the reading direction's cosine and sine (float16, (2, h, w), unit length)
  held.json              the quarters and their split, the answered blocks, the unknown radius with its measurement,
                         the recipe and the seed
  scores.json            each quarter's detector on its own quarter of the 2000 sheets (`quarter_scores`)
built in a fresh folder beside it and renamed into its place (`runs.replace_folder`).  Nothing in it is a time, and
the .npz files are written with a fixed date (`save_npz`): the same inputs give the same bytes.

The labels, the lots and the sheets (with their placements) are read first; then the four trainings and the scoring
run in one exclusive turn on the Mac's GPU (`sfzmaps.gpu`).  On the CPU there is no turn.
"""
import dataclasses
import gc
import json
import logging
import math
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from sfzmaps import gpu
from sfzmaps.solver import grid, paths, runs
from sfzmaps.solver.codereader import labels
from sfzmaps.solver.textnet import boxes, data, predict, scores, stage, train

log = logging.getLogger(__name__)

SEED = 0
MODELS_DIR = "models"
CHANCE_FILE, KIND_FILE, DIRECTION_FILE = "chance.png", "kind.npz", "direction.npz"
HELD_FILE, SCORES_FILE = "held.json", "scores.json"
NPZ_DATE = (1980, 1, 1, 0, 0, 0)     # the date every member of an .npz is written with (the earliest zip allows)
RADIUS_QUANTILES = (0.5, 0.9, 0.99)  # of the code words' lengths, recorded beside the longest


# --- what is held out -----------------------------------------------------------------------------------------------

def held_blocks() -> list[str]:
    """The answered blocks: `labels.held_out_blocks` of every label (`labels.load_labels`, without crops), sorted."""
    return sorted(labels.held_out_blocks(labels.load_labels(crops=False)))


def block_area(blocks, lot_block: np.ndarray, grown: np.ndarray, col0: int = 0, row0: int = 0) -> data.GridArea:
    """The grid px whose nearest lot is in one of `blocks`, as a `data.GridArea` cut to the rows and columns holding
    any.  `grown`: each grid px's nearest lot (-1 for none) over a window of the grid whose top-left px is (col0,
    row0); `lot_block`: each lot's block, by lot index."""
    held_lot = np.isin(np.asarray(lot_block).astype(str), sorted(blocks))
    mask = np.where(grown >= 0, held_lot[np.maximum(grown, 0)], False)
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return data.GridArea(np.zeros((0, 0), bool), col0, row0)
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return data.GridArea(np.ascontiguousarray(mask[r0:r1, c0:c1]), col0 + c0, row0 + r0)


def load_block_area(blocks) -> data.GridArea:
    """`block_area` of `blocks` over the whole grid, from the lots stage's output (lots.npz `grown`, and each lot's
    block number from table.parquet)."""
    lot_block = pd.read_parquet(paths.LOTS_DIR / "table.parquet", columns=["block_num"])["block_num"].astype(str)
    with np.load(paths.LOTS_DIR / "lots.npz") as z:
        grown = z["grown"]
    return block_area(blocks, lot_block.to_numpy(), grown)


def measure_radius(sheets: list[data.SheetData], areas=()) -> dict:
    """The unknown radius (see the module) from `sheets`, the words whose centre px (their rectangle's corners' mean,
    rounded, kept on the page) is in one of `areas` left out: `px` the radius; `longest` the word that sets it (its
    series, sheet, Terra's text, length in px and centre); `words` the code words measured and `left_out` those left
    out; `quantiles` of the measured lengths (RADIUS_QUANTILES).  ValueError when no code word is measured."""
    code = data.KINDS["zoning code"]
    found = []          # (length, sheet index, word index, centre)
    left_out = 0
    for i, sd in enumerate(sheets):
        for j in np.flatnonzero(sd.kinds == code):
            rect = sd.words[j].astype(float)
            x, y = np.clip(np.rint(rect.mean(axis=0)).astype(int), 0, [sd.width - 1, sd.height - 1])
            if any(data.in_area(sd, a, int(y), int(x), 1, 1)[0, 0] for a in areas):
                left_out += 1
                continue
            found.append((float(np.hypot(*(rect[1] - rect[0]))), i, int(j), (int(x), int(y))))
    if not found:
        raise ValueError("no Terra word of kind zoning code to measure the unknown radius by")
    length, i, j, centre = max(found, key=lambda f: f[0])
    sd = sheets[i]
    text = boxes.terra_boxes(sd.series, sd.sheet)[j]["text"]      # the sheet's words are Terra's boxes, in order
    lengths = np.array([f[0] for f in found])
    return {"px": math.ceil(round(length, 6)),
            "longest": {"series": sd.series, "sheet": sd.sheet, "text": text, "length_px": round(length, 3),
                        "centre": list(centre)},
            "words": len(found), "left_out": left_out,
            "quantiles": {str(q): round(float(np.quantile(lengths, q)), 3) for q in RADIUS_QUANTILES}}


# --- the outputs on the 2000 sheets ----------------------------------------------------------------------------------

def page_quarters(sd: data.SheetData) -> np.ndarray:
    """The quarter of every px of a sheet's page by its map point (int8, `data.Quarter`), data.BAND rows at a time."""
    out = np.full((sd.height, sd.width), -1, np.int8)
    for y0 in range(0, sd.height, data.BAND):
        h = min(data.BAND, sd.height - y0)
        band = out[y0:y0 + h]
        for q in range(len(paths.QUARTERS)):
            band[data.in_area(sd, data.Quarter(q), y0, 0, h, sd.width)] = q
    if (out < 0).any():
        raise ValueError(f"{sd.series} {sd.sheet}: px in no quarter")
    return out


def sheet_outputs(nets: list, ink: np.ndarray, quarter: np.ndarray, device: str) -> predict.Outputs:
    """A page's outputs (`predict.outputs`), each px's from `nets[q]` for its quarter q (`page_quarters`): each net
    scores only the tiles holding px of its quarter, and gives those px the numbers a whole page's scoring gives."""
    h, w = ink.shape
    out = predict.Outputs(np.zeros((h, w), np.uint8), np.full((h, w), predict.NO_KIND, np.uint8),
                          np.zeros((4, h, w), np.float16), np.zeros((2, h, w), np.float16))
    for q, net in enumerate(nets):
        mine = quarter == q
        if not mine.any():
            continue
        got = predict.outputs(net, ink, device, only=mine)
        out.chance[mine] = got.chance[mine]
        out.kind[mine] = got.kind[mine]
        out.kind_p[:, mine] = got.kind_p[:, mine]
        out.direction[:, mine] = got.direction[:, mine]
    return out


def save_npz(f: Path, **arrays) -> None:
    """`np.savez_compressed`'s file, its members dated NPZ_DATE (numpy dates them now): the same arrays, the same
    bytes."""
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
        for name, a in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", NPZ_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            with z.open(info, "w", force_zip64=True) as member:
                np.lib.format.write_array(member, np.asanyarray(a), allow_pickle=False)


def write_outputs(folder: Path, out: predict.Outputs) -> None:
    """A sheet's three files (see the module) in `folder`, made."""
    folder.mkdir(parents=True)
    if not cv2.imwrite(str(folder / CHANCE_FILE), out.chance):
        raise OSError(f"can't write {folder / CHANCE_FILE}")
    save_npz(folder / KIND_FILE, kind=out.kind, p=out.kind_p)
    save_npz(folder / DIRECTION_FILE, direction=out.direction)


def _tally(sd: data.SheetData, chance: np.ndarray, text: np.ndarray, unknown: np.ndarray,
           scope: np.ndarray) -> scores._Tally:
    """One sheet's counts on the px of `scope`, as `scores.score_sheet` counts them, the long lines aside."""
    ink = data.page_ink(sd)
    scored = ink & ~unknown & scope
    t = scores._Tally(sheets=1)
    t.pos += np.bincount(chance[scored & text], minlength=256)
    t.neg += np.bincount(chance[scored & ~text], minlength=256)
    scores._add_words(t, sd, chance, ink & scope, scope)
    return t


def quarter_scores(sheets: list[data.SheetData], chances: dict[str, np.ndarray],
                   quarters: dict[str, np.ndarray]) -> dict:
    """Each quarter's detector on its own held-out quarter: the one measure (`scores`) on the px of the sheets in that
    quarter (their chance from that detector alone), the long lines aside (they take minutes a page to find): the ink
    px's AUC and log loss, the text ink found, the ink outside every box called text, and Terra's words found by kind
    (a word counted in the quarter holding its centre px, judged on its ink there).  Unknown px are the sheets' own
    (the reader's radius).  Counts are whole numbers, divided at the end; JSON-able."""
    tallies = {q: scores._Tally() for q in paths.QUARTERS}
    for sd in sheets:
        text, unknown = data.page_text(sd)
        for q, name in enumerate(paths.QUARTERS):
            scope = quarters[sd.sheet] == q
            if scope.any():
                tallies[name].add(_tally(sd, chances[sd.sheet], text, unknown, scope))
    out = {}
    for name, t in tallies.items():
        s = t.scores()
        s.pop("long_lines")
        out[name] = s
    return {"where": "each quarter's detector on the 2000 sheets' px in its own held-out quarter",
            "called": scores.CALLED, "quarters": out}


# --- the stage ------------------------------------------------------------------------------------------------------

def build_reader_text(device: str | None = None, recipe: train.Recipe | None = None, seed: int = SEED) -> dict:
    """Train the four detectors by `recipe` (None: `stage.DEFAULT_RECIPE`) on `device` (None: `train.default_device()`)
    and score every 2000 sheet with them (see the module).  Returns the steps, the device, the sheets trained on and
    scored, the unknown radius, the answered blocks' count, the scores and the seconds taken by each part."""
    recipe = stage.DEFAULT_RECIPE if recipe is None else recipe
    device = device or train.default_device()
    seconds = {}
    t0 = time.perf_counter()
    blocks = held_blocks()
    area = load_block_area(blocks)
    sheets = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False)
    radius = measure_radius(sheets, (area,))
    sheets = [dataclasses.replace(sd, point_radius=radius["px"]) for sd in sheets]
    by_name = {sd.sheet: sd for sd in sheets if sd.series == paths.SERIES}
    missing = [name for name in paths.SHEETS if name not in by_name]
    if missing:
        raise ValueError(f"2000 sheets not among the training sheets: {', '.join(missing)}")
    scored = [by_name[name] for name in paths.SHEETS]
    quarters = {sd.sheet: page_quarters(sd) for sd in scored}
    seconds["load"] = time.perf_counter() - t0
    log.info("%d sheets of %d maps read; %d answered blocks held out; unknown radius %d px (%s %s, %s) (%.0f s)",
             len(sheets), len(paths.TEXT_TRAIN_SERIES), len(blocks), radius["px"], radius["longest"]["series"],
             radius["longest"]["sheet"], radius["longest"]["text"], seconds["load"])
    part = runs.fresh_folder(paths.READER_TEXT_DIR)
    (part / MODELS_DIR).mkdir()
    chances = {}
    with gpu.gpu_turn(True, device):
        t0 = time.perf_counter()
        for q, name in enumerate(paths.QUARTERS):
            log.info("%s held out: training on %s", name, device)
            net = train.fit(data.hold_out(sheets, (data.Quarter(q), area)), recipe, device, seed)
            train.save_model(part / MODELS_DIR / f"{name}.pt", net, recipe, device, seed)
            del net
            gc.collect()
            if gpu.on_gpu(device):
                torch.mps.empty_cache()     # the training's buffers, before the next one makes its own
        seconds["train"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        nets = [predict.load_model(part / MODELS_DIR / f"{name}.pt", device) for name in paths.QUARTERS]
        for sd in scored:
            t1 = time.perf_counter()
            out = sheet_outputs(nets, data.page_ink(sd), quarters[sd.sheet], device)
            write_outputs(part / sd.sheet, out)
            chances[sd.sheet] = out.chance
            log.info("%s scored (%.0f s)", sd.sheet, time.perf_counter() - t1)
        del nets
        seconds["predict"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = quarter_scores(scored, chances, quarters)
    seconds["score"] = time.perf_counter() - t0
    held = {"quarters": list(paths.QUARTERS), "split": list(grid.QUARTER_SPLIT), "blocks": blocks,
            "block_grid_px": int(area.mask.sum()), "unknown_radius": radius, "recipe": dataclasses.asdict(recipe),
            "seed": seed}
    (part / HELD_FILE).write_text(json.dumps(held, indent=1) + "\n")
    (part / SCORES_FILE).write_text(json.dumps(got, indent=1) + "\n")
    runs.replace_folder(part, paths.READER_TEXT_DIR)
    return {"iterations": recipe.iterations, "device": device, "trained_on": len(sheets), "scored": len(scored),
            "blocks": len(blocks), "radius": radius, "scores": got, "seconds": seconds}
