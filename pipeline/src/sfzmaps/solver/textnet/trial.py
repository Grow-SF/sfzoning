"""Trying recipes for the text detector (`sfzmaps solver text-trial`): each trained on every sheet Terra read of the
training maps (`paths.TEXT_TRAIN_SERIES`) with the validation area held out, and scored on that area alone.  Not a
stage: each recipe's run goes in its own experiment folder (`folder`: data/solver/experiments/text-<Recipe.name()>/,
with -seed=<n> after it for a seed other than 0), never a stage's output and never shipped:

  model.pt      the weights, with the recipe they came from (`train.save_model`)
  recipe.json   the recipe's settings (`train.Recipe(**...)` reads them back)
  scores.json   the scores on the validation area (`scores.score_sheets`, "validation"; no times in it)
  minutes.json  the minutes each part took: load (reading the sheets, once for all the recipes of a run), train,
                predict and score (the first recipe's score also finds each sheet's long lines, which the later ones
                reuse: `scores.line_ink`)

The sheets are read once for all the recipes.  The recipes run one at a time, each training and predicting in its own
exclusive turn on the Mac's GPU (`sfzmaps.gpu`), and scored on the CPU after it.  Only the tiles holding px of the
validation area are predicted (`predict.chance`'s `only`); they get the chance a whole page's scoring would give them.

A recipe is written as "name=value" settings over today's (`parse_recipe`); a recipe that runs is made afresh in a
folder beside its own and renamed into place (`runs.replace_folder`).  With `reuse`, each recipe's folder must already
hold its model.pt (that recipe's and that seed's): it is scored, not trained, and the minutes its training took are
kept from the folder's minutes.json.
"""
import dataclasses
import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from sfzmaps import gpu
from sfzmaps.solver import paths, runs
from sfzmaps.solver.textnet import data, predict, scores, train

log = logging.getLogger(__name__)

PREFIX = "text-"                  # an experiment folder of the text detector's (`folder`)
FILES = ("model.pt", "recipe.json", "scores.json", "minutes.json")
PARTS = ("load", "train", "predict", "score")      # minutes.json's parts, in order
TRUE, FALSE = ("true", "yes", "on", "1"), ("false", "no", "off", "0")


def folder(recipe: train.Recipe, seed: int = 0) -> Path:
    """The experiment folder of a recipe's trial with `seed`: EXPERIMENTS_DIR/text-<recipe.name()>, with -seed=<seed>
    after it when the seed is not 0."""
    return runs.EXPERIMENTS_DIR / f"{PREFIX}{recipe.name()}{f'-seed={seed}' if seed else ''}"


def _value(field: dataclasses.Field, text: str):
    """A recipe setting's value from its text, by the setting's type."""
    if field.type is bool:
        low = text.strip().lower()
        if low in TRUE or low in FALSE:
            return low in TRUE
        raise ValueError(f"recipe: {field.name} is True or False, not {text!r}")
    try:
        return field.type(text)
    except ValueError:
        raise ValueError(f"recipe: {field.name} is {'a whole number' if field.type is int else 'a number'}, not "
                         f"{text!r}") from None


def parse_recipe(settings) -> train.Recipe:
    """A recipe from settings over today's (`train.Recipe()`): "name=value" strings, a list of them or one string of
    them apart by spaces; none (or the one word "default") is today's recipe.  A switch is true / false (yes / no,
    on / off, 1 / 0).  An unknown or repeated name, or a value the recipe refuses, is a ValueError."""
    if isinstance(settings, str):
        settings = settings.split()
    if list(settings) == ["default"]:
        settings = []
    fields = {f.name: f for f in dataclasses.fields(train.Recipe)}
    chosen = {}
    for item in settings:
        name, eq, text = item.partition("=")
        name = name.strip()
        if not eq or not name:
            raise ValueError(f"recipe: {item!r} is not name=value")
        if name not in fields:
            raise ValueError(f"recipe: no setting {name!r}: one of {', '.join(fields)}")
        if name in chosen:
            raise ValueError(f"recipe: {name} is set twice")
        chosen[name] = _value(fields[name], text)
    return train.Recipe(**chosen)


def _write_json(f: Path, obj) -> None:
    """Write `obj` to `f` as JSON, whole: to a file beside it first, then renamed into place."""
    part = f.with_name(f".{f.name}.{os.getpid()}.part")
    try:
        part.write_text(json.dumps(obj, indent=1) + "\n")
        os.replace(part, f)
    finally:
        part.unlink(missing_ok=True)


def _check_saved(f: Path, recipe: train.Recipe, seed: int) -> None:
    """Stop (FileNotFoundError, ValueError) unless `f` is a model trained by `recipe` with `seed`."""
    if not f.is_file():
        raise FileNotFoundError(f"can't reuse {f}: there is no model there (run the trial without --reuse)")
    saved = torch.load(f, map_location="cpu")
    if "recipe" not in saved or train.Recipe(**saved["recipe"]).name() != recipe.name() \
            or saved.get("seed") != seed:
        raise ValueError(f"can't reuse {f}: it was trained by another recipe or seed than {recipe.name()}, seed "
                         f"{seed}")


def _minutes(f: Path) -> dict:
    """A folder's minutes.json as it was, {} without one."""
    return json.loads(f.read_text()) if f.is_file() else {}


def validation_scopes(sheets: list[data.SheetData]) -> dict[tuple[str, str], np.ndarray]:
    """Each sheet's px in the validation area (`scores.scored_px`), packed 8 px a byte along each row, for the sheets
    with any."""
    out = {}
    for sd in sheets:
        scope = scores.scored_px(sd, "validation")
        if scope.any():
            out[sd.series, sd.sheet] = np.packbits(scope, axis=1)
    return out


def validation_chances(net, sheets: list[data.SheetData], scopes: dict, device: str) -> dict:
    """The chance of text (`predict.chance`) on each sheet with px in the validation area (`validation_scopes`),
    predicted only on the tiles holding some."""
    out = {}
    for sd in sheets:
        key = (sd.series, sd.sheet)
        if key in scopes:
            only = np.unpackbits(scopes[key], axis=1, count=sd.width).view(bool)
            out[key] = predict.chance(net, data.page_ink(sd), device, only=only)
    return out


def run_trials(recipes: list[train.Recipe], device: str | None = None, seed: int = 0,
               reuse: bool = False) -> list[dict]:
    """Try each recipe in turn (see the module) on `device` (None: `train.default_device()`) with `seed`.  Every
    recipe's folder must be its own, and with `reuse` hold its model (checked before anything is read).  Returns, per
    recipe, its name, folder, scores and minutes."""
    device = device or train.default_device()
    plan = [(r, folder(r, seed)) for r in recipes]
    seen = set()
    for r, out in plan:
        if out in seen:
            raise ValueError(f"two of the recipes are {r.name()} (they would write one folder, {out})")
        seen.add(out)
        if reuse:
            _check_saved(out / FILES[0], r, seed)
    t0 = time.perf_counter()
    sheets = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=True)
    scopes = validation_scopes(sheets)
    load = (time.perf_counter() - t0) / 60
    log.info("%d sheets read, %d with px in the validation area (%.1f min)", len(sheets), len(scopes), load)
    results = []
    for recipe, out in plan:
        minutes = {"load": load}
        if reuse:
            work = out
            minutes.update({k: v for k, v in _minutes(out / FILES[3]).items() if k == "train"})
        else:
            work = runs.fresh_folder(out)
        with gpu.gpu_turn(True, device):
            if not reuse:
                log.info("%s: training on %s", recipe.name(), device)
                t0 = time.perf_counter()
                net = train.fit(sheets, recipe, device, seed)
                train.save_model(work / FILES[0], net, recipe, device, seed)
                minutes["train"] = (time.perf_counter() - t0) / 60
                del net
                gc.collect()
                if gpu.on_gpu(device):
                    torch.mps.empty_cache()
            t0 = time.perf_counter()
            train.settle_torch()
            net = predict.load_model(work / FILES[0], device)
            chances = validation_chances(net, sheets, scopes, device)
            minutes["predict"] = (time.perf_counter() - t0) / 60
            del net
        t0 = time.perf_counter()
        got = scores.score_sheets(chances, "validation", sheets)
        minutes["score"] = (time.perf_counter() - t0) / 60
        del chances
        minutes = {k: round(minutes[k], 3) for k in PARTS if k in minutes}
        _write_json(work / FILES[1], dataclasses.asdict(recipe))
        _write_json(work / FILES[2], got)
        _write_json(work / FILES[3], minutes)
        if not reuse:
            runs.replace_folder(work, out)
        log.info("%s: words found %s, long-line ink called text %s", recipe.name(), got["all"]["words_found"],
                 got["all"]["long_lines"]["called_text"])
        results.append({"name": recipe.name(), "folder": out, "scores": got, "minutes": minutes})
    return results
