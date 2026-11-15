"""Trying recipes for the text detector (`sfzmaps solver text-trial`): each trained on every sheet Terra read of the
training maps (`paths.TEXT_TRAIN_SERIES`, or some of them: `training_maps`) with the validation area held out, and
scored on that area alone (and, when asked, on whole test drawings).  Not a stage: each recipe's run goes in its own
experiment folder (`folder`: data/solver/experiments/text-<Recipe.name()>/, with -train=<maps> after it when trained
on some of the maps only and -seed=<n> for a seed other than 0), never a stage's output and never shipped:

  model.pt          the weights, with the recipe they came from (`train.save_model`)
  recipe.json       the recipe's settings (`train.Recipe(**...)` reads them back)
  scores.json       the scores on the validation area (`scores.score_sheets`, "validation"; no times in it)
  test_scores.json  with test maps asked for, the scores on every px of their sheets ("whole", as the scores stage
                    takes them, `score_stage`)
  minutes.json      the minutes each part took, by run: "plain", or "prefetch" for a run with the batches drawn ahead
                    (`Recipe.prefetch`, which trains the same weights into the same folder, so the two runs' minutes
                    are kept side by side): load (reading the sheets, once for all the recipes of a run), train,
                    predict and score (the first recipe's score also finds each sheet's long lines, which the later
                    ones reuse: `scores.line_ink`), and test (predicting and scoring the test drawings)

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

PREFIX = runs.TEXT_PREFIX         # an experiment folder of the text detector's (`folder`; a solve never takes one)
FILES = ("model.pt", "recipe.json", "scores.json", "minutes.json")
TEST_FILE = "test_scores.json"
PARTS = ("load", "train", "predict", "score", "test")      # minutes.json's parts of a run, in order
RUNS = ("plain", "prefetch")                                # minutes.json's runs, by `Recipe.prefetch`
TRUE, FALSE = ("true", "yes", "on", "1"), ("false", "no", "off", "0")


def training_maps(series=None) -> tuple[str, ...]:
    """The maps a trial trains on: `series` (None: every training map, `paths.TEXT_TRAIN_SERIES`), in the order of
    paths.TEXT_TRAIN_SERIES (the order the sheets are read in, so one set of maps trains one way).  A map that is not
    a training map, or none, is a ValueError."""
    if series is None:
        return tuple(paths.TEXT_TRAIN_SERIES)
    return _maps(series, paths.TEXT_TRAIN_SERIES, "training")


def test_maps(series) -> tuple[str, ...]:
    """The test drawings a trial is also scored on (`paths.TEXT_TEST_SERIES`, in that order); none for none.  A map
    that is not a test drawing is a ValueError."""
    return _maps(series, paths.TEXT_TEST_SERIES, "test") if series else ()


def _maps(series, allowed, kind: str) -> tuple[str, ...]:
    asked = set(series)
    unknown = sorted(asked - set(allowed))
    if unknown:
        raise ValueError(f"not a {kind} map: {', '.join(unknown)} (the {kind} maps are {', '.join(allowed)})")
    if not asked:
        raise ValueError(f"no {kind} map")
    return tuple(s for s in allowed if s in asked)


def folder(recipe: train.Recipe, seed: int = 0, train_series=None) -> Path:
    """The experiment folder of a recipe's trial with `seed`, trained on `train_series` (`training_maps`):
    EXPERIMENTS_DIR/text-<recipe.name()>, with -train=<the maps, joined by "+"> after it when not every training map
    and -seed=<seed> when the seed is not 0."""
    maps = training_maps(train_series)
    some = f"-train={'+'.join(maps)}" if maps != tuple(paths.TEXT_TRAIN_SERIES) else ""
    return runs.EXPERIMENTS_DIR / f"{PREFIX}{recipe.name()}{some}{f'-seed={seed}' if seed else ''}"


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
    """A folder's minutes.json as it was, by run (RUNS); {} without one.  A file of one run's parts alone (as the
    first baseline's was written) is the plain run's."""
    got = json.loads(f.read_text()) if f.is_file() else {}
    return got if set(got) <= set(RUNS) else {RUNS[0]: got}


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


def run_trials(recipes: list[train.Recipe], device: str | None = None, seed: int = 0, reuse: bool = False,
               train_series=None, test_series=()) -> list[dict]:
    """Try each recipe in turn (see the module) on `device` (None: `train.default_device()`) with `seed`, trained on
    `train_series` (None: every training map; `training_maps`) and, with `test_series`, scored on those test drawings
    as well (`test_maps`).  Every recipe's folder must be its own, and with `reuse` hold its model (checked before
    anything is read).  Returns, per recipe, its name, folder, scores (and test scores, or None) and this run's
    minutes."""
    device = device or train.default_device()
    maps, tests = training_maps(train_series), test_maps(test_series)
    plan = [(r, folder(r, seed, maps)) for r in recipes]
    seen = set()
    for r, out in plan:
        if out in seen:
            raise ValueError(f"two of the recipes are {r.name()} (they would write one folder, {out})")
        seen.add(out)
        if reuse:
            _check_saved(out / FILES[0], r, seed)
    t0 = time.perf_counter()
    sheets = data.load_sheets(maps, hold_out_validation=True)
    scopes = validation_scopes(sheets)
    test_sheets = data.load_sheets(tests, hold_out_validation=False) if tests else []
    load = (time.perf_counter() - t0) / 60
    log.info("%d sheets read, %d with px in the validation area, %d of test drawings (%.1f min)", len(sheets),
             len(scopes), len(test_sheets), load)
    results = []
    for recipe, out in plan:
        minutes, run = {"load": load}, RUNS[recipe.prefetch]
        before = _minutes(out / FILES[3])       # the other run's minutes are kept (the same weights)
        if reuse:
            work = out
            minutes.update({k: v for k, v in before.get(run, {}).items() if k == "train"})
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
            t0 = time.perf_counter()
            test_chances = {(sd.series, sd.sheet): predict.chance(net, data.page_ink(sd), device)
                            for sd in test_sheets}
            test_seconds = time.perf_counter() - t0
            del net
        t0 = time.perf_counter()
        got = scores.score_sheets(chances, "validation", sheets)
        minutes["score"] = (time.perf_counter() - t0) / 60
        del chances
        tested = None
        if tests:
            t0 = time.perf_counter()
            tested = scores.score_sheets(test_chances, "whole", test_sheets)
            minutes["test"] = (test_seconds + time.perf_counter() - t0) / 60
            _write_json(work / TEST_FILE, tested)
        del test_chances
        minutes = {k: round(minutes[k], 3) for k in PARTS if k in minutes}
        every = {**before, run: minutes}
        _write_json(work / FILES[1], dataclasses.asdict(recipe))
        _write_json(work / FILES[2], got)
        _write_json(work / FILES[3], {k: every[k] for k in RUNS if k in every})
        if not reuse:
            runs.replace_folder(work, out)
        log.info("%s: words found %s, long-line ink called text %s", recipe.name(), got["all"]["words_found"],
                 got["all"]["long_lines"]["called_text"])
        results.append({"name": recipe.name(), "folder": out, "scores": got, "test_scores": tested,
                        "minutes": minutes})
    return results
