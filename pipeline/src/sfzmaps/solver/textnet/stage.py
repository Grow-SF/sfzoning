"""The text detector, as its build stage makes it (`sfzmaps solver text`, stage `solver-text`): DEFAULT_RECIPE trained
on every sheet Terra read of the training maps (`paths.TEXT_TRAIN_SERIES`), the validation area included (the recipe
was chosen with it held out, `trial`), and then every 2000 sheet's page scored with it (`paths.SHEETS`, the sheets the
walls read the chance of text on).  The sheets are read first, without their placements (the training needs none);
then the training and the scoring run in one exclusive turn on the Mac's GPU (`sfzmaps.gpu`; nothing else computes on
it meanwhile, so the numbers are the same every run).  On the CPU there is no turn.

Writes data/solver/text/model.pt and chance/<sheet>.png (`paths.TEXT_DIR`), nothing else: built in a fresh folder
beside it and renamed into its place (`runs.replace_folder`), so a run that stops leaves the last one whole.  Its
scores on the test drawings are another stage's (`score_stage.build_scores`, stage `solver-text-scores`).
"""
import gc
import logging
import time

import cv2
import torch

from sfzmaps import gpu
from sfzmaps.solver import grid, paths, runs
from sfzmaps.solver.textnet import data, predict, train

log = logging.getLogger(__name__)

# The recipe the detector trains with, chosen by trials with the validation area held out (`trial`; the steps and their
# scores are in pipeline/README.md): the network at half size, the loss on the ink and the px next to it, 4,000 steps
# with the learning rate rising over the first 10%, the batches drawn ahead.  On the validation area its three seeds
# find 94.03-94.92% of the words and call 4.12-4.15% of the long-line ink text (today's recipe, `Recipe()`: 94.15% and
# 4.10%); its trial trained in 6.2 min on the Mac's GPU, today's in 45.7.
DEFAULT_RECIPE = train.Recipe(iterations=4000, pct_start=0.1, ink_only="ink+1px", half_size=True, prefetch=True)


def build_text(recipe: train.Recipe | None = None, device: str | None = None, seed: int = 0) -> dict:
    """Train the detector by `recipe` (None: DEFAULT_RECIPE) on `device` (None: `train.default_device()`) and score
    every 2000 sheet's page with it (see the module).  Returns the steps, the device, the sheets trained on and
    scored, and the seconds taken by each part."""
    recipe = DEFAULT_RECIPE if recipe is None else recipe
    device = device or train.default_device()
    seconds = {}
    t0 = time.perf_counter()
    sheets = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False, placement=False)
    trained = len(sheets)
    seconds["load"] = time.perf_counter() - t0
    log.info("%d sheets of %d maps read (%.0f s)", trained, len(paths.TEXT_TRAIN_SERIES), seconds["load"])
    part = runs.fresh_folder(paths.TEXT_DIR)
    with gpu.gpu_turn(True, device):
        log.info("training on %s", device)
        t0 = time.perf_counter()
        net = train.fit(sheets, recipe, device, seed)
        train.save_model(part / paths.TEXT_MODEL.name, net, recipe, device, seed)
        seconds["train"] = time.perf_counter() - t0
        del net, sheets
        gc.collect()
        if gpu.on_gpu(device):
            torch.mps.empty_cache()     # the training's buffers, before the scoring makes its own
        t0 = time.perf_counter()
        net = predict.load_model(part / paths.TEXT_MODEL.name, device)     # as the scores stage reads it
        chances = part / paths.TEXT_CHANCE_DIR.name
        chances.mkdir()
        for name in paths.SHEETS:
            t1 = time.perf_counter()
            ink = data.load_page(name) < grid.INK_BELOW
            if not cv2.imwrite(str(chances / f"{name}.png"), predict.chance(net, ink, device)):
                raise OSError(f"can't write {chances / f'{name}.png'}")
            log.info("%s scored (%.0f s)", name, time.perf_counter() - t1)
        seconds["predict"] = time.perf_counter() - t0
    runs.replace_folder(part, paths.TEXT_DIR)
    return {"iterations": recipe.iterations, "device": device, "trained_on": trained, "scored": len(paths.SHEETS),
            "seconds": seconds}
