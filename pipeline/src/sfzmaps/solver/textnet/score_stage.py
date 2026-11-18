"""The text detector's scores on the test drawings, as their build stage makes them (`sfzmaps solver text-scores`,
stage `solver-text-scores`, after `solver-text`): every sheet Terra read of the test maps (`paths.TEXT_TEST_SERIES`,
whole drawings the detector never trained on) scored with the stage's model (`paths.TEXT_MODEL`) in one
exclusive turn on the Mac's GPU, then the one measure (`scores.score_sheets`) on every px of them ("whole").  The
sheets are read with their placements: the long lines are measured in each sheet's own feet.

Writes data/solver/text-scores/scores.json (`paths.TEXT_SCORES_DIR`), nothing else: built in a fresh folder beside it
and renamed into its place (`runs.replace_folder`).  It holds no time, so the same chances give the same bytes.
"""
import json
import logging
import time

from sfzmaps import gpu
from sfzmaps.solver import paths, runs
from sfzmaps.solver.textnet import data, predict, scores, train

log = logging.getLogger(__name__)

SCORES_FILE = "scores.json"


def build_scores(device: str | None = None) -> dict:
    """Score the stage's model on the test drawings and write TEXT_SCORES_DIR/scores.json (see the module).  Returns
    the scores and the seconds taken by each part."""
    device = device or train.default_device()
    model_file = paths.TEXT_MODEL
    if not model_file.is_file():
        raise FileNotFoundError(f"can't read {model_file}: train the detector first (`sfzmaps solver text`, stage "
                                "solver-text)")
    seconds = {}
    t0 = time.perf_counter()
    sheets = data.load_sheets(paths.TEXT_TEST_SERIES, hold_out_validation=False)
    seconds["load"] = time.perf_counter() - t0
    chances = {}
    with gpu.gpu_turn(True, device):
        t0 = time.perf_counter()
        train.settle_torch()
        net = predict.load_model(model_file, device)
        for sd in sheets:
            chances[sd.series, sd.sheet] = predict.chance(net, data.page_ink(sd), device)
            log.info("%s %s scored", sd.series, sd.sheet)
        seconds["predict"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = scores.score_sheets(chances, "whole", sheets)
    seconds["score"] = time.perf_counter() - t0
    part = runs.fresh_folder(paths.TEXT_SCORES_DIR)
    (part / SCORES_FILE).write_text(json.dumps(got, indent=1) + "\n")
    runs.replace_folder(part, paths.TEXT_SCORES_DIR)
    return {"scores": got, "seconds": seconds}
