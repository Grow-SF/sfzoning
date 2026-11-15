"""One quarter's text detector, as its build stage makes it (`sfzmaps solver text Q`, stage `solver-text-<Q>`): the
quarter's folder made afresh, every sheet's targets read with the quarter held out, then the model trained and every
sheet's page scored with it, both in one exclusive turn on the Mac's GPU (`sfzmaps.gpu`; nothing else computes on it
meanwhile, so the numbers are the same every run).  On the CPU there is no turn.

Writes data/solver/text-<Q>/model.pt and chance/<sheet>.png, nothing else.  The scores on the held-out quarters are
another stage's (`evaluate.build_scores`, stage `solver-text-scores`).
"""
import gc
import logging
import shutil
import time

import torch

from sfzmaps import gpu
from sfzmaps.solver import paths
from sfzmaps.solver.textnet import data, predict, train

log = logging.getLogger(__name__)


def build_text(q: str, iterations: int | None = None, device: str | None = None, seed: int = 0) -> dict:
    """Train the detector that holds out quarter `q` for `iterations` steps (None: train.ITERATIONS; at least 1) on
    `device` (None: `train.default_device()`) and score every sheet's page with it (see the module).  Returns the
    quarter, the steps, the device, the sheets scored and the seconds taken by each part."""
    data.quarter_index(q)
    if iterations is not None and iterations < 1:
        raise ValueError(f"iterations must be at least 1, not {iterations}")
    iterations = train.ITERATIONS if iterations is None else iterations
    device = device or train.default_device()
    shutil.rmtree(paths.text_dir(q), ignore_errors=True)
    seconds = {}
    t0 = time.perf_counter()
    sheets = train.quarter_targets(q)
    seconds["targets"] = time.perf_counter() - t0
    with gpu.gpu_turn(True, device):
        log.info("training on %s", device)
        t0 = time.perf_counter()
        net = train.fit(sheets, iterations, device, seed)
        train.save_model(q, net, iterations, device, seed)
        seconds["train"] = time.perf_counter() - t0
        del net, sheets
        gc.collect()
        if gpu.on_gpu(device):
            torch.mps.empty_cache()     # the training's buffers, before the scoring makes its own
        t0 = time.perf_counter()
        scored = predict.predict_sheets(q, device)
        seconds["predict"] = time.perf_counter() - t0
    return {"quarter": q, "iterations": iterations, "device": device, "sheets": scored, "seconds": seconds}
