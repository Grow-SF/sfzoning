import logging
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import ndimage

from sfzmaps.sheetwarp import probabilistic
from sfzmaps.sheetwarp.probabilistic import (ITERS, ProbabilisticInputs, StopWhenSettled, block_line_weights,
                                             probabilistic_correction)
from sfzmaps.sheetwarp.street_edges_and_later_blocks import unexplained_weight
from sfzmaps.sheetwarp.targets import Targets
from sfzmaps.transform import Correction

# the CPU everywhere (CI's path); the Mac's GPU too where there is one
DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(),
                                                                reason="no GPU (mps) here"))]
OFF_THE_PRINT = 1.5          # px: how far today's placement puts the synthetic sheet's block sides off their lines


def _synthetic_sheet():
    """A 200 px crop printed with 1 px block lines every 50 px each way; its block-side points, every 2 px along
    them and up to 0.8 px either side of them (a print is never exact), and three control points, all placed
    OFF_THE_PRINT px right of and below where the print has them: the inputs of a probabilistic correction, its
    targets and block-line weights."""
    h = w = 200
    lines = (50, 100, 150)
    along = np.arange(14, 187, 2.0)
    running_down = np.zeros((h, w), bool)        # ink running at 90 degrees: map 9
    running_down[10:190, list(lines)] = True
    running_across = running_down.T.copy()       # ink running at 0 degrees: map 0
    weights = np.zeros((18, h, w), np.float16)
    weights[9][running_down], weights[0][running_across] = 1, 1
    maps = np.full((18, h, w), 255, np.uint8)
    for b, centres in ((9, running_down), (0, running_across)):
        maps[b] = np.minimum(ndimage.distance_transform_edt(~centres), 255).astype(np.uint8)
    either_side = np.random.default_rng(5).uniform(-0.8, 0.8, len(lines) * len(along) * 2)
    down = np.array([(x, y) for x in lines for y in along], float)
    across = np.array([(x, y) for y in lines for x in along], float)
    down[:, 0] += either_side[:len(down)]
    across[:, 1] += either_side[len(down):]
    points = np.concatenate([down, across]) + OFF_THE_PRINT
    bins = np.repeat([9, 0], [len(down), len(across)])
    control_on_the_print = np.array([[50.0, 50.0], [150.0, 50.0], [100.0, 150.0]], np.float32)
    inputs = ProbabilisticInputs(P=points.astype(np.float32), bin=bins, street_px=20.0, origin=np.array([0, 0]),
                                 gbase=control_on_the_print + OFF_THE_PRINT, gpx=control_on_the_print, tau_j=0.01)
    return inputs, Targets(maps=maps, origin=np.array([0, 0]), page_shape=(h, w)), weights


@pytest.mark.parametrize("device", DEVICES)
def test_the_probabilistic_correction_repeats_bit_for_bit(device, monkeypatch):
    monkeypatch.setenv("SFZMAPS_WARP_DEVICE", device)
    inputs, targets, weights = _synthetic_sheet()
    first_run = probabilistic_correction(inputs, targets, weights)
    second_run = probabilistic_correction(inputs, targets, weights)
    assert isinstance(first_run.grid, np.ndarray) and first_run.grid.dtype == np.float32
    assert first_run.grid.shape == (2, 5, 5) and np.isfinite(first_run.grid).all()
    assert np.array_equal(first_run.grid, second_run.grid)

    # and it is a correction: it brings every block-side point within 1 px of its printed line
    def within_1px(points):
        col, row = np.round(points).astype(int).T
        return targets.maps[inputs.bin, row, col] <= 1

    corrected = inputs.P + Correction(first_run.grid, first_run.spacing, first_run.origin).u(inputs.P)
    assert within_1px(inputs.P).mean() < 0.6 and within_1px(corrected).all()


def test_a_stage_stops_when_its_loss_settles(monkeypatch, caplog):
    monkeypatch.setenv("SFZMAPS_WARP_DEVICE", "cpu")
    inputs, targets, weights = _synthetic_sheet()
    with caplog.at_level(logging.INFO, logger=probabilistic.__name__):
        probabilistic_correction(inputs, targets, weights)
    steps = [int(m) for m in re.findall(r"stage \d+: .*?steps (\d+)", caplog.text)]
    assert len(steps) >= 3                                   # every stage, from half a street width to sigma
    # the synthetic sheet's loss flattens soon in every stage: each stops when it has settled, not at the cap
    assert all(50 <= n < ITERS for n in steps), steps


def test_the_stop_rule_compares_the_loss_every_25_steps_from_50_on():
    def stop_step(loss_after):
        """the step a stage stops at, given its loss after each step"""
        rule = StopWhenSettled()
        return next((step for step in range(ITERS) if rule.settled(step, loss_after(step))), ITERS)

    assert stop_step(lambda step: 3.0) == 50                          # flat from the start: not before step 50
    assert stop_step(lambda step: -2.0) == 50                         # relative to the loss's size, whatever its sign
    assert stop_step(lambda step: 1.0 + max(0, 100 - step)) == 125   # falls until step 100, flat after
    assert stop_step(lambda step: 1.0 - 3e-6 * step) == 50            # falls 7.5e-5 of itself every 25 steps: settled
    assert stop_step(lambda step: np.exp(-1e-5 * step)) == ITERS      # falls 2.5e-4 of itself every 25 steps: never
    assert stop_step(lambda step: 1.0 + 0.01 * step) == 50            # rising: it fell by less than 1e-4


def test_block_line_weights_favour_the_thin_strokes():
    page = np.full((200, 200), 255, np.uint8)
    for x in (30, 60, 90, 120):          # thin vertical lines, 1 px
        page[10:190, x] = 0
    for y in (40, 100, 160):             # heavy horizontal bars, 6 px
        page[y - 3:y + 3, 10:190] = 0
    targets = Targets(maps=np.zeros((18, 200, 200), np.uint8), origin=np.array([0, 0]), page_shape=(200, 200))
    weights = block_line_weights(SimpleNamespace(page=page, series="s", name="n"), targets)
    assert weights.shape == (18, 200, 200) and weights.dtype == np.float16
    # thin centres: the vertical lines' columns, between the bars; vertical ink is 90 degrees, map 9
    thin_centres = weights[9][50:90, [30, 60, 90, 120]].astype(float)
    # heavy centres: the bars' middle rows (an even width has two), away from the lines; horizontal ink is map 0
    heavy_centres = np.stack([weights[0][y - 1:y + 1, 40:80].max(axis=0) for y in (40, 100, 160)]).astype(float)
    assert thin_centres.mean() > heavy_centres.mean()
    assert thin_centres.mean() - heavy_centres.mean() > 0.5
    assert (thin_centres > 0).all()            # every thin centre pixel is there to be compared


def test_street_edge_points_count_by_how_little_blocks_explain_them():
    s = 30.0
    w = unexplained_weight(np.array([0.0, 5.0, 300.0]), s)
    assert w[0] == 0
    assert 0 < w[1] < 0.1
    assert w[2] > 0.999
