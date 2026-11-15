import logging
import re
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

from sfzmaps.sheetwarp import device, probabilistic
from sfzmaps.sheetwarp.probabilistic import (ITERS, ProbabilisticInputs, StopWhenSettled, block_line_weights,
                                             probabilistic_correction)
from sfzmaps.sheetwarp.street_edges_and_later_blocks import unexplained_weight
from sfzmaps.sheetwarp.sheet import NothingToFit
from sfzmaps.sheetwarp.targets import Targets, ink_centres
from sfzmaps.transform import Correction

# the CPU everywhere (CI's path); the Mac's GPU too where there is one
DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(),
                                                                reason="no GPU (mps) here"))]
OFF_THE_PRINT = 1.5          # px: how far today's placement puts the synthetic sheet's block sides off their lines


def _synthetic_sheet(off_the_print=OFF_THE_PRINT, either_side_px=0.8):
    """A 200 px crop printed with 1 px block lines every 50 px each way; its block-side points, every 2 px along
    them and up to `either_side_px` either side of them (a print is never exact), and three control points, all
    placed `off_the_print` px right of and below where the print has them: the inputs of a probabilistic
    correction, its targets and block-line weights."""
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
    either_side = np.random.default_rng(5).uniform(-either_side_px, either_side_px, len(lines) * len(along) * 2)
    down = np.array([(x, y) for x in lines for y in along], float)
    across = np.array([(x, y) for y in lines for x in along], float)
    down[:, 0] += either_side[:len(down)]
    across[:, 1] += either_side[len(down):]
    points = np.concatenate([down, across]) + off_the_print
    bins = np.repeat([9, 0], [len(down), len(across)])
    control_on_the_print = np.array([[50.0, 50.0], [150.0, 50.0], [100.0, 150.0]], np.float32)
    inputs = ProbabilisticInputs(P=points.astype(np.float32), bin=bins, street_px=20.0, origin=np.array([0, 0]),
                                 gbase=control_on_the_print + off_the_print, gpx=control_on_the_print, tau_j=0.01)
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
    targets = Targets(maps=np.zeros((18, 200, 200), np.uint8), origin=np.array([0, 0]), page_shape=(200, 200),
                      ink_centres=ink_centres(page < 128))
    weights = block_line_weights(SimpleNamespace(series="s", name="n"), targets)
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


def _todays_stroke_width_fit(widths):
    """the stroke-width EM as block_line_weights ran it on every centre pixel: each one's chance of the thin group"""
    m = np.array([np.percentile(widths, 25), np.percentile(widths, 95)])
    s = np.array([1.0, 1.0])
    pi = np.array([0.7, 0.3])
    for _ in range(200):
        lk = pi[None] * np.exp(-(widths[:, None] - m[None]) ** 2 / (2 * s[None] ** 2)) / (np.sqrt(2 * np.pi) * s[None])
        r = lk / np.maximum(lk.sum(1, keepdims=True), 1e-300)
        pi = r.mean(0)
        m = (r * widths[:, None]).sum(0) / r.sum(0)
        s = np.sqrt((r * (widths[:, None] - m[None]) ** 2).sum(0) / r.sum(0)) + 1e-3
    thin = int(np.argmin(m))
    return r[:, thin], m[thin], m[1 - thin], pi[thin]


def test_the_stroke_width_fit_on_the_histogram_equals_the_fit_on_every_pixel():
    # widths as an ink skeleton has them: twice the distance to the paper, sqrt of whole numbers; mostly thin lines
    rng = np.random.default_rng(11)
    widths = 2.0 * np.sqrt(np.r_[rng.choice([1, 2, 4, 5], 30000, p=[0.5, 0.3, 0.15, 0.05]),
                                 rng.choice([9, 10, 13, 16, 18, 25], 6000)].astype(float))
    thin_share, *groups = probabilistic.stroke_width_groups(widths)
    todays_share, *todays_groups = _todays_stroke_width_fit(widths)
    # the same fit summed in another order: alike to 1e-11 of each chance (the far tail's tiny chances included)
    assert np.allclose(thin_share, todays_share, rtol=1e-9, atol=0)
    assert np.allclose(groups, todays_groups, rtol=1e-12)
    # what the weights keep of it: float16 by way of float32, as they are stored
    assert np.array_equal(thin_share.astype(np.float32).astype(np.float16),
                          todays_share.astype(np.float32).astype(np.float16))


def _todays_block_line_weights(ink):
    """block_line_weights as it was before the ink centres were shared: per pixel, scipy's distances"""
    half = ndimage.distance_transform_edt(ink)
    sk = skeletonize(ink)
    weight = np.zeros(ink.shape, np.float32)
    weight[sk] = _todays_stroke_width_fit(2.0 * half[sk])[0]
    f = ndimage.gaussian_filter(ink.astype(np.float32), 1.0)
    gx, gy = ndimage.sobel(f, 1), ndimage.sobel(f, 0)
    Jxx, Jxy, Jyy = (ndimage.gaussian_filter(v, 2.0) for v in (gx * gx, gx * gy, gy * gy))
    cdir = (0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy)) + 90.0) % 180
    W8 = np.zeros((18, *ink.shape), np.float16)
    for i, b in enumerate(np.arange(0, 180, 10.0)):
        W8[i] = (weight * (np.abs((cdir - b + 90) % 180 - 90) <= 20.0)).astype(np.float16)
    return W8


def test_block_line_weights_from_the_shared_ink_centres_equal_todays():
    page = np.full((260, 400), 255, np.uint8)
    for k, width in enumerate((1, 1, 2, 3, 6)):
        cv2.line(page, (15, 20 + 45 * k), (385, 40 + 45 * k), 0, width)
        cv2.line(page, (30 + 75 * k, 10), (20 + 75 * k, 250), 0, width)
    cv2.circle(page, (200, 130), 60, 0, 2)
    cv2.putText(page, "R-4 C-2", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 4)
    ink = page < 128
    targets = Targets(maps=np.zeros((18, *ink.shape), np.uint8), origin=np.array([0, 0]), page_shape=ink.shape,
                      ink_centres=ink_centres(ink))
    weights = block_line_weights(SimpleNamespace(series="s", name="n"), targets)
    assert weights.dtype == np.float16 and (weights > 0).any()
    assert np.array_equal(weights, _todays_block_line_weights(ink))


def test_a_spread_that_collapses_to_zero_still_ends_the_fit(monkeypatch):
    # every block-side point exactly on its printed line and every control point on its match: every distance is 0,
    # so both EMs' sigma fall to 0 and their next round divides 0 by 0.  The points' NaN sigma never let the stages
    # reach it (the fit ran forever); the control points' made the correction NaN.
    monkeypatch.setenv("SFZMAPS_WARP_DEVICE", "cpu")
    inputs, targets, weights = _synthetic_sheet(off_the_print=0.0, either_side_px=0.0)
    stages = []
    todays_blur = device.blur

    def blur_counting_stages(maps, s, on):
        stages.append(s)
        assert len(stages) <= 40, "the stages never end"
        return todays_blur(maps, s, on)

    monkeypatch.setattr(device, "blur", blur_counting_stages)
    correction = probabilistic_correction(inputs, targets, weights)
    assert np.isfinite(correction.grid).all()
    assert len(stages) <= 6                # from half the street width (10 px) down to sigma, halving


@pytest.mark.filterwarnings("ignore::RuntimeWarning")          # the mean of no distances
def test_a_spread_that_cannot_be_estimated_has_nothing_to_fit(monkeypatch):
    # no block-side point (kind 0) to measure the spread on: its estimate is not a number
    monkeypatch.setenv("SFZMAPS_WARP_DEVICE", "cpu")
    inputs, targets, weights = _synthetic_sheet()
    inputs.wt, inputs.kind = np.ones(len(inputs.P), np.float32), np.full(len(inputs.P), 2)
    with pytest.raises(NothingToFit, match="spread"):
        probabilistic_correction(inputs, targets, weights)


def test_the_observation_count_does_not_depend_on_torch_s_threads():
    wt = np.random.default_rng(1).uniform(0, 1, 40_000).astype(np.float32) ** 3    # torch's float32 sum of these
    counts, threads_before = set(), torch.get_num_threads()                        # differs by its thread count
    for threads in (1, 2, 3, 5, 8, 10):
        torch.set_num_threads(threads)
        counts.add(probabilistic.observation_count(wt, len(wt), 4))
    torch.set_num_threads(threads_before)
    assert len(counts) == 1
    assert counts.pop() == float(np.sum(wt, dtype=np.float64)) + 4
    assert probabilistic.observation_count(None, 1000, 4) == 1004


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no GPU (mps) here")
def test_a_fit_gives_back_the_gpu_memory_it_used(monkeypatch):
    monkeypatch.setenv("SFZMAPS_WARP_DEVICE", "mps")
    inputs, targets, weights = _synthetic_sheet()
    torch.zeros(1, device="mps").cpu()                             # the framework's own start-up buffers
    before = torch.mps.driver_allocated_memory()
    probabilistic_correction(inputs, targets, weights)
    assert torch.mps.current_allocated_memory() <= 1_000_000
    assert torch.mps.driver_allocated_memory() <= before + 1_000_000


def test_a_fit_still_runs_on_the_cpu_after_giving_back_memory(monkeypatch):
    monkeypatch.setenv("SFZMAPS_WARP_DEVICE", "cpu")
    inputs, targets, weights = _synthetic_sheet()
    assert np.isfinite(probabilistic_correction(inputs, targets, weights).grid).all()
