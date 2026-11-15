from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sfzmaps.sheetwarp import first
from sfzmaps.sheetwarp.sheet import NothingToFit
from sfzmaps.sheetwarp.targets import SidePoints, Targets

# the CPU everywhere (CI's path); the Mac's GPU too where there is one
DEVICES = ["cpu", pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(),
                                                                reason="no GPU (mps) here"))]


def _sides(n_points, sid):
    px = np.column_stack([np.arange(n_points, dtype=float) + 10, np.full(n_points, 10.0)])
    return SidePoints(pts=px, sid=np.asarray(sid), ta=px - [3, 0], tb=px + [3, 0], px=px, pa=px - [3, 0], pb=px + [3, 0])


def test_propagation_with_every_side_held_has_nothing_to_move():
    sides = _sides(10, [0] * 5 + [1] * 5)
    targets = Targets(maps=np.zeros((18, 40, 40), np.uint8), origin=np.array([0, 0]), page_shape=(40, 40))
    with pytest.raises(NothingToFit):
        first.propagated_shifts(SimpleNamespace(series="s", name="n"), sides, targets, np.array([True, True]))


def test_propagation_with_no_held_side_on_the_crop_has_nothing_to_fit(tmp_path):
    # nothing held to carry the shifts out from (the search for the nearest solved points would ask for none)
    (tmp_path / "blocks_px.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    sides = _sides(10, [0] * 5 + [1] * 5)
    targets = Targets(maps=np.zeros((18, 40, 40), np.uint8), origin=np.array([0, 0]), page_shape=(40, 40))
    sheet = SimpleNamespace(series="s", name="n", folder=tmp_path)
    with pytest.raises(NothingToFit, match="no held"):
        first.propagated_shifts(sheet, sides, targets, np.array([False, False]))


def _first_correction_inputs(n_held_line, n_held_cp):
    targets = Targets(maps=np.zeros((18, 120, 120), np.uint8), origin=np.array([0, 0]), page_shape=(120, 120))
    shifts = first._Shifts(free=np.array([[60.0, 60.0], [62.0, 60.0]], np.float32),
                           free_n=np.array([[0.0, 1.0], [0.0, 1.0]], np.float32),
                           free_off=np.array([1.0, 1.0], np.float32))
    held = first._Inputs(held_line=np.tile(np.float32([[20.0, 20.0]]), (n_held_line, 1)),
                         held_n=np.tile(np.float32([[0.0, 1.0]]), (n_held_line, 1)),
                         held_cp=np.tile(np.float32([[100.0, 100.0]]), (n_held_cp, 1)),
                         corner_rec=np.zeros((0, 2), np.float32), corner_scan=np.zeros((0, 2), np.float32))
    return targets, shifts, held


def _run_first_correction(monkeypatch, n_held_line, n_held_cp):
    targets, shifts, held = _first_correction_inputs(n_held_line, n_held_cp)
    monkeypatch.setattr(first, "propagated_shifts", lambda *a: shifts)
    monkeypatch.setattr(first, "_inputs", lambda *a: held)
    sheet = SimpleNamespace(series="s", name="n")
    return first.first_correction(sheet, None, targets, None)


def test_the_first_correction_with_no_held_block_side_points_has_nothing_to_fit(monkeypatch):
    # the mean of an empty term is NaN, which would make the whole correction NaN
    with pytest.raises(NothingToFit, match="no held"):
        _run_first_correction(monkeypatch, 0, 3)


def test_the_first_correction_with_no_held_control_points_is_still_finite(monkeypatch):
    # the empty mean's NaN loss has an empty gradient into the grid, so the grid stays finite
    correction = _run_first_correction(monkeypatch, 3, 0)
    assert np.isfinite(correction.grid).all()


@pytest.mark.parametrize("device", DEVICES)
def test_the_first_correction_repeats_bit_for_bit(device, monkeypatch):
    monkeypatch.setenv("SFZMAPS_WARP_DEVICE", device)
    # thousands of held points on the same 4 grid nodes: added up in any other order, their pull changes in the bits
    first_run = _run_first_correction(monkeypatch, 5000, 3)
    second_run = _run_first_correction(monkeypatch, 5000, 3)
    assert isinstance(first_run.grid, np.ndarray) and first_run.grid.dtype == np.float32
    assert np.array_equal(first_run.grid, second_run.grid)


def test_held_sides_leaves_out_points_near_a_sides_ends():
    # 1 point per px (ft_per_px 2: points every 2 ft): the 3 points at each end of a side are within 2 px of it
    sheet = SimpleNamespace(series="s", name="n", ft_per_px=2.0, ink=np.zeros((40, 40), bool), block_line_half_width=2.0)
    sides = _sides(20, [0] * 10 + [1] * 10)
    d = np.zeros(20)
    d[[0, 1, 9]] = 5.0           # side 0: far from its line only at its ends
    d[15] = 5.0                  # side 1: far from its line in the middle
    held = first.held_sides(sheet, sides, d)
    assert held.tolist() == [True, False]
