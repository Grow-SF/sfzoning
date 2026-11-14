from types import SimpleNamespace

import numpy as np
import pytest

from sfzmaps.sheetwarp import first
from sfzmaps.sheetwarp.sheet import NothingToFit
from sfzmaps.sheetwarp.targets import SidePoints, Targets


def _sides(n_points, sid):
    px = np.column_stack([np.arange(n_points, dtype=float) + 10, np.full(n_points, 10.0)])
    return SidePoints(pts=px, sid=np.asarray(sid), ta=px - [3, 0], tb=px + [3, 0], px=px, pa=px - [3, 0], pb=px + [3, 0])


def test_propagation_with_every_side_held_has_nothing_to_move():
    sides = _sides(10, [0] * 5 + [1] * 5)
    targets = Targets(maps=np.zeros((18, 40, 40), np.uint8), origin=np.array([0, 0]), page_shape=(40, 40))
    with pytest.raises(NothingToFit):
        first.propagated_shifts(SimpleNamespace(series="s", name="n"), sides, targets, np.array([True, True]))


def test_held_sides_leaves_out_points_near_a_sides_ends():
    # 1 point per px (ft_per_px 2: points every 2 ft): the 3 points at each end of a side are within 2 px of it
    sheet = SimpleNamespace(series="s", name="n", ft_per_px=2.0, ink=np.zeros((40, 40), bool), block_line_half_width=2.0)
    sides = _sides(20, [0] * 10 + [1] * 10)
    d = np.zeros(20)
    d[[0, 1, 9]] = 5.0           # side 0: far from its line only at its ends
    d[15] = 5.0                  # side 1: far from its line in the middle
    held = first.held_sides(sheet, sides, d)
    assert held.tolist() == [True, False]
