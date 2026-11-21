import numpy as np
import rasterio.features
from shapely.geometry import Polygon

from sfzmaps.frame import decide


def test_cut_level_is_the_cost_ratio():
    assert decide.cut_level(3.0, 1.0) == 0.25


def test_level_itself_is_not_cut():
    q = np.full((5, 5), 0.25)
    assert not decide.decide(q, np.zeros((5, 5)), level=0.25).outside.any()


def test_hole_inside_the_map_is_filled():
    q = np.zeros((50, 50))
    q[:3] = 1
    q[20:25, 20:25] = 1
    d = decide.decide(q, np.zeros((50, 50)))
    assert d.outside[:3].all() and not d.outside[20:25, 20:25].any() and d.filled_ink == 25


def test_island_in_the_margin_is_dropped():
    q = np.ones((50, 50))
    q[10:40, 10:40] = 0
    q[2, 2] = 0
    d = decide.decide(q, np.zeros((50, 50)))
    assert d.outside[2, 2] and d.dropped_ink == 1


def test_two_maps_kept_only_when_listed():
    q = np.ones((40, 80))
    q[5:35, 5:35] = 0
    q[5:35, 45:75] = 0
    assert decide.decide(q, np.ones((40, 80)), maps=1).outside[20, 60]
    assert not decide.decide(q, np.ones((40, 80)), maps=2).outside[20, 60]


def test_diagonal_touch_does_not_reach_the_border():
    q = np.zeros((10, 10))
    q[0, 0] = 1
    q[1, 1] = 1
    d = decide.decide(q, np.zeros((10, 10)))
    assert d.outside[0, 0] and not d.outside[1, 1]


def test_all_cut_and_none_cut():
    assert decide.decide(np.ones((8, 8)), np.zeros((8, 8))).outside.all()
    assert not decide.decide(np.zeros((8, 8)), np.zeros((8, 8))).outside.any()


def test_outline_reproduces_the_region():
    outside = np.ones((40, 50), bool)
    outside[5:35, 5:45] = False
    outside[5:15, 30:45] = True  # L-shape
    rings = decide.outline(outside)
    raster = rasterio.features.rasterize([(Polygon(r), 1) for r in rings], out_shape=outside.shape)
    np.testing.assert_array_equal(raster.astype(bool), ~outside)
    assert len(rings[0]) <= 7
    assert decide.outline(np.ones((8, 8), bool)) == []
    assert len(decide.outline(np.zeros((8, 8), bool))[0]) <= 5
