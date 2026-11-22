import numpy as np
import rasterio.features
from shapely.geometry import Polygon

from sfzmaps.frame import decide


def _p(q) -> np.ndarray:
    """Three-class chances from a two-class q = P(outside): map 1 - q, outside q, removed block 0."""
    q = np.asarray(q, float)
    return np.stack([1 - q, q, 0 * q])


def _framed() -> np.ndarray:
    """50 x 50 chances: all map, but a frame along the top three rows."""
    p = np.zeros((3, 50, 50))
    p[0] = 1
    p[:, :3] = np.array([1e-3, 1 - 2e-3, 1e-3])[:, None, None]
    return p


def test_cut_level_is_even_odds():
    assert decide.cut_level() == 0.5


def test_level_itself_is_not_cut():
    q = np.full((5, 5), 0.5)
    assert not decide.decide(_p(q), np.zeros((5, 5))).deleted.any()


def test_hole_inside_the_map_is_filled():
    q = np.zeros((50, 50))
    q[:3] = 1
    q[20:25, 20:25] = 1
    d = decide.decide(_p(q), np.zeros((50, 50)))
    assert d.outside[:3].all() and not d.deleted[20:25, 20:25].any() and d.filled_ink == 25


def test_island_in_the_margin_is_dropped():
    q = np.ones((50, 50))
    q[10:40, 10:40] = 0
    q[2, 2] = 0
    d = decide.decide(_p(q), np.zeros((50, 50)))
    assert d.outside[2, 2] and d.dropped_ink == 1


def test_filled_ink_counts_only_what_stays_map():
    """A cut px enclosed by an island that is then dropped is not put back: its ink is dropped, not filled."""
    q = np.ones((50, 50))
    q[10:40, 10:40] = 0
    q[1:6, 1:6] = 0
    q[3, 3] = 1
    d = decide.decide(_p(q), np.zeros((50, 50)))
    assert d.outside[1:6, 1:6].all() and d.filled_ink == 0 and d.dropped_ink == 25


def test_two_maps_kept_only_when_listed():
    q = np.ones((40, 80))
    q[5:35, 5:35] = 0
    q[5:35, 45:75] = 0
    assert decide.decide(_p(q), np.ones((40, 80)), maps=1).outside[20, 60]
    assert not decide.decide(_p(q), np.ones((40, 80)), maps=2).deleted[20, 60]


def test_diagonal_touch_does_not_reach_the_border():
    q = np.zeros((10, 10))
    q[0, 0] = 1
    q[1, 1] = 1
    d = decide.decide(_p(q), np.zeros((10, 10)))
    assert d.outside[0, 0] and not d.deleted[1, 1]


def test_all_cut_and_none_cut():
    assert decide.decide(_p(np.ones((8, 8))), np.zeros((8, 8))).outside.all()
    assert not decide.decide(_p(np.zeros((8, 8))), np.zeros((8, 8))).deleted.any()


def test_removed_block_mid_map_is_a_hole():
    p = _framed()
    p[:, 20:26, 20:26] = np.array([0.1, 0.1, 0.8])[:, None, None]                    # a compass rose mid-map
    d = decide.decide(p, np.zeros((50, 50)))
    assert d.removed[22, 22] and not d.outside[22, 22] and d.deleted[22, 22] and d.removed_ink == 36
    assert d.outside[:3].all() and int(d.deleted[3:].sum()) == 36                      # nothing else deleted


def test_map_called_frame_mid_map_is_kept():                                          # the 1948 commercial band
    p = _framed()
    p[:, 20:26, 20:26] = np.array([0.05, 0.9, 0.05])[:, None, None]
    d = decide.decide(p, np.zeros((50, 50)))
    assert not d.deleted[22, 22] and d.filled_ink == 36


def test_removed_block_on_the_frame_is_outside():
    """A block the cut joins to the frame is outside the frame, not a hole."""
    p = _framed()
    p[:, 3:9, 20:26] = np.array([0.1, 0.1, 0.8])[:, None, None]
    d = decide.decide(p, np.zeros((50, 50)))
    assert d.outside[3:9, 20:26].all() and not d.removed.any() and d.removed_ink == 0


def test_outline_has_the_removed_hole():
    deleted = np.ones((40, 40), bool); deleted[5:35, 5:35] = False; deleted[15:20, 15:20] = True
    polys = decide.outline(deleted)
    raster = rasterio.features.rasterize(
        [(Polygon(q["exterior"], q["holes"]), 1) for q in polys], out_shape=deleted.shape)
    np.testing.assert_array_equal(raster.astype(bool), ~deleted)


def test_outline_reproduces_the_region():
    outside = np.ones((40, 50), bool)
    outside[5:35, 5:45] = False
    outside[5:15, 30:45] = True  # L-shape
    polys = decide.outline(outside)
    raster = rasterio.features.rasterize(
        [(Polygon(q["exterior"], q["holes"]), 1) for q in polys], out_shape=outside.shape)
    np.testing.assert_array_equal(raster.astype(bool), ~outside)
    assert len(polys[0]["exterior"]) <= 7 and polys[0]["holes"] == []
    assert decide.outline(np.ones((8, 8), bool)) == []
    assert len(decide.outline(np.zeros((8, 8), bool))[0]["exterior"]) <= 5
