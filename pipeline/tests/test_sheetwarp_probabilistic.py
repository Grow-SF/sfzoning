from types import SimpleNamespace

import numpy as np

from sfzmaps.sheetwarp.probabilistic import block_line_weights
from sfzmaps.sheetwarp.street_edges_and_later_blocks import unexplained_weight
from sfzmaps.sheetwarp.targets import Targets


def test_block_line_weights_favour_the_thin_strokes():
    page = np.full((200, 200), 255, np.uint8)
    for x in (30, 60, 90, 120):          # thin vertical lines, 1 px
        page[10:190, x] = 0
    for y in (40, 100, 160):             # heavy horizontal bars, 6 px
        page[y - 3:y + 3, 10:190] = 0
    targets = Targets(maps=np.zeros((18, 200, 200), np.uint8), origin=np.array([0, 0]), page_shape=(200, 200))
    weights = block_line_weights(SimpleNamespace(page=page, series="s", name="n"), targets)
    assert weights.shape == (18, 200, 200) and weights.dtype == np.float16
    thin_centres = weights[9][50:90, [30, 60, 90, 120]]       # vertical ink: 90 degrees, map 9; clear of the bars
    heavy_centres = weights[0][[40, 100, 160]][:, 40:80]   # horizontal ink: 0 degrees, map 0
    assert thin_centres.mean() > 0.9
    assert heavy_centres.mean() < 0.1


def test_street_edge_points_count_by_how_little_blocks_explain_them():
    s = 30.0
    w = unexplained_weight(np.array([0.0, 5.0, 300.0]), s)
    assert w[0] == 0
    assert 0 < w[1] < 0.1
    assert w[2] > 0.999
