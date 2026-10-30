"""Block-shape extraction (linework.extract_blocks) on synthetic and real scans.

Real-crop hand counts
---------------------
Counting rule (applied identically by eye and in code): a block is any closed
shape -- outlined, solid, hatched, stippled, or a mix -- separated from every
neighbour by a continuous white channel (a street or an alley; even a 2-3 px
hairline gap between two outlines counts).  Shapes split only by a single
printed line (an alley drawn as one line, a lot line, a district line through
a block) are ONE block.  Shapes touching the crop edge are excluded
(``extract_blocks`` drops every block whose material touches the image border
or leaves the mask, so partial blocks never produce corners).

crop_1948.png: page 1948/7 (work/1948/1/page.png)[1660:3200, 1380:2850],
  grayscale.  North Beach, Chestnut..Pacific x Mason..Kearny: outlined white
  blocks, solid-black commercial blocks and strips touching white ones,
  a diagonal-hatched strip (block 48), stippled (cross-hatch) industrial
  blocks, Columbus Ave cutting blocks into triangles, faded dotted outlines.
  Hand count by block row (top to bottom, named by printed block numbers):
    48-49 row 9, 66-68 row 8, 71-76 row 8, 89-94 row 13, 97-101 row 7,
    118-122 row 9, 125-129 row 6, 149-152 row 8, 155-158 row 13  -> 81.
crop_1968.png: 1968 page 8 (work/1968/20180102110513967-p08/page.png)
  [2900:4400, 2400:3900].  Mission / Inner Mission, Chattanooga..Treat:
  1-bit outlined blocks split into strips by alleys, heavy R-/C-/P district
  boundaries running down street centrelines (Orange Alley, Virgil, Poplar,
  Treat, South Van Ness) and around blocks, bold R-3/R-4/R-5/C-2 lettering
  inside blocks and straddling streets, street names inside streets.
  Hand count by horizontal band (y px in the crop):
    0-130: 7, 160-380: 12, 400-620: 14, 640-860: 16, 880-980: 7,
    990-1095: 7, 1115-1220: 6, 1230-1330: 6, 1355-1450: 7  -> 82.
"""
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from sfzmaps import linework
from synthetic import make_sheet

FIX = Path(__file__).parent / "fixtures"

HAND_COUNT_1948 = 81
HAND_COUNT_1968 = 82


def _grid_blocks(n=3, w=400.0, h=260.0, street=60.0, x0=6_005_000.0, y0=2_115_000.0):
    rows, geoms = [], []
    for i in range(n):
        for j in range(n):
            x = x0 + j * (w + street)
            y = y0 + i * (h + street)
            rows.append(f"{i * n + j + 1:04d}")
            geoms.append(box(x, y, x + w, y + h))
    return gpd.GeoDataFrame({"block_num": rows}, geometry=geoms, crs="EPSG:2227")


def _wobble(p: np.ndarray) -> np.ndarray:
    """Mild smooth paper distortion: sheet px -> undistorted px."""
    x, y = p[..., 0], p[..., 1]
    return np.stack([x + 4.0 * np.sin(y / 300.0), y + 3.0 * np.cos(x / 250.0)], axis=-1)


def _truth_px(sheet, blocks) -> list[Polygon]:
    out = []
    for g in blocks.geometry:
        dense = np.asarray(g.segmentize(5.0).exterior.coords)
        out.append(Polygon(sheet.world_to_px(dense)))
    return out


def _check_synthetic(style):
    blocks = _grid_blocks()
    sheet = make_sheet(blocks, px_per_ft=0.55, rotation_deg=8.0, distortion=_wobble, style=style)
    polys = linework.extract_blocks(sheet.img)
    assert len(polys) == 9
    for t in _truth_px(sheet, blocks):
        best = max(t.intersection(p).area / t.union(p).area for p in polys)
        assert best > 0.9


def test_extract_blocks_synthetic_outline():
    _check_synthetic("outline")


def test_extract_blocks_synthetic_filled():
    _check_synthetic("filled")


def test_make_sheet_px_to_world_round_trips():
    blocks = _grid_blocks()
    sheet = make_sheet(blocks, px_per_ft=0.55, rotation_deg=33.0, distortion=_wobble, style="outline",
                       shift_blocks={"0005": (30.0, -20.0)})
    world = np.array([[6_005_100.0, 2_115_100.0], [6_005_900.0, 2_115_700.0]])
    back = sheet.true_px_to_world(sheet.world_to_px(world))
    assert np.abs(back - world).max() < 1e-3
    assert {lb.text for lb in sheet.labels} == set(blocks.block_num)
    # The shifted block's label sits at its shifted centroid.
    lb = next(lb for lb in sheet.labels if lb.text == "0005")
    c = blocks.geometry[blocks.block_num == "0005"].iloc[0].centroid
    w = sheet.true_px_to_world(np.array([[lb.x, lb.y]], float))[0]
    assert np.hypot(w[0] - (c.x + 30.0), w[1] - (c.y - 20.0)) < 4.0


def test_block_corners_px_rectangles():
    polys = [box(0, 0, 100, 60), Polygon([(200, 0), (300, 0), (300, 80), (250, 80), (250, 40), (200, 40)])]
    c = linework.block_corners_px(polys, street_px=20.0)
    assert c.shape == (4 + 6, 2)


def _real(name):
    img = cv2.imread(str(FIX / name), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    return linework.extract_blocks(img)


def test_extract_blocks_real_1948_crop():
    polys = _real("crop_1948.png")
    assert abs(len(polys) - HAND_COUNT_1948) <= 1


def test_extract_blocks_real_1968_crop():
    polys = _real("crop_1968.png")
    assert abs(len(polys) - HAND_COUNT_1968) <= 1
    areas = np.array([p.area for p in polys])
    # Controller ruling R12: at most one merged block.  Known merge: the C-2
    # block (overlay label 70, x~1000-1415, y~1230-1325), where a street
    # segment boxed in by a jogging district line around the "C-2" lettering
    # is read as block and bridges the two blocks on either side.
    assert (areas > 1.5 * np.median(areas)).sum() <= 1


def test_extract_blocks_street_margin_stays_street():
    """crop_1948_margin.png: page 1948/1 [2497:3197, 2880:3580], grayscale.  The
    solid block left of the vertical street (interior point (300, 420)) has its
    right edge (ink) at x = 344.  Specks and a ragged kerb break the long street
    runs near that edge, so a ~7 px white margin was once read as block material
    (it joins the alley mouth) and the block grew into the street to x ~ 351."""
    from shapely.geometry import Point

    polys = _real("crop_1948_margin.png")
    hit = [p for p in polys if p.contains(Point(300, 420))]
    assert len(hit) == 1
    assert abs(max(x for x, _ in hit[0].exterior.coords) - 344) <= 2


def test_extract_blocks_exclude_region():
    """exclude=True marks excluded pixels (same polarity as reading.exclusion_mask)."""
    blocks = _grid_blocks()
    sheet = make_sheet(blocks, px_per_ft=0.55, rotation_deg=0.0, distortion=lambda p: p, style="outline")
    h, w = sheet.img.shape
    exclude = np.zeros((h, w), bool)
    exclude[:, : w // 2] = True
    polys = linework.extract_blocks(sheet.img, exclude)
    assert polys
    assert all(p.bounds[0] > w // 2 for p in polys)
    assert len(polys) < 9


def test_block_corners_px_ignores_text_notch():
    # 400 x 200 px rectangle with a 6 px deep, 30 px wide notch on the top edge.
    ring = [(0, 0), (180, 0), (180, 6), (210, 6), (210, 0), (400, 0), (400, 200), (0, 200)]
    c = linework.block_corners_px([Polygon(ring)], street_px=25.0)
    assert len(c) == 4


def test_make_sheet_rejects_non_contractive_distortion():
    # p -> 3p - 2*500 has fixed point 500 but expands errors 3x per step.
    with pytest.raises(ValueError, match="did not converge"):
        make_sheet(_grid_blocks(), px_per_ft=0.55, rotation_deg=0.0,
                   distortion=lambda p: 3.0 * p - 1000.0, style="outline")


def _px_grid(n, w, h, street):
    from shapely.geometry import box as _box
    return [_box(j * (w + street), i * (h + street), j * (w + street) + w, i * (h + street) + h)
            for i in range(n) for j in range(n)]


def test_estimate_street_px_wide_streets_measured():
    px, measured = linework.estimate_street_px(_px_grid(3, 1500, 1000, 200))
    assert measured and abs(px - 200) < 5


def test_estimate_street_px_fallback_not_measured():
    from shapely.geometry import box as _box
    assert linework.estimate_street_px([_box(0, 0, 10, 10)]) == (linework.DEFAULT_STREET_PX, False)
    assert linework.estimate_street_px([]) == (linework.DEFAULT_STREET_PX, False)


# -- corner quality ---------------------------------------------------------------------

def test_block_corner_features_rows_match_block_corners_px():
    polys = [box(0, 0, 100, 60), Polygon([(200, 0), (300, 0), (300, 80), (250, 80), (250, 40), (200, 40)])]
    f = linework.block_corner_features(polys, street_px=20.0)
    c = linework.block_corners_px(polys, street_px=20.0)
    assert np.allclose(np.column_stack([f["x"], f["y"]]), c)
    assert list(f["n_corners"]) == [4] * 4 + [6] * 6
    assert list(f["block"]) == [0] * 4 + [1] * 6


def test_corner_quality_clean_rectangle_is_one():
    f = linework.block_corner_features([box(0, 0, 400, 250)], street_px=25.0)
    assert np.allclose(linework.corner_quality(f), 1.0)
    assert len(linework.good_corners_px([box(0, 0, 400, 250)], street_px=25.0)) == 4


def test_corner_quality_short_jog_is_poor():
    # 400 x 250 block with a 30 px (1.2 street widths) step in its top edge: the jog's
    # corners are fixed by a short edge; the four outer corners stay good.
    ring = [(0, 0), (200, 0), (200, 30), (400, 30), (400, 250), (0, 250)]
    f = linework.block_corner_features([Polygon(ring)], street_px=25.0)
    q = linework.corner_quality(f)
    jog = f["x"] == 200
    assert jog.sum() == 2 and (q[jog] < linework.MIN_CORNER_QUALITY).all()
    assert (q[~jog] >= linework.MIN_CORNER_QUALITY).all()


def test_corner_quality_acute_and_odd_angles_are_poor():
    # A triangle at a diagonal street: its 45-degree corners turn 135 degrees.
    tri = Polygon([(0, 0), (400, 0), (0, 400)])
    f = linework.block_corner_features([tri], street_px=25.0)
    q = linework.corner_quality(f)
    ad = np.abs(f["turn_deg"] - 90)
    assert (q[ad > 30] == 0).all() and (q[ad < 1] > 0.9).all()


def test_corner_quality_complex_block_is_poor():
    # A staircase outline (a block merged with a bold district line drawn around it):
    # 20 corners, every edge long and every turn square, yet none is trusted.
    pts = [(0.0, 0.0)]
    for _ in range(9):
        x, y = pts[-1]
        pts += [(x + 80, y), (x + 80, y + 80)]
    pts += [(0.0, pts[-1][1])]
    poly = Polygon(pts)
    f = linework.block_corner_features([poly], street_px=25.0)
    assert f["n_corners"][0] == 20
    assert (linework.corner_quality(f) < linework.MIN_CORNER_QUALITY).all()
    assert len(linework.good_corners_px([poly], street_px=25.0)) == 0


def test_corner_quality_scale_free():
    # The same block at two scan scales (street px scaling with it) scores alike.
    from shapely.affinity import scale as _scale
    p = Polygon([(0, 0), (200, 0), (200, 30), (400, 30), (400, 250), (0, 250)])
    q1 = linework.corner_quality(linework.block_corner_features([p], street_px=25.0))
    q2 = linework.corner_quality(linework.block_corner_features(
        [_scale(p, 0.6, 0.6, origin=(0, 0))], street_px=15.0))
    assert np.allclose(q1, q2)


def test_good_corners_px_empty():
    assert linework.good_corners_px([], street_px=25.0).shape == (0, 2)


# -- scale-aware constants --------------------------------------------------------------

def test_sheet_scale_band():
    lo, hi = linework.TUNED_STREET_PX
    assert linework.sheet_scale(lo) == linework.sheet_scale(hi) == linework.sheet_scale(25.0) == 1.0
    assert abs(linework.sheet_scale(16.0) - 16.0 / lo) < 1e-9
    assert abs(linework.sheet_scale(40.0) - 40.0 / hi) < 1e-9
    assert linework.sheet_scale(2.0) == linework.SCALE_RANGE[0]
    assert linework.sheet_scale(float("nan")) == 1.0


def test_extract_blocks_synthetic_coarse_scan():
    """A coarse scan (~4 ft/px: 64 ft streets are ~16 px): the constants scale down."""
    blocks = _grid_blocks(street=64.0)
    sheet = make_sheet(blocks, px_per_ft=0.25, rotation_deg=8.0, distortion=_wobble, style="outline")
    polys = linework.extract_blocks(sheet.img)
    assert len(polys) == 9
    for t in _truth_px(sheet, blocks):
        best = max(t.intersection(p).area / t.union(p).area for p in polys)
        assert best > 0.95   # fixed (unscaled) constants: 0.947 on one block
