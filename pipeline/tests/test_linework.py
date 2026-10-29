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
