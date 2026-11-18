import cv2
import numpy as np
import pytest
from scipy import ndimage
from skimage.morphology import skeletonize

from sfzmaps import config
from sfzmaps.sheetwarp import sheet, targets
from sfzmaps.transform import Correction


def test_placed_sheets_needs_every_placed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_DIR", tmp_path)
    whole = tmp_path / "1921" / "whole"
    short = tmp_path / "1921" / "short"
    for folder, names in ((whole, sheet.PLACED_FILES),
                          (short, [n for n in sheet.PLACED_FILES if n != "refine.json"])):
        folder.mkdir(parents=True)
        for n in names:
            (folder / n).write_text("")
    assert sheet.placed_sheets("1921") == ["whole"]
    assert sheet.placed_sheets("no-such-series") == []


def _targets():
    maps = np.zeros((18, 10, 10), np.uint8)
    maps[0] = 5   # running along x (0 degrees)
    maps[9] = 7   # running along y (90 degrees)
    return targets.Targets(maps=maps, origin=np.array([100, 200]), page_shape=(1000, 1000))


def test_distances_reads_the_map_of_each_points_direction():
    px = np.array([[103.0, 203.0], [104.0, 204.0], [50.0, 50.0]])
    step = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    d = targets.distances(_targets(), px, px - step, px + step)
    assert d[0] == 5 and d[1] == 7 and np.isinf(d[2])


def test_measure_with_a_correction_moves_the_points_first():
    tg = _targets()
    tg.maps[0, :, 5:] = 9
    px = np.array([[101.0, 203.0], [102.0, 205.0]])
    pa, pb = px - [3, 0], px + [3, 0]
    sides = targets.SidePoints(pts=px, sid=np.zeros(2, int), ta=pa, tb=pb, px=px, pa=pa, pb=pb)
    shift = np.array([3.0, 0.0])
    grid = np.stack([np.full((50, 50), 3.0), np.zeros((50, 50))]).astype(np.float32)
    correction = Correction(grid=grid, spacing=100.0, origin=np.array([-1000.0, -1000.0]))
    assert np.allclose(correction.u(px), shift)
    moved = targets.distances(tg, px + shift, pa + shift, pb + shift)
    assert (targets.measure(tg, sides, correction) == moved).all()
    assert (targets.measure(tg, sides) != moved).any()


# -- the distance transform and the crop's ink centres, against today's scipy versions --------------------------
@pytest.mark.parametrize("shape, paper_share", [((300, 400), 0.5), ((300, 400), 0.01), ((1000, 1200), 3e-6),
                                                ((3, 5000), 0.0)])
def test_the_distance_transform_equals_scipys(shape, paper_share):
    rng = np.random.default_rng(int(paper_share * 1e4))
    mask = rng.random(shape) >= paper_share
    mask[0, 0] = False                   # one paper pixel at least: (3, 5000) is 4999 px from its far end
    assert np.array_equal(sheet.distance_transform(mask), ndimage.distance_transform_edt(mask))
    assert np.array_equal(sheet.distance_transform(mask, cap=255),
                          np.minimum(ndimage.distance_transform_edt(mask), 255))


def test_the_distance_transform_equals_scipys_just_under_the_exact_limit():
    mask = np.ones((360, 360), bool)
    mask[0, 0] = False                   # the far corner is ~509 px away, just under EXACT_DISTANCE_BELOW_PX
    assert 500 < ndimage.distance_transform_edt(mask).max() < sheet.EXACT_DISTANCE_BELOW_PX
    assert np.array_equal(sheet.distance_transform(mask), ndimage.distance_transform_edt(mask))


def test_the_distance_transform_of_a_mask_without_paper_or_ink_equals_scipys():
    for mask in (np.ones((40, 60), bool), np.zeros((40, 60), bool)):
        assert np.array_equal(sheet.distance_transform(mask), ndimage.distance_transform_edt(mask))
        assert np.array_equal(sheet.distance_transform(mask, cap=20),
                              np.minimum(ndimage.distance_transform_edt(mask), 20))


def _page(only_across_and_down=False):
    """ink on paper: strokes 1-7 px wide every which way, rings and lettering; or only lines across and down (most
    directions then have no ink at all)"""
    page = np.full((300, 420), 255, np.uint8)
    for k, width in enumerate((1, 2, 3, 5, 7)):
        cv2.line(page, (20, 30 + 50 * k), (400, 30 + 50 * k), 0, width)
        cv2.line(page, (30 + 80 * k, 15), (30 + 80 * k, 290), 0, width)
    if not only_across_and_down:
        for k, width in enumerate((1, 2, 4)):
            cv2.line(page, (10, 290 - 20 * k), (410, 20 + 40 * k), 0, width)
            cv2.circle(page, (120 + 90 * k, 150), 25 + 10 * k, 0, width)
        cv2.putText(page, "BLOCK 1300", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    return page < 128


def _todays_direction_maps(ink, block_line_half_width):
    """fit_targets' maps as they were computed before the ink centres were shared (scipy's distances)"""
    centres = skeletonize(ink) & (ndimage.distance_transform_edt(ink) <= block_line_half_width)
    f = ndimage.gaussian_filter(ink.astype(np.float32), 1.0)
    gx, gy = ndimage.sobel(f, 1), ndimage.sobel(f, 0)
    Jxx, Jxy, Jyy = (ndimage.gaussian_filter(v, 2.0) for v in (gx * gx, gx * gy, gy * gy))
    cdir = (0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy)) + 90.0) % 180
    maps = np.zeros((18, *ink.shape), np.uint8)
    for i, b in enumerate(np.arange(0, 180, 10.0)):
        m = centres & (np.abs((cdir - b + 90) % 180 - 90) <= 20.0)
        maps[i] = np.minimum(ndimage.distance_transform_edt(~m), 255).astype(np.uint8)
    return maps


@pytest.mark.parametrize("only_across_and_down", [False, True])
def test_direction_maps_from_the_shared_ink_centres_equal_todays(only_across_and_down):
    ink = _page(only_across_and_down)
    maps = targets.direction_maps(targets.ink_centres(ink), 1.5)
    assert maps.dtype == np.uint8 and maps.shape == (18, *ink.shape)
    assert np.array_equal(maps, _todays_direction_maps(ink, 1.5))
    assert (maps == 0).any()
    if only_across_and_down:
        assert (maps == 255).any()          # a direction with no ink at all: scipy's distances, capped
