import numpy as np

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
