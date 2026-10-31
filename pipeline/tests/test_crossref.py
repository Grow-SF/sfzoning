import json
import math

import cv2
import numpy as np

from sfzmaps import crossref
from sfzmaps.score import SheetScore


def _score(passed, p50, p95=8.0, limit=5.0):
    s = SheetScore.empty(None if passed else "holdout_median")
    s.passed, s.holdout_p50_ft, s.holdout_p95_ft = passed, p50, p95
    s.pass_median_limit_ft, s.pass_p95_limit_ft = limit, 12.0
    return s


def test_better_prefers_passing_and_lower_error():
    assert crossref.better(_score(False, 6.0), _score(True, 4.0))
    assert not crossref.better(_score(True, 3.0), _score(False, 2.0))     # never lose a pass
    assert not crossref.better(_score(True, 3.0), _score(True, 4.0))
    assert crossref.better(_score(False, 7.0), _score(False, 6.0))
    assert not crossref.better(_score(False, 6.0), _score(False, 6.0))
    assert not crossref.better(_score(False, 6.0), None)


def test_better_places_an_unplaced_sheet_only_when_close():
    unplaced = SheetScore.empty("insufficient_anchors")
    assert math.isnan(unplaced.holdout_p50_ft)
    assert crossref.better(unplaced, _score(False, 9.0, 20.0))            # within 3x the limits
    assert not crossref.better(unplaced, _score(False, 40.0, 90.0))       # placed, but way off


def test_candidates_same_number_nearest_edition_first():
    numbers = {("1972", "a"): "1", ("1968", "b"): "1", ("1980", "c"): "1", ("1960", "d"): "1",
               ("1968", "e"): "2"}
    passed = {("1968", "b"), ("1980", "c"), ("1960", "d"), ("1968", "e")}
    assert crossref.candidates("1972", "1", passed, numbers) == [("1968", "b"), ("1980", "c"),
                                                                 ("1960", "d")]
    assert crossref.candidates("1972", None, passed, numbers) == []


def _page(tmp, name, img):
    d = tmp / name
    d.mkdir()
    cv2.imwrite(str(d / "page.png"), img)
    (d / "page.json").write_text(json.dumps({"kind": "map", "sheet_number": "1", "mask_boxes": []}))
    return d


def test_match_recovers_a_shifted_scaled_copy(tmp_path):
    rng = np.random.default_rng(0)
    img = np.full((1600, 2000), 255, np.uint8)
    for _ in range(400):   # irregular linework, so features are distinctive
        x, y = rng.integers(50, 1950), rng.integers(50, 1550)
        cv2.line(img, (int(x), int(y)), (int(x + rng.integers(-120, 120)), int(y + rng.integers(-120, 120))),
                 0, int(rng.integers(2, 6)))
    A = np.array([[0.97, 0.0, 40.0], [0.0, 0.97, -25.0]])
    moved = cv2.warpAffine(img, A, (2000, 1600), borderValue=255)
    m = crossref.match(_page(tmp_path, "f", img), _page(tmp_path, "r", moved))
    assert m is not None
    H, n = m
    assert n >= crossref.MIN_INLIERS
    p = cv2.perspectiveTransform(np.array([[[1000.0, 800.0]]]), H)[0, 0]
    assert np.allclose(p, A @ [1000.0, 800.0, 1.0], atol=3.0)


def test_match_rejects_unrelated_pages(tmp_path):
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, (1600, 2000)).astype(np.uint8) * 255
    b = rng.integers(0, 2, (1600, 2000)).astype(np.uint8) * 255
    assert crossref.match(_page(tmp_path, "a", a), _page(tmp_path, "b", b)) is None


def test_backup_and_restore_round_trip(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "transform.json").write_text("old")
    (d / "score.json").write_text("old-score")
    saved = crossref.backup(d)
    (d / "transform.json").write_text("new")
    (d / "rough.json").write_text("new-rough")      # did not exist before: must be removed
    crossref.restore(d, saved)
    assert (d / "transform.json").read_text() == "old"
    assert (d / "score.json").read_text() == "old-score"
    assert not (d / "rough.json").exists()
    assert not saved.exists()


def test_plausible_homography_rejects_degenerate_fits():
    assert crossref.plausible_homography(np.array([[0.98, 0.01, 40], [-0.01, 0.98, -20], [0, 0, 1.0]]))
    assert not crossref.plausible_homography(np.diag([0.001, 0.001, 1.0]))          # collapsed
    assert not crossref.plausible_homography(np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]]))  # skewed
    assert not crossref.plausible_homography(np.array([[1.0, 0, 0], [0, 1.0, 0], [1e-3, 0, 1.0]]))


def test_warp_refuses_a_degenerate_transform(tmp_path):
    from sfzmaps import runner
    from sfzmaps.transform import GCP, SheetTransform
    img = np.full((400, 400), 255, np.uint8)
    png = tmp_path / "p.png"
    cv2.imwrite(str(png), img)
    P = np.array([[0, 0], [399, 0], [0, 399], [399, 399]], float)
    W = P * 0.0005 + 6_000_000.0                       # 0.0005 ft/px, as the stuck run had
    t = SheetTransform([GCP(*p, *q, "auto") for p, q in zip(P, W)], smoothing=1.0).fit()
    import pytest
    with pytest.raises(RuntimeError, match="refusing to warp"):
        runner._warp_checked(png, t, [], tmp_path / "w.tif")
