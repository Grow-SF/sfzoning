import time

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from sfzmaps import anchors, config, linework, refine
from sfzmaps.sources.datasf import Reference, block_corners
from sfzmaps.transform import GCP, SheetTransform
from synthetic import make_sheet

PX_PER_FT = 0.55
STRETCH_FT = 25.0  # peak paper-stretch displacement


def _irregular_grid(n=20, seed=3, x0=6_005_000.0, y0=2_115_000.0):
    """n x n rectangular blocks with uneven block and street sizes (not periodic)."""
    rng = np.random.default_rng(seed)
    ws, hs = rng.uniform(300, 500, n), rng.uniform(220, 320, n)
    sx, sy = rng.uniform(50, 80, n), rng.uniform(50, 80, n)
    xs = x0 + np.concatenate([[0], np.cumsum(ws + sx)[:-1]])
    ys = y0 + np.concatenate([[0], np.cumsum(hs + sy)[:-1]])
    rows, geoms = [], []
    for i in range(n):
        for j in range(n):
            rows.append(f"{i * n + j + 1:04d}")
            geoms.append(box(xs[j], ys[i], xs[j] + ws[j], ys[i] + hs[i]))
    return gpd.GeoDataFrame({"block_num": rows}, geometry=geoms, crs="EPSG:2227")


def _paper_stretch(p):
    """Smooth low-frequency paper stretch, sheet px -> undistorted px, |d| <= 25 ft."""
    a = STRETCH_FT * PX_PER_FT / np.sqrt(2)
    x, y = p[..., 0], p[..., 1]
    return np.stack([x + a * np.sin(x / 1700 + 0.3) * np.cos(y / 2100),
                     y + a * np.cos(x / 1900) * np.sin(y / 1500 + 1.0)], axis=-1)


def _ref(blocks):
    inter = gpd.GeoDataFrame({"node_cnn": [], "streets": []}, geometry=[], crs="EPSG:2227")
    return Reference(blocks, block_corners(blocks), inter, None)


def _perturbed_start(sheet, ref, shift_ft, rot_deg):
    """Task 8 rough fit from the sheet's labels, then its world side moved by a rigid
    rotation about the GCP centre plus a shift."""
    fit = anchors.rough_fit(anchors.match_labels(sheet.labels, ref))
    W = np.array([[g.wx, g.wy] for g in fit.transform.gcps])
    c = W.mean(0)
    th = np.radians(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Wp = (W - c) @ R.T + c + np.asarray(shift_ft)
    gcps = [GCP(g.px, g.py, float(w[0]), float(w[1]), "anchor")
            for g, w in zip(fit.transform.gcps, Wp)]
    return SheetTransform(gcps, smoothing=config.TPS_SMOOTHING).fit()


def _inliers(res):
    return [m for m in res.matches if m.label == "inlier"]


def _inlier_errors(res, sheet):
    px = np.array([[m.px, m.py] for m in _inliers(res)])
    return np.linalg.norm(res.transform.to_world(px) - sheet.true_px_to_world(px), axis=1)


@pytest.fixture(scope="module")
def big():
    blocks = _irregular_grid()
    sheet = make_sheet(blocks, px_per_ft=PX_PER_FT, rotation_deg=7.0,
                       distortion=_paper_stretch, style="outline")
    corners = linework.block_corners_px(linework.extract_blocks(sheet.img))
    return blocks, sheet, _ref(blocks), corners


@pytest.fixture(scope="module")
def big_result(big):
    blocks, sheet, ref, corners = big
    start = _perturbed_start(sheet, ref, (40.0 / np.sqrt(2), 40.0 / np.sqrt(2)), 1.0)
    t = time.perf_counter()
    res = refine.refine(corners, ref.corners, start)
    return res, time.perf_counter() - t


def test_refine_converges_on_synthetic(big, big_result):
    _, sheet, _, _ = big
    res, _ = big_result
    err = _inlier_errors(res, sheet)
    assert np.quantile(err, 0.95) < 3.0


def test_refine_uses_many_points(big_result):
    res, _ = big_result
    assert len(_inliers(res)) > 200


def test_refine_labels_every_corner_once(big, big_result):
    _, _, _, corners = big
    res, _ = big_result
    assert len(res.matches) == len(corners)
    assert np.allclose([[m.px, m.py] for m in res.matches], corners)
    assert {m.label for m in res.matches} <= {"inlier", "outlier", "no_equivalent"}
    assert 1 <= res.iterations <= 30


def test_refine_final_transform_is_auto_corners_only(big_result):
    res, _ = big_result
    assert all(g.source == "auto" for g in res.transform.gcps)
    assert len(res.transform.gcps) == len(_inliers(res))


def test_refine_is_fast(big_result):
    res, secs = big_result
    assert secs < 30.0, f"refine took {secs:.1f}s"


def test_refine_half_street_offset_does_not_lock_wrong_side(big):
    """Start off by about half a street width (~32 ft) in x with no rotation."""
    _, sheet, ref, corners = big
    start = _perturbed_start(sheet, ref, (32.0, 0.0), 0.0)
    res = refine.refine(corners, ref.corners, start)
    assert np.quantile(_inlier_errors(res, sheet), 0.95) < 3.0
    assert len(_inliers(res)) > 200


def test_refine_recovers_large_rough_offset(big):
    """~100 ft and ~1.9 deg off (corners of the sheet ~250 ft off): beyond the first
    radius, so the coarse translation vote has to bring it in."""
    _, sheet, ref, corners = big
    start = _perturbed_start(sheet, ref, (-10.5, 99.5), -1.9)
    res = refine.refine(corners, ref.corners, start)
    assert np.quantile(_inlier_errors(res, sheet), 0.95) < 3.0
    assert len(_inliers(res)) > 0.9 * len(corners)


def test_refine_survives_spurious_corners(big):
    """~35% extra corners scattered around real ones (text notches, lettering)."""
    _, sheet, ref, corners = big
    rng = np.random.default_rng(7)
    k = int(0.35 * len(corners))
    a = corners[rng.integers(0, len(corners), k)]
    extra = a + rng.normal(0, 25, (k, 2))  # px: scattered ~45 ft about real corners
    noisy = np.vstack([corners, extra])
    start = _perturbed_start(sheet, ref, (28.0, -28.0), -1.0)
    res = refine.refine(noisy, ref.corners, start)
    assert len(res.matches) == len(noisy)
    assert np.quantile(_inlier_errors(res, sheet), 0.95) < 3.0
    assert len(_inliers(res)) > 200


# --- small, image-free cases ---------------------------------------------------

def _small_case():
    blocks = _irregular_grid(n=6, seed=1)
    ref = _ref(blocks)
    W = np.array([[p.x, p.y] for p in ref.corners.geometry])
    c = W.mean(0)
    # pixel = similarity of world (y down), 0.5 px/ft
    P = np.c_[(W[:, 0] - c[0]) * 0.5 + 1500, -(W[:, 1] - c[1]) * 0.5 + 1000]
    return ref, W, P, c


def _start_from(P, W, shift):
    idx = [0, len(P) // 3, 2 * len(P) // 3, len(P) - 1, len(P) // 2]
    return SheetTransform([GCP(*P[i], *(W[i] + shift), "anchor") for i in idx],
                          smoothing=config.TPS_SMOOTHING).fit()


def test_refine_far_corner_is_no_equivalent():
    ref, W, P, c = _small_case()
    far = np.array([[P[:, 0].max() + 400, P[:, 1].mean()]])  # ~800 ft east of everything
    res = refine.refine(np.vstack([P, far]), ref.corners, _start_from(P, W, (20.0, -15.0)))
    assert res.matches[-1].label == "no_equivalent"
    assert sum(m.label == "inlier" for m in res.matches) >= 0.95 * len(P)
    assert all(m.dist_ft < 1.0 for m in res.matches[:-1] if m.label == "inlier")


def test_refine_does_not_chase_moved_block():
    """A scan block drawn 20 ft off its modern place stays where the paper puts it."""
    ref, W, P, c = _small_case()
    nums = np.asarray(ref.corners["block_num"])
    moved = nums == "0015"
    shift_px = np.array([20.0 * 0.5, 0.0])  # 20 ft east on paper
    P2 = P.copy()
    P2[moved] += shift_px
    res = refine.refine(P2, ref.corners, _start_from(P, W, (10.0, 10.0)))
    labels = np.array([m.label for m in res.matches])
    assert not (labels[moved] == "inlier").any()
    # true world of the moved corners is modern + 20 ft east; the warp must not pull
    # them back onto the modern corners
    got = res.transform.to_world(P2[moved])
    want = W[moved] + [20.0, 0.0]
    assert np.linalg.norm(got - want, axis=1).max() < 4.0


def test_refine_too_few_matches_keeps_anchors():
    ref, W, P, c = _small_case()
    start = _start_from(P, W, (0.0, 0.0))
    res = refine.refine(P[:5], ref.corners, start)
    assert any(g.source == "anchor" for g in res.transform.gcps)


def test_refine_deterministic():
    ref, W, P, c = _small_case()
    a = refine.refine(P, ref.corners, _start_from(P, W, (20.0, 5.0)))
    b = refine.refine(P, ref.corners, _start_from(P, W, (20.0, 5.0)))
    assert a.matches == b.matches and a.iterations == b.iterations


def test_refine_no_corners_returns_start():
    ref, W, P, c = _small_case()
    start = _start_from(P, W, (0.0, 0.0))
    res = refine.refine(np.empty((0, 2)), ref.corners, start)
    assert res.matches == [] and res.iterations == 0 and res.transform is start
