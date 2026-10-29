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


# --- R16 carry-overs ---------------------------------------------------------

def _periodic_grid(n=10, w=300.0, h=200.0, street=60.0, x0=6_005_000.0, y0=2_115_000.0):
    """Identical blocks on a regular lattice: every block looks like its neighbour."""
    geoms, rows = [], []
    for i in range(n):
        for j in range(n):
            x, y = x0 + j * (w + street), y0 + i * (h + street)
            rows.append(f"{i * n + j + 1:04d}")
            geoms.append(box(x, y, x + w, y + h))
    return gpd.GeoDataFrame({"block_num": rows}, geometry=geoms, crs="EPSG:2227")


def _px_of(W, c):
    return np.c_[(W[:, 0] - c[0]) * 0.5 + 1500, -(W[:, 1] - c[1]) * 0.5 + 1000]


def test_vote_reach_is_clamped_to_rough_tol():
    assert config.ROUGH_TOL_FT == 150.0
    assert refine._vote_reach((120, 60, 30, 15)) == 150.0
    assert refine._vote_reach((80, 40)) == 120.0


def test_vote_ambiguous_on_periodic_grid_beyond_reach():
    """Start 180 ft off on a lattice with a 360 ft period: inside the vote window the
    offsets -120 and +120 (one street / one block over) are equally likely. The vote must
    not pick one silently: it returns zero and says why."""
    ref = _ref(_periodic_grid())
    W = np.array([[p.x, p.y] for p in ref.corners.geometry])
    P = _px_of(W, W.mean(0))
    start = _start_from(P, W, (180.0, 0.0))
    shift, status = refine._vote_shift(start.to_world(P), refine.cKDTree(W), 150.0)
    assert status in ("ambiguous", "edge")
    assert np.allclose(shift, 0.0)
    res = refine.refine(P, ref.corners, start)
    assert res.vote_status in ("ambiguous", "edge")
    assert res.vote_shift_ft == (0.0, 0.0)


def test_vote_clear_peak_on_periodic_grid_within_reach():
    ref = _ref(_periodic_grid())
    W = np.array([[p.x, p.y] for p in ref.corners.geometry])
    P = _px_of(W, W.mean(0))
    start = _start_from(P, W, (40.0, 0.0))
    shift, status = refine._vote_shift(start.to_world(P), refine.cKDTree(W), 150.0)
    assert status == "ok"
    assert np.allclose(shift, (-40.0, 0.0), atol=5.0)


def test_refine_records_vote(big_result):
    res, _ = big_result
    assert res.vote_status == "ok"
    assert np.hypot(*res.vote_shift_ft) > 20.0  # start was 40 ft off


def test_refine_12ft_moved_block_not_inlier():
    """A block drawn 12 ft off: inside the final 15 ft radius, so it pairs, but it must not
    be an inlier, and the warp must keep it where the paper puts it."""
    ref, W, P, c = _small_case()
    moved = np.asarray(ref.corners["block_num"]) == "0015"
    P2 = P.copy()
    P2[moved] += [12.0 * 0.5, 0.0]
    res = refine.refine(P2, ref.corners, _start_from(P, W, (10.0, 10.0)))
    labels = np.array([m.label for m in res.matches])
    assert not (labels[moved] == "inlier").any()
    got = res.transform.to_world(P2[moved])
    assert np.linalg.norm(got - (W[moved] + [12.0, 0.0]), axis=1).max() < 3.0


def test_refine_converged_flag(big, big_result):
    res, _ = big_result
    assert res.converged is True
    _, sheet, ref, corners = big
    start = _perturbed_start(sheet, ref, (40.0 / np.sqrt(2), 40.0 / np.sqrt(2)), 1.0)
    short = refine.refine(corners, ref.corners, start, max_iter=2)
    assert short.converged is False and short.iterations == 2


def test_refine_empty_ref_corners():
    ref, W, P, c = _small_case()
    start = _start_from(P, W, (0.0, 0.0))
    res = refine.refine(P, ref.corners.iloc[:0], start)
    assert len(res.matches) == len(P)
    assert all(m.label == "no_equivalent" and m.dist_ft == np.inf for m in res.matches)
    assert res.transform is start and res.iterations == 0


def test_refine_non_finite_corners():
    ref, W, P, c = _small_case()
    P2 = np.vstack([P[:20], [[np.nan, 5.0]], P[20:], [[np.inf, -np.inf]]])
    res = refine.refine(P2, ref.corners, _start_from(P, W, (10.0, -5.0)))
    assert len(res.matches) == len(P2)
    for i in (20, len(P2) - 1):
        assert res.matches[i].label == "no_equivalent" and res.matches[i].dist_ft == np.inf
    finite = [m for i, m in enumerate(res.matches) if i not in (20, len(P2) - 1)]
    assert sum(m.label == "inlier" for m in finite) >= 0.95 * len(P)


def test_refine_labels_and_dist_from_final_transform(big):
    """Every label and dist_ft is measured with the returned transform."""
    _, sheet, ref, corners = big
    rng = np.random.default_rng(11)
    extra = corners[rng.integers(0, len(corners), 150)] + rng.normal(0, 20, (150, 2))
    noisy = np.vstack([corners, extra])
    start = _perturbed_start(sheet, ref, (28.0, -28.0), -1.0)
    res = refine.refine(noisy, ref.corners, start)
    W = np.array([[p.x, p.y] for p in ref.corners.geometry])
    tree = refine.cKDTree(W)
    world = res.transform.to_world(noisy)
    dn, _ = tree.query(world)
    for m, w, d in zip(res.matches, world, dn):
        assert m.dist_ft == pytest.approx(np.hypot(w[0] - m.wx, w[1] - m.wy), abs=1e-6)
        if m.label == "no_equivalent":
            assert d > 15.0 and m.dist_ft == pytest.approx(d, abs=1e-6)
        elif m.label == "outlier":
            assert d <= 15.0 and m.dist_ft == pytest.approx(d, abs=1e-6)


# --- change classification (Task 10) --------------------------------------------

def _labels(res):
    return np.array([m.label for m in res.matches])


@pytest.fixture(scope="module")
def shifted():
    """The big sheet with block 0045 drawn 30 ft east of its modern place."""
    blocks = _irregular_grid()
    sheet = make_sheet(blocks, px_per_ft=PX_PER_FT, rotation_deg=7.0,
                       distortion=_paper_stretch, style="outline",
                       shift_blocks={"0045": (30.0, 0.0)})
    corners = linework.block_corners_px(linework.extract_blocks(sheet.img))
    ref = _ref(blocks)
    start = _perturbed_start(sheet, ref, (40.0 / np.sqrt(2), 40.0 / np.sqrt(2)), 1.0)
    res = refine.refine(corners, ref.corners, start)
    return blocks, sheet, ref, corners, res, refine.classify_changes(res)


def _block_corner_px(sheet, blocks, num, shift):
    g = blocks.loc[blocks["block_num"] == num, "geometry"].iloc[0]
    xy = np.asarray(g.exterior.coords)[:-1] + shift
    return sheet.world_to_px(xy)


def _match_near(res, px, tol=4.0):
    P = np.array([[m.px, m.py] for m in res.matches])
    d = np.linalg.norm(P - px, axis=1)
    assert d.min() < tol, "corner not detected"
    return res.matches[int(np.argmin(d))]


def test_shifted_block_is_real_change(shifted):
    from sfzmaps import score
    blocks, sheet, ref, corners, before, res = shifted
    px = _block_corner_px(sheet, blocks, "0045", (30.0, 0.0))
    ms = [_match_near(res, p) for p in px]
    assert [m.label for m in ms] == ["real_change"] * 4
    for m in ms:  # offset historic - modern is the drawn shift
        w = res.transform.to_world([[m.px, m.py]])[0]
        assert np.hypot(w[0] - m.wx - 30.0, w[1] - m.wy) < 3.0
    s = score.score(res)
    assert s.passed is True and s.reason is None
    assert s.n_real_change >= 4
    # the warp near 0045 is where the paper puts it, not dragged onto the modern block
    truth = sheet.true_px_to_world(px)
    assert np.linalg.norm(res.transform.to_world(px) - truth, axis=1).max() < 5.0
    assert np.linalg.norm(res.transform.to_world(px) - before.transform.to_world(px),
                          axis=1).max() < 5.0


def test_shifted_block_real_change_excluded_from_fit(shifted):
    *_, res = shifted
    rc = {(m.px, m.py) for m in res.matches if m.label == "real_change"}
    assert rc
    assert not rc & {(g.px, g.py) for g in res.transform.gcps}
    assert all(g.source == "auto" for g in res.transform.gcps)
    assert len(res.transform.gcps) == sum(m.label == "inlier" for m in res.matches)


def test_classify_unshifted_sheet_has_no_real_change(big_result):
    res, _ = big_result
    out = refine.classify_changes(res)
    assert (_labels(out) == "real_change").sum() == 0
    assert (_labels(out) == _labels(res)).all()


@pytest.mark.parametrize("shift_ft", [12.0, 20.0, 40.0])
def test_classify_small_moved_block(shift_ft):
    ref, W, P, c = _small_case()
    moved = np.asarray(ref.corners["block_num"]) == "0015"
    P2 = P.copy()
    P2[moved] += [shift_ft * 0.5, 0.0]
    res = refine.classify_changes(refine.refine(P2, ref.corners, _start_from(P, W, (10.0, 10.0))))
    labels = _labels(res)
    assert (labels[moved] == "real_change").all()
    assert (labels[~moved] != "real_change").all()
    for m, w in zip(np.array(res.matches)[moved], W[moved]):
        assert (m.wx, m.wy) == (w[0], w[1])  # its own modern corner, not across the street


def test_isolated_outlier_not_real_change():
    """Large residuals that do not form a coherent cluster stay outlier/no_equivalent:
    one lone corner, two corners agreeing (< min_cluster), and three nearby corners
    moved in different directions."""
    ref, W, P, c = _small_case()
    P2 = P.copy()
    nums = np.asarray(ref.corners["block_num"])
    lone = np.flatnonzero(nums == "0003")[0]
    pair = np.flatnonzero(nums == "0034")[:2]
    mixed = np.flatnonzero(nums == "0022")[:3]
    P2[lone] += [0.0, 10.0 * 0.5]                    # 10 ft north
    P2[pair] += [9.0 * 0.5, 0.0]                     # two corners 9 ft east
    P2[mixed] += np.array([[12.0, 0.0], [0.0, 12.0], [-9.0, -9.0]]) * [0.5, -0.5]
    res = refine.classify_changes(refine.refine(P2, ref.corners, _start_from(P, W, (5.0, 5.0))))
    labels = _labels(res)
    assert not (labels == "real_change").any()
    for i in [lone, *pair, *mixed]:
        assert labels[i] in ("outlier", "no_equivalent")


def test_changes_geojson():
    ref, W, P, c = _small_case()
    moved = np.asarray(ref.corners["block_num"]) == "0015"
    P2 = P.copy()
    P2[moved] += [20.0 * 0.5, 0.0]
    far = np.array([[P[:, 0].max() + 400, P[:, 1].mean()],
                    [P[:, 0].max() + 430, P[:, 1].mean()],
                    [P[:, 0].max() + 415, P[:, 1].mean() + 30],
                    [P[:, 0].min() - 400, P[:, 1].mean()]])
    res = refine.classify_changes(
        refine.refine(np.vstack([P2, far]), ref.corners, _start_from(P, W, (10.0, 10.0))))
    gj = refine.changes_geojson(res)
    assert gj["type"] == "FeatureCollection"
    assert gj["crs"]["properties"]["name"] == "urn:ogc:def:crs:EPSG::2227"
    rc = [f for f in gj["features"] if f["properties"]["kind"] == "real_change"]
    ne = [f for f in gj["features"] if f["properties"]["kind"] == "no_equivalent"]
    assert len(rc) == 4
    for f in rc:
        p = f["properties"]
        assert p["block_num"] == "0015"
        assert p["dx_ft"] == pytest.approx(20.0, abs=1.5) and abs(p["dy_ft"]) < 1.5
        assert f["geometry"]["type"] == "Point"
    assert sorted(f["properties"]["n_corners"] for f in ne) == [1, 3]
    assert all(f["geometry"]["type"] == "Polygon" for f in ne)
