import heapq
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

from sfzmaps import linework
from sfzmaps.sheetwarp import first
from sfzmaps.sheetwarp.sheet import NothingToFit
from sfzmaps.sheetwarp.targets import SidePoints, Targets

# the CPU everywhere (CI's path); the Mac's GPU too where there is one
DEVICES = ["cpu", pytest.param("mps", marks=[pytest.mark.mps,
                                              pytest.mark.skipif(not torch.backends.mps.is_available(),
                                                                 reason="no GPU (mps) here")])]


def _sides(n_points, sid):
    px = np.column_stack([np.arange(n_points, dtype=float) + 10, np.full(n_points, 10.0)])
    return SidePoints(pts=px, sid=np.asarray(sid), ta=px - [3, 0], tb=px + [3, 0], px=px, pa=px - [3, 0], pb=px + [3, 0])


def test_propagation_with_every_side_held_has_nothing_to_move():
    sides = _sides(10, [0] * 5 + [1] * 5)
    targets = Targets(maps=np.zeros((18, 40, 40), np.uint8), origin=np.array([0, 0]), page_shape=(40, 40))
    with pytest.raises(NothingToFit):
        first.propagated_shifts(SimpleNamespace(series="s", name="n"), sides, targets, np.array([True, True]))


def test_propagation_with_no_held_side_on_the_crop_has_nothing_to_fit(tmp_path):
    # nothing held to carry the shifts out from (the search for the nearest solved points would ask for none)
    (tmp_path / "blocks_px.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    sides = _sides(10, [0] * 5 + [1] * 5)
    targets = Targets(maps=np.zeros((18, 40, 40), np.uint8), origin=np.array([0, 0]), page_shape=(40, 40))
    sheet = SimpleNamespace(series="s", name="n", folder=tmp_path)
    with pytest.raises(NothingToFit, match="no held"):
        first.propagated_shifts(sheet, sides, targets, np.array([False, False]))


def _first_correction_inputs(n_held_line, n_held_cp):
    targets = Targets(maps=np.zeros((18, 120, 120), np.uint8), origin=np.array([0, 0]), page_shape=(120, 120))
    shifts = first._Shifts(free=np.array([[60.0, 60.0], [62.0, 60.0]], np.float32),
                           free_n=np.array([[0.0, 1.0], [0.0, 1.0]], np.float32),
                           free_off=np.array([1.0, 1.0], np.float32))
    held = first._Inputs(held_line=np.tile(np.float32([[20.0, 20.0]]), (n_held_line, 1)),
                         held_n=np.tile(np.float32([[0.0, 1.0]]), (n_held_line, 1)),
                         held_cp=np.tile(np.float32([[100.0, 100.0]]), (n_held_cp, 1)),
                         corner_rec=np.zeros((0, 2), np.float32), corner_scan=np.zeros((0, 2), np.float32))
    return targets, shifts, held


def _run_first_correction(monkeypatch, n_held_line, n_held_cp):
    targets, shifts, held = _first_correction_inputs(n_held_line, n_held_cp)
    monkeypatch.setattr(first, "propagated_shifts", lambda *a: shifts)
    monkeypatch.setattr(first, "_inputs", lambda *a: held)
    sheet = SimpleNamespace(series="s", name="n")
    return first.first_correction(sheet, None, targets, None)


def test_the_first_correction_with_no_held_block_side_points_has_nothing_to_fit(monkeypatch):
    # the mean of an empty term is NaN, which would make the whole correction NaN
    with pytest.raises(NothingToFit, match="no held"):
        _run_first_correction(monkeypatch, 0, 3)


def test_the_first_correction_with_no_held_control_points_is_still_finite(monkeypatch):
    # the empty mean's NaN loss has an empty gradient into the grid, so the grid stays finite
    correction = _run_first_correction(monkeypatch, 3, 0)
    assert np.isfinite(correction.grid).all()


@pytest.mark.parametrize("device", DEVICES)
def test_the_first_correction_repeats_bit_for_bit(device, monkeypatch):
    monkeypatch.setenv("SFZMAPS_WARP_DEVICE", device)
    # thousands of held points on the same 4 grid nodes: added up in any other order, their pull changes in the bits
    first_run = _run_first_correction(monkeypatch, 5000, 3)
    second_run = _run_first_correction(monkeypatch, 5000, 3)
    assert isinstance(first_run.grid, np.ndarray) and first_run.grid.dtype == np.float32
    assert np.array_equal(first_run.grid, second_run.grid)


def test_held_sides_leaves_out_points_near_a_sides_ends():
    # 1 point per px (ft_per_px 2: points every 2 ft): the 3 points at each end of a side are within 2 px of it
    sheet = SimpleNamespace(series="s", name="n", ft_per_px=2.0, ink=np.zeros((40, 40), bool),
                            distance_to_paper=np.zeros((40, 40)), block_line_half_width=2.0)
    sides = _sides(20, [0] * 10 + [1] * 10)
    d = np.zeros(20)
    d[[0, 1, 9]] = 5.0           # side 0: far from its line only at its ends
    d[15] = 5.0                  # side 1: far from its line in the middle
    held = first.held_sides(sheet, sides, d)
    assert held.tolist() == [True, False]


def test_side_members_by_one_sort_equal_each_sides_mask():
    rng = np.random.default_rng(7)
    sid = rng.integers(0, 300, 20000)            # in no order; some of the 320 sides have no points
    keep = rng.random(20000) < 0.8
    members = first.side_members(sid, keep, 320)
    masked = [np.flatnonzero((sid == s) & keep) for s in range(320)]
    assert len(members) == 320
    assert all(np.array_equal(a, b) and a.dtype == b.dtype for a, b in zip(members, masked))
    assert sum(len(m) == 0 for m in members) >= 20


def _todays_propagated_shifts(sides, targets, held_side, street_px):
    """propagated_shifts as it was before its sides were grouped by a sort and its nearest-first heap was built by
    one query: a mask per side, a query per side"""
    TG = targets.maps
    B, h, w = TG.shape
    sid, P = sides.sid, sides.px.copy()
    dv = sides.pb - sides.pa
    tng = dv / np.maximum(np.linalg.norm(dv, axis=1), 1e-9)[:, None]
    nrm = np.column_stack([-tng[:, 1], tng[:, 0]])
    pbin = np.round((np.degrees(np.arctan2(dv[:, 1], dv[:, 0])) % 180) / 10.0).astype(int) % B
    ok = np.isfinite(P).all(1) & (P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h)
    ns = sid.max() + 1
    members = [np.flatnonzero((sid == s) & ok) for s in range(ns)]
    shift = np.full((len(P), 2), np.nan)
    solved_side = np.zeros(ns, bool)
    for s in range(ns):
        if held_side[s] and len(members[s]):
            shift[members[s]] = 0.0
            solved_side[s] = True
    pending = [s for s in range(ns) if not solved_side[s] and len(members[s])]
    reach = np.arange(-(street_px / 2 - 0.5), street_px / 2 - 0.5 + 0.01, 0.5)

    def solved_tree():
        idx = np.flatnonzero(~np.isnan(shift[:, 0]))
        return cKDTree(P[idx]), idx

    tree, sidx = solved_tree()
    heap = [(float(tree.query(P[members[s]])[0].min()), s) for s in pending]
    heapq.heapify(heap)
    k = 0
    while heap:
        _, s = heapq.heappop(heap)
        if solved_side[s]:
            continue
        m = members[s]
        dd, ii = tree.query(P[m].mean(0)[None], k=min(30, len(sidx)))
        dd, ii = np.atleast_1d(dd[0]), np.atleast_1d(ii[0])
        wts = 1.0 / np.maximum(dd, 1.0) ** 2
        prior = (shift[sidx[ii]] * wts[:, None]).sum(0) / wts.sum()
        q = P[m][:, None, :] + prior[None, None, :] + nrm[m][:, None, :] * reach[None, :, None]
        xi = np.clip(np.round(q[..., 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(q[..., 1]).astype(int), 0, h - 1)
        dmap = TG[pbin[m][:, None], yi, xi].astype(float)
        best = np.argmin(dmap + 1e-3 * np.abs(reach)[None, :], axis=1)
        hit = dmap[np.arange(len(m)), best] <= 1.0
        if hit.any():
            nn_ = np.median(nrm[m], axis=0)
            sv = prior + float(np.median(reach[best[hit]])) * nn_ / max(np.linalg.norm(nn_), 1e-9)
        else:
            sv = prior
        shift[m] = sv
        solved_side[s] = True
        k += 1
        if k % 50 == 0:
            tree, sidx = solved_tree()
            heap = [(float(tree.query(P[members[t]])[0].min()), t) for t in range(ns)
                    if not solved_side[t] and len(members[t])]
            heapq.heapify(heap)
    free = ok & ~held_side[sid]
    free_off = np.sum(shift[free] * nrm[free], axis=1)
    return P[free].astype(np.float32), nrm[free].astype(np.float32), free_off.astype(np.float32)


def test_propagated_shifts_are_todays(tmp_path):
    # a 10 x 10 grid of 60 px blocks with 20 px streets, printed 1 px wide; today's placement puts every side up to
    # 4 px off its line, by a smooth shift; a quarter of the sides held: 300 to carry outward, 6 times 50
    (tmp_path / "blocks_px.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    h = w = 830
    across, down = np.zeros((h, w), bool), np.zeros((h, w), bool)
    pts, sid, ends = [], [], []
    for row in range(10):
        for col in range(10):
            x0, y0 = 20 + 80 * col, 20 + 80 * row
            for a, b in (((x0, y0), (x0 + 60, y0)), ((x0 + 60, y0), (x0 + 60, y0 + 60)),
                         ((x0 + 60, y0 + 60), (x0, y0 + 60)), ((x0, y0 + 60), (x0, y0))):
                along = np.linspace(0, 1, 61)[:, None]
                line = np.round(np.asarray(a) + along * (np.subtract(b, a))).astype(int)
                (across if a[1] == b[1] else down)[line[:, 1], line[:, 0]] = True
                pts.append(np.asarray(a) + along * np.subtract(b, a))
                sid.append(np.full(61, len(ends)))
                ends.append((a, b))
    maps = np.full((18, h, w), 255, np.uint8)
    maps[0] = np.minimum(ndimage.distance_transform_edt(~across), 255)
    maps[9] = np.minimum(ndimage.distance_transform_edt(~down), 255)
    targets = Targets(maps=maps, origin=np.array([0, 0]), page_shape=(h, w))
    pts, sid = np.concatenate(pts), np.concatenate(sid)
    tangent = np.concatenate([np.repeat([np.subtract(b, a) / 20.0], 61, axis=0) for a, b in ends])
    placed = pts + np.c_[3 * np.sin(pts[:, 1] / 130.0), 4 * np.cos(pts[:, 0] / 170.0)]
    sides = SidePoints(pts=pts, sid=sid, ta=pts - tangent, tb=pts + tangent, px=placed, pa=placed - tangent,
                       pb=placed + tangent)
    held = np.random.default_rng(4).random(len(ends)) < 0.25
    sheet = SimpleNamespace(series="s", name="n", folder=tmp_path)
    shifts = first.propagated_shifts(sheet, sides, targets, held)
    free, free_n, free_off = _todays_propagated_shifts(sides, targets, held, linework.DEFAULT_STREET_PX)
    assert np.array_equal(shifts.free, free) and np.array_equal(shifts.free_n, free_n)
    assert np.array_equal(shifts.free_off, free_off)
    assert (np.abs(free_off) > 1).mean() > 0.5        # most free sides really are moved
