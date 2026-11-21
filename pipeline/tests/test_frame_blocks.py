import json
import math

import numpy as np
import pytest

from sfzmaps.frame import blocks
from sfzmaps.frame.geometry import Frame, cost_field
from sfzmaps.frame.shape import Params

MAP_P = np.array([0.95, 0.04, 0.01], np.float32)
BLOCK_P = np.array([0.08, 0.02, 0.90], np.float32)


def _octagon(cx, cy, r):
    """A regular octagon of inner radius r (the labels' SECTION badges and roses are drawn so)."""
    R = r / math.cos(math.pi / 8)
    return [(cx + R * math.cos(math.pi / 8 + k * math.pi / 4), cy + R * math.sin(math.pi / 8 + k * math.pi / 4))
            for k in range(8)]


def _labels(tmp_path, sheets):
    for (series, sheet), remove in sheets.items():
        d = tmp_path / series
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{sheet}.json").write_text(json.dumps({"polygons": [[[0, 0], [10, 0], [10, 10]]], "remove": remove}))
    return sorted(sheets)


def test_block_bounds_class_each_label_by_overlap(tmp_path):
    rect = [(100, 100), (140, 100), (140, 120), (100, 120)]  # 40 x 20
    turned = [(300, 300), (300 + 60 * math.cos(0.3), 300 + 60 * math.sin(0.3)),
              (300 + 60 * math.cos(0.3) - 10 * math.sin(0.3), 300 + 60 * math.sin(0.3) + 10 * math.cos(0.3)),
              (300 - 10 * math.sin(0.3), 300 + 10 * math.cos(0.3))]  # 60 x 10, turned
    clipped = [(500, 500), (536, 500), (540, 504), (540, 530), (500, 530)]  # a rectangle with a cut corner
    badge = _octagon(700, 700, 24)  # 48 across
    labelled = _labels(tmp_path, {("1948", "1"): [rect, badge], ("1921", "x"): [turned, clipped]})
    b = blocks.block_bounds(labelled, root=tmp_path)
    assert b.rect_max == pytest.approx((1.5 * 60, 1.5 * 30), abs=0.5)
    major, minor = b.ellipse_max
    assert 1.5 * 48 - 1 <= minor <= major <= 1.5 * 52


def test_block_bounds_with_no_round_label_fit_no_ellipse(tmp_path):
    labelled = _labels(tmp_path, {("1948", "1"): [[(0, 0), (40, 0), (40, 20), (0, 20)]], ("1948", "2"): []})
    b = blocks.block_bounds(labelled, root=tmp_path)
    assert b.ellipse_max == (0.0, 0.0)


def _blank_page(h, w):
    lum = np.ones((h, w), np.float32)
    p = np.broadcast_to(MAP_P[:, None, None], (3, h, w)).copy()
    return lum, p


def test_blocks_are_bounded_by_the_labels():
    # a round p2 blob of ink on blank paper, 5x the largest labelled round block: the ellipse (and any rectangle)
    # fitted to it stays within the bounds
    bounds = blocks.BlockBounds(rect_max=(45.0, 30.0), ellipse_max=(60.0, 60.0))
    h, w = 600, 600
    lum, p = _blank_page(h, w)
    Y, X = np.mgrid[0:h, 0:w]
    disc = np.hypot(X + 0.5 - 300, Y + 0.5 - 300) <= 150
    lum[disc] = 0.5
    p[:, disc] = BLOCK_P[:, None]
    D = cost_field(p, lum)
    got = blocks.block_shapes(p, D, np.ones((h, w), bool), Frame(), bounds, Params(0.0, 0.0, 0.0))
    assert got
    for b in got:
        if b.kind == "ellipse":
            assert 2 * max(b.p) <= 60.0 + 1e-6 and 2 * min(b.p) <= 60.0 + 1e-6
        else:
            u0, u1, v0, v1 = b.p
            assert max(u1 - u0, v1 - v0) <= 45.0 + 1e-6 and min(u1 - u0, v1 - v0) <= 30.0 + 1e-6
    assert got[0].kind == "ellipse"  # the bigger allowed shape takes the most of the blob


def _patch(lum, p, m, rng):
    lum[m] = 1.0 - (rng.random(int(m.sum())) < 0.2)
    p[:, m] = BLOCK_P[:, None]


def test_map_shaped_false_block_is_refused():
    # map ink everywhere; an L of it called "removed block" and, elsewhere, a solid rectangle of the same area called
    # so too.  The L's own px gain more than lambda_block, but no rectangle or ellipse takes it without the map in its
    # crook, so it stays; the rectangle goes.
    h, w = 700, 900
    rng = np.random.default_rng(0)
    lum = (1.0 - (rng.random((h, w)) < 0.2)).astype(np.float32)
    p = np.broadcast_to(MAP_P[:, None, None], (3, h, w)).copy()
    L = np.zeros((h, w), bool)
    L[100:300, 100:120] = True
    L[280:300, 100:300] = True
    R = np.zeros((h, w), bool)
    R[450:470, 450:830] = True
    assert L.sum() == R.sum()
    _patch(lum, p, L, rng)
    _patch(lum, p, R, rng)
    D = cost_field(p, lum)
    arm = -D[100:300, 100:120].sum()
    lam = 1.5 * arm  # between one arm's gain and the whole L's
    assert -D[L].sum() > lam
    bounds = blocks.BlockBounds(rect_max=(600.0, 100.0), ellipse_max=(300.0, 300.0))
    got = blocks.block_shapes(p, D, np.ones((h, w), bool), Frame(), bounds, Params(0.0, 0.0, lam))
    removed = blocks.blocks_mask(got, (h, w))
    assert not (removed & L).any()
    assert (removed & R).sum() >= 0.95 * R.sum()


def test_blocks_are_rescored_inside_the_kept_region():
    # a legend half outside the kept region: only what is kept counts, and below lambda it is not kept
    h, w = 400, 400
    rng = np.random.default_rng(1)
    lum, p = _blank_page(h, w)
    m = np.zeros((h, w), bool)
    m[100:200, 100:300] = True
    _patch(lum, p, m, rng)
    D = cost_field(p, lum)
    cands = blocks.candidates(p, D, np.ones((h, w), bool), Frame(),
                              blocks.BlockBounds((400.0, 400.0), (0.0, 0.0)))
    full = -D[m].sum()
    kept = np.ones((h, w), bool)
    kept[:, 200:] = False
    removed, acc = blocks.accept(cands, D, kept, 0.6 * full)
    assert acc == [] and not removed.any()
    removed, acc = blocks.accept(cands, D, kept, 0.4 * full)
    assert len(acc) == 1 and not (removed & ~kept).any()
