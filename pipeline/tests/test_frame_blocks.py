import json
import math

import numpy as np
import pytest

from sfzmaps.frame import blocks, shape
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


def _turned(X, Y, cx, cy, deg):
    """(u, v) of px centres about (cx, cy), turned by `deg` degrees."""
    t = math.radians(deg)
    dx, dy = X + 0.5 - cx, Y + 0.5 - cy
    return math.cos(t) * dx + math.sin(t) * dy, -math.sin(t) * dx + math.cos(t) * dy


def _ruled_box_page(phi, seed=0):
    """A 1000 x 1000 page of blank paper (called map) with a ruled 600 x 400 box of 20% ink turned by `phi` about
    (500, 500); the chances call the box "removed" over a band along its diagonal covering three quarters of it (the
    corners off the band, 300 x 200 triangles, are called map).  (lum, p, box mask)."""
    h = w = 1000
    rng = np.random.default_rng(seed)
    lum, p = _blank_page(h, w)
    Y, X = np.mgrid[0:h, 0:w]
    u, v = _turned(X, Y, 500, 500, phi)
    box = (np.abs(u) < 300) & (np.abs(v) < 200)
    rule = box & ((np.abs(u) >= 295) | (np.abs(v) >= 195))
    lum[box] = 1.0 - (rng.random(int(box.sum())) < 0.2)
    lum[rule] = 0.0
    a, b = (u + 300) / 600, (v + 200) / 400  # 0..1 across the box, left to right and top to bottom
    band = box & (np.abs(a - b) < 0.5)  # leaves out the top-right and bottom-left corners, each 1/8 of the box
    p[:, band] = BLOCK_P[:, None]
    p[:, box & ~band] = MAP_P[:, None]
    return lum, p, box


def _the_block(p, lum, frame, bounds):
    h, w = lum.shape
    cands = blocks.candidates(p, cost_field(p, lum), np.ones((h, w), bool), frame, bounds)
    assert cands
    return cands[0]


def _angle_off(theta, phi):
    """The angle between a rectangle turned by theta and one turned by phi (a rectangle repeats every 90 degrees)."""
    return abs((theta - phi + 45.0) % 90.0 - 45.0)


@pytest.mark.parametrize("phi", [0.0, 0.5])
def test_a_box_is_square_to_the_frame_when_the_blob_is_diagonal(phi):
    # the frame is turned phi degrees and so is the box (phi 0: a page with no framed side, square to the page); the
    # blob called "removed" runs along the box's diagonal, so its own moment angle is far from the frame's: the
    # rectangle still takes the frame's angle
    lum, p, _ = _ruled_box_page(phi)
    b = _the_block(p, lum, Frame(phi=phi), blocks.BlockBounds((900.0, 600.0), (0.0, 0.0)))
    assert b.kind == "rect"
    assert _angle_off(b.theta, phi) <= 0.5


def test_ink_read_mostly_map_is_not_a_box():
    # a legend-sized blob of ink the chances read as 34% label box and 62% map: at even odds it is map, so the box
    # step finds no block there and the whole decision keeps its ink
    h, w = 400, 400
    rng = np.random.default_rng(3)
    lum, p = _blank_page(h, w)
    m = np.zeros((h, w), bool)
    m[150:250, 120:280] = True
    _patch(lum, p, m, rng)
    p[:, m] = np.array([0.62, 0.04, 0.34], np.float32)[:, None]
    bounds = blocks.BlockBounds((400.0, 400.0), (400.0, 400.0))
    assert blocks.candidates(p, cost_field(p, lum), np.ones((h, w), bool), Frame(), bounds) == []
    dec, fr, ns, bs = shape.decide(p, lum, Params(lam_notch=1e4, beta=3.0, lam_block=1e3), bounds)
    assert bs == [] and not dec.removed.any()
    assert not (dec.deleted & m).any()


def test_an_ellipse_still_takes_its_own_angle():
    # a thin ellipse of ink turned 30 degrees, called "removed": found as an ellipse near 30 degrees
    h = w = 600
    rng = np.random.default_rng(2)
    lum, p = _blank_page(h, w)
    Y, X = np.mgrid[0:h, 0:w]
    u, v = _turned(X, Y, 300, 300, 30.0)
    m = (u / 200) ** 2 + (v / 40) ** 2 <= 1.0
    _patch(lum, p, m, rng)
    b = _the_block(p, lum, Frame(), blocks.BlockBounds((600.0, 120.0), (600.0, 120.0)))
    assert b.kind == "ellipse"
    assert abs((b.theta - 30.0 + 90.0) % 180.0 - 90.0) <= 3.0
