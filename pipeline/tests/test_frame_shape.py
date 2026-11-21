import numpy as np
import pytest
from synthetic_frame import make_page

from sfzmaps.frame import shape

SIDES = ("top", "bottom", "left", "right")


def _fit(D, notch_aware=False):
    return shape.fit_sides_arcs(shape.frame_search(D, notch_aware), D)


# The precision tests use a 1800 x 2400 page: on 900 x 1200 the lines' 1-px rounding alone moves a side 1-2 px, the
# angle 0.05 degrees and (3.4 px of radius per px of side) a fillet 5 px, so the brief's tolerances held on some
# seeds only.
BIG = dict(h=1800, w=2400, offset={"top": 160.0, "bottom": 140.0, "left": 180.0, "right": 150.0})


def test_cost_field_sign():
    p = np.array([[0.9, 0.1, 0.9], [0.05, 0.85, 0.05], [0.05, 0.05, 0.05]], np.float32)[:, None, :]
    lum = np.array([[0.0, 0.0, 1.0]], np.float32)  # map ink, frame ink, paper
    D = shape.cost_field(p, lum)
    assert D.dtype == np.float32
    assert D[0, 0] > 0 > D[0, 1]  # map ink is cheaper kept, frame ink cheaper deleted
    assert D[0, 2] == 0  # paper costs nothing either way


def test_recovers_a_rotated_frame():
    # 1.2 degrees, the top side bowed 10 px inward, 120-px fillets, a legend and a badge inside
    lum, p, tr = make_page(phi=1.2, bow_top=10.0, arc=120.0, blocks=True, **BIG)
    fr = _fit(shape.cost_field(p, lum))
    assert abs(fr.phi - 1.2) <= 0.05
    for s in ("bottom", "left", "right"):
        assert abs(fr.offset[s] - tr["offset"][s]) <= 2, (s, fr.offset[s], tr["offset"][s])
    assert abs(fr.offset["top"] + fr.bow["top"] - (tr["offset"]["top"] + 10.0)) <= 2  # the bowed side's middle
    assert abs(fr.bow["top"] - 10.0) <= 2
    wrong = fr.mask(lum.shape) ^ tr["K"]
    assert wrong.mean() < 0.002


def test_notch_aware_start_keeps_a_side_past_a_ruled_notch():
    # a ruled box 250 px deep at the bottom, turned 1 degree off the frame: the plain rectangle takes its depth (the
    # box's ink outweighs the map beside it); the notch-aware start keeps the bottom side at the frame, up to its
    # 16-px grid plus the plain refinement's drift (24 + 8 + 4 px)
    c = np.cos(np.radians(1.2))
    notch = dict(side="bottom", psi=2.2, a0=112.0, a1=375.0, d=70 - c * 450 + 250)
    lum, p, tr = make_page(phi=1.2, blocks=True, notch=notch)
    D = shape.cost_field(p, lum)
    plain, aware = shape.frame_search(D, False), shape.frame_search(D, True)
    assert plain.offset["bottom"] - tr["offset"]["bottom"] > 200
    assert abs(aware.offset["bottom"] - tr["offset"]["bottom"]) <= 16 + 36
    assert abs(aware.phi - 1.2) <= 0.1


def test_fillets_recovered_within_3px():
    lum, p, tr = make_page(phi=-0.8, arc=120.0, seed=1, **BIG)
    fr = _fit(shape.cost_field(p, lum))
    for c in shape.CORNERS:
        assert 117 <= fr.arc[c] <= 123, (c, fr.arc[c])
    for s in SIDES:  # the sides are not bowed into the arcs
        assert abs(fr.bow[s]) <= 2.0, (s, fr.bow[s])
        assert abs(fr.offset[s] - tr["offset"][s]) <= 2, (s, fr.offset[s], tr["offset"][s])
    assert (fr.mask(lum.shape) ^ tr["K"]).mean() < 0.002


@pytest.mark.parametrize("notch_aware", [False, True])
def test_frameless_page_is_the_page_rectangle(notch_aware):
    rng = np.random.default_rng(0)
    D = (rng.random((900, 1200)) < 0.2).astype(np.float32) * rng.uniform(0.1, 1.0, (900, 1200)).astype(np.float32)
    fr = _fit(D, notch_aware)
    assert fr.phi == 0.0
    assert all(fr.offset[s] == 0.0 for s in SIDES)
    assert fr.mask(D.shape).all()


@pytest.mark.parametrize("notch_aware", [False, True])
def test_all_ties_take_the_simplest_shape(notch_aware):
    D = np.zeros((900, 1200), np.float32)
    fr = _fit(D, notch_aware)
    assert fr.phi == 0.0
    assert all(fr.offset[s] == 0.0 and fr.tilt[s] == 0.0 and fr.bow[s] == 0.0 for s in SIDES)
    assert all(r == 0.0 for r in fr.arc.values())
    assert fr.mask(D.shape).all()


def test_ties_on_a_framed_side_take_zero_tilt_and_bow():
    # 12 px of blank paper between each rule and the map: every tilt or bow that stays on the paper ties
    lum, p, tr = make_page(phi=0.4, gap=12, seed=2)
    fr = _fit(shape.cost_field(p, lum))
    for s in SIDES:
        assert fr.tilt[s] == 0.0 and fr.bow[s] == 0.0, (s, fr.tilt[s], fr.bow[s])
    ink, kept = 1 - lum, fr.mask(lum.shape)
    assert ink[kept & ~tr["K"]].sum() == 0  # no rule ink kept
    assert ink[~kept & tr["K"]].sum() == 0  # no map ink cut


def test_blank_strip_called_outside_does_not_move_the_side():
    lum, p, tr = make_page(phi=0.6, blank_strip=300, seed=3)
    fr = _fit(shape.cost_field(p, lum))
    assert abs(fr.offset["left"] - tr["offset"]["left"]) <= 2, fr.offset["left"]


@pytest.mark.parametrize("notch_aware", [False, True])
def test_all_ink_page_does_not_crash(notch_aware):
    rng = np.random.default_rng(4)
    p = rng.dirichlet((1.0, 1.0, 1.0), (900, 1200)).transpose(2, 0, 1).astype(np.float32)
    lum = np.zeros((900, 1200), np.float32)
    fr = _fit(shape.cost_field(p, lum), notch_aware)
    m = fr.mask(lum.shape)
    assert m.shape == lum.shape and m.dtype == bool
    assert abs(fr.phi) <= shape.ANGLE_MAX + 0.2


def test_partly_framed_page_keeps_its_corners():
    h, w = 900, 1200
    lum, p, tr = make_page(h, w, phi=1.0, offset={"top": 70.0, "left": 85.0, "bottom": None, "right": None}, seed=5)
    fr = _fit(shape.cost_field(p, lum))
    assert abs(fr.phi - 1.0) <= 0.1
    for s in ("top", "left"):
        assert abs(fr.offset[s] - tr["offset"][s]) <= 2, (s, fr.offset[s])
    for s in ("bottom", "right"):
        assert fr.offset[s] == 0.0 and fr.tilt[s] == 0.0 and fr.bow[s] == 0.0
    assert fr.arc[("bottom", "right")] == 0.0
    m = fr.mask((h, w))
    assert m[-1, w // 4:].all() and m[h // 4:, -1].all()  # nothing cut along the page-edge sides, corner kept
    assert (m ^ tr["K"]).mean() < 0.002
