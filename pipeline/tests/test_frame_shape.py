import logging

import numpy as np
import pytest
from scipy import ndimage as ndi
from synthetic_frame import make_page

from sfzmaps.frame import decide as pixel
from sfzmaps.frame import rules
from sfzmaps.frame import shape
from sfzmaps.frame.blocks import BlockBounds

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
    # a ruled box 250 px deep at the bottom, turned 1 degree off the frame, half ink inside: the plain rectangle takes
    # most of its depth (the box's ink outweighs the map beside it); the notch-aware start keeps the bottom side at
    # the frame, up to its 16-px grid plus the plain refinement's drift (24 + 8 + 4 px)
    c = np.cos(np.radians(1.2))
    notch = dict(side="bottom", psi=2.2, a0=112.0, a1=375.0, d=70 - c * 450 + 250, ink=0.5)
    lum, p, tr = make_page(phi=1.2, blocks=True, notch=notch)
    D = shape.cost_field(p, lum)
    plain, aware = shape.frame_search(D, False), shape.frame_search(D, True)
    assert plain.offset["bottom"] - tr["offset"]["bottom"] > 150
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


# ---------------------------------------------------------------------------------------------------------------
# The whole decision
# ---------------------------------------------------------------------------------------------------------------

# make_page's legend is 150 x 110 px and its badge 60 across: the labels' largest, widened by half
BOUNDS = BlockBounds(rect_max=(225.0, 165.0), ellipse_max=(90.0, 90.0))
FITTED = shape.Params(lam_notch=1e4, beta=3.0, lam_block=1e3)  # the spike's fitted numbers


def _notch_page():
    c = np.cos(np.radians(1.2))
    notch = dict(side="bottom", psi=2.2, a0=112.0, a1=375.0, d=70 - c * 450 + 250)
    return make_page(phi=1.2, blocks=True, notch=notch)


def test_notchless_page_takes_the_plain_start():
    # without a notch the decision takes the plain start and no notch.  (At the old 3:1 cost the notch-aware start
    # came out degrees off here, its credited notches taking back the rule wedges a wrong angle leaves in; at even
    # odds that trap is gone from this page, and no notch cost brings it back without false notches winning.)
    lum, p, tr = make_page(phi=0.7, bow_top=10.0, arc=120.0, blocks=True, **BIG)
    params = shape.Params(lam_notch=3e3, beta=3.0, lam_block=1e3)
    timing = {}
    dec, fr, ns, bs = shape.decide(p, lum, params, BOUNDS, timing=timing)
    assert timing["start"] == "plain"
    assert abs(fr.phi - 0.7) <= 0.1
    assert ns == []
    assert ((~dec.deleted) ^ (tr["cls"] == 0)).mean() < 0.002


def test_decision_is_one_region_with_only_block_holes():
    lum, p, tr = _notch_page()
    dec, fr, ns, bs = shape.decide(p, lum, FITTED, BOUNDS)
    assert len(ns) == 1 and ns[0].side == "bottom"
    assert sorted(b.kind for b in bs) == ["ellipse", "rect"]
    kept = ~dec.deleted
    assert not (dec.outside & dec.removed).any()
    _, n = ndi.label(kept, structure=np.ones((3, 3)))
    assert n == 1

    def border_pieces(m):  # 4-connected pieces of m: (labels, count, the labels touching the page's border)
        lab, n = ndi.label(m)
        return lab, n, set(np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])) - {0}

    lab, n, border = border_pieces(dec.outside)
    assert border == set(range(1, n + 1))  # outside reaches the page's border
    lab, n, border = border_pieces(dec.deleted)
    holes = np.isin(lab, [k for k in range(1, n + 1) if k not in border])
    assert holes.any() and (holes <= dec.removed).all()  # the kept region's only holes are removed blocks
    assert (kept ^ (tr["cls"] == 0)).mean() < 0.005
    assert dec.filled_ink == 0.0 and dec.dropped_ink == 0.0
    assert dec.removed_ink == pytest.approx(float((1 - lum)[dec.removed].sum()), rel=1e-6)


def test_frameless_page_only_loses_its_blocks():
    lum, p, tr = make_page(offset=dict.fromkeys(SIDES), blocks=True)
    dec, fr, ns, bs = shape.decide(p, lum, FITTED, BOUNDS)
    assert all(fr.offset[s] == 0.0 for s in SIDES) and ns == []
    assert not dec.outside.any()
    truth = tr["cls"] == 2
    assert (dec.removed & truth).sum() >= 0.999 * truth.sum()
    assert (dec.removed & ~truth).sum() <= 0.05 * truth.sum()


def test_multi_map_page_keeps_the_per_pixel_decision(caplog):
    lum, p, tr = make_page(blocks=True)
    with caplog.at_level(logging.INFO, logger="sfzmaps.frame.shape"):
        dec, fr, ns, bs = shape.decide(p, lum, FITTED, BOUNDS, maps=2)
    want = pixel.decide(rules.share(p, lum), lum, 2)  # rules are shared on every page, several maps too (amendment 5)
    assert (dec.outside == want.outside).all() and (dec.removed == want.removed).all()
    assert "per-pixel" in caplog.text


def test_side_ink_left_ranks_a_missed_rule():
    # the fitted left side sits 5 px outside the truth, on the 5-px rule the chances call frame
    lum, p, tr = make_page(phi=0.6, seed=2)
    fr = shape.Frame(phi=0.6, offset=dict(tr["offset"]))
    assert all(v == 0.0 for v in shape.side_ink_left(p, lum, fr).values())
    fr.offset["left"] -= 5
    left = shape.side_ink_left(p, lum, fr)
    assert max(left, key=left.get) == "left"
    assert left["left"] > 0.9 * 4 * 700  # the 4-px strip along the side's ~740 px
    assert left["top"] <= 4 * 5 and left["bottom"] <= 4 * 5  # the corners where they meet the missed rule
    assert left["right"] == 0.0


def test_cache_reuses_the_work_that_does_not_depend_on_the_numbers(monkeypatch):
    lum, p, tr = _notch_page()
    other = shape.Params(lam_notch=1e4, beta=0.0, lam_block=1e4)
    fresh = shape.decide(p, lum, other, BOUNDS)
    cache = {}
    shape.decide(p, lum, FITTED, BOUNDS, cache=cache)
    calls = []
    real = shape.frame_search
    monkeypatch.setattr(shape, "frame_search", lambda *a, **k: calls.append(a[1:]) or real(*a, **k))
    again = shape.decide(p, lum, other, BOUNDS, cache=cache)
    assert calls == []  # both starts came from the cache (same lambda_notch)
    assert (again[0].outside == fresh[0].outside).all() and (again[0].removed == fresh[0].removed).all()
    assert again[1] == fresh[1] and again[2] == fresh[2]
    shape.decide(p, lum, shape.Params(lam_notch=2e4, beta=3.0, lam_block=1e3), BOUNDS, cache=cache)
    assert calls == [(True, 2e4)]  # a new lambda_notch: only the notch-aware start again


def test_cache_refuses_another_page():
    lum, p, _ = make_page(h=300, w=400, offset={"top": 30.0, "bottom": 30.0, "left": 30.0, "right": 30.0})
    cache = {}
    shape.decide(p, lum, FITTED, BOUNDS, cache=cache)
    shape.decide(p, lum, FITTED, BOUNDS, cache=cache)            # the same page again: fine
    other = p.copy()
    other[:, 150:160, 200:210] = other[:, 150:160, 200:210][::-1]
    with pytest.raises(ValueError, match="another page"):
        shape.decide(other, lum, FITTED, BOUNDS, cache=cache)
    with pytest.raises(ValueError, match="another page"):
        shape.decide(p[:, :200], lum[:200], FITTED, BOUNDS, cache=cache)


def test_a_frame_rule_half_read_as_map_does_not_tilt_the_cut():
    # a level frame; the chances call the right half of the top rule map (as on 2000-digital-su p11, where the model's
    # chance of deletion along the top rule fell from 0.97 at its left to 0.17-0.38 at its right)
    lum, p, tr = make_page(phi=0.0, seed=6)
    top = int(tr["offset"]["top"])
    rule = np.zeros(lum.shape, bool)
    rule[top - 5:top] = lum[top - 5:top] < 0.5  # the 5-px rule just outside the kept region
    right = rule.copy()
    right[:, : lum.shape[1] // 2] = False
    p = p.copy()
    p[:, right] = np.array([0.8, 0.18, 0.02], np.float32)[:, None]
    dec, fr, ns, bs = shape.decide(p, lum, FITTED, BOUNDS)
    assert abs(fr.phi) <= 0.1, fr.phi
    assert fr.bow["top"] == 0.0 and fr.tilt["top"] == 0.0, (fr.bow["top"], fr.tilt["top"])
    assert dec.deleted[rule].all(), (~dec.deleted[rule]).sum()  # the whole top rule


def test_a_weakly_read_long_frame_rule_is_cut_at_even_odds():
    # a level frame whose rules read 0.3 (map 0.7) everywhere, its map ink read 0.02: on the model's chances even odds
    # keep the frame; on the learned chances (amendment 6, a fit like the measured one) the frame goes and the map stays
    from sfzmaps.frame import learned

    lum, p, tr = make_page(phi=0.0, seed=7)
    ink = 1.0 - lum
    frame, mapink = (tr["cls"] == 1) & (ink > 0), (tr["cls"] == 0) & (ink > 0)
    p = p.copy()
    p[:, tr["cls"] == 1] = np.array([0.7, 0.3, 0.0], np.float32)[:, None]
    p[:, mapink] = np.array([0.98, 0.02, 0.0], np.float32)[:, None]
    measured = learned.Chances(rule=(-6.03, 2.17, 1.46), ink=(0.16, 1.0))

    def left_and_cut(q):
        dec, *_ = shape.decide(q, lum, FITTED, BOUNDS)
        return ink[frame & ~dec.deleted].sum() / ink[frame].sum(), ink[mapink & dec.deleted].sum() / ink[mapink].sum()

    left, cut = left_and_cut(p)
    assert left > 0.5                                                   # without the learned chances: kept
    left, cut = left_and_cut(learned.apply(p, lum, measured))
    assert left < 0.01 and cut < 0.002
