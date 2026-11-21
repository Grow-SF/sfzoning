import math

import numpy as np

from sfzmaps.frame import paper
from sfzmaps.frame.geometry import Frame
from sfzmaps.frame.layout import LONG_PATCHES

H, W = 600, 1110  # one patch: 10 px
R = math.ceil(max(H, W) / LONG_PATCHES)
MAP_P = np.array([0.95, 0.04, 0.01], np.float32)
OUT_P = np.array([0.04, 0.95, 0.01], np.float32)


def _page(columns, called_out_to, seed=0):
    """A page whose left part is drawn column by column: `columns` is a list of (x0, x1, kind) with kind "ink"
    (solid), "map" (20% ink) or "paper"; the rest of the page is map.  The chances call every column left of
    `called_out_to` outside, the rest map."""
    rng = np.random.default_rng(seed)
    lum = (1.0 - (rng.random((H, W)) < 0.2)).astype(np.float32)
    for x0, x1, kind in columns:
        if kind == "ink":
            lum[:, x0:x1] = 0.0
        elif kind == "paper":
            lum[:, x0:x1] = 1.0
    p = np.empty((3, H, W), np.float32)
    p[:] = MAP_P[:, None, None]
    p[:, :, :called_out_to] = OUT_P[:, None, None]
    return lum, p


def _left_framed(first_kept):
    """phi = 0, only the left side framed, keeping columns >= first_kept."""
    return Frame(phi=0.0, offset={"top": 0.0, "bottom": 0.0, "left": float(first_kept), "right": 0.0})


def _first_kept(fr):
    m = fr.mask((H, W))
    cols = np.flatnonzero(m.all(0))
    assert (m.any(0) == m.all(0)).all()  # a straight vertical line
    return int(cols[0])


def test_on_paper_moves_a_side_off_a_thin_rule():
    # margin, a 3-px rule at 100..102, paper to 109, map from 109; the fit keeps from 102 (2 px into the rule)
    lum, p = _page([(0, 100, "paper"), (100, 103, "ink"), (103, 109, "paper")], called_out_to=103)
    fr, ns, bs = paper.on_paper(_left_framed(102), [], [], p, lum)
    assert _first_kept(fr) == 103
    assert fr.tilt["left"] == 0.0


def test_on_paper_picks_the_inner_gap_not_the_margin():
    # margin, bar (4 px), gap, thin rule (2 px), gap, map; the fit starts in the gap between the bar and the rule.
    # The window reaches the blank margin outside the bar, but the cut ends on the paper inside the thin rule.
    lum, p = _page([(0, 66, "paper"), (66, 70, "ink"), (70, 76, "paper"), (76, 78, "ink"), (78, 84, "paper")],
                   called_out_to=82)
    assert 73 - R < 66  # the margin is within reach
    fr, _, _ = paper.on_paper(_left_framed(73), [], [], p, lum)
    assert _first_kept(fr) == 78


def test_on_paper_leaves_ties_and_page_edges_alone():
    # the left side sits in a blank band wider than the window: every position ties
    lum, p = _page([(0, 200, "paper")], called_out_to=150)
    start = _left_framed(150)
    fr, _, _ = paper.on_paper(start, [], [], p, lum)
    assert fr.offset == start.offset and fr.tilt == start.tilt
    # a page with no frame at all: ink right at the page edge, called outside; nothing moves
    lum, p = _page([(0, 3, "ink")], called_out_to=10)
    fr, _, _ = paper.on_paper(Frame(), [], [], p, lum)
    assert fr.offset == Frame().offset and fr.tilt == Frame().tilt


def test_on_paper_stays_within_one_patch():
    # ink called outside up to column 240, map after; the fit keeps from 200: the slide goes inward by less than a
    # patch, every px it changes within one patch of the line
    lum, p = _page([(0, 240, "map")], called_out_to=240)
    start = _left_framed(200)
    fr, _, _ = paper.on_paper(start, [], [], p, lum)
    changed = start.mask((H, W)) ^ fr.mask((H, W))
    assert changed.any()
    cols = np.flatnonzero(changed.any(0))
    assert cols.min() >= 200 - R and cols.max() < 200 + R


def test_on_paper_moves_notch_and_block_sides():
    from sfzmaps.frame.blocks import Block
    from sfzmaps.frame.notches import Notch

    # a ruled box at the top whose fitted inner side sits 2 px inside its 3-px rule; a block whose right side
    # leaves 3 px of its ruled border kept
    lum, p = _page([], called_out_to=0, seed=1)
    lum[:, :] = 1.0
    lum[0:120, 300:303] = lum[0:120, 400:403] = 0.0
    lum[117:120, 300:403] = 0.0  # the box's rule; its inner side (kept from row 120)
    p[:, 0:120, 300:403] = OUT_P[:, None, None]
    lum[300:400, 600:603] = lum[300:400, 697:700] = 0.0
    lum[300:303, 600:700] = lum[397:400, 600:700] = 0.0  # a block's ruled border, 600..700 x 300..400
    lum[320:380, 620:680] = 0.5
    p[:, 300:400, 600:700] = np.array([0.08, 0.02, 0.9], np.float32)[:, None, None]
    lum[130:160, 200:500] = lum[420:460, 560:740] = 0.6  # map ink nearby, called map
    # the notch in frame coordinates (psi = 0: along = x + 0.5 - W/2, e = y + 0.5 - H/2 from the top)
    n = Notch("top", 0.0, 300 - W / 2, 403 - W / 2, 118 - H / 2)
    b = Block("rect", 650.0, 350.0, 0.0, (-50.0, 47.0, -50.0, 50.0))
    fr, ns, bs = paper.on_paper(Frame(), [n], [b], p, lum)
    assert round(ns[0].d - (120 - H / 2)) == 0
    assert round(bs[0].p[1]) == 50
    assert n.d == 118 - H / 2 and b.p[1] == 47.0  # the inputs are not changed


def test_on_paper_never_turns_a_notch_inside_out():
    # a 9-px notch over map ink called map, with map ink past it: shrinking it to nothing would keep the most map,
    # but its side stops at the minimum width instead of passing the other side
    from sfzmaps.frame.notches import NOTCH_MIN_PX, Notch

    lum, p = _page([], called_out_to=0, seed=2)
    n = Notch("top", 0.0, 300 - W / 2, 309 - W / 2, 200 - H / 2)
    fr, ns, _ = paper.on_paper(Frame(), [n], [], p, lum)
    assert ns[0].a1 - ns[0].a0 >= NOTCH_MIN_PX
