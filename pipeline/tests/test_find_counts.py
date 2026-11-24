"""The counts a read is judged by (`find.counts`): each printed block found, joined, cut up or lost (exactly one of
them), open land left out, the depth fitted on the choosing crops, how far found block edges lie from the hand ones,
and the zoning-border statements agreed with, each read by its own statement's chance."""
from pathlib import Path

import numpy as np
import pytest

from sfzmaps.find import counts, handlabels
from sfzmaps.find.cuts import BORDER, EDGE, Cuts
from sfzmaps.find.handlabels import CropLabels


def _truth(blocks: dict[int, tuple[slice, slice]], shape, kinds=None) -> CropLabels:
    area = np.zeros(shape, np.int32)
    for i, (r, c) in blocks.items():
        area[r, c] = i
    kinds = kinds or {i: "block" for i in blocks}
    street = (area == 0).astype(np.uint8)
    return CropLabels(street, area, kinds, np.zeros(shape, np.uint8), [])


SHAPE = (60, 80)
C = (slice(5, 25), slice(5, 35))        # found: 600 px, its area also holds B
B = (slice(5, 25), slice(37, 42))       # joined: 100 px, all in C's area, which is mostly C
D = (slice(30, 50), slice(5, 35))       # cut up: three areas of 200 px
E = (slice(30, 50), slice(45, 61))      # lost: no area
O = (slice(52, 58), slice(40, 70))      # open land


def _found() -> np.ndarray:
    found = np.zeros(SHAPE, np.int32)
    found[5:25, 5:42] = 1                       # C, the street between, and B
    for k, c0 in enumerate((5, 15, 25)):
        found[30:50, c0:c0 + 10] = 2 + k
    found[52:58, 5:31] = 5                      # on hand street: an extra area
    found[52:58, 40:70] = 6                     # on the open land
    return found


def test_each_block_is_exactly_one_thing():
    truth = _truth({1: C, 2: B, 3: D, 4: E}, SHAPE)
    got = counts.blocks(_found(), truth)
    assert counts.block_status(_found(), truth) == {1: "found", 2: "joined", 3: "cut_up", 4: "lost"}
    assert (got["found"], got["joined"], got["cut_up"], got["lost"]) == (1, 1, 1, 1)
    assert got["found"] + got["joined"] + got["cut_up"] + got["lost"] == got["blocks"] == 4
    assert got["extra"] == 2                    # the area on hand street, and the one on O, street in this truth


def test_open_land_is_not_counted():
    with_open = _truth({1: C, 2: B, 3: D, 4: E, 5: O}, SHAPE, {1: "block", 2: "block", 3: "block", 4: "block",
                                                               5: "open"})
    without = _truth({1: C, 2: B, 3: D, 4: E}, SHAPE)
    without.street[O] = handlabels.NONE         # the same ground with no label at all
    got = counts.blocks(_found(), with_open)
    assert got == counts.blocks(_found(), without)
    assert got["blocks"] == 4 and got["extra"] == 1
    assert 5 not in counts.block_status(_found(), with_open)


def test_found_share_is_the_mean_over_eras():
    by_era = {"a": [{"found": 1, "blocks": 1}, {"found": 0, "blocks": 3}], "b": [{"found": 3, "blocks": 3}]}
    assert counts.found_share(by_era) == pytest.approx((1 / 4 + 1) / 2)
    assert counts.era_shares(by_era) == {"a": 0.25, "b": 1.0}


H, W, RIDGE = 40, 60, 30


def test_fit_depth_takes_the_smallest_best():
    # the faint-stretch case (a column of block edge at 0.9 dipping to 0.6 between two blocks), the left block made of
    # three parts parted by two low bumps of street chance 25/255, floors 10/255, 12.5/255 and 12.5/255: below 12.5/255
    # it is cut up; the right block's floor 50.5/255 keeps it apart up to 0.6 - 50.5/255 = 102.5/255
    street = np.ones((H, W))
    street[2:-2, 2:RIDGE] = 12.5 / 255
    street[2:-2, 2:11] = 10 / 255
    street[2:-2, [11, 20]] = 25 / 255
    street[2:-2, RIDGE + 1:-2] = 50.5 / 255
    street[2:-2, RIDGE] = 0.0
    edge = np.zeros((H, W))
    edge[2:-2, RIDGE] = 0.9
    edge[18:21, RIDGE] = 0.6
    border = np.zeros((H, W))
    read = (np.stack([street, edge, border]).astype(np.float32), np.ones((H, W), bool))
    truth = _truth({1: (slice(2, H - 2), slice(2, RIDGE)), 2: (slice(2, H - 2), slice(RIDGE + 1, W - 2))}, (H, W))
    depth, share = counts.fit_depth([read], [truth], ["lined-grey"])
    assert depth == 13 / 255
    assert share == 1.0


def test_edge_feet_measure_found_blocks_only():
    shape = (60, 50)
    truth = _truth({1: (slice(10, 30), slice(10, 40)), 2: (slice(40, 55), slice(5, 45))}, shape)
    edge = np.zeros(shape, np.uint8)
    for i in (1, 2):
        a = truth.area == i
        inner = np.zeros_like(a)
        inner[1:-1, 1:-1] = a[1:-1, 1:-1] & a[:-2, 1:-1] & a[2:, 1:-1] & a[1:-1, :-2] & a[1:-1, 2:]
        edge[a & ~inner] = 1
    truth = CropLabels(truth.street, truth.area, truth.kinds, edge, [])
    found = np.zeros(shape, np.int32)
    found[8:32, 8:42] = 1                       # the first block grown 2 px every way; the second lost
    got = counts.edge_feet(found, truth, 3.0)
    assert counts.block_status(found, truth) == {1: "found", 2: "lost"}
    assert got["median"] == 6.0 and got["p90"] == 6.0
    assert got["px"] == int(edge[truth.area == 1].sum())


def _cuts(lines, width) -> Cuts:
    """Cuts along row 0 of an item `width` px wide: each (start column, px, answer, says, weight)."""
    px, cut = [], []
    for i, (c0, n, *_rest) in enumerate(lines):
        px.append(np.arange(c0, c0 + n))
        cut.append(np.full(n, i, np.int32))
    m = len(lines)
    return Cuts(np.concatenate(px).astype(np.int64), np.concatenate(cut), np.ones(sum(len(p) for p in px), np.float32),
                np.array([a for *_x, a, _s, _w in lines], np.uint8), np.array([s for *_x, s, _w in lines], bool),
                np.array([w for *_x, w in lines], np.float32), np.full((m, 2), -1, np.int32), np.zeros(0, np.int64))


def test_border_counts_are_per_cut():
    chance = np.zeros((1, 40), np.float32)
    chance[0, 0:5] = 0.3            # "crosses": 1 - 0.7^5 = 0.83, agreed though no px is over one half
    chance[0, 6:9] = 0.1            # "crosses": 1 - 0.9^3 = 0.27, not agreed
    chance[0, 10:13] = 0.3          # "none": 0.7^3 = 0.34, not agreed though no px is over one half
    chance[0, 14:16] = 0.1          # "none": 0.9^2 = 0.81, agreed
    chance[0, 17:19] = (0.0, 1.0)   # "none" with a px at 1 as stored: not agreed (and no NaN)
    chance[0, 20:23] = 0.9          # "crosses", but weight 0: not counted
    chance[0, 24:27] = 0.9          # a block-edge cut: not counted
    cuts = _cuts([(0, 5, BORDER, True, 1.0), (6, 3, BORDER, True, 1.0), (10, 3, BORDER, False, 1.0),
                  (14, 2, BORDER, False, 1.0), (17, 2, BORDER, False, 1.0), (20, 3, BORDER, True, 0.0),
                  (24, 3, EDGE, True, 1.0)], 40)
    assert counts.borders(chance, cuts) == {"crosses": 2, "crosses_agreed": 1, "none": 3, "none_agreed": 1}


def test_cuts_in_a_box_are_those_wholly_inside_it():
    width = 20
    # cut 0 inside the box (rows 2-5, columns 3-9), cut 1 half out of it, cut 2 a zoning-border cut unused
    px = np.array([3 * width + 4, 3 * width + 5, 4 * width + 9, 4 * width + 10, 5 * width + 6], np.int64)
    cuts = Cuts(px, np.array([0, 0, 1, 1, 2], np.int32), np.ones(5, np.float32), np.array([BORDER] * 3, np.uint8),
                np.array([True, True, False]), np.array([1, 1, 0], np.float32), np.full((3, 2), -1, np.int32),
                np.zeros(0, np.int64))
    got = counts.cuts_in_box(cuts, width, (3, 2, 7, 4))
    assert list(got.cut) == [0, 0] and list(got.px) == [1 * 7 + 1, 1 * 7 + 2] and len(got.answer) == 1


def _crop(set_: str) -> handlabels.Crop:
    return handlabels.Crop(set_, "s", "p1", (0, 0, 10, 10), ((1, 1),), (), (), (), (), Path("s/p1__0_0_10_10.json"))


def test_the_chooser_takes_choosing_crops_only():
    with pytest.raises(ValueError, match="test"):
        counts.chooser([_crop("choose"), _crop("test")], "cpu")
    with pytest.raises(ValueError, match="no choose crop"):
        counts.chooser([], "cpu")
