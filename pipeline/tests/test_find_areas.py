"""Closed areas from a read (`find.areas`): a faint stretch of block edge still parts two blocks at a small depth, an
area reaches only where "not street" is the likelier answer, a read that calls everything street closes no block, and
every depth read at once gives what each depth read alone gives."""
import numpy as np

from sfzmaps.find import areas, counts
from sfzmaps.find.handlabels import CropLabels

H, W = 40, 60
RIDGE = 30                  # the column of block edge between the two blocks


def _two_blocks(left=0.05, right=0.2, dip_rows=(18, 21)):
    """A street ring two px wide round two blocks (left floor `left`, right floor `right`, as street chances) parted
    by a column of block edge at 0.9 that dips to 0.6 for 3 px."""
    street = np.ones((H, W))
    street[2:-2, 2:RIDGE] = left
    street[2:-2, RIDGE:-2] = right
    street[2:-2, RIDGE] = 0.0
    edge = np.zeros((H, W))
    edge[2:-2, RIDGE] = 0.9
    edge[dip_rows[0]:dip_rows[1], RIDGE] = 0.6
    return street, edge, np.ones((H, W), bool)


def _hand_two_blocks() -> CropLabels:
    area = np.zeros((H, W), np.int32)
    area[2:-2, 2:RIDGE] = 1
    area[2:-2, RIDGE + 1:-2] = 2
    street = (area == 0).astype(np.uint8)
    return CropLabels(street, area, {1: "block", 2: "block"}, np.zeros((H, W), np.uint8), [])


def test_a_faint_stretch_still_separates_two_blocks():
    street, edge, keep = _two_blocks()
    # the right block's floor is 0.2 and the dip 0.6: a depth up to 0.4 keeps it its own minimum
    assert areas.areas(street, edge, keep, 0.5).max() == 1
    two = areas.areas(street, edge, keep, 0.2)
    assert two.max() == 2
    assert len(np.unique(two[2:-2, 2:RIDGE])) == 1 and len(np.unique(two[2:-2, RIDGE + 1:-2])) == 1
    assert two[5, 5] != two[5, RIDGE + 5]


def test_a_block_reaches_only_where_not_street_is_likelier():
    # two blocks across a street 20 px wide whose street chance climbs from 0.1 at the kerb to 0.9 in the middle
    street = np.full((H, W), 0.1)
    ramp = np.r_[np.linspace(0.1, 0.9, 10), np.linspace(0.9, 0.1, 10)]
    street[:, 20:40] = ramp
    street[:, 40:] = 0.12                       # the far block's floor a little higher: its own minimum all the same
    edge = np.zeros((H, W))
    keep = np.ones((H, W), bool)
    found = areas.areas(street, edge, keep, 0.1)
    np.testing.assert_array_equal(found > 0, street < 0.5)
    assert found.max() == 2
    assert len(np.unique(found[:, :30][found[:, :30] > 0])) == 1
    assert len(np.unique(found[:, 30:][found[:, 30:] > 0])) == 1


def test_everything_street_finds_no_block():
    rng = np.random.default_rng(0)
    street = np.ones((H, W))
    edge = rng.uniform(0, 1, (H, W))
    found = areas.areas(street, edge, np.ones((H, W), bool), 0.1)
    assert not found.any()
    c = counts.blocks(found, _hand_two_blocks())
    assert c["blocks"] == 2 and c["lost"] == 2 and c["found"] == 0


def test_off_the_map_is_no_ground():
    # a block touching the frame cut is still its own area (off the map is not low ground that would drain it)
    street, edge, keep = _two_blocks()
    keep[:, :4] = False
    found = areas.areas(street * keep, edge * keep, keep, 0.2)
    assert found.max() == 2 and not found[~keep].any()


def test_every_depth_at_once_is_each_depth_alone():
    rng = np.random.default_rng(3)
    depths = [i / 255 for i in range(1, 255)]
    for trial in range(4):
        street = np.rint(rng.uniform(0, 1, (36, 44)) ** 3 * 255) / 255
        edge = np.rint(rng.uniform(0, 1, (36, 44)) ** 4 * 255) / 255
        keep = np.ones(street.shape, bool)
        if trial % 2:
            keep[:, :5] = False
            street[~keep] = edge[~keep] = 0
        got = {}
        for which, found in areas.areas_by_depth(street, edge, keep, depths):
            for i in which:
                assert i not in got
                got[i] = found
        assert sorted(got) == list(range(len(depths)))
        for i, d in enumerate(depths):
            np.testing.assert_array_equal(got[i], areas.areas(street, edge, keep, d), err_msg=f"depth {d}")
