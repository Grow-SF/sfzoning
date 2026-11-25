"""The areas of a read (`find.areas`): the 4-connected pieces of ground the model calls not street.  A street band parts
two blocks; one px of it called not street joins them (the method: a gap in the street answer joins blocks); blocks
touching only at a corner stay two; ground off the frame cut is in no area and joins nothing; a chance of exactly one
half is not street."""
import numpy as np

from sfzmaps.find import areas, counts
from sfzmaps.find.handlabels import CropLabels

H, W = 30, 50
BAND = slice(20, 26)        # the street band between the two blocks, columns 20-25


def _two_blocks():
    """Two blocks (street chance 0.1) parted by a street band (0.9) six px wide, a street ring two px wide round them,
    all inside the frame cut."""
    street = np.full((H, W), 0.9)
    street[2:-2, 2:BAND.start] = 0.1
    street[2:-2, BAND.stop:-2] = 0.1
    return street, np.ones((H, W), bool)


def test_a_street_band_parts_two_blocks_and_one_gap_in_it_joins_them():
    street, keep = _two_blocks()
    found = areas.areas(street, keep)
    assert found.dtype == np.int32 and found.max() == 2
    np.testing.assert_array_equal(found > 0, street <= 0.5)
    assert len(np.unique(found[2:-2, 2:BAND.start])) == 1 and len(np.unique(found[2:-2, BAND.stop:-2])) == 1
    assert found[5, 5] == 1 and found[5, BAND.stop + 1] == 2          # numbered by their first px, row by row
    bridged = street.copy()
    bridged[14, BAND] = 0.3                                            # one row of the band called not street
    joined = areas.areas(bridged, keep)
    assert joined.max() == 1 and joined[5, 5] == joined[5, BAND.stop + 1] == joined[14, 22] == 1


def test_blocks_touching_only_at_a_corner_are_two_areas():
    street = np.ones((20, 20))
    street[2:10, 2:10] = 0.0
    street[10:18, 10:18] = 0.0                  # its corner (10, 10) meets the first's (9, 9) diagonally
    found = areas.areas(street, np.ones((20, 20), bool))
    assert found.max() == 2 and found[9, 9] != found[10, 10]
    c = counts.blocks(found, _key(((slice(2, 10), slice(2, 10)), (slice(10, 18), slice(10, 18))), (20, 20)))
    assert c["found"] == 2


def test_ground_off_the_frame_cut_is_in_no_area_and_joins_nothing():
    street, keep = _two_blocks()
    street[:, BAND] = 0.1                       # the band called not street ...
    keep[:, BAND] = False                       # ... but off the frame cut
    keep[:, :4] = False
    found = areas.areas(street, keep)
    assert not found[~keep].any()
    assert found.max() == 2 and found[5, 5] != found[5, BAND.stop + 1]


def test_a_chance_of_one_half_is_not_street():
    street, keep = _two_blocks()
    street[14, BAND] = 0.5                      # exactly one half: not street, so it bridges
    assert areas.areas(street, keep).max() == 1
    street[14, BAND] = np.nextafter(0.5, 1.0)   # just over one half: street, so it does not
    assert areas.areas(street, keep).max() == 2


def test_everything_street_finds_no_area():
    found = areas.areas(np.ones((H, W)), np.ones((H, W), bool))
    assert not found.any()
    c = counts.blocks(found, _key(((slice(2, 28), slice(2, 20)), (slice(2, 28), slice(26, 48))), (H, W)))
    assert c["blocks"] == 2 and c["lost"] == 2 and c["found"] == 0


def _key(boxes, shape) -> CropLabels:
    area = np.zeros(shape, np.int32)
    for i, (r, c) in enumerate(boxes, 1):
        area[r, c] = i
    street = (area == 0).astype(np.uint8)
    return CropLabels(street, area, {i: "block" for i in range(1, len(boxes) + 1)}, np.zeros(shape, np.uint8), [])
