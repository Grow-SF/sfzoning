"""The printed areas and their measure against today's blocks: merged where a street was missed, shattered where one
was invented."""
import numpy as np

from sfzmaps.solver.hatchnet import areas

SHAPE = (60, 80)


def page(streets=(), crop=None):
    """(boundary_prob, crop, today's blocks): two blocks, columns 5-35 and 45-75, rows 5-55; `streets` are the
    boundary columns."""
    p = np.zeros(SHAPE, np.float32)
    for c in streets:
        p[:, c] = 1.0
    block = np.zeros(SHAPE, np.int32)
    block[5:55, 5:35] = 1
    block[5:55, 45:75] = 2
    return p, np.ones(SHAPE, bool) if crop is None else crop, block


def test_two_blocks_split_by_a_street():
    p, crop, block = page(streets=range(36, 44))
    m = areas.against_blocks(areas.areas(p, crop), block)
    assert (m["areas"], m["blocks"]) == (2, 2)
    assert (m["merged_areas"], m["blocks_in_merged"], m["shattered_blocks"]) == (0, 0, 0)
    assert m["merged_px_share"] == 0.0


def test_missing_street_merges_blocks():
    p, crop, block = page()
    a = areas.areas(p, crop)
    m = areas.against_blocks(a, block)
    assert (m["areas"], m["merged_areas"], m["blocks_in_merged"]) == (1, 1, 2)
    assert m["merged_px_share"] == 1.0 and m["largest_area_px_share"] == 1.0
    assert areas.merged(a, block)[a > 0].all()


def test_invented_street_shatters_a_block():
    p, crop, block = page(streets=[*range(36, 44), 12, 20, 28])        # block 1 cut into four parts, none over half
    m = areas.against_blocks(areas.areas(p, crop), block)
    assert m["shattered_blocks"] == 1 and m["merged_areas"] == 0


def test_no_boundary_at_all_is_one_area():
    p, crop, block = page()
    a = areas.areas(p, crop)
    assert a.max() == 1 and (a == 1).all()


def test_area_at_the_crop_edge_counts():
    crop = np.ones(SHAPE, bool)
    crop[:, 60:] = False                                               # the frame cut ends inside block 2
    p, crop, block = page(streets=range(36, 44), crop=crop)
    a = areas.areas(p, crop)
    assert (a[:, 60:] == 0).all()
    assert areas.against_blocks(a, block)["areas"] == 2


def test_block_outside_the_crop_is_left_out():
    crop = np.ones(SHAPE, bool)
    crop[:, 40:] = False                                               # block 2 is wholly off the frame cut
    p, crop, block = page(streets=range(36, 44), crop=crop)
    m = areas.against_blocks(areas.areas(p, crop), block)
    assert (m["blocks"], m["shattered_blocks"]) == (1, 0)


def test_picture_tints_areas_and_marks_merged_red():
    p, crop, block = page()
    a = areas.areas(p, crop)
    img = areas.picture(np.ones(SHAPE, np.float32), a, areas.merged(a, block))
    assert img.shape == (*SHAPE, 3) and img.dtype == np.uint8
    b, g, r = (int(v) for v in img[30, 20])
    assert r > g and r > b                                              # a merged area is red
