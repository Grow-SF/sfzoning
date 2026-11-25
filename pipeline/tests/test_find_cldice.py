"""The finder's unbroken-line terms (`sfzmaps.find.cldice`): soft skeletons and the clDice recall and precision.
Moved with the module from the 1921 hatching reader's tests (`test_hatchnet_topo.py`, branch
`hatch-experiments-archive`)."""
import torch

from sfzmaps.find import cldice

ITERS = 32          # the skeleton depth for a 64 px tile (half the tile, as `find.train.unbroken_loss` takes it);
                    # every test shape is wider


def _street(shape, rows, cols, inside=0.95, outside=0.01):
    p = torch.full(shape, outside)
    p[..., rows, cols] = inside
    return p


def test_skeleton_of_thin_line():
    p = torch.zeros(1, 64, 160)
    p[0, 30:35, 20:140] = 1.0                       # a bar 5 px wide and 120 px long
    s = cldice.soft_skeleton(p, ITERS)[0]
    assert s.shape == p.shape[1:]
    assert float(s[:31].abs().max()) == 0.0 and float(s[34:].abs().max()) == 0.0     # only its middle 3 px
    kept = (s[31:34] > 0.5).any(0)
    assert int(kept.sum()) >= 0.95 * 120, int(kept.sum())


def test_skeleton_of_soft_line_stays_inside():
    p = torch.zeros(1, 64, 160)
    p[0, 30:35, 20:140] = torch.tensor([0.3, 0.8, 0.95, 0.8, 0.3])[:, None]     # a soft ridge across the bar
    s = cldice.soft_skeleton(p, ITERS)[0]
    assert float(s[:30].abs().max()) == 0.0 and float(s[35:].abs().max()) == 0.0
    assert float(s[32, 25:135].min()) > 0.5                                       # its crest is the skeleton


def test_recall_penalises_gap():
    shape = (1, 32, 100)
    centre = torch.zeros(shape)
    centre[0, 16, 10:90] = 1.0
    corridor = torch.zeros(shape)
    corridor[0, 14:19, 10:90] = 1.0
    whole = _street(shape, slice(14, 19), slice(10, 90))
    gap = whole.clone()
    gap[0, 14:19, 50] = 0.01                        # one px column missing across the street
    r_whole, _ = cldice.cldice_terms(whole, centre, corridor, ITERS)
    r_gap, _ = cldice.cldice_terms(gap, centre, corridor, ITERS)
    assert float(r_gap) < float(r_whole)


def test_precision_penalises_blob_off_centre():
    shape = (1, 128, 200)
    centre = torch.zeros(shape)
    centre[0, 20, 10:190] = 1.0
    corridor = torch.zeros(shape)                   # 0 off the corridor: today's blocks or ignored px alike
    corridor[0, 18:23, 10:190] = 1.0
    street = _street(shape, slice(18, 23), slice(10, 190))
    blob = street.clone()
    blob[0, 70:110, 120:160] = 0.95                 # a 40 x 40 predicted street far from every centre line
    _, before = cldice.cldice_terms(street, centre, corridor, ITERS)
    _, after = cldice.cldice_terms(blob, centre, corridor, ITERS)
    assert float(after) < float(before), (float(before), float(after))
    # precision is over every px (the region), labelled or not: the blob's px are all ignored here
    region = torch.ones(shape)
    _, after_region = cldice.cldice_terms(blob, centre, corridor, ITERS, region=region)
    assert float(after_region) == float(after)


def test_empty_terms_are_one():
    z = torch.zeros(1, 16, 16)
    r, p = cldice.cldice_terms(z, z, z, 4)
    assert float(r) == 1.0 and float(p) == 1.0      # no centre px: nothing to recall; no predicted skeleton
