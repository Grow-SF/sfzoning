"""Soft skeletons and the clDice terms (Shit et al., "clDice: a novel topology-preserving loss function for tubular
structure segmentation", CVPR 2021), for the finder's unbroken-line terms (`find.train`).  Moved here unchanged from the
1921 hatching reader (`solver.hatchnet.topo`, branch `hatch-experiments-archive`).

`soft_skeleton(p, iters)`: the paper's soft skeleton, iterated min and max pooling.  A soft erosion is the least of a
px and its four side neighbours (min pools 3 x 1 and 1 x 3), a soft dilation the greatest over the 3 x 3 square; each
of `iters` rounds erodes the map once more and adds what its opening (erode, then dilate) loses, so a band k px wide
is reduced to its middle once `iters` reaches about k / 2.  Pooling pads with nothing (max pools' own padding): a
street that leaves the tile is not eroded from the tile's edge.

`cldice_terms(p, centre, corridor, iters, region)`:
  recall     sum(centre p) / sum(centre): how much of the centre lines (weights `centre`) the predicted street covers;
             a gap in the street along a centre line lowers it
  precision  sum(S region corridor) / sum(S region), S = soft_skeleton(p): how much of the predicted street's
             skeleton, over every px of `region` (labelled or not), lies in a street corridor (weights `corridor`);
             predicted street away from every corridor lowers it
A term whose denominator is 0 (no centre px; no predicted skeleton) is 1: there is nothing to miss.
"""
import torch
import torch.nn.functional as F


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """(n, 1, h, w): each px the least of itself and its four side neighbours."""
    return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1)))


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """(n, 1, h, w): each px the greatest over its 3 x 3 square."""
    return F.max_pool2d(x, 3, 1, 1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeleton(p: torch.Tensor, iters: int) -> torch.Tensor:
    """The soft skeleton of the maps `p` (..., h, w) in [0, 1], of their shape (Shit et al.'s soft-skel)."""
    shape = p.shape
    img = p.reshape(-1, 1, *shape[-2:])
    skel = F.relu(img - soft_open(img))
    for _ in range(iters):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel.reshape(shape)


def ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """num / den, and 1 where den is 0."""
    return torch.where(den > 0, num / den.clamp_min(torch.finfo(den.dtype).tiny), torch.ones_like(den))


def cldice_sums(p: torch.Tensor, centre: torch.Tensor, corridor: torch.Tensor, iters: int,
                region: torch.Tensor | None = None) -> torch.Tensor:
    """The four sums of `cldice_terms` (recall's numerator and denominator, precision's), summed over every map in
    `p` (..., h, w): pooled sums, so several tiles' terms add up to the terms of them all."""
    s = soft_skeleton(p, iters)
    if region is not None:
        s = s * region
    return torch.stack([(centre * p).sum(), centre.sum(), (s * corridor).sum(), s.sum()])


def cldice_terms(p: torch.Tensor, centre: torch.Tensor, corridor: torch.Tensor, iters: int,
                 region: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(recall, precision) of the predicted street `p` against the centre lines and corridors (see the module)."""
    s = cldice_sums(p, centre, corridor, iters, region)
    return ratio(s[0], s[1]), ratio(s[2], s[3])
