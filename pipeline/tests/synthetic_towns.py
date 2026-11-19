"""Synthetic hatched towns for the learned streets' tests (`sfzmaps.solver.hatchnet.train`).

A town is a grid of `block`-px blocks between white `street`-px streets, the grid shifted by a random phase; each block
has a black outline and a diagonal hatch every `period` px (one family, at one angle, as 1R).  Its labels are those
`hatchnet.data` writes:

  block     today's blocks (the first target): 1 on a block, 0 on a street
  street    the street labels: 1 street, 0 not street, every px labelled (no 255)
  street_w  1 on every labelled px
  centre    1 on each labelled street's centre line (the middle row or column of its band) where the label is street

Two kinds of label noise, each on its own:
  - `theta`, `phi`: each px's label flipped independently: a printed block px is labelled street with probability
    theta, a printed street px labelled block with probability phi (the noise model's own two rates);
  - `unprinted`: the share of blocks that today's map cuts with a street the print does not show: a vertical strip
    `street` px wide down the block's middle, from street to street, labelled street (with its centre line) and taken
    out of today's blocks, while the ink stays the block's outline and hatching.
"""
import numpy as np

from sfzmaps.solver.hatchnet import data


def _grid(seed: int, shape, block: int, street: int):
    rng = np.random.default_rng(seed)
    h, w = shape
    step = block + street
    oy, ox = (int(v) for v in rng.integers(0, step, 2))
    yy, xx = np.mgrid[0:h, 0:w]
    return rng, yy, xx, oy, ox, step


def printed_blocks(seed: int, shape=(192, 192), block: int = 40, street: int = 8) -> np.ndarray:
    """Where `town(seed, shape, block=block, street=street)` prints a block (bool)."""
    _, yy, xx, oy, ox, step = _grid(seed, shape, block, street)
    return ((yy - oy) % step < block) & ((xx - ox) % step < block)


def town(seed: int, shape=(192, 192), block: int = 40, street: int = 8, period: int = 6, ft_per_px: float = 5.0,
         theta: float = 0.0, phi: float = 0.0, unprinted: float = 0.0) -> data.SheetRasters:
    rng, yy, xx, oy, ox, step = _grid(seed, shape, block, street)
    ry, rx = (yy - oy) % step, (xx - ox) % step
    inside = (ry < block) & (rx < block)
    edge = inside & ((ry == 0) | (ry == block - 1) | (rx == 0) | (rx == block - 1))
    hatch = inside & ((yy + xx) % period == 0)
    ink = np.where(edge | hatch, 0.0, 1.0).astype(np.float32)
    printed = ~inside
    centre = printed & ((ry == block + street // 2) | (rx == block + street // 2))
    today = inside.copy()
    if unprinted:
        by, bx = (yy - oy) // step + 1, (xx - ox) // step + 1          # each px's block (row, column), from 0
        cut = rng.random((by.max() + 1, bx.max() + 1)) < unprinted      # the blocks today's map cuts
        # each cut's column in its block, at random clear of the block's side outlines: where today's map cuts a
        # block is not in the print, so nothing in the ink can tell the network where the strips are
        lo = rng.integers(2, block - street - 1, cut.shape)[by, bx]
        strip = inside & (rx >= lo) & (rx < lo + street) & cut[by, bx]
        today &= ~strip
        centre |= strip & (rx == lo + street // 2)
    label = ~today
    if theta or phi:
        u = rng.random(shape)
        label = np.where(printed, u >= phi, u < theta)
        today = ~label
    centre &= label
    return data.SheetRasters(f"town{seed}", ink, today.astype(np.uint8), np.full(shape, -1, np.int32),
                             np.ones(shape, bool), ft_per_px, street=label.astype(np.uint8),
                             centre=centre.astype(np.uint8), street_w=np.ones(shape, np.float16))
