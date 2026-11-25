"""No labels where the sheet draws no ink.  A sheet prints only its own part of the city; the rest of its frame is blank
paper over which the lots would still put labels.  The rule (the owner, 2026-11-23: "don't draw lots where there isn't
ink"):

- a block-edge cut is used only if there is ink on at least one of its px (`INK_BELOW`; `use`);
- an outline px of a block (a block px with a 4-neighbour off the blocks) is inked if one of its block-edge cuts is
  used;
- every other label (a px's street-or-not label and its "no block edge" statement) is used only where the outline px
  nearest to it is inked (`near_inked`).

Reads and writes nothing."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from sfzmaps.find.cuts import Cuts

INK_BELOW = 0.5                 # a px is ink where ink is the more likely reading: luminance under half (exact on the
                                # 1-bit scans the lots label)


def inked(c: Cuts, lum: np.ndarray) -> np.ndarray:
    """Bool (n,): the cuts with ink on at least one of their px."""
    return np.bincount(c.cut, lum.ravel()[c.px] < INK_BELOW, minlength=len(c.weight)) > 0


def near_inked(block: np.ndarray, inked_px: np.ndarray) -> np.ndarray:
    """Bool, the shape of the grid window `block` (bool, the blocks): whether the outline px nearest each px is inked,
    `inked_px` (flat window indexes) being the inside px of the used block-edge cuts.  All False when the window has
    no outline."""
    bare = np.pad(~block, 1, constant_values=False)        # beyond the window: not known to be off the blocks
    outline = block & (bare[:-2, 1:-1] | bare[2:, 1:-1] | bare[1:-1, :-2] | bare[1:-1, 2:])
    if not outline.any():
        return np.zeros(block.shape, bool)
    ink = np.zeros(block.size, bool)
    ink[inked_px] = True
    near = ndimage.distance_transform_edt(~outline, return_distances=False, return_indices=True)
    return ink.reshape(block.shape)[near[0], near[1]]


def use(c: Cuts, lum: np.ndarray) -> tuple[Cuts, dict]:
    """The cuts with weight 1 where used (ink on one of their px, `inked`) and 0 where not, and the counts
    `edge_used` and `edge_dropped`."""
    used = inked(c, lum)
    return c.with_weight(used.astype(np.float32)), {"edge_used": int(used.sum()), "edge_dropped": int((~used).sum())}
