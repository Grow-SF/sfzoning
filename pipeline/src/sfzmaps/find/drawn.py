"""No labels where the sheet draws no ink.  A sheet prints only its own part of the city; the rest of its frame is blank
paper over which the lots would still put labels.  The rule (the owner, 2026-11-23: "don't draw lots where there isn't
ink"):

- an edge cut is used only if there is ink on at least one of its px (`INK_BELOW`);
- an outline px of a block (a block px with a 4-neighbour off the blocks) is inked if one of its edge cuts is used;
- every other label (a px's street-or-not label, its "no block edge" / "no zoning border" statements, a zoning-border
  cut) is used only where the outline px nearest to it is inked: a zoning-border cut needs both its lots' nearest
  outline px inked, and a cut saying "a zoning border crosses" needs ink on the cut itself as well.

Reads and writes nothing."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from sfzmaps.find.cuts import BORDER, EDGE, Cuts

INK_BELOW = 0.5                 # a px is ink where ink is the more likely reading: luminance under half (exact on the
                                # 1-bit scans the lots label)


def inked(c: Cuts, lum: np.ndarray) -> np.ndarray:
    """Bool (n,): the cuts with ink on at least one of their px."""
    return np.bincount(c.cut, lum.ravel()[c.px] < INK_BELOW, minlength=len(c.answer)) > 0


def near_inked(block: np.ndarray, inked_px: np.ndarray) -> np.ndarray:
    """Bool, the shape of the grid window `block` (bool, the blocks): whether the outline px nearest each px is inked,
    `inked_px` (flat window indexes) being the inside px of the used edge cuts.  All False when the window has no
    outline."""
    bare = np.pad(~block, 1, constant_values=False)        # beyond the window: not known to be off the blocks
    outline = block & (bare[:-2, 1:-1] | bare[2:, 1:-1] | bare[1:-1, :-2] | bare[1:-1, 2:])
    if not outline.any():
        return np.zeros(block.shape, bool)
    ink = np.zeros(block.size, bool)
    ink[inked_px] = True
    near = ndimage.distance_transform_edt(~outline, return_distances=False, return_indices=True)
    return ink.reshape(block.shape)[near[0], near[1]]


def use(c: Cuts, lum: np.ndarray, near: np.ndarray) -> tuple[Cuts, dict]:
    """The cuts with weight 1 where used and 0 where not (see the module), and the counts: edge cuts used and dropped;
    "crosses" cuts used, dropped for no ink on the cut, dropped for a bare block side; "none" cuts used and dropped."""
    has_ink = inked(c, lum)
    edge, border = c.answer == EDGE, c.answer == BORDER
    flat = near.ravel()
    sides = np.ones(len(c.answer), bool)
    for j in (0, 1):
        end = c.ends[:, j]
        sides &= np.where(end >= 0, flat[np.maximum(end, 0)], True)
    used = np.where(edge, has_ink, sides & (~c.says | has_ink))
    crosses, none = border & c.says, border & ~c.says
    counts = {"edge_used": int((edge & used).sum()), "edge_dropped": int((edge & ~used).sum()),
              "crosses_used": int((crosses & used).sum()),
              "crosses_dropped_no_ink_on_cut": int((crosses & sides & ~has_ink).sum()),
              "crosses_dropped_bare_side": int((crosses & ~sides).sum()),
              "none_used": int((none & used).sum()), "none_dropped": int((none & ~used).sum())}
    return c.with_weight(used.astype(np.float32)), counts
