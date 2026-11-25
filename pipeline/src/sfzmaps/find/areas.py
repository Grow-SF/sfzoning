"""The areas of a read (blocks, and other ground that is not street): the connected pieces of ground the model calls not
street.

`areas`: the px inside the frame cut (`keep`) where street is not the more likely answer (the street chance at most one
half) are joined into pieces through their four side neighbours (`skimage.measure.label`, connectivity 1): two pieces
touching only at a corner are two areas.  Nothing else decides an area: not the block-edge chance, no depth, no
setting.  This is the method, and it is meant: a gap in the street answer joins the blocks on either side of it into
one area, and only the street answer parts them.  Px off the frame cut are in no area and join nothing.  Areas are
numbered 1, 2, ... in the order their first px comes, row by row; 0 is street or off the frame cut.

Reads and writes nothing."""
from __future__ import annotations

import numpy as np
from skimage.measure import label


def _renumber(lab: np.ndarray) -> np.ndarray:
    """Int32: `lab`'s labels above 0 as 1, 2, ... in the order of their first px, row by row."""
    ids, first = np.unique(lab, return_index=True)
    first, ids = first[ids > 0], ids[ids > 0]
    lut = np.zeros(int(lab.max()) + 1 if lab.size else 1, np.int32)
    lut[ids[np.argsort(first)]] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[lab]


def areas(street: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Int32 (h, w): the areas of a read, from its street chance (h, w) and frame cut `keep` (h, w), 0 none (see the
    module)."""
    ground = (np.asarray(street) <= 0.5) & np.asarray(keep, bool)
    return _renumber(label(ground, connectivity=1))
