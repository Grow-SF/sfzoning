"""From the network's three chances per pixel to the deletion mask (design spec part 1 and part 2, as amended by
"removed blocks").

The chances are p = (p0, p1, p2): map, outside the frame, removed block (a legend or title block, a compass rose or a
"SECTION N" badge printed inside the frame, deleted wherever it sits).  There is no cost ratio (amendment 6: the
earlier 3:1, frame ink left in counted three times map ink cut, was a guess, not a decision): the more likely answer
wins, so a pixel is deleted where p1 + p2 > p0, the `level` of one half, and the cost field is ink x (p0 - p1 - p2).
The chances it is applied to are the learned ones (`learned.apply`: how often a whole printed rule, and other ink,
really is frame or box given the model's reading, fitted on the training sheets).

The pixel-wise answer can leave holes or stray islands, so the decision applies the definitions:
1. Outside the frame means connected to the page edge.  Outside is the part of the cut (p1 + p2 > level) reachable
   from the border.
2. Every other pixel cannot be outside the frame, by definition.  It is removed iff p2 > level: the same cost rule
   with the impossible outside-frame mass p1 returned to map.  Not conditioned (p2 / (p0 + p2) > level): where the
   network confidently but wrongly calls map "frame", p0 and p2 are both tiny and their ratio is noise, while
   returning p1 to map keeps the earlier behaviour there (a commercial band called frame mid-map is kept).  The cut
   pixels neither outside nor removed are filled back to map.
3. The inside (neither outside nor removed) is reduced to the `maps` largest pieces; the rest is outside.
Removed blocks are therefore the only holes in the kept region, and only where the network says "removed block", not
merely "not map".

Outside uses 4-connectivity and inside 8-connectivity, the complementary pair (Rosenfeld 1979) that keeps the two
regions from crossing at a diagonal and makes the inside's outline a simple set of rings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio.features
from affine import Affine
from scipy import ndimage as ndi
from shapely.geometry import shape

# Even odds (amendment 6).  Not knobs: the names stay so the cost field's modules need no change.
COST_FRAME = 1.0
COST_MAP = 1.0

_FOUR = ndi.generate_binary_structure(2, 1)
_EIGHT = ndi.generate_binary_structure(2, 2)


def cut_level(cost_frame: float = COST_FRAME, cost_map: float = COST_MAP) -> float:
    """The chance of "not map" above which deleting is the more likely answer: one half."""
    return cost_map / (cost_frame + cost_map)


@dataclass(frozen=True)
class Decision:
    outside: np.ndarray  # bool: outside the frame (the cut reaching the border, and inside pieces beyond `maps`)
    removed: np.ndarray  # bool: removed blocks, enclosed by the map
    filled_ink: float  # ink of cut pixels that end up map (neither outside nor removed)
    dropped_ink: float  # ink of inside pieces beyond the `maps` largest
    removed_ink: float  # ink of the removed blocks

    @property
    def deleted(self) -> np.ndarray:
        """Bool, True = deleted: outside the frame or a removed block."""
        return self.outside | self.removed


def _ink(ink: np.ndarray, where: np.ndarray) -> float:
    return float(ink[where].sum(dtype=np.float64))


def decide(p: np.ndarray, lum: np.ndarray, maps: int = 1, level: float | None = None) -> Decision:
    """The deletion for the chances `p` (`(3, h, w)`: map, outside, removed block) of a page with luminance `lum`
    holding `maps` maps (see the module)."""
    if level is None:
        level = cut_level()
    p = np.asarray(p)
    ink = 1.0 - np.asarray(lum, np.float32)
    cut = p[1] + p[2] > level

    labels, n = ndi.label(cut, structure=_FOUR)
    border = np.zeros(n + 1, bool)  # a lookup table: label -> touches the page's border
    for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        border[edge] = True
    border[0] = False
    outside = border[labels]
    removed = ~outside & (p[2] > level)

    inside = ~outside & ~removed
    comp, m = ndi.label(inside, structure=_EIGHT)
    sizes = np.bincount(comp.ravel(), minlength=m + 1)[1:]
    order = np.argsort(-sizes, kind="stable")  # ties: lower label first
    keep = np.zeros(m + 1, bool)
    keep[order[:maps] + 1] = True
    kept = keep[comp]
    dropped = inside & ~kept
    return Decision(outside | dropped, removed, _ink(ink, cut & kept), _ink(ink, dropped), _ink(ink, removed))


def outline(deleted: np.ndarray) -> list[dict]:
    """The kept region (`~deleted`) as polygons in pixel-corner (x, y) coordinates, one dict per polygon:
    {"exterior": [(x, y), ...], "holes": [[(x, y), ...], ...]}.  Holes are where removed blocks sit."""
    inside = ~np.asarray(deleted, bool)
    if not inside.any():
        return []
    polys = []
    for geom, val in rasterio.features.shapes(
        inside.astype(np.uint8), mask=inside, connectivity=8, transform=Affine.identity()
    ):
        if val != 1:
            continue
        poly = shape(geom)
        polys.append({"exterior": _ring(poly.exterior), "holes": [_ring(r) for r in poly.interiors]})
    return polys


def _ring(ring) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in ring.simplify(0).coords]
