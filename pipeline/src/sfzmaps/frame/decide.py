"""From q = P(outside the printed frame) to the deletion mask.

Cutting a pixel costs (1 - q) * COST_MAP in expectation (map ink lost if it
was really inside); keeping it costs q * COST_FRAME (frame ink left in). By
linearity of expectation the pixel-wise best choice is to cut where
q * COST_FRAME > (1 - q) * COST_MAP, i.e. q > cost_map / (cost_frame + cost_map).

The pixel-wise answer can leave holes or stray islands, so we apply the
definition: "outside the frame" means connected to the page edge. Outside is
the cut region reachable from the border; any cut pixels enclosed by the map
are filled back to inside. Inside is then reduced to the `maps` largest
pieces. Outside uses 4-connectivity and inside 8-connectivity, the
complementary pair (Rosenfeld 1979) that keeps the two regions from crossing
at a diagonal and makes the inside's outline a simple set of rings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio.features
from affine import Affine
from scipy import ndimage as ndi
from shapely.geometry import shape

# Owner 2026-11-20: frame ink left in is 3x worse than map ink cut.
COST_FRAME = 3.0
COST_MAP = 1.0

_FOUR = ndi.generate_binary_structure(2, 1)
_EIGHT = ndi.generate_binary_structure(2, 2)


def cut_level(cost_frame: float = COST_FRAME, cost_map: float = COST_MAP) -> float:
    """The q above which cutting is cheaper in expectation."""
    return cost_map / (cost_frame + cost_map)


@dataclass(frozen=True)
class Decision:
    outside: np.ndarray  # bool, True = deleted
    filled_ink: float  # ink of cut pixels put back (enclosed by the map)
    dropped_ink: float  # ink of inside pieces beyond the `maps` largest


def decide(q: np.ndarray, lum: np.ndarray, maps: int = 1, level: float | None = None) -> Decision:
    if level is None:
        level = cut_level()
    ink = 1.0 - np.asarray(lum, dtype=float)
    cut = np.asarray(q) > level

    labels, n = ndi.label(cut, structure=_FOUR)
    if n:
        border = np.unique(
            np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        )
        border = border[border > 0]
        outside = np.isin(labels, border)
    else:
        outside = np.zeros_like(cut)
    filled = cut & ~outside
    filled_ink = float(ink[filled].sum())

    inside = ~outside
    if not inside.any():
        return Decision(np.ones_like(cut), filled_ink, 0.0)
    comp, m = ndi.label(inside, structure=_EIGHT)
    sizes = np.bincount(comp.ravel(), minlength=m + 1)[1:]
    order = np.argsort(-sizes, kind="stable")  # ties: lower label first
    keep = order[:maps] + 1
    dropped = inside & ~np.isin(comp, keep)
    return Decision(outside | dropped, filled_ink, float(ink[dropped].sum()))


def outline(outside: np.ndarray) -> list[list[tuple[float, float]]]:
    """The inside region's exterior rings in pixel-corner (x, y) coordinates."""
    inside = ~np.asarray(outside, bool)
    if not inside.any():
        return []
    rings = []
    for geom, val in rasterio.features.shapes(
        inside.astype(np.uint8), mask=inside, connectivity=8, transform=Affine.identity()
    ):
        if val != 1:
            continue
        ring = shape(geom).exterior.simplify(0)
        rings.append([(float(x), float(y)) for x, y in ring.coords])
    return rings
