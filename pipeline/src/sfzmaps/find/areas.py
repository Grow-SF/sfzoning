"""Closed areas (blocks, and other ground that is not street) from a read: a watershed over the finder's chances.

The ground's height at a px is `height`: 1 - (1 - street)(1 - edge), the chance that the px is street or a block edge
runs there.  `areas`: the markers are the connected pieces of `skimage.morphology.h_minima(height, depth)` (the hollows
at least `depth` deep), the basins `skimage.segmentation.watershed(height, markers, mask=keep)`; an area is a basin's
px where street < 0.5 (not street is the more likely answer), and a basin with no such px is dropped.  Off `keep` (off
the map) the ground counts as the highest there is: no hollow drains off the map.  Areas are numbered 1, 2, ... in the
order their first px comes, row by row; 0 is no area.

`areas_by_depth` gives `areas` at many depths at once, each depth's exactly as `areas` gives it: h_minima keeps a
hollow (a regional minimum of the height, its px of one value m) at depth d when every path from it to lower ground
climbs to at least m + d, so each hollow's own depth (that climb, `_depths`) says at once which depths keep it.  The
climbs come from the watershed of every hollow: a path between two hollows' basins climbs at least the lower of the
two px heights across their common side, and within a basin every px is reached from its hollow without climbing
above it.  One watershed per distinct set of markers, not one per depth.

Reads and writes nothing."""
from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence

import numpy as np
from skimage.measure import label
from skimage.morphology import h_minima, local_minima
from skimage.segmentation import watershed

OFF_MAP = 1.0               # the height off `keep`: the highest there is


def height(street: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Float64: 1 - (1 - street)(1 - edge)."""
    return 1 - (1 - np.asarray(street, np.float64)) * (1 - np.asarray(edge, np.float64))


def _ground(street: np.ndarray, edge: np.ndarray, keep: np.ndarray) -> np.ndarray:
    g = height(street, edge)
    g[~np.asarray(keep, bool)] = OFF_MAP
    return g


def _renumber(lab: np.ndarray) -> np.ndarray:
    """Int32: `lab`'s labels above 0 as 1, 2, ... in the order of their first px, row by row."""
    ids, first = np.unique(lab, return_index=True)
    first, ids = first[ids > 0], ids[ids > 0]
    lut = np.zeros(int(lab.max()) + 1 if lab.size else 1, np.int32)
    lut[ids[np.argsort(first)]] = np.arange(1, len(ids) + 1, dtype=np.int32)
    return lut[lab]


def _close(ground: np.ndarray, markers: np.ndarray, street: np.ndarray, keep: np.ndarray) -> np.ndarray:
    basins = watershed(ground, markers, mask=np.asarray(keep, bool))
    basins[np.asarray(street) >= 0.5] = 0
    return _renumber(basins)


def areas(street: np.ndarray, edge: np.ndarray, keep: np.ndarray, depth: float) -> np.ndarray:
    """Int32 (h, w): the closed areas of the read (street and edge chances, (h, w) each) at `depth`, 0 none (see the
    module)."""
    ground = _ground(street, edge, keep)
    with warnings.catch_warnings():     # skimage's reconstruction sets an array's shape, which numpy 2.5 deprecates
        warnings.filterwarnings("ignore", "Setting the shape on a NumPy array", DeprecationWarning)
        minima = h_minima(ground, depth)
    markers = label(minima, connectivity=2)
    return _close(ground, markers, street, keep)


def _depths(ground: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The hollows of `ground` (int64 (h, w): 0 none, else the hollow's number) and the depth of each (float64, by
    number - 1): the climb from its height m to the lowest pass to lower ground, infinite for a lowest hollow (see the
    module)."""
    hollows = label(local_minima(ground, connectivity=2), connectivity=2).astype(np.int64)
    n = int(hollows.max())
    floor = np.zeros(n + 1)
    floor[hollows.ravel()] = ground.ravel()                       # a hollow's px all have its height
    basins = watershed(ground, hollows, connectivity=2)
    a_all, b_all, w_all = [], [], []
    h, w = ground.shape
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):              # every pair of 8-neighbours, once
        ys, xs = slice(0, h - dy), slice(max(0, -dx), w - max(0, dx))
        yt, xt = slice(dy, h), slice(max(0, dx), w - max(0, -dx))
        a, b = basins[ys, xs].ravel(), basins[yt, xt].ravel()
        cross = a != b
        a_all.append(np.minimum(a, b)[cross])
        b_all.append(np.maximum(a, b)[cross])
        w_all.append(np.maximum(ground[ys, xs], ground[yt, xt]).ravel()[cross])
    a, b, wt = np.concatenate(a_all), np.concatenate(b_all), np.concatenate(w_all)
    order = np.lexsort((b, a, wt))                                # by pass height, then by the pair: deterministic
    a, b, wt = a[order], b[order], wt[order]
    parent = np.arange(n + 1)
    low = floor.copy()                                            # each group's lowest height
    waiting = {i: [i] for i in range(1, n + 1)}                   # each group's hollows whose depth is not yet known
    depth = np.full(n + 1, np.inf)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, v in zip(a.tolist(), b.tolist(), wt.tolist()):
        ri, rj = root(i), root(j)
        if ri == rj:
            continue
        if low[rj] < low[ri]:
            ri, rj = rj, ri                                       # ri holds the lower ground (or as low)
        if low[rj] > low[ri]:                                     # rj's waiting hollows first reach lower ground here
            for k in waiting.pop(rj):
                depth[k] = v - floor[k]
        else:
            waiting[ri].extend(waiting.pop(rj))
        parent[rj] = ri
    return hollows, depth[1:]


def areas_by_depth(street: np.ndarray, edge: np.ndarray, keep: np.ndarray,
                   depths: Sequence[float]) -> Iterator[tuple[list[int], np.ndarray]]:
    """For each distinct set of markers among `depths`: the indexes into `depths` it serves and the areas (`areas`
    at each of those depths, exactly; see the module), in the order of each set's first depth."""
    ground = _ground(street, edge, keep)
    hollows, depth = _depths(ground)
    span = float(ground.max() - ground.min())
    groups: dict[bytes, list[int]] = {}
    for i, d in enumerate(depths):
        kept = (depth >= d) & (d <= span)          # h_minima finds nothing at a depth above the ground's whole span
        groups.setdefault(kept.tobytes(), []).append(i)
    for key, which in groups.items():
        kept = np.r_[False, np.frombuffer(key, bool)]
        yield which, _close(ground, np.where(kept[hollows], hollows, 0), street, keep)
