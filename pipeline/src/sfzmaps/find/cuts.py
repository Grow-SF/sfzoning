"""Lines as statements about short cuts: the block edges and zoning borders the finder learns, from a series' lots.

A cut is a short run of page px across a place where the lots say a line should be: a block side, the line between two
lots, or the ground between two lots facing each other across a street.  Its statement is "a line crosses this cut"
(`says` True) or "none does" (False); which px of the cut the line is on, the model finds itself.  Each px carries a
prior: where the placement puts the line, falling off as a Gaussian of the sheet's measured placement spread
(`spread_px`), so a cut is as long as the placement is unsure.  A cut with any px off the page or off the frame cut is
dropped whole.  Each cut names the blocks it speaks about (`Blocks`: 4-connected pieces of the grid's blocks in the
sheet's window), so that it can be weighted by whether the sheet draws them (`find.drawn`, `Cuts.weighted`).

Reads a sheet's placement and data/warp/<series>/<sheet>/scores.json (the placement's measured miss).  Writes nothing.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates

from sfzmaps.find import lotlabels
from sfzmaps.find.pages import SheetId
from sfzmaps.solver import grid, hatching

EDGE, BORDER = 0, 1                 # Cuts.answer: a block edge, a zoning border
REACH_SIGMAS = 4.0                  # a cut runs this many sigma past where the line can be: numerics only, the prior
                                    # there is 3e-4 of its peak
MEDIAN_RADIUS = 1.1774              # the median distance from the centre of a 2-D normal, in sigma
LATTICE = hatching.LATTICE          # grid px between the points where the placement itself is evaluated (as
                                    # hatching.PageWorld reads a sheet the other way)
MISS = {"warp": None, "work": "today"}   # whose control-point miss (scores.json) is a placement's: the warp's is the
                                         # one it chose; the work placement is the warp's uncorrected starting point


@dataclass(frozen=True)
class Cuts:
    """Cuts on one page, px of every cut one after the other."""
    px: np.ndarray          # int64 (m,): flat page index (row * w + col) of each cut px
    cut: np.ndarray         # int32 (m,): the cut each px is in
    prior: np.ndarray       # float32 (m,): 1 where the line most likely is, falling off
    answer: np.ndarray      # uint8 (n,): EDGE or BORDER
    says: np.ndarray        # bool (n,): True "a line crosses this cut", False "none does"
    weight: np.ndarray      # float32 (n,): how much the cut counts (`weighted`; 1 as made)
    blocks: np.ndarray      # int32 (n, 2): the blocks (`Blocks` ids) it speaks about: an edge cut its inside block and
                            # -1, a zoning-border cut the blocks of its two lots

    @staticmethod
    def empty() -> Cuts:
        return Cuts(np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros(0, np.uint8),
                    np.zeros(0, bool), np.zeros(0, np.float32), np.zeros((0, 2), np.int32))

    @staticmethod
    def joined(a: Cuts, b: Cuts) -> Cuts:
        """`a`'s cuts, then `b`'s."""
        return Cuts(np.r_[a.px, b.px], np.r_[a.cut, b.cut + len(a.answer)].astype(np.int32), np.r_[a.prior, b.prior],
                    np.r_[a.answer, b.answer], np.r_[a.says, b.says], np.r_[a.weight, b.weight],
                    np.r_[a.blocks, b.blocks].astype(np.int32))

    def weighted(self, drawn: np.ndarray) -> Cuts:
        """The cuts weighted by the chance each of their blocks is drawn (`drawn[block]`): an edge cut by its block's,
        a zoning-border cut by the product of its two blocks'."""
        d = np.r_[np.asarray(drawn, np.float64), 1.0]                            # a block id of -1 reads 1
        w = d[self.blocks[:, 0]] * d[self.blocks[:, 1]]
        return Cuts(self.px, self.cut, self.prior, self.answer, self.says, w.astype(np.float32), self.blocks)

    def on(self, shape: tuple[int, int], which: np.ndarray | None = None) -> np.ndarray:
        """Bool (h, w): the page px in any cut (`which`: bool (n,), only those cuts)."""
        out = np.zeros(shape[0] * shape[1], bool)
        out[self.px if which is None else self.px[which[self.cut]]] = True
        return out.reshape(shape)


def spread_px(sid: SheetId, which: str = "warp") -> float:
    """The sheet's placement spread in page px: the median control-point miss of the placement (`MISS`, from the
    warp's scores.json) over `MEDIAN_RADIUS`, combined in quadrature with the solver grid's own 4 ft / sqrt(12) in the
    sheet's page px (the placement's median feet per px)."""
    scores = json.loads((lotlabels.PLACEMENTS["warp"] / sid[0] / sid[1] / "scores.json").read_text())
    miss = scores[f"control_miss_p50_{MISS[which] or scores['chosen']}"]
    ft_per_px = lotlabels.placement(sid, which).median_scale_ft_per_px(lotlabels.page_shape(sid))
    return float(np.hypot(miss / MEDIAN_RADIUS, grid.FT_PER_PX / math.sqrt(12) / ft_per_px))


def reach_px(sigma: float) -> int:
    return math.ceil(REACH_SIGMAS * sigma)


class _GridPage:
    """Page px (x, y) of solver grid points (col, row) in a window of the grid, as the placement's `to_pixel` puts
    them: its uncorrected placement's px plus its correction there.  The uncorrected placement is smooth and is read
    at every LATTICE grid px (`grid_to_pixel`, to_pixel for a large batch on a page of `shape`), bilinear between
    (within 0.003 px of to_pixel on the 2000 sheets); the correction can change sharply between those points (by up
    to 7 px) and is added at each point exactly."""

    def __init__(self, tr, shape: tuple[int, int], r0: int, r1: int, c0: int, c1: int):
        self.correction = tr.correction
        base = tr if tr.correction is None else tr.uncorrected()
        self.cols = np.arange(c0 - LATTICE, c1 + 2 * LATTICE, LATTICE, dtype=float)
        self.rows = np.arange(r0 - LATTICE, r1 + 2 * LATTICE, LATTICE, dtype=float)
        cc, rr = np.meshgrid(self.cols, self.rows)
        p = np.asarray(base.grid_to_pixel(grid.to_map(np.c_[cc.ravel(), rr.ravel()]), shape), float)
        self.nodes = p.reshape(rr.shape + (2,))

    def at(self, colrow: np.ndarray) -> np.ndarray:
        colrow = np.asarray(colrow, float).reshape(-1, 2)
        rc = [(colrow[:, 1] - self.rows[0]) / LATTICE, (colrow[:, 0] - self.cols[0]) / LATTICE]
        p = np.stack([map_coordinates(self.nodes[..., k], rc, order=1, mode="nearest") for k in (0, 1)], -1)
        if self.correction is not None:
            ok = np.isfinite(p).all(axis=1)
            p[ok] = p[ok] + self.correction.u(p[ok])
        return p


@dataclass(frozen=True)
class Blocks:
    """The blocks of a sheet's window of the grid (`hatching._slice`): the 4-connected pieces of the grid's blocks,
    numbered 0 .. count - 1 in scan order.  `id` (int32, the window's shape): each block px its block, each px off the
    blocks its nearest block (-1 when the window holds no block)."""
    window: tuple[int, int, int, int]
    id: np.ndarray
    count: int


def blocks(block: np.ndarray, tr, keep: np.ndarray) -> Blocks:
    """The `Blocks` of the grid's blocks `block` (bool) in the window the sheet's kept px reach."""
    r0, r1, c0, c1 = hatching._slice(tr, keep, block.shape)
    return _blocks(block[r0:r1, c0:c1], (r0, r1, c0, c1))


def _blocks(win: np.ndarray, window) -> Blocks:
    lab, count = ndimage.label(win)                     # the default structure: 4-connected
    ids = (lab - 1).astype(np.int32)
    if count:
        near = ndimage.distance_transform_edt(lab == 0, return_distances=False, return_indices=True)
        ids = ids[near[0], near[1]]
    return Blocks(window, ids, int(count))


def _window(tr, keep: np.ndarray, grid_shape) -> tuple[tuple[int, int, int, int], _GridPage]:
    """The part of the grid the sheet's kept px reach (`hatching._slice`), and its page px."""
    r0, r1, c0, c1 = hatching._slice(tr, keep, grid_shape)
    return (r0, r1, c0, c1), _GridPage(tr, keep.shape, r0, r1, c0, c1)


def _pairs(a: np.ndarray, b: np.ndarray | None = None, *, test) -> tuple[np.ndarray, np.ndarray]:
    """Every 4-neighbour pair (p, q) of px of a window (flat indexes), across then down, for which `test(a[p], a[q])`
    (or `test(a[p], a[q], b[p], b[q])`) holds."""
    idx = np.arange(a.size).reshape(a.shape)
    ps, qs = [], []
    for sp, sq in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                   ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        ok = test(a[sp], a[sq]) if b is None else test(a[sp], a[sq], b[sp], b[sq])
        ps.append(idx[sp][ok])
        qs.append(idx[sq][ok])
    return np.concatenate(ps), np.concatenate(qs)


def _centres(flat: np.ndarray, width: int, r0: int, c0: int) -> np.ndarray:
    """Grid (col, row) of the centres of window px `flat`."""
    r, c = np.divmod(flat, width)
    return np.c_[c + c0 + 0.5, r + r0 + 0.5]


def _lay(a: np.ndarray, b: np.ndarray, d: np.ndarray, answer: int, says: np.ndarray, owners: np.ndarray, sigma: float,
         reach: int, shape: tuple[int, int], keep: np.ndarray) -> Cuts:
    """Cuts from page points `a` to `b` (equal for a cut across one place) along unit directions `d`, `reach` steps
    past both ends.  A step is one px along the direction's longer axis, so a cut's px are distinct and touching;
    prior 1 from a to b and exp(-t^2 / 2 sigma^2) at distance t past them."""
    n = len(a)
    if not n:
        return Cuts.empty()
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.abs(d).max(axis=1)
        step = d / m[:, None]
        span = np.rint(np.linalg.norm(b - a, axis=1) * m)
    good = np.isfinite(step).all(axis=1) & np.isfinite(a).all(axis=1) & np.isfinite(span)
    span = np.where(good, span, 0).astype(np.int64)
    length = span + 2 * reach + 1
    cut = np.repeat(np.arange(n), length)
    k = np.arange(len(cut)) - np.repeat(np.cumsum(length) - length, length) - reach
    pos = a[cut] + k[:, None] * step[cut]
    with np.errstate(invalid="ignore"):
        col, row = np.rint(pos[:, 0]), np.rint(pos[:, 1])
        inside = (col >= 0) & (col < shape[1]) & (row >= 0) & (row < shape[0])
    col, row = np.where(inside, col, 0).astype(np.int64), np.where(inside, row, 0).astype(np.int64)
    inside &= keep[row, col]
    good &= np.bincount(cut[~inside], minlength=n) == 0
    t = (np.maximum(-k, 0) + np.maximum(k - span[cut], 0)) / m[cut]
    on = good[cut]
    renumber = np.cumsum(good) - 1
    return Cuts((row * shape[1] + col)[on], renumber[cut[on]].astype(np.int32),
                np.exp(-t[on] ** 2 / (2 * sigma ** 2)).astype(np.float32), np.full(int(good.sum()), answer, np.uint8),
                np.asarray(says, bool)[good], np.ones(int(good.sum()), np.float32),
                np.asarray(owners, np.int32).reshape(-1, 2)[good])


def _across(p: np.ndarray, q: np.ndarray, width: int, r0: int, c0: int, gp: _GridPage, answer: int,
            says: np.ndarray, owners: np.ndarray, sigma: float, reach: int, shape, keep) -> Cuts:
    """A cut across the place between window px p and q, from p towards q: centred on their midpoint."""
    cp, cq = _centres(p, width, r0, c0), _centres(q, width, r0, c0)
    xy = gp.at(np.r_[(cp + cq) / 2, cp, cq])
    mid, d = xy[:len(p)], xy[2 * len(p):] - xy[len(p):2 * len(p)]
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return _lay(mid, mid, d, answer, says, owners, sigma, reach, shape, keep)


def edge_cuts(g: lotlabels.Ground, tr, shape: tuple[int, int], keep: np.ndarray, sigma: float,
              reach: int | None = None) -> Cuts:
    """A cut across every block side: one for each 4-neighbour pair of grid px u in a block and v not, centred on
    their midpoint and running from u towards v on the page, `reach` px each way (None: ceil(4 sigma)); each says a
    block edge crosses it."""
    reach = reach_px(sigma) if reach is None else reach
    (r0, r1, c0, c1), gp = _window(tr, keep, g.block.shape)
    block = g.block[r0:r1, c0:c1]
    ids = _blocks(block, (r0, r1, c0, c1)).id.ravel()
    p, q = _pairs(block, test=lambda x, y: x != y)
    flip = ~block.ravel()[p]
    u, v = np.where(flip, q, p), np.where(flip, p, q)
    owners = np.c_[ids[u], np.full(len(u), -1)]
    return _across(u, v, block.shape[1], r0, c0, gp, EDGE, np.ones(len(u), bool), owners, sigma, reach, shape, keep)


def border_cuts(lot: np.ndarray, codes: np.ndarray, tr, shape: tuple[int, int], keep: np.ndarray, sigma: float,
                reach: int | None = None, block: np.ndarray | None = None) -> Cuts:
    """Cuts saying whether a zoning border crosses, from the lots (`lot`, the grid's lot index, -1 none) and each
    lot's code (`codes[i]`, "" unknown: no statement).  Two lots touching: a cut across every 4-neighbour pair of
    their px, as `edge_cuts`.  Across a street: for every 4-neighbour pair of grid px off the lots whose nearest lots
    differ, a cut on the straight line from one's nearest lot px to the other's, `reach` px past both (prior 1 between
    the two lot px).  Each says a zoning border crosses when the two lots' codes differ.  `block`: the grid's blocks
    the cuts name (None: the lots with their enclosed holes filled)."""
    reach = reach_px(sigma) if reach is None else reach
    (r0, r1, c0, c1), gp = _window(tr, keep, lot.shape)
    win = lot[r0:r1, c0:c1]
    bid = _blocks(ndimage.binary_fill_holes(win >= 0) if block is None else block[r0:r1, c0:c1],
                  (r0, r1, c0, c1)).id.ravel()                                  # each px's block
    names, ids = np.unique(np.asarray(codes, str), return_inverse=True)
    ids = np.where(names[ids] == "", -1, ids).astype(np.int64)
    code = np.where(win >= 0, ids[np.maximum(win, 0)], -1)                       # each px's code id, -1 none
    width = win.shape[1]

    p, q = _pairs(win, code, test=lambda a, b, ca, cb: (a >= 0) & (b >= 0) & (a != b) & (ca >= 0) & (cb >= 0))
    cp, cq = code.ravel()[p], code.ravel()[q]
    touching = _across(p, q, width, r0, c0, gp, BORDER, cp != cq, np.c_[bid[p], bid[q]], sigma, reach, shape, keep)

    if not (win >= 0).any():
        return touching
    near = ndimage.distance_transform_edt(win < 0, return_distances=False, return_indices=True)
    near = (near[0] * width + near[1]).ravel()                                   # each px's nearest lot px
    nlot = np.where(win.ravel() < 0, win.ravel()[near], -1).reshape(win.shape)   # off the lots: the nearest lot
    p, q = _pairs(nlot, test=lambda a, b: (a >= 0) & (b >= 0) & (a != b))
    a, b = near[p], near[q]
    ca, cb = code.ravel()[a], code.ravel()[b]
    known = (ca >= 0) & (cb >= 0)
    a, b, ca, cb = a[known], b[known], ca[known], cb[known]
    lo, hi = np.minimum(a, b), np.maximum(a, b)                                  # many pairs share two lot px
    _, first = np.unique(lo * win.size + hi, return_index=True)
    lo, hi, same = lo[first], hi[first], ca[first] == cb[first]
    ends = gp.at(np.r_[_centres(lo, width, r0, c0), _centres(hi, width, r0, c0)])
    pa, pb = ends[:len(lo)], ends[len(lo):]
    d = (pb - pa) / np.linalg.norm(pb - pa, axis=1, keepdims=True)
    street = _lay(pa, pb, d, BORDER, ~same, np.c_[bid[lo], bid[hi]], sigma, reach, shape, keep)
    return Cuts.joined(touching, street)
