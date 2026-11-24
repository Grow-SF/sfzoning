"""Block edges as statements about short cuts: where the lots say a block side is, the finder learns that a block edge
crosses.

A cut is a short run of page px across a block side the lots give; its statement is "a block edge crosses this cut",
and which px of the cut the edge is on, the model finds itself.  Each px carries a prior: where the placement puts the
side, falling off as a Gaussian of the sheet's measured placement spread (`spread_px`), so a cut is as long as the
placement is unsure.  A cut with any px off the page or off the frame cut is dropped whole.  Each cut names the grid px
of the block it is across the side of (`Cuts.inside`), so that it can be kept only where the sheet draws the block
(`find.drawn`).  `relaid` lays a square's cuts again on a tile showing the square at another size, by the same
arithmetic (`_lay`).

Reads a sheet's placement and data/warp/<series>/<sheet>/scores.json (the placement's measured miss).  Writes nothing.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from sfzmaps.find import lotlabels
from sfzmaps.find.pages import SheetId
from sfzmaps.solver import grid, hatching

REACH_SIGMAS = 4.0                  # a cut runs this many sigma past where the line can be: numerics only, the prior
                                    # there is 3e-4 of its peak
MEDIAN_RADIUS = 1.1774              # the median distance from the centre of a 2-D normal, in sigma
LATTICE = hatching.LATTICE          # grid px between the points where the placement itself is evaluated (as
                                    # hatching.PageWorld reads a sheet the other way)
MISS = {"warp": None, "work": "today"}   # whose control-point miss (scores.json) is a placement's: the warp's is the
                                         # one it chose; the work placement is the warp's uncorrected starting point


@dataclass(frozen=True)
class Cuts:
    """Block-edge cuts on one page, px of every cut one after the other; each says a block edge crosses it."""
    px: np.ndarray          # int64 (m,): flat page index (row * w + col) of each cut px
    cut: np.ndarray         # int32 (m,): the cut each px is in
    prior: np.ndarray       # float32 (m,): 1 where the line most likely is, falling off
    weight: np.ndarray      # float32 (n,): how much the cut counts (1 as made; `find.drawn` sets 0 for a cut not used)
    inside: np.ndarray      # int32 (n,): the block's grid px the cut is across the side of, as a flat index of the
                            # sheet's grid window (`window`); -1 when not known (a cut read from a label file)
    dropped: np.ndarray     # int64: the page px (flat) of the cuts the lots call for that were dropped for leaving the
                            # page or the frame cut, those on the page

    @staticmethod
    def empty() -> Cuts:
        return Cuts(np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros(0, np.float32),
                    np.zeros(0, np.int32), np.zeros(0, np.int64))

    def with_weight(self, weight: np.ndarray) -> Cuts:
        return Cuts(self.px, self.cut, self.prior, np.asarray(weight, np.float32), self.inside, self.dropped)

    def only(self, keep: np.ndarray) -> Cuts:
        """The cuts `keep` (bool (n,)) says to keep, renumbered in order; `dropped` as it is."""
        keep = np.asarray(keep, bool)
        on = keep[self.cut]
        renumber = (np.cumsum(keep) - 1).astype(np.int32)
        return Cuts(self.px[on], renumber[self.cut[on]], self.prior[on], self.weight[keep], self.inside[keep],
                    self.dropped)

    def on(self, shape: tuple[int, int], which: np.ndarray | None = None, dropped: bool = False) -> np.ndarray:
        """Bool (h, w): the page px in any cut (`which`: bool (n,), only those cuts; `dropped`: and the px of the cuts
        dropped at the frame cut)."""
        out = np.zeros(shape[0] * shape[1], bool)
        out[self.px if which is None else self.px[which[self.cut]]] = True
        if dropped:
            out[self.dropped] = True
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


def window(tr, keep: np.ndarray, grid_shape) -> tuple[int, int, int, int]:
    """The grid rows and columns (r0, r1, c0, c1) the sheet's kept px reach (`hatching._slice`): the window every cut's
    `ends` index."""
    return hatching._slice(tr, keep, grid_shape)


def _window(tr, keep: np.ndarray, grid_shape) -> tuple[tuple[int, int, int, int], _GridPage]:
    """The part of the grid the sheet's kept px reach (`hatching._slice`), and its page px."""
    r0, r1, c0, c1 = window(tr, keep, grid_shape)
    return (r0, r1, c0, c1), _GridPage(tr, keep.shape, r0, r1, c0, c1)


def _pairs(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every 4-neighbour pair (p, q) of px of a window (flat indexes) whose values in `a` differ, across then down."""
    idx = np.arange(a.size).reshape(a.shape)
    ps, qs = [], []
    for sp, sq in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                   ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        ok = a[sp] != a[sq]
        ps.append(idx[sp][ok])
        qs.append(idx[sq][ok])
    return np.concatenate(ps), np.concatenate(qs)


def _centres(flat: np.ndarray, width: int, r0: int, c0: int) -> np.ndarray:
    """Grid (col, row) of the centres of window px `flat`."""
    r, c = np.divmod(flat, width)
    return np.c_[c + c0 + 0.5, r + r0 + 0.5]


def _lay(mid: np.ndarray, d: np.ndarray, inside: np.ndarray, sigma: float, reach: int, shape: tuple[int, int],
         keep: np.ndarray) -> Cuts:
    """Cuts across page points `mid` along unit directions `d`, `reach` steps each way.  A step is one px along the
    direction's longer axis, so a cut's px are distinct and touching; prior exp(-t^2 / 2 sigma^2) at distance t from
    `mid`."""
    n = len(mid)
    if not n:
        return Cuts.empty()
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.abs(d).max(axis=1)
        step = d / m[:, None]
    good = np.isfinite(step).all(axis=1) & np.isfinite(mid).all(axis=1)
    cut = np.repeat(np.arange(n), 2 * reach + 1)
    k = np.tile(np.arange(-reach, reach + 1), n)
    pos = mid[cut] + k[:, None] * step[cut]
    with np.errstate(invalid="ignore"):
        col, row = np.rint(pos[:, 0]), np.rint(pos[:, 1])
        inbounds = (col >= 0) & (col < shape[1]) & (row >= 0) & (row < shape[0])
    col, row = np.where(inbounds, col, 0).astype(np.int64), np.where(inbounds, row, 0).astype(np.int64)
    on_page = inbounds.copy()
    inbounds &= keep[row, col]
    good &= np.bincount(cut[~inbounds], minlength=n) == 0
    lost = on_page & ~good[cut]
    t = np.abs(k) / m[cut]
    on = good[cut]
    renumber = np.cumsum(good) - 1
    return Cuts((row * shape[1] + col)[on], renumber[cut[on]].astype(np.int32),
                np.exp(-t[on] ** 2 / (2 * sigma ** 2)).astype(np.float32), np.ones(int(good.sum()), np.float32),
                np.asarray(inside, np.int32)[good], (row * shape[1] + col)[lost])


def relaid(c: Cuts, side: int, size: int, sigma: float) -> Cuts:
    """The cuts `c` of a square `side` px wide (`px` index it flat) laid again by `_lay` on a tile of `size` px that
    shows the square at the factor f = size / side.  Each cut's centre (its px of the highest prior, prior 1 as laid;
    the first of a tie) is moved to the tile px it falls in, its direction kept (the long axis of its px), its sigma
    `sigma` times f and its reach `reach_px` of that; its weight and `inside` carry over.  A cut that no longer lies
    wholly in the tile is dropped, as one leaving the page is (`dropped` is empty, as in a window)."""
    n = len(c.weight)
    if not n:
        return Cuts.empty()
    f = size / side
    r, q = np.divmod(c.px, side)
    count = np.bincount(c.cut, minlength=n)
    mean_q, mean_r = (np.bincount(c.cut, v, n) / np.maximum(count, 1) for v in (q, r))
    dq, dr = q - mean_q[c.cut], r - mean_r[c.cut]
    sqq, srr, sqr = (np.bincount(c.cut, v, n) for v in (dq * dq, dr * dr, dq * dr))
    angle = 0.5 * np.arctan2(2 * sqr, sqq - srr)                  # the long axis of each cut's px
    order = np.lexsort((-c.prior, c.cut))
    first = order[np.r_[True, np.diff(c.cut[order]) != 0]] if len(order) else order
    centre = np.full((n, 2), np.nan)                              # a cut with no px is not laid
    centre[c.cut[first]] = np.c_[q[first], r[first]]
    mid = np.floor((centre + 0.5) * f)
    s = sigma * f
    laid = _lay(mid, np.c_[np.cos(angle), np.sin(angle)], np.arange(n), s, reach_px(s), (size, size),
                np.ones((size, size), bool))
    kept = laid.inside
    return Cuts(laid.px, laid.cut, laid.prior, c.weight[kept], c.inside[kept], np.zeros(0, np.int64))


def edge_cuts(g: lotlabels.Ground, tr, shape: tuple[int, int], keep: np.ndarray, sigma: float,
              reach: int | None = None) -> Cuts:
    """A cut across every block side: one for each 4-neighbour pair of grid px u in a block and v not, centred on
    their midpoint and running from u towards v on the page, `reach` px each way (None: ceil(4 sigma)); each says a
    block edge crosses it."""
    reach = reach_px(sigma) if reach is None else reach
    (r0, r1, c0, c1), gp = _window(tr, keep, g.block.shape)
    block = g.block[r0:r1, c0:c1]
    p, q = _pairs(block)
    flip = ~block.ravel()[p]
    u, v = np.where(flip, q, p), np.where(flip, p, q)
    width = block.shape[1]
    cu, cv = _centres(u, width, r0, c0), _centres(v, width, r0, c0)
    xy = gp.at(np.r_[(cu + cv) / 2, cu, cv])
    mid, d = xy[:len(u)], xy[2 * len(u):] - xy[len(u):2 * len(u)]
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return _lay(mid, d, u, sigma, reach, shape, keep)
