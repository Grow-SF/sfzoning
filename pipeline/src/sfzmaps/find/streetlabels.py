"""The street labels on the solver's grid, made from today's blocks and today's street centre lines: every px off
today's blocks that lies within the learned half-widths of its nearest centre line, short of that side's block, is
street, weighted by how often a street of its type is that wide (`street_labels`); today's blocks are not street.
`street_grid` makes them for the whole grid, with their counts.  The finder's lot labels are these
(`find.lotlabels`).

Moved here unchanged from the 1921 hatching reader's training data (`solver.hatchnet.data`, branch
`hatch-experiments-archive`).  Reads only the centre lines' file it is given (`lines_from`); writes nothing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import shapely
from scipy import ndimage

from sfzmaps.solver import grid

UNKNOWN = 255                           # `block` where the place is not this sheet's to label
ROWS = 512                              # page rows laid at once (memory only: the result doesn't depend on it)
STREET, NOT_STREET = 1, 0               # `street`'s labels; UNKNOWN (255) is ignored
SUBPIXEL_BITS = 4                       # line points go to cv2 to 1/16 px (as `lots.lot_rasters` fills the lots)
KIND_NONE = "(none)"                    # the street type of a line with none (freeways, The Embarcadero, ...)
POOLED = "(pooled)"                     # the half-widths of every street type together
CONFIDENCE = 0.95                       # a street type keeps its own half-widths only when their distribution differs
                                        # from all types' by more than the DKW band at this confidence for its number
                                        # of lines (the one setting the learning needs: the usual 95%)


@dataclass(frozen=True)
class Lines:
    """Today's street centre lines: each one's geometry (EPSG:2227 ft), street type and street."""
    geoms: np.ndarray
    kinds: np.ndarray
    streets: np.ndarray


def _type_from_name(name, known: set[str]) -> str:
    """A line with no `st_type`: its name's last word when that is a type other lines have (OCTAVIA ST -> ST), else
    KIND_NONE (freeways, THE EMBARCADERO, BROADWAY)."""
    last = name.split()[-1] if isinstance(name, str) and name.split() else ""
    return last if last in known else KIND_NONE


def lines_from(path: Path) -> Lines:
    """Today's street centre lines (DataSF's, data/ref/streets.parquet, EPSG:2227 ft as the solver's grid), every row:
    the ones no longer active too, since 1921's streets are older than today's list.  Each one's street type
    (`st_type`; where missing, `_type_from_name`) and street (`streetname`; a line with none is its own street)."""
    df = pd.read_parquet(path)
    geoms = np.array([shapely.from_wkb(v) if isinstance(v, (bytes, bytearray)) else v for v in df["geometry"]],
                     dtype=object)
    n = len(df)
    kinds = df["st_type"].astype(object).to_numpy() if "st_type" in df else np.full(n, None, dtype=object)
    names = df["streetname"].astype(object).to_numpy() if "streetname" in df else np.full(n, None, dtype=object)
    known = {k for k in kinds if isinstance(k, str) and k}
    kinds = np.array([k if isinstance(k, str) and k else _type_from_name(nm, known) for k, nm in zip(kinds, names)],
                     dtype=object)
    streets = np.array([s if isinstance(s, str) and s else f"#{i}" for i, s in enumerate(names)], dtype=object)
    keep = ~(shapely.is_missing(geoms) | shapely.is_empty(geoms)) if n else np.zeros(0, bool)
    return Lines(geoms[keep], kinds[keep], streets[keep])


@dataclass(frozen=True)
class Centres:
    """Today's centre lines on the solver's grid, segment by segment (two consecutive vertices of a line), each
    8-connected and 1 px wide (cv2, to 1/16 px), a later segment over an earlier one.  Grid px (c, r) covers [c, c+1)
    x [r, r+1) and cv2 takes integer coordinates as px centres, so the vertices go in at `to_grid - 0.5`.  A line off
    the grid draws nothing."""
    seg: np.ndarray            # int32 grid: the segment of each centre px, -1 none
    direction: np.ndarray      # (n, 2): each segment's unit direction on the grid (col, row)
    line: np.ndarray           # (n,): its line (index into the lines given)
    street: np.ndarray         # (n,): its street (index into the distinct streets)
    kind: np.ndarray           # (n,): its street type (index into `kinds`)
    kinds: tuple[str, ...]


def centre_segments(lines, shape: tuple[int, int], kinds=None, streets=None) -> Centres:
    """`Centres` of `lines` (EPSG:2227 ft) on a grid of `shape` from the solver grid's corner; `kinds`: each line's
    street type (KIND_NONE when None), `streets`: each line's street (each line its own when None)."""
    lines = np.asarray(lines, dtype=object)
    n = len(lines)
    kinds = np.full(n, KIND_NONE, dtype=object) if kinds is None else np.asarray(kinds, dtype=object)
    streets = np.arange(n).astype(str) if streets is None else np.asarray(streets, dtype=object)
    kind_names, kind_of = np.unique(kinds.astype(str), return_inverse=True) if n else (np.array([KIND_NONE]),
                                                                                       np.zeros(0, np.int64))
    street_of = np.unique(streets.astype(str), return_inverse=True)[1] if n else np.zeros(0, np.int64)
    seg = np.full(shape, -1, np.int32)
    none = np.zeros(0, np.int64)
    if n:
        parts, part_line = shapely.get_parts(lines, return_index=True)
        is_line = shapely.get_type_id(parts) == 1
        parts, part_line = parts[is_line], part_line[is_line]
    if not n or not len(parts):
        return Centres(seg, np.zeros((0, 2)), none, none, none, tuple(str(k) for k in kind_names))
    xy, part = shapely.get_coordinates(parts, return_index=True)
    g = grid.to_grid(xy) - 0.5
    same = part[1:] == part[:-1]
    a, b, line = g[:-1][same], g[1:][same], part_line[part[:-1][same]]
    v = b - a
    length = np.hypot(v[:, 0], v[:, 1])
    ok = length > 0
    a, b, v, length, line = a[ok], b[ok], v[ok], length[ok], line[ok]
    scale = 1 << SUBPIXEL_BITS
    pa, pb = np.rint(a * scale).astype(np.int64), np.rint(b * scale).astype(np.int64)
    for k in range(len(a)):
        cv2.line(seg, (int(pa[k, 0]), int(pa[k, 1])), (int(pb[k, 0]), int(pb[k, 1])), int(k), 1, cv2.LINE_8,
                 SUBPIXEL_BITS)
    return Centres(seg, v / length[:, None], line, street_of[line], kind_of[line], tuple(str(k) for k in kind_names))


@dataclass(frozen=True)
class Sides:
    """The centre px off today's blocks and each one's two half-widths (`side_widths`)."""
    rows: np.ndarray
    cols: np.ndarray
    seg: np.ndarray            # each one's segment
    width: np.ndarray          # (n, 2) grid px: side 0 along the normal (-dy, dx), side 1 against it; inf: unbounded


def _normals(direction: np.ndarray) -> np.ndarray:
    return np.c_[-direction[:, 1], direction[:, 0]]


def side_widths(block: np.ndarray, centres: Centres, area: np.ndarray | None, near: np.ndarray, rows: np.ndarray,
                cols: np.ndarray) -> np.ndarray:
    """Each centre px's half-width on each side (n, 2), in grid px: a ray from the px along its segment's normal, one
    px along its major axis per step, to the first block px it meets.  The side is unbounded (inf) when the ray first
    leaves the grid or the data `area` (None: the grid), or enters px nearer another street's centre line than its own
    (`near`: the nearest centre px of every px, `distance_transform_edt` indices): past there the ray runs along or
    into another street (down a cross street at a junction, across a plaza, out over water to another street), and a
    block it meets measures that street, not this one's width."""
    h, w = block.shape
    sid = centres.seg[rows, cols]
    own = centres.street[sid]
    normal = _normals(centres.direction[sid])
    width = np.full((len(rows), 2), np.inf)
    start = np.arange(len(rows)) if area is None else np.flatnonzero(area[rows, cols])
    for side, sign in enumerate((1.0, -1.0)):
        major = np.abs(normal).max(axis=1)
        step = sign * normal / major[:, None]
        act, k = start, 0
        while act.size:
            k += 1
            x = np.rint(cols[act] + k * step[act, 0]).astype(np.int64)
            y = np.rint(rows[act] + k * step[act, 1]).astype(np.int64)
            on = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x, y = np.where(on, x, 0), np.where(on, y, 0)
            if area is not None:
                on &= area[y, x]
            hit = on & block[y, x]
            width[act[hit], side] = k / major[act[hit]]
            same = centres.street[centres.seg[near[0][y, x], near[1][y, x]]] == own[act]
            act = act[on & ~hit & same]
    return width


@dataclass(frozen=True)
class HalfWidths:
    """The half-widths learned from the centre px bounded on both sides (grid px, sorted): every type's together
    (`pooled`) and the types that keep their own; `report` per type."""
    pooled: np.ndarray
    own: dict
    report: dict

    def of(self, kind: str) -> np.ndarray:
        return self.own.get(kind, self.pooled)


def survival(samples: np.ndarray, d) -> np.ndarray:
    """S(d) = P(half-width > d): the share of `samples` (sorted) beyond d; 0 with no samples.  Strictly beyond, so a
    px at d is street as often as a street's block lies past it: a side whose block is at w labels px below w."""
    d = np.asarray(d, float)
    if not len(samples):
        return np.zeros(d.shape)
    return (len(samples) - np.searchsorted(samples, d, side="right")) / len(samples)


def _ecdf_gap(a: np.ndarray, b: np.ndarray) -> float:
    """sup |F_a - F_b| of two sorted samples (at their values, where the step functions jump)."""
    v = np.union1d(a, b)
    return float(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b)).max())


def _ft(s: np.ndarray) -> dict:
    if not len(s):
        return {"median_ft": None, "p90_ft": None, "max_ft": None}
    f = s * grid.FT_PER_PX
    return {"median_ft": float(np.quantile(f, 0.5)), "p90_ft": float(np.quantile(f, 0.9)), "max_ft": float(f[-1])}


def learn_half_widths(width: np.ndarray, sid: np.ndarray, centres: Centres, per_type: bool = True) -> HalfWidths:
    """The half-widths of the centre px bounded on both sides, both sides' (the empirical distribution: nothing
    smoothed).  Per street type: its own when its two-sided px come from lines enough to tell it apart, i.e. its
    empirical CDF differs from every type's together by more than the DKW band sqrt(ln(2 / (1 - CONFIDENCE)) /
    (2 n)), n its lines (the px of one line are not independent); else the pooled ones.  `per_type` False: every type
    the pooled ones, with no test."""
    both = np.isfinite(width).all(axis=1)
    kind, line = centres.kind[sid], centres.line[sid]
    pooled = np.sort(width[both].ravel())
    own, report = {}, {POOLED: {"px": int(both.sum()), "lines": int(len(np.unique(line[both]))), **_ft(pooled)}}
    if not per_type:
        return HalfWidths(pooled, own, report)
    for k, name in enumerate(centres.kinds):
        m = both & (kind == k)
        s = np.sort(width[m].ravel())
        n = int(len(np.unique(line[m])))
        band = float(np.sqrt(np.log(2.0 / (1.0 - CONFIDENCE)) / (2.0 * n))) if n else None
        gap = _ecdf_gap(s, pooled) if len(s) else None
        keep = bool(band is not None and gap > band)
        if keep:
            own[name] = s
        report[name] = {"px": int(m.sum()), "lines": n, "own": keep, "ecdf_gap": gap, "dkw_band": band, **_ft(s)}
    return HalfWidths(pooled, own, report)


@dataclass(frozen=True)
class StreetLabels:
    street: np.ndarray         # uint8: 1 street, 0 not street, 255 ignored
    weight: np.ndarray         # float16: each label's weight (1 not street, (0, 1] street, 0 ignored)
    centre: np.ndarray         # uint8: 1 on the centre px used (those off today's blocks)
    sides: Sides
    learned: HalfWidths


def street_labels(block_world: np.ndarray, centres: Centres, area: np.ndarray | None = None,
                  learned: HalfWidths | None = None, per_type: bool = True) -> StreetLabels:
    """The street labels on the solver's 4 ft grid.  Each px x off today's blocks belongs to its nearest centre px c
    (`distance_transform_edt` indices), at distance d, on the side of c's segment that x lies on, with that side's
    block distance w (`side_widths`; inf with no block).  On every side x is street with weight S(d) = P(half-width >
    d) (`survival`) of c's street type's learned half-widths (`learn_half_widths`, from this grid's two-sided px unless
    `learned` is given) while d < w: the side's own block cuts it, and a far block (across water or open land) never
    makes full-weight street, since S is ~0 there.  Ignored where S(d) is 0 (beyond the largest half-width learned)
    and past the side's block.  A weight, never a 1/2 target.  Today's blocks are not street (weight 1).  A centre px
    on a block (a park road, a street since built over) is no centre: it would label nothing off the block.  No width
    is picked: block distances are measured, the weights learned.  `per_type`: as `learn_half_widths`."""
    block = np.asarray(block_world, bool)
    h, w = block.shape
    kept = (centres.seg >= 0) & ~block
    street = np.where(block, NOT_STREET, UNKNOWN).astype(np.uint8)
    weight = block.astype(np.float16)
    rows, cols = np.nonzero(kept)
    sid = centres.seg[rows, cols]
    if not len(rows):
        sides = Sides(rows, cols, sid, np.zeros((0, 2)))
        learned = learned if learned is not None else learn_half_widths(sides.width, sid, centres, per_type)
        return StreetLabels(street, weight, kept.astype(np.uint8), sides, learned)
    near = ndimage.distance_transform_edt(~kept, return_distances=False, return_indices=True)
    width = side_widths(block, centres, area, near, rows, cols)
    sides = Sides(rows, cols, sid, width)
    learned = learned if learned is not None else learn_half_widths(width, sid, centres, per_type)
    index = np.full(block.shape, -1, np.int32)
    index[rows, cols] = np.arange(len(rows), dtype=np.int32)
    normal = _normals(centres.direction[sid])
    kind = centres.kind[sid]
    samples = [learned.of(name) for name in centres.kinds]
    xs = np.arange(w, dtype=np.int64)
    for y0 in range(0, h, ROWS):
        band = slice(y0, min(h, y0 + ROWS))
        ri, ci = near[0][band], near[1][band]
        i = index[ri, ci]
        vy = (np.arange(band.start, band.stop, dtype=np.int64)[:, None] - ri).astype(float)
        vx = (xs - ci).astype(float)
        side = ((vx * normal[i, 0] + vy * normal[i, 1]) < 0).astype(np.int64)
        half = width[i, side]
        d = np.sqrt(vx * vx + vy * vy)
        inside = d < half
        wt = np.zeros(d.shape)
        kk = np.where(inside, kind[i], -1)
        for k in np.unique(kk[inside]):
            m = kk == k
            wt[m] = survival(samples[k], d[m])
        off = ~block[band]
        street[band][off & (wt > 0)] = STREET
        weight[band][off] = wt[off].astype(np.float16)
    return StreetLabels(street, weight, kept.astype(np.uint8), sides, learned)


def street_grid(lines: Lines, blocks: np.ndarray, area: np.ndarray | None = None,
                per_type: bool = True) -> tuple[np.ndarray, np.ndarray, dict]:
    """The street labels and weights on the whole solver grid (`street_labels` of today's blocks and `lines`, rays
    kept to the data `area`) and their counts: lines, centre px (those on blocks dropped), the centre px bounded on
    both sides, one or none, px per label, the soft street px and the street weight's sum, and the learned
    half-widths per street type (`per_type`: as `learn_half_widths`)."""
    centres = centre_segments(lines.geoms, blocks.shape, lines.kinds, lines.streets)
    lab = street_labels(blocks, centres, area, per_type=per_type)
    fin = np.isfinite(lab.sides.width).sum(axis=1)
    on = lab.street == STREET
    stats = {"lines": int(len(lines.geoms)), "centre_grid_px": int(len(lab.sides.rows)),
             "centre_on_blocks_px": int(np.count_nonzero((centres.seg >= 0) & blocks)),
             "centre_sides": {"two_sided_px": int((fin == 2).sum()), "one_sided_px": int((fin == 1).sum()),
                              "unbounded_px": int((fin == 0).sum())},
             "street_grid_px": int(np.count_nonzero(on)),
             "soft_street_grid_px": int(np.count_nonzero(on & (lab.weight < 1))),
             "street_weight_sum_grid": float(lab.weight[on].astype(np.float64).sum()),
             "not_street_grid_px": int(np.count_nonzero(lab.street == NOT_STREET)),
             "ignored_grid_px": int(np.count_nonzero(lab.street == UNKNOWN)),
             "half_widths": lab.learned.report}
    return lab.street, lab.weight, stats
