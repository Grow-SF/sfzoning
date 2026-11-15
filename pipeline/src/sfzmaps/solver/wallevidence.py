"""What the merged page's ink says about each pair of neighbouring lots: is a printed district line drawn between
them?  Nothing here decides anything: every measure is a column of features that the wall fit (`walls`) weighs.

The ink is read through its strokes (`strokes.npz`: the raw page's centre lines, text and lines alike, each with its
median width, width spread, length and spur tag).  A pair's border is its border px: both px of every 4-neighbour px
pair where `grown` changes between the pair's two lots.  Every feature is a mean over the border px.  Distances go
into hat functions over DISTANCE_KNOTS (ft); "near" below means hats of the distance from the border px.

The feature groups, in column order (GROUPS):
  stroke evidence   (60)  the nearest stroke (where strokes cross, the widest is drawn on top): near x hats of its
                          median width (WIDTH_KNOTS), and near x its letter share, its log length, its steadiness
                          (width spread / median width, capped at STEADINESS_CAP) and its spur tag
  whole line        (30)  the nearest stroke's whole line (`whole_lines`): near x its log length, whether it closes on
                          itself, the strokes of other lines ending on it, its sharpest turn, its width
  straight closing  (30)  the nearest straight closing (`straight_closings`: from a loose stroke end to the nearest
                          point of another stroke within STRAIGHT_REACH): near x [1, its log length, how well it
                          carries on the end's direction, the smaller width at its ends, its share over lots]
  enclosure         (16)  whether the strokes and the closing paths separate the two lots' regions (`enclosure`), one
                          column per level of ENCLOSURE_WIDTHS x ENCLOSURE_CLOSINGS; the city limit is a wall at
                          every level
  closing path      (24)  the nearest closing path (`closings`: the shortest way over street ground and lot lines
                          from a loose end to another stroke or the city limit, within CLOSING_REACH): near x [1, its
                          log length, how well it carries on, the smaller width at its ends]
  profile           (48)  for each width of PROFILE_LEVELS, the nearest stroke at least that wide: near, and near
                          (within 26 ft) x its letter share and log length

Widths are scaled by WIDTH_SCALE and capped at WIDTH_CAP; lengths enter as log(1 + ft) over a fixed scale, so the
fit's ridge sees columns of like size.  The city limit is the edge of the mapped area (`sheets.png` > 0), the
raster's own edge included.
"""
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from sfzmaps.solver import grid

log = logging.getLogger(__name__)

DISTANCE_KNOTS = np.array([0.0, 6.0, 14.0, 26.0, 42.0, 64.0])   # ft from a border px to a stroke or closing
WIDTH_KNOTS = np.array([0.0, 8.0, 14.0, 20.0, 28.0, 40.0])      # ft, a stroke's median width
PROFILE_LEVELS = (0.0, 6.0, 12.0, 18.0, 24.0, 32.0)              # ft, the least stroke width of each profile level
ENCLOSURE_WIDTHS = (6.0, 12.0, 18.0, 26.0)                       # ft, the least stroke width drawn as a wall
ENCLOSURE_CLOSINGS = (0.0, 60.0, 150.0, 400.0)                   # ft, the longest closing drawn as a wall
CLOSING_REACH = 512.0       # ft: how far a closing path is searched for (a search reach; the fit weighs the length)
STRAIGHT_REACH = 250.0      # ft: the same for a straight closing
DIRECTION_FT = 30.0         # a stroke end's direction for whole lines: towards its point this far along the stroke
END_POINTS = 4              # ... and for a closing path: towards the stroke's point this many points in
LOT_SAMPLE_FT = 4.0         # a straight closing's share over lots is sampled about this often
CITY_LIMIT_WIDTH = 40.0     # ft: the width a closing on the city limit counts at its far end
STEADINESS_CAP = 3.0        # width spread / median width, capped
WIDTH_SCALE = 40.0          # ft: a width enters as width / WIDTH_SCALE ...
WIDTH_CAP = 1.5             # ... capped at this
NEAR_FT = 26.0              # the profile's letter share and length count while the stroke is within this
BAND_ROWS = 1024            # grid rows of loose ends per closing job (memory and parallelism only)

GROUPS = (("stroke evidence", 60), ("whole line", 30), ("straight closing", 30), ("enclosure", 16),
          ("closing path", 24), ("profile", 48))
N_FEATURES = sum(k for _, k in GROUPS)


def hats(x, knots) -> np.ndarray:
    """Hat functions of `x` (n,) over `knots` (k,), as float32 (n, k): each row is 1 at its knot and falls linearly to
    0 at the next ones, the rows sum to 1, and x outside the knots counts as the end knot."""
    knots = np.asarray(knots, float)
    x = np.clip(np.asarray(x, float), knots[0], knots[-1])
    out = np.zeros((len(x), len(knots)), np.float32)
    j = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, len(knots) - 2)
    t = (x - knots[j]) / (knots[j + 1] - knots[j])
    rows = np.arange(len(x))
    out[rows, j] = 1 - t
    out[rows, j + 1] += t
    return out


def _px(xy) -> np.ndarray:
    """Map points (n, 2) -> the grid px (col, row) holding them, int32."""
    return np.floor(grid.to_grid(np.asarray(xy, float))).astype(np.int32)


def _draw(raster: np.ndarray, xy, value: int) -> None:
    """Draw the path `xy` (map ft) on `raster` as a 1 px 8-connected line of `value`."""
    cv2.polylines(raster, [_px(xy).reshape(-1, 1, 2)], False, int(value), 1)


def stroke_raster(strokes, shape) -> np.ndarray:
    """Each stroke's id on the grid (int32, -1 where none), as 1 px centre lines drawn narrowest first, so where
    strokes cross the widest is on top."""
    sid = np.full(shape, -1, np.int32)
    xy, off = strokes["xy"], strokes["off"]
    for k in np.argsort(strokes["wmed"], kind="stable"):
        _draw(sid, xy[off[k]:off[k + 1]], k)
    return sid


def city_limit(mapped: np.ndarray) -> np.ndarray:
    """The px of the mapped area with a 4-neighbour outside it (the raster's edge counts as outside)."""
    m = np.pad(np.asarray(mapped, bool), 1)
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    return m[1:-1, 1:-1] & ~inner


class Border:
    """Every pair's border px: rows, cols and the pair row (`pair`) of both px of each 4-neighbour px pair where
    `grown` changes between the pair's two lots; `mean(v)` is each pair's mean of a value per border px."""

    def __init__(self, lots):
        grown, n = lots.grown, lots.n
        keys = lots.a.astype(np.int64) * n + lots.b
        self.pairs = len(keys)
        rows, cols, pair = [], [], []
        for d in (0, 1):
            ga, gb = (grown[:, :-1], grown[:, 1:]) if d == 0 else (grown[:-1], grown[1:])
            m = ga != gb
            r, c = np.nonzero(m)
            lo = np.minimum(ga[m], gb[m]).astype(np.int64)
            hi = np.maximum(ga[m], gb[m]).astype(np.int64)
            del m
            k = np.minimum(np.searchsorted(keys, lo * n + hi), len(keys) - 1)
            ok = keys[k] == lo * n + hi
            for dr, dc in ((0, 0), (d, 1 - d)):
                rows.append((r[ok] + dr).astype(np.int32))
                cols.append((c[ok] + dc).astype(np.int32))
                pair.append(k[ok].astype(np.int32))
        self.rows, self.cols, self.pair = np.concatenate(rows), np.concatenate(cols), np.concatenate(pair)
        self.count = np.maximum(np.bincount(self.pair, minlength=self.pairs), 1).astype(float)

    def mean(self, values) -> np.ndarray:
        """Each pair's mean over its border px of `values` (one per border px)."""
        return np.bincount(self.pair, values, minlength=self.pairs) / self.count

    def nearest(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each border px, the id of the nearest px of `ids` that holds one (>= 0) and its distance in ft
        (`cv2.distanceTransformWithLabels`, L2 with a 5 x 5 mask); -1 and inf where `ids` holds none."""
        hit = ids >= 0
        if not hit.any():
            return np.full(len(self.rows), -1, np.int32), np.full(len(self.rows), np.inf, np.float32)
        dist, lab = cv2.distanceTransformWithLabels((~hit).view(np.uint8), cv2.DIST_L2, 5,
                                                    labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(hit)
        del hit
        to_id = np.full(int(lab.max()) + 1, -1, np.int32)
        to_id[lab[ys, xs]] = ids[ys, xs]
        near = to_id[lab[self.rows, self.cols]]
        return near, dist[self.rows, self.cols] * np.float32(grid.FT_PER_PX)

    def near_hats(self, near, dist) -> np.ndarray:
        """Hats of each border px's distance over DISTANCE_KNOTS, 0 where there is nothing (near < 0)."""
        ok = near >= 0
        return hats(np.where(ok, dist, DISTANCE_KNOTS[-1]), DISTANCE_KNOTS) * ok[:, None]


def _width(w) -> np.ndarray:
    return np.clip(np.asarray(w, float) / WIDTH_SCALE, 0, WIDTH_CAP)


def letter_share(strokes, text: np.ndarray) -> np.ndarray:
    """Per stroke, the share of its points (on the raster `text`) that are text (True); 0 for a stroke with no
    point on the raster."""
    xy, off = strokes["xy"], strokes["off"]
    n = len(off) - 1
    cr = _px(xy)
    h, w = text.shape
    on = (cr[:, 0] >= 0) & (cr[:, 0] < w) & (cr[:, 1] >= 0) & (cr[:, 1] < h)
    t = np.zeros(len(cr))
    t[on] = text[cr[on, 1], cr[on, 0]]
    sid = np.repeat(np.arange(n), np.diff(off))
    num, den = np.bincount(sid, t, n), np.bincount(sid, on.astype(float), n)
    return np.where(den > 0, num / np.maximum(den, 1), 0.0)


def text_rings(boxes: list[dict], quad, cuts) -> list[np.ndarray]:
    """A sheet's text boxes as page px rings (4, 2): each box's word rectangle (`textnet.data.word_rect`: the axis box
    turned to its reading angle), kept when its centre is inside the sheet's crop quad and in none of its cut boxes
    ([x0, y0, x1, y1], both ends inside)."""
    from sfzmaps.solver.textnet.data import word_rect
    q = np.asarray(quad, float)
    out = []
    for b in boxes:
        ring = word_rect(b["box"], float(b.get("angle_deg") or 0.0))
        cx, cy = ring.mean(0)
        inside = all((q[(e + 1) % 4, 0] - q[e, 0]) * (cy - q[e, 1]) - (q[(e + 1) % 4, 1] - q[e, 1]) * (cx - q[e, 0])
                     >= 0 for e in range(4))
        if inside and not any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in cuts):
            out.append(ring)
    return out


def text_raster(rings_map: list[np.ndarray], shape) -> np.ndarray:
    """The grid px (bool) whose centres are inside any of the rings (map ft)."""
    out = np.zeros(shape, np.uint8)
    for ring in rings_map:
        pts = np.rint((grid.to_grid(ring) - 0.5) * 16).astype(np.int32)
        cv2.fillPoly(out, [pts], 1, shift=4)
    return out.view(bool)


def _direction(xy, off, k: int, at_start: bool) -> np.ndarray:
    """The unit direction from stroke k's end into the stroke, towards its point DIRECTION_FT along it."""
    p = xy[off[k]:off[k + 1]].astype(float)
    if not at_start:
        p = p[::-1]
    d = np.cumsum(np.r_[0, np.linalg.norm(np.diff(p, axis=0), axis=1)])
    j = int(np.searchsorted(d, min(DIRECTION_FT, d[-1])))
    v = p[min(j, len(p) - 1)] - p[0]
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.zeros(2)


def loose_ends(strokes) -> list[tuple[int, int]]:
    """Every loose end (a node with this stroke alone; loops have none) as (stroke, 0 for its first end / 1 for its
    last): first ends in stroke order, then last ends."""
    a, b = strokes["a"], strokes["b"]
    degree = np.bincount(np.r_[a, b])
    return ([(k, 0) for k in range(len(a)) if degree[a[k]] == 1 and a[k] != b[k]]
            + [(k, 1) for k in range(len(a)) if degree[b[k]] == 1 and a[k] != b[k]])


def whole_lines(strokes) -> dict[str, np.ndarray]:
    """Strokes joined into whole lines: at every node the stroke ends are paired by how straight they carry on (the
    straightest pairing first, each end used once), and strokes joined through paired ends make one line.  Returns
    `line` (per stroke) and per line: `length` (ft), `width` (the length-weighted median of its strokes' median
    widths), `turn` (its sharpest pairing: 1 + cos of the two ends' directions, 0 straight on, 2 doubling back),
    `closed` (it closes on itself) and `stems` (the nodes it shares with strokes of other lines)."""
    xy, off, a, b = strokes["xy"], strokes["off"], strokes["a"], strokes["b"]
    wmed, length = strokes["wmed"].astype(float), strokes["length"].astype(float)
    n = len(a)
    ends: dict[int, list] = {}
    for k in range(n):
        if a[k] == b[k]:
            continue
        ends.setdefault(int(a[k]), []).append((k, _direction(xy, off, k, True)))
        ends.setdefault(int(b[k]), []).append((k, _direction(xy, off, k, False)))
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    turn_in = np.zeros(n)
    for es in ends.values():
        if len(es) < 2:
            continue
        cand = sorted((1 + float(np.dot(es[i][1], es[j][1])), i, j)
                      for i in range(len(es)) for j in range(i + 1, len(es)))
        used = set()
        for t, i, j in cand:
            if i in used or j in used:
                continue
            used |= {i, j}
            ki, kj = es[i][0], es[j][0]
            parent[find(ki)] = find(kj)
            turn_in[ki] = max(turn_in[ki], t)
            turn_in[kj] = max(turn_in[kj], t)
    root = np.array([find(i) for i in range(n)])
    _, line = np.unique(root, return_inverse=True)
    n_lines = int(line.max()) + 1 if n else 0
    order = np.lexsort((wmed, line))
    counts = np.bincount(line, minlength=n_lines)
    width = np.zeros(n_lines)
    start = 0
    for li in range(n_lines):
        k = order[start:start + counts[li]]
        c = np.cumsum(length[k])
        width[li] = wmed[k][np.searchsorted(c, c[-1] / 2)]
        start += counts[li]
    turn = np.zeros(n_lines)
    np.maximum.at(turn, line, turn_in)
    closed = np.zeros(n_lines, bool)
    closed[line[a == b]] = True
    # every node of a line of two or more strokes met an even number of times: it closes on itself
    node_line, node_count = np.unique(np.c_[np.r_[line, line], np.r_[a, b]], axis=0, return_counts=True)
    odd = np.bincount(node_line[:, 0], (node_count % 2 == 1).astype(float), n_lines) > 0
    closed |= (counts > 1) & ~odd
    stems = np.zeros(n_lines)
    for es in ends.values():
        ls = {int(line[k]) for k, _ in es}
        if len(ls) > 1:
            stems[list(ls)] += 1
    return {"line": line, "length": np.bincount(line, length, n_lines), "width": width, "turn": turn,
            "closed": closed, "stems": stems}


def straight_closings(strokes, lot: np.ndarray) -> np.ndarray:
    """From every loose end, the straight gap to the nearest point of any other stroke within STRAIGHT_REACH.  Rows
    (x0, y0, x1, y1, gap ft, align, end's width, other stroke's width, share over lots, stroke): align is +1 when the
    gap carries straight on from the end; the share over lots is of points every LOT_SAMPLE_FT along the gap."""
    xy, off, wmed = strokes["xy"].astype(float), strokes["off"], strokes["wmed"].astype(float)
    sid_pt = np.repeat(np.arange(len(off) - 1), np.diff(off))
    tree = cKDTree(xy)
    h, w = lot.shape
    rows = []
    for k, e in loose_ends(strokes):
        p0 = xy[off[k]] if e == 0 else xy[off[k + 1] - 1]
        into = _direction(xy, off, k, e == 0)
        idx = np.array(tree.query_ball_point(p0, STRAIGHT_REACH), np.int64)
        idx = idx[sid_pt[idx] != k]
        if not len(idx):
            continue
        dd = np.linalg.norm(xy[idx] - p0, axis=1)
        best = idx[np.lexsort((idx, dd))[0]]
        gap = float(np.linalg.norm(xy[best] - p0))
        if gap <= 0:
            continue
        align = float(-np.dot((xy[best] - p0) / gap, into))
        t = np.linspace(0, 1, max(2, int(gap / LOT_SAMPLE_FT)))[:, None]
        cr = _px(p0 + t * (xy[best] - p0))
        on = (cr[:, 0] >= 0) & (cr[:, 0] < w) & (cr[:, 1] >= 0) & (cr[:, 1] < h)
        over = float((lot[cr[on, 1], cr[on, 0]] >= 0).mean()) if on.any() else 0.0
        rows.append((*p0, *xy[best], gap, align, wmed[k], wmed[sid_pt[best]], over, k))
    return np.array(rows, float).reshape(-1, 10)


def _close_band(free: np.ndarray, targets: np.ndarray, row0: int, ends: list, reach_px: int) -> list:
    """The closing paths of the loose ends in one band of grid rows (`free` and `targets` are those rows, from grid
    row `row0`).  `ends`: (index, stroke, row, col, its neighbours, end point, point inside) per loose end."""
    h, w = free.shape
    out = []
    for i, k, r, c, nbr, p, q in ends:
        r -= row0
        r0, r1, c0, c1 = max(r - reach_px, 0), min(r + reach_px + 1, h), max(c - reach_px, 0), min(c + reach_px + 1, w)
        if not (r0 <= r < r1 and c0 <= c < c1):
            continue
        tgt = targets[r0:r1, c0:c1]
        own = np.isin(tgt, nbr)
        cost = np.where(free[r0:r1, c0:c1] | own, 1.0, np.inf)
        cost[r - r0, c - c0] = 1.0
        mcp = MCP_Geometric(cost, fully_connected=True)
        cum, _ = mcp.find_costs([(r - r0, c - c0)])
        ok = (tgt >= 0) & ~own & (cum <= reach_px)
        if not ok.any():
            continue
        rr, cc = np.nonzero(ok)
        j = int(np.argmin(cum[rr, cc]))
        end = (rr[j], cc[j])
        path = np.array(mcp.traceback(end))
        world = grid.to_map(np.c_[path[:, 1] + c0, path[:, 0] + r0 + row0] + 0.5)
        v_end, v_gap = p - q, world[-1] - world[0]
        align = float(np.dot(v_end, v_gap) / (np.linalg.norm(v_end) * np.linalg.norm(v_gap) + 1e-9))
        out.append((i, k, int(tgt[end]), float(cum[end]) * grid.FT_PER_PX, align, world.astype(np.float32)))
    return out


def _close_band_job(job):
    return _close_band(*job)


def closings(strokes, lot: np.ndarray, mapped: np.ndarray, sid: np.ndarray | None = None, workers: int = 1,
             band_rows: int = BAND_ROWS) -> dict[str, np.ndarray]:
    """From every loose end, the shortest way (`MCP_Geometric`, 8-connected) over street ground and lot lines (a lot
    px with a 4-neighbour of another lot), never through a lot, to another stroke or to the city limit, within
    CLOSING_REACH.  Another stroke is one sharing no node with this one; the strokes sharing a node may be crossed.
    Returns `src` (the loose end's stroke), `tgt` (the stroke reached; the number of strokes for the city limit),
    `gap` (ft), `align` (cos between the end's direction and the path's) and the paths (`xy` map ft, `off`).
    `sid`: the stroke raster (`stroke_raster`), made when None.  The loose ends are worked in bands of `band_rows`
    grid rows, on `workers` processes; the result is the same for any band size and number of workers.  The
    processes are spawned, so with workers > 1 the calling program's main module must be safe to import (behind
    `if __name__ == "__main__"`)."""
    t0 = time.perf_counter()
    xy, off, a, b = strokes["xy"].astype(float), strokes["off"], strokes["a"], strokes["b"]
    n = len(a)
    h, w = lot.shape
    reach = int(CLOSING_REACH / grid.FT_PER_PX)
    free = lot < 0
    free[:, 1:] |= lot[:, 1:] != lot[:, :-1]
    free[:, :-1] |= lot[:, :-1] != lot[:, 1:]
    free[1:, :] |= lot[1:, :] != lot[:-1, :]
    free[:-1, :] |= lot[:-1, :] != lot[1:, :]
    targets = stroke_raster(strokes, lot.shape) if sid is None else sid.copy()
    targets[city_limit(mapped)] = n
    free |= targets >= 0
    by_node: dict[int, list[int]] = {}
    for k in range(n):
        by_node.setdefault(int(a[k]), []).append(k)
        by_node.setdefault(int(b[k]), []).append(k)
    ends = []
    for i, (k, e) in enumerate(loose_ends(strokes)):
        p = xy[off[k]] if e == 0 else xy[off[k + 1] - 1]
        q = xy[min(off[k] + END_POINTS, off[k + 1] - 1)] if e == 0 else xy[max(off[k + 1] - 1 - END_POINTS, off[k])]
        c, r = _px(p[None])[0]
        nbr = np.array(sorted(set(by_node[int(a[k])]) | set(by_node[int(b[k])])), np.int32)
        ends.append((i, k, int(r), int(c), nbr, p, q))
    jobs = []
    for band in sorted({e[2] // band_rows for e in ends}):
        mine = [e for e in ends if e[2] // band_rows == band]
        lo = max(0, min(e[2] for e in mine) - reach)
        hi = min(h, max(e[2] for e in mine) + reach + 1)
        jobs.append((free[lo:hi], targets[lo:hi], lo, mine, reach))
    del free, targets
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as ex:
            found = [x for part in ex.map(_close_band_job, jobs) for x in part]
    else:
        found = [x for job in jobs for x in _close_band(*job)]
    found.sort(key=lambda x: x[0])
    log.info("closings: %d loose ends, %d closed (%.0f s)", len(ends), len(found), time.perf_counter() - t0)
    routes = [x[5] for x in found]
    return {"src": np.array([x[1] for x in found], np.int64), "tgt": np.array([x[2] for x in found], np.int64),
            "gap": np.array([x[3] for x in found], np.float32), "align": np.array([x[4] for x in found], np.float32),
            "xy": np.vstack(routes) if routes else np.zeros((0, 2), np.float32),
            "off": np.cumsum([0] + [len(r) for r in routes]).astype(np.int64)}


class LotRegions:
    """Each lot's region: the region holding most of the lot's px (wall px not counted).  Built once per lot raster,
    called once per region raster."""

    def __init__(self, lot: np.ndarray, n_lots: int):
        self.mask = lot >= 0
        self.lot_px = lot[self.mask].astype(np.int64)
        self.n_lots = n_lots

    def __call__(self, region: np.ndarray, n_regions: int) -> np.ndarray:
        """`region`: the region of each px (0 = a wall or outside the mapped area), as `cv2.connectedComponents`
        numbers them.  Returns each lot's region (0 = none)."""
        rk = region[self.mask].astype(np.int64)
        ok = rk > 0
        key, count = np.unique(self.lot_px[ok] * n_regions + rk[ok], return_counts=True)
        lot_of, region_of = key // n_regions, key % n_regions
        order = np.lexsort((-count, lot_of))
        first = np.r_[True, lot_of[order][1:] != lot_of[order][:-1]]
        out = np.zeros(self.n_lots, np.int64)
        out[lot_of[order][first]] = region_of[order][first]
        return out


def separated(region: np.ndarray, a, b) -> np.ndarray:
    """Pairs whose two lots are both placed and in different regions."""
    return (region[a] != region[b]) & (region[a] > 0) & (region[b] > 0)


def enclosure(lots, mapped: np.ndarray, sid: np.ndarray, wmed, found: dict) -> np.ndarray:
    """Whether each pair's lots are walled apart, per level (float32 (pairs, 16), column = width level x closing
    level).  At a level the walls are the city limit, every stroke at least that wide (`sid`, `wmed`) and every
    closing path (`found`, from `closings`) up to that long from a stroke that wide; the regions are what the walls
    leave 4-connected on the mapped area, and each lot is in the region holding most of its px (`LotRegions`)."""
    t0 = time.perf_counter()
    wmed = np.asarray(wmed, float)
    place = LotRegions(lots.lot, lots.n)
    limit = city_limit(mapped)
    width = np.where(sid >= 0, wmed[np.maximum(sid, 0)].astype(np.float32), np.float32(-1))
    out = np.zeros((len(lots.a), len(ENCLOSURE_WIDTHS) * len(ENCLOSURE_CLOSINGS)), np.float32)
    col = 0
    for wl in ENCLOSURE_WIDTHS:
        strokes_wall = limit | (width >= wl)
        for gl in ENCLOSURE_CLOSINGS:
            wall = strokes_wall.astype(np.uint8)
            if gl > 0:
                for i in np.flatnonzero((found["gap"] <= gl) & (wmed[found["src"]] >= wl)):
                    _draw(wall, found["xy"][found["off"][i]:found["off"][i + 1]], 1)
            n, region = cv2.connectedComponents(((wall == 0) & mapped).view(np.uint8), connectivity=4)
            del wall
            out[:, col] = separated(place(region, n), lots.a, lots.b)
            del region
            log.info("enclosure: width >= %.0f ft, closings <= %.0f ft: %d regions, %.1f%% of pairs apart",
                     wl, gl, n, 100 * out[:, col].mean())
            col += 1
    log.info("enclosure: %.0f s", time.perf_counter() - t0)
    return out


def border_features(lots, mapped: np.ndarray, strokes, letter: np.ndarray, workers: int = 1) -> np.ndarray:
    """The border evidence of every pair of neighbouring lots: float32 (pairs, N_FEATURES), in the row order of
    `lots.a` / `lots.b` and the column order of GROUPS.  `lots`: `lots.Lots` (the rasters on the grid from its
    top-left corner); `mapped`: the mapped area (bool, the lots' raster shape); `strokes`: strokes.npz's arrays;
    `letter`: each stroke's letter share; `workers`: processes for the closing paths."""
    t0 = time.perf_counter()
    shape = lots.lot.shape
    wmed = strokes["wmed"].astype(float)
    log_length = np.log1p(strokes["length"].astype(float))
    letter = np.asarray(letter, float)
    border = Border(lots)
    sid = stroke_raster(strokes, shape)
    log.info("border px %d, strokes drawn (%.0f s)", len(border.pair), time.perf_counter() - t0)
    cols = []

    near, dist = border.nearest(sid)
    ok = near >= 0
    k = np.where(ok, near, 0)
    Hd = border.near_hats(near, dist)
    Hw = hats(wmed[k], WIDTH_KNOTS)
    per = [letter[k], log_length[k] / 8.0,
           np.clip(strokes["wiqr"][k] / np.maximum(wmed[k], 1.0), 0, STEADINESS_CAP), strokes["spur"][k].astype(float)]
    for i in range(len(DISTANCE_KNOTS)):                            # stroke evidence
        cols += [border.mean(Hd[:, i] * Hw[:, j]) for j in range(len(WIDTH_KNOTS))]
        cols += [border.mean(Hd[:, i] * v) for v in per]
    del Hw, per
    lines = whole_lines(strokes)
    ln = lines["line"][k]
    per = [np.log1p(lines["length"][ln]) / 9.0, lines["closed"][ln].astype(float), np.log1p(lines["stems"][ln]) / 3.0,
           lines["turn"][ln] / 2.0, _width(lines["width"][ln])]
    for i in range(len(DISTANCE_KNOTS)):                            # whole line
        cols += [border.mean(Hd[:, i] * v) for v in per]
    del per, ln, Hd
    log.info("stroke evidence and whole lines (%.0f s)", time.perf_counter() - t0)

    straight = straight_closings(strokes, lots.lot)
    cid = np.full(shape, -1, np.int32)
    for i in np.argsort(-straight[:, 4], kind="stable"):            # short gaps drawn last, on top
        _draw(cid, straight[i, :4].reshape(2, 2), i)
    near_c, dist_c = border.nearest(cid)
    del cid
    Hc = border.near_hats(near_c, dist_c)
    c = straight[np.maximum(near_c, 0)] if len(straight) else np.zeros((len(near_c), 10))
    per = [np.ones(len(c)), np.log1p(c[:, 4]) / 5.5, c[:, 5], _width(np.minimum(c[:, 6], c[:, 7])), c[:, 8]]
    for i in range(len(DISTANCE_KNOTS)):                            # straight closing
        cols += [border.mean(Hc[:, i] * v) for v in per]
    del Hc, c, per
    log.info("straight closings: %d (%.0f s)", len(straight), time.perf_counter() - t0)

    found = closings(strokes, lots.lot, mapped, sid=sid, workers=workers)
    sep = enclosure(lots, mapped, sid, wmed, found)                 # enclosure
    cols += list(sep.T)
    cid = np.full(shape, -1, np.int32)
    for i in np.argsort(-found["gap"], kind="stable"):
        _draw(cid, found["xy"][found["off"][i]:found["off"][i + 1]], i)
    near_c, dist_c = border.nearest(cid)
    del cid
    Hc = border.near_hats(near_c, dist_c)
    j = np.maximum(near_c, 0)
    far_width = np.r_[wmed, CITY_LIMIT_WIDTH]
    if len(found["src"]):
        per = [np.ones(len(j)), np.log1p(found["gap"][j]) / 6.0, found["align"][j].astype(float),
               _width(np.minimum(wmed[found["src"][j]], far_width[found["tgt"][j]]))]
    else:
        per = [np.zeros(len(j))] * 4
    for i in range(len(DISTANCE_KNOTS)):                            # closing path
        cols += [border.mean(Hc[:, i] * v) for v in per]
    del Hc, per
    log.info("closing paths (%.0f s)", time.perf_counter() - t0)

    for lv in PROFILE_LEVELS:                                       # profile
        if lv > 0:
            near, dist = border.nearest(np.where((sid >= 0) & (wmed[np.maximum(sid, 0)] >= lv), sid, -1))
        k = np.where(near >= 0, near, 0)
        Hd = border.near_hats(near, dist)
        cols += [border.mean(Hd[:, i]) for i in range(len(DISTANCE_KNOTS))]
        close = Hd[:, DISTANCE_KNOTS < NEAR_FT].sum(1)
        cols += [border.mean(close * letter[k]), border.mean(close * log_length[k] / 8.0)]
        del Hd, close
    log.info("profile (%.0f s)", time.perf_counter() - t0)
    F = np.stack(cols, 1).astype(np.float32)
    assert F.shape[1] == N_FEATURES
    return F
