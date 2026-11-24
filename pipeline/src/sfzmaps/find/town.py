"""Made-up towns, every size drawn from measurements of real sheets, and the design's made-up case: whether the
zoning-border answer learns from cut statements alone which ink is the zoning border.

The measures (`measure`, written by `sfzmaps find measure-town` to inputs/find/town-measures.json, read by
`measures`): 101 quantiles (px) of each of SIZES, from built label files (`lotlabels.sheet_labels`), used cuts only
(weight > 0):
  street_width         the between-lots run of each cut across a street (its px of prior 1 hold street-labelled
                       px): those px run from one lot's px centre to the other's (the street's px and one of each
                       lot's)
  block_side           the runs of block px (street-or-not label 0) along rows and columns with street (label 1) at
                       both ends
  block_line_width     the ink run along each block-edge cut under its best px
  zoning_border_width  the same for each "a zoning border crosses" cut
A zoning border printed beside a block line is measured together with it, so that width is the pair's.
A cut's best px is its ink px of highest prior, and of those the one in the longest ink run (across a street every
px between the lots has prior 1, and the zoning border, where the sheet prints one there, is the heavier line).
`spread_px` is the mean of those sheets' placement spreads (`cuts.spread_px`, the files' `sigma`).

A town (`draw`): axis-aligned blocks of two to four lots side by side, streets between, every block outlined, lot
lines not printed.  Zoning borders (KINDS): down a street's middle, beside a block line (in the street, against the
outline), or along a lot line inside a block.  Some streets between block rows carry one the town's whole width,
cutting the rows into bands; more cross each band from edge to edge, along a lot line only in a band one block row
deep (where it meets zoning borders at both ends).  Each is the kind placed least so far in the town among those with
a free place, ties at random: equal shares as far as the blocks allow.  The districts (`codes`) are the parts the
zoning borders cut the town into.  The print is offset from the lots by a placement error drawn from a normal of the
measured spread, in whole px as a 1-bit scan prints.  The block lines' and the zoning borders' widths are one draw:
the same quantile of both lists.  A zoning border is printed only where its print touches no cut
its lots say none crosses (the labels' rule: never a label that might not match the print; a zoning border down a
narrow street can touch the lots beside it), and lots no printed zoning border divides are one district.  The town
lies on the solver grid's top-left corner, one page px a grid px (`PLACEMENT`).

The made-up case (`fit`, `chances`): the network learns from the towns' block-edge and zoning-border cuts alone.

Reads built label files (data/find/labels/<series>/<sheet>.npz) and their scans.  Writes nothing itself."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sfzmaps import config
from sfzmaps.find import cuts, drawn, losses, lotlabels, model, pages
from sfzmaps.solver import grid
from sfzmaps.solver.hatchnet.data import NOT_STREET, STREET

SIZES = ("street_width", "block_side", "block_line_width", "zoning_border_width")
KINDS = ("middle", "beside", "lot line")      # down a street's middle, beside a block line, along a lot line
QUANTILES = 101
MEASURES = "town-measures.json"
SHAPE = (192, 192)          # test setting: a town's side, px
LOTS = (2, 4)               # a block's lots, fewest and most
LR = 1e-2                   # engineering setting: Adam's step size for the made-up case: of 1e-3, 3e-3, 1e-2, the
                            # smallest whose training loss stops falling within its 400 steps (1e-3 was still falling
                            # fast: 3.6 to 0.7 a statement)


def measures_path() -> Path:
    return config.INPUTS_DIR / "find" / MEASURES


def measures(path: Path | None = None) -> dict:
    """The measures file (`measures_path`): SIZES as lists of QUANTILES px, and `spread_px`."""
    found = json.loads((path or measures_path()).read_text())
    missing = [k for k in (*SIZES, "spread_px") if k not in found]
    if missing:
        raise ValueError(f"{path or measures_path()}: no {', '.join(missing)}")
    return found


# --- measuring real sheets --------------------------------------------------------------------------------------------

def _ink_runs(cut: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Per cut px: the length of the run of ink px along its cut that holds it (0 off the ink).  A cut's px are
    consecutive and touching (`cuts`)."""
    new = np.r_[True, (cut[1:] != cut[:-1]) | (ink[1:] != ink[:-1])]
    run = np.cumsum(new) - 1
    return np.where(ink, np.bincount(run)[run], 0)


def _best(cut: np.ndarray, prior: np.ndarray, run: np.ndarray, n: int) -> np.ndarray:
    """Per cut: the index of its best px (see the module), -1 for a cut with no ink."""
    idx = np.flatnonzero(run > 0)
    order = idx[np.lexsort((-run[idx], -prior[idx], cut[idx]))]
    first = np.r_[True, cut[order][1:] != cut[order][:-1]] if len(order) else np.zeros(0, bool)
    best = np.full(n, -1, np.int64)
    best[cut[order][first]] = order[first]
    return best


def _block_runs(street: np.ndarray, weight: np.ndarray) -> np.ndarray:
    block, road = (street == NOT_STREET) & (weight > 0), (street == STREET) & (weight > 0)
    out = []
    for b, r in ((block, road), (block.T, road.T)):
        d = np.diff(np.pad(b, ((0, 0), (1, 1))).astype(np.int8), axis=1)
        rows, start = np.nonzero(d == 1)
        _, end = np.nonzero(d == -1)                    # row-major: each row's runs start before they end
        ok = (start > 0) & (end < b.shape[1])
        rows, start, end = rows[ok], start[ok], end[ok]
        ok = r[rows, start - 1] & r[rows, end]
        out.append((end - start)[ok])
    return np.concatenate(out)


def sizes(z, lum: np.ndarray) -> dict[str, np.ndarray]:
    """SIZES (see the module) of one sheet: `z` its label file's arrays (street, weight, cuts_*), `lum` its scan."""
    px, cut, prior = z["cuts_px"], z["cuts_cut"], z["cuts_prior"]
    answer, says, used = z["cuts_answer"], z["cuts_says"], z["cuts_weight"] > 0
    n = len(answer)
    between = np.bincount(cut, prior == 1, minlength=n)
    border = (answer == cuts.BORDER) & used
    run = _ink_runs(cut, lum.ravel()[px] < drawn.INK_BELOW)
    best = _best(cut, prior, run, n)

    def width(which):
        b = best[which]
        return run[b[b >= 0]]

    on_street = np.bincount(cut, (prior == 1) & (z["street"].ravel()[px] == STREET), minlength=n) > 0
    return {"street_width": between[border & on_street],
            "block_side": _block_runs(z["street"], z["weight"]),
            "block_line_width": width((answer == cuts.EDGE) & used),
            "zoning_border_width": width(border & says)}


def measure(series: str, sheets: list[str], which: str = "warp") -> dict:
    """The measures (see the module) from the series' sheets' built label files, made through placement `which`
    (`lotlabels.load_sheet_labels`, whose errors name the command that makes a file): SIZES as QUANTILES px each,
    `spread_px`, and `counts` (how many values each list was made from)."""
    found = {k: [] for k in SIZES}
    sigma = []
    for h in sheets:
        lum = pages.load_scan((series, h))
        z = lotlabels.load_sheet_labels((series, h), which, lum.shape)
        got = sizes(z, lum)
        sigma.append(float(z["sigma"]))
        for k in SIZES:
            found[k].append(got[k])
    out = {}
    for k in SIZES:
        v = np.concatenate(found[k])
        out[k] = [float(q) for q in np.quantile(v, np.linspace(0, 1, QUANTILES))]
    out["spread_px"] = float(np.mean(sigma))
    out["counts"] = {k: int(sum(len(a) for a in found[k])) for k in SIZES}
    return out


# --- drawing a town ---------------------------------------------------------------------------------------------------

OFFSET = -0.25              # the town's placement puts grid px c's centre at page x c + 0.5 + OFFSET: no cut px falls
                            # on a half px, where rounding would put two of a cut's px on one (`cuts._lay`)


class _Corner:
    """The town's placement: page px (x, y) (px centres at integers) at (X0 + 4 (x - OFFSET), Y1 - 4 (y - OFFSET))
    ft, on the solver grid's top-left corner, so grid px (col, row) is page px (col, row) (as
    `find_fixtures.Flat(4.0, OFFSET, OFFSET)`)."""

    correction = None

    def to_world(self, px) -> np.ndarray:
        px = np.asarray(px, float).reshape(-1, 2) - OFFSET
        return np.c_[grid.X0 + px[:, 0] * grid.FT_PER_PX, grid.Y1 - px[:, 1] * grid.FT_PER_PX]

    def to_pixel(self, world) -> np.ndarray:
        w = np.asarray(world, float).reshape(-1, 2)
        return np.c_[(w[:, 0] - grid.X0) / grid.FT_PER_PX, (grid.Y1 - w[:, 1]) / grid.FT_PER_PX] + OFFSET

    def grid_to_pixel(self, world, page_shape) -> np.ndarray:
        return self.to_pixel(world)


PLACEMENT = _Corner()


@dataclass(frozen=True)
class Town:
    lum: np.ndarray             # float32 (h, w): the print, 1 paper, 0 ink
    lot: np.ndarray             # int32 (h, w): the lots on the grid under PLACEMENT, -1 none
    codes: np.ndarray           # str (lots,): each lot's zoning district
    block_line: np.ndarray      # bool (h, w): where the block outlines are printed
    border_line: np.ndarray     # bool (h, w): where the zoning borders are printed
    shift: tuple[int, int]      # (dx, dy): the print's offset from the lots, px
    kinds: tuple[str, ...]      # each printed zoning border's kind (KINDS)
    widths: tuple[int, int]     # the block lines' and the zoning borders' width, px


def _at(q, u: float) -> int:
    """The size at level u (0 to 1) of its quantiles (px, whole, at least 1)."""
    return max(1, int(round(float(np.interp(u * (len(q) - 1), np.arange(len(q)), q)))))


def _size(rng: np.random.Generator, q) -> int:
    """A size drawn from its quantiles."""
    return _at(q, rng.random())


def _spans(rng: np.random.Generator, m: dict, n: int) -> list[tuple[int, int]]:
    """Blocks (start, end) along one axis of n px, streets between, from a block cut by the town's edge."""
    side = _size(rng, m["block_side"])
    x, out = -int(rng.integers(0, side)), []
    while x < n:
        out.append((x, x + side))
        x += side + max(1, _size(rng, m["street_width"]) - 2)       # its px between the lots' (see the module)
        side = _size(rng, m["block_side"])
    return out


def _line(g: float, width: int) -> slice:
    """The px of a line `width` px wide centred on grid coordinate g (grid px c runs from c to c + 1; page
    coordinate g + OFFSET)."""
    lo = int(np.floor(g + OFFSET - (width - 1) / 2 + 0.5))
    return slice(lo, lo + width)


def _across(lo: float, hi: float, where: str, line: int, heavy: int) -> slice:
    """The px across a street, between block sides at grid coordinates lo and hi, of a zoning border `heavy` px wide
    down its middle or beside the block line (`line` px wide) at lo ("low") or at hi ("high")."""
    if where == "middle":
        return _line((lo + hi) / 2, heavy)
    if where == "low":
        start = _line(lo, line).stop
        return slice(start, start + heavy)
    stop = _line(hi, line).start
    return slice(stop - heavy, stop)


class _Canvas:
    """Masks a little larger than the page, so a print offset by `shift` still covers the whole page."""

    def __init__(self, h: int, w: int, shift: tuple[int, int]):
        self.h, self.w, self.shift = h, w, shift
        self.pad = max(abs(shift[0]), abs(shift[1])) + 1

    def new(self) -> np.ndarray:
        return np.zeros((self.h + 2 * self.pad, self.w + 2 * self.pad), bool)

    def box(self, a: np.ndarray, rows: slice, cols: slice) -> None:
        """Ink page rows x cols (clipped to the canvas)."""
        p = self.pad
        r0, r1 = max(0, rows.start + p), max(0, rows.stop + p)
        c0, c1 = max(0, cols.start + p), max(0, cols.stop + p)
        a[r0:r1, c0:c1] = True

    def page(self, a: np.ndarray, printed: bool) -> np.ndarray:
        """The page's part of a mask: as drawn, or (printed) where the print puts it."""
        p = self.pad
        dx, dy = self.shift if printed else (0, 0)
        return a[p - dy:p - dy + self.h, p - dx:p - dx + self.w]


def _outlines(cv: _Canvas, cols, rows, width: int) -> np.ndarray:
    a = cv.new()
    for y0, y1 in rows:
        for x0, x1 in cols:
            top, bottom, left, right = _line(y0, width), _line(y1, width), _line(x0, width), _line(x1, width)
            across, down = slice(left.start, right.stop), slice(top.start, bottom.stop)
            for rr, cc in ((top, across), (bottom, across), (down, left), (down, right)):
                cv.box(a, rr, cc)
    return a


def _bands(rng: np.random.Generator, n_rows: int) -> list[list[int]]:
    """Block rows grouped by the streets between them that carry a zoning border (each street between two block rows
    does with chance one half)."""
    out = [[0]]
    for j in range(1, n_rows):
        if rng.random() < 0.5:
            out.append([j])
        else:
            out[-1].append(j)
    return out


def ground(t: Town) -> lotlabels.Ground:
    """The town's street-or-not labels on its grid: street off the lots, block on them, every px labelled."""
    block = t.lot >= 0
    return lotlabels.Ground(t.lot, block, np.where(block, NOT_STREET, STREET).astype(np.uint8),
                            np.ones(block.shape, np.float16))


def border_cuts(lot: np.ndarray, codes: np.ndarray, sigma: float) -> cuts.Cuts:
    return cuts.border_cuts(lot, codes, PLACEMENT, lot.shape, np.ones(lot.shape, bool), sigma)


def draw(rng: np.random.Generator, m: dict, shape: tuple[int, int] = SHAPE) -> Town:
    """A made-up town of `shape` px (see the module), every size drawn from the measures `m`.  Lots run side by side
    across the blocks' width; the town is laid on its side in half the draws."""
    flip = bool(rng.random() < 0.5)
    h, w = (shape[1], shape[0]) if flip else shape
    u = rng.random()                    # one draw for both widths: the zoning border is the heavier line, as measured
    line, heavy = _at(m["block_line_width"], u), _at(m["zoning_border_width"], u)
    sigma = float(m["spread_px"])
    shift = (int(np.rint(rng.normal(0, sigma))), int(np.rint(rng.normal(0, sigma))))
    cols, rows = _spans(rng, m, w), _spans(rng, m, h)
    cv = _Canvas(h, w, shift)

    lot = np.full((h, w), -1, np.int32)
    centre, row_of, lot_lines = [], [], {}
    for j, (y0, y1) in enumerate(rows):
        for i, (x0, x1) in enumerate(cols):
            k = min(int(rng.integers(LOTS[0], LOTS[1] + 1)), x1 - x0)
            xs = [x0 + round(q * (x1 - x0) / k) for q in range(k + 1)]
            lot_lines[i, j] = xs[1:-1]
            for a, b in zip(xs, xs[1:]):
                ra, rb, ca, cb = max(y0, 0), min(y1, h), max(a, 0), min(b, w)
                if ra < rb and ca < cb:
                    lot[ra:rb, ca:cb] = len(centre)
                    centre.append((a + b) / 2)
                    row_of.append(j)
    centre, row_of = np.array(centre), np.array(row_of)

    # zoning borders: down the streets between bands of block rows, then across each band; each the kind placed least
    # so far among those with a free place, ties at random (equal shares as far as the blocks allow)
    bands = _bands(rng, len(rows))
    band_of = np.zeros(len(rows), int)
    for b, rs in enumerate(bands):
        band_of[rs] = b
    placed = dict.fromkeys(KINDS, 0)

    def pick(free: dict):
        open_ = [k for k in KINDS if free[k]]
        if not open_:
            return None
        least = min(placed[k] for k in open_)
        tied = [k for k in open_ if placed[k] == least]
        kind = tied[int(rng.integers(len(tied)))]
        placed[kind] += 1
        return kind, free[kind][int(rng.integers(len(free[kind])))]

    full = slice(-cv.pad, w + cv.pad)
    pieces, splits = [], [[] for _ in bands]                 # (kind, mask); each band's split x
    tall = slice(-cv.pad, h + cv.pad)
    between = [tall]                                         # the px rows of the zoning borders between the bands
    for b in range(1, len(bands)):
        lo, hi = rows[bands[b][0] - 1][1], rows[bands[b][0]][0]
        kind, where = pick({"middle": ["middle"], "beside": ["low", "high"], "lot line": []})
        across = _across(lo, hi, where, line, heavy)
        a = cv.new()
        cv.box(a, across, full)
        pieces.append((kind, a))
        between.append(across)
    between.append(tall)
    for b, rs in enumerate(bands):
        down = slice(between[b].start, between[b + 1].stop)       # from the zoning border above to the one below
        free = {"middle": [("middle", i) for i in range(len(cols) - 1)],
                "beside": [(side, i) for i in range(len(cols) - 1) for side in ("low", "high")],
                "lot line": [("lot", x) for i in range(len(cols)) for x in lot_lines[i, rs[0]]] if len(rs) == 1 else []}
        for _ in range(int(rng.integers(0, len(cols)))):
            got = pick(free)
            if got is None:
                break
            kind, (where, at) = got
            if where == "lot":
                free["lot line"].remove((where, at))
                x, across = at, _line(at, heavy)
            else:                                       # one zoning border to a street
                free["middle"] = [p for p in free["middle"] if p[1] != at]
                free["beside"] = [p for p in free["beside"] if p[1] != at]
                lo, hi = cols[at][1], cols[at + 1][0]
                x, across = (lo + hi) / 2, _across(lo, hi, where, line, heavy)
            a = cv.new()
            cv.box(a, down, across)
            pieces.append((kind, a))
            splits[b].append(x)

    # the districts: each band cut at its splits; then the borders whose print would contradict the lots go
    cell = np.array([band_of[r] * (len(cols) + 1) + int(np.sum(np.array(splits[band_of[r]]) < c))
                     for r, c in zip(row_of, centre)], np.int64)
    parent = np.arange(int(cell.max(initial=0)) + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    alive = [True] * len(pieces)
    keep = np.ones((h, w), bool)
    r0, r1, c0, c1 = cuts.window(PLACEMENT, keep, lot.shape)
    win = lot[r0:r1, c0:c1].ravel()
    while True:
        codes = np.array([f"D{root(i)}" for i in cell], dtype=str)
        c = border_cuts(lot, codes, sigma)
        drawn_ink, none_px = np.zeros((h, w), bool), c.on((h, w), ~c.says)
        drop = []
        for k, (_, a) in enumerate(pieces):
            if alive[k]:
                drawn_ink |= cv.page(a, False)
                if (cv.page(a, True) & none_px).any():
                    drop.append(k)
        gap = c.says & ~(np.bincount(c.cut, drawn_ink.ravel()[c.px], minlength=len(c.answer)) > 0)
        if not drop and not gap.any():
            break
        for k in drop:
            alive[k] = False
        for p, q in win[c.ends[gap]]:                  # lots no zoning border divides any more: one district
            parent[root(cell[p])] = root(cell[q])

    block_line = cv.page(_outlines(cv, cols, rows, line), True).copy()
    border_line = np.zeros((h, w), bool)
    for k, (_, a) in enumerate(pieces):
        if alive[k]:
            border_line |= cv.page(a, True)
    kinds = tuple(kind for k, (kind, _) in enumerate(pieces) if alive[k])
    lum = np.where(block_line | border_line, 0, 1).astype(np.float32)
    if flip:
        lum, lot, block_line, border_line = lum.T.copy(), lot.T.copy(), block_line.T.copy(), border_line.T.copy()
        shift = shift[::-1]
    return Town(lum, lot, codes, block_line, border_line, shift, kinds, (line, heavy))


def labels(t: Town, sigma: float) -> dict:
    """The town's labels as a sheet's label file holds them (`lotlabels.sheet_labels`): street, weight and cuts_*,
    the block-edge cuts then the zoning-border cuts, every one used."""
    g = ground(t)
    keep = np.ones(t.lot.shape, bool)
    c = cuts.Cuts.joined(cuts.edge_cuts(g, PLACEMENT, t.lot.shape, keep, sigma), border_cuts(t.lot, t.codes, sigma))
    return {"street": g.street, "weight": g.weight.astype(np.float32), "cuts_px": c.px, "cuts_cut": c.cut,
            "cuts_prior": c.prior, "cuts_answer": c.answer, "cuts_says": c.says, "cuts_weight": c.weight}


# --- the made-up case -------------------------------------------------------------------------------------------------

def _tile_cuts(lab: dict, shape, y0: int, x0: int, tile: int, offset: int):
    """The cuts wholly inside a tile, their px as indexes into the batch's flat logits (`offset` that tile's first
    channel's first px), each in its answer's channel."""
    rows, cols = np.divmod(lab["cuts_px"], shape[1])
    n = len(lab["cuts_answer"])
    inside = (rows >= y0) & (rows < y0 + tile) & (cols >= x0) & (cols < x0 + tile)
    whole = np.bincount(lab["cuts_cut"], ~inside, minlength=n) == 0
    on = whole[lab["cuts_cut"]]
    renumber = np.cumsum(whole) - 1
    chan = np.where(lab["cuts_answer"] == cuts.EDGE, model.EDGE, model.BORDER)[lab["cuts_cut"][on]]
    px = offset + chan * tile * tile + (rows[on] - y0) * tile + (cols[on] - x0)
    return px, renumber[lab["cuts_cut"][on]], lab["cuts_prior"][on], lab["cuts_says"][whole], \
        lab["cuts_weight"][whole]


def fit(towns: list[Town], sigma: float, *, width: int, tile: int, batch: int, steps: int,
        seed: int) -> model.FindNet:
    """The network trained on the towns' block-edge and zoning-border cuts alone (`labels`): each step a batch of
    tiles drawn from the towns, the loss the cuts' likelihood (`losses.cut_nll`) over their count (`statements`).
    CPU; the caller sets deterministic algorithms."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    labs = [labels(t, sigma) for t in towns]
    net = model.FindNet(width)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    net.train()
    for _ in range(steps):
        xs, parts, n = [], [], 0
        for b in range(batch):
            k = int(rng.integers(len(towns)))
            h, w = towns[k].lum.shape
            y0, x0 = int(rng.integers(0, h - tile + 1)), int(rng.integers(0, w - tile + 1))
            xs.append(towns[k].lum[y0:y0 + tile, x0:x0 + tile])
            px, cut, prior, says, weight = _tile_cuts(labs[k], (h, w), y0, x0, tile, b * len(model.ANSWERS) * tile ** 2)
            parts.append((px, cut + n, prior, says, weight))
            n += len(says)
        px, cut, prior, says, weight = (torch.from_numpy(np.concatenate(p)) for p in zip(*parts))
        logit = net(torch.from_numpy(np.stack(xs))[:, None]).reshape(-1)
        loss = losses.cut_nll(logit, px, cut.long(), prior, says, weight) / max(1.0, losses.statements(weight))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


@torch.no_grad()
def chances(net: model.FindNet, lum: np.ndarray) -> np.ndarray:
    """float32 (3, h, w): each answer's chance at every px of a print (h, w multiples of model.MULTIPLE)."""
    net.eval()
    return torch.sigmoid(net(torch.from_numpy(np.asarray(lum, np.float32))[None, None]))[0].numpy()
