"""Hand crops: a person's marks on a crop of a scan, the labels made from them, and the ground left out of training
under the test and choosing crops.

A crop file is inputs/find-labels/<set>/<series>/<sheet>__<x0>_<y0>_<w>_<h>.json (the format and the reading of the
maps: inputs/find-labels/README.md).  It is drawn one of two ways:
- seeds and strokes (for maps whose streets are open paper): `make_labels` turns them into per-px labels by nine
  rules: paper by Otsu's threshold, openings as paper (where the map's owner has said a street runs through printed
  ink), strokes and limits as barriers, street as the paper 4-connected to a street seed, areas as the 8-connected
  islands holding a block or open seed, and pockets (street paper the street reaches only through a pin-hole, as a
  hatching cell behind a hole in a thin outline) joined to the one area they touch (`find_pockets`);
- outlines (for maps where heavy lines and letters join the blocks): each typed polygon is an area, the rest street.
  Outline crops are for scoring only (`choose` and `test`): a corner typed by eye is a px or two off the printed line.
This is a drawing aid for hand labels only, checked by eye on its picture (`look.crop`): its 3 px barrier and opening
width, its 4-connected street fill and 8-connected islands, PINHOLE, and the 2 px an outline's corner may lie
outside the box are parts of the tool, and nothing like it runs when a sheet is read.

`reserved` is the ground under every `choose` and `test` crop, on any sheet's page, which training leaves out.

Reads the crop files and (for `reserved`) the sheets' placements.  Writes nothing."""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.draw import line as draw_line
from skimage.filters import threshold_otsu

from sfzmaps import config
from sfzmaps.find import lotlabels
from sfzmaps.find.pages import SheetId
from sfzmaps.solver import hatching

SETS = ("train", "choose", "test")
HELD_OUT = ("choose", "test")       # the sets whose ground training leaves out
SEED_MARKS = ("street", "blocks", "open", "strokes", "limits", "openings")
OUTLINE_MARKS = ("polygons", "open_polygons")
KEYS = {"box", "notes", "drawn_by", *SEED_MARKS, *OUTLINE_MARKS}
BARRIER = 3                         # tool setting: a stroke's, limit's or opening's width, px
OUTSIDE = 2                         # tool setting: how far an outline's corner may lie outside the box, px (a block
                                    # cut by the crop's side is typed with corners on or just past the box's edge)
LARGEST = 5                         # tool setting: how many of the largest unseeded islands are always warned of
PINHOLE = 3                         # tool setting: street paper reached only through a neck narrower than this, px,
                                    # is a pocket
RING_STEP = 8                       # engineering setting: px between the points of a crop's box sent to another page
log = logging.getLogger(__name__)
STREET, AREA, NONE = 1, 0, 255      # the street label; NONE also marks edge px with no label
FOUR = ndimage.generate_binary_structure(2, 1)
EIGHT = ndimage.generate_binary_structure(2, 2)

Point = tuple[int, int]


def root_dir() -> Path:
    """inputs/find-labels, read when called (tests point config.INPUTS_DIR elsewhere)."""
    return config.INPUTS_DIR / "find-labels"


@dataclass(frozen=True)
class Crop:
    """One crop file: its set, sheet, box (x0, y0, w, h) and marks, all in page px as stored."""
    set: str
    series: str
    sheet: str
    box: tuple[int, int, int, int]
    street: tuple[Point, ...]
    blocks: tuple[Point, ...]
    open: tuple[Point, ...]
    strokes: tuple[tuple[Point, ...], ...]
    limits: tuple[tuple[Point, ...], ...]
    path: Path
    polygons: tuple[tuple[Point, ...], ...] = ()
    open_polygons: tuple[tuple[Point, ...], ...] = ()
    openings: tuple[tuple[Point, ...], ...] = ()

    @property
    def outlined(self) -> bool:
        """True for an outline crop (polygons), False for a seed crop."""
        return bool(self.polygons or self.open_polygons)


def _points(path: Path, key: str, value) -> tuple[Point, ...]:
    try:
        return tuple((int(round(x)), int(round(y))) for x, y in value)
    except (TypeError, ValueError):
        raise ValueError(f"{path}: `{key}` must be a list of [x, y]") from None


def read_crop(path: Path, set_: str) -> Crop:
    """The crop file at `path` (in `set_`).  ValueError for a name not `<sheet>__<x0>_<y0>_<w>_<h>.json`, a name and
    `box` that disagree, an unknown key, a mark that is not [x, y] points, a polygon of fewer than 3 corners, marks of
    both kinds (seeds and strokes, outlines) or of neither, an outline crop in the `train` set, a file that is not
    JSON, and a crop on a duplicate scan (never read)."""
    path = Path(path)
    sheet, _, tail = path.stem.rpartition("__")
    try:
        named = tuple(int(v) for v in tail.split("_"))
    except ValueError:
        named = ()
    if not sheet or len(named) != 4:
        raise ValueError(f"{path}: expected a name <sheet>__<x0>_<y0>_<w>_<h>.json")
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not JSON: {e}") from None
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a JSON object")
    unknown = sorted(set(d) - KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {', '.join(unknown)} (expected {', '.join(sorted(KEYS))})")
    try:
        box = tuple(int(v) for v in d.get("box", ()))
    except (TypeError, ValueError):
        raise ValueError(f"{path}: `box` must be [x0, y0, w, h], whole numbers; it is {d.get('box')!r}") from None
    if box != named:
        raise ValueError(f"{path}: its name says box {list(named)}, the file {list(box)}")
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError(f"{path}: the box's width and height must be positive")
    try:
        lines = {k: tuple(_points(path, k, s) for s in d.get(k, [])) for k in ("strokes", "limits", "openings",
                                                                       *OUTLINE_MARKS)}
    except TypeError:
        raise ValueError(f"{path}: strokes, limits, openings and outlines must be lists of [x, y] lists") from None
    seeded, outlined = any(d.get(k) for k in SEED_MARKS), any(d.get(k) for k in OUTLINE_MARKS)
    if seeded == outlined:
        raise ValueError(f"{path}: {'marks of both kinds' if seeded else 'no marks'}; a crop is drawn either with "
                         f"seeds and strokes ({', '.join(SEED_MARKS)}) or with outlines ({', '.join(OUTLINE_MARKS)})")
    if outlined and set_ == "train":
        raise ValueError(f"{path}: outline crops may only be in choose or test (they are for scoring only)")
    for k in OUTLINE_MARKS:
        for i, poly in enumerate(lines[k]):
            if len(poly) < 3:
                raise ValueError(f"{path}: {k}[{i}] has {len(poly)} corners; an outline needs at least 3")
    dup = hatching.duplicate_scans(path.parent.name)
    if sheet in dup:
        raise ValueError(f"{path}: {sheet} is a duplicate scan of {dup[sheet]}, which is never read; draw the crop on "
                         f"that scan")
    return Crop(set_, path.parent.name, sheet, box, _points(path, "street", d.get("street", [])),
                _points(path, "blocks", d.get("blocks", [])), _points(path, "open", d.get("open", [])),
                lines["strokes"], lines["limits"], path, lines["polygons"], lines["open_polygons"],
                lines["openings"])


def crop_files(sets=SETS, root: Path | None = None) -> list[tuple[Path, str]]:
    """(path, set) of every crop file of `sets` under `root` (None: inputs/find-labels), sorted by path, unread.
    ValueError for an unknown set."""
    root = root_dir() if root is None else Path(root)
    bad = sorted(set(sets) - set(SETS))
    if bad:
        raise ValueError(f"unknown crop sets {', '.join(bad)} (expected {', '.join(SETS)})")
    return sorted((f, s) for s in sets for f in (root / s).glob("*/*.json"))


def crops(sets=SETS, root: Path | None = None) -> list[Crop]:
    """Every crop of `sets` under `root` (None: inputs/find-labels), sorted by path.  ValueError (naming the file) for
    the first bad file (`read_crop`)."""
    return [read_crop(f, s) for f, s in crop_files(sets, root)]


def crop_scan(lum: np.ndarray, crop: Crop) -> np.ndarray:
    """The crop's box of the page `lum`.  ValueError if the box runs off the page."""
    x0, y0, w, h = crop.box
    if x0 < 0 or y0 < 0 or x0 + w > lum.shape[1] or y0 + h > lum.shape[0]:
        raise ValueError(f"{crop.path}: box {list(crop.box)} runs off the page ({lum.shape[1]} x {lum.shape[0]} px)")
    return lum[y0:y0 + h, x0:x0 + w]


@dataclass(frozen=True)
class CropLabels:
    """A crop's labels: `street` (1 street, 0 area, 255 a stroke or limit), `area` (0 none, else the area's number),
    `kinds` (area number: "block" or "open"), `edge` (1 edge, 0 not, 255 no label) and `warnings` (page px)."""
    street: np.ndarray          # uint8 (h, w)
    area: np.ndarray            # int32 (h, w)
    kinds: dict[int, str]
    edge: np.ndarray            # uint8 (h, w)
    warnings: list[str]


def _barrier(lines, x0: int, y0: int, shape: tuple[int, int]) -> np.ndarray:
    """The polylines `lines` (page px) drawn BARRIER px wide on a crop of `shape` whose corner is page (x0, y0)."""
    out = np.zeros(shape, bool)
    for pts in lines:
        pts = [(x - x0, y - y0) for x, y in pts]
        for (xa, ya), (xb, yb) in zip(pts, pts[1:] if len(pts) > 1 else pts):
            rr, cc = draw_line(ya, xa, yb, xb)
            ok = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
            out[rr[ok], cc[ok]] = True
    return ndimage.binary_dilation(out, np.ones((BARRIER, BARRIER), bool)) if out.any() else out


def _next_to(mask: np.ndarray) -> np.ndarray:
    """Px 8-adjacent to `mask` (or on it)."""
    return ndimage.binary_dilation(mask, EIGHT) if mask.any() else mask


def _edges(area: np.ndarray) -> np.ndarray:
    """1 on area px (area > 0) 8-adjacent to a px not of the same area (px off the crop do not count), 0 elsewhere."""
    h, w = area.shape
    padded = np.pad(area, 1, mode="edge")
    other = np.zeros(area.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                other |= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] != area
    return ((area > 0) & other).astype(np.uint8)


def _no_edge_on_the_crop_side(edge: np.ndarray) -> None:
    edge[[0, -1], :] = NONE
    edge[:, [0, -1]] = NONE


def make_labels(lum_crop: np.ndarray, crop: Crop, smooth: bool = True) -> CropLabels:
    """The crop's labels from its marks (the module's rules, as inputs/find-labels/README.md states them): an outline
    crop's by `outline_labels`, a seed crop's by `seed_labels` (`smooth`: with its pockets joined).  ValueError naming the
    file if `lum_crop` is not the box's size."""
    lum = np.asarray(lum_crop, np.float32)
    w, h = crop.box[2:]
    if lum.shape != (h, w):
        raise ValueError(f"{crop.path}: the crop is {lum.shape}, its box {w} x {h}")
    return outline_labels(crop) if crop.outlined else seed_labels(lum, crop, smooth)


def outline_labels(crop: Crop) -> CropLabels:
    """An outline crop's labels: each polygon an area, numbered in file order (`polygons`, then `open_polygons`),
    filled on the crop (px centres at integers) and clipped to it; street 1 outside every polygon, 0 inside; edge 1 on
    area px 8-adjacent to anything not the same area, 255 on the crop's outermost rows and columns.  ValueError naming
    the file for a corner more than OUTSIDE px outside the box and for two polygons sharing a px (naming both)."""
    x0, y0, w, h = crop.box
    area = np.zeros((h, w), np.int32)
    kinds: dict[int, str] = {}
    polys = [(f"polygons[{i}]", "block", p) for i, p in enumerate(crop.polygons)] + \
            [(f"open_polygons[{i}]", "open", p) for i, p in enumerate(crop.open_polygons)]
    for name, kind, poly in polys:
        pts = np.array(poly, np.int64) - (x0, y0)
        off = (pts < -OUTSIDE).any(axis=1) | (pts[:, 0] > w - 1 + OUTSIDE) | (pts[:, 1] > h - 1 + OUTSIDE)
        if off.any():
            x, y = poly[int(np.argmax(off))]
            raise ValueError(f"{crop.path}: {name} has a corner ({x}, {y}) more than {OUTSIDE} px outside the box")
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        mine = mask.astype(bool)
        if (mine & (area > 0)).any():
            first = polys[int(area[mine & (area > 0)][0]) - 1][0]
            raise ValueError(f"{crop.path}: {first} and {name} overlap; outlines may touch no common px")
        kinds[len(kinds) + 1] = kind
        area[mine] = len(kinds)
    edge = _edges(area)
    _no_edge_on_the_crop_side(edge)
    return CropLabels(np.where(area > 0, AREA, STREET).astype(np.uint8), area, kinds, edge, [])


def _deepest(islands: np.ndarray, i: int, depth: np.ndarray) -> tuple[int, int]:
    """The crop px (x, y) of island `i` farthest from its outside."""
    yy, xx = np.unravel_index(int(np.argmax(np.where(islands == i, depth, -1))), islands.shape)
    return int(xx), int(yy)


@dataclass(frozen=True)
class Pocket:
    """Street paper the street reaches only through a neck narrower than PINHOLE: its number in `find_pockets`'s
    labels, px count, the crop px (x, y) of a px well inside it, what it touches ("street seed", "one area", "no area"
    or "two or more areas") and the area it joins (0: none)."""
    number: int
    size: int
    at: tuple[int, int]
    case: str
    joins: int


def find_pockets(flooded: np.ndarray, area: np.ndarray,
                 street_seeds: list[tuple[int, int]]) -> tuple[np.ndarray, list[Pocket]]:
    """The pockets of the street's flooded paper `flooded` (int32 labels, 0 none, and one `Pocket` each).  The part the
    street reaches through passages at least PINHOLE px wide: the px whose PINHOLE x PINHOLE neighbourhood is all
    flooded (outside the crop counts as flooded: streets run off it), flooded 4-connected from those within one px of
    a street seed (crop px (x, y)), grown by one PINHOLE x PINHOLE step inside `flooded`.  The rest, in 8-connected
    pieces, are pockets.  A pocket joins the area it touches when it touches exactly one and holds no street seed."""
    square = np.ones((PINHOLE, PINHOLE), bool)
    h, w = flooded.shape
    wide, _ = ndimage.label(ndimage.binary_erosion(flooded, square, border_value=1), FOUR)
    near = {int(v) for x, y in street_seeds for v in wide[max(0, y - 1):y + 2, max(0, x - 1):x + 2].ravel() if v}
    reached = ndimage.binary_dilation(np.isin(wide, sorted(near)), square) & flooded
    pockets, n = ndimage.label(flooded & ~reached, EIGHT)
    if not n:
        return pockets, []
    ys, xs = np.nonzero(pockets)
    number = pockets[ys, xs]
    padded = np.pad(area, 1)
    pairs = [np.zeros((0, 2), np.int64)]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                a = padded[ys + 1 + dy, xs + 1 + dx]
                pairs.append(np.c_[number[a > 0], a[a > 0]])
    pairs = np.unique(np.vstack(pairs).astype(np.int64), axis=0)
    touching = np.bincount(pairs[:, 0], minlength=n + 1)
    first = np.zeros(n + 1, np.int64)
    first[pairs[::-1, 0]] = pairs[::-1, 1]
    seeded = {int(pockets[y, x]) for x, y in street_seeds}
    sizes = np.bincount(pockets.ravel(), minlength=n + 1)
    depth = ndimage.distance_transform_edt(np.pad(pockets > 0, 1))[1:-1, 1:-1]
    where = ndimage.maximum_position(depth, pockets, np.arange(1, n + 1))
    out = []
    for i in range(1, n + 1):
        case = "street seed" if i in seeded else {0: "no area", 1: "one area"}.get(int(touching[i]),
                                                                                  "two or more areas")
        y, x = where[i - 1]
        out.append(Pocket(i, int(sizes[i]), (int(x), int(y)), case, int(first[i]) if case == "one area" else 0))
    return pockets, out


def seed_labels(lum: np.ndarray, crop: Crop, smooth: bool = True) -> CropLabels:
    """A seed crop's labels by the nine rules (`smooth` False: without rule 6, the pockets).  ValueError naming the
    file and the seed for a seed off the crop, on a stroke or limit, a street seed not on paper, a block or open seed
    the street reaches, and two seeds in one island.  Warnings: each unseeded island larger than the smallest seeded
    one, then the LARGEST largest unseeded islands whatever was seeded, then (`smooth`) the LARGEST largest pockets that
    joined an area (page px of a px well inside each)."""
    x0, y0, w, h = crop.box
    stroke = _barrier(crop.strokes, x0, y0, lum.shape)
    limit = _barrier(crop.limits, x0, y0, lum.shape)
    barrier = stroke | limit
    paper = ((lum > threshold_otsu(lum)) | _barrier(crop.openings, x0, y0, lum.shape)) & ~barrier
    seeds = [("street", p) for p in crop.street] + [("block", p) for p in crop.blocks] + \
            [("open", p) for p in crop.open]
    for kind, (x, y) in seeds:
        if not (x0 <= x < x0 + w and y0 <= y < y0 + h):
            raise ValueError(f"{crop.path}: {kind} seed ({x}, {y}) lies off the crop")
        if barrier[y - y0, x - x0]:
            raise ValueError(f"{crop.path}: {kind} seed ({x}, {y}) lies on a stroke or limit; move it off")
    for x, y in crop.street:
        if not paper[y - y0, x - x0]:
            raise ValueError(f"{crop.path}: street seed ({x}, {y}) is not on paper; move it onto the street's paper")
    pieces, _ = ndimage.label(paper, FOUR)
    street = np.isin(pieces, sorted({int(pieces[y - y0, x - x0]) for x, y in crop.street}))
    for kind, (x, y) in seeds:
        if kind != "street" and street[y - y0, x - x0]:
            raise ValueError(f"{crop.path}: {kind} seed ({x}, {y}) is reached by the street fill (its paper joins a "
                             f"street seed's); close the gap with a stroke")
    islands, n = ndimage.label(~street & ~barrier, EIGHT)
    area = np.zeros(lum.shape, np.int32)
    kinds: dict[int, str] = {}
    seeded: dict[int, tuple[str, Point]] = {}
    for kind, (x, y) in seeds:
        if kind == "street":
            continue
        i = int(islands[y - y0, x - x0])
        if i in seeded:
            raise ValueError(f"{crop.path}: {seeded[i][0]} seed {seeded[i][1]} and {kind} seed ({x}, {y}) lie in "
                             f"one island; draw a stroke between them")
        seeded[i] = (kind, (x, y))
        kinds[len(kinds) + 1] = kind
        area[islands == i] = len(kinds)
    sizes = np.bincount(islands.ravel(), minlength=n + 1)
    unseeded = sorted((i for i in range(1, n + 1) if i not in seeded), key=lambda i: (-sizes[i], i))
    smallest = min((sizes[i] for i in seeded), default=None)
    warnings = []
    if unseeded:
        depth = ndimage.distance_transform_edt(np.pad(islands > 0, 1))[1:-1, 1:-1]
        for i in sorted(i for i in unseeded if smallest is not None and sizes[i] > smallest):
            x, y = _deepest(islands, i, depth)
            warnings.append(f"possible missed block at {x + x0}, {y + y0}")
        for i in unseeded[:LARGEST]:
            x, y = _deepest(islands, i, depth)
            warnings.append(f"unseeded island of {int(sizes[i])} px at {x + x0}, {y + y0}: a block left unseeded is "
                            f"labelled street")
    if smooth:
        pockets, found = find_pockets(street, area, [(x - x0, y - y0) for x, y in crop.street])
        joined = [p for p in found if p.joins]
        if joined:
            area = area.copy()
            area[pockets > 0] = np.r_[0, [p.joins for p in found]][pockets[pockets > 0]]
        for p in sorted(joined, key=lambda p: (-p.size, p.number))[:LARGEST]:
            warnings.append(f"pocket of {p.size} px at {p.at[0] + x0}, {p.at[1] + y0} joined its block: the street "
                            f"reaches it only through a gap under {PINHOLE} px; if it is a real dead end, give it a "
                            f"street seed")
    label = np.where(barrier, NONE, np.where(area > 0, AREA, STREET)).astype(np.uint8)
    edge = _edges(area)
    edge[stroke] = 1
    edge[_next_to(limit)] = NONE
    _no_edge_on_the_crop_side(edge)
    return CropLabels(label, area, kinds, edge, warnings)


def ring(box: tuple[int, int, int, int], step: int = RING_STEP) -> np.ndarray:
    """(n, 2) px centres round the box (x0, y0, w, h): each side, clockwise from the top-left corner, every `step` px
    from its first corner (the next side starts at its last)."""
    x0, y0, w, h = box
    x1, y1 = x0 + w - 1, y0 + h - 1
    out = []
    for (xa, ya), (xb, yb) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(abs(xb - xa), abs(yb - ya))
        t = np.arange(0, max(n, 1), step, dtype=float)
        out.append(np.c_[xa + np.sign(xb - xa) * t, ya + np.sign(yb - ya) * t])
    return np.vstack(out)


def reserved(sid: SheetId, shape: tuple[int, int], margin: int, root: Path | None = None, which: str = "warp",
             placements: Callable[[SheetId], object] | None = None) -> np.ndarray:
    """Bool (h, w) of the page `sid`: True on the ground under any `choose` or `test` crop of any sheet, grown by
    `margin` px (a square of side 2 margin + 1).  Points round each crop's box (`ring`) go to the ground through its
    sheet's placement (`placements`, None: `lotlabels.placement` with `which`) and from the ground onto this page
    through this sheet's; the polygon of the finite ones, in order, is filled.  When none is finite the crop's ground
    is not on this page; when only some are, a warning is logged naming the crop, the sheet and the count.
    FileNotFoundError naming a crop on a sheet with no placement."""
    get = (lambda s: lotlabels.placement(s, which)) if placements is None else placements
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    found: dict[SheetId, object] = {}

    def tr_of(s: SheetId, crop: Crop | None):
        if s not in found:
            try:
                found[s] = get(s)
            except FileNotFoundError:
                if crop is None:
                    raise
                raise FileNotFoundError(f"{crop.path}: sheet {s[0]}/{s[1]} has no placement; crops must lie on "
                                        f"placed sheets") from None
        return found[s]

    held = crops(HELD_OUT, root)
    if not held:
        return out.astype(bool)
    here = tr_of(sid, None)
    for c in held:
        pts = ring(c.box)
        px = here.to_pixel(tr_of((c.series, c.sheet), c).to_world(pts))
        ok = np.isfinite(px).all(axis=1)
        if not ok.any():
            continue
        if not ok.all():
            log.warning("%s: %d of %d points round its box could not be placed on %s/%s; reserved the polygon of the "
                        "rest", c.path, int((~ok).sum()), len(ok), sid[0], sid[1])
        cv2.fillPoly(out, [np.rint(px[ok]).astype(np.int32)], 1)
    if margin > 0:
        out = cv2.dilate(out, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return out.astype(bool)
