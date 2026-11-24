"""Hand crops: a person's seeds, strokes and limits on a crop of a scan, the labels made from them, and the ground
left out of training under the test and choosing crops.

A crop file is inputs/find-labels/<set>/<series>/<sheet>__<x0>_<y0>_<w>_<h>.json (the format and the reading of the
maps: inputs/find-labels/README.md).  `make_labels` turns its marks into per-px labels by seven rules: paper by Otsu's
threshold, strokes and limits as barriers, street as the paper 4-connected to a street seed, areas as the 8-connected
islands holding a block or open seed.  This is a drawing aid for hand labels only, checked by eye on its picture
(`look.crop`): its 3 px barrier width and its 4-connected street fill and 8-connected islands are parts of the tool,
and nothing like it runs when a sheet is read.

`reserved` is the ground under every `choose` and `test` crop, on any sheet's page, which training leaves out.

Reads the crop files and (for `reserved`) the sheets' placements.  Writes nothing."""
from __future__ import annotations

import json
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
KEYS = {"box", "street", "blocks", "open", "strokes", "limits", "notes", "drawn_by"}
BARRIER = 3                         # tool setting: a stroke's or limit's width, px
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


def _points(path: Path, key: str, value) -> tuple[Point, ...]:
    try:
        return tuple((int(round(x)), int(round(y))) for x, y in value)
    except (TypeError, ValueError):
        raise ValueError(f"{path}: `{key}` must be a list of [x, y]") from None


def read_crop(path: Path, set_: str) -> Crop:
    """The crop file at `path` (in `set_`).  ValueError for a name not `<sheet>__<x0>_<y0>_<w>_<h>.json`, a name and
    `box` that disagree, an unknown key, or a mark that is not [x, y] points."""
    path = Path(path)
    sheet, _, tail = path.stem.rpartition("__")
    try:
        named = tuple(int(v) for v in tail.split("_"))
    except ValueError:
        named = ()
    if not sheet or len(named) != 4:
        raise ValueError(f"{path}: expected a name <sheet>__<x0>_<y0>_<w>_<h>.json")
    d = json.loads(path.read_text())
    unknown = sorted(set(d) - KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {', '.join(unknown)} (expected {', '.join(sorted(KEYS))})")
    box = tuple(int(v) for v in d.get("box", ()))
    if box != named:
        raise ValueError(f"{path}: its name says box {list(named)}, the file {list(box)}")
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError(f"{path}: the box's width and height must be positive")
    lines = {k: tuple(_points(path, k, s) for s in d.get(k, [])) for k in ("strokes", "limits")}
    return Crop(set_, path.parent.name, sheet, box, _points(path, "street", d.get("street", [])),
                _points(path, "blocks", d.get("blocks", [])), _points(path, "open", d.get("open", [])),
                lines["strokes"], lines["limits"], path)


def crops(sets=SETS, root: Path | None = None) -> list[Crop]:
    """Every crop file of `sets` under `root` (None: inputs/find-labels), sorted by path.  ValueError for an unknown
    set, a bad file (`read_crop`) or a crop on a duplicate scan (never read)."""
    root = root_dir() if root is None else Path(root)
    bad = sorted(set(sets) - set(SETS))
    if bad:
        raise ValueError(f"unknown crop sets {', '.join(bad)} (expected {', '.join(SETS)})")
    found = sorted((f, s) for s in sets for f in (root / s).glob("*/*.json"))
    out = [read_crop(f, s) for f, s in found]
    skip: dict[str, dict] = {}
    for c in out:
        if c.series not in skip:
            skip[c.series] = hatching.duplicate_scans(c.series)
        if c.sheet in skip[c.series]:
            raise ValueError(f"{c.path}: {c.sheet} is a duplicate scan of {skip[c.series][c.sheet]}, which is never "
                             f"read; draw the crop on that scan")
    return out


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


def make_labels(lum_crop: np.ndarray, crop: Crop) -> CropLabels:
    """The crop's labels from its marks (the module's rules, as inputs/find-labels/README.md states them).  ValueError
    naming the file and the seed for a seed off the crop, on a stroke or limit, a street seed not on paper, a block or
    open seed the street reaches, and two seeds in one island."""
    lum = np.asarray(lum_crop, np.float32)
    x0, y0, w, h = crop.box
    if lum.shape != (h, w):
        raise ValueError(f"{crop.path}: the crop is {lum.shape}, its box {w} x {h}")
    stroke = _barrier(crop.strokes, x0, y0, lum.shape)
    limit = _barrier(crop.limits, x0, y0, lum.shape)
    barrier = stroke | limit
    paper = (lum > threshold_otsu(lum)) & ~barrier
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
    smallest = min((sizes[i] for i in seeded), default=None)
    warnings = []
    if smallest is not None:
        missed = [i for i in range(1, n + 1) if i not in seeded and sizes[i] > smallest]
        if missed:
            depth = ndimage.distance_transform_edt(np.pad(islands > 0, 1))[1:-1, 1:-1]
            for i in missed:
                yy, xx = np.unravel_index(int(np.argmax(np.where(islands == i, depth, -1))), lum.shape)
                warnings.append(f"possible missed block at {int(xx) + x0}, {int(yy) + y0}")
    label = np.where(barrier, NONE, np.where(area > 0, AREA, STREET)).astype(np.uint8)
    edge = np.zeros(lum.shape, np.uint8)
    padded = np.pad(area, 1, mode="edge")
    other = np.zeros(lum.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                other |= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] != area
    edge[(area > 0) & other] = 1
    edge[stroke] = 1
    edge[_next_to(limit)] = NONE
    edge[[0, -1], :] = NONE
    edge[:, [0, -1]] = NONE
    return CropLabels(label, area, kinds, edge, warnings)


def reserved(sid: SheetId, shape: tuple[int, int], margin: int, root: Path | None = None, which: str = "warp",
             placements: Callable[[SheetId], object] | None = None) -> np.ndarray:
    """Bool (h, w) of the page `sid`: True on the ground under any `choose` or `test` crop of any sheet, grown by
    `margin` px (a square of side 2 margin + 1).  Each crop's box corners and side midpoints go to the ground through
    its sheet's placement (`placements`, None: `lotlabels.placement` with `which`) and from the ground onto this page
    through this sheet's; the polygon they make is filled (points the placement cannot bring onto this page are left
    out; with fewer than three left the crop's ground is not on it).  FileNotFoundError naming a crop on a sheet with
    no placement."""
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
        x0, y0, cw, ch = c.box
        x1, y1, xm, ym = x0 + cw - 1, y0 + ch - 1, x0 + (cw - 1) / 2, y0 + (ch - 1) / 2
        ring = np.array([(x0, y0), (xm, y0), (x1, y0), (x1, ym), (x1, y1), (xm, y1), (x0, y1), (x0, ym)], float)
        px = here.to_pixel(tr_of((c.series, c.sheet), c).to_world(ring))
        px = px[np.isfinite(px).all(axis=1)]
        if len(px) >= 3:
            cv2.fillPoly(out, [np.rint(px).astype(np.int32)], 1)
    if margin > 0:
        out = cv2.dilate(out, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return out.astype(bool)
