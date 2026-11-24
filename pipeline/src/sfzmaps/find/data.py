"""What the finder trains on: a sheet's lot labels and a hand crop's labels as one kind of item (`Labelled`), cut into
tiles and turned.

A `Labelled` is a scan (or a part of one) with a label per px for each of the two answers (street, block edge), 1 yes
/ 0 no / 255 no label, a weight per px for the street answer, and the block-edge cut statements of `find.cuts` (None
for a hand crop):
- a sheet labelled from lots (`sheet_labelled`): street and its weight from the label file; block edge 0 on its
  `edge_none` px, 255 elsewhere (an edge is said to be there only by a cut);
- a hand crop (`crop_labelled`): street and block edge from `handlabels.make_labels`, street weight 1 where labelled.
  The item is the crop grown by `model.REACH` px on every side where the page allows: the scan is shown there, so the
  network sees the print round a crop's edge as it does when a page is read, but nothing is labelled there.
The ground under the `choose` and `test` crops, grown by `model.REACH` (`handlabels.reserved`), has no label on any
sheet or training crop, and a cut touching it is dropped whole; the scan is still shown there.

Each item knows its print size (`Labelled.ft_per_px`, through its sheet's placement: a sheet's over its whole page, a
crop's at its box) and a sheet its cuts' `sigma` (the label file's).  `print_sizes` measures the print sizes the finder
is used on, every placed sheet of every series of the label sources.

`window` cuts a square out of an item (a cut is kept only when every px of it lies in the square); `resized` shows a
square at another size, its labels and cuts with it; `turned` turns an item and every label a number of quarter turns.

Reads data/find/labels/<series>/<sheet>.npz (made by `sfzmaps find labels`), the scans, the crop files and the sheets'
placements (the reserved ground, the items' scales; `print_sizes` every sheet's).  Writes nothing."""
from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from sfzmaps.find import cuts as cutting, handlabels, lotlabels, model, pages
from sfzmaps.find.cuts import Cuts
from sfzmaps.find.pages import SheetId

NONE = 255                  # no label
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Labelled:
    """A scan and its labels, all (h, w) in the item's own px; `cuts.px` index the item's px flat (row * w + col)."""
    name: str
    era: str
    lum: np.ndarray             # float32: luminance as stored, 1 paper to 0 ink
    street: np.ndarray          # uint8: 1 street, 0 not, 255 no label
    street_w: np.ndarray        # float32: the street label's weight, 0 where there is none
    edge: np.ndarray            # uint8: 1 a block edge runs here, 0 none does, 255 no label
    cuts: Cuts | None           # the block-edge cut statements; None for a hand crop
    left_out: float = 0.0       # a training crop's share of px left without a label as reserved ground (0 otherwise)
    set: str | None = None      # a hand crop's set (train, choose, test); None for a sheet
    ft_per_px: float = math.nan  # the print's feet per px, through the sheet's placement (`lotlabels.ft_per_px`)
    sigma: float = math.nan     # the cuts' sigma, px (the label file's); NaN for a hand crop

    @property
    def hand(self) -> bool:
        """True for a hand crop (or a tile of one)."""
        return self.set is not None


def sheet_labelled(sid: SheetId, which: str = "warp") -> Labelled:
    """The sheet's lot labels (data/find/labels/<series>/<sheet>.npz) as a `Labelled` (see the module), with the
    reserved ground through placement `which` left unlabelled and every cut with a px on it dropped.
    FileNotFoundError or ValueError naming the command that makes the label file (`lotlabels.load_sheet_labels`):
    missing, made through another placement, or not fitting the scan."""
    lum = pages.load_scan(sid)
    lab = lotlabels.load_sheet_labels(sid, which, lum.shape)
    found = lotlabels.file_cuts(lab)
    held = handlabels.reserved(sid, lum.shape, model.REACH, which=which)
    street = np.where(held, NONE, lab["street"]).astype(np.uint8)
    street_w = np.where(held | (street == NONE), 0, lab["weight"]).astype(np.float32)
    edge = np.where(lab["edge_none"] & ~held, 0, NONE).astype(np.uint8)
    off_held = np.bincount(found.cut[held.ravel()[found.px]], minlength=len(found.weight)) == 0
    scale = lotlabels.page_ft_per_px(lotlabels.placement(sid, which), lum.shape)
    return Labelled(f"{sid[0]}/{sid[1]}", pages.source(sid[0]).era, lum, street, street_w, edge, found.only(off_held),
                    ft_per_px=scale, sigma=float(lab["sigma"]))


def grown(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """The page slices of the box (x0, y0, w, h) grown by model.REACH px on every side where a page of `shape` allows,
    and the box's own slices within that: a crop as the trainer shows it and as `find.counts` reads it."""
    x0, y0, w, h = box
    gx0, gy0 = max(0, x0 - model.REACH), max(0, y0 - model.REACH)
    gx1, gy1 = min(shape[1], x0 + w + model.REACH), min(shape[0], y0 + h + model.REACH)
    return ((slice(gy0, gy1), slice(gx0, gx1)),
            (slice(y0 - gy0, y0 - gy0 + h), slice(x0 - gx0, x0 - gx0 + w)))


def crop_name(crop: handlabels.Crop) -> str:
    return f"{crop.set}/{crop.series}/{crop.path.stem}"


def crop_labelled(crop: handlabels.Crop) -> Labelled:
    """A hand crop's labels as a `Labelled` (see the module): the crop grown by `model.REACH` px where the page
    allows (`grown`).  For a `train` crop, the px on reserved ground are left without a label, and one line is logged
    naming the crop and the share of its px left out; a `choose` or `test` crop is the reserved ground and is never
    masked."""
    sid = (crop.series, crop.sheet)
    page = pages.load_scan(sid)
    lab = handlabels.make_labels(handlabels.crop_scan(page, crop), crop)
    outer, inner = grown(crop.box, page.shape)
    gy0, gy1, gx0, gx1 = outer[0].start, outer[0].stop, outer[1].start, outer[1].stop
    shape = (gy1 - gy0, gx1 - gx0)
    street, edge = (np.full(shape, NONE, np.uint8) for _ in range(2))
    street[inner] = lab.street
    edge[inner] = lab.edge
    left_out = 0.0
    if crop.set not in handlabels.HELD_OUT:
        held = handlabels.reserved(sid, page.shape, model.REACH)[gy0:gy1, gx0:gx1]
        left_out = float(held[inner].mean())
        for a in (street, edge):
            a[held] = NONE
        if left_out:
            log.info("%s: %.3f of its px lie on reserved ground and are left without a label", crop_name(crop),
                     left_out)
    street_w = (street != NONE).astype(np.float32)
    scale = lotlabels.ft_per_px(lotlabels.placement(sid), crop.box)
    return Labelled(crop_name(crop), pages.source(crop.series).era, np.ascontiguousarray(page[gy0:gy1, gx0:gx1]),
                    street, street_w, edge, None, left_out, crop.set, ft_per_px=scale)


def print_sizes(which: str = "warp") -> tuple[dict[str, list[float]], list[str]]:
    """The print sizes the finder is used on: for every series of the label sources (`pages.sources`), in order, the
    ft per px over the whole page (`lotlabels.page_ft_per_px`) of each of its sheets (`pages.sheets`: duplicate scans
    left out) that has a placement `which` on disk; and the series with no such sheet, left out."""
    sizes, left_out = {}, []
    for series in pages.sources():
        found = []
        for sid in pages.sheets(series):
            try:
                tr = lotlabels.placement(sid, which)
            except FileNotFoundError:
                continue
            found.append(lotlabels.page_ft_per_px(tr, lotlabels.page_shape(sid)))
        if found:
            sizes[series] = found
        else:
            left_out.append(series)
    return sizes, left_out


def size_summary(sizes: dict[str, list[float]]) -> dict[str, dict]:
    """Per series: how many sheets were measured and their least, median and most ft per px."""
    return {s: {"count": len(v), "least": float(min(v)), "median": float(np.median(v)), "most": float(max(v))}
            for s, v in sizes.items()}


def labelled(item: Labelled) -> np.ndarray:
    """Bool (h, w): the px that carry a label: a street weight above 0, a block-edge label, or a px of a used cut
    (weight above 0)."""
    out = (item.street_w > 0) | (item.edge != NONE)
    if item.cuts is not None and len(item.cuts.weight):
        out |= item.cuts.on(item.lum.shape, which=item.cuts.weight > 0)
    return out


def cuts_without_px(c: Cuts) -> int:
    """How many used cuts (weight above 0) have no px: a malformed statement, which `cut_boxes` (and so every tile)
    leaves out."""
    return int(((c.weight > 0) & (np.bincount(c.cut, minlength=len(c.weight)) == 0)).sum())


@dataclass(frozen=True)
class CutBoxes:
    """Each used cut's px run (`start`, `stop` into `order`, which lists the px of each cut together) and its box of
    rows and columns, inclusive: so a window finds the cuts wholly inside it without reading every px."""
    order: np.ndarray
    start: np.ndarray
    stop: np.ndarray
    r0: np.ndarray
    r1: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    used: np.ndarray            # the cuts' numbers in `Cuts`


def cut_boxes(c: Cuts, width: int) -> CutBoxes:
    """The `CutBoxes` of `c`'s used cuts (weight above 0) with a px (`cuts_without_px` counts those without) on an
    item `width` px wide."""
    size = np.bincount(c.cut, minlength=len(c.weight))
    use = (c.weight > 0) & (size > 0)
    used = np.flatnonzero(use)
    idx = np.flatnonzero(use[c.cut])
    order = idx[np.argsort(c.cut[idx], kind="stable")]
    counts = size[used]
    stop = np.cumsum(counts)
    start = stop - counts
    r, q = np.divmod(c.px[order], width)
    if len(order):
        r0, r1 = np.minimum.reduceat(r, start), np.maximum.reduceat(r, start)
        c0, c1 = np.minimum.reduceat(q, start), np.maximum.reduceat(q, start)
    else:
        r0 = r1 = c0 = c1 = np.zeros(0, np.int64)
    return CutBoxes(order, start, stop, r0, r1, c0, c1, used)


def window(item: Labelled, y0: int, x0: int, side: int, boxes: CutBoxes | None = None) -> Labelled:
    """The square of `side` px of `item` from row y0, column x0, as a `Labelled`.  A cut is kept only when every px of
    it lies in the square, and only a used one (weight above 0); its px are re-indexed to the square.  `boxes`: the
    item's `cut_boxes`, made here when None.  ValueError for a square that leaves the item."""
    h, w = item.lum.shape
    if y0 < 0 or x0 < 0 or y0 + side > h or x0 + side > w:
        raise ValueError(f"{item.name}: a {side} px square at ({y0}, {x0}) leaves the item ({h} x {w} px)")
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    found = None
    if item.cuts is not None:
        c = item.cuts
        b = cut_boxes(c, w) if boxes is None else boxes
        inside = (b.r0 >= y0) & (b.r1 < y0 + side) & (b.c0 >= x0) & (b.c1 < x0 + side)
        start, stop = b.start[inside], b.stop[inside]
        length = stop - start
        first = np.repeat(start - (np.cumsum(length) - length), length)
        take = b.order[first + np.arange(int(length.sum()))]
        r, q = np.divmod(c.px[take], w)
        which = b.used[inside]
        found = Cuts((r - y0) * side + (q - x0), np.repeat(np.arange(len(which), dtype=np.int32), length),
                     c.prior[take], c.weight[which], c.inside[which], np.zeros(0, np.int64))
    return dataclasses.replace(item, lum=item.lum[sl], street=item.street[sl], street_w=item.street_w[sl],
                               edge=item.edge[sl], cuts=found)


def resized(item: Labelled, y0: int, x0: int, side: int, size: int, boxes: CutBoxes | None = None) -> Labelled:
    """The square of `side` px of `item` from row y0, column x0 (`window`) shown as a tile of `size` px, the print
    enlarged (size above side) or shrunk by the factor f = size / side: the scan area-averaged when shrinking and
    linear when enlarging (cv2); the street label, its weight and "no label" at each tile px those of the source px
    under its centre; the block-edge label likewise when enlarging, and when shrinking "no label" where any source px
    the tile px covers has none, else 1 where any is an edge, else 0 (a one-px edge stays unbroken); the cuts laid
    again at the new size (`cuts.relaid`, with the item's `sigma`).  The tile's `ft_per_px` and `sigma` are the
    item's at the new size.  A `side` equal to `size` gives `window`'s square itself.  ValueError for a square that
    leaves the item, or cuts with no sigma."""
    src = window(item, y0, x0, side, boxes)
    if side == size:
        return src
    f = size / side
    under = ((2 * np.arange(size) + 1) * side) // (2 * size)      # the source px under each tile px's centre
    pick = np.ix_(under, under)
    shrink = size < side
    lum = cv2.resize(src.lum, (size, size), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    if shrink:
        covers = {v: cv2.resize((src.edge == v).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0
                  for v in (NONE, 1)}
        edge = np.where(covers[NONE], NONE, np.where(covers[1], 1, 0)).astype(np.uint8)
    else:
        edge = src.edge[pick]
    found = None
    if src.cuts is not None:
        if not math.isfinite(item.sigma):
            raise ValueError(f"{item.name}: its cuts have no sigma; they cannot be laid again at another size")
        found = cutting.relaid(src.cuts, side, size, item.sigma)
    return dataclasses.replace(src, lum=np.ascontiguousarray(lum, np.float32), street=src.street[pick],
                               street_w=src.street_w[pick], edge=edge, cuts=found, ft_per_px=item.ft_per_px / f,
                               sigma=item.sigma * f)


def turned(tile: Labelled, k: int) -> Labelled:
    """`tile` and every label turned k quarter turns (`np.rot90`, counter-clockwise), the cut px re-indexed."""
    def rot(a):
        return np.ascontiguousarray(np.rot90(a, k))
    found = tile.cuts
    if found is not None:
        h, w = tile.lum.shape
        where = np.empty(h * w, np.int64)
        where[rot(np.arange(h * w).reshape(h, w)).ravel()] = np.arange(h * w)      # old flat px -> new flat px
        found = dataclasses.replace(found, px=where[found.px])
    return dataclasses.replace(tile, lum=rot(tile.lum), street=rot(tile.street), street_w=rot(tile.street_w),
                               edge=rot(tile.edge), cuts=found)
