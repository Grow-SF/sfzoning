"""What the finder trains on: a sheet's lot labels and a hand crop's labels as one kind of item (`Labelled`), cut into
tiles and turned.

A `Labelled` is a scan (or a part of one) with a label per px for each of the three answers, 1 yes / 0 no / 255 no
label, a weight per px for the street answer, and the cut statements of `find.cuts` (None for a hand crop):
- a sheet labelled from lots (`sheet_labelled`): street and its weight from the label file; block edge 0 on its
  `edge_none` px, zoning border 0 on its `border_none` px, 255 elsewhere (a line is said to be there only by a cut);
- a hand crop (`crop_labelled`): street and block edge from `handlabels.make_labels`, street weight 1 where labelled;
  zoning border 0 on the whole crop when the series draws none (`pages.Source.borders` "none"), else no label.  The
  item is the crop grown by `model.REACH` px on every side where the page allows: the scan is shown there, so the
  network sees the print round a crop's edge as it does when a page is read, but nothing is labelled there.
The ground under the `choose` and `test` crops, grown by `model.REACH` (`handlabels.reserved`), has no label on any
sheet or training crop, and a cut touching it is dropped whole; the scan is still shown there.

`window` cuts a square out of an item (a cut is kept only when every px of it lies in the square); `turned` turns an
item and every label a number of quarter turns.

Reads data/find/labels/<series>/<sheet>.npz (made by `sfzmaps find labels`), the scans, the crop files and (for the
reserved ground) the sheets' placements.  Writes nothing."""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import numpy as np

from sfzmaps.find import cuts as cutlib, handlabels, lotlabels, model, pages
from sfzmaps.find.cuts import Cuts
from sfzmaps.find.pages import SheetId

NONE = 255                  # no label
ANSWERS = model.ANSWERS
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
    border: np.ndarray          # uint8: 1 a zoning border runs here, 0 none does, 255 no label
    cuts: Cuts | None           # the cut statements; None for a hand crop
    left_out: float = 0.0       # a training crop's share of px left without a label as reserved ground (0 otherwise)
    set: str | None = None      # a hand crop's set (train, choose, test); None for a sheet

    @property
    def hand(self) -> bool:
        """True for a hand crop (or a tile of one)."""
        return self.set is not None


def _no_cut_on(c: Cuts, ground: np.ndarray) -> Cuts:
    """`c` less every cut with a px on `ground` (bool, flat page), the rest renumbered in order."""
    n = len(c.answer)
    keep = np.bincount(c.cut[ground[c.px]], minlength=n) == 0
    on = keep[c.cut]
    renumber = (np.cumsum(keep) - 1).astype(np.int32)
    return Cuts(c.px[on], renumber[c.cut[on]], c.prior[on], c.answer[keep], c.says[keep], c.weight[keep],
                c.ends[keep], c.dropped)


def sheet_labelled(sid: SheetId, which: str = "warp") -> Labelled:
    """The sheet's lot labels (data/find/labels/<series>/<sheet>.npz) as a `Labelled` (see the module), with the
    reserved ground through placement `which` left unlabelled.  FileNotFoundError naming the command that makes a
    missing label file; ValueError when the file was made through another placement or does not fit the scan."""
    f = lotlabels.sheet_path(sid)
    if not f.is_file():
        raise FileNotFoundError(f"{f}: no labels; make them with `sfzmaps find labels --series {sid[0]} "
                                f"--sheet {sid[1]}`")
    with np.load(f) as z:
        lab = {k: z[k] for k in z.files}
    if str(lab["placement"]) != which:
        raise ValueError(f"{f}: made through the {lab['placement']} placement, not {which}")
    lum = pages.load_scan(sid)
    if lab["street"].shape != lum.shape:
        raise ValueError(f"{f}: labels are {lab['street'].shape}, the scan {lum.shape}")
    n = len(lab["cuts_answer"])
    found = Cuts(lab["cuts_px"].astype(np.int64), lab["cuts_cut"].astype(np.int32),
                 lab["cuts_prior"].astype(np.float32), lab["cuts_answer"].astype(np.uint8),
                 lab["cuts_says"].astype(bool), lab["cuts_weight"].astype(np.float32),
                 np.full((n, 2), -1, np.int32), np.zeros(0, np.int64))          # the file holds no `ends`
    held = handlabels.reserved(sid, lum.shape, model.REACH, which=which)
    street = np.where(held, NONE, lab["street"]).astype(np.uint8)
    street_w = np.where(held | (street == NONE), 0, lab["weight"]).astype(np.float32)
    edge = np.where(lab["edge_none"] & ~held, 0, NONE).astype(np.uint8)
    border = np.where(lab["border_none"] & ~held, 0, NONE).astype(np.uint8)
    return Labelled(f"{sid[0]}/{sid[1]}", pages.source(sid[0]).era, lum, street, street_w, edge, border,
                    _no_cut_on(found, held.ravel()))


def crop_name(crop: handlabels.Crop) -> str:
    return f"{crop.set}/{crop.series}/{crop.path.stem}"


def crop_labelled(crop: handlabels.Crop) -> Labelled:
    """A hand crop's labels as a `Labelled` (see the module): the crop grown by `model.REACH` px where the page
    allows.  For a `train` crop, the px on reserved ground are left without a label, and one line is logged naming
    the crop and the share of its px left out; a `choose` or `test` crop is the reserved ground and is never masked."""
    sid = (crop.series, crop.sheet)
    page = pages.load_scan(sid)
    lab = handlabels.make_labels(handlabels.crop_scan(page, crop), crop)
    x0, y0, w, h = crop.box
    gx0, gy0 = max(0, x0 - model.REACH), max(0, y0 - model.REACH)
    gx1, gy1 = min(page.shape[1], x0 + w + model.REACH), min(page.shape[0], y0 + h + model.REACH)
    inner = (slice(y0 - gy0, y0 - gy0 + h), slice(x0 - gx0, x0 - gx0 + w))
    shape = (gy1 - gy0, gx1 - gx0)
    street, edge, border = (np.full(shape, NONE, np.uint8) for _ in range(3))
    street[inner] = lab.street
    edge[inner] = lab.edge
    if pages.source(crop.series).borders == "none":
        border[inner] = 0
    left_out = 0.0
    if crop.set not in handlabels.HELD_OUT:
        held = handlabels.reserved(sid, page.shape, model.REACH)[gy0:gy1, gx0:gx1]
        left_out = float(held[inner].mean())
        for a in (street, edge, border):
            a[held] = NONE
        if left_out:
            log.info("%s: %.3f of its px lie on reserved ground and are left without a label", crop_name(crop),
                     left_out)
    street_w = (street != NONE).astype(np.float32)
    return Labelled(crop_name(crop), pages.source(crop.series).era, np.ascontiguousarray(page[gy0:gy1, gx0:gx1]),
                    street, street_w, edge, border, None, left_out, crop.set)


def labelled(item: Labelled, answers=ANSWERS) -> np.ndarray:
    """Bool (h, w): the px that carry a label of one of `answers`: a street weight above 0, a block-edge or
    zoning-border label, or a px of a used cut (weight above 0) of those answers."""
    out = np.zeros(item.lum.shape, bool)
    if "street" in answers:
        out |= item.street_w > 0
    if "edge" in answers:
        out |= item.edge != NONE
    if "border" in answers:
        out |= item.border != NONE
    if item.cuts is not None and len(item.cuts.answer):
        c = item.cuts
        mine = np.isin(c.answer, [a for a, name in ((cutlib.EDGE, "edge"), (cutlib.BORDER, "border"))
                                  if name in answers])
        out |= c.on(item.lum.shape, which=(c.weight > 0) & mine)
    return out


def cuts_without_px(c: Cuts) -> int:
    """How many used cuts (weight above 0) have no px: a malformed statement, which `cut_boxes` (and so every tile)
    leaves out."""
    return int(((c.weight > 0) & (np.bincount(c.cut, minlength=len(c.answer)) == 0)).sum())


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
    size = np.bincount(c.cut, minlength=len(c.answer))
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
                     c.prior[take], c.answer[which], c.says[which], c.weight[which], c.ends[which],
                     np.zeros(0, np.int64))
    return dataclasses.replace(item, lum=item.lum[sl], street=item.street[sl], street_w=item.street_w[sl],
                               edge=item.edge[sl], border=item.border[sl], cuts=found)


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
                               edge=rot(tile.edge), border=rot(tile.border), cuts=found)
