"""The counts a read of the finder is judged by, on the hand crops: how many printed blocks it finds, joins, cuts up or
loses, and how many zoning-border statements of the April 2000 table it agrees with.

- `blocks`: every hand area of kind `block` is exactly one of, tested in this order: `lost` (more than half its px lie
  in no found area), `found` (one found area holds more than half its px, and more than half of that area's px are
  it), `joined` (the found area holding most of it lies mostly outside it), `cut_up` (otherwise).  `extra`: the found
  areas more than half of whose px are hand street.  Open land is not counted and makes nothing extra.  Truth is
  `handlabels.make_labels`; px it leaves without a label count as no block and no street.
- `found_share`: the mean over the map eras of found / blocks (each era's crops summed), so an era with few blocks
  weighs as much as one with many.
- `fit_depth`: the watershed depth (`find.areas`) among i / 255, i = 1..254 (every level a stored read can tell
  apart) with the highest `found_share`, the smallest of a tie.
- `edge_feet`: how far, in feet, the hand block-edge px of found blocks lie from the nearest side of a found area.
- `borders`: of the used zoning-border cuts (`find.cuts`), how many of those that say "a zoning border crosses" and of
  those that say "none does" the read agrees with: the chance it gives the cut's own statement
  (`losses.statement_chance`) over one half, the more likely answer.

`chooser` scores a checkpoint by the `choose` crops' `found_share` at their fitted depth (the trainer keeps the best);
`count` reads every `test` crop with a run's kept model at the depth fitted on the `choose` crops.  A crop is read with
model.REACH px of scan round it where the page allows and scored on its own box only.

Reads the scans and frame cuts, the crop files, and for `count` each test crop's sheet placement (its ft per px, for
scoring only) and its sheet's label file data/find/labels/<series>/<sheet>.npz (the zoning-border cuts lying wholly in
the crop, for scoring only).  `count` writes data/find/runs/<run>/reads/test/<crop>.npz and look/test/<crop>.png."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

from sfzmaps.find import areas, cuts as cutlib, data, handlabels, look, losses, lotlabels, model, pages, read
from sfzmaps.find.cuts import Cuts
from sfzmaps.find.handlabels import CropLabels

log = logging.getLogger(__name__)

DEPTHS = tuple(i / read.LEVELS for i in range(1, read.LEVELS))
OUTCOMES = ("found", "joined", "cut_up", "lost")
TINY = 1e-12                # numerics only: a stored chance of 0 or 1 is read as this far inside (0, 1) for its logit


# --- the blocks -------------------------------------------------------------------------------------------------------

def _table(found: np.ndarray, truth: CropLabels) -> tuple[np.ndarray, np.ndarray]:
    """(hand area x found area) px counts, and each found area's size."""
    fa, ta = np.asarray(found).ravel(), np.asarray(truth.area).ravel()
    nf, nt = int(fa.max(initial=0)) + 1, int(ta.max(initial=0)) + 1
    joint = np.bincount(ta.astype(np.int64) * nf + fa, minlength=nt * nf).reshape(nt, nf)
    return joint, np.bincount(fa, minlength=nf)


def block_status(found: np.ndarray, truth: CropLabels) -> dict[int, str]:
    """Each hand block's outcome (`OUTCOMES`; see the module), by its area number."""
    joint, size = _table(found, truth)
    out = {}
    for i, kind in sorted(truth.kinds.items()):
        if kind != "block":
            continue
        row = joint[i]
        n = int(row.sum())
        if 2 * int(row[0]) > n or n == 0:
            out[i] = "lost"
            continue
        a = int(np.argmax(row[1:])) + 1              # the found area holding most of it (the first of a tie)
        held, whole = int(row[a]), int(size[a])
        if 2 * held > n and 2 * held > whole:
            out[i] = "found"
        elif 2 * (whole - held) > whole:
            out[i] = "joined"
        else:
            out[i] = "cut_up"
    return out


def blocks(found: np.ndarray, truth: CropLabels) -> dict:
    """The hand blocks (`blocks`), how many of them are each outcome, and the `extra` found areas (see the module)."""
    status = block_status(found, truth)
    out = {"blocks": len(status), **{k: list(status.values()).count(k) for k in OUTCOMES}}
    fa = np.asarray(found).ravel()
    size = np.bincount(fa)
    on_street = np.bincount(fa[np.asarray(truth.street).ravel() == handlabels.STREET], minlength=len(size))
    out["extra"] = int((2 * on_street[1:] > size[1:]).sum())
    return out


def _summed(dicts: list[dict]) -> dict:
    return {k: int(sum(d[k] for d in dicts)) for k in ("blocks", *OUTCOMES, "extra")}


def era_shares(dicts_by_era: dict[str, list[dict]]) -> dict[str, float]:
    """Each era's found / blocks, its crops summed; an era with no block is left out."""
    out = {}
    for era in sorted(dicts_by_era):
        n = sum(d["blocks"] for d in dicts_by_era[era])
        if n:
            out[era] = sum(d["found"] for d in dicts_by_era[era]) / n
    return out


def found_share(dicts_by_era: dict[str, list[dict]]) -> float:
    """The mean over eras of found / blocks (`era_shares`).  ValueError when no era has a block."""
    shares = era_shares(dicts_by_era)
    if not shares:
        raise ValueError("no hand block in any crop: nothing to find")
    return float(np.mean(list(shares.values())))


def _by_era(dicts: list[dict], eras: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for d, era in zip(dicts, eras):
        out.setdefault(era, []).append(d)
    return out


def fit_depth(reads: list[tuple[np.ndarray, np.ndarray]], truths: list[CropLabels],
              eras: list[str]) -> tuple[float, float]:
    """The depth among DEPTHS with the highest `found_share` over the crops, the smallest of a tie, and that share.
    `reads`: each crop's chances (3, h, w), as stored (`read.stored`), and its frame cut `keep` (h, w), on its box;
    `truths` and `eras` the crops' hand labels and map eras, in the same order."""
    if not (len(reads) == len(truths) == len(eras)) or not reads:
        raise ValueError(f"{len(reads)} reads, {len(truths)} truths, {len(eras)} eras: expected the same, at least 1")
    per_depth: list[list[dict | None]] = [[None] * len(reads) for _ in DEPTHS]
    for k, ((chances, keep), truth) in enumerate(zip(reads, truths)):
        for which, found in areas.areas_by_depth(chances[model.STREET], chances[model.EDGE], keep, DEPTHS):
            c = blocks(found, truth)
            for i in which:
                per_depth[i][k] = c
    best, best_share = 0, -1.0
    for i, dicts in enumerate(per_depth):
        share = found_share(_by_era(dicts, eras))
        if share > best_share:
            best, best_share = i, share
    return DEPTHS[best], best_share


# --- the block edges --------------------------------------------------------------------------------------------------

def _sides(found: np.ndarray) -> np.ndarray:
    """Bool: found-area px 8-adjacent to a px not of the same area (px off the crop do not count)."""
    h, w = found.shape
    padded = np.pad(found, 1, mode="edge")
    other = np.zeros(found.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                other |= padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] != found
    return (found > 0) & other


def edge_feet(found: np.ndarray, truth: CropLabels, ft_per_px: float) -> dict:
    """Of the hand block-edge px (edge 1) of the blocks `block_status` calls found: how many (`px`), and the median
    and 90th percentile (`median`, `p90`, ft) of their distance to the nearest px on a side of a found area (None when
    there is no such px or no side)."""
    found = np.asarray(found)
    mine = [i for i, s in block_status(found, truth).items() if s == "found"]
    on = (np.asarray(truth.edge) == 1) & np.isin(truth.area, mine)
    sides = _sides(found)
    out = {"px": int(on.sum()), "median": None, "p90": None}
    if on.any() and sides.any():
        d = ndimage.distance_transform_edt(~sides)[on] * ft_per_px
        out["median"], out["p90"] = float(np.median(d)), float(np.percentile(d, 90))
    return out


def ft_per_px(tr, box: tuple[int, int, int, int]) -> float:
    """The feet per page px at the box (x0, y0, w, h) through the placement `tr` (`to_world`, ft): the mean over its
    four sides of the side's length in feet over its length in px."""
    x0, y0, w, h = box
    corners = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], float)
    world = np.asarray(tr.to_world(corners), float)
    feet = np.linalg.norm(world - np.roll(world, -1, axis=0), axis=1)
    return float(np.mean(feet / np.array([w, h, w, h], float)))


# --- the zoning borders -----------------------------------------------------------------------------------------------

def cuts_in_box(c: Cuts, page_width: int, box: tuple[int, int, int, int]) -> Cuts:
    """The used cuts (weight above 0) of a page `page_width` px wide lying wholly in the box (x0, y0, w, h), their px
    re-indexed to the box (row * w + col), in their order."""
    x0, y0, w, h = box
    r, q = np.divmod(c.px, page_width)
    inbox = (r >= y0) & (r < y0 + h) & (q >= x0) & (q < x0 + w)
    n = len(c.answer)
    size = np.bincount(c.cut, minlength=n)
    keep = (size > 0) & (np.bincount(c.cut[inbox], minlength=n) == size) & (c.weight > 0)
    on = keep[c.cut]
    renumber = (np.cumsum(keep) - 1).astype(np.int32)
    return Cuts((r[on] - y0) * w + (q[on] - x0), renumber[c.cut[on]], c.prior[on], c.answer[keep], c.says[keep],
                c.weight[keep], c.ends[keep], np.zeros(0, np.int64))


def borders(border_chance: np.ndarray, c: Cuts) -> dict:
    """Of the used zoning-border cuts (answer BORDER, weight above 0) on the item whose zoning-border chances are
    `border_chance` (h, w; `c.px` index it flat): how many say "a zoning border crosses" (`crosses`) and how many of
    those the read agrees with (`crosses_agreed`), and the same for "none does" (`none`, `none_agreed`).  Agreeing:
    the chance the read gives the cut's own statement (`losses.statement_chance`) is over one half."""
    use = (c.answer == cutlib.BORDER) & (c.weight > 0)
    p = np.clip(np.asarray(border_chance, np.float64).ravel(), TINY, 1 - TINY)
    logit = torch.from_numpy(np.log(p) - np.log1p(-p))
    with torch.no_grad():
        chance = losses.statement_chance(logit, torch.from_numpy(c.px.astype(np.int64)),
                                         torch.from_numpy(c.cut.astype(np.int64)),
                                         torch.from_numpy(c.prior.astype(np.float64)),
                                         torch.from_numpy(c.says.astype(bool))).numpy()
    agreed = chance > 0.5
    return {"crosses": int((use & c.says).sum()), "crosses_agreed": int((use & c.says & agreed).sum()),
            "none": int((use & ~c.says).sum()), "none_agreed": int((use & ~c.says & agreed).sum())}


def _sheet_border_cuts(sid: pages.SheetId, shape: tuple[int, int]) -> Cuts | None:
    """The sheet's zoning-border cuts from its label file; None when it has no label file.  ValueError when the file
    does not fit the page."""
    f = lotlabels.sheet_path(sid)
    if not f.is_file():
        return None
    with np.load(f) as z:
        if z["street"].shape != tuple(shape):
            raise ValueError(f"{f}: labels are {z['street'].shape}, the page {tuple(shape)}")
        n = len(z["cuts_answer"])
        return Cuts(z["cuts_px"].astype(np.int64), z["cuts_cut"].astype(np.int32), z["cuts_prior"].astype(np.float32),
                    z["cuts_answer"].astype(np.uint8), z["cuts_says"].astype(bool),
                    z["cuts_weight"].astype(np.float32), np.full((n, 2), -1, np.int32), np.zeros(0, np.int64))


# --- reading the crops ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CropRead:
    """A hand crop ready to read: its scan and frame cut grown by model.REACH px where the page allows (`read.grown`),
    its box within them (`inner`), the page's shape, its era and its hand labels."""
    crop: handlabels.Crop
    era: str
    lum: np.ndarray
    keep: np.ndarray
    inner: tuple[slice, slice]
    page_shape: tuple[int, int]
    truth: CropLabels


def prepare(crops: list[handlabels.Crop]) -> list[CropRead]:
    """Each crop as a `CropRead` (one page in memory at a time).  ValueError or FileNotFoundError naming a crop that
    cannot be read."""
    out, page = [], None
    for c in crops:
        sid = (c.series, c.sheet)
        if page is None or page[0] != sid:
            lum = pages.load_scan(sid)
            page = (sid, lum, pages.load_keep(sid, lum.shape))
        _, lum, keep = page
        truth = handlabels.make_labels(handlabels.crop_scan(lum, c), c)
        outer, inner = read.grown(c.box, lum.shape)
        out.append(CropRead(c, pages.source(c.series).era, np.ascontiguousarray(lum[outer]),
                            np.ascontiguousarray(keep[outer]), inner, lum.shape, truth))
    return out


def read_one(net: model.FindNet, p: CropRead, device) -> tuple[np.ndarray, np.ndarray]:
    """The crop's chances on its box as stored (`read.stored`), and its frame cut there."""
    chances = read.read_page(net, p.lum, p.keep, device)[(slice(None), *p.inner)]
    return read.stored(chances), p.keep[p.inner]


def network(state_dict: dict) -> model.FindNet:
    """A FindNet holding `state_dict` (its width from the head), made without touching torch's generator."""
    with torch.random.fork_rng(devices=[]):
        net = model.FindNet(int(state_dict["head.weight"].shape[1]))
    net.load_state_dict(state_dict)
    return net


def _only(crops: list[handlabels.Crop], set_: str) -> None:
    if not crops:
        raise ValueError(f"no {set_} crop (inputs/find-labels/{set_}/)")
    bad = [f"{c.path} ({c.set})" for c in crops if c.set != set_]
    if bad:
        raise ValueError(f"only {set_} crops are read here, not {', '.join(bad)}")


def chooser(crops: list[handlabels.Crop], device) -> Callable[[int, dict], float]:
    """A checkpoint's score for the trainer (`train.fit`'s on_checkpoint): every crop read with the checkpoint's
    weights, and `fit_depth`'s share.  The crops are read from disk once, here.  ValueError when there is no crop or
    one is not in the `choose` set."""
    _only(crops, "choose")
    ready = prepare(crops)

    def score(step: int, state_dict: dict) -> float:
        t0 = time.perf_counter()
        net = network(state_dict)
        reads = [read_one(net, p, device) for p in ready]
        depth, share = fit_depth(reads, [p.truth for p in ready], [p.era for p in ready])
        log.info("step %d: the choosing crops find %.3f of their blocks (mean over eras) at depth %d/255; read and "
                 "scored in %.1f s", step, share, round(depth * read.LEVELS), time.perf_counter() - t0)
        return share

    return score


# --- a run's counts ---------------------------------------------------------------------------------------------------

def count(state_dict: dict, choose: list[handlabels.Crop], test: list[handlabels.Crop], device,
          folder: Path) -> dict:
    """The counts of a run's kept model (see the module): the depth fitted on the `choose` crops, then every `test`
    crop read at it; each test crop's read saved to <folder>/reads/test/<crop file stem>.npz and its picture to
    <folder>/look/test/<crop file stem>.png.  Returns what counts.json holds (no clock time, no absolute path).
    ValueError when a set has no crop, holds a crop of another set, or two of its crops share a file stem."""
    _only(choose, "choose")
    _only(test, "test")
    for crops in (choose, test):
        stems = [c.path.stem for c in crops]
        twice = sorted({s for s in stems if stems.count(s) > 1})
        if twice:
            raise ValueError(f"crops sharing a file stem (their pictures would overwrite): {', '.join(twice)}")
    net = network(state_dict)
    ready = prepare(choose)
    reads = [read_one(net, p, device) for p in ready]
    depth, share = fit_depth(reads, [p.truth for p in ready], [p.era for p in ready])
    at_depth = [blocks(areas.areas(c[model.STREET], c[model.EDGE], k, depth), p.truth) for (c, k), p in
                zip(reads, ready)]
    out = {"depth": depth, "depth_level": round(depth * read.LEVELS),
           "choose": {"found_share": share, "era_shares": era_shares(_by_era(at_depth, [p.era for p in ready])),
                      "crops": [data.crop_name(p.crop) for p in ready]}}
    rows, no_file = [], []
    for c in test:
        p = prepare([c])[0]
        sid = (c.series, c.sheet)
        chances, keep = read_one(net, p, device)
        read.save(folder / "reads" / "test" / f"{c.path.stem}.npz", chances)
        found = areas.areas(chances[model.STREET], chances[model.EDGE], keep, depth)
        status = block_status(found, p.truth)
        got = blocks(found, p.truth)
        scale = ft_per_px(lotlabels.placement(sid), c.box)
        zb = None
        if pages.source(c.series).borders in lotlabels.TABLE_BORDERS:
            sheet_cuts = _sheet_border_cuts(sid, p.page_shape)
            if sheet_cuts is None:
                no_file.append(data.crop_name(c))
            else:
                zb = borders(chances[model.BORDER], cuts_in_box(sheet_cuts, p.page_shape[1], c.box))
        picture = Path("look") / "test" / f"{c.path.stem}.png"
        (folder / picture).parent.mkdir(parents=True, exist_ok=True)
        x0, y0, w, h = c.box
        title = (f"test / {c.series} / {c.sheet}  page x {x0}-{x0 + w}, y {y0}-{y0 + h}; {p.era}; depth "
                 f"{out['depth_level']}/255; blocks {got['blocks']}: found {got['found']}, joined {got['joined']}, "
                 f"cut up {got['cut_up']}, lost {got['lost']}; extra areas {got['extra']}")
        cv2.imwrite(str(folder / picture), look.read_picture(p.lum[p.inner], chances, found, p.truth, status, title))
        rows.append({"name": data.crop_name(c), "era": p.era, "counts": got,
                     "edge_feet": edge_feet(found, p.truth, scale), "ft_per_px": scale, "zoning_borders": zb,
                     "picture": str(picture)})
        log.info("%s: %s", data.crop_name(c), got)
    by_era = _by_era([r["counts"] for r in rows], [r["era"] for r in rows])
    zb_rows = [r for r in rows if r["zoning_borders"] is not None]
    zb_by_era = _by_era([r["zoning_borders"] for r in zb_rows], [r["era"] for r in zb_rows])
    out["test"] = {"found_share": found_share(by_era), "era_shares": era_shares(by_era),
                   "eras": {e: _summed(d) for e, d in sorted(by_era.items())},
                   "pooled": _summed([r["counts"] for r in rows]),
                   "zoning_borders": {"eras": {e: _sum_keys(d) for e, d in sorted(zb_by_era.items())},
                                      "pooled": _sum_keys([r["zoning_borders"] for r in zb_rows]),
                                      "no_label_file": no_file},
                   "crops": rows}
    return out


def _sum_keys(dicts: list[dict]) -> dict:
    keys = ("crosses", "crosses_agreed", "none", "none_agreed")
    return {k: int(sum(d[k] for d in dicts)) for k in keys}
