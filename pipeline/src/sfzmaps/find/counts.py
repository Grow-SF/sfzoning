"""The counts a read of the finder is judged by, on the hand crops: how many printed blocks it finds, joins, cuts up or
loses, holds whole, and where it puts the street and the block boundary.

- `blocks`: every hand area of kind `block` is exactly one of, tested in this order: `lost` (more than half its px lie
  in no found area), `found` (one found area holds more than half its px, and more than half of that area's px are
  it), `joined` (the found area holding most of it lies mostly outside it), `cut_up` (otherwise).  `extra`: the found
  areas more than half of whose px are hand street.  Open land is not counted and makes nothing extra.  Truth is
  `handlabels.make_labels`; px it leaves without a label count as no block and no street.
- `found_share`: the mean over the map eras of found / blocks (each era's crops summed), so an era with few blocks
  weighs as much as one with many.
- `panoptic_parts` / `panoptic_summed`: panoptic quality, how well the found areas match the answer key's areas
  (blocks and open land), counting stray areas and missed ones: a found area and a key area match when their overlap
  (px in both over px in either) is over one half; quality = the matched overlaps summed / (matched + unmatched found
  areas / 2 + unmatched key areas / 2).
- `fit_depth`: the watershed depth (`find.areas`) among i / 255, i = 1..254 (every level a stored read can tell
  apart) with the highest panoptic quality (the mean over the map eras, each era's crops summed), the smallest of a
  tie.  The share of blocks found does not see stray areas or cut-up blocks; panoptic quality does.
- `whole`: per hand block, the share of it its biggest found area holds, and whether it is held whole (WHOLE, a
  reporting level); a block split 60/40 is `found` but not whole.
- `ground`: of the hand street px, how many the read calls street, and of the hand area px how many it calls not
  street (its street chance over one half, the more likely answer).
- `boundary_feet`: from each hand block-boundary px, the distance in feet to the nearest px the read calls a block
  edge (its block-edge chance over one half): the median, 90th and 99th percentile.
- `model_edge_feet`: the other half: from each px the read calls a block edge on the crop's labelled ground, the
  distance in feet to the nearest answer-key boundary px; the median, 90th and 99th percentile.
- `edge_feet`: how far, in feet, the hand block-edge px of found blocks lie from the nearest side of a found area.

`chooser` scores a checkpoint by the `choose` crops' `found_share` at their fitted depth (and logs their panoptic
quality there), for the trainer's log (the
held-out trace; the trainer keeps the last step whatever it scores); `count` reads every `test` crop with a run's
kept model at the depth fitted on the `choose` crops (the choosing crops' one use in a count).  Both refuse a choosing
crop lying on a test crop's ground (`handlabels.refuse_choose_on_test`).  A crop is read with model.REACH px of scan
round it where the page allows (`data.grown`) and scored on its own box only.

Reads the scans and frame cuts, the crop files, every held-out crop's sheet placement (for the choosing crops'
ground), and for `count` each test crop's sheet placement (its ft per px, for scoring only).  `count` writes
data/find/runs/<run>/reads/test/<series>__<crop>.npz and look/test/<series>__<crop>.png."""
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

from sfzmaps.find import areas, data, handlabels, look, lotlabels, model, pages, read
from sfzmaps.find.handlabels import CropLabels

log = logging.getLogger(__name__)

DEPTHS = tuple(i / read.LEVELS for i in range(1, read.LEVELS))
OUTCOMES = ("found", "joined", "cut_up", "lost")
WHOLE = 0.9                 # reporting level only, never used to choose anything: a hand block is held "whole" when
                            # its biggest found area holds at least this share of it and it is at least this share of
                            # that area
PIECE = 0.1                 # reporting level only: a found area holding at least this share of a hand block is one of
                            # the pieces it lies in


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


def whole(found: np.ndarray, truth: CropLabels) -> dict:
    """Per hand block (`per_block`, by area number): the share of it its biggest found area holds (`held`; 0 when no
    found area touches it), whether it is held whole (that area holds at least WHOLE of it and it is at least WHOLE of
    that area) and how many found areas hold PIECE or more of it (`pieces`); and the counts: `blocks`, `whole`, and
    `in_two_or_more` (blocks with two or more such pieces)."""
    joint, size = _table(found, truth)
    per = []
    for i, kind in sorted(truth.kinds.items()):
        if kind != "block":
            continue
        row = joint[i]
        n = int(row.sum())
        held, is_whole, pieces = 0.0, False, 0
        if n and row[1:].any():
            a = int(np.argmax(row[1:])) + 1          # the biggest found area in it (the first of a tie)
            held = int(row[a]) / n
            is_whole = held >= WHOLE and int(row[a]) / int(size[a]) >= WHOLE
            pieces = int((row[1:] >= PIECE * n).sum())
        per.append({"block": int(i), "held": float(held), "whole": bool(is_whole), "pieces": pieces})
    return {"blocks": len(per), "whole": sum(b["whole"] for b in per),
            "in_two_or_more": sum(b["pieces"] >= 2 for b in per), "per_block": per}


def whole_summed(dicts: list[dict]) -> dict:
    """`whole` of several crops together: the counts summed, the share of blocks held whole (`whole_share`) and the
    median of every block's `held` (`median_held`); None for a share or median of no block."""
    per = [b for d in dicts for b in d["per_block"]]
    out = {k: int(sum(d[k] for d in dicts)) for k in ("blocks", "whole", "in_two_or_more")}
    out["whole_share"] = out["whole"] / out["blocks"] if out["blocks"] else None
    out["median_held"] = float(np.median([b["held"] for b in per])) if per else None
    return out


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
        raise ValueError("no answer-key block in any crop: nothing to find")
    return float(np.mean(list(shares.values())))


def _by_era(dicts: list[dict], eras: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for d, era in zip(dicts, eras):
        out.setdefault(era, []).append(d)
    return out


def panoptic_parts(found: np.ndarray, truth: CropLabels) -> dict:
    """The parts of panoptic quality of one crop.  The answer key's areas are every area of the key (`truth.area` above
    0: blocks and open land), the found areas those of the read (`found` above 0); px where the key has no label
    (`truth.street` 255) are left out of both.  A key area and a found area overlap by the px in both over the px in
    either, and MATCH when that is over one half: not a picked level, the least overlap at which an area can match at
    most one other, so matching needs no assignment step.  `matched` (pairs), `overlap_sum` (their overlaps summed),
    `found_unmatched`, `key_unmatched`; `panoptic_summed` makes the quality."""
    labelled = np.asarray(truth.street) != handlabels.NONE
    ka, fa = np.asarray(truth.area)[labelled].astype(np.int64), np.asarray(found)[labelled].astype(np.int64)
    nk, nf = int(ka.max(initial=0)) + 1, int(fa.max(initial=0)) + 1
    full = np.bincount(ka * nf + fa, minlength=nk * nf).reshape(nk, nf)       # row 0 / column 0: no area
    table, key_size, found_size = full[1:, 1:], full[1:].sum(axis=1), full[:, 1:].sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        overlap = table / (key_size[:, None] + found_size[None, :] - table)
    match = np.nan_to_num(overlap) > 0.5
    keys, founds = key_size > 0, found_size > 0
    return {"matched": int(match.sum()), "overlap_sum": float(overlap[match].sum()),
            "found_unmatched": int((founds & ~match.any(axis=0)).sum()),
            "key_unmatched": int((keys & ~match.any(axis=1)).sum())}


def panoptic_summed(dicts: list[dict]) -> dict:
    """`panoptic_parts` of several crops summed, with `quality` = overlap_sum / (matched + found_unmatched / 2 +
    key_unmatched / 2); None when that is 0 / 0 (no area found and none in the key)."""
    out = {k: sum(d[k] for d in dicts) for k in ("matched", "overlap_sum", "found_unmatched", "key_unmatched")}
    den = out["matched"] + out["found_unmatched"] / 2 + out["key_unmatched"] / 2
    out["quality"] = out["overlap_sum"] / den if den else None
    return out


def era_qualities(dicts_by_era: dict[str, list[dict]]) -> dict[str, float]:
    """Per era (its crops' `panoptic_parts` summed) the panoptic quality; an era with none to give left out."""
    out = {}
    for era in sorted(dicts_by_era):
        q = panoptic_summed(dicts_by_era[era])["quality"]
        if q is not None:
            out[era] = q
    return out


def panoptic_quality(dicts_by_era: dict[str, list[dict]]) -> float:
    """The mean over eras of the panoptic quality (`era_qualities`).  ValueError when no era has an area."""
    q = era_qualities(dicts_by_era)
    if not q:
        raise ValueError("no answer-key area and no found area in any crop: nothing to match")
    return float(np.mean(list(q.values())))


def fit_depth(reads: list[tuple[np.ndarray, np.ndarray]], truths: list[CropLabels],
              eras: list[str]) -> tuple[float, float, float]:
    """The depth among DEPTHS with the highest `panoptic_quality` over the crops (how well the found areas match the
    answer key's areas, counting stray areas and missed ones), the smallest of a tie; that quality; and the
    `found_share` at that depth.  `reads`: each crop's chances (2, h, w), as stored (`read.stored`), and its frame cut
    `keep` (h, w), on its box; `truths` and `eras` the crops' answer keys and map eras, in the same order."""
    if not (len(reads) == len(truths) == len(eras)) or not reads:
        raise ValueError(f"{len(reads)} reads, {len(truths)} truths, {len(eras)} eras: expected the same, at least 1")
    per_depth: list[list[tuple | None]] = [[None] * len(reads) for _ in DEPTHS]
    for k, ((chances, keep), truth) in enumerate(zip(reads, truths)):
        for which, found in areas.areas_by_depth(chances[model.STREET], chances[model.EDGE], keep, DEPTHS):
            c = (blocks(found, truth), panoptic_parts(found, truth))
            for i in which:
                per_depth[i][k] = c
    best, best_q = 0, -1.0
    for i, dicts in enumerate(per_depth):
        q = panoptic_quality(_by_era([p for _, p in dicts], eras))
        if q > best_q:
            best, best_q = i, q
    return DEPTHS[best], best_q, found_share(_by_era([b for b, _ in per_depth[best]], eras))


# --- the street and the block boundary -------------------------------------------------------------------------------

def ground(street_chance: np.ndarray, truth: CropLabels) -> dict:
    """Of the hand street px (street 1: `street_px`), how many the read calls street (its street chance over one half,
    the more likely answer: `street_called_street`); of the hand area px (street 0, block or open land: `area_px`), how
    many it calls not street (`area_called_not_street`).  Counts; `ground_summed` makes the shares."""
    called = np.asarray(street_chance) > 0.5
    on_street, on_area = truth.street == handlabels.STREET, truth.street == handlabels.AREA
    return {"street_px": int(on_street.sum()), "street_called_street": int(called[on_street].sum()),
            "area_px": int(on_area.sum()), "area_called_not_street": int((~called[on_area]).sum())}


def ground_summed(dicts: list[dict]) -> dict:
    """`ground` of several crops summed, with the shares `street_share` (street_called_street / street_px) and
    `area_share` (area_called_not_street / area_px); None for a share of no px."""
    out = {k: int(sum(d[k] for d in dicts)) for k in ("street_px", "street_called_street", "area_px",
                                                     "area_called_not_street")}
    out["street_share"] = out["street_called_street"] / out["street_px"] if out["street_px"] else None
    out["area_share"] = out["area_called_not_street"] / out["area_px"] if out["area_px"] else None
    return out


def boundary_distances(edge_chance: np.ndarray, truth: CropLabels, ft_per_px: float) -> np.ndarray | None:
    """Float64: from each hand block-boundary px (edge 1), in row order, the distance in feet to the nearest px the
    read calls a block edge (its block-edge chance over one half, the more likely answer); None when it calls no px of
    the crop an edge."""
    called = np.asarray(edge_chance) > 0.5
    if not called.any():
        return None
    return ndimage.distance_transform_edt(~called)[np.asarray(truth.edge) == 1] * float(ft_per_px)


def _percentiles(d: np.ndarray) -> dict:
    if not len(d):
        return {"median": None, "p90": None, "p99": None}
    return {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "p99": float(np.percentile(d, 99))}


def boundary_feet(edge_chance: np.ndarray, truth: CropLabels, ft_per_px: float) -> dict:
    """One half of the boundary measure, answer key to model: the answer key's block-boundary px (edge 1: `px`) and
    the median, 90th and 99th percentile (`median`, `p90`, `p99`, ft) of their `boundary_distances`.  When the read
    calls no px an edge, `no_model_edge` is True and the percentiles are None (no distance to give); likewise None for
    a crop with no answer-key boundary px.  It does not see a block edge the model draws where the answer key has
    none; the other half, `model_edge_feet`, does."""
    d = boundary_distances(edge_chance, truth, ft_per_px)
    px = int((np.asarray(truth.edge) == 1).sum())
    return {"px": px, "no_model_edge": d is None, **_percentiles(np.zeros(0) if d is None else d)}


def _pooled(distances: list[np.ndarray | None], px: list[int]) -> dict:
    """Several crops' distances together: `px` all their px, `px_measured` those of the crops that could be measured
    (distances not None), and the percentiles over every measured px together."""
    got = [d for d in distances if d is not None]
    d = np.concatenate(got) if got else np.zeros(0)
    return {"px": int(sum(px)), "px_measured": int(len(d)), **_percentiles(d)}


def boundary_feet_pooled(distances: list[np.ndarray | None], px: list[int]) -> dict:
    """`boundary_feet` of several crops together (`distances` each crop's `boundary_distances`, `px` its answer-key
    boundary px): `px` all of them, `px_measured` those on crops where the read calls an edge,
    `crops_with_no_model_edge`, and the percentiles over every measured px together."""
    return {**_pooled(distances, px), "crops_with_no_model_edge": sum(d is None for d in distances)}


def model_edge_distances(edge_chance: np.ndarray, truth: CropLabels, ft_per_px: float) -> np.ndarray | None:
    """Float64: from each px the read calls a block edge (its block-edge chance over one half, the more likely
    answer) on the crop's labelled ground (edge not 255), in row order, the distance in feet to the nearest answer-key
    boundary px (edge 1); None when the answer key has no boundary px or the read calls no such px an edge."""
    edge = np.asarray(truth.edge)
    key = edge == 1
    called = (np.asarray(edge_chance) > 0.5) & (edge != handlabels.NONE)
    if not key.any() or not called.any():
        return None
    return ndimage.distance_transform_edt(~key)[called] * float(ft_per_px)


def model_edge_feet(edge_chance: np.ndarray, truth: CropLabels, ft_per_px: float) -> dict:
    """The other half of the boundary measure (`boundary_feet` is the first), model to answer key: the px the read
    calls a block edge on the crop's labelled ground (`px`) and the median, 90th and 99th percentile (ft) of their
    `model_edge_distances`.  `no_model_edge`: the read calls no such px an edge; `no_answer_key_boundary`: the answer
    key has no boundary px; either way the percentiles are None (no distance to give)."""
    edge = np.asarray(truth.edge)
    px = int(((np.asarray(edge_chance) > 0.5) & (edge != handlabels.NONE)).sum())
    d = model_edge_distances(edge_chance, truth, ft_per_px)
    return {"px": px, "no_model_edge": px == 0, "no_answer_key_boundary": not (edge == 1).any(),
            **_percentiles(np.zeros(0) if d is None else d)}


def model_edge_feet_pooled(distances: list[np.ndarray | None], px: list[int]) -> dict:
    """`model_edge_feet` of several crops together (`distances` each crop's `model_edge_distances`, `px` its model
    edge px), as `boundary_feet_pooled`: `crops_not_measured` counts the crops with no model edge px or no answer-key
    boundary px."""
    return {**_pooled(distances, px), "crops_not_measured": sum(d is None for d in distances)}


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
    there is no such px or no side).  Unlike `boundary_feet`, which measures every hand boundary px against the px the
    block-edge answer calls an edge, this measures the closed areas (`find.areas`) and only the blocks they find."""
    found = np.asarray(found)
    mine = [i for i, s in block_status(found, truth).items() if s == "found"]
    on = (np.asarray(truth.edge) == 1) & np.isin(truth.area, mine)
    sides = _sides(found)
    out = {"px": int(on.sum()), "median": None, "p90": None}
    if on.any() and sides.any():
        d = ndimage.distance_transform_edt(~sides)[on] * ft_per_px
        out["median"], out["p90"] = float(np.median(d)), float(np.percentile(d, 90))
    return out


# --- reading the crops ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CropRead:
    """A hand crop ready to read: its scan and frame cut grown by model.REACH px where the page allows (`data.grown`),
    its box within them (`inner`), its era and its hand labels."""
    crop: handlabels.Crop
    era: str
    lum: np.ndarray
    keep: np.ndarray
    inner: tuple[slice, slice]
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
        outer, inner = data.grown(c.box, lum.shape)
        out.append(CropRead(c, pages.source(c.series).era, np.ascontiguousarray(lum[outer]),
                            np.ascontiguousarray(keep[outer]), inner, truth))
    return out


def read_one(net: model.FindNet, p: CropRead, device) -> tuple[np.ndarray, np.ndarray]:
    """The crop's chances on its box as stored (`read.stored`), and its frame cut there."""
    chances = read.read_page(net, p.lum, p.keep, device)[(slice(None), *p.inner)]
    return read.stored(chances), p.keep[p.inner]


def network(state_dict: dict) -> model.FindNet:
    """A FindNet holding `state_dict` (its width from the head), made without touching torch's generator.  ValueError,
    in one line, for a model saved with another number of answers (the three of before 2026-11-23)."""
    n = int(state_dict["head.weight"].shape[0])
    if n != len(model.ANSWERS):
        raise ValueError(f"the model gives {n} answers: it predates the two-answer finder (street and block edge); "
                         f"train it again with `sfzmaps find train`")
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


def chooser(crops: list[handlabels.Crop], device, test: list[handlabels.Crop] | None = None,
            placements=None) -> Callable[[int, dict], float]:
    """A checkpoint's score for the trainer's log (`train.fit`'s on_checkpoint; it chooses nothing): every crop read
    with the checkpoint's weights, and `fit_depth`'s share.  The crops are read from disk once, here.  ValueError
    when there is no crop, one is not in the `choose` set, or one lies on the ground of a `test` crop (`test`, None:
    every test crop; `handlabels.refuse_choose_on_test` with `placements`)."""
    _only(crops, "choose")
    handlabels.refuse_choose_on_test(crops, handlabels.crops(("test",)) if test is None else test, placements)
    ready = prepare(crops)

    def score(step: int, state_dict: dict) -> float:
        t0 = time.perf_counter()
        net = network(state_dict)
        reads = [read_one(net, p, device) for p in ready]
        depth, quality, share = fit_depth(reads, [p.truth for p in ready], [p.era for p in ready])
        log.info("step %d: the choosing crops find %.3f of their blocks (mean over eras) at depth %d/255, panoptic "
                 "quality %.3f; read and scored in %.1f s", step, share, round(depth * read.LEVELS), quality,
                 time.perf_counter() - t0)
        return share

    return score


# --- a run's counts ---------------------------------------------------------------------------------------------------


def crop_file_name(c: handlabels.Crop) -> str:
    """<series>__<crop file stem>: a test crop's read and picture are named so (as `look.write_crop` names a crop)."""
    return f"{c.series}__{c.path.stem}"


def count(net: model.FindNet, choose: list[handlabels.Crop], test: list[handlabels.Crop], device, folder: Path,
          placements=None) -> dict:
    """The counts of a run's kept model `net` (see the module): the depth fitted on the `choose` crops, then every
    `test` crop read at it; each test crop's read saved to <folder>/reads/test/<series>__<crop file stem>.npz and its
    picture to <folder>/look/test/<series>__<crop file stem>.png.  Returns what counts.json holds (no clock time, no
    absolute path), each measure per crop, per era and pooled.  ValueError when a set has no crop or holds a crop of
    another set, or a choosing crop lies on a test crop's ground (`handlabels.refuse_choose_on_test`, with
    `placements`)."""
    _only(choose, "choose")
    _only(test, "test")
    handlabels.refuse_choose_on_test(choose, test, placements)
    ready = prepare(choose)
    reads = [read_one(net, p, device) for p in ready]
    depth, quality, share = fit_depth(reads, [p.truth for p in ready], [p.era for p in ready])
    at_depth = [areas.areas(c[model.STREET], c[model.EDGE], k, depth) for c, k in reads]
    choose_eras = [p.era for p in ready]
    out = {"depth": depth, "depth_level": round(depth * read.LEVELS),
           "choose": {"panoptic_quality": quality,
                      "era_panoptic": era_qualities(_by_era([panoptic_parts(f, p.truth) for f, p in
                                                             zip(at_depth, ready)], choose_eras)),
                      "found_share": share,
                      "era_shares": era_shares(_by_era([blocks(f, p.truth) for f, p in zip(at_depth, ready)],
                                                       choose_eras)),
                      "crops": [data.crop_name(p.crop) for p in ready]}}
    rows, distances, model_distances = [], [], []
    for c in test:
        p = prepare([c])[0]
        sid = (c.series, c.sheet)
        chances, keep = read_one(net, p, device)
        name = crop_file_name(c)
        stored = Path("reads") / "test" / f"{name}.npz"
        read.save(folder / stored, chances)
        found = areas.areas(chances[model.STREET], chances[model.EDGE], keep, depth)
        status = block_status(found, p.truth)
        got = blocks(found, p.truth)
        scale = lotlabels.ft_per_px(lotlabels.placement(sid), c.box)
        d = boundary_distances(chances[model.EDGE], p.truth, scale)
        distances.append(d)
        bf = boundary_feet(chances[model.EDGE], p.truth, scale)
        model_distances.append(model_edge_distances(chances[model.EDGE], p.truth, scale))
        mf = model_edge_feet(chances[model.EDGE], p.truth, scale)
        held = whole(found, p.truth)
        picture = Path("look") / "test" / f"{name}.png"
        (folder / picture).parent.mkdir(parents=True, exist_ok=True)
        x0, y0, w, h = c.box
        title = (f"test / {c.series} / {c.sheet}  page x {x0}-{x0 + w}, y {y0}-{y0 + h}; {p.era}; depth "
                 f"{out['depth_level']}/255; blocks {got['blocks']}: found {got['found']}, whole {held['whole']}, "
                 f"joined {got['joined']}, cut up {got['cut_up']}, lost {got['lost']}; extra areas {got['extra']}")
        cv2.imwrite(str(folder / picture), look.read_picture(p.lum[p.inner], chances, found, p.truth, status, title))
        parts = panoptic_parts(found, p.truth)
        rows.append({"name": data.crop_name(c), "era": p.era, "counts": got, "whole": held,
                     "panoptic": {**parts, "quality": panoptic_summed([parts])["quality"]},
                     "ground": ground(chances[model.STREET], p.truth), "boundary_feet": bf, "model_edge_feet": mf,
                     "edge_feet": edge_feet(found, p.truth, scale), "ft_per_px": scale, "read": str(stored),
                     "picture": str(picture)})
        log.info("%s: %s", data.crop_name(c), got)
    eras = [r["era"] for r in rows]
    by_era = _by_era([r["counts"] for r in rows], eras)
    ground_by_era = _by_era([r["ground"] for r in rows], eras)
    whole_by_era = _by_era([r["whole"] for r in rows], eras)
    feet_by_era = _by_era([(d, r["boundary_feet"]["px"]) for d, r in zip(distances, rows)], eras)
    feet = {e: boundary_feet_pooled([d for d, _ in v], [n for _, n in v]) for e, v in sorted(feet_by_era.items())}
    model_by_era = _by_era([(d, r["model_edge_feet"]["px"]) for d, r in zip(model_distances, rows)], eras)
    model_feet = {e: model_edge_feet_pooled([d for d, _ in v], [n for _, n in v])
                  for e, v in sorted(model_by_era.items())}
    panoptic_by_era = _by_era([r["panoptic"] for r in rows], eras)
    out["test"] = {"panoptic": {"mean_over_eras": panoptic_quality(panoptic_by_era),
                                "eras": {e: panoptic_summed(d) for e, d in sorted(panoptic_by_era.items())},
                                "pooled": panoptic_summed([r["panoptic"] for r in rows])},
                   "found_share": found_share(by_era), "era_shares": era_shares(by_era),
                   "eras": {e: _summed(d) for e, d in sorted(by_era.items())},
                   "pooled": _summed([r["counts"] for r in rows]),
                   "whole": {"eras": {e: whole_summed(d) for e, d in sorted(whole_by_era.items())},
                             "pooled": whole_summed([r["whole"] for r in rows])},
                   "ground": {"eras": {e: ground_summed(d) for e, d in sorted(ground_by_era.items())},
                              "pooled": ground_summed([r["ground"] for r in rows])},
                   "boundary_feet": {"eras": feet,
                                     "pooled": boundary_feet_pooled(distances, [r["boundary_feet"]["px"]
                                                                               for r in rows])},
                   "model_edge_feet": {"eras": model_feet,
                                       "pooled": model_edge_feet_pooled(model_distances, [r["model_edge_feet"]["px"]
                                                                                          for r in rows])},
                   "crops": rows}
    return out
