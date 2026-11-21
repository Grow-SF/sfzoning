"""The frame finder's run (issue #71; design spec amendment 3, "the combined design"): the layout's held-out chances,
the temperature, the shape decision's three numbers, the scores, the final heads and every map page's outputs.  It
orchestrates only: the layout is `layout`, the decision `shape` (per-pixel: `decide`), the scores `scores` and the
temperature `calibrate`.

`held_out` scores every labelled sheet with layout heads that never saw it:
1. `layout.extract_all` over every map page (and the labelled sheets): the frozen backbone's reduced features, on the
   caller's device (`extract_features`: the CLI runs it first, alone in the GPU's turn, so here it finds them);
2. leave one sheet out: for each labelled sheet and each of `layout.HEAD_SEEDS`, a head trained on every other sheet
   (`leave_one_out`; cpu, on the process pool), giving the sheet's patch logits per seed.  Each is upsampled to the
   page (`layout.page_logits`) and the seeds' tempered softmaxes are averaged (`layout.combine`).  One temperature is
   fitted on ink-weighted samples of every sheet's upsampled logits (`calibrate`);
3. the shape decision's numbers (`shape.Params`) are chosen by the lowest pooled realised cost (summed over the held
   sheets) in a coordinate-wise search over `GRID` from `START` (controller ruling C8; the full grid is 48 points
   and hours): each number in turn takes each of its GRID values with the others fixed, the lowest pooled cost is
   kept (ties keep the current value, then the earlier GRID value), and rounds repeat until one changes nothing, at
   most `SEARCH_ROUNDS`.  Every visited point and its pooled cost go to the summary.  The block bounds a held sheet
   is decided with are fitted on the other sheets' labels (`blocks.block_bounds`), as the heads are;
4. per sheet, the scores of the per-pixel cut on the layout chances (`decide.decide`) and of the shape decision at
   the chosen numbers (its outputs are written), the frame ink the chances still see along each fitted side
   (`shape.side_ink_left`), pooled means with sheet-bootstrap intervals, pooled ink shares, and paired differences
   against `BASELINE` where its files exist;
5. everything goes to `folds.json`.

The page work runs on `PROCESSES` spawned worker processes, each pinned to one maths thread (heads: torch takes
`layout.HEAD_THREADS`).  A job carries everything it needs (the page's path, its patch logits, the temperature,
the label folder), and its worker loads the page, makes the chances and lets them go when the job ends: so one
page's chances and shape cache (`shape.decide`'s `cache`) live in one worker at a time, and a search job runs every
parameter point it needs on that one cache.  `PROCESSES = 1` runs everything in this process.

`train_final` trains `layout.HEAD_SEEDS` heads on every labelled sheet; `predict_all` decides every other map page
with them at the held-out run's temperature and numbers; the labelled sheets keep their held-out outputs.

Outputs, all under `out` (default `data/frame`):
- `reducer.npz` and `features/<series>/<sheet>.npy`: the layout's features (`layout.extract_all`);
- `<series>/<sheet>/outside.png` (`OUTSIDE_PNG`): 1-bit, the deletion (`Decision.deleted`: outside the frame or a
  removed block);
- `<series>/<sheet>/outside_prob.npz` (`PROB_FILE`): key `q`, the layout's chance of deletion p1 + p2 as uint8
  levels round(255 q), for reading and placement to weigh ink by;
- `<series>/<sheet>/erased.png` (`ERASED_PNG`): the scan in its own mode (1-bit as 8-bit grey) with the deleted px
  painted plain paper, the per-channel median of the kept px: for looking only;
- `<series>/<sheet>/outline.json` (`OUTLINE_FILE`): {"polygons": the kept region (`decide.outline`), "maps": n,
  "shape": the fitted frame, notches and blocks, or null where the per-pixel cut decided (a page of several maps)};
- `folds.json` (`FOLDS_FILE`): the held-out run, deleted when a run starts, so a run that stops part way leaves no
  summary for the final training to trust;
- `model/head-<seed>.pt` and `model/model.json` (`MODEL_DIR`): the final heads, the numbers, the temperature, the
  block bounds and the reducer they were trained on (its sha256);
- `pages.json` (`PAGES_FILE`): every map page's side ink left and dropped ink, with both ranks, the pages to look at.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import time
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sfzmaps.frame import blocks as blk
from sfzmaps.frame import calibrate, data, decide, layout, scores, shape
from sfzmaps.frame.data import DATA_DIR, OUT, SheetId
from sfzmaps.solver import threads

log = logging.getLogger(__name__)

# the search grid from the spike (computational), and the coordinate search's start and round limit (ruling C8)
GRID = {"lam_notch": (0, 1e3, 1e4, 1e5), "beta": (0, 1, 3, 10), "lam_block": (0, 1e3, 1e4)}
START = shape.Params(lam_notch=1e4, beta=3, lam_block=1e3)
SEARCH_ROUNDS = 3
PROCESSES = 4           # computational: worker processes for the page decisions and the heads
THREADS = 4             # computational: torch threads in this process (upsampling, head logits)
MIN_FREE_GB = 20        # computational safety margin: `predict_all` refuses with less free disk under `out`
BASELINE = DATA_DIR / "frame-cmp-none"   # the old two-pass model's held-out outputs (spec part 7)

FOLDS_FILE = "folds.json"
MODEL_DIR = "model"
REDUCER_FILE = "reducer.npz"          # `layout.extract_all`'s file
OUTSIDE_PNG = "outside.png"
PROB_FILE = "outside_prob.npz"
ERASED_PNG = "erased.png"
OUTLINE_FILE = "outline.json"
PAGES_FILE = "pages.json"
BASELINE_FILE = "map_prob.npz"      # the two-pass run's file name
COMPARED = ("frame_kept", "map_cut", "cost")   # the scores compared with the baseline
PAIRED = ("frame_kept", "map_cut", "cost", "cost_layout", "cost_edge")   # shape - per-pixel, paired


class DiskFull(RuntimeError):
    """Less than `MIN_FREE_GB` free where the outputs go."""


def sheet_dir(out: Path, sid: SheetId) -> Path:
    return out / sid[0] / sid[1]


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lum(sid: SheetId) -> np.ndarray:
    return data.load_luminance(data.page_path(*sid, work=data.WORK))


def _cls(sid: SheetId, shape_: tuple[int, int], root: Path | None = None) -> np.ndarray:
    root = data.LABELS if root is None else root
    return data.class_mask(data.load_label(*sid, root=root), data.load_remove(*sid, root=root), shape_)


# ---------------------------------------------------------------------------------------------------------------
# The process pool
# ---------------------------------------------------------------------------------------------------------------

_STATE: dict = {}   # a worker's shared inputs, set once per worker (`pool_map`'s `state`)


def _init_worker(state: dict) -> None:
    """A spawned worker: one thread for OpenCV and torch (BLAS is pinned by the environment it started with), its
    warnings shown, and the shared inputs."""
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    _STATE.clear()
    _STATE.update(state)


@contextlib.contextmanager
def _pinned_environment():
    """`threads.ONE_THREAD` (and its quiet pools) in this process's environment while it lasts, so the workers it
    spawns start pinned; this process's own libraries read theirs when they loaded, and keep them."""
    pins = {**threads.ONE_THREAD, **threads.QUIET_POOLS}
    saved = {k: os.environ.get(k) for k in pins}
    os.environ.update(pins)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _apply(fn: Callable, args: tuple, item):
    return fn(item, *args)


def pool_map(fn: Callable, items: Sequence, *args, state: dict | None = None, label: str = "") -> list:
    """`[fn(item, *args) for item in items]`, in order, on up to `PROCESSES` spawned workers (here when
    `PROCESSES` <= 1 or there is one item).  `fn` must be a module-level function; `state` is given to every worker
    once (`_STATE`).  With a `label`, each result is logged as it arrives."""
    items = list(items)
    state = state or {}
    n = len(items)
    t0 = time.perf_counter()

    def note(i: int) -> None:
        if label:
            log.info("%s %d/%d (%.0f s)", label, i, n, time.perf_counter() - t0)

    if PROCESSES <= 1 or n <= 1:
        _STATE.clear()
        _STATE.update(state)
        try:
            out = []
            for i, item in enumerate(items, 1):
                out.append(fn(item, *args))
                note(i)
            return out
        finally:
            _STATE.clear()
    with _pinned_environment(), concurrent.futures.ProcessPoolExecutor(
            min(PROCESSES, n), mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker,
            initargs=(state,)) as ex:
        out = []
        for i, r in enumerate(ex.map(partial(_apply, fn, args), items), 1):
            out.append(r)
            note(i)
        return out


# ---------------------------------------------------------------------------------------------------------------
# The layout: features, heads, chances
# ---------------------------------------------------------------------------------------------------------------


def feature_pages(labelled: Sequence[SheetId]) -> list[SheetId]:
    """The pages the reducer is fitted over: every map page and the labelled sheets, sorted.  The held-out run, the
    final training and the prediction use the same list, so the features are extracted once."""
    return sorted(set(data.map_pages(data.PAGE_KINDS, data.WORK)) | set(labelled))


def extract_features(labelled: Sequence[SheetId], device: str, out: Path = OUT) -> None:
    """`layout.extract_all` over `feature_pages(labelled)` on `device` (a no-op when they are all there): the only
    GPU work of the run, so the caller can hold the GPU's turn for this alone."""
    layout.extract_all(feature_pages(labelled), device, out)


def load_features(out: Path, sid: SheetId) -> np.ndarray:
    return np.load(layout.features_dir(out) / sid[0] / f"{sid[1]}.npy")


def _features_and_targets(labelled: Sequence[SheetId], out: Path) -> tuple[dict, dict]:
    """Each labelled sheet's features and patch targets (`layout.patch_targets`), in sorted order."""
    feats, targets = {}, {}
    for sid in labelled:
        f = load_features(out, sid)
        lum = _lum(sid)
        g = layout.grid(lum.shape)
        if f.shape[1:] != g:
            raise ValueError(f"{sid[0]}/{sid[1]}: features are {f.shape[1:]}, the page's grid {g}: extract again")
        feats[sid] = f
        targets[sid] = layout.patch_targets(_cls(sid, lum.shape), lum, g)
    return feats, targets


def _head_job(job: tuple[SheetId | None, int]) -> tuple:
    """One head of `layout.train_head` on every sheet in the worker's `feats` but `held` (all of them when `held` is
    None): (held, seed, the held sheet's patch logits or, for None, the head's weights as arrays, seconds)."""
    held, seed = job
    feats, targets = _STATE["feats"], _STATE["targets"]
    rest = [s for s in feats if s != held]
    t0 = time.perf_counter()
    head = layout.train_head([feats[s] for s in rest], [targets[s] for s in rest], seed)
    seconds = time.perf_counter() - t0
    if held is None:
        return None, seed, {k: v.detach().cpu().numpy().copy() for k, v in head.state_dict().items()}, seconds
    return held, seed, layout.head_logits(head, feats[held]), seconds


def leave_one_out(feats: dict, targets: dict, seeds: Sequence[int]) -> tuple[dict, list[dict]]:
    """`layout.leave_one_out` on the process pool: for each sheet, the patch logits of heads trained on every other
    sheet, one per seed (in `seeds` order); and each head's training time."""
    seeds = list(seeds)
    jobs = [(sid, s) for sid in feats for s in seeds]
    done = pool_map(_head_job, jobs, state={"feats": feats, "targets": targets}, label="held-out head")
    logits = {sid: [None] * len(seeds) for sid in feats}
    seconds = []
    for held, seed, z, sec in done:
        logits[held][seeds.index(seed)] = z
        seconds.append({"series": held[0], "sheet": held[1], "seed": seed, "seconds": sec})
        log.info("head without %s/%s, seed %d: %.1f s", *held, seed, sec)
    return logits, seconds


def chances(zs: Sequence[np.ndarray], shape_: tuple[int, int], temperature: float) -> np.ndarray:
    """`layout.combine` of the patch logits `zs` upsampled to the page (`layout.page_logits`), the same sums in the
    same order, but one upsampled logits array alive at a time."""
    total = None
    for z in zs:
        q = layout.combine([layout.page_logits(z, shape_)], temperature)
        if total is None:
            total = q
        else:
            total += q
    total /= np.float32(len(zs))
    return total


def sample_patch_logits(zs: Sequence[np.ndarray], cls: np.ndarray, lum: np.ndarray, rng,
                        n: int) -> tuple[np.ndarray, np.ndarray]:
    """`calibrate.sample` of the patch logits `zs` upsampled to the page, the same px, but one upsampled seed alive at
    a time."""
    idx = calibrate.draw(lum, rng, n)
    z = np.stack([layout.page_logits(a, lum.shape).reshape(a.shape[0], -1)[:, idx] for a in zs]).astype(np.float32)
    return z, np.asarray(cls).ravel()[idx].astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------
# A page's work, in a worker
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class PageJob:
    """What a worker needs for one page: its id and scan, each head's patch logits, the temperature, the number of
    maps on it, the block bounds, and the label folder (labelled sheets: to score)."""

    sid: SheetId
    page: Path
    zs: tuple
    temperature: float
    maps: int
    bounds: blk.BlockBounds
    labels: Path | None = None


def _load(job: PageJob) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    lum = data.load_luminance(job.page)
    p = chances(job.zs, lum.shape, job.temperature)
    cls = _cls(job.sid, lum.shape, job.labels) if job.labels is not None else None
    return lum, p, cls


def search_page(job: PageJob, points: Sequence[shape.Params]) -> list[dict]:
    """The shape decision's realised cost (`scores.realised`) on one labelled page at each of `points`, on one cache."""
    lum, p, cls = _load(job)
    cache: dict = {}
    out = []
    for pt in points:
        dec, *_ = shape.decide(p, lum, pt, job.bounds, job.maps, cache=cache)
        out.append(scores.realised(dec.deleted, cls, lum))
    return out


def _shape_json(fr, ns, bs) -> dict:
    """The fitted frame, notches and blocks as JSON (page px and degrees; see `geometry.Frame`, `notches.Notch`,
    `blocks.Block`)."""
    frame = {"phi": fr.phi, "offset": fr.offset, "tilt": fr.tilt, "bow": fr.bow,
             "arc": {"-".join(c): r for c, r in fr.arc.items()}}
    return {"frame": frame, "notches": [dataclasses.asdict(n) for n in ns],
            "blocks": [dataclasses.asdict(b) for b in bs]}


def _baseline_scores(job: PageJob, lum: np.ndarray, cls: np.ndarray, baseline: Path | None) -> dict | None:
    """The old two-pass model's chances for the page under `baseline` (key `map`, P(inside), read as (map, 1 - map,
    0)), decided and scored as the new ones with no temperature: `COMPARED`; None where it has no file."""
    if baseline is None:
        return None
    f = sheet_dir(baseline, job.sid) / BASELINE_FILE
    if not f.exists():
        return None
    with np.load(f) as z:
        inside = z["map"].astype(np.float32)
    if inside.shape != lum.shape:
        log.warning("%s/%s: baseline is %s, the page %s; left out", *job.sid, inside.shape, lum.shape)
        return None
    p = np.stack([inside, 1.0 - inside, np.zeros_like(inside)])
    s = scores.sheet_scores(p, decide.decide(p, lum, job.maps).deleted, cls, lum)
    return {k: s[k] for k in COMPARED}


def final_page(job: PageJob, params: shape.Params, out: Path, baseline: Path | None) -> dict:
    """A labelled page at the chosen numbers: the shape decision (its outputs written) and the per-pixel cut, both
    scored, the side ink left, and the baseline's scores (or None)."""
    lum, p, cls = _load(job)
    pix = decide.decide(p, lum, job.maps)
    dec, fr, ns, bs = shape.decide(p, lum, params, job.bounds, job.maps)
    write_outputs(out, job.sid, p, dec, job.maps, page=job.page, fitted=(fr, ns, bs) if job.maps == 1 else None)
    side = shape.side_ink_left(p, lum, fr)
    return {"series": job.sid[0], "sheet": job.sid[1], "maps": job.maps,
            "scores": scores.sheet_scores(p, dec.deleted, cls, lum),
            "pixel": scores.sheet_scores(p, pix.deleted, cls, lum),
            "side_ink_left": side, "side_ink_left_total": float(sum(side.values())),
            "filled_ink": dec.filled_ink, "dropped_ink": dec.dropped_ink, "removed_ink": dec.removed_ink,
            "notches": len(ns), "blocks": len(bs), "phi": fr.phi,
            "bounds": dataclasses.asdict(job.bounds),
            "baseline": _baseline_scores(job, lum, cls, baseline)}


def predict_page(job: PageJob, params: shape.Params, out: Path) -> dict:
    """An unlabelled page: the shape decision, its outputs written; its ink figures and the side ink left."""
    lum, p, _ = _load(job)
    dec, fr, ns, bs = shape.decide(p, lum, params, job.bounds, job.maps)
    write_outputs(out, job.sid, p, dec, job.maps, page=job.page, fitted=(fr, ns, bs) if job.maps == 1 else None)
    side = shape.side_ink_left(p, lum, fr)
    return {"filled_ink": dec.filled_ink, "dropped_ink": dec.dropped_ink, "removed_ink": dec.removed_ink,
            "side_ink_left": side, "side_ink_left_total": float(sum(side.values())), "notches": len(ns),
            "blocks": len(bs)}


def write_outputs(out: Path, sid: SheetId, p: np.ndarray, decision: decide.Decision, maps: int,
                  page: Path | None = None, fitted: tuple | None = None) -> None:
    """The page's four files (see the module): `p` its (3, h, w) chances, `decision` the cut made from them, `page`
    its scan (default: under `data.WORK`), `fitted` the shape decision's (frame, notches, blocks) or None."""
    d = sheet_dir(out, sid)
    d.mkdir(parents=True, exist_ok=True)
    deleted = decision.deleted
    page = data.page_path(*sid, work=data.WORK) if page is None else page

    def replace(name: str, write) -> None:
        tmp = d / (name + ".tmp")
        write(tmp)
        os.replace(tmp, d / name)

    replace(OUTSIDE_PNG, lambda f: Image.fromarray(deleted).save(f, format="PNG"))
    q = np.round(np.clip(p[1] + p[2], 0.0, 1.0) * 255.0).astype(np.uint8)

    def save_q(f: Path) -> None:
        with open(f, "wb") as fh:
            np.savez_compressed(fh, q=q)
    replace(PROB_FILE, save_q)

    with Image.open(page) as im:
        # 1-bit as 8-bit grey (paper's median may not be 0 or 1); a palette has no median of its indices
        a = np.array(im.convert({"1": "L", "P": "RGB"}.get(im.mode, im.mode)))
    if a.shape[:2] != deleted.shape:
        raise ValueError(f"{sid}: page is {a.shape[:2]}, the decision {deleted.shape}")
    kept = a[~deleted] if not deleted.all() else a.reshape(-1, *a.shape[2:])   # nothing kept: the whole page's
    a[deleted] = np.round(np.median(kept, axis=0)).astype(a.dtype)
    replace(ERASED_PNG, lambda f: Image.fromarray(a).save(f, format="PNG"))

    outline = {"polygons": decide.outline(deleted), "maps": maps,
               "shape": _shape_json(*fitted) if fitted is not None else None}
    replace(OUTLINE_FILE, lambda f: f.write_text(json.dumps(outline, default=_json_default)))


# ---------------------------------------------------------------------------------------------------------------
# The held-out run
# ---------------------------------------------------------------------------------------------------------------


def _interval(t: tuple[float, float, float]) -> dict:
    return {"mean": t[0], "lo": t[1], "hi": t[2]}


def search_params(jobs: Sequence[PageJob]) -> tuple[shape.Params, list[dict], int]:
    """The coordinate search (see the module) on the held-out `jobs`: (the chosen numbers, every visited point with
    its pooled `frame_kept`, `map_cut` and `cost` in visiting order, the rounds run)."""
    visited: dict[shape.Params, dict] = {}

    def evaluate(points: list[shape.Params]) -> None:
        need = [pt for pt in dict.fromkeys(points) if pt not in visited]
        if not need:
            return
        per_page = pool_map(search_page, jobs, tuple(need), label=f"search, {len(need)} points per page")
        for i, pt in enumerate(need):
            rows = [r[i] for r in per_page]
            visited[pt] = {**dataclasses.asdict(pt), **{k: float(sum(r[k] for r in rows))
                                                        for k in ("frame_kept", "map_cut", "cost")}}
            log.info("lambda_notch %g, beta %g, lambda_block %g: pooled cost %.1f", pt.lam_notch, pt.beta,
                     pt.lam_block, visited[pt]["cost"])

    current, rounds = START, 0
    for _ in range(SEARCH_ROUNDS):
        rounds += 1
        changed = False
        for name, values in GRID.items():
            points = [dataclasses.replace(current, **{name: v}) for v in values]
            evaluate([current, *points])
            best = current
            for pt in points:
                if visited[pt]["cost"] < visited[best]["cost"]:
                    best = pt
            if best != current:
                current, changed = best, True
        if not changed:
            break
    return current, list(visited.values()), rounds


def _pooled(rows: list[dict]) -> dict:
    return {"means": {k: _interval(scores.bootstrap_mean([r[k] for r in rows], scores.BOOT_N)) for k in rows[0]},
            "shares": scores.pooled_shares(rows)}


def _paired(a: dict[SheetId, dict], b: dict[SheetId, dict], keys: Sequence[str]) -> dict:
    return {k: _interval(scores.paired_difference({s: v[k] for s, v in a.items()}, {s: v[k] for s, v in b.items()},
                                                  scores.BOOT_N)) for k in keys}


def held_out(labelled: Sequence[SheetId], device: str, out: Path = OUT) -> dict:
    """Score every labelled sheet held out (see the module); returns what it writes to `out/folds.json`.  Missing
    features are extracted on `device` (the CLI does that first, in the GPU's turn); the rest runs on cpu."""
    labelled = sorted(set(labelled))
    if len(labelled) < 2:
        raise ValueError(f"{len(labelled)} labelled sheet(s): leaving one out needs at least 2")
    (out / FOLDS_FILE).unlink(missing_ok=True)
    torch.set_num_threads(THREADS)
    counts = data.map_counts(data.PAGE_KINDS)
    extract_features(labelled, device, out)
    feats, targets = _features_and_targets(labelled, out)

    seeds = list(layout.HEAD_SEEDS)
    loo, head_seconds = leave_one_out(feats, targets, seeds)

    rng = np.random.default_rng(0)            # one generator for the run, sheets in sorted order
    samples = []
    for sid in labelled:
        lum = _lum(sid)
        samples.append(sample_patch_logits(loo[sid], _cls(sid, lum.shape), lum, rng, calibrate.N_CAL))
    temperature = calibrate.fit_temperature(samples)
    t_lo, t_hi = calibrate.temperature_interval(samples, calibrate.CAL_BOOT)
    reliability = calibrate.reliability(samples, temperature)
    del samples
    log.info("temperature %.3f (%.3f-%.3f)", temperature, t_lo, t_hi)

    jobs = [PageJob(sid=sid, page=data.page_path(*sid, work=data.WORK), zs=tuple(loo[sid]), temperature=temperature,
                    maps=counts[sid], labels=data.LABELS,
                    bounds=blk.block_bounds([s for s in labelled if s != sid], root=data.LABELS))
            for sid in labelled]
    params, search, rounds = search_params(jobs)
    log.info("numbers: lambda_notch %g, beta %g, lambda_block %g (%d points, %d rounds)", params.lam_notch,
             params.beta, params.lam_block, len(search), rounds)

    rows = pool_map(final_page, jobs, params, out, BASELINE, label="held-out sheet")
    old = {}
    sheets = []
    for r in rows:
        sid = (r["series"], r["sheet"])
        b = r.pop("baseline")
        if b is not None:
            old[sid] = b
        sheets.append(r)
        sc, px = r["scores"], r["pixel"]
        log.info("%s/%s: shape left in %s, cut %s, cost %.1f; per-pixel left in %s, cut %s, cost %.1f", *sid,
                 scores.percent(sc["frame_kept_share"]), scores.percent(sc["map_cut_share"]), sc["cost"],
                 scores.percent(px["frame_kept_share"]), scores.percent(px["map_cut_share"]), px["cost"])
    new = {(r["series"], r["sheet"]): r["scores"] for r in sheets}
    pixel = {(r["series"], r["sheet"]): r["pixel"] for r in sheets}
    result = {
        "params": dataclasses.asdict(params),
        "grid": {k: list(v) for k, v in GRID.items()},
        "start": dataclasses.asdict(START),
        "search": search,
        "rounds": rounds,
        "head_seeds": seeds,
        "head_seconds": head_seconds,
        "temperature": temperature,
        "temperature_interval": [t_lo, t_hi],
        "reliability": reliability,
        "sheets": sheets,
        "pooled": {"shape": _pooled(list(new.values())), "pixel": _pooled(list(pixel.values()))},
        "shape_vs_pixel": _paired(new, pixel, PAIRED),
        "side_ink_left": _interval(scores.bootstrap_mean([r["side_ink_left_total"] for r in sheets],
                                                         scores.BOOT_N)),
        "baseline": {"dir": str(BASELINE),
                     "sheets": [{"series": s, "sheet": h, **v} for (s, h), v in sorted(old.items())],
                     "paired": _paired({s: new[s] for s in old}, old, COMPARED) if old else {}},
    }
    _write_json(out / FOLDS_FILE, result)
    return result


# ---------------------------------------------------------------------------------------------------------------
# The final heads and every page
# ---------------------------------------------------------------------------------------------------------------


def train_final(labelled: Sequence[SheetId], device: str, out: Path, held: dict) -> None:
    """`layout.HEAD_SEEDS` heads trained on every labelled sheet, saved to `out/model/` with the held-out run's
    numbers and temperature (`held`, its `folds.json`), the block bounds of every label and the reducer's sha256;
    `model.json` is deleted first and written last, so a run that stops part way leaves no model to load.  The
    features are extracted first if they are missing (on `device`; the CLI extracts them first, in the GPU's turn)."""
    labelled = sorted(set(labelled))
    d = out / MODEL_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.json").unlink(missing_ok=True)
    torch.set_num_threads(THREADS)
    extract_features(labelled, device, out)
    feats, targets = _features_and_targets(labelled, out)
    seeds = list(layout.HEAD_SEEDS)
    done = pool_map(_head_job, [(None, s) for s in seeds], state={"feats": feats, "targets": targets},
                    label="final head")
    seconds = []
    for _, seed, weights, sec in done:
        log.info("final head, seed %d: %.1f s", seed, sec)
        seconds.append({"seed": seed, "seconds": sec})
        tmp = d / f"head-{seed}.pt.tmp"
        torch.save({k: torch.from_numpy(v) for k, v in weights.items()}, tmp)
        os.replace(tmp, d / f"head-{seed}.pt")
    bounds = blk.block_bounds(labelled, root=data.LABELS)
    _write_json(d / "model.json", {
        "head_seeds": seeds, "head_seconds": seconds, "components": layout.COMPONENTS,
        "params": {k: held["params"][k] for k in ("lam_notch", "beta", "lam_block")},
        "temperature": float(held["temperature"]), "bounds": dataclasses.asdict(bounds),
        "reducer": {"file": REDUCER_FILE, "sha256": _sha256(out / REDUCER_FILE)},
        "sheets": [list(s) for s in labelled], "level": decide.cut_level()})


def load_model(out: Path) -> tuple[list, dict]:
    """The final heads (eval, cpu) and `model.json`, from `out/model/`."""
    d = out / MODEL_DIR
    if not (d / "model.json").exists():
        raise FileNotFoundError(f"{d / 'model.json'} is missing: train the final model first (`sfzmaps frame`)")
    meta = json.loads((d / "model.json").read_text())
    if "head_seeds" not in meta:
        raise FileNotFoundError(f"{d / 'model.json'} is the old full-resolution model's: train the final heads "
                                "first (`sfzmaps frame`)")
    heads = []
    for s in meta["head_seeds"]:
        head = layout.new_head()
        head.load_state_dict(torch.load(d / f"head-{s}.pt", map_location="cpu", weights_only=True))
        heads.append(head.eval())
    return heads, meta


def check_disk(out: Path) -> None:
    """Raise `DiskFull` with less than `MIN_FREE_GB` free under `out` (or its nearest existing parent)."""
    where = out
    while not where.exists() and where != where.parent:
        where = where.parent
    free = shutil.disk_usage(where).free
    if free < MIN_FREE_GB * 1e9:
        raise DiskFull(f"only {free / 1e9:.1f} GB free under {where}; predicting every page needs {MIN_FREE_GB} GB "
                       "free (a safety margin): free some space first")


def rank_pages(rows: list[dict]) -> list[dict]:
    """The rows ordered by `side_ink_left_total`, highest first (ties: more `dropped_ink` first, then as given),
    each with its `side_rank` and its `dropped_rank` (by `dropped_ink`, highest first), both from 1."""
    rows = sorted(rows, key=lambda r: (-r["side_ink_left_total"], -r["dropped_ink"]))
    for i, r in enumerate(rows, 1):
        r["side_rank"] = i
    for i, r in enumerate(sorted(rows, key=lambda r: -r["dropped_ink"]), 1):
        r["dropped_rank"] = i
    return rows


def predict_all(device: str, out: Path = OUT) -> None:
    """Every map page that is not labelled, decided with the final heads at the held-out numbers; the labelled sheets
    keep their held-out outputs.  Then `PAGES_FILE` (`rank_pages`; the labelled sheets' rows from `folds.json`,
    marked `held_out`).  Refuses (`DiskFull`) with less than `MIN_FREE_GB` free, and (RuntimeError) when the reducer
    is not the one the heads were trained on.  Missing features are extracted on `device` (the CLI does that first,
    in the GPU's turn)."""
    check_disk(out)
    heads, meta = load_model(out)
    torch.set_num_threads(THREADS)
    labelled = sorted(set(data.labelled_sheets(data.LABELS)))
    extract_features(labelled, device, out)
    if _sha256(out / REDUCER_FILE) != meta["reducer"]["sha256"]:
        raise RuntimeError(f"{out / REDUCER_FILE} is not the reducer the heads were trained on (the page list "
                           "changed?): run the held-out run and the final training again")
    params = shape.Params(**meta["params"])
    bounds = blk.BlockBounds(tuple(meta["bounds"]["rect_max"]), tuple(meta["bounds"]["ellipse_max"]))
    counts = data.map_counts(data.PAGE_KINDS)
    pages = data.map_pages(data.PAGE_KINDS, data.WORK)
    for sid in pages:
        if sid in labelled and not (sheet_dir(out, sid) / OUTSIDE_PNG).exists():
            log.warning("%s/%s is labelled but has no held-out outputs: run `sfzmaps frame --held-out`", *sid)
    todo = [sid for sid in pages if sid not in set(labelled)]
    jobs = []
    for sid in todo:
        f = load_features(out, sid)
        jobs.append(PageJob(sid=sid, page=data.page_path(*sid, work=data.WORK),
                            zs=tuple(layout.head_logits(h, f) for h in heads), temperature=float(meta["temperature"]),
                            maps=counts[sid], bounds=bounds))
    stats = pool_map(predict_page, jobs, params, out, label="page")
    rows = [{"series": s, "sheet": h, **v, "held_out": False} for (s, h), v in zip(todo, stats)]
    summary = out / FOLDS_FILE
    if summary.exists():
        on_page = set(pages)
        rows += [{"series": r["series"], "sheet": r["sheet"], "filled_ink": r["filled_ink"],
                  "dropped_ink": r["dropped_ink"], "removed_ink": r["removed_ink"],
                  "side_ink_left": r.get("side_ink_left"), "side_ink_left_total": r["side_ink_left_total"],
                  "held_out": True}
                 for r in json.loads(summary.read_text())["sheets"] if (r["series"], r["sheet"]) in on_page]
    else:
        log.warning("%s is missing: the labelled sheets are left out of %s", summary, PAGES_FILE)
    _write_json(out / PAGES_FILE, rank_pages(rows))


# ---------------------------------------------------------------------------------------------------------------
# The owner's summary
# ---------------------------------------------------------------------------------------------------------------


def _fmt(iv: dict, digits: int = 1) -> str:
    return f"{iv['mean']:.{digits}f} [{iv['lo']:.{digits}f}, {iv['hi']:.{digits}f}]"


def _shares_line(name: str, sh: dict) -> str:
    pct = scores.percent
    return (f"  {name}: frame/legend ink left in {pct(sh['frame_kept_share'])} (outside the frame "
            f"{pct(sh['outside_kept_share'])}, removed blocks {pct(sh['block_kept_share'])}); "
            f"map ink cut {pct(sh['map_cut_share'])}")


def format_scores(result: dict) -> str:
    """The held-out run for the owner: the ink shares first (pooled over every sheet's ink), then the numbers, the
    costs with 90% sheet-bootstrap intervals, the comparisons and each sheet."""
    pct = scores.percent
    lo, hi = result["temperature_interval"]
    p = result["params"]
    shp, pix = result["pooled"]["shape"], result["pooled"]["pixel"]
    lines = [
        f"held-out: {len(result['sheets'])} sheets, each laid out by {len(result['head_seeds'])} heads trained "
        f"without it",
        "pooled over every sheet's ink:",
        _shares_line("straight-line shape", shp["shares"]),
        _shares_line("per-pixel cut      ", pix["shares"]),
        f"shape numbers: lambda_notch {p['lam_notch']:g}, beta {p['beta']:g}, lambda_block {p['lam_block']:g} "
        f"({len(result['search'])} points searched, {result['rounds']} rounds); temperature "
        f"{result['temperature']:.3f} ({lo:.3f}-{hi:.3f})",
        f"means over sheets [90% sheet-bootstrap interval], cost at frame:map {decide.COST_FRAME:g}:"
        f"{decide.COST_MAP:g}:",
    ]
    for name, pooled in (("shape    ", shp), ("per-pixel", pix)):
        m = pooled["means"]
        lines.append(f"  {name}: cost {_fmt(m['cost'])} (layout {_fmt(m['cost_layout'])}, edge "
                     f"{_fmt(m['cost_edge'])}); frame ink left {_fmt(m['frame_kept'])}, map ink cut "
                     f"{_fmt(m['map_cut'])}")
    lines.append("shape - per-pixel, paired: " + ", ".join(f"{k} {_fmt(v)}" for k, v in result["shape_vs_pixel"].items()))
    lines.append(f"side ink left (confident frame ink along the fitted sides): {_fmt(result['side_ink_left'])}")
    base = result["baseline"]
    if base["sheets"]:
        lines.append(f"baseline (two-pass, {len(base['sheets'])} sheets), shape - old: "
                     + ", ".join(f"{k} {_fmt(v)}" for k, v in base["paired"].items()))
    else:
        lines.append("baseline: no shared sheets")
    lines.append("per sheet (shape | per-pixel):")
    for s in result["sheets"]:
        sc, px = s["scores"], s["pixel"]
        lines.append(f"  {s['series']}/{s['sheet']}: left in {pct(sc['frame_kept_share'])}, cut "
                     f"{pct(sc['map_cut_share'])}, cost {sc['cost']:.0f} (layout {sc['cost_layout']:.0f}, edge "
                     f"{sc['cost_edge']:.0f}), side ink left {s['side_ink_left_total']:.0f} | left in "
                     f"{pct(px['frame_kept_share'])}, cut {pct(px['map_cut_share'])}, cost {px['cost']:.0f}")
    lines.append("search (pooled cost):")
    for r in result["search"]:
        lines.append(f"  lambda_notch {r['lam_notch']:g}, beta {r['beta']:g}, lambda_block {r['lam_block']:g}: "
                     f"{r['cost']:.0f}")
    return "\n".join(lines)
