"""The frame finder's run (issue #71; design spec amendments 3 to 6): the model fit, the dev and test scores, and
every map page's outputs.  It orchestrates only: the layout is `layout`, the decision `shape` (per-pixel: `decide`),
the scores `scores` and the temperature `calibrate`.

The training sheets are the labels under `data.LABELS`; the test sheets, under `data.TEST_LABELS`, are never trained
on and never used to fit a number (`data.check_split` refuses a sheet in both, before anything else runs).

`fit` makes the model:
1. `layout.extract_all` over every map page and the training and test sheets: the frozen backbone's reduced
   features, on the run's device.  Reused when the page list is the reducer's; only `fit` may fit a reducer;
2. the training sheets dealt round-robin, in sorted order, into `TEMPERATURE_FOLDS` folds; per fold,
   `layout.HEAD_SEEDS` heads trained on the other folds (one padded batch on the run's device, one head after
   another), and the held sheets' patch logits per seed kept: the dev logits, `model/dev_logits.npz`;
3. the final heads, one per seed, on every training sheet.  Steps 1 to 3 run inside the caller's `turn` (the CLI's GPU
   turn);
4. one temperature fitted on ink-weighted samples of the dev logits upsampled to each page (`calibrate`), with its
   sheet-bootstrap interval;
5. the learned chances (`learned`, amendment 6): each training sheet's dev chances and label give its rule and ink
   rows (on the workers), and `learned.fit` on all of them the two curves;
6. `model/model.json`, written last: the temperature, the learned chances (`"chances"`, with the rules and ink they
   were fitted on), the folds, the block bounds of every label, the reducer record (sha256 and page list) and the
   training sheets.  A fit deletes it first, so one that stops part way leaves no model.
A model whose heads, dev logits, training sheets and reducer record are current but which has no learned chances (an
older model's) gets steps 5 and 6 only: the heads, dev logits and temperature are kept.

The model is stale (`model_stale`) when `model.json` is missing or has no learned chances, or its training sheets or
reducer record are not the current ones.  The shape decision's numbers are the constants `PARAMS`, not searched.

Every page's chances are corrected by `learned.apply` before the decision (`shape.decide`, or the per-pixel one on a
page of several maps), which cuts at even odds; the corrected chances are the ones written.  `score_dev` cuts each
training sheet from its dev logits (block bounds from its fold's other sheets, as its heads); `score_test` cuts each
test sheet with the final heads.  Both score the cuts against the labels apart by what is
deleted (`scores.split_scores`: frame ink, label-box ink, map ink cut by the frame and by boxes), and write under
`out/dev/` or `out/test/` the sheets' outputs, `scores.json` (per sheet and pooled) and `fits.json` (each sheet's
fitted frame, notches and boxes, `fit_json`; compared between runs by `fit_changes`).  `format_split` is the table for
the owner.

`predict_all` decides every map page that is not a training sheet with the final heads; the training sheets take
their dev outputs.

The page work (the shape decisions) runs on `PROCESSES` spawned worker processes, each pinned to one maths thread.  A
job carries everything it needs (the page's path, its patch logits, the temperature, the label folder), and its
worker loads the page, makes the chances and lets them go when the job ends.  `PROCESSES = 1` runs everything in this
process.

Outputs, all under `out` (default `data/frame`):
- `reducer.npz` and `features/<series>/<sheet>.npy`: the layout's features (`layout.extract_all`);
- `[dev/|test/]<series>/<sheet>/outside.png` (`OUTSIDE_PNG`): 1-bit, the deletion (`Decision.deleted`: outside the
  frame or a removed block);
- `.../outside_prob.npz` (`PROB_FILE`): key `q`, the learned chance of deletion p1 + p2 as uint8 levels
  round(255 q), for reading and placement to weigh ink by;
- `.../erased.png` (`ERASED_PNG`): the scan in its own mode (1-bit as 8-bit grey) with the deleted px painted plain
  paper, the per-channel median of the kept px: for looking only;
- `.../outline.json` (`OUTLINE_FILE`): {"polygons": the kept region (`decide.outline`), "maps": n, "shape": the
  fitted frame, notches and blocks, or null where the per-pixel cut decided (a page of several maps)};
- `dev/` and `test/` (`DEV_DIR`, `TEST_DIR`): `scores.json` (`SCORES_FILE`) and `fits.json` (`FITS_FILE`), deleted
  when a scoring run starts;
- `model/head-<seed>.pt`, `model/dev_logits.npz` (`DEV_LOGITS`: per training sheet "series/sheet", float32
  (seeds, classes, rows, cols)) and `model/model.json` (`MODEL_DIR`);
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
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sfzmaps.frame import blocks as blk
from sfzmaps.frame import calibrate, data, decide, layout, learned, rules, scores, shape
from sfzmaps.frame.data import OUT, SheetId
from sfzmaps.frame.geometry import CORNERS, SIDES
from sfzmaps.solver import threads

log = logging.getLogger(__name__)

TEMPERATURE_FOLDS = 4   # computational: folds of the training sheets for the dev logits and the temperature
# the shape decision's numbers behind the cuts the owner approved (spec amendment 4); constants, not searched
PARAMS = shape.Params(lam_notch=1e4, beta=3, lam_block=1e3)
PROCESSES = 4           # computational: worker processes for the page decisions
THREADS = 4             # computational: torch threads in this process (upsampling, head logits)
MIN_FREE_GB = 20        # computational safety margin: `predict_all` refuses with less free disk under `out`

MODEL_DIR = "model"
DEV_LOGITS = "dev_logits.npz"
DEV_DIR = "dev"
TEST_DIR = "test"
SCORES_FILE = "scores.json"
FITS_FILE = "fits.json"
REDUCER_FILE = "reducer.npz"          # `layout.extract_all`'s file
OUTSIDE_PNG = "outside.png"
PROB_FILE = "outside_prob.npz"
ERASED_PNG = "erased.png"
OUTLINE_FILE = "outline.json"
PAGES_FILE = "pages.json"
PAGE_FILES = (OUTSIDE_PNG, PROB_FILE, ERASED_PNG, OUTLINE_FILE)


class DiskFull(RuntimeError):
    """Less than `MIN_FREE_GB` free where the outputs go."""


class StaleFeatures(RuntimeError):
    """The model is stale (`model_stale`): its features, reducer or training sheets are not the current ones."""


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


def _name(sid: SheetId) -> str:
    return f"{sid[0]}/{sid[1]}"


def _names(pages: Sequence[SheetId]) -> list[str]:
    return [_name(s) for s in pages]


def _sheet_lists() -> tuple[list[SheetId], list[SheetId]]:
    """(training sheets, test sheets), each sorted, from the label folders as they are now."""
    return (sorted(set(data.labelled_sheets(data.LABELS))), sorted(set(data.test_sheets(data.TEST_LABELS))))


# ---------------------------------------------------------------------------------------------------------------
# The process pool
# ---------------------------------------------------------------------------------------------------------------

def _init_worker() -> None:
    """A spawned worker: one thread for OpenCV and torch (BLAS is pinned by the environment it started with), and
    its warnings shown."""
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")


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


def pool_map(fn: Callable, items: Sequence, *args, label: str = "",
             describe: Callable[[object], str] | None = None) -> list:
    """`[fn(item, *args) for item in items]`, in order, on up to `PROCESSES` spawned workers (here when
    `PROCESSES` <= 1 or there is one item).  `fn` must be a module-level function.  With a `label`, each result is
    logged as it completes (one line, with `describe(result)` when given)."""
    items = list(items)
    n = len(items)
    t0 = time.perf_counter()

    def note(i: int, r) -> None:
        if label:
            log.info("%s %d/%d (%.0f s)%s", label, i, n, time.perf_counter() - t0,
                     f": {describe(r)}" if describe is not None else "")

    if PROCESSES <= 1 or n <= 1:
        out = []
        for i, item in enumerate(items, 1):
            out.append(fn(item, *args))
            note(i, out[-1])
        return out
    with _pinned_environment(), concurrent.futures.ProcessPoolExecutor(
            min(PROCESSES, n), mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker) as ex:
        futures = {ex.submit(fn, item, *args): k for k, item in enumerate(items)}
        out = [None] * n
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            out[futures[f]] = f.result()
            note(i, out[futures[f]])
        return out


# ---------------------------------------------------------------------------------------------------------------
# The layout: features, heads, chances
# ---------------------------------------------------------------------------------------------------------------


def feature_pages(sheets: Sequence[SheetId]) -> list[SheetId]:
    """The pages the reducer is fitted over: every map page and `sheets` (the training and test sheets), sorted.  The
    fit, the scoring and the prediction use the same list, so the features are extracted once."""
    return sorted(set(data.map_pages(data.PAGE_KINDS, data.WORK)) | set(sheets))


def extract_features(sheets: Sequence[SheetId], device: str, out: Path = OUT, record: dict | None = None) -> dict:
    """`layout.extract_all` over `feature_pages(sheets)` on `device` (a no-op when they are all there); returns the
    reducer's record (`file`, `sha256`, `pages`).  With a `record` (a model's), it refuses (`StaleFeatures`) before
    extracting when the page list differs, and after when the reducer is not that record's: only `fit` fits a
    reducer."""
    pages = feature_pages(sheets)
    if record is not None and record.get("pages") != _names(pages):
        raise StaleFeatures("the pages changed since the model was fitted: refit it")
    layout.extract_all(pages, device, out)
    now = {"file": REDUCER_FILE, "sha256": _sha256(out / REDUCER_FILE), "pages": _names(pages)}
    if record is not None and now["sha256"] != record["sha256"]:
        raise StaleFeatures(f"{out / REDUCER_FILE} is not the reducer the model was fitted on: refit it")
    return now


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


def _train(feats: dict, targets: dict, rest: Sequence[SheetId], seed: int, device: str) -> tuple:
    t0 = time.perf_counter()
    head = layout.train_head([feats[s] for s in rest], [targets[s] for s in rest], seed, _head_device(device))
    return head, time.perf_counter() - t0


def _head_device(device: str) -> str:
    """Heads train on the run's device: mps where it is the device, else cpu."""
    return "mps" if device == "mps" else "cpu"


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
    maps on it, the block bounds, the learned chances, and the label folder (labelled sheets: to score)."""

    sid: SheetId
    page: Path
    zs: tuple
    temperature: float
    maps: int
    bounds: blk.BlockBounds
    chances: learned.Chances
    labels: Path | None = None


def _load(job: PageJob) -> tuple[np.ndarray, np.ndarray, dict]:
    """The page's luminance, its learned chances (`learned.apply` on the model's) and a `shape.decide` cache holding
    its rule labels, found once for both."""
    lum = data.load_luminance(job.page)
    labels = rules.rule_labels(lum)
    p = learned.apply(chances(job.zs, lum.shape, job.temperature), lum, job.chances, labels)
    return lum, p, {"rules": labels}


def _r(x) -> float:
    return round(float(x), 3)


def fit_json(fr, ns, bs) -> dict:
    """A sheet's fit as plain JSON numbers rounded to 3 decimals: the frame's `angle` (degrees), per side its offset,
    tilt and bow, or "page edge" where the side is unframed, and per corner its arc (`geometry.Frame`); the notches
    and the boxes (`notches.Notch`, `blocks.Block`, without their gains)."""
    sides = {s: {"offset": _r(fr.offset[s]), "tilt": _r(fr.tilt[s]), "bow": _r(fr.bow[s])} if fr.framed(s)
             else "page edge" for s in SIDES}
    return {"frame": {"angle": _r(fr.phi), "sides": sides, "arcs": {"-".join(c): _r(fr.arc[c]) for c in CORNERS}},
            "notches": [{"side": n.side, "psi": _r(n.psi), "a0": _r(n.a0), "a1": _r(n.a1), "d": _r(n.d)}
                        for n in ns],
            "boxes": [{"kind": b.kind, "cx": _r(b.cx), "cy": _r(b.cy), "theta": _r(b.theta),
                       "p": [_r(v) for v in b.p]} for b in bs]}


def _shape_json(fr, ns, bs) -> dict:
    """The fitted frame, notches and blocks as JSON (page px and degrees; see `geometry.Frame`, `notches.Notch`,
    `blocks.Block`)."""
    frame = {"phi": fr.phi, "offset": fr.offset, "tilt": fr.tilt, "bow": fr.bow,
             "arc": {"-".join(c): r for c, r in fr.arc.items()}}
    return {"frame": frame, "notches": [dataclasses.asdict(n) for n in ns],
            "blocks": [dataclasses.asdict(b) for b in bs]}


def score_page(job: PageJob, params: shape.Params, out: Path) -> dict:
    """A labelled page (its label under `job.labels`): the shape decision at `params`, its outputs written under
    `out`, scored against the label (`scores.split_scores`); its fit (`fit_json`), ink figures and side ink left."""
    lum, p, cache = _load(job)
    dec, fr, ns, bs = shape.decide(p, lum, params, job.bounds, job.maps, cache=cache)
    del cache
    fitted = (fr, ns, bs) if job.maps == 1 else None
    write_outputs(out, job.sid, p, dec, job.maps, page=job.page, fitted=fitted)
    side = shape.side_ink_left(p, lum, fr)
    split = scores.split_scores(dec.outside, dec.removed, data.load_label(*job.sid, root=job.labels),
                                data.load_remove(*job.sid, root=job.labels), lum)
    return {"series": job.sid[0], "sheet": job.sid[1], "maps": job.maps, "split": split,
            "filled_ink": dec.filled_ink, "dropped_ink": dec.dropped_ink, "removed_ink": dec.removed_ink,
            "side_ink_left": side, "side_ink_left_total": float(sum(side.values())),
            "bounds": dataclasses.asdict(job.bounds), "fit": fit_json(fr, ns, bs)}


def predict_page(job: PageJob, params: shape.Params, out: Path) -> dict:
    """An unlabelled page: the shape decision, its outputs written; its ink figures and the side ink left."""
    lum, p, cache = _load(job)
    dec, fr, ns, bs = shape.decide(p, lum, params, job.bounds, job.maps, cache=cache)
    del cache
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


@dataclasses.dataclass(frozen=True, eq=False)
class RowsJob:
    """What a worker needs for one training sheet's learned-chance rows: its id, scan and label folder, its dev
    logits (each seed's) and the temperature."""

    sid: SheetId
    page: Path
    zs: tuple
    temperature: float
    labels: Path


def chance_rows(job: RowsJob) -> dict:
    """A training sheet's `learned.rule_rows` and `learned.ink_rows` on its dev chances and its label."""
    lum = data.load_luminance(job.page)
    p = chances(job.zs, lum.shape, job.temperature)
    cls = _cls(job.sid, lum.shape, job.labels)
    labels = rules.rule_labels(lum)
    return {"series": job.sid[0], "sheet": job.sid[1], "rules": learned.rule_rows(p, lum, cls, labels),
            "ink": learned.ink_rows(p, lum, cls, labels)}


def _rows_line(r: dict) -> str:
    return (f"{r['series']}/{r['sheet']}: {len(r['rules'])} rules (ink {r['rules'][:, 0].sum():.0f}), other ink "
            f"{r['ink'][:, 0].sum():.0f}")


def fit_chances(train: Sequence[SheetId], dev: dict, temperature: float) -> dict:
    """The learned chances (`learned.fit`) from every training sheet's dev chances (`dev`: its dev logits) at
    `temperature` and its label, the rows made on `PROCESSES` workers (one log line per sheet): the two curves and
    what they were fitted on, as `model.json` keeps them."""
    t0 = time.perf_counter()
    jobs = [RowsJob(sid=sid, page=data.page_path(*sid, work=data.WORK), zs=tuple(dev[sid]),
                    temperature=float(temperature), labels=data.LABELS) for sid in train]
    rows = pool_map(chance_rows, jobs, label="chances sheet", describe=_rows_line)
    rule = np.concatenate([r["rules"] for r in rows])
    ink = np.sum([r["ink"] for r in rows], axis=0)
    got = learned.fit(rule, ink)
    a, b, g = got.rule
    i0, i1 = got.ink
    log.info("learned chances from %d sheets (%d rules) in %.0f s: rules sigmoid(%.3f + %.3f logit(reading) + %.3f "
             "log(length px)); other ink sigmoid(%.3f + %.3f logit(reading)), even odds at a reading of %.3f",
             len(rows), len(rule), time.perf_counter() - t0, a, b, g, i0, i1, 1.0 / (1.0 + np.exp(i0 / i1)))
    return {**got.to_json(), "fitted_on": {"sheets": len(rows), "rules": int(len(rule)),
                                           "rule_ink": float(rule[:, 0].sum()), "other_ink": float(ink[:, 0].sum())}}


# ---------------------------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------------------------


def _save_head(head, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(head.state_dict(), tmp)
    os.replace(tmp, path)


def fit(device: str, out: Path = OUT,
        turn: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext) -> dict:
    """Fit the model (see the module): the dev logits, the temperature, the final heads and the learned chances;
    returns what it writes to `out/model/model.json`.  Refuses (ValueError) a sheet that is both a training and a test
    sheet, before anything is read or written.  The features (extracted, or refitted when the page list changed: the
    only place that may) and the heads run on `device` inside `turn()` (the CLI's GPU turn); the rest runs on cpu
    after it.  A model current but for its learned chances keeps its heads, dev logits and temperature and gets the
    chances only (no turn taken)."""
    train, test = _sheet_lists()
    data.check_split(train, test)
    if len(train) < TEMPERATURE_FOLDS:
        raise ValueError(f"{len(train)} training sheet(s): {TEMPERATURE_FOLDS} folds need at least as many")
    d = out / MODEL_DIR
    torch.set_num_threads(THREADS)
    if _heads_stale(out) is None:
        meta = json.loads((d / "model.json").read_text())
        if "chances" not in meta:
            log.info("the heads, dev logits and temperature (%.3f) are current: fitting the learned chances only",
                     meta["temperature"])
            meta["chances"] = fit_chances(train, load_dev_logits(out), float(meta["temperature"]))
            meta["level"] = decide.cut_level()
            _write_json(d / "model.json", meta)
            return json.loads((d / "model.json").read_text())
    (d / "model.json").unlink(missing_ok=True)
    seeds = list(layout.HEAD_SEEDS)
    folds = [train[k::TEMPERATURE_FOLDS] for k in range(TEMPERATURE_FOLDS)]
    dev: dict[SheetId, np.ndarray] = {}
    seconds = []
    with turn():
        reducer = extract_features(train + test, device, out)
        feats, targets = _features_and_targets(train, out)
        for k, held in enumerate(folds):
            rest = [s for s in train if s not in set(held)]
            heads = []
            for seed in seeds:
                head, sec = _train(feats, targets, rest, seed, device)
                heads.append(head)
                seconds.append({"fold": k, "seed": seed, "seconds": sec})
                log.info("fold %d/%d, seed %d: head on %d sheets, %.1f s on %s", k + 1, TEMPERATURE_FOLDS, seed,
                         len(rest), sec, _head_device(device))
            for sid in held:
                dev[sid] = np.stack([layout.head_logits(h, feats[sid]) for h in heads]).astype(np.float32)
        d.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            head, sec = _train(feats, targets, train, seed, device)
            seconds.append({"fold": None, "seed": seed, "seconds": sec})
            log.info("final head, seed %d: on %d sheets, %.1f s on %s", seed, len(train), sec, _head_device(device))
            _save_head(head, d / f"head-{seed}.pt")
    tmp = d / (DEV_LOGITS + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **{_name(sid): z for sid, z in sorted(dev.items())})
    os.replace(tmp, d / DEV_LOGITS)

    t0 = time.perf_counter()
    rng = np.random.default_rng(0)            # one generator for the run, sheets in sorted order
    samples = []
    for sid in train:
        lum = _lum(sid)
        samples.append(sample_patch_logits(list(dev[sid]), _cls(sid, lum.shape), lum, rng, calibrate.N_CAL))
    temperature = calibrate.fit_temperature(samples)
    t_lo, t_hi = calibrate.temperature_interval(samples, calibrate.CAL_BOOT, centre=temperature)
    reliability = calibrate.reliability(samples, temperature)
    del samples
    log.info("temperature %.3f (%.3f-%.3f), from the dev logits of %d sheets in %.0f s", temperature, t_lo, t_hi,
             len(train), time.perf_counter() - t0)
    fitted = fit_chances(train, dev, temperature)

    meta = {"head_seeds": seeds, "head_seconds": seconds, "components": layout.COMPONENTS,
            "temperature": temperature, "temperature_interval": [t_lo, t_hi], "reliability": reliability,
            "chances": fitted, "folds": [[list(s) for s in f] for f in folds],
            "bounds": dataclasses.asdict(blk.block_bounds(train, root=data.LABELS)), "reducer": reducer,
            "sheets": [list(s) for s in train], "params": dataclasses.asdict(PARAMS), "level": decide.cut_level()}
    _write_json(d / "model.json", meta)
    return json.loads((d / "model.json").read_text())


def model_stale(out: Path = OUT) -> str | None:
    """Why the model under `out/model/` is stale, or None when it is current: `model.json` missing (or an older
    model's), its training sheets not the labels' now, its reducer record not the pages' now or not `reducer.npz`,
    a file it needs missing, or no learned chances (`fit` then fits only those)."""
    reason = _heads_stale(out)
    if reason is None and "chances" not in json.loads((out / MODEL_DIR / "model.json").read_text()):
        return (f"{out / MODEL_DIR / 'model.json'} has no learned chances (amendment 6): they are fitted on its dev "
                "logits, its heads kept")
    return reason


def _heads_stale(out: Path) -> str | None:
    """`model_stale` but for the learned chances: why the heads, dev logits, temperature, training sheets or reducer
    record under `out/model/` are not current, or None."""
    d = out / MODEL_DIR
    f = d / "model.json"
    if not f.exists():
        return f"{f} is missing"
    meta = json.loads(f.read_text())
    if not {"folds", "reducer", "head_seeds"} <= set(meta):
        return f"{f} is an older model's (no folds or reducer record)"
    train, test = _sheet_lists()
    had = [tuple(s) for s in meta["sheets"]]
    if had != train:
        added, gone = _names(sorted(set(train) - set(had))), _names(sorted(set(had) - set(train)))
        return (f"the training sheets changed since the fit ({len(added)} added, e.g. {added[:3]}; {len(gone)} gone, "
                f"e.g. {gone[:3]})")
    rec = meta["reducer"]
    now = _names(feature_pages(train + test))
    if rec.get("pages") != now:
        added, gone = sorted(set(now) - set(rec.get("pages", []))), sorted(set(rec.get("pages", [])) - set(now))
        return (f"the pages changed since the fit ({len(added)} added, e.g. {added[:3]}; {len(gone)} gone, e.g. "
                f"{gone[:3]}): the reducer would be refitted")
    r = out / REDUCER_FILE
    if not r.exists() or _sha256(r) != rec.get("sha256"):
        return f"{r} is not the reducer the model was fitted on"
    for need in [d / DEV_LOGITS, *(d / f"head-{s}.pt" for s in meta["head_seeds"])]:
        if not need.exists():
            return f"{need} is missing"
    return None


def _require_current(out: Path) -> None:
    reason = model_stale(out)
    if reason is not None:
        raise StaleFeatures(f"{reason}: refit the model first (`sfzmaps frame` fits it when it is stale)")


def load_model(out: Path) -> tuple[list, dict]:
    """The final heads (eval, cpu) and `model.json`, from `out/model/`."""
    d = out / MODEL_DIR
    if not (d / "model.json").exists():
        raise FileNotFoundError(f"{d / 'model.json'} is missing: fit the model first (`sfzmaps frame`)")
    meta = json.loads((d / "model.json").read_text())
    if "head_seeds" not in meta:
        raise FileNotFoundError(f"{d / 'model.json'} is the old full-resolution model's: fit the model first "
                                "(`sfzmaps frame`)")
    heads = []
    for s in meta["head_seeds"]:
        head = layout.new_head()
        head.load_state_dict(torch.load(d / f"head-{s}.pt", map_location="cpu", weights_only=True))
        heads.append(head.eval())
    return heads, meta


def load_dev_logits(out: Path = OUT) -> dict[SheetId, np.ndarray]:
    """Each training sheet's dev logits (`DEV_LOGITS`): float32 (seeds, classes, rows, cols), from the heads of the
    fold that held it out."""
    with np.load(out / MODEL_DIR / DEV_LOGITS) as z:
        return {tuple(k.split("/", 1)): z[k] for k in z.files}


def _bounds(meta_bounds: dict) -> blk.BlockBounds:
    return blk.BlockBounds(tuple(meta_bounds["rect_max"]), tuple(meta_bounds["ellipse_max"]))


# ---------------------------------------------------------------------------------------------------------------
# The dev and test scores
# ---------------------------------------------------------------------------------------------------------------


def _deleted_boxes(boxes: list[dict]) -> int:
    """The labelled boxes most of whose ink is deleted (a reporting count; each box's share is in `scores.json`)."""
    return sum(1 for b in boxes if b["deleted_share"] > 0.5)


def _split_line(name: str, shares: dict, boxes: list[dict]) -> str:
    pct = scores.percent
    return (f"{name}: frame left {pct(shares['frame_left_share'])} | boxes left {pct(shares['box_left_share'])} "
            f"({_deleted_boxes(boxes)} of {len(boxes)} deleted) | map cut by frame "
            f"{pct(shares['map_cut_frame_share'])}, by boxes {pct(shares['map_cut_boxes_share'])}")


def _sheet_line(row: dict) -> str:
    return _split_line(f"{row['series']}/{row['sheet']}", scores.pooled_split([row["split"]]), row["split"]["boxes"])


def format_split(result: dict) -> str:
    """A `score_dev` or `score_test` result for the owner: one line per sheet, then the pooled line (summed ink over
    summed ink)."""
    rows = result["sheets"]
    lo, hi = result["temperature_interval"]
    lines = [f"{result['set']}: {len(rows)} sheets, temperature {result['temperature']:.3f} ({lo:.3f}-{hi:.3f})"]
    lines += [_sheet_line(r) for r in rows]
    lines.append(_split_line(f"pooled ({len(rows)} sheets)", result["pooled"],
                             [b for r in rows for b in r["split"]["boxes"]]))
    return "\n".join(lines)


def _score_set(name: str, jobs: list[PageJob], where: Path, meta: dict) -> dict:
    """`score_page` on every job (`PROCESSES` workers, one log line per finished sheet), then `SCORES_FILE` and
    `FITS_FILE` under `where`; returns what `SCORES_FILE` holds."""
    for f in (SCORES_FILE, FITS_FILE):
        (where / f).unlink(missing_ok=True)
    rows = pool_map(score_page, jobs, PARAMS, where, label=f"{name} sheet", describe=_sheet_line)
    fits = {f"{r['series']}/{r['sheet']}": r.pop("fit") for r in rows}
    _write_json(where / FITS_FILE, fits)
    _write_json(where / SCORES_FILE, {
        "set": name, "temperature": meta["temperature"], "temperature_interval": meta["temperature_interval"],
        "params": dataclasses.asdict(PARAMS), "sheets": rows,
        "pooled": scores.pooled_split([r["split"] for r in rows])})
    return json.loads((where / SCORES_FILE).read_text())


def score_dev(out: Path = OUT) -> dict:
    """Each training sheet cut from its dev logits at the model's temperature, corrected by the learned chances, with
    block bounds from its fold's other sheets (as its heads), and scored against its label; written under `out/dev/` (see the module).  Refuses
    (`StaleFeatures`) a stale model."""
    _require_current(out)
    meta = json.loads((out / MODEL_DIR / "model.json").read_text())
    torch.set_num_threads(THREADS)
    train = [tuple(s) for s in meta["sheets"]]
    logits = load_dev_logits(out)
    counts = data.map_counts(data.PAGE_KINDS)
    jobs = []
    for fold in meta["folds"]:
        held = [tuple(s) for s in fold]
        bounds = blk.block_bounds([s for s in train if s not in set(held)], root=data.LABELS)
        jobs += [PageJob(sid=sid, page=data.page_path(*sid, work=data.WORK), zs=tuple(logits[sid]),
                         temperature=float(meta["temperature"]), maps=counts[sid], bounds=bounds,
                         chances=learned.Chances.from_json(meta["chances"]), labels=data.LABELS)
                 for sid in held]
    jobs.sort(key=lambda j: j.sid)
    return _score_set("dev", jobs, out / DEV_DIR, meta)


def score_test(out: Path = OUT) -> dict:
    """Each test sheet cut by the final heads at the model's temperature, corrected by the learned chances, and
    scored against its label (under
    `data.TEST_LABELS`); written under `out/test/` (see the module).  Refuses (`StaleFeatures`) a stale model."""
    _require_current(out)
    heads, meta = load_model(out)
    torch.set_num_threads(THREADS)
    _, test = _sheet_lists()
    counts = data.map_counts(data.PAGE_KINDS)
    bounds = _bounds(meta["bounds"])
    jobs = []
    for sid in test:
        f = load_features(out, sid)
        jobs.append(PageJob(sid=sid, page=data.page_path(*sid, work=data.WORK),
                            zs=tuple(layout.head_logits(h, f) for h in heads), temperature=float(meta["temperature"]),
                            maps=counts[sid], bounds=bounds, chances=learned.Chances.from_json(meta["chances"]),
                            labels=data.TEST_LABELS))
    return _score_set("test", jobs, out / TEST_DIR, meta)


def fit_changes(before: Path, after: Path) -> list[str]:
    """The sheets, in both `fits.json` files, whose frame (angle, sides, arcs) or notches differ between them; the
    boxes are not compared."""
    a, b = json.loads(Path(before).read_text()), json.loads(Path(after).read_text())
    return sorted(k for k in a.keys() & b.keys()
                  if a[k]["frame"] != b[k]["frame"] or a[k]["notches"] != b[k]["notches"])


# ---------------------------------------------------------------------------------------------------------------
# Every page
# ---------------------------------------------------------------------------------------------------------------


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


def _copy_dev_outputs(out: Path, sid: SheetId) -> bool:
    """The training sheet's dev outputs copied to its place under `out`; False (nothing copied) when any is missing."""
    src, dst = sheet_dir(out / DEV_DIR, sid), sheet_dir(out, sid)
    if not all((src / n).exists() for n in PAGE_FILES):
        return False
    dst.mkdir(parents=True, exist_ok=True)
    for n in PAGE_FILES:
        shutil.copyfile(src / n, dst / (n + ".tmp"))
        os.replace(dst / (n + ".tmp"), dst / n)
    return True


def predict_all(device: str, out: Path = OUT,
                turn: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext) -> None:
    """Every map page that is not a training sheet, decided with the final heads at the model's temperature and
    `PARAMS`; the training sheets take their dev outputs (`sfzmaps frame --dev`).  Then `PAGES_FILE` (`rank_pages`;
    the training sheets' rows from `dev/scores.json`, marked `held_out`).  Refuses (`DiskFull`) with less than
    `MIN_FREE_GB` free, and (`StaleFeatures`, before any extraction) a stale model.  Missing feature files are
    extracted on `device` inside `turn()`."""
    check_disk(out)
    _require_current(out)
    heads, meta = load_model(out)
    torch.set_num_threads(THREADS)
    train, test = _sheet_lists()
    with turn():
        extract_features(train + test, device, out, meta["reducer"])
    bounds = _bounds(meta["bounds"])
    counts = data.map_counts(data.PAGE_KINDS)
    pages = data.map_pages(data.PAGE_KINDS, data.WORK)
    for sid in pages:
        if sid in set(train) and not _copy_dev_outputs(out, sid):
            log.warning("%s/%s is a training sheet with no dev outputs: run `sfzmaps frame --dev`", *sid)
    todo = [sid for sid in pages if sid not in set(train)]
    jobs = []
    for sid in todo:
        f = load_features(out, sid)
        jobs.append(PageJob(sid=sid, page=data.page_path(*sid, work=data.WORK),
                            zs=tuple(layout.head_logits(h, f) for h in heads), temperature=float(meta["temperature"]),
                            maps=counts[sid], bounds=bounds, chances=learned.Chances.from_json(meta["chances"])))
    stats = pool_map(predict_page, jobs, PARAMS, out, label="page")
    rows = [{"series": s, "sheet": h, **v, "held_out": False} for (s, h), v in zip(todo, stats)]
    summary = out / DEV_DIR / SCORES_FILE
    if summary.exists():
        on_page = set(pages)
        rows += [{"series": r["series"], "sheet": r["sheet"], "filled_ink": r["filled_ink"],
                  "dropped_ink": r["dropped_ink"], "removed_ink": r["removed_ink"],
                  "side_ink_left": r.get("side_ink_left"), "side_ink_left_total": r["side_ink_left_total"],
                  "held_out": True}
                 for r in json.loads(summary.read_text())["sheets"] if (r["series"], r["sheet"]) in on_page]
    else:
        log.warning("%s is missing: the training sheets are left out of %s", summary, PAGES_FILE)
    _write_json(out / PAGES_FILE, rank_pages(rows))
