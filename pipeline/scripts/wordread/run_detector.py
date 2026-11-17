"""Task 8's detector experiments: train the text detector with the word heads on one quarter held out, form its words
from the votes (`wordread.words.core_words`), place and read them, and measure them end to end.  Everything is written
to data/solver/experiments/wordread/detector/runs/<name>-<q>-s<seed>/ (never committed).

    uv run python scripts/wordread/run_detector.py check --quarter ne                  # two short runs repeat? memory
    uv run python scripts/wordread/run_detector.py train --name a --quarter ne --seed 0 [--doubt]
    uv run python scripts/wordread/run_detector.py fit --name a --quarter ne --seed 0 --own   # spreads and prior
    uv run python scripts/wordread/run_detector.py words --name a --quarter ne --seed 0
    uv run python scripts/wordread/run_detector.py read --name a --quarter ne --seed 0
    uv run python scripts/wordread/run_detector.py measure --name a --quarter ne --seed 0
    uv run python scripts/wordread/run_detector.py page --name a --quarter ne --seed 0

The steps, each reading the one before from the run's folder:
  train    `variant.run` with the word heads (`train.Recipe(heads="words")`, the stage's recipe, the per-ink head
           taught by the label rule lines-unknown, ruling R2), the quarter and the answered blocks held out; with
           `--doubt` Terra's doubt weighs each word (`wordread.doubt`, the "box" weights).  The run's folder holds
           variant's files (model.pt, chance/, the gate's scores.json, run.json) and words/<sheet>.npz, the word
           heads' outputs over every 2000 page (`predict.word_outputs`, tiles stitched).
  fit      the word model's spreads (`words.measure_spreads`) and prior (`words.prior_decisions`, then
           `words.fit_word_prior`: a first prior on the "word" decisions alone, then every decision the search
           takes with it), on predictions against Terra's words in the training quarters only: never the held-out
           quarter, never an answered block (`fit_area`), never the owner's answers.  `--own`: this run's own
           predictions on its training quarters (optimistic: it trained there; Phase 1 only).  Otherwise (`--from`
           runs) the cross-fit: each of the other folds' detectors' predictions on ITS held-out quarter, never this
           fold's quarter.  Writes fit/{spreads.json, prior.json, decisions.parquet}.
  words    each 2000 sheet's words in the held-out quarter: `core_words` on the page's word outputs, the px outside
           the quarter not a word's (`predict.NOT_SCORED`); placed as `wordboxes.sheet_boxes` places its boxes
           (`wordboxes.place`), the boxes whose map point is in the quarter kept.  Writes boxes/boxes.parquet
           (wordboxes.COLUMNS then words.EXTRA_COLUMNS) and boxes/summary.json.
  read     the boxes read by the current fold readers (`textread.read.build_read`, data/solver/text-train), into
           read/ (a text-read folder: reads.npz, reads.parquet, answered.parquet).
  measure  the headline (read right at our boxes over the quarter's answered labels with a code, a missed box wrong;
           `measures.headline`), the box numbers against Terra's words of the quarter (`measures.box_numbers`) and
           the read numbers: headline.json, box_numbers.json, read_numbers.json.
  page     the Mission Street review page (`wordread.review`) with this run's boxes in its quarter; the other quarters
           drawn faded, "fold not yet trained" (no quarter is shown from a model that trained on it); written to
           data/solver/experiments/wordread/review/<name>/.

CPU work runs in up to WORKERS processes of one thread; every training and prediction takes the GPU's exclusive turn.
$0: nothing here calls a paid service.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "textlines"))
import variant  # noqa: E402

from sfzmaps import gpu  # noqa: E402
from sfzmaps.solver import paths, runs  # noqa: E402
from sfzmaps.solver.textnet import data, reader_stage, stage, train, zones  # noqa: E402

log = logging.getLogger(__name__)

ROOT = runs.EXPERIMENTS_DIR / "wordread" / "detector"
RUNS = ROOT / "runs"
WORKERS = 4
CHECK_STEPS = 100        # each of the two short runs of `check`


def run_name(name: str, quarter: str, seed: int) -> str:
    return f"{name}-{quarter}-s{seed}"


def run_dir(name: str, quarter: str, seed: int, root: Path | None = None) -> Path:
    return (root or RUNS) / run_name(name, quarter, seed)


def word_recipe(**changes) -> train.Recipe:
    """The stage's recipe with the word heads, the per-ink head by the label rule lines-unknown (ruling R2)."""
    return dataclasses.replace(stage.DEFAULT_RECIPE, labels="lines-unknown", heads="words", **changes)


def held_areas(quarter: str) -> tuple:
    """The areas a fold holds out: its quarter and the answered blocks."""
    return (data.Quarter(paths.QUARTERS.index(quarter)), reader_stage.load_block_area(reader_stage.held_blocks()))


def _state_hash(net) -> str:
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# --- the first mps run: two short runs, bit for bit ----------------------------------------------------------------

def check(quarter: str, steps: int = CHECK_STEPS, seed: int = 0, device: str | None = None, sheets=None,
          areas=None, doubt: bool = False, out: Path | None = None) -> dict:
    """Two short trainings of the word recipe (`steps` steps each, the same seed, deterministic mode) on the sheets
    with the quarter's areas held out: whether their weights and learned loss settings are the same bit for bit, and
    the most memory the GPU's allocator held over both (`variant.Peak`).  Writes check-<quarter>.json under ROOT."""
    device = device or train.default_device()
    if sheets is None:
        sheets = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False)
    areas = held_areas(quarter) if areas is None else tuple(areas)
    from sfzmaps.solver.wordread import doubt as doubt_mod
    from sfzmaps.solver.wordread import shrink
    prepared = shrink.prepare(zones.prepare(sheets, areas, variant.LINES))
    if doubt:
        prepared = doubt_mod.prepare(prepared, doubt_mod.load_weights(quarter), "box")
    held = data.hold_out(prepared, areas)
    recipe = word_recipe(iterations=steps)
    got = []
    t0 = time.perf_counter()
    with gpu.gpu_turn(True, device), variant.Peak(device) as peak:
        for _ in range(2):
            net = train.fit(held, recipe, device, seed)
            got.append({"hash": _state_hash(net), "learned": net.learned_word_loss,
                        "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}})
            del net
            if gpu.on_gpu(device):
                import torch
                torch.mps.empty_cache()
    diff = max(float((a - got[1]["state"][k]).abs().max()) if a.is_floating_point() else
               float((a != got[1]["state"][k]).any()) for k, a in got[0]["state"].items())
    result = {"quarter": quarter, "seed": seed, "steps": steps, "device": device, "recipe": dataclasses.asdict(recipe),
              "doubt": doubt, "same_bits": got[0]["hash"] == got[1]["hash"], "max_abs_weight_diff": diff,
              "learned_same": got[0]["learned"] == got[1]["learned"], "learned": [g["learned"] for g in got],
              "hashes": [g["hash"] for g in got], "mps_peak": peak.most if peak.on else None,
              "seconds": round(time.perf_counter() - t0, 1)}
    f = (out or ROOT) / f"check-{quarter}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(result, indent=1) + "\n")
    return result


# --- training -------------------------------------------------------------------------------------------------------

def train_run(name: str, quarter: str, seed: int, doubt: bool = False, recipe: train.Recipe | None = None,
              device: str | None = None, sheets=None, areas=None, score=None, root: Path | None = None,
              doubt_weights=None) -> Path:
    """`variant.run` of the word recipe (or `recipe`) with the fold's areas held out, into RUNS (or `root`)."""
    recipe = word_recipe() if recipe is None else recipe
    if recipe.heads != "words":
        raise ValueError(f"a detector run here has the word heads, not heads {recipe.heads!r}")
    return variant.run(name, recipe, quarter, seed, device=device, sheets=sheets, areas=areas, score=score,
                       doubt=doubt, root=root or RUNS, doubt_weights=doubt_weights)




# --- what the steps read of a run and of a sheet ----------------------------------------------------------------------

def outputs(run: Path, sheet: str):
    """The word heads' outputs of a 2000 sheet's page in a run's folder (`variant.save_word_outputs`)."""
    return variant.load_word_outputs(Path(run) / variant.WORDS / f"{sheet}.npz")


_RECORDS: dict = {}


def load_sheet(sheet: str) -> data.SheetData:
    """A 2000 sheet's SheetData with its placement (for the quarters and the held-out areas) and Terra's words, no
    area held out."""
    if "terra" not in _RECORDS:
        from sfzmaps.solver.textnet import boxes
        _RECORDS["terra"] = boxes.terra_records()
    recs = [r for r in _RECORDS["terra"] if r.get("series") == paths.SERIES and r.get("sheet") == sheet]
    return data.sheet_data(paths.SERIES, sheet, hold_out_validation=False, terra=recs)


def fit_area(sd: data.SheetData, quarters, blocks) -> np.ndarray:
    """The px of a page a fit may read, bool (h, w): in one of `quarters` (numbers, by each px's map point,
    `reader_stage.page_quarters`) and in no answered block (`blocks`, a `data.GridArea`; None: none held)."""
    out = np.isin(reader_stage.page_quarters(sd), list(quarters))
    if blocks is not None:
        for y0 in range(0, sd.height, data.BAND):
            h = min(data.BAND, sd.height - y0)
            out[y0:y0 + h] &= ~data.in_area(sd, blocks, y0, 0, h, sd.width)
    return out


def windows(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Rectangles (y0, x0, h, w) that cover the px of `mask` exactly, each wholly inside it: the page cut in four,
    and each part again, until a part is wholly in or wholly out (a summed-area table counts its px)."""
    h, w = mask.shape
    sat = np.zeros((h + 1, w + 1), np.int64)
    sat[1:, 1:] = mask.astype(np.int64).cumsum(0).cumsum(1)
    out, todo = [], [(0, 0, h, w)]
    while todo:
        y0, x0, y1, x1 = todo.pop()
        n = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
        if n == 0:
            continue
        if n == (y1 - y0) * (x1 - x0):
            out.append((y0, x0, y1 - y0, x1 - x0))
            continue
        ym, xm = (y0 + y1) // 2, (x0 + x1) // 2
        for a, b, c, d in ((y0, x0, ym, xm), (y0, xm, ym, x1), (ym, x0, y1, xm), (ym, xm, y1, x1)):
            if c > a and d > b:
                todo.append((a, b, c, d))
    return sorted(out)


def word_ids(sd: data.SheetData) -> np.ndarray:
    """1 + the Terra word whose rectangle holds each px of the page (later over earlier), 0 on none: the word
    targets' tag_id over the whole page (the cores are the words' own rectangles, `shrink.prepare`)."""
    from sfzmaps.solver.wordread import shrink
    return data._cores(shrink.prepare([sd])[0], 0, 0, sd.height, sd.width)


# --- the word model's spreads and prior ------------------------------------------------------------------------------

_SHARED: dict = {}      # read once before the workers fork


def spread_pages(sources, sheets, blocks, sheet_fn=load_sheet, outputs_fn=outputs):
    """`words.measure_spreads`' pages: of each source (run folder, quarter numbers) and sheet, the windows of the
    page the fit may read (`fit_area`, `windows`) with the word heads' geo and tag there, Terra's word ids and its
    rectangles (moved into the window)."""
    for run, quarters in sources:
        for name in sheets:
            sd = sheet_fn(name)
            area = fit_area(sd, quarters, blocks)
            if not area.any():
                continue
            o = outputs_fn(run, name)
            wid = word_ids(sd)
            rects = sd.words.astype(np.float64)
            for y0, x0, h, w in windows(area):
                win = (slice(y0, y0 + h), slice(x0, x0 + w))
                yield {"geo": o.geo[(slice(None), *win)], "tag": o.tag[(slice(None), *win)], "word_id": wid[win],
                       "words": rects - [x0, y0]}


def masked_inputs(o, area: np.ndarray, wid: np.ndarray | None = None):
    """The word outputs with the px outside `area` not a word's (core logit `predict.NOT_SCORED`, so their votes
    weigh nothing: chance 0 in double precision) and, with `wid`, no Terra word there."""
    from sfzmaps.solver.textnet import predict
    core = np.where(area, o.core, np.float32(predict.NOT_SCORED)).astype(np.float32)
    return core, (None if wid is None else np.where(area, wid, 0))


def reach_window(area: np.ndarray, geo: np.ndarray) -> tuple[int, int, int, int]:
    """The window (y0, x0, y1, x1) of the page that holds every box the votes of `area` can make: the area's
    bounding box grown by the reach of its votes (`geo` as `predict.WordOutputs` holds it: a vote's box is centred
    its offset (geo 0-1) from its px, of its length and height (geo 2-3) at any axis, so it lies within |offset| +
    half its diagonal of the px; the reach is the most of that over the area), cut to the page.  Outside it no vote
    is the area's, every one weighs 0, so the words formed in it are the page's (to the EMs' numerical stopping)."""
    ys, xs = np.nonzero(area)
    h, w = area.shape
    if not len(ys):
        return 0, 0, 0, 0
    g = geo[:4, ys, xs].astype(np.float64)
    reach = int(np.ceil(float((np.hypot(g[0], g[1]) + 0.5 * np.hypot(g[2], g[3])).max())))
    return (max(int(ys.min()) - reach, 0), max(int(xs.min()) - reach, 0), min(int(ys.max()) + 1 + reach, h),
            min(int(xs.max()) + 1 + reach, w))


def sheet_decisions(run, name, quarters, spreads, prior, blocks, sheet_fn=load_sheet, outputs_fn=outputs):
    """`words.prior_decisions` on one page of one source, the px outside the fit's area not a word's.  A proposal
    made only of such px (every vote's chance exactly 0: its gain is exactly 0 and its label 0, the background
    explaining it alone) is no decision in the area and is left out; any proposal holding a px of the area stays."""
    from sfzmaps.solver.wordread import words
    import pandas as pd
    sd = sheet_fn(name)
    area = fit_area(sd, quarters, blocks)
    if not area.any():
        return pd.DataFrame(columns=["kind", "gain", "label", "sheet", "source"])
    o = outputs_fn(run, name)
    core, wid = masked_inputs(o, area, word_ids(sd))
    y0, x0, y1, x1 = reach_window(area, o.geo)
    win = (slice(y0, y1), slice(x0, x1))
    d = words.prior_decisions(core[win], o.geo[(slice(None), *win)], o.tag[(slice(None), *win)], spreads, wid[win],
                              prior)
    outside = d["kind"].isin(["proposal", "word"]) & (d["gain"] == 0.0) & (d["label"] == 0.0)
    d = d[~outside].reset_index(drop=True)
    d["sheet"], d["source"] = name, Path(run).name
    return d


def _decisions_job(args):
    import cv2
    cv2.setNumThreads(1)
    run, name, quarters, spreads, prior = args
    t0 = time.perf_counter()
    d = sheet_decisions(run, name, quarters, spreads, prior, _SHARED["blocks"], _SHARED["sheet_fn"],
                        _SHARED["outputs_fn"])
    log.info("%s %s: %d decisions (%.0f s)", Path(run).name, name, len(d), time.perf_counter() - t0)
    return d


def _pool_map(fn, jobs, workers):
    from multiprocessing import get_context
    if workers <= 1:
        return [fn(j) for j in jobs]
    with get_context("fork").Pool(min(workers, len(jobs))) as pool:
        return pool.map(fn, jobs, chunksize=1)


def fit(out: Path, sources, sheets=paths.SHEETS, workers: int = WORKERS, blocks="answered",
        sheet_fn=load_sheet, outputs_fn=outputs) -> dict:
    """The word model's spreads and prior (see the module) from `sources` ((run folder, quarter numbers) each: the
    run's predictions on those quarters of `sheets`), the answered blocks (`blocks`, a `data.GridArea`; "answered":
    `reader_stage`'s) never read.  Writes `out`/{spreads.json, prior.json, decisions.parquet}; returns {"spreads",
    "prior"}."""
    import pandas as pd
    from sfzmaps.solver.wordread import words
    if isinstance(blocks, str):
        blocks = reader_stage.load_block_area(reader_stage.held_blocks())
    sources = [(Path(r), tuple(int(q) for q in qs)) for r, qs in sources]
    t0 = time.perf_counter()
    spreads = words.measure_spreads(spread_pages(sources, sheets, blocks, sheet_fn, outputs_fn))
    measured = time.perf_counter() - t0
    log.info("spreads %s (%.0f s)", spreads, measured)
    _SHARED.update(blocks=blocks, sheet_fn=sheet_fn, outputs_fn=outputs_fn)
    jobs = [(run, name, qs) for run, qs in sources for name in sheets]
    t0 = time.perf_counter()
    first = pd.concat(_pool_map(_decisions_job, [(*j, spreads, None) for j in jobs], workers), ignore_index=True)
    first_prior = words.fit_word_prior(first["gain"], first["label"])
    log.info("first prior %s from %d word decisions (%.0f s)", first_prior, len(first), time.perf_counter() - t0)
    t0 = time.perf_counter()
    every = pd.concat(_pool_map(_decisions_job, [(*j, spreads, first_prior) for j in jobs], workers),
                      ignore_index=True)
    prior = words.fit_word_prior(every["gain"], every["label"])
    log.info("prior %s from %d decisions (%.0f s)", prior, len(every), time.perf_counter() - t0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spreads.json").write_text(json.dumps(dataclasses.asdict(spreads), indent=1) + "\n")
    (out / "prior.json").write_text(json.dumps({
        "prior": dataclasses.asdict(prior), "first_prior": dataclasses.asdict(first_prior),
        "sources": [[str(r), [paths.QUARTERS[q] for q in qs]] for r, qs in sources],
        "decisions": {k: int(v) for k, v in every["kind"].value_counts().items()},
        "first_decisions": len(first), "seconds": {"spreads": round(measured, 1)}}, indent=1) + "\n")
    pd.concat([first.assign(round=1), every.assign(round=2)], ignore_index=True).to_parquet(
        out / "decisions.parquet", index=False)
    return {"spreads": spreads, "prior": prior}


def load_fit(folder: Path):
    """`fit`'s spreads and prior back from its folder."""
    from sfzmaps.solver.wordread import words
    sp = json.loads((Path(folder) / "spreads.json").read_text())
    sp = {k: tuple(v) if isinstance(v, list) else v for k, v in sp.items()}      # JSON's lists: the tuples back
    return words.Spreads(**sp), words.WordPrior(**json.loads((Path(folder) / "prior.json").read_text())["prior"])


# --- the words of the held-out quarter, placed -------------------------------------------------------------------

def wordboxes_placer(name: str, centres: np.ndarray) -> dict:
    """`wordboxes.place` of a 2000 sheet's box centres, with its placement and the lots and mapped area read once."""
    from sfzmaps.solver import lots, page, wordboxes
    if "lots" not in _SHARED:
        _SHARED["lots"], _SHARED["mapped"] = lots.load_lots(), page.load_sheets() > 0
    transform, footprint = wordboxes.placement(name)
    return wordboxes.place(centres, name, transform, footprint, _SHARED["lots"], _SHARED["mapped"])


def wordboxes_corner_cols() -> list[str]:
    from sfzmaps.solver import wordboxes
    return list(wordboxes.CORNER_COLS)


def sheet_words(run, name: str, q: int, spreads, prior, sheet_fn=load_sheet, outputs_fn=outputs,
                placer=wordboxes_placer):
    """One page's words in quarter `q` (see the module): `core_words` on its word outputs with the px outside the
    quarter not a word's, placed (`placer`), the boxes whose map point is in the quarter kept.  The table, and its
    counts."""
    from sfzmaps.solver.wordread import words
    sd = sheet_fn(name)
    inq = reader_stage.page_quarters(sd) == q
    if not inq.any():
        return words._table([]), {"formed": 0, "off_map": 0, "other_quarter": 0, "kept": 0}
    o = outputs_fn(run, name)
    core, _ = masked_inputs(o, inq)
    y0, x0, y1, x1 = reach_window(inq, o.geo)
    win = (slice(y0, y1), slice(x0, x1))
    t = words.core_words(core[win], o.geo[(slice(None), *win)], o.tag[(slice(None), *win)], data.page_ink(sd)[win],
                         spreads, prior, kind_p=o.kind_p[(slice(None), *win)], sheet=name)
    for c in ("page_x", *(k for k in wordboxes_corner_cols() if k.endswith("x"))):
        t[c] += x0
    for c in ("page_y", *(k for k in wordboxes_corner_cols() if k.endswith("y"))):
        t[c] += y0
    placed = placer(name, t[["page_x", "page_y"]].to_numpy(float))
    kept = t.iloc[np.asarray(placed["i"], int)].copy()
    for c in ("quarter", "x", "y", "lab_lot", "street"):
        kept[c] = np.asarray(placed[c])
    kept = kept.astype({"quarter": "int8", "lab_lot": "int32", "street": bool, "x": "float64", "y": "float64"})
    mine = kept[kept["quarter"] == q]
    return mine, {"formed": len(t), "off_map": len(t) - len(kept), "other_quarter": len(kept) - len(mine),
                  "kept": len(mine)}


def _words_job(args):
    import cv2
    cv2.setNumThreads(1)
    run, name, q, spreads, prior = args
    t0 = time.perf_counter()
    t, counts = sheet_words(run, name, q, spreads, prior, _SHARED["sheet_fn"], _SHARED["outputs_fn"],
                            _SHARED["placer"])
    counts["seconds"] = round(time.perf_counter() - t0, 1)
    log.info("%s %s: %s", Path(run).name, name, counts)
    return name, t, counts


def form_words(run: Path, quarter: str, spreads, prior, sheets=paths.SHEETS, workers: int = WORKERS,
               sheet_fn=load_sheet, outputs_fn=outputs, placer=wordboxes_placer, add: bool = False) -> Path:
    """Every sheet's words in the run's held-out quarter (`sheet_words`), written to `run`/boxes/boxes.parquet
    (sorted by sheet and box number; wordboxes.COLUMNS then words.EXTRA_COLUMNS) with summary.json (each sheet's
    counts and the spreads and prior used).  With `add`, the boxes of the other sheets already there are kept (made
    with the same spreads and prior, else ValueError).  Returns the folder."""
    import pandas as pd
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.wordread import words
    q = paths.QUARTERS.index(quarter)
    if placer is wordboxes_placer and "lots" not in _SHARED:
        from sfzmaps.solver import lots, page
        _SHARED["lots"], _SHARED["mapped"] = lots.load_lots(), page.load_sheets() > 0
    _SHARED.update(sheet_fn=sheet_fn, outputs_fn=outputs_fn, placer=placer)
    folder = Path(run) / "boxes"
    kept, counts = [], {}
    if add and (folder / "boxes.parquet").is_file():
        before = json.loads((folder / "summary.json").read_text())
        made = json.loads(json.dumps([dataclasses.asdict(spreads), dataclasses.asdict(prior)]))
        if [before["spreads"], before["prior"]] != made:
            raise ValueError(f"{folder} holds boxes made with other spreads or prior: make them all again")
        old = pd.read_parquet(folder / "boxes.parquet")
        kept = [old[~old["sheet"].isin(list(sheets))]]
        counts = {k: v for k, v in before["sheets"].items() if k not in sheets}
    got = _pool_map(_words_job, [(Path(run), name, q, spreads, prior) for name in sheets], workers)
    tables = [t for t in kept if len(t)] + [t for _, t, _ in got if len(t)]
    boxes = (pd.concat(tables, ignore_index=True) if tables else words._table([]))
    boxes = boxes.sort_values(["sheet", "i"], kind="stable").reset_index(drop=True)[
        [*wordboxes.COLUMNS, *words.EXTRA_COLUMNS]]
    folder.mkdir(parents=True, exist_ok=True)
    boxes.to_parquet(folder / "boxes.parquet", index=False)
    (folder / "summary.json").write_text(json.dumps({
        "quarter": quarter, "boxes": len(boxes),
        "sheets": dict(sorted({**counts, **{n: c for n, _, c in got}}.items())),
        "spreads": dataclasses.asdict(spreads), "prior": dataclasses.asdict(prior)}, indent=1) + "\n")
    return folder


# --- reading and measuring ----------------------------------------------------------------------------------------

def read(run: Path, workers: int = WORKERS) -> Path:
    """Every box of `run`/boxes read by the current fold readers (data/solver/text-train), as the stage reads
    `solver-boxes`' boxes (`textread.read.build_read`), into `run`/read (a text-read folder)."""
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.textread import read as R
    from sfzmaps.solver.textread import store
    old = wordboxes.BOXES_DIR, store.READ_DIR
    wordboxes.BOXES_DIR, store.READ_DIR = Path(run) / "boxes", Path(run) / "read"
    try:
        R.build_read(workers)
    finally:
        wordboxes.BOXES_DIR, store.READ_DIR = old
    return Path(run) / "read"


def run_boxes(run: Path):
    """The run's boxes, indexed by their row (the order `read` keys them by)."""
    import pandas as pd
    b = pd.read_parquet(Path(run) / "boxes" / "boxes.parquet").reset_index(drop=True)
    b["row"] = np.arange(len(b))
    return b.set_index("row", drop=False)


def measure(run: Path, quarter: str, reads_dir: Path | None = None, terra=None) -> dict:
    """The run's measures in its quarter (see the module): headline.json, labels.parquet (each answered label right
    or wrong, for the paired intervals), box_numbers.json and read_numbers.json in `run`.  `reads_dir`: the
    text-read folder (`run`/read when None)."""
    from sfzmaps.solver.wordread import measures, review
    q = paths.QUARTERS.index(quarter)
    reads_dir = Path(reads_dir or Path(run) / "read")
    boxes = run_boxes(run)
    reads = measures.reads_from_store(reads_dir)
    answers = measures.answers_from_store(reads_dir)
    answers = answers[answers["quarter"] == q].reset_index(drop=True)
    terra = measures.terra_words(paths.SHEETS) if terra is None else terra
    terra = terra[terra["quarter"] == q].reset_index(drop=True)
    head = measures.headline(boxes, reads, answers, paths.QUARTERS)
    measures.label_right(boxes, reads, answers).to_parquet(Path(run) / "labels.parquet", index=False)
    bn = measures.box_numbers(boxes, terra, reads)
    for v in bn.values():
        v.pop("by_quarter", None)
    forms = review.load_forms()
    rn = {str(iou): measures.read_numbers(boxes, terra, reads, forms=forms, answers=answers, iou=iou)
          for iou in (0.5, 0.3)}
    for f, obj in (("headline.json", head), ("box_numbers.json", bn), ("read_numbers.json", rn)):
        (Path(run) / f).write_text(json.dumps(obj, indent=1, default=float) + "\n")
    return {"headline": head, "box_numbers": bn, "read_numbers": rn}


# --- the review page -------------------------------------------------------------------------------------------------

WORD_REASONS = {
    "below_level": "most of its ink has a word-core chance below 0.5: the word heads saw no word there",
    "merged_with_word": "the box holding most of its core ink also holds another Terra word's ink",
    "merged_into_line": "the box holding it, holding no other word, is more than 3x the word's length",
    "split": "its core ink fell in two or more of our boxes",
    "wrong_size": "one box holds it, alone and not a line, but the box doesn't fit the word",
}
NOT_TRAINED = "fold not yet trained"


def today_numbers(sheet: str, quarter: str) -> dict:
    """Today's page numbers (data/solver/boxes, text-read, the fold detectors' chance and levels) on `sheet`,
    restricted to `quarter`'s boxes and Terra words: the "before" of a page showing one quarter.  Written to
    review/today-<quarter>/numbers.json (no page)."""
    import cv2
    from sfzmaps.solver import grid, wordboxes
    from sfzmaps.solver.textread import store
    from sfzmaps.solver.wordread import measures, review
    q = paths.QUARTERS.index(quarter)
    f = review.experiments_dir() / f"today-{quarter}" / "numbers.json"
    if f.is_file():
        return json.loads(f.read_text())
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    boxes = boxes[(boxes["sheet"] == sheet) & (boxes["quarter"] == q)]
    terra = review.terra_on_sheet(sheet)
    terra = terra[terra["quarter"] == q].reset_index(drop=True)
    chance = cv2.imread(str(paths.READER_TEXT_DIR / sheet / "chance.png"), cv2.IMREAD_GRAYSCALE)
    ink = data.load_page(sheet) < grid.INK_BELOW
    on_map = review.on_the_map(sheet, terra[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2).mean(1))
    answers = measures.answers_from_store()
    answers = answers[(answers["sheet"] == sheet) & (answers["quarter"] == q)].reset_index(drop=True)
    d = review.page_data(sheet, boxes, review.reads_for_page(), terra, chance, None, ink=ink, on_map=on_map,
                         answers=answers, forms=review.load_forms(), run=f"today-{quarter}")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(d["numbers"], indent=1) + "\n")
    del store
    return d["numbers"]


def page_lines(run: Path, quarter: str, fit_note: str) -> list[str]:
    """The lines naming what drew the page."""
    info = json.loads((Path(run) / "run.json").read_text())
    rec = info["recipe"]
    from sfzmaps import config
    try:
        rel = Path(run).resolve().relative_to(config.PIPELINE_DIR.resolve())
    except ValueError:
        rel = Path(run)
    return [
        f"Red ink and our boxes in {quarter.upper()}: the detector with the word heads in {rel} (model.pt; "
        f"{rec['iterations']} iterations, crop {rec['crop']} px, half size {rec['half_size']}, label rule "
        f"{rec['labels']}, doubt {info.get('doubt', False)}, seed {info['seed']}), trained with {quarter.upper()} "
        "and the answered blocks held out.  Red: ink whose word-core chance is 0.5 or more.",
        f"Words: the vote mixture of the word heads (wordread.words.core_words); its spreads and prior {fit_note}.",
        "Readings: the current fold readers (data/solver/text-train, the reader that never saw the box's quarter), "
        "and their 0/180 classifiers; the missed words read at Terra's box by the same readers on the CPU.",
        f"The other quarters are drawn faded: {NOT_TRAINED} (no quarter is shown from a detector that trained on it).",
        "Terra: its saved word boxes and strings (text-on-every-sheet).  Numbers 'vs last run' are against today's "
        f"boxes and reads on this sheet's {quarter.upper()} only (review/today-{quarter}).",
    ]


def build_page(run: Path, name: str, quarter: str, fit_note: str, sheet: str | None = None,
               reads_dir: Path | None = None) -> Path:
    """The review page of the run's boxes in its quarter on the Mission sheet (see the module), written to
    review/<name>/ (review.html, numbers.json, missed.json) and added to review/runs.json."""
    from sfzmaps.solver import grid
    from sfzmaps.solver.textread import store
    from sfzmaps.solver.wordread import measures, review
    q = paths.QUARTERS.index(quarter)
    sheet = sheet or review.mission_sheet()
    reads_dir = Path(reads_dir or Path(run) / "read")
    boxes = run_boxes(run)
    boxes = boxes[(boxes["sheet"] == sheet) & (boxes["quarter"] == q)]
    terra = review.terra_on_sheet(sheet)
    terra = terra[terra["quarter"] == q].reset_index(drop=True)
    sd = load_sheet(sheet)
    quarters = reader_stage.page_quarters(sd)
    inq = quarters == q
    o = outputs(run, sheet)
    core = np.where(inq, np.round(255 / (1 + np.exp(-o.core.astype(np.float64)))), 0).astype(np.uint8)
    ink = data.load_page(sheet) < grid.INK_BELOW
    comps = review.box_components(boxes, core, ink)
    on_map = review.on_the_map(sheet, terra[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2).mean(1))
    missed = review.missed_words(boxes, terra, review.MATCH_IOU)
    terra_reads = review.terra_box_reads(sheet, terra, missed, store.TRAIN_DIR)
    answers = measures.answers_from_store(reads_dir)
    answers = answers[(answers["sheet"] == sheet) & (answers["quarter"] == q)].reset_index(drop=True)
    notes = []
    for k, qn in enumerate(paths.QUARTERS):
        ys, xs = np.nonzero(quarters[::16, ::16] == k)
        if k != q and len(ys):
            notes.append((16 * float(np.median(xs)), 16 * float(np.median(ys)), f"{qn.upper()}: {NOT_TRAINED}"))
    before = today_numbers(sheet, quarter)
    d = review.page_data(sheet, boxes, review.reads_for_page(reads_dir), terra, core, before, ink=ink,
                         levels={qn: review.TEXT_CHANCE for qn in paths.QUARTERS}, on_map=on_map,
                         terra_reads=terra_reads, answers=answers, forms=review.load_forms(),
                         models=page_lines(run, quarter, fit_note), run=f"{name} (before: today, {quarter} only)",
                         comps=comps, visible=inq, reason_text=WORD_REASONS, notes=notes)
    root = review.experiments_dir()
    folder = root / name
    out = review.render(d, folder / "review.html")
    (folder / "numbers.json").write_text(json.dumps(d["numbers"], indent=1) + "\n")
    (folder / "missed.json").write_text(json.dumps(d["missed"], indent=1, ensure_ascii=False) + "\n")
    index = root / "runs.json"
    order = json.loads(index.read_text()) if index.is_file() else []
    index.write_text(json.dumps([r for r in order if r != name] + [name], indent=1) + "\n")
    return out


# --- the command line ------------------------------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--quarter", default="ne")
    c.add_argument("--steps", type=int, default=CHECK_STEPS)
    c.add_argument("--then-train", default=None, help="train this arm's run after the check (same sheets)")
    c.add_argument("--seed", type=int, default=0)
    parsers = {k: sub.add_parser(k) for k in ("train", "fit", "words", "read", "measure", "page")}
    for p in parsers.values():
        p.add_argument("--name", required=True)
        p.add_argument("--quarter", required=True, choices=paths.QUARTERS)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--workers", type=int, default=WORKERS)
    parsers["train"].add_argument("--doubt", action="store_true")
    parsers["fit"].add_argument("--own", action="store_true",
                                help="this run's own predictions on its training quarters (Phase 1, optimistic)")
    parsers["fit"].add_argument("--from", dest="sources", nargs="*", default=[],
                                help="other folds' runs (folder names), each read on its own held-out quarter")
    parsers["words"].add_argument("--sheets", nargs="*", default=None,
                                  help="these 2000 sheets only (default every one)")
    parsers["words"].add_argument("--add", action="store_true", help="keep the other sheets' boxes already made")
    parsers["words"].add_argument("--fit-dir", type=Path, default=None,
                                  help="the spreads and prior from this folder (default the run's fit/)")
    parsers["page"].add_argument("--page-name", default=None)
    parsers["page"].add_argument("--fit-note", default="")
    a = ap.parse_args()
    if a.cmd == "check":
        sheets = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False)
        print(json.dumps({k: v for k, v in check(a.quarter, a.steps, sheets=sheets).items() if k != "recipe"},
                         indent=1))
        if a.then_train:
            print(train_run(a.then_train, a.quarter, a.seed, sheets=sheets))
        return
    run = run_dir(a.name, a.quarter, a.seed)
    q = paths.QUARTERS.index(a.quarter)
    if a.cmd == "train":
        print(train_run(a.name, a.quarter, a.seed, a.doubt))
    elif a.cmd == "fit":
        if a.own == bool(a.sources):
            sys.exit("fit: give --own or --from <runs>, one of the two")
        if a.own:
            sources = [(run, tuple(k for k in range(len(paths.QUARTERS)) if k != q))]
        else:
            sources = []
            for r in a.sources:
                other = json.loads((RUNS / r / "run.json").read_text())["quarter"]
                if other == a.quarter:
                    sys.exit(f"fit: {r} held out {other}, this fold's own quarter")
                sources.append((RUNS / r, (paths.QUARTERS.index(other),)))
        got = fit(run / "fit", sources, workers=a.workers)
        print(json.dumps({k: dataclasses.asdict(v) for k, v in got.items()}, indent=1))
    elif a.cmd == "words":
        spreads, prior = load_fit(a.fit_dir or run / "fit")
        print(form_words(run, a.quarter, spreads, prior, sheets=tuple(a.sheets or paths.SHEETS), workers=a.workers,
                         add=a.add))
    elif a.cmd == "read":
        print(read(run, a.workers))
    elif a.cmd == "measure":
        got = measure(run, a.quarter)
        print(json.dumps({"headline": {k: v for k, v in got["headline"].items() if k != "by_quarter"},
                          "boxes": got["box_numbers"]}, indent=1, default=float))
    elif a.cmd == "page":
        print(build_page(run, a.page_name or run.name, a.quarter, a.fit_note))


if __name__ == "__main__":
    main()
