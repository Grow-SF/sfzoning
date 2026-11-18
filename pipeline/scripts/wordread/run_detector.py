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


WORD_HEADS = ("words", "words_aff", "words_code")    # train.Recipe.heads with the word heads (+ affinity, + code)


def word_recipe(**changes) -> train.Recipe:
    """The stage's recipe with the word heads (`heads` "words" unless changed), the per-ink head by the label rule
    lines-unknown (ruling R2)."""
    return dataclasses.replace(stage.DEFAULT_RECIPE, **{"labels": "lines-unknown", "heads": "words", **changes})


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
    if recipe.heads not in WORD_HEADS:
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


def word_ids(sd: data.SheetData) -> np.ndarray:
    """1 + the Terra word whose rectangle holds each px of the page (later over earlier), 0 on none: the word
    targets' tag_id over the whole page (the cores are the words' own rectangles, `shrink.prepare`)."""
    from sfzmaps.solver.wordread import shrink
    return data._cores(shrink.prepare([sd])[0], 0, 0, sd.height, sd.width)


# --- the word model's spreads and prior ------------------------------------------------------------------------------

_SHARED: dict = {}      # read once before the workers fork


def spread_pages(sources, sheets, blocks, sheet_fn=load_sheet, outputs_fn=outputs, tally: dict | None = None):
    """`words.measure_spreads`' pages: of each source (run folder, quarter numbers) and sheet, the whole page's
    word heads' geo, tag and core logit, Terra's word ids and rectangles, and the px the fit may read as the page's mask
    (`fit_area`: a word with a px outside it is left out whole, the background is the mask's px in no word).
    `tally` collects, per source and sheet, the area's px and the Terra words wholly inside it and across its edge."""
    for run, quarters in sources:
        for name in sheets:
            sd = sheet_fn(name)
            area = fit_area(sd, quarters, blocks)
            if not area.any():
                continue
            o = outputs_fn(run, name)
            wid = word_ids(sd)
            rects = sd.words.astype(np.float64)
            if tally is not None:
                inside, outside = np.unique(wid[area & (wid > 0)]), np.unique(wid[~area & (wid > 0)])
                across = np.intersect1d(inside, outside)
                tally[f"{Path(run).name}/{name}"] = {"area_px": int(area.sum()),
                                                     "words": int(len(inside) - len(across)),
                                                     "across_edge": int(len(across))}
            yield {"geo": o.geo, "tag": o.tag, "core": o.core, "word_id": wid, "words": rects, "mask": area}


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
        sheet_fn=load_sheet, outputs_fn=outputs, rounds: int = 2) -> dict:
    """The word model's spreads and prior (see the module) from `sources` ((run folder, quarter numbers) each: the
    run's predictions on those quarters of `sheets`), the answered blocks (`blocks`, a `data.GridArea`; "answered":
    `reader_stage`'s) never read.  `rounds` 1: the first prior alone (a draft's; the search's decisions not taken).
    Writes `out`/{spreads.json, prior.json, decisions.parquet}; returns {"spreads", "prior"}."""
    import pandas as pd
    from sfzmaps.solver.wordread import words
    if isinstance(blocks, str):
        blocks = reader_stage.load_block_area(reader_stage.held_blocks())
    sources = [(Path(r), tuple(int(q) for q in qs)) for r, qs in sources]
    t0 = time.perf_counter()
    tally: dict = {}
    spreads = words.measure_spreads(spread_pages(sources, sheets, blocks, sheet_fn, outputs_fn, tally))
    measured = time.perf_counter() - t0
    log.info("spreads %s (%.0f s)", spreads, measured)
    _SHARED.update(blocks=blocks, sheet_fn=sheet_fn, outputs_fn=outputs_fn)
    jobs = [(run, name, qs) for run, qs in sources for name in sheets]
    t0 = time.perf_counter()
    first = pd.concat(_pool_map(_decisions_job, [(*j, spreads, None) for j in jobs], workers), ignore_index=True)
    first_prior = words.fit_word_prior(first["gain"], first["label"])
    log.info("first prior %s from %d word decisions (%.0f s)", first_prior, len(first), time.perf_counter() - t0)
    t0 = time.perf_counter()
    if rounds == 1:
        every, prior = first, first_prior
    else:
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
        "first_decisions": len(first), "seconds": {"spreads": round(measured, 1)},
        "spread_words": {"whole_in_area": sum(c["words"] for c in tally.values()),
                         "across_edge": sum(c["across_edge"] for c in tally.values()), "by_page": tally}},
        indent=1) + "\n")
    pd.concat([first.assign(round=1), every.assign(round=2)], ignore_index=True).to_parquet(
        out / "decisions.parquet", index=False)
    return {"spreads": spreads, "prior": prior}


def load_fit(folder: Path):
    """`fit`'s spreads and prior back from its folder."""
    from sfzmaps.solver.wordread import words
    def tuples(v):                                         # JSON's lists, nested too: the tuples back
        return tuple(tuples(x) for x in v) if isinstance(v, list) else v
    sp = {k: tuples(v) for k, v in json.loads((Path(folder) / "spreads.json").read_text()).items()}
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


def vs_today(run: Path, quarter: str | None, reads_dir: Path | None = None, terra=None, sheets=None) -> dict:
    """The run's boxes against today's (data/solver/boxes, read by the same current fold readers, text-read) in the
    quarter, over the same answered labels: both headlines, the paired difference (run minus today) with its 95%
    interval resampling whole blocks (`measures.paired_labels`), and both box numbers against Terra's words (on
    `sheets`, every 2000 sheet when None: give the sheets the run's boxes cover).  `quarter` None: every quarter (the
    headline over every answered label with a code, by quarter too).  Writes `run`/vs_today.json."""
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.textread import store
    from sfzmaps.solver.wordread import measures
    qs = list(range(len(paths.QUARTERS))) if quarter is None else [paths.QUARTERS.index(quarter)]
    reads_dir = Path(reads_dir or Path(run) / "read")
    sheets = list(sheets or paths.SHEETS)
    answers = measures.answers_from_store(reads_dir)
    answers = answers[answers["quarter"].isin(qs) & answers["sheet"].isin(sheets)].reset_index(drop=True)
    today_answers = measures.answers_from_store(store.READ_DIR)
    today_answers = today_answers[today_answers["quarter"].isin(qs)
                                  & today_answers["sheet"].isin(sheets)].reset_index(drop=True)
    if not answers[["sheet", "point_x", "point_y", "truth"]].equals(today_answers[["sheet", "point_x", "point_y",
                                                                                     "truth"]]):
        raise ValueError("the run's answered labels are not today's, in the same order")
    boxes = run_boxes(run)
    boxes = boxes[boxes["sheet"].isin(sheets)]
    reads = measures.reads_from_store(reads_dir)
    today = wordboxes.load_boxes().reset_index(drop=True)
    today["row"] = np.arange(len(today))
    today = today.set_index("row", drop=False)
    today = today[today["sheet"].isin(sheets)]           # every quarter's: the labels' holding box as today
    today_reads = measures.reads_from_store(store.READ_DIR)
    mine = measures.label_right(boxes, reads, answers)
    theirs = measures.label_right(today, today_reads, answers)
    terra = measures.terra_words(sheets) if terra is None else terra
    terra = terra[terra["quarter"].isin(qs) & terra["sheet"].isin(sheets)].reset_index(drop=True)
    out = {"quarter": quarter or "all", "sheets": sheets, "labels": int(len(mine)),
           "headline": measures.headline(boxes, reads, answers, paths.QUARTERS),
           "today_headline": measures.headline(today, today_reads, answers, paths.QUARTERS),
           "paired_run_minus_today": measures.paired_labels(mine, theirs),
           "boxes": measures.box_numbers(boxes, terra, reads)[0.5],
           "today_boxes": measures.box_numbers(today[today["quarter"].isin(qs)], terra, today_reads)[0.5]}
    if quarter is not None:
        for key in ("boxes", "today_boxes"):
            out[key].pop("by_quarter", None)
    else:
        out["paired_by_quarter"] = {qn: measures.paired_labels(mine[mine["quarter"] == k].reset_index(drop=True),
                                                               theirs[theirs["quarter"] == k].reset_index(drop=True))
                                    for k, qn in enumerate(paths.QUARTERS) if (mine["quarter"] == k).any()}
    (Path(run) / "vs_today.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def vs_run(run: Path, other: Path, quarter: str | None, terra=None, sheets=None) -> dict:
    """The run's boxes against another run's (each read by its own read store) over the same answered labels in the
    quarter (None: every quarter): both headlines, the paired difference (run minus other, `measures.paired_labels`,
    by quarter too with every quarter) and both box numbers against Terra's words.  Writes `run`/vs_<other>.json."""
    from sfzmaps.solver.wordread import measures
    qs = list(range(len(paths.QUARTERS))) if quarter is None else [paths.QUARTERS.index(quarter)]
    sheets = list(sheets or paths.SHEETS)
    sides = {}
    for key, r in (("run", Path(run)), ("other", Path(other))):
        ans = measures.answers_from_store(r / "read")
        ans = ans[ans["quarter"].isin(qs) & ans["sheet"].isin(sheets)].reset_index(drop=True)
        boxes = run_boxes(r)
        sides[key] = (ans, boxes[boxes["sheet"].isin(sheets)], measures.reads_from_store(r / "read"))
    cols = ["sheet", "point_x", "point_y", "truth"]
    if not sides["run"][0][cols].equals(sides["other"][0][cols]):
        raise ValueError(f"{Path(run).name} and {Path(other).name}: not the same answered labels in the same order")
    answers = sides["run"][0]
    right = {k: measures.label_right(b, rd, answers) for k, (_, b, rd) in sides.items()}
    terra = measures.terra_words(sheets) if terra is None else terra
    terra = terra[terra["quarter"].isin(qs) & terra["sheet"].isin(sheets)].reset_index(drop=True)
    out = {"quarter": quarter or "all", "run": Path(run).name, "other": Path(other).name, "labels": len(answers),
           "headline": measures.headline(sides["run"][1], sides["run"][2], answers, paths.QUARTERS),
           "other_headline": measures.headline(sides["other"][1], sides["other"][2], answers, paths.QUARTERS),
           "paired_run_minus_other": measures.paired_labels(right["run"], right["other"]),
           "boxes": measures.box_numbers(sides["run"][1], terra, sides["run"][2])[0.5],
           "other_boxes": measures.box_numbers(sides["other"][1][sides["other"][1]["quarter"].isin(qs)], terra,
                                               sides["other"][2])[0.5]}
    if quarter is None:
        mine, theirs = right["run"], right["other"]
        out["paired_by_quarter"] = {qn: measures.paired_labels(mine[mine["quarter"] == k].reset_index(drop=True),
                                                               theirs[theirs["quarter"] == k].reset_index(drop=True))
                                    for k, qn in enumerate(paths.QUARTERS) if (mine["quarter"] == k).any()}
    (Path(run) / f"vs_{Path(other).name}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


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


# --- the draft page: every quarter of one sheet from its own held-out detector --------------------------------------

DRAFT = ROOT / "draft"
DRAFT_FIT_PAGES = {"se": "2000-digital-zoning-p10", "nw": "2000-digital-zoning-p04", "sw": "2000-digital-zoning-p13"}
DRAFT_NOTE = ("DRAFT, not final: the word heads' centre votes are uninformative (each px votes for about itself); "
              "the spreads and the prior are provisional (measured and fitted on one page per source fold, the SE, "
              "NW and SW detectors each on its own held-out quarter, never NE or an answered block; the first prior "
              "only, from the proposals); Task 6's vote calibration (round 5) is pending")


def draft(sheet: str = "2000-digital-zoning-p07", name: str = "a", seed: int = 0, workers: int = WORKERS,
          fit_dir: Path | None = None, tag: str = "", note: str = DRAFT_NOTE) -> Path:
    """A draft of the review page for `sheet` (see DRAFT_NOTE): the spreads and the first prior on DRAFT_FIT_PAGES
    (or `fit_dir`'s), every quarter's words formed from its own held-out detector (the quarters in parallel), placed,
    read by the current fold readers, and the page written to review/draft-boxes/ (with `tag`, draft-<tag>/ and
    review/draft-boxes-<tag>/); `note` says on the page what is not final."""
    from sfzmaps.solver import lots, page
    out = DRAFT if not tag else ROOT / f"draft-{tag}"
    if fit_dir is None:
        fit_dir = out / "fit"
        if not (fit_dir / "prior.json").is_file():
            sources = [(run_dir(name, q, seed), (paths.QUARTERS.index(q),)) for q in DRAFT_FIT_PAGES]
            fit(fit_dir, sources, sheets=tuple(DRAFT_FIT_PAGES.values()), workers=min(workers, 3), rounds=1)
    spreads, prior = load_fit(fit_dir)
    sd = load_sheet(sheet)
    present = np.unique(reader_stage.page_quarters(sd))
    runs_ = {int(k): run_dir(name, paths.QUARTERS[k], seed) for k in present}
    _SHARED["lots"], _SHARED["mapped"] = lots.load_lots(), page.load_sheets() > 0
    _SHARED.update(sheet_fn=load_sheet, outputs_fn=outputs, placer=wordboxes_placer)
    t0 = time.perf_counter()
    got = _pool_map(_words_job, [(runs_[k], sheet, k, spreads, prior) for k in runs_], workers)
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.wordread import words
    import pandas as pd
    tables = [t for _, t, _ in got if len(t)]
    boxes = pd.concat(tables, ignore_index=True) if tables else words._table([])
    boxes = boxes.sort_values(["quarter", "i"], kind="stable").reset_index(drop=True)[
        [*wordboxes.COLUMNS, *words.EXTRA_COLUMNS]]
    (out / "boxes").mkdir(parents=True, exist_ok=True)
    boxes.to_parquet(out / "boxes" / "boxes.parquet", index=False)
    (out / "boxes" / "summary.json").write_text(json.dumps({
        "sheet": sheet, "boxes": len(boxes), "by_quarter": {paths.QUARTERS[k]: c for k, (_, _, c) in
                                                             zip(runs_, got)},
        "detectors": {paths.QUARTERS[k]: r.name for k, r in runs_.items()}, "fit": str(fit_dir),
        "seconds": round(time.perf_counter() - t0, 1)}, indent=1) + "\n")
    log.info("draft words: %d boxes (%.0f s)", len(boxes), time.perf_counter() - t0)
    read(out, min(workers, 4))
    return build_draft_page(out, sheet, runs_, "draft-boxes" + (f"-{tag}" if tag else ""), note)


def build_draft_page(folder: Path, sheet: str, runs_: dict, page_name: str = "draft-boxes",
                     note: str = DRAFT_NOTE) -> Path:
    """The draft page (see `draft`): every quarter's boxes from its own detector, red = ink whose word-core chance
    (from the px's quarter's detector) is 0.5 or more, the reasons read on the boxes' px (`review.box_components`),
    "before" = today's numbers on the whole sheet (review/baseline)."""
    from sfzmaps.solver import grid
    from sfzmaps.solver.textread import store
    from sfzmaps.solver.wordread import measures, review
    boxes = run_boxes(folder)
    boxes = boxes[boxes["sheet"] == sheet]
    terra = review.terra_on_sheet(sheet)
    sd = load_sheet(sheet)
    quarters = reader_stage.page_quarters(sd)
    core = np.zeros(quarters.shape, np.uint8)
    for k, run in runs_.items():
        mine = quarters == k
        o = outputs(run, sheet)
        core[mine] = np.round(255 / (1 + np.exp(-o.core[mine].astype(np.float64)))).astype(np.uint8)
        del o
    ink = data.load_page(sheet) < grid.INK_BELOW
    comps = review.box_components(boxes, core, ink)
    on_map = review.on_the_map(sheet, terra[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2).mean(1))
    missed = review.missed_words(boxes, terra, review.MATCH_IOU)
    terra_reads = review.terra_box_reads(sheet, terra, missed, store.TRAIN_DIR)
    answers = measures.answers_from_store(folder / "read")
    answers = answers[answers["sheet"] == sheet].reset_index(drop=True)
    root = review.experiments_dir()
    before = json.loads((root / "baseline" / "numbers.json").read_text())
    lines = [note + ".",
             "Red ink and our boxes: each quarter from its own held-out detector with the sign-free word heads (arm "
             "a, seed 0: " + ", ".join(f"{paths.QUARTERS[k].upper()} {r.name}" for k, r in runs_.items())
             + "), trained with that quarter and the answered blocks held out.  Red: ink whose word-core chance is 0.5 "
             "or more.",
             "Words: the vote mixture of the word heads (wordread.words.core_words), placed as today's boxes are.",
             "Readings: the current fold readers (data/solver/text-train) and their 0/180 classifiers; the missed "
             "words read at Terra's box by the same readers.",
             "Numbers 'vs last run' are against today's boxes and reads on the whole sheet (review/baseline)."]
    d = review.page_data(sheet, boxes, review.reads_for_page(folder / "read"), terra, core, before, ink=ink,
                         levels={qn: review.TEXT_CHANCE for qn in paths.QUARTERS}, on_map=on_map,
                         terra_reads=terra_reads, answers=answers, forms=review.load_forms(), models=lines,
                         run="DRAFT boxes from the word heads (not final; before: today)", comps=comps,
                         reason_text=WORD_REASONS)
    out = root / page_name
    page_path = review.render(d, out / "review.html", about=f"<h3>Draft</h3><p><b>{review._e(note)}.</b></p>")
    (out / "numbers.json").write_text(json.dumps(d["numbers"], indent=1) + "\n")
    (out / "missed.json").write_text(json.dumps(d["missed"], indent=1, ensure_ascii=False) + "\n")
    b = d["numbers"]["boxes"]["0.5"]
    print(json.dumps({"page": str(page_path), "mb": round(page_path.stat().st_size / 1e6, 2),
                      "boxes": {k: b[k] for k in ("terra", "ours", "matched", "missed", "missed_reasons",
                                                  "ours_unmatched_text", "ours_unmatched_none")},
                      "headline": d["numbers"].get("headline", {}).get("share")}, indent=1))
    return page_path


# --- crude-map: words as the core head's even-odds components (a lower-bound check; no tags, no votes) -------------

CRUDE_NOTE = ("crude-map: each word is a connected component (8 px around) of the px the core head gives even odds "
              "or better (P(core) > 1/2, the MAP decision, no picked level); its box the smallest rectangle of the "
              "component's px along the core-weighted mean of the learned axis; no tags and no votes, so touching "
              "words merge: a lower-bound check of whether the new detector's boxes can move the headline")


# --- crude-tags: crude-map's components split by their tags --------------------------------------------------------

TAGS_NOTE = ("crude-tags: crude-map's components (P(core) > 1/2, 8 px around) each split by the word tags: mean-shift "
             "in tag space with a Gaussian kernel whose sd is the within-word tag spread measured for that detector on "
             "its own training quarters' Terra words (answered blocks out); each cluster's 8-connected pieces one "
             "word, "
             "its box the smallest rectangle along its px' core-weighted mean axis; no votes")
MS_STEPS = 500          # numerical only: mean-shift stops after this many steps ...
MS_TOL = 1e-4           # ... or once no point moves by this share of the bandwidth
MS_SAME = 1e-2          # numerical only: points whose modes lie within this share of the bandwidth share a mode (a
#                         point stops within MS_TOL of its mode; two modes of a Gaussian kernel density lie at least
#                         about a bandwidth apart, so any share between the two gives the same clusters)
BIN = 0.5               # numerical only: tags are pooled in cells of this share of the bandwidth (their weighted means
#                         the data), so the kernel's sums are over cells, not px


def tag_spread(run, quarters, blocks, sheets=paths.SHEETS, sheet_fn=load_sheet, outputs_fn=outputs) -> dict:
    """The within-word tag spread of a detector on `quarters` (its training quarters) outside the answered blocks: the
    root mean square, per channel, of each Terra word's core px' tags about the word's core-weighted mean tag
    (weighted by core chance), over every word with a px in the area."""
    num = den = 0.0
    words_n = 0
    for name in sheets:
        sd = sheet_fn(name)
        area = fit_area(sd, quarters, blocks)
        wid = word_ids(sd)
        on = area & (wid > 0)
        if not on.any():
            continue
        o = outputs_fn(run, name)
        ys, xs = np.nonzero(on)
        j = wid[ys, xs]
        order = np.argsort(j, kind="stable")
        j, ys, xs = j[order], ys[order], xs[order]
        start = np.flatnonzero(np.r_[True, j[1:] != j[:-1]])
        w = 1 / (1 + np.exp(-o.core[ys, xs].astype(np.float64)))
        t = o.tag[:, ys, xs].astype(np.float64)
        sw = np.add.reduceat(w, start)
        cnt = np.diff(np.r_[start, len(j)])
        mean = np.stack([np.add.reduceat(w * t[c], start) / sw for c in range(len(t))])
        r = t - np.repeat(mean, cnt, axis=1)
        num += float((w * (r ** 2).sum(0)).sum())
        den += float(w.sum()) * len(t)
        words_n += len(start)
    return {"sigma": float(np.sqrt(num / den)), "words": words_n}


def mean_shift(points: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """Cluster labels (0, 1, ...) of `points` (n, k) by Gaussian mean-shift of bandwidth `sigma` over the weighted
    points: every point climbs to its mode; points whose modes coincide (MS_SAME) share a label."""
    x = points.copy()
    for _ in range(MS_STEPS):
        d2 = ((x[:, None, :] - points[None, :, :]) ** 2).sum(-1)
        kern = weights[None, :] * np.exp(-d2 / (2 * sigma ** 2))
        new = kern @ points / kern.sum(1, keepdims=True)
        moved = float(np.abs(new - x).max())
        x = new
        if moved < MS_TOL * sigma:
            break
    labels = np.full(len(x), -1)
    k = 0
    for i in range(len(x)):
        if labels[i] >= 0:
            continue
        same = (np.abs(x - x[i]).max(1) < MS_SAME * sigma) & (labels < 0)
        labels[same] = k
        k += 1
    return labels


def split_by_tags(lab: np.ndarray, tag: np.ndarray, w_core: np.ndarray, sigma: float) -> np.ndarray:
    """crude-tags' words: each component of `lab` (int, 0 off them) split by mean-shift of its px' tags (pooled in
    cells of BIN bandwidths, weighted by core chance), each cluster's 8-connected pieces one word; a new label image
    (0 off the words)."""
    import cv2
    out = np.zeros_like(lab)
    ys, xs = np.nonzero(lab)
    comp = lab[ys, xs]
    order = np.argsort(comp, kind="stable")
    ys, xs, comp = ys[order], xs[order], comp[order]
    start = np.r_[np.flatnonzero(np.r_[True, comp[1:] != comp[:-1]]), len(comp)]
    nxt = 1
    for a, b in zip(start[:-1], start[1:]):
        yy, xx = ys[a:b], xs[a:b]
        t = tag[:, yy, xx].T.astype(np.float64)
        w = w_core[yy, xx].astype(np.float64)
        cell = np.floor(t / (BIN * sigma)).astype(np.int64)
        keys, inv = np.unique(cell, axis=0, return_inverse=True)
        inv = inv.ravel()
        cw = np.bincount(inv, w, len(keys))
        cm = np.stack([np.bincount(inv, w * t[:, c], len(keys)) for c in range(t.shape[1])], 1) / cw[:, None]
        cl = mean_shift(cm, cw, sigma)[inv] if len(keys) > 1 else np.zeros(len(inv), int)
        y0, x0 = yy.min(), xx.min()
        for c in np.unique(cl):
            sel = cl == c
            m = np.zeros((yy.max() - y0 + 1, xx.max() - x0 + 1), np.uint8)
            m[yy[sel] - y0, xx[sel] - x0] = 1
            n, pieces = cv2.connectedComponents(m, connectivity=8)
            for p in range(1, n):
                py, px = np.nonzero(pieces == p)
                out[py + y0, px + x0] = nxt
                nxt += 1
    return out


# --- crude-aff: the core px joined by the learned same-word affinity ------------------------------------------------

AFF_NOTE = ("crude-aff: the px the core head gives even odds or better (P(core) > 1/2) joined to their right and "
            "lower neighbours where the affinity head gives even odds or better that the two are of one word "
            "(P(same) > 1/2, the even-odds call); each word a connected component of that graph (4 px around: the "
            "head has no diagonal pairs), its box the smallest rectangle along its px' core-weighted mean axis; no "
            "tags and no votes")
AFF_FIELD = "aff"       # predict.WordOutputs' field of the affinity logits (2, h, w): to its right, to the px below


def aff_labels(on: np.ndarray, right: np.ndarray, down: np.ndarray) -> np.ndarray:
    """crude-aff's words: the connected components of the graph whose nodes are the `on` px and whose edges join a
    px to its right neighbour where `right` (logit) > 0 and to the px below where `down` > 0, both ends on; a label
    image (0 off them, 1 ... in the order of each component's first px, row by row)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    h, w = on.shape
    idx = np.full((h, w), -1, np.int64)
    n = int(on.sum())
    idx[on] = np.arange(n)
    er = on[:, :-1] & on[:, 1:] & (right[:, :-1] > 0)
    ed = on[:-1] & on[1:] & (down[:-1] > 0)
    a = np.r_[idx[:, :-1][er], idx[:-1][ed]]
    b = np.r_[idx[:, 1:][er], idx[1:][ed]]
    _, comp = connected_components(coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n)), directed=False)
    first = np.full(comp.max() + 1 if n else 0, n, np.int64)
    np.minimum.at(first, comp, np.arange(n))
    rank = np.empty(len(first), np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    out = np.zeros((h, w), np.int32)
    out[on] = rank[comp] + 1
    return out


def aff_counts(run, quarters, blocks, sheets=paths.SHEETS, sheet_fn=load_sheet, outputs_fn=outputs) -> dict:
    """The affinity's taught px pairs (right and down neighbours) on `quarters` (a detector's training quarters)
    outside the answered blocks, both px in that area: "same" both px of one Terra word, "apart" a word px beside the
    background or another word (background beside background is not taught, as in training); all of them, and those
    whose two px are both core px (P(core) > 1/2, as crude-aff reads them).  With the log odds log(same / apart) of
    each: the class prior the balanced loss leaves out (its logit is the odds under equal classes)."""
    n = {"same": 0, "apart": 0, "same_core": 0, "apart_core": 0}
    for name in sheets:
        sd = sheet_fn(name)
        area = fit_area(sd, quarters, blocks)
        if not area.any():
            continue
        wid = word_ids(sd)
        on = outputs_fn(run, name).core > 0
        for sa, sb in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
            a, b = wid[sa], wid[sb]
            both = area[sa] & area[sb]
            core = on[sa] & on[sb]
            same = both & (a > 0) & (a == b)
            apart = both & ((a > 0) | (b > 0)) & (a != b)
            n["same"] += int(same.sum())
            n["apart"] += int(apart.sum())
            n["same_core"] += int((same & core).sum())
            n["apart_core"] += int((apart & core).sum())
    n["log_odds"] = float(np.log(n["same"] / n["apart"]))
    n["log_odds_core"] = float(np.log(n["same_core"] / n["apart_core"]))
    return n


# --- crude-kind: crude-map's components split where the kind head's MAP kind changes --------------------------------

KIND_NOTE = ("crude-kind: the px the core head gives even odds or better (P(core) > 1/2) joined to their 8 neighbours "
             "only when both have the same MAP kind (the argmax of the per-ink head's kinds: zoning code, block "
             "number, street name, other; with the nearest-ink variant, each px takes the MAP kind of its nearest "
             "ink px, an exact Euclidean distance transform); each word a connected component of that graph, its box "
             "the smallest rectangle along its px' core-weighted mean axis; no tags, no votes, no threshold")
KIND_MODES = ("map", "nearest-ink")


def kind_map(kind_p: np.ndarray, ink: np.ndarray, mode: str = "map") -> np.ndarray:
    """Each px's MAP kind (uint8 (h, w), `data.KINDS`' order: the argmax of `kind_p` (4, h, w), the softmax of the
    per-ink head's kind channels 1-4); with `mode` "nearest-ink", each px's kind is that of the ink px nearest it (exact
    Euclidean distance; ties as `scipy.ndimage.distance_transform_edt` breaks them), the ink's own kept."""
    from scipy import ndimage
    if mode not in KIND_MODES:
        raise ValueError(f"kind mode is one of {', '.join(KIND_MODES)}, not {mode!r}")
    k = np.argmax(kind_p, axis=0).astype(np.uint8)
    if mode == "map" or not ink.any():
        return k
    _, (iy, ix) = ndimage.distance_transform_edt(~ink, return_indices=True)
    return k[iy, ix]


def kind_labels(on: np.ndarray, kinds: np.ndarray) -> np.ndarray:
    """crude-kind's words: the connected components (8 px around) of the `on` px joined only to neighbours of the same
    kind (`kinds`), a label image (int32, 0 off them): each kind's own components, numbered kind by kind."""
    import cv2
    out = np.zeros(on.shape, np.int32)
    nxt = 0
    for k in np.unique(kinds[on]):
        n, lab = cv2.connectedComponents((on & (kinds == k)).astype(np.uint8), connectivity=8)
        sel = lab > 0
        out[sel] = lab[sel] + nxt
        nxt += n - 1
    return out


def kind_check(name: str = "a", quarter: str = "ne", seed: int = 0) -> dict:
    """The kind head's MAP kind against Terra's on the held-out quarter's px inside Terra's words (a word's px as
    `data._paint` fills its rectangle, the later word's where two meet): on the ink px, on the core px that are not
    ink (P(core) > 1/2), and each mode of `kind_map`; per kind of Terra word too.  Writes runs/kind-check-<run>.json."""
    from sfzmaps.solver.textnet import data as D
    q = paths.QUARTERS.index(quarter)
    run = run_dir(name, quarter, seed)
    tally = {m: {w: np.zeros((4, 4), np.int64) for w in ("ink", "core_not_ink")} for m in KIND_MODES}
    for sheet in paths.SHEETS:
        sd = load_sheet(sheet)
        inq = reader_stage.page_quarters(sd) == q
        if not inq.any():
            continue
        word = D._words(sd, 0, 0, sd.height, sd.width)
        terra = np.concatenate([[255], sd.kinds]).astype(np.uint8)[word]
        o = outputs(run, sheet)
        ink = data.page_ink(sd)
        at = {"ink": inq & (word > 0) & ink, "core_not_ink": inq & (word > 0) & ~ink & (o.core > 0)}
        for m in KIND_MODES:
            k = kind_map(o.kind_p, ink, m)
            for w, sel in at.items():
                np.add.at(tally[m][w], (terra[sel].astype(int), k[sel].astype(int)), 1)
    names = list(D.KINDS)
    out = {"run": run.name, "quarter": quarter}
    for m in KIND_MODES:
        for w, c in tally[m].items():
            out[f"{m}/{w}"] = {"px": int(c.sum()), "agree": round(float(np.trace(c) / max(c.sum(), 1)), 4),
                               "by_terra_kind": {names[i]: {"px": int(c[i].sum()),
                                                            "agree": round(float(c[i, i] / max(c[i].sum(), 1)), 4)}
                                                 for i in range(4)},
                               "confusion_rows_terra_cols_map": c.tolist()}
    (RUNS / f"kind-check-{run.name}.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def crude_map_words(run, name: str, q: int, sheet_fn=load_sheet, outputs_fn=outputs, placer=wordboxes_placer,
                    sigma: float | None = None, aff: bool = False, aff_prior: float = 0.0,
                    kind: str | None = None):
    """One page's crude-map words in quarter `q` (CRUDE_NOTE; with `sigma`, the tag bandwidth, crude-tags'
    (TAGS_NOTE); with `aff`, crude-aff's (AFF_NOTE), the affinity logits plus `aff_prior` (the class prior's log odds,
    `aff_counts`); with `kind` (a `kind_map` mode), crude-kind's (KIND_NOTE)), placed as `wordboxes` places its
    boxes; the boxes whose map point is in the quarter kept.  The
    table (wordboxes.COLUMNS) and its counts."""
    import cv2
    import pandas as pd
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.wordread import words
    sd = sheet_fn(name)
    inq = reader_stage.page_quarters(sd) == q
    o = outputs_fn(run, name)
    on = inq & (o.core > 0)                      # sigmoid(core) > 1/2
    if not on.any():
        return pd.DataFrame(columns=wordboxes.COLUMNS), {"formed": 0, "off_map": 0, "other_quarter": 0, "kept": 0}
    ink = data.page_ink(sd)
    if aff:
        same = getattr(o, AFF_FIELD)
        lab = aff_labels(on, same[0] + aff_prior, same[1] + aff_prior)
    elif kind is not None:
        lab = kind_labels(on, kind_map(o.kind_p, ink, kind))
    else:
        _, lab = cv2.connectedComponents(on.astype(np.uint8), connectivity=8)
    if sigma is not None:
        lab = split_by_tags(lab, o.tag, 1 / (1 + np.exp(-o.core.astype(np.float64))), sigma)
    ys, xs = np.nonzero(lab)
    comp = lab[ys, xs]
    order = np.argsort(comp, kind="stable")
    ys, xs, comp = ys[order], xs[order], comp[order]
    start = np.flatnonzero(np.r_[True, comp[1:] != comp[:-1]])
    count = np.diff(np.r_[start, len(comp)])
    w = 1 / (1 + np.exp(-o.core[ys, xs].astype(np.float64)))
    two = np.radians(2 * o.geo[4, ys, xs].astype(np.float64))
    theta = 0.5 * np.arctan2(np.add.reduceat(w * np.sin(two), start), np.add.reduceat(w * np.cos(two), start))
    theta = np.mod(theta, np.pi)
    th = np.repeat(theta, count)
    px, py = xs + 0.5, ys + 0.5                  # px centres, as wordboxes.group measures its boxes
    a = px * np.cos(th) - py * np.sin(th)        # along u = (cos, -sin), the axis (y down)
    b = px * np.sin(th) + py * np.cos(th)        # along f = (sin, cos), towards the letters' feet
    a0, a1 = np.minimum.reduceat(a, start) - 0.5, np.maximum.reduceat(a, start) + 0.5
    b0, b1 = np.minimum.reduceat(b, start) - 0.5, np.maximum.reduceat(b, start) + 0.5
    mean_w = np.add.reduceat(w, start) / count
    rows = []
    for k in range(len(start)):
        u = np.array([np.cos(theta[k]), -np.sin(theta[k])])
        f = np.array([np.sin(theta[k]), np.cos(theta[k])])
        g = np.array([u * a0[k] + f * b0[k], u * a1[k] + f * b0[k], u * a1[k] + f * b1[k], u * a0[k] + f * b1[k]])
        t, ink_px, kinds = words._inked(g, ink, o.kind_p)
        centre = g.mean(0)
        _, _, _, (ta0, ta1, tb0, tb1) = words._frame(t)
        rows.append({"sheet": name, "i": k, "level": 0.5, "quarter": -1, "page_x": float(centre[0]),
                     "page_y": float(centre[1]), "axis": float(np.degrees(theta[k])), "length": float(a1[k] - a0[k]),
                     "height": float(b1[k] - b0[k]), "t_length": float(ta1 - ta0), "t_height": float(tb1 - tb0),
                     "mean_chance": float(mean_w[k]), "ink_px": int(ink_px),
                     **{f"p_{kn}": float(kinds[c]) for c, kn in enumerate(wordboxes.KIND_NAMES)},
                     "x": np.nan, "y": np.nan, "lab_lot": -1, "street": False,
                     **dict(zip(wordboxes.CORNER_COLS, [*g.reshape(8).tolist(), *t.reshape(8).tolist()]))})
    t = pd.DataFrame(rows, columns=wordboxes.COLUMNS)
    placed = placer(name, t[["page_x", "page_y"]].to_numpy(float))
    kept = t.iloc[np.asarray(placed["i"], int)].copy()
    for c in ("quarter", "x", "y", "lab_lot", "street"):
        kept[c] = np.asarray(placed[c])
    kept = kept.astype({"i": "int32", "quarter": "int8", "lab_lot": "int32", "street": bool, "x": "float64",
                        "y": "float64", "ink_px": "int64"})
    mine = kept[kept["quarter"] == q]
    return mine, {"formed": len(t), "off_map": len(t) - len(kept), "other_quarter": len(kept) - len(mine),
                  "kept": len(mine)}


def _crude_job(args):
    import cv2
    cv2.setNumThreads(1)
    run, name, q, sigma, aff, aff_prior, *kind = args
    t0 = time.perf_counter()
    t, counts = crude_map_words(run, name, q, _SHARED["sheet_fn"], _SHARED["outputs_fn"], _SHARED["placer"],
                                sigma, aff, aff_prior, kind[0] if kind else None)
    counts["seconds"] = round(time.perf_counter() - t0, 1)
    log.info("crude %s %s: %s", Path(run).name, name, counts)
    return name, t, counts


def crude(name: str = "a", quarter: str = "ne", seed: int = 0, workers: int = WORKERS, tags: bool = False,
          aff: bool = False, against: str | None = None, prior: str | None = None, read_max: int | None = None,
          kind: str | None = None) -> dict:
    """The crude-map chain (CRUDE_NOTE; crude-tags' with `tags`, crude-aff's with `aff`) on the quarter's every sheet
    from `name`'s held-out detector: words, placed, read by the current fold readers, and against today's
    (`vs_today`); with `against` (a run folder's name) against that run's boxes too (`vs_run`).  crude-aff with
    `prior` "all" or "core": the affinity logits plus the class prior's log odds measured on each detector's training
    quarters (`aff_counts`: every taught pair, or those of two core px).  `read_max`: no read nor score when the
    boxes number more (the boxes and their counts kept).  Writes runs/crude-<map|tags|aff>[-prior-<prior>]-<quarter>-
    s<seed>/."""
    if sum((tags, aff, kind is not None)) > 1:
        raise ValueError("crude: one of tags, aff and kind")
    import pandas as pd
    from sfzmaps.solver import lots, page, wordboxes
    quarters = paths.QUARTERS if quarter == "all" else (quarter,)
    srcs = {paths.QUARTERS.index(qn): run_dir(name, qn, seed) for qn in quarters}
    what = ("tags" if tags else (f"aff-prior-{prior}" if prior else "aff") if aff
            else f"kind-{kind}" if kind is not None else "map")
    usual = "aff" if aff else "a"                         # the arm each kind was first run from: no suffix
    out = RUNS / f"crude-{what}{'' if name == usual else '-from-' + name}-{quarter}-s{seed}"
    priors = {}
    if aff and prior:
        blocks = reader_stage.load_block_area(reader_stage.held_blocks())
        for q, src in srcs.items():
            priors[q] = aff_counts(src, tuple(k for k in range(len(paths.QUARTERS)) if k != q), blocks)
            log.info("%s: affinity pairs on its training quarters %s", src.name, priors[q])
    sigmas = {}
    if tags:
        blocks = reader_stage.load_block_area(reader_stage.held_blocks())
        for q, src in srcs.items():
            sigmas[q] = tag_spread(src, tuple(k for k in range(len(paths.QUARTERS)) if k != q), blocks)
            log.info("%s: within-word tag spread %s", src.name, sigmas[q])
    _SHARED["lots"], _SHARED["mapped"] = lots.load_lots(), page.load_sheets() > 0
    _SHARED.update(sheet_fn=load_sheet, outputs_fn=outputs, placer=wordboxes_placer)
    shift = {q: (v["log_odds_core"] if prior == "core" else v["log_odds"]) for q, v in priors.items()}
    got = _pool_map(_crude_job, [(src, s, q, sigmas[q]["sigma"] if tags else None, aff, shift.get(q, 0.0), kind)
                                 for q, src in srcs.items() for s in paths.SHEETS], workers)
    tables = [t for _, t, _ in got if len(t)]
    boxes = pd.concat(tables, ignore_index=True).sort_values(["sheet", "i"], kind="stable").reset_index(drop=True)
    (out / "boxes").mkdir(parents=True, exist_ok=True)
    boxes[wordboxes.COLUMNS].to_parquet(out / "boxes" / "boxes.parquet", index=False)
    (out / "boxes" / "summary.json").write_text(json.dumps({
        "note": TAGS_NOTE if tags else AFF_NOTE if aff else KIND_NOTE if kind is not None else CRUDE_NOTE,
        "kind_mode": kind,
        "detectors": {paths.QUARTERS[q]: r.name for q, r in srcs.items()},
        "tag_spread": {paths.QUARTERS[q]: v for q, v in sigmas.items()},
        "affinity_pairs": {paths.QUARTERS[q]: v for q, v in priors.items()}, "prior": prior, "boxes": len(boxes),
        "sheets": [[n, c] for n, _, c in got]}, indent=1) + "\n")
    if read_max is not None and len(boxes) > read_max:
        log.info("%s: %d boxes, over %d: not read", out.name, len(boxes), read_max)
        return {"boxes_made": len(boxes), "not_read": True}
    read(out, workers)
    got = vs_today(out, None if quarter == "all" else quarter)
    if against:
        got["vs_" + against] = vs_run(out, RUNS / against, None if quarter == "all" else quarter)
    return got


# --- code spotting: the code head's classes read straight off the px ------------------------------------------------

SPOT_NOTE = ("code spotting: each px's likeliest class of the code head (heads words_code); a label's code is the "
             "class of the px its point lies on (the px whose square holds it) when that class is a code; no code "
             "there (the background or another text) counts wrong; no boxes, no word formation, no reader")


def code_at(code: np.ndarray, x: float, y: float) -> int:
    """The class of the px whose square holds the page point (x, y) (px (r, c) spans c..c+1, r..r+1); NO_CODE off the
    page."""
    from sfzmaps.solver.textnet import predict
    r, c = int(np.floor(y)), int(np.floor(x))
    if not (0 <= r < code.shape[0] and 0 <= c < code.shape[1]):
        return predict.NO_CODE
    return int(code[r, c])


def code_component(code: np.ndarray, x: float, y: float, cache: dict) -> np.ndarray | None:
    """The 8-connected component of the px sharing the class at the page point (x, y) that holds it, bool (the page's
    shape), or None when the class there is no code (`codes.FIRST_CODE` on)."""
    import cv2
    from sfzmaps.solver.wordread import codes
    c = code_at(code, x, y)
    if c < codes.FIRST_CODE or c >= model_classes():
        return None
    if c not in cache:
        cache[c] = cv2.connectedComponents((code == c).astype(np.uint8), connectivity=8)[1]
    lab = cache[c]
    return lab == lab[int(np.floor(y)), int(np.floor(x))]


def model_classes() -> int:
    from sfzmaps.solver.textnet import model
    return model.CODE_CLASSES


def words_in(component: np.ndarray, words: np.ndarray) -> list[int]:
    """The indices of the page rectangles `words` ((n, 4, 2) px corners, as `data._paint` fills them) more than half of
    whose px lie in the component (the even-odds call)."""
    if component is None:
        return []
    ys, xs = np.nonzero(component)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lo, hi = words.min(1), words.max(1)
    near = np.flatnonzero((hi[:, 0] >= x0) & (lo[:, 0] < x1) & (hi[:, 1] >= y0) & (lo[:, 1] < y1))
    out = []
    for k in near:
        (wx0, wy0), (wx1, wy1) = lo[k], hi[k] + 1
        rect = data._paint(words[k:k + 1], np.ones(1, np.int32), int(wy0), int(wx0), int(wy1 - wy0),
                           int(wx1 - wx0)) > 0
        sub = np.zeros_like(rect)
        ya, yb, xa, xb = max(wy0, 0), min(wy1, component.shape[0]), max(wx0, 0), min(wx1, component.shape[1])
        sub[ya - wy0:yb - wy0, xa - wx0:xb - wx0] = component[ya:yb, xa:xb]
        if rect.any() and (rect & sub).sum() * 2 > rect.sum():
            out.append(int(k))
    return out


def spot(run, quarter: str | None, against: tuple = (), out: Path | None = None) -> dict:
    """Code spotting (SPOT_NOTE) on the quarter's answered code labels from the run's word outputs (`run` a folder;
    or {quarter number: folder}, each quarter's labels from its own run, `quarter` None): the headline, the paired
    difference against today's boxes and reads and against each of `against` (run folders' names, their boxes and
    reads; by quarter too with several quarters), the wrong labels by cause (no code at the point; a wrong code in a
    component holding at most one of Terra's words; a wrong code in a component holding two or more, merged) and the
    right ones in merged components.  Writes spot.json and spot_labels.parquet in `out` (`run` when None)."""
    import pandas as pd
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.textread import store
    from sfzmaps.solver.wordread import codes, measures
    runs = dict(run) if isinstance(run, dict) else {paths.QUARTERS.index(quarter): Path(run)}
    out_dir = Path(out or run)
    names = codes.classes()
    answers = measures.answers_from_store(store.READ_DIR)
    answers = answers[answers["quarter"].isin(list(runs))].reset_index(drop=True)
    rows = []
    for (sheet, q), a in answers.groupby(["sheet", "quarter"], sort=True):
        o = outputs(runs[q], sheet)
        if o.code is None:
            raise ValueError(f"{Path(runs[q]).name}: its word outputs hold no code (heads words_code)")
        sd = load_sheet(sheet)
        cache: dict = {}
        for k, lab in a.iterrows():
            c = code_at(o.code, lab["point_x"], lab["point_y"])
            comp = code_component(o.code, lab["point_x"], lab["point_y"], cache)
            inside = words_in(comp, sd.words)
            got = names[c] if codes.FIRST_CODE <= c < len(names) else None
            rows.append({"label": int(k), "sheet": sheet, "point_x": lab["point_x"], "point_y": lab["point_y"],
                         "truth": lab["truth"], "terra": lab["terra"], "block": lab["block"], "quarter": int(q),
                         "class": c, "code": got, "p": float(o.code_p[int(np.floor(lab["point_y"])),
                                                                     int(np.floor(lab["point_x"]))]),
                         "component_px": int(comp.sum()) if comp is not None else 0,
                         "component_words": len(inside), "right": float(got == lab["truth"])})
    t = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
    t["cause"] = np.where(t["right"] > 0, np.where(t["component_words"] >= 2, "right, merged component", "right"),
                          np.where(t["code"].isna(), "no code at the point",
                                   np.where(t["component_words"] >= 2, "wrong code, merged component",
                                            "wrong code, own component")))
    mine = pd.DataFrame({"label": t["label"], "row": -1, "right": t["right"],
                         "block": [b if isinstance(b, str) and b else f"label{k}" for k, b in zip(t["label"],
                                                                                                   t["block"])],
                         "quarter": t["quarter"]})
    today = wordboxes.load_boxes().reset_index(drop=True)
    today["row"] = np.arange(len(today))
    today = today.set_index("row", drop=False)
    sides = {"today": measures.label_right(today, measures.reads_from_store(store.READ_DIR), answers)}
    for other in against:
        r = RUNS / other
        theirs = measures.answers_from_store(r / "read")
        theirs = theirs[theirs["quarter"].isin(list(runs))].reset_index(drop=True)
        cols = ["sheet", "point_x", "point_y", "truth"]
        if not theirs[cols].equals(answers[cols]):
            raise ValueError(f"{other}: not the same answered labels in the same order")
        sides[other] = measures.label_right(run_boxes(r), measures.reads_from_store(r / "read"), answers)
    out = {"note": SPOT_NOTE, "runs": {paths.QUARTERS[q]: Path(r).name for q, r in runs.items()},
           "quarter": quarter or "all", "labels": len(t),
           "right": int(t["right"].sum()), "share": round(float(t["right"].mean()), 4),
           "ci": measures.cluster_interval(mine["right"].to_numpy(float), mine["block"].to_numpy(str)),
           "causes": {k: int(v) for k, v in t["cause"].value_counts().items()},
           "others": {k: {"right": int(d["right"].sum()), "share": round(float(d["right"].mean()), 4)}
                      for k, d in sides.items()},
           "paired_minus": {k: measures.paired_labels(mine, d) for k, d in sides.items()}}
    if len(runs) > 1:
        out["by_quarter"] = {
            paths.QUARTERS[q]: {"labels": int((t["quarter"] == q).sum()), "right": int(t.loc[t["quarter"] == q,
                                                                                            "right"].sum()),
                                "causes": {k: int(v) for k, v in t.loc[t["quarter"] == q, "cause"].value_counts()
                                           .items()},
                                **{f"paired_minus_{k}": measures.paired_labels(
                                    mine[mine["quarter"] == q].reset_index(drop=True),
                                    d[d["quarter"] == q].reset_index(drop=True)) for k, d in sides.items()}}
            for q in runs}
    out_dir.mkdir(parents=True, exist_ok=True)
    t.to_parquet(out_dir / "spot_labels.parquet", index=False)
    (out_dir / "spot.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def code_counts(quarter: str) -> dict:
    """Each class's Terra words on the 2000 sheets' training px of the fold holding `quarter` out: the other quarters
    outside the answered blocks (`fit_area`), a word counted by the px at its rectangle's centre."""
    from sfzmaps.solver.textnet import boxes
    from sfzmaps.solver.wordread import codes
    q = paths.QUARTERS.index(quarter)
    keep = tuple(k for k in range(len(paths.QUARTERS)) if k != q)
    blocks = reader_stage.load_block_area(reader_stage.held_blocks())
    records = boxes.terra_records()
    sheets = codes.prepare([load_sheet(s) for s in paths.SHEETS], records=records)

    def area(sd):
        a = fit_area(sd, keep, blocks)
        c = np.round(sd.words.mean(1)).astype(int)
        c[:, 0], c[:, 1] = np.clip(c[:, 0], 0, sd.width - 1), np.clip(c[:, 1], 0, sd.height - 1)
        return a[c[:, 1], c[:, 0]]
    n = codes.counts(sheets, area)
    allw = codes.counts(sheets)
    names = codes.classes()
    return {"training_words": {names[k]: int(n[k]) for k in range(len(names))},
            "all_words": {names[k]: int(allw[k]) for k in range(len(names))}}


# --- the crude-kind review page: three box sets at one zoom -----------------------------------------------------------

SETS = (("kind", "crude-kind"), ("map", "crude-map"), ("today", "today"))
PAGE_TITLE = "Mission New Boxes"
CAUSES = ROOT.parent / "causes"          # label_causes.py's labels-<run>.parquet: each wrong label's cause


def split_kinds(run, sheet: str, q: int, mode: str = "map") -> list[str]:
    """The kind crude-kind gave each of a page's words in quarter `q` (by `i`, the order `crude_map_words` numbers
    them: by label), `data.KINDS`' names."""
    sd = load_sheet(sheet)
    o = outputs(run, sheet)
    on = (reader_stage.page_quarters(sd) == q) & (o.core > 0)
    kinds = kind_map(o.kind_p, data.page_ink(sd), mode)
    flat = kind_labels(on, kinds).ravel()
    sel = flat > 0
    _, first = np.unique(flat[sel], return_index=True)
    names = list(data.KINDS)
    return [names[int(k)] for k in kinds.ravel()[sel][first]]


def label_causes_of(name: str, sheet: str):
    """label_causes.py's wrong labels of a run (`name`; "today" for today's boxes) on `sheet`, or None."""
    import pandas as pd
    f = CAUSES / f"labels-{name}.parquet"
    if not f.is_file():
        return None
    c = pd.read_parquet(f)
    return c[c["sheet"] == sheet].reset_index(drop=True)


def build_kind_page(sheet: str = "2000-digital-zoning-p07", kind_run: str = "crude-kind-map-all-s0",
                    map_run: str = "crude-map-all-s0", page_name: str = "crude-kind") -> Path:
    """The review page of crude-kind's boxes on `sheet` (every quarter from its own a-*-s0 detector and fold reader),
    the outlines coloured by the kind the split gave each word; a segmented control flips the boxes (and what a tap
    tells, and the wrong labels with their causes) to crude-map's or today's at the same zoom.  One image under all
    three (the ink; the text ink layer red where the word-core chance is 0.5 or more); the folded panels are
    crude-kind's against today's."""
    import html as H
    from sfzmaps.solver import grid, wordboxes
    from sfzmaps.solver.textread import store
    from sfzmaps.solver.wordread import measures, review
    root = review.experiments_dir()
    terra = review.terra_on_sheet(sheet)
    sd = load_sheet(sheet)
    quarters = reader_stage.page_quarters(sd)
    runs_ = {k: run_dir("a", qn, 0) for k, qn in enumerate(paths.QUARTERS)}
    core = np.zeros(quarters.shape, np.uint8)
    for k, run in runs_.items():
        mine = quarters == k
        o = outputs(run, sheet)
        core[mine] = np.round(255 / (1 + np.exp(-o.core[mine].astype(np.float64)))).astype(np.uint8)
        del o
    ink = data.load_page(sheet) < grid.INK_BELOW
    on_map = review.on_the_map(sheet, terra[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2).mean(1))
    kind_dir, map_dir = RUNS / kind_run, RUNS / map_run
    today = wordboxes.load_boxes().reset_index(drop=True)
    today["row"] = np.arange(len(today))
    today = today.set_index("row", drop=False)
    sets = {"kind": (run_boxes(kind_dir), review.reads_for_page(kind_dir / "read")),
            "map": (run_boxes(map_dir), review.reads_for_page(map_dir / "read")),
            "today": (today, review.reads_for_page(store.READ_DIR))}
    sets = {k: (b[b["sheet"] == sheet], r) for k, (b, r) in sets.items()}
    missed = sorted(set().union(*(review.missed_words(b, terra, review.MATCH_IOU) for b, _ in sets.values())))
    terra_reads = review.terra_box_reads(sheet, terra, np.array(missed, int), store.TRAIN_DIR)
    answers = measures.answers_from_store(kind_dir / "read")
    answers = answers[answers["sheet"] == sheet].reset_index(drop=True)
    before = json.loads((root / "baseline" / "numbers.json").read_text())
    forms = review.load_forms()
    causes = {"kind": label_causes_of(kind_run, sheet), "map": label_causes_of(map_run, sheet),
              "today": label_causes_of("today", sheet)}
    lines = [KIND_NOTE + ".",
             "Text ink: the word-core chance 0.5 or more, each quarter from its own held-out detector (arm a, seed 0: "
             + ", ".join(f"{paths.QUARTERS[k].upper()} {r.name}" for k, r in runs_.items()) + ").",
             "Readings: the current fold readers (data/solver/text-train) and their 0/180 classifiers; the missed "
             "words read at Terra's box by the same readers.  Today's boxes: data/solver/boxes, read as the store "
             "holds them.",
             "Wrong labels' causes: label_causes.py (experiments/wordread/causes/labels-<run>.parquet).",
             "The panels: crude-kind against today's boxes on this sheet (review/baseline)."]
    ds = {}
    for key, title in SETS:
        b, r = sets[key]
        ds[key] = review.page_data(sheet, b, r, terra, core, before, ink=ink,
                                   levels={qn: review.TEXT_CHANCE for qn in paths.QUARTERS}, on_map=on_map,
                                   terra_reads=terra_reads, answers=answers, forms=forms, models=lines,
                                   run=f"{title} (each quarter from its own held-out detector)",
                                   comps=review.box_components(b, core, ink), reason_text=WORD_REASONS,
                                   causes=causes[key])
    # crude-kind's outlines by the kind its split gave each word
    kb = sets["kind"][0]
    given = {}
    for k, run in runs_.items():
        names = split_kinds(run, sheet, k)
        for row, i in kb.loc[kb["quarter"] == k, "i"].items():
            given[int(row)] = names[int(i)]
    for rec in ds["kind"]["boxes"]:
        rec["split_kind"] = given.get(int(rec["row"]))
        if rec["split_kind"]:
            rec["kind"] = {"zoning code": "zoning", "block number": "block", "street name": "street",
                           "other": "other"}[rec["split_kind"]]
    ds["kind"]["pixels"] = review.palette_image(ink, review.text_ink_of(ink, core), [], [])
    full = {"kind": json.loads((kind_dir / "vs_today.json").read_text()),
            "map": json.loads((map_dir / "vs_today.json").read_text())}
    vs_map = json.loads((kind_dir / f"vs_{map_run}.json").read_text())
    rows = []
    for key, title in SETS:
        n = ds[key]["numbers"]
        bx, hd = n["boxes"]["0.5"], n.get("headline", {})
        whole = (full["kind"]["headline"] if key == "kind" else full["map"]["headline"] if key == "map"
                 else full["kind"]["today_headline"])
        rows.append(f"<tr><th>{title}</th><td>{hd.get('right', '–')}/{hd.get('n', '–')}</td>"
                    f"<td>{whole['right']}/{whole['n']}</td><td>{bx['ours']:,}</td><td>{bx['matched']:,}</td>"
                    f"<td>{bx['missed']:,}</td><td>{bx['missed_reasons'].get('merged_with_word', 0):,}</td>"
                    f"<td>{bx['ours_unmatched_none']:,}</td></tr>")
    pk, pm = full["kind"]["paired_run_minus_today"], vs_map["paired_run_minus_other"]
    about = (
        f"<h3>The versions</h3><p>{H.escape(KIND_NOTE)}.</p>"
        "<p>crude-kind's outlines take the kind its split gave each word; crude-map's and today's the box's likeliest "
        "kind.  The image under all three is one.</p>"
        '<div class="tblwrap"><table class="cmp"><tr><th></th><th>labels right, this sheet</th><th>labels right, '
        "all 699</th><th>boxes</th><th>matched (IoU 0.5)</th><th>missed</th><th>merged with a word</th>"
        "<th>read as no text</th></tr>" + "".join(rows) + "</table></div>"
        f'<p>All 699 labels, paired by whole blocks: crude-kind minus today {pk["diff"]:+.3f} '
        f'[{pk["lo"]:+.3f}, {pk["hi"]:+.3f}]; minus crude-map {pm["diff"]:+.3f} [{pm["lo"]:+.3f}, {pm["hi"]:+.3f}].</p>'
        "<h3>crude-kind against today's boxes</h3>")
    out = root / page_name
    page = review.render(ds["kind"], out / "review.html", sets=[(k, t, ds[k]) for k, t in SETS], title=PAGE_TITLE,
                         about=about)
    (out / "numbers.json").write_text(json.dumps({k: ds[k]["numbers"] for k, _ in SETS}, indent=1) + "\n")
    return page


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
    sp = sub.add_parser("spot")
    sp.add_argument("--name", default="code")
    sp.add_argument("--quarter", default="ne", choices=(*paths.QUARTERS, "all"))
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--against", nargs="*", default=[])
    cc = sub.add_parser("code-counts")
    cc.add_argument("--quarter", default="ne", choices=paths.QUARTERS)
    cr = sub.add_parser("crude")
    cr.add_argument("--quarter", default="ne", choices=(*paths.QUARTERS, "all"))
    cr.add_argument("--seed", type=int, default=0)
    cr.add_argument("--workers", type=int, default=WORKERS)
    cr.add_argument("--tags", action="store_true", help="crude-tags: split the components by the word tags")
    cr.add_argument("--aff", action="store_true", help="crude-aff: join the core px by the learned affinity")
    cr.add_argument("--name", default="a", help="the detector arm (its folds' runs <name>-<quarter>-s<seed>)")
    cr.add_argument("--against", default=None, help="a run folder's name: paired against its boxes too")
    cr.add_argument("--prior", default=None, choices=("all", "core"),
                    help="crude-aff: add the class prior's log odds measured on the training quarters' pairs")
    cr.add_argument("--read-max", type=int, default=None, help="read and score only up to this many boxes")
    cr.add_argument("--kind", default=None, choices=KIND_MODES,
                    help="crude-kind: join core px only to neighbours of the same MAP kind (this kind map)")
    kp = sub.add_parser("kind-page")
    kp.add_argument("--sheet", default="2000-digital-zoning-p07")
    kc = sub.add_parser("kind-check")
    kc.add_argument("--name", default="a")
    kc.add_argument("--quarter", default="ne", choices=paths.QUARTERS)
    kc.add_argument("--seed", type=int, default=0)
    dr = sub.add_parser("draft")
    dr.add_argument("--sheet", default="2000-digital-zoning-p07")
    dr.add_argument("--workers", type=int, default=WORKERS)
    dr.add_argument("--fit-dir", type=Path, default=None)
    dr.add_argument("--tag", default="")
    dr.add_argument("--note", default=DRAFT_NOTE)
    parsers = {k: sub.add_parser(k) for k in ("train", "fit", "words", "read", "measure", "page")}
    for p in parsers.values():
        p.add_argument("--name", required=True)
        p.add_argument("--quarter", required=True, choices=paths.QUARTERS)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--workers", type=int, default=WORKERS)
    parsers["train"].add_argument("--doubt", action="store_true")
    parsers["train"].add_argument("--heads", default="words", choices=WORD_HEADS)
    parsers["fit"].add_argument("--own", action="store_true",
                                help="this run's own predictions on its training quarters (Phase 1, optimistic)")
    parsers["fit"].add_argument("--from", dest="sources", nargs="*", default=[],
                                help="other folds' runs (folder names), each read on its own held-out quarter")
    parsers["fit"].add_argument("--sheets", nargs="*", default=None, help="these 2000 sheets only (default every one)")
    parsers["fit"].add_argument("--rounds", type=int, default=2, choices=(1, 2),
                                help="1: the first prior alone (a draft's)")
    parsers["words"].add_argument("--sheets", nargs="*", default=None,
                                  help="these 2000 sheets only (default every one)")
    parsers["words"].add_argument("--add", action="store_true", help="keep the other sheets' boxes already made")
    parsers["words"].add_argument("--fit-dir", type=Path, default=None,
                                  help="the spreads and prior from this folder (default the run's fit/)")
    parsers["measure"].add_argument("--today", action="store_true",
                                    help="against today's boxes over the same labels (vs_today.json)")
    parsers["measure"].add_argument("--sheets", nargs="*", default=None)
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
    if a.cmd == "kind-page":
        p = build_kind_page(a.sheet)
        print(json.dumps({"page": str(p), "mb": round(p.stat().st_size / 1e6, 2)}))
        return
    if a.cmd == "kind-check":
        print(json.dumps(kind_check(a.name, a.quarter, a.seed), indent=1))
        return
    if a.cmd == "spot":
        if a.quarter == "all":
            got = spot({q: run_dir(a.name, qn, a.seed) for q, qn in enumerate(paths.QUARTERS)}, None,
                       tuple(a.against), RUNS / f"spot-{a.name}-all-s{a.seed}")
        else:
            got = spot(run_dir(a.name, a.quarter, a.seed), a.quarter, tuple(a.against))
        print(json.dumps(got, indent=1, default=float))
        return
    if a.cmd == "code-counts":
        got = code_counts(a.quarter)
        (RUNS / f"code-counts-{a.quarter}.json").write_text(json.dumps(got, indent=1) + "\n")
        print(json.dumps(got, indent=1))
        return
    if a.cmd == "crude":
        got = crude(a.name, a.quarter, a.seed, a.workers, a.tags, a.aff, a.against, a.prior, a.read_max, a.kind)
        if got.get("not_read"):
            print(json.dumps(got))
            return
        print(json.dumps({"labels": got["labels"], "paired_run_minus_today": got["paired_run_minus_today"],
                          "run": {k: got["headline"][k] for k in ("n", "right", "share", "ci", "with_box")},
                          "today": {k: got["today_headline"][k] for k in ("n", "right", "share", "ci", "with_box")},
                          "boxes": {k: got["boxes"][k] for k in ("terra", "ours", "matched", "missed",
                                                                 "missed_reasons")},
                          "today_boxes": {k: got["today_boxes"][k] for k in ("terra", "ours", "matched", "missed",
                                                                             "missed_reasons")}},
                         indent=1, default=float))
        return
    if a.cmd == "draft":
        print(draft(a.sheet, workers=a.workers, fit_dir=a.fit_dir, tag=a.tag, note=a.note))
        return
    run = run_dir(a.name, a.quarter, a.seed)
    q = paths.QUARTERS.index(a.quarter)
    if a.cmd == "train":
        print(train_run(a.name, a.quarter, a.seed, a.doubt, recipe=word_recipe(heads=a.heads)))
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
        got = fit(run / "fit", sources, sheets=tuple(a.sheets or paths.SHEETS), workers=a.workers, rounds=a.rounds)
        print(json.dumps({k: dataclasses.asdict(v) for k, v in got.items()}, indent=1))
    elif a.cmd == "words":
        spreads, prior = load_fit(a.fit_dir or run / "fit")
        print(form_words(run, a.quarter, spreads, prior, sheets=tuple(a.sheets or paths.SHEETS), workers=a.workers,
                         add=a.add))
    elif a.cmd == "read":
        print(read(run, a.workers))
    elif a.cmd == "measure":
        if a.today:
            got = vs_today(run, a.quarter, sheets=a.sheets or None)
            print(json.dumps({k: got[k] for k in ("labels", "paired_run_minus_today")}, indent=1, default=float))
            print(json.dumps({"run": {k: got["headline"][k] for k in ("n", "right", "share", "ci")},
                              "today": {k: got["today_headline"][k] for k in ("n", "right", "share", "ci")}},
                             indent=1, default=float))
            return
        got = measure(run, a.quarter)
        print(json.dumps({"headline": {k: v for k, v in got["headline"].items() if k != "by_quarter"},
                          "boxes": got["box_numbers"]}, indent=1, default=float))
    elif a.cmd == "page":
        print(build_page(run, a.page_name or run.name, a.quarter, a.fit_note))


if __name__ == "__main__":
    main()
