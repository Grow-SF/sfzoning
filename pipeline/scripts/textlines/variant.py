"""Train one detector with one quarter (and the owner's answered blocks) held out, predict that quarter's px on the 2000
sheets (and on other maps' sheets, `--extra`), gate it on the text-vs-line gate, and draw the picture panel and the
missed-text audit.  What it is for: trying a training change on one held-out quarter, scored on what it did not see.

    uv run python scripts/textlines/variant.py control --quarter ne --seed 0 --labels lines-unknown \
        --extra 1990-1996-zoning
    uv run python scripts/textlines/variant.py composites --quarter ne --seed 0 --labels lines-unknown \
        --composites P --extra 1990-1996-zoning

With `--composites P` (and the label rule lines-unknown, which the composites start from) the bank of clean words and
line pieces (`composite.Bank.build`) is built from the training sheets with the quarter and the answered blocks held
out, seeded by the run's seed, before the GPU's turn, and handed to the training.

With `--heads words` the detector has the word heads too (`train.Recipe.heads="words"`): the training sheets get their
word cores (`wordread.shrink.prepare`, each word's own rectangle) and, with `--doubt`, Terra's doubt as each word's
weight (`wordread.doubt.prepare` of the quarter's "box" weights, `doubt.load_weights`); after the chance, in the same
GPU turn, every 2000 sheet's whole page is scored by the word heads (`predict.word_outputs`) and written to words/
for the word formation (`scripts/wordread/run_detector.py`).  `--root` writes the run under another folder.

Writes `data/solver/experiments/textlines/runs/<name>-<quarter>-s<seed>/`:
  model.pt        the detector (`train.save_model`)
  chance/         the chance of text x 255 of each 2000 sheet, `<sheet>.png`, in the quarter's px and 0 elsewhere; each
                  extra map's sheets in `<series>/<sheet>.png`
  scores.json, scores.md, blocks.npz   the gate (`linegate.write`) on the quarter's px of the 2000 sheets
  extra/<series>/{scores.json, scores.md, blocks.npz}   the gate on each extra map's px in the quarter: it has no table,
                  so no block and no M1a (block -1, weight 0): read its M1b, M1c and M2
  panel.png, audit.png, audit.json   `panel.render` (this run beside the held-out detectors and the old quarter
                  models) and `audit.sheet` (64 missed letter-sized pieces in the quarter)
  run.json        the recipe, seed, device, the minutes training and predicting took, git head, `git status
                  --porcelain -- src` at the start; with composites the bank's sizes (`Bank.sizes`) and what the
                  composites were (`Bank.tally`); `doubt`; on the Mac's GPU the most memory the training held
                  (`mps_peak`, sampled)
  words/<sheet>.npz   with the word heads: the word heads' outputs over the whole page (`save_word_outputs`)

One process, one GPU turn around the training and the prediction (nothing else computes on the GPU meanwhile); the gate
runs after it, on the CPU, one thread, about 7 GB at the peak: never two at once.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
import argparse  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.setNumThreads(1)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sfzmaps import gpu  # noqa: E402
from sfzmaps.solver import paths, runs  # noqa: E402
from sfzmaps.solver.textnet import composite, data, linegate, predict, reader_stage, stage, train, zones  # noqa: E402

RUNS_DIR = runs.EXPERIMENTS_DIR / "textlines" / "runs"
WORDS = "words"           # the run's folder of the word heads' outputs
PEAK_EVERY = 0.05         # s between samples of the GPU's memory while training (a record only)
BASELINES = runs.EXPERIMENTS_DIR / "textlines" / "baselines"
LINES = zones.LINES_DIR
AUDIT_N = 64
BANK_THREADS = 4          # sheets the bank is built from at once


def run_name(name: str, quarter: str, seed: int) -> str:
    return f"{name}-{quarter}-s{seed}"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False,
                          cwd=Path(__file__).resolve().parents[2]).stdout.strip()


class ExtraGeo:
    """The gate's Geo for another map's sheets: the quarter by each px's map point, no block (-1) and no border weight
    (there is no table to weigh it by), so M1a has no px there."""

    def quarter(self, sd: data.SheetData) -> np.ndarray:
        return reader_stage.page_quarters(sd).astype(np.uint8)

    def block(self, sd: data.SheetData) -> np.ndarray:
        return np.full((sd.height, sd.width), -1, np.int32)

    def border_weight(self, sd: data.SheetData) -> np.ndarray:
        return np.zeros((sd.height, sd.width), np.float32)


class Peak:
    """The most memory torch's GPU allocator held (`torch.mps.current_allocated_memory` and
    `driver_allocated_memory`), sampled every PEAK_EVERY s by a thread while the body runs: a record, deciding
    nothing.  Off the Mac's GPU it samples nothing."""

    def __init__(self, device):
        self.on = gpu.on_gpu(device)
        self.most = {"allocated_gb": 0.0, "driver_gb": 0.0}

    def __enter__(self):
        import threading
        self.stop = threading.Event()
        if self.on:
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()
        return self

    def _sample(self):
        import torch
        while not self.stop.wait(PEAK_EVERY):
            self.most["allocated_gb"] = max(self.most["allocated_gb"], torch.mps.current_allocated_memory() / 1e9)
            self.most["driver_gb"] = max(self.most["driver_gb"], torch.mps.driver_allocated_memory() / 1e9)

    def __exit__(self, *exc):
        self.stop.set()
        if self.on:
            self.thread.join()
        return False


def save_word_outputs(f: Path, out) -> None:
    """A page's word outputs (`predict.WordOutputs`) as one .npz: chance (uint8), geo in float32 (a distance may run
    past float16's range), every other field the outputs hold (not None) in float16."""
    f.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in out._asdict().items() if v is not None}
    np.savez_compressed(f, **{k: v if k == "chance" else v.astype(np.float32 if k == "geo" else np.float16)
                              for k, v in arrays.items()})


def load_word_outputs(f: Path) -> predict.WordOutputs:
    """`save_word_outputs`' file back, the floats as float32."""
    with np.load(f) as z:
        return predict.WordOutputs(**{k: z[k] if k == "chance" else z[k].astype(np.float32)
                                      for k in predict.WordOutputs._fields if k in z.files})


def _by_sheet(chances: dict[tuple[str, str], np.ndarray], series: str) -> linegate.ChanceSource:
    return lambda sheet: chances[series, sheet]       # noqa: E731


def score(out: Path, quarter: str, scored: list[data.SheetData], chances: dict, extra: list[data.SheetData]) -> None:
    """The real scoring: the gate on `scored` (the 2000 sheets) by `CityGeo`, and on each extra map's sheets by
    `ExtraGeo`; the panel and the audit (see the module).  `chances`: (series, sheet) -> the page's chance."""
    from sfzmaps.solver.textnet import linegate as lg
    import audit
    import panel

    geo = lg.CityGeo()
    result, tallies = lg.gate_sheets(scored, _by_sheet(chances, paths.SERIES), (quarter,), geo)
    if lg.LEARNED_RULES.is_file():
        found = json.loads(lg.LEARNED_RULES.read_text()).get("run / font.npz")
        if found:
            result["rules"]["font.npz"] = lg.summarise_rules(found)
    lg.write(result, tallies, out)
    for series in dict.fromkeys(sd.series for sd in extra):
        mine = [sd for sd in extra if sd.series == series]
        r, t = lg.gate_sheets(mine, _by_sheet(chances, series), (quarter,), ExtraGeo())
        lg.write(r, t, out / "extra" / series)
    ink = {sd.sheet: data.page_ink(sd) for sd in scored}
    panel.page_ink = lambda sheet: ink[sheet]
    srcs = {"this run": _by_sheet(chances, paths.SERIES),
            "r328": lg.folder_source(BASELINES / "r328", "nested"),
            "old quarters": lg.folder_source(paths.SOLVER_DIR / f"text-{quarter}" / "chance", "flat")}
    panel.render(srcs, out / "panel.png",
                 f"{out.name}: red = ink called text (chance >= 0.5), grey = other ink; this run's chance is 0 "
                 f"outside {quarter}; the old quarters column is the {quarter} model everywhere")
    cands, total = audit.candidates(scored, srcs["this run"], (quarter,), geo)
    items = audit.draw(cands, AUDIT_N, 0)
    audit.sheet(items, out / "audit.png", lambda sheet: ink[sheet])
    (out / "audit.json").write_text(json.dumps({"letter_sized_pieces": total, "missed_below_0.5": len(cands),
                                                "items": items}, indent=1) + "\n")


def run(name: str, recipe: train.Recipe, quarter: str, seed: int, device: str | None = None,
        sheets: list[data.SheetData] | None = None, areas: tuple | None = None, score=None, bank=None,
        extra_series: tuple[str, ...] = (), doubt: bool = False, root: Path | None = None,
        doubt_weights=None) -> Path:
    """Train `recipe` with `quarter` and `areas` (default: the answered blocks) held out on `sheets` (default: every
    sheet Terra read of the training maps), predict the quarter's px of the 2000 sheets (0 elsewhere) and of
    `extra_series`' sheets, and score them with `score(out, quarter, scored, chances, extra)` (default `score`: the gate,
    the panel and the audit).  With `recipe.composites` the composites come from `bank`, or (None) from the bank of
    the training sheets with the same areas held out (`composite.Bank.build`, seeded `seed`), built before the GPU's
    turn.  With the word heads (`recipe.heads` "words") the sheets get their cores (`wordread.shrink.prepare`) and,
    with `doubt`, the quarter's "box" doubt weights (`doubt_weights`, or `wordread.doubt.load_weights(quarter)` when
    None); the 2000 sheets' word outputs are written to words/.  `root`: the folder the run's folder is made in
    (RUNS_DIR when None).  Returns the run's folder (see the module)."""
    if recipe.composites > 0 and recipe.labels != "lines-unknown":
        raise ValueError("composites start from the crops of the label rule lines-unknown (and its sheets' line px): "
                         f"run them with labels lines-unknown, not {recipe.labels}")
    if bank is not None and not recipe.composites:
        raise ValueError("a bank is for a recipe with composites: this one has none")
    if quarter not in paths.QUARTERS:
        raise ValueError(f"quarters are {paths.QUARTERS}, not {quarter!r}")
    words = recipe.heads in ("words", "words_aff")
    if doubt and not words:
        raise ValueError("doubt weighs the words in the word heads' loss: give a recipe with heads words")
    score = globals()["score"] if score is None else score
    q = paths.QUARTERS.index(quarter)
    device = device or train.default_device()
    out = (RUNS_DIR if root is None else Path(root)) / run_name(name, quarter, seed)
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", "src")
    t0 = time.perf_counter()
    if sheets is None:
        sheets = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False)
    if areas is None:
        areas = (data.Quarter(q), reader_stage.load_block_area(reader_stage.held_blocks()))
    areas = tuple(areas)
    if recipe.labels == "lines-unknown":
        prepared = zones.prepare(sheets, areas, LINES)
    else:
        radius = reader_stage.measure_radius(sheets, areas)["px"]
        prepared = [dataclasses.replace(sd, point_radius=radius) for sd in sheets]
    if words:
        from sfzmaps.solver.wordread import doubt as doubt_mod
        from sfzmaps.solver.wordread import shrink
        prepared = shrink.prepare(prepared)
        if doubt:
            weights = doubt_mod.load_weights(quarter) if doubt_weights is None else doubt_weights
            prepared = doubt_mod.prepare(prepared, weights, "box")
    scored = [sd for sd in sheets if sd.series == paths.SERIES]
    extra = [sd for sd in sheets if sd.series in extra_series]
    missing = [s for s in extra_series if not any(sd.series == s for sd in sheets)]
    if missing or not scored:
        raise ValueError(f"no sheets of {', '.join(missing) or paths.SERIES} among the sheets given")
    quarters = {(sd.series, sd.sheet): reader_stage.page_quarters(sd) == q for sd in [*scored, *extra]}
    loaded = time.perf_counter() - t0
    t0 = time.perf_counter()
    if recipe.composites > 0 and bank is None:
        bank = composite.Bank.build(prepared, areas, seed, threads=BANK_THREADS)
    banked = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)
    chances: dict[tuple[str, str], np.ndarray] = {}
    worded = 0.0
    with gpu.gpu_turn(True, device):
        t0 = time.perf_counter()
        with Peak(device) as peak:
            net = train.fit(data.hold_out(prepared, areas), recipe, device, seed, bank=bank)
        train.save_model(out / "model.pt", net, recipe, device, seed)
        trained = time.perf_counter() - t0
        t0 = time.perf_counter()
        for sd in [*scored, *extra]:
            chance = predict.chance(net, data.page_ink(sd), device, only=quarters[sd.series, sd.sheet])
            chance[~quarters[sd.series, sd.sheet]] = 0
            chances[sd.series, sd.sheet] = chance
            f = out / "chance" / (f"{sd.sheet}.png" if sd.series == paths.SERIES else f"{sd.series}/{sd.sheet}.png")
            f.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(f), chance):
                raise OSError(f"can't write {f}")
        predicted = time.perf_counter() - t0
        if words:
            t0 = time.perf_counter()
            for sd in scored:
                save_word_outputs(out / WORDS / f"{sd.sheet}.npz", predict.word_outputs(net, data.page_ink(sd), device))
            worded = time.perf_counter() - t0
        del net
        gc.collect()
        if gpu.on_gpu(device):
            import torch
            torch.mps.empty_cache()
    del prepared
    t0 = time.perf_counter()
    score(out, quarter, scored, chances, extra)
    scoring = time.perf_counter() - t0
    (out / "run.json").write_text(json.dumps({
        "name": name, "quarter": quarter, "seed": seed, "device": device, "recipe": dataclasses.asdict(recipe),
        "extra_series": list(extra_series), "git_head": head, "git_status_src": dirty,
        "bank": None if bank is None else bank.sizes(), "doubt": bool(doubt),
        "mps_peak": {k: round(v, 3) for k, v in peak.most.items()} if peak.on else None,
        "composites_made": None if bank is None else dict(sorted(bank.tally.items())),
        "minutes": {"load": round(loaded / 60, 2), "bank": round(banked / 60, 2), "train": round(trained / 60, 2),
                    "predict": round(predicted / 60, 2), "word_outputs": round(worded / 60, 2),
                    "score": round(scoring / 60, 2)}}, indent=1) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("name")
    ap.add_argument("--quarter", required=True, choices=paths.QUARTERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default="boxes", choices=("boxes", "lines-unknown"))
    ap.add_argument("--composites", type=float, default=0.0,
                    help="the share of crops that are composites (needs --labels lines-unknown)")
    ap.add_argument("--extra", action="append", default=[], help="another map's series to predict and gate too")
    ap.add_argument("--heads", default=None, choices=train.CHOICES["heads"],
                    help="the recipe's heads (default: the stage's); words adds the word heads")
    ap.add_argument("--doubt", action="store_true", help="Terra's doubt weighs each word (needs --heads words)")
    ap.add_argument("--root", type=Path, default=None, help="the folder the run's folder is made in")
    a = ap.parse_args()
    if a.composites and a.labels != "lines-unknown":
        sys.exit("--composites start from the crops of the label rule lines-unknown: add --labels lines-unknown")
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    recipe = dataclasses.replace(stage.DEFAULT_RECIPE, labels=a.labels, composites=a.composites,
                                 **({"heads": a.heads} if a.heads else {}))
    print(run(a.name, recipe, a.quarter, a.seed, extra_series=tuple(a.extra), doubt=a.doubt, root=a.root))


if __name__ == "__main__":
    main()
