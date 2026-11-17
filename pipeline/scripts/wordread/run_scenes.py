"""Task 13's scenes (item 8): build the scene bank of one fold (`wordread.scenes.SceneBank`, the quarter and the answered
blocks held out), write what it measured, a contact sheet of scenes with their masks, and the words a crop holds, real
and scene; and the run step Task 8's runner calls when the kept detector exists.  Everything is written to
data/solver/experiments/wordread/scenes/<quarter>/ (never committed).  CPU only; $0.

    uv run python scripts/wordread/run_scenes.py build --quarter ne [--seed 0] [--threads 2]

The run step (`scene_run`): the kept word recipe with `scenes` at the measured share (`density.json`'s "share"), its
scenes from this fold's bank, trained and measured as Task 8's arms are (`variant.run` through the bank it takes).

The keep rule (Task 8's, written before the runs): scenes are kept if, on NE over seeds 0 and 1, end to end (the
detector plus a reader retrained on its boxes with Task 4's kept options), the headline rises with the paired label
interval excluding zero; the per-ink gate's measures (m1a_in_box, m2, m2_clean_words, m1c) lose nothing beyond the
control's noise; and the controller's look at the review page agrees.  A kept item is confirmed on all four folds at
seed 0.  The decision goes in experiments/wordread/scenes/decision.md.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sfzmaps.solver import paths, runs  # noqa: E402
from sfzmaps.solver.textnet import data, reader_stage, train  # noqa: E402
from sfzmaps.solver.wordread import scenes, shrink  # noqa: E402

cv2.setNumThreads(1)
log = logging.getLogger(__name__)

ROOT = runs.EXPERIMENTS_DIR / "wordread" / "scenes"
SHEET_CROPS = 12          # scenes on the contact sheet
DENSITY_CROPS = 2000      # crops of each kind drawn to count the words a crop holds (a sample size; its error reported)
COLUMNS = 3               # scenes a row on the contact sheet


def held_areas(quarter: str) -> tuple:
    """The areas a fold holds out: its quarter and the answered blocks."""
    return (data.Quarter(paths.QUARTERS.index(quarter)), reader_stage.load_block_area(reader_stage.held_blocks()))


def bank_for(quarter: str, seed: int, sheets=None, areas=None, threads: int = 2) -> scenes.SceneBank:
    """The fold's scene bank: every sheet Terra read of the training maps (or `sheets`) with the quarter and the
    answered blocks (or `areas`) held out, `threads` sheets at a time."""
    sheets = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False) if sheets is None else sheets
    return scenes.SceneBank.build(sheets, held_areas(quarter) if areas is None else tuple(areas), seed,
                                  threads=threads)


def scene_recipe(share: float, kept: train.Recipe | None = None) -> train.Recipe:
    """The kept word recipe (default Task 8's `run_detector.word_recipe()`) with `share` of its crops scenes."""
    if kept is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import run_detector
        kept = run_detector.word_recipe()
    return dataclasses.replace(kept, scenes=float(share))


def measured_share(quarter: str, root: Path | None = None) -> float:
    """The scene share measured for the fold (`density.json`'s "share", written by `build`)."""
    return float(json.loads(((root or ROOT) / quarter / "density.json").read_text())["share"])


def scene_run(quarter: str, seed: int, share: float | None = None, kept: train.Recipe | None = None,
              bank=None, name: str = "scenes", **kwargs) -> Path:
    """Task 8's run step for scenes: the kept recipe with `share` (default the measured one) of its crops scenes,
    trained by `variant.run` with the fold's bank (built here, before the GPU's turn, when not given) and measured
    as Task 8's arms are; `kwargs` as `variant.run` takes them (device, sheets, areas, root, doubt, ...).  Needs
    `variant.run` to take a bank for a recipe with scenes (today it takes one only with composites)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "textlines"))
    import variant
    recipe = scene_recipe(measured_share(quarter) if share is None else share, kept)
    if bank is None:
        bank = bank_for(quarter, seed, kwargs.get("sheets"), kwargs.get("areas"))
    return variant.run(name, recipe, quarter, seed, bank=bank, **kwargs)


# --- what a crop holds -----------------------------------------------------------------------------------------------

def words_in(crop: dict) -> int:
    """The words a crop holds: its distinct word ids with a core px in it."""
    return int(len(np.unique(crop["tag_id"][crop["tag_id"] > 0])))


def density(bank: scenes.SceneBank, sheets: list, seed: int, n: int = DENSITY_CROPS, size: int = train.CROP) -> dict:
    """Words a crop holds, real (drawn as `train.draw` draws them, the bank's areas held out) and scene, at the same
    scales and turns; and the pasteable share of the usable words.  The share of scene crops is the pasteable share
    (see the report: the word density alone does not fix it when scenes centre as the real crops do)."""
    held = shrink.prepare(data.hold_out(sheets, bank.sheets[0].held))
    pools = train.centre_pools(held)
    rng = np.random.default_rng([seed, 21])
    real, scene = [], []
    for _ in range(n):
        pool = pools[0] if rng.random() < train.TEXT_SHARE else pools[1]
        i, y, x = pool[rng.integers(len(pool))]
        scale = float(np.exp(rng.uniform(np.log(train.SCALE[0]), np.log(train.SCALE[1]))))
        turns = int(rng.integers(4))
        real.append(words_in(data.crop_targets(held[i], y, x, size, scale, turns, words=True)))
        scene.append(words_in(bank.crop(rng, size, scale, turns)))
    usable = sum(m.summary["words"]["usable"] for m in bank.measured.values())
    pasteable = sum(m.summary["words"]["pasteable"] for m in bank.measured.values())

    def stat(v):
        v = np.asarray(v, float)
        return {"mean": round(float(v.mean()), 4), "se": round(float(v.std(ddof=1) / math.sqrt(len(v))), 4),
                "q50": float(np.median(v)), "q90": float(np.quantile(v, 0.9))}
    return {"crops": n, "size": size, "real_words_per_crop": stat(real), "scene_words_per_crop": stat(scene),
            "usable_words": usable, "pasteable_words": pasteable, "share": round(pasteable / usable, 4)}


# --- the contact sheet -----------------------------------------------------------------------------------------------

def _tile(t: dict, caption: str) -> np.ndarray:
    """One scene as two panels: its ink and lessons (black ink; red text; blue line ink inside a word's rectangle,
    known not text; pale yellow the paper inside a word's rectangle, unknown), and its cores by word (a colour a word)
    with each word's centre as its core px' geometry put it (white dots: px + offset x height)."""
    ink, text, known, core, tag = t["ink"], t["text"] > 0, t["known"], t["core"] > 0, t["tag_id"]
    h, w = ink.shape
    a = np.full((h, w, 3), 255, np.uint8)
    a[ink] = (0, 0, 0)
    a[~known] = (170, 240, 250)
    a[core & ink & ~text] = (230, 120, 0)
    a[text] = (0, 0, 220)
    b = np.full((h, w, 3), 255, np.uint8)
    b[ink] = (190, 190, 190)
    palette = np.random.default_rng(1).integers(40, 230, (int(tag.max()) + 1, 3)).astype(np.uint8)
    b[core] = palette[tag[core]]
    geo = t["geo"]
    for k in np.unique(tag[core]):
        rows, cols = np.nonzero(tag == k)
        height = np.exp(geo[3, rows, cols])
        cx = float(np.mean(cols + geo[0, rows, cols] * height))
        cy = float(np.mean(rows + geo[1, rows, cols] * height))
        cv2.circle(b, (int(round(cx)), int(round(cy))), 3, (255, 255, 255), -1)
        cv2.circle(b, (int(round(cx)), int(round(cy))), 3, (0, 0, 0), 1)
    tile = np.concatenate([a, np.full((h, 4, 3), 255, np.uint8), b], 1)
    bar = np.full((22, tile.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, caption, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([bar, tile], 0)


def contact_sheet(bank: scenes.SceneBank, seed: int, out: Path, n: int = SHEET_CROPS, size: int = train.CROP) -> dict:
    """`n` scenes drawn as the training draws them (scale and turns), each beside its masks (`_tile`), in rows of
    COLUMNS: written to `out` (png).  Returns what each held."""
    rng = np.random.default_rng([seed, 13])
    tiles, held = [], []
    for k in range(n):
        scale = float(np.exp(rng.uniform(np.log(train.SCALE[0]), np.log(train.SCALE[1]))))
        turns = int(rng.integers(4))
        t = bank.crop(rng, size, scale, turns)
        last = bank.last
        sd = bank.sheets[last["sheet"]]
        labels = [f"{p['what']}:{bank.words[sd.series].text[p['word']]}" for p in last["labels"]]
        held.append({"scene": k, "series": sd.series, "sheet": sd.sheet, "window": list(last["window"]),
                     "scale": round(scale, 3), "turns": turns, "words": words_in(t), "labels": labels,
                     "segments": {n_: int((last["segments"]["kind"] == i).sum())
                                  for i, n_ in enumerate(scenes.LINE_KINDS)}})
        tiles.append(_tile(t, f"{k}: {sd.series} {sd.sheet}  x{scale:.2f} turns {turns}  {words_in(t)} words"))
    rows = [np.concatenate(tiles[i:i + COLUMNS] + [np.full_like(tiles[0], 255)] * (COLUMNS - len(tiles[i:i + COLUMNS])),
                           1) for i in range(0, n, COLUMNS)]
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), np.concatenate(rows, 0)):
        raise OSError(f"can't write {out}")
    return {"png": str(out), "scenes": held}


def build(quarter: str, seed: int, threads: int, root: Path | None = None) -> Path:
    """Build the fold's bank and write measured.json, sheet.png (+ sheet.json) and density.json."""
    out = (root or ROOT) / quarter
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sheets = data.load_sheets(paths.TEXT_TRAIN_SERIES, hold_out_validation=False)
    loaded = time.perf_counter() - t0
    t0 = time.perf_counter()
    bank = bank_for(quarter, seed, sheets, threads=threads)
    built = time.perf_counter() - t0
    (out / "measured.json").write_text(json.dumps({
        "quarter": quarter, "seed": seed, "minutes": {"load": round(loaded / 60, 2), "bank": round(built / 60, 2)},
        "maps": bank.sizes()}, indent=1, default=float) + "\n")
    sheet = contact_sheet(bank, seed, out / "sheet.png")
    (out / "sheet.json").write_text(json.dumps(sheet, indent=1, default=float) + "\n")
    t0 = time.perf_counter()
    dens = density(bank, sheets, seed)
    dens["minutes"] = round((time.perf_counter() - t0) / 60, 2)
    dens["tally"] = dict(sorted(bank.tally.items()))
    (out / "density.json").write_text(json.dumps(dens, indent=1, default=float) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the fold's bank: measures, contact sheet and word density")
    b.add_argument("--quarter", required=True, choices=paths.QUARTERS)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--threads", type=int, default=2)
    b.add_argument("--root", type=Path, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(build(a.quarter, a.seed, a.threads, a.root))


if __name__ == "__main__":
    main()
