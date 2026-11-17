"""Build the Mission Street review page (`sfzmaps.solver.wordread.review`) for one run of our word boxes and the text
reader, and write it to data/solver/experiments/wordread/review/<run>/:

  review.html   the page (one self-contained file; the controller publishes it to one Artifact and updates it)
  numbers.json  its top-line numbers: the next run's "before"
  missed.json   every Terra word no box matches, with its reason code and details

and add the run to review/runs.json (the runs in the order their pages were built).  CPU only, one thread, $0.

    uv run python scripts/wordread/review_page.py                      # the baseline: data/solver/boxes, text-read
    uv run python scripts/wordread/review_page.py --run <name> --boxes <boxes.parquet> --reads <text-read folder> \\
        [--levels <levels.json>] [--chance <folder of <sheet>/chance.png>] [--readers <text-train folder>] \\
        [--before <run>] [--sheet <sheet>]

A run's text-read folder holds reads.npz, reads.parquet (with each box's best paths) and answered.parquet, as
`textread.read` writes them; its boxes are `wordboxes`' boxes.parquet, in the row order the reads are keyed by.
"Before" is the run named by --before, else the run listed last in runs.json other than this one.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from sfzmaps import config  # noqa: E402
from sfzmaps.solver import grid, paths, wordboxes  # noqa: E402
from sfzmaps.solver.textnet import data  # noqa: E402
from sfzmaps.solver.textread import store  # noqa: E402
from sfzmaps.solver.wordread import measures, review  # noqa: E402

BASELINE = "baseline"


def rel(p: Path) -> Path:
    """A path as the page names it: relative to the pipeline folder when inside it."""
    try:
        return Path(p).resolve().relative_to(config.PIPELINE_DIR.resolve())
    except ValueError:
        return Path(p)


def models_lines(sheet: str, chance_dir: Path, readers: Path, levels: dict, boxes: pd.DataFrame,
                 terra: pd.DataFrame, lengths: dict) -> list[str]:
    """The lines naming what drew the page."""
    out = []
    held = chance_dir / "held.json"
    if held.is_file():
        h = json.loads(held.read_text())
        rec = h.get("recipe", {})
        out.append(f"Red ink and our boxes: the four held-out text detectors in {rel(chance_dir) / 'models'} "
                   f"({', '.join(h.get('quarters', paths.QUARTERS))}.pt; {rec.get('iterations')} iterations, "
                   f"crop {rec.get('crop')} px, seed {h.get('seed')}, unknown radius "
                   f"{h.get('unknown_radius', {}).get('px')} px: r{h.get('unknown_radius', {}).get('px')}), each px "
                   "from the detector that never saw its quarter of the city or the answered blocks "
                   f"({rel(chance_dir) / sheet / 'chance.png'}).")
    else:
        out.append(f"Red ink and our boxes: the fold detectors' chance in {rel(chance_dir) / sheet / 'chance.png'}, "
                   "each px from the detector that never saw its quarter.")
    per_q = boxes["quarter"].value_counts().to_dict()
    tq = terra["quarter"].value_counts().to_dict()
    out.append("Boxes grouped at each quarter's level (" + ", ".join(f"{q} {levels[q]:g}" for q in paths.QUARTERS)
               + "); on this sheet, boxes by quarter " + ", ".join(f"{q} {per_q.get(k, 0)}"
                                                                     for k, q in enumerate(paths.QUARTERS))
               + "; Terra's words by quarter " + ", ".join(f"{q} {tq.get(k, 0)}" for k, q in enumerate(paths.QUARTERS))
               + ".")
    th = readers / "held.json"
    if th.is_file():
        t = json.loads(th.read_text())["train"]
        q0 = t[paths.QUARTERS[0]]
        out.append(f"Readings: the four fold readers in {rel(readers)} ({', '.join(paths.QUARTERS)}.pt; "
                   f"{q0['reader']['steps']} steps, synthetic codes {q0['reader']['synth']}, seed {q0['seed']}) and "
                   f"their 0/180 classifiers (<q>_direction.pt, {q0['direction']['steps']} steps), each box read by "
                   "the reader that never saw its quarter; the missed words read at Terra's box by the same fold "
                   "readers on the CPU.")
    out.append("Terra: its saved word boxes and strings (text-on-every-sheet). The sheet: "
               + ", ".join(f"{s} {round(v):,} ft" for s, v in sorted(lengths.items(), key=lambda kv: -kv[1])[:3])
               + " of DataSF's MISSION ST centreline inside each footprint; the page shows the first.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=BASELINE)
    ap.add_argument("--boxes", type=Path, default=wordboxes.BOXES_DIR / "boxes.parquet")
    ap.add_argument("--reads", type=Path, default=store.READ_DIR)
    ap.add_argument("--levels", type=Path, default=wordboxes.BOXES_DIR / "levels.json")
    ap.add_argument("--chance", type=Path, default=paths.READER_TEXT_DIR)
    ap.add_argument("--readers", type=Path, default=store.TRAIN_DIR)
    ap.add_argument("--before", default=None)
    ap.add_argument("--sheet", default=None)
    args = ap.parse_args()
    t0 = time.perf_counter()

    lengths = review.mission_lengths()
    sheet = args.sheet or review.mission_sheet()
    print(f"sheet {sheet}: MISSION ST " + ", ".join(f"{s} {v:,.0f} ft" for s, v in
                                                     sorted(lengths.items(), key=lambda kv: -kv[1]) if v > 0))
    boxes = pd.read_parquet(args.boxes).reset_index(drop=True)
    boxes = boxes[boxes["sheet"] == sheet]
    reads = review.reads_for_page(args.reads)
    lacking = [r for r in boxes.index if reads.get(r, {}).get("text") is None]
    if lacking:
        raise SystemExit(f"{len(lacking)} boxes have no best path in {args.reads}/reads.parquet")
    terra = review.terra_on_sheet(sheet)
    chance = cv2.imread(str(args.chance / sheet / "chance.png"), cv2.IMREAD_GRAYSCALE)
    ink = data.load_page(sheet) < grid.INK_BELOW
    if chance is None or chance.shape != ink.shape:
        raise SystemExit(f"no chance of text the page's size at {args.chance / sheet / 'chance.png'}")
    levels = review.load_levels(args.levels)
    on_map = review.on_the_map(sheet, terra[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2).mean(1))
    missed = review.missed_words(boxes, terra, review.MATCH_IOU)
    terra_reads = review.terra_box_reads(sheet, terra, missed, args.readers)
    print(f"{len(boxes)} boxes, {len(terra)} Terra words, {len(missed)} missed, read at Terra's box "
          f"({time.perf_counter() - t0:.0f} s)")
    answered = args.reads / "answered.parquet"
    answers = measures.answers_from_store(args.reads) if answered.is_file() else None

    root = review.experiments_dir()
    index = root / "runs.json"
    order = json.loads(index.read_text()) if index.is_file() else []
    prev = args.before or next((r for r in reversed(order) if r != args.run), None)
    before = json.loads((root / prev / "numbers.json").read_text()) if prev else None

    d = review.page_data(sheet, boxes, reads, terra, chance, before, ink=ink, levels=levels, on_map=on_map,
                         terra_reads=terra_reads, answers=answers, forms=review.load_forms(),
                         models=models_lines(sheet, args.chance, args.readers, levels, boxes, terra, lengths),
                         run=args.run + (f" (before: {prev})" if prev else ""))
    folder = root / args.run
    out = review.render(d, folder / "review.html")
    (folder / "numbers.json").write_text(json.dumps(d["numbers"], indent=1) + "\n")
    (folder / "missed.json").write_text(json.dumps(d["missed"], indent=1, ensure_ascii=False) + "\n")
    index.write_text(json.dumps([r for r in order if r != args.run] + [args.run], indent=1) + "\n")
    b = d["numbers"]["boxes"][str(review.MATCH_IOU)]
    print(json.dumps({"page": str(out), "mb": round(out.stat().st_size / 1e6, 2), "before": prev,
                      "boxes": {k: b[k] for k in ("terra", "ours", "matched", "missed", "ours_unmatched_text",
                                                  "ours_unmatched_none", "matched_to_missed", "missed_reasons")},
                      "reads": d["numbers"]["reads"]["all"], "headline": d["numbers"].get("headline", {}).get("share"),
                      "red_px": d["text_ink_px"], "seconds": round(time.perf_counter() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
