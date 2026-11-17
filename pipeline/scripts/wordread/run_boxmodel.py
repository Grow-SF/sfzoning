"""Measure the learned box model's targets on today's boxes, and build or fit its examples
(`sfzmaps.solver.wordread.boxmodel`), writing data/solver/experiments/wordread/boxmodel/:

  gaps.json            per held-out quarter, the same-line gaps of its training words (`boxmodel.same_line_gaps`: to
                       the next word on each word's line of text, weighted by the chance it is that); a record only,
                       the joined neighbours don't use them
  counts.json          today's boxes (data/solver/boxes) against every Terra word: the expected number of boxes holding
                       0, 1, 2 and 3+ words (the sums of their Poisson-binomial count distributions; also the counts by
                       each box's likeliest count, for display), overall, by quarter and on the Mission Street sheet;
                       per held-out quarter what its model trains on (boxes, joined neighbours, K), the quantiles of its
                       same-line gaps, and with `--examples` its merge model (`boxmodel.fit_merge`'s coefficients, its
                       chance's tail mass by separation) and the geometry of today's merges against the joined pairs'
  examples_<q>.npz     (`--examples`) the examples of the model holding q out, and their .parquet
  <q>_seed<s>.pt       (`fit`; Task 8's arm (c)) a fitted box model, and its log <q>_seed<s>.json

    uv run python scripts/wordread/run_boxmodel.py measure [--examples] [--quarters ne ...]
    uv run python scripts/wordread/run_boxmodel.py fit --quarter ne --seed 0 [--device mps] [--steps N]

`measure` is CPU only, one thread, $0, and trains nothing.  The words are `solver-text-crops`' rows (every Terra word
of the 2000 sheets, with the held-out areas its crop touches); the chance of text is the fold detectors'
(`text-reader/<sheet>/chance.png`); without the word heads' predictions (Task 8) the crops have two channels.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sfzmaps.solver import paths, wordboxes  # noqa: E402
from sfzmaps.solver.textnet import boxes as textboxes  # noqa: E402
from sfzmaps.solver.textread import store  # noqa: E402
from sfzmaps.solver.wordread import boxmodel as B  # noqa: E402

BINS = ("0", "1", "2", "3+")


class Sheets:
    """sheet -> `boxmodel.SheetInputs`, read when asked, the last one kept."""

    def __init__(self):
        self.last: dict = {}

    def __getitem__(self, sheet: str) -> B.SheetInputs:
        if sheet not in self.last:
            self.last.clear()
            self.last[sheet] = B.load_sheet(sheet)
        return self.last[sheet]


def binned(dists: list[np.ndarray]) -> dict:
    """The expected number of boxes holding 0, 1, 2 and 3+ words (sums of their count distributions), and the number
    by each box's likeliest count (display only)."""
    expected = np.zeros(4)
    likeliest = np.zeros(4, int)
    for d in dists:
        expected += [d[0], d[1:2].sum(), d[2:3].sum(), d[3:].sum()]
        likeliest[min(int(np.argmax(d)), 3)] += 1
    return {"boxes": len(dists), "expected": dict(zip(BINS, np.round(expected, 2).tolist())),
            "likeliest": dict(zip(BINS, likeliest.tolist()))}


def load_words() -> pd.DataFrame:
    rows = pd.read_parquet(store.CROPS_DIR / "crops.parquet")
    terra = B.terra_frame(rows)
    recs = textboxes.terra_records()
    for sheet in paths.SHEETS:
        have = int((terra["sheet"] == sheet).sum())
        want = len(textboxes.terra_boxes(paths.SERIES, sheet, recs))
        if have != want:
            raise SystemExit(f"{sheet}: {have} words in solver-text-crops, Terra has {want}: rebuild text-crops")
    return terra


def measure(args) -> None:
    from sfzmaps.solver.wordread import readercrops, review
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    t0 = time.perf_counter()
    terra = load_words()
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    sheets = Sheets()
    shapes, dists = {}, [None] * len(boxes)
    turned_s = turned_n = 0.0
    for sheet in sorted(set(boxes["sheet"])):
        inputs = sheets[sheet]
        shapes[sheet] = inputs.shape
        mine = boxes[boxes["sheet"] == sheet]
        got = B.box_targets(mine, terra[terra["sheet"] == sheet], inputs.ink)
        for label, t in zip(mine.index, got):
            dists[label] = t["count"]
            turned_s += float((t["shares"] * t["turned"]).sum())
            turned_n += float(t["shares"].sum())
    mission = review.mission_sheet()
    quarter = boxes["quarter"].to_numpy()
    out = {"today": {"all": binned(dists),
                     "by_quarter": {qn: binned([dists[i] for i in np.flatnonzero(quarter == q)])
                                    for q, qn in enumerate(paths.QUARTERS)},
                     "mission_sheet": {"sheet": mission,
                                       **binned([dists[i] for i in np.flatnonzero(boxes["sheet"] == mission)])},
                     "largest_count_with_any_chance": int(max(len(d) - 1 for d in dists)),
                     "turned_share": turned_s / turned_n if turned_n else None},
           "words": int(len(terra)), "boxes": int(len(boxes)), "folds": {}}
    touch = readercrops.HeldAreas()
    gaps = {}
    for qn in args.quarters:
        q = paths.QUARTERS.index(qn)
        words = B.training_words(terra, q)
        keep = B.training_boxes(boxes, terra, quarter=q, touch=touch, shapes=shapes)
        d = [dists[i] for i in keep.index]
        gaps[qn] = B.gap_distribution(B.same_line_gaps(words))
        out["folds"][qn] = {"training_words": int(len(words)), "boxes": int(len(keep)), **binned(d),
                            "joined": int(round(sum(float(x[2:].sum()) for x in d))),
                            "K": int(max([len(x) - 1 for x in d] + [2])),
                            "same_line_gaps": {"pairs": gaps[qn]["n"], "weight_sum": round(gaps[qn]["weight_sum"], 2),
                                               "quantiles": gaps[qn]["quantiles"]}}
        print(qn, json.dumps(out["folds"][qn]))
    B.OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.quarters:
        B.GAPS_FILE.write_text(json.dumps({"definition": " ".join(B.same_line_gaps.__doc__.split()),
                                           "words": "solver-text-crops' Terra words of the 2000 sheets, each "
                                                    "quarter's training words", "quarters": gaps}) + "\n")
    if args.examples:
        for qn in args.quarters:
            q = paths.QUARTERS.index(qn)
            t1 = time.perf_counter()
            words = B.training_words(terra, q)
            keep = B.training_boxes(boxes, terra, quarter=q, touch=touch, shapes=shapes)
            held = terra[~terra.index.isin(words.index)]
            ex = B.box_examples(keep, words, sheets, q, touch=touch, held_words=held)
            ex.save(B.OUT_DIR / f"examples_{qn}")
            src = ex.meta["source"].value_counts().to_dict()
            out["folds"][qn]["examples"] = {"n": len(ex), "boxes": int(src.get("box", 0)),
                                            "joined": int(src.get("joined", 0)), "K": ex.K,
                                            "channels": list(ex.channels), "columns": int(ex.flat.shape[2]),
                                            "weight_sum": float(ex.weight.sum()),
                                            "seconds": round(time.perf_counter() - t1, 1)}
            out["folds"][qn]["merge"] = ex.info
            print(qn, "examples", json.dumps(out["folds"][qn]["examples"]))
            print(qn, "merge", json.dumps(ex.info))
    out["seconds"] = round(time.perf_counter() - t0, 1)
    (B.OUT_DIR / "counts.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["today"], indent=1))


def fit(args) -> None:
    """Task 8's arm (c): fit the box model holding `--quarter` out on its saved examples."""
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    ex = B.Examples.load(B.OUT_DIR / f"examples_{args.quarter}")
    recipe = {**B.RECIPE, "seed": args.seed, "device": args.device}
    if args.steps:
        recipe["steps"] = args.steps
    net = B.fit_boxmodel(ex, recipe)
    name = f"{args.quarter}_seed{args.seed}"
    B.save_boxnet(B.OUT_DIR / f"{name}.pt", net, quarter=args.quarter, recipe=recipe, examples=len(ex))
    (B.OUT_DIR / f"{name}.json").write_text(json.dumps(net.fit_log, indent=1) + "\n")
    print(json.dumps(net.fit_log["steps"][-1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--quarters", nargs="*", default=list(paths.QUARTERS), choices=paths.QUARTERS)
    m.add_argument("--examples", action="store_true")
    f = sub.add_parser("fit")
    f.add_argument("--quarter", required=True, choices=paths.QUARTERS)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", default="cpu")
    f.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    measure(args) if args.cmd == "measure" else fit(args)


if __name__ == "__main__":
    main()
