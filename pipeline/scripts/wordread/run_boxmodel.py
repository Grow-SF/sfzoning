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
    uv run python scripts/wordread/run_boxmodel.py stack [--arm a --seed 0]       # the word heads, stacked by quarter
    uv run python scripts/wordread/run_boxmodel.py train [--quarters ne ..] [--seed 0] [--device mps]

`stack` writes stacked/<sheet>/{chance,core,axis,tag,quarter}.npy and stacked.json: each px's word-head outputs from
the detector run (`detector/runs/<arm>-<q>-s<seed>`) that held out its quarter, so every input looks like a held-out
one.  `train` writes runs/<q>-s<seed>/: model.pt (`boxmodel.save_boxnet`), fit.json (the recipe, the curve: the
count's cross-entropy, the boundary's, the thresholds), examples.parquet (the examples' meta) and heldout.json (the
count's calibration and the boundary columns on the held-out quarter's boxes off the answered blocks, and the Mission
sheet's merged_with_word boxes of that quarter).  One fold after another, each fit in its own exclusive GPU turn.

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

from sfzmaps.solver import grid, paths, wordboxes  # noqa: E402
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


FORMS = {"offsets": tuple(B.OFFSET_FEATURES), "split": tuple(B.MERGE_FEATURES)}


def heldout_boxes(boxes: pd.DataFrame, terra: pd.DataFrame, q: int, touch, shapes: dict) -> pd.DataFrame:
    """Quarter q's boxes whose crop touches no answered block and that meet no answered-block word's rectangle."""
    mine = boxes[boxes["quarter"].to_numpy() == q]
    block = terra[terra["touch_blocks"].to_numpy(bool)]
    keep = np.ones(len(mine), bool)
    sheet_of = mine["sheet"].astype(str).to_numpy()
    for sheet in sorted(set(sheet_of)):
        idx = np.flatnonzero(sheet_of == sheet)
        rects = mine.iloc[idx][B.T_COLS].to_numpy(float).reshape(-1, 4, 2)
        theirs = block[block["sheet"] == sheet][B.C_COLS].to_numpy(float).reshape(-1, 4, 2)
        if len(theirs):
            keep[idx[np.unique(wordboxes.pairs(rects, theirs)[0])]] = False
        for k in idx[keep[idx]]:
            corners, length, height, axis = B._box_geometry(mine.iloc[k])
            keep[k] = not touch(sheet, B._footprint(corners, length, height, axis, shapes[sheet]),
                                shapes[sheet])["touch_blocks"]
    return mine[keep]


def merge_forms(keep, words, boxes, terra, q: int, touch, shapes, sheets) -> dict:
    """Each merge-model form (FORMS) fitted on the fold's training boxes and words and scored by its negative log
    likelihood on the held-out quarter's pairs (its words outside the answered blocks, labelled by its boxes that
    touch no answered block): the choice of form is made held out."""
    train_pairs, train_label = B.merge_data(keep, words, sheets)
    test_words = terra[(terra["quarter"].to_numpy() == q) & ~terra["touch_blocks"].to_numpy(bool)]
    test_pairs, test_label = B.merge_data(heldout_boxes(boxes, terra, q, touch, shapes), test_words, sheets)
    out = {}
    for name, feats in FORMS.items():
        fit = B.fit_merge(train_pairs, train_label, features=feats)
        nll = B.merge_nll(fit, test_pairs, test_label)
        out[name] = {"train_nll": fit["nll"], "heldout_nll": nll, "heldout_pairs": int(len(test_pairs)),
                     "heldout_merged": float(test_label.sum()), "heldout_nll_per_merged": nll / max(test_label.sum(),
                                                                                                    1e-12),
                     "converged": fit["converged"]}
    return out


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
            sq = ex.meta["squeezed"].to_numpy(bool)
            out["folds"][qn]["examples"] = {"n": len(ex), "boxes": int(src.get("box", 0)),
                                            "joined": int(src.get("joined", 0)), "K": ex.K,
                                            "channels": list(ex.channels), "columns": int(ex.flat.shape[2]),
                                            "weight_sum": float(ex.weight.sum()),
                                            "squeezed": {s: int((sq & (ex.meta["source"] == s)).sum())
                                                         for s in ("box", "joined")},
                                            "seconds": round(time.perf_counter() - t1, 1)}
            out["folds"][qn]["merge"] = ex.info
            out["folds"][qn]["merge_form"] = merge_forms(keep, words, boxes, terra, q, touch, shapes, sheets)
            print(qn, "examples", json.dumps(out["folds"][qn]["examples"]))
            print(qn, "merge", json.dumps(ex.info))
            print(qn, "form", json.dumps(out["folds"][qn]["merge_form"]))
        if args.quarters:
            total = {form: sum(out["folds"][qn]["merge_form"][form]["heldout_nll"] for qn in args.quarters)
                     for form in FORMS}
            out["merge_form"] = {"heldout_nll": total, "kept": min(total, key=total.get),
                                 "in_use": "split" if tuple(B.MERGE_FEATURES) == FORMS["split"] else "offsets"}
            print("forms", json.dumps(out["merge_form"]))
    out["seconds"] = round(time.perf_counter() - t0, 1)
    (B.OUT_DIR / "counts.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["today"], indent=1))


# --- the word heads' inputs, stacked by quarter, and training ---------------------------------------------------------

DETECTOR_RUNS = B.OUT_DIR.parent / "detector" / "runs"
STACK_DIR = B.OUT_DIR / "stacked"
RUNS_DIR = B.OUT_DIR / "runs"
STACKED = ("chance", "core", "axis", "tag", "quarter")
MISSED = B.OUT_DIR.parent / "review" / "baseline" / "missed.json"


def stack(args) -> None:
    """Every 2000 page's box-model inputs from the detector runs `<arm>-<q>-s<seed>`, each px from the run that held
    out its quarter (`reader_stage.page_quarters`: the px's map point): the chance (uint8), the core chance (the
    sigmoid of the core logit), the axis (degrees), the tags (float16), and the quarter map, as .npy (memory-mapped
    when read)."""
    from types import SimpleNamespace

    from scipy.special import expit

    from sfzmaps.solver.textnet import data, reader_stage
    cv2.setNumThreads(1)
    t0 = time.perf_counter()
    runs = {q: DETECTOR_RUNS / f"{args.arm}-{q}-s{args.seed}" for q in paths.QUARTERS}
    for q, f in runs.items():
        if not (f / "words").is_dir():
            raise SystemExit(f"no word outputs at {f / 'words'}")
    record = {"runs": {q: str(f.relative_to(B.OUT_DIR.parent)) for q, f in runs.items()}, "sheets": {}}
    for sheet in paths.SHEETS:
        out = STACK_DIR / sheet
        if not args.force and all((out / f"{k}.npy").is_file() for k in STACKED):
            continue
        h, w = data.load_page(sheet).shape
        sd = SimpleNamespace(series=paths.SERIES, sheet=sheet, height=h, width=w,
                             world=data.placement_samples(data.load_transform(sheet), (h, w)))
        quarter = reader_stage.page_quarters(sd)
        chance = np.zeros((h, w), np.uint8)
        core = np.zeros((h, w), np.float16)
        axis = np.zeros((h, w), np.float16)
        tag = None
        for k, q in enumerate(paths.QUARTERS):
            m = quarter == k
            if not m.any():
                continue
            with np.load(runs[q] / "words" / f"{sheet}.npz") as z:
                chance[m] = z["chance"][m]
                core[m] = expit(z["core"][m].astype(np.float32)).astype(np.float16)
                axis[m] = z["geo"][4][m].astype(np.float16)
                t = z["tag"]
                if tag is None:
                    tag = np.zeros((len(t), h, w), np.float16)
                tag[:, m] = t[:, m]
        out.mkdir(parents=True, exist_ok=True)
        for name, arr in (("chance", chance), ("core", core), ("axis", axis), ("tag", tag), ("quarter", quarter)):
            np.save(out / f"{name}.npy", arr)
        record["sheets"][sheet] = {q: int((quarter == k).sum()) for k, q in enumerate(paths.QUARTERS)}
        print(sheet, record["sheets"][sheet], round(time.perf_counter() - t0, 1), flush=True)
    (STACK_DIR / "stacked.json").write_text(json.dumps(record, indent=1) + "\n")


class Stacked:
    """sheet -> `boxmodel.SheetInputs` with the stacked word heads (`stack`), memory-mapped, the last one kept."""

    def __init__(self):
        self.last: dict = {}

    def __getitem__(self, sheet: str) -> B.SheetInputs:
        if sheet not in self.last:
            from sfzmaps.solver.textnet import data
            self.last.clear()
            f = STACK_DIR / sheet
            mm = {k: np.load(f / f"{k}.npy", mmap_mode="r") for k in ("chance", "core", "axis", "tag")}
            self.last[sheet] = B.SheetInputs(ink=data.load_page(sheet) < grid.INK_BELOW, chance=mm["chance"],
                                             core=mm["core"], axis=mm["axis"], tag=mm["tag"])
        return self.last[sheet]


def count_report(T: np.ndarray, lp: np.ndarray, prior: np.ndarray) -> dict:
    """The count's calibration on held-out boxes: `T` their exact targets (n, K_t + 1), `lp` the model's log
    probabilities (`box_logp`), `prior` the training examples' mean target over 0, 1, 2, 3+.  At 0, 1, 2, 3+: the mean
    predicted chance and the mean target; the cross-entropy of the targets (model, and the prior for scale); and the
    reliability of P(count >= 2) by deciles of the prediction (report only)."""
    t = np.concatenate([T[:, :3], T[:, 3:].sum(1, keepdims=True)], 1) if T.shape[1] > 3 else np.pad(
        T, ((0, 0), (0, 4 - T.shape[1])))
    p = np.exp(B.collapse_counts(lp, 2))
    ce = float(-(t * np.log(np.maximum(p, 1e-300))).sum(1).mean())
    ce_prior = float(-(t * np.log(np.maximum(prior, 1e-300))[None]).sum(1).mean())
    two_p, two_t = p[:, 2:].sum(1), t[:, 2:].sum(1)
    order = np.argsort(two_p, kind="stable")
    deciles = []
    for part in np.array_split(order, 10):
        if len(part):
            deciles.append({"n": int(len(part)), "predicted": round(float(two_p[part].mean()), 4),
                            "target": round(float(two_t[part].mean()), 4)})
    return {"n": int(len(T)), "bins": list(BINS),
            "mean_predicted": [round(float(v), 4) for v in p.mean(0)],
            "mean_target": [round(float(v), 4) for v in t.mean(0)],
            "cross_entropy": round(ce, 4), "cross_entropy_prior": round(ce_prior, 4),
            "two_or_more_reliability": deciles}


def boundary_report(bounds: list[np.ndarray], blp: list[np.ndarray]) -> dict:
    """The boundary columns on held-out boxes: per column the target t (soft) against the predicted chance p: the
    mean binary cross-entropy (and of the base rate, for scale), the Brier score, and, calling a boundary at p > 0.5
    (report only), the expected accuracy, recall (sum t [p > .5] / sum t) and precision."""
    t = np.concatenate(bounds).astype(float)
    p = np.exp(np.concatenate([b[:, 1] for b in blp]))
    rate = t.mean()
    eps = 1e-12
    bce = -(t * np.log(np.maximum(p, eps)) + (1 - t) * np.log(np.maximum(1 - p, eps))).mean()
    bce0 = -(t * np.log(max(rate, eps)) + (1 - t) * np.log(max(1 - rate, eps))).mean()
    call = p > 0.5
    return {"columns": int(len(t)), "boundary_share": round(float(rate), 5), "bce": round(float(bce), 5),
            "bce_base_rate": round(float(bce0), 5), "brier": round(float(((p - t) ** 2).mean()), 5),
            "accuracy_at_half": round(float((t * call + (1 - t) * ~call).mean()), 5),
            "recall_at_half": round(float((t * call).sum() / max(t.sum(), eps)), 4),
            "precision_at_half": round(float((t * call).sum() / max(call.sum(), eps)), 4)}


def train(args) -> None:
    """Per held-out quarter: the examples of the training quarters (today's boxes against Terra, plus the merge-drawn
    joined pairs) on the stacked inputs, the box model trained on the GPU (`fit_boxmodel`, RECIPE), and scored on its
    held-out quarter (`held_out_examples`: the quarter's boxes off the answered blocks against every word off them);
    the Mission sheet's merged_with_word boxes (review/baseline/missed.json) each scored by its own quarter's model."""
    from sfzmaps.solver.wordread import readercrops, review
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    terra = load_words()
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    sheets = Stacked()
    shapes = {s: tuple(np.load(STACK_DIR / s / "quarter.npy", mmap_mode="r").shape) for s in paths.SHEETS}
    touch = readercrops.HeldAreas()
    mission = review.mission_sheet()
    for qn in args.quarters:
        q = paths.QUARTERS.index(qn)
        out = RUNS_DIR / f"{qn}-s{args.seed}"
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        words = B.training_words(terra, q)
        keep = B.training_boxes(boxes, terra, quarter=q, touch=touch, shapes=shapes)
        held = terra[~terra.index.isin(words.index)]
        ex = B.box_examples(keep, words, sheets, q, touch=touch, held_words=held)
        ex.meta.to_parquet(out / "examples.parquet", index=False)
        print(qn, "examples", len(ex), "K", ex.K, "channels", ex.channels, round(time.perf_counter() - t0, 1),
              flush=True)
        recipe = {**B.RECIPE, "seed": args.seed, "device": args.device}
        if args.steps:
            recipe["steps"] = args.steps
        net = B.fit_boxmodel(ex, recipe)
        about = {"quarter": qn, "recipe": recipe, "examples": len(ex), "K": ex.K, "channels": list(ex.channels),
                 "stacking": "each px's inputs from the detector run that held out its quarter (stacked.json)",
                 "merge": ex.info.get("merge")}
        B.save_boxnet(out / "model.pt", net, **about)
        (out / "fit.json").write_text(json.dumps({**about, "log": net.fit_log}, indent=1) + "\n")
        print(qn, "trained", json.dumps(net.fit_log["steps"][-1]), flush=True)
        prior = np.concatenate([ex.count[:, :3], ex.count[:, 3:].sum(1, keepdims=True)], 1).mean(0)
        (out / "prior.json").write_text(json.dumps({"bins": list(BINS), "prior": prior.tolist()}) + "\n")
        evaluate_fold(qn, net, prior, out, boxes, terra, sheets, touch, shapes, mission, t0)


def evaluate_fold(qn: str, net, prior: np.ndarray, out, boxes, terra, sheets, touch, shapes, mission: str,
                  t0: float) -> dict:
    """The model holding `qn` out scored on that quarter (heldout.json; per box, heldout.parquet): the count's
    calibration and the boundary columns on the quarter's boxes off the answered blocks; how often its P(count >= 2)
    passes its P(count = 1), by whether the target says two words or more (report only); and the Mission sheet's
    merged_with_word boxes of the quarter (answered blocks included: no model saw them)."""
    q = paths.QUARTERS.index(qn)
    test = heldout_boxes(boxes, terra, q, touch, shapes)
    hx = B.held_out_examples(test, terra[~terra["touch_blocks"].to_numpy(bool)], sheets)
    lp, blp = B.box_logp(net, [hx.crop(i) for i in range(len(hx))])
    report = {"quarter": qn, "boxes": int(len(hx)), "squeezed": int(hx.meta["squeezed"].sum()),
              "count": count_report(hx.count, lp, prior),
              "boundary": boundary_report([hx.target(i)[1] for i in range(len(hx))], blp)}
    target_two = hx.count[:, 2:].sum(1)
    two = np.flatnonzero(target_two > 0.5)
    report["boundary_in_boxes_likelier_two_or_more"] = boundary_report([hx.target(i)[1] for i in two],
                                                                       [blp[i] for i in two])
    pc = np.exp(B.collapse_counts(lp, 2))
    above = pc[:, 2:].sum(1) > pc[:, 1]
    report["two_or_more_above_one"] = {
        "all": [int(above.sum()), int(len(above))],
        "target_two_or_more_likelier": [int(above[target_two > 0.5].sum()), int((target_two > 0.5).sum())],
        "target_two_or_more_less_likely": [int(above[target_two <= 0.5].sum()), int((target_two <= 0.5).sum())]}
    pd.DataFrame({"box": hx.meta["box"], "sheet": hx.meta["sheet"],
                  **{f"target_{b}": np.concatenate([hx.count[:, :3], hx.count[:, 3:].sum(1, keepdims=True)], 1)[:, k]
                     if hx.count.shape[1] > 3 else np.pad(hx.count, ((0, 0), (0, 4 - hx.count.shape[1])))[:, k]
                     for k, b in enumerate(BINS)},
                  **{f"p_{b}": pc[:, k] for k, b in enumerate(BINS)}}).to_parquet(out / "heldout.parquet",
                                                                                 index=False)
    mw = [m for m in json.loads(MISSED.read_text()) if m["reason"] == "merged_with_word" and m.get("best_box")]
    labels = sorted({int(m["best_box"][1:]) for m in mw})
    mine = boxes.loc[[b for b in labels if boxes.at[b, "sheet"] == mission and boxes.at[b, "quarter"] == q]]
    report["mission_merged_with_word"] = {"boxes": 0, "two_or_more_above_one": 0, "rows": []}
    if len(mine):
        mx = B.held_out_examples(mine, terra, sheets)
        mlp, _ = B.box_logp(net, [mx.crop(i) for i in range(len(mx))])
        mp = np.exp(B.collapse_counts(mlp, 2))
        report["mission_merged_with_word"] = {
            "boxes": int(len(mine)), "two_or_more_above_one": int((mp[:, 2:].sum(1) > mp[:, 1]).sum()),
            "target_two_or_more_mean": round(float(mx.count[:, 2:].sum(1).mean()), 4),
            "rows": [{"box": int(b), "p": [round(float(v), 4) for v in row]} for b, row in zip(mine.index, mp)]}
    report["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "heldout.json").write_text(json.dumps(report, indent=1) + "\n")
    print(qn, "held out", json.dumps({k: report[k] for k in ("boxes", "squeezed")}),
          json.dumps(report["count"]["mean_predicted"]), json.dumps(report["count"]["mean_target"]),
          "ce", report["count"]["cross_entropy"], report["count"]["cross_entropy_prior"],
          "boundary", json.dumps(report["boundary"]), "above", json.dumps(report["two_or_more_above_one"]),
          "mission", report["mission_merged_with_word"]["two_or_more_above_one"], "/",
          report["mission_merged_with_word"]["boxes"], flush=True)
    return report


def evaluate(args) -> None:
    """`evaluate_fold` again for saved models (runs/<q>-s<seed>/model.pt and prior.json)."""
    from sfzmaps.solver.wordread import readercrops, review
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    terra = load_words()
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    shapes = {s: tuple(np.load(STACK_DIR / s / "quarter.npy", mmap_mode="r").shape) for s in paths.SHEETS}
    touch = readercrops.HeldAreas()
    mission = review.mission_sheet()
    for qn in args.quarters:
        out = RUNS_DIR / f"{qn}-s{args.seed}"
        prior = np.array(json.loads((out / "prior.json").read_text())["prior"])
        evaluate_fold(qn, B.load_boxnet(out / "model.pt"), prior, out, boxes, terra, Stacked(), touch, shapes,
                      mission, time.perf_counter())


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
    st = sub.add_parser("stack")
    st.add_argument("--arm", default="a")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--force", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--quarters", nargs="*", default=list(paths.QUARTERS), choices=paths.QUARTERS)
    ev.add_argument("--seed", type=int, default=0)
    tr = sub.add_parser("train")
    tr.add_argument("--quarters", nargs="*", default=list(paths.QUARTERS), choices=paths.QUARTERS)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--device", default="mps")
    tr.add_argument("--steps", type=int, default=None)
    f = sub.add_parser("fit")
    f.add_argument("--quarter", required=True, choices=paths.QUARTERS)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--device", default="cpu")
    f.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    {"measure": measure, "stack": stack, "train": train, "evaluate": evaluate, "fit": fit}[args.cmd](args)


if __name__ == "__main__":
    main()
