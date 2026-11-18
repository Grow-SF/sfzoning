"""Task 4's reader experiments: train one quarter's text reader with the read-time distribution of crops, read our word
boxes with it, and write what it scored, to data/solver/experiments/wordread/runs/<name>-<q>-s<seed>/.

    uv run python scripts/wordread/run_reader.py run --name base --quarter ne --seed 0
    uv run python scripts/wordread/run_reader.py run --name a --quarter ne --seed 0 --own --sloppy
    uv run python scripts/wordread/run_reader.py run --name b --quarter ne --seed 0 --own --sloppy --mask
    uv run python scripts/wordread/run_reader.py decide                  # the keep rule, on NE at seeds 0 and 1
    uv run python scripts/wordread/run_reader.py assemble --name a       # the four fold runs -> a text-read folder

The arms (no share is set by hand; the owner's rule, 11-17).  The reader reads only our boxes, so its training inputs
are drawn as read time sees them:
  base  today's recipe: the real crops and 40% synthetic code crops (the stage's own, unchanged).
  own + sloppy   (a) every training Terra word of the 2000 maps is cut with our boxes' measured offsets (`sloppy_crop`:
        an offset near zero, a clean Terra crop, comes up at its measured frequency; so the word's own clean crop leaves
        the real pool: the sloppy crop stands for it), plus every real own-box crop (`own_box_crops`, the kinds drawn
        in the training words' measured shares: `kind_weights="terra"`).  Each pool is drawn in proportion to its
        size: a pool's share of a draw is its count over the counts of the real, own-box and sloppy pools, times the
        share the synthetic crops leave (`natural_shares`); the synthetic crops keep today's recipe's share.
  mask  (b) = (a) and every crop blanked outside the ink touching its own rectangle (training), and our boxes read so.
The 0/180 classifier is the stage's (the recipe is unchanged), shared by every arm, so it adds no difference between them.

Each run writes labels.parquet (per answered label of the quarter: its box, right or wrong, block), headline.json,
box_numbers.json, read_numbers.json, run.json (recipe, shares, pool sizes, training log, the reading at Terra's boxes),
and reader.pt.  The reading runs on the CPU in up to WORKERS processes of one thread; every training is `train.fit`'s
exclusive GPU turn.

The keep rule (written before any run; `decide`): a variant replaces the current reader when, on NE over seeds 0 and 1,
  (1) its headline rises over base, with the paired label-bootstrap interval (`measures.paired_labels`, whole blocks
      resampled) excluding zero, on the labels' mean over the two seeds, and the rise holds at each seed;
  (2) the reading at Terra's boxes (closed top-1 of the answered labels at Terra's box, the classifier weighting the
      crops) does not fall by more than its seed noise: the variant's mean over the seeds minus base's mean is not below
      minus the spread of base's two seeds;
  (3) the controller's look at the review page agrees (not decided here).
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sfzmaps import config  # noqa: E402
from sfzmaps.solver import paths, runs  # noqa: E402
from sfzmaps.solver.textread import ctc, model, store, train  # noqa: E402
from sfzmaps.solver.textread import read as R  # noqa: E402
from sfzmaps.solver.wordread import measures  # noqa: E402

log = logging.getLogger(__name__)

ROOT = runs.EXPERIMENTS_DIR / "wordread"
RUNS = ROOT / "runs"
CACHE = ROOT / "reader" / "cache"
WORKERS = 4
THREADS = 4          # building the bank's sheets
FIRST_SEEDS = (0, 1)
ARMS = {"base": {}, "a": {"own": True, "sloppy": True}, "b": {"own": True, "sloppy": True, "mask": True}}


def run_name(name: str, quarter: str, seed: int) -> str:
    return f"{name}-{quarter}-s{seed}"


def natural_shares(n_real: int, n_own: int, n_sloppy: int, synth_share: float = 0.0) -> tuple[float, float]:
    """The shares of a draw taken from the own-box and the sloppy crops when each pool is drawn in proportion to its
    count (the real pool the rest): the synthetic crops take `synth_share`, the others split what is left by their
    counts.  Nothing is set: only counted."""
    total = n_real + n_own + n_sloppy
    if total <= 0:
        raise ValueError("no crops")
    rest = 1.0 - synth_share
    return rest * n_own / total, rest * n_sloppy / total


# --- the real inputs (replaced in tests) ------------------------------------------------------------------------------

def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=config.PIPELINE_DIR, capture_output=True, text=True).stdout.strip()


class Env:
    """What a run takes from the world: the training crops, the extra crops, the boxes' crops, the answered labels."""

    def __init__(self, boxes_dir: Path | None = None):
        self.boxes_dir = Path(boxes_dir) if boxes_dir else paths.SOLVER_DIR / "boxes"

    def boxes(self) -> pd.DataFrame:
        f = self.boxes_dir / "boxes.parquet"
        if not f.is_file():
            raise FileNotFoundError(f"can't read {f}")
        b = pd.read_parquet(f).reset_index(drop=True)
        b["row"] = np.arange(len(b))
        return b

    def codes(self) -> R.CodeList:
        return R.CodeList.load()

    def answers(self) -> pd.DataFrame:
        return measures.answers_from_store()

    def answered_crops(self) -> tuple[pd.DataFrame, store.Crops]:
        return pd.read_parquet(store.CROPS_DIR / "answered.parquet"), store.Crops.load(store.CROPS_DIR / "answered.npz")

    def terra(self, boxes: pd.DataFrame) -> pd.DataFrame:
        return measures.terra_words(sorted(set(boxes["sheet"])))

    def training(self, q: int, drop_terra_words: bool):
        """The quarter's pool as `build_train` makes it; with `drop_terra_words` without the 2000 maps' Terra words
        (their sloppy crops stand for them).  Returns (pool, rows, n_dropped)."""
        real, rows = train.load_inputs()
        name = paths.QUARTERS[q]
        idx = store.training_rows(rows, q)
        dropped = 0
        if drop_terra_words:
            word = ((rows["series"] == paths.SERIES) & (rows["word"] >= 0) & rows["text"].notna()
                    & (rows["text"].astype(str) != "")).to_numpy()
            keep = ~word[idx]
            dropped = int((~keep).sum())
            idx = idx[keep]
        syn = store.Crops.load(store.CROPS_DIR / f"synth_{name}.npz")
        syn_rows = pd.read_parquet(store.CROPS_DIR / f"synth_{name}.parquet")
        pool = train.Pool(real, rows, idx, syn, list(syn_rows["text"]), train.monitor_words(real, rows, q))
        return pool, rows, dropped

    def direction_pool(self, q: int) -> train.Pool:
        real, rows = train.load_inputs()
        return train.Pool(real, rows, store.training_rows(rows, q, text_only=True))

    def direction(self, name: str):
        return model.load(store.TRAIN_DIR / f"{name}_direction.pt", "direction")

    def extra(self, q: int, seed: int, own: bool, sloppy: bool, mask: bool, boxes: pd.DataFrame, rows: pd.DataFrame):
        """The own-box and sloppy crops and the masks (`wordread.readercrops`), by the quarter's measured offsets."""
        from sfzmaps.solver.textnet import composite, data, reader_stage, zones
        from sfzmaps.solver.wordread import readercrops as RC
        t0 = time.perf_counter()
        pages = RC.DiskPages()
        sha = hashlib.sha1(pd.util.hash_pandas_object(boxes[["sheet", "g0x", "g0y", "g2x", "g2y"]]).values.tobytes()
                           ).hexdigest()[:10]
        CACHE.mkdir(parents=True, exist_ok=True)
        tf = CACHE / "terra_frame.pkl"
        if tf.is_file():
            terra = pickle.loads(tf.read_bytes())
        else:
            terra = RC.terra_frame(pd.read_parquet(store.CROPS_DIR / "crops.parquet"))
            terra[list(RC.LINE_COLUMNS)] = RC.line_measures(terra)
            tf.write_bytes(pickle.dumps(terra))
        df = CACHE / f"dist-{paths.QUARTERS[q]}-{sha}.pkl"
        if df.is_file():
            dist = pickle.loads(df.read_bytes())
        else:
            dist = RC.offset_distribution(boxes, terra, quarter=q, pages=pages)
            df.write_bytes(pickle.dumps(dist))
        log.info("the offsets of %d pairs (%.0f s)", dist["n"], time.perf_counter() - t0)
        own_c, slop_c, masks = [], [], None
        if own:
            own_c = RC.own_box_crops(boxes, terra, quarter=q, pages=pages, mask=mask)
        if sloppy:
            areas = (data.Quarter(q), reader_stage.load_block_area(reader_stage.held_blocks()))
            sheets = data.load_sheets((paths.SERIES,), hold_out_validation=False)
            prepared = zones.prepare(sheets, areas, zones.LINES_DIR, THREADS)
            bank = composite.Bank.build(prepared, areas, seed, THREADS)
            patches = RC.piece_patches(bank, quarter=q, pages=pages)
            del bank, prepared, sheets
            words = terra[RC._training_words(terra, q)]
            lines = RC.calibrate_lines(patches, dist, words, pages=pages, seed=seed)
            log.info("line pieces calibrated: %s", {k: lines.report[k] for k in ("thickness_tv", "cover_tv",
                                                                                  "mean_keep", "pieces_drawable")})
            slop_c = RC.sloppy_crops(terra, dist, lines, seed, quarter=q, pages=pages, mask=mask)
        if mask:
            mf = CACHE / "real_masks.pkl"
            if mf.is_file():
                masks = pickle.loads(mf.read_bytes())
            else:
                masks = RC.real_masks(rows, pages=None)
                mf.write_bytes(pickle.dumps(masks))
        log.info("extra crops: %d own-box, %d sloppy (%.0f s)", len(own_c), len(slop_c), time.perf_counter() - t0)
        return RC.extra_pool(own_c, slop_c, kind_weights="terra" if own_c else None, kind_shares=dist["kind_shares"],
                             real_masks=masks)

    def read(self, qname: str, reader, direction, boxes_q: pd.DataFrame, mask: bool, ans_jobs: list, workers: int):
        """Every box of `boxes_q` and every answered crop (`ans_jobs`: (key, quarter name, crop)) read, as
        `read.build_read` reads them.  Returns {key: read_one's dict}: boxes by their row, answered ones by key."""
        _NETS.update(reader=reader, direction=direction, qname=qname, codes=self.codes())
        jobs = [(s, part.to_dict("records"), mask) for s, part in boxes_q.groupby("sheet", sort=True)]
        chunks = [ans_jobs[k:k + 200] for k in range(0, len(ans_jobs), 200)]
        with get_context("fork").Pool(workers, initializer=_init) as pool:
            got = R._run(pool, R._box_job, jobs)
            got.update(R._run(pool, R._read_job, chunks))
        return got


_NETS: dict = {}


def _init() -> None:
    """A worker: one thread, and the run's networks (made before the fork) in place of the stage's four."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    R._M["readers"] = {_NETS["qname"]: _NETS["reader"]}
    R._M["directions"] = {_NETS["qname"]: _NETS["direction"]}
    R._M["codes"] = _NETS["codes"]


# --- measuring a reading ----------------------------------------------------------------------------------------------

def reads_of(boxes_q: pd.DataFrame, got: dict, codes: R.CodeList) -> dict:
    """`measures`' reads: {row: {"closed": {code: p}, "top": ...}} for the boxes of `boxes_q` (keyed by `row`)."""
    out = {}
    for row in boxes_q["row"]:
        g = got[int(row)]
        c = R.combine(g, R.weight(g), codes)
        op = R.open_by_code(c["post"], codes)
        top = min(op.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        out[int(row)] = {"closed": dict(zip(codes.codes, np.exp(c["log_closed"]).tolist())), "top": top}
    return out


def terra_reading(ans_all: pd.DataFrame, qnum: int, got: dict, codes: R.CodeList) -> dict:
    """The reading at Terra's boxes: the quarter's scored answered labels with a code truth (`measures`' own rule:
    closed top-1), read at Terra's box in Terra's direction and on the axis alone (the classifier weighting each)."""
    d = ans_all.copy()
    d["scored"] = [isinstance(a, str) and (a == "" or a in codes.codes) for a in d["answer"]]
    hit_dir, hit_axis = [], []
    for k, a in enumerate(d.itertuples(index=False)):
        if a.quarter != qnum or not a.scored or a.answer == "" or a.crop < 0:
            continue
        g = got[("ans", k)]
        c = R.combine(g, R.weight(g), codes)
        hit_dir.append(float(np.argmax(c["log_closed"]) == codes.codes.index(a.answer)))
        _, turned = ctc.axis_and_turn(a.angle)
        ga = g if not turned else {"reads": g["reads"][::-1], "p_classifier": g["p_classifier"][::-1]}
        c = R.combine(ga, R.weight(ga), codes)
        hit_axis.append(float(np.argmax(c["log_closed"]) == codes.codes.index(a.answer)))
    out = {"n": len(hit_dir)}
    for key, h in (("terra_direction", hit_dir), ("axis_only", hit_axis)):
        out[key] = round(float(np.mean(h)), 4) if h else None
        out[key + "_ci"] = R.boot_interval(np.array(h)) if h else None
    return out


# --- the run ----------------------------------------------------------------------------------------------------------

def run(name: str, quarter: str, seed: int, own: bool = False, sloppy: bool = False, mask: bool = False,
        boxes_dir: Path | None = None, *, env: Env | None = None, recipe: dict | None = None,
        device: str | None = None, workers: int = WORKERS, out_root: Path | None = None,
        train_direction: bool = False) -> Path:
    """Train quarter `quarter`'s reader with the options and read its boxes of `boxes_dir` (see the module); write
    <out_root or RUNS>/<name>-<quarter>-s<seed>/ and return it.  `recipe` replaces `train.READER` (tests);
    `train_direction`: train the 0/180 classifier at this seed too (else the stage's, shared by every arm)."""
    if quarter not in paths.QUARTERS:
        raise ValueError(f"quarters are {paths.QUARTERS}, not {quarter!r}")
    if mask and not (own or sloppy):
        raise ValueError("the mask arm is the read-time distribution masked: give own and sloppy")
    env = env or Env(boxes_dir)
    q = paths.QUARTERS.index(quarter)
    device = device or train.default_device()
    out = (out_root or RUNS) / run_name(name, quarter, seed)
    t0 = time.perf_counter()
    torch.set_num_threads(train.THREADS)
    cv2.setNumThreads(1)
    boxes = env.boxes()
    codes = env.codes()
    pool, rows, dropped = env.training(q, drop_terra_words=bool(sloppy))
    base_recipe = dict(recipe or train.READER)
    extra = None
    shares = {"own": 0.0, "sloppy": 0.0}
    if own or sloppy:
        extra = env.extra(q, seed, own, sloppy, mask, boxes, rows)
        n_own, n_slop = len(extra["own"]), len(extra["sloppy"])
        shares["own"], shares["sloppy"] = natural_shares(pool.n_real, n_own, n_slop,
                                                         train.SYNTH_SHARE if pool.n_synth else 0.0)
    fit_recipe = {**base_recipe, "own": shares["own"], "sloppy": shares["sloppy"], "mask": bool(mask)}
    log.info("%s: real %d (%d Terra words replaced), synth %d, shares %s", run_name(name, quarter, seed), pool.n_real,
             dropped, pool.n_synth, shares)
    net, lg = train.fit("reader", pool, fit_recipe, device, seed, tag=run_name(name, quarter, seed), extra=extra)
    if train_direction:
        dnet, _ = train.fit("direction", env.direction_pool(q), train.DIRECTION, device, seed,
                            tag=run_name(name, quarter, seed) + " direction")
    else:
        dnet = env.direction(quarter)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "reader.pt", net, quarter=quarter, recipe=fit_recipe, seed=seed)
    trained = time.perf_counter() - t0

    # reading
    boxes_q = boxes[boxes["quarter"] == q]
    ans_all, ans_crops = env.answered_crops()
    ans_jobs = [(("ans", k), quarter, ans_crops[int(c)]) for k, (qq, c) in enumerate(zip(ans_all["quarter"],
                                                                                         ans_all["crop"]))
                if c >= 0 and qq == q]
    got = env.read(quarter, net, dnet, boxes_q, bool(mask), ans_jobs, workers)
    reads = reads_of(boxes_q, got, codes)
    answers = env.answers()
    answers = answers[answers["quarter"] == q].reset_index(drop=True)
    labels = measures.label_right(boxes_q.set_index("row", drop=False), reads, answers)
    labels.to_parquet(out / "labels.parquet", index=False)
    head = measures.headline(boxes_q.set_index("row", drop=False), reads, answers, paths.QUARTERS)
    terra = env.terra(boxes)
    terra = terra[terra["quarter"] == q].reset_index(drop=True)
    bq = boxes_q.set_index("row", drop=False)
    bn = measures.box_numbers(bq, terra, reads)
    rn = {str(iou): measures.read_numbers(bq, terra, reads, forms=dict(zip(codes.forms, codes.form_code)),
                                          answers=answers, iou=iou) for iou in (0.5, 0.3)}
    tr = terra_reading(ans_all, q, got, codes)
    (out / "headline.json").write_text(json.dumps(head, indent=1) + "\n")
    (out / "box_numbers.json").write_text(json.dumps(bn, indent=1, default=float) + "\n")
    (out / "read_numbers.json").write_text(json.dumps(rn, indent=1, default=float) + "\n")
    info = {"name": name, "quarter": quarter, "seed": seed, "own": own, "sloppy": sloppy, "mask": mask,
            "recipe": fit_recipe, "shares": shares, "pools": {"real": pool.n_real, "terra_words_replaced": dropped,
                                                               "synth": pool.n_synth, "own": pool.n_own or
                                                               (len(extra["own"]) if extra else 0),
                                                               "sloppy": len(extra["sloppy"]) if extra else 0},
            "classifier": "trained at this seed" if train_direction else "the stage's (text-train)",
            "terra_reading": tr, "device": device, "train_log": lg, "boxes": str(env.boxes_dir)
            if hasattr(env, "boxes_dir") else None, "git": _git("rev-parse", "HEAD"),
            "src_dirty": _git("status", "--porcelain", "--", "src"), "seconds_train": round(trained, 1),
            "seconds": round(time.perf_counter() - t0, 1), "torch": torch.__version__}
    (out / "run.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
    log.info("%s: headline %s/%s (%.4f), at Terra's boxes %s (%.0f s)", out.name, head["right"], head["n"],
             head["share"] or 0, tr["terra_direction"], time.perf_counter() - t0)
    return out


# --- the decision -----------------------------------------------------------------------------------------------------

def load_run(name: str, quarter: str, seed: int, root: Path | None = None) -> dict:
    d = (root or RUNS) / run_name(name, quarter, seed)
    return {"dir": d, "labels": pd.read_parquet(d / "labels.parquet"),
            "run": json.loads((d / "run.json").read_text()), "headline": json.loads((d / "headline.json").read_text())}


def averaged(runs_: list) -> pd.DataFrame:
    """The labels' right-or-wrong averaged over runs (seeds) of the same labels."""
    base = runs_[0]["labels"][["label", "row", "block", "quarter"]].copy()
    base["right"] = np.mean([r["labels"]["right"].to_numpy(float) for r in runs_], axis=0)
    return base


def decide_rule(base: list, variant: list) -> dict:
    """The keep rule's numbers (conditions 1 and 2; see the module) for a variant's runs against base's, runs of the
    same quarter at seeds in the same order."""
    paired = measures.paired_labels(averaged(variant), averaged(base))
    per_seed = [measures.paired_labels(v["labels"], b["labels"]) for v, b in zip(variant, base)]
    tb = [b["run"]["terra_reading"]["terra_direction"] for b in base]
    tv = [v["run"]["terra_reading"]["terra_direction"] for v in variant]
    noise = max(tb) - min(tb)
    drop = float(np.mean(tv) - np.mean(tb))
    one = paired["lo"] > 0
    each = all(p["diff"] > 0 for p in per_seed)
    two = drop >= -noise
    return {"paired_mean_of_seeds": paired, "paired_by_seed": per_seed, "headline_rises": bool(one and each),
            "terra_base": tb, "terra_variant": tv, "terra_change": drop, "terra_seed_noise": noise,
            "terra_ok": bool(two), "keep_before_the_look": bool(one and each and two)}


def decide(quarter: str = "ne", seeds=FIRST_SEEDS, root: Path | None = None) -> dict:
    base = [load_run("base", quarter, s, root) for s in seeds]
    out = {"quarter": quarter, "seeds": list(seeds)}
    for arm in ("a", "b"):
        try:
            v = [load_run(arm, quarter, s, root) for s in seeds]
        except FileNotFoundError:
            continue
        out[arm] = {"headline": [r["headline"]["share"] for r in v], "ci": [r["headline"]["ci"] for r in v],
                    **decide_rule(base, v)}
    out["base"] = {"headline": [r["headline"]["share"] for r in base], "ci": [r["headline"]["ci"] for r in base]}
    return out


# --- the review page's reads, from the four fold runs -----------------------------------------------------------------

def assemble(name: str, out: Path, seed: int = 0, mask: bool | None = None, workers: int = WORKERS) -> Path:
    """The four fold runs `name` at `seed`: their readers (and the stage's classifiers) put in a folder and every box
    read with `read.build_read` (`mask` as the runs were trained when None), written to `out` as a text-read folder
    (`review_page.py --reads`)."""
    import shutil
    tmp = out.parent / (out.name + "-readers")
    tmp.mkdir(parents=True, exist_ok=True)
    for qn in paths.QUARTERS:
        d = RUNS / run_name(name, qn, seed)
        shutil.copy(d / "reader.pt", tmp / f"{qn}.pt")
        shutil.copy(store.TRAIN_DIR / f"{qn}_direction.pt", tmp / f"{qn}_direction.pt")
        if mask is None:
            mask = bool(json.loads((d / "run.json").read_text())["mask"])
    old = store.TRAIN_DIR, store.READ_DIR
    store.TRAIN_DIR, store.READ_DIR = tmp, out
    try:
        R.build_read(workers, mask=bool(mask))
    finally:
        store.TRAIN_DIR, store.READ_DIR = old
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--name", required=True)
    r.add_argument("--quarter", required=True)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--own", action="store_true")
    r.add_argument("--sloppy", action="store_true")
    r.add_argument("--mask", action="store_true")
    r.add_argument("--boxes-dir", type=Path, default=None)
    d = sub.add_parser("decide")
    d.add_argument("--quarter", default="ne")
    a = sub.add_parser("assemble")
    a.add_argument("--name", required=True)
    a.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "run":
        print(run(args.name, args.quarter, args.seed, args.own, args.sloppy, args.mask, args.boxes_dir))
    elif args.cmd == "decide":
        print(json.dumps(decide(args.quarter), indent=1, default=float))
    else:
        print(assemble(args.name, ROOT / "reader" / f"text-read-{args.name}", args.seed))


if __name__ == "__main__":
    main()
