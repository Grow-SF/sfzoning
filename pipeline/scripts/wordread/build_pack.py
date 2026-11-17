"""San Francisco's context pack for the 2000 map: build it, read our matched boxes, measure each fold's spreads and fit
the CTC temperature and the kinds' weights (Task 10 of the map-text plan; `sfzmaps.solver.wordread.context`).  CPU
only, at most 2 processes x 1 thread, $0.

    uv run python scripts/wordread/build_pack.py build    # the pack's streets, blocks and code list
    uv run python scripts/wordread/build_pack.py read     # our boxes matched to Terra's words, read (frames kept)
    uv run python scripts/wordread/build_pack.py fit      # per IoU level and fold: spreads, fit, held-out scores

Everything lands in data/solver/experiments/wordread/packs/ (never committed):
  sf-2000.parquet        the pack; after `fit`, with the IoU-0.5 spreads of every fold and of "all"
  sf-2000.json           its counts
  sf-2000-words.parquet  our boxes matched to Terra's words at IoU >= 0.3 (Task 1's rule, `wordread.measures`: a box
                         to its best word), with Terra's string and touches, the box's place and kind_p; and their reads
  sf-2000-reads.npz      each box's two reads' log-probabilities (frames x classes, as cut and turned), as
                         `solver-text-read` reads it (its quarter's reader; the classifier's q)
  sf-2000-fit.json       per IoU level (0.3, 0.5, 0.7) and held-out quarter q:
                           - the spreads measured on the level's words of the other three quarters (none touching q or
                             an answered block; the codes on Terra's code words there) by EM
                             (`context.measure_spreads`);
                           - tau and w per kind fitted on those words (`context.fit_prior`);
                           - on q's own words (none in or touching an answered block): the mean log posterior of Terra's
                             string (the pack, the ink at tau alone, the ink as read), top-1 against Terra's string
                             (the pack's decode: its likeliest named string, the best path when OTHER wins; the ink at
                             tau over the same strings; the best path), a reliability table (the decode's posterior,
                             binned for display, against agreement), the words whose Terra string is off the list and
                             whose ink beats every listed string (do they keep it?), and the brief's example (every
                             listed string's ink 1e-6, an off-list string's 0.95) on q's real code-word priors.

Terra's strings are agreement, not truth: these numbers say how the pack moves the reads toward Terra's, never how
many of the owner's answered labels are read right (the headline, Task 11).
"""
import json
import logging
import math
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np
import pandas as pd

from sfzmaps.solver import paths
from sfzmaps.solver.wordread import context as C

log = logging.getLogger("build_pack")

YEAR = 2000
WORKERS = 2
LEVELS = (0.3, 0.5, 0.7)          # the IoU levels screened; the pack keeps PACK_LEVEL's spreads
PACK_LEVEL = 0.5
PACK_FILE = C.PACKS_DIR / f"sf-{YEAR}.parquet"
SUMMARY_FILE = C.PACKS_DIR / f"sf-{YEAR}.json"
WORDS_FILE = C.PACKS_DIR / f"sf-{YEAR}-words.parquet"
READS_FILE = C.PACKS_DIR / f"sf-{YEAR}-reads.npz"
FIT_FILE = C.PACKS_DIR / f"sf-{YEAR}-fit.json"
JOBS_DIR = C.PACKS_DIR / f"sf-{YEAR}-fit-jobs"      # each fit job's result as it lands (a rerun skips those done)
KIND_P = ["p_zoning", "p_block", "p_street", "p_other"]        # wordboxes' columns, in C.KIND_ORDER
TOUCH = (*[f"touch_{q}" for q in paths.QUARTERS], "touch_blocks")   # text-crops' held-out areas a word touches
BINS = np.linspace(0.0, 1.0, 11)  # the reliability table's display bins
OFF_LIST = "ZZZZ"                 # the brief's example's off-list string (no code, street or block prints it)


def _held_blocks() -> set[str]:
    from sfzmaps.solver.textnet import reader_stage
    return set(reader_stage.held_blocks())


def _crop_rows() -> pd.DataFrame:
    """The 2000 drawing's Terra words in text-crops (its rows with a word)."""
    from sfzmaps.solver.textread import store
    rows = pd.read_parquet(store.CROPS_DIR / "crops.parquet")
    return rows[(rows["series"] == paths.SERIES) & (rows["word"] >= 0)].reset_index(drop=True)


def _on_map(df: pd.DataFrame, corners: np.ndarray) -> pd.DataFrame:
    """`C.boxes_on_map` of each row's corners through its sheet's placement."""
    from sfzmaps.solver.textnet import data
    out = pd.DataFrame(index=df.index, columns=["x", "y", "axis"], dtype=float)
    for sheet, idx in df.groupby("sheet", sort=True).groups.items():
        pos = df.index.get_indexer(idx)
        out.loc[idx] = C.boxes_on_map(corners[pos], data.load_transform(sheet)).to_numpy()
    return out.astype(float)


def _fold(words: pd.DataFrame, held: set[str], q: int | None) -> pd.DataFrame:
    """The words a fold may learn from: none in or touching an answered block, none in or touching q."""
    keep = ~words["touch_blocks"].astype(bool) & ~words["block"].astype(str).isin(held)
    if q is not None:
        keep &= (words["quarter"].astype(int) != q) & ~words[f"touch_{paths.QUARTERS[q]}"].astype(bool)
    return words[keep]


def _scored(words: pd.DataFrame, held: set[str], q: int) -> pd.DataFrame:
    """q's own words, none in or touching an answered block."""
    return words[(words["quarter"].astype(int) == q) & ~words["touch_blocks"].astype(bool)
                 & ~words["block"].astype(str).isin(held)]


# --- build ------------------------------------------------------------------------------------------------------------

def build() -> dict:
    t0 = time.perf_counter()
    pack = C.build_sf_pack(YEAR)
    pack.save(PACK_FILE)
    out = {**pack.counts(), "seconds": round(time.perf_counter() - t0, 1)}
    SUMMARY_FILE.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


# --- read -------------------------------------------------------------------------------------------------------------

_M: dict = {}


def _init() -> None:
    import cv2
    import torch

    from sfzmaps.solver.textread import model, store
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    _M["readers"] = {q: model.load(store.TRAIN_DIR / f"{q}.pt") for q in paths.QUARTERS}
    _M["directions"] = {q: model.load(store.TRAIN_DIR / f"{q}_direction.pt", "direction") for q in paths.QUARTERS}


def _read_sheet(job) -> list[dict]:
    """Every matched box of one sheet: its crop (`crops.box_crop`), read as cut and turned by its quarter's reader;
    both reads' log-probabilities, the classifier's q on the crop as cut (`read.weight`) and the likelier way's best
    path."""
    import torch

    from sfzmaps.solver.textnet import data
    from sfzmaps.solver.textread import crops as K
    from sfzmaps.solver.textread import ctc, model
    sheet, items = job
    page = data.load_page(sheet)
    out = []
    for it in items:
        crop = K.box_crop(page, np.asarray(it["corners"], float), it["t_length"], it["t_height"], it["axis"])
        reader, direction = _M["readers"][it["q"]], _M["directions"][it["q"]]
        lps, best, p_cls = [], [], []
        with torch.inference_mode():
            for c in (crop, np.ascontiguousarray(np.rot90(crop, 2))):
                x, n = model.to_input([c])
                lp = ctc.log_softmax(reader(x)[0, :n[0]].numpy())
                lps.append(lp.numpy())
                best.append(model.best_path(lp))
                p_cls.append(float(torch.sigmoid(direction(x))[0]))
        out.append({"key": it["key"], "reads": lps, "q": p_cls[0], "best": best[0] if p_cls[0] < 0.5 else best[1]})
    return out


def matched_words() -> pd.DataFrame:
    """Our boxes matched to Terra's words at IoU >= min(LEVELS) (a box to its best word), with Terra's string (`text`,
    in the reader's alphabet), its kind, block, quarter and touches, the box's IoU, place on the map and kind_p."""
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.wordread import measures
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    terra = measures.terra_words(paths.SHEETS)
    terra["word"] = terra.groupby("sheet").cumcount()
    rows = _crop_rows()
    terra = terra.merge(rows[["sheet", "word", "terra", "text", "block", "quarter", *TOUCH]]
                        .rename(columns={"text": "truth", "quarter": "word_quarter"}), on=["sheet", "word"],
                        how="left", validate="one_to_one")
    if not (terra["terra"] == terra["text"]).all():
        raise ValueError("Terra's words and text-crops' rows disagree: rebuilt at different times?")
    ti, bj, iou = measures._overlaps(boxes, terra)
    word, best = measures._best_word(ti, bj, iou, len(boxes))
    b = np.flatnonzero((best >= min(LEVELS)) & (word >= 0))
    t = terra.iloc[word[b]].reset_index(drop=True)
    keep = (t["truth"].notna() & (t["truth"] != "")).to_numpy()
    b, t = b[keep], t[keep].reset_index(drop=True)
    bx = boxes.iloc[b].reset_index(drop=True)
    words = pd.DataFrame({"row": b, "sheet": bx["sheet"], "box_quarter": bx["quarter"].astype(int),
                          "text": t["truth"], "kind": t["kind"], "block": t["block"].fillna("").astype(str),
                          "quarter": t["word_quarter"].astype(int), "iou": best[b],
                          **{c: t[c].astype(bool) for c in TOUCH}})
    g = bx[measures.G_COLS].to_numpy(float).reshape(-1, 4, 2)
    words = pd.concat([words, _on_map(words, g)], axis=1)
    for k, c in zip(KIND_P, C.KIND_ORDER):
        words[f"kind_p_{c}"] = bx[k].to_numpy(float)
    words["t_corners"] = list(bx[[f"t{i}{a}" for i in range(4) for a in "xy"]].to_numpy(float))
    words["t_length"], words["t_height"], words["axis_page"] = (bx["t_length"].to_numpy(), bx["t_height"].to_numpy(),
                                                               bx["axis"].to_numpy())
    return words


def read() -> dict:
    t0 = time.perf_counter()
    words = matched_words()
    jobs = []
    for sheet, part in words.groupby("sheet", sort=True):
        jobs.append((sheet, [{"key": int(k), "corners": np.asarray(r["t_corners"]).reshape(4, 2),
                              "t_length": float(r["t_length"]), "t_height": float(r["t_height"]),
                              "axis": float(r["axis_page"]), "q": paths.QUARTERS[int(r["box_quarter"])]}
                             for k, r in part.iterrows()]))
    with get_context("fork").Pool(WORKERS, initializer=_init) as pool:
        got = {g["key"]: g for part in pool.imap(_read_sheet, jobs) for g in part}
    words["q"] = [got[int(k)]["q"] for k in words.index]
    words["best"] = [got[int(k)]["best"] for k in words.index]
    reads = [got[int(k)]["reads"] for k in words.index]
    lens = np.array([[r[0].shape[0], r[1].shape[0]] for r in reads], np.int64)
    np.savez_compressed(READS_FILE, lens=lens, one=np.concatenate([r[0] for r in reads]).astype(np.float64),
                        two=np.concatenate([r[1] for r in reads]).astype(np.float64))
    words.drop(columns=["t_corners"]).to_parquet(WORDS_FILE, index=False)
    return {"words": int(len(words)), "by_level": {lv: int((words["iou"] >= lv).sum()) for lv in LEVELS},
            "seconds": round(time.perf_counter() - t0, 1)}


def load_words() -> pd.DataFrame:
    """The read words, each with its two reads (`reads`)."""
    words = pd.read_parquet(WORDS_FILE)
    with np.load(READS_FILE) as z:
        lens, one, two = z["lens"], z["one"], z["two"]
    o1, o2 = np.r_[0, np.cumsum(lens[:, 0])], np.r_[0, np.cumsum(lens[:, 1])]
    words["reads"] = [(one[o1[i]:o1[i + 1]], two[o2[i]:o2[i + 1]]) for i in range(len(words))]
    words["kind_p"] = list(words[[f"kind_p_{c}" for c in C.KIND_ORDER]].to_numpy(float))
    return words


# --- fit --------------------------------------------------------------------------------------------------------------

_F: dict = {}


def _init_fit() -> None:
    import torch
    torch.set_num_threads(1)


def _weights(w: dict) -> dict:
    """The fitted weights for every listing kind (a kind no training word listed: 0, it lifts nothing)."""
    return {k: float(w.get(k, 0.0)) for k in (C.CODE, C.BLOCK, C.STREET)}


def _priors(pack: C.Pack, words: pd.DataFrame, sp: C.Spreads) -> list:
    return [C.prior(pack, {"x": x, "y": y, "axis": a}, kp, sp)
            for x, y, a, kp in zip(words["x"], words["y"], words["axis"], words["kind_p"])]


def _decode(post: dict, best: str) -> tuple[str, float]:
    top, p = max(post.items(), key=lambda kv: (kv[1], kv[0] != C.OTHER))
    return (best if top == C.OTHER else top), math.exp(p)


def score_fold(test: pd.DataFrame, tau: float, w: dict) -> dict:
    """q's own words (see the module)."""
    if not len(test):
        return {}
    ll = {"pack": C.truth_loglik(test, tau, w), "ink_at_tau": C.truth_loglik(test, tau, dict.fromkeys(w, 0.0)),
          "ink_as_read": C.truth_loglik(test, 1.0, dict.fromkeys(w, 0.0))}
    rows = []
    for r in test.itertuples(index=False):
        names = r.prior.strings if r.prior is not None else []
        ink = C.ink_logp(r.reads, r.q, sorted(set(names) | {r.truth}), tau)
        listed = {s: ink[s] for s in names}
        top, p = _decode(C.posterior(listed, r.prior, w), r.best)
        top_ink, _ = _decode(C.posterior(listed, None, {}), r.best)
        off = r.truth not in names
        margin = ink[r.truth] - max(listed.values()) if off and listed else (math.inf if off else -math.inf)
        named = C.posterior(ink, r.prior, w)
        keeps = max(((v, s) for s, v in named.items() if s != C.OTHER))[1] == r.truth
        rows.append({"kind": r.kind, "right_pack": top == r.truth, "right_ink": top_ink == r.truth,
                     "right_best": r.best == r.truth, "p": p, "off_list": off, "margin": margin,
                     "keeps": keeps if off else None})
    d = pd.DataFrame(rows)
    out = {"n": int(len(d)), "truth_off_list": int(d["off_list"].sum()), "by_kind": {}}
    for kind in ["all", *C.KIND_ORDER]:
        sel = np.ones(len(d), bool) if kind == "all" else (d["kind"] == kind).to_numpy()
        if not sel.any():
            continue
        out["by_kind"][kind] = {"n": int(sel.sum()), **{f"mean_loglik_{k}": round(float(v[sel].mean()), 4)
                                                         for k, v in ll.items()},
                                **{f"top1_{k}": round(float(d.loc[sel, f"right_{k}"].mean()), 4)
                                   for k in ("pack", "ink", "best")}}
    stronger = d[d["off_list"] & (d["margin"] > 0)]
    out["off_list_stronger_ink"] = {"n": int(len(stronger)), "keep_their_reading": int(stronger["keeps"].sum()),
                                    "by_kind": stronger.groupby("kind")["keeps"].agg(["size", "sum"]).astype(int)
                                    .to_dict("index")}
    out["reliability"] = reliability(d)
    return out


def reliability(d: pd.DataFrame) -> list[dict]:
    """The pack decode's posterior, in display bins, against the share that agrees with Terra."""
    idx = np.clip(np.digitize(d["p"], BINS) - 1, 0, len(BINS) - 2)
    return [{"bin": [float(BINS[i]), float(BINS[i + 1])], "n": int((idx == i).sum()),
             "mean_p": round(float(d.loc[idx == i, "p"].mean()), 4) if (idx == i).any() else None,
             "agree": round(float(d.loc[idx == i, "right_pack"].mean()), 4) if (idx == i).any() else None}
            for i in range(len(BINS) - 1)]


def brief_example(test: pd.DataFrame, w: dict) -> dict:
    """The brief's example on q's real code-word priors (their kind_p): every listed string's ink 1e-6, an off-list
    string's 0.95 (the rest of the ink elsewhere); does the off-list string win, and by how much."""
    codes = test[test["kind"] == C.CODE]
    wins, margins = [], []
    for pr in codes["prior"]:
        if pr is None or not pr.strings:
            continue
        ink = {**{s: math.log(1e-6) for s in pr.strings}, OFF_LIST: math.log(0.95)}
        post = C.posterior(ink, pr, w)
        best_listed = max(v for s, v in post.items() if s not in (OFF_LIST, C.OTHER))
        wins.append(post[OFF_LIST] > best_listed)
        margins.append(post[OFF_LIST] - best_listed)
    kp = np.array(list(codes["kind_p"])) if len(codes) else np.zeros((0, 4))
    return {"code_words": len(wins), "off_list_wins": int(sum(wins)),
            "min_log_margin": round(float(min(margins)), 3) if margins else None,
            "median_kind_p": dict(zip(C.KIND_ORDER, np.round(np.median(kp, 0), 4).tolist())) if len(kp) else None}


def _fit_job(job) -> dict:
    level, q = job
    pack, words, code_words, held = _F["pack"], _F["words"], _F["code_words"], _F["held"]
    t0 = time.perf_counter()
    lw = words[words["iou"] >= level]
    train = _fold(lw, held, q)
    sp, _ = C.measure_spreads(pack, train, _fold(code_words, held, q), held_blocks=held, quarter=q)
    if q is None:
        return {"level": level, "fold": "all", "spreads": sp}
    test = _scored(lw, held, q).copy()
    train = train.assign(prior=_priors(pack, train, sp), truth=train["text"])
    test["prior"], test["truth"] = _priors(pack, test, sp), test["text"]
    got = C.fit_prior(train, held_blocks=held, quarter=q, joint=True)
    out = {"level": level, "fold": paths.QUARTERS[q], "spreads": sp, "fit": got,
           "train_by_kind": train["kind"].value_counts().to_dict(),
           "held_out": score_fold(test, got["tau"], _weights(got["w"])),
           "brief_example": brief_example(test, _weights(got["w"])),
           "seconds": round(time.perf_counter() - t0, 1)}
    log.info("IoU %.1f %s: tau %.3f w %s (%.0f s)", level, paths.QUARTERS[q], got["tau"],
             {k: round(v, 3) for k, v in got["w"].items()}, out["seconds"])
    return out


def fit() -> dict:
    t0 = time.perf_counter()
    held = _held_blocks()
    pack = C.Pack.load(PACK_FILE)
    rows = _crop_rows()
    code_words = rows[(rows["kind"] == C.CODE) & rows["text"].notna()].reset_index(drop=True)
    _F.update(pack=pack, words=load_words(), held=held, code_words=code_words)
    jobs = [(PACK_LEVEL, None)] + [(lv, q) for lv in LEVELS for q in range(len(paths.QUARTERS))]
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def job_file(job):
        return JOBS_DIR / f"{job[0]}-{'all' if job[1] is None else paths.QUARTERS[job[1]]}.pkl"

    todo = [j for j in jobs if not job_file(j).is_file()]      # a job already done is kept (resumable)
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("fork"), initializer=_init_fit) as pool:
        running = {pool.submit(_fit_job, j): j for j in todo}
        for fut in as_completed(running):                      # a dead worker raises here, never hangs
            job_file(running[fut]).write_bytes(pickle.dumps(fut.result()))
    got = [pickle.loads(job_file(j).read_bytes()) for j in jobs]
    got.sort(key=lambda g: (g["level"], C.FOLDS.index(g["fold"])))
    pack.spreads = {"all": next(g["spreads"] for g in got if g["fold"] == "all")}
    pack.spreads.update({g["fold"]: g["spreads"] for g in got if g["level"] == PACK_LEVEL and g["fold"] != "all"})
    pack.spreads = {f: pack.spreads[f] for f in C.FOLDS}
    pack.save(PACK_FILE)
    SUMMARY_FILE.write_text(json.dumps(pack.counts(), indent=1, default=float) + "\n")
    out = {"levels": {}}
    for g in got:
        if g["fold"] == "all":
            continue
        out["levels"].setdefault(str(g["level"]), {})[g["fold"]] = {
            "spreads": g["spreads"].summary(), **{k: g[k] for k in ("fit", "train_by_kind", "held_out",
                                                                       "brief_example", "seconds")}}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    FIT_FILE.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return {"levels": {lv: {f: {"tau": v["fit"]["tau"], "w": v["fit"]["w"]} for f, v in d.items()}
                       for lv, d in out["levels"].items()}, "seconds": out["seconds"]}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    what = sys.argv[1] if len(sys.argv) > 1 else "build"
    result = {"build": build, "read": read, "fit": fit}[what](*sys.argv[2:])
    print(json.dumps(result, indent=1, default=float)[:6000])
