"""San Francisco's context pack for the 2000 map: build it, read our boxes, measure each fold's spreads and fit the
CTC temperature and the kinds' weights (Task 10 of the map-text plan; `sfzmaps.solver.wordread.context`).  CPU only,
at most 2 processes x 1 thread, $0.

    uv run python scripts/wordread/build_pack.py build    # the pack's streets, blocks and code list
    uv run python scripts/wordread/build_pack.py read     # every (our box, Terra word) pair with its weight; boxes read
    uv run python scripts/wordread/build_pack.py fit      # per held-out quarter: spreads, fits, held-out scores
    uv run python scripts/wordread/build_pack.py blocks   # the block numbers the pack overrules, and example crops

No IoU anywhere: every pair of one of our boxes and a Terra word on its sheet counts with the chance the box holds the
word, the product of the word's share of its ink inside our box's (trimmed) rectangle and our box's share of its ink
inside the word's rectangle (`wordread.boxmodel.ink_shares` both ways, on the page's ink, `grid.INK_BELOW`); a pair
whose product is 0 adds nothing.  A pair's target is the part of Terra's string inside our box: its characters spread
evenly along the word's rectangle (no per-character widths are measured), each inside by the share of its ink px in our
box (`context.char_shares`), the likelihood the expectation over the contiguous runs they make
(`context.substring_targets`); a word wholly inside keeps its whole string.

Everything lands in data/solver/experiments/wordread/packs/ (never committed):
  sf-2000.parquet        the pack; after `fit`, with the spreads of every fold and of "all"
  sf-2000.json           its counts
  sf-2000-pairs.parquet  the pairs: the box, the word's string (in the reader's alphabet), kind, block, quarter and
                         touches, the weight, the box's place on the map and kind_p
  sf-2000-reads.npz      each paired box's two reads' log-probabilities (frames x classes, as cut and turned), as
                         `solver-text-read` reads it (its quarter's reader; the classifier's q), and its best path
  sf-2000-fit.json       per held-out quarter q, fitted on the pairs whose word lies in the other three quarters
                         (touching neither q nor an answered block; the codes on Terra's code words there):
                           - the spreads (`context.measure_spreads`, weighted EM);
                           - tau and w per kind, sequentially (tau on the ink alone, then w) and jointly
                             (`context.fit_prior`), the mode kept by the pooled held-out NLL;
                           - a temperature on the combined log posterior, fitted on the other three quarters' held-out
                             predictions (`context.fit_temperature`), so q's own words never set it;
                           - on q's own pairs (none in or touching an answered block), weighted: the NLL of Terra's
                             string, top-1 against it (the pack's decode: its likeliest listed string, the best path
                             when OTHER wins; the ink at tau over the same strings; the best path), the reliability
                             (the decode's posterior in display bins against agreement), the off-list strings whose
                             ink beats every listed string (kept?), and the brief's example on q's code-pair priors.
  block_disagreements/   the held-out block-number pairs holding their whole word, Terra's string off the list and its
                         ink stronger, that the pack overrules: their edge distances, how their numbers relate to today's
                         blocks
                         (summary.json), and 10 example crops (ours and the page around it, with index.json)

Terra's strings are agreement, not truth: these numbers say how the pack moves the reads toward Terra's, never how
many of the owner's answered labels are read right (the headline, Task 11).
"""
import json
import logging
import math
import pickle
import re
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
PACK_FILE = C.PACKS_DIR / f"sf-{YEAR}.parquet"
SUMMARY_FILE = C.PACKS_DIR / f"sf-{YEAR}.json"
PAIRS_FILE = C.PACKS_DIR / f"sf-{YEAR}-pairs.parquet"
READS_FILE = C.PACKS_DIR / f"sf-{YEAR}-reads.npz"
FIT_FILE = C.PACKS_DIR / f"sf-{YEAR}-fit.json"
JOBS_DIR = C.PACKS_DIR / f"sf-{YEAR}-fit-jobs"      # each fold's result as it lands (a rerun skips those done)
BLOCKS_DIR = C.PACKS_DIR / "block_disagreements"
KIND_P = ["p_zoning", "p_block", "p_street", "p_other"]        # wordboxes' columns, in C.KIND_ORDER
TOUCH = (*[f"touch_{q}" for q in paths.QUARTERS], "touch_blocks")   # text-crops' held-out areas a word touches
T_COLS = [f"t{i}{a}" for i in range(4) for a in "xy"]              # a box's trimmed rectangle
BINS = np.linspace(0.0, 1.0, 11)  # the reliability table's display bins
OFF_LIST = "ZZZZ"                 # the brief's example's off-list string (no code, street or block prints it)
MODES = ("sequential", "joint")
EXAMPLES = 10


def _held_blocks() -> set[str]:
    from sfzmaps.solver.textnet import reader_stage
    return set(reader_stage.held_blocks())


def _crop_rows() -> pd.DataFrame:
    """The 2000 drawing's Terra words in text-crops (its rows with a word)."""
    from sfzmaps.solver.textread import store
    rows = pd.read_parquet(store.CROPS_DIR / "crops.parquet")
    return rows[(rows["series"] == paths.SERIES) & (rows["word"] >= 0)].reset_index(drop=True)


def _fold(words: pd.DataFrame, held: set[str], q: int | None) -> pd.DataFrame:
    """The pairs a fold may learn from: the word in or touching no answered block, nor in or touching q."""
    keep = ~words["touch_blocks"].astype(bool) & ~words["block"].astype(str).isin(held)
    if q is not None:
        keep &= (words["quarter"].astype(int) != q) & ~words[f"touch_{paths.QUARTERS[q]}"].astype(bool)
    return words[keep]


def _scored(words: pd.DataFrame, held: set[str], q: int) -> pd.DataFrame:
    """q's own pairs, the word in or touching no answered block."""
    return words[(words["quarter"].astype(int) == q) & ~words["touch_blocks"].astype(bool)
                 & ~words["block"].astype(str).isin(held)]


def _wmean(x, w) -> float | None:
    w = np.asarray(w, float)
    return round(float((np.asarray(x, float) * w).sum() / w.sum()), 4) if w.sum() > 0 else None


# --- build ------------------------------------------------------------------------------------------------------------

def build() -> dict:
    t0 = time.perf_counter()
    pack = C.build_sf_pack(YEAR)
    pack.save(PACK_FILE)
    out = {**pack.counts(), "seconds": round(time.perf_counter() - t0, 1)}
    SUMMARY_FILE.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


# --- read: the pairs and the reads ------------------------------------------------------------------------------------

_M: dict = {}


def _init_read() -> None:
    import cv2
    import torch

    from sfzmaps.solver.textread import model, store
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    _M["readers"] = {q: model.load(store.TRAIN_DIR / f"{q}.pt") for q in paths.QUARTERS}
    _M["directions"] = {q: model.load(store.TRAIN_DIR / f"{q}_direction.pt", "direction") for q in paths.QUARTERS}


def _sheet_job(job) -> dict:
    """One sheet: every (box, Terra word) pair with its weight (see the module), and every paired box read as cut
    and turned by its quarter's reader (both reads' log-probabilities, the classifier's q on the crop as cut, the
    likelier way's best path)."""
    import torch

    from sfzmaps.solver import grid
    from sfzmaps.solver.textnet import data
    from sfzmaps.solver.textread import crops as K
    from sfzmaps.solver.textread import ctc, model
    from sfzmaps.solver.wordread import boxmodel, measures
    sheet, boxes, terra = job
    page = data.load_page(sheet)
    ink = page < grid.INK_BELOW
    rects = boxes[T_COLS].to_numpy(float).reshape(-1, 4, 2)
    words = terra[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2)
    bi, wj, s_word = boxmodel.ink_shares(rects, words, ink)        # the word's share of its ink inside the box
    wj2, bi2, s_box = boxmodel.ink_shares(words, rects, ink)       # the box's share of its ink inside the word
    box_share = dict(zip(zip(bi2.tolist(), wj2.tolist()), s_box.tolist()))
    pairs = [(int(b), int(w), float(s) * box_share.get((int(b), int(w)), 0.0)) for b, w, s in zip(bi, wj, s_word)]
    pairs = [p for p in pairs if p[2] > 0]
    truths = terra["truth"].tolist()
    word_ink: dict[int, tuple] = {}
    targets = []
    for b, w, _ in pairs:                       # the part of the word inside the box (`context.substring_targets`)
        t = truths[w]
        if not isinstance(t, str) or not t:
            targets.append([])
            continue
        if w not in word_ink:
            word_ink[w] = boxmodel._ink_of(words[w], ink)
        targets.append(C.substring_targets(t, C.char_shares(t, words[w], rects[b], *word_ink[w])))
    reads = {}
    for b in sorted({p[0] for p in pairs}):
        r = boxes.iloc[b]
        corners = r[T_COLS].to_numpy(float).reshape(4, 2)
        crop = K.box_crop(page, corners, r["t_length"], r["t_height"], r["axis"])
        quarter = paths.QUARTERS[int(r["quarter"])]
        lps, best, p_cls = [], [], []
        with torch.inference_mode():
            for c in (crop, np.ascontiguousarray(np.rot90(crop, 2))):
                x, n = model.to_input([c])
                lp = ctc.log_softmax(_M["readers"][quarter](x)[0, :n[0]].numpy())
                lps.append(lp.numpy())
                best.append(model.best_path(lp))
                p_cls.append(float(torch.sigmoid(_M["directions"][quarter](x))[0]))
        reads[int(boxes.index[b])] = {"reads": lps, "q": p_cls[0], "best": best[0] if p_cls[0] < 0.5 else best[1]}
    return {"sheet": sheet, "pairs": [(int(boxes.index[b]), int(terra.index[w]), s, json.dumps(tg))
                                      for (b, w, s), tg in zip(pairs, targets)], "reads": reads}


def read() -> dict:
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.textnet import data
    from sfzmaps.solver.wordread import measures
    t0 = time.perf_counter()
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    terra = measures.terra_words(paths.SHEETS)
    terra["word"] = terra.groupby("sheet").cumcount()
    rows = _crop_rows()
    terra = terra.merge(rows[["sheet", "word", "terra", "text", "block", "quarter", *TOUCH]]
                        .rename(columns={"text": "truth", "quarter": "word_quarter"}), on=["sheet", "word"],
                        how="left", validate="one_to_one")
    if not (terra["terra"] == terra["text"]).all():
        raise ValueError("Terra's words and text-crops' rows disagree: rebuilt at different times?")
    jobs = [(s, boxes[boxes["sheet"] == s], terra[terra["sheet"] == s]) for s in paths.SHEETS]
    got = []
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("fork"), initializer=_init_read) as pool:
        for fut in as_completed([pool.submit(_sheet_job, j) for j in jobs]):
            got.append(fut.result())
    reads = {k: v for g in got for k, v in g["reads"].items()}
    pairs = pd.DataFrame([p for g in got for p in g["pairs"]], columns=["box", "word", "weight", "targets"])
    pairs = pairs.sort_values(["box", "word"]).reset_index(drop=True)
    t = terra.loc[pairs["word"]].reset_index(drop=True)
    bx = boxes.loc[pairs["box"]].reset_index(drop=True)
    out = pd.DataFrame({"box": pairs["box"], "word": pairs["word"], "weight": pairs["weight"],
                        "targets": pairs["targets"], "sheet": bx["sheet"],
                        "box_quarter": bx["quarter"].astype(int), "text": t["truth"], "kind": t["kind"],
                        "block": t["block"].fillna("").astype(str), "quarter": t["word_quarter"].astype(int),
                        **{c: t[c].astype(bool) for c in TOUCH}})
    out = out[out["text"].notna() & (out["text"] != "") & (out["targets"] != "[]")].reset_index(drop=True)
    on_map = []
    for sheet, part in boxes.loc[sorted(set(out["box"]))].groupby("sheet", sort=True):
        g = part[measures.G_COLS].to_numpy(float).reshape(-1, 4, 2)
        on_map.append(C.boxes_on_map(g, data.load_transform(sheet)).set_index(part.index))
    on_map = pd.concat(on_map)
    out = pd.concat([out, on_map.loc[out["box"]].reset_index(drop=True)], axis=1)
    for k, c in zip(KIND_P, C.KIND_ORDER):
        out[f"kind_p_{c}"] = boxes.loc[out["box"], k].to_numpy(float)
    out["q"] = [reads[b]["q"] for b in out["box"]]
    out["best"] = [reads[b]["best"] for b in out["box"]]
    keys = np.array(sorted(reads), np.int64)
    lens = np.array([[reads[k]["reads"][0].shape[0], reads[k]["reads"][1].shape[0]] for k in keys], np.int64)
    np.savez_compressed(READS_FILE, boxes=keys, lens=lens,
                        one=np.concatenate([reads[k]["reads"][0] for k in keys]).astype(np.float64),
                        two=np.concatenate([reads[k]["reads"][1] for k in keys]).astype(np.float64))
    out.to_parquet(PAIRS_FILE, index=False)
    whole = [json.loads(tg) == [[s, 1.0]] for tg, s in zip(out["targets"], out["text"])]
    return {"pairs": int(len(out)), "boxes": int(out["box"].nunique()), "words": int(out["word"].nunique()),
            "weight": round(float(out["weight"].sum()), 1), "whole_word_pairs": int(sum(whole)),
            "whole_word_weight": round(float(out.loc[whole, "weight"].sum()), 1),
            "seconds": round(time.perf_counter() - t0, 1)}


def load_pairs() -> pd.DataFrame:
    """The pairs, each with its box's two reads (`reads`) and kind_p."""
    pairs = pd.read_parquet(PAIRS_FILE)
    with np.load(READS_FILE) as z:
        keys, lens, one, two = z["boxes"], z["lens"], z["one"], z["two"]
    o1, o2 = np.r_[0, np.cumsum(lens[:, 0])], np.r_[0, np.cumsum(lens[:, 1])]
    by = {int(k): (one[o1[i]:o1[i + 1]], two[o2[i]:o2[i + 1]]) for i, k in enumerate(keys)}
    pairs["reads"] = [by[int(b)] for b in pairs["box"]]
    pairs["kind_p"] = list(pairs[[f"kind_p_{c}" for c in C.KIND_ORDER]].to_numpy(float))
    pairs["targets"] = [[(s, float(p)) for s, p in json.loads(tg)] for tg in pairs["targets"]]
    pairs["truth"] = [tg[0][0] for tg in pairs["targets"]]       # the likeliest part (the word itself when whole)
    return pairs


# --- fit --------------------------------------------------------------------------------------------------------------

_F: dict = {}


def _init_fit() -> None:
    import torch
    torch.set_num_threads(1)


def _weights(w: dict) -> dict:
    """The fitted weights for every listing kind (a kind no training pair listed: 0, it lifts nothing)."""
    return {k: float(w.get(k, 0.0)) for k in (C.CODE, C.BLOCK, C.STREET)}


def _priors(pack: C.Pack, words: pd.DataFrame, sp: C.Spreads) -> list:
    """Each row's prior, computed once per box."""
    by = {}
    for b, x, y, a, kp in zip(words["box"], words["x"], words["y"], words["axis"], words["kind_p"]):
        if b not in by:
            by[b] = C.prior(pack, {"x": x, "y": y, "axis": a}, kp, sp)
    return [by[b] for b in words["box"]]


def _decode(labels: list, logpost: np.ndarray, best: str) -> tuple[str, float]:
    """The likeliest hypothesis (ties: a named string before OTHER) as a string, and its probability."""
    order = sorted(range(len(labels)), key=lambda i: (-logpost[i], labels[i] == C.OTHER))
    top = labels[order[0]]
    return (best if top == C.OTHER else top), float(math.exp(logpost[order[0]]))


def _agree(top: str, r: dict) -> float:
    """How far a decode agrees with Terra at this pair: the chance of the part of the word it holds that it equals."""
    return float(sum(p for s, p in r["parts"] if s == top))


def held_out(test: pd.DataFrame, tau: float, w: dict) -> dict:
    """q's own pairs under (tau, w): per row the log posterior over its listed strings and its targets then OTHER
    (the NLL of the expected target), the decode's posterior over the listed strings then OTHER (`posterior`), and
    the facts the scores need (the off-list facts of the likeliest target, `truth`)."""
    rows = C.truth_rows(test, tau, w)
    out = []
    for (lp, k), r in zip(rows, test.itertuples(index=False)):
        names = r.prior.strings if r.prior is not None else []
        ink = C.ink_logp(r.reads, r.q, sorted(set(names) | {r.truth}), tau)
        listed = {s: ink[s] for s in names}
        post = C.posterior(listed, r.prior, w)
        alone = C.posterior(listed, None, {})
        off = r.truth not in names
        margin = (ink[r.truth] - max(listed.values()) if listed else math.inf) if off else -math.inf
        named = C.posterior(ink, r.prior, w)
        out.append({"lp": lp, "k": k, "parts": list(r.targets), "labels": list(post),
                    "post": np.array(list(post.values())),
                    "ink_top": _decode(list(alone), np.array(list(alone.values())), r.best)[0],
                    "best": r.best, "truth": r.truth, "kind": r.kind, "weight": float(r.weight), "off": off,
                    "margin": margin,
                    "keeps": (max(((v, s) for s, v in named.items() if s != C.OTHER))[1] == r.truth) if off else None,
                    "box": int(r.box), "sheet": r.sheet, "x": r.x, "y": r.y,
                    "top_block": max(r.prior.items.get(C.BLOCK, []), key=lambda it: it[0])[1] if r.prior is not None
                    and r.prior.items.get(C.BLOCK) else None})
    return {"rows": out}


def score(rows: list[dict], temperature: float = 1.0) -> dict:
    """The held-out scores (see the module) of `held_out` rows at a temperature on the combined posterior."""
    if not rows:
        return {}
    v = np.array([r["weight"] for r in rows])
    nll = np.array([-C.target_logp(C.tempered(r["lp"], temperature), r["k"]) for r in rows])
    dec = [_decode(r["labels"], C.tempered(r["post"], temperature), r["best"]) for r in rows]
    right = np.array([_agree(d[0], r) for d, r in zip(dec, rows)], float)
    p = np.array([d[1] for d in dec])
    out = {"pairs": len(rows), "weight": round(float(v.sum()), 2), "nll": _wmean(nll, v), "by_kind": {}}
    for kind in ["all", *C.KIND_ORDER]:
        sel = np.ones(len(rows), bool) if kind == "all" else np.array([r["kind"] == kind for r in rows])
        if not sel.any():
            continue
        out["by_kind"][kind] = {"weight": round(float(v[sel].sum()), 2), "nll": _wmean(nll[sel], v[sel]),
                                "top1_pack": _wmean(right[sel], v[sel]),
                                "top1_ink": _wmean([_agree(r["ink_top"], r) for r, s in zip(rows, sel) if s], v[sel]),
                                "top1_best": _wmean([_agree(r["best"], r) for r, s in zip(rows, sel) if s], v[sel])}
    idx = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    out["reliability"] = [{"bin": [float(BINS[i]), float(BINS[i + 1])], "weight": round(float(v[idx == i].sum()), 2),
                           "mean_p": _wmean(p[idx == i], v[idx == i]), "agree": _wmean(right[idx == i], v[idx == i])}
                          for i in range(len(BINS) - 1) if (idx == i).any()]
    strong = [r for r in rows if r["off"] and r["margin"] > 0]
    out["off_list_stronger_ink"] = {"pairs": len(strong), "weight": round(sum(r["weight"] for r in strong), 2),
                                    "kept_weight": round(sum(r["weight"] for r in strong if r["keeps"]), 2),
                                    "kept_pairs": sum(bool(r["keeps"]) for r in strong),
                                    "by_kind": {k: [sum(r["kind"] == k for r in strong),
                                                    sum(bool(r["keeps"]) and r["kind"] == k for r in strong)]
                                                for k in C.KIND_ORDER if any(r["kind"] == k for r in strong)}}
    return out


def brief_example(test: pd.DataFrame, w: dict) -> dict:
    """The brief's example on q's code pairs' real priors (their kind_p): every listed string's ink 1e-6, an off-list
    string's 0.95; does the off-list string win, and by how much (each box once)."""
    codes = test[test["kind"] == C.CODE].drop_duplicates("box")
    wins, margins = [], []
    for pr in codes["prior"]:
        if pr is None or not pr.strings:
            continue
        post = C.posterior({**{s: math.log(1e-6) for s in pr.strings}, OFF_LIST: math.log(0.95)}, pr, w)
        best_listed = max(v for s, v in post.items() if s not in (OFF_LIST, C.OTHER))
        wins.append(post[OFF_LIST] > best_listed)
        margins.append(post[OFF_LIST] - best_listed)
    return {"code_boxes": len(wins), "off_list_wins": int(sum(wins)),
            "min_log_margin": round(float(min(margins)), 3) if margins else None}


def _fit_job(q) -> dict:
    pack, words, code_words, held = _F["pack"], _F["words"], _F["code_words"], _F["held"]
    t0 = time.perf_counter()
    train = _fold(words, held, q)
    sp, _ = C.measure_spreads(pack, train, _fold(code_words, held, q), held_blocks=held, quarter=q)
    if q is None:
        return {"fold": "all", "spreads": sp}
    test = _scored(words, held, q).copy()
    train = train.assign(prior=_priors(pack, train, sp))
    test["prior"] = _priors(pack, test, sp)
    can = C.readable(test)              # a truth the box can't emit at all is no observation of it (counted)
    impossible = {"pairs": int((~can).sum()), "weight": round(float(test.loc[~can, "weight"].sum()), 3)}
    test = test[can]
    got = C.fit_prior(train, held_blocks=held, quarter=q, joint=True)
    fits = {"sequential": got, "joint": got["joint"]}
    out = {"fold": paths.QUARTERS[q], "spreads": sp, "fit": got, "held_out_impossible": impossible,
           "train_weight_by_kind": train.groupby("kind")["weight"].sum().round(2).to_dict(),
           "held_out": {m: held_out(test, f["tau"], _weights(f["w"])) for m, f in fits.items()},
           "brief_example": {m: brief_example(test, _weights(f["w"])) for m, f in fits.items()},
           "seconds": round(time.perf_counter() - t0, 1)}
    log.info("%s: sequential tau %.3f w %s; joint tau %.3f w %s (%.0f s)", paths.QUARTERS[q], got["tau"],
             {k: round(v, 2) for k, v in got["w"].items()}, got["joint"]["tau"],
             {k: round(v, 2) for k, v in got["joint"]["w"].items()}, out["seconds"])
    return out


def fit() -> dict:
    t0 = time.perf_counter()
    held = _held_blocks()
    pack = C.Pack.load(PACK_FILE)
    rows = _crop_rows()
    code_words = rows[(rows["kind"] == C.CODE) & rows["text"].notna()].reset_index(drop=True)
    _F.update(pack=pack, words=load_pairs(), held=held, code_words=code_words)
    jobs = [None] + list(range(len(paths.QUARTERS)))
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    def job_file(q):
        return JOBS_DIR / f"{'all' if q is None else paths.QUARTERS[q]}.pkl"

    todo = [q for q in jobs if not job_file(q).is_file()]      # a fold already done is kept (resumable)
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("fork"), initializer=_init_fit) as pool:
        running = {pool.submit(_fit_job, q): q for q in todo}
        for fut in as_completed(running):                      # a dead worker raises here, never hangs
            job_file(running[fut]).write_bytes(pickle.dumps(fut.result()))
    got = {g["fold"]: g for g in (pickle.loads(job_file(q).read_bytes()) for q in jobs)}
    pack.spreads = {f: got[f]["spreads"] for f in C.FOLDS}
    pack.save(PACK_FILE)
    SUMMARY_FILE.write_text(json.dumps(pack.counts(), indent=1, default=float) + "\n")
    folds = list(paths.QUARTERS)
    pooled = {m: [r for f in folds for r in got[f]["held_out"][m]["rows"]] for m in MODES}
    pooled_nll = {m: score(pooled[m])["nll"] for m in MODES}
    mode = min(MODES, key=lambda m: pooled_nll[m])
    temps = {}
    for f in folds:                                            # q's temperature from the other three quarters
        others = [r for g in folds if g != f for r in got[g]["held_out"][mode]["rows"]]
        temps[f] = C.fit_temperature([(r["lp"], r["k"]) for r in others], [r["weight"] for r in others])
    tempered_rows = [r for f in folds for r in got[f]["held_out"][mode]["rows"]]
    tempered_scores = {f: score(got[f]["held_out"][mode]["rows"], temps[f]["temperature"]) for f in folds}
    out = {"pooled_nll": pooled_nll, "mode": mode, "folds": {}}
    for f in folds:
        g = got[f]
        out["folds"][f] = {"spreads": g["spreads"].summary(), "fit": g["fit"],
                           "held_out_impossible": g["held_out_impossible"],
                           "train_weight_by_kind": g["train_weight_by_kind"],
                           "held_out": {m: score(g["held_out"][m]["rows"]) for m in MODES},
                           "temperature": temps[f], "held_out_tempered": tempered_scores[f],
                           "brief_example": g["brief_example"], "seconds": g["seconds"]}
    out["pooled"] = {m: score(pooled[m]) for m in MODES}
    out["pooled_tempered"] = {"nll": _wmean([-C.target_logp(C.tempered(r["lp"], temps[f]["temperature"]), r["k"])
                                             for f in folds for r in got[f]["held_out"][mode]["rows"]],
                                            [r["weight"] for r in tempered_rows]),
                              "reliability": _pooled_reliability(got, folds, mode, temps)}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    FIT_FILE.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return {"mode": mode, "pooled_nll": pooled_nll, "temperatures": {f: t["temperature"] for f, t in temps.items()},
            "folds": {f: {m: {"tau": got[f]["fit"]["tau"] if m == "sequential" else got[f]["fit"]["joint"]["tau"],
                              "w": got[f]["fit"]["w"] if m == "sequential" else got[f]["fit"]["joint"]["w"]}
                          for m in MODES} for f in folds}, "seconds": out["seconds"]}


def _pooled_reliability(got, folds, mode, temps) -> list[dict]:
    """The reliability over every fold's held-out pairs, each fold at its own temperature."""
    p, right, v = [], [], []
    for f in folds:
        t = temps[f]["temperature"]
        for r in got[f]["held_out"][mode]["rows"]:
            top, pr = _decode(r["labels"], C.tempered(r["post"], t), r["best"])
            p.append(pr)
            right.append(_agree(top, r))
            v.append(r["weight"])
    p, right, v = np.array(p), np.array(right), np.array(v)
    idx = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    return [{"bin": [float(BINS[i]), float(BINS[i + 1])], "weight": round(float(v[idx == i].sum()), 2),
             "mean_p": _wmean(p[idx == i], v[idx == i]), "agree": _wmean(right[idx == i], v[idx == i])}
            for i in range(len(BINS) - 1) if (idx == i).any()]


# --- blocks: the block numbers the pack overrules ---------------------------------------------------------------------

def _digits(s: str) -> tuple[str, str]:
    m = re.match(r"^(\d+)([A-Z]*)$", str(s))
    return (m.group(1).lstrip("0") or "0", m.group(2)) if m else (str(s), "")


def blocks() -> dict:
    """The held-out block-number pairs holding their whole word, whose Terra string is off the list with stronger ink
    and that the kept mode overrules (see the module): how near the box sits to its block's edge, how Terra's number
    relates to the block the pack chose and to today's blocks, and EXAMPLES crops (the heaviest pairs)."""
    import cv2
    import shapely

    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.textnet import data
    from sfzmaps.solver.textread import crops as K
    fit_out = json.loads(FIT_FILE.read_text())
    mode = fit_out["mode"]
    pack = C.Pack.load(PACK_FILE)
    rows = []
    for f in paths.QUARTERS:
        g = pickle.loads((JOBS_DIR / f"{f}.pkl").read_bytes())
        rows += [r for r in g["held_out"][mode]["rows"] if r["kind"] == C.BLOCK and r["off"] and r["margin"] > 0
                 and not r["keeps"] and r["parts"] == [(r["truth"], 1.0)]]      # the whole word in the box
    numbers = {}
    for r_, forms in enumerate(pack.forms_of_block):
        for fm in forms:
            numbers.setdefault(fm, []).append(r_)
    stems = {}
    for fm in numbers:
        stems.setdefault(_digits(fm)[0], set()).add(fm)
    out = []
    for r in rows:
        pt = shapely.Point(r["x"], r["y"])
        top = sorted(r["top_block"])[0] if r["top_block"] else None
        holding = [i for i in pack.block_tree.query(pt, predicate="intersects")]
        edge = float(min(pack.blocks.geometry.values[i].boundary.distance(pt) for i in holding)) if holding else None
        t_stem, t_suffix = _digits(r["truth"])
        if top is not None and t_stem == _digits(top)[0]:
            relation = "same number, other suffix"
        elif r["truth"] in numbers:
            relation = "another block today"
        else:
            relation = "no block today"
        other = min((pack.blocks.geometry.values[i].distance(pt) for i in numbers.get(r["truth"], [])), default=None)
        split = top is not None and len(stems.get(_digits(top)[0], ())) > 1
        one_char = top is not None and len(top) == len(r["truth"]) and \
            sum(a != b for a, b in zip(top, r["truth"])) == 1           # a misread of one character, e.g. B as 8
        out.append({**{k: r[k] for k in ("sheet", "box", "truth", "weight", "x", "y")}, "pack_block": top,
                    "edge_ft": edge, "relation": relation, "truth_block_ft": other, "pack_block_has_suffix_kin": split,
                    "one_character_apart": one_char})
    d = pd.DataFrame(out)
    BLOCKS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"mode": mode, "pairs": int(len(d)), "weight": round(float(d["weight"].sum()), 2) if len(d) else 0.0}
    if len(d):
        e = d["edge_ft"].dropna().to_numpy()
        summary.update({
            "edge_ft_quantiles": {f"q{p}": round(float(np.quantile(e, p / 100)), 1) for p in (10, 25, 50, 75, 90)},
            "edge_ft_display_counts": {f"<= {c} ft": int((e <= c).sum()) for c in (10, 25, 50, 100)},
            "relation": d.groupby("relation")["weight"].agg(["size", "sum"]).round(2).to_dict("index"),
            "pack_block_has_suffix_kin": int(d["pack_block_has_suffix_kin"].sum()),
            "one_character_apart": {"pairs": int(d["one_character_apart"].sum()),
                                    "weight": round(float(d.loc[d["one_character_apart"], "weight"].sum()), 2)},
            "truth_block_ft_quantiles": {f"q{p}": round(float(np.quantile(d["truth_block_ft"].dropna(), p / 100)), 1)
                                         for p in (10, 50, 90)} if d["truth_block_ft"].notna().any() else None})
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    examples = d.sort_values(["weight", "sheet", "box"], ascending=[False, True, True]).head(EXAMPLES)
    index = []
    for k, r in enumerate(examples.itertuples(index=False)):
        b = boxes.loc[r.box]
        page = data.load_page(r.sheet)
        corners = b[T_COLS].to_numpy(float).reshape(4, 2)
        crop = K.box_crop(page, corners, b["t_length"], b["t_height"], b["axis"])
        x0, y0 = np.floor(corners.min(0) - 2 * max(b["t_length"], 40)).astype(int)
        x1, y1 = np.ceil(corners.max(0) + 2 * max(b["t_length"], 40)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, page.shape[1]), min(y1, page.shape[0])
        around = cv2.cvtColor(page[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        cv2.polylines(around, [np.round(corners - [x0, y0]).astype(np.int32)], True, (0, 0, 255), 1)
        stem = f"{k:02d}-{r.sheet[-3:]}-box{r.box}"
        cv2.imwrite(str(BLOCKS_DIR / f"{stem}-crop.png"), crop)
        cv2.imwrite(str(BLOCKS_DIR / f"{stem}-around.png"), around)
        index.append({"file": stem, "sheet": r.sheet, "box": int(r.box), "terra": r.truth, "pack_block": r.pack_block,
                      "relation": r.relation, "edge_ft": r.edge_ft, "weight": round(float(r.weight), 3)})
    summary["examples"] = index
    (BLOCKS_DIR / "index.json").write_text(json.dumps(index, indent=1, default=float) + "\n")
    (BLOCKS_DIR / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    what = sys.argv[1] if len(sys.argv) > 1 else "build"
    result = {"build": build, "read": read, "fit": fit, "blocks": blocks}[what](*sys.argv[2:])
    print(json.dumps(result, indent=1, default=float)[:8000])
