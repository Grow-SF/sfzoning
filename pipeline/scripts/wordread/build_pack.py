"""San Francisco's context pack for the 2000 map: build it, read our boxes, measure each fold's spreads and fit the
CTC temperature, the kinds' weights and the read-time temperature (Task 10 of the map-text plan;
`sfzmaps.solver.wordread.context`).  CPU only, at most 2 processes x 1 thread, $0.

    uv run python scripts/wordread/build_pack.py build    # the pack's streets, blocks and code list
    uv run python scripts/wordread/build_pack.py read     # every (our box, Terra word) pair with its weight; boxes read
    uv run python scripts/wordread/build_pack.py rows     # each paired box's target: what it holds of its words
    uv run python scripts/wordread/build_pack.py fit      # per held-out quarter: spreads, fits, held-out scores
    uv run python scripts/wordread/build_pack.py blocks   # the block numbers the pack overrules, and example crops

No IoU anywhere.  The spreads are measured on pairs: every pair of one of our boxes and a Terra word on its sheet counts
with the chance the box holds the word, the product of the word's share of its ink inside our box's (trimmed) rectangle
and our box's share of its ink inside the word's rectangle (`wordread.boxmodel.ink_shares` both ways, on the page's ink,
`grid.INK_BELOW`); a pair whose product is 0 adds nothing.  Everything read is fitted and scored on boxes
(`context.box_targets`): a box's target is the text it holds of the Terra words it overlaps, read as one line (the
line's place across the box drawn from its ink inside the words: words of one line together in reading order, stacked
labels apart), each word held by its share across the reading and each character inside by its share along it; a box's
weight is the chance it holds any of that text times its ink's share inside the words' rectangles.

Everything lands in data/solver/experiments/wordread/packs/ (never committed):
  sf-2000.parquet        the pack; after `fit`, with the spreads of every fold and of "all", and each quarter's fit
                         (`context.ReadFit`: mode, tau, w, temperature) for `context.read_posterior`
  sf-2000.json           its counts
  sf-2000-pairs.parquet  the pairs: the box, the word's string (in the reader's alphabet), kind, block, quarter and
                         touches, the weight, the box's place on the map and kind_p
  sf-2000-reads.npz      each paired box's two reads' log-probabilities (frames x classes, as cut and turned), as
                         `solver-text-read` reads it (its quarter's reader; the classifier's q), and its best path
  sf-2000-rows.parquet   a row per paired box: its words in reading order, its targets, weight and what it touches
  sf-2000-fit.json       per held-out quarter q, fitted on the rows of the other three quarters (touching neither q nor
                         an answered block; the spreads on their pairs, the codes on Terra's code words there):
                           - the spreads (`context.measure_spreads`, weighted EM, leave-one-box-out bandwidths);
                           - tau and w per kind, sequentially (tau on the ink alone, then w) and jointly
                             (`context.fit_prior`);
                           - per mode, a temperature on the read-time posterior (the strings the prior lists, the
                             best path, OTHER) fitted on the other three quarters' held-out vectors
                             (`context.fit_temperature`, each target at its slot, OTHER when not named); the mode with
                             the lower NLL there after its own temperature is q's;
                           - on q's own rows (none in or touching an answered block), weighted: the NLL of Terra's
                             string and of its slot, top-1 against it (the likeliest named string; the ink at tau over
                             the same strings; the best path), the reliability (the read-time vector's likeliest slot,
                             in display bins, against the chance the target is in it), the off-list strings whose ink
                             beats every listed string (kept?, whole words apart from parts), and the brief's example
                             (listed ink 1e-6, an off-list string's 0.95) on q's code, block and street boxes.
  block_disagreements/   the held-out block-number boxes holding their whole word, Terra's string off the list and
                         its ink stronger, that the pack overrules: their edge distances, how their numbers relate to
                         today's blocks (summary.json), and 10 example crops (ours and the page around it, with
                         index.json)

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
ROWS_FILE = C.PACKS_DIR / f"sf-{YEAR}-rows.parquet"
FIT_FILE = C.PACKS_DIR / f"sf-{YEAR}-fit.json"
JOBS_DIR = C.PACKS_DIR / f"sf-{YEAR}-fit-jobs"      # each fold's result as it lands (a rerun skips those done)
BLOCKS_DIR = C.PACKS_DIR / "block_disagreements"
KIND_P = ["p_zoning", "p_block", "p_street", "p_other"]        # wordboxes' columns, in C.KIND_ORDER
TOUCH = (*[f"touch_{q}" for q in paths.QUARTERS], "touch_blocks")   # text-crops' held-out areas a word touches
META = ("block", "blocks", "quarter", *TOUCH)                      # what `context.refuse_answered` checks
T_COLS = [f"t{i}{a}" for i in range(4) for a in "xy"]              # a box's trimmed rectangle
BINS = np.linspace(0.0, 1.0, 11)  # the reliability table's display bins
OFF_LIST = "ZZZZ"                 # the brief's example's off-list string (no code, street or block prints it)
MODES = C.MODES
EXAMPLES = 10


def _held_blocks() -> set[str]:
    from sfzmaps.solver.textnet import reader_stage
    return set(reader_stage.held_blocks())


def _crop_rows() -> pd.DataFrame:
    """The 2000 drawing's Terra words in text-crops (its rows with a word)."""
    from sfzmaps.solver.textread import store
    rows = pd.read_parquet(store.CROPS_DIR / "crops.parquet")
    return rows[(rows["series"] == paths.SERIES) & (rows["word"] >= 0)].reset_index(drop=True)


def _answered(words: pd.DataFrame, held: set[str]) -> np.ndarray:
    """Which rows are in or touch an answered block (any of a box row's words)."""
    bad = words["touch_blocks"].astype(bool).to_numpy() | words["block"].astype(str).isin(held).to_numpy()
    if "blocks" in words:
        bad |= np.array([any(str(b) in held for b in bs) for bs in words["blocks"]], bool)
    return bad


def _fold(words: pd.DataFrame, held: set[str], q: int | None) -> pd.DataFrame:
    """The rows a fold may learn from: in or touching no answered block, nor in or touching q."""
    keep = ~_answered(words, held)
    if q is not None:
        keep &= (words["quarter"].astype(int) != q).to_numpy() & ~words[f"touch_{paths.QUARTERS[q]}"].astype(bool)
    return words[keep]


def _scored(words: pd.DataFrame, held: set[str], q: int) -> pd.DataFrame:
    """q's own rows, in or touching no answered block."""
    return words[(words["quarter"].astype(int) == q).to_numpy() & ~_answered(words, held)]


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


def _terra() -> pd.DataFrame:
    """Terra's words on the 2000 sheets with their text-crops rows (`truth`: the string in the reader's alphabet,
    block, quarter, touches)."""
    from sfzmaps.solver.wordread import measures
    terra = measures.terra_words(paths.SHEETS)
    terra["word"] = terra.groupby("sheet").cumcount()
    rows = _crop_rows()
    terra = terra.merge(rows[["sheet", "word", "terra", "text", "block", "quarter", *TOUCH]]
                        .rename(columns={"text": "truth", "quarter": "word_quarter"}), on=["sheet", "word"],
                        how="left", validate="one_to_one")
    if not (terra["terra"] == terra["text"]).all():
        raise ValueError("Terra's words and text-crops' rows disagree: rebuilt at different times?")
    return terra


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
    return {"sheet": sheet, "pairs": [(int(boxes.index[b]), int(terra.index[w]), s) for b, w, s in pairs],
            "reads": reads}


def read() -> dict:
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.textnet import data
    from sfzmaps.solver.wordread import measures
    t0 = time.perf_counter()
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    terra = _terra()
    jobs = [(s, boxes[boxes["sheet"] == s], terra[terra["sheet"] == s]) for s in paths.SHEETS]
    got = []
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("fork"), initializer=_init_read) as pool:
        for fut in as_completed([pool.submit(_sheet_job, j) for j in jobs]):
            got.append(fut.result())
    reads = {k: v for g in got for k, v in g["reads"].items()}
    pairs = pd.DataFrame([p for g in got for p in g["pairs"]], columns=["box", "word", "weight"])
    pairs = pairs.sort_values(["box", "word"]).reset_index(drop=True)
    t = terra.loc[pairs["word"]].reset_index(drop=True)
    bx = boxes.loc[pairs["box"]].reset_index(drop=True)
    out = pd.DataFrame({"box": pairs["box"], "word": pairs["word"], "weight": pairs["weight"], "sheet": bx["sheet"],
                        "box_quarter": bx["quarter"].astype(int), "text": t["truth"], "kind": t["kind"],
                        "block": t["block"].fillna("").astype(str), "quarter": t["word_quarter"].astype(int),
                        **{c: t[c].astype(bool) for c in TOUCH}})
    out = out[out["text"].notna() & (out["text"] != "")].reset_index(drop=True)
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
    return {"pairs": int(len(out)), "boxes": int(out["box"].nunique()), "words": int(out["word"].nunique()),
            "weight": round(float(out["weight"].sum()), 1), "seconds": round(time.perf_counter() - t0, 1)}


def _reads() -> dict:
    """Each read box's two reads (log-probabilities, frames x classes)."""
    with np.load(READS_FILE) as z:
        keys, lens, one, two = z["boxes"], z["lens"], z["one"], z["two"]
    o1, o2 = np.r_[0, np.cumsum(lens[:, 0])], np.r_[0, np.cumsum(lens[:, 1])]
    return {int(k): (one[o1[i]:o1[i + 1]], two[o2[i]:o2[i + 1]]) for i, k in enumerate(keys)}


def load_pairs() -> pd.DataFrame:
    """The pairs (the spreads' words)."""
    return pd.read_parquet(PAIRS_FILE, columns=["box", "word", "weight", "sheet", "box_quarter", "text", "kind",
                                                "block", "quarter", *TOUCH, "x", "y", "axis"])


# --- rows: each box's target ------------------------------------------------------------------------------------------

def _rows_job(job) -> list[dict]:
    """One sheet's paired boxes: each box's target (`context.box_targets`, its words' texts, rectangles and ink on the
    page), the chance it holds a non-empty run of them, its ink's share inside the union of their rectangles, and its
    weight, their product."""
    import shapely

    from sfzmaps.solver import grid
    from sfzmaps.solver.textnet import data
    from sfzmaps.solver.wordread import boxmodel, measures
    sheet, rects, terra, pairs, frames = job
    ink = data.load_page(sheet) < grid.INK_BELOW
    word_ink: dict[int, tuple] = {}
    out = []
    for box, part in pairs.groupby("box", sort=True):
        rect = rects[box]
        ws = part["word"].tolist()
        wr = [terra.loc[w, measures.C_COLS].to_numpy(float).reshape(4, 2) for w in ws]
        for w, r in zip(ws, wr):
            if w not in word_ink:
                word_ink[w] = boxmodel._ink_of(r, ink)
        bx, by = boxmodel._ink_of(rect, ink)
        if len(bx):
            in_words = np.zeros(len(bx), bool)
            for r in wr:
                in_words |= C._in_quad(r, bx, by)
            s_box = float(in_words.mean())
        else:                                     # no ink: the share of its area
            union = shapely.union_all([shapely.Polygon(r) for r in wr])
            s_box = C._area_share(shapely.Polygon(rect), union)
        texts = [str(terra.loc[w, "truth"]) for w in ws]
        got = C.box_targets([(t, r, *word_ink[w]) for t, r, w in zip(texts, wr, ws)], rect, frames[box], (bx, by))
        heavy = int(np.argmax(part["weight"].to_numpy()))
        out.append({"box": int(box), "words": json.dumps([int(ws[k]) for k in got["order"]]),
                    "texts": json.dumps([texts[k] for k in got["order"]]), "n_words": len(ws),
                    "targets": json.dumps(got["targets"]), "dropped": got["dropped"],
                    "p_nonempty": got["p_nonempty"], "s_box": s_box, "weight": got["p_nonempty"] * s_box,
                    "pair_weight": float(part["weight"].sum()), "across": json.dumps(got["across"]),
                    "lines": json.dumps(got["lines"]),
                    "heaviest_word": int(ws[heavy])})
    return out


def rows() -> dict:
    """Each paired box's row (see the module) to ROWS_FILE: its words in reading order and their targets, weight,
    kind/block/quarter (its heaviest pair's), every block its words lie in (`blocks`) and every held-out area any of
    them touches (a word in another quarter than the row's touches that quarter)."""
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.wordread import measures
    t0 = time.perf_counter()
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    terra = _terra()
    pairs = load_pairs()
    if not (boxes.loc[pairs["box"], "sheet"].to_numpy() == pairs["sheet"].to_numpy()).all() or \
            not (terra.loc[pairs["word"], "truth"].to_numpy() == pairs["text"].to_numpy()).all():
        raise ValueError("the pairs were read from other boxes or Terra rows than today's: rerun `read`")
    with np.load(READS_FILE) as z:
        frames = dict(zip(z["boxes"].tolist(), z["lens"][:, 0].tolist()))
    jobs = []
    for s in paths.SHEETS:
        part = pairs[pairs["sheet"] == s]
        if not len(part):
            continue
        rect = {int(b): boxes.loc[b, T_COLS].to_numpy(float).reshape(4, 2) for b in part["box"].unique()}
        words = terra.loc[sorted(set(part["word"])), ["truth", *measures.C_COLS]]
        jobs.append((s, rect, words, part[["box", "word", "weight"]], {b: frames[b] for b in rect}))
    got = []
    with ProcessPoolExecutor(WORKERS, mp_context=get_context("fork")) as pool:
        for fut in as_completed([pool.submit(_rows_job, j) for j in jobs]):
            got += fut.result()
    out = pd.DataFrame(got).sort_values("box").reset_index(drop=True)
    first = pairs.drop_duplicates("box").set_index("box")
    heavy = pairs.drop_duplicates("word").set_index("word").loc[out["heaviest_word"]]   # a word's own fields
    out["kind"] = heavy["kind"].to_numpy()
    out["block"] = heavy["block"].astype(str).to_numpy()
    out["quarter"] = heavy["quarter"].astype(int).to_numpy()
    by_box = pairs.groupby("box")
    out["blocks"] = [json.dumps(sorted(set(by_box.get_group(b)["block"].astype(str)))) for b in out["box"]]
    for c in TOUCH:
        out[c] = [bool(by_box.get_group(b)[c].any()) for b in out["box"]]
    for k, qn in enumerate(paths.QUARTERS):             # a word in another quarter: the row touches it
        other = [bool((by_box.get_group(b)["quarter"].astype(int) == k).any()) and q != k
                 for b, q in zip(out["box"], out["quarter"])]
        out[f"touch_{qn}"] = out[f"touch_{qn}"].to_numpy() | np.array(other, bool)
    for c in ("sheet", "box_quarter", "x", "y", "axis"):
        out[c] = first.loc[out["box"], c].to_numpy()
    full = pd.read_parquet(PAIRS_FILE, columns=["box", "q", "best", *[f"kind_p_{c}" for c in C.KIND_ORDER]])
    full = full.drop_duplicates("box").set_index("box")
    for c in full.columns:
        out[c] = full.loc[out["box"], c].to_numpy()
    out.to_parquet(ROWS_FILE, index=False)
    n_targets = out["targets"].map(lambda t: len(json.loads(t)))
    return {"boxes": int(len(out)), "weight": round(float(out["weight"].sum()), 1),
            "pair_weight": round(float(out["pair_weight"].sum()), 1),
            "multi_word_boxes": int((out["n_words"] > 1).sum()),
            "multi_word_weight": round(float(out.loc[out["n_words"] > 1, "weight"].sum()), 1),
            "no_readable_target": int((n_targets == 0).sum()),
            "whole": int(sum(json.loads(t) == [[s, 1.0]] for t, s in
                             zip(out["targets"], out["texts"].map(lambda x: " ".join(json.loads(x)))))),
            "targets_max": int(n_targets.max()), "seconds": round(time.perf_counter() - t0, 1)}


def load_rows() -> pd.DataFrame:
    """The box rows, each with its two reads (`reads`), kind_p, targets, the likeliest of them (`truth`) and its
    words' blocks (`blocks`)."""
    out = pd.read_parquet(ROWS_FILE)
    reads = _reads()
    out["reads"] = [reads[int(b)] for b in out["box"]]
    out["kind_p"] = list(out[[f"kind_p_{c}" for c in C.KIND_ORDER]].to_numpy(float))
    out["targets"] = [[(s, float(p)) for s, p in json.loads(tg)] for tg in out["targets"]]
    out["blocks"] = [json.loads(b) for b in out["blocks"]]
    out["truth"] = [tg[0][0] if tg else "" for tg in out["targets"]]
    return out


# --- fit --------------------------------------------------------------------------------------------------------------

_F: dict = {}


def _init_fit() -> None:
    import torch
    torch.set_num_threads(1)


def _weights(w: dict) -> dict:
    """The fitted weights for every listing kind (a kind no training row listed: 0, it lifts nothing)."""
    return {k: float(w.get(k, 0.0)) for k in (C.CODE, C.BLOCK, C.STREET)}


def _priors(pack: C.Pack, words: pd.DataFrame, sp: C.Spreads) -> list:
    """Each row's prior, computed once per box."""
    by = {}
    for b, x, y, a, kp in zip(words["box"], words["x"], words["y"], words["axis"], words["kind_p"]):
        if b not in by:
            by[b] = C.prior(pack, {"x": x, "y": y, "axis": a}, kp, sp)
    return [by[b] for b in words["box"]]


def _top_named(labels: list, logpost: np.ndarray) -> str:
    """The likeliest named string (ties: the first by name); OTHER is no string."""
    return min(((-v, s) for s, v in zip(labels, logpost) if s != C.OTHER))[1]


def _agree(top: str, r: dict) -> float:
    """How far a decode agrees with Terra at this box: the chance of the text it holds that it equals."""
    return float(sum(p for s, p in r["parts"] if s == top))


def held_out(test: pd.DataFrame, tau: float, w: dict) -> dict:
    """q's own rows under (tau, w): per row the string-space log posterior with its targets named (`lp`, `k`: the
    fit's likelihood), the read-time vector (`labels`, `post`: the prior's strings and the best path, then OTHER;
    `slots`: the targets on it), the decodes, the off-list facts of the likeliest target (`truth`) and what
    `refuse_answered` checks."""
    rows = C.truth_rows(test, tau, w)
    out = []
    for (lp, k), r in zip(rows, test.itertuples(index=False)):
        names = r.prior.strings if r.prior is not None else []
        read_names = sorted(set(names) | {r.best})
        ink = C.ink_logp(r.reads, r.q, sorted(set(read_names) | {s for s, _ in r.targets}), tau)
        post = C.posterior({s: ink[s] for s in read_names}, r.prior, w)
        labels = list(post)
        alone = C.posterior({s: ink[s] for s in read_names}, None, {})
        off = r.truth not in names
        listed = [ink[s] for s in names]
        margin = (ink[r.truth] - max(listed) if listed else math.inf) if off else -math.inf
        named = C.posterior({s: ink[s] for s in sorted(set(names) | {r.truth})}, r.prior, w)
        out.append({"lp": lp, "k": k, "parts": list(r.targets), "dropped": float(r.dropped), "labels": labels,
                    "post": np.array(list(post.values())),
                    "slots": C.slot_targets(labels, r.targets, float(r.dropped)),
                    "decode": _top_named(labels, np.array(list(post.values()))),
                    "ink_top": _top_named(list(alone), np.array(list(alone.values()))),
                    "best": r.best, "truth": r.truth, "kind": r.kind, "weight": float(r.weight), "off": off,
                    "margin": margin, "n_words": int(r.n_words),
                    "whole": len(r.targets) == 1 and r.targets[0][1] == 1.0 and r.dropped == 0,
                    "keeps": (_top_named(list(named), np.array(list(named.values()))) == r.truth) if off else None,
                    "box": int(r.box), "sheet": r.sheet, "x": r.x, "y": r.y,
                    "top_block": max(r.prior.items.get(C.BLOCK, []), key=lambda it: it[0])[1] if r.prior is not None
                    and r.prior.items.get(C.BLOCK) else None,
                    **{c: getattr(r, c) for c in META}})
    return {"rows": out}


def _reliability(p, right, v) -> list[dict]:
    p, right, v = np.asarray(p, float), np.asarray(right, float), np.asarray(v, float)
    idx = np.clip(np.digitize(p, BINS) - 1, 0, len(BINS) - 2)
    return [{"bin": [round(float(BINS[i]), 1), round(float(BINS[i + 1]), 1)],
             "weight": round(float(v[idx == i].sum()), 2), "mean_p": _wmean(p[idx == i], v[idx == i]),
             "agree": _wmean(right[idx == i], v[idx == i])}
            for i in range(len(BINS) - 1) if (idx == i).any()]


def _slot_top(r: dict, temperature: float) -> tuple[float, float]:
    """The read-time vector's likeliest slot at a temperature: its probability and the chance the target is in it."""
    t = C.tempered(r["post"], temperature)
    top = int(np.argmax(t))
    return float(math.exp(t[top])), float(sum(math.exp(lp) for k, lp in r["slots"] if k == top))


def _keeps(strong: list[dict]) -> dict:
    return {"pairs": len(strong), "weight": round(sum(r["weight"] for r in strong), 2),
            "kept_pairs": sum(bool(r["keeps"]) for r in strong),
            "kept_weight": round(sum(r["weight"] for r in strong if r["keeps"]), 2),
            "by_kind": {k: [sum(r["kind"] == k for r in strong),
                            sum(bool(r["keeps"]) and r["kind"] == k for r in strong)]
                        for k in C.KIND_ORDER if any(r["kind"] == k for r in strong)}}


def score(rows: list[dict], temperature: float = 1.0) -> dict:
    """The held-out scores (see the module) of `held_out` rows, the read-time vector at a temperature."""
    if not rows:
        return {}
    v = np.array([r["weight"] for r in rows])
    nll = np.array([-C.target_logp(r["lp"], r["k"]) for r in rows])
    nll_slot = np.array([-C.target_logp(C.tempered(r["post"], temperature), r["slots"]) for r in rows])
    right = np.array([_agree(r["decode"], r) for r in rows], float)
    out = {"boxes": len(rows), "weight": round(float(v.sum()), 2), "nll": _wmean(nll, v),
           "nll_slot": _wmean(nll_slot, v), "by_kind": {}}
    for kind in ["all", *C.KIND_ORDER]:
        sel = np.ones(len(rows), bool) if kind == "all" else np.array([r["kind"] == kind for r in rows])
        if not sel.any():
            continue
        out["by_kind"][kind] = {"weight": round(float(v[sel].sum()), 2), "nll": _wmean(nll[sel], v[sel]),
                                "nll_slot": _wmean(nll_slot[sel], v[sel]), "top1_pack": _wmean(right[sel], v[sel]),
                                "top1_ink": _wmean([_agree(r["ink_top"], r) for r, s in zip(rows, sel) if s], v[sel]),
                                "top1_best": _wmean([_agree(r["best"], r) for r, s in zip(rows, sel) if s], v[sel])}
    top = [_slot_top(r, temperature) for r in rows]
    out["reliability"] = _reliability([t[0] for t in top], [t[1] for t in top], v)
    strong = [r for r in rows if r["off"] and r["margin"] > 0]
    out["off_list_stronger_ink"] = {"all": _keeps(strong), "whole": _keeps([r for r in strong if r["whole"]]),
                                    "whole_one_word": _keeps([r for r in strong if r["whole"] and r["n_words"] == 1]),
                                    "part": _keeps([r for r in strong if not r["whole"]])}
    return out


def brief_example(test: pd.DataFrame, w: dict) -> dict:
    """The brief's example on q's boxes' real priors (their kind_p), per kind of the box's heaviest word: every listed
    string's ink 1e-6, an off-list string's 0.95; does the off-list string win, and by how much (log posterior, before
    the read-time temperature, which divides every margin and changes no winner)."""
    out = {}
    for kind in (C.CODE, C.BLOCK, C.STREET):
        wins, margins = [], []
        for pr in test.loc[test["kind"] == kind, "prior"]:
            if pr is None or not pr.strings:
                continue
            post = C.posterior({**{s: math.log(1e-6) for s in pr.strings}, OFF_LIST: math.log(0.95)}, pr, w)
            best_listed = max(v for s, v in post.items() if s not in (OFF_LIST, C.OTHER))
            wins.append(post[OFF_LIST] > best_listed)
            margins.append(post[OFF_LIST] - best_listed)
        m = np.array(margins)
        out[kind] = {"boxes": len(wins), "off_list_wins": int(sum(wins)),
                     "min_log_margin": round(float(m.min()), 3) if len(m) else None,
                     "losing_margins_q10_q50": [round(float(np.quantile(m[m <= 0], p)), 2) for p in (0.1, 0.5)]
                     if (m <= 0).any() else None}
    return out


def _fit_job(q) -> dict:
    pack, pairs, words, code_words, held = _F["pack"], _F["pairs"], _F["rows"], _F["code_words"], _F["held"]
    t0 = time.perf_counter()
    sp, _ = C.measure_spreads(pack, _fold(pairs, held, q), _fold(code_words, held, q), held_blocks=held, quarter=q)
    if q is None:
        return {"fold": "all", "spreads": sp}
    train = _fold(words, held, q)
    test = _scored(words, held, q).copy()
    train = train.assign(prior=_priors(pack, train, sp))
    test["prior"] = _priors(pack, test, sp)
    can = C.readable(test)              # a box with no target its reads can emit is no observation of it (counted)
    impossible = {"boxes": int((~can).sum()), "weight": round(float(test.loc[~can, "weight"].sum()), 3)}
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


def _mode_fits(got: dict, held: set[str]) -> dict:
    """Per quarter q and mode: the read-time temperature fitted on the other three quarters' held-out vectors (none
    touching q), and the NLL there after it; q's mode is the one with the lower NLL."""
    out = {}
    for qi, f in enumerate(paths.QUARTERS):
        per = {}
        for m in MODES:
            others = [r for g in paths.QUARTERS if g != f for r in got[g]["held_out"][m]["rows"] if not r[f"touch_{f}"]]
            meta = pd.DataFrame([{c: r[c] for c in META} for r in others])
            v = np.array([r["weight"] for r in others])
            t = C.fit_temperature([(r["post"], r["slots"]) for r in others], meta, v, held_blocks=held, quarter=qi)
            per[m] = {**t, "weight": round(float(v.sum()), 2), "nll_per_weight": t["nll"] / float(v.sum()),
                      "nll_per_weight_at_1": t["nll_at_1"] / float(v.sum())}
        mode = min(MODES, key=lambda m: per[m]["nll_per_weight"])
        out[f] = {"mode": mode, "by_mode": per}
    return out


def fit() -> dict:
    t0 = time.perf_counter()
    held = _held_blocks()
    pack = C.Pack.load(PACK_FILE)
    crops = _crop_rows()
    code_words = crops[(crops["kind"] == C.CODE) & crops["text"].notna()].reset_index(drop=True)
    _F.update(pack=pack, pairs=load_pairs(), rows=load_rows(), held=held, code_words=code_words)
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
    folds = list(paths.QUARTERS)
    modes = _mode_fits(got, held)
    chosen = {}
    for f in folds:
        m = modes[f]["mode"]
        fit_m = got[f]["fit"] if m == "sequential" else got[f]["fit"]["joint"]
        chosen[f] = C.ReadFit(mode=m, tau=float(fit_m["tau"]), w=_weights(fit_m["w"]),
                              temperature=float(modes[f]["by_mode"][m]["temperature"]))
    pack.spreads = {f: got[f]["spreads"] for f in C.FOLDS}
    pack.fits = chosen
    pack.save(PACK_FILE)
    SUMMARY_FILE.write_text(json.dumps(pack.counts(), indent=1, default=float) + "\n")
    out = {"fits": {f: {"mode": c.mode, "tau": c.tau, "w": c.w, "temperature": c.temperature}
                    for f, c in chosen.items()}, "mode_choice": modes, "folds": {}}
    for f in folds:
        g, c = got[f], chosen[f]
        out["folds"][f] = {"spreads": g["spreads"].summary(), "fit": g["fit"],
                           "held_out_impossible": g["held_out_impossible"],
                           "train_weight_by_kind": g["train_weight_by_kind"],
                           "held_out": {m: score(g["held_out"][m]["rows"]) for m in MODES},
                           "held_out_kept": score(g["held_out"][c.mode]["rows"], c.temperature),
                           "brief_example": g["brief_example"], "seconds": g["seconds"]}
    kept = [(r, chosen[f].temperature) for f in folds for r in got[f]["held_out"][chosen[f].mode]["rows"]]
    v = np.array([r["weight"] for r, _ in kept])
    top = [_slot_top(r, t) for r, t in kept]
    out["pooled_kept"] = {
        "weight": round(float(v.sum()), 2),
        "nll": _wmean([-C.target_logp(r["lp"], r["k"]) for r, _ in kept], v),
        "nll_slot_at_1": _wmean([-C.target_logp(r["post"], r["slots"]) for r, _ in kept], v),
        "nll_slot": _wmean([-C.target_logp(C.tempered(r["post"], t), r["slots"]) for r, t in kept], v),
        "top1_pack": _wmean([_agree(r["decode"], r) for r, _ in kept], v),
        "top1_ink": _wmean([_agree(r["ink_top"], r) for r, _ in kept], v),
        "top1_best": _wmean([_agree(r["best"], r) for r, _ in kept], v),
        "reliability": _reliability([t[0] for t in top], [t[1] for t in top], v),
        "reliability_at_1": _reliability(*zip(*[_slot_top(r, 1.0) for r, _ in kept]), v)}
    out["pooled"] = {m: score([r for f in folds for r in got[f]["held_out"][m]["rows"]]) for m in MODES}
    out["seconds"] = round(time.perf_counter() - t0, 1)
    FIT_FILE.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return {"fits": out["fits"], "nll_per_weight": {f: {m: round(modes[f]["by_mode"][m]["nll_per_weight"], 4)
                                                        for m in MODES} for f in folds},
            "pooled_kept": {k: v for k, v in out["pooled_kept"].items() if not k.startswith("reliability")},
            "seconds": out["seconds"]}


# --- blocks: the block numbers the pack overrules ---------------------------------------------------------------------

def _digits(s: str) -> tuple[str, str]:
    m = re.match(r"^(\d+)([A-Z]*)$", str(s))
    return (m.group(1).lstrip("0") or "0", m.group(2)) if m else (str(s), "")


def blocks() -> dict:
    """The held-out block-number boxes holding their whole word, whose Terra string is off the list with stronger ink
    and that each quarter's kept mode overrules (see the module): how near the box sits to its block's edge, how
    Terra's number relates to the block the pack chose and to today's blocks, and EXAMPLES crops (the heaviest)."""
    import cv2
    import shapely

    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.textnet import data
    from sfzmaps.solver.textread import crops as K
    fit_out = json.loads(FIT_FILE.read_text())
    pack = C.Pack.load(PACK_FILE)
    rows = []
    for f in paths.QUARTERS:
        g = pickle.loads((JOBS_DIR / f"{f}.pkl").read_bytes())
        rows += [r for r in g["held_out"][fit_out["fits"][f]["mode"]]["rows"] if r["kind"] == C.BLOCK and r["off"]
                 and r["margin"] > 0 and not r["keeps"] and r["whole"]]
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
    summary = {"modes": {f: fit_out["fits"][f]["mode"] for f in paths.QUARTERS}, "boxes": int(len(d)),
               "weight": round(float(d["weight"].sum()), 2) if len(d) else 0.0}
    if len(d):
        e = d["edge_ft"].dropna().to_numpy()
        summary.update({
            "edge_ft_quantiles": {f"q{p}": round(float(np.quantile(e, p / 100)), 1) for p in (10, 25, 50, 75, 90)},
            "edge_ft_display_counts": {f"<= {c} ft": int((e <= c).sum()) for c in (10, 25, 50, 100)},
            "relation": d.groupby("relation")["weight"].agg(["size", "sum"]).round(2).to_dict("index"),
            "pack_block_has_suffix_kin": int(d["pack_block_has_suffix_kin"].sum()),
            "one_character_apart": {"boxes": int(d["one_character_apart"].sum()),
                                    "weight": round(float(d.loc[d["one_character_apart"], "weight"].sum()), 2)},
            "truth_block_ft_quantiles": {f"q{p}": round(float(np.quantile(d["truth_block_ft"].dropna(), p / 100)), 1)
                                         for p in (10, 50, 90)} if d["truth_block_ft"].notna().any() else None})
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    examples = d.sort_values(["weight", "sheet", "box"], ascending=[False, True, True]).head(EXAMPLES) if len(d) else d
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
    result = {"build": build, "read": read, "rows": rows, "fit": fit, "blocks": blocks}[what](*sys.argv[2:])
    print(json.dumps(result, indent=1, default=float)[:8000])
