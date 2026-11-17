"""San Francisco's context pack for the 2000 map: build it, measure its spreads, and fit the prior's weight and escape
(Task 10 of the map-text plan; `sfzmaps.solver.wordread.context`).  CPU only, at most 2 processes x 1 thread, $0.

    uv run python scripts/wordread/build_pack.py build    # -> data/solver/experiments/wordread/packs/sf-2000.parquet
    uv run python scripts/wordread/build_pack.py fit      # -> .../packs/sf-2000-fit.json (needs the pack)
    uv run python scripts/wordread/build_pack.py refit    # the fits again from the kept read words (no reading)

build: `context.build_sf_pack(2000)` (DataSF's centrelines, the Assessor's blocks, the 2000 code list; no lot codes),
  then the spreads (`context.measure_spreads`), measured on the 2000 drawing's Terra words (`text-crops`' rows of
  kind street name or block number, in the reader's alphabet), never a word in or touching an answered block: "all"
  on every quarter, and per quarter q on the words lying outside q and touching no part of it (`touch_<q>`).  Writes
  the pack and sf-2000.json (its counts, the spreads, how many words name something in the pack and how many name
  what they sit by).

fit: our boxes (`solver-boxes`) matched to Terra's words at IoU >= 0.5 (Task 1's rule, `wordread.measures`: a box to
  its best word), each read by its quarter's reader both ways, mixed by the 0/180 classifier, as `solver-text-read`
  reads it; the ink of every listed string of the box's four priors (one per held-out quarter's spreads) and of
  Terra's string, exactly by CTC.  Then per held-out quarter q: w and e fitted on the words of the other three
  quarters (outside q, touching neither q nor an answered block), and scored on q's own words (none in or touching an
  answered block): the mean log posterior of Terra's string, and top-1 against Terra's string with the pack against
  the ink's own choice over the same strings and the reader's best path alone, by kind; and the log-likelihood's
  profiles in w and in e.  Writes sf-2000-fit.json, and the read words to sf-2000-fit-words.parquet.

Terra's strings are agreement, not truth: these numbers say how the pack moves the reads toward Terra's, never how
many of the owner's answered labels are read right (the headline, Task 11).
"""
import json
import logging
import math
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

from sfzmaps.solver import paths
from sfzmaps.solver.wordread import context as C

log = logging.getLogger("build_pack")

YEAR = 2000
WORKERS = 2
IOU = 0.5
PACK_FILE = C.PACKS_DIR / f"sf-{YEAR}.parquet"
SUMMARY_FILE = C.PACKS_DIR / f"sf-{YEAR}.json"
FIT_FILE = C.PACKS_DIR / f"sf-{YEAR}-fit.json"
WORDS_FILE = C.PACKS_DIR / f"sf-{YEAR}-fit-words.parquet"   # the read words, kept for `refit` and later tasks
KIND_P = ["p_zoning", "p_block", "p_street", "p_other"]        # wordboxes' columns, in C.KIND_ORDER
TOUCH = (*[f"touch_{q}" for q in paths.QUARTERS], "touch_blocks")   # text-crops' held-out areas a word touches


def _held_blocks() -> set[str]:
    from sfzmaps.solver.textnet import reader_stage
    return set(reader_stage.held_blocks())


def _crop_rows() -> pd.DataFrame:
    """The 2000 drawing's Terra words in text-crops (its rows with a word), indexed by (sheet, word)."""
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


def _fold_words(words: pd.DataFrame, held: set[str], q: int | None) -> pd.DataFrame:
    """The words a prior for held-out quarter q (None: every quarter) may learn from: none in or touching an answered
    block, and none in or touching q."""
    keep = ~words["touch_blocks"].astype(bool) & ~words["block"].astype(str).isin(held)
    if q is not None:
        keep &= (words["quarter"].astype(int) != q) & ~words[f"touch_{paths.QUARTERS[q]}"].astype(bool)
    return words[keep]


# --- build ------------------------------------------------------------------------------------------------------------

def build() -> dict:
    from sfzmaps.solver.textnet import boxes as textboxes
    t0 = time.perf_counter()
    pack = C.build_sf_pack(YEAR)
    log.info("pack: %d centrelines, %d blocks, %d code forms (%.0f s)", len(pack.streets), len(pack.blocks),
             len(pack.codes), time.perf_counter() - t0)
    held = _held_blocks()
    rows = _crop_rows()
    words = rows[rows["kind"].isin(["street name", "block number"]) & rows["text"].notna()].reset_index(drop=True)
    corners = np.array([textboxes.word_rect((a, b, c, d), ang) for a, b, c, d, ang in
                        zip(words["box_x0"], words["box_y0"], words["box_x1"], words["box_y1"], words["angle"])])
    words = pd.concat([words, _on_map(words, corners)], axis=1)
    found = {}
    for fold in C.FOLDS:
        q = None if fold == "all" else paths.QUARTERS.index(fold)
        mine = _fold_words(words, held, q)
        pack.spreads[fold], off = C.measure_spreads(pack, mine, held_blocks=held, quarter=q)
        found[fold] = {k: {"words": int((mine["kind"] == k).sum()),
                           "found": int(off.loc[mine["kind"] == k, "found"].sum()),
                           "here": int(off.loc[mine["kind"] == k, "here"].sum())}
                       for k in ("street name", "block number")}
        if fold == "all":
            unnamed = mine.loc[(mine["kind"] == "street name") & ~off["found"], "text"].value_counts()
    pack.save(PACK_FILE)
    summary = {**pack.counts(), "words_measured": found,
               "street_words_naming_no_street": {"distinct": int(len(unnamed)),
                                                 "top": unnamed.head(25).to_dict()},
               "answered_blocks": len(held), "seconds": round(time.perf_counter() - t0, 1)}
    SUMMARY_FILE.write_text(json.dumps(summary, indent=1) + "\n")
    return summary


# --- fit: reading our boxes -------------------------------------------------------------------------------------------

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
    """Every matched box of one sheet: its crop (`crops.box_crop`), read both ways by its quarter's reader; the ink
    of each candidate string (`context.ink_logp`), mixed by the classifier's q on the crop as cut (`read.weight`);
    each way's best path."""
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
                lps.append(lp)
                best.append(model.best_path(lp))
                p_cls.append(float(torch.sigmoid(direction(x))[0]))
        out.append({"key": it["key"], "ink": C.ink_logp(lps, p_cls[0], it["strings"]), "q": p_cls[0], "best": best})
    return out


def matched_words(pack: C.Pack, held: set[str], sheets=paths.SHEETS) -> pd.DataFrame:
    """Our boxes matched to Terra's words at IoU >= IOU (a box to its best word), with Terra's string (`truth`, in the
    reader's alphabet), its kind, block, quarter and touches, the box's place on the map and its kind_p, and the
    box's four priors (`prior_<q>`, with the spreads measured with q left out)."""
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.wordread import measures
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    boxes = boxes[boxes["sheet"].isin(sheets)]
    terra = measures.terra_words(sheets)
    terra["word"] = terra.groupby("sheet").cumcount()
    rows = _crop_rows()
    terra = terra.merge(rows[["sheet", "word", "terra", "text", "block", "quarter", *TOUCH]]
                        .rename(columns={"text": "truth", "quarter": "word_quarter"}), on=["sheet", "word"],
                        how="left", validate="one_to_one")
    if not (terra["terra"] == terra["text"]).all():
        raise ValueError("Terra's words and text-crops' rows disagree: rebuilt at different times?")
    ti, bj, iou = measures._overlaps(boxes, terra)
    word, best = measures._best_word(ti, bj, iou, len(boxes))
    b = np.flatnonzero((best >= IOU) & (word >= 0))
    t = terra.iloc[word[b]].reset_index(drop=True)
    b, b_iou = boxes.index.to_numpy()[b], best[b]  # the box's row of boxes.parquet (reads.npz's row), its IoU
    keep = (t["truth"].notna() & (t["truth"] != "")).to_numpy()
    b, b_iou, t = b[keep], b_iou[keep], t[keep].reset_index(drop=True)
    bx = boxes.loc[b].reset_index(drop=True)
    words = pd.DataFrame({"row": b, "sheet": bx["sheet"], "box_quarter": bx["quarter"].astype(int),
                          "truth": t["truth"], "kind": t["kind"], "block": t["block"].fillna("").astype(str),
                          "quarter": t["word_quarter"].astype(int), "iou": b_iou,
                          **{c: t[c].astype(bool) for c in TOUCH}})
    g = bx[measures.G_COLS].to_numpy(float).reshape(-1, 4, 2)
    words = pd.concat([words, _on_map(words, g)], axis=1)
    words["kind_p"] = list(bx[KIND_P].to_numpy(float))
    words["t_corners"] = list(bx[[f"t{i}{a}" for i in range(4) for a in "xy"]].to_numpy(float).reshape(-1, 4, 2))
    words["t_length"], words["t_height"], words["axis_page"] = (bx["t_length"].to_numpy(), bx["t_height"].to_numpy(),
                                                               bx["axis"].to_numpy())
    for qn in paths.QUARTERS:
        words[f"prior_{qn}"] = [C.prior(pack, {"x": x, "y": y, "axis": a}, kp, pack.spreads[qn])
                                for x, y, a, kp in zip(words["x"], words["y"], words["axis"], words["kind_p"])]
    return words


def read_words(words: pd.DataFrame) -> pd.DataFrame:
    """`ink`, `q` and `best` for every word (`_read_sheet`), in WORKERS processes of one thread."""
    jobs = []
    for sheet, part in words.groupby("sheet", sort=True):
        items = []
        for k, r in part.iterrows():
            strings = sorted({r["truth"]} | {s for qn in paths.QUARTERS for s in r[f"prior_{qn}"] if s != C.ESCAPE})
            items.append({"key": int(k), "corners": r["t_corners"], "t_length": float(r["t_length"]),
                          "t_height": float(r["t_height"]), "axis": float(r["axis_page"]),
                          "q": paths.QUARTERS[int(r["box_quarter"])], "strings": strings})
        jobs.append((sheet, items))
    with get_context("fork").Pool(WORKERS, initializer=_init) as pool:
        got = {g["key"]: g for part in pool.imap(_read_sheet, jobs) for g in part}
    words = words.copy()
    words["ink"] = [got[int(k)]["ink"] for k in words.index]
    words["q_180"] = [got[int(k)]["q"] for k in words.index]
    words["best"] = [got[int(k)]["best"][0] if got[int(k)]["q"] < 0.5 else got[int(k)]["best"][1]
                     for k in words.index]
    return words


def check_against_store(words: pd.DataFrame) -> dict:
    """Our ink of the code forms against `text-read/reads.npz`'s mixture at the same boxes: the reading is the
    stage's (max absolute difference of the probabilities)."""
    from sfzmaps.solver.textread import store
    with np.load(store.READ_DIR / "reads.npz") as z:
        forms, post = [str(f) for f in z["forms"]], z["post"]
    diffs = []
    for r in words.itertuples(index=False):
        for k, f in enumerate(forms):
            if f in r.ink:
                diffs.append(abs(math.exp(r.ink[f]) - float(post[r.row, k])))
    return {"compared": len(diffs), "max_abs_diff": float(max(diffs)) if diffs else None}


def _decode(post: dict, best: str) -> str:
    top = max(post.items(), key=lambda kv: (kv[1], kv[0] != C.OTHER))[0]
    return best if top == C.OTHER else top


def score_fold(words: pd.DataFrame, qn: str, w: float, e: float) -> dict:
    """q's own words: the mean log posterior of Terra's string (pack at (w, e) against the ink alone) and top-1
    against Terra's string (the pack's decode: its likeliest listed string, or the best path when OTHER wins;
    against the ink's own decode over the same strings, w = 0; and against the best path alone), by kind."""
    col = f"prior_{qn}"
    fw = words.assign(prior=words[col])
    out = {}
    for kind in ["all", *C.KIND_ORDER]:
        d = fw if kind == "all" else fw[fw["kind"] == kind]
        if not len(d):
            continue
        pack_ll, ink_ll = C.truth_loglik(d, w, e), C.truth_loglik(d, 0.0, 0.5)
        tops = {"pack": [], "ink_listed": [], "best_path": []}
        for r in d.itertuples(index=False):
            listed = {s: r.ink[s] for s in r.prior if s != C.ESCAPE}
            tops["pack"].append(_decode(C.posterior(listed, r.prior, w, e), r.best) == r.truth)
            tops["ink_listed"].append(_decode(C.posterior(listed, r.prior, 0.0, 0.5), r.best) == r.truth)
            tops["best_path"].append(r.best == r.truth)
        out[kind] = {"n": int(len(d)), "truth_listed": int(sum(t in p for t, p in zip(d["truth"], d["prior"]))),
                     "mean_loglik_pack": round(float(np.mean(pack_ll)), 4),
                     "mean_loglik_ink": round(float(np.mean(ink_ll)), 4),
                     **{f"top1_{k}": round(float(np.mean(v)), 4) for k, v in tops.items()}}
    return out


PROFILE_W = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


PROFILE_E = (1.125e-07, 1e-05, 1e-04, 1e-03, 3e-03, 1e-02, 3e-02, 0.1)
WORD_COLS = ["row", "sheet", "box_quarter", "truth", "kind", "block", "quarter", "iou", *TOUCH, "x", "y", "axis",
             "q_180", "best"]
JSON_COLS = ["ink", *[f"prior_{qn}" for qn in paths.QUARTERS]]


def save_words(words: pd.DataFrame, f) -> None:
    """The read words (`read_words`): their columns, with each ink and prior as JSON (-inf as null)."""
    out = words[WORD_COLS].copy()
    out["kind_p"] = [json.dumps([float(v) for v in kp]) for kp in words["kind_p"]]
    for c in JSON_COLS:
        out[c] = [json.dumps({k: (v if math.isfinite(v) else None) for k, v in d.items()}) for d in words[c]]
    out.to_parquet(f, index=False)


def load_words(f) -> pd.DataFrame:
    words = pd.read_parquet(f)
    words["kind_p"] = [np.array(json.loads(s)) for s in words["kind_p"]]
    for c in JSON_COLS:
        words[c] = [{k: (-math.inf if v is None else v) for k, v in json.loads(s).items()} for s in words[c]]
    return words


def fit(*sheets: str) -> dict:
    """See the module; with `sheets`, a smoke run on those sheets alone (sf-2000-fit-smoke.json).  The read words are
    kept in sf-2000-fit-words.parquet (`refit` fits again from them without reading)."""
    t0 = time.perf_counter()
    pack = C.Pack.load(PACK_FILE)
    held = _held_blocks()
    words = matched_words(pack, held, sheets or paths.SHEETS)
    log.info("%d matched words; reading (%.0f s)", len(words), time.perf_counter() - t0)
    words = read_words(words)
    if not sheets:
        save_words(words, WORDS_FILE)
    out = {"iou": IOU, "matched_words": int(len(words)), "by_kind": words["kind"].value_counts().to_dict(),
           "reading_check": check_against_store(words)}
    out.update(fit_folds(words, held))
    out["seconds"] = round(time.perf_counter() - t0, 1)
    (FIT_FILE if not sheets else C.PACKS_DIR / f"sf-{YEAR}-fit-smoke.json").write_text(
        json.dumps(out, indent=1, default=float) + "\n")
    return out


def refit() -> dict:
    """`fit` again from the kept read words (no reading)."""
    t0 = time.perf_counter()
    words = load_words(WORDS_FILE)
    out = json.loads(FIT_FILE.read_text()) if FIT_FILE.is_file() else {}
    out.update(fit_folds(words, _held_blocks()))
    out["refit_seconds"] = round(time.perf_counter() - t0, 1)
    FIT_FILE.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out


def fit_folds(words: pd.DataFrame, held: set[str]) -> dict:
    """Per held-out quarter: the fit on the other three, the log-likelihood's profiles in w (at the fitted e) and in e
    (at the fitted w), and the scores on the quarter's own words."""
    out = {"folds": {}}
    for q, qn in enumerate(paths.QUARTERS):
        train = _fold_words(words, held, q).assign(prior=lambda d: d[f"prior_{qn}"])
        if not len(train):
            continue
        got = C.fit_prior(train[["truth", "ink", "prior", "block", "quarter", "touch_blocks"]], held_blocks=held,
                          quarter=q)
        test = words[(words["quarter"] == q) & ~words["touch_blocks"] & ~words["block"].isin(held)]
        out["folds"][qn] = {"fit": got, "train_by_kind": train["kind"].value_counts().to_dict(),
                            "profile_w_at_fitted_e": {w: round(float(C.truth_loglik(train, w, got["e"]).sum()), 2)
                                                      for w in PROFILE_W},
                            "profile_e_at_fitted_w": {e: round(float(C.truth_loglik(train, got["w"], e).sum()), 2)
                                                      for e in PROFILE_E},
                            "held_out": score_fold(test, qn, got["w"], got["e"])}
        log.info("%s: w %.3f e %.4g (n %d)", qn, got["w"], got["e"], got["n"])
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    what = sys.argv[1] if len(sys.argv) > 1 else "build"
    result = {"build": build, "fit": fit, "refit": refit}[what](*sys.argv[2:])
    print(json.dumps(result, indent=1, default=float)[:6000])
