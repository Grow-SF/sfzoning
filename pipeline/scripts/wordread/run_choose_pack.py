"""Task 11's read-time experiments: (5) the reader choosing between versions of each box, and (4) the context packs.
No training.  Outputs go to data/solver/experiments/wordread/choose-pack/<name>-<quarter>-s<seed>/ (never committed).

    uv run python scripts/wordread/run_choose_pack.py choose --name prelim --quarter ne --seed 0
    uv run python scripts/wordread/run_choose_pack.py decide --name prelim --quarter ne      # NE, seeds 0 and 1
    uv run python scripts/wordread/run_choose_pack.py page --name prelim --seed 0            # Mission sheet's page

Item (5), box choice (`wordread.choose`): each box gets its versions (as found, split at the widest gap, joined with the
nearest box, a character trimmed at either end), each version's words are read by the reader that never saw the box's
quarter (CPU, one thread per worker), and the version is chosen.  Two arms are scored against reading the boxes as found:
  plain   the version whose words' log posteriors sum highest (`choose(params=None)`: no fitted weight at all);
  fitted  each version's own quality (`choose.fit_quality`: a fractional logistic regression of its label weight on
          its features, a split's quality the product of its halves'; the version of the highest quality is kept, no
          shares between versions), fitted per fold on the TRAINING quarters' boxes outside the answered blocks,
          soft-labelled by Terra's words' ink (`choose.label_weights`), never by the owner's answers.
Every box is read once by its own quarter's reader (the run's reader for the held-out quarter: `--reader-name`, at the
run's seed, else the stage's `<q>.pt`), so a training quarter's scores are as out-of-sample as the held-out quarter's.
`--boxes-dir`, `--reader-name` and `--reader-root` take another run's boxes and readers (Task 8's).

The keep rule (written before the runs; brief of Task 11): (5) is kept if, on NE over the kept models' two seeds, the
headline rises with the paired label interval excluding zero and the review page agrees.  `decide` prints those numbers
and decides nothing: a run of today's boxes and readers is PRELIMINARY.

Item (4), the pack (`pack_read_fn`): a word's reading is `context.read_posterior` (Task 10's recipe: the prior at the word's own
place on the map, its fold's tau, w and temperature); `choose(..., read_fn=pack_read_fn(...))`.
Run on today's NE boxes as found (`run_choose_pack.py pack`, `pack --page`): PRELIMINARY.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
import torch  # noqa: E402

from sfzmaps import config  # noqa: E402
from sfzmaps.solver import grid, paths, runs  # noqa: E402
from sfzmaps.solver.textnet import data  # noqa: E402
from sfzmaps.solver.textread import crops as K  # noqa: E402
from sfzmaps.solver.textread import model, store  # noqa: E402
from sfzmaps.solver.textread import read as R  # noqa: E402
from sfzmaps.solver.wordread import choose as C  # noqa: E402
from sfzmaps.solver.wordread import context, measures  # noqa: E402

log = logging.getLogger(__name__)

ROOT = runs.EXPERIMENTS_DIR / "wordread"
OUT = ROOT / "choose-pack"
WORKERS = 4
T_COLS = [f"t{i}{a}" for i in range(4) for a in "xy"]
G_COLS = measures.G_COLS
C_COLS = measures.C_COLS
HOLD = [f"touch_{q}" for q in paths.QUARTERS]
ARMS = ("without", "plain", "fitted")


# --- geometry of a chosen version ---------------------------------------------------------------------------------------

def _frame(box: dict):
    corners = np.array([[box[f"t{i}x"], box[f"t{i}y"]] for i in range(4)], float)
    a = math.radians(float(box["axis"]))
    return corners.mean(0), np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])


def _extent(points: np.ndarray, c, u, v) -> tuple[float, float, float, float]:
    p, q = (points - c) @ u, (points - c) @ v
    return float(p.min()), float(p.max()), float(q.min()), float(q.max())


def _rect(c, u, v, p0, p1, q0, q1) -> np.ndarray:
    return np.array([c + p * u + q * v for p, q in ((p0, q0), (p1, q0), (p1, q1), (p0, q1))])


def grouped_rect(box: dict, cand: C.Candidate, k: int) -> np.ndarray:
    """The grouped rectangle (4, 2) a version's k-th word takes, the one a label's point is held by.  The box's own
    grouped rectangle for the box as found and the trims; for a split, each half of it, cut where the words are cut (the
    cut halfway across the gap); for a join, the box's grouped rectangle grown to hold the joined words."""
    g = np.array([[box[f"g{i}x"], box[f"g{i}y"]] for i in range(4)], float)
    if cand.name in ("as_found", "trim_start", "trim_end"):
        return g
    c, u, v = _frame(box)
    gp0, gp1, gq0, gq1 = _extent(g, c, u, v)
    rp0, rp1, rq0, rq1 = _extent(cand.rects[k], c, u, v)
    if cand.name == "join_nearest":
        return _rect(c, u, v, min(gp0, rp0), max(gp1, rp1), min(gq0, rq0), max(gq1, rq1))
    a, b = _extent(cand.rects[0], c, u, v), _extent(cand.rects[1], c, u, v)
    mid = (a[1] + b[0]) / 2
    p0, p1 = (min(gp0, rp0), mid) if k == 0 else (mid, max(gp1, rp1))
    return _rect(c, u, v, p0, p1, min(gq0, rq0), max(gq1, rq1))


# --- one box's versions, read (in workers) ------------------------------------------------------------------------------

_W: dict = {}


def _init() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)


def read_versions(page: np.ndarray, ink: np.ndarray, rows: list[dict], mine: list[int], reader, direction,
                  forms: list[str], widths: dict) -> dict:
    """For each box of `rows` at the positions `mine` (rows of one sheet, boxes.parquet's columns): its versions
    (`choose.candidates` against the other boxes of the sheet) and each version's words read, `{"cands": [...],
    "gots": [[`read.read_one`'s dict per word] per version]}`, by the row.  The box as found is cut as the stage cuts it."""
    out = {}
    for k in mine:
        box = rows[k]
        cands = C.candidates(box, ink, rows[:k] + rows[k + 1:], widths)
        gots = []
        for cand in cands:
            got = []
            for r in cand.rects:
                if cand.name == "as_found":
                    crop = K.box_crop(page, np.array([[box[f"t{i}x"], box[f"t{i}y"]] for i in range(4)], float),
                                      box["t_length"], box["t_height"], box["axis"])
                else:
                    length, height = C.dims(r, float(box["axis"]))
                    crop = K.box_crop(page, r, length, height, box["axis"])
                got.append(R.read_one(reader, direction, crop, forms))
            gots.append(got)
        out[int(box["row"])] = {"cands": cands, "gots": gots}
    return out


def _version_job(args) -> dict:
    sheet, qname, rows, mine = args
    page = data.load_page(sheet)
    return read_versions(page, page < grid.INK_BELOW, rows, mine, _W["readers"][qname], _W["directions"][qname],
                         _W["forms"], C.load_char_widths(held_out=qname))


def versions_of_quarter(boxes: pd.DataFrame, qi: int, reader_path: Path, workers: int, cache: Path) -> dict:
    """Every box of quarter `qi`: its versions and their reads (`read_versions`), by row, read by the reader at
    `reader_path` and the stage's 0/180 classifier of the quarter, cached beside the run's other outputs."""
    qname = paths.QUARTERS[qi]
    key = hashlib.sha1(reader_path.read_bytes() + pd.util.hash_pandas_object(boxes[T_COLS + ["quarter"]]).values
                       .tobytes() + (store.TRAIN_DIR / f"{qname}_direction.pt").read_bytes()
                       + json.dumps(C.load_char_widths(held_out=qname), sort_keys=True).encode()).hexdigest()[:12]
    f = cache / f"versions-{qname}-{key}.pkl"
    if f.is_file():
        return pickle.loads(f.read_bytes())
    t0 = time.perf_counter()
    codes = R.CodeList.load()
    _W.update(readers={qname: model.load(reader_path)}, forms=codes.forms,
              directions={qname: model.load(store.TRAIN_DIR / f"{qname}_direction.pt", "direction")})
    jobs = []
    for sheet, part in boxes.groupby("sheet", sort=True):
        rows = part.to_dict("records")
        mine = [k for k, r in enumerate(rows) if int(r["quarter"]) == qi]
        if mine:
            jobs.append((sheet, qname, rows, mine))
    out: dict = {}
    with get_context("fork").Pool(workers, initializer=_init) as pool:
        for part in pool.imap(_version_job, jobs):
            out.update(part)
    cache.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(out))
    log.info("%s: %d boxes' versions read with %s (%.0f s)", qname, len(out), reader_path.name,
             time.perf_counter() - t0)
    return out


# --- scoring the versions -----------------------------------------------------------------------------------------------

def open_log_post(got: dict, codes: R.CodeList) -> dict[str, float]:
    """A word's reading as `choose` takes it: each code's log posterior (its forms summed), "other" and "none", over the
    ink alone."""
    c = R.combine(got, R.weight(got), codes)
    with np.errstate(divide="ignore"):
        return {h: math.log(p) if p > 0 else -math.inf for h, p in R.open_by_code(c["post"], codes).items()}


def read_fn_of(entry: dict, codes: R.CodeList):
    """`choose`'s `read_fn` over one box's stored reads: the word rectangle (an array of the entry's own) to its log
    posterior."""
    by = {r.tobytes(): got for cand, gots in zip(entry["cands"], entry["gots"]) for r, got in zip(cand.rects, gots)}
    memo: dict = {}

    def read(corners, box):
        key = np.asarray(corners).tobytes()
        if key not in by:
            raise KeyError("a word that was not read")
        if key not in memo:
            memo[key] = open_log_post(by[key], codes)
        return memo[key]
    return read


def terra_index(terra: pd.DataFrame) -> dict:
    """Per sheet: the STRtree of Terra's word polygons, their positions in `terra` and their rectangles."""
    out = {}
    for sheet, t in terra.groupby("sheet", sort=True):
        rects = t[C_COLS].to_numpy(float).reshape(-1, 4, 2)
        out[sheet] = (shapely.STRtree([shapely.Polygon(r) for r in rects]), t.index.to_numpy(), rects)
    return out


def terra_near(index: dict, sheet: str, cands) -> list[int]:
    """The positions (in `terra`) of Terra's words whose rectangle meets any version of the box (an overlap of any
    size: a geometric inclusion, nothing decided)."""
    if sheet not in index:
        return []
    tree, pos, _ = index[sheet]
    hull = shapely.unary_union([shapely.Polygon(r) for c in cands for r in c.rects])
    return sorted(int(pos[i]) for i in tree.query(hull, predicate="intersects"))


def feature_rows(boxes: pd.DataFrame, versions: dict, terra: pd.DataFrame, codes: R.CodeList, ink_of=None) -> pd.DataFrame:
    """One row per version of every box in `versions`, `choice_features`'s columns plus `box` (the row), `quarter`,
    `weight` (`choose.label_weights`: each version's chance of holding exactly one Terra word, by ink), `answered` (some
    Terra word meeting the box touches an answered block) and `touch_<q>` (some touches quarter q), for the fold fits.
    `ink_of(sheet)` the page's ink (default: the page's px darker than grid.INK_BELOW)."""
    ink_of = ink_of or (lambda sheet: data.load_page(sheet) < grid.INK_BELOW)
    index = terra_index(terra)
    by_row = boxes.set_index("row", drop=False)
    out = []
    for sheet in sorted(set(boxes.loc[sorted(versions), "sheet"])):
        ink = ink_of(sheet)
        for row in sorted(r for r in versions if by_row.loc[r, "sheet"] == sheet):
            box, e = by_row.loc[row].to_dict(), versions[row]
            near = terra_near(index, sheet, e["cands"])
            sub = terra.loc[near] if near else terra.iloc[:0]
            w = C.label_weights(e["cands"], [r.reshape(4, 2) for r in sub[C_COLS].to_numpy(float)], ink)
            f = C.choice_features(box, e["cands"], read_fn_of(e, codes))
            f["box"], f["quarter"], f["weight"] = row, int(box["quarter"]), [w[c] for c in f["cand"]]
            f["answered"] = bool(sub["touch_blocks"].any()) if near else False
            for h in HOLD:
                f[h] = bool(sub[h].any()) if near else False
            out.append(f)
    return pd.concat(out, ignore_index=True)


FIT_COLS = ["box", "cand", "word_scores", "gap", "across", "kind", "weight", "held_out", "answered"]


def fold_fit(rows: pd.DataFrame, q: int) -> dict:
    """Fold q's quality parameters: `fit_quality` on the boxes of the other quarters outside the answered blocks and not
    meeting a Terra word that touches quarter q (with the box model's `word_count1` when the rows carry it)."""
    use = rows[(rows["quarter"] != q) & ~rows["answered"] & ~rows[HOLD[q]]].copy()
    use["held_out"] = False
    return C.fit_quality(use[FIT_COLS + [c for c in ("word_count1",) if c in use]])


def calibration(params: dict, rows: pd.DataFrame) -> dict:
    """Per version: the mean predicted quality (`choose.predict_quality`) against the mean label weight, and the
    count, over `rows` (a report, decides nothing)."""
    q = C.predict_quality(params, rows.reset_index(drop=True))
    d = pd.DataFrame({"cand": rows["cand"].to_numpy(), "quality": q, "weight": rows["weight"].to_numpy(float)})
    g = d.groupby("cand")
    return {c: {"mean_quality": round(float(g["quality"].mean()[c]), 4),
                "mean_weight": round(float(g["weight"].mean()[c]), 4), "n": int(g.size()[c])} for c in g.groups}


# --- the chosen boxes ---------------------------------------------------------------------------------------------------

def choose_all(boxes: pd.DataFrame, versions: dict, codes: R.CodeList, params: dict | None) -> dict[int, str]:
    """The version every box in `versions` keeps (`choose.choose`: with `params` the highest quality; None: the plain
    sum of log posteriors)."""
    by_row = boxes.set_index("row", drop=False)
    return {row: C.choose(by_row.loc[row].to_dict(), e["cands"], read_fn_of(e, codes), params=params)[0]
            for row, e in versions.items()}


def assemble(boxes: pd.DataFrame, versions: dict, picks: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    """The boxes table after the choice: each box of `boxes` as its chosen version's words (a split gives two rows, a
    join one row of both words; a word chosen twice, as two boxes joined with each other, once), with its own columns
    but the trimmed rectangle (`t<i>x/y`, `t_length`, `t_height`), the grouped one (`grouped_rect`), `row` (the new
    position) and `src_row`, `choice`.  Also {new row: the word's read} for `read.combine`."""
    rows, gots, seen = [], {}, set()
    for src, box in boxes.set_index("row", drop=False).iterrows():
        box = box.to_dict()
        if src not in versions:
            continue
        e = versions[src]
        k = next(i for i, c in enumerate(e["cands"]) if c.name == picks[src])
        cand = e["cands"][k]
        for w, rect in enumerate(cand.rects):
            key = (box["sheet"], tuple(np.round(rect.reshape(-1), 3)))
            if cand.name != "as_found":
                if key in seen:
                    continue
                seen.add(key)
            new = dict(box)
            if cand.name != "as_found":
                new["t_length"], new["t_height"] = C.dims(rect, float(box["axis"]))
            for i in range(4):
                new[f"t{i}x"], new[f"t{i}y"] = (box[f"t{i}x"], box[f"t{i}y"]) if cand.name == "as_found" \
                    else rect[i]
            g = grouped_rect(box, cand, w)
            for i in range(4):
                new[f"g{i}x"], new[f"g{i}y"] = g[i]
            new["src_row"], new["choice"] = int(src), cand.name
            gots[len(rows)] = e["gots"][k][w]
            rows.append(new)
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["row"] = np.arange(len(out))
    return out, gots


def reads_of(gots: dict, codes: R.CodeList) -> dict:
    """`measures`' reads {row: {"closed": {code: p}, "top": hypothesis}} from each row's read."""
    out = {}
    for row, g in gots.items():
        c = R.combine(g, R.weight(g), codes)
        op = R.open_by_code(c["post"], codes)
        out[int(row)] = {"closed": dict(zip(codes.codes, np.exp(c["log_closed"]).tolist())),
                         "top": min(op.items(), key=lambda kv: (-kv[1], kv[0]))[0]}
    return out


def score_arm(table: pd.DataFrame, gots: dict, qnum: int, answers: pd.DataFrame, terra: pd.DataFrame,
              codes: R.CodeList) -> dict:
    """An arm's numbers on quarter `qnum`: the labels (per answered label, right or wrong), the headline, the box
    numbers and the read numbers at IoU 0.5 and 0.3."""
    reads = reads_of(gots, codes)
    bq = table[table["quarter"] == qnum].set_index("row", drop=False)
    labels = measures.label_right(bq, reads, answers)
    return {"labels": labels, "headline": measures.headline(bq, reads, answers, paths.QUARTERS),
            "box_numbers": measures.box_numbers(bq, terra, reads),
            "read_numbers": {str(i): measures.read_numbers(bq, terra, reads, forms=dict(zip(codes.forms,
                                                                                          codes.form_code)),
                                                           answers=answers, iou=i) for i in (0.5, 0.3)}}


def write_text_read(table: pd.DataFrame, gots: dict, codes: R.CodeList, folder: Path) -> Path:
    """A text-read folder (`review_page.py --reads`) of the chosen boxes: reads.npz, reads.parquet (each word's best
    paths) and the stage's answered.parquet."""
    n, F = len(table), len(codes.forms)
    post, log_closed = np.zeros((n, F + 2)), np.zeros((n, len(codes.codes)))
    lf, ln, po, pc, summary = np.zeros((n, 2, F)), np.zeros((n, 2)), np.zeros((n, 2)), np.zeros(n), []
    for k in range(n):
        g = gots[k]
        q = R.weight(g)
        c = R.combine(g, q, codes)
        post[k], log_closed[k], pc[k] = c["post"], c["log_closed"], q
        for d in (0, 1):
            lf[k, d], ln[k, d], po[k, d] = g["reads"][d]
        top = min(R.open_by_code(c["post"], codes).items(), key=lambda kv: (-kv[1], kv[0]))
        summary.append({"row": k, "sheet": table["sheet"].iloc[k], "i": int(table["i"].iloc[k]),
                        "quarter": int(table["quarter"].iloc[k]), "axis": float(table["axis"].iloc[k]),
                        "best_0": g["best"][0], "best_180": g["best"][1], "p_classifier": q, "top": top[0],
                        "p_top": float(top[1])})
    folder.mkdir(parents=True, exist_ok=True)
    store.save_npz(folder / "reads.npz", row=np.arange(n, dtype=np.int64), forms=np.array(codes.forms),
                   form_code=np.array(codes.form_code), codes=np.array(codes.codes), log_forms=lf, log_none=ln,
                   p_other=po, p_classifier=pc, post=post, log_closed=log_closed)
    pd.DataFrame(summary).to_parquet(folder / "reads.parquet", index=False)
    pd.read_parquet(store.READ_DIR / "answered.parquet").to_parquet(folder / "answered.parquet", index=False)
    return folder


# --- the pack (item 4) --------------------------------------------------------------------------------------------------

def read_lp(reader, direction, crop: np.ndarray) -> tuple[list, float, list[str]]:
    """A crop's log-probabilities as cut and turned a half turn (frames x classes each), the classifier's p(upside down)
    of the crop as cut, and each read's best path: what `context.ink_logp` and the recipe's `strings` take."""
    from sfzmaps.solver.textread import ctc
    out = []
    with torch.inference_mode():
        for c in (crop, np.ascontiguousarray(np.rot90(crop, 2))):
            x, n = model.to_input([c])
            out.append(ctc.log_softmax(reader(x)[0, :n[0]].numpy()))
            if len(out) == 1:
                q = float(torch.sigmoid(direction(x))[0])
    return out, q, [model.best_path(lp) for lp in out]


def box_kind_p(box: dict) -> np.ndarray:
    """The box's kind probabilities in `context.KIND_ORDER` (the detector's `p_<kind>`)."""
    return np.array([float(box.get(f"p_{k}", 0.0)) for k in C.KIND_NAMES])


def pack_read_fn(lp_fn, pack, transforms: dict, base_strings=()):
    """`choose`'s `read_fn` with the context pack, by Task 10's recipe (`context.read_posterior`): a word's calibrated
    log posterior over the strings its prior lists, `base_strings` and the reader's best paths, and OTHER.
    `lp_fn(corners, box)` returns (the two reads' log-probabilities, q, the two best paths); the word is placed on the
    map through `transforms[box["sheet"]]` and read with its box's own quarter's fit (`box["quarter"]`).  Give no
    `prior_fn` to `choose`: the prior is in here."""
    def read(corners, box):
        reads, q, best = lp_fn(corners, box)
        return context.read_posterior(
            pack, paths.QUARTERS[int(box["quarter"])], lambda names, tau: context.ink_logp(reads, q, names, tau),
            {**context.box_on_map(corners, transforms[box["sheet"]]), "kind_p": box_kind_p(box)},
            sorted(set(base_strings) | set(best)))
    return read


def pack_job(args) -> dict:
    """One sheet's boxes of one quarter (rows of boxes.parquet) read as found, once: `read.read_one`'s dict by the row
    (`got`, as the stage reads), the pack's posterior (`post`: string -> log p, OTHER included) and the two best paths."""
    sheet, qname, rows, mine = args
    page = data.load_page(sheet)
    reader, direction = _W["readers"][qname], _W["directions"][qname]
    forms, pack, transforms = _W["forms"], _W["pack"], _W["transforms"]
    from sfzmaps.solver.textread import ctc
    base = [*forms, ""]
    out = {}
    for k in mine:
        b = rows[k]
        corners = np.array([[b[f"t{i}x"], b[f"t{i}y"]] for i in range(4)], float)
        crop = K.box_crop(page, corners, b["t_length"], b["t_height"], b["axis"])
        reads, q, best = read_lp(reader, direction, crop)
        got = {"reads": [ctc.read(lp, forms) for lp in reads], "best": best, "p_classifier": [q, float("nan")]}
        fn = pack_read_fn(lambda c, bx: (reads, q, best), pack, transforms, base)
        out[int(b["row"])] = {"got": got, "post": fn(corners, b), "best": best, "q": q}
    return out


def top_text(post: dict, best: list, q: float) -> str | None:
    """What a pack posterior reads: its likeliest named string ("" for no text), None when OTHER is likeliest (the
    pack names none of the strings); a posterior None: the best path in the classifier's direction."""
    if post is None:
        return best[1] if q > 0.5 else best[0]
    k = max(sorted(post), key=lambda h: post[h])
    return None if k == context.OTHER else k


def pack_reads(post: dict, codes: R.CodeList) -> dict:
    """`measures`' read of a pack posterior: `closed` the codes' probability (their forms' posterior renormalised over
    the codes), `top` "none" (the empty string likeliest), else the likeliest string (OTHER: "other")."""
    mass = np.zeros(len(codes.codes))
    for form, c in zip(codes.forms, codes.code_of):
        if form in post:
            mass[c] += math.exp(post[form])
    tot = float(mass.sum())
    closed = mass / tot if tot > 0 else np.full(len(codes.codes), 1.0 / len(codes.codes))
    k = max(sorted(post), key=lambda h: post[h])
    return {"closed": dict(zip(codes.codes, closed.tolist())), "top": "none" if k == "" else
            ("other" if k == context.OTHER else k)}


TERRA_KIND = {"street name": "street", "block number": "block", "zoning code": "code"}


def norm_text(t) -> str:
    return " ".join(str(t).replace("\u200b", "").upper().split())


def text_numbers(boxes_q: pd.DataFrame, terra_q: pd.DataFrame, texts: dict[str, dict], iou: float = 0.5) -> dict:
    """Against Terra's words, per kind (street names, block numbers, zoning codes): of the boxes matched to a Terra word of
    that kind at `iou` (each box to its best word: `measures`' rule), how many read exactly its string.  `texts`:
    {arm: {box row: the string read, or None}}; `boxes_q` indexed by row."""
    ti, bj, v = measures._overlaps(boxes_q, terra_q)
    word, best = measures._best_word(ti, bj, v, len(boxes_q))
    out = {}
    for arm, tx in texts.items():
        res = {k: {"matched": 0, "same": 0} for k in TERRA_KIND.values()}
        for pos in np.flatnonzero((best >= iou) & (word >= 0)):
            kind = TERRA_KIND.get(terra_q["kind"].iloc[word[pos]])
            if kind is None:
                continue
            res[kind]["matched"] += 1
            read = tx.get(int(boxes_q.index[pos]))
            res[kind]["same"] += int(read is not None and norm_text(read) == norm_text(terra_q["text"].iloc[word[pos]]))
        out[arm] = res
    return out


def write_pack_text_read(boxes: pd.DataFrame, found: dict, codes: R.CodeList, folder: Path) -> Path:
    """A text-read folder (`review_page.py --reads`) of the pack's reads of the boxes in `found` (row -> `pack_job`'s
    entry): each box's posterior over the codes' forms, "other" (OTHER and every string that is no code form's) and "none"
    (the empty string); `best_0`/`best_180` hold what the pack reads (`top_text`; the best path where OTHER is likeliest),
    so the page's text is the pack's."""
    rows = sorted(found)
    F = len(codes.forms)
    post, log_closed, pc, summary = np.zeros((len(rows), F + 2)), np.zeros((len(rows), len(codes.codes))), \
        np.zeros(len(rows)), []
    for k, r in enumerate(rows):
        e = found[r]
        p = e["post"]
        pr = np.array([math.exp(p[f]) if f in p else 0.0 for f in codes.forms])
        none = math.exp(p[""]) if "" in p else 0.0
        post[k, :F], post[k, F], post[k, F + 1] = pr, max(0.0, 1.0 - float(pr.sum()) - none), none
        read = pack_reads(p, codes)
        log_closed[k] = np.log(np.maximum(list(read["closed"].values()), 1e-300))
        pc[k] = e["q"]
        text = top_text(p, e["best"], e["q"])
        text = text if text is not None else (e["best"][1] if e["q"] > 0.5 else e["best"][0])
        b = boxes.loc[r]
        summary.append({"row": r, "sheet": b["sheet"], "i": int(b["i"]), "quarter": int(b["quarter"]),
                        "axis": float(b["axis"]), "best_0": text, "best_180": text, "p_classifier": e["q"],
                        "top": read["top"], "p_top": float(max(post[k]))})
    folder.mkdir(parents=True, exist_ok=True)
    store.save_npz(folder / "reads.npz", row=np.array(rows, np.int64), forms=np.array(codes.forms),
                   form_code=np.array(codes.form_code), codes=np.array(codes.codes), post=post, log_closed=log_closed,
                   p_classifier=pc)
    pd.DataFrame(summary).to_parquet(folder / "reads.parquet", index=False)
    pd.read_parquet(store.READ_DIR / "answered.parquet").to_parquet(folder / "answered.parquet", index=False)
    return folder


# --- the run ------------------------------------------------------------------------------------------------------------

def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=config.PIPELINE_DIR, capture_output=True, text=True).stdout.strip()


def run_name(name: str, quarter: str, seed: int) -> str:
    return f"{name}-{quarter}-s{seed}"


def reader_path(qname: str, held: str, seed: int, reader_name: str, reader_root: Path) -> Path:
    """The reader that reads quarter `qname`'s boxes: the run's own at the held-out quarter, else the stage's."""
    own = reader_root / f"{reader_name}-{qname}-s{seed}" / "reader.pt"
    return own if own.is_file() else store.TRAIN_DIR / f"{qname}.pt"


def terra_frame() -> pd.DataFrame:
    """Terra's words with their rectangles and the held-out areas they touch (`readercrops.terra_frame`)."""
    from sfzmaps.solver.wordread import readercrops as RC
    f = ROOT / "reader" / "cache" / "terra_frame.pkl"
    if f.is_file():
        return pickle.loads(f.read_bytes())
    return RC.terra_frame(pd.read_parquet(store.CROPS_DIR / "crops.parquet"))


def run(name: str, quarter: str, seed: int, boxes_dir: Path | None = None, reader_name: str = "base",
        reader_root: Path | None = None, workers: int = WORKERS, out_root: Path | None = None) -> Path:
    """Item (5) on fold `quarter` at `seed` (see the module): write <out_root or OUT>/<name>-<quarter>-s<seed>/ and
    return it."""
    if quarter not in paths.QUARTERS:
        raise ValueError(f"quarters are {paths.QUARTERS}, not {quarter!r}")
    q = paths.QUARTERS.index(quarter)
    t0 = time.perf_counter()
    out = (out_root or OUT) / run_name(name, quarter, seed)
    out.mkdir(parents=True, exist_ok=True)
    boxes_dir = Path(boxes_dir) if boxes_dir else paths.SOLVER_DIR / "boxes"
    reader_root = Path(reader_root) if reader_root else ROOT / "runs"
    boxes = pd.read_parquet(boxes_dir / "boxes.parquet").reset_index(drop=True)
    boxes["row"] = np.arange(len(boxes))
    codes = R.CodeList.load()
    torch.set_num_threads(1)
    versions, readers = {}, {}
    for qi, qname in enumerate(paths.QUARTERS):
        readers[qname] = reader_path(qname, quarter, seed, reader_name, reader_root)
        versions.update(versions_of_quarter(boxes, qi, readers[qname], workers, out.parent / "cache"))
    terra_all = terra_frame()
    rows = feature_rows(boxes, versions, terra_all, codes)
    rows.to_parquet(out / "version_features.parquet", index=False)
    params = {qn: fold_fit(rows, qi) for qi, qn in enumerate(paths.QUARTERS)}
    (out / "params.json").write_text(json.dumps(params, indent=1) + "\n")

    picks = {"without": {r: "as_found" for r in versions}, "plain": choose_all(boxes, versions, codes, None)}
    fitted = {}
    for qi, qn in enumerate(paths.QUARTERS):
        mine = {r: e for r, e in versions.items() if int(boxes["quarter"].iloc[r]) == qi}
        fitted.update(choose_all(boxes, mine, codes, params[qn]))
    picks["fitted"] = fitted
    order = sorted(versions)
    pd.DataFrame({"row": order, "quarter": boxes["quarter"].to_numpy()[order],
                  **{arm: [picks[arm][r] for r in order] for arm in ARMS}}).to_parquet(out / "picks.parquet", index=False)

    answers = measures.answers_from_store()
    answers = answers[answers["quarter"] == q].reset_index(drop=True)
    terra = measures.terra_words(sorted(set(boxes["sheet"])))
    terra = terra[terra["quarter"] == q].reset_index(drop=True)
    result = {}
    for arm in ARMS:
        table, gots = assemble(boxes, versions, picks[arm])
        if arm == "fitted":
            table.to_parquet(out / "chosen_boxes.parquet", index=False)
            pickle.dump(gots, (out / "chosen_gots.pkl").open("wb"))
        s = score_arm(table, gots, q, answers, terra, codes)
        s["labels"].to_parquet(out / f"labels_{arm}.parquet", index=False)
        result[arm] = s
    paired = {arm: measures.paired_labels(result[arm]["labels"], result["without"]["labels"])
              for arm in ("plain", "fitted")}
    wins = {arm: dict(Counter(c for r, c in picks[arm].items() if int(boxes["quarter"].iloc[r]) == q))
            for arm in ARMS}
    truth_kind = rows[rows["quarter"] == q].groupby("cand")["weight"].sum().round(1).to_dict()
    info = {"name": name, "quarter": quarter, "seed": seed, "preliminary": True, "boxes": str(boxes_dir),
            "readers": {k: str(v) for k, v in readers.items()}, "params": params[quarter],
            "headline": {a: {k: result[a]["headline"][k] for k in ("n", "right", "share", "ci", "with_box")}
                         for a in ARMS},
            "paired_vs_without": paired, "wins": wins, "terra_label_weight": truth_kind,
            "box_numbers": {a: {str(i): {k: result[a]["box_numbers"][i][k] for k in
                                         ("terra", "ours", "matched", "missed", "missed_reasons",
                                          "ours_unmatched_text", "ours_unmatched_none")} for i in (0.5, 0.3)}
                            for a in ARMS},
            "read_numbers": {a: result[a]["read_numbers"] for a in ARMS},
            "fit_boxes": {qn: params[qn]["boxes"] for qn in paths.QUARTERS},
            "calibration": {"training": calibration(params[quarter], rows[(rows["quarter"] != q) & ~rows["answered"]
                                                                          & ~rows[HOLD[q]]]),
                            "held_out": calibration(params[quarter], rows[rows["quarter"] == q])},
            "git": _git("rev-parse", "HEAD"), "seconds": round(time.perf_counter() - t0, 1)}
    (out / "run.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
    base = ROOT / "runs" / run_name(reader_name, quarter, seed) / "labels.parquet"
    if base.is_file():
        info["matches_reader_run_labels"] = bool((pd.read_parquet(base)["right"].to_numpy()
                                                  == result["without"]["labels"]["right"].to_numpy()).all())
        (out / "run.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
    log.info("%s: without %s, plain %s, fitted %s of %s (%.0f s)", out.name,
             result["without"]["headline"]["right"], result["plain"]["headline"]["right"],
             result["fitted"]["headline"]["right"], result["without"]["headline"]["n"], time.perf_counter() - t0)
    return out


def run_pack(name: str, quarter: str, seed: int, boxes_dir: Path | None = None, reader_name: str = "base",
             reader_root: Path | None = None, pack_file: Path | None = None, workers: int = WORKERS,
             out_root: Path | None = None) -> Path:
    """Item (4) on fold `quarter` at `seed`, the boxes as found: every box of the quarter (and every box of the review sheet
    in the other quarters) read once by its own quarter's reader, and read again with the context pack
    (`context.read_posterior`, the box's own quarter's fit).  Writes <out_root or OUT>/<name>-<quarter>-s<seed>/:
    labels_{without,pack}.parquet, run.json (headline, paired, the read numbers against Terra, the labels changed) and
    the posteriors (`found.pkl`) for the review page."""
    from sfzmaps.solver.wordread import review
    from sfzmaps.transform import SheetTransform
    q = paths.QUARTERS.index(quarter)
    t0 = time.perf_counter()
    out = (out_root or OUT) / run_name(name, quarter, seed)
    out.mkdir(parents=True, exist_ok=True)
    boxes_dir = Path(boxes_dir) if boxes_dir else paths.SOLVER_DIR / "boxes"
    reader_root = Path(reader_root) if reader_root else ROOT / "runs"
    pack_file = Path(pack_file) if pack_file else context.PACKS_DIR / "sf-2000.parquet"
    boxes = pd.read_parquet(boxes_dir / "boxes.parquet").reset_index(drop=True)
    boxes["row"] = np.arange(len(boxes))
    codes = R.CodeList.load()
    pack = context.Pack.load(pack_file)
    transforms = {}
    for sheet in sorted(set(boxes["sheet"])):
        wd = config.DATA_DIR / "warp" / paths.SERIES / sheet
        transforms[sheet] = SheetTransform.from_json(json.loads((wd / "transform.json").read_text()), folder=wd)
    page_sheet = review.mission_sheet()
    torch.set_num_threads(1)
    found: dict = {}
    for qi, qname in enumerate(paths.QUARTERS):
        _W.update(readers={qname: model.load(reader_path(qname, quarter, seed, reader_name, reader_root))},
                  directions={qname: model.load(store.TRAIN_DIR / f"{qname}_direction.pt", "direction")},
                  forms=codes.forms, pack=pack, transforms=transforms)
        jobs = []
        for sheet, part in boxes.groupby("sheet", sort=True):
            rows = part.to_dict("records")
            mine = [k for k, r in enumerate(rows) if int(r["quarter"]) == qi and (qi == q or sheet == page_sheet)]
            if mine:
                jobs.append((sheet, qname, rows, mine))
        with get_context("fork").Pool(workers, initializer=_init) as pool:
            for part in pool.imap(pack_job, jobs):
                found.update(part)
        log.info("%s: %d boxes read with the pack (%.0f s)", qname, sum(1 for r in found if boxes["quarter"].iloc[r] == qi),
                 time.perf_counter() - t0)
    pickle.dump(found, (out / "found.pkl").open("wb"))
    answers = measures.answers_from_store()
    answers = answers[answers["quarter"] == q].reset_index(drop=True)
    terra = measures.terra_words(sorted(set(boxes["sheet"])))
    terra = terra[terra["quarter"] == q].reset_index(drop=True)
    bq = boxes[boxes["quarter"] == q].set_index("row", drop=False)
    rows_q = [int(r) for r in bq["row"]]
    reads = {"without": reads_of({r: found[r]["got"] for r in rows_q}, codes),
             "pack": {r: pack_reads(found[r]["post"], codes) for r in rows_q}}
    res = {}
    for arm in ("without", "pack"):
        labels = measures.label_right(bq, reads[arm], answers)
        labels.to_parquet(out / f"labels_{arm}.parquet", index=False)
        res[arm] = {"labels": labels, "headline": measures.headline(bq, reads[arm], answers, paths.QUARTERS),
                    "box_numbers": measures.box_numbers(bq, terra, reads[arm]),
                    "read_numbers": {str(i): measures.read_numbers(bq, terra, reads[arm], forms=dict(zip(
                        codes.forms, codes.form_code)), answers=answers, iou=i) for i in (0.5, 0.3)}}
    texts = {"without": {r: top_text(None, found[r]["best"], found[r]["q"]) for r in rows_q},
             "pack": {r: top_text(found[r]["post"], found[r]["best"], found[r]["q"]) for r in rows_q}}
    w, p = res["without"]["labels"], res["pack"]["labels"]
    info = {"name": name, "quarter": quarter, "seed": seed, "preliminary": True, "boxes": str(boxes_dir),
            "pack": str(pack_file), "fit": {k: [v.mode, v.tau, v.temperature, v.w] for k, v in pack.fits.items()},
            "headline": {a: {k: res[a]["headline"][k] for k in ("n", "right", "share", "ci", "with_box")}
                         for a in res},
            "paired_pack_vs_without": measures.paired_labels(p, w),
            "labels_changed": {"right_to_wrong": int(((w["right"] == 1) & (p["right"] == 0)).sum()),
                               "wrong_to_right": int(((w["right"] == 0) & (p["right"] == 1)).sum())},
            "text_numbers_vs_terra": {str(i): text_numbers(bq, terra, texts, i) for i in (0.5, 0.3)},
            "pack_other_likeliest": float(np.mean([texts["pack"][r] is None for r in rows_q])),
            "read_numbers": {a: res[a]["read_numbers"] for a in res},
            "box_numbers": {a: {str(i): {k: res[a]["box_numbers"][i][k] for k in
                                         ("matched", "ours_unmatched_text", "ours_unmatched_none")} for i in (0.5, 0.3)}
                            for a in res},
            "git": _git("rev-parse", "HEAD"), "seconds": round(time.perf_counter() - t0, 1)}
    base = ROOT / "runs" / run_name(reader_name, quarter, seed) / "labels.parquet"
    if base.is_file():
        info["without_matches_reader_run_labels"] = bool((pd.read_parquet(base)["right"].to_numpy()
                                                          == w["right"].to_numpy()).all())
    (out / "run.json").write_text(json.dumps(info, indent=1, default=float) + "\n")
    log.info("%s: without %s, pack %s of %s (%.0f s)", out.name, res["without"]["headline"]["right"],
             res["pack"]["headline"]["right"], res["without"]["headline"]["n"], time.perf_counter() - t0)
    return out


def pack_page(name: str, quarter: str, seed: int, boxes_dir: Path | None = None, run_page: str = "pack-prelim") -> None:
    """The Mission sheet's review page of the pack's reads of the boxes as found."""
    d = OUT / run_name(name, quarter, seed)
    boxes_dir = Path(boxes_dir) if boxes_dir else paths.SOLVER_DIR / "boxes"
    boxes = pd.read_parquet(boxes_dir / "boxes.parquet").reset_index(drop=True)
    found = pickle.loads((d / "found.pkl").read_bytes())
    folder = write_pack_text_read(boxes, found, R.CodeList.load(), d / "text-read")
    subprocess.run([sys.executable, str(Path(__file__).parent / "review_page.py"), "--run", run_page, "--boxes",
                    str(boxes_dir / "boxes.parquet"), "--reads", str(folder), "--before", "baseline"], check=True,
                   cwd=config.PIPELINE_DIR)


def decide(name: str, quarter: str = "ne", seeds=(0, 1), root: Path | None = None) -> dict:
    """The keep rule's numbers for (5) on `quarter` over `seeds`: each arm's headline against reading as found, the
    paired label interval on the mean of the seeds and at each seed.  The rule's other condition (the review page agrees)
    is the controller's look.  The verdict is left to the caller: a run on today's boxes is preliminary."""
    d = [(root or OUT) / run_name(name, quarter, s) for s in seeds]
    out = {"quarter": quarter, "seeds": list(seeds), "preliminary": True}
    for arm in ("plain", "fitted"):
        a = [pd.read_parquet(x / f"labels_{arm}.parquet") for x in d]
        b = [pd.read_parquet(x / "labels_without.parquet") for x in d]
        mean = lambda ls: ls[0].assign(right=np.mean([x["right"].to_numpy(float) for x in ls], axis=0))  # noqa: E731
        out[arm] = {"mean_of_seeds": measures.paired_labels(mean(a), mean(b)),
                    "by_seed": [measures.paired_labels(x, y) for x, y in zip(a, b)],
                    "ci_excludes_zero": bool(measures.paired_labels(mean(a), mean(b))["lo"] > 0)}
    return out


def page(name: str, quarter: str, seed: int, run_page: str = "choose-prelim") -> None:
    """The Mission sheet's review page of the fitted choice's boxes (every quarter's boxes by its own fold's choice)."""
    d = OUT / run_name(name, quarter, seed)
    table = pd.read_parquet(d / "chosen_boxes.parquet")
    gots = pickle.loads((d / "chosen_gots.pkl").read_bytes())
    folder = write_text_read(table, gots, R.CodeList.load(), d / "text-read")
    subprocess.run([sys.executable, str(Path(__file__).parent / "review_page.py"), "--run", run_page, "--boxes",
                    str(d / "chosen_boxes.parquet"), "--reads", str(folder), "--before", "baseline"], check=True,
                   cwd=config.PIPELINE_DIR)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("choose")
    r.add_argument("--name", required=True)
    r.add_argument("--quarter", default="ne")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--boxes-dir", type=Path, default=None)
    r.add_argument("--reader-name", default="base")
    r.add_argument("--reader-root", type=Path, default=None)
    r.add_argument("--workers", type=int, default=WORKERS)
    d = sub.add_parser("decide")
    d.add_argument("--name", required=True)
    d.add_argument("--quarter", default="ne")
    p = sub.add_parser("page")
    p.add_argument("--name", required=True)
    p.add_argument("--quarter", default="ne")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default="choose-prelim")
    k = sub.add_parser("pack")
    k.add_argument("--name", default="pack")
    k.add_argument("--quarter", default="ne")
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--boxes-dir", type=Path, default=None)
    k.add_argument("--reader-name", default="base")
    k.add_argument("--reader-root", type=Path, default=None)
    k.add_argument("--pack", type=Path, default=None)
    k.add_argument("--workers", type=int, default=WORKERS)
    k.add_argument("--page", action="store_true", help="build the review page from an existing run")
    args = ap.parse_args()
    if args.cmd == "choose":
        print(run(args.name, args.quarter, args.seed, args.boxes_dir, args.reader_name, args.reader_root, args.workers))
    elif args.cmd == "decide":
        print(json.dumps(decide(args.name, args.quarter), indent=1, default=float))
    elif args.cmd == "page":
        page(args.name, args.quarter, args.seed, args.run)
    elif args.page:
        pack_page(args.name, args.quarter, args.seed, args.boxes_dir)
    else:
        print(run_pack(args.name, args.quarter, args.seed, args.boxes_dir, args.reader_name, args.reader_root, args.pack,
                       args.workers))


if __name__ == "__main__":
    main()
