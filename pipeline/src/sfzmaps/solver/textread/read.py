"""The text reader's readings (stage `solver-text-read`, `sfzmaps solver text-read`): every word box of `solver-boxes`
read by the reader of its quarter (the one that never saw that quarter), both ways along its axis, and the measures.

A box's crop (`crops.box_crop`): its ink-trimmed rectangle along its axis, read at 0 degrees (the axis's own angle)
and at 180 (the crop turned a half turn); for trials only, blanked outside the box's own px first (`mask`).  Each
read gives log p(form | crop) for every printed form of the code list (`solver-text-crops`' prior.json `forms`: the
template reader's code list), log p(none | crop) (no text: the empty string) and p(other | crop) (every other
string), exactly (`ctc`).  The reading is their mixture, with q = p(180 degrees) the quarter's 0/180 classifier's
probability that the 0-degree crop is upside down (`weight`).  Batch 1 on the CPU, one thread per process: a crop's
numbers never depend on its batch or the machine's cores.

Writes data/solver/text-read/ (`store.READ_DIR`):
  reads.npz      one entry per row of boxes.parquet, in its order:
                   row (n,)                the box's row in boxes.parquet
                   forms (F,), form_code (F,), codes (K,)   the printed forms, each form's code, the codes (sorted)
                   log_forms (n, 2, F)     log p(form | crop) read at 0 and at 180 degrees
                   log_none (n, 2)         log p(none | crop) at 0 and 180
                   p_other (n, 2)          p(other | crop) at 0 and 180
                   p_classifier (n,)       q: the classifier's p(the 0-degree crop is upside down), the mixture's
                                           weight on the 180-degree read
                   post (n, F + 2)         the mixture: each form, then "other", then "none" (sums to 1)
                   log_closed (n, K)       log p(code | box, a code): the mixture's forms by code, renormalised
  reads.parquet  one row per box: row, sheet, i, quarter, axis, best_0 and best_180 (each read's best path),
                 p_classifier,
                 p_codes (the code list's mass), p_other, p_none, top (the likeliest of the codes, "other" and
                 "none") and p_top, closed_top and closed_p (the likeliest code given a code)
  answered.parquet  each answered label: its truth, quarter, its readings at Terra's box (`terra_*`: Terra's
                 direction, as spike S2 read it; `axis_*`: the axis alone, as our boxes are read; each weighted by
                 the classifier on its own 0-degree crop) and at the box holding its point (`box_*`; `box_row` -1
                 when no box holds it); `<prefix>p_classifier` the weight each used
  measures.json  the measures (`measures`)
"""
import json
import logging
import math
import time
from multiprocessing import get_context

import cv2
import numpy as np
import pandas as pd
import torch

from sfzmaps.solver import grid, paths, runs, wordboxes
from sfzmaps.solver.textnet import data
from sfzmaps.solver.textread import crops as K
from sfzmaps.solver.textread import ctc, model, store

log = logging.getLogger(__name__)

WORKERS = 4
FLOOR = 1e-15              # -log p is floored at 34.5 nats a label (spike S2's)
BOOT = 2000
BOOT_SEED = 20261116
NEAR = 400                 # px: boxes whose centre is this near a label's point are tested for holding it
SPIKE = {"top1_closed_terra_boxes": [0.956, 0.973], "seeds": 3, "what": "spike S2's reader C (se held out), "
         "the owner's answered labels with Terra's box, closed list, Terra's direction"}
LLM_CHAIN = {"top1": 0.969, "what": "the LLM chain (the plan's figure)"}
HELD_KINDS = ("zoning code", "block number", "street name", "other", "none")


# --- the code list --------------------------------------------------------------------------------------------------

class CodeList:
    """The printed forms (sorted), each form's code, the codes (sorted) and each form's code index."""

    def __init__(self, forms: dict[str, str]):
        self.forms = sorted(forms)
        self.form_code = [forms[f] for f in self.forms]
        self.codes = sorted(set(self.form_code))
        self.code_of = np.array([self.codes.index(c) for c in self.form_code], np.int64)

    @classmethod
    def load(cls) -> "CodeList":
        return cls(json.loads((store.CROPS_DIR / "prior.json").read_text())["forms"])


# --- reading, in worker processes -----------------------------------------------------------------------------------

_M: dict = {}


def _init() -> None:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    _M["readers"] = {q: model.load(store.TRAIN_DIR / f"{q}.pt") for q in paths.QUARTERS}
    _M["directions"] = {q: model.load(store.TRAIN_DIR / f"{q}_direction.pt", "direction") for q in paths.QUARTERS}
    _M["codes"] = CodeList.load()


def read_one(reader, direction, crop: np.ndarray, forms: list[str]) -> dict:
    """One crop read as cut and turned a half turn: each read (`ctc.read`), each best path, and the classifier's
    p(upside down) of the crop as cut and of the turned crop."""
    out = {"reads": [], "best": [], "p_classifier": []}
    with torch.inference_mode():
        for c in (crop, np.ascontiguousarray(np.rot90(crop, 2))):
            x, n = model.to_input([c])
            lp = ctc.log_softmax(reader(x)[0, :n[0]].numpy())
            out["reads"].append(ctc.read(lp, forms))
            out["best"].append(model.best_path(lp))
            out["p_classifier"].append(float(torch.sigmoid(direction(x))[0]))
    return out


def _read_job(job) -> list[dict]:
    """Read a list of (key, quarter name, crop) with each quarter's networks."""
    forms = _M["codes"].forms
    return [{"key": key, **read_one(_M["readers"][q], _M["directions"][q], crop, forms)} for key, q, crop in job]


def components(ink: np.ndarray, chance: np.ndarray, level: float, gate: np.ndarray | None = None) -> dict:
    """The components `wordboxes.group` makes boxes of at `level`: 8-connected px at or above it in the held-out
    detectors' chance and within INK_R px of ink (`gate`, made when None), holding MIN_PX ink px or more, in label
    order (a box's `i` counts them).  `lab` the label image; per kept component its `label`, `ink_px` and `box` (x0,
    y0, x1, y1: page px, ends excluded)."""
    gate = wordboxes.ink_gate(ink) if gate is None else gate
    mask = (chance >= int(np.ceil(level * 255 - 1e-9))) & gate
    _, lab = cv2.connectedComponents(mask.view(np.uint8), connectivity=8)
    ys, xs = np.nonzero(mask)
    lp = lab[ys, xs]
    order = np.argsort(lp, kind="stable")
    lp, ys, xs = lp[order], ys[order], xs[order]
    if not len(lp):
        return {"lab": lab, "label": np.zeros(0, lab.dtype), "ink_px": np.zeros(0, np.int64),
                "box": np.zeros((0, 4), np.int64)}
    start = np.flatnonzero(np.r_[True, lp[1:] != lp[:-1]])
    n_ink = np.add.reduceat(ink[ys, xs].astype(np.int64), start)
    kept = np.flatnonzero(n_ink >= wordboxes.MIN_PX)
    box = np.stack([np.minimum.reduceat(xs, start), np.minimum.reduceat(ys, start),
                    np.maximum.reduceat(xs, start) + 1, np.maximum.reduceat(ys, start) + 1], 1).astype(np.int64)
    return {"lab": lab, "label": lp[start][kept], "ink_px": n_ink[kept], "box": box[kept]}


def grown(lab: np.ndarray, labels, boxes: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """The px of the components `labels` of `lab` (`boxes` their x0, y0, x1, y1), grown by wordboxes.INK_R px (a
    disc): a bool window and its top-left page px (x0, y0)."""
    r = wordboxes.INK_R
    h, w = lab.shape
    boxes = np.asarray(boxes).reshape(-1, 4)
    x0, y0 = max(int(boxes[:, 0].min()) - r, 0), max(int(boxes[:, 1].min()) - r, 0)
    x1, y1 = min(int(boxes[:, 2].max()) + r, w), min(int(boxes[:, 3].max()) + r, h)
    own = np.isin(lab[y0:y1, x0:x1], np.asarray(labels)).view(np.uint8)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(own, disc) > 0, (x0, y0)


def word_masks(sheet: str, rows: list[dict], page: np.ndarray | None = None,
               chance: np.ndarray | None = None) -> dict:
    """Each box's own px: the component `wordboxes.group` grouped it from (`components` at its `level`, the box's
    `i`-th), grown by wordboxes.INK_R px (`grown`).  `rows`: boxes.parquet rows of `sheet` (with `row`, `i`, `level`,
    `ink_px`); `page` and `chance` read from disk when None.  {row: (bool window, (x0, y0) its top-left page px)}.
    ValueError when a box's component holds another number of ink px than the box (the boxes were grouped from other
    chance)."""
    page = data.load_page(sheet) if page is None else page
    ink = page < grid.INK_BELOW
    chance = wordboxes._read_chance(sheet, ink.shape) if chance is None else chance
    gate = wordboxes.ink_gate(ink)
    out = {}
    for level in sorted({float(x["level"]) for x in rows}):
        c = components(ink, chance, level, gate)
        for x in (x for x in rows if float(x["level"]) == level):
            i = int(x["i"])
            if i >= len(c["label"]) or int(c["ink_px"][i]) != int(x["ink_px"]):
                raise ValueError(f"{sheet}: box {x['row']} ({x['ink_px']} ink px) is not its component {i} at level "
                                 f"{level} of this chance")
            out[x["row"]] = grown(c["lab"], c["label"][i:i + 1], c["box"][i:i + 1])
    return out


def box_crops(page: np.ndarray, rows: list[dict], masks: dict | None = None) -> list[np.ndarray]:
    """The crops of boxes (boxes.parquet rows with `row`) of one page (`crops.box_crop`); with `masks`
    (`word_masks`), each paper outside its box's own px (`crops.mask_outside` of the mask cut as the box is)."""
    out = []
    for r in rows:
        corners = np.array([[r[f"t{i}x"], r[f"t{i}y"]] for i in range(4)], float)
        crop = K.box_crop(page, corners, r["t_length"], r["t_height"], r["axis"])
        if masks is not None:
            win, origin = masks[r["row"]]
            crop = K.mask_outside(crop, K.box_mask(win, corners, r["t_length"], r["t_height"], r["axis"], origin))
        out.append(crop)
    return out


def _box_job(args) -> list[dict]:
    """Every box of one sheet (rows of boxes.parquet), cut from its page (with a third item True: blanked outside its
    own px, `word_masks`) and read."""
    sheet, rows, *mask = args
    page = data.load_page(sheet)
    masks = word_masks(sheet, rows, page) if mask and mask[0] else None
    return _read_job([(r["row"], paths.QUARTERS[int(r["quarter"])], c)
                      for r, c in zip(rows, box_crops(page, rows, masks), strict=True)])


def _run(pool, fn, jobs) -> dict:
    return {r["key"]: r for part in pool.imap(fn, jobs) for r in part}


# --- posteriors -----------------------------------------------------------------------------------------------------

def weight(got: dict) -> float:
    """q, the weight on a reading's second read (the crop turned a half turn): the quarter's 0/180 classifier's
    probability that the first crop is upside down, alone."""
    return float(got["p_classifier"][0])


def combine(got: dict, q: float, codes: CodeList) -> dict:
    """A reading's mixture (`ctc.mixture`) and closed posterior (`ctc.closed_log`) at weight q on the second read."""
    reads = tuple(got["reads"])
    return {"post": ctc.mixture(q, reads), "log_closed": ctc.closed_log(q, reads, codes.code_of, len(codes.codes))}


def open_by_code(post: np.ndarray, codes: CodeList) -> dict[str, float]:
    """The mixture's mass by hypothesis: each code (its forms summed), "other", "none"."""
    by = np.zeros(len(codes.codes))
    np.add.at(by, codes.code_of, post[:-2])
    return {**dict(zip(codes.codes, by.tolist())), "other": float(post[-2]), "none": float(post[-1])}


def label_scores(open_p: dict[str, float], log_closed: np.ndarray, codes: CodeList, truth: str) -> dict:
    """One label's measures against its truth (a code, or "" for no code there): open top-1, top-3 and -log p (no
    code: "none" or "other" is right, their mass summed); closed top-1, top-3 and -log p (codes only; a "no code"
    truth has none)."""
    ranked = sorted(open_p.items(), key=lambda kv: (-kv[1], kv[0]))
    if truth == "":
        p = open_p["none"] + open_p["other"]
        o1, o3 = ranked[0][0] in ("none", "other"), any(h in ("none", "other") for h, _ in ranked[:3])
    else:
        p = open_p.get(truth, 0.0)
        o1, o3 = ranked[0][0] == truth, any(h == truth for h, _ in ranked[:3])
    out = {"open_top1": float(o1), "open_top3": float(o3), "open_nll": -math.log(max(p, FLOOR)),
           "open_best": ranked[0][0], "open_p": ranked[0][1]}
    order = np.lexsort((np.arange(len(codes.codes)), -log_closed))
    if truth in codes.codes:
        k = codes.codes.index(truth)
        out.update(closed_top1=float(order[0] == k), closed_top3=float(k in order[:3]),
                   closed_nll=-max(float(log_closed[k]), math.log(FLOOR)))
    else:
        out.update(closed_top1=np.nan, closed_top3=np.nan, closed_nll=np.nan)
    out.update(closed_best=codes.codes[order[0]], closed_p=float(np.exp(log_closed[order[0]])))
    return out


# --- matching the answered labels to our boxes ----------------------------------------------------------------------

def holding_box(boxes: pd.DataFrame, sheet: str, x: float, y: float) -> int:
    """The row of the box of `boxes` on `sheet` whose grouped rectangle holds the page point (x, y) (edges included;
    of several, the one whose centre is nearest), or -1."""
    mine = boxes[boxes["sheet"] == sheet]
    if not len(mine):
        return -1
    g = mine[[f"g{i}{a}" for i in range(4) for a in "xy"]].to_numpy(float).reshape(-1, 4, 2)
    c = g.mean(1)
    near = np.flatnonzero(np.abs(c - [x, y]).max(1) < NEAR)
    hits = [k for k in near if cv2.pointPolygonTest(g[k].astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)),
                                                   False) >= 0]
    if not hits:
        return -1
    best = min(hits, key=lambda k: (float(np.hypot(*(c[k] - [x, y]))), k))
    return int(mine.index[best])


# --- measures -------------------------------------------------------------------------------------------------------

def boot_interval(x: np.ndarray, seed: int = BOOT_SEED) -> list[float] | None:
    """The 95% interval of the mean of `x` over BOOT resamples of its entries (seeded)."""
    x = np.asarray(x, float)
    if not len(x):
        return None
    idx = np.random.default_rng(seed).integers(0, len(x), (BOOT, len(x)))
    s = x[idx].mean(1)
    return [round(float(np.quantile(s, 0.025)), 4), round(float(np.quantile(s, 0.975)), 4)]


def summarise(df: pd.DataFrame, prefix: str) -> dict:
    """top-1, top-3 and -log p (open and closed) of the rows with a reading (`<prefix>closed_best` set), with 95%
    intervals; the closed measures on code truths only."""
    out = {"n": int(len(df))}
    if not len(df):
        return out
    for kind in ("closed", "open"):
        for m in ("top1", "top3", "nll"):
            x = df[f"{prefix}{kind}_{m}"].to_numpy(float)
            x = x[np.isfinite(x)]
            out[f"{kind}_{m}"] = round(float(x.mean()), 4) if len(x) else None
            out[f"{kind}_{m}_ci"] = boot_interval(x)
        out[f"{kind}_n"] = int(np.isfinite(df[f"{prefix}{kind}_top1"].to_numpy(float)).sum())
    return out


def by_quarter(df: pd.DataFrame, fn) -> dict:
    return {"all": fn(df), **{n: fn(df[df["quarter"] == q]) for q, n in enumerate(paths.QUARTERS)}}


def measures(ans: pd.DataFrame) -> dict:
    """The measures on the owner's answered labels (rows of answered.parquet with a truth in the code list or "")."""
    scored = ans[ans["scored"]]

    def terra(d):
        t = d[d["terra_crop"] >= 0]
        return {"terra_direction": summarise(t, "terra_"), "axis_only": summarise(t, "axis_")}

    def ours(d):
        found = d[d["box_row"] >= 0]
        out = {"labels": int(len(d)), "box_found": int(len(found)), "box_missed": int(len(d) - len(found)),
               "found": summarise(found, "box_")}
        # every scored label, a missed box counted wrong for a code and right for "no code there" (closed: the code
        # truths only)
        no_code = d["truth"].to_numpy() == ""
        for kind in ("closed", "open"):
            x = np.where(d["box_row"].to_numpy() >= 0, d[f"box_{kind}_top1"].to_numpy(float), no_code.astype(float))
            if kind == "closed":
                x = x[~no_code]
            out[f"every_label_{kind}_top1"] = round(float(x.mean()), 4) if len(x) else None
            out[f"every_label_{kind}_top1_ci"] = boot_interval(x)
        both = d[(d["terra_crop"] >= 0) & (d["box_row"] >= 0) & (d["truth"] != "")]
        out["same_labels"] = {"n": int(len(both)),
                              "terra_box_axis_closed_top1": round(float(both["axis_closed_top1"].mean()), 4)
                              if len(both) else None,
                              "our_box_closed_top1": round(float(both["box_closed_top1"].mean()), 4)
                              if len(both) else None,
                              "difference_ci": boot_interval((both["box_closed_top1"] - both["axis_closed_top1"])
                                                             .to_numpy(float))}
        return out

    return {"answered": int(len(ans)), "scored": int(len(scored)),
            "no_answer": int(sum(not isinstance(a, str) for a in ans["answer"])),
            "outside_code_list": sorted({a for a, s in zip(ans["answer"], ans["scored"])
                                         if isinstance(a, str) and not s}),
            "no_code_answers": int((scored["truth"] == "").sum()), "flagged": int(scored["flagged"].sum()),
            "terra_boxes": by_quarter(scored, terra), "our_boxes": by_quarter(scored, ours),
            "unflagged": {"terra_boxes": terra(scored[~scored["flagged"]]),
                          "our_boxes": ours(scored[~scored["flagged"]])},
            "compare": {"spike_s2": SPIKE, "llm_chain": LLM_CHAIN}}


def held_out_words(rows: pd.DataFrame, got: dict, codes: CodeList) -> dict:
    """Each quarter's reader on its own quarter's words at Terra's boxes on the 2000 scan (their centre in it, touching
    no answered block; cut in Terra's direction, weighted by the classifier): by kind, the mean mass of the code list,
    of "other" and of "none"; and of the code words whose Terra string is a printed form, the closed top-1 against
    it."""
    out = {}
    for kind in HELD_KINDS:
        sel = rows[rows["kind"] == kind]
        mass = []
        hits = []
        for i in sel.index:
            g = got[("held", int(i))]
            c = combine(g, weight(g), codes)
            mass.append([c["post"][:-2].sum(), c["post"][-2], c["post"][-1]])
            text = rows.at[i, "text"]
            if kind == "zoning code" and text in codes.forms:
                k = codes.code_of[codes.forms.index(text)]
                hits.append(float(np.argmax(c["log_closed"]) == k))
        m = np.array(mass).reshape(-1, 3)
        out[kind] = {"n": int(len(sel)), "p_codes": round(float(m[:, 0].mean()), 4) if len(m) else None,
                     "p_other": round(float(m[:, 1].mean()), 4) if len(m) else None,
                     "p_none": round(float(m[:, 2].mean()), 4) if len(m) else None}
        if kind == "zoning code":
            out[kind]["in_code_list"] = len(hits)
            out[kind]["closed_top1_vs_terra"] = round(float(np.mean(hits)), 4) if hits else None
    return out


# --- the stage ------------------------------------------------------------------------------------------------------

def build_read(workers: int = WORKERS, mask: bool = False) -> dict:
    """Read every box and the answered labels, measure, and write READ_DIR (see the module).  `mask` (for trials, off
    in the stage): each box's crop blanked outside its own px (`word_masks`, which reads the held-out detectors'
    chance) before it is read."""
    t0 = time.perf_counter()
    codes = CodeList.load()
    boxes = wordboxes.load_boxes().reset_index(drop=True)
    boxes["row"] = np.arange(len(boxes))
    ans = pd.read_parquet(store.CROPS_DIR / "answered.parquet")
    ans_crops = store.Crops.load(store.CROPS_DIR / "answered.npz")
    rows = pd.read_parquet(store.CROPS_DIR / "crops.parquet")
    real = store.Crops.load(store.CROPS_DIR / "crops.npz")
    held = rows[(rows["series"] == paths.SERIES) & ~rows["touch_blocks"].astype(bool)
                & rows["kind"].isin(HELD_KINDS)]
    jobs_boxes = [(s, part.to_dict("records"), mask) for s, part in boxes.groupby("sheet", sort=True)]
    ans_job = [(("ans", int(k)), paths.QUARTERS[int(q)], ans_crops[int(c)])
               for k, (q, c) in enumerate(zip(ans["quarter"], ans["crop"])) if c >= 0]
    held_job = [(("held", int(i)), paths.QUARTERS[int(q)], real[int(i)]) for i, q in zip(held.index, held["quarter"])]
    chunks = [ans_job[k:k + 200] for k in range(0, len(ans_job), 200)] + \
        [held_job[k:k + 500] for k in range(0, len(held_job), 500)]
    with get_context("fork").Pool(workers, initializer=_init) as pool:
        got = _run(pool, _box_job, jobs_boxes)
        got.update(_run(pool, _read_job, chunks))
    log.info("%d boxes, %d answered crops and %d held-out words read in %.0f s", len(boxes), len(ans_job),
             len(held_job), time.perf_counter() - t0)

    # every box
    n, F = len(boxes), len(codes.forms)
    log_forms, log_none, p_other = np.zeros((n, 2, F)), np.zeros((n, 2)), np.zeros((n, 2))
    p_cls = np.zeros(n)
    post, log_closed = np.zeros((n, F + 2)), np.zeros((n, len(codes.codes)))
    summary = []
    for k, r in enumerate(boxes.itertuples(index=False)):
        g = got[k]
        q = weight(g)
        c = combine(g, q, codes)
        for d in (0, 1):
            log_forms[k, d], log_none[k, d], p_other[k, d] = g["reads"][d]
        p_cls[k] = q
        post[k], log_closed[k] = c["post"], c["log_closed"]
        op = open_by_code(c["post"], codes)
        top = min(op.items(), key=lambda kv: (-kv[1], kv[0]))
        ct = int(np.argmax(c["log_closed"]))
        summary.append({"row": k, "sheet": r.sheet, "i": int(r.i), "quarter": int(r.quarter), "axis": float(r.axis),
                        "best_0": g["best"][0], "best_180": g["best"][1], "p_classifier": q,
                        "p_codes": float(c["post"][:-2].sum()), "p_other": float(c["post"][-2]),
                        "p_none": float(c["post"][-1]), "top": top[0], "p_top": float(top[1]),
                        "closed_top": codes.codes[ct], "closed_p": float(np.exp(c["log_closed"][ct]))})

    # the answered labels
    ans = ans.copy()
    ans["truth"] = ans["answer"]               # a code, "" (no code there) or missing (no answer)
    ans["scored"] = [isinstance(a, str) and (a == "" or a in codes.codes) for a in ans["answer"]]
    ans["terra_crop"] = ans["crop"]
    ans["box_row"] = [holding_box(boxes, s, x, y) for s, x, y in zip(ans["sheet"], ans["point_x"], ans["point_y"])]
    cols: dict[str, list] = {}
    for k, a in enumerate(ans.itertuples(index=False)):
        truth = a.truth if a.scored else None
        found = {}
        if a.terra_crop >= 0:
            g = got[("ans", k)]
            # Terra's direction (spike S2's crop): the crop as cut first, weighted by the classifier on it
            found["terra_"] = (combine(g, weight(g), codes), weight(g))
            # the axis alone: the 0-degree crop is the axis's, Terra's crop or its half turn, the classifier on it
            _, turned = ctc.axis_and_turn(a.angle)
            g_axis = g if not turned else {"reads": g["reads"][::-1], "p_classifier": g["p_classifier"][::-1]}
            found["axis_"] = (combine(g_axis, weight(g_axis), codes), weight(g_axis))
        if a.box_row >= 0:
            found["box_"] = ({"post": post[a.box_row], "log_closed": log_closed[a.box_row]}, p_cls[a.box_row])
        for prefix in ("terra_", "axis_", "box_"):
            if prefix in found and truth is not None:
                c, q = found[prefix]
                s = label_scores(open_by_code(c["post"], codes), c["log_closed"], codes, truth)
                s["p_classifier"] = q
            else:
                s = {}
            for key in ("open_top1", "open_top3", "open_nll", "open_best", "open_p", "closed_top1", "closed_top3",
                        "closed_nll", "closed_best", "closed_p", "p_classifier"):
                cols.setdefault(prefix + key, []).append(s.get(key, None if key.endswith("best") else np.nan))
    for key, v in cols.items():
        ans[key] = v
    got_measures = measures(ans)
    got_measures["held_out_quarter_words"] = {
        qn: held_out_words(held[held["quarter"] == q], got, codes)
        for q, qn in enumerate(paths.QUARTERS)}
    got_measures["boxes"] = {"n": int(n), "top": pd.Series([s["top"] for s in summary]).value_counts()
                             .head(15).to_dict(),
                             "mean_p_classifier": round(float(p_cls.mean()), 4)}
    part = runs.fresh_folder(store.READ_DIR)
    store.save_npz(part / "reads.npz", row=boxes["row"].to_numpy(np.int64), forms=np.array(codes.forms),
                   form_code=np.array(codes.form_code), codes=np.array(codes.codes), log_forms=log_forms,
                   log_none=log_none, p_other=p_other, p_classifier=p_cls,
                   post=post, log_closed=log_closed)
    pd.DataFrame(summary).to_parquet(part / "reads.parquet", index=False)
    ans.drop(columns=["terra_crop"]).to_parquet(part / "answered.parquet", index=False)
    (part / "measures.json").write_text(json.dumps(got_measures, indent=1, default=float) + "\n")
    runs.replace_folder(part, store.READ_DIR)
    return {**got_measures, "seconds": round(time.perf_counter() - t0, 1)}
