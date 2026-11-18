"""The end-to-end measures of finding and reading words on the 2000 sheets.

The headline: read right at our own boxes, over every answered label with a code (a label whose box is missing counts
wrong, never dropped).  A label goes to the box whose grouped rectangle holds its point (`textread.read.holding_box`'s
rule), and, of several, to the one the reader is surest of (the highest closed posterior; ties: nearest centre).

Also, against Terra's word boxes (`box_numbers`, `read_numbers`), the difference of two runs (`compare`) and the paired
interval of two arms over answered labels, resampling blocks (`paired_labels`).

A reading (`reads[row]`, `row` a label of the boxes' index) is a dict: `closed` {code: probability} (the posterior of
the codes given a code, as `reads.npz`'s `log_closed` stores it) and `top` (the open hypothesis the reader likes best:
a code, "other" or "none").  `reads_from_store` builds it from `data/solver/text-read`.

Terra's words are a DataFrame: sheet, text, kind, quarter and corners c0x, c0y .. c3x, c3y (page px).
"""
import json
from collections import Counter

import cv2
import numpy as np
import pandas as pd

from sfzmaps import config
from sfzmaps.solver import grid, paths, wordboxes
from sfzmaps.solver.textnet import boxes as textboxes
from sfzmaps.solver.textread import read as R
from sfzmaps.transform import SheetTransform

SEED = 20261117
BOOT = 2000
SPLIT_PART = 0.1       # a box meets a word at least this much to be a part of it
SPLIT_WHOLE = 0.5      # ... and no box meets it this much: the word is split
G_COLS = [f"g{i}{a}" for i in range(4) for a in "xy"]
C_COLS = [f"c{i}{a}" for i in range(4) for a in "xy"]


# --- inputs from the stores -------------------------------------------------------------------------------------------

def reads_from_store(folder=None) -> dict:
    """`reads` for every box of `text-read`: {row: {"closed": {code: p}, "top": str}}."""
    folder = folder or R.store.READ_DIR
    with np.load(folder / "reads.npz") as z:
        codes, log_closed, row = [str(c) for c in z["codes"]], z["log_closed"], z["row"]
    top = pd.read_parquet(folder / "reads.parquet")["top"].tolist()
    p = np.exp(log_closed)
    return {int(r): {"closed": dict(zip(codes, p[k].tolist())), "top": top[k]} for k, r in enumerate(row)}


def answers_from_store(folder=None) -> pd.DataFrame:
    """The owner's answered labels with a code (the rows of `text-read/answered.parquet` scored, truth not ""):
    sheet, point, truth, quarter, block, terra (Terra's string)."""
    folder = folder or R.store.READ_DIR
    a = pd.read_parquet(folder / "answered.parquet")
    a = a[a["scored"] & (a["truth"] != "")]
    return a[["sheet", "point_x", "point_y", "truth", "quarter", "block", "terra"]].reset_index(drop=True)


def terra_words(sheets) -> pd.DataFrame:
    """Terra's words (every kind) on `sheets` of the 2000 series, as a DataFrame (see the module)."""
    rows = []
    for sheet in sheets:
        found = textboxes.terra_boxes(paths.SERIES, sheet)
        if not found:
            continue
        rects = np.array([textboxes.word_rect(b["box"], float(b.get("angle_deg") or 0.0)) for b in found],
                         float).reshape(-1, 4, 2)
        warp = config.DATA_DIR / "warp" / paths.SERIES / sheet
        transform = SheetTransform.from_json(json.loads((warp / "transform.json").read_text()), folder=warp)
        quarter = grid.quarter_of(transform.to_world(rects.mean(1)))
        for b, r, q in zip(found, rects, quarter):
            rows.append({"sheet": sheet, "text": b["text"], "kind": b["kind"], "quarter": int(q),
                         **dict(zip(C_COLS, r.reshape(8).tolist()))})
    return pd.DataFrame(rows, columns=["sheet", "text", "kind", "quarter", *C_COLS])


# --- the headline ----------------------------------------------------------------------------------------------------

def holding_boxes(boxes: pd.DataFrame, sheet: str, x: float, y: float) -> list:
    """The index labels of every box on `sheet` whose grouped rectangle holds the page point (edges included)."""
    mine = boxes[boxes["sheet"] == sheet]
    if not len(mine):
        return []
    g = mine[G_COLS].to_numpy(float).reshape(-1, 4, 2)
    near = np.flatnonzero(np.abs(g.mean(1) - [x, y]).max(1) < R.NEAR)
    return [mine.index[k] for k in near
            if cv2.pointPolygonTest(g[k].astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0]


def _sure(read: dict | None) -> float:
    return max(read["closed"].values()) if read and read.get("closed") else 0.0


def _closed_best(read: dict | None):
    """The likeliest code of a reading (ties: the first by name), or None."""
    c = read.get("closed") if read else None
    return min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] if c else None


def label_box(boxes: pd.DataFrame, reads: dict, sheet, x, y):
    """The box a label's point goes to (see the module), or None."""
    hits = holding_boxes(boxes, sheet, x, y)
    if not hits:
        return None
    g = boxes.loc[hits, G_COLS].to_numpy(float).reshape(-1, 4, 2).mean(1)
    dist = np.hypot(*(g - [x, y]).T)
    k = min(range(len(hits)), key=lambda i: (-_sure(reads.get(hits[i])), dist[i], i))
    return hits[k]


def label_right(boxes: pd.DataFrame, reads: dict, answers: pd.DataFrame) -> pd.DataFrame:
    """Per answered label of `answers` with a code truth: `label` (its row in `answers`), `row` (its box, or -1),
    `right` (0 or 1), `block` (its block; one of its own when none) and `quarter`."""
    out = []
    has_block = "block" in answers.columns
    for k, a in enumerate(answers.to_dict("records")):
        if not isinstance(a["truth"], str) or a["truth"] == "":
            continue
        row = label_box(boxes, reads, a["sheet"], a["point_x"], a["point_y"])
        block = a["block"] if has_block else ""
        out.append({"label": k, "row": -1 if row is None else row,
                    "right": float(row is not None and _closed_best(reads.get(row)) == a["truth"]),
                    "block": block if isinstance(block, str) and block else f"label{k}",
                    "quarter": int(a["quarter"])})
    return pd.DataFrame(out, columns=["label", "row", "right", "block", "quarter"])


def cluster_interval(right: np.ndarray, block: np.ndarray, boot: int = BOOT, seed: int = SEED):
    """The 95% interval of the share right, resampling whole blocks with replacement (seeded)."""
    if not len(right):
        return None
    names, inv = np.unique(block, return_inverse=True)
    hit = np.bincount(inv, weights=right, minlength=len(names))
    cnt = np.bincount(inv, minlength=len(names)).astype(float)
    idx = np.random.default_rng([seed, 1]).integers(0, len(names), (boot, len(names)))
    s = hit[idx].sum(1) / cnt[idx].sum(1)
    return [round(float(np.quantile(s, 0.025)), 4), round(float(np.quantile(s, 0.975)), 4)]


def _share(d: pd.DataFrame) -> dict:
    n = int(len(d))
    with_box = d["row"] >= 0
    return {"n": n, "right": int(d["right"].sum()), "share": round(float(d["right"].mean()), 4) if n else None,
            "with_box": int(with_box.sum()), "right_with_box": int(d.loc[with_box, "right"].sum())}


def headline(boxes: pd.DataFrame, reads: dict, answers: pd.DataFrame, quarters: tuple[str, ...]) -> dict:
    """Read right at our own boxes over the answered labels with a code (a label with no box counts wrong): `n`,
    `right`, `share`, `ci` (95%, resampling blocks), `ci_labels` (resampling labels), `with_box`, `right_with_box`,
    and `by_quarter` (the same by quarter name; `quarters` names the quarter numbers in order)."""
    d = label_right(boxes, reads, answers)
    out = _share(d)
    out["ci"] = cluster_interval(d["right"].to_numpy(float), d["block"].to_numpy(str))
    out["ci_labels"] = R.boot_interval(d["right"].to_numpy(float))
    out["by_quarter"] = {name: _share(d[d["quarter"] == q]) for q, name in enumerate(quarters)}
    return out


# --- the boxes against Terra's words --------------------------------------------------------------------------------

def _overlaps(boxes: pd.DataFrame, terra: pd.DataFrame):
    """(Terra position, box position, IoU) of every overlapping pair (positions in the frames as given)."""
    ti, bj, iou = [], [], []
    t_pos = np.arange(len(terra))
    b_pos = np.arange(len(boxes))
    for sheet, t in terra.groupby("sheet", sort=True):
        sel = (boxes["sheet"] == sheet).to_numpy()
        if not sel.any():
            continue
        tp = t[C_COLS].to_numpy(float).reshape(-1, 4, 2)
        bp = boxes.loc[sel, G_COLS].to_numpy(float).reshape(-1, 4, 2)
        i, j, inter = wordboxes.pairs(tp, bp)
        ta, ba = wordboxes.poly_area(tp), wordboxes.poly_area(bp)
        ti.append(t_pos[terra.index.get_indexer(t.index)][i])
        bj.append(b_pos[sel][j])
        iou.append(inter / (ta[i] + ba[j] - inter))
    if not ti:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(ti), np.concatenate(bj), np.concatenate(iou)


def _best_word(ti, bj, iou, n_boxes):
    """For every box: its best Terra word (-1: none) and the IoU with it."""
    word, best = np.full(n_boxes, -1), np.zeros(n_boxes)
    for k in np.lexsort((ti, iou)):             # ascending IoU: a box's highest is written last
        word[bj[k]], best[bj[k]] = ti[k], iou[k]
    return word, best


def geometric_reason(t: int, ti, bj, iou, best_word) -> str:
    """Why Terra word `t` has no box, from the geometry alone: "split" (two or more boxes each meet it at IoU >=
    SPLIT_PART and none at SPLIT_WHOLE), "merged" (the box meeting it best meets another word better), "none" (no
    box meets it at all) or "other"."""
    mine = ti == t
    if not mine.any():
        return "none"
    v = iou[mine]
    if (v >= SPLIT_PART).sum() >= 2 and not (v >= SPLIT_WHOLE).any():
        return "split"
    if best_word[bj[mine][int(np.argmax(v))]] != t:
        return "merged"
    return "other"


def box_numbers(boxes: pd.DataFrame, terra: pd.DataFrame, reads: dict, iou=(0.5, 0.3), reason_fn=None,
                quarters: tuple[str, ...] = paths.QUARTERS) -> dict:
    """How our boxes meet Terra's words, at each IoU of `iou` (the result's keys): `terra` words, `ours` boxes,
    `matched` (a box is matched to its best word only, and only when it meets it at the IoU: a merged box matches one
    word; a Terra word is matched when some box is), `missed` (the rest, each counted once) with `missed_reasons` (a
    Counter), `ours_unmatched_text` and `ours_unmatched_none` (boxes matched to no word, read as text or as no text),
    `ours_near_missed` (of those, the boxes that meet a missed word at IoU >= SPLIT_PART), `matched_to_missed` (the
    plain ratio matched / missed; None when none is missed), and `by_quarter` (the same
    by quarter name, from the `quarter` columns of both).

    The reasons are "split", "merged", "none" and "other" (`geometric_reason`); with `reason_fn(t, ti, bj, iou,
    best_word) -> str` (t: the word's row; ti, bj, iou: every overlapping pair; best_word: each box's best word) its
    codes are counted instead."""
    boxes = boxes.copy()
    terra = terra.copy()
    ti, bj, v = _overlaps(boxes, terra)
    nb = len(boxes)
    word, best = _best_word(ti, bj, v, nb)
    is_text = np.array([reads.get(r, {}).get("top", "none") != "none" for r in boxes.index], bool)
    fn = reason_fn or geometric_reason
    tq = terra["quarter"].to_numpy() if "quarter" in terra else None
    bq = boxes["quarter"].to_numpy() if "quarter" in boxes else None
    out = {}
    for level in iou:
        won = (best >= level) & (word >= 0)
        matched = np.zeros(len(terra), bool)
        matched[word[won]] = True
        reasons = {int(t): fn(int(t), ti, bj, v, word) for t in np.flatnonzero(~matched)}
        near_missed = np.zeros(nb, bool)
        sel = ~matched[ti] & (v >= SPLIT_PART)
        near_missed[bj[sel]] = True

        def count(wsel, bsel):
            return {"terra": int(wsel.sum()), "ours": int(bsel.sum()), "matched": int((matched & wsel).sum()),
                    "missed": int((~matched & wsel).sum()),
                    "missed_reasons": Counter(r for t, r in reasons.items() if wsel[t]),
                    "ours_unmatched_text": int((~won & bsel & is_text).sum()),
                    "ours_unmatched_none": int((~won & bsel & ~is_text).sum()),
                    "ours_near_missed": int((~won & bsel & near_missed).sum()),
                    "matched_to_missed": (round(float((matched & wsel).sum()) / float((~matched & wsel).sum()), 4)
                                          if (~matched & wsel).any() else None)}

        res = count(np.ones(len(terra), bool), np.ones(nb, bool))
        if tq is not None and bq is not None:
            res["by_quarter"] = {name: count(tq == q, bq == q) for q, name in enumerate(quarters)}
        out[level] = res
    return out


def _code_of(text, forms: dict | None):
    """The code a Terra string is a printed form of (`forms`: form -> code; None: the string itself), or None."""
    t = " ".join(str(text).replace("​", "").upper().split())
    return (t or None) if forms is None else forms.get(t)


def read_numbers(boxes: pd.DataFrame, terra: pd.DataFrame, reads: dict, forms: dict | None = None,
                 answers: pd.DataFrame | None = None, iou: float = 0.5) -> dict:
    """On our boxes matched to a Terra word at `iou` (each to its best word): `matched_boxes`; of those whose Terra
    string is a printed form (`forms`: form -> code; None: the string is the code), `same_as_terra` (our likeliest code
    is its code) and `differs`; `not_a_form` (the rest).  With `answers` (see `answers_from_store`; `terra` is Terra's
    string for the label), `codes_right_vs_answers`: of the labels whose box is matched to a Terra word, `n`, how many
    our code gets right (`ours`) and how many Terra's string does (`terra`)."""
    boxes = boxes.copy()
    ti, bj, v = _overlaps(boxes, terra)
    word, best = _best_word(ti, bj, v, len(boxes))
    same = differs = other = 0
    for pos in np.flatnonzero((best >= iou) & (word >= 0)):
        code = _code_of(terra["text"].iloc[word[pos]], forms)
        mine = _closed_best(reads.get(boxes.index[pos]))
        if code is None or mine is None:
            other += 1
        elif code == mine:
            same += 1
        else:
            differs += 1
    out = {"matched_boxes": same + differs + other, "same_as_terra": same, "differs": differs, "not_a_form": other,
           "codes_right_vs_answers": None}
    if answers is not None:
        n = ours = theirs = 0
        for r in label_right(boxes, reads, answers).itertuples(index=False):
            if r.row < 0:
                continue
            pos = boxes.index.get_loc(r.row)
            if best[pos] < iou or word[pos] < 0:
                continue
            n += 1
            ours += int(r.right)
            theirs += int(_code_of(answers["terra"].iloc[r.label], forms) == answers["truth"].iloc[r.label])
        out["codes_right_vs_answers"] = {"n": n, "ours": ours, "terra": theirs}
    return out


# --- runs against each other ----------------------------------------------------------------------------------------

def compare(now: dict, before: dict) -> dict:
    """`now` minus `before`, for every number the two share (dicts and Counters are walked; other keys are dropped)."""
    out = {}
    for k, v in now.items():
        if k not in before:
            continue
        w = before[k]
        if isinstance(v, dict) and isinstance(w, dict):
            d = compare(v, w)
            if d:
                out[k] = d
        elif isinstance(v, (int, float)) and isinstance(w, (int, float)) and not isinstance(v, bool):
            out[k] = v - w
    return out


def paired_labels(a: pd.DataFrame, b: pd.DataFrame, boot: int = BOOT, seed: int = 0) -> dict:
    """Two arms' right-or-wrong per answered label (columns `right` and `block`; the same labels in the same order) as
    a difference of shares (a minus b) with its 95% interval, resampling whole blocks: `diff`, `lo`, `hi`."""
    if len(a) != len(b) or (a["block"].to_numpy(str) != b["block"].to_numpy(str)).any():
        raise ValueError("the two arms must score the same labels, in the same blocks")
    ra, rb = a["right"].to_numpy(float), b["right"].to_numpy(float)
    names, inv = np.unique(a["block"].to_numpy(str), return_inverse=True)
    gain = np.bincount(inv, weights=ra - rb, minlength=len(names))
    cnt = np.bincount(inv, minlength=len(names)).astype(float)
    idx = np.random.default_rng([seed, 2]).integers(0, len(names), (boot, len(names)))
    s = gain[idx].sum(1) / cnt[idx].sum(1)
    return {"diff": float((ra - rb).mean()), "lo": float(np.quantile(s, 0.025)), "hi": float(np.quantile(s, 0.975))}
