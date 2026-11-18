"""Why answered code labels are read wrong (Task 8, a measurement): each wrong label sorted into one cause by Terra's
words and their ink inside the box the label's point goes to (`measures.label_box`), for a run's boxes and for today's.

A Terra word's ink is the page's ink px (`data.page_ink`) whose centres lie in its rectangle; a box HOLDS a word when
more than half of that ink lies in the box (the even-odds call: the word is more in it than out).  The label's code
word is the Terra word whose rectangle holds the label's point (none: "no Terra word at the point").

  1 no_box      no box holds the label's point
  2 merged      the box holds 2+ Terra words; split by the other word's kind (the held word other than the code word
                with the most of its ink in the box; "no code word held" when the box holds two others) and by where
                it lies from the code word: "stacked" when its centre is farther across the code word's axis than
                along it, "along" otherwise
  3 partial     the box does not hold the code word: half of its ink or less is in the box
  4 clean       the box holds the code word alone and meets its rectangle at IoU >= 0.5 (`box_numbers`' match); the
                reader got it wrong. Sub-count: Terra's own string is wrong against the owner's answer too
  5 other       the rest: a loose box (holds the code word alone, IoU < 0.5), a box holding one other word, no Terra
                word at the point

Each label also gets the largest ink share in its box of any Terra word it does not hold (how near the call is).

    uv run python scripts/wordread/label_causes.py --run crude-map-all-s0 [--examples 5]

Writes experiments/wordread/causes/: causes-<run>.json (counts per quarter and in total, the run's and today's),
labels-<run>.parquet and labels-today.parquet (each wrong label), and per cause up to `examples` crops of the run's
wrong labels (ink grey, Terra's words blue and the code word green, the box red, the label's point a cross) with
index.html."""
from __future__ import annotations

import argparse
import base64
import html
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "textlines")]
import run_detector as RD  # noqa: E402

from sfzmaps.solver import paths  # noqa: E402
from sfzmaps.solver.textnet import data  # noqa: E402
from sfzmaps.solver.wordread import measures  # noqa: E402

log = logging.getLogger(__name__)
OUT = RD.ROOT.parent / "causes"
CAUSES = ("no_box", "merged", "partial", "clean", "other")
MATCH_IOU = 0.5          # box_numbers' first level: a box matches a Terra word at IoU 0.5 (measures.box_numbers)
SUB = 4                  # fixed-point bits for the polygons' px (cv2.fillPoly's shift): 1/16 px
SCALE = 3                # the crops' display zoom (nearest px)
MARGIN = 1.0             # the crops' margin around what they show, in the code word's heights


def _mask(poly: np.ndarray, x0: int, y0: int, h: int, w: int) -> np.ndarray:
    """The px of the window (x0, y0, w, h) whose centres lie in the page polygon `poly` (4, 2), bool."""
    m = np.zeros((h, w), np.uint8)
    pts = np.round((poly - [x0 + 0.5, y0 + 0.5]) * (1 << SUB)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_8, shift=SUB)
    return m > 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    from shapely.geometry import Polygon
    pa, pb = Polygon(a).buffer(0), Polygon(b).buffer(0)
    u = pa.union(pb).area
    return float(pa.intersection(pb).area / u) if u > 0 else 0.0


def _window(polys, h: int, w: int):
    lo = np.floor(np.min([p.min(0) for p in polys], 0)).astype(int) - 1
    hi = np.ceil(np.max([p.max(0) for p in polys], 0)).astype(int) + 1
    x0, y0 = max(lo[0], 0), max(lo[1], 0)
    return x0, y0, min(hi[0], w) - x0, min(hi[1], h) - y0


def label_cause(box: np.ndarray | None, point, words: pd.DataFrame, ink: np.ndarray) -> dict:
    """One wrong label's cause (the module) from its box's corners (None: no box), its point and the sheet's Terra
    words (`measures.terra_words` rows), with the page's ink."""
    rects = words[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2)
    code = next((k for k in range(len(rects))
                 if cv2.pointPolygonTest(rects[k].astype(np.float32).reshape(-1, 1, 2),
                                         (float(point[0]), float(point[1])), False) >= 0), None)
    out = {"cause": "no_box", "detail": "", "code_kind": None if code is None else words["kind"].iloc[code],
           "code_text": None if code is None else words["text"].iloc[code], "code_share": np.nan, "code_iou": np.nan,
           "held": 0, "other_kind": None, "other_text": None, "placement": None, "largest_unheld_share": np.nan}
    if box is None:
        return out
    near = [k for k in range(len(rects))
            if rects[k].min(0)[0] < box.max(0)[0] and rects[k].max(0)[0] > box.min(0)[0]
            and rects[k].min(0)[1] < box.max(0)[1] and rects[k].max(0)[1] > box.min(0)[1]]
    shares = {}
    h, w = ink.shape
    for k in near:
        x0, y0, ww, hh = _window([rects[k]], h, w)
        if ww <= 0 or hh <= 0:
            continue
        wm = _mask(rects[k], x0, y0, hh, ww) & ink[y0:y0 + hh, x0:x0 + ww]
        if not wm.any():
            continue
        shares[k] = float((wm & _mask(box, x0, y0, hh, ww)).sum() / wm.sum())
    held = sorted((k for k, s in shares.items() if s > 0.5), key=lambda k: -shares[k])
    unheld = [s for k, s in shares.items() if s <= 0.5]
    out.update(held=len(held), largest_unheld_share=max(unheld) if unheld else 0.0)
    if code is not None:
        out["code_share"] = shares.get(code, 0.0)
        out["code_iou"] = _iou(box, rects[code])
    if len(held) >= 2:
        others = [k for k in held if k != code]
        o = others[0]
        out.update(cause="merged", other_kind=words["kind"].iloc[o], other_text=words["text"].iloc[o])
        if code is None or code not in held:
            out["detail"] = "no code word held"
        else:
            u = rects[code][1] - rects[code][0]
            u = u / max(np.hypot(*u), 1e-9)
            d = rects[o].mean(0) - rects[code].mean(0)
            along, across = abs(d @ u), abs(d @ np.array([-u[1], u[0]]))
            out["placement"] = "stacked" if across > along else "along"
            out["detail"] = f"code+{words['kind'].iloc[o]}"
        return out
    if code is None:
        out.update(cause="other", detail="no Terra word at the point")
        return out
    if out["code_share"] <= 0.5:
        out.update(cause="partial", detail="holds another word" if held else "")
        return out
    if out["code_iou"] >= MATCH_IOU:
        out.update(cause="clean")
        return out
    out.update(cause="other", detail="loose box (holds the code word alone, IoU < 0.5)")
    return out


def causes(boxes: pd.DataFrame, reads: dict, answers: pd.DataFrame, terra: pd.DataFrame, ink_fn) -> pd.DataFrame:
    """Every wrong label of `answers` with its cause (`label_cause`), its box's read and the owner's and Terra's
    strings."""
    right = measures.label_right(boxes, reads, answers)
    rows = []
    for sheet, a in answers.assign(k=np.arange(len(answers))).groupby("sheet", sort=True):
        r = right.set_index("label").loc[a["k"]]
        wrong = a[(r["right"] == 0).to_numpy()]
        if not len(wrong):
            continue
        ink = ink_fn(sheet)
        words = terra[terra["sheet"] == sheet].reset_index(drop=True)
        for lab in wrong.to_dict("records"):
            row = int(r.loc[lab["k"], "row"])
            box = None if row < 0 else boxes.loc[row, measures.G_COLS].to_numpy(float).reshape(4, 2)
            got = label_cause(box, (lab["point_x"], lab["point_y"]), words, ink)
            read = measures._closed_best(reads.get(row)) if row >= 0 else None
            rows.append({"label": lab["k"], "sheet": sheet, "quarter": paths.QUARTERS[int(lab["quarter"])],
                         "point_x": lab["point_x"], "point_y": lab["point_y"], "truth": lab["truth"],
                         "terra": lab["terra"], "terra_right": lab["terra"] == lab["truth"], "row": row,
                         "read": read, **({} if box is None else dict(zip(measures.G_COLS, box.reshape(8)))), **got})
    return pd.DataFrame(rows)


def counts(t: pd.DataFrame, n_labels: dict) -> dict:
    """The causes' counts per quarter and in total (with the labels and wrong labels each has), the merged ones by
    the other word's kind and placement, the clean ones whose Terra string is wrong too, and the others by detail."""
    def one(d: pd.DataFrame, n: int) -> dict:
        m = d[d["cause"] == "merged"]
        return {"labels": n, "wrong": len(d), **{c: int((d["cause"] == c).sum()) for c in CAUSES},
                "merged_by_kind": {k: int(v) for k, v in m["detail"].value_counts().items()},
                "merged_by_placement": {k: int(v) for k, v in m["placement"].fillna("-").value_counts().items()},
                "merged_kind_placement": {f"{k} {p}": int(v) for (k, p), v in
                                          m.groupby([m["detail"], m["placement"].fillna("-")]).size().items()},
                "clean_terra_wrong_too": int(((d["cause"] == "clean") & ~d["terra_right"]).sum()),
                "partial_holding_another_word": int(((d["cause"] == "partial")
                                                     & (d["detail"] == "holds another word")).sum()),
                "other_by_detail": {k: int(v) for k, v in d.loc[d["cause"] == "other", "detail"].value_counts()
                                    .items()}}
    out = {q: one(t[t["quarter"] == q] if len(t) else t, n_labels[q]) for q in paths.QUARTERS}
    out["all"] = one(t, sum(n_labels.values()))
    return out


def crop(t: dict, words: pd.DataFrame, ink: np.ndarray) -> bytes:
    """A PNG of one wrong label (the module's colours), SCALE times its px."""
    rects = words[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2)
    p = np.array([t["point_x"], t["point_y"]], float)
    box = (np.array([t[c] for c in measures.G_COLS], float).reshape(4, 2)
           if all(c in t and np.isfinite(t[c]) for c in measures.G_COLS) else None)
    height = 12.0
    code = [k for k in range(len(rects)) if cv2.pointPolygonTest(rects[k].astype(np.float32).reshape(-1, 1, 2),
                                                                 (float(p[0]), float(p[1])), False) >= 0]
    if code:
        height = float(np.hypot(*(rects[code[0]][3] - rects[code[0]][0])))
    shown = [p[None, :] + [[-height, -height], [height, height]]] + ([box] if box is not None else [])
    shown += [rects[k] for k in code]
    lo = np.min([s.min(0) for s in shown], 0) - MARGIN * height * 2
    hi = np.max([s.max(0) for s in shown], 0) + MARGIN * height * 2
    h, w = ink.shape
    x0, y0 = int(max(np.floor(lo[0]), 0)), int(max(np.floor(lo[1]), 0))
    x1, y1 = int(min(np.ceil(hi[0]), w)), int(min(np.ceil(hi[1]), h))
    img = np.where(ink[y0:y1, x0:x1, None], np.uint8(110), np.uint8(255)).repeat(3, 2)
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)

    def draw(poly, colour, thick):
        pts = np.round(((poly - [x0, y0]) * SCALE) * (1 << SUB)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, colour, thick, cv2.LINE_AA, shift=SUB)
    for k in range(len(rects)):
        r = rects[k]
        if r.max(0)[0] >= x0 and r.min(0)[0] <= x1 and r.max(0)[1] >= y0 and r.min(0)[1] <= y1:
            draw(r, (40, 170, 40) if k in code else (200, 120, 30), 1)
    if box is not None:
        draw(box, (30, 30, 220), 2)
    c = np.round((p - [x0, y0]) * SCALE).astype(int)
    cv2.drawMarker(img, (int(c[0]), int(c[1])), (180, 0, 180), cv2.MARKER_CROSS, 4 * SCALE, 2)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def write_examples(t: pd.DataFrame, terra: pd.DataFrame, ink_fn, n: int, out: Path, title: str, table: dict) -> Path:
    """Up to `n` crops per cause (a seeded draw) and an index.html with the counts; the page's path."""
    rng = np.random.default_rng(0)
    parts = [f"<!doctype html><meta charset=utf-8><title>Label causes</title><style>body{{font:14px sans-serif;"
             f"margin:16px;background:#fff;color:#111}}figure{{display:inline-block;margin:6px;vertical-align:top;"
             f"max-width:420px}}img{{max-width:420px;border:1px solid #ccc}}figcaption{{font-size:12px}}"
             f"table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:2px 6px;text-align:right}}"
             f"</style><h1>{html.escape(title)}</h1><p>Each wrong answered code label in one cause (see "
             f"scripts/wordread/label_causes.py). Grey: the page's ink; blue: Terra's words; green: the code word "
             f"(the Terra word at the label's point); red: the box the label's point goes to; purple cross: the "
             f"label's point. Up to {n} examples per cause, drawn at random (seed 0).</p>"]
    parts.append("<table><tr><th></th>" + "".join(f"<th>{q}</th>" for q in table) + "</tr>")
    for key in ("labels", "wrong", *CAUSES, "clean_terra_wrong_too"):
        parts.append(f"<tr><th>{key}</th>" + "".join(f"<td>{table[q][key]}</td>" for q in table) + "</tr>")
    parts.append("</table>")
    for cause in CAUSES:
        d = t[t["cause"] == cause]
        parts.append(f"<h2>{cause} ({len(d)})</h2>")
        pick = d.iloc[np.sort(rng.choice(len(d), min(n, len(d)), replace=False))] if len(d) else d
        for k, r in enumerate(pick.to_dict("records")):
            words = terra[terra["sheet"] == r["sheet"]].reset_index(drop=True)
            png = crop(r, words, ink_fn(r["sheet"]))
            f = out / cause / f"{k}-{r['sheet']}-{int(r['point_x'])}-{int(r['point_y'])}.png"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(png)
            cap = (f"{r['sheet']} {r['quarter']}: owner {r['truth']!s}, read {r['read']!s}, Terra {r['terra']!s}"
                   + (f"; {r['detail']}" if r["detail"] else "")
                   + (f" ({r['placement']})" if isinstance(r.get("placement"), str) else "")
                   + (f"; code word ink in box {r['code_share']:.2f}, IoU {r['code_iou']:.2f}"
                      if np.isfinite(r["code_share"]) else ""))
            parts.append(f"<figure><img alt='{html.escape(cause)}' src='data:image/png;base64,"
                         f"{base64.b64encode(png).decode()}'><figcaption>{html.escape(cap)}</figcaption></figure>")
    page = out / "index.html"
    page.write_text("\n".join(parts) + "\n")
    return page


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="crude-map-all-s0", help="a run folder's name under the detector's runs")
    ap.add_argument("--examples", type=int, default=5)
    a = ap.parse_args()
    from sfzmaps.solver import wordboxes
    from sfzmaps.solver.textread import store
    run = RD.RUNS / a.run
    answers = measures.answers_from_store(run / "read")
    today_answers = measures.answers_from_store(store.READ_DIR)
    cols = ["sheet", "point_x", "point_y", "truth"]
    if not answers[cols].equals(today_answers[cols]):
        raise ValueError("the run's answered labels are not today's, in the same order")
    terra = measures.terra_words(paths.SHEETS)
    inks: dict = {}

    def ink_fn(sheet):
        if sheet not in inks:
            inks[sheet] = data.page_ink(RD.load_sheet(sheet))
        return inks[sheet]
    today = wordboxes.load_boxes().reset_index(drop=True)
    today["row"] = np.arange(len(today))
    today = today.set_index("row", drop=False)
    n_labels = {q: int((answers["quarter"] == k).sum()) for k, q in enumerate(paths.QUARTERS)}
    mine = causes(RD.run_boxes(run), measures.reads_from_store(run / "read"), answers, terra, ink_fn)
    theirs = causes(today, measures.reads_from_store(store.READ_DIR), answers, terra, ink_fn)
    OUT.mkdir(parents=True, exist_ok=True)
    mine.to_parquet(OUT / f"labels-{a.run}.parquet", index=False)
    theirs.to_parquet(OUT / "labels-today.parquet", index=False)
    both = mine[["label", "cause"]].merge(theirs[["label", "cause"]], on="label", how="outer",
                                         suffixes=("_run", "_today")).fillna("right")
    moves = {f"{a_} -> {b_}": int(v) for (a_, b_), v in both.groupby(["cause_today", "cause_run"]).size().items()}
    res = {"run": a.run, "note": __doc__.split("\n\n")[1], "run_causes": counts(mine, n_labels),
           "today_causes": counts(theirs, n_labels), "today_to_run": moves}
    (OUT / f"causes-{a.run}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    page = write_examples(mine, terra, ink_fn, a.examples, OUT / a.run, f"Why labels are wrong: {a.run}",
                         res["run_causes"])
    write_examples(theirs, terra, ink_fn, a.examples, OUT / "today", "Why labels are wrong: today's boxes",
                   res["today_causes"])
    print(json.dumps({"page": str(page), "run": res["run_causes"]["all"], "today": res["today_causes"]["all"]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
