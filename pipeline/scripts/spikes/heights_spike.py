"""Heights spike: our zoning-trained text detector + word boxes + text reader on 2000 HEIGHT sheets, no retraining.

A throwaway measurement, not shipped code.  From pipeline/:

    uv run python scripts/spikes/heights_spike.py predict          # the detector's chance (one GPU turn), PNGs
    scripts/spikes/heights_spike.sh read                           # boxes, crops, model C readings, the page
    scripts/spikes/heights_spike.sh grammar                        # the grammar-constrained reading of matched boxes
    uv run python scripts/spikes/heights_spike.py control          # block numbers: the in-distribution control

SFZ_HEIGHTS_TIGHT=0 crops the S3 box as it is (the first run, kept as loose_* in OUT); the default trims it to its ink.

`predict`: the `solver-text` detector (data/solver/text/model.pt) as it stands on SHEETS' page.png (ink: page <
grid.INK_BELOW, as `textnet.stage`), written to OUT/chance/<sheet>.png.
`read` (docTR overlay): spike S3's word boxes at its level I0.80 (chance >= 0.8, px within 4 px of ink, components of
>= 20 ink px: `s3_boxes.group`, the speck-run measure) and, for comparison, plain L0.80; each box cropped along its axis
as spike S2 cuts a word (`s2_ctc/crops.rectify`'s padding, 32 px tall) and read by S2's model C both ways along the
axis (0 and 180 degrees): the direction with the higher best-path (greedy) probability is kept.  Also the closed-list
top reading over S2's zoning code forms (meaningless for heights; kept because it is free).  The reference: the saved
LLM height reads (data/work/2000-digital-height/<sheet>/district_labels.json: code and page point), used only to score.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")

import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sfzmaps import config  # noqa: E402
from sfzmaps.solver import grid, paths  # noqa: E402
from sfzmaps.solver.codereader import labels as L  # noqa: E402
from sfzmaps.solver.textnet import data  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "s2_ctc"))
import s3_boxes as S3  # noqa: E402

cv2.setNumThreads(1)
SERIES = "2000-digital-height"
SHEETS = ("2000-digital-height-p01", "2000-digital-height-p07", "2000-digital-height-p11")
KIND = {"2000-digital-height-p01": "dense downtown", "2000-digital-height-p07": "mixed",
        "2000-digital-height-p11": "residential"}
OUT = config.DATA_DIR / "solver" / "experiments" / "spike-heights"
VARIANTS = ("I0.80", "L0.80")
MAIN = "I0.80"
PAD_X, PAD_Y = 0.4, 0.2       # word heights of page around the box (as s2_ctc/crops.py)
CROP_H = 32
MAX_WIDTH = 1024
TIGHT = os.environ.get("SFZ_HEIGHTS_TIGHT", "1") == "1"   # trim each box to its ink before cropping
MATCH_R = 30                  # px: a saved height point belongs to a box within this
HEIGHT_RE = re.compile(r"^(\d{2,3}(-\d{2,3})?-?[A-Z](-\d)?|OS)$")
TITLE = ("Heights spike: our zoning-trained detector + text reader on 2000 height sheets, no retraining. "
         "Free reading / prob / saved LLM height code.")


def norm(s: str) -> str:
    return re.sub(r"[\s\-.]", "", s.upper())


# --- the detector ---------------------------------------------------------------------------------------------------

def predict() -> None:
    from sfzmaps import gpu
    from sfzmaps.solver.textnet import predict as pr
    from sfzmaps.solver.textnet import train
    device = train.default_device()
    (OUT / "chance").mkdir(parents=True, exist_ok=True)
    with gpu.gpu_turn(True, device):
        train.settle_torch()
        net = pr.load_model(paths.TEXT_MODEL, device)
        for sheet in SHEETS:
            ink = data.load_page(sheet, SERIES) < grid.INK_BELOW
            cv2.imwrite(str(OUT / "chance" / f"{sheet}.png"), pr.chance(net, ink, device))
            print(sheet, flush=True)


# --- boxes, crops, readings -----------------------------------------------------------------------------------------

def boxes(sheet: str, page: np.ndarray, variant: str) -> dict:
    chance = cv2.imread(str(OUT / "chance" / f"{sheet}.png"), cv2.IMREAD_GRAYSCALE)
    level = float(variant[1:])
    if variant[0] == "I":
        ink = page < grid.INK_BELOW
        k = 2 * S3.INK_R + 1
        gate = cv2.dilate(ink.view(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).view(bool)
        return S3.group(chance, level, gate=gate, min_px=S3.MIN_PX, ink=ink)
    return S3.group(chance, level)


def crop_of(page: np.ndarray, corners: np.ndarray, axis: float, length: float, height: float) -> np.ndarray:
    """The box cut along its axis (reading left to right at `axis`), padded as S2 pads a word, CROP_H px tall."""
    cx, cy = corners.mean(0) - 0.5
    if TIGHT:   # the box trimmed to the ink inside it (the chance spreads past the ink: S2 learned ink-tight boxes)
        bw, bh = max(2, round(length)), max(2, round(height))
        a0 = L.frame((cx, cy), bw, bh, axis)
        ink = L.sample(page, a0, (bh, bw), 255) < grid.INK_BELOW
        rows, cols = np.flatnonzero(ink.any(1)), np.flatnonzero(ink.any(0))
        if len(rows) and len(cols):
            u0, u1, v0, v1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
            cx, cy = a0 @ np.array([(u0 + u1 - 1) / 2, (v0 + v1 - 1) / 2, 1.0])
            length, height = float(u1 - u0), float(v1 - v0)
    height = max(height, 4.0)
    w = max(4, round(length + 2 * PAD_X * height))
    h = max(4, round(height * (1 + 2 * PAD_Y)))
    a = L.frame((cx, cy), w, h, axis)
    raw = L.sample(page, a, (h, w), 255)
    out_w = min(MAX_WIDTH, max(4, round(w * CROP_H / h)))
    return cv2.resize(raw, (out_w, CROP_H), interpolation=cv2.INTER_AREA if h > CROP_H else cv2.INTER_LINEAR)


def saved_points(sheet: str) -> list[dict]:
    return json.loads((config.WORK_DIR / SERIES / sheet / "district_labels.json").read_text())["labels"]


def near_box(corners: np.ndarray, x: float, y: float) -> float:
    """Signed distance of a point to a box (positive inside)."""
    return cv2.pointPolygonTest(corners.astype(np.float32).reshape(-1, 1, 2), (float(x), float(y)), True)


def read_sheet(job) -> dict:
    sheet, variant = job
    import torch
    import torch.nn.functional as F
    import common as C
    import infer as I
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = I.load("C")
    forms = json.loads((C.INPUTS / "prior3.json").read_text())["forms"]
    page = data.load_page(sheet, SERIES)
    G = boxes(sheet, page, variant)
    rows, crops = [], []
    with torch.inference_mode():
        for i in range(len(G["axis"])):
            crop = crop_of(page, G["corners"][i], G["axis"][i], G["length"][i], G["height"][i])
            both = [crop, np.ascontiguousarray(np.rot90(crop, 2))]
            best = None
            for d, c in zip((0, 180), both):
                x, n = C.to_input([c])
                lp = F.log_softmax(C.forward(model, x)[0, :n[0]].double(), -1)
                bp = float(lp.max(-1).values.sum())
                if best is None or bp > best[0]:
                    best = (bp, d, lp, c)
            bp, d, lp, c = best
            text = C.best_path(lp)
            p_str = float(np.exp(C.string_log_probs(lp, [text])[0]))
            post = C.code_posterior(lp, forms)
            top = max(((k, v) for k, v in post.items() if k not in ("none", "other")), key=lambda kv: kv[1])
            rows.append({"sheet": sheet, "variant": variant, "i": i, "text": text, "p_path": math.exp(bp),
                         "p_str": p_str, "dir": d, "axis": float(G["axis"][i]), "length": float(G["length"][i]),
                         "height": float(G["height"][i]), "cx": float(G["corners"][i][:, 0].mean()),
                         "cy": float(G["corners"][i][:, 1].mean()), "closed_top": top[0], "closed_p": top[1],
                         "p_none": post["none"], "p_other": post["other"]})
            crops.append(c)
    # the saved height points: each to the nearest box within MATCH_R (box centre distance breaks ties)
    pts = saved_points(sheet)
    for r in rows:
        r["saved"] = ""
    matched = []
    for p in pts:
        best, bi = None, -1
        for r in rows:
            if abs(r["cx"] - p["x"]) > 400 or abs(r["cy"] - p["y"]) > 400:
                continue
            s = near_box(G["corners"][r["i"]], p["x"], p["y"])
            if s >= -MATCH_R:
                key = (-min(s, 0.0), math.hypot(r["cx"] - p["x"], r["cy"] - p["y"]))
                if best is None or key < best:
                    best, bi = key, r["i"]
        matched.append({"code": p["code"], "x": p["x"], "y": p["y"], "box": bi})
        if bi >= 0:
            rows[bi]["saved"] = rows[bi]["saved"] + ("|" if rows[bi]["saved"] else "") + p["code"]
    return {"sheet": sheet, "variant": variant, "rows": rows, "crops": crops, "points": matched,
            "corners": G["corners"]}


def read() -> None:
    jobs = [(s, v) for v in VARIANTS for s in SHEETS]
    with Pool(4) as pool:
        res = pool.map(read_sheet, jobs)
    with open(OUT / "readings.pkl", "wb") as f:
        pickle.dump(res, f)
    report(res)


# --- the page and results.md ----------------------------------------------------------------------------------------

def stats(res: list[dict], variant: str) -> dict:
    out = {}
    for r in res:
        if r["variant"] != variant:
            continue
        rows, pts = r["rows"], r["points"]
        hl = [x for x in rows if HEIGHT_RE.match(x["text"])]
        found = [p for p in pts if p["box"] >= 0]
        exact = [p for p in found if rows[p["box"]]["text"] == p["code"]]
        loose = [p for p in found if norm(rows[p["box"]]["text"]) == norm(p["code"])]
        out[r["sheet"]] = {"boxes": len(rows), "height_like": len(hl), "points": len(pts), "with_box": len(found),
                           "exact": len(exact), "loose": len(loose),
                           "boxes_on_points": len({p["box"] for p in found}),
                           "height_like_on_points": sum(bool(HEIGHT_RE.match(rows[b]["text"]))
                                                        for b in {p["box"] for p in found})}
    tot = {k: sum(v[k] for v in out.values()) for k in next(iter(out.values()))}
    out["all"] = tot
    return out


def contact(res: list[dict]) -> None:
    import common as C
    rng = np.random.default_rng(20261116)
    picks = []
    for r in res:
        if r["variant"] != MAIN:
            continue
        rows = r["rows"]
        on = [x["i"] for x in rows if x["saved"]]
        hl = [x["i"] for x in rows if HEIGHT_RE.match(x["text"]) and not x["saved"]]
        rest = [x["i"] for x in rows if not x["saved"] and not HEIGHT_RE.match(x["text"]) and x["text"]]
        hi = [i for i in rest if rows[i]["p_path"] >= 0.5]
        lo = [i for i in rest if rows[i]["p_path"] < 0.5]
        want = [(on, 18), (hl, 6), (hi, 6), (lo, 6)]
        for pool, k in want:
            if pool:
                for i in rng.choice(pool, min(k, len(pool)), replace=False):
                    picks.append((r, int(i)))
    crops, caps = [], []
    for r, i in picks:
        x = r["rows"][i]
        crops.append(r["crops"][i])
        sv = x["saved"] or "-"
        ok = ("=" if norm(x["text"]) == norm(sv.split("|")[0]) else "x") if x["saved"] else ""
        caps.append(f"{x['text'] or '(empty)'} / {x['p_path']:.2f} / {sv} {ok}\n"
                    f"{r['sheet'][-3:]} dir{x['dir']} ax{x['axis']:.0f} closed:{x['closed_top']}")
    C.contact_sheet(crops, caps, OUT / "contact.png", cols=4, scale=2, cell_w=330, title=TITLE[:110])
    # the title in full on two lines (cv2 draws one line)
    img = cv2.imread(str(OUT / "contact.png"), cv2.IMREAD_GRAYSCALE)
    head = np.full((44, img.shape[1]), 255, np.uint8)
    cv2.putText(head, TITLE[:95], (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
    cv2.putText(head, TITLE[95:], (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
    img = np.vstack([head, img[30:]])
    cv2.imwrite(str(OUT / "contact.png"), img)
    print(f"contact: {len(crops)} crops", flush=True)


def report(res: list[dict]) -> None:
    contact(res)
    lines = ["# Heights spike results", "",
             "Detector (`solver-text` model.pt, trained on zoning sheets) + S3 word boxes + S2 model C, no retraining, "
             "on three 2000 height sheets.  Saved LLM height reads (district_labels.json) used only to score.", ""]
    for v in VARIANTS:
        st = stats(res, v)
        lines += [f"## {v}{' (main)' if v == MAIN else ''}", "",
                  "| sheet | kind | boxes | height-like readings | saved height labels | with a box | free = saved "
                  "(exact) | free = saved (ignoring - and spaces) |", "|---|---|---|---|---|---|---|---|"]
        for s, d in st.items():
            pct = lambda a, b: f"{a} ({100 * a / b:.0f}%)" if b else str(a)  # noqa: E731
            lines.append(f"| {s} | {KIND.get(s, '')} | {d['boxes']} | {pct(d['height_like'], d['boxes'])} | "
                         f"{d['points']} | {pct(d['with_box'], d['points'])} | {pct(d['exact'], d['points'])} | "
                         f"{pct(d['loose'], d['points'])} |")
        a = st["all"]
        lines += ["", f"Boxes holding a saved label: {a['boxes_on_points']}, of which height-like reading: "
                      f"{a['height_like_on_points']}.", ""]
    (OUT / "results_auto.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def grammar_strings() -> list[str]:
    """A generic height grammar (no saved answer in it): every number 10-499, a letter, with or without the hyphen;
    and OS."""
    out = ["OS"]
    for n in range(10, 500):
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            out += [f"{n}-{c}", f"{n}{c}"]
    return out


def grammar_sheet(job) -> list[dict]:
    """The grammar-constrained reading of every box holding a saved height point: both directions along its axis, the
    string of highest CTC probability over `grammar_strings`; its share of the grammar's mass and the grammar's mass."""
    sheet, rows, crops = job
    import torch
    import common as C
    import infer as I
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    import torch.nn.functional as F
    model = I.load("C")
    strings = grammar_strings()
    out = []
    with torch.inference_mode():
        for x, crop in zip(rows, crops):
            best = None
            for d, c in ((x["dir"], crop), ((x["dir"] + 180) % 360, np.ascontiguousarray(np.rot90(crop, 2)))):
                t, n = C.to_input([c])
                lp = F.log_softmax(C.forward(model, t)[0, :n[0]].double(), -1)
                ls = np.concatenate([C.string_log_probs(lp, strings[k:k + 4000]) for k in range(0, len(strings), 4000)])
                k = int(np.argmax(ls))
                mass = float(np.exp(ls).sum())
                if best is None or ls[k] > best[0]:
                    best = (float(ls[k]), d, strings[k], float(np.exp(ls[k])) / max(mass, 1e-300), mass)
            out.append({"sheet": sheet, "i": x["i"], "saved": x["saved"], "free": x["text"], "g_text": best[2],
                        "g_dir": best[1], "g_share": best[3], "g_mass": best[4]})
    return out


def grammar() -> None:
    with open(OUT / "readings.pkl", "rb") as f:
        res = pickle.load(f)
    jobs = []
    for r in res:
        if r["variant"] != MAIN:
            continue
        idx = [x["i"] for x in r["rows"] if x["saved"]]
        jobs.append((r["sheet"], [r["rows"][i] for i in idx], [r["crops"][i] for i in idx]))
    with Pool(3) as pool:
        got = [g for part in pool.map(grammar_sheet, jobs) for g in part]
    with open(OUT / "grammar.pkl", "wb") as f:
        pickle.dump(got, f)
    lines = ["## Grammar-constrained reading (boxes holding a saved height point, I0.80)", "",
             "Model C's CTC probability over a generic height grammar (any number 10-499, hyphen optional, one letter; "
             "and OS: 25,481 strings, no saved answer in it), both directions, best string kept.  A box holding two "
             "saved points counts as right when it equals either.", "",
             "| sheet | boxes | grammar top = saved (ignoring -) | free reading = saved |", "|---|---|---|---|"]
    for s in SHEETS + ("all",):
        g = [x for x in got if s == "all" or x["sheet"] == s]
        ok = sum(any(norm(x["g_text"]) == norm(c) for c in x["saved"].split("|")) for x in g)
        fr = sum(any(norm(x["free"]) == norm(c) for c in x["saved"].split("|")) for x in g)
        lines.append(f"| {s} | {len(g)} | {ok} ({100 * ok / max(len(g), 1):.0f}%) | {fr} |")
    # by kind of code, and by the grammar share (calibration)
    kinds = {"OS": lambda c: c == "OS", "40-X": lambda c: c == "40-X", "other": lambda c: c not in ("OS", "40-X")}
    lines += ["", "| saved code | boxes | grammar top right |", "|---|---|---|"]
    for k, f in kinds.items():
        g = [x for x in got if f(x["saved"].split("|")[0])]
        ok = sum(any(norm(x["g_text"]) == norm(c) for c in x["saved"].split("|")) for x in g)
        lines.append(f"| {k} | {len(g)} | {ok} ({100 * ok / max(len(g), 1):.0f}%) |")
    lines += ["", "| grammar top's share of the grammar mass | boxes | right |", "|---|---|---|"]
    for lo, hi in ((0.9, 1.01), (0.5, 0.9), (0.0, 0.5)):
        g = [x for x in got if lo <= x["g_share"] < hi]
        ok = sum(any(norm(x["g_text"]) == norm(c) for c in x["saved"].split("|")) for x in g)
        lines.append(f"| {lo:.1f}-{min(hi, 1):.1f} | {len(g)} | {ok} ({100 * ok / max(len(g), 1):.0f}%) |")
    wrong = collections_counter((x["saved"].split("|")[0], x["g_text"]) for x in got
                                if not any(norm(x["g_text"]) == norm(c) for c in x["saved"].split("|")))
    lines += ["", "Most common wrong grammar readings (saved -> read): "
              + ", ".join(f"{a} -> {b} ({n})" for (a, b), n in wrong.most_common(12))]
    (OUT / "results_grammar.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def collections_counter(it):
    import collections
    return collections.Counter(it)


def control(readings: str = "readings.pkl") -> None:
    """The in-distribution control: the free reading of the box centred within 15 px of each saved block-number point
    (labels.json, LLM-read; scoring only) against its saved text."""
    with open(OUT / readings, "rb") as f:
        res = pickle.load(f)
    for r in res:
        if r["variant"] != MAIN:
            continue
        labs = json.loads((config.WORK_DIR / SERIES / r["sheet"] / "labels.json").read_text())["labels"]
        bn = [lab for lab in labs if lab["kind"] == "block_number"]
        rows = r["rows"]
        ctr = np.array([[x["cx"], x["cy"]] for x in rows])
        found = ok = 0
        for lab in bn:
            d = np.hypot(ctr[:, 0] - lab["x"], ctr[:, 1] - lab["y"])
            i = int(np.argmin(d))
            if d[i] < 15:
                found += 1
                ok += rows[i]["text"] == lab["text"].strip()
        print(r["sheet"], "block numbers", len(bn), "box centre within 15 px", found, "free == saved", ok)


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "grammar":
        grammar()
        return
    if cmd == "control":
        control(*sys.argv[2:])
        return
    if cmd == "predict":
        predict()
    elif cmd == "read":
        read()
    elif cmd == "report":
        with open(OUT / "readings.pkl", "rb") as f:
            report(pickle.load(f))
    else:
        raise SystemExit(f"unknown {cmd}")


if __name__ == "__main__":
    main()
