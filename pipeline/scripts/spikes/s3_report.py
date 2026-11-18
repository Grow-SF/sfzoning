"""Spike S3's report: the tables (results.md, results.json), the curve plot and the crop pages, from `s3_boxes.py
measure`'s per_sheet.pkl and keep_L<MID>.pkl.  Findings are the person's text in findings.md, appended last."""
import json
import pickle

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import s3_boxes as S
from sfzmaps import config

OUT = S.OUT
KIND_ALL = ("all",) + S.KINDS
RNG_SEED = 0


def load():
    with open(OUT / "per_sheet.pkl", "rb") as f:
        per = pickle.load(f)
    f = OUT / "per_sheet_ink.pkl"
    if f.is_file():
        with open(f, "rb") as fh:
            for k, v in pickle.load(fh).items():
                per[k].update(v)
    f = OUT / "per_sheet_speck.pkl"
    if f.is_file():
        with open(f, "rb") as fh:
            for k, v in pickle.load(fh).items():
                per[k].update(v)
    return per


def all_variants() -> list[str]:
    return (S.variants() + (S.ink_variants() if (OUT / "per_sheet_ink.pkl").is_file() else [])
            + (S.speck_variants() if (OUT / "per_sheet_speck.pkl").is_file() else []))


def group_of(key: str) -> str:
    series = key.split("/")[0]
    return next(g for g, s in S.GROUPS if s == series)


def kind_sel(r: dict, kind: str) -> np.ndarray:
    return np.ones(len(r["tkind"]), bool) if kind == "all" else r["tkind"] == S.KINDS.index(kind)


# --- statistics: bootstrap over sheets -------------------------------------------------------------------------------

class Boot:
    """Sheet-cluster bootstrap: the sheets are resampled with replacement (BOOT times, fixed seed)."""

    def __init__(self, n: int):
        self.idx = np.random.default_rng(RNG_SEED).integers(0, n, (S.BOOT, n))

    def ratio(self, num: np.ndarray, den: np.ndarray):
        """value and 95% interval of sum(num) / sum(den) over sheets."""
        if den.sum() == 0:
            return None, None, None
        d = den[self.idx].sum(1)
        ok = d > 0
        b = num[self.idx].sum(1)[ok] / d[ok]
        return float(num.sum() / den.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    def quantile(self, arrays: list[np.ndarray], q: float):
        whole = np.concatenate(arrays) if arrays else np.zeros(0)
        if not len(whole):
            return None, None, None
        out = []
        for row in self.idx:
            c = np.concatenate([arrays[i] for i in row])
            if len(c):
                out.append(np.percentile(c, q))
        return float(np.percentile(whole, q)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def fmt(t, pct=True, d=1):
    v, lo, hi = t
    if v is None:
        return "n/a"
    s = 100 if pct else 1
    return f"{v * s:.{d}f} [{lo * s:.{d}f}, {hi * s:.{d}f}]"


# --- the numbers ---------------------------------------------------------------------------------------------------

def compute(per: dict) -> dict:
    keys = list(per)
    boot = Boot(len(keys))
    groups = [group_of(k) for k in keys]
    res = {"sheets": len(keys), "bootstrap": f"{S.BOOT} resamples of the sheets, 95% percentile interval",
           "variants": {}, "per_group": {}, "sol": {}, "scope": {}}
    for g in [g for g, _ in S.GROUPS]:
        res["scope"][g] = {"sheets": groups.count(g),
                           "terra_words": {k: int(sum(kind_sel(per[x]["L0.50"], k).sum() for x, gg in zip(keys, groups)
                                                      if gg == g)) for k in KIND_ALL},
                           "scope_px": int(sum(per[x]["n_scope_px"] for x, gg in zip(keys, groups) if gg == g))}
    for v in all_variants():
        rs = [per[k][v] for k in keys]
        o = {"kinds": {}}
        for kind in KIND_ALL:
            sel = [kind_sel(r, kind) for r in rs]
            n = np.array([s.sum() for s in sel], float)
            best = [r["tbest"][s] for r, s in zip(rs, sel)]
            err = [r["terr"][s] for r, s in zip(rs, sel)]
            k = {"n": int(n.sum()), "recall": {}, "axis": {}, "usable": {}, "split": None, "in_merged": None}
            for thr in S.IOUS:
                hit = np.array([(b >= thr).sum() for b in best], float)
                k["recall"][f"{thr:.1f}"] = boot.ratio(hit, n)
            for thr in (0.3, 0.5):
                m = [e[b >= thr] for b, e in zip(best, err)]
                k["axis"][f"{thr:.1f}"] = {"median": boot.quantile(m, 50), "p90": boot.quantile(m, 90),
                                           "n": int(sum(len(x) for x in m))}
            # orthogonal flips: matched (IoU 0.3) words whose axis is a quarter turn off (error >= 75 deg), as the
            # axis of a vertical stack of upright letters is its elongation, not its reading direction
            m3 = [(b >= S.MAIN_IOU) for b in best]
            k["flip"] = boot.ratio(np.array([((e >= 75) & m).sum() for e, m in zip(err, m3)], float),
                                   np.array([m.sum() for m in m3], float))
            k["axis_noflip"] = {"median": boot.quantile([e[m & (e < 75)] for e, m in zip(err, m3)], 50),
                                "p90": boot.quantile([e[m & (e < 75)] for e, m in zip(err, m3)], 90)}
            for lim in (5, 10, 20):      # found at IoU 0.3 with an axis within lim degrees, of ALL the kind's words
                ok = np.array([((b >= S.MAIN_IOU) & (e <= lim)).sum() for b, e in zip(best, err)], float)
                k["usable"][str(lim)] = boot.ratio(ok, n)
            k["split"] = boot.ratio(np.array([r["tsplit"][s].sum() for r, s in zip(rs, sel)], float), n)
            k["in_merged"] = boot.ratio(np.array([r["tmerged"][s].sum() for r, s in zip(rs, sel)], float), n)
            # elongated words only (Terra's length / height >= 2.5), where an axis is well defined
            el = [s & (r["taspect"] >= 2.5) for r, s in zip(rs, sel)]
            ne = np.array([s.sum() for s in el], float)
            k["elongated"] = {"n": int(ne.sum()),
                              "recall": boot.ratio(np.array([(r["tbest"][s] >= S.MAIN_IOU).sum()
                                                             for r, s in zip(rs, el)], float), ne),
                              "axis_median": boot.quantile([r["terr"][s][r["tbest"][s] >= S.MAIN_IOU]
                                                            for r, s in zip(rs, el)], 50),
                              "axis_p90": boot.quantile([r["terr"][s][r["tbest"][s] >= S.MAIN_IOU]
                                                         for r, s in zip(rs, el)], 90)}
            o["kinds"][kind] = k
        # the grouped boxes
        ng = np.array([len(r["gbest"]) for r in rs], float)
        o["boxes"] = {"n": int(ng.sum())}
        for thr in (0.1, 0.3):
            un = np.array([(r["gbest"] < thr).sum() for r in rs], float)
            unk = np.array([((r["gbest"] < thr) & ~r["gunk"]).sum() for r in rs], float)
            o["boxes"][f"unmatched_{thr:.1f}"] = int(un.sum())
            o["boxes"][f"unmatched_{thr:.1f}_not_near_unknown"] = int(unk.sum())
            o["boxes"][f"unmatched_{thr:.1f}_share"] = boot.ratio(un, ng)
        o["boxes"]["merged"] = boot.ratio(np.array([r["gmerged"].sum() for r in rs], float), ng)
        o["boxes"]["per_terra_word"] = float(ng.sum() / max(1, sum(len(r["tbest"]) for r in rs)))
        res["variants"][v] = o
    # per group, the zoning code and all kinds at IoU 0.3 and 0.5
    TEST = "test (1980+1972+1975)"
    for g in [g for g, _ in S.GROUPS] + [TEST]:
        mem = [i for i, gg in enumerate(groups) if (gg == g if g != TEST else gg != "2000 val")]
        b = Boot(len(mem))
        o = {}
        for v in all_variants():
            rs = [per[keys[i]][v] for i in mem]
            for kind in ("all", "zoning code"):
                sel = [kind_sel(r, kind) for r in rs]
                n = np.array([s.sum() for s in sel], float)
                hit = np.array([(r["tbest"][s] >= S.MAIN_IOU).sum() for r, s in zip(rs, sel)], float)
                ax = [r["terr"][s][r["tbest"][s] >= S.MAIN_IOU] for r, s in zip(rs, sel)]
                er = [r["terr"][s] for r, s in zip(rs, sel)]
                ok = np.array([((r["tbest"][s] >= S.MAIN_IOU) & (e <= 10)).sum() for r, s, e in zip(rs, sel, er)],
                              float)
                o[f"{v}/{kind}"] = {"n": int(n.sum()), "recall_0.3": b.ratio(hit, n), "usable_10": b.ratio(ok, n),
                                    "axis_median": b.quantile(ax, 50), "axis_p90": b.quantile(ax, 90)}
        res["per_group"][g] = o
        for r_ in ("r15", "r30"):
            tot = {f: int(sum(per[keys[i]]["sol"][r_][f] for i in mem)) for f in
                   ("points", "no_terra", "no_terra_zoning", "no_terra_but_box")}
            res["sol"].setdefault(g, {})[r_] = tot
    return res


# --- the plot (PIL: no plotting library is a dependency) ---------------------------------------------------------------

COLORS = {"all": (60, 60, 60), "zoning code": (200, 40, 40), "block number": (40, 110, 200),
          "street name": (40, 150, 70), "other": (200, 140, 20)}


def font(size):
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def panel(draw, box, title, xs, series, xlabel, ylabel, ymax=1.0, xfmt="{:.2f}"):
    x0, y0, x1, y1 = box
    f, fs = font(15), font(12)
    draw.text((x0, y0 - 28), title, fill=(0, 0, 0), font=f)
    draw.rectangle(box, outline=(120, 120, 120))
    xmin, xmax = min(xs), max(xs)

    def px(x):
        return x0 + (x - xmin) / (xmax - xmin) * (x1 - x0)

    def py(y):
        return y1 - y / ymax * (y1 - y0)

    for t in np.linspace(0, ymax, 6):
        draw.line([(x0, py(t)), (x1, py(t))], fill=(225, 225, 225))
        draw.text((x0 - 38, py(t) - 7), f"{t:.2f}" if ymax <= 1 else f"{t:.0f}", fill=(60, 60, 60), font=fs)
    for x in xs:
        draw.text((px(x) - 12, y1 + 4), xfmt.format(x), fill=(60, 60, 60), font=fs)
    draw.text(((x0 + x1) / 2 - 30, y1 + 22), xlabel, fill=(0, 0, 0), font=fs)
    draw.text((x0 - 40, y0 - 14), ylabel, fill=(0, 0, 0), font=fs)
    for i, (name, ys, ci) in enumerate(series):
        col = COLORS[name]
        pts = [(px(x), py(y)) for x, y in zip(xs, ys) if y is not None]
        if ci:
            for x, (lo, hi) in zip(xs, ci):
                if lo is not None:
                    draw.line([(px(x), py(lo)), (px(x), py(hi))], fill=col)
        if len(pts) > 1:
            draw.line(pts, fill=col, width=2)
        for p in pts:
            draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=col)
        draw.text((x1 + 10, y0 + 6 + 18 * i), name, fill=col, font=f)


def plot(res: dict) -> None:
    W, H = 2100, 560
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    lv = [f"L{x:.2f}" for x in S.LEVELS]
    xs = list(S.LEVELS)
    V = res["variants"]
    ser = [(k, [V[v]["kinds"][k]["recall"]["0.3"][0] for v in lv],
            [(V[v]["kinds"][k]["recall"]["0.3"][1], V[v]["kinds"][k]["recall"]["0.3"][2]) for v in lv])
           for k in KIND_ALL]
    panel(d, (60, 60, 560, 440), "Recall at IoU 0.3, by level", xs, ser, "level", "share of Terra's words")
    ious = [float(x) for x in S.IOUS]
    for j, v in enumerate(("L0.50", "L0.90")):
        ser = [(k, [V[v]["kinds"][k]["recall"][f"{t:.1f}"][0] for t in ious], None) for k in KIND_ALL]
        panel(d, (760 + 700 * j, 60, 1160 + 700 * j - 0, 440), f"Recall by IoU threshold, level {v[1:]}", ious, ser,
              "IoU threshold", "share of Terra's words", xfmt="{:.1f}")
    im.save(OUT / "curves_recall.png")
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    for j, (stat, title) in enumerate((("median", "Axis error, median (deg), matched at IoU 0.3"),
                                       ("p90", "Axis error, 90th percentile (deg)"))):
        ser = [(k, [V[v]["kinds"][k]["axis"]["0.3"][stat][0] for v in lv],
                [(V[v]["kinds"][k]["axis"]["0.3"][stat][1], V[v]["kinds"][k]["axis"]["0.3"][stat][2]) for v in lv])
               for k in KIND_ALL]
        panel(d, (60 + 700 * j, 60, 560 + 700 * j, 440), title, xs, ser, "level", "degrees", ymax=90.0)
    ser = [(k, [V[v]["kinds"][k]["usable"]["10"][0] for v in lv],
            [(V[v]["kinds"][k]["usable"]["10"][1], V[v]["kinds"][k]["usable"]["10"][2]) for v in lv])
           for k in KIND_ALL]
    panel(d, (1460, 60, 1860, 440), "Found at IoU 0.3 with axis within 10 deg, of all words", xs, ser, "level",
          "share of Terra's words")
    im.save(OUT / "curves_axis.png")
    iv = [v for v in S.ink_variants() if v in V]
    if iv:
        im = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(im)
        xs = [float(v[1:]) for v in iv]
        for j, t in enumerate(("0.3", "0.5")):
            ser = [(k, [V[v]["kinds"][k]["recall"][t][0] for v in iv],
                    [(V[v]["kinds"][k]["recall"][t][1], V[v]["kinds"][k]["recall"][t][2]) for v in iv])
                   for k in KIND_ALL]
            panel(d, (60 + 700 * j, 60, 560 + 700 * j, 440), f"Ink-gated (I), recall at IoU {t}, by level", xs, ser,
                  "level", "share of Terra's words")
        ser = [(k, [V[v]["kinds"][k]["axis"]["0.3"]["p90"][0] for v in iv], None) for k in KIND_ALL]
        panel(d, (1460, 60, 1860, 440), "Ink-gated (I), axis error 90th percentile (deg)", xs, ser, "level",
              "degrees", ymax=90.0)
        im.save(OUT / "curves_ink.png")


# --- the crops -------------------------------------------------------------------------------------------------------

TILE = 210
CAP = 30


def page_gray(key: str) -> np.ndarray:
    series, sheet = key.split("/")
    g = cv2.imread(str(config.WORK_DIR / series / sheet / "page.png"), cv2.IMREAD_GRAYSCALE)
    return g


_pages: dict[str, np.ndarray] = {}


def tile(key: str, keep: dict, subject: np.ndarray, subject_is: str, caption: str) -> Image.Image:
    """A crop around `subject` (4, 2) with Terra's boxes (red) and the grouped boxes (green) drawn on the scan."""
    if key not in _pages:
        _pages.clear()
        _pages[key] = page_gray(key)
    g = _pages[key]
    h, w = g.shape
    lo, hi = subject.min(0), subject.max(0)
    ctr = (lo + hi) / 2
    win = max(110.0, 1.6 * float(max(hi - lo)))
    x0, y0 = int(round(ctr[0] - win / 2)), int(round(ctr[1] - win / 2))
    n = int(round(win))
    canvas = np.full((n, n), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + n, w), min(y0 + n, h)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = g[sy0:sy1, sx0:sx1]
    s = TILE / n
    img = cv2.cvtColor(cv2.resize(canvas, (TILE, TILE), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)
    org = np.array([x0, y0], float)

    def draw(polys, color, thick):
        for p in polys:
            if (p.max(0) < org).any() or (p.min(0) > org + n).any():
                continue
            cv2.polylines(img, [np.round((p - org) * s).astype(np.int32).reshape(-1, 1, 2)], True, color, thick,
                          cv2.LINE_AA)

    draw(keep["T"], (255, 40, 40), 1)
    draw(keep["G"], (0, 150, 0), 1)
    draw([subject], (255, 40, 40) if subject_is == "T" else (0, 150, 0), 2)
    out = Image.new("RGB", (TILE, TILE + CAP), "white")
    out.paste(Image.fromarray(img), (0, 0))
    ImageDraw.Draw(out).text((3, TILE + 2), caption, fill=(0, 0, 0), font=font(11))
    return out


def montage(tiles: list[Image.Image], cols: int, name: str) -> None:
    rows = (len(tiles) + cols - 1) // cols
    im = Image.new("RGB", (cols * (TILE + 6) + 6, rows * (TILE + CAP + 6) + 6), (235, 235, 235))
    for i, t in enumerate(tiles):
        im.paste(t, (6 + (i % cols) * (TILE + 6), 6 + (i // cols) * (TILE + CAP + 6)))
    im.save(OUT / name)


def pick(rng, items: list, n: int) -> list:
    if len(items) <= n:
        return items
    return [items[i] for i in sorted(rng.choice(len(items), n, replace=False))]


def crops(keep_all: dict, tag: str, only: tuple = ()) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    short = lambda k: k.split("/")[0][:4] + "/" + k.split("-p")[-1][-2:] if "-p" in k else k.split("/")[0] + "/" + k.split("p")[-1]
    cats: dict[str, list] = {n: [] for n in ("unmatched", "thin", "split", "merged", "missed_zoning", "bad_axis")}
    for key, k in keep_all.items():
        for j in np.flatnonzero(k["gin"]):
            if k["gbest"][j] < 0.1:
                if k["gaspect"][j] >= 8:
                    cats["thin"].append((key, "G", j))
                if not k["gunk"][j]:
                    cats["unmatched"].append((key, "G", j))
            if k["gmerged"][j]:
                cats["merged"].append((key, "G", j))
        for j in np.flatnonzero(k["tin"]):
            if k["tsplit"][j]:
                cats["split"].append((key, "T", j))
            if k["tkind"][j] == 0 and k["tbest"][j] < 0.1:
                cats["missed_zoning"].append((key, "T", j))
            if k["tkind"][j] == 0 and k["tbest"][j] >= 0.3 and k["terr"][j] > 30:
                cats["bad_axis"].append((key, "T", j))
    counts = {n: len(v) for n, v in cats.items()}
    sizes = {"unmatched": 30, "thin": 12, "split": 12, "merged": 12, "missed_zoning": 12, "bad_axis": 12}
    for name, items in cats.items():
        if only and name not in only:
            continue
        tiles = []
        for key, which, j in pick(rng, items, sizes[name]):
            k = keep_all[key]
            if which == "G":
                cap = (f"{short(key)} box {k['gaspect'][j]:.1f}:1 chance {k['gmean'][j]:.2f}"
                       + (f" best IoU {k['gbest'][j]:.2f}" if name != "merged" else ""))
                tiles.append(tile(key, k, k["G"][j], "G", cap))
            else:
                cap = f"{short(key)} {S.KINDS[k['tkind'][j]]}"
                if name == "bad_axis":
                    cap += f" err {k['terr'][j]:.0f} deg"
                tiles.append(tile(key, k, k["T"][j], "T", cap))
        montage(tiles, 6, f"crops_{name}_{tag}.png")
    return counts


# --- the markdown ----------------------------------------------------------------------------------------------------

CAPTIONS = {
    "curves_recall.png": ("Terra's words found by grouped boxes: left, by level at IoU 0.3; middle and right, by IoU "
                          "threshold at level 0.5 and 0.9, each kind a line, bars are the 95% sheet-bootstrap "
                          "interval.  Look for where each curve is flat (the level or threshold does not matter "
                          "there) and where a kind drops."),
    "curves_ink.png": ("The ink-gated variant (I): recall at IoU 0.3 and 0.5 by level, and the axis error's 90th "
                       "percentile.  Compare with curves_recall.png: the same levels, the chance counted only "
                       "within 4 px of ink.  Look for whether the recall curve now holds up at the low levels."),
    "curves_axis.png": ("The axis of the matched words, by level: median and 90th percentile of |angle difference| "
                        "modulo 180 deg, and the share of all of Terra's words found with an axis within 10 deg.  "
                        "Look for the level where the tail (90th percentile) stops falling."),
}


def crop_captions(tag: str) -> dict:
    lv = f"{tag[1]}.{tag[2:]}" if tag.startswith("L") else tag
    return {
        f"crops_unmatched_{tag}.png": f"30 grouped boxes (green) at level {lv} with no Terra word of IoU 0.1 or more "
        "(red boxes are Terra's), not near Sol's points or the other passes' boxes.  Terra misses text, so judge by "
        "eye: real text Terra missed (a miss of the reference, not of the grouping), or ink that is not lettering "
        "(a false box).  Random, seed 0, from every scope.",
        f"crops_thin_{tag}.png": f"Unmatched grouped boxes (level {lv}) at least 8:1 long, the subject in green: "
        "district lines, leader lines and rules read as text.  Look for whether the box lies along a line with no "
        "letters.",
        f"crops_split_{tag}.png": f"Terra words (subject, thick red) that two or more grouped boxes (green, level "
        f"{lv}) each mostly lie inside: a word cut by the grouping.  Look for the gap that cuts it (a space, a "
        "street line through the word).",
        f"crops_merged_{tag}.png": f"Grouped boxes (subject, thick green, level {lv}) holding two or more of "
        "Terra's words each mostly inside it.  Look for what joins them (touching labels, a line, a leader, a block "
        "of text).",
        f"crops_missed_zoning_{tag}.png": f"Zoning-code words of Terra's (thick red) that no grouped box of level "
        f"{lv} overlaps with IoU 0.1 or more.  A box may still cover them inside a much larger blob.  Look at "
        "whether the scan holds a code and what the green boxes around it are.",
        f"crops_bad_axis_{tag}.png": f"Zoning-code words found (IoU 0.3 or more, level {lv}) whose grouped box's "
        "axis differs from Terra's by more than 30 deg.  Look for which one is right: a short word whose px are "
        "round, a code next to a line, or Terra's angle off.",
    }


def headline(res: dict, w) -> None:
    V, PG = res["variants"], res["per_group"]
    test = "test (1980+1972+1975)"
    w("## Headline (revised after review)\n")
    w("What is pre-specified and what is post hoc.  Pre-specified (the brief): plain grouping L at levels 0.3 to 0.8, "
      "by kind, over IoU 0.1 to 0.7; hysteresis H (the one optional variant).  **Post hoc, added after looking at the "
      "crops of this same scoring data**: levels 0.9 and 0.95; the ink gate I (and its radius, 4 px); the "
      "speck-cutoff grid; the quarter-turn and elongated analyses.  MIN_PX = 20 was set before any look and never "
      "tuned (sensitivity below).\n")
    w("### 1. Plain grouping, the pre-specified headline (all 105 sheets pooled, zoning code, IoU 0.3)\n")
    w("Recall and, beside it, the share found with the axis within 10 deg (of ALL the kind's words).\n")
    w("| level | zoning: found | zoning: found, axis within 10 deg | all kinds: found | all kinds: axis within 10 deg |"
      "\n|---|---|---|---|---|")
    for lv in S.LEVELS:
        v = f"L{lv:.2f}"
        z, a = V[v]["kinds"]["zoning code"], V[v]["kinds"]["all"]
        tag = "" if lv <= 0.8 else " (post hoc)"
        w(f"| {lv:.2f}{tag} | {fmt(z['recall']['0.3'])} | {fmt(z['usable']['10'])} | {fmt(a['recall']['0.3'])} | "
          f"{fmt(a['usable']['10'])} |")
    w("\nOver the pre-specified range 0.3 to 0.8 there is no flat part: both columns climb at every step (see the "
      "intervals).  The curve flattens only at 0.8 to 0.9, and 0.9 is post hoc.  So plain grouping has no level at "
      "which the cutoff does not matter within the pre-specified range.\n")
    w("### 2. A choice made on one part, reported on the other\n")
    w("Each family's setting is chosen on the **design** part, the 2000 map's validation area (13 sheets, 348 zoning "
      "codes: held out of the detector's training), by the highest share of zoning codes found at IoU 0.3 with the axis "
      "within 10 deg; the number reported is on the **test** part, the three test drawings 1980, 1972 and 1975 "
      "pooled (92 sheets, 6,527 zoning codes), which played no part in the choice.  Settings: plain = every level "
      "0.3 to 0.95; ink-gated (new, speck cutoff in ink px) = level x cutoff grid of the speck run.  The 2000 "
      "validation part is small, so the choice is noisy, and the pictures that suggested the ink gate came from "
      "the test part (the gate itself is therefore still a post-hoc idea, only its setting is held out).\n")
    w("| family | chosen on design | design: found / axis within 10 deg | test: found | test: axis within 10 deg |"
      "\n|---|---|---|---|---|")
    fams = {"plain L (levels 0.3 to 0.95)": [f"L{x:.2f}" for x in S.LEVELS],
            "plain L, pre-specified levels only (0.3 to 0.8)": [f"L{x:.2f}" for x in S.LEVELS if x <= 0.8],
            "ink gate I, speck cutoff in ink px (level x cutoff)": [v for v in S.speck_variants() if v[0] == "I"],
            "ink gate I as first measured (mask-px cutoff 20)": S.ink_variants()}
    chosen = {}
    for name, vs in fams.items():
        vs = [v for v in vs if f"{v}/zoning code" in PG["2000 val"]]
        if not vs:
            continue
        best = max(vs, key=lambda v: PG["2000 val"][f"{v}/zoning code"]["usable_10"][0] or 0)
        chosen[name] = best
        d, t = PG["2000 val"][f"{best}/zoning code"], PG[test][f"{best}/zoning code"]
        w(f"| {name} | {best} | {fmt(d['recall_0.3'], True, 0)} / {fmt(d['usable_10'], True, 0)} | "
          f"{fmt(t['recall_0.3'])} | {fmt(t['usable_10'])} |")
    w("\nThe chosen setting on each test drawing separately (zoning code, found / axis within 10 deg):\n")
    w("| family | " + " | ".join(S.paths.TEXT_TEST_SERIES) + " |\n|---|---|---|---|")
    for name, best in chosen.items():
        w(f"| {name} ({best}) | " + " | ".join(
            f"{fmt(PG[g][f'{best}/zoning code']['recall_0.3'], True, 0)} / "
            f"{fmt(PG[g][f'{best}/zoning code']['usable_10'], True, 0)}" for g in S.paths.TEXT_TEST_SERIES) + " |")
    w("")
    w("### 3. Speck cutoff sensitivity (all 105 sheets pooled)\n")
    w("L: components of at least N px of the chance mask; I: ink-gated, components holding at least N px of INK "
      "(see the ink-gate check below).  N = 20 for L is the cutoff used everywhere above.  Zoning code, IoU 0.3; "
      "boxes: all grouped boxes in scope; unmatched: share with no Terra word at IoU 0.1; zoning split: share of "
      "zoning codes cut into two or more boxes.\n")
    w("| variant | zoning found | zoning, axis within 10 deg | all kinds found | boxes | unmatched share | zoning split |"
      "\n|---|---|---|---|---|---|---|")
    for k in "LI":
        for lv in S.SPECK_LEVELS:
            for m in S.SPECK_MIN:
                v = f"{k}{lv:.2f}m{m}"
                if v not in V:
                    continue
                z, a, b = V[v]["kinds"]["zoning code"], V[v]["kinds"]["all"], V[v]["boxes"]
                w(f"| {v} | {fmt(z['recall']['0.3'])} | {fmt(z['usable']['10'])} | {fmt(a['recall']['0.3'])} | "
                  f"{b['n']} | {fmt(b['unmatched_0.1_share'])} | {fmt(z['split'])} |")
    w("")
    if "I0.80m0" in V:
        old = V["I0.80"]["boxes"]["n"]
        new5 = V["I0.80m5"]["boxes"]["n"]
        new20 = V["I0.80m20"]["boxes"]["n"]
        w("#### Ink-gate check: do 1 to 3 px specks pass?\n")
        w(f"Yes, as first measured: I0.80 with the 20 px cutoff counted the dilated mask (a 1 px speck of ink becomes "
          f"a disc of about 60 px of mask, which the detector's chance fills), so it kept {old} boxes, against {new5} "
          f"when a box must hold at least 5 ink px and {new20} at 20 ink px.  That was a flaw of the measure, not of "
          "the detector: the cutoff now counts ink px for the I variants (variants named `I..m<N>`), and the family "
          "'ink gate I, speck cutoff in ink px' above uses it.  `crops_unmatched_I080.png` and the I numbers in the "
          "tables further down are the first measurement (mask-px cutoff).\n")
    w("")


def md(res: dict, counts: dict, mid: str) -> str:
    V = res["variants"]
    L = []
    w = L.append
    w("# Spike S3: word boxes grouped from the detector's text chance, against Terra's words\n")
    w(f"{res['sheets']} sheets.  Intervals: {res['bootstrap']}.  Level is the cutoff on the chance; every number is a "
      "curve over it.  L = components of px at or above the level; H = the same, keeping components with a px "
      f">= {S.SEED} (hysteresis); self = Terra's own boxes through the same code (sanity check).  Components under "
      f"{S.MIN_PX} px are dropped.  A Terra word is found at IoU t when a grouped box overlaps it with IoU >= t "
      "(rotated rectangles).\n")
    headline(res, w)
    w("## Scope\n")
    w("| scope | sheets | scored px | Terra words (centre in scope): all / zoning code / block number / street name / "
      "other |\n|---|---|---|---|")
    for g, s in res["scope"].items():
        t = s["terra_words"]
        w(f"| {g} | {s['sheets']} | {s['scope_px']:,} | " + " / ".join(str(t[k]) for k in KIND_ALL) + " |")
    w("\nThe 2000 map is scored on the validation area of the detector's training (held out); 1980, 1972 and 1975 "
      "are whole drawings the detector never saw.\n")
    sf = V["self"]["kinds"]["all"]
    w("## Sanity check: Terra's boxes against themselves\n")
    w(f"Recall {sf['recall']['0.7'][0]:.3f} at IoU 0.7 (and {sf['recall']['0.1'][0]:.3f} at 0.1), axis error median "
      f"{sf['axis']['0.5']['median'][0]:.3f} deg, 90th percentile {sf['axis']['0.5']['p90'][0]:.3f} deg, over "
      f"{sf['n']} words.\n")
    lv = [f"L{x:.2f}" for x in S.LEVELS]
    hv = [f"H{x:.2f}" for x in S.LEVELS if x < S.SEED]
    iv = [v for v in S.ink_variants() if v in V]

    def table(title, vs, cell, kinds=KIND_ALL):
        w(f"### {title}\n")
        w("| variant | " + " | ".join(kinds) + " |\n|---|" + "---|" * len(kinds))
        for v in vs:
            w(f"| {v} | " + " | ".join(cell(V[v]["kinds"][k]) for k in kinds) + " |")
        w("")

    w("## Recall: the share of Terra's words found\n")
    w("Counts of words: " + ", ".join(f"{k} {V['L0.50']['kinds'][k]['n']}" for k in KIND_ALL) + ".\n")
    table("At IoU 0.3, by level (percent)", lv + hv + iv, lambda k: fmt(k["recall"]["0.3"]))
    for v in ("L0.50", "L0.80", "L0.90"):
        w(f"### Level {v[1:]} by IoU threshold (percent)\n")
        w("| IoU | " + " | ".join(KIND_ALL) + " |\n|---|" + "---|" * len(KIND_ALL))
        for t in S.IOUS:
            w(f"| {t:.1f} | " + " | ".join(fmt(V[v]["kinds"][k]["recall"][f"{t:.1f}"]) for k in KIND_ALL) + " |")
        w("")
    w("## Axis error on the matched words\n")
    w("|angle difference| modulo 180 deg, words matched at IoU 0.3.  Terra's angle is a whole degree from a language "
      "model, itself an estimate; the numbers include its error.\n")
    table("Median (deg)", lv + hv + iv, lambda k: fmt(k["axis"]["0.3"]["median"], False))
    table("90th percentile (deg)", lv + hv + iv, lambda k: fmt(k["axis"]["0.3"]["p90"], False))
    table("Found at IoU 0.3 with the axis within 10 deg, of ALL Terra's words (percent)", lv + hv + iv,
          lambda k: fmt(k["usable"]["10"]))
    table("Same within 5 deg (percent)", lv, lambda k: fmt(k["usable"]["5"]))
    table("Same within 20 deg (percent)", lv, lambda k: fmt(k["usable"]["20"]))
    w("Words at least 2.5 times as long as high (Terra's box), where an axis is well defined: recall at IoU 0.3 and "
      "the axis median / 90th percentile (deg):\n")
    w("| variant | " + " | ".join(KIND_ALL) + " |\n|---|" + "---|" * len(KIND_ALL))
    for v in lv + iv:
        w(f"| {v} | " + " | ".join(
            f"{fmt(V[v]['kinds'][k]['elongated']['recall'], True, 0)}; {V[v]['kinds'][k]['elongated']['axis_median'][0]:.1f}"
            f" / {V[v]['kinds'][k]['elongated']['axis_p90'][0]:.1f}" for k in KIND_ALL) + " |")
    w("")
    w("### Quarter-turn flips among the matched words (IoU 0.3)\n")
    w("The share of matched words whose axis is 75 deg or more off (the box's elongation is not the word's reading "
      "direction: a vertical stack of upright letters such as R / - / 1, or a one-glyph word), and the 90th "
      "percentile of the rest (deg).\n")
    table("Flipped share (percent)", lv + iv, lambda k: fmt(k["flip"]))
    table("90th percentile of the axis error without the flipped (deg)", lv + iv,
          lambda k: fmt(k["axis_noflip"]["p90"], False))
    w("## Split and merged words\n")
    table("Terra words split (two or more boxes each mostly inside the word), percent of the words", lv + hv + iv,
          lambda k: fmt(k["split"]))
    table("Terra words inside a merged box (a box holding two or more Terra words), percent of the words", lv + hv + iv,
          lambda k: fmt(k["in_merged"]))
    w("Terra's own boxes overlap each other in places (self row), so a floor of a few percent is not the grouping's.\n")
    w("## Grouped boxes with no Terra word\n")
    w("| variant | grouped boxes | per Terra word | IoU < 0.1: count | share | not near Sol's points or other passes' "
      "boxes | IoU < 0.3: count | merged boxes (share) |\n|---|---|---|---|---|---|---|---|")
    for v in lv + hv + iv:
        b = V[v]["boxes"]
        w(f"| {v} | {b['n']} | {b['per_terra_word']:.2f} | {b['unmatched_0.1']} | {fmt(b['unmatched_0.1_share'])} | "
          f"{b['unmatched_0.1_not_near_unknown']} | {b['unmatched_0.3']} | {fmt(b['merged'])} |")
    w("\nTerra misses text, so these are not all false: see the crop pages below.\n")
    w("## By scope: all kinds and zoning codes, at IoU 0.3 (recall percent; axis median / 90th percentile deg)\n")
    gs = [g for g, _ in S.GROUPS]
    w("| variant, kind | " + " | ".join(gs) + " |\n|---|" + "---|" * len(gs))
    for v in lv + iv:
        for kind in ("all", "zoning code"):
            cells = []
            for g in gs:
                c = res["per_group"][g][f"{v}/{kind}"]
                cells.append(f"{fmt(c['recall_0.3'], True, 0)}; {c['axis_median'][0]:.1f} / {c['axis_p90'][0]:.1f}"
                             if c["n"] else "n/a")
            w(f"| {v} {kind} | " + " | ".join(cells) + " |")
    w("")
    w("## Terra's own misses (Sol's label points with no Terra word nearby)\n")
    w("Sol's label points in the scope.  A point has a Terra word nearby when a Terra word's box (any kind) lies "
      "within R px of it; `zoning` counts words of the zoning-code kind only (Sol's points are district-label "
      "points, so a zoning-code word is what should be near).  `+ box`: of the points with no Terra word, how many "
      f"have a grouped box ({mid}) within R px: text Terra missed that the detector found.\n")
    w("| scope | R px | points | no Terra word | share | + a grouped box | no zoning-code word | share |\n"
      "|---|---|---|---|---|---|---|---|")
    for g in gs:
        for r in ("r15", "r30"):
            t = res["sol"][g][r]
            n = max(1, t["points"])
            w(f"| {g} | {r[1:]} | {t['points']} | {t['no_terra']} | {100 * t['no_terra'] / n:.1f}% | "
              f"{t['no_terra_but_box']} | {t['no_terra_zoning']} | {100 * t['no_terra_zoning'] / n:.1f}% |")
    w("")
    w("## Terra's degenerate boxes (reference quality)\n")
    w("Terra's word rectangles (`word_rect`) whose short side is under 4 px: a slanted word whose axis box does not "
      "fix its shape (a thin diagonal bar).  They can't be matched by any grouping.  Level I0.80, in the scope.\n")
    with open(OUT / "keep_I080.pkl", "rb") as fh:
        kp = pickle.load(fh)
    w("| kind | words | degenerate | share | found at IoU 0.3 (degenerate) | found at IoU 0.3 (others) |\n"
      "|---|---|---|---|---|---|")
    for ki, kn in enumerate(KIND_ALL):
        n = d = fd = fo = 0
        for key, k in kp.items():
            c = k["T"]
            short = np.minimum(np.linalg.norm(c[:, 1] - c[:, 0], axis=1), np.linalg.norm(c[:, 3] - c[:, 0], axis=1))
            sel = k["tin"] & (np.ones(len(c), bool) if kn == "all" else (k["tkind"] == ki - 1))
            deg = sel & (short < 4)
            n += int(sel.sum())
            d += int(deg.sum())
            fd += int((deg & (k["tbest"] >= S.MAIN_IOU)).sum())
            fo += int((sel & ~deg & (k["tbest"] >= S.MAIN_IOU)).sum())
        w(f"| {kn} | {n} | {d} | {100 * d / max(n, 1):.1f}% | {100 * fd / max(d, 1):.0f}% | "
          f"{100 * fo / max(n - d, 1):.1f}% |")
    w("")
    w("## Pictures\n")
    caps = dict(CAPTIONS)
    for tag in counts:
        caps.update({n: c for n, c in crop_captions(tag).items() if (OUT / n).is_file()})
    for name, cap in caps.items():
        tag = name.removesuffix(".png").rsplit("_", 1)[-1]
        n = counts.get(tag, {}).get(name.removeprefix("crops_").removesuffix(f"_{tag}.png"))
        w(f"![{name}]({name})\n\n**{name}**: {cap}" + (f"  ({n} such in all.)" if n is not None else "") + "\n")
    f = OUT / "findings.md"
    if f.is_file():
        w(f.read_text())
    return "\n".join(L)


def report() -> None:
    per = load()
    res = compute(per)
    plot(res)
    mid = f"L{S.MID:.2f}"
    counts = {}
    for tag, only in (("I080", ()), ("L050", ("unmatched", "thin"))):
        with open(OUT / f"keep_{tag}.pkl", "rb") as f:
            counts[tag] = crops(pickle.load(f), tag, only)
    res["crop_candidates"] = counts
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    (OUT / "results.md").write_text(md(res, counts, mid))
    print("done")
