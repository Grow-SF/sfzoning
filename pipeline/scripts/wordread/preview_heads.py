"""A preview of the word heads on the Mission Street sheet (2000-digital-zoning-p07), before any word is formed: each
quarter drawn from its own held-out detector (arm (a), seed 0, the sign-free heads), what the heads say, Terra's and
today's boxes for comparison.  CPU only, $0.  Writes data/solver/experiments/wordread/review/preview-heads/
{review.html, numbers.json}; the controller publishes the page.

    uv run python scripts/wordread/preview_heads.py [--sheet 2000-digital-zoning-p07] [--name a] [--seed 0]

Layers (each quarter's px from the detector that held that quarter out; a quarter without one drawn faded):
  ink          the page's ink (px darker than grid.INK_BELOW), as the detector reads it
  core         the paper tinted blue by the core chance (sigmoid of the core logit): no level, the tint is the chance
  tags         the paper tinted by the tag's colour, with the core chance as its opacity: the 4-D tag projected on its
               first three principal axes (a PCA weighted by core chance, fitted on that quarter's px of the sheet),
               each axis at +-2.5 of its sd mapped to 0..255 (display scaling)
  axis ticks   the predicted axis (sign-free, modulo 180) at every TICK-th px each way, a TICK_LEN px tick, its
               opacity the core chance (drawn in 10 opacity steps, display only)
  outlines     Terra's word boxes (red) and today's boxes (data/solver/boxes, teal)
The ink is drawn over the tints, darkened from them, so a word's letters carry its tag colour.

The top line, per quarter: the axis error and the size ratios against Terra's words at their core px (each word's
rectangle px), per px and per word (a word: the core-weighted circular mean of its px' axes, the weighted geometric
means of their lengths and heights, the weighted mean of their centre votes), as `axis_size.py` measures them; the
Terra words whose px' tags fall into two clusters (DISPLAY ONLY: per word, 2-means of its core-weighted tags; two
clusters when the gap between the two means is wider than the gap between the word's mean tag and that of the Terra
word nearest it on the page, read by the same detector: the word's tags then differ inside it more than from its
neighbour's); today's merged_with_word on the same sheet and quarter.  Tap a Terra box for its numbers.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
           "OPENCV_FOR_THREADS_NUM"):
    os.environ.setdefault(_k, "1")
import argparse  # noqa: E402
import base64  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_detector as RD  # noqa: E402

from sfzmaps.solver import paths, wordboxes  # noqa: E402
from sfzmaps.solver.textnet import data, reader_stage  # noqa: E402
from sfzmaps.solver.wordread import measures, review  # noqa: E402

SHEET = "2000-digital-zoning-p07"
TICK = 16            # px between axis ticks each way (legibility only)
TICK_LEN = 10        # px
TILE = 2048
JPEG_QUALITY = 85
PCA_SPAN = 2.5       # sd of a principal axis mapped to the colour's full range (display scaling)
KMEANS_STEPS = 20
CORE_RGB = np.array([43, 108, 176], np.float32)
INK_SHADE = 0.42     # the ink drawn as its px' colour times this
FADED_PAPER, FADED_INK = (238, 238, 234), (206, 206, 202)
NOT_TRAINED = "fold not yet trained"
TERRA_COL, TODAY_COL, TICK_COL = "#d22222", "#178f8f", "#111111"


def pca_colours(tag: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(n, 3) colours in 0..1 of tags (k, n) on their first three principal axes, weighted by `w` (see the module);
    each axis's sign set so its largest coefficient is positive."""
    t = tag.astype(np.float64)
    ww = w.astype(np.float64) / max(float(w.sum()), 1e-12)
    mu = (t * ww).sum(1, keepdims=True)
    d = t - mu
    cov = (d * ww) @ d.T
    val, vec = np.linalg.eigh(cov)
    order = np.argsort(val)[::-1][:3]
    val, vec = np.maximum(val[order], 1e-12), vec[:, order]
    vec *= np.sign(vec[np.argmax(np.abs(vec), 0), np.arange(3)])
    z = (vec.T @ d) / (PCA_SPAN * np.sqrt(val)[:, None])
    return np.clip(0.5 + 0.5 * z, 0, 1).T


def two_clusters(tag: np.ndarray, w: np.ndarray) -> dict:
    """Display only (see the module): a word's mean tag (weighted by `w`), and the gap between the two means of a
    2-means of its tags (k, n), seeded at the tag farthest from the weighted mean and the tag farthest from that, with
    the smaller cluster's weight share."""
    t = tag.T.astype(np.float64)
    w = w.astype(np.float64)
    mu = (t * w[:, None]).sum(0) / max(w.sum(), 1e-12)
    if len(t) < 2 or w.sum() <= 0:
        return {"mean": mu, "gap": 0.0, "minor": 0.0}
    a = t[np.argmax(((t - mu) ** 2).sum(1))]
    b = t[np.argmax(((t - a) ** 2).sum(1))]
    cent = np.stack([a, b])
    for _ in range(KMEANS_STEPS):
        lab = np.argmin(((t[:, None, :] - cent[None]) ** 2).sum(-1), 1)
        new = np.stack([(t[lab == j] * w[lab == j, None]).sum(0) / max(w[lab == j].sum(), 1e-12) for j in (0, 1)])
        if np.allclose(new, cent):
            break
        cent = new
    if (lab == 0).all() or (lab == 1).all():
        return {"mean": mu, "gap": 0.0, "minor": 0.0}
    return {"mean": mu, "gap": float(np.linalg.norm(cent[0] - cent[1])),
            "minor": float(min(w[lab == 0].sum(), w[lab == 1].sum()) / w.sum())}


def word_frame(rect):
    r = np.asarray(rect, float)
    e = r[1] - r[0]
    return float(np.degrees(np.arctan2(-e[1], e[0])) % 180), float(np.hypot(*e)), float(np.hypot(*(r[3] - r[0]))), \
        r.mean(0)


def _jpeg_tiles(rgb: np.ndarray, layer: str) -> str:
    out = []
    h, w = rgb.shape[:2]
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            part = rgb[y0:y0 + TILE, x0:x0 + TILE]
            buf = io.BytesIO()
            Image.fromarray(part).save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
            out.append(f'<img class="tile {layer}" alt="" width="{part.shape[1]}" height="{part.shape[0]}" '
                       f'style="left:{x0}px;top:{y0}px" src="data:image/jpeg;base64,'
                       f'{base64.b64encode(buf.getvalue()).decode("ascii")}">')
    return "".join(out)


def _path(polys) -> str:
    return "".join("M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in np.asarray(p, float).reshape(-1, 2)) + "Z"
                   for p in polys)


def build(sheet: str = SHEET, name: str = "a", seed: int = 0) -> Path:
    t0 = time.perf_counter()
    sd = RD.load_sheet(sheet)
    h, w = sd.height, sd.width
    ink = data.page_ink(sd)
    quarters = reader_stage.page_quarters(sd)
    wid = RD.word_ids(sd)
    shown, runs = np.zeros((h, w), bool), {}
    core = np.zeros((h, w), np.float32)
    colour = np.zeros((h, w, 3), np.float32)
    axis = np.zeros((h, w), np.float32)
    geo_px = {}                                   # quarter -> decoded geo at that quarter's word px
    for k, qn in enumerate(paths.QUARTERS):
        mine = quarters == k
        run = RD.run_dir(name, qn, seed)
        f = run / "words" / f"{sheet}.npz"
        if not mine.any() or not f.is_file():
            continue
        o = RD.outputs(run, sheet)
        if o.geo.shape[0] != 5:
            raise ValueError(f"{run} holds outputs of another layout ({o.geo.shape[0]} geo channels)")
        runs[qn] = run
        shown |= mine
        c = 1 / (1 + np.exp(-o.core[mine].astype(np.float64)))
        core[mine] = c
        colour[mine] = pca_colours(o.tag[:, mine], c)
        axis[mine] = o.geo[4][mine]
        ys, xs = np.nonzero(mine & (wid > 0))
        geo_px[k] = (ys, xs, o.geo[:, ys, xs].astype(np.float64), 1 / (1 + np.exp(-o.core[ys, xs].astype(np.float64))),
                     o.tag[:, ys, xs])
        del o
    # the views: paper tinted, ink darkened from its tint; faded where no held-out detector
    c3 = core[..., None]
    views = {"tags": (1 - c3) * 255 + c3 * colour * 255,
             "core": (1 - c3) * 255 + c3 * CORE_RGB[None, None, :],
             "ink": np.full((h, w, 3), 255, np.float32)}
    rgb = {}
    for key, v in views.items():
        v = np.where(ink[..., None], v * INK_SHADE if key != "ink" else 80.0, v)
        v[~shown & ~ink] = FADED_PAPER
        v[~shown & ink] = FADED_INK
        rgb[key] = np.clip(np.rint(v), 0, 255).astype(np.uint8)
    del views
    # the axis ticks: every TICK-th px, opacity the core chance in 10 steps
    gy, gx = np.mgrid[TICK // 2:h:TICK, TICK // 2:w:TICK]
    gy, gx = gy.ravel(), gx.ravel()
    keep = shown[gy, gx]
    gy, gx = gy[keep], gx[keep]
    op = np.rint(core[gy, gx] * 10).astype(int)
    th = np.radians(axis[gy, gx].astype(np.float64))
    dx, dy = np.cos(th) * TICK_LEN / 2, -np.sin(th) * TICK_LEN / 2
    ticks = {}
    for level in range(1, 11):
        s = op == level
        ticks[level] = "".join(f"M{x - a:.1f} {y - b:.1f}L{x + a:.1f} {y + b:.1f}"
                               for x, y, a, b in zip(gx[s], gy[s], dx[s], dy[s]))
    # Terra's words and today's boxes
    terra = review.terra_on_sheet(sheet)
    rects = terra[measures.C_COLS].to_numpy(float).reshape(-1, 4, 2)
    today = wordboxes.load_boxes()
    today = today[today["sheet"] == sheet].reset_index(drop=True)
    today_polys = today[measures.G_COLS].to_numpy(float).reshape(-1, 4, 2)
    _, _, _, t_best = review.matching(today, terra)
    # each Terra word's numbers, from its quarter's own detector at its px
    words, agg = [], {}
    for j in range(len(terra)):
        ang, length, height, centre = word_frame(rects[j])
        cx, cy = np.clip(np.rint(centre).astype(int), 0, [w - 1, h - 1])
        k = int(quarters[cy, cx])
        rec = {"poly": np.round(rects[j], 1).tolist(), "text": str(terra["text"].iloc[j]),
               "kind": str(terra["kind"].iloc[j]), "quarter": paths.QUARTERS[k],
               "today_iou": round(float(t_best[j]), 3),
               "shown": k in geo_px}
        if k in geo_px:
            ys, xs, g, c, tg = geo_px[k]
            s = wid[ys, xs] == j + 1
            if s.any():
                g, c, tg = g[:, s], c[s], tg[:, s]
                err = np.abs((g[4] - ang + 90) % 180 - 90)
                ce = np.hypot(xs[s] + g[0] - centre[0], ys[s] + g[1] - centre[1]) / height
                m = 0.5 * np.degrees(np.arctan2((c * np.sin(np.radians(2 * g[4]))).sum(),
                                                (c * np.cos(np.radians(2 * g[4]))).sum()))
                wl = float(np.exp((c * np.log(g[2])).sum() / c.sum()) / length)
                wh = float(np.exp((c * np.log(g[3])).sum() / c.sum()) / height)
                wc = np.array([(c * (xs[s] + g[0])).sum(), (c * (ys[s] + g[1])).sum()]) / c.sum()
                cl = two_clusters(tg, c)
                rec.update(px=int(s.sum()), core_mean=round(float(c.mean()), 3),
                           axis_err_px=round(float(np.median(err)), 2),
                           axis_err_word=round(float(abs((m - ang + 90) % 180 - 90)), 2),
                           length_px=round(float(np.median(g[2] / length)), 3), length_word=round(wl, 3),
                           height_px=round(float(np.median(g[3] / height)), 3), height_word=round(wh, 3),
                           centre_px=round(float(np.median(ce)), 3),
                           centre_word=round(float(np.hypot(*(wc - centre)) / height), 3), cluster=cl,
                           centre_xy=centre, q=k)
                a = agg.setdefault(k, {"px_err": [], "px_len": [], "px_hgt": [], "w_err": [], "w_len": [],
                                       "w_hgt": [], "w_ctr": [], "two": 0, "words": 0})
                a["px_err"].append(err)
                a["px_len"].append(g[2] / length)
                a["px_hgt"].append(g[3] / height)
                a["w_err"].append(rec["axis_err_word"])
                a["w_len"].append(wl)
                a["w_hgt"].append(wh)
                a["w_ctr"].append(rec["centre_word"])
                a["words"] += 1
        words.append(rec)
    # display only: two tag clusters when the word's inner gap is wider than its gap to its nearest Terra word's tag
    for k in agg:
        idx = [i for i, r in enumerate(words) if r.get("q") == k]
        xy = np.array([words[i]["centre_xy"] for i in idx]).reshape(-1, 2)
        means = np.array([words[i]["cluster"]["mean"] for i in idx]).reshape(len(idx), -1)
        two = 0
        for n, i in enumerate(idx):
            dist = np.hypot(*(xy - xy[n]).T)
            dist[n] = np.inf
            cl = words[i]["cluster"]
            if len(idx) > 1:
                nn = int(np.argmin(dist))
                apart = float(np.linalg.norm(means[n] - means[nn]))
                cl.update(neighbour=words[idx[nn]]["text"], neighbour_gap=round(apart, 3), two=bool(cl["gap"] > apart))
            else:
                cl.update(neighbour=None, neighbour_gap=None, two=False)
            two += int(cl["two"])
        agg[k]["two"] = two
    for r in words:
        r.pop("centre_xy", None)
        r.pop("q", None)
        if "cluster" in r:
            r["cluster"] = {"gap": round(r["cluster"]["gap"], 3), "minor": round(r["cluster"]["minor"], 3),
                            "neighbour": r["cluster"]["neighbour"], "neighbour_gap": r["cluster"]["neighbour_gap"],
                            "two": r["cluster"]["two"]}
    top = {}
    for k, a in sorted(agg.items()):
        qn = paths.QUARTERS[k]
        pe = np.concatenate(a["px_err"])
        today_n = RD.today_numbers(sheet, qn)
        top[qn] = {"detector": str(runs[qn].name), "terra_words": a["words"],
                   "px_axis_err_median": round(float(np.median(pe)), 2),
                   "px_within_10": round(float((pe <= 10).mean()), 4),
                   "px_length_ratio_median": round(float(np.median(np.concatenate(a["px_len"]))), 3),
                   "px_height_ratio_median": round(float(np.median(np.concatenate(a["px_hgt"]))), 3),
                   "word_axis_err_median": round(float(np.median(a["w_err"])), 2),
                   "word_within_10": round(float((np.asarray(a["w_err"]) <= 10).mean()), 4),
                   "word_length_ratio_median": round(float(np.median(a["w_len"])), 3),
                   "word_height_ratio_median": round(float(np.median(a["w_hgt"])), 3),
                   "word_centre_err_median": round(float(np.median(a["w_ctr"])), 3),
                   "two_tag_clusters": a["two"],
                   "today_merged_with_word": int(today_n["boxes"]["0.5"]["missed_reasons"].get("merged_with_word", 0)),
                   "today_missed": int(today_n["boxes"]["0.5"]["missed"])}
    notes = []
    for k, qn in enumerate(paths.QUARTERS):
        ys, xs = np.nonzero(quarters[::16, ::16] == k)
        if len(ys) and qn not in runs:
            notes.append((16 * float(np.median(xs)), 16 * float(np.median(ys)), f"{qn.upper()}: {NOT_TRAINED}"))
    out = review.experiments_dir() / "preview-heads"
    out.mkdir(parents=True, exist_ok=True)
    html = render(sheet, rgb, ticks, rects, today_polys, words, top, notes, runs)
    (out / "review.html").write_text(html, encoding="utf-8")
    (out / "numbers.json").write_text(json.dumps({"sheet": sheet, "quarters": top, "tick_px": TICK,
                                                  "cluster_rule": "display only: 2-means of a word's core-weighted "
                                                                  "tags; two when the means' gap is wider than the "
                                                                  "gap to the nearest Terra word's mean tag"},
                                                 indent=1) + "\n")
    print(json.dumps({"page": str(out / "review.html"), "mb": round((out / "review.html").stat().st_size / 1e6, 2),
                      "quarters": top, "seconds": round(time.perf_counter() - t0)}, indent=1))
    return out / "review.html"


def _e(s) -> str:
    return review._e(s)


def render(sheet, rgb, ticks, rects, today_polys, words, top, notes, runs) -> str:
    h, w = rgb["ink"].shape[:2]
    tiles = "".join(_jpeg_tiles(rgb[k], k) for k in ("tags", "core", "ink"))
    ov = [f'<svg id="ov" xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for level, d in ticks.items():
        if d:
            ov.append(f'<path class="ln tick" d="{d}" stroke="{TICK_COL}" stroke-opacity="{level / 10:.1f}" '
                      'stroke-width="1.2" fill="none"/>')
    ov.append(f'<path class="ln today" d="{_path(today_polys)}" stroke="{TODAY_COL}" stroke-width="1" fill="none"/>')
    ov.append(f'<path class="ln terra" d="{_path(rects)}" stroke="{TERRA_COL}" stroke-width="1" fill="none"/>')
    for x, y, t in notes:
        ov.append(f'<text x="{x:.0f}" y="{y:.0f}" font-size="72" fill="{review.NOTE}" text-anchor="middle" '
                  f'font-family="-apple-system,Helvetica,Arial,sans-serif">{_e(t)}</text>')
    ov.append("</svg>")
    rows = "".join(
        f'<tr><td><b>{qn.upper()}</b></td><td class="num">{t["terra_words"]}</td>'
        f'<td class="num">{t["px_axis_err_median"]:.1f}° · {100 * t["px_within_10"]:.0f}%</td>'
        f'<td class="num">{t["word_axis_err_median"]:.1f}° · {100 * t["word_within_10"]:.0f}%</td>'
        f'<td class="num">{t["word_length_ratio_median"]:.2f} / {t["word_height_ratio_median"]:.2f}</td>'
        f'<td class="num">{t["word_centre_err_median"]:.2f}</td>'
        f'<td class="num">{t["two_tag_clusters"]}</td>'
        f'<td class="num">{t["today_merged_with_word"]} of {t["today_missed"]}</td></tr>' for qn, t in top.items())
    table = ('<table class="rs"><tr><td>quarter</td><td class="num">Terra words</td><td class="num">axis err, px '
             '(median · within 10°)</td><td class="num">axis err, word</td><td class="num">length / height ratio, '
             'word</td><td class="num">centre err, word (heights)</td><td class="num">words with two tag clusters'
             '</td><td class="num">today: merged with a word, of missed</td></tr>' + rows + '</table>')
    what = (
        f"<p>Each quarter is drawn from its own held-out detector (arm (a), seed 0: "
        f"{', '.join(f'{q.upper()} {r.name}' for q, r in runs.items())}), the sign-free word heads. Nothing here is a "
        "word yet: no word is formed, placed or read; this shows what the heads give.</p>"
        "<p><b>Tags</b> (default view): the paper tinted by each px's 4-D tag projected on its first three principal "
        "axes (a PCA weighted by core chance, fitted on that quarter's px of this sheet; display scaling ±2.5 sd), "
        "its opacity the core chance. Touching words are told apart where their colours differ. <b>Core</b>: the paper "
        "tinted blue by the core chance itself, no level. <b>Ink</b>: the page's ink alone. In the two tinted views "
        "the letters are drawn darkened from their tint.</p>"
        f"<p><b>Axis ticks</b>: the predicted axis (no reading sign) at every {TICK}th px each way, {TICK_LEN} px "
        "long, its opacity the core chance (10 steps; the grid is for legibility only). <b>Outlines</b>: Terra's "
        "words (red) and today's boxes (teal).</p>"
        "<p><b>Two tag clusters</b> is DISPLAY ONLY: per Terra word, 2-means of its px' tags weighted by core chance; "
        "counted when the gap between the two means is wider than the gap between the word's mean tag and that of "
        "the Terra word nearest it on the page (read by the same detector): its tags differ inside it more than from "
        "its neighbour's. Such a word is one the tags would split (two words in one Terra box, or one word the heads "
        "see as two).</p>"
        "<p>Axis error and the size ratios are against Terra's word at its own px (its rectangle's px): per px, and "
        "per word (the core-weighted circular mean of the axes, the weighted geometric mean of the lengths and "
        "heights, the weighted mean of the centre votes, the centre's error in word heights). "
        "<i>Today: merged with a word</i> counts the Terra words today's boxes miss because a box merged them with "
        "another word, on this sheet and quarter.</p>")
    legend = (f'<span class="sw" style="background:{TERRA_COL}"></span> Terra\'s word box · '
              f'<span class="sw" style="background:{TODAY_COL}"></span> today\'s box · '
              f'<span class="sw" style="background:rgb({",".join(str(int(v)) for v in CORE_RGB)})"></span> core '
              'chance (Core view) · ticks: the axis')
    data_js = json.dumps({"shape": [h, w], "words": words}, ensure_ascii=False,
                         separators=(",", ":")).replace("</", "<\\/")
    return (
        '<title>Word Heads Preview</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
        f"<style>{review.STYLE}{EXTRA_STYLE}</style>\n"
        '<div class="wrap"><h1>The Mission Street sheet: what the word heads see</h1>'
        f'<p class="sub">Sheet <b>{_e(sheet)}</b> · {w:,} × {h:,} px at full resolution · a preview before word '
        'formation, readings and the review page\'s numbers.</p>'
        f'<div class="grid"><section class="box" style="grid-column:1/-1"><h2>Top line, this sheet</h2>{table}'
        '</section><section class="box"><h2>Legend</h2>' + legend + '</section>'
        f'<section class="box" style="grid-column:1/-1"><h2>What each layer is</h2>{what}</section></div>'
        '<div class="viewer"><div class="bar"><button id="zin" aria-label="Zoom in">+</button>'
        '<button id="zout" aria-label="Zoom out">−</button><button id="zfit">Fit</button>'
        '<label><input type="radio" name="v" value="tags" checked> Tags</label>'
        '<label><input type="radio" name="v" value="core"> Core</label>'
        '<label><input type="radio" name="v" value="ink"> Ink</label>'
        '<label><input type="checkbox" id="c-tick" checked> ticks</label>'
        '<label><input type="checkbox" id="c-terra" checked> Terra</label>'
        '<label><input type="checkbox" id="c-today"> today</label>'
        '<span id="zl" class="hint"></span><span class="hint">Drag or pinch to move and zoom; tap a Terra box '
        'for its numbers.</span></div>'
        f'<div id="vp"><div id="stage" class="v-tags" style="width:{w}px;height:{h}px">{tiles}{"".join(ov)}</div>'
        '<div id="info"></div></div></div></div>\n'
        f'<script type="application/json" id="data">{data_js}</script>\n<script>{PREVIEW_SCRIPT}</script>\n')


EXTRA_STYLE = """
#stage img.tile { display: none; image-rendering: auto; }
#stage.v-tags img.tags, #stage.v-core img.core, #stage.v-ink img.ink { display: block; }
#stage.no-tick .tick, #stage.no-terra .terra, #stage.no-today .today { display: none; }
.bar label { display: inline-flex; gap: 4px; align-items: center; font-size: 14px; min-height: 36px; padding: 0 4px; }
"""

PREVIEW_SCRIPT = """
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const W = D.shape[1], H = D.shape[0], MAXS = 24;
  const vp = document.getElementById('vp'), st = document.getElementById('stage');
  const info = document.getElementById('info'), zl = document.getElementById('zl');
  let s = 1, tx = 0, ty = 0, minS = 0.05;
  function apply() {
    st.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + s + ')';
    zl.textContent = (s >= 1 ? s.toFixed(1) : s.toFixed(2)) + 'x';
  }
  function fit() {
    const r = vp.getBoundingClientRect();
    s = Math.min(r.width / W, r.height / H); minS = s * 0.5;
    tx = (r.width - W * s) / 2; ty = (r.height - H * s) / 2; apply();
  }
  function zoomAt(px, py, f) {
    const ns = Math.min(MAXS, Math.max(minS, s * f));
    tx = px - (px - tx) * ns / s; ty = py - (py - ty) * ns / s; s = ns; apply();
  }
  function centre(f) { const r = vp.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, f); }
  document.getElementById('zin').onclick = () => centre(2);
  document.getElementById('zout').onclick = () => centre(0.5);
  document.getElementById('zfit').onclick = fit;
  document.querySelectorAll('input[name=v]').forEach(r => r.onchange = () => {
    st.classList.remove('v-tags', 'v-core', 'v-ink'); st.classList.add('v-' + r.value);
  });
  [['c-tick', 'no-tick'], ['c-terra', 'no-terra'], ['c-today', 'no-today']].forEach(([id, cls]) => {
    const c = document.getElementById(id);
    const set = () => st.classList.toggle(cls, !c.checked); c.onchange = set; set();
  });
  vp.addEventListener('wheel', e => {
    e.preventDefault();
    const r = vp.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)));
  }, { passive: false });
  const pts = new Map(); let moved = 0, startT = 0, pinch = null;
  vp.addEventListener('pointerdown', e => {
    if (info.contains(e.target)) return;
    vp.setPointerCapture(e.pointerId);
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pts.size === 1) { moved = 0; startT = Date.now(); vp.classList.add('drag'); }
    if (pts.size === 2) pinch = null;
  });
  vp.addEventListener('pointermove', e => {
    if (!pts.has(e.pointerId)) return;
    const p = pts.get(e.pointerId), r = vp.getBoundingClientRect();
    if (pts.size === 1) {
      tx += e.clientX - p.x; ty += e.clientY - p.y; moved += Math.abs(e.clientX - p.x) + Math.abs(e.clientY - p.y);
      p.x = e.clientX; p.y = e.clientY; apply(); return;
    }
    p.x = e.clientX; p.y = e.clientY; moved += 10;
    const [a, b] = [...pts.values()];
    const d = Math.hypot(a.x - b.x, a.y - b.y), mx = (a.x + b.x) / 2 - r.left, my = (a.y + b.y) / 2 - r.top;
    if (pinch) { tx += mx - pinch.mx; ty += my - pinch.my; zoomAt(mx, my, d / pinch.d); }
    pinch = { d: d, mx: mx, my: my };
  });
  function up(e) {
    if (!pts.has(e.pointerId)) return;
    const single = pts.size === 1;
    pts.delete(e.pointerId); pinch = null;
    if (!pts.size) vp.classList.remove('drag');
    if (single && moved < 8 && Date.now() - startT < 600) {
      const r = vp.getBoundingClientRect();
      tap((e.clientX - r.left - tx) / s, (e.clientY - r.top - ty) / s);
    }
  }
  vp.addEventListener('pointerup', up); vp.addEventListener('pointercancel', up);
  vp.addEventListener('dblclick', e => {
    const r = vp.getBoundingClientRect(); zoomAt(e.clientX - r.left, e.clientY - r.top, 2);
  });
  function inside(poly, x, y) {
    let c = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  }
  const ENT = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' };
  const esc = t => String(t == null ? '' : t).replace(/[&<>"]/g, c => ENT[c]);
  const f2 = v => v == null ? '–' : (+v).toFixed(2);
  const row = (k, v) => '<tr><td>' + k + '</td><td>' + v + '</td></tr>';
  function wordHtml(m) {
    let t = '<h3>Terra: “' + esc(m.text) + '” (' + esc(m.kind) + ', ' + m.quarter.toUpperCase() + ')</h3><table>';
    if (m.px == null) return t + row('heads', m.shown ? 'no px of its rectangle on the page' : 'not shown') +
      row('today', 'best IoU ' + f2(m.today_iou)) + '</table>';
    const c = m.cluster;
    return t + row('core chance', 'mean ' + f2(m.core_mean) + ' over its ' + m.px + ' px') +
      row('axis error', 'median ' + f2(m.axis_err_px) + '° over its px · ' + f2(m.axis_err_word) + '° as a word') +
      row('length / Terra', 'median ' + f2(m.length_px) + ' per px · ' + f2(m.length_word) + ' as a word') +
      row('height / Terra', 'median ' + f2(m.height_px) + ' per px · ' + f2(m.height_word) + ' as a word') +
      row('centre error', 'median ' + f2(m.centre_px) + ' word heights per px vote · ' + f2(m.centre_word) +
          ' as a word') +
      row('tag clusters', (c.two ? '<b>two</b>' : 'one') + ' (display only: inner gap ' + f2(c.gap) +
          ', the smaller part ' + Math.round(100 * c.minor) + '%; gap to the nearest word “' + esc(c.neighbour) +
          '” ' + f2(c.neighbour_gap) + ')') +
      row('today', 'best IoU with today\\'s boxes ' + f2(m.today_iou)) + '</table>';
  }
  function tap(x, y) {
    const hits = D.words.filter(m => inside(m.poly, x, y)).map(wordHtml);
    if (!hits.length) { info.classList.remove('on'); return; }
    info.innerHTML = '<button class="x" aria-label="Close">×</button><div class="pt">page px ' + Math.round(x) +
      ', ' + Math.round(y) + '</div>' + hits.join('');
    info.querySelector('.x').onclick = () => info.classList.remove('on');
    info.classList.add('on');
  }
  fit();
  window.addEventListener('resize', () => { if (s <= minS * 2.01) fit(); });
})();
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", default=SHEET)
    ap.add_argument("--name", default="a")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.sheet, a.name, a.seed)


if __name__ == "__main__":
    main()
