"""Word formation (`wordread.words.core_words`) on a page-sized synthetic page: its cost (time and peak memory per
stage) and its boxes against the words painted on it.

The page (a 2000 sheet's size by default, 5100 x 3300 px) holds words on a grid: codes (60 x 20 px), street names
(200 x 26), at 0, 30 and 90 degrees, a sixth of them touching in pairs; and fragments (8 x 8 px patches at core chance
0.3 voting for their own small boxes).  The background is at core chance 0.001.  The votes are the word heads' as
`model.word_geometry` decodes them (Task 5): each px's offset to its word's centre in px, the length and height in px,
and the axis in degrees in [0, 180); noisy (offsets 0.1 heights, log sizes 0.05, axes 3 degrees, tags 0.3), drawn
once per 2 x 2 px; the spreads match them (cell 4) and the prior is (1, -60).  Each box is matched to the painted word
or fragment it meets best (IoU, for display only), and the summary counts: painted words with one box, with more than
one (and whether those boxes are splits of it or near copies of each other: their pairwise IoU and direction
difference), with none; boxes meeting nothing painted; and each box's share of its area on painted words or fragments
other than its own (claims across neighbours).  For each pair of boxes on one painted word: whether the search
weighed joining them (their supports meet) and the join's exact score (> 0: the search should have joined them).
`core_words` runs once, end to end, its stages (proposals, search, final fit) timed as they run, with the process's
peak memory after each.

  uv run python scripts/wordread/page_synthetic.py [--width 5100 --height 3300 --fragments 2000 --seed 0]

Writes data/solver/experiments/wordread/page_synthetic/<seed>/summary.json and boxes.parquet (never committed)."""
import argparse
import json
import logging
import math
import resource
import sys
import time

import cv2
import numpy as np

from sfzmaps.solver import paths
from sfzmaps.solver.wordread import words

log = logging.getLogger("page_synthetic")
OUT = paths.SOLVER_DIR / "experiments" / "wordread" / "page_synthetic"
SPREADS = words.Spreads(centre=0.1, length=0.05, height=0.05, angle=3.0, tag=0.3, cell=4.0, edge=0.3,
                        core_share=0.15, outlier=0.01, centre_bias=(0.0, 0.0), length_bias=0.0, height_bias=0.0,
                        angle_bias=0.0,
                        background_centre=1.0, background_length=(math.log(20.0), 0.5),
                        background_height=(math.log(10.0), 0.5), background_tag=1.0,
                        background_tag_mean=(0.0, 0.0, 0.0, 0.0))
PRIOR = words.WordPrior(slope=1.0, log_prior=-60.0)


def peak_gb() -> float:
    """The process's peak resident memory so far (macOS reports bytes, Linux KiB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 30 if sys.platform == "darwin" else peak / 2 ** 20


def page(width: int, height: int, fragments: int, seed: int):
    """The synthetic page: core logit, geo (6, h, w), tag (4, h, w), ink, and the painted rectangles with their kind
    ("word" or "fragment")."""
    rng = np.random.default_rng([seed, 11])
    logit = np.full((height, width), np.log(1e-3 / (1 - 1e-3)), np.float32)
    geo = np.empty((5, height, width), np.float32)                  # as model.word_geometry decodes the heads
    geo[2] = np.exp(rng.normal(math.log(20.0), 0.5, (height, width)))
    geo[3] = np.exp(rng.normal(math.log(10.0), 0.5, (height, width)))
    geo[:2] = rng.normal(0, 1.0, (2, height, width)) * geo[3]
    geo[4] = rng.uniform(0, 180.0, (height, width))
    tag = rng.normal(0, 1.0, (4, height, width)).astype(np.float32)
    ink = np.zeros((height, width), bool)
    painted = []

    def paint(cx, cy, length, h, angle, w, tag_vec, kind, noise=0.1, size_noise=0.05, angle_noise=3.0,
              tag_noise=0.3):
        a = np.deg2rad(angle)
        v, n = np.array([np.cos(a), -np.sin(a)]), np.array([np.sin(a), np.cos(a)])
        c = np.array([cx, cy])
        corners = np.array([c - v * length / 2 - n * h / 2, c + v * length / 2 - n * h / 2,
                            c + v * length / 2 + n * h / 2, c - v * length / 2 + n * h / 2])
        x0, y0 = np.maximum(np.floor(corners.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(corners.max(0)).astype(int) + 1, [width, height])
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        p = np.stack([xx, yy], -1).astype(float)
        pa, pb = (p - corners[0]) @ v, (p - corners[0]) @ n
        on = (pa >= 0) & (pa <= length) & (pb >= 0) & (pb <= h)
        if not on.any():
            return
        q = p[on]
        _, b = np.unique((yy[on] // 2) * width + xx[on] // 2, return_inverse=True)
        draws = b.max() + 1
        off = (c - q) / h + rng.normal(0, noise, (draws, 2))[b]
        sub = (slice(y0, y1), slice(x0, x1))
        length_px = length * np.exp(rng.normal(0, size_noise, draws)[b])
        height_px = h * np.exp(rng.normal(0, size_noise, draws)[b])
        geo[0][sub][on], geo[1][sub][on] = off[:, 0] * height_px, off[:, 1] * height_px
        geo[2][sub][on], geo[3][sub][on] = length_px, height_px
        geo[4][sub][on] = (angle + rng.normal(0, angle_noise, draws)[b]) % 180.0
        for ch in range(4):
            tag[ch][sub][on] = tag_vec[ch] + rng.normal(0, tag_noise, int(on.sum()))
        logit[sub][on] = np.log(w / (1 - w))
        ink[sub][on & (np.abs(pb - h / 2) < h * 0.3) & (pa > 3) & (pa < length - 3)] = True
        painted.append((corners, kind))

    for gy in range(60, height - 60, 75):
        for gx in range(80, width - 160, 150):
            kind = rng.integers(0, 6)
            tv = rng.normal(0, 3, 4)
            if kind == 0:
                paint(gx + 60, gy, 200, 26, 0, 0.95, tv, "word")
            elif kind == 1:                                  # a touching pair, one tag
                paint(gx, gy, 60, 20, 0, 0.95, tv, "word")
                paint(gx + 60, gy, 60, 20, 0, 0.95, tv, "word")
            elif kind == 2:
                paint(gx + 40, gy, 60, 20, 30, 0.95, tv, "word")
            elif kind == 3:
                paint(gx + 30, gy, 20, 60, 90, 0.95, tv, "word")
            else:
                paint(gx + 30, gy, 60, 20, 0, 0.95, tv, "word")
    for _ in range(fragments):
        paint(rng.uniform(10, width - 10), rng.uniform(10, height - 10), 8, 8, 0, 0.3, rng.normal(0, 3, 4),
              "fragment", noise=0.3)
    return logit, geo, tag, ink, painted


def area(c) -> float:
    return 0.5 * abs(float(np.cross(c[1] - c[0], c[3] - c[0])))


def inter(a, b) -> float:
    return float(cv2.intersectConvexConvex(np.float32(a), np.float32(b))[0])


def iou(a, b) -> float:
    i = inter(a, b)
    return float(i / (area(a) + area(b) - i)) if i > 0 else 0.0


def match(boxes: np.ndarray, painted: list) -> dict:
    """Each box's best painted rectangle (IoU, display only) and the counts of the module's summary."""
    centres = np.array([c.mean(0) for c, _ in painted])
    kinds = np.array([k for _, k in painted])
    best = []
    for g in boxes:
        near = np.flatnonzero(np.abs(centres - g.mean(0)).max(1) < 300)
        scores = [iou(g, painted[j][0]) for j in near]
        best.append((int(near[int(np.argmax(scores))]), max(scores)) if scores and max(scores) > 0 else (-1, 0.0))
    owner = np.array([b for b, _ in best])
    per = np.bincount(owner[owner >= 0], minlength=len(painted))
    # claims across neighbours: the share of each box's area on painted words or fragments other than its own
    across = []
    for g, (j, _) in zip(boxes, best):
        near = np.flatnonzero(np.abs(centres - g.mean(0)).max(1) < 300)
        on = sum(inter(g, painted[k][0]) for k in near if k != j)
        across.append(on / area(g) if area(g) > 0 else 0.0)
    across = np.array(across)
    words_ = kinds == "word"
    many = np.flatnonzero(words_ & (per > 1))
    pairs = []
    for j in many:                                   # the boxes sharing a painted word: copies, or its parts?
        mine = np.flatnonzero(owner == j)
        for x in range(len(mine)):
            for y in range(x + 1, len(mine)):
                a, b = boxes[mine[x]], boxes[mine[y]]
                da = np.degrees(np.arctan2(*(a[1] - a[0])[::-1]) - np.arctan2(*(b[1] - b[0])[::-1]))
                pairs.append({"word": int(j), "kind": str(kinds[j]), "boxes": [int(mine[x]), int(mine[y])],
                              "iou": iou(a, b),
                              "turn": float(abs((da + 180) % 360 - 180)),
                              "ious_with_word": [best[mine[x]][1], best[mine[y]][1]],
                              "across": [float(across[mine[x]]), float(across[mine[y]])]})
    return {"painted_words": int(words_.sum()), "painted_fragments": int((~words_).sum()), "boxes": len(boxes),
            "words_with_one_box": int((words_ & (per == 1)).sum()), "words_with_more": int(len(many)),
            "words_with_none": int((words_ & (per == 0)).sum()), "boxes_on_nothing": int((owner < 0).sum()),
            "boxes_on_fragments": int(np.isin(owner, np.flatnonzero(~words_)).sum()),
            "across_quantiles": ({q: float(np.quantile(across, q)) for q in (0.5, 0.9, 0.99, 1.0)}
                                 if len(across) else {}),
            "boxes_across": [{"box": int(k), "best": int(owner[k]), "iou": float(best[k][1]),
                              "across": float(across[k])} for k in np.argsort(-across)[:50]],
            "pairs_sharing_a_word": pairs}


def timed(stages: dict, name: str, fn, count=None):
    """`fn` wrapped to log its time and the peak memory after it into stages[name] (and the extra counts)."""
    def run(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        stages[name] = {"seconds": round(time.perf_counter() - t0, 1), "peak_gb": round(peak_gb(), 2),
                        **(count(out) if count else {})}
        log.info("%s: %s", name, stages[name])
        return out
    return run


def joins(V, final: list, boxes: np.ndarray, pairs: list) -> list:
    """For each pair of boxes sharing a painted word: whether the search weighed joining them (their supports meet)
    and the join's exact score change (the one-word fit from each start, the better kept, every other word held):
    > 0 means the search should have joined them."""
    centres = np.array([words._box(w["prm"]).mean(0) for w in final])
    scale = PRIOR.slope / SPREADS.cell
    out = []
    for pair in pairs:
        a, b = (int(np.argmin(np.abs(centres - boxes[k].mean(0)).sum(1))) for k in pair["boxes"])
        held = [w for k, w in enumerate(final) if k not in (a, b)]
        best = -np.inf
        for start in (final[a], final[b]):
            new = words._fit(V, [start], held)
            d, _ = words._change(V, [final[a], final[b]], new, held)
            best = max(best, d)
        out.append({**pair, "weighed": bool(words._overlap(V, final[a]["prm"], final[b]["prm"])),
                    "join_score": float(scale * best - PRIOR.log_prior)})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--width", type=int, default=5100)
    ap.add_argument("--height", type=int, default=3300)
    ap.add_argument("--fragments", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t0 = time.perf_counter()
    logit, geo, tag, ink, painted = page(args.width, args.height, args.fragments, args.seed)
    stages = {"draw": {"seconds": round(time.perf_counter() - t0, 1), "peak_gb": round(peak_gb(), 2)}}
    log.info("%d painted on %d x %d in %.0f s", len(painted), args.width, args.height, stages["draw"]["seconds"])
    # one end-to-end run of core_words, its stages timed as they run
    fits = [0]
    kept = {}
    real = {name: getattr(words, name) for name in ("_proposals", "_search", "_final", "_fit")}

    def fit(*a, **k):
        fits[0] += 1
        return real["_fit"](*a, **k)

    def final(V, found):
        out = real["_final"](V, found)
        kept.update(V=V, final=out)
        return out

    words._fit = fit
    words._proposals = timed(stages, "proposals", real["_proposals"], lambda P: {"basins": int(len(P.first))})
    words._search = timed(stages, "search", real["_search"], lambda f: {"words": len(f), "fits": fits[0]})
    words._final = timed(stages, "final", final, lambda f: {"words": len(f)})
    try:
        t0 = time.perf_counter()
        df = words.core_words(logit, geo, tag, ink, SPREADS, PRIOR)
        stages["core_words"] = {"seconds": round(time.perf_counter() - t0, 1), "peak_gb": round(peak_gb(), 2),
                                "boxes": len(df), "fits": fits[0]}
        log.info("core_words end to end: %s", stages["core_words"])
    finally:
        for name, fn in real.items():
            setattr(words, name, fn)
    boxes = df[[f"g{i}{a}" for i in range(4) for a in "xy"]].to_numpy(float).reshape(-1, 4, 2)
    matched = match(boxes, painted)
    matched["pairs_sharing_a_word"] = joins(kept["V"], kept["final"], boxes, matched["pairs_sharing_a_word"])
    summary = {"stages": stages, "match": matched, "spreads": SPREADS.__dict__, "prior": PRIOR.__dict__,
               "page": [args.width, args.height], "seed": args.seed}
    out = OUT / str(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    df.to_parquet(out / "boxes.parquet", index=False)
    log.info("match: %s", {k: v for k, v in summary["match"].items()
                           if k not in ("pairs_sharing_a_word", "boxes_across")})
    for pair in matched["pairs_sharing_a_word"]:
        log.info("pair on painted word %d (%s): IoU %.2f, turn %.1f, IoUs with it %s, across %s, join weighed %s, "
                 "join score %.1f", pair["word"], pair["kind"], pair["iou"], pair["turn"],
                 [round(x, 2) for x in pair["ious_with_word"]], [round(x, 2) for x in pair["across"]],
                 pair["weighed"], pair["join_score"])
    return summary


if __name__ == "__main__":
    main()
