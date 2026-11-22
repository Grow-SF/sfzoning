"""Issue #71 trial: the learned "more likely reading wins" (owner 2026-11-21: no thresholds).

From the training sheets read by heads that did not train on them (`issue71_dev.py`, data/frame-dev/tables.json):
  rules      the chance that a whole rule is frame or box, given the model's mean reading along it and its length
             (a logistic fit, each rule weighted by its ink);
  other ink  the chance that ink on no rule is frame or box, given the model's reading (a logistic fit on the same
             sheets' ink).
On a page, every rule's ink takes its rule's learned chance, other ink its learned chance, and the straight-line cut
runs at even odds: the more likely answer wins, with no cost ratio.  Compared with the committed cut (3x) on the same
heads (48 labels).  Throwaway; prints as it goes.

    FRAME_HEADS=frame-look48 uv run python scripts/spikes/issue71_learned.py
"""
import json
import os
import subprocess

import numpy as np
import torch
from scipy import ndimage as ndi
from scipy.optimize import minimize

from sfzmaps.frame import blocks as blk
from sfzmaps.frame import data, decide, geometry, layout, look, paper, rules, shape
from sfzmaps.frame import run as fr

SRC = data.DATA_DIR / os.environ.get("FRAME_HEADS", "frame-look48")
OUT = data.DATA_DIR / os.environ.get("FRAME_OUT", "frame-learned")
DEV = data.DATA_DIR / "frame-dev" / "tables.json"
BEFORE = data.DATA_DIR / "frame-test48" / "rows.json"
TEST = data.LABELS.parent / "frame-labels-test"
T = 1.540
PARAMS = shape.Params(lam_notch=1e4, beta=3, lam_block=1e3)
FLAGGED = [("2000-digital-su", "2000-digital-su-p11"), ("1975", "20180102114142444-p18"), ("1948", "4"),
           ("1980-1990-su", "1980-1990-su-p01")]
EPS = 1e-4
BINS = np.array([0, .05, .15, .25, .35, .45, .55, .65, .75, .85, .95, 1.0])


def logit(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def fit_logistic(X, y, w):
    """Weighted logistic regression with soft targets y in [0, 1]: the coefficients."""
    X = np.asarray(X, float); y = np.asarray(y, float); w = np.asarray(w, float) / np.sum(w)

    def nll(b):
        z = X @ b
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)))

    return minimize(nll, np.zeros(X.shape[1]), method="BFGS").x


def fits():
    res = json.load(open(DEV))
    r = np.array([row for s in res for row in s["rules"]], float)          # ink, reading, really, length, horizontal
    Xr = np.stack([np.ones(len(r)), logit(r[:, 1]), np.log(r[:, 3])], 1)
    br = fit_logistic(Xr, r[:, 2], r[:, 0])
    ri = np.sum([s["rest_ink"] for s in res], 0); rt = np.sum([s["rest_true"] for s in res], 0)
    mid = (BINS[:-1] + BINS[1:]) / 2
    bo = fit_logistic(np.stack([np.ones(len(mid)), logit(mid)], 1), rt / np.maximum(ri, 1e-9), ri)
    return br, bo


def learned(p, lum, br, bo):
    """The chances after the two learned corrections: (3, h, w)."""
    q = p[1] + p[2]
    c = 1.0 / (1.0 + np.exp(-(bo[0] + bo[1] * logit(q))))                   # ink on no rule
    lab, n = rules.rule_labels(lum)
    if n:
        ink = (1.0 - lum).astype(np.float64).ravel()
        flat = lab.ravel()
        tot = np.bincount(flat, weights=ink, minlength=n + 1)
        m = np.bincount(flat, weights=ink * q.ravel(), minlength=n + 1) / np.maximum(tot, 1e-9)
        length = np.ones(n + 1)
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            if sl is not None:
                length[i] = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        cr = 1.0 / (1.0 + np.exp(-(br[0] + br[1] * logit(m) + br[2] * np.log(length))))
        on = lab > 0
        c[on] = cr[lab[on]]
    share2 = np.where(q > 1e-6, p[2] / np.maximum(q, 1e-6), 0.0)
    return np.stack([1.0 - c, c * (1.0 - share2), c * share2]).astype(np.float32)


def one(job):
    sid, zs, bounds, maps, scored, br, bo = job
    torch.set_num_threads(1)
    lum = data.load_luminance(data.page_path(*sid))
    p = learned(fr.chances(zs, lum.shape, T), lum, br, bo)
    decide.COST_FRAME = geometry.COST_FRAME = paper.COST_FRAME = 1.0          # even odds
    decide.cut_level.__defaults__ = (1.0, 1.0)
    dec, frm, ns, bs = shape.decide(p, lum, PARAMS, bounds, maps)
    fr.write_outputs(OUT, sid, p, dec, maps, page=data.page_path(*sid), fitted=(frm, ns, bs))
    _, deleted, cut = look._load_outputs(OUT, sid)
    look.sheet_picture(lum, deleted.astype(np.float32), deleted, 1300, cut).save(OUT / f"{sid[0]}_{sid[1][-6:]}.png")
    r = {"phi": round(frm.phi, 2), "notches": len(ns), "boxes": [(b.kind, round(b.theta, 1), round(b.gain)) for b in bs]}
    if scored:
        polys, rem = data.load_label(*sid, TEST), data.load_remove(*sid, TEST)
        cls = data.class_mask(polys, rem, lum.shape)
        ink = 1.0 - lum
        d = dec.deleted
        r.update(frame_ink=float(ink[cls == 1].sum()), frame_left=float(ink[(cls == 1) & ~d].sum()),
                 box_ink=float(ink[cls == 2].sum()), box_left=float(ink[(cls == 2) & ~d].sum()),
                 map_ink=float(ink[cls == 0].sum()), map_cut=float(ink[(cls == 0) & d].sum()),
                 per_box=[round(float(ink[(data.class_mask(polys, [x], lum.shape) == 2) & d].sum()
                                      / max(ink[data.class_mask(polys, [x], lum.shape) == 2].sum(), 1e-9)), 2) for x in rem])
    return {"sheet": f"{sid[0]}/{sid[1]}", "learned": r}


def main():
    br, bo = fits()
    print("rules: chance = sigmoid(%.2f + %.2f logit(reading) + %.2f log(length px))" % tuple(br))
    for m in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        print("   a rule read %.2f: %s" % (m, ", ".join("%d px -> %.0f%%" % (L, 100 / (1 + np.exp(-(br[0] + br[1] * logit(m) + br[2] * np.log(L))))) for L in (250, 1000, 3000))))
    print("other ink: chance = sigmoid(%.2f + %.2f logit(reading)); even odds at a reading of %.2f" % (bo[0], bo[1], 1 / (1 + np.exp(bo[0] / bo[1]))), flush=True)
    files = subprocess.run(["git", "ls-files", "inputs/frame-labels"], capture_output=True, text=True).stdout.split()
    train = sorted((f.split("/")[-2], f.split("/")[-1][:-5]) for f in files if f.endswith(".json"))
    heads = []
    for sd in torch.load(SRC / "heads.pt"):
        hd = layout.new_head(); hd.load_state_dict(sd); heads.append(hd.eval())
    bounds = blk.block_bounds(train, root=data.LABELS)
    counts = data.map_counts(data.PAGE_KINDS)
    test = sorted((p.parent.name, p.stem) for p in TEST.glob("*/*.json"))
    jobs = [(s, tuple(layout.head_logits(hd, fr.load_features(data.OUT, s)) for hd in heads), bounds, counts[s],
             s in test, br, bo) for s in test + FLAGGED]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = fr.pool_map(one, jobs, label="page")
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
    before = {r["sheet"]: r["now"] for r in json.load(open(BEFORE))}

    def fmt(m):
        s = f"angle {m['phi']:+.2f}"
        if "frame_ink" in m:
            s += f" frame left {100 * m['frame_left'] / max(m['frame_ink'], 1e-9):.2f}%"
            if m["box_ink"]:
                s += f" boxes left {100 * m['box_left'] / m['box_ink']:.0f}% {m.get('per_box', '')}"
            s += f" map cut {100 * m['map_cut'] / m['map_ink']:.3f}%"
        return s + f" boxes {m['boxes']}"

    for r in rows:
        print(f"{r['sheet'][-26:]}: | 3x: {fmt(before[r['sheet']])} | learned: {fmt(r['learned'])}", flush=True)
    for name in ("3x", "learned"):
        ms = [before[r["sheet"]] if name == "3x" else r["learned"] for r in rows if "frame_ink" in r["learned"]]
        s = {k: sum(m[k] for m in ms) for k in ("frame_ink", "frame_left", "box_ink", "box_left", "map_ink", "map_cut")}
        print(f"POOLED {name}: frame ink left {100 * s['frame_left'] / s['frame_ink']:.2f}%, box ink left "
              f"{100 * s['box_left'] / s['box_ink']:.1f}%, map ink cut {100 * s['map_cut'] / s['map_ink']:.3f}%", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
