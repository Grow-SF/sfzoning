"""Issue #71, the learned "more likely reading wins" (owner 2026-11-21: no thresholds; "Yes both").  Step 1, a
measurement: on the training sheets, each read by heads that did not train on it, how often is a whole rule (or a
whole blob read as a label box) really what the model's reading says?

  1. dev chances: the committed training sheets dealt round-robin into FOLDS folds; per fold, HEAD_SEEDS heads on the
     other folds; the held sheets' patch logits saved under data/frame-dev/logits/.
  2. per training sheet: every rule (`rules.rule_labels`) with its ink, the model's mean chances along it and the
     share of its ink the label calls frame or box; every blob the model reads as "box" (the most likely class) with
     its ink, mean box chance and the share of its ink the label calls box; and the same for ink on no rule, binned by
     the chance of deletion.
  3. the tables: how often the reading is right, by the reading's strength.

    uv run python scripts/spikes/issue71_dev.py
"""
import json
import subprocess
import time

import numpy as np
import torch
from scipy import ndimage as ndi

from sfzmaps import gpu
from sfzmaps.frame import data, layout, rules
from sfzmaps.frame import run as fr

OUT = data.DATA_DIR / "frame-dev"
FOLDS = 4
T = 1.540
BINS = np.array([0, .05, .15, .25, .35, .45, .55, .65, .75, .85, .95, 1.0001])


def train_sheets():
    files = subprocess.run(["git", "ls-files", "inputs/frame-labels"], capture_output=True, text=True).stdout.split()
    return sorted((f.split("/")[-2], f.split("/")[-1][:-5]) for f in files if f.endswith(".json"))


def zpath(sid):
    return OUT / "logits" / f"{sid[0]}__{sid[1]}.npz"


def dev_logits(train):
    todo = [s for s in train if not zpath(s).exists()]
    if not todo:
        return
    feats, targets = fr._features_and_targets(train, data.OUT)
    (OUT / "logits").mkdir(parents=True, exist_ok=True)
    with gpu.gpu_turn(True, "mps"):
        for k in range(FOLDS):
            held = train[k::FOLDS]
            rest = [s for s in train if s not in held]
            t0 = time.perf_counter()
            heads = [fr._train(feats, targets, rest, seed, "mps")[0] for seed in layout.HEAD_SEEDS]
            for s in held:
                np.savez_compressed(zpath(s), z=np.stack([layout.head_logits(h, feats[s]) for h in heads]).astype(np.float32))
            print(f"fold {k}: {len(held)} sheets held, heads on {len(rest)} in {time.perf_counter() - t0:.0f} s", flush=True)


def tables(sid):
    torch.set_num_threads(1)
    lum = data.load_luminance(data.page_path(*sid))
    ink = (1.0 - lum).astype(np.float64)
    p = fr.chances(list(np.load(zpath(sid))["z"]), lum.shape, T)
    cls = data.class_mask(data.load_label(*sid), data.load_remove(*sid), lum.shape)
    lab, n = rules.rule_labels(lum)
    flat = lab.ravel()
    w = ink.ravel()
    tot = np.bincount(flat, weights=w, minlength=n + 1)
    q = (p[1] + p[2]).ravel().astype(np.float64)
    mq = np.bincount(flat, weights=w * q, minlength=n + 1)
    tr = np.bincount(flat, weights=w * (cls.ravel() != 0), minlength=n + 1)
    sl = ndi.find_objects(lab)
    rule_rows = []
    for i in range(1, n + 1):
        if tot[i] <= 0 or sl[i - 1] is None:
            continue
        hh, ww = sl[i - 1][0].stop - sl[i - 1][0].start, sl[i - 1][1].stop - sl[i - 1][1].start
        rule_rows.append([float(tot[i]), float(mq[i] / tot[i]), float(tr[i] / tot[i]), int(max(hh, ww)), int(ww >= hh)])
    # ink on no rule, by the chance of deletion
    off = flat == 0
    b = np.digitize(q[off], BINS) - 1
    rest_ink = np.bincount(b, weights=w[off], minlength=len(BINS) - 1)[:len(BINS) - 1]
    rest_true = np.bincount(b, weights=w[off] * (cls.ravel()[off] != 0), minlength=len(BINS) - 1)[:len(BINS) - 1]
    # blobs the model reads as "box" (box is the most likely class)
    box = (p[2] > p[0]) & (p[2] > p[1])
    bl, m = ndi.label(box, structure=np.ones((3, 3), bool))
    blob_rows = []
    if m:
        bf = bl.ravel()
        bt = np.bincount(bf, weights=w, minlength=m + 1)
        bp = np.bincount(bf, weights=w * p[2].ravel(), minlength=m + 1)
        by = np.bincount(bf, weights=w * (cls.ravel() == 2), minlength=m + 1)
        px = np.bincount(bf, minlength=m + 1)
        for i in range(1, m + 1):
            if px[i] >= 50 and bt[i] > 0:
                blob_rows.append([float(bt[i]), float(bp[i] / bt[i]), float(by[i] / bt[i]), int(px[i])])
    return {"sheet": f"{sid[0]}/{sid[1]}", "rules": rule_rows, "rest_ink": rest_ink.tolist(),
            "rest_true": rest_true.tolist(), "blobs": blob_rows}


def show(title, rows, xi=1, yi=2, wi=0):
    a = np.array(rows, float)
    print(title)
    if not len(a):
        print("  none")
        return
    b = np.digitize(a[:, xi], BINS) - 1
    for k in range(len(BINS) - 1):
        m = b == k
        if m.any():
            wsum = a[m, wi].sum()
            print(f"  reading {BINS[k]:.2f}-{min(BINS[k + 1], 1):.2f}: {int(m.sum()):5d} objects, ink {wsum:11.0f}, really so: "
                  f"{100 * (a[m, wi] * a[m, yi]).sum() / wsum:5.1f}% of ink; objects mostly so: "
                  f"{100 * (a[m, yi] > 0.5).mean():5.1f}%", flush=True)


def main():
    train = train_sheets()
    print(len(train), "training sheets", flush=True)
    dev_logits(train)
    res = fr.pool_map(tables, train, label="sheet")
    (OUT / "tables.json").write_text(json.dumps(res))
    show("RULES: the model's mean chance that a rule is frame or box -> how often it is", [r for s in res for r in s["rules"]])
    long_ = [r for s in res for r in s["rules"] if r[3] >= 1000]
    show("RULES at least 1000 px long", long_)
    show("BLOBS the model reads as box: its mean box chance -> how often it is a labelled box", [r for s in res for r in s["blobs"]])
    ri = np.sum([s["rest_ink"] for s in res], 0)
    rt = np.sum([s["rest_true"] for s in res], 0)
    print("INK ON NO RULE, by the model's chance of deletion -> share really frame or box")
    for k in range(len(BINS) - 1):
        if ri[k] > 0:
            print(f"  {BINS[k]:.2f}-{min(BINS[k + 1], 1):.2f}: ink {ri[k]:12.0f}, really so {100 * rt[k] / ri[k]:5.1f}%", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
