"""Issue #71, owner's option 2 for the tilted frame on 2000-digital-su p11: "treat a straight printed rule as one
thing: if most of it is frame, all of it is".  Throwaway trial; prints as it goes.

A rule: a connected stroke of ink that is straight and long along the page's axes (within the frame search's angle
range) and thin across.  Each rule's ink then shares one set of chances, the ink-weighted mean of the model's chances
along it, and the straight-line cut runs on those.  Two readings are compared with the cut as it is ("before"):
  one   every rule takes its mean chances (a rule is one thing, either way);
  most  only a rule whose mean chance of deletion is over one half takes its mean (most of it is frame: all of it is).

    uv run python scripts/spikes/issue71_rules.py p11     # the flagged pages
    uv run python scripts/spikes/issue71_rules.py test    # the 22 test sheets, scored
"""
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

from sfzmaps.frame import blocks as blk
from sfzmaps.frame import data, layout, look, shape
from sfzmaps.frame import run as fr

SRC = data.DATA_DIR / "frame-look39"
OUT = data.DATA_DIR / os.environ.get("RULE_OUT", "frame-rules")
MODES = tuple(os.environ.get("RULE_MODES", "before,one,most").split(","))
TEST = data.LABELS.parent / "frame-labels-test"
T = 1.540
PARAMS = shape.Params(lam_notch=1e4, beta=3, lam_block=1e3)
RUN = int(os.environ.get("RULE_RUN", 201))        # px: a rule is ink that runs at least this far along its direction (trial setting)
ACROSS = 61      # px: ... and is thinner than this across it (trial setting; the thickest labelled bars are ~30 px)
ANGLES = np.arange(-shape.ANGLE_MAX, shape.ANGLE_MAX + 1e-9, 0.5)
FLAGGED = [("2000-digital-su", "2000-digital-su-p11"), ("1975", "20180102114142444-p18"), ("1948", "4"),
           ("1980-1990-su", "1980-1990-su-p01")]


def rule_labels(lum):
    """(labels, n): every px of a rule carries its rule's number (1..n), other px 0."""
    ink = ((1.0 - lum) > 0.5).astype(np.uint8)
    h, w = ink.shape
    c = ((w - 1) / 2, (h - 1) / 2)
    H, V = np.zeros_like(ink), np.zeros_like(ink)
    for a in ANGLES:
        M = cv2.getRotationMatrix2D(c, float(a), 1.0)
        R = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_NEAREST)
        Mi = cv2.invertAffineTransform(M)
        H |= cv2.warpAffine(cv2.morphologyEx(R, cv2.MORPH_OPEN, np.ones((1, RUN), np.uint8)), Mi, (w, h), flags=cv2.INTER_NEAREST)
        V |= cv2.warpAffine(cv2.morphologyEx(R, cv2.MORPH_OPEN, np.ones((RUN, 1), np.uint8)), Mi, (w, h), flags=cv2.INTER_NEAREST)
    tall = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((ACROSS, 1), np.uint8))
    wide = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, ACROSS), np.uint8))
    Hr = H & ink & (1 - tall)
    Vr = V & ink & (1 - wide)
    nh, lh = cv2.connectedComponents(Hr, connectivity=8)
    nv, lv = cv2.connectedComponents(Vr, connectivity=8)
    lab = lh.astype(np.int32)
    lab[(lv > 0) & (lab == 0)] = lv[(lv > 0) & (lab == 0)] + (nh - 1)
    return lab, nh - 1 + nv - 1


def pooled(p, lum, lab, n, mode):
    """p with each rule's px set to the rule's ink-weighted mean chances (`mode` "one": every rule; "most": only
    rules whose mean chance of deletion is over one half).  Returns (p, the rules changed)."""
    ink = (1.0 - lum).astype(np.float64).ravel()
    flat = lab.ravel()
    tot = np.bincount(flat, weights=ink, minlength=n + 1)
    means = np.stack([np.bincount(flat, weights=ink * p[k].ravel(), minlength=n + 1) / np.maximum(tot, 1e-9)
                      for k in range(3)])
    use = np.ones(n + 1, bool) if mode == "one" else (means[1] + means[2]) > 0.5
    use[0] = False
    m = use[lab]
    q = p.copy()
    for k in range(3):
        q[k][m] = means[k][lab[m]]
    return q, int(use.sum())


def one(job):
    sid, zs, bounds, maps, scored = job
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    lum = data.load_luminance(data.page_path(*sid))
    p0 = fr.chances(zs, lum.shape, T)
    t = time.perf_counter()
    lab, n = rule_labels(lum)
    secs = time.perf_counter() - t
    row = {"sheet": f"{sid[0]}/{sid[1]}", "rules": n, "rule_seconds": round(secs, 1)}
    if scored:
        polys, rem = data.load_label(*sid, TEST), data.load_remove(*sid, TEST)
        cls = data.class_mask(polys, rem, lum.shape)
        ink = 1.0 - lum
    for mode in MODES:
        p, changed = (p0, 0) if mode == "before" else pooled(p0, lum, lab, n, mode)
        dec, frm, ns, bs = shape.decide(p, lum, PARAMS, bounds, maps)
        out = OUT / mode
        fr.write_outputs(out, sid, p, dec, maps, page=data.page_path(*sid), fitted=(frm, ns, bs))
        _, deleted, polys_cut = look._load_outputs(out, sid)
        look.sheet_picture(lum, deleted.astype(np.float32), deleted, 1300, polys_cut).save(out / f"{sid[0]}_{sid[1][-6:]}.png")
        r = {"phi": round(frm.phi, 2), "bow": {s: round(frm.bow[s], 1) for s in shape.SIDES if frm.framed(s)},
             "notches": len(ns), "boxes": len(bs), "rules_changed": changed}
        if scored:
            d = dec.deleted
            r.update(frame_ink=float(ink[cls == 1].sum()), frame_left=float(ink[(cls == 1) & ~d].sum()),
                     box_ink=float(ink[cls == 2].sum()), box_left=float(ink[(cls == 2) & ~d].sum()),
                     map_ink=float(ink[cls == 0].sum()), map_cut=float(ink[(cls == 0) & d].sum()))
        row[mode] = r
    return row


def main():
    what = sys.argv[1]
    train = sorted({tuple(f.split("/")[-2:]) for f in subprocess.run(
        ["git", "ls-files", "inputs/frame-labels"], capture_output=True, text=True).stdout.split() if f.endswith(".json")})
    train = [(s, h[:-5]) for s, h in train]              # the committed labels only: the heads' own 39
    assert len(train) == 39, len(train)
    heads = []
    for sd in torch.load(SRC / "heads.pt"):
        hd = layout.new_head(); hd.load_state_dict(sd); heads.append(hd.eval())
    bounds = blk.block_bounds(train, root=data.LABELS)
    counts = data.map_counts(data.PAGE_KINDS)
    pages = FLAGGED if what == "p11" else sorted((p.parent.name, p.stem) for p in TEST.glob("*/*.json"))
    jobs = [(s, tuple(layout.head_logits(hd, fr.load_features(data.OUT, s)) for hd in heads), bounds, counts[s],
             what == "test") for s in pages]
    rows = fr.pool_map(one, jobs, label="page")
    (OUT / f"{what}.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        line = f"{r['sheet'][-24:]}: {r['rules']} rules ({r['rule_seconds']} s)"
        for mode in MODES:
            m = r[mode]
            line += f" | {mode}: angle {m['phi']:+.2f}"
            if "frame_ink" in m:
                line += (f" frame left {100 * m['frame_left'] / max(m['frame_ink'], 1e-9):.2f}%"
                         f" boxes left {100 * m['box_left'] / m['box_ink']:.0f}%" if m["box_ink"] else
                         f" frame left {100 * m['frame_left'] / max(m['frame_ink'], 1e-9):.2f}%")
                line += f" map cut {100 * m['map_cut'] / m['map_ink']:.3f}%"
            else:
                line += f" bows {m['bow']} boxes {m['boxes']}"
        print(line, flush=True)
    if what == "test":
        for mode in MODES:
            s = {k: sum(r[mode][k] for r in rows) for k in ("frame_ink", "frame_left", "box_ink", "box_left", "map_ink", "map_cut")}
            print(f"POOLED {mode}: frame ink left {100 * s['frame_left'] / s['frame_ink']:.2f}%, box ink left "
                  f"{100 * s['box_left'] / s['box_ink']:.1f}%, map ink cut {100 * s['map_cut'] / s['map_ink']:.3f}%", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
