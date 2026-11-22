"""Issue #71 trial (owner 2026-11-21: "The boxes should be aligned with the frame"; "The 3x rule is not a rule ... I
still don't want thresholds").  Two changes tried on the 22 test sheets and the four flagged pages, against the cut as
committed (rules shared, 1f83ee4):
  square   rectangular label boxes take the frame's angle only (no angle from the model's blob);
  likely   square, and the more likely reading wins everywhere: frame-or-box against map at even odds (no 3x cost).
Throwaway; prints as it goes.  Nothing in the package is changed: the settings are swapped inside each worker.

    uv run python scripts/spikes/issue71_likely.py
"""
import json
import os
import subprocess

import numpy as np
import torch

from sfzmaps.frame import blocks as blk
from sfzmaps.frame import data, decide, geometry, layout, look, paper, shape
from sfzmaps.frame import run as fr

SRC = data.DATA_DIR / os.environ.get("FRAME_HEADS", "frame-look39")
OUT = data.DATA_DIR / os.environ.get("FRAME_OUT", "frame-likely")
BEFORE = os.environ.get("LIKELY_BEFORE", "frame-rules/test.json:one")     # a rows file and the mode in it
TEST = data.LABELS.parent / "frame-labels-test"
T = 1.540
PARAMS = shape.Params(lam_notch=1e4, beta=3, lam_block=1e3)
FLAGGED = [("2000-digital-su", "2000-digital-su-p11"), ("1975", "20180102114142444-p18"), ("1948", "4"),
           ("1980-1990-su", "1980-1990-su-p01")]
MODES = tuple(os.environ.get("LIKELY_MODES", "square,likely").split(","))   # "now": the package as it is


def settings(mode):
    if mode == "now":
        return
    blk.MOMENT_ANGLES = ()                                  # rectangles: the frame's angle only
    cf = 1.0 if mode == "likely" else 3.0
    decide.COST_FRAME = geometry.COST_FRAME = paper.COST_FRAME = cf
    decide.cut_level.__defaults__ = (cf, 1.0)


def one(job):
    sid, zs, bounds, maps, scored = job
    torch.set_num_threads(1)
    lum = data.load_luminance(data.page_path(*sid))
    p = fr.chances(zs, lum.shape, T)
    row = {"sheet": f"{sid[0]}/{sid[1]}"}
    if scored:
        polys, rem = data.load_label(*sid, TEST), data.load_remove(*sid, TEST)
        cls = data.class_mask(polys, rem, lum.shape)
        box_masks = [data.class_mask(polys, [q], lum.shape) == 2 for q in rem]
        ink = 1.0 - lum
    for mode in MODES:
        settings(mode)
        dec, frm, ns, bs = shape.decide(p, lum, PARAMS, bounds, maps)
        out = OUT / mode
        fr.write_outputs(out, sid, p, dec, maps, page=data.page_path(*sid), fitted=(frm, ns, bs))
        _, deleted, cut = look._load_outputs(out, sid)
        look.sheet_picture(lum, deleted.astype(np.float32), deleted, 1300, cut).save(out / f"{sid[0]}_{sid[1][-6:]}.png")
        r = {"phi": round(frm.phi, 2), "notches": len(ns),
             "boxes": [(b.kind, round(b.theta, 1), round(b.gain)) for b in bs]}
        if scored:
            d = dec.deleted
            r.update(frame_ink=float(ink[cls == 1].sum()), frame_left=float(ink[(cls == 1) & ~d].sum()),
                     box_ink=float(ink[cls == 2].sum()), box_left=float(ink[(cls == 2) & ~d].sum()),
                     map_ink=float(ink[cls == 0].sum()), map_cut=float(ink[(cls == 0) & d].sum()),
                     map_cut_boxes=float(ink[(cls == 0) & dec.removed].sum()),
                     per_box=[round(float(ink[m & d].sum() / max(ink[m].sum(), 1e-9)), 2) for m in box_masks])
        row[mode] = r
    return row


def main():
    train = sorted({tuple(f.split("/")[-2:]) for f in subprocess.run(
        ["git", "ls-files", "inputs/frame-labels"], capture_output=True, text=True).stdout.split() if f.endswith(".json")})
    train = [(s, h[:-5]) for s, h in train]
    heads = []
    for sd in torch.load(SRC / "heads.pt"):
        hd = layout.new_head(); hd.load_state_dict(sd); heads.append(hd.eval())
    bounds = blk.block_bounds(train, root=data.LABELS)
    counts = data.map_counts(data.PAGE_KINDS)
    test = sorted((p.parent.name, p.stem) for p in TEST.glob("*/*.json"))
    jobs = [(s, tuple(layout.head_logits(hd, fr.load_features(data.OUT, s)) for hd in heads), bounds, counts[s],
             s in test) for s in test + FLAGGED]
    rows = fr.pool_map(one, jobs, label="page")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rows.json").write_text(json.dumps(rows, indent=1))
    bfile, bmode = BEFORE.split(":")
    before = {r["sheet"]: r[bmode] for r in json.load(open(data.DATA_DIR / bfile)) if "frame_ink" in r[bmode]}
    for r in rows:
        line = r["sheet"][-26:] + ":"
        ms = [("before", before[r["sheet"]])] if r["sheet"] in before else []
        for name, m in ms + [(k, r[k]) for k in MODES]:
            line += f" | {name}: angle {m['phi']:+.2f}"
            if "frame_ink" in m:
                line += f" frame left {100 * m['frame_left'] / max(m['frame_ink'], 1e-9):.2f}%"
                if m["box_ink"]:
                    line += f" boxes left {100 * m['box_left'] / m['box_ink']:.0f}%"
                    if "per_box" in m:
                        line += f" {m['per_box']}"
                line += f" map cut {100 * m['map_cut'] / m['map_ink']:.3f}%"
            if name != "before":
                line += f" boxes {m['boxes']}"
        print(line, flush=True)
    scored = [r for r in rows if "frame_ink" in r[MODES[0]]]
    for name in ("before",) + MODES:
        ms = [before[r["sheet"]] if name == "before" else r[name] for r in scored]
        s = {k: sum(m[k] for m in ms) for k in ("frame_ink", "frame_left", "box_ink", "box_left", "map_ink", "map_cut")}
        print(f"POOLED {name}: frame ink left {100 * s['frame_left'] / s['frame_ink']:.2f}%, box ink left "
              f"{100 * s['box_left'] / s['box_ink']:.1f}%, map ink cut {100 * s['map_cut'] / s['map_ink']:.3f}%", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
