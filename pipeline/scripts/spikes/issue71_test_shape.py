"""Issue #71: the straight-line shape decision scored on the 22 never-trained test sheets, with the heads trained on
the 39 training labels (saved by issue71_look39.py), beside the simple cut's scores.  Throwaway; prints as it goes.
"""
import json
import sys

import numpy as np
import torch

from sfzmaps.frame import blocks as blk
from sfzmaps.frame import data, layout, look, scores, shape
from sfzmaps.frame import run as fr

SRC = data.DATA_DIR / "frame-look39"
OUT = data.DATA_DIR / "frame-test-shape"
FEATS = data.OUT
T = 1.540
TEST = data.LABELS.parent / "frame-labels-test"
PARAMS = shape.Params(lam_notch=1e4, beta=3, lam_block=1e3)     # the first spike's numbers


def one(job):
    sid, zs, bounds, maps = job
    lum = data.load_luminance(data.page_path(*sid))
    p = fr.chances(zs, lum.shape, T)
    dec, frm, ns, bs = shape.decide(p, lum, PARAMS, bounds, maps)
    fr.write_outputs(OUT, sid, p, dec, maps, page=data.page_path(*sid), fitted=(frm, ns, bs))
    cls = data.class_mask(data.load_label(*sid, TEST), data.load_remove(*sid, TEST), lum.shape)
    sc = scores.sheet_scores(p, dec.deleted, cls, lum)
    return {"series": sid[0], "sheet": sid[1], "left": sc["frame_kept_share"], "cut": sc["map_cut_share"],
            "frame_kept": sc["frame_kept"], "map_cut": sc["map_cut"], "notches": len(ns), "blocks": len(bs),
            "phi": frm.phi}


def main():
    train = sorted(data.labelled_sheets())
    test = sorted((p.parent.name, p.stem) for p in TEST.glob("*/*.json"))
    heads = []
    for sd in torch.load(SRC / "heads.pt"):
        h = layout.new_head(); h.load_state_dict(sd); heads.append(h.eval())
    bounds = blk.block_bounds(train, root=data.LABELS)
    counts = data.map_counts(data.PAGE_KINDS)
    jobs = [(s, tuple(layout.head_logits(h, fr.load_features(FEATS, s)) for h in heads), bounds, counts[s])
            for s in test]
    rows = fr.pool_map(one, jobs, label="test sheet")
    old = {(r["series"], r["sheet"]): r for r in json.load(open(SRC / "test_scores.json"))}
    for r in rows:
        o = old[(r["series"], r["sheet"])]
        print(f"{r['series']}/{r['sheet']}: shape left in {100 * r['left']:.2f}% cut {100 * r['cut']:.3f}% "
              f"({r['notches']} notches, {r['blocks']} blocks, angle {r['phi']:.2f}) | simple left in "
              f"{100 * o['left']:.2f}% cut {100 * o['cut']:.3f}%", flush=True)
    (OUT / "test_scores.json").write_text(json.dumps(rows, indent=1))
    look.series_mosaics(OUT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
