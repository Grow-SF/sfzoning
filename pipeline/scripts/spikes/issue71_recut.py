"""Issue #71, plan Task 0: the pages the owner flagged earlier, cut again with the current model (the heads trained on
the 39 training labels, saved by issue71_look39.py) and the straight-line cut at the approved numbers.  Unscored:
these pages have no labels.  Throwaway; prints as it goes.
"""
import json
import os

import numpy as np
import torch
from PIL import Image

from sfzmaps.frame import blocks as blk
from sfzmaps.frame import data, layout, look, shape
from sfzmaps.frame import run as fr

SRC = data.DATA_DIR / "frame-look39"
OUT = data.DATA_DIR / os.environ.get("FRAME_OUT", "frame-recut")     # FRAME_OUT: another folder, to keep a run's files
T = 1.540
PARAMS = shape.Params(lam_notch=1e4, beta=3, lam_block=1e3)
PAGES = [("1975", "20180102114142444-p18"), ("2000-digital-su", "2000-digital-su-p11"), ("1948", "4"),
         ("1980-1990-su", "1980-1990-su-p01")]


def one(job):
    sid, zs, bounds, maps = job
    lum = data.load_luminance(data.page_path(*sid))
    p = fr.chances(zs, lum.shape, T)
    dec, frm, ns, bs = shape.decide(p, lum, PARAMS, bounds, maps)
    fr.write_outputs(OUT, sid, p, dec, maps, page=data.page_path(*sid), fitted=(frm, ns, bs))
    _, deleted, polys = look._load_outputs(OUT, sid)
    pic = look.sheet_picture(lum, deleted.astype(np.float32), deleted, 1600, polys)
    pic.save(OUT / f"{sid[0]}_{sid[1][-6:]}.png")
    return {"sheet": f"{sid[0]}/{sid[1]}", "phi": frm.phi, "framed": [s for s in shape.SIDES if frm.framed(s)],
            "notches": len(ns), "blocks": [(b.kind, round(b.theta, 1)) for b in bs]}


def main():
    train = sorted(data.labelled_sheets())
    assert not set(PAGES) & set(train)
    heads = []
    for sd in torch.load(SRC / "heads.pt"):
        h = layout.new_head(); h.load_state_dict(sd); heads.append(h.eval())
    bounds = blk.block_bounds(train, root=data.LABELS)
    counts = data.map_counts(data.PAGE_KINDS)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, tuple(layout.head_logits(h, fr.load_features(data.OUT, s)) for h in heads), bounds, counts[s])
            for s in PAGES]
    for r in fr.pool_map(one, jobs, label="page"):
        print(json.dumps(r), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
