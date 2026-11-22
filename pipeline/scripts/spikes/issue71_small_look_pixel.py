"""Issue #71: the same 44 pages as the small look, cut the SIMPLE way: the layout's chances, cut where deleting is
cheaper, outside = what connects to the file's edge, one kept region (`decide.decide`).  No shape fitting.  Heads on
the original 21 labels, saved so later looks don't retrain.  Throwaway.
"""
import json
import time

import numpy as np
import torch

from sfzmaps import gpu
from sfzmaps.frame import data, decide, layout, look
from sfzmaps.frame import run as fr

OUT = data.DATA_DIR / "frame-small-look-pixel"
FEATS = data.OUT
T = 1.540
OLD21 = [tuple(x) for x in json.load(open("/Users/sbuss/.claude/jobs/0ba5fc75/tmp/old21.json"))]


def main():
    lab = set(data.labelled_sheets())
    done = sorted((p.parent.parent.name, p.parent.name)
                  for p in (data.DATA_DIR / "frame-small-look").glob("*/*/outside.png"))
    feats, targets = fr._features_and_targets(OLD21, FEATS)
    t0 = time.perf_counter()
    with gpu.gpu_turn(True, "mps"):
        heads = [fr._train(feats, targets, OLD21, seed, "mps")[0] for seed in layout.HEAD_SEEDS]
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save([h.state_dict() for h in heads], OUT / "heads21.pt")
    print(f"3 heads in {time.perf_counter() - t0:.0f} s", flush=True)
    counts = data.map_counts(data.PAGE_KINDS)
    for i, sid in enumerate(done, 1):
        lum = data.load_luminance(data.page_path(*sid))
        p = fr.chances([layout.head_logits(h, fr.load_features(FEATS, sid)) for h in heads], lum.shape, T)
        d = decide.decide(p, lum, counts[sid])
        fr.write_outputs(OUT, sid, p, d, counts[sid], page=data.page_path(*sid))
        print(f"{i}/{len(done)} {sid[0]}/{sid[1]}", flush=True)
    look.series_mosaics(OUT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
