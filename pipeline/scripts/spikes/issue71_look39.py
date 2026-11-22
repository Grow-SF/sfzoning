"""Issue #71: heads retrained on ALL current training labels (39), the simple cut (layout chances, outside = what
connects to the file's edge), on the same 44 look pages and on the never-trained test sheets, which are scored.
Throwaway; prints as it goes.
"""
import json
import time

import numpy as np
import torch

from sfzmaps import gpu
from sfzmaps.frame import data, decide, layout, look, scores
from sfzmaps.frame import run as fr

OUT = data.DATA_DIR / "frame-look39"
FEATS = data.OUT
T = 1.540
TEST = data.LABELS.parent / "frame-labels-test"


def main():
    train = sorted(data.labelled_sheets())
    test = sorted((p.parent.name, p.stem) for p in TEST.glob("*/*.json"))
    assert not set(train) & set(test)
    pages = sorted((p.parent.parent.name, p.parent.name)
                   for p in (data.DATA_DIR / "frame-small-look").glob("*/*/outside.png"))
    feats, targets = fr._features_and_targets(train, FEATS)
    t0 = time.perf_counter()
    with gpu.gpu_turn(True, "mps"):
        heads = [fr._train(feats, targets, train, seed, "mps")[0] for seed in layout.HEAD_SEEDS]
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save([h.state_dict() for h in heads], OUT / "heads.pt")
    print(f"3 heads on {len(train)} labels in {time.perf_counter() - t0:.0f} s", flush=True)
    counts = data.map_counts(data.PAGE_KINDS)
    rows = []
    for i, sid in enumerate(test + [p for p in pages if p not in train and p not in test], 1):
        lum = data.load_luminance(data.page_path(*sid))
        p = fr.chances([layout.head_logits(h, fr.load_features(FEATS, sid)) for h in heads], lum.shape, T)
        d = decide.decide(p, lum, counts[sid])
        fr.write_outputs(OUT, sid, p, d, counts[sid], page=data.page_path(*sid))
        if sid in test:
            cls = data.class_mask(data.load_label(*sid, TEST), data.load_remove(*sid, TEST), lum.shape)
            sc = scores.sheet_scores(p, d.deleted, cls, lum)
            rows.append({"series": sid[0], "sheet": sid[1], "left": sc["frame_kept_share"], "cut": sc["map_cut_share"],
                         "frame_kept": sc["frame_kept"], "map_cut": sc["map_cut"]})
            print(f"TEST {sid[0]}/{sid[1]}: frame/legend ink left in {100 * sc['frame_kept_share']:.2f}%, "
                  f"map ink cut {100 * sc['map_cut_share']:.3f}%", flush=True)
        else:
            print(f"{i} {sid[0]}/{sid[1]}", flush=True)
    (OUT / "test_scores.json").write_text(json.dumps(rows, indent=1))
    look.series_mosaics(OUT)
    print("done", flush=True)


if __name__ == "__main__":
    main()
