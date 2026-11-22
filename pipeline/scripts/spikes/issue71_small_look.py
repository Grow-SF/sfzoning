"""Issue #71: the small eye-first look.  Final heads on all labels (3 seeds), two unlabelled pages per series, the
shape decision at the best numbers seen in the stopped held-out run, one mosaic per series.  Throwaway.

    pixi run uv run python scripts/spikes/issue71_small_look.py
"""
import logging
import time

from sfzmaps import gpu
from sfzmaps.frame import blocks as blk
from sfzmaps.frame import data, layout, look, shape
from sfzmaps.frame import run as fr

OUT = data.DATA_DIR / "frame-small-look"
FEATS = data.OUT                      # the features the stopped run extracted for every page
T = 1.540                             # the stopped run's fitted temperature
PARAMS = shape.Params(lam_notch=1e3, beta=10, lam_block=1e3)   # the best point it had seen
PER_SERIES = 2


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    labelled = sorted(data.labelled_sheets())
    lab = set(labelled)
    by = {}
    for sid in data.map_pages(data.PAGE_KINDS, data.WORK):
        if sid not in lab:
            by.setdefault(sid[0], []).append(sid)
    pick = []
    for series, sids in sorted(by.items()):
        idx = sorted({0, len(sids) // 2})[:PER_SERIES]
        pick += [sids[i] for i in idx]
    feats, targets = fr._features_and_targets(labelled, FEATS)
    t0 = time.perf_counter()
    with gpu.gpu_turn(True, "mps"):
        heads = [fr._train(feats, targets, labelled, seed, "mps")[0] for seed in layout.HEAD_SEEDS]
    print(f"3 heads in {time.perf_counter() - t0:.0f} s")
    bounds = blk.block_bounds(labelled, root=data.LABELS)
    counts = data.map_counts(data.PAGE_KINDS)
    jobs = [fr.PageJob(sid=s, page=data.page_path(*s, work=data.WORK),
                       zs=tuple(layout.head_logits(h, fr.load_features(FEATS, s)) for h in heads),
                       temperature=T, maps=counts[s], bounds=bounds) for s in pick]
    t0 = time.perf_counter()
    rows = fr.pool_map(fr.predict_page, jobs, PARAMS, OUT, label="page")
    print(f"{len(jobs)} pages decided in {time.perf_counter() - t0:.0f} s")
    for s, r in zip(pick, rows):
        print(f"{s[0]}/{s[1]}: notches {r['notches']}, blocks {r['blocks']}, side ink left {r['side_ink_left_total']:.0f}")
    look.series_mosaics(OUT)


if __name__ == "__main__":
    main()
