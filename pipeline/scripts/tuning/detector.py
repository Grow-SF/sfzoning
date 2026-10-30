"""Misplacement signals per sheet vs the answer key (the key is only compared, never used
to decide). Optional --work DIR reads an older copy of the work tree.

    uv run python scripts/tuning/detector.py 1948 1960
    uv run python scripts/tuning/detector.py --work /path/to/old/work 1960
"""
import json
import sys
import warnings

import numpy as np

from sfzmaps import config, pages, placement, reading, runner
from sfzmaps.evaluate import answer_key_distances
from sfzmaps.sources import datasf
from sfzmaps.transform import SheetTransform

warnings.filterwarnings("ignore")
args = sys.argv[1:]
if args[:1] == ["--work"]:
    from pathlib import Path
    config.WORK_DIR = Path(args[1])
    args = args[2:]
ref = datasf.load_reference()
for series in args:
    for page in [p for p in pages.discover() if p.series == series]:
        d = config.WORK_DIR / series / page.sheet_id
        if not (d / "transform.json").exists() or not (d / "refine.json").exists():
            continue
        t = SheetTransform.from_json(json.loads((d / "transform.json").read_text()))
        ms = json.loads((d / "refine.json").read_text())["matches"]
        lab = [m["label"] for m in ms if m["px"] is not None]
        inl_frac = lab.count("inlier") / max(len(lab), 1)
        ag = placement.label_agreement(t, reading.load_labels(d / "labels.json"), ref)
        sc = json.loads((d / "score.json").read_text())
        misplaced = ((ag.n >= config.MISPLACED_MIN_LABELS and ag.fraction < config.MISPLACED_MAX_LABEL_AGREEMENT)
                     or inl_frac < config.MISPLACED_MIN_INLIER_FRAC)
        key = runner._answer_key_gcps(page, d / "page.png") if runner.answer_key_path(page) else None
        kd = float(np.median(answer_key_distances(t, key))) if key else float("nan")
        print(f"{series}/{page.sheet_id:8s} inlier_frac {inl_frac:.2f}  label_agree {ag.fraction:.2f} (n {ag.n})  "
              f"misplaced {'YES' if misplaced else 'no '}  key_med {kd:.1f}")
