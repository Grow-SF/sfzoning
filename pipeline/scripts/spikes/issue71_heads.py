"""Issue #71: train the three layout heads on every committed training label and save them (throwaway helper for the
trials; the package's own `fit` replaces it).

    FRAME_HEADS=frame-look48 uv run python scripts/spikes/issue71_heads.py
"""
import os
import subprocess
import time

import torch

from sfzmaps import gpu
from sfzmaps.frame import data, layout
from sfzmaps.frame import run as fr

OUT = data.DATA_DIR / os.environ["FRAME_HEADS"]


def main():
    files = subprocess.run(["git", "ls-files", "inputs/frame-labels"], capture_output=True, text=True).stdout.split()
    train = sorted((f.split("/")[-2], f.split("/")[-1][:-5]) for f in files if f.endswith(".json"))
    test = {(p.parent.name, p.stem) for p in (data.LABELS.parent / "frame-labels-test").glob("*/*.json")}
    assert not set(train) & test
    feats, targets = fr._features_and_targets(train, data.OUT)
    t0 = time.perf_counter()
    with gpu.gpu_turn(True, "mps"):
        heads = [fr._train(feats, targets, train, seed, "mps")[0] for seed in layout.HEAD_SEEDS]
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save([h.state_dict() for h in heads], OUT / "heads.pt")
    (OUT / "train.txt").write_text("\n".join(f"{s}/{h}" for s, h in train))
    print(f"3 heads on {len(train)} labels in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
