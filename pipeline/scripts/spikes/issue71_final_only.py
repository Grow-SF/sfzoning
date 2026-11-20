"""Issue #71: the final model only (no measuring folds) — train both passes on every labelled sheet, predict every
unlabelled map page, draw the series mosaics.  For judging by eye before spending time on held-out scores.

    uv run python scripts/spikes/issue71_final_only.py [--device mps]
"""

import argparse
import logging

from sfzmaps import gpu
from sfzmaps.frame import data, look
from sfzmaps.frame import run as frun
from sfzmaps.frame.train import Recipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with gpu.gpu_turn(True, a.device):
        frun.train_final(data.labelled_sheets(), Recipe(), a.device, out=data.OUT)
        frun.predict_all(a.device, out=data.OUT)
        for p in look.series_mosaics(data.OUT):
            print("mosaic", p)


if __name__ == "__main__":
    main()
