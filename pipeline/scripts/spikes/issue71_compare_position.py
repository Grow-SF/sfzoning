"""Issue #71: held-out folds for each position-channel setting, the same folds, seed and recipe, each into its own
folder; then the ink scores side by side (the owner's goal is frame ink only, so no area numbers).

    uv run python scripts/spikes/issue71_compare_position.py
"""

import json
import logging

from sfzmaps import gpu
from sfzmaps.frame import data
from sfzmaps.frame import run as frun
from sfzmaps.frame.train import Recipe

SETTINGS = {"none": (), "valid": ("valid",)}
KEYS = ("log_loss_frame_band", "log_loss_map_band", "log_loss_all_ink", "frame_called_map", "map_called_frame")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = data.DATA_DIR if hasattr(data, "DATA_DIR") else data.OUT.parent
    results = {}
    with gpu.gpu_turn(True, "mps"):
        for name, channels in SETTINGS.items():
            data.POSITION_CHANNELS = channels
            out = root / f"frame-cmp-{name}"
            results[name] = frun.run_folds(data.labelled_sheets(), Recipe(), "mps", out=out)
    print("\nheld-out ink scores (ink-weighted means over all 18 sheets; lower is better)")
    print(f"{'setting':8s} " + " ".join(f"{k:>20s}" for k in KEYS))
    for name, r in results.items():
        print(f"{name:8s} " + " ".join(f"{r['overall'][k]['mean']:20.4f}" for k in KEYS))
    for series in sorted(results["none"]["series"]):
        for name, r in results.items():
            m = r["series"][series]
            print(f"  {series:22s} {name:6s} " + " ".join(f"{m[k]['mean']:8.4f}" for k in KEYS))
    print(json.dumps({n: r["overall"] for n, r in results.items()})[:0])


if __name__ == "__main__":
    main()
