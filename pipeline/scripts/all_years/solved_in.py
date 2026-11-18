"""Which series have a solve (solve/runs/final.npz) under a solver root, one per line; with --wait, waits until every
series of paths.ALL_YEARS has one (checking every 30 s), printing each as it lands.

    uv run python scripts/all_years/solved_in.py ROOT [--wait]"""
import sys
import time
from pathlib import Path

from sfzmaps.solver import paths


def done(root: Path) -> list[str]:
    return [s for s in paths.ALL_YEARS if (root / s / "solve" / "runs" / "final.npz").is_file()]


if __name__ == "__main__":
    root, wait = Path(sys.argv[1]), "--wait" in sys.argv
    seen = set()
    while True:
        for s in done(root):
            if s not in seen:
                print(s, flush=True)
                seen.add(s)
        if not wait or len(seen) == len(paths.ALL_YEARS):
            break
        time.sleep(30)
