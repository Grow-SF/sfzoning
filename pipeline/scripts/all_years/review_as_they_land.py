"""Make each series' review entry as soon as its solve lands under a solver root (review_page.py --series S, one at a
time, one process), until every series (2000 and paths.ALL_YEARS) has one.  Checks every 60 s; prints each entry made.

    uv run python scripts/all_years/review_as_they_land.py ROOT"""
import subprocess
import sys
import time
from pathlib import Path

from sfzmaps.solver import paths

HERE = Path(__file__).parent


def solved(root: Path, s: str) -> bool:
    folder = root if s == paths.SERIES_2000 else root / s
    return (folder / "solve" / "runs" / "final.npz").is_file() and (folder / "solve" / "report.json").is_file()


def main():
    root = Path(sys.argv[1])
    want = [paths.SERIES_2000, *paths.ALL_YEARS]
    done = set(sys.argv[2:])
    while len(done) < len(want):
        for s in want:
            if s in done or not solved(root, s):
                continue
            time.sleep(20)                       # a solve folder is renamed into place whole; let it settle
            r = subprocess.run([sys.executable, str(HERE / "review_page.py"), "--solve-root", str(root), "--series", s],
                               capture_output=True, text=True)
            print(s, "made" if r.returncode == 0 else f"FAILED: {r.stderr.strip()[-300:]}", flush=True)
            done.add(s)
        time.sleep(60)


if __name__ == "__main__":
    main()
