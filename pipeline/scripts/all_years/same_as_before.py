"""Compare a copy of solver outputs taken before a change with the outputs now (byte for byte; an .npz by its arrays
when the bytes differ, a .parquet by its table), to show the 2000 map's stages are unchanged.

    uv run python scripts/all_years/same_as_before.py BEFORE_DIR NOW_DIR"""
import filecmp
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    before, now = Path(sys.argv[1]), Path(sys.argv[2])
    diff, same = [], 0
    for f in sorted(before.rglob("*")):
        if f.is_dir():
            continue
        g = now / f.relative_to(before)
        if not g.exists():
            diff.append(("missing", str(g)))
            continue
        if filecmp.cmp(f, g, shallow=False):
            same += 1
            continue
        if f.suffix == ".npz":
            a, b = np.load(f, allow_pickle=True), np.load(g, allow_pickle=True)
            if set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files):
                same += 1
                continue
            diff.append(("arrays", str(g), [k for k in a.files if k not in b.files or not np.array_equal(a[k], b[k])]))
        elif f.suffix == ".parquet" and pd.read_parquet(f).equals(pd.read_parquet(g)):
            same += 1
        else:
            diff.append(("bytes", str(g)))
    print(f"{same} files the same; differences: {diff or 'none'}")


if __name__ == "__main__":
    main()
