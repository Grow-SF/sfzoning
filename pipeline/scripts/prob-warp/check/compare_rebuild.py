"""Compare rebuilt prob-warp outputs with the rescued originals, file by file.
  identical   same bytes
  same data   bytes differ but every array / pixel / JSON value is equal
  DIFFERS     with the largest absolute difference per array (npz), pixel share differing (png)
Logs (*.log) hold clock times and are skipped.  usage: compare_rebuild.py ORIG_DIR NEW_DIR [ORIG_DIR NEW_DIR ...]"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(a, b):
    if sha(a) == sha(b):
        return "identical", ""
    if a.suffix == ".npz":
        A, B = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
        if set(A.files) != set(B.files):
            return "DIFFERS", f"keys {sorted(set(A.files) ^ set(B.files))}"
        worst = []
        for k in A.files:
            x, y = A[k], B[k]
            if x.shape != y.shape:
                worst.append(f"{k}: shape {x.shape} vs {y.shape}")
            elif x.dtype.kind in "fc":
                d = np.nanmax(np.abs(x.astype(float) - y.astype(float))) if x.size else 0.0
                nan = int((np.isnan(x) != np.isnan(y)).sum())
                if d > 0 or nan:
                    worst.append(f"{k}: max |diff| {d:.3g}" + (f", nan mismatch {nan}" if nan else ""))
            elif not np.array_equal(x, y):
                worst.append(f"{k}: {int((x != y).sum())} of {x.size} differ")
        return ("same data", "") if not worst else ("DIFFERS", "; ".join(worst[:4]))
    if a.suffix == ".png":
        import cv2
        x, y = cv2.imread(str(a), cv2.IMREAD_UNCHANGED), cv2.imread(str(b), cv2.IMREAD_UNCHANGED)
        if x is None or y is None or x.shape != y.shape:
            return "DIFFERS", "shape"
        n = int((x != y).any(-1).sum()) if x.ndim == 3 else int((x != y).sum())
        return ("same data", "") if n == 0 else ("DIFFERS", f"{n / (x.shape[0] * x.shape[1]):.4%} of pixels")
    if a.suffix == ".json":
        return ("same data", "") if json.loads(a.read_text()) == json.loads(b.read_text()) else ("DIFFERS", "values")
    return "DIFFERS", "bytes"


args = sys.argv[1:]
for orig, new in zip(args[::2], args[1::2]):
    orig, new = Path(orig), Path(new)
    print(f"== {orig.name}")
    for a in sorted(orig.rglob("*")):
        if a.is_dir() or a.suffix == ".log":
            continue
        b = new / a.relative_to(orig)
        if not b.exists():
            print(f"  MISSING in rebuild  {a.relative_to(orig)}")
            continue
        verdict, detail = compare(a, b)
        print(f"  {verdict:10s} {a.relative_to(orig)}" + (f"  ({detail})" if detail else ""))
    extra = [b.relative_to(new) for b in sorted(new.rglob("*")) if b.is_file() and b.suffix != ".log"
             and not (orig / b.relative_to(new)).exists()]
    if extra:
        print("  only in rebuild (lost from the original): " + ", ".join(map(str, extra)))
